"""Integration tests: existing sim preserved + connected Operations→ROI workflow."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import main
import server
from roi import run_roi_analysis
from roi import period_runner as pr
from roi.engine import _deep_merge, config_hash, operations_from_config
from roi.models import AnalysisConfig, ROIAssumptions
from roi.validation import ROIValidationError, validate_request


def small_config():
    cfg = main.load_yaml("config.yaml")
    cfg["task_generation"]["num_tasks"] = 8
    return cfg


def ops_block(scenarios, params=None, one_day=None, run_id="run-test-1"):
    """Build an `operations` block as the server would after a /simulate run."""
    return {
        "run_id": run_id,
        "timestamp": "2026-07-05 13:01",
        "params": params or {"grid_kw": 10.5, "farm_pv_kw": 5.0, "panel_w": 650,
                             "tractors": 3, "chargers": 2, "charger_kw": 6.6,
                             "battery_kwh": 44.8, "num_tasks": 8, "seed": 42},
        "one_day": one_day or {},
        "scenarios": scenarios,
    }


SMART = {"id": "smart", "label": "Smart", "charging_strategy": "smart",
         "tractor_pv_enabled": False, "load_shedding": False}
FULL = {"id": "full_smart", "label": "Full smart", "charging_strategy": "smart_with_swap",
        "tractor_pv_enabled": True, "load_shedding": True}


class TestExistingBehaviour(unittest.TestCase):
    def test_one_day_simulation_still_runs(self):
        cfg = small_config()
        out = main.run_simulation(cfg, ["full_smart"], start_date="2026-06-01", seed=42)
        s = out["scenarios"][0]["summary"]
        self.assertIsInstance(s["completed_tasks"], int)
        self.assertIsInstance(s["total_cost_eur"], float)
        self.assertEqual(len(out["scenarios"][0]["timeseries"]), 96)

    def test_dashboard_endpoint_still_works(self):
        cfg = small_config()
        results = server.run_scenarios(cfg, [{"name": "smart", "charging_strategy": "smart",
                                              "tractor_pv_enabled": False, "load_shedding": False}])
        self.assertEqual(len(results), 1)
        self.assertIn("total_cost_eur", results[0])
        self.assertIn("tasks", results[0])

    def test_roi_disabled_leaves_scenario_results_unchanged(self):
        cfg = small_config()
        with_roi = main.run_simulation(cfg, ["smart"], start_date="2026-06-01", seed=7)
        cfg_no_roi = copy.deepcopy(cfg)
        cfg_no_roi.pop("roi", None)
        without_roi = main.run_simulation(cfg_no_roi, ["smart"], start_date="2026-06-01", seed=7)
        self.assertAlmostEqual(with_roi["scenarios"][0]["summary"]["total_cost_eur"],
                               without_roi["scenarios"][0]["summary"]["total_cost_eur"], places=9)


class TestServerRunStore(unittest.TestCase):
    def test_simulate_stores_run_and_returns_identifier(self):
        cfg = small_config()
        scen = [SMART]
        results = server.run_scenarios(cfg, scen)
        ident = server._store_ops_run({"num_tasks": 8, "seed": 42}, scen, cfg, results)
        self.assertIn("operations_run_id", ident)
        # Latest run is retrievable by its id; a wrong id is rejected.
        self.assertIsNotNone(server._get_ops_run(ident["operations_run_id"]))
        self.assertIsNone(server._get_ops_run("nope-unknown"))

    def test_roi_without_run_id_is_unusable(self):
        # Engine refuses to analyse when there are no Operations scenarios.
        cfg = small_config()
        with self.assertRaises(ROIValidationError):
            run_roi_analysis(cfg, {"operations": {"scenarios": []}, "investments": ["farm_pv"]})

    def test_config_hash_changes_when_operations_control_changes(self):
        cfg = small_config()
        h1 = config_hash(cfg)
        cfg2 = copy.deepcopy(cfg)
        cfg2["grid"]["max_power_kw"] = 7.0     # an Operations control changed
        self.assertNotEqual(h1, config_hash(cfg2))


class TestConnectedROI(unittest.TestCase):
    def test_roi_returns_valid_json_with_new_shape(self):
        cfg = small_config()
        report = run_roi_analysis(cfg, {
            "start": "2026-06-01", "end": "2026-06-02", "period_mode": "exact", "horizon": 5,
            "investments": ["electric_fleet", "farm_pv", "tractor_roof_pv", "portfolio"],
            "operations": ops_block([SMART, FULL]),
        })
        json.dumps(report, allow_nan=False)      # strictly JSON-safe
        self.assertIn("long_term", report)
        self.assertIn("portfolios", report)
        self.assertEqual(report["meta"]["operations_run_id"], "run-test-1")

    def test_all_operations_scenarios_appear_in_long_term(self):
        cfg = small_config()
        report = run_roi_analysis(cfg, {
            "start": "2026-06-01", "end": "2026-06-02", "period_mode": "exact",
            "investments": ["farm_pv"], "operations": ops_block([SMART, FULL]),
        })
        names = [r["scenario_name"] for r in report["long_term"]]
        self.assertEqual(set(names), {"Smart", "Full smart"})

    def test_investments_labelled_by_source_scenario(self):
        cfg = small_config()
        report = run_roi_analysis(cfg, {
            "start": "2026-06-01", "end": "2026-06-02", "period_mode": "exact",
            "investments": ["electric_fleet", "farm_pv", "tractor_roof_pv"],
            "operations": ops_block([SMART, FULL]),
        })
        ids = {(i["id"], i["scenario_id"]) for i in report["investments"]}
        # Every investment names its source scenario; roof PV only for the roof scenario.
        self.assertIn(("electric_fleet", "smart"), ids)
        self.assertIn(("farm_pv", "full_smart"), ids)
        self.assertIn(("tractor_roof_pv", "full_smart"), ids)
        self.assertNotIn(("tractor_roof_pv", "smart"), ids)      # Smart has no roof
        for i in report["investments"]:
            self.assertIn("—", i["name"])                       # "<inv> — <scenario>"

    def test_no_scenario_outside_operations(self):
        cfg = small_config()
        report = run_roi_analysis(cfg, {
            "start": "2026-06-01", "end": "2026-06-02", "period_mode": "exact",
            "investments": ["farm_pv", "portfolio"], "operations": ops_block([SMART]),
        })
        allowed = {"smart"}
        for i in report["investments"] + report["portfolios"]:
            self.assertIn(i["scenario_id"], allowed)

    def test_operations_values_used_not_request(self):
        # A stray farm_pv_kw in the request must NOT change the operational basis.
        cfg = small_config()
        report = run_roi_analysis(cfg, {
            "start": "2026-06-01", "end": "2026-06-02", "period_mode": "exact",
            "investments": ["farm_pv"], "farm_pv_kw": 999,     # ignored by ROI
            "operations": ops_block([SMART]),
        })
        self.assertEqual(report["operations_basis"]["farm_pv_kwp"], 5.0)
        self.assertEqual(report["operations_basis"]["n_tractors"], 3)
        self.assertEqual(report["operations_basis"]["seed"], 42)

    def test_default_portfolio_scenario_is_best_completion(self):
        cfg = small_config()
        one_day = {"smart": {"completed_tasks": 4}, "full_smart": {"completed_tasks": 7}}
        report = run_roi_analysis(cfg, {
            "start": "2026-06-01", "end": "2026-06-02", "period_mode": "exact",
            "investments": ["portfolio"],
            "operations": ops_block([SMART, FULL], one_day=one_day),
        })
        self.assertEqual(report["default_portfolio_scenario"], "full_smart")

    def test_representative_year_reflects_seasonal_pv(self):
        cfg = small_config()
        june = main.run_simulation(cfg, ["full_smart"], start_date="2026-06-01", seed=42)
        june_farm_gen = june["scenarios"][0]["summary"]["farm_pv_generated_kwh"]
        report = run_roi_analysis(cfg, {
            "start": "2026-01-01", "end": "2026-12-31", "period_mode": "representative_month",
            "investments": ["farm_pv"], "operations": ops_block([FULL]),
        })
        annual = next(r["farm_pv_generated_kwh"] for r in report["long_term"]
                      if r["scenario_id"] == "full_smart")
        self.assertLess(annual, june_farm_gen * 365.0 * 0.95)
        self.assertEqual(report["meta"]["days_represented"], 365)


class TestAssumptionsAndValidation(unittest.TestCase):
    def test_assumptions_loaded_from_merged_config(self):
        # ROI assumptions come from config, not hard-coded JS — verify the demo values.
        a = ROIAssumptions.from_config(small_config())
        self.assertAlmostEqual(a.diesel.fuel_price_eur_per_litre, 1.55)
        self.assertAlmostEqual(a.electric_fleet.electric_tractor_purchase_eur, 120000.0)
        self.assertAlmostEqual(a.farm_pv.capex_eur_per_kwp, 950.0)

    def test_local_style_override_wins(self):
        # config.local.yaml is deep-merged over config.yaml (same logic as the server).
        base = small_config()
        merged = _deep_merge(base, {"roi": {"diesel": {"fuel_price_eur_per_litre": 2.10}}})
        a = ROIAssumptions.from_config(merged)
        self.assertAlmostEqual(a.diesel.fuel_price_eur_per_litre, 2.10)

    def test_demonstration_values_produce_nonempty_financials(self):
        cfg = small_config()
        report = run_roi_analysis(cfg, {
            "start": "2026-06-01", "end": "2026-06-02", "period_mode": "exact",
            "investments": ["farm_pv"], "operations": ops_block([FULL]),
        })
        farm = next(i for i in report["investments"] if i["id"] == "farm_pv")
        self.assertNotEqual(farm["metrics"].get("status"), "input_required")
        self.assertIsNotNone(farm["metrics"]["npv_eur"])

    def test_zero_capex_is_input_required_not_zero_payback(self):
        cfg = small_config()
        report = run_roi_analysis(cfg, {
            "start": "2026-06-01", "end": "2026-06-02", "period_mode": "exact",
            "investments": ["farm_pv", "portfolio"],
            "assumptions": {"farm_pv": {"capex_eur_per_kwp": 0}},
            "operations": ops_block([FULL]),
        })
        farm = next(i for i in report["investments"] if i["id"] == "farm_pv")
        self.assertEqual(farm["metrics"]["status"], "input_required")
        self.assertIsNone(farm["metrics"]["npv_eur"])
        self.assertIsNone(farm["metrics"]["simple_payback_years"])
        self.assertTrue(farm["metrics"]["missing"])
        # Portfolio must exclude the incomplete farm-PV stage.
        pf = report["portfolios"][0]
        self.assertIn("farm_pv", pf["metrics"].get("excluded_ids", []))

    def test_invalid_request_rejected(self):
        a = ROIAssumptions.from_config(small_config())
        with self.assertRaises(ROIValidationError):
            validate_request("2026-06-10", "2026-06-01", a, ["farm_pv"])


class TestExportAndCLI(unittest.TestCase):
    def test_export_includes_operations_metadata(self):
        from roi import export
        cfg = small_config()
        report = run_roi_analysis(cfg, {
            "start": "2026-06-01", "end": "2026-06-02", "period_mode": "exact",
            "investments": ["farm_pv", "portfolio"], "operations": ops_block([FULL]),
        })
        with tempfile.TemporaryDirectory() as td:
            export.write_all(report, Path(td))
            summary = (Path(td) / "roi_summary.csv").read_text(encoding="utf-8")
            assumptions = json.loads((Path(td) / "roi_assumptions.json").read_text(encoding="utf-8"))
        self.assertIn("operations_run_id", summary)
        self.assertIn("run-test-1", summary)
        self.assertEqual(assumptions["export_meta"]["operations_run_id"], "run-test-1")
        self.assertTrue(assumptions["export_meta"]["demonstration_assumptions"])

    def test_cli_creates_output_files(self):
        from roi import __main__ as roi_cli
        with tempfile.TemporaryDirectory() as td:
            rc = roi_cli.main([
                "--config", "config.yaml", "--start", "2026-06-01", "--end", "2026-06-02",
                "--period-mode", "exact", "--horizon", "5", "--scenarios", "smart,full_smart",
                "--investments", "electric_fleet,portfolio", "--out", td,
            ])
            self.assertEqual(rc, 0)
            for name in ("roi_summary.csv", "roi_cashflows.csv",
                         "roi_assumptions.json", "roi_report.json"):
                self.assertTrue((Path(td) / name).exists(), f"{name} not written")


if __name__ == "__main__":
    unittest.main()
