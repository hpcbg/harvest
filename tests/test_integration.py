"""Integration tests: existing sim preserved + ROI engine end-to-end."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import main
import server
from roi import run_roi_analysis
from roi import period_runner as pr
from roi.models import AnalysisConfig, ROIAssumptions
from roi.validation import ROIValidationError, validate_request


def small_config():
    cfg = main.load_yaml("config.yaml")
    cfg["task_generation"]["num_tasks"] = 8
    return cfg


DEMO_ASSUMPTIONS = {
    "diesel": {"fuel_price_eur_per_litre": 1.5, "travel_litres_per_100km": 22,
               "pto_litres_per_hour": 8, "idle_litres_per_hour": 2},
    "electric_fleet": {"electric_tractor_purchase_eur": 90000,
                       "diesel_equivalent_purchase_eur": 60000,
                       "charger_capex_eur_each": 4000, "charger_installation_eur_each": 1500},
    "farm_pv": {"capex_eur_per_kwp": 900},
    "tractor_roof_pv": {"capex_eur_per_tractor": 1200},
    "outages": {"enabled": True, "frequency_per_year": 6, "average_duration_hours": 3,
                "value_of_lost_load_eur_per_kwh": 5, "islanding_enabled": True,
                "critical_load_kw": 0.5, "backup_battery_kwh": 15, "backup_max_power_kw": 5,
                "islanding_capex_eur": 3000, "backup_capex_eur": 6000},
}


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


class TestROIEndToEnd(unittest.TestCase):
    def test_roi_returns_valid_json(self):
        cfg = small_config()
        report = run_roi_analysis(cfg, {
            "start": "2026-06-01", "end": "2026-06-02", "period_mode": "exact",
            "horizon": 5, "investments": ["electric_fleet", "farm_pv", "portfolio"],
            "assumptions": DEMO_ASSUMPTIONS,
        })
        # Must be strictly JSON-safe (no NaN / Infinity).
        json.dumps(report, allow_nan=False)
        self.assertIn("meta", report)
        self.assertTrue(report["investments"])
        self.assertIn("portfolio", report)
        self.assertEqual(report["meta"]["period_mode"], "exact")

    def test_exact_multiday_aggregates(self):
        cfg = small_config()
        analysis = AnalysisConfig(period_mode="exact")
        variant = pr.make_variant_config(cfg, "full_smart", 5.0, 650.0)
        totals = pr.run_period(variant, "full_smart", analysis, "2026-06-01", "2026-06-02")
        self.assertEqual(totals.simulations_run, 2)
        self.assertEqual(totals.days_represented, 2)
        self.assertGreater(totals.grid_kwh, 0.0)

    def test_representative_year_reflects_seasonal_pv(self):
        # A full-year representative analysis must NOT equal the June day × 365.
        cfg = small_config()
        june = main.run_simulation(cfg, ["full_smart"], start_date="2026-06-01", seed=42)
        june_farm_gen = june["scenarios"][0]["summary"]["farm_pv_generated_kwh"]
        report = run_roi_analysis(cfg, {
            "start": "2026-01-01", "end": "2026-12-31", "period_mode": "representative_month",
            "horizon": 5, "investments": ["farm_pv"], "assumptions": DEMO_ASSUMPTIONS,
        })
        annual_farm_gen = report["operational_summary"]["farm_pv_generated_kwh"]
        # Seasonal winter months pull the annual average below a pure summer projection.
        self.assertLess(annual_farm_gen, june_farm_gen * 365.0 * 0.95)
        self.assertEqual(report["meta"]["days_represented"], 365)

    def test_invalid_request_rejected(self):
        a = ROIAssumptions.from_config(small_config())
        with self.assertRaises(ROIValidationError):
            validate_request("2026-06-10", "2026-06-01", a, ["farm_pv"])
        with self.assertRaises(ROIValidationError):
            a2 = ROIAssumptions.from_config(small_config())
            a2.analysis.financial_horizon_years = 0
            validate_request("2026-01-01", "2026-12-31", a2, ["farm_pv"])


class TestCLI(unittest.TestCase):
    def test_cli_creates_output_files(self):
        from roi import __main__ as roi_cli
        with tempfile.TemporaryDirectory() as td:
            rc = roi_cli.main([
                "--config", "config.yaml", "--start", "2026-06-01", "--end", "2026-06-02",
                "--period-mode", "exact", "--horizon", "5",
                "--investments", "electric_fleet,portfolio", "--out", td,
            ])
            self.assertEqual(rc, 0)
            for name in ("roi_summary.csv", "roi_cashflows.csv",
                         "roi_assumptions.json", "roi_report.json"):
                self.assertTrue((Path(td) / name).exists(), f"{name} not written")


if __name__ == "__main__":
    unittest.main()
