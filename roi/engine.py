"""
roi.engine
==========
Top-level orchestrator for a connected Operations → ROI analysis.

``run_roi_analysis(config, request)`` takes the **saved Operations run** (its
merged config + selected scenarios, supplied by the server as
``request["operations"]``) and, for *every* scenario the operator selected on the
Operations page:

* runs the minimum set of paired operational simulations over the ROI period,
* produces a long-term operational comparison row,
* builds each requested investment (labelled by its source scenario),
* assembles a sequential, no-double-count portfolio.

Operational values (fleet, PV, chargers, tasks, seed, tariffs, scenarios) always
come from the saved run — never from independently editable ROI fields.  The ROI
request only carries the period, the financial horizon and the financial /
investment assumptions.
"""

from __future__ import annotations

import copy
import hashlib
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from . import calculator as calc
from . import investments as inv
from . import period_runner as pr
from .models import OperationalTotals, ROIAssumptions
from .reliability import analyze_reliability
from .validation import ROIValidationError, collect_warnings, validate_request

# Short display names used when labelling an investment by its source scenario.
_INV_SHORT = {
    "electric_fleet": "Electric fleet",
    "farm_pv": "Fixed farm PV",
    "tractor_roof_pv": "Tractor-roof PV",
    "backup_islanding": "Backup / islanding",
}


# ─────────────────────────────────────────────────────────────────────────────
# Request → config folding
# ─────────────────────────────────────────────────────────────────────────────

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _apply_request_to_config(config: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
    """Fold ROI-level (financial/period) overrides into ``config['roi']``.

    Only financial and period fields are honoured here — operational values are
    never taken from the request.
    """
    cfg = copy.deepcopy(config)
    roi = cfg.setdefault("roi", {})
    roi.setdefault("analysis", {})
    roi.setdefault("financial", {})

    if request.get("assumptions"):
        cfg["roi"] = _deep_merge(cfg["roi"], request["assumptions"])
        roi = cfg["roi"]
        roi.setdefault("analysis", {})
        roi.setdefault("financial", {})

    a = roi["analysis"]
    if request.get("period_mode"):
        a["period_mode"] = request["period_mode"]
    if request.get("horizon") is not None:
        a["financial_horizon_years"] = int(request["horizon"])
    if request.get("currency"):
        a["currency"] = request["currency"]
    if request.get("representative_days_per_month") is not None:
        a["representative_days_per_month"] = int(request["representative_days_per_month"])
    if request.get("start"):
        a["default_start_date"] = request["start"]
    if request.get("end"):
        a["default_end_date"] = request["end"]
    if request.get("discount_rate") is not None:
        roi["financial"]["discount_rate_pct"] = float(request["discount_rate"])
    return cfg


def operations_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build an ``operations`` block from a config (used by the CLI / fallback).

    Uses every scenario defined in the config as the selected set.
    """
    scenarios = []
    for s in cfg.get("scenarios", []):
        scenarios.append({
            "id": s["name"], "name": s["name"], "label": s["name"].replace("_", " ").title(),
            "charging_strategy": s.get("charging_strategy", "smart"),
            "tractor_pv_enabled": bool(s.get("tractor_pv_enabled", False)),
            "load_shedding": bool(s.get("load_shedding", False)),
            "use_marl": bool(s.get("use_marl", False)),
        })
    return {"run_id": None, "timestamp": None, "scenarios": scenarios,
            "params": _params_from_config(cfg), "one_day": {}}


def _params_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    stations = cfg.get("charging", {}).get("stations", [])
    fleet = cfg.get("tractors", {}).get("fleet", [])
    return {
        "grid_kw": float(cfg.get("grid", {}).get("max_power_kw", 0.0)),
        "farm_pv_kw": float(cfg.get("pv", {}).get("farm_fixed_peak_kw", 0.0)),
        "panel_w": float(cfg.get("tractor_pv", {}).get("panel_peak_w", 0.0)),
        "tractors": len(fleet),
        "chargers": len(stations),
        "charger_kw": float(stations[0]["max_power_kw"]) if stations else 0.0,
        "battery_kwh": float(cfg.get("tractors", {}).get("model", {}).get("battery_capacity_kwh", 0.0)),
        "num_tasks": int(cfg.get("task_generation", {}).get("num_tasks", 0)),
        "seed": int(cfg.get("task_generation", {}).get("seed", 42)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario variant runner (caches the paired simulations)
# ─────────────────────────────────────────────────────────────────────────────

class _ScenarioContext:
    """Runs & caches the operational variants for one Operations scenario."""

    def __init__(self, cfg, a, start, end, farm_kwp, roof_w, scenario_def, sim_counter):
        self.cfg = cfg
        self.a = a
        self.start = start
        self.end = end
        self.farm_kwp = farm_kwp
        self.roof_w = roof_w
        self.scn = scenario_def
        self.name = scenario_def.get("id") or scenario_def.get("name")
        self.roof_capable = bool(scenario_def.get("tractor_pv_enabled")) and roof_w > 0
        self._cache: Dict[Tuple[bool, bool], OperationalTotals] = {}
        self._sim_counter = sim_counter

    def get(self, farm_on: bool, roof_on: bool, capture_profile: bool = False) -> OperationalTotals:
        roof_on = bool(roof_on and self.roof_capable)
        key = (bool(farm_on), roof_on)
        if key in self._cache:
            return self._cache[key]
        variant_cfg = pr.make_variant_config(
            self.cfg, self.scn,
            self.farm_kwp if farm_on else 0.0,
            self.roof_w if roof_on else 0.0,
        )
        totals = pr.run_period(
            variant_cfg, self.name, self.a.analysis, self.start, self.end,
            variant_name=f"{self.name}:farm={farm_on},roof={roof_on}",
            capture_profile=capture_profile,
        )
        self._sim_counter[0] += totals.simulations_run
        self._cache[key] = totals
        return totals


def _reliability_for(a: ROIAssumptions, cfg: Dict[str, Any],
                     profile: List[Dict[str, Any]], has_pv: bool) -> Dict[str, Any]:
    n_tr = len(cfg.get("tractors", {}).get("fleet", []))
    battery = float(cfg.get("tractors", {}).get("model", {}).get("battery_capacity_kwh", 0.0))
    v2l_kw = float(cfg.get("v2l", {}).get("max_discharge_kw", 6.6))
    return analyze_reliability(profile, a.outages, n_tr, battery, v2l_kw, has_pv)


def _scenario_builds(ctx, a, n_tractors, n_chargers, n_roof, farm_kwp, roof_w,
                     reliability) -> Dict[str, inv._Built]:
    """Build standalone + marginal investments for one scenario from cached totals."""
    v_full = ctx.get(True, True)
    v_no_pv = ctx.get(False, False)
    v_farm_only = ctx.get(True, False)        # farm on, roof off
    v_no_farm = ctx.get(False, True)          # farm off, roof as scenario

    builds: Dict[str, inv._Built] = {}
    builds["electric_fleet"] = inv.electric_vs_diesel(v_no_pv, a, n_tractors, n_chargers)
    builds["_m_electric"] = builds["electric_fleet"]

    if farm_kwp > 0:
        builds["farm_pv"] = inv.farm_pv(v_no_farm, v_full, a, farm_kwp)
        builds["_m_farm"] = inv.farm_pv(v_no_pv, v_farm_only, a, farm_kwp,
                                        name="Fixed farm PV (marginal)")
    if ctx.roof_capable:
        builds["tractor_roof_pv"] = inv.roof_pv(v_farm_only, v_full, a, n_roof, roof_w)
        builds["_m_roof"] = builds["tractor_roof_pv"]

    builds["backup_islanding"] = inv.backup_islanding(reliability, a)
    builds["_m_backup"] = builds["backup_islanding"]
    return builds


def _scenario_portfolio(builds: Dict[str, inv._Built], a: ROIAssumptions,
                        include_backup: bool) -> inv._Built:
    stages = [builds["_m_electric"]]
    order = ["diesel", "electric_fleet"]
    if "_m_farm" in builds:
        stages.append(builds["_m_farm"])
        order.append("farm_pv")
    if "_m_roof" in builds:
        stages.append(builds["_m_roof"])
        order.append("tractor_roof_pv")
    if include_backup:
        stages.append(builds["_m_backup"])
        order.append("backup_islanding")
    return inv.portfolio(stages, a, order)


# ─────────────────────────────────────────────────────────────────────────────
# Result assembly helpers
# ─────────────────────────────────────────────────────────────────────────────

def _labelled(build: inv._Built, scn: Dict[str, Any]) -> Dict[str, Any]:
    d = build.result.to_dict()
    label = scn.get("label") or scn.get("name")
    short = _INV_SHORT.get(build.result.id, build.result.name)
    d["name"] = f"{short} — {label}"
    d["scenario_id"] = scn.get("id") or scn.get("name")
    d["scenario_name"] = label
    d["complete"] = build.complete
    return d


def _long_term_row(scn: Dict[str, Any], t: OperationalTotals,
                   reliability: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    completed = t.tasks_completed
    period_factor = (t.days_represented / 365.0) if t.days_represented else 0.0
    row = {
        "scenario_id": scn.get("id") or scn.get("name"),
        "scenario_name": scn.get("label") or scn.get("name"),
        "days_represented": t.days_represented,
        "simulations_run": t.simulations_run,
        "grid_kwh_annual": round(t.grid_kwh, 1),
        "grid_kwh_period": round(t.grid_kwh * period_factor, 1),
        "grid_cost_annual": round(t.grid_cost_eur, 2),
        "grid_cost_period": round(t.grid_cost_eur * period_factor, 2),
        "farm_pv_generated_kwh": round(t.farm_pv_generated_kwh, 1),
        "tractor_pv_generated_kwh": round(t.tractor_pv_generated_kwh, 1),
        "pv_used_kwh": round(t.pv_used_kwh, 1),
        "tasks_completed": round(completed, 1),
        "tasks_missed": round(t.tasks_missed, 1),
        "total_tasks": round(t.total_tasks, 1),
        "task_completion_pct": round(100.0 * completed / t.total_tasks, 1) if t.total_tasks else None,
        "cost_per_completed_task": round(t.grid_cost_eur / completed, 3) if completed else None,
        "grid_kwh_per_completed_task": round(t.grid_kwh / completed, 2) if completed else None,
        "operating_hours": round(t.operating_hours, 1),
        "transit_distance_km": round(t.transit_distance_km, 1),
        "charger_energy_kwh": round(t.tractor_charge_input_kwh, 1),
        "peak_grid_kw": round(t.peak_grid_kw, 2),
        "downtime_pct": round(t.downtime_pct, 1),
    }
    if reliability is not None:
        row["reliability"] = {
            "expected_outage_hours_per_year": reliability.get("expected_outage_hours_per_year"),
            "expected_unserved_energy_kwh": reliability.get("expected_unserved_energy_kwh"),
            "critical_load_coverage_pct": reliability.get("critical_load_coverage_pct"),
            "avoided_outage_cost_eur": reliability.get("avoided_outage_cost_eur"),
        }
    return row


def _best_completion_scenario(ops: Dict[str, Any], scenarios: List[Dict[str, Any]]) -> str:
    """Scenario id with the best one-day task completion (fallback: first)."""
    one_day = ops.get("one_day") or {}
    best_id, best_val = None, -1.0
    for scn in scenarios:
        sid = scn.get("id") or scn.get("name")
        summary = one_day.get(sid) or {}
        val = float(summary.get("completed_tasks", -1))
        if val > best_val:
            best_val, best_id = val, sid
    return best_id or (scenarios[0].get("id") or scenarios[0].get("name"))


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def config_hash(cfg: Dict[str, Any]) -> str:
    """Stable short hash of the operational parts of a config (for run identity)."""
    import json
    keys = ("grid", "pv", "tractor_pv", "tractors", "charging", "energy_consumers",
            "tariffs", "task_generation", "scenarios", "v2l", "marl", "prediction")
    subset = {k: cfg.get(k) for k in keys if k in cfg}
    blob = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def run_roi_analysis(config: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
    """Run a full connected ROI analysis and return a JSON-safe response dict."""
    cfg = _apply_request_to_config(config, request)
    a = ROIAssumptions.from_config(cfg)
    demo = bool(cfg.get("roi", {}).get("demonstration_assumptions", False))

    ops = request.get("operations") or operations_from_config(cfg)
    scenarios: List[Dict[str, Any]] = ops.get("scenarios") or []
    if not scenarios:
        raise ROIValidationError("No Operations scenarios to analyse — run Operations first.")

    start = request.get("start") or a.analysis.default_start_date
    end = request.get("end") or a.analysis.default_end_date
    selected: List[str] = list(request.get("investments") or
                               ["electric_fleet", "farm_pv", "tractor_roof_pv", "portfolio"])
    validate_request(start, end, a, selected)

    farm_kwp = float(cfg.get("pv", {}).get("farm_fixed_peak_kw", 0.0))
    roof_w = float(cfg.get("tractor_pv", {}).get("panel_peak_w", 0.0))
    fleet = cfg.get("tractors", {}).get("fleet", [])
    stations = cfg.get("charging", {}).get("stations", [])
    n_tractors = len(fleet)
    n_chargers = len(stations)
    n_roof = a.roof_pv.equipped_tractors or sum(
        1 for t in fleet if t.get("has_pv_roof", False)) or n_tractors

    start_d = datetime.fromisoformat(str(start)[:10]).date()
    end_d = datetime.fromisoformat(str(end)[:10]).date()
    period_days = (end_d - start_d).days + 1
    method, _ = pr.resolve_method(start_d, end_d, a.analysis)
    include_backup = a.outages.enabled or "backup_islanding" in selected

    sim_counter = [0]
    long_term: List[Dict[str, Any]] = []
    investments_out: List[Dict[str, Any]] = []
    portfolios_out: List[Dict[str, Any]] = []
    reliability_map: Dict[str, Any] = {}
    contexts: Dict[str, _ScenarioContext] = {}
    builds_by_scn: Dict[str, Dict[str, inv._Built]] = {}

    for scn in scenarios:
        sid = scn.get("id") or scn.get("name")
        ctx = _ScenarioContext(cfg, a, start, end, farm_kwp, roof_w, scn, sim_counter)
        contexts[sid] = ctx

        v_full = ctx.get(True, True, capture_profile=True)
        rel = _reliability_for(a, cfg, v_full.profile,
                               has_pv=(farm_kwp > 0 or ctx.roof_capable))
        reliability_map[sid] = rel
        long_term.append(_long_term_row(scn, v_full, rel if a.outages.enabled else None))

        builds = _scenario_builds(ctx, a, n_tractors, n_chargers, n_roof,
                                  farm_kwp, roof_w, rel)
        builds_by_scn[sid] = builds

        for inv_id in ("electric_fleet", "farm_pv", "tractor_roof_pv", "backup_islanding"):
            if inv_id in selected and inv_id in builds:
                investments_out.append(_labelled(builds[inv_id], scn))

        if "portfolio" in selected:
            pf = _scenario_portfolio(builds, a, include_backup)
            portfolios_out.append(_labelled(pf, scn))

    default_scn = _best_completion_scenario(ops, scenarios)

    warnings = collect_warnings(a, selected, period_days, farm_kwp, roof_w)
    if demo:
        warnings.insert(0, "Demonstration assumptions — replace with supplier quotations "
                           "and local operating data before making an investment decision.")

    sensitivity: Dict[str, Any] = {}
    if a.sensitivity.enabled and "portfolio" in selected and default_scn in builds_by_scn:
        sensitivity = _run_sensitivity(
            cfg, a, contexts[default_scn], reliability_map[default_scn],
            n_tractors, n_chargers, n_roof, farm_kwp, roof_w, include_backup, default_scn)

    params = ops.get("params") or _params_from_config(cfg)
    operations_basis = {
        "run_id": ops.get("run_id"),
        "timestamp": ops.get("timestamp"),
        "grid_kw": params.get("grid_kw"),
        "farm_pv_kwp": params.get("farm_pv_kw"),
        "roof_panel_w": params.get("panel_w"),
        "n_tractors": params.get("tractors"),
        "n_chargers": params.get("chargers"),
        "charger_kw": params.get("charger_kw"),
        "battery_kwh": params.get("battery_kwh"),
        "num_tasks": params.get("num_tasks"),
        "seed": params.get("seed"),
        "scenarios": [{"id": s.get("id") or s.get("name"),
                       "name": s.get("label") or s.get("name")} for s in scenarios],
        "one_day": ops.get("one_day") or {},
    }

    export_meta = {
        "operations_run_id": ops.get("run_id"),
        "operations_timestamp": ops.get("timestamp"),
        "operations_scenarios": [s.get("label") or s.get("name") for s in scenarios],
        "roi_period": {"start": str(start), "end": str(end), "method": method,
                       "days_represented": long_term[0]["days_represented"] if long_term else 0},
        "financial_horizon_years": a.analysis.financial_horizon_years,
        "assumption_profile": "demonstration" if demo else "custom",
        "demonstration_assumptions": demo,
    }

    return _json_safe({
        "meta": {
            "operations_run_id": ops.get("run_id"),
            "operations_timestamp": ops.get("timestamp"),
            "start_date": str(start),
            "end_date": str(end),
            "period_mode": method,
            "requested_period_mode": a.analysis.period_mode,
            "simulations_run": sim_counter[0],
            "days_represented": long_term[0]["days_represented"] if long_term else 0,
            "period_days": period_days,
            "annualisation": "totals scaled to a 365-day year",
            "financial_horizon_years": a.analysis.financial_horizon_years,
            "discount_rate_pct": a.financial.discount_rate_pct,
            "currency": a.analysis.currency,
            "seed": a.analysis.base_seed,
            "demonstration_assumptions": demo,
            "scenarios": [s.get("label") or s.get("name") for s in scenarios],
            "default_portfolio_scenario": default_scn,
            "seasonal_pv_model": (
                "monthly factor applied (static PV backend)"
                if str(cfg.get("prediction", {}).get("pv", {}).get("backend", "static")) == "static"
                else "seasonal predictor backend"),
        },
        "assumptions": a.to_dict(),
        "operations_basis": operations_basis,
        "long_term": long_term,
        "investments": investments_out,
        "portfolios": portfolios_out,
        "default_portfolio_scenario": default_scn,
        "reliability": reliability_map,
        "sensitivity": sensitivity,
        "warnings": warnings,
        "export_meta": export_meta,
    })


# ─────────────────────────────────────────────────────────────────────────────
# One-way sensitivity (reuses cached operational totals — no simulation re-runs)
# ─────────────────────────────────────────────────────────────────────────────

def _perturb(a: ROIAssumptions, field_path: str, factor: float) -> ROIAssumptions:
    a2 = copy.deepcopy(a)
    group, attr = field_path.split(".")
    grp = getattr(a2, group)
    setattr(grp, attr, getattr(grp, attr) * factor)
    return a2


def _portfolio_metrics(a, ctx, reliability, counts, include_backup):
    n_tractors, n_chargers, n_roof, farm_kwp, roof_w = counts
    builds = _scenario_builds(ctx, a, n_tractors, n_chargers, n_roof,
                              farm_kwp, roof_w, reliability)
    pb = _scenario_portfolio(builds, a, include_backup)
    m = pb.result.metrics
    return {"npv_eur": m.get("npv_eur"), "simple_payback_years": m.get("simple_payback_years"),
            "roi_pct": m.get("roi_pct")}


def _run_sensitivity(cfg, a, ctx, reliability, n_tractors, n_chargers, n_roof,
                     farm_kwp, roof_w, include_backup, scenario_id) -> Dict[str, Any]:
    counts = (n_tractors, n_chargers, n_roof, farm_kwp, roof_w)
    var = a.sensitivity.variation_pct / 100.0
    low_f, high_f = 1.0 - var, 1.0 + var
    base = _portfolio_metrics(a, ctx, reliability, counts, include_backup)

    params = [
        ("diesel_price", "diesel.fuel_price_eur_per_litre", False),
        ("electricity_price", "financial.electricity_price_escalation_pct", False),
        ("farm_pv_capex", "farm_pv.capex_eur_per_kwp", False),
        ("electric_tractor_capex", "electric_fleet.electric_tractor_purchase_eur", False),
        ("discount_rate", "financial.discount_rate_pct", False),
        ("outage_frequency", "outages.frequency_per_year", True),
        ("value_of_lost_load", "outages.value_of_lost_load_eur_per_kwh", True),
    ]
    rows = []
    for label, path, needs_rel in params:
        results = {}
        for tag, factor in (("low", low_f), ("high", high_f)):
            a2 = _perturb(a, path, factor)
            rel2 = reliability
            if needs_rel:
                rel2 = _reliability_for(a2, cfg, ctx.get(True, True).profile,
                                        has_pv=(farm_kwp > 0 or ctx.roof_capable))
            results[tag] = _portfolio_metrics(a2, ctx, rel2, counts, include_backup)
        rows.append({"parameter": label, "field": path,
                     "low": results["low"], "high": results["high"]})
    return {"variation_pct": a.sensitivity.variation_pct, "scenario_id": scenario_id,
            "base": base, "parameters": rows}


def _json_safe(obj: Any) -> Any:
    import math
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj
