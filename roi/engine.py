"""
roi.engine
==========
Top-level orchestrator for an ROI analysis.

``run_roi_analysis(config, request)`` runs the minimum set of paired operational
simulations (period_runner), builds every requested investment
(investments.py), assembles the sequential portfolio, runs the one-way
sensitivity analysis (reusing operational totals — no simulation re-runs) and
returns a JSON-safe response dict.

The engine is deliberately free of HTTP / HTML concerns; the server and the CLI
both call it.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from . import calculator as calc
from . import investments as inv
from . import period_runner as pr
from .models import OperationalTotals, ROIAssumptions
from .reliability import analyze_reliability
from .validation import ROIValidationError, collect_warnings, validate_request

PORTFOLIO_ORDER = ["diesel", "electric_fleet", "farm_pv", "tractor_roof_pv", "backup_islanding"]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _apply_request_to_config(config: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
    """Fold request-level assumption/horizon overrides into ``config['roi']``."""
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
    if request.get("scenario"):
        a["scenario"] = request["scenario"]
    if request.get("representative_days_per_month") is not None:
        a["representative_days_per_month"] = int(request["representative_days_per_month"])
    if request.get("start"):
        a["default_start_date"] = request["start"]
    if request.get("end"):
        a["default_end_date"] = request["end"]
    if request.get("discount_rate") is not None:
        roi["financial"]["discount_rate_pct"] = float(request["discount_rate"])
    return cfg


class _Variants:
    """Lazily runs & caches operational totals for the config variants needed."""

    def __init__(self, cfg: Dict[str, Any], a: ROIAssumptions,
                 start: str, end: str, farm_kwp: float, roof_w: float):
        self.cfg = cfg
        self.a = a
        self.start = start
        self.end = end
        self.farm_kwp = farm_kwp
        self.roof_w = roof_w
        self._cache: Dict[Tuple[bool, bool], OperationalTotals] = {}
        self.total_simulations = 0

    def get(self, farm_on: bool, roof_on: bool, capture_profile: bool = False) -> OperationalTotals:
        key = (farm_on, roof_on)
        if key in self._cache:
            return self._cache[key]
        farm_kwp = self.farm_kwp if farm_on else 0.0
        roof_w = self.roof_w if roof_on else 0.0
        variant_cfg = pr.make_variant_config(self.cfg, self.a.analysis.scenario, farm_kwp, roof_w)
        name = f"farm={'on' if farm_on else 'off'},roof={'on' if roof_on else 'off'}"
        totals = pr.run_period(
            variant_cfg, self.a.analysis.scenario, self.a.analysis,
            self.start, self.end, variant_name=name, capture_profile=capture_profile,
        )
        self.total_simulations += totals.simulations_run
        self._cache[key] = totals
        return totals


def _build_investments(
    variants: _Variants,
    a: ROIAssumptions,
    n_tractors: int,
    n_chargers: int,
    n_roof: int,
    farm_kwp: float,
    roof_w: float,
    reliability: Dict[str, Any],
) -> Dict[str, inv._Built]:
    """Build every standalone + marginal investment from cached totals.

    Pure w.r.t. ``a`` (given fixed operational totals + reliability), so the
    sensitivity analysis can call it repeatedly with perturbed assumptions
    without re-running any simulation.
    """
    v_full = variants.get(True, True)
    v_no_pv = variants.get(False, False)
    v_no_roof = variants.get(True, False)
    v_no_farm = variants.get(False, True)

    out: Dict[str, inv._Built] = {}
    # Standalone
    out["electric_fleet"] = inv.electric_vs_diesel(v_no_pv, a, n_tractors, n_chargers)
    out["farm_pv"] = inv.farm_pv(v_no_farm, v_full, a, farm_kwp)
    out["tractor_roof_pv"] = inv.roof_pv(v_no_roof, v_full, a, n_roof, roof_w)
    out["backup_islanding"] = inv.backup_islanding(reliability, a)

    # Marginal (sequential) stages for the portfolio
    out["_m_electric"] = out["electric_fleet"]
    out["_m_farm"] = inv.farm_pv(v_no_pv, v_no_roof, a, farm_kwp,
                                 inv_id="farm_pv", name="Fixed farm PV (marginal)")
    out["_m_roof"] = out["tractor_roof_pv"]
    out["_m_backup"] = out["backup_islanding"]
    return out


def _portfolio_from(builds: Dict[str, inv._Built], a: ROIAssumptions,
                    include_backup: bool) -> inv._Built:
    stages = [builds["_m_electric"], builds["_m_farm"], builds["_m_roof"]]
    order = ["diesel", "electric_fleet", "farm_pv", "tractor_roof_pv"]
    if include_backup:
        stages.append(builds["_m_backup"])
        order.append("backup_islanding")
    return inv.portfolio(stages, a, order)


def _reliability_for(a: ROIAssumptions, cfg: Dict[str, Any],
                     profile: List[Dict[str, Any]], has_pv: bool) -> Dict[str, Any]:
    n_tr = len(cfg.get("tractors", {}).get("fleet", []))
    battery = float(cfg.get("tractors", {}).get("model", {}).get("battery_capacity_kwh", 0.0))
    v2l_kw = float(cfg.get("v2l", {}).get("max_discharge_kw", 6.6))
    return analyze_reliability(profile, a.outages, n_tr, battery, v2l_kw, has_pv)


def run_roi_analysis(config: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
    """Run a full ROI analysis and return a JSON-safe response dict.

    Raises :class:`ROIValidationError` on invalid input (server → HTTP 400).
    ``config`` should already carry any dashboard simulation overrides
    (grid/fleet/PV sliders); ``request`` carries ROI-level fields.
    """
    cfg = _apply_request_to_config(config, request)
    a = ROIAssumptions.from_config(cfg)

    start = request.get("start") or a.analysis.default_start_date
    end = request.get("end") or a.analysis.default_end_date
    selected: List[str] = list(request.get("investments") or
                               ["electric_fleet", "farm_pv", "tractor_roof_pv", "portfolio"])

    validate_request(start, end, a, selected)

    farm_kwp = float(cfg.get("pv", {}).get("farm_fixed_peak_kw", 0.0))
    roof_w = float(cfg.get("tractor_pv", {}).get("panel_peak_w", 0.0))
    n_tractors = len(cfg.get("tractors", {}).get("fleet", []))
    n_chargers = len(cfg.get("charging", {}).get("stations", []))
    n_roof = a.roof_pv.equipped_tractors or sum(
        1 for t in cfg.get("tractors", {}).get("fleet", []) if t.get("has_pv_roof", False)
    ) or n_tractors

    start_d = datetime.fromisoformat(str(start)[:10]).date()
    end_d = datetime.fromisoformat(str(end)[:10]).date()
    period_days = (end_d - start_d).days + 1
    method, _ = pr.resolve_method(start_d, end_d, a.analysis)

    variants = _Variants(cfg, a, start, end, farm_kwp, roof_w)
    v_full = variants.get(True, True, capture_profile=True)   # operations + reliability profile
    reliability = _reliability_for(a, cfg, v_full.profile, has_pv=(farm_kwp > 0 or roof_w > 0))

    builds = _build_investments(variants, a, n_tractors, n_chargers, n_roof,
                                farm_kwp, roof_w, reliability)

    include_backup = a.outages.enabled or "backup_islanding" in selected
    portfolio_build = _portfolio_from(builds, a, include_backup)

    # ── Assemble response ──────────────────────────────────────────────────
    investments_out = []
    for inv_id in ("electric_fleet", "farm_pv", "tractor_roof_pv", "backup_islanding"):
        if inv_id in selected:
            investments_out.append(builds[inv_id].result.to_dict())

    portfolio_out: Dict[str, Any] = {}
    if "portfolio" in selected:
        portfolio_out = portfolio_build.result.to_dict()

    warnings = collect_warnings(a, selected, period_days, farm_kwp, roof_w)
    if v_full.tasks_missed > 0.5:
        warnings.append(
            f"Baseline and candidate service levels differ (~{v_full.tasks_missed:.1f} "
            "tasks/yr uncompleted) — compare cost per completed task, not totals."
        )

    sensitivity = {}
    if a.sensitivity.enabled:
        sensitivity = _run_sensitivity(
            cfg, a, variants, reliability, n_tractors, n_chargers, n_roof,
            farm_kwp, roof_w, v_full.profile, include_backup,
        )

    op = v_full.to_dict()
    op["scenario"] = a.analysis.scenario

    return _json_safe({
        "meta": {
            "start_date": str(start),
            "end_date": str(end),
            "period_mode": method,
            "requested_period_mode": a.analysis.period_mode,
            "simulations_run": variants.total_simulations,
            "days_represented": v_full.days_represented,
            "period_days": period_days,
            "annualisation": "totals scaled to a 365-day year",
            "financial_horizon_years": a.analysis.financial_horizon_years,
            "discount_rate_pct": a.financial.discount_rate_pct,
            "currency": a.analysis.currency,
            "seed": a.analysis.base_seed,
            "seasonal_pv_model": (
                "monthly factor applied (static PV backend)"
                if str(cfg.get("prediction", {}).get("pv", {}).get("backend", "static")) == "static"
                else "seasonal predictor backend"
            ),
        },
        "assumptions": a.to_dict(),
        "operational_summary": op,
        "reliability": reliability,
        "investments": investments_out,
        "portfolio": portfolio_out,
        "sensitivity": sensitivity,
        "warnings": warnings,
    })


# ─────────────────────────────────────────────────────────────────────────────
# One-way sensitivity (reuses operational totals — no simulation re-runs)
# ─────────────────────────────────────────────────────────────────────────────

def _perturb(a: ROIAssumptions, field_path: str, factor: float) -> ROIAssumptions:
    """Return a copy of the assumptions with one field scaled by ``factor``."""
    a2 = copy.deepcopy(a)
    group, attr = field_path.split(".")
    grp = getattr(a2, group)
    setattr(grp, attr, getattr(grp, attr) * factor)
    return a2


def _portfolio_metrics(cfg, a, variants, reliability, counts, profile, include_backup):
    n_tractors, n_chargers, n_roof, farm_kwp, roof_w = counts
    builds = _build_investments(variants, a, n_tractors, n_chargers, n_roof,
                                farm_kwp, roof_w, reliability)
    pb = _portfolio_from(builds, a, include_backup)
    m = pb.result.metrics
    return {
        "npv_eur": m.get("npv_eur"),
        "simple_payback_years": m.get("simple_payback_years"),
        "roi_pct": m.get("roi_pct"),
    }


def _run_sensitivity(cfg, a, variants, reliability, n_tractors, n_chargers, n_roof,
                     farm_kwp, roof_w, profile, include_backup) -> Dict[str, Any]:
    counts = (n_tractors, n_chargers, n_roof, farm_kwp, roof_w)
    var = a.sensitivity.variation_pct / 100.0
    low_f, high_f = 1.0 - var, 1.0 + var

    base = _portfolio_metrics(cfg, a, variants, reliability, counts, profile, include_backup)

    params = [
        ("diesel_price", "diesel.fuel_price_eur_per_litre", False),
        ("electricity_price", "financial.electricity_price_escalation_pct", True),
        ("farm_pv_capex", "farm_pv.capex_eur_per_kwp", False),
        ("electric_tractor_capex", "electric_fleet.electric_tractor_purchase_eur", False),
        ("discount_rate", "financial.discount_rate_pct", True),
        ("outage_frequency", "outages.frequency_per_year", True),
        ("value_of_lost_load", "outages.value_of_lost_load_eur_per_kwh", True),
    ]

    rows = []
    for label, path, needs_reliability in params:
        results = {}
        for tag, factor in (("low", low_f), ("high", high_f)):
            a2 = _perturb(a, path, factor)
            rel2 = reliability
            if needs_reliability and path.startswith("outages."):
                rel2 = _reliability_for(a2, cfg, profile, has_pv=(farm_kwp > 0 or roof_w > 0))
            results[tag] = _portfolio_metrics(cfg, a2, variants, rel2, counts, profile, include_backup)
        rows.append({
            "parameter": label,
            "field": path,
            "low": results["low"],
            "high": results["high"],
        })

    return {"variation_pct": a.sensitivity.variation_pct, "base": base, "parameters": rows}


# ─────────────────────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Recursively replace NaN / Infinity with None so JSON stays valid."""
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
