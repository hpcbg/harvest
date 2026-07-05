"""
roi.period_runner
=================
Runs / aggregates the pilot6 simulator across an operating period and returns
annualised :class:`OperationalTotals` for one configuration variant.

It never re-implements the simulation loop — it calls
``main.run_simulation`` (the reusable API) once per simulated day and
aggregates the structured results.

Period modes (section 5 of the spec)
-------------------------------------
* ``exact``                – one deterministic run per calendar day in range.
* ``representative_month`` – N representative days per covered month, weighted
  by the number of applicable calendar days.
* ``auto``                 – exact for periods ≤ ``auto_exact_max_days`` days,
  otherwise representative-month.

Seasonality
-----------
The default ``static`` PV backend has no seasonal variation, so a naive yearly
analysis would just repeat the June day 365×.  To make a one-year analysis
reflect real seasonal PV differences, this runner multiplies the *generation*
of each simulated day by a documented monthly factor (``_seasonal_factor``)
when the active PV backend is ``static``.  Installed capacity (and therefore
CAPEX) is unaffected — only monthly yield changes.  When a genuinely seasonal
predictor backend (``stub`` / ``openmeteo`` / ``nn``) is active the factor is
1.0, so seasonality is not double-counted.

Deterministic seed rule
------------------------
``daily_seed = base_seed + YYYYMMDD`` (e.g. base 42 on 2026-06-01 → 42+20260601).
"""

from __future__ import annotations

import calendar
import copy
import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import main as sim_main

from .models import AnalysisConfig, OperationalTotals

STEP_MIN = 15  # matches config.simulation.time_step_minutes; used for kWh scaling


# ─────────────────────────────────────────────────────────────────────────────
# Period resolution
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(value: Any) -> date:
    from datetime import datetime
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def resolve_method(start: date, end: date, analysis: AnalysisConfig) -> Tuple[str, int]:
    """Return ``(method, calendar_days)`` for the requested period."""
    n_days = (end - start).days + 1
    mode = analysis.period_mode
    if mode == "auto":
        method = "exact" if n_days <= analysis.auto_exact_max_days else "representative_month"
    else:
        method = mode
    return method, n_days


def _seasonal_factor(month: int, apply: bool) -> float:
    """Monthly PV generation factor, peak in June, trough in December.

    Ranges ~0.25 (mid-winter) to 1.0 (mid-summer) — a ~4× swing consistent with
    the seasonal behaviour documented for the WeatherStub backend.  Returns 1.0
    when seasonal scaling should not be applied.
    """
    if not apply:
        return 1.0
    # cos peaks at month 6 (June); shift so June→1.0, December→~0.25.
    seasonal = 0.5 + 0.5 * math.cos(2.0 * math.pi * (month - 6) / 12.0)
    return 0.25 + 0.75 * seasonal


def _representative_days(start: date, end: date, method: str,
                         days_per_month: int) -> List[Tuple[date, float]]:
    """List of ``(day, weight)`` where weight is calendar days represented."""
    if method == "exact":
        days: List[Tuple[date, float]] = []
        cur = start
        while cur <= end:
            days.append((cur, 1.0))
            cur += timedelta(days=1)
        return days

    # representative_month
    out: List[Tuple[date, float]] = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        y, m = cur.year, cur.month
        month_first = date(y, m, 1)
        month_last = date(y, m, calendar.monthrange(y, m)[1])
        lo = max(month_first, start)
        hi = min(month_last, end)
        covered = (hi - lo).days + 1
        if covered > 0:
            n = max(1, days_per_month)
            # Spread representative days roughly evenly through the covered span.
            weight = covered / n
            for k in range(n):
                offset = int((k + 0.5) * covered / n)
                rep_day = lo + timedelta(days=min(offset, covered - 1))
                out.append((rep_day, weight))
        # advance to first of next month
        if m == 12:
            cur = date(y + 1, 1, 1)
        else:
            cur = date(y, m + 1, 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Variant construction
# ─────────────────────────────────────────────────────────────────────────────

def make_variant_config(
    base_config: Dict[str, Any],
    scenario: str,
    farm_pv_kwp: Optional[float],
    roof_panel_w: Optional[float],
) -> Dict[str, Any]:
    """Return a deep-copied config with a single ROI scenario and PV toggles.

    ``farm_pv_kwp`` / ``roof_panel_w`` set the installed capacities for this
    variant (use 0 to switch an asset off; ``None`` to keep the config value).
    All other parameters — seed, fleet, chargers, tariffs — are untouched, so
    paired baseline/candidate runs differ only in the asset under study.
    """
    cfg = copy.deepcopy(base_config)

    if farm_pv_kwp is not None:
        cfg["pv"]["farm_fixed_peak_kw"] = float(farm_pv_kwp)
    if roof_panel_w is not None:
        cfg.setdefault("tractor_pv", {})["panel_peak_w"] = float(roof_panel_w)

    roof_on = (roof_panel_w is None and cfg.get("tractor_pv", {}).get("panel_peak_w", 0) > 0) \
        or (roof_panel_w is not None and roof_panel_w > 0)

    # Find the requested scenario definition to inherit its charging strategy.
    src = None
    for s in cfg.get("scenarios", []):
        if s.get("name") == scenario:
            src = s
            break
    if src is None:
        src = {"name": scenario, "charging_strategy": "smart"}

    variant_scenario = {
        "name": src.get("name", scenario),
        "charging_strategy": src.get("charging_strategy", "smart"),
        "tractor_pv_enabled": bool(roof_on),
        "load_shedding": bool(src.get("load_shedding", False)),
        "use_marl": bool(src.get("use_marl", False)),
    }
    cfg["scenarios"] = [variant_scenario]
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Period run
# ─────────────────────────────────────────────────────────────────────────────

def _pv_backend_is_static(config: Dict[str, Any]) -> bool:
    return str(config.get("prediction", {}).get("pv", {}).get("backend", "static")) == "static"


def run_period(
    variant_config: Dict[str, Any],
    scenario: str,
    analysis: AnalysisConfig,
    start: Any,
    end: Any,
    variant_name: str = "",
    capture_profile: bool = False,
) -> OperationalTotals:
    """Aggregate the simulator over the period and return annualised totals."""
    start_d = _parse_date(start)
    end_d = _parse_date(end)
    method, n_days = resolve_method(start_d, end_d, analysis)
    apply_seasonal = _pv_backend_is_static(variant_config)

    base_farm_pv = float(variant_config.get("pv", {}).get("farm_fixed_peak_kw", 0.0))
    base_roof_w = float(variant_config.get("tractor_pv", {}).get("panel_peak_w", 0.0))

    rep_days = _representative_days(start_d, end_d, method, analysis.representative_days_per_month)

    totals = OperationalTotals(variant=variant_name)
    days_represented = 0.0
    sims = 0
    dt_h = STEP_MIN / 60.0

    profiles: List[Tuple[float, List[Dict[str, Any]]]] = []
    factors_used: List[float] = []

    for day, weight in rep_days:
        factor = _seasonal_factor(day.month, apply_seasonal)
        factors_used.append(factor)
        day_cfg = copy.deepcopy(variant_config)
        day_cfg["pv"]["farm_fixed_peak_kw"] = base_farm_pv * factor
        day_cfg.setdefault("tractor_pv", {})["panel_peak_w"] = base_roof_w * factor

        daily_seed = analysis.base_seed + int(day.strftime("%Y%m%d"))
        result = sim_main.run_simulation(
            day_cfg,
            selected_scenarios=[scenario],
            start_date=day,
            seed=daily_seed,
        )
        scen = result["scenarios"][0]
        s = scen["summary"]
        sims += 1
        days_represented += weight

        totals.grid_kwh += s["total_grid_kwh"] * weight
        totals.grid_farm_kwh += s.get("grid_farm_kwh", 0.0) * weight
        totals.grid_tractor_kwh += s.get("grid_tractor_kwh", 0.0) * weight
        totals.grid_cost_eur += s["total_cost_eur"] * weight
        totals.tractor_charge_cost_eur += s.get("tractor_charge_cost_eur", 0.0) * weight
        totals.tractor_charge_input_kwh += s.get("tractor_charge_input_kwh", 0.0) * weight
        totals.farm_pv_generated_kwh += s.get("farm_pv_generated_kwh", 0.0) * weight
        totals.tractor_pv_generated_kwh += s.get("tractor_pv_generated_kwh", 0.0) * weight
        totals.pv_used_kwh += s.get("total_pv_used_kwh", 0.0) * weight
        totals.farm_pv_used_kwh += s.get("farm_pv_used_kwh", 0.0) * weight
        totals.tractor_pv_used_kwh += s.get("tractor_pv_used_kwh", 0.0) * weight
        totals.transit_distance_km += s.get("fleet_transit_distance_km", 0.0) * weight
        totals.work_distance_km += s.get("fleet_work_distance_km", 0.0) * weight
        totals.operating_hours += s.get("fleet_operating_hours", 0.0) * weight
        totals.pto_hours += s.get("fleet_pto_hours", 0.0) * weight
        totals.idle_hours += s.get("fleet_idle_hours", 0.0) * weight
        totals.tasks_completed += s.get("completed_tasks", 0) * weight
        totals.tasks_missed += s.get("missed_tasks", 0) * weight
        totals.total_tasks += s.get("total_tasks", 0) * weight

        # Exported / curtailed PV: generation not consumed on-site.
        gen = s.get("farm_pv_generated_kwh", 0.0) + s.get("tractor_pv_generated_kwh", 0.0)
        unused = max(0.0, gen - s.get("total_pv_used_kwh", 0.0))
        totals.curtailed_pv_kwh += unused * weight

        if capture_profile:
            profiles.append((factor, scen["timeseries"]))

    # Annualise to a 365-day year for the financial model.
    scale = (365.0 / days_represented) if days_represented > 0 else 0.0
    for attr in ("grid_kwh", "grid_farm_kwh", "grid_tractor_kwh", "grid_cost_eur",
                 "tractor_charge_cost_eur", "tractor_charge_input_kwh",
                 "farm_pv_generated_kwh", "tractor_pv_generated_kwh", "pv_used_kwh",
                 "farm_pv_used_kwh", "tractor_pv_used_kwh", "exported_pv_kwh",
                 "curtailed_pv_kwh", "transit_distance_km", "work_distance_km",
                 "operating_hours", "pto_hours", "idle_hours", "tasks_completed",
                 "tasks_missed", "total_tasks"):
        setattr(totals, attr, getattr(totals, attr) * scale)

    totals.days_represented = int(round(days_represented))
    totals.simulations_run = sims

    if capture_profile and profiles:
        mean_factor = sum(factors_used) / len(factors_used)
        # Representative profile = the day whose seasonal factor is closest to
        # the annual mean (an average-irradiance day, best for expected-value use).
        best = min(profiles, key=lambda p: abs(p[0] - mean_factor))
        totals.profile = best[1]

    return totals
