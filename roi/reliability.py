"""
roi.reliability
==============
Grid-outage / reliability analysis based on an expected-value model over the
simulated 15-minute power profile.

Physical rule enforced here (section 11 of the spec)
----------------------------------------------------
Farm PV and tractor-roof PV provide **no** outage benefit unless an
islanding-capable inverter (or equivalent microgrid mode) is enabled.  With
``islanding_enabled=False`` the PV contribution to backup supply is exactly
zero — ordinary grid-tied PV disconnects during an outage.

Source labelling
----------------
The operational engine does **not** simulate a stationary backup battery, so
its contribution is an *analytical* backup model.  PV availability and critical
load come from the simulated profile.  ``result["sources"]`` records this so the
dashboard can show simulated-vs-analytical provenance without mixing them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import OutageAssumptions


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _evaluate(
    profile: List[Dict[str, Any]],
    o: OutageAssumptions,
    islanding_enabled: bool,
    storage_energy_kwh: float,
    storage_power_kw: float,
    default_critical_kw: float,
) -> Dict[str, Any]:
    """Expected outage exposure & cost for one backup configuration."""
    freq = o.frequency_per_year
    duration = o.average_duration_hours

    crit_series: List[float] = []
    pv_to_crit_series: List[float] = []
    for row in profile:
        crit = float(row.get("critical_load_kw", 0.0) or 0.0)
        if crit <= 0:
            crit = default_critical_kw
        pv = float(row.get("total_pv_kw", 0.0) or 0.0)
        island_pv = pv if islanding_enabled else 0.0
        crit_series.append(crit)
        pv_to_crit_series.append(min(crit, island_pv))

    crit_mean = _mean(crit_series) if crit_series else default_critical_kw
    pv_cover_mean = _mean(pv_to_crit_series) if pv_to_crit_series else 0.0
    residual_mean = max(0.0, crit_mean - pv_cover_mean)

    # Storage (battery + V2L) supplies the residual up to its power cap and a
    # per-outage energy budget.
    storage_supply_kw = min(storage_power_kw, residual_mean)
    if duration > 0 and storage_supply_kw > 0:
        energy_from_storage = min(storage_energy_kwh, storage_supply_kw * duration)
        autonomy_hours = energy_from_storage / storage_supply_kw
    else:
        autonomy_hours = 0.0
    autonomy_hours = min(autonomy_hours, duration)

    unserved_kw_while_backed = max(0.0, residual_mean - storage_supply_kw)
    unserved_energy_per_outage = (
        unserved_kw_while_backed * autonomy_hours
        + residual_mean * (duration - autonomy_hours)
    )
    unsupported_hours_per_outage = (
        duration if unserved_kw_while_backed > 1e-9 else (duration - autonomy_hours)
    )

    expected_outage_hours = freq * duration
    expected_unserved_energy = unserved_energy_per_outage * freq
    expected_unsupported_hours = unsupported_hours_per_outage * freq
    critical_energy_per_outage = crit_mean * duration
    coverage_pct = (
        100.0 * (1.0 - unserved_energy_per_outage / critical_energy_per_outage)
        if critical_energy_per_outage > 0 else 100.0
    )
    expected_task_disruptions = freq if unsupported_hours_per_outage > 1e-9 else 0.0

    cost = (
        expected_unserved_energy * o.value_of_lost_load_eur_per_kwh
        + expected_unsupported_hours * o.downtime_cost_eur_per_hour
        + expected_unsupported_hours * o.task_disruption_cost_eur_per_hour
    )

    return {
        "expected_outage_hours_per_year": expected_outage_hours,
        "expected_unserved_energy_kwh": expected_unserved_energy,
        "critical_load_coverage_pct": max(0.0, min(100.0, coverage_pct)),
        "autonomous_hours": autonomy_hours,
        "expected_task_disruptions": expected_task_disruptions,
        "expected_annual_outage_cost_eur": cost,
        "critical_load_mean_kw": crit_mean,
        "pv_backup_mean_kw": pv_cover_mean,
    }


def analyze_reliability(
    profile: List[Dict[str, Any]],
    o: OutageAssumptions,
    n_tractors: int,
    tractor_battery_kwh: float,
    v2l_max_discharge_kw: float,
    has_pv: bool,
) -> Dict[str, Any]:
    """Return reliability metrics + avoided outage cost (candidate vs no-backup)."""
    warnings: List[str] = []

    # Candidate storage resources.
    battery_usable = (
        o.backup_battery_kwh
        * (o.backup_battery_usable_pct / 100.0)
        * max(0.0, 1.0 - o.backup_reserve_soc_pct / 100.0)
    )
    v2l_energy = 0.0
    v2l_power = 0.0
    if o.v2l_enabled:
        # Analytical V2L headroom: usable tractor energy above the V2L SOC floor,
        # assuming an average available state of charge of 60 %.
        avail_soc_fraction = max(0.0, 0.60 - o.v2l_min_tractor_soc_pct / 100.0)
        v2l_energy = n_tractors * tractor_battery_kwh * avail_soc_fraction
        v2l_power = n_tractors * v2l_max_discharge_kw

    storage_energy = battery_usable + v2l_energy
    storage_power = o.backup_max_power_kw + v2l_power

    candidate = _evaluate(profile, o, o.islanding_enabled, storage_energy,
                          storage_power, o.critical_load_kw)
    # Baseline: grid-tied, no islanding, no backup → PV and storage give nothing.
    baseline = _evaluate(profile, o, False, 0.0, 0.0, o.critical_load_kw)

    avoided = (baseline["expected_annual_outage_cost_eur"]
               - candidate["expected_annual_outage_cost_eur"])

    if has_pv and not o.islanding_enabled:
        warnings.append(
            "Grid-connected PV is assumed to disconnect during an outage and "
            "therefore provides no backup-power benefit (islanding disabled)."
        )
    if o.backup_battery_kwh > 0:
        warnings.append(
            "Stationary backup-battery results are an analytical model — the "
            "operational engine does not simulate a fixed battery."
        )

    result = dict(candidate)
    result.update({
        "baseline_expected_outage_cost_eur": baseline["expected_annual_outage_cost_eur"],
        "avoided_outage_cost_eur": avoided,
        "islanding_enabled": o.islanding_enabled,
        "storage_energy_kwh": storage_energy,
        "storage_power_kw": storage_power,
        "sources": {
            "critical_load": "simulated_profile",
            "pv_availability": "simulated_profile",
            "stationary_battery": "analytical",
            "v2l": "analytical",
        },
        "warnings": warnings,
    })
    return result
