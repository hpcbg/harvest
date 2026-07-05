"""
roi.investments
==============
Builds per-investment economic results from annualised operational totals and
the user's assumptions.  Every investment is compared against an appropriate
baseline using **paired simulations** (the differences in energy / cost between
two otherwise-identical runs), so savings reflect the real simulator behaviour
rather than a blended average price.

Investments
-----------
* :func:`electric_vs_diesel`  – matched-service electric-vs-diesel counterfactual.
* :func:`farm_pv`             – fixed farm PV (baseline PV = 0).
* :func:`roof_pv`             – tractor-roof PV (baseline panel = 0).
* :func:`backup_islanding`    – reliability investment (see :mod:`roi.reliability`).
* :func:`portfolio`           – sequential incremental combination, no double-count.

Each builder returns an :class:`InvestmentResult` plus the year-0 ``net_capex``
and the per-year net cash flows, so the portfolio can compose marginal stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import calculator as calc
from .models import (
    CashflowRow, InvestmentResult, OperationalTotals,
    ROIAssumptions,
)


@dataclass
class _Built:
    result: InvestmentResult
    net_capex: float
    year_nets: List[float]          # net cash flow for years 1..horizon
    complete: bool = True           # False → excluded from the portfolio


def _incomplete(inv_id: str, name: str, net_capex: float,
                missing: List[str], extra: Dict[str, Any] = None) -> "_Built":
    """Result for an investment whose required inputs are missing.

    Financial metrics are left as ``None`` ("Input required") — no zero-year
    payback, no NPV — and the investment is excluded from the portfolio.
    """
    metrics: Dict[str, Any] = {
        "status": "input_required",
        "missing": missing,
        "net_capex_eur": round(net_capex, 2) if net_capex else None,
        "npv_eur": None,
        "irr_pct": None,
        "simple_payback_years": None,
        "discounted_payback_years": None,
        "roi_pct": None,
    }
    if extra:
        metrics.update({k: v for k, v in extra.items()})
    result = InvestmentResult(id=inv_id, name=name, metrics=metrics,
                              annual_cashflows=[], warnings=list(missing))
    return _Built(result=result, net_capex=net_capex, year_nets=[], complete=False)


def _farm_grid_cost(t: OperationalTotals) -> float:
    """Annual grid cost attributable to farm (non-tractor) loads."""
    return max(0.0, t.grid_cost_eur - t.tractor_charge_cost_eur)


# ─────────────────────────────────────────────────────────────────────────────
# Shared cash-flow finaliser
# ─────────────────────────────────────────────────────────────────────────────

def _finalize(
    inv_id: str,
    name: str,
    net_capex: float,
    year_rows: List[CashflowRow],
    discount_rate_pct: float,
    extra_metrics: Dict[str, Any],
    warnings: List[str],
) -> _Built:
    """Assemble year-0 CAPEX + yearly rows and compute all financial metrics."""
    rows: List[CashflowRow] = []
    year0 = CashflowRow(year=0, net_cash_flow=-net_capex, replacement_cost=0.0)
    rows.append(year0)
    rows.extend(year_rows)

    cum = 0.0
    cum_disc = 0.0
    for row in rows:
        row.discount_factor = calc.discount_factor(discount_rate_pct, row.year)
        row.discounted_cash_flow = row.net_cash_flow * row.discount_factor
        cum += row.net_cash_flow
        cum_disc += row.discounted_cash_flow
        row.cumulative_cash_flow = cum
        row.cumulative_discounted_cash_flow = cum_disc

    flows = [row.net_cash_flow for row in rows]
    cumulative_net_benefit = sum(r.net_cash_flow for r in year_rows)  # excludes year 0

    metrics = {
        "net_capex_eur": calc.sanitize(net_capex, 2),
        "npv_eur": calc.sanitize(calc.npv(discount_rate_pct, flows), 2),
        "irr_pct": calc.sanitize(calc.irr(flows), 2),
        "simple_payback_years": calc.sanitize(calc.simple_payback(flows), 2),
        "discounted_payback_years": calc.sanitize(calc.discounted_payback(flows, discount_rate_pct), 2),
        "roi_pct": calc.sanitize(calc.roi_pct(net_capex, cumulative_net_benefit), 1),
        "cumulative_net_benefit_eur": calc.sanitize(cumulative_net_benefit, 2),
    }
    metrics.update({k: (calc.sanitize(v, 2) if isinstance(v, (int, float)) else v)
                    for k, v in extra_metrics.items()})

    result = InvestmentResult(id=inv_id, name=name, metrics=metrics,
                              annual_cashflows=rows, warnings=warnings)
    return _Built(result=result, net_capex=net_capex,
                  year_nets=[r.net_cash_flow for r in year_rows])


# ─────────────────────────────────────────────────────────────────────────────
# Electric vs diesel
# ─────────────────────────────────────────────────────────────────────────────

def diesel_fuel_litres(
    transit_distance_km: float,
    pto_hours: float,
    idle_hours: float,
    a: ROIAssumptions,
) -> Dict[str, float]:
    """Diesel fuel breakdown for a matched workload (litres)."""
    d = a.diesel
    travel = transit_distance_km * d.travel_litres_per_100km / 100.0
    work = pto_hours * d.pto_litres_per_hour
    idle = idle_hours * d.idle_litres_per_hour
    return {"travel": travel, "work": work, "idle": idle, "total": travel + work + idle}


def electric_vs_diesel(
    electric_totals: OperationalTotals,
    a: ROIAssumptions,
    n_tractors: int,
    n_chargers: int,
    baseline_name: str = "diesel fleet",
) -> _Built:
    """Matched-service electric-fleet-vs-diesel comparison.

    ``electric_totals`` are the annual operational totals of the electric fleet
    charging from the grid (no PV) — the same workload a diesel fleet would do.
    The comparison is per *completed service*: diesel performs the identical
    task set, so service levels match by construction.
    """
    fin = a.financial
    ef = a.electric_fleet
    horizon = a.analysis.financial_horizon_years

    litres = diesel_fuel_litres(
        electric_totals.transit_distance_km,
        electric_totals.pto_hours,
        electric_totals.idle_hours,
        a,
    )
    diesel_fuel_cost = litres["total"] * a.diesel.fuel_price_eur_per_litre
    diesel_maint = electric_totals.operating_hours * a.diesel.maintenance_eur_per_hour
    electric_charge_cost = electric_totals.tractor_charge_cost_eur
    electric_maint = electric_totals.operating_hours * ef.electric_maintenance_eur_per_hour

    annual_operating_savings = (
        diesel_fuel_cost + diesel_maint - electric_charge_cost - electric_maint
    )

    electric_capex = (
        ef.electric_tractor_purchase_eur * n_tractors
        + (ef.charger_capex_eur_each + ef.charger_installation_eur_each) * n_chargers
    )
    diesel_capex = ef.diesel_equivalent_purchase_eur * n_tractors
    incremental_capex = electric_capex - diesel_capex
    net_capex = incremental_capex - ef.grant_eur

    # Required-input check → "Input required" rather than a misleading zero result.
    missing: List[str] = []
    if a.diesel.fuel_price_eur_per_litre <= 0:
        missing.append("Diesel fuel price is required for the electric-fleet investment.")
    if a.diesel.travel_litres_per_100km <= 0 and a.diesel.pto_litres_per_hour <= 0:
        missing.append("Diesel fuel consumption (travel and/or PTO) is required.")
    if ef.electric_tractor_purchase_eur <= 0:
        missing.append("E-tractor price (€/tractor) is required for the electric-fleet investment.")
    if ef.diesel_equivalent_purchase_eur <= 0:
        missing.append("Diesel tractor price (€/tractor) is required for the electric-fleet investment.")
    if ef.charger_capex_eur_each <= 0:
        missing.append("Charger CAPEX (€/charger) is required for the electric-fleet investment.")
    if missing:
        return _incomplete("electric_fleet",
                           "Electric tractors and charging infrastructure",
                           net_capex, missing,
                           {"incremental_capex_eur": round(incremental_capex, 2)})

    warnings: List[str] = []
    if electric_totals.tasks_missed > 0.5:
        warnings.append(
            f"Electric fleet leaves ~{electric_totals.tasks_missed:.1f} tasks/yr "
            "uncompleted; the diesel counterfactual is charged the same unserved "
            "workload so the comparison stays matched-service."
        )
    if a.diesel.fuel_price_eur_per_litre <= 0:
        warnings.append("Diesel fuel price is zero — operating savings are unset (input required).")

    year_rows: List[CashflowRow] = []
    for y in range(1, horizon + 1):
        fuel = calc.escalate(diesel_fuel_cost, fin.diesel_price_escalation_pct, y)
        elec = calc.escalate(electric_charge_cost, fin.electricity_price_escalation_pct, y)
        d_maint = calc.escalate(diesel_maint, fin.maintenance_escalation_pct, y)
        e_maint = calc.escalate(electric_maint, fin.maintenance_escalation_pct, y)
        replacement = 0.0
        if ef.battery_replacement_year is not None and y == ef.battery_replacement_year:
            replacement = ef.battery_replacement_eur
        revenue = ef.residual_value_eur if y == horizon else 0.0

        baseline_cost = fuel + d_maint
        candidate_cost = elec + e_maint
        net = baseline_cost - candidate_cost - replacement + revenue
        year_rows.append(CashflowRow(
            year=y,
            baseline_operating_cost=baseline_cost,
            candidate_operating_cost=candidate_cost,
            fuel_cost=fuel,
            electricity_cost=elec,
            maintenance=e_maint - d_maint,
            revenue=revenue,
            replacement_cost=replacement,
            net_cash_flow=net,
        ))

    co2_avoided = litres["total"] * a.diesel.co2_kg_per_litre
    extra = {
        "annual_distance_km": electric_totals.transit_distance_km,
        "annual_operating_hours": electric_totals.operating_hours,
        "annual_diesel_litres": litres["total"],
        "annual_diesel_cost_eur": diesel_fuel_cost,
        "annual_electric_charge_cost_eur": electric_charge_cost,
        "annual_maintenance_difference_eur": diesel_maint - electric_maint,
        "annual_operating_savings_eur": annual_operating_savings,
        "incremental_capex_eur": incremental_capex,
        "electric_capex_eur": electric_capex,
        "diesel_capex_eur": diesel_capex,
        "annual_co2_avoided_kg": co2_avoided,
        "tasks_completed_per_year": electric_totals.tasks_completed,
        "tasks_missed_per_year": electric_totals.tasks_missed,
    }
    return _finalize("electric_fleet",
                     "Electric tractors and charging infrastructure",
                     net_capex, year_rows, fin.discount_rate_pct, extra, warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Farm PV
# ─────────────────────────────────────────────────────────────────────────────

def farm_pv(
    baseline: OperationalTotals,
    candidate: OperationalTotals,
    a: ROIAssumptions,
    farm_pv_kwp: float,
    inv_id: str = "farm_pv",
    name: str = "Fixed farm PV installation",
) -> _Built:
    """Farm PV ROI from a paired baseline(PV=0)/candidate(PV=kWp) comparison."""
    fin = a.financial
    p = a.farm_pv
    horizon = a.analysis.financial_horizon_years

    # Avoided grid purchases = reduction in annual grid cost (time-step priced).
    avoided_grid_cost = baseline.grid_cost_eur - candidate.grid_cost_eur
    pv_generated = candidate.farm_pv_generated_kwh
    self_consumed = max(0.0, candidate.farm_pv_used_kwh)
    unused = max(0.0, pv_generated - self_consumed)
    exported = unused if p.export_enabled else 0.0
    curtailed = 0.0 if p.export_enabled else unused
    export_rev_base = exported * p.feed_in_tariff_eur_per_kwh

    equipment_capex = farm_pv_kwp * p.capex_eur_per_kwp + p.fixed_installation_eur
    net_capex = equipment_capex - p.grant_eur
    annual_om = p.annual_om_eur + p.annual_om_pct_of_capex / 100.0 * equipment_capex

    if p.capex_eur_per_kwp <= 0:
        return _incomplete(inv_id, name, net_capex,
                           ["Farm PV CAPEX per kWp is required."],
                           {"installed_kwp": farm_pv_kwp})

    warnings: List[str] = []

    year_rows: List[CashflowRow] = []
    for y in range(1, horizon + 1):
        # Yield degrades; the value of each kWh escalates with electricity price.
        yield_factor = calc.degrade(1.0, p.annual_degradation_pct, y)
        price_factor = (1.0 + fin.electricity_price_escalation_pct / 100.0) ** (y - 1)
        benefit = avoided_grid_cost * yield_factor * price_factor
        export_rev = export_rev_base * yield_factor * price_factor
        om = calc.escalate(annual_om, fin.general_cost_escalation_pct, y)
        replacement = 0.0
        if p.inverter_replacement_year is not None and y == p.inverter_replacement_year:
            replacement = p.inverter_replacement_eur
        residual = p.residual_value_eur if y == horizon else 0.0
        net = benefit + export_rev - om - replacement + residual
        year_rows.append(CashflowRow(
            year=y,
            baseline_operating_cost=calc.escalate(baseline.grid_cost_eur, fin.electricity_price_escalation_pct, y),
            candidate_operating_cost=calc.escalate(candidate.grid_cost_eur, fin.electricity_price_escalation_pct, y),
            electricity_cost=-benefit,
            maintenance=om,
            revenue=export_rev + residual,
            replacement_cost=replacement,
            net_cash_flow=net,
        ))

    extra = {
        "installed_kwp": farm_pv_kwp,
        "annual_pv_generation_kwh": pv_generated,
        "annual_self_consumed_kwh": self_consumed,
        "annual_exported_kwh": exported,
        "annual_curtailed_kwh": curtailed,
        "pv_utilisation_pct": (100.0 * self_consumed / pv_generated) if pv_generated > 0 else None,
        "annual_avoided_grid_cost_eur": avoided_grid_cost,
        "annual_export_revenue_eur": export_rev_base,
        "annual_om_eur": annual_om,
        "annual_net_benefit_eur": avoided_grid_cost + export_rev_base - annual_om,
    }
    return _finalize(inv_id, name, net_capex, year_rows,
                     fin.discount_rate_pct, extra, warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Tractor-roof PV
# ─────────────────────────────────────────────────────────────────────────────

def roof_pv(
    baseline: OperationalTotals,
    candidate: OperationalTotals,
    a: ROIAssumptions,
    n_equipped: int,
    roof_panel_w: float,
    inv_id: str = "tractor_roof_pv",
    name: str = "Tractor-roof PV panels",
) -> _Built:
    """Roof PV ROI from a paired baseline(panel=0)/candidate(panel=W) comparison."""
    fin = a.financial
    p = a.roof_pv
    horizon = a.analysis.financial_horizon_years

    avoided_grid_cost = baseline.grid_cost_eur - candidate.grid_cost_eur
    avoided_charging_cost = baseline.tractor_charge_cost_eur - candidate.tractor_charge_cost_eur
    grid_avoided_kwh = baseline.grid_kwh - candidate.grid_kwh
    roof_generated = candidate.tractor_pv_generated_kwh
    extra_tasks = candidate.tasks_completed - baseline.tasks_completed

    equipment_capex = (p.capex_eur_per_tractor + p.installation_eur_per_tractor) * n_equipped
    net_capex = equipment_capex - p.grant_eur
    annual_om = p.annual_om_eur_per_tractor * n_equipped

    if p.capex_eur_per_tractor <= 0:
        return _incomplete(inv_id, name, net_capex,
                           ["Roof PV CAPEX per tractor is required."],
                           {"equipped_tractors": n_equipped})

    warnings: List[str] = []

    year_rows: List[CashflowRow] = []
    for y in range(1, horizon + 1):
        yield_factor = calc.degrade(1.0, p.annual_degradation_pct, y)
        price_factor = (1.0 + fin.electricity_price_escalation_pct / 100.0) ** (y - 1)
        benefit = avoided_grid_cost * yield_factor * price_factor
        om = calc.escalate(annual_om, fin.general_cost_escalation_pct, y)
        replacement = 0.0
        if p.replacement_year is not None and y == p.replacement_year:
            replacement = p.replacement_eur_per_tractor * n_equipped
        residual = p.residual_value_eur if y == horizon else 0.0
        net = benefit - om - replacement + residual
        year_rows.append(CashflowRow(
            year=y,
            electricity_cost=-benefit,
            maintenance=om,
            revenue=residual,
            replacement_cost=replacement,
            net_cash_flow=net,
        ))

    extra = {
        "equipped_tractors": n_equipped,
        "annual_roof_pv_generation_kwh": roof_generated,
        "annual_grid_charging_avoided_kwh": grid_avoided_kwh,
        "annual_electricity_cost_avoided_eur": avoided_grid_cost,
        "annual_charging_cost_avoided_eur": avoided_charging_cost,
        "additional_tasks_per_year": extra_tasks,
        "annual_om_eur": annual_om,
        "annual_net_benefit_eur": avoided_grid_cost - annual_om,
    }
    return _finalize(inv_id, name, net_capex, year_rows,
                     fin.discount_rate_pct, extra, warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Backup / islanding (reliability)
# ─────────────────────────────────────────────────────────────────────────────

def backup_islanding(
    reliability: Dict[str, Any],
    a: ROIAssumptions,
    inv_id: str = "backup_islanding",
    name: str = "Islanding / backup capability",
) -> _Built:
    """Reliability investment whose benefit is avoided outage cost per year."""
    fin = a.financial
    o = a.outages
    horizon = a.analysis.financial_horizon_years

    annual_benefit = float(reliability.get("avoided_outage_cost_eur", 0.0) or 0.0)
    net_capex = o.islanding_capex_eur + o.backup_capex_eur - o.grant_eur

    if (o.islanding_capex_eur + o.backup_capex_eur) <= 0:
        return _incomplete(inv_id, name, net_capex,
                           ["Islanding and/or backup CAPEX is required for the backup investment."],
                           {"annual_avoided_outage_cost_eur": round(annual_benefit, 2)})

    warnings: List[str] = list(reliability.get("warnings", []))

    year_rows: List[CashflowRow] = []
    for y in range(1, horizon + 1):
        benefit = calc.escalate(annual_benefit, fin.general_cost_escalation_pct, y)
        year_rows.append(CashflowRow(
            year=y,
            outage_loss=-benefit,
            net_cash_flow=benefit,
        ))

    extra = {
        "annual_avoided_outage_cost_eur": annual_benefit,
        "expected_outage_hours_per_year": reliability.get("expected_outage_hours_per_year"),
        "expected_unserved_energy_kwh": reliability.get("expected_unserved_energy_kwh"),
        "critical_load_coverage_pct": reliability.get("critical_load_coverage_pct"),
    }
    return _finalize(inv_id, name, net_capex, year_rows,
                     fin.discount_rate_pct, extra, warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio (sequential incremental — avoids double-counting)
# ─────────────────────────────────────────────────────────────────────────────

def portfolio(
    stages: List[_Built],
    a: ROIAssumptions,
    order: List[str],
) -> _Built:
    """Combine marginal stages into one portfolio result.

    ``stages`` are the *marginal* builds (electric-vs-diesel, farm-PV over the
    electric baseline, roof-PV over the farm-PV baseline, backup) — each already
    measured only against the previous stage, so summing their yearly net cash
    flows does not double-count overlapping savings.
    """
    fin = a.financial
    horizon = a.analysis.financial_horizon_years

    # Only complete stages contribute — incomplete (input-required) investments are
    # excluded so the portfolio is never based on missing assumptions.
    included = [s for s in stages if s.complete]
    excluded = [s for s in stages if not s.complete]
    total_capex = sum(s.net_capex for s in included)

    year_rows: List[CashflowRow] = []
    for y in range(1, horizon + 1):
        net = sum(s.year_nets[y - 1] for s in included if len(s.year_nets) >= y)
        year_rows.append(CashflowRow(year=y, net_cash_flow=net))

    warnings = [
        "Combined portfolio uses sequential incremental baselines "
        f"({' → '.join(order)}); each stage counts only its marginal benefit, so "
        "the total is not the sum of the standalone results.",
    ]
    for s in excluded:
        warnings.append(f"Excluded from portfolio (input required): {s.result.name}.")
    extra = {
        "investment_order": order,
        "stage_ids": [s.result.id for s in included],
        "excluded_ids": [s.result.id for s in excluded],
        "total_capex_eur": total_capex,
    }
    built = _finalize("portfolio", "Combined HARVEST portfolio",
                      total_capex, year_rows, fin.discount_rate_pct, extra, warnings)
    if not included:
        built.complete = False
    return built
