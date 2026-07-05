"""
roi.validation
==============
Central validation layer for ROI requests.  ``validate_request`` raises
``ROIValidationError`` (a plain ``ValueError`` subclass) with a human-readable
message; the server maps this to HTTP 400 and never leaks a traceback.

``collect_warnings`` produces the non-fatal advisory messages the dashboard is
required to surface (derived distances, single-day periods, PV-without-islanding,
zero/missing inputs, service-level differences, standalone-overlap, …).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List

from .models import ROIAssumptions


class ROIValidationError(ValueError):
    """Raised when an ROI request is structurally or numerically invalid."""


def _parse_date(value: str, field_name: str) -> date:
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except (ValueError, TypeError):
        raise ROIValidationError(f"{field_name} is not a valid ISO date: {value!r}")


def validate_request(
    start: str,
    end: str,
    assumptions: ROIAssumptions,
    selected_investments: List[str],
) -> None:
    """Validate a fully-parsed ROI request. Raises ROIValidationError on failure."""
    start_d = _parse_date(start, "start date")
    end_d = _parse_date(end, "end date")
    if end_d < start_d:
        raise ROIValidationError("End date must not precede start date.")

    a = assumptions
    if a.analysis.financial_horizon_years < 1:
        raise ROIValidationError("Financial horizon must be at least one year.")

    mode = a.analysis.period_mode
    if mode not in ("exact", "representative_month", "auto"):
        raise ROIValidationError(
            f"Unsupported analysis mode {mode!r} "
            "(expected exact | representative_month | auto)."
        )
    if a.analysis.representative_days_per_month < 1:
        raise ROIValidationError("representative_days_per_month must be at least 1.")

    # Percentages within reasonable mathematical bounds.
    for label, value in [
        ("discount rate", a.financial.discount_rate_pct),
        ("electricity price escalation", a.financial.electricity_price_escalation_pct),
        ("diesel price escalation", a.financial.diesel_price_escalation_pct),
        ("general cost escalation", a.financial.general_cost_escalation_pct),
    ]:
        if not (-100.0 < value < 1000.0):
            raise ROIValidationError(f"{label} ({value}%) is out of a sensible range.")

    for label, value in [
        ("farm PV degradation", a.farm_pv.annual_degradation_pct),
        ("roof PV degradation", a.roof_pv.annual_degradation_pct),
    ]:
        if not (0.0 <= value < 100.0):
            raise ROIValidationError(f"{label} ({value}%) must be within 0–100%.")

    # Non-negativity: CAPEX, prices, capacities.
    negatives: List[str] = []

    def _check(name: str, value: float) -> None:
        if value < 0:
            negatives.append(f"{name} cannot be negative ({value}).")

    _check("diesel fuel price", a.diesel.fuel_price_eur_per_litre)
    _check("diesel travel consumption", a.diesel.travel_litres_per_100km)
    _check("diesel PTO consumption", a.diesel.pto_litres_per_hour)
    _check("diesel idle consumption", a.diesel.idle_litres_per_hour)
    _check("electric tractor purchase", a.electric_fleet.electric_tractor_purchase_eur)
    _check("diesel equivalent purchase", a.electric_fleet.diesel_equivalent_purchase_eur)
    _check("charger CAPEX", a.electric_fleet.charger_capex_eur_each)
    _check("charger installation", a.electric_fleet.charger_installation_eur_each)
    _check("farm PV CAPEX/kWp", a.farm_pv.capex_eur_per_kwp)
    _check("farm PV installation", a.farm_pv.fixed_installation_eur)
    _check("roof PV CAPEX/tractor", a.roof_pv.capex_eur_per_tractor)
    _check("outage frequency", a.outages.frequency_per_year)
    _check("outage duration", a.outages.average_duration_hours)
    _check("value of lost load", a.outages.value_of_lost_load_eur_per_kwh)
    _check("backup battery capacity", a.outages.backup_battery_kwh)
    _check("backup max power", a.outages.backup_max_power_kw)
    _check("critical load", a.outages.critical_load_kw)
    if negatives:
        raise ROIValidationError(" ".join(negatives))

    valid_ids = {
        "electric_fleet", "farm_pv", "tractor_roof_pv", "backup_islanding", "portfolio",
    }
    unknown = [i for i in selected_investments if i not in valid_ids]
    if unknown:
        raise ROIValidationError(f"Unknown investment id(s): {', '.join(unknown)}.")
    if not selected_investments:
        raise ROIValidationError("Select at least one investment to analyse.")


def collect_warnings(
    assumptions: ROIAssumptions,
    selected_investments: List[str],
    period_days: int,
    farm_pv_kwp: float,
    roof_panel_w: float,
) -> List[str]:
    """Non-fatal advisory warnings shown in the dashboard."""
    warns: List[str] = []
    a = assumptions

    if period_days <= 1:
        warns.append(
            "Operational period covers a single day — seasonal PV variation is "
            "not represented. Use a longer range or representative-month mode."
        )

    if "farm_pv" in selected_investments and farm_pv_kwp <= 0:
        warns.append("Farm PV analysis requires a non-zero installed PV capacity.")
    if "tractor_roof_pv" in selected_investments and roof_panel_w <= 0:
        warns.append("Tractor-roof PV analysis requires a non-zero roof-panel power.")

    if a.outages.enabled and not a.outages.islanding_enabled and (
        farm_pv_kwp > 0 or roof_panel_w > 0
    ):
        warns.append(
            "Grid-connected PV is assumed to disconnect during an outage and "
            "therefore provides no backup-power benefit (islanding disabled)."
        )

    # Zero/missing financial inputs → results are placeholders.
    if "electric_fleet" in selected_investments:
        if a.diesel.fuel_price_eur_per_litre <= 0:
            warns.append("Diesel fuel price is zero — electric-vs-diesel savings are unset (input required).")
        if a.electric_fleet.electric_tractor_purchase_eur <= 0 and \
           a.electric_fleet.diesel_equivalent_purchase_eur <= 0:
            warns.append("Tractor purchase prices are zero — incremental CAPEX is unset (input required).")
    if "farm_pv" in selected_investments and a.farm_pv.capex_eur_per_kwp <= 0:
        warns.append("Farm PV CAPEX per kWp is zero — payback/NPV are unset (input required).")
    if "tractor_roof_pv" in selected_investments and a.roof_pv.capex_eur_per_tractor <= 0:
        warns.append("Roof PV CAPEX per tractor is zero — payback/NPV are unset (input required).")

    if "portfolio" in selected_investments:
        warns.append(
            "Standalone investment values are each measured against their own "
            "baseline and are not necessarily additive — see the combined "
            "portfolio result, which avoids double-counting overlapping savings."
        )

    return warns
