"""
roi.models
==========
Typed data structures for the ROI & Investment module: economic/technical
assumptions, aggregated operational totals, per-year cash-flow rows, per-
investment results and the top-level API response.

The assumption dataclasses know how to build themselves from the ``roi`` section
of the merged HARVEST config (``from_config``), so economic assumptions are
never hard-coded in Python source — they live in ``config.yaml`` /
``config.local.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def _f(d: Dict[str, Any], key: str, default: float) -> float:
    v = d.get(key, default)
    return default if v is None else float(v)


def _opt_int(d: Dict[str, Any], key: str) -> Optional[int]:
    v = d.get(key)
    return None if v in (None, "", "null") else int(v)


# ─────────────────────────────────────────────────────────────────────────────
# Assumption groups
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    period_mode: str = "auto"                 # exact | representative_month | auto
    auto_exact_max_days: int = 90
    representative_days_per_month: int = 1
    default_start_date: str = "2026-01-01"
    default_end_date: str = "2026-12-31"
    financial_horizon_years: int = 10
    currency: str = "EUR"
    scenario: str = "full_smart"              # operating strategy used for ROI runs
    base_seed: int = 42

    @classmethod
    def from_config(cls, roi: Dict[str, Any]) -> "AnalysisConfig":
        a = roi.get("analysis", {}) or {}
        return cls(
            period_mode=str(a.get("period_mode", "auto")),
            auto_exact_max_days=int(a.get("auto_exact_max_days", 90)),
            representative_days_per_month=int(a.get("representative_days_per_month", 1)),
            default_start_date=str(a.get("default_start_date", "2026-01-01")),
            default_end_date=str(a.get("default_end_date", "2026-12-31")),
            financial_horizon_years=int(a.get("financial_horizon_years", 10)),
            currency=str(a.get("currency", "EUR")),
            scenario=str(a.get("scenario", "full_smart")),
            base_seed=int(a.get("base_seed", 42)),
        )


@dataclass
class FinancialAssumptions:
    discount_rate_pct: float = 6.0
    general_cost_escalation_pct: float = 2.0
    electricity_price_escalation_pct: float = 2.5
    diesel_price_escalation_pct: float = 2.5
    maintenance_escalation_pct: float = 2.0

    @classmethod
    def from_config(cls, roi: Dict[str, Any]) -> "FinancialAssumptions":
        f = roi.get("financial", {}) or {}
        return cls(
            discount_rate_pct=_f(f, "discount_rate_pct", 6.0),
            general_cost_escalation_pct=_f(f, "general_cost_escalation_pct", 2.0),
            electricity_price_escalation_pct=_f(f, "electricity_price_escalation_pct", 2.5),
            diesel_price_escalation_pct=_f(f, "diesel_price_escalation_pct", 2.5),
            maintenance_escalation_pct=_f(f, "maintenance_escalation_pct",
                                          _f(f, "general_cost_escalation_pct", 2.0)),
        )


@dataclass
class ServiceValue:
    uncompleted_task_cost_eur: float = 0.0
    downtime_cost_eur_per_hour: float = 0.0

    @classmethod
    def from_config(cls, roi: Dict[str, Any]) -> "ServiceValue":
        s = roi.get("service_value", {}) or {}
        return cls(
            uncompleted_task_cost_eur=_f(s, "uncompleted_task_cost_eur", 0.0),
            downtime_cost_eur_per_hour=_f(s, "downtime_cost_eur_per_hour", 0.0),
        )


@dataclass
class ElectricFleetAssumptions:
    electric_tractor_purchase_eur: float = 0.0
    diesel_equivalent_purchase_eur: float = 0.0
    electric_maintenance_eur_per_hour: float = 0.0
    charger_capex_eur_each: float = 0.0
    charger_installation_eur_each: float = 0.0
    grant_eur: float = 0.0
    residual_value_eur: float = 0.0
    battery_replacement_year: Optional[int] = None
    battery_replacement_eur: float = 0.0

    @classmethod
    def from_config(cls, roi: Dict[str, Any]) -> "ElectricFleetAssumptions":
        e = roi.get("electric_fleet", {}) or {}
        return cls(
            electric_tractor_purchase_eur=_f(e, "electric_tractor_purchase_eur", 0.0),
            diesel_equivalent_purchase_eur=_f(e, "diesel_equivalent_purchase_eur", 0.0),
            electric_maintenance_eur_per_hour=_f(e, "electric_maintenance_eur_per_hour", 0.0),
            charger_capex_eur_each=_f(e, "charger_capex_eur_each", 0.0),
            charger_installation_eur_each=_f(e, "charger_installation_eur_each", 0.0),
            grant_eur=_f(e, "grant_eur", 0.0),
            residual_value_eur=_f(e, "residual_value_eur", 0.0),
            battery_replacement_year=_opt_int(e, "battery_replacement_year"),
            battery_replacement_eur=_f(e, "battery_replacement_eur", 0.0),
        )


@dataclass
class DieselAssumptions:
    fuel_price_eur_per_litre: float = 0.0
    travel_litres_per_100km: float = 0.0
    pto_litres_per_hour: float = 0.0
    idle_litres_per_hour: float = 0.0
    maintenance_eur_per_hour: float = 0.0
    co2_kg_per_litre: float = 0.0

    @classmethod
    def from_config(cls, roi: Dict[str, Any]) -> "DieselAssumptions":
        d = roi.get("diesel", {}) or {}
        return cls(
            fuel_price_eur_per_litre=_f(d, "fuel_price_eur_per_litre", 0.0),
            travel_litres_per_100km=_f(d, "travel_litres_per_100km", 0.0),
            pto_litres_per_hour=_f(d, "pto_litres_per_hour", 0.0),
            idle_litres_per_hour=_f(d, "idle_litres_per_hour", 0.0),
            maintenance_eur_per_hour=_f(d, "maintenance_eur_per_hour", 0.0),
            co2_kg_per_litre=_f(d, "co2_kg_per_litre", 0.0),
        )


@dataclass
class FarmPVAssumptions:
    capex_eur_per_kwp: float = 0.0
    fixed_installation_eur: float = 0.0
    annual_om_pct_of_capex: float = 0.0
    annual_om_eur: float = 0.0
    annual_degradation_pct: float = 0.5
    feed_in_tariff_eur_per_kwh: float = 0.0
    export_enabled: bool = False
    grant_eur: float = 0.0
    inverter_replacement_year: Optional[int] = None
    inverter_replacement_eur: float = 0.0
    residual_value_eur: float = 0.0
    lifetime_years: int = 25

    @classmethod
    def from_config(cls, roi: Dict[str, Any]) -> "FarmPVAssumptions":
        p = roi.get("farm_pv", {}) or {}
        return cls(
            capex_eur_per_kwp=_f(p, "capex_eur_per_kwp", 0.0),
            fixed_installation_eur=_f(p, "fixed_installation_eur", 0.0),
            annual_om_pct_of_capex=_f(p, "annual_om_pct_of_capex", 0.0),
            annual_om_eur=_f(p, "annual_om_eur", 0.0),
            annual_degradation_pct=_f(p, "annual_degradation_pct", 0.5),
            feed_in_tariff_eur_per_kwh=_f(p, "feed_in_tariff_eur_per_kwh", 0.0),
            export_enabled=bool(p.get("export_enabled", False)),
            grant_eur=_f(p, "grant_eur", 0.0),
            inverter_replacement_year=_opt_int(p, "inverter_replacement_year"),
            inverter_replacement_eur=_f(p, "inverter_replacement_eur", 0.0),
            residual_value_eur=_f(p, "residual_value_eur", 0.0),
            lifetime_years=int(p.get("lifetime_years", 25)),
        )


@dataclass
class RoofPVAssumptions:
    capex_eur_per_tractor: float = 0.0
    installation_eur_per_tractor: float = 0.0
    annual_om_eur_per_tractor: float = 0.0
    annual_degradation_pct: float = 0.5
    replacement_year: Optional[int] = None
    replacement_eur_per_tractor: float = 0.0
    grant_eur: float = 0.0
    equipped_tractors: Optional[int] = None
    residual_value_eur: float = 0.0

    @classmethod
    def from_config(cls, roi: Dict[str, Any]) -> "RoofPVAssumptions":
        p = roi.get("tractor_roof_pv", {}) or {}
        return cls(
            capex_eur_per_tractor=_f(p, "capex_eur_per_tractor", 0.0),
            installation_eur_per_tractor=_f(p, "installation_eur_per_tractor", 0.0),
            annual_om_eur_per_tractor=_f(p, "annual_om_eur_per_tractor", 0.0),
            annual_degradation_pct=_f(p, "annual_degradation_pct", 0.5),
            replacement_year=_opt_int(p, "replacement_year"),
            replacement_eur_per_tractor=_f(p, "replacement_eur_per_tractor", 0.0),
            grant_eur=_f(p, "grant_eur", 0.0),
            equipped_tractors=_opt_int(p, "equipped_tractors"),
            residual_value_eur=_f(p, "residual_value_eur", 0.0),
        )


@dataclass
class OutageAssumptions:
    enabled: bool = False
    frequency_per_year: float = 0.0
    average_duration_hours: float = 0.0
    value_of_lost_load_eur_per_kwh: float = 0.0
    downtime_cost_eur_per_hour: float = 0.0
    task_disruption_cost_eur_per_hour: float = 0.0
    islanding_enabled: bool = False
    critical_load_kw: float = 0.0
    backup_battery_kwh: float = 0.0
    backup_battery_usable_pct: float = 80.0
    backup_reserve_soc_pct: float = 10.0
    backup_max_power_kw: float = 0.0
    v2l_enabled: bool = False
    v2l_min_tractor_soc_pct: float = 30.0
    islanding_capex_eur: float = 0.0
    backup_capex_eur: float = 0.0
    grant_eur: float = 0.0
    seed: int = 42

    @classmethod
    def from_config(cls, roi: Dict[str, Any]) -> "OutageAssumptions":
        o = roi.get("outages", {}) or {}
        return cls(
            enabled=bool(o.get("enabled", False)),
            frequency_per_year=_f(o, "frequency_per_year", 0.0),
            average_duration_hours=_f(o, "average_duration_hours", 0.0),
            value_of_lost_load_eur_per_kwh=_f(o, "value_of_lost_load_eur_per_kwh", 0.0),
            downtime_cost_eur_per_hour=_f(o, "downtime_cost_eur_per_hour", 0.0),
            task_disruption_cost_eur_per_hour=_f(o, "task_disruption_cost_eur_per_hour", 0.0),
            islanding_enabled=bool(o.get("islanding_enabled", False)),
            critical_load_kw=_f(o, "critical_load_kw", 0.0),
            backup_battery_kwh=_f(o, "backup_battery_kwh", 0.0),
            backup_battery_usable_pct=_f(o, "backup_battery_usable_pct", 80.0),
            backup_reserve_soc_pct=_f(o, "backup_reserve_soc_pct", 10.0),
            backup_max_power_kw=_f(o, "backup_max_power_kw", 0.0),
            v2l_enabled=bool(o.get("v2l_enabled", False)),
            v2l_min_tractor_soc_pct=_f(o, "v2l_min_tractor_soc_pct", 30.0),
            islanding_capex_eur=_f(o, "islanding_capex_eur", 0.0),
            backup_capex_eur=_f(o, "backup_capex_eur", 0.0),
            grant_eur=_f(o, "grant_eur", 0.0),
            seed=int(o.get("seed", 42)),
        )


@dataclass
class SensitivityConfig:
    enabled: bool = True
    variation_pct: float = 20.0

    @classmethod
    def from_config(cls, roi: Dict[str, Any]) -> "SensitivityConfig":
        s = roi.get("sensitivity", {}) or {}
        return cls(
            enabled=bool(s.get("enabled", True)),
            variation_pct=_f(s, "variation_pct", 20.0),
        )


@dataclass
class ROIAssumptions:
    """The complete bundle of assumptions used by one ROI analysis."""
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    financial: FinancialAssumptions = field(default_factory=FinancialAssumptions)
    service_value: ServiceValue = field(default_factory=ServiceValue)
    electric_fleet: ElectricFleetAssumptions = field(default_factory=ElectricFleetAssumptions)
    diesel: DieselAssumptions = field(default_factory=DieselAssumptions)
    farm_pv: FarmPVAssumptions = field(default_factory=FarmPVAssumptions)
    roof_pv: RoofPVAssumptions = field(default_factory=RoofPVAssumptions)
    outages: OutageAssumptions = field(default_factory=OutageAssumptions)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ROIAssumptions":
        roi = (config or {}).get("roi", {}) or {}
        return cls(
            analysis=AnalysisConfig.from_config(roi),
            financial=FinancialAssumptions.from_config(roi),
            service_value=ServiceValue.from_config(roi),
            electric_fleet=ElectricFleetAssumptions.from_config(roi),
            diesel=DieselAssumptions.from_config(roi),
            farm_pv=FarmPVAssumptions.from_config(roi),
            roof_pv=RoofPVAssumptions.from_config(roi),
            outages=OutageAssumptions.from_config(roi),
            sensitivity=SensitivityConfig.from_config(roi),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis": asdict(self.analysis),
            "financial": asdict(self.financial),
            "service_value": asdict(self.service_value),
            "electric_fleet": asdict(self.electric_fleet),
            "diesel": asdict(self.diesel),
            "farm_pv": asdict(self.farm_pv),
            "roof_pv": asdict(self.roof_pv),
            "outages": asdict(self.outages),
            "sensitivity": asdict(self.sensitivity),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Operational + result structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OperationalTotals:
    """Annualised operational aggregates for one config variant.

    All energy is kWh/yr, costs are EUR/yr, distances km/yr, hours hr/yr.
    ``profile`` holds a representative 15-minute series (list of dicts) used by
    the reliability model.  ``source`` records how the numbers were produced.
    """
    variant: str = ""
    grid_kwh: float = 0.0
    grid_farm_kwh: float = 0.0
    grid_tractor_kwh: float = 0.0
    grid_cost_eur: float = 0.0
    tractor_charge_cost_eur: float = 0.0
    tractor_charge_input_kwh: float = 0.0
    farm_pv_generated_kwh: float = 0.0
    tractor_pv_generated_kwh: float = 0.0
    pv_used_kwh: float = 0.0
    farm_pv_used_kwh: float = 0.0
    tractor_pv_used_kwh: float = 0.0
    exported_pv_kwh: float = 0.0
    curtailed_pv_kwh: float = 0.0
    transit_distance_km: float = 0.0
    work_distance_km: float = 0.0
    operating_hours: float = 0.0
    pto_hours: float = 0.0
    idle_hours: float = 0.0
    tasks_completed: float = 0.0
    tasks_missed: float = 0.0
    total_tasks: float = 0.0
    days_represented: int = 0
    simulations_run: int = 0
    profile: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("profile", None)   # profile is large; not part of the JSON summary
        return d


@dataclass
class CashflowRow:
    year: int
    baseline_operating_cost: float = 0.0
    candidate_operating_cost: float = 0.0
    fuel_cost: float = 0.0
    electricity_cost: float = 0.0
    maintenance: float = 0.0
    outage_loss: float = 0.0
    revenue: float = 0.0
    replacement_cost: float = 0.0
    net_cash_flow: float = 0.0
    discount_factor: float = 1.0
    discounted_cash_flow: float = 0.0
    cumulative_cash_flow: float = 0.0
    cumulative_discounted_cash_flow: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


@dataclass
class InvestmentResult:
    id: str
    name: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    annual_cashflows: List[CashflowRow] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "metrics": self.metrics,
            "annual_cashflows": [r.to_dict() for r in self.annual_cashflows],
            "warnings": self.warnings,
        }
