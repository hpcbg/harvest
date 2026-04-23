from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math
import copy

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from task_generator import generate_tasks


# ============================================================
# Utilities
# ============================================================

def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def parse_hhmm(value: str) -> time:
    hh, mm = map(int, value.split(":"))
    return time(hour=hh, minute=mm)


def euclidean_distance_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt(v: Any, decimals: int = 2) -> str:
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


# ============================================================
# Data models
# ============================================================

@dataclass
class Task:
    task_id: str
    name: str
    location: Tuple[float, float]
    earliest_start: datetime
    latest_finish: datetime
    duration_minutes: int
    distance_km: float
    priority: str
    can_wait: bool
    uses_pto: bool
    pto_power_kw: float
    assigned_tractor_id: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    is_active: bool = False
    is_done: bool = False

    def duration_hours(self) -> float:
        return self.duration_minutes / 60.0


@dataclass
class Charger:
    charger_id: str
    location: Tuple[float, float]
    max_power_kw: float


@dataclass
class EnergyConsumer:
    consumer_id: str
    power_kw: float
    priority: str          # critical | high | normal | low
    start: Optional[time] = None
    end: Optional[time] = None
    always_on: bool = False

    def is_active(self, now: datetime, shed_low: bool = False) -> bool:
        """Return True if consumer is drawing power at this moment.

        When shed_low=True, 'low' and 'normal' priority non-critical loads
        are suppressed to reduce grid demand.
        """
        if shed_low and self.priority in ("low", "normal") and not self.always_on:
            return False

        if self.always_on:
            return True
        if self.start is None or self.end is None:
            return False
        current = now.time()
        # Handle midnight-crossing windows (e.g. 22:00 – 06:00)
        if self.start <= self.end:
            return self.start <= current < self.end
        return current >= self.start or current < self.end


@dataclass
class TractorModel:
    name: str
    battery_capacity_kwh: float
    swappable_capacity_kwh: float
    module_capacity_kwh: float
    modules_total: int
    modules_swappable: int
    max_charge_power_kw: float
    full_charge_time_h: float
    nominal_power_kw: float
    max_power_kw: float
    pto_power_kw: float
    max_speed_kmh: float
    eco_speed_kmh: float
    charging_efficiency: float
    driving_kwh_per_km: float
    idle_kwh_per_h: float


@dataclass
class TractorPVConfig:
    panel_peak_w: float      # Wp per tractor
    field_derating: float    # fraction of peak when working
    parked_derating: float   # fraction of peak when parked / charging


@dataclass
class Tractor:
    tractor_id: str
    soc_percent: float
    location: Tuple[float, float]
    enabled: bool
    has_pv_roof: bool = False
    current_task_id: Optional[str] = None
    is_charging: bool = False
    assigned_charger_id: Optional[str] = None
    requested_charge_power_kw: float = 0.0
    actual_charge_power_kw: float = 0.0
    battery_swaps_count: int = 0
    # accumulated downtime (hours spent idle waiting for charge or tasks)
    idle_hours: float = 0.0

    def battery_kwh(self, model: TractorModel) -> float:
        return (self.soc_percent / 100.0) * model.battery_capacity_kwh

    def set_battery_kwh(self, model: TractorModel, value_kwh: float) -> None:
        self.soc_percent = clamp(100.0 * value_kwh / model.battery_capacity_kwh, 0.0, 100.0)

    def pv_output_kw(self, pv_cfg: TractorPVConfig, pv_shape: float) -> float:
        """Instantaneous roof PV output in kW, given normalised solar irradiance 0-1."""
        if not self.has_pv_roof:
            return 0.0
        peak_kw = pv_cfg.panel_peak_w / 1000.0
        derating = pv_cfg.parked_derating if self.current_task_id is None else pv_cfg.field_derating
        return peak_kw * pv_shape * derating


@dataclass
class ScenarioDef:
    name: str
    charging_strategy: str   # naive | night_only | smart | smart_with_swap
    tractor_pv_enabled: bool
    load_shedding: bool


@dataclass
class SimulationConfig:
    start_time: datetime
    end_time: datetime
    step_minutes: int
    grid_max_power_kw: float
    tractors_model: TractorModel
    tractors: List[Tractor]
    chargers: List[Charger]
    tasks: List[Task]
    consumers: List[EnergyConsumer]
    pv_profile: Dict[int, float]          # normalised 0-1 hourly shape
    farm_fixed_peak_kw: float             # farm array installed capacity
    tractor_pv_cfg: TractorPVConfig
    objective_weights: Dict[str, float]
    scenario_def: ScenarioDef


@dataclass
class StepMetrics:
    timestamp: datetime
    scenario: str
    # Power flows
    farm_fixed_pv_kw: float
    tractor_pv_kw: float
    total_pv_kw: float
    farm_load_kw: float           # non-tractor consumers
    tractor_charge_kw: float
    total_demand_kw: float        # farm_load + tractor_charge
    grid_kw: float                # max(0, demand - total_pv)
    # Energy / cost accumulators (per step)
    grid_energy_kwh: float
    pv_energy_used_kwh: float
    cost_eur: float
    # Task state
    completed_tasks: int
    missed_tasks: int
    # Fleet state
    average_soc_percent: float
    tractors_charging: int
    tractors_working: int
    tractors_idle: int


# ============================================================
# Tariff
# ============================================================

class TariffModel:
    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self.energy_periods = config_dict["tariffs"]["energy_periods"]

    @staticmethod
    def _time_in_interval(t: time, start_str: str, end_str: str) -> bool:
        start = parse_hhmm(start_str)
        end = parse_hhmm(end_str)
        if start <= end:
            return start <= t < end
        return t >= start or t < end

    def get_energy_price(self, now: datetime) -> float:
        if now.weekday() >= 5:
            valle = self.energy_periods.get("valle", {})
            if valle.get("weekend_all_day_valle", False):
                return float(valle["price_eur_per_kwh"])

        for _, period_cfg in self.energy_periods.items():
            for interval in period_cfg.get("intervals", []):
                if self._time_in_interval(now.time(), interval["start"], interval["end"]):
                    return float(period_cfg["price_eur_per_kwh"])

        return float(self.energy_periods["llano"]["price_eur_per_kwh"])


# ============================================================
# PV model
# ============================================================

class PVModel:
    """Handles both the fixed farm array and per-tractor roof panels."""

    def __init__(self, normalised_profile: Dict[int, float], farm_fixed_peak_kw: float) -> None:
        self.profile = normalised_profile
        self.farm_fixed_peak_kw = farm_fixed_peak_kw

    def shape_at(self, now: datetime) -> float:
        """Normalised irradiance 0-1 for this hour."""
        return float(self.profile.get(now.hour, 0.0))

    def farm_fixed_kw(self, now: datetime) -> float:
        return self.farm_fixed_peak_kw * self.shape_at(now)

    def tractor_fleet_kw(
        self,
        tractors: List[Tractor],
        pv_cfg: TractorPVConfig,
        now: datetime,
        enabled: bool,
    ) -> float:
        if not enabled:
            return 0.0
        shape = self.shape_at(now)
        return sum(tr.pv_output_kw(pv_cfg, shape) for tr in tractors if tr.enabled)


# ============================================================
# Scheduler
# ============================================================

class Scheduler:
    def __init__(self, config: SimulationConfig) -> None:
        self.cfg = config

    def assign_tasks(self, now: datetime) -> None:
        pending = [
            t for t in self.cfg.tasks
            if not t.is_done
            and t.assigned_tractor_id is None
            and now >= t.earliest_start
            and now <= t.latest_finish
        ]
        priority_order = {"urgent": 0, "normal": 1, "flexible": 2}
        pending.sort(key=lambda x: (priority_order.get(x.priority, 99), x.latest_finish))

        free = [
            tr for tr in self.cfg.tractors
            if tr.enabled and tr.current_task_id is None and not tr.is_charging
        ]

        for task in pending:
            best, best_score = None, float("inf")
            for tr in free:
                required_kwh = self._task_energy(task)
                if tr.battery_kwh(self.cfg.tractors_model) < required_kwh + 3.0:
                    continue
                dist_m = euclidean_distance_m(tr.location, task.location)
                # Penalise low-SOC tractors so they are kept for urgent tasks
                soc_penalty = max(0.0, (50.0 - tr.soc_percent)) * 5.0
                score = dist_m + soc_penalty
                if score < best_score:
                    best_score, best = score, tr
            if best:
                task.assigned_tractor_id = best.tractor_id
                best.current_task_id = task.task_id
                free.remove(best)

    def _task_energy(self, task: Task) -> float:
        model = self.cfg.tractors_model
        return (task.distance_km * model.driving_kwh_per_km
                + (task.duration_hours() * task.pto_power_kw if task.uses_pto else 0.0))

    def maybe_swap_battery(self, now: datetime) -> None:
        strategy = self.cfg.scenario_def.charging_strategy
        if strategy not in {"smart", "smart_with_swap"}:
            return
        urgent_pending = [
            t for t in self.cfg.tasks
            if not t.is_done and t.assigned_tractor_id is None
            and t.priority == "urgent" and now <= t.latest_finish
        ]
        if not urgent_pending:
            return
        if strategy == "smart_with_swap":
            for tr in self.cfg.tractors:
                if tr.current_task_id is None and tr.soc_percent < 20.0:
                    tr.soc_percent = min(80.0, tr.soc_percent + 50.0)
                    tr.battery_swaps_count += 1
                    break

    def allocate_charging(self, now: datetime, net_available_kw: float) -> None:
        """Assign chargers to idle tractors based on active strategy."""
        for tr in self.cfg.tractors:
            tr.is_charging = False
            tr.assigned_charger_id = None
            tr.requested_charge_power_kw = 0.0
            tr.actual_charge_power_kw = 0.0

        idle = [
            tr for tr in self.cfg.tractors
            if tr.enabled and tr.current_task_id is None and tr.soc_percent < 90.0
        ]
        strategy = self.cfg.scenario_def.charging_strategy

        if strategy == "naive":
            self._charge_naive(idle)
        elif strategy == "night_only":
            self._charge_night_only(now, idle)
        else:
            self._charge_smart(now, idle, net_available_kw)

    # ── Charging strategies ─────────────────────────────────────────────────

    def _charge_naive(self, idle: List[Tractor]) -> None:
        chargers = list(self.cfg.chargers)
        for tr in idle:
            if not chargers:
                break
            ch = chargers.pop(0)
            req = min(ch.max_power_kw, self.cfg.tractors_model.max_charge_power_kw)
            tr.is_charging = True
            tr.assigned_charger_id = ch.charger_id
            tr.requested_charge_power_kw = req
            tr.actual_charge_power_kw = req

    def _charge_night_only(self, now: datetime, idle: List[Tractor]) -> None:
        if not (now.hour >= 22 or now.hour < 8):
            return
        available = self.cfg.grid_max_power_kw
        chargers = list(self.cfg.chargers)
        idle.sort(key=lambda tr: tr.soc_percent)
        for tr in idle:
            if not chargers or available <= 0.5:
                break
            ch = chargers.pop(0)
            req = min(ch.max_power_kw, self.cfg.tractors_model.max_charge_power_kw)
            alloc = min(req, available)
            tr.is_charging = True
            tr.assigned_charger_id = ch.charger_id
            tr.requested_charge_power_kw = req
            tr.actual_charge_power_kw = alloc
            available -= alloc

    def _charge_smart(self, now: datetime, idle: List[Tractor], net_available_kw: float) -> None:
        available = max(0.0, net_available_kw)
        chargers = list(self.cfg.chargers)

        def priority_key(tr: Tractor) -> Tuple[int, float]:
            urgent_flag = 1
            nearest_deadline = 1e9
            for task in self.cfg.tasks:
                if task.is_done or task.assigned_tractor_id is not None:
                    continue
                if now <= task.latest_finish:
                    remaining = (task.latest_finish - now).total_seconds() / 60.0
                    nearest_deadline = min(nearest_deadline, remaining)
                    if task.priority == "urgent":
                        urgent_flag = 0
            return (urgent_flag, tr.soc_percent + 0.001 * nearest_deadline)

        idle.sort(key=priority_key)
        for tr in idle:
            if not chargers or available <= 0.5:
                break
            ch = chargers.pop(0)
            req = min(ch.max_power_kw, self.cfg.tractors_model.max_charge_power_kw)
            alloc = min(req, available)
            if alloc <= 0.5:
                continue
            tr.is_charging = True
            tr.assigned_charger_id = ch.charger_id
            tr.requested_charge_power_kw = req
            tr.actual_charge_power_kw = alloc
            available -= alloc


# ============================================================
# Simulator
# ============================================================

class Simulator:
    def __init__(self, config_dict: Dict[str, Any], scenario_def: ScenarioDef) -> None:
        self.raw = copy.deepcopy(config_dict)
        self.scenario_def = scenario_def
        self.config = self._build_config()
        self.scheduler = Scheduler(self.config)
        self.tariff = TariffModel(config_dict)
        self.pv_model = PVModel(
            normalised_profile={int(k): float(v) for k, v in self.raw["pv"]["profile"].items()},
            farm_fixed_peak_kw=float(self.raw["pv"]["farm_fixed_peak_kw"]),
        )
        self.metrics: List[StepMetrics] = []

    # ── Config builder ──────────────────────────────────────────────────────

    def _build_config(self) -> SimulationConfig:
        cfg = self.raw
        mc = cfg["tractors"]["model"]
        model = TractorModel(
            name=mc["name"],
            battery_capacity_kwh=float(mc["battery_capacity_kwh"]),
            swappable_capacity_kwh=float(mc["swappable_capacity_kwh"]),
            module_capacity_kwh=float(mc["module_capacity_kwh"]),
            modules_total=int(mc["modules_total"]),
            modules_swappable=int(mc["modules_swappable"]),
            max_charge_power_kw=float(mc["max_charge_power_kw"]),
            full_charge_time_h=float(mc["full_charge_time_h"]),
            nominal_power_kw=float(mc["nominal_power_kw"]),
            max_power_kw=float(mc["max_power_kw"]),
            pto_power_kw=float(mc["pto_power_kw"]),
            max_speed_kmh=float(mc["max_speed_kmh"]),
            eco_speed_kmh=float(mc["eco_speed_kmh"]),
            charging_efficiency=float(mc["charging_efficiency"]),
            driving_kwh_per_km=float(mc["driving_kwh_per_km"]),
            idle_kwh_per_h=float(mc["idle_kwh_per_h"]),
        )

        tpv_raw = cfg.get("tractor_pv", {})
        tractor_pv_cfg = TractorPVConfig(
            panel_peak_w=float(tpv_raw.get("panel_peak_w", 650)),
            field_derating=float(tpv_raw.get("field_derating", 0.70)),
            parked_derating=float(tpv_raw.get("parked_derating", 0.95)),
        )

        tractors = [
            Tractor(
                tractor_id=tr["id"],
                soc_percent=float(tr["initial_soc_percent"]),
                location=(float(tr["initial_location"]["x"]), float(tr["initial_location"]["y"])),
                enabled=bool(tr["enabled"]),
                has_pv_roof=bool(tr.get("has_pv_roof", False)),
            )
            for tr in cfg["tractors"]["fleet"]
        ]

        chargers = [
            Charger(
                charger_id=ch["id"],
                location=(float(ch["location"]["x"]), float(ch["location"]["y"])),
                max_power_kw=float(ch["max_power_kw"]),
            )
            for ch in cfg["charging"]["stations"]
        ]

        tg = cfg.get("task_generation", {})
        if tg.get("mode", "static") == "generated":
            task_items = generate_tasks(
                start_date=cfg["simulation"]["start_time"],
                num_tasks=int(tg.get("num_tasks", 12)),
                seed=int(tg.get("seed", 42)),
                map_width_m=float(cfg["farm"]["map"]["width_m"]),
                map_height_m=float(cfg["farm"]["map"]["height_m"]),
            )
        else:
            task_items = cfg.get("tasks", [])

        tasks = [
            Task(
                task_id=t.get("id", t.get("task_id")),
                name=t["name"],
                location=(float(t["location"]["x"]), float(t["location"]["y"])),
                earliest_start=parse_dt(t["earliest_start"]),
                latest_finish=parse_dt(t["latest_finish"]),
                duration_minutes=int(t["duration_minutes"]),
                distance_km=float(t["distance_km"]),
                priority=str(t["priority"]),
                can_wait=bool(t["can_wait"]),
                uses_pto=bool(t["uses_pto"]),
                pto_power_kw=float(t["pto_power_kw"]),
            )
            for t in task_items
        ]

        consumers: List[EnergyConsumer] = []
        for c in cfg.get("energy_consumers", []):
            schedule = c.get("schedule")
            consumers.append(EnergyConsumer(
                consumer_id=c["id"],
                power_kw=float(c["power_kw"]),
                priority=str(c["priority"]),
                start=parse_hhmm(schedule["start"]) if schedule else None,
                end=parse_hhmm(schedule["end"]) if schedule else None,
                always_on=bool(c.get("always_on", False)),
            ))

        return SimulationConfig(
            start_time=parse_dt(cfg["simulation"]["start_time"]),
            end_time=parse_dt(cfg["simulation"]["end_time"]),
            step_minutes=int(cfg["simulation"]["time_step_minutes"]),
            grid_max_power_kw=float(cfg["grid"]["max_power_kw"]),
            tractors_model=model,
            tractors=tractors,
            chargers=chargers,
            tasks=tasks,
            consumers=consumers,
            pv_profile={int(k): float(v) for k, v in cfg["pv"]["profile"].items()},
            farm_fixed_peak_kw=float(cfg["pv"]["farm_fixed_peak_kw"]),
            tractor_pv_cfg=tractor_pv_cfg,
            objective_weights=cfg["scheduler"]["objective_weights"],
            scenario_def=self.scenario_def,
        )

    # ── Main loop ───────────────────────────────────────────────────────────

    def run(self) -> List[StepMetrics]:
        now = self.config.start_time
        dt_h = self.config.step_minutes / 60.0
        shed = self.scenario_def.load_shedding
        pv_roof_on = self.scenario_def.tractor_pv_enabled

        while now < self.config.end_time:
            # --- PV generation ---
            farm_fixed_kw = self.pv_model.farm_fixed_kw(now)
            tractor_pv_kw = self.pv_model.tractor_fleet_kw(
                self.config.tractors, self.config.tractor_pv_cfg, now, pv_roof_on
            )
            total_pv_kw = farm_fixed_kw + tractor_pv_kw

            # --- Farm load (non-tractor consumers) ---
            farm_load_kw = sum(
                c.power_kw for c in self.config.consumers if c.is_active(now, shed_low=shed)
            )

            # --- Headroom available for charging ---
            # grid cap + PV surplus - farm load
            net_available_kw = self.config.grid_max_power_kw + total_pv_kw - farm_load_kw

            # --- Scheduling ---
            self.scheduler.maybe_swap_battery(now)
            self.scheduler.assign_tasks(now)
            self._progress_tasks(now, dt_h, pv_roof_on)
            self.scheduler.allocate_charging(now, net_available_kw)
            self._apply_charging(dt_h)
            self._accumulate_idle(dt_h)

            # --- Power accounting ---
            tractor_charge_kw = sum(tr.actual_charge_power_kw for tr in self.config.tractors)
            total_demand_kw = farm_load_kw + tractor_charge_kw
            grid_kw = max(0.0, total_demand_kw - total_pv_kw)
            pv_used_kwh = min(total_demand_kw, total_pv_kw) * dt_h
            grid_energy_kwh = grid_kw * dt_h
            cost_eur = grid_energy_kwh * self.tariff.get_energy_price(now)

            # --- Task & fleet counters ---
            completed = sum(1 for t in self.config.tasks if t.is_done)
            missed = sum(1 for t in self.config.tasks if not t.is_done and now > t.latest_finish)
            avg_soc = (sum(tr.soc_percent for tr in self.config.tractors)
                       / max(1, len(self.config.tractors)))
            n_charging = sum(1 for tr in self.config.tractors if tr.is_charging)
            n_working = sum(1 for tr in self.config.tractors if tr.current_task_id is not None)
            n_idle = len(self.config.tractors) - n_charging - n_working

            self.metrics.append(StepMetrics(
                timestamp=now,
                scenario=self.scenario_def.name,
                farm_fixed_pv_kw=farm_fixed_kw,
                tractor_pv_kw=tractor_pv_kw,
                total_pv_kw=total_pv_kw,
                farm_load_kw=farm_load_kw,
                tractor_charge_kw=tractor_charge_kw,
                total_demand_kw=total_demand_kw,
                grid_kw=grid_kw,
                grid_energy_kwh=grid_energy_kwh,
                pv_energy_used_kwh=pv_used_kwh,
                cost_eur=cost_eur,
                completed_tasks=completed,
                missed_tasks=missed,
                average_soc_percent=avg_soc,
                tractors_charging=n_charging,
                tractors_working=n_working,
                tractors_idle=n_idle,
            ))

            now += timedelta(minutes=self.config.step_minutes)

        return self.metrics

    # ── Step helpers ────────────────────────────────────────────────────────

    def _progress_tasks(self, now: datetime, dt_h: float, pv_roof_on: bool) -> None:
        model = self.config.tractors_model
        tasks_by_id = {t.task_id: t for t in self.config.tasks}

        for tr in self.config.tractors:
            if tr.current_task_id is None:
                # Idle: drain at idle rate, offset by roof PV if enabled
                shape = self.pv_model.shape_at(now)
                roof_kw = tr.pv_output_kw(self.config.tractor_pv_cfg, shape) if pv_roof_on else 0.0
                net_drain = max(0.0, model.idle_kwh_per_h * dt_h - roof_kw * dt_h)
                battery = tr.battery_kwh(model) - net_drain
                tr.set_battery_kwh(model, battery)
                continue

            task = tasks_by_id[tr.current_task_id]
            if task.started_at is None:
                task.started_at = now
                task.is_active = True

            elapsed_h = (now - task.started_at).total_seconds() / 3600.0
            total_h = task.duration_hours()

            total_energy = task.distance_km * model.driving_kwh_per_km
            if task.uses_pto:
                total_energy += task.duration_hours() * task.pto_power_kw
            per_hour_drain = total_energy / max(total_h, 1e-6)

            # Tractor roof PV partially offsets field energy draw
            shape = self.pv_model.shape_at(now)
            roof_kw = tr.pv_output_kw(self.config.tractor_pv_cfg, shape) if pv_roof_on else 0.0
            net_per_hour = max(0.0, per_hour_drain - roof_kw)

            battery = tr.battery_kwh(model) - net_per_hour * dt_h
            tr.set_battery_kwh(model, battery)
            tr.location = task.location

            if elapsed_h + dt_h >= total_h:
                task.finished_at = now + timedelta(hours=dt_h)
                task.is_done = True
                task.is_active = False
                tr.current_task_id = None

    def _apply_charging(self, dt_h: float) -> None:
        model = self.config.tractors_model
        for tr in self.config.tractors:
            if tr.is_charging and tr.actual_charge_power_kw > 0:
                added = tr.actual_charge_power_kw * model.charging_efficiency * dt_h
                tr.set_battery_kwh(model, tr.battery_kwh(model) + added)

    def _accumulate_idle(self, dt_h: float) -> None:
        for tr in self.config.tractors:
            if tr.enabled and tr.current_task_id is None and not tr.is_charging:
                tr.idle_hours += dt_h

    # ── Summary & export ────────────────────────────────────────────────────

    def summarize(self) -> Dict[str, Any]:
        total_grid_kwh = sum(m.grid_energy_kwh for m in self.metrics)
        total_pv_kwh = sum(m.pv_energy_used_kwh for m in self.metrics)
        total_demand_kwh = sum(m.total_demand_kw * (self.config.step_minutes / 60.0)
                               for m in self.metrics)
        total_cost = sum(m.cost_eur for m in self.metrics)
        peak_grid_kw = max((m.grid_kw for m in self.metrics), default=0.0)
        total_farm_pv_gen = sum(
            m.farm_fixed_pv_kw * (self.config.step_minutes / 60.0) for m in self.metrics
        )
        total_tractor_pv_gen = sum(
            m.tractor_pv_kw * (self.config.step_minutes / 60.0) for m in self.metrics
        )

        completed = self.metrics[-1].completed_tasks if self.metrics else 0
        missed = self.metrics[-1].missed_tasks if self.metrics else 0
        total_tasks = len(self.config.tasks)

        total_possible_tractor_h = (
            len(self.config.tractors)
            * (self.config.end_time - self.config.start_time).total_seconds() / 3600.0
        )
        total_idle_h = sum(tr.idle_hours for tr in self.config.tractors)
        downtime_pct = 100.0 * total_idle_h / max(1.0, total_possible_tractor_h)

        pv_share_pct = 100.0 * total_pv_kwh / max(1e-6, total_demand_kwh)
        swaps = sum(tr.battery_swaps_count for tr in self.config.tractors)

        return {
            "scenario": self.scenario_def.name,
            "charging_strategy": self.scenario_def.charging_strategy,
            "tractor_pv": self.scenario_def.tractor_pv_enabled,
            "load_shedding": self.scenario_def.load_shedding,
            # Task KPIs
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "missed_tasks": missed,
            "task_completion_pct": round(100.0 * completed / max(1, total_tasks), 1),
            # Energy KPIs
            "total_demand_kwh": round(total_demand_kwh, 2),
            "total_grid_kwh": round(total_grid_kwh, 2),
            "total_pv_used_kwh": round(total_pv_kwh, 2),
            "farm_pv_generated_kwh": round(total_farm_pv_gen, 2),
            "tractor_pv_generated_kwh": round(total_tractor_pv_gen, 2),
            "pv_self_use_share_pct": round(pv_share_pct, 1),
            "peak_grid_kw": round(peak_grid_kw, 2),
            # Cost KPIs
            "total_cost_eur": round(total_cost, 2),
            "cost_per_completed_task_eur": round(total_cost / max(1, completed), 2),
            # Fleet KPIs
            "tractor_downtime_pct": round(downtime_pct, 1),
            "battery_swaps": swaps,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "timestamp": m.timestamp,
            "scenario": m.scenario,
            "farm_fixed_pv_kw": m.farm_fixed_pv_kw,
            "tractor_pv_kw": m.tractor_pv_kw,
            "total_pv_kw": m.total_pv_kw,
            "farm_load_kw": m.farm_load_kw,
            "tractor_charge_kw": m.tractor_charge_kw,
            "total_demand_kw": m.total_demand_kw,
            "grid_kw": m.grid_kw,
            "grid_energy_kwh": m.grid_energy_kwh,
            "pv_energy_used_kwh": m.pv_energy_used_kwh,
            "cost_eur": m.cost_eur,
            "completed_tasks": m.completed_tasks,
            "missed_tasks": m.missed_tasks,
            "average_soc_percent": m.average_soc_percent,
            "tractors_charging": m.tractors_charging,
            "tractors_working": m.tractors_working,
            "tractors_idle": m.tractors_idle,
        } for m in self.metrics])

    def task_schedule_dataframe(self) -> pd.DataFrame:
        rows = [{
            "task_id": t.task_id,
            "name": t.name,
            "priority": t.priority,
            "uses_pto": t.uses_pto,
            "earliest_start": t.earliest_start,
            "latest_finish": t.latest_finish,
            "duration_minutes": t.duration_minutes,
            "distance_km": t.distance_km,
            "assigned_tractor": t.assigned_tractor_id,
            "started_at": t.started_at,
            "finished_at": t.finished_at,
            "is_done": t.is_done,
        } for t in self.config.tasks]
        return pd.DataFrame(rows).sort_values("earliest_start").reset_index(drop=True)


# ============================================================
# Plots
# ============================================================

PALETTE = {
    "grid":        "#e05c5c",
    "pv_farm":     "#f5c842",
    "pv_tractor":  "#f59842",
    "farm_load":   "#5c8ae0",
    "charge":      "#5ce07a",
    "soc":         "#9b5ce0",
    "cost":        "#e05ca0",
    "completed":   "#3ab56e",
    "missed":      "#e05c5c",
}


def plot_scenario_power(df: pd.DataFrame, out_dir: Path, name: str,
                        task_df: Optional[pd.DataFrame] = None) -> None:
    s = df[df["scenario"] == name].copy()

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 1, figure=fig, hspace=0.45)

    # --- Top: power flows ---
    ax1 = fig.add_subplot(gs[0])
    ax1.stackplot(s["timestamp"],
                  s["farm_fixed_pv_kw"], s["tractor_pv_kw"],
                  labels=["Farm PV (kW)", "Tractor roof PV (kW)"],
                  colors=[PALETTE["pv_farm"], PALETTE["pv_tractor"]], alpha=0.7)
    ax1.plot(s["timestamp"], s["grid_kw"], color=PALETTE["grid"], lw=1.5, label="Grid draw (kW)")
    ax1.plot(s["timestamp"], s["total_demand_kw"], color=PALETTE["charge"],
             lw=1.2, ls="--", label="Total demand (kW)")
    ax1.set_ylabel("kW")
    ax1.set_title(f"Power flows — {name}")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Middle: SOC + fleet state ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(s["timestamp"], s["average_soc_percent"], color=PALETTE["soc"], lw=1.5, label="Avg SOC (%)")
    ax2.axhline(20, color="red", ls=":", lw=0.8, label="Low (20%)")
    ax2.axhline(90, color="green", ls=":", lw=0.8, label="Full (90%)")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("%")
    ax2.set_title("Fleet average SOC")
    ax2b = ax2.twinx()
    ax2b.bar(s["timestamp"], s["tractors_working"], width=0.008, color="#5c8ae0", alpha=0.4, label="Working")
    ax2b.bar(s["timestamp"], s["tractors_charging"], width=0.008, color=PALETTE["charge"], alpha=0.4,
             bottom=s["tractors_working"], label="Charging")
    ax2b.bar(s["timestamp"], s["tractors_idle"], width=0.008, color="grey", alpha=0.3,
             bottom=s["tractors_working"] + s["tractors_charging"], label="Idle")
    ax2b.set_ylabel("Tractors")
    ax2b.set_ylim(0, len(df["tractors_working"].unique()) * 3)
    lines2, labels2 = ax2.get_legend_handles_labels()
    bars2, blabels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines2 + bars2, labels2 + blabels2, loc="upper left", fontsize=7)
    ax2.grid(True, alpha=0.3)

    # --- Bottom: cumulative cost + task completion ---
    ax3 = fig.add_subplot(gs[2])
    cumcost = s["cost_eur"].cumsum()
    ax3.plot(s["timestamp"], cumcost, color=PALETTE["cost"], lw=1.5, label="Cumulative cost (€)")
    ax3.set_ylabel("€")
    ax3.set_title("Cumulative energy cost & tasks")
    ax3b = ax3.twinx()
    if task_df is not None and not task_df.empty:
        ax3b.step(s["timestamp"], s["completed_tasks"], where="post",
                  color=PALETTE["completed"], lw=1.2, label="Completed tasks")
        ax3b.step(s["timestamp"], s["missed_tasks"], where="post",
                  color=PALETTE["missed"], lw=1.2, ls=":", label="Missed tasks")
    ax3b.set_ylabel("Tasks")
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines3b, labels3b = ax3b.get_legend_handles_labels()
    ax3.legend(lines3 + lines3b, labels3 + labels3b, loc="upper left", fontsize=7)
    ax3.grid(True, alpha=0.3)

    fig.savefig(out_dir / f"{name}_detail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kpi_comparison(summaries: List[Dict[str, Any]], out_dir: Path) -> None:
    df = pd.DataFrame(summaries)
    scenarios = df["scenario"].tolist()
    x = list(range(len(scenarios)))
    w = 0.38

    def bar_chart(ax, values, title, ylabel, color="#5c8ae0", fmt_str="{:.1f}"):
        bars = ax.bar(x, values, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                    fmt_str.format(val), ha="center", va="bottom", fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.suptitle("Scenario KPI Comparison", fontsize=14, fontweight="bold", y=1.01)

    # Row 0 – Energy
    bar_chart(axes[0, 0], df["total_grid_kwh"], "Grid energy consumed", "kWh", "#e05c5c")
    bar_chart(axes[0, 1], df["total_pv_used_kwh"], "PV energy used on-site", "kWh", PALETTE["pv_farm"])
    bar_chart(axes[0, 2], df["pv_self_use_share_pct"], "PV self-use share", "%", PALETTE["pv_tractor"],
              fmt_str="{:.1f}")

    # Row 1 – Cost & power
    bar_chart(axes[1, 0], df["total_cost_eur"], "Total energy cost", "€", PALETTE["cost"])
    bar_chart(axes[1, 1], df["cost_per_completed_task_eur"], "Cost per completed task", "€/task",
              "#c05c9b")
    bar_chart(axes[1, 2], df["peak_grid_kw"], "Peak grid draw", "kW", "#e07a5c")

    # Row 2 – Tasks & fleet
    # Grouped: completed vs missed
    ax = axes[2, 0]
    bars_c = ax.bar([xi - w / 2 for xi in x], df["completed_tasks"], w,
                    label="Completed", color=PALETTE["completed"], alpha=0.85, edgecolor="white")
    bars_m = ax.bar([xi + w / 2 for xi in x], df["missed_tasks"], w,
                    label="Missed", color=PALETTE["missed"], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Tasks", fontsize=9)
    ax.set_title("Task completion", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    bar_chart(axes[2, 1], df["task_completion_pct"], "Task completion rate", "%",
              PALETTE["completed"], fmt_str="{:.1f}")
    bar_chart(axes[2, 2], df["tractor_downtime_pct"], "Tractor idle / downtime", "%",
              "grey", fmt_str="{:.1f}")

    fig.tight_layout()
    fig.savefig(out_dir / "kpi_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pv_breakdown(summaries: List[Dict[str, Any]], out_dir: Path) -> None:
    """Stacked bar: grid vs farm PV vs tractor PV energy for every scenario."""
    df = pd.DataFrame(summaries)
    scenarios = df["scenario"].tolist()
    x = list(range(len(scenarios)))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, df["total_grid_kwh"], label="Grid", color=PALETTE["grid"], alpha=0.85)
    ax.bar(x, df["farm_pv_generated_kwh"], bottom=0,
           label="Farm PV generated", color=PALETTE["pv_farm"], alpha=0.7)
    # show tractor PV on top of farm PV bar (it's additive generation)
    ax.bar(x, df["tractor_pv_generated_kwh"], bottom=df["farm_pv_generated_kwh"],
           label="Tractor roof PV generated", color=PALETTE["pv_tractor"], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.set_ylabel("kWh")
    ax.set_title("Energy source breakdown per scenario")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pv_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Console helpers
# ============================================================

def print_farm_banner(config: Dict[str, Any]) -> None:
    tg = config.get("task_generation", {})
    fleet = config["tractors"]["fleet"]
    chargers = config["charging"]["stations"]
    consumers = config.get("energy_consumers", [])
    pv = config["pv"]
    tpv = config.get("tractor_pv", {})
    grid_cap = config["grid"]["max_power_kw"]
    n_pv_roof = sum(1 for t in fleet if t.get("has_pv_roof", False))

    farm_pv_peak = float(pv["farm_fixed_peak_kw"])
    panel_w = float(tpv.get("panel_peak_w", 0))
    tractor_pv_peak_kw = n_pv_roof * panel_w / 1000.0
    total_pv_peak = farm_pv_peak + tractor_pv_peak_kw

    always_on_kw = sum(c["power_kw"] for c in consumers if c.get("always_on", False))
    peak_consumer_kw = sum(c["power_kw"] for c in consumers)

    print(f"\n{'═' * 68}")
    print(f"  {config['project']['name']}  v{config['project']['version']}")
    print(f"{'═' * 68}")
    print(f"  {'Simulation window':<30} {config['simulation']['start_time']} → {config['simulation']['end_time']}")
    print(f"  {'Time step':<30} {config['simulation']['time_step_minutes']} min")
    print(f"{'─' * 68}")
    print(f"  {'Grid cap':<30} {grid_cap:.1f} kW")
    print(f"{'─' * 68}  PV")
    print(f"  {'Farm fixed array (peak)':<30} {farm_pv_peak:.1f} kW")
    print(f"  {'Tractor roof panels':<30} {n_pv_roof} × {panel_w:.0f} W = {tractor_pv_peak_kw:.2f} kW peak")
    print(f"  {'Total PV installed':<30} {total_pv_peak:.2f} kW peak")
    print(f"{'─' * 68}  Fleet")
    print(f"  {'Tractors':<30} {len(fleet)} (model: {config['tractors']['model']['name']})")
    print(f"  {'Chargers':<30} {len(chargers)} × {chargers[0]['max_power_kw']} kW")
    print(f"  {'Battery capacity':<30} {config['tractors']['model']['battery_capacity_kwh']} kWh / tractor")
    print(f"{'─' * 68}  Farm loads")
    print(f"  {'Always-on load':<30} {always_on_kw:.2f} kW")
    print(f"  {'Peak simultaneous load':<30} {peak_consumer_kw:.2f} kW")
    for c in consumers:
        sched = c.get("schedule", {})
        when = "always" if c.get("always_on") else f"{sched.get('start','?')}–{sched.get('end','?')}"
        print(f"    {c['id']:<32} {c['power_kw']:.1f} kW  [{c['priority']}]  {when}")
    print(f"{'─' * 68}  Tasks")
    print(f"  {'Mode':<30} {tg.get('mode','static')}")
    print(f"  {'Count':<30} {tg.get('num_tasks', len(config.get('tasks', [])))}  (seed={tg.get('seed','n/a')})")
    print(f"{'═' * 68}\n")


def print_scenario_result(summary: Dict[str, Any]) -> None:
    name = summary["scenario"]
    flags = []
    if summary["tractor_pv"]:
        flags.append("tractor-PV")
    if summary["load_shedding"]:
        flags.append("load-shed")
    flag_str = "  [" + ", ".join(flags) + "]" if flags else ""
    print(f"\n  ┌─ {name}  ({summary['charging_strategy']}){flag_str}")
    kpis = [
        ("Tasks completed", f"{summary['completed_tasks']}/{summary['total_tasks']}  ({summary['task_completion_pct']}%)"),
        ("Tasks missed", str(summary["missed_tasks"])),
        ("Grid energy", f"{summary['total_grid_kwh']} kWh"),
        ("PV self-use", f"{summary['total_pv_used_kwh']} kWh  ({summary['pv_self_use_share_pct']}% of demand)"),
        ("Farm PV gen.", f"{summary['farm_pv_generated_kwh']} kWh"),
        ("Tractor PV gen.", f"{summary['tractor_pv_generated_kwh']} kWh"),
        ("Peak grid draw", f"{summary['peak_grid_kw']} kW"),
        ("Total cost", f"€ {summary['total_cost_eur']}"),
        ("Cost / task done", f"€ {summary['cost_per_completed_task_eur']}"),
        ("Tractor downtime", f"{summary['tractor_downtime_pct']}%"),
        ("Battery swaps", str(summary["battery_swaps"])),
    ]
    for label, value in kpis:
        print(f"  │  {label:<26} {value}")
    print(f"  └{'─' * 50}")


def print_task_schedule(task_df: pd.DataFrame) -> None:
    W = 102
    print(f"\n{'─' * W}")
    print(f"{'ID':<12} {'Name':<26} {'Pri':<10} {'Start':<18} {'Deadline':<18} {'Dur(m)':<8} {'Assigned':<14} Done")
    print(f"{'─' * W}")
    for _, r in task_df.iterrows():
        done = "✓" if r["is_done"] else ("✗" if pd.isna(r["started_at"]) else "~")
        assigned = r["assigned_tractor"] if r["assigned_tractor"] else "—"
        print(f"{r['task_id']:<12} {r['name']:<26} {r['priority']:<10} "
              f"{str(r['earliest_start'])[:16]:<18} {str(r['latest_finish'])[:16]:<18} "
              f"{r['duration_minutes']:<8} {assigned:<14} {done}")
    print(f"{'─' * W}")


# ============================================================
# Config loader
# ============================================================

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Main
# ============================================================

def main() -> None:
    config_path = Path("config.yaml")
    outputs_dir = Path("outputs")
    ensure_dir(outputs_dir)

    config = load_yaml(config_path)
    print_farm_banner(config)

    # Build scenario definitions from config
    scenario_defs: List[ScenarioDef] = [
        ScenarioDef(
            name=s["name"],
            charging_strategy=s["charging_strategy"],
            tractor_pv_enabled=bool(s.get("tractor_pv_enabled", False)),
            load_shedding=bool(s.get("load_shedding", False)),
        )
        for s in config.get("scenarios", [])
    ]

    all_frames: List[pd.DataFrame] = []
    all_summaries: List[Dict[str, Any]] = []

    for sdef in scenario_defs:
        print(f"▶  Running: {sdef.name}  "
              f"(strategy={sdef.charging_strategy}, "
              f"tractor_pv={sdef.tractor_pv_enabled}, "
              f"shed={sdef.load_shedding}) ...", end="", flush=True)

        sim = Simulator(config, sdef)
        sim.run()

        df = sim.to_dataframe()
        summary = sim.summarize()
        task_df = sim.task_schedule_dataframe()

        all_frames.append(df)
        all_summaries.append(summary)

        save_dataframe_csv(df, outputs_dir / f"timeseries_{sdef.name}.csv")
        save_dataframe_csv(task_df, outputs_dir / f"task_schedule_{sdef.name}.csv")
        plot_scenario_power(df, outputs_dir, sdef.name, task_df)

        print(" done.")
        print_scenario_result(summary)
        print_task_schedule(task_df)

    # Combined outputs
    combined_df = pd.concat(all_frames, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)

    save_dataframe_csv(combined_df, outputs_dir / "timeseries_all_scenarios.csv")
    save_summary_csv(all_summaries, outputs_dir / "scenario_summary.csv")

    plot_kpi_comparison(all_summaries, outputs_dir)
    plot_pv_breakdown(all_summaries, outputs_dir)

    print(f"\n{'═' * 68}")
    print("  Outputs saved to ./outputs/")
    print("    timeseries_<scenario>.csv")
    print("    task_schedule_<scenario>.csv")
    print("    <scenario>_detail.png        (power / SOC / cost per scenario)")
    print("    kpi_comparison.png           (9-panel KPI dashboard)")
    print("    pv_breakdown.png             (energy source stacked bar)")
    print("    scenario_summary.csv")
    print(f"{'═' * 68}\n")


def save_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
