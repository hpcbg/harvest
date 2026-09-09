"""
Protocol-free farm physics for the device simulators.

Deliberately much simpler than HARVEST's ``main.Simulator``: the goal is
believable, reactive telemetry to exercise the protocol clients end-to-end,
not energy-accurate scheduling.  Time is accelerated (default 60x: one real
second is one simulated minute) so charging and tariff changes are visible in
a live demo.

Standalone -- imports nothing beyond the standard library, so the simulators
run in a container without HARVEST's numpy/pandas stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Normalised PV irradiance by hour (mirrors config.yaml's pv.profile default).
_DEFAULT_PV_PROFILE = {
    6: 0.02, 7: 0.04, 8: 0.16, 9: 0.36, 10: 0.6, 11: 0.84,
    12: 1.0, 13: 1.0, 14: 0.88, 15: 0.64, 16: 0.4, 17: 0.18, 18: 0.04,
}

# Spanish-style tariff windows (mirrors config.yaml's tariffs section).
_TARIFF_WINDOWS = [
    # (start_h, end_h, code)  0=valle 1=llano 2=punta
    (0, 8, 0), (8, 10, 1), (10, 14, 2), (14, 18, 1), (18, 22, 2), (22, 24, 1),
]
DEFAULT_PRICES = {0: 0.15, 1: 0.17, 2: 0.20}

LEVEL_FACTOR = {0: 0.0, 1: 0.5, 2: 1.0}
CHARGING_EFFICIENCY = 0.95


@dataclass
class TractorSim:
    id: str
    capacity_kwh: float = 44.8
    soc_pct: float = 70.0
    pos: Tuple[float, float] = (50.0, 50.0)
    available: bool = True
    charge_request: bool = False
    v2l_kw: float = 0.0
    docked_charger: Optional[str] = None

    @property
    def energy_kwh(self) -> float:
        return self.capacity_kwh * self.soc_pct / 100.0

    @property
    def charging(self) -> bool:
        return self.docked_charger is not None and self.charge_request and self.soc_pct < 100.0

    @property
    def discharging(self) -> bool:
        return self.docked_charger is not None and self.v2l_kw > 0.0


@dataclass
class ChargerSim:
    id: str
    rated_kw: float = 6.6
    level: int = 2                       # 0=off 1=half 2=full (writable)
    docked: Optional[str] = None
    power_kw: float = 0.0                # >0 charging, <0 V2L discharge


@dataclass
class LoadSim:
    id: str
    base_kw: float = 1.0
    shed: bool = False                   # writable
    always_on: bool = False
    start_h: float = 0.0
    end_h: float = 24.0
    power_kw: float = 0.0

    def active(self, hour: float) -> bool:
        if self.shed:
            return False
        if self.always_on:
            return True
        if self.start_h <= self.end_h:
            return self.start_h <= hour < self.end_h
        return hour >= self.start_h or hour < self.end_h   # midnight crossing


@dataclass
class FarmState:
    tractors: List[TractorSim] = field(default_factory=list)
    chargers: List[ChargerSim] = field(default_factory=list)
    loads: List[LoadSim] = field(default_factory=list)
    grid_cap_kw: float = 10.5
    pv_peak_kw: float = 5.0
    prices: Dict[int, float] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    clock_min: float = 8.0 * 60          # start mid-morning so PV/loads are alive
    pv_kw: float = 0.0
    grid_draw_kw: float = 0.0
    tariff_code: int = 1

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "FarmState":
        """Build from a HARVEST ``config.yaml`` dict; defaults where absent."""
        cfg = cfg or {}
        state = cls()
        state.grid_cap_kw = float(cfg.get("grid", {}).get("max_power_kw", state.grid_cap_kw))
        state.pv_peak_kw = float(cfg.get("pv", {}).get("farm_fixed_peak_kw", state.pv_peak_kw))

        model = cfg.get("tractors", {}).get("model", {})
        cap = float(model.get("battery_capacity_kwh", 44.8))
        fleet = cfg.get("tractors", {}).get("fleet") or [
            {"id": f"tractor_{i+1}", "initial_soc_percent": 70} for i in range(3)
        ]
        for entry in fleet:
            if not entry.get("enabled", True):
                continue
            loc = entry.get("initial_location", {})
            state.tractors.append(TractorSim(
                id=str(entry["id"]),
                capacity_kwh=cap,
                soc_pct=float(entry.get("initial_soc_percent", 70)),
                pos=(float(loc.get("x", 50)), float(loc.get("y", 50))),
            ))

        stations = cfg.get("charging", {}).get("stations") or [
            {"id": "charger_1", "max_power_kw": 6.6},
            {"id": "charger_2", "max_power_kw": 6.6},
        ]
        for entry in stations:
            state.chargers.append(ChargerSim(
                id=str(entry["id"]),
                rated_kw=float(entry.get("max_power_kw", 6.6)),
            ))

        consumers = cfg.get("energy_consumers") or [
            {"id": "electric_fence", "power_kw": 0.2, "always_on": True},
            {"id": "cold_storage", "power_kw": 1.2,
             "schedule": {"start": "08:00", "end": "20:00"}},
            {"id": "workshop_tools", "power_kw": 2.5,
             "schedule": {"start": "08:00", "end": "17:00"}},
        ]
        for entry in consumers:
            sched = entry.get("schedule") or {}
            state.loads.append(LoadSim(
                id=str(entry["id"]),
                base_kw=float(entry.get("power_kw", 1.0)),
                always_on=bool(entry.get("always_on", False)),
                start_h=_parse_hour(sched.get("start", 0)),
                end_h=_parse_hour(sched.get("end", 24)),
            ))
        return state

    # ---- lookups ------------------------------------------------------------
    def tractor(self, tid: str) -> Optional[TractorSim]:
        return next((t for t in self.tractors if t.id == tid), None)

    # ---- physics ------------------------------------------------------------
    def tick(self, sim_dt_s: float) -> None:
        self.clock_min = (self.clock_min + sim_dt_s / 60.0) % (24 * 60)
        hour = self.clock_min / 60.0
        dt_h = sim_dt_s / 3600.0

        self.pv_kw = self.pv_peak_kw * _DEFAULT_PV_PROFILE.get(int(hour), 0.0)
        self.tariff_code = next(
            code for lo, hi, code in _TARIFF_WINDOWS if lo <= hour < hi
        )

        # Docking: requesting tractors take the first free charger.
        docked_ids = {c.docked for c in self.chargers if c.docked}
        for t in self.tractors:
            if t.docked_charger and not (t.charge_request or t.v2l_kw > 0):
                for c in self.chargers:
                    if c.docked == t.id:
                        c.docked = None
                t.docked_charger = None
            elif (t.docked_charger is None and t.available
                  and (t.charge_request or t.v2l_kw > 0) and t.id not in docked_ids):
                free = next((c for c in self.chargers if c.docked is None), None)
                if free is not None:
                    free.docked = t.id
                    t.docked_charger = free.id
                    docked_ids.add(t.id)

        # Charging / V2L energy flow.
        charge_total = 0.0
        v2l_total = 0.0
        for c in self.chargers:
            c.power_kw = 0.0
            t = self.tractor(c.docked) if c.docked else None
            if t is None:
                continue
            if t.v2l_kw > 0 and t.soc_pct > 5.0:
                out = min(t.v2l_kw, c.rated_kw)
                t.soc_pct = max(0.0, t.soc_pct - out * dt_h / t.capacity_kwh * 100.0)
                c.power_kw = -out
                v2l_total += out
            elif t.charge_request and t.soc_pct < 100.0:
                p = c.rated_kw * LEVEL_FACTOR.get(c.level, 1.0)
                t.soc_pct = min(
                    100.0,
                    t.soc_pct + p * CHARGING_EFFICIENCY * dt_h / t.capacity_kwh * 100.0,
                )
                c.power_kw = p
                charge_total += p

        load_total = 0.0
        for l in self.loads:
            l.power_kw = l.base_kw if l.active(hour) else 0.0
            load_total += l.power_kw

        self.grid_draw_kw = load_total + charge_total - self.pv_kw - v2l_total

    @property
    def price_eur_per_kwh(self) -> float:
        return self.prices.get(self.tariff_code, 0.15)


def _parse_hour(value) -> float:
    """Accept 7, '07:00', '19:30' or datetime.time (YAML parses 06:00 oddly)."""
    if hasattr(value, "hour"):
        return value.hour + value.minute / 60.0
    if isinstance(value, (int, float)):
        # YAML reads unquoted 06:00 as sexagesimal int 360 (minutes).
        return float(value) / 60.0 if value > 24 else float(value)
    text = str(value)
    if ":" in text:
        h, m = text.split(":")[:2]
        return int(h) + int(m) / 60.0
    return float(text)
