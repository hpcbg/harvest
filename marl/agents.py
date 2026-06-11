from __future__ import annotations

from .base import BaseAgent, TractorObservation, ChargerObservation, LoadObservation

# ── Tractor action space ─────────────────────────────────────────────────────
TRACTOR_IDLE = 0
TRACTOR_REQUEST_CHARGE = 1

# ── Charger action space ─────────────────────────────────────────────────────
CHARGER_OFF = 0
CHARGER_LOW = 1    # 50 % of max rated power
CHARGER_FULL = 2   # 100 % of max rated power

# ── Load action space ────────────────────────────────────────────────────────
LOAD_ON = 0
LOAD_OFF = 1


class TractorAgent(BaseAgent):
    """Rule-based policy: requests charging when SOC is low and conditions are favourable.

    Thresholds mirror the smart-charging heuristics in Scheduler._charge_smart()
    but expressed as per-agent decisions rather than a centralised allocator.
    """

    def __init__(self, agent_id: str, charge_threshold_soc: float = 90.0) -> None:
        super().__init__(agent_id)
        self._thresh = charge_threshold_soc / 100.0

    def act(self, obs: TractorObservation) -> int:
        if obs.soc_norm >= self._thresh:
            return TRACTOR_IDLE
        if obs.has_task:
            return TRACTOR_IDLE
        # Emergency — charge regardless of tariff or headroom
        if obs.soc_norm < 0.20:
            return TRACTOR_REQUEST_CHARGE
        # Valle (cheapest night window) — always opportunistic
        if obs.tariff_norm == 0.0:
            return TRACTOR_REQUEST_CHARGE
        # Daytime PV surplus with grid headroom
        if obs.pv_shape > 0.30 and obs.net_power_norm > 0.15:
            return TRACTOR_REQUEST_CHARGE
        # Llano (mid tariff) with comfortable grid headroom
        if obs.tariff_norm <= 0.5 and obs.net_power_norm > 0.40:
            return TRACTOR_REQUEST_CHARGE
        return TRACTOR_IDLE


class ChargingStationAgent(BaseAgent):
    """Rule-based policy: decides charging power level given grid conditions."""

    def act(self, obs: ChargerObservation) -> int:
        if obs.net_power_norm <= 0.05:
            return CHARGER_OFF
        # Throttle to half during peak tariff when headroom is tight
        if obs.tariff_norm >= 1.0 and obs.net_power_norm < 0.25:
            return CHARGER_LOW
        return CHARGER_FULL


class LoadAgent(BaseAgent):
    """Rule-based policy: sheds deferrable loads during peak tariff and grid stress."""

    def act(self, obs: LoadObservation) -> int:
        if obs.priority in ("critical", "high"):
            return LOAD_ON
        # Shed low/normal priority during punta with constrained headroom
        if obs.tariff_norm >= 1.0 and obs.net_power_norm < 0.15:
            return LOAD_OFF
        return LOAD_ON
