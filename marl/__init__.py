from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseAgent, TractorObservation, ChargerObservation, LoadObservation
from .agents import TractorAgent, ChargingStationAgent, LoadAgent
from .environment import MARLEnvironment, MARLStepLog


def build_marl_engine(
    sim_cfg: Any,
    raw_cfg: Dict[str, Any],
    tariff_model: Any,
    pv_model: Any,
) -> Optional[MARLEnvironment]:
    """Factory: build a MARLEnvironment from the merged config dict.

    Returns None when marl.enabled is False or the section is absent, so
    the caller can fall back to the rule-based Scheduler transparently.
    """
    marl_cfg = raw_cfg.get("marl", {})
    if not marl_cfg.get("enabled", False):
        return None

    agent_cfg = marl_cfg.get("agents", {})
    charge_thresh = float(
        agent_cfg.get("tractors", {}).get("charge_threshold_soc", 90.0)
    )
    managed_priorities = set(
        agent_cfg.get("loads", {}).get("managed_priorities", ["low", "normal"])
    )

    # One TractorAgent per enabled tractor
    tractor_agents: Dict[str, TractorAgent] = {}
    if agent_cfg.get("tractors", {}).get("enabled", True):
        for tr in sim_cfg.tractors:
            if tr.enabled:
                tractor_agents[tr.tractor_id] = TractorAgent(
                    tr.tractor_id, charge_threshold_soc=charge_thresh
                )

    # One ChargingStationAgent per charger
    charger_agents: Dict[str, ChargingStationAgent] = {}
    if agent_cfg.get("charging_stations", {}).get("enabled", True):
        for ch in sim_cfg.chargers:
            charger_agents[ch.charger_id] = ChargingStationAgent(ch.charger_id)

    # LoadAgents only for deferrable (non-always-on) consumers of managed priority
    load_agents: Dict[str, LoadAgent] = {}
    if agent_cfg.get("loads", {}).get("enabled", True):
        for c in sim_cfg.consumers:
            if c.priority in managed_priorities and not c.always_on:
                load_agents[c.consumer_id] = LoadAgent(c.consumer_id)

    return MARLEnvironment(
        cfg=sim_cfg,
        tractor_agents=tractor_agents,
        charger_agents=charger_agents,
        load_agents=load_agents,
        tariff_model=tariff_model,
        pv_model=pv_model,
    )


__all__ = [
    "build_marl_engine",
    "MARLEnvironment",
    "MARLStepLog",
    "BaseAgent",
    "TractorAgent",
    "ChargingStationAgent",
    "LoadAgent",
    "TractorObservation",
    "ChargerObservation",
    "LoadObservation",
]
