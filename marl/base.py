from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class TractorObservation:
    agent_id: str
    soc_norm: float           # soc_percent / 100
    is_charging: bool
    has_task: bool
    task_urgency: float       # 0=none  0.33=flexible  0.66=normal  1.0=urgent
    deadline_norm: float      # minutes_to_nearest_open_deadline / 1440, clamped [0,1]
    tariff_norm: float        # 0=valle  0.5=llano  1.0=punta
    net_power_norm: float     # net_available_kw / grid_max_kw, clamped [0,1]
    pv_shape: float           # normalised solar irradiance 0-1
    hour_norm: float          # hour / 24.0


@dataclass
class ChargerObservation:
    agent_id: str
    is_occupied: bool
    connected_soc_norm: float  # SOC of connected tractor; 0.0 if free
    net_power_norm: float
    tariff_norm: float
    pv_shape: float
    hour_norm: float


@dataclass
class LoadObservation:
    agent_id: str
    priority: str              # critical | high | normal | low
    power_kw: float
    net_power_norm: float
    tariff_norm: float
    hour_norm: float


class BaseAgent(ABC):
    """Abstract base for all MARL agents.

    Subclasses implement act() for rule-based or learned policies.
    learn() is a no-op here; a PPO subclass would override it.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    @abstractmethod
    def act(self, observation: Any) -> Any:
        ...

    def reset(self) -> None:
        pass

    def learn(self, obs: Any, action: Any, reward: float, next_obs: Any) -> None:
        pass
