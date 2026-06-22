from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import TractorObservation, ChargerObservation, LoadObservation
from .agents import (
    TractorAgent,
    ChargingStationAgent,
    LoadAgent,
    TRACTOR_REQUEST_CHARGE,
    CHARGER_OFF,
    CHARGER_LOW,
    CHARGER_FULL,
    LOAD_OFF,
)


@dataclass
class MARLStepLog:
    """One 15-min step of MARL agent decisions, observations, and outcomes."""
    timestamp: datetime
    # Per-agent actions
    tractor_actions:  Dict[str, int]    # TRACTOR_IDLE=0 | REQUEST_CHARGE=1
    charger_actions:  Dict[str, int]    # CHARGER_OFF=0 | LOW=1 | FULL=2
    load_actions:     Dict[str, int]    # LOAD_ON=0 | LOAD_OFF=1
    # Key observations (tractor agents only — most informative)
    tractor_soc:      Dict[str, float]  # tractor_id -> soc_percent at decision time
    tractor_charge_kw: Dict[str, float] # tractor_id -> actual charge kW allocated
    tractor_obs:      Dict[str, Dict]   # tractor_id -> subset of TractorObservation fields
    # Environment snapshot
    net_power_norm:   float
    tariff_norm:      float
    pv_shape:         float
    # Outcomes (filled by record_step)
    grid_kw:          float = 0.0
    cost_eur:         float = 0.0
    completed_tasks:  int   = 0
    v2l_discharge_kw: float = 0.0       # power supplied by tractor batteries (V2L)
    # Rewards (filled by record_step)
    rewards:          Dict[str, float] = field(default_factory=dict)
    cost_reward:      float = 0.0
    peak_reward:      float = 0.0
    task_reward:      float = 0.0


class MARLEnvironment:
    """Multi-agent environment that wraps the farm simulation state.

    Replaces Scheduler.allocate_charging() with per-agent decisions and
    handles optional load shedding via LoadAgent policies.
    Task assignment remains rule-based (Scheduler.assign_tasks is unchanged).

    Agents
    ------
    - TractorAgent   — one per enabled tractor (decides: request charge or idle)
    - ChargingStationAgent — one per charger   (decides: off / low / full power)
    - LoadAgent      — one per deferrable consumer (decides: on or off)

    Reward
    ------
    compute_step_rewards() returns a per-agent scalar for use in future PPO
    training.  Rule-based agents ignore it; it is computed every step so the
    data pipeline is ready when a learned policy is plugged in.
    """

    def __init__(
        self,
        cfg: Any,                                    # SimulationConfig
        tractor_agents: Dict[str, TractorAgent],
        charger_agents: Dict[str, ChargingStationAgent],
        load_agents: Dict[str, LoadAgent],
        tariff_model: Any,                           # TariffModel
        pv_model: Any,                               # PVModel
    ) -> None:
        self.cfg = cfg
        self.tractor_agents = tractor_agents
        self.charger_agents = charger_agents
        self.load_agents = load_agents
        self._tariff = tariff_model
        self._pv = pv_model
        # ── Step logging ──────────────────────────────────────────────────────
        self.step_log: List[MARLStepLog] = []
        self._pending_log: Optional[Dict] = None
        self._pending_load_actions: Dict[str, int] = {}

    # ── Normalisation helpers ────────────────────────────────────────────────

    def _tariff_norm(self, now: datetime) -> float:
        price = self._tariff.get_energy_price(now)
        prices = [float(v["price_eur_per_kwh"]) for v in self._tariff.energy_periods.values()]
        lo, hi = min(prices), max(prices)
        return 0.5 if hi == lo else (price - lo) / (hi - lo)

    def _net_norm(self, net_available_kw: float) -> float:
        return max(0.0, min(1.0, net_available_kw / max(self.cfg.grid_max_power_kw, 1.0)))

    def _task_urgency(self, tractor: Any) -> float:
        if tractor.current_task_id is None:
            return 0.0
        task = next((t for t in self.cfg.tasks if t.task_id == tractor.current_task_id), None)
        if task is None:
            return 0.0
        return {"flexible": 0.33, "normal": 0.66, "urgent": 1.0}.get(task.priority, 0.66)

    def _deadline_norm(self, now: datetime) -> float:
        open_tasks = [
            t for t in self.cfg.tasks
            if not t.is_done and t.assigned_tractor_id is None
        ]
        if not open_tasks:
            return 1.0
        nearest_min = min(
            (t.effective_deadline() - now).total_seconds() / 60.0
            for t in open_tasks
        )
        return max(0.0, min(1.0, nearest_min / 1440.0))

    # ── Observation builders ─────────────────────────────────────────────────

    def _obs_tractor(
        self, tr: Any, now: datetime, net_available_kw: float, pv_shape: float
    ) -> TractorObservation:
        return TractorObservation(
            agent_id=tr.tractor_id,
            soc_norm=tr.soc_percent / 100.0,
            is_charging=tr.is_charging,
            has_task=tr.current_task_id is not None,
            task_urgency=self._task_urgency(tr),
            deadline_norm=self._deadline_norm(now),
            tariff_norm=self._tariff_norm(now),
            net_power_norm=self._net_norm(net_available_kw),
            pv_shape=pv_shape,
            hour_norm=now.hour / 24.0,
        )

    def _obs_charger(
        self, ch: Any, now: datetime, net_available_kw: float, pv_shape: float
    ) -> ChargerObservation:
        connected = next(
            (tr for tr in self.cfg.tractors if tr.assigned_charger_id == ch.charger_id),
            None,
        )
        return ChargerObservation(
            agent_id=ch.charger_id,
            is_occupied=connected is not None,
            connected_soc_norm=(connected.soc_percent / 100.0) if connected else 0.0,
            net_power_norm=self._net_norm(net_available_kw),
            tariff_norm=self._tariff_norm(now),
            pv_shape=pv_shape,
            hour_norm=now.hour / 24.0,
        )

    def _obs_load(
        self, consumer: Any, now: datetime, net_available_kw: float
    ) -> LoadObservation:
        return LoadObservation(
            agent_id=consumer.consumer_id,
            priority=consumer.priority,
            power_kw=consumer.power_kw,
            net_power_norm=self._net_norm(net_available_kw),
            tariff_norm=self._tariff_norm(now),
            hour_norm=now.hour / 24.0,
        )

    # ── Core environment steps ───────────────────────────────────────────────

    def apply_charging(self, now: datetime, net_available_kw: float) -> None:
        """MARL-driven charging allocation — replaces Scheduler.allocate_charging()."""
        for tr in self.cfg.tractors:
            tr.is_charging = False
            tr.assigned_charger_id = None
            tr.requested_charge_power_kw = 0.0
            tr.actual_charge_power_kw = 0.0

        pv_shape    = self._pv.shape_at(now)
        tariff_norm = self._tariff_norm(now)
        net_norm    = self._net_norm(net_available_kw)

        # ── collect tractor actions + observations for logging ─────────────
        tractor_actions: Dict[str, int]  = {}
        tractor_soc:     Dict[str, float] = {}
        tractor_obs_log: Dict[str, Dict]  = {}

        requesting: List[Any] = []
        for tr in self.cfg.tractors:
            tractor_soc[tr.tractor_id] = tr.soc_percent
            if not tr.enabled or tr.current_task_id is not None or tr.soc_percent >= 90.0 \
                    or getattr(tr, "is_discharging", False):
                tractor_actions[tr.tractor_id] = 0   # TRACTOR_IDLE
                continue
            agent = self.tractor_agents.get(tr.tractor_id)
            if agent is None:
                tractor_actions[tr.tractor_id] = 0
                continue
            obs    = self._obs_tractor(tr, now, net_available_kw, pv_shape)
            action = agent.act(obs)
            tractor_actions[tr.tractor_id] = action
            tractor_obs_log[tr.tractor_id] = {
                "soc_norm":       obs.soc_norm,
                "tariff_norm":    obs.tariff_norm,
                "net_power_norm": obs.net_power_norm,
                "pv_shape":       obs.pv_shape,
                "has_task":       obs.has_task,
            }
            if action == TRACTOR_REQUEST_CHARGE:
                requesting.append(tr)

        # ── default all chargers to OFF, update as they are allocated ──────
        charger_actions: Dict[str, int] = {ch.charger_id: CHARGER_OFF
                                           for ch in self.cfg.chargers}

        if requesting:
            has_urgent = any(
                not t.is_done and t.assigned_tractor_id is None and t.priority == "urgent"
                for t in self.cfg.tasks
            )
            requesting.sort(key=lambda tr: (0 if has_urgent else 1, tr.soc_percent))

            available_kw  = max(0.0, net_available_kw)
            free_chargers = list(self.cfg.chargers)

            for tr in requesting:
                if not free_chargers or available_kw <= 0.5:
                    break
                ch     = free_chargers.pop(0)
                agent  = self.charger_agents.get(ch.charger_id)
                ch_obs = self._obs_charger(ch, now, available_kw, pv_shape)
                action = agent.act(ch_obs) if agent else CHARGER_FULL
                charger_actions[ch.charger_id] = action

                if action == CHARGER_OFF:
                    free_chargers.insert(0, ch)
                    continue

                max_req = min(ch.max_power_kw, self.cfg.tractors_model.max_charge_power_kw)
                req     = max_req * (0.5 if action == CHARGER_LOW else 1.0)
                alloc   = min(req, available_kw)
                if alloc <= 0.5:
                    continue

                tr.is_charging              = True
                tr.assigned_charger_id      = ch.charger_id
                tr.requested_charge_power_kw = req
                tr.actual_charge_power_kw   = alloc
                available_kw -= alloc

        # ── store pending log entry (outcomes filled in record_step) ───────
        self._pending_log = {
            "tractor_actions":  tractor_actions,
            "tractor_soc":      tractor_soc,
            "tractor_charge_kw": {tr.tractor_id: tr.actual_charge_power_kw
                                  for tr in self.cfg.tractors},
            "tractor_obs":      tractor_obs_log,
            "charger_actions":  charger_actions,
            "net_power_norm":   net_norm,
            "tariff_norm":      tariff_norm,
            "pv_shape":         pv_shape,
        }
        self._pending_load_actions = {}   # reset; filled by consumer_is_active calls

    def consumer_is_active(
        self, consumer: Any, now: datetime, net_available_kw: float
    ) -> bool:
        """Return whether a load agent wants this consumer on.

        Consumers not managed by a LoadAgent are always passed through unchanged;
        the caller still applies the normal schedule check.
        """
        agent = self.load_agents.get(consumer.consumer_id)
        if agent is None:
            return True
        obs    = self._obs_load(consumer, now, net_available_kw)
        action = agent.act(obs)
        self._pending_load_actions[consumer.consumer_id] = action   # log it
        return action != LOAD_OFF

    def record_step(
        self,
        now: datetime,
        grid_kw: float,
        cost_eur: float,
        completed: int,
        weights: Dict[str, float],
        v2l_kw: float = 0.0,
    ) -> None:
        """Finalise the pending log entry with step outcomes and rewards.

        Call this once per simulation step AFTER power accounting, e.g.::

            if self.marl_engine:
                self.marl_engine.record_step(
                    now, grid_kw, cost_eur, completed,
                    self.config.objective_weights,
                )
        """
        if self._pending_log is None:
            return

        rewards = self.compute_step_rewards(grid_kw, cost_eur, completed, weights)

        w_cost = weights.get("energy_cost",   1.0)
        w_peak = weights.get("peak_power",    3.0)
        w_task = weights.get("task_completion", 10.0)

        cost_r = -cost_eur * w_cost
        peak_r = -max(0.0, grid_kw - self.cfg.grid_max_power_kw * 0.8) * w_peak
        task_r = completed * 0.01 * w_task

        # Include all managed load agents: -1 = not scheduled this step
        complete_load_actions = {lid: -1 for lid in self.load_agents}
        complete_load_actions.update(self._pending_load_actions)

        entry = MARLStepLog(
            timestamp=now,
            tractor_actions=self._pending_log["tractor_actions"],
            charger_actions=self._pending_log["charger_actions"],
            load_actions=complete_load_actions,
            tractor_soc=self._pending_log["tractor_soc"],
            tractor_charge_kw=self._pending_log["tractor_charge_kw"],
            tractor_obs=self._pending_log["tractor_obs"],
            net_power_norm=self._pending_log["net_power_norm"],
            tariff_norm=self._pending_log["tariff_norm"],
            pv_shape=self._pending_log["pv_shape"],
            grid_kw=grid_kw,
            cost_eur=cost_eur,
            completed_tasks=completed,
            v2l_discharge_kw=v2l_kw,
            rewards=rewards,
            cost_reward=cost_r,
            peak_reward=peak_r,
            task_reward=task_r,
        )
        self.step_log.append(entry)
        self._pending_log = None

    # ── Reward computation (for future RL training) ──────────────────────────

    def compute_step_rewards(
        self,
        grid_kw: float,
        cost_eur: float,
        completed: int,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Scalar reward per agent for this time step.

        Rule-based agents ignore this; it is computed each step so the data
        pipeline is in place when a learned policy replaces the heuristics.
        """
        cost_r = -cost_eur * weights.get("energy_cost", 1.0)
        peak_r = (
            -max(0.0, grid_kw - self.cfg.grid_max_power_kw * 0.8)
            * weights.get("peak_power", 3.0)
        )
        task_r = completed * 0.01 * weights.get("unserved_tasks", 10.0)
        team_r = cost_r + peak_r + task_r

        rewards: Dict[str, float] = {}
        for tr in self.cfg.tractors:
            soc_bonus = (
                -abs(tr.soc_percent / 100.0 - 0.6) * weights.get("battery_stress", 1.0) * 0.05
            )
            rewards[tr.tractor_id] = team_r + soc_bonus
        for ch in self.cfg.chargers:
            rewards[ch.charger_id] = team_r
        for aid in self.load_agents:
            rewards[aid] = team_r
        return rewards
