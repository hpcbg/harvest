"""
predictor/static.py
===================
StaticPVPredictor  – wraps the existing config.yaml hourly profile dict.
StaticLoadPredictor – wraps a list of EnergyConsumer-style dicts.

These are the *zero-change* drop-in replacements that make main.py accept
the new interface while preserving all existing behaviour.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Dict, List, Optional

from .base import BasePVPredictor, BaseLoadPredictor


class StaticPVPredictor(BasePVPredictor):
    """
    Reads the normalised hourly PV profile from config.yaml and returns it
    unchanged.  Identical behaviour to the original PVModel.

    Parameters
    ----------
    normalised_profile : dict[int, float]
        Hour (0-23) → normalised irradiance (0-1).  From ``pv.profile``.
    farm_fixed_peak_kw : float
        Peak installed capacity of the farm fixed array.
    """

    def __init__(
        self,
        normalised_profile: Dict[int, float],
        farm_fixed_peak_kw: float,
    ) -> None:
        super().__init__(farm_fixed_peak_kw)
        self._profile = {int(k): float(v) for k, v in normalised_profile.items()}

    def predict_shape(self, now: datetime) -> float:
        return self._profile.get(now.hour, 0.0)


class StaticLoadPredictor(BaseLoadPredictor):
    """
    Reconstructs expected farm load from the ``energy_consumers`` list in
    config.yaml, using each consumer's schedule and power.

    This mirrors exactly what Simulator._compute_farm_load() does so it
    can be used as a look-ahead forecast without running the full simulator.
    """

    def __init__(self, consumer_configs: List[Dict[str, Any]]) -> None:
        self._consumers = consumer_configs

    @staticmethod
    def _parse_t(s: str) -> time:
        h, m = map(int, s.split(":"))
        return time(h, m)

    @staticmethod
    def _active(consumer: Dict[str, Any], now: datetime) -> bool:
        if consumer.get("always_on", False):
            return True
        sched = consumer.get("schedule")
        if not sched:
            return False
        start = StaticLoadPredictor._parse_t(sched["start"])
        end   = StaticLoadPredictor._parse_t(sched["end"])
        t = now.time()
        if start <= end:
            return start <= t < end
        return t >= start or t < end   # midnight-crossing

    def predict_load_kw(self, now: datetime) -> float:
        return sum(
            c["power_kw"] for c in self._consumers if self._active(c, now)
        )
