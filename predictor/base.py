"""
predictor/base.py
=================
Abstract base classes for HARVEST prediction modules.

Design goals
------------
* Drop-in replacement for the static PVModel profile: any predictor that
  implements BasePVPredictor can be passed to PVModel and the rest of the
  simulator is unchanged.
* Feature-extensible: subclasses can accept as many additional features as
  needed (weather, SOC history, task calendar) without modifying the interface.
* Backend-agnostic: works with rule tables, sklearn models, TensorFlow/Keras,
  or external weather APIs equally well.

Typical call sequence (within simulator step):
    shape = pv_predictor.predict_shape(now)           # 0.0 – 1.0
    farm_kw = pv_predictor.predict_farm_kw(now)       # kW
    load_kw = load_predictor.predict_load_kw(now)     # kW

For 24-hour look-ahead scheduling:
    daily = pv_predictor.predict_day(date)            # dict[hour -> float]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────
# PV predictor hierarchy
# ─────────────────────────────────────────────────────────────

class BasePVPredictor(ABC):
    """
    Predicts PV generation for a farm fixed array.

    All subclasses must implement ``predict_shape``.  The higher-level
    helpers (``predict_farm_kw``, ``predict_day``) are provided as
    conveniences and can be overridden when the subclass has a more
    efficient batch implementation.
    """

    def __init__(self, farm_fixed_peak_kw: float) -> None:
        self.farm_fixed_peak_kw = farm_fixed_peak_kw

    @abstractmethod
    def predict_shape(self, now: datetime) -> float:
        """
        Return normalised irradiance in [0, 1] for the given timestamp.

        This is the single value the simulator's ``PVModel.shape_at()`` uses.
        A value of 1.0 means the array is at peak rated output.
        """

    def predict_farm_kw(self, now: datetime) -> float:
        """Return expected farm array output in kW."""
        return self.farm_fixed_peak_kw * self.predict_shape(now)

    def predict_day(self, day: date) -> Dict[int, float]:
        """
        Return a dict mapping hour (0-23) to normalised shape value.

        Default implementation calls ``predict_shape`` 24 times.
        Override for efficiency if the backend supports batch queries.
        """
        return {
            h: self.predict_shape(datetime(day.year, day.month, day.day, h))
            for h in range(24)
        }

    def mape(self, actuals: Dict[int, float], predicted: Dict[int, float]) -> float:
        """Mean Absolute Percentage Error — useful for TPI3 validation."""
        errors = []
        for h, actual in actuals.items():
            if actual > 0:
                errors.append(abs(actual - predicted.get(h, 0.0)) / actual)
        return float(sum(errors) / len(errors) * 100) if errors else 0.0


# ─────────────────────────────────────────────────────────────
# Farm load predictor hierarchy
# ─────────────────────────────────────────────────────────────

class BaseLoadPredictor(ABC):
    """
    Predicts aggregate farm load (non-tractor consumers) in kW.

    Mirrors the paper's output-packages prediction: the same FFNN
    architecture can predict when irrigation pumps, HVAC, workshop etc.
    are likely to be active, allowing the scheduler to reserve headroom
    in advance instead of reacting step-by-step.
    """

    @abstractmethod
    def predict_load_kw(self, now: datetime) -> float:
        """Return expected farm load in kW at the given timestamp."""

    def predict_day_load(self, day: date) -> Dict[int, float]:
        """
        Return a dict mapping hour (0-23) to expected farm load in kW.
        Default implementation calls ``predict_load_kw`` 24 times.
        """
        return {
            h: self.predict_load_kw(datetime(day.year, day.month, day.day, h))
            for h in range(24)
        }


# ─────────────────────────────────────────────────────────────
# Combined forecast bundle (convenience wrapper)
# ─────────────────────────────────────────────────────────────

class ForecastBundle:
    """
    Packages a PV predictor and a load predictor together.

    The scheduler can hold a single ForecastBundle and call helpers such as
    ``net_available_kw`` to get the expected headroom at any future timestamp
    — used for pro-active charging decisions.
    """

    def __init__(
        self,
        pv: BasePVPredictor,
        load: BaseLoadPredictor,
        grid_max_kw: float,
    ) -> None:
        self.pv   = pv
        self.load = load
        self.grid_max_kw = grid_max_kw

    def net_available_kw(self, now: datetime) -> float:
        """
        Estimate of charging headroom at ``now``:
            grid_cap + PV_forecast – load_forecast
        """
        return self.grid_max_kw + self.pv.predict_farm_kw(now) - self.load.predict_load_kw(now)

    def best_charging_window(self, day: date, duration_hours: float) -> Optional[int]:
        """
        Return the starting hour of the cheapest / most available charging
        window of ``duration_hours`` length within the given day.

        A simple greedy scan — a real MARL agent would use this as a feature,
        not a hard decision.
        """
        headroom = {
            h: self.net_available_kw(datetime(day.year, day.month, day.day, h))
            for h in range(24)
        }
        dur = max(1, int(duration_hours))
        best_h, best_val = 0, -1e9
        for h in range(24 - dur + 1):
            window_val = sum(headroom[h + i] for i in range(dur))
            if window_val > best_val:
                best_val, best_h = window_val, h
        return best_h
