"""
predictor/weather.py
====================
Weather-aware PV predictor.

Two implementations:
  1. ``OpenMeteoPVPredictor``  — fetches real hourly shortwave radiation from
     the Open-Meteo API (free, no key required).  Used when internet is
     available.
  2. ``WeatherStubPredictor``  — returns a seasonally-modulated static
     profile.  Used as fallback or in offline/simulation mode.

The Open-Meteo implementation is intentionally lightweight: one HTTPS GET
per day, result cached for 24 h.  The cache is stored in a simple JSON file
next to the config so it survives server restarts.

Why Open-Meteo?
---------------
* Free, no API key, GDPR-compliant (EU-hosted)
* Returns ``shortwave_radiation`` in W/m² at hourly resolution
* Works for any lat/lon — suitable for the Pilot 6 farm location

Usage example in config.yaml
-----------------------------
    prediction:
      pv:
        backend: openmeteo          # or: static | nn
        latitude: 41.69
        longitude: 23.31
        cache_file: pv_weather_cache.json

Integration with main.py
------------------------
The Simulator constructor checks ``cfg.get("prediction", {}).get("pv", {}).get("backend")``
and instantiates the appropriate predictor.  If no backend is set, it falls
back to StaticPVPredictor (zero change for existing users).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlopen
from urllib.error import URLError

import numpy as np

from .base import BasePVPredictor

logger = logging.getLogger(__name__)

# Open-Meteo endpoint
_OM_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&hourly=shortwave_radiation"
    "&forecast_days=2"
    "&timezone=auto"
)


def _fetch_openmeteo(lat: float, lon: float, timeout: int = 8) -> Dict[int, float]:
    """
    Fetch today's hourly shortwave radiation (W/m²) from Open-Meteo.
    Returns a dict mapping hour (0-23) to W/m².
    """
    url = _OM_URL.format(lat=lat, lon=lon)
    try:
        with urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except URLError as e:
        logger.warning("Open-Meteo fetch failed: %s", e)
        return {}

    times       = data["hourly"]["time"]                  # ISO strings
    radiation   = data["hourly"]["shortwave_radiation"]   # W/m²
    today       = datetime.now().date().isoformat()

    result: Dict[int, float] = {}
    for ts, rad in zip(times, radiation):
        if ts.startswith(today):
            h = int(ts[11:13])
            result[h] = float(rad) if rad is not None else 0.0
    return result


def _radiation_to_shape(radiation: Dict[int, float], peak_radiation_wm2: float = 900.0) -> Dict[int, float]:
    """Normalise W/m² to 0-1 shape values."""
    return {
        h: min(1.0, r / peak_radiation_wm2)
        for h, r in radiation.items()
    }


# ─────────────────────────────────────────────────────────────
# Open-Meteo predictor (real weather)
# ─────────────────────────────────────────────────────────────

class OpenMeteoPVPredictor(BasePVPredictor):
    """
    PV predictor backed by real weather forecasts from Open-Meteo.

    The forecast is fetched at most once per calendar day and cached to a
    JSON file.  On fetch failure the predictor falls back to a seasonal
    static profile so the simulator always gets a value.

    Parameters
    ----------
    farm_fixed_peak_kw   : Peak array capacity
    latitude, longitude  : Farm location (decimal degrees)
    cache_file           : Path to the JSON cache file
    peak_radiation_wm2   : Clear-sky peak irradiance for normalisation
    """

    def __init__(
        self,
        farm_fixed_peak_kw: float,
        latitude: float,
        longitude: float,
        cache_file: Path = Path("pv_weather_cache.json"),
        peak_radiation_wm2: float = 900.0,
    ) -> None:
        super().__init__(farm_fixed_peak_kw)
        self._lat   = latitude
        self._lon   = longitude
        self._cache_file = Path(cache_file)
        self._peak_wm2   = peak_radiation_wm2
        self._cache: Dict[str, Dict[int, float]] = {}   # date_str → {hour: shape}
        self._load_cache()

    # ── cache helpers ─────────────────────────────────────────

    def _load_cache(self) -> None:
        if self._cache_file.exists():
            try:
                self._cache = json.loads(self._cache_file.read_text())
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        try:
            self._cache_file.write_text(json.dumps(self._cache))
        except Exception as e:
            logger.warning("Could not save weather cache: %s", e)

    def _get_shapes_for_date(self, d: date) -> Dict[int, float]:
        key = d.isoformat()
        if key in self._cache:
            return self._cache[key]

        logger.info("Fetching Open-Meteo forecast for %s (lat=%.3f lon=%.3f)", key, self._lat, self._lon)
        radiation = _fetch_openmeteo(self._lat, self._lon)
        if radiation:
            shapes = _radiation_to_shape(radiation, self._peak_wm2)
            self._cache[key] = shapes
            self._save_cache()
            return shapes

        # Fallback: seasonal static profile
        logger.warning("Weather fetch failed — using seasonal fallback for %s", key)
        from .static import StaticPVPredictor
        angle  = 2 * np.pi * (d.month - 6) / 12
        factor = 0.575 + 0.425 * np.cos(angle)
        # Simple bell curve
        profile = {
            h: float(max(0.0, factor * np.exp(-0.5 * ((h - 12) / 2.5) ** 2)))
            for h in range(24)
        }
        peak = max(profile.values()) or 1.0
        shapes = {h: v / peak for h, v in profile.items()}
        self._cache[key] = shapes
        self._save_cache()
        return shapes

    # ── interface ─────────────────────────────────────────────

    def predict_shape(self, now: datetime) -> float:
        shapes = self._get_shapes_for_date(now.date())
        # Interpolate between hourly values for sub-hour timestamps
        h  = now.hour
        sh = shapes.get(h, 0.0)
        sh_next = shapes.get(h + 1, sh)
        frac = now.minute / 60.0
        return float(sh + (sh_next - sh) * frac)

    def predict_day(self, day: date) -> Dict[int, float]:
        return self._get_shapes_for_date(day)


# ─────────────────────────────────────────────────────────────
# Seasonal stub predictor (offline fallback)
# ─────────────────────────────────────────────────────────────

class WeatherStubPredictor(BasePVPredictor):
    """
    Offline PV predictor using a seasonally-modulated Gaussian bell curve.

    No network access required.  Used when:
    - Running in fully offline simulation mode
    - Unit testing
    - Open-Meteo is unavailable

    The bell curve is parameterised:
        shape(h) = seasonal_factor * exp(-0.5 * ((h - peak_hour) / width) ** 2)
    """

    def __init__(
        self,
        farm_fixed_peak_kw: float,
        peak_hour: float = 12.0,
        width_h: float = 2.5,
    ) -> None:
        super().__init__(farm_fixed_peak_kw)
        self._peak_hour = peak_hour
        self._width_h   = width_h

    def _seasonal(self, month: int) -> float:
        angle = 2 * np.pi * (month - 6) / 12
        return float(0.575 + 0.425 * np.cos(angle))

    def predict_shape(self, now: datetime) -> float:
        sf   = self._seasonal(now.month)
        raw  = sf * np.exp(-0.5 * ((now.hour - self._peak_hour) / self._width_h) ** 2)
        return float(np.clip(raw, 0.0, 1.0))
