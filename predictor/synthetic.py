"""
predictor/synthetic.py
======================
Synthetic dataset generator for HARVEST prediction module training.

Direct analogy to the paper's simulate.py but generalised for two targets:
  1. PV irradiance shape   (0-1 normalised, replaces "input packages")
  2. Farm aggregate load   (kW, replaces "output packages")

The generator uses the same truncated-Gaussian mixture model from the paper:

    f(x) = w0 + sum_k  w_k * NIT(x, mu_k, sigma_k)

where NIT is the integral of a truncated Gaussian over [x, x+1].

The key difference from the paper is that we generate two independent
time series with physically meaningful parameters:
  - PV: single mid-day peak (e.g. μ=12h, σ=2h) modulated by season
  - Load: multiple daily peaks matching the farm consumer schedule

Usage
-----
    gen = SyntheticDataGenerator.from_config(cfg)
    df  = gen.generate(weeks=12, seed=42)
    df.to_csv("train_dataset.csv", index=False)

The output DataFrame has columns:
    hour, day_of_week, month, pv_shape, farm_load_kw
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import truncnorm


# ─────────────────────────────────────────────────────────────
# Core mixture-of-Gaussians component (from the paper)
# ─────────────────────────────────────────────────────────────

class _GaussianMixture:
    """
    Truncated-Gaussian mixture expected value function.

    Directly ported from Expected class in simulate.py, renamed for clarity.
    """

    def __init__(
        self,
        total: float,
        means: List[float],
        stds: List[float],
        raw_weights: List[float],
        n_bins: int,
    ) -> None:
        if len(means) != len(stds):
            raise ValueError(f"means length {len(means)} != stds length {len(stds)}")
        if len(raw_weights) != len(means) + 1:
            raise ValueError(
                f"raw_weights must have len(means)+1 elements "
                f"({len(means)+1}), got {len(raw_weights)}"
            )
        self._total    = total
        self._means    = np.array(means, dtype=float)
        self._stds     = np.array(stds,  dtype=float)
        self._n_bins   = n_bins
        self._lo, self._hi = 0, n_bins - 1

        w = np.array(raw_weights, dtype=float)
        denom = np.sum(w[1:]) + w[0] * n_bins
        self._weights = w / denom * total

    def _nit(self, x: int, mu: float, sigma: float) -> float:
        """Integral of truncated Gaussian over [x, x+1]."""
        a = (self._lo - mu) / sigma
        b = (self._hi - mu) / sigma
        return float(
            truncnorm.cdf(x + 1, a=a, b=b, loc=mu, scale=sigma)
            - truncnorm.cdf(x,   a=a, b=b, loc=mu, scale=sigma)
        )

    def expected(self, x: int) -> float:
        """Return expected value at bin x."""
        if not (0 <= x < self._n_bins):
            raise ValueError(f"x={x} outside [0, {self._n_bins-1}]")
        val = self._weights[0]
        for w, m, s in zip(self._weights[1:], self._means, self._stds):
            val += w * self._nit(x, m, s)
        return val

    def sample_poisson(self, x: int, rng: np.random.Generator) -> int:
        """Draw a Poisson sample with lambda = expected(x)."""
        lam = max(0.0, self.expected(x))
        return int(rng.poisson(lam))


# ─────────────────────────────────────────────────────────────
# Seasonal modulation
# ─────────────────────────────────────────────────────────────

def _seasonal_pv_factor(month: int) -> float:
    """
    Monthly factor for PV peak irradiance (Northern Hemisphere).
    month: 1-12 → factor in [0.15, 1.0]
    Simple cosine approximation centred on June (month 6).
    """
    angle = 2 * np.pi * (month - 6) / 12
    return float(0.575 + 0.425 * np.cos(angle))


# ─────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────

@dataclass
class SyntheticDataGenerator:
    """
    Generates synthetic hourly datasets for training HARVEST predictors.

    Parameters mirror the paper's sim.yaml but are split into PV and load
    domains and extended with a month feature for seasonal variation.

    PV parameters
    -------------
    pv_peak_kw          : Peak PV array output (kW)
    pv_means_h          : Hour means for the daily irradiance bell curve(s)
    pv_stds_h           : Corresponding std-devs
    pv_weights_h        : Mixture weights (len = len(pv_means_h) + 1)
    pv_noise_frac       : Gaussian noise fraction added to each sample

    Load parameters
    ---------------
    load_peak_kw        : Maximum simultaneous farm load (kW)
    load_means_h        : Hour means for daily load peaks
    load_stds_h         : Corresponding std-devs
    load_weights_h      : Mixture weights
    load_means_d        : Day-of-week means
    load_stds_d         : Day-of-week std-devs
    load_weights_d      : Day-of-week mixture weights
    load_noise_frac     : Gaussian noise fraction
    """

    pv_peak_kw:      float       = 5.0
    pv_means_h:      List[float] = field(default_factory=lambda: [12.0])
    pv_stds_h:       List[float] = field(default_factory=lambda: [2.5])
    pv_weights_h:    List[float] = field(default_factory=lambda: [0.05, 10.0])
    pv_noise_frac:   float       = 0.08

    load_peak_kw:    float       = 13.3   # sum of all consumers
    load_means_h:    List[float] = field(default_factory=lambda: [7.0, 13.0, 20.0])
    load_stds_h:     List[float] = field(default_factory=lambda: [1.0, 2.0, 1.0])
    load_weights_h:  List[float] = field(default_factory=lambda: [1.0, 5.0, 3.0, 4.0])
    load_means_d:    List[float] = field(default_factory=lambda: [2.0])
    load_stds_d:     List[float] = field(default_factory=lambda: [1.0])
    load_weights_d:  List[float] = field(default_factory=lambda: [1.0, 5.0])
    load_noise_frac: float       = 0.10

    # ── factory ──────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SyntheticDataGenerator":
        """Build from the HARVEST config.yaml dict."""
        pv    = cfg.get("pv", {})
        pred  = cfg.get("prediction", {})
        pv_p  = pred.get("pv", {})
        ld_p  = pred.get("load", {})

        # Derive peak load from consumer list
        consumers = cfg.get("energy_consumers", [])
        peak_load = sum(c["power_kw"] for c in consumers)

        return cls(
            pv_peak_kw      = float(pv.get("farm_fixed_peak_kw", 5.0)),
            pv_means_h      = pv_p.get("means_h",   [12.0]),
            pv_stds_h       = pv_p.get("stds_h",    [2.5]),
            pv_weights_h    = pv_p.get("weights_h", [0.05, 10.0]),
            pv_noise_frac   = float(pv_p.get("noise_frac", 0.08)),
            load_peak_kw    = peak_load or 13.3,
            load_means_h    = ld_p.get("means_h",   [7.0, 13.0, 20.0]),
            load_stds_h     = ld_p.get("stds_h",    [1.0, 2.0, 1.0]),
            load_weights_h  = ld_p.get("weights_h", [1.0, 5.0, 3.0, 4.0]),
            load_means_d    = ld_p.get("means_d",   [2.0]),
            load_stds_d     = ld_p.get("stds_d",    [1.0]),
            load_weights_d  = ld_p.get("weights_d", [1.0, 5.0]),
            load_noise_frac = float(ld_p.get("noise_frac", 0.10)),
        )

    # ── public API ────────────────────────────────────────────

    def generate(
        self,
        weeks: int = 12,
        seed: int = 42,
        start: datetime = datetime(2026, 1, 1),
    ) -> pd.DataFrame:
        """
        Generate ``weeks`` weeks of hourly synthetic data.

        Returns a DataFrame with columns:
            timestamp, hour, day_of_week, month,
            pv_shape, pv_kw, farm_load_kw
        """
        rng = np.random.default_rng(seed)
        rows: List[Dict[str, Any]] = []

        n_hours = weeks * 7 * 24
        ts = start

        # Build day-level load modulator once per week
        for _ in range(weeks):
            month = ts.month
            seasonal = _seasonal_pv_factor(month)

            pv_daily = _GaussianMixture(
                self.pv_peak_kw * seasonal,
                self.pv_means_h,
                self.pv_stds_h,
                self.pv_weights_h,
                n_bins=24,
            )

            load_weekly = _GaussianMixture(
                self.load_peak_kw,
                self.load_means_d,
                self.load_stds_d,
                self.load_weights_d,
                n_bins=7,
            )

            for d in range(7):
                daily_load_budget = load_weekly.expected(d)

                load_hourly = _GaussianMixture(
                    daily_load_budget,
                    self.load_means_h,
                    self.load_stds_h,
                    self.load_weights_h,
                    n_bins=24,
                )

                for h in range(24):
                    pv_mean    = max(0.0, pv_daily.expected(h))
                    pv_noise   = rng.normal(0, pv_mean * self.pv_noise_frac + 1e-6)
                    pv_kw      = float(np.clip(pv_mean + pv_noise, 0.0, self.pv_peak_kw))
                    pv_shape   = pv_kw / max(self.pv_peak_kw, 1e-6)

                    load_mean  = max(0.0, load_hourly.expected(h))
                    load_noise = rng.normal(0, load_mean * self.load_noise_frac + 1e-6)
                    load_kw    = float(np.clip(load_mean + load_noise, 0.0, self.load_peak_kw))

                    rows.append({
                        "timestamp":    ts.isoformat(),
                        "hour":         h,
                        "day_of_week":  d,
                        "month":        month,
                        "pv_shape":     round(pv_shape, 4),
                        "pv_kw":        round(pv_kw,    3),
                        "farm_load_kw": round(load_kw,  3),
                    })
                    ts += timedelta(hours=1)

        return pd.DataFrame(rows)

    def save(
        self,
        output_path,
        weeks: int = 12,
        seed: int = 42,
        fmt: str = "csv",
    ) -> None:
        """Generate and save dataset.  fmt: 'csv' or 'npz'."""
        df = self.generate(weeks=weeks, seed=seed)
        output_path = Path(output_path) if not hasattr(output_path, "stem") else output_path
        if fmt == "npz":
            features = ["hour", "day_of_week", "month"]
            targets  = ["pv_shape", "farm_load_kw"]
            import numpy as _np
            _np.savez(
                output_path,
                dataset_x=df[features].values.astype("float32"),
                dataset_y=df[targets].values.astype("float32"),
                columns_x=features,
                columns_y=targets,
            )
        else:
            df.to_csv(output_path, index=False)


from pathlib import Path


if __name__ == "__main__":
    import argparse, yaml

    p = argparse.ArgumentParser(description="Generate HARVEST synthetic training data")
    p.add_argument("--config",  type=Path, default=Path("config.yaml"))
    p.add_argument("--weeks",   type=int,  default=12)
    p.add_argument("--seed",    type=int,  default=42)
    p.add_argument("--output",  type=Path, default=Path("train_data.npz"))
    p.add_argument("--fmt",     choices=["csv", "npz"], default="npz")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    gen = SyntheticDataGenerator.from_config(cfg)
    gen.save(args.output, weeks=args.weeks, seed=args.seed, fmt=args.fmt)
    print(f"Saved {args.weeks}-week dataset → {args.output}")
