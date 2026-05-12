#!/usr/bin/env python
"""
generate_prediction_overview.py
================================
Generates images/prediction_overview.png for the HARVEST README.

Usage (from the harvest/ root folder):
    python generate_prediction_overview.py

Requirements: same as requirements.txt (numpy, scipy, matplotlib, pyyaml).
The predictor/ package must be present in the same folder.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yaml

# ── ensure predictor package is importable ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from predictor import build_predictors, ForecastBundle
from predictor.static import StaticLoadPredictor
from predictor.synthetic import SyntheticDataGenerator, _seasonal_pv_factor
from predictor.weather import WeatherStubPredictor

# ── load config ───────────────────────────────────────────────────────────────
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# ── output path ───────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent / "images"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "prediction_overview.png"

# ── shared style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.color":        "#CCCCCC",
})

C = {
    "pv":    "#EF9F27",
    "load":  "#C0312F",
    "head":  "#028090",
    "syn":   "#BA7517",
}

hours = list(range(24))

# ── data ──────────────────────────────────────────────────────────────────────

# Static PV from config profile
static_pv = [
    cfg["pv"]["profile"][h] * cfg["pv"]["farm_fixed_peak_kw"]
    for h in hours
]

# Static farm load from consumer schedules
load_pred = StaticLoadPredictor(cfg["energy_consumers"])
static_load = [
    load_pred.predict_load_kw(datetime(2026, 6, 1, h))
    for h in hours
]

# Seasonal bell-curve helper
def bell_pv(h: int, month: int, peak: float = 5.0) -> float:
    sf = _seasonal_pv_factor(month)
    return peak * sf * np.exp(-0.5 * ((h - 12) / 2.5) ** 2)

# Noisy synthetic PV samples (to show training data texture)
def noisy_pv(seed: int, month: int = 6, peak: float = 5.0):
    rng = np.random.default_rng(seed)
    vals = []
    for h in hours:
        base  = peak * _seasonal_pv_factor(month) * np.exp(-0.5 * ((h - 12) / 2.5) ** 2)
        noise = rng.normal(0, base * 0.12 + 0.02)
        vals.append(max(0.0, base + noise))
    return vals

# Seasonal stub predictor
stub = WeatherStubPredictor(5.0)
stub_vals = [stub.predict_farm_kw(datetime(2026, 6, 1, h)) for h in hours]

# ForecastBundle headroom
pv_pred, ld_pred = build_predictors(cfg)
bundle   = ForecastBundle(pv_pred, ld_pred, grid_max_kw=cfg["grid"]["max_power_kw"])
headroom = [bundle.net_available_kw(datetime(2026, 6, 1, h)) for h in hours]
best_h   = bundle.best_charging_window(date(2026, 6, 1), duration_hours=2)

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 8))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

# ── panel 1: PV seasonal variation ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

months      = [1,       3,        6,        9,        12     ]
month_names = ["Jan",   "Mar",    "Jun",    "Sep",    "Dec"  ]
m_colors    = ["#1A3A6B","#2E6FAA","#EF9F27","#D45A1A","#0D2A50"]

for m, mn, mc in zip(months, month_names, m_colors):
    ax1.plot(
        hours, [bell_pv(h, m) for h in hours],
        color=mc,
        lw=2.5 if mn == "Jun" else 1.5,
        alpha=1.0 if mn == "Jun" else 0.7,
        label=mn,
    )
ax1.fill_between(hours, static_pv, alpha=0.15, color=C["pv"])
ax1.plot(hours, static_pv, color=C["pv"], lw=2.5, ls="--", label="Config (Jun)")
ax1.set_title("PV Generation — Seasonal Variation", fontweight="bold", fontsize=11)
ax1.set_xlabel("Hour of day")
ax1.set_ylabel("Farm array output (kW)")
ax1.set_xticks(range(0, 24, 3))
ax1.legend(fontsize=8, ncol=3)
ax1.set_ylim(0, 6)

# ── panel 2: farm load profile ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

consumers_def = [
    ("Electric fence",    0.2, "#888780", [(0,  24)]),
    ("Irrigation (morn)", 3.0, "#2E6FAA", [(6,  8) ]),
    ("Irrigation (eve)",  3.0, "#2E6FAA", [(19, 21)]),
    ("Cold storage",      1.2, "#028090", [(8,  20)]),
    ("Workshop tools",    2.5, "#BA7517", [(8,  17)]),
    ("Office HVAC",       1.5, "#EF9F27", [(8,  18)]),
    ("Barn doors",        0.5, "#3A7A0E", [(7,  8) ]),
    ("Lighting",          0.8, "#C0312F", [(20, 23)]),
    ("Security lighting", 0.3, "#6B21A8", [(22, 24), (0, 6)]),
]

bottom = np.zeros(24)
for name, kw, color, windows in consumers_def:
    vals = np.zeros(24)
    for s, e in windows:
        vals[s:e] = kw
    ax2.bar(hours, vals, bottom=bottom, color=color, width=0.9, alpha=0.85, label=name)
    bottom += vals

ax2.plot(hours, static_load, color="black", lw=2, ls="--", label="Total load")
ax2.set_title("Farm Load Profile — All Consumers", fontweight="bold", fontsize=11)
ax2.set_xlabel("Hour of day")
ax2.set_ylabel("Load (kW)")
ax2.set_xticks(range(0, 24, 3))
ax2.legend(fontsize=7, ncol=2, loc="upper right")
ax2.set_ylim(0, 11)

# ── panel 3: predictor backends comparison ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

for i, seed in enumerate([42, 77, 123, 200]):
    ax3.plot(
        hours, noisy_pv(seed),
        color=C["syn"], alpha=0.35, lw=1.2,
        label="Synthetic samples" if i == 0 else None,
    )
ax3.plot(hours, static_pv,  color=C["pv"],    lw=2.5,          label="Static profile (config)")
ax3.plot(hours, stub_vals,  color="#185FA5",   lw=2.0, ls=":",  label="WeatherStub (seasonal)")
ax3.set_title("PV Predictor Backends — Training Data", fontweight="bold", fontsize=11)
ax3.set_xlabel("Hour of day")
ax3.set_ylabel("Farm array output (kW)")
ax3.set_xticks(range(0, 24, 3))
ax3.legend(fontsize=9)
ax3.set_ylim(0, 6.5)
ax3.text(
    0.98, 0.97,
    "backend: static | stub | nn | openmeteo",
    transform=ax3.transAxes, ha="right", va="top",
    fontsize=8, color="#888780", style="italic",
)

# ── panel 4: ForecastBundle charging headroom ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

head_pos = [max(0.0, h) for h in headroom]
head_neg = [min(0.0, h) for h in headroom]

ax4.bar(hours, head_pos, color=C["head"], alpha=0.7, width=0.9, label="Available headroom")
ax4.bar(hours, head_neg, color=C["load"], alpha=0.7, width=0.9, label="Grid pressure")
ax4.axhline(0, color="black", lw=0.8)
ax4.axvline(best_h, color=C["pv"], lw=2, ls="--", alpha=0.9,
            label=f"Best 2h window: {best_h}:00")
ax4.axvspan(best_h, best_h + 2, alpha=0.12, color=C["pv"])

# Tariff bands (background shading)
tariff_bands = [
    (0,  8,  "#2E6FAA", "valle\n0.15€",  4  ),
    (10, 14, "#C0312F", "punta\n0.20€",  12 ),
    (18, 22, "#C0312F", "punta\n0.20€",  20 ),
    (8,  10, "#EF9F27", None,             None),
    (14, 18, "#EF9F27", None,             None),
    (22, 24, "#EF9F27", None,             None),
]
for s, e, col, label, lx in tariff_bands:
    ax4.axvspan(s, e, alpha=0.06, color=col, zorder=0)
    if label:
        ax4.text(lx, 14.5, label, ha="center", fontsize=7, color=col)

ax4.set_title("ForecastBundle — Charging Headroom", fontweight="bold", fontsize=11)
ax4.set_xlabel("Hour of day")
ax4.set_ylabel("Available power (kW)")
ax4.set_xticks(range(0, 24, 3))
ax4.legend(fontsize=9)
ax4.set_ylim(-3, 16)

# ── save ──────────────────────────────────────────────────────────────────────
fig.suptitle("HARVEST Prediction Module — Overview", fontsize=13, fontweight="bold", y=1.01)
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_path}")
