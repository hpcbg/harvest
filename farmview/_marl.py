from __future__ import annotations

from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from ._colors import (
    MARL_TRACTOR_CMAP, MARL_CHARGER_CMAP, MARL_LOAD_CMAP,
    REWARD_COST_COLOR, REWARD_PEAK_COLOR, REWARD_TASK_COLOR,
    TARIFF_VALLE_BG, TARIFF_LLANO_BG, TARIFF_PUNTA_BG,
    TRACTOR_STATUS_COLOR, EVENT_COLORS,
)

# Action labels for the legend
_TRACTOR_ACTION_LABELS = {0: "Idle", 1: "Request charge"}
_CHARGER_ACTION_LABELS = {0: "Off", 1: "Low (50%)", 2: "Full"}
_LOAD_ACTION_LABELS    = {0: "On", 1: "Shed (off)"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_hours(step_log) -> list:
    t0 = step_log[0].timestamp
    return [(s.timestamp - t0).total_seconds() / 3600.0 for s in step_log]


def _tariff_bg(tariff_norm: float) -> str:
    if tariff_norm == 0.0:   return TARIFF_VALLE_BG
    if tariff_norm >= 1.0:   return TARIFF_PUNTA_BG
    return TARIFF_LLANO_BG


def _shade_tariff(ax, hours: list, step_log) -> None:
    """Fill x-axis background with tariff-period colour."""
    for i in range(len(step_log) - 1):
        ax.axvspan(hours[i], hours[i + 1],
                   facecolor=_tariff_bg(step_log[i].tariff_norm),
                   alpha=0.55, zorder=0)
    if step_log:
        ax.axvspan(hours[-1], hours[-1] + 0.25,
                   facecolor=_tariff_bg(step_log[-1].tariff_norm),
                   alpha=0.55, zorder=0)


def _draw_event_markers(ax, sim_t0, events, y_frac: float = 0.97) -> None:
    """Draw vertical dashed lines at event times, with a short label at the top."""
    if not events:
        return
    for ev in events:
        h = (ev.timestamp - sim_t0).total_seconds() / 3600.0
        color = EVENT_COLORS.get(ev.event_type, "#7F8C8D")
        ax.axvline(h, color=color, lw=1.2, linestyle="--", alpha=0.8, zorder=5)
        ylim = ax.get_ylim()
        ypos = ylim[0] + (ylim[1] - ylim[0]) * y_frac
        ax.text(h + 0.05, ypos, ev.label,
                fontsize=5, color=color, va="top", ha="left",
                rotation=90, zorder=6)


def _build_heatmap(step_log, tractor_ids, charger_ids, load_ids):
    """Return a 2-D array (n_agents × n_steps) of action integers.

    Load values in the log are -1 (not scheduled), 0 (ON), 1 (shed).
    We shift them by +1 to give imshow the range [0, 2].
    """
    n_tr, n_ch, n_ld = len(tractor_ids), len(charger_ids), len(load_ids)
    n_agents = n_tr + n_ch + n_ld
    steps    = len(step_log)
    mat = np.zeros((n_agents, steps), dtype=float)
    for t, log in enumerate(step_log):
        for r, tid in enumerate(tractor_ids):
            mat[r, t] = log.tractor_actions.get(tid, 0)
        for r, cid in enumerate(charger_ids):
            mat[n_tr + r, t] = log.charger_actions.get(cid, 0)
        for r, lid in enumerate(load_ids):
            raw = log.load_actions.get(lid, -1)
            mat[n_tr + n_ch + r, t] = raw + 1   # shift: -1→0, 0→1, 1→2
    return mat


# ── sub-panel renderers ───────────────────────────────────────────────────────

def _draw_heatmap(ax, step_log, tractor_ids, charger_ids, load_ids, events=None) -> None:
    hours   = _to_hours(step_log)
    mat     = _build_heatmap(step_log, tractor_ids, charger_ids, load_ids)
    n_tr, n_ch, n_ld = len(tractor_ids), len(charger_ids), len(load_ids)
    n_agents = n_tr + n_ch + n_ld

    agent_labels = (
        [f"T {tid.replace('tractor_', '')}" for tid in tractor_ids]
        + [f"CH{cid.replace('charger_', '')}" for cid in charger_ids]
        + [f"LD {lid[:9]}" for lid in load_ids]
    )

    x0, x1 = hours[0], hours[-1] + 0.25
    ax.set_xlim(x0, x1 + 0.3)
    ax.set_ylim(-0.5, n_agents - 0.5)

    cmaps  = (
        [mcolors.ListedColormap(MARL_TRACTOR_CMAP)] * n_tr
        + [mcolors.ListedColormap(MARL_CHARGER_CMAP)] * n_ch
        + [mcolors.ListedColormap(MARL_LOAD_CMAP)] * n_ld
    )
    # Load is 3-state (0–2), tractor 2-state (0–1), charger 3-state (0–2)
    vmaxes = [1] * n_tr + [2] * n_ch + [2] * n_ld

    for r in range(n_agents):
        row = mat[r, :].reshape(1, -1)
        ax.imshow(
            row,
            aspect="auto",
            extent=[x0, x1, r - 0.5, r + 0.5],
            vmin=0, vmax=vmaxes[r],
            cmap=cmaps[r],
            origin="lower",
            interpolation="nearest",
            zorder=1,
        )

    # Group separator lines
    if n_tr and n_ch:
        ax.axhline(n_tr - 0.5, color="white", lw=2, zorder=3)
    if n_ch and n_ld:
        ax.axhline(n_tr + n_ch - 0.5, color="white", lw=2, zorder=3)

    ax.set_yticks(range(n_agents))
    ax.set_yticklabels(agent_labels, fontsize=7)
    ax.set_xlabel("Hour of day", fontsize=8)
    ax.set_title("Agent Decisions Over Time", fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(axis="x", labelsize=7)

    # Group annotations on right margin
    group_info = []
    if n_tr:
        group_info.append((n_tr / 2 - 0.5, "Tractors", TRACTOR_STATUS_COLOR["charging"]))
    if n_ch:
        group_info.append((n_tr + n_ch / 2 - 0.5, "Chargers", "#1A6EBD"))
    if n_ld:
        group_info.append((n_tr + n_ch + n_ld / 2 - 0.5, "Loads", "#E67E22"))
    for row_center, label, color in group_info:
        ax.annotate(
            label,
            xy=(1.004, row_center / n_agents),
            xycoords="axes fraction",
            fontsize=6.5, color=color, va="center", ha="left", rotation=90,
        )

    # Action legend — unambiguous colors across all three agent groups
    legend_patches = [
        mpatches.Patch(facecolor=MARL_TRACTOR_CMAP[1], label="Tractor: request charge"),
        mpatches.Patch(facecolor=MARL_TRACTOR_CMAP[0], label="Tractor: idle"),
        mpatches.Patch(facecolor=MARL_CHARGER_CMAP[2], label="Charger: full"),
        mpatches.Patch(facecolor=MARL_CHARGER_CMAP[1], label="Charger: low 50%"),
        mpatches.Patch(facecolor=MARL_CHARGER_CMAP[0], label="Charger: off"),
        mpatches.Patch(facecolor=MARL_LOAD_CMAP[2],    label="Load: shed"),
        mpatches.Patch(facecolor=MARL_LOAD_CMAP[1],    label="Load: on"),
        mpatches.Patch(facecolor=MARL_LOAD_CMAP[0],    label="Load: off-shift"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=5.5,
              ncol=2, framealpha=0.9, title="Actions", title_fontsize=6)

    if events:
        _draw_event_markers(ax, step_log[0].timestamp, events, y_frac=0.85)


def _draw_soc_traces(ax, step_log, tractor_ids, events=None) -> None:
    hours = _to_hours(step_log)
    _shade_tariff(ax, hours, step_log)

    tractor_colors = ["#2E86C1", "#E67E22", "#27AE60", "#8E44AD", "#E74C3C"]
    for i, tid in enumerate(tractor_ids):
        color  = tractor_colors[i % len(tractor_colors)]
        label  = tid.replace("tractor_", "Tractor ")
        soc    = [log.tractor_soc.get(tid, 0) for log in step_log]
        ch_kw  = [log.tractor_charge_kw.get(tid, 0) for log in step_log]

        ax.plot(hours, soc, color=color, lw=1.8, label=label, zorder=3)

        # Dots where the tractor is actually charging
        ch_h = [h for h, kw in zip(hours, ch_kw) if kw > 0]
        ch_s = [soc[j] for j, kw in enumerate(ch_kw) if kw > 0]
        if ch_h:
            ax.scatter(ch_h, ch_s, s=18, color=color, marker="o",
                       zorder=4, alpha=0.85)

    ax.axhline(90, color="#BDC3C7", lw=0.8, linestyle="--", zorder=2)
    ax.axhline(20, color="#E74C3C", lw=0.8, linestyle=":",  zorder=2)
    ax.set_xlim(hours[0], hours[-1] + 0.3)
    ax.set_ylim(0, 108)
    ax.set_ylabel("SOC (%)", fontsize=8)
    ax.set_xlabel("Hour of day", fontsize=8)
    ax.set_title("Tractor Battery State (SOC)", fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7)

    if events:
        _draw_event_markers(ax, step_log[0].timestamp, events, y_frac=0.97)

    # Combined legend: SOC lines + tariff bands
    tariff_patches = [
        mpatches.Patch(facecolor=TARIFF_VALLE_BG, edgecolor="#BDC3C7", label="Valle (cheap 00–08h)"),
        mpatches.Patch(facecolor=TARIFF_LLANO_BG, edgecolor="#BDC3C7", label="Llano (mid)"),
        mpatches.Patch(facecolor=TARIFF_PUNTA_BG, edgecolor="#BDC3C7", label="Punta (peak)"),
    ]
    soc_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=soc_handles + tariff_patches,
              loc="lower right", fontsize=6, ncol=2, framealpha=0.9)


def _draw_rewards(ax, step_log, events=None) -> None:
    hours = _to_hours(step_log)
    has_data = any(
        log.cost_reward != 0 or log.task_reward != 0 or log.peak_reward != 0
        for log in step_log
    )

    if not has_data:
        ax.text(0.5, 0.5,
                "No reward data.\nCall marl_engine.record_step() in the simulation loop.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="#7F8C8D")
        ax.axis("off")
        return

    cost_r = [log.cost_reward for log in step_log]
    peak_r = [log.peak_reward for log in step_log]
    task_r = [log.task_reward for log in step_log]
    total  = [c + p + t for c, p, t in zip(cost_r, peak_r, task_r)]
    cum    = list(np.cumsum(total))

    # Stacked bar chart
    ax.bar(hours, cost_r, width=0.20, color=REWARD_COST_COLOR, alpha=0.75, label="Cost penalty")
    bottom_1 = cost_r
    ax.bar(hours, peak_r, width=0.20, bottom=bottom_1,
           color=REWARD_PEAK_COLOR, alpha=0.75, label="Peak penalty")
    bottom_2 = [c + p for c, p in zip(cost_r, peak_r)]
    ax.bar(hours, task_r, width=0.20, bottom=bottom_2,
           color=REWARD_TASK_COLOR, alpha=0.75, label="Task reward")

    ax.axhline(0, color="#7F8C8D", lw=0.6)

    # Cumulative on twin axis
    ax2 = ax.twinx()
    ax2.plot(hours, cum, color="#1A252F", lw=1.5, label="Cumulative")
    ax2.set_ylabel("Cumulative reward", fontsize=7, color="#1A252F")
    ax2.tick_params(labelsize=6, colors="#1A252F")

    ax.set_xlim(hours[0], hours[-1] + 0.25)
    ax.set_xlabel("Hour of day", fontsize=8)
    ax.set_ylabel("Step reward", fontsize=8)
    ax.set_title("Reward Decomposition", fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper left", ncol=1, framealpha=0.9)

    if events:
        _draw_event_markers(ax, step_log[0].timestamp, events, y_frac=0.97)


def _draw_grid_power(ax, step_log, events=None) -> None:
    hours     = _to_hours(step_log)
    grid_vals = [log.grid_kw   for log in step_log]
    pv_vals   = [log.pv_shape * 100 for log in step_log]   # scale to 0-100 for dual axis

    ax.fill_between(hours, grid_vals, alpha=0.45, color="#E74C3C")
    ax.plot(hours, grid_vals, color="#E74C3C", lw=1.2, label="Grid draw (kW)")

    ax2 = ax.twinx()
    ax2.plot(hours, pv_vals, color="#F1C40F", lw=1.5, linestyle="--",
             label="PV irradiance (%)")
    ax2.set_ylabel("PV irradiance (%)", fontsize=7, color="#B8860B")
    ax2.set_ylim(0, 130)
    ax2.tick_params(labelsize=6, colors="#B8860B")

    ax.set_xlabel("Hour of day", fontsize=8)
    ax.set_ylabel("Grid draw (kW)", fontsize=8)
    ax.set_title("Grid Power & PV Irradiance", fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper left", framealpha=0.9)

    if events:
        _draw_event_markers(ax, step_log[0].timestamp, events, y_frac=0.97)


# ── public API ────────────────────────────────────────────────────────────────

def render_marl_log(
    step_log,
    output=None,
    title: str = "MARL Agent Decisions",
    dpi: int = 150,
    events=None,
) -> plt.Figure:
    """Visualise a list of ``MARLStepLog`` entries.

    Produces a 3-row dashboard:

    * **Row 0** — agent decision heatmap (tractors / chargers / loads over time)
    * **Row 1** — tractor SOC traces with tariff-band background
    * **Row 2** — reward decomposition (left) + grid power & PV (right)

    Parameters
    ----------
    step_log
        List of ``MARLStepLog`` objects from ``MARLEnvironment.step_log``.
    output
        File path to save the PNG (or None to skip saving).
    title
        Figure title.
    dpi
        Output resolution.
    """
    if not step_log:
        raise ValueError("step_log is empty — run a MARL scenario first")

    tractor_ids = sorted(step_log[0].tractor_actions.keys())
    charger_ids = sorted(step_log[0].charger_actions.keys())
    # Collect load_ids across all steps so agents that first appear mid-day are included
    load_ids    = sorted({lid for s in step_log for lid in s.load_actions})
    n_agents    = len(tractor_ids) + len(charger_ids) + len(load_ids)

    heatmap_h = max(n_agents * 0.5, 2.5)   # proportional to number of agents

    fig = plt.figure(figsize=(16, 11), facecolor="#F0F2F5")
    gs  = GridSpec(
        3, 2, figure=fig,
        height_ratios=[heatmap_h, 3.5, 2.5],
        width_ratios=[2, 1],
        left=0.08, right=0.96, top=0.93, bottom=0.07,
        hspace=0.42, wspace=0.30,
    )

    ax_heat   = fig.add_subplot(gs[0, :])    # full-width heatmap
    ax_soc    = fig.add_subplot(gs[1, :])    # full-width SOC traces
    ax_reward = fig.add_subplot(gs[2, 0])    # reward decomposition
    ax_grid   = fig.add_subplot(gs[2, 1])    # grid power + PV

    _draw_heatmap(ax_heat,  step_log, tractor_ids, charger_ids, load_ids, events=events)
    _draw_soc_traces(ax_soc, step_log, tractor_ids, events=events)
    _draw_rewards(ax_reward, step_log, events=events)
    _draw_grid_power(ax_grid, step_log, events=events)

    fig.suptitle(title, fontsize=12, fontweight="bold", color="#1A252F", y=0.97)

    if output:
        fig.savefig(
            output, dpi=dpi, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    return fig
