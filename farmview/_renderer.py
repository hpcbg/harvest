from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch, Rectangle, Wedge
from matplotlib.gridspec import GridSpec

from ._colors import (
    PHASE_COLOR, PHASE_LABEL,
    PRIORITY_EDGE_COLOR, PRIORITY_EDGE_WIDTH,
    TASK_TYPE_MARKER, DEFAULT_MARKER,
    TRACTOR_STATUS_COLOR, TRACTOR_STATUS_LABEL,
    SOC_HIGH, SOC_MID, SOC_LOW,
    FARM_BG, DEPOT_BG, CHARGER_COLOR,
    EVENT_COLORS,
)

_DEFAULT_MAP_W = 800.0
_DEFAULT_MAP_H = 500.0

# Tractor body size in map units (~3m × 2m upscaled for readability)
_BODY_W = 30
_BODY_H = 20


# ── helpers ───────────────────────────────────────────────────────────────────

def _soc_color(soc: float) -> str:
    if soc >= 60:   return SOC_HIGH
    if soc >= 30:   return SOC_MID
    return SOC_LOW


def _tractor_status(tr, tasks_by_id: Dict) -> str:
    if not tr.enabled:
        return "offline"
    if tr.current_task_id is not None:
        task = tasks_by_id.get(tr.current_task_id)
        if task:
            return "transit" if task.phase == "TRANSIT" else "executing"
    return "charging" if tr.is_charging else "idle"


def _task_marker_style(name: str):
    first = name.lower().split()[0]
    return TASK_TYPE_MARKER.get(first, DEFAULT_MARKER)


# ── farm background ───────────────────────────────────────────────────────────

def _draw_farm_bg(ax, map_w: float, map_h: float) -> None:
    ax.set_facecolor(FARM_BG)
    for x in range(0, int(map_w) + 1, 100):
        ax.axvline(x, color="#C8E6C9", lw=0.5, zorder=0)
    for y in range(0, int(map_h) + 1, 100):
        ax.axhline(y, color="#C8E6C9", lw=0.5, zorder=0)
    ax.add_patch(Rectangle(
        (0, 0), map_w, map_h,
        fill=False, edgecolor="#4CAF50", linewidth=2.5, zorder=1,
    ))


def _draw_depot(ax, chargers) -> None:
    if not chargers:
        return
    xs = [c.location[0] for c in chargers]
    ys = [c.location[1] for c in chargers]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    ax.add_patch(FancyBboxPatch(
        (cx - 55, cy - 50), 110, 100,
        boxstyle="round,pad=5",
        facecolor=DEPOT_BG, edgecolor="#BDC3C7",
        linewidth=1.5, linestyle="--", alpha=0.8, zorder=1,
    ))
    ax.text(cx, cy + 55, "Depot",
            ha="center", va="bottom", fontsize=7,
            color="#7F8C8D", fontstyle="italic", zorder=2)


# ── charger stations ──────────────────────────────────────────────────────────

def _draw_charger(ax, ch, label_above: bool = False) -> None:
    x, y = ch.location
    hw, hh = 9, 12
    ax.add_patch(FancyBboxPatch(
        (x - hw, y - hh), hw * 2, hh * 2,
        boxstyle="round,pad=1.5",
        facecolor=CHARGER_COLOR, edgecolor="white",
        linewidth=1.5, zorder=3,
    ))
    ax.text(x, y + 1, "⚡", fontsize=8,
            ha="center", va="center", color="#F1C40F", fontweight="bold", zorder=4)
    label_y  = y + hh + 4  if label_above else y - hh - 4
    label_va = "bottom"     if label_above else "top"
    ax.text(x, label_y,
            ch.charger_id.replace("charger_", "CH"),
            fontsize=5.5, ha="center", va=label_va,
            color=CHARGER_COLOR, fontweight="bold", zorder=4)


def _draw_charger_line(ax, tr, ch) -> None:
    tx, ty = tr.location
    cx, cy = ch.location
    ax.annotate(
        "", xy=(cx, cy), xytext=(tx, ty),
        arrowprops=dict(
            arrowstyle="-", color="#2980B9", lw=1.3, linestyle="dashed",
        ),
        zorder=2,
    )


# ── task markers ──────────────────────────────────────────────────────────────

def _draw_tasks(ax, tasks, injected_task_ids=None) -> None:
    for task in tasks:
        x, y = task.location
        marker, _ = _task_marker_style(task.name)
        phase_color = PHASE_COLOR.get(task.phase, "#9DA8B7")
        edge_color  = PRIORITY_EDGE_COLOR.get(task.priority, "#95A5A6")
        size = 260 if task.phase != "DONE" else 190

        # Priority ring (outer, larger)
        ax.scatter([x], [y], s=size * 1.75, marker=marker,
                   c=edge_color, zorder=2, linewidths=0)
        # Phase fill (inner)
        ax.scatter([x], [y], s=size, marker=marker,
                   c=phase_color, edgecolors="white", linewidths=1.0,
                   zorder=3, alpha=0.95)

        # DONE: checkmark overlay
        if task.phase == "DONE":
            ax.text(x, y, "✓", fontsize=6, ha="center", va="center",
                    color="white", fontweight="bold", zorder=4)

        # TRANSIT / EXECUTING: progress wedge arc
        if task.phase in ("TRANSIT", "EXECUTING"):
            pct = (task.transit_progress_pct
                   if task.phase == "TRANSIT" else task.progress_pct)
            if pct > 0:
                angle = pct / 100.0 * 360.0
                ax.add_patch(Wedge(
                    (x, y), r=12, theta1=90 - angle, theta2=90, width=3.5,
                    facecolor=phase_color, edgecolor="white",
                    linewidth=0.7, alpha=0.8, zorder=4,
                ))

        # Injected tasks: purple outer ring to mark dynamic origin
        if injected_task_ids and task.task_id in injected_task_ids:
            ax.scatter([x], [y], s=size * 3.2, marker="o",
                       c="#8E44AD", zorder=1, linewidths=0, alpha=0.45)

        # Task ID label below
        ax.text(x, y - 15, task.task_id.replace("task_", "#"),
                fontsize=5, ha="center", va="top", color="#2C3E50", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.3))


# ── tractor symbols ───────────────────────────────────────────────────────────

def _draw_tractor(ax, tr, tasks_by_id: Dict, model) -> None:
    x, y = tr.location
    status = _tractor_status(tr, tasks_by_id)
    sc = TRACTOR_STATUS_COLOR[status]

    bx = x - _BODY_W / 2
    by = y - _BODY_H / 2

    # Main body
    ax.add_patch(FancyBboxPatch(
        (bx, by), _BODY_W, _BODY_H,
        boxstyle="round,pad=1.5",
        facecolor=sc, edgecolor="white", linewidth=2.0, alpha=0.88, zorder=5,
    ))

    # PV roof strip (yellow bar at top of body)
    if tr.has_pv_roof:
        ax.add_patch(Rectangle(
            (bx + 2, by + _BODY_H - 3), _BODY_W - 9, 3,
            facecolor="#F1C40F", edgecolor="#E67E22", linewidth=0.6, zorder=6,
        ))

    # Battery gauge (vertical bar on right side of body)
    gw, gh = 5, _BODY_H - 5
    gx, gy = bx + _BODY_W - gw - 1.5, by + 2.5
    ax.add_patch(Rectangle(
        (gx, gy), gw, gh,
        facecolor="#1C2833", edgecolor="white", linewidth=0.5, zorder=6,
    ))
    fill_h = max(0.3, gh * (tr.soc_percent / 100.0))
    ax.add_patch(Rectangle(
        (gx + 0.5, gy + 0.5), gw - 1, fill_h - 0.5,
        facecolor=_soc_color(tr.soc_percent), edgecolor="none", zorder=7,
    ))
    ax.text(gx + gw / 2, gy + gh / 2, f"{tr.soc_percent:.0f}",
            fontsize=4, ha="center", va="center", color="white",
            fontweight="bold", rotation=90, zorder=8)

    # Tractor icon: bold short label centred in the body area left of the gauge
    emoji_cx = bx + (_BODY_W - gw - 2) / 2
    ax.text(emoji_cx, by + _BODY_H / 2,
            tr.tractor_id.replace("tractor_", "T"),
            fontsize=8, ha="center", va="center",
            color="white", fontweight="bold", zorder=7)

    # Charging bolt indicator (top-left corner)
    if status == "charging":
        ax.text(bx + 3.5, by + _BODY_H - 4, "⚡",
                fontsize=6, color="#F1C40F", ha="center", va="top", zorder=8)

    # Offline: cross-out overlay
    if status == "offline":
        for x1, y1, x2, y2 in [(bx, by, bx+_BODY_W, by+_BODY_H),
                                 (bx+_BODY_W, by, bx, by+_BODY_H)]:
            ax.plot([x1, x2], [y1, y2], color="white", lw=2.5, zorder=9)
        ax.text(x - _BODY_W * 0.10, by + _BODY_H / 2, "X",
                fontsize=9, ha="center", va="center",
                color="white", fontweight="bold", zorder=10)

    # Label below body
    lbl_color = "#922B21" if status == "offline" else "#1A252F"
    ax.text(x, by - 5,
            tr.tractor_id.replace("tractor_", "T"),
            fontsize=6.5, ha="center", va="top", color=lbl_color, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5), zorder=8)


def _draw_transit_arrow(ax, tr, task) -> None:
    tx, ty = tr.location
    qx, qy = task.location
    dx, dy = qx - tx, qy - ty
    dist = math.hypot(dx, dy)
    if dist < 5:
        return
    # Nudge arrow start to body edge so it doesn't overlap the tractor icon
    body_r = math.hypot(_BODY_W / 2, _BODY_H / 2)
    sx = tx + dx / dist * body_r
    sy = ty + dy / dist * body_r
    ax.annotate(
        "", xy=(qx, qy), xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="-|>", color=TRACTOR_STATUS_COLOR["transit"],
            lw=1.5, linestyle="dashed", mutation_scale=10,
        ),
        zorder=2,
    )


# ── fleet panel (right side) ──────────────────────────────────────────────────

def _draw_fleet_panel(ax, tractors, tasks_by_id: Dict, model, events=None) -> None:
    n = len(tractors)
    if n == 0:
        return

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#F4F6F8")

    ax.text(0.5, 0.988, "Fleet Status",
            ha="center", va="top", fontsize=9, fontweight="bold", color="#1A252F",
            transform=ax.transAxes)

    events_h = 0.0
    if events:
        events_h = min(0.04 * len(events) + 0.04, 0.30)

    pad = 0.018
    header_h = 0.055
    card_h = (1.0 - header_h - pad - events_h) / n

    for i, tr in enumerate(tractors):
        status = _tractor_status(tr, tasks_by_id)
        sc = TRACTOR_STATUS_COLOR[status]

        cy_top = 1.0 - header_h - i * card_h
        cy_bot = cy_top - card_h + pad
        h = card_h - pad * 1.5   # usable card height

        # Card background with coloured border
        ax.add_patch(FancyBboxPatch(
            (0.04, cy_bot), 0.92, h,
            boxstyle="round,pad=0.008",
            facecolor="white", edgecolor=sc, linewidth=2,
            transform=ax.transAxes, zorder=2,
        ))

        # Tractor name (top-left)
        ax.text(0.09, cy_bot + h * 0.80,
                tr.tractor_id.replace("tractor_", "Tractor "),
                fontsize=7.5, fontweight="bold", color="#1A252F",
                transform=ax.transAxes, va="center")

        # Status (top-right)
        stxt = TRACTOR_STATUS_LABEL.get(status, status.title())
        if status == "charging" and tr.actual_charge_power_kw > 0:
            stxt += f"  {tr.actual_charge_power_kw:.1f} kW"
        ax.text(0.95, cy_bot + h * 0.80, stxt,
                fontsize=6.5, color=sc, ha="right", va="center",
                transform=ax.transAxes)

        # Battery bar (horizontal, middle of card)
        bx0, bw = 0.07, 0.69
        by0 = cy_bot + h * 0.34
        bh  = h * 0.28

        # Background
        ax.add_patch(Rectangle(
            (bx0, by0), bw, bh,
            facecolor="#ECF0F1", edgecolor="#BDC3C7", linewidth=0.8,
            transform=ax.transAxes, zorder=3,
        ))
        # Terminal nub
        ax.add_patch(Rectangle(
            (bx0 + bw, by0 + bh * 0.25), 0.014, bh * 0.5,
            facecolor="#7F8C8D", edgecolor="none",
            transform=ax.transAxes, zorder=3,
        ))
        # SOC fill
        fill_w = bw * (tr.soc_percent / 100.0)
        ax.add_patch(Rectangle(
            (bx0 + 0.004, by0 + 0.003), max(0, fill_w - 0.004), bh - 0.006,
            facecolor=_soc_color(tr.soc_percent), edgecolor="none",
            transform=ax.transAxes, zorder=4,
        ))
        # SOC label inside bar
        txt_color = "white" if tr.soc_percent > 25 else "#2C3E50"
        ax.text(bx0 + bw / 2, by0 + bh / 2, f"{tr.soc_percent:.0f}%",
                fontsize=7, ha="center", va="center", color=txt_color,
                fontweight="bold", transform=ax.transAxes, zorder=5)
        # kWh label right of bar
        kwh = tr.soc_percent / 100.0 * model.battery_capacity_kwh
        ax.text(bx0 + bw + 0.038, by0 + bh / 2, f"{kwh:.1f} kWh",
                fontsize=6, ha="left", va="center", color="#566573",
                transform=ax.transAxes, zorder=3)

        # PV + active task info (bottom of card)
        sub = []
        if tr.has_pv_roof:
            sub.append("☀ PV")
        if tr.current_task_id:
            task = tasks_by_id.get(tr.current_task_id)
            if task:
                pct = (task.progress_pct if task.phase == "EXECUTING"
                       else task.transit_progress_pct)
                sub.append(f"→ {task.name[:14]}  {pct:.0f}%")
        if sub:
            ax.text(0.08, cy_bot + h * 0.10, "  ".join(sub),
                    fontsize=5.5, color="#566573", va="center",
                    transform=ax.transAxes)

    # Event log (bottom of panel, only when events were fired)
    if events:
        ey_top = events_h
        ax.add_patch(FancyBboxPatch(
            (0.04, 0.005), 0.92, ey_top - 0.01,
            boxstyle="round,pad=0.006",
            facecolor="#F8F9FA", edgecolor="#BDC3C7", linewidth=1.0,
            transform=ax.transAxes, zorder=2,
        ))
        ax.text(0.5, ey_top - 0.008,
                "Plan changes",
                fontsize=6, ha="center", va="top", fontweight="bold",
                color="#566573", transform=ax.transAxes)
        for k, ev in enumerate(events):
            ey = ey_top - 0.032 - k * 0.038
            if ey < 0.01:
                break
            ec = EVENT_COLORS.get(ev.event_type, "#7F8C8D")
            ax.add_patch(mpatches.Circle(
                (0.08, ey + 0.005), 0.025,
                facecolor=ec, transform=ax.transAxes, zorder=3,
            ))
            ts = ev.timestamp.strftime("%H:%M")
            ax.text(0.14, ey + 0.005, f"{ts}  {ev.label}",
                    fontsize=5.5, va="center", color="#2C3E50",
                    transform=ax.transAxes)


# ── legend ─────────────────────────────────────────────────────────────────────

def _draw_legend(ax) -> None:
    handles = []
    for phase, color in PHASE_COLOR.items():
        handles.append(mpatches.Patch(
            facecolor=color, edgecolor="white", linewidth=0.5,
            label=PHASE_LABEL[phase],
        ))
    for status, color in TRACTOR_STATUS_COLOR.items():
        handles.append(mpatches.Patch(
            facecolor=color,
            label=f"Tractor: {TRACTOR_STATUS_LABEL[status]}",
        ))
    for priority, color in PRIORITY_EDGE_COLOR.items():
        handles.append(mlines.Line2D(
            [], [], color=color,
            linewidth=PRIORITY_EDGE_WIDTH[priority] + 0.5,
            label=f"Priority: {priority}",
        ))
    for key, (marker, label) in TASK_TYPE_MARKER.items():
        handles.append(mlines.Line2D(
            [], [], marker=marker, color="none",
            markerfacecolor="#7F8C8D", markeredgecolor="#7F8C8D",
            markersize=5.5, label=f"Task: {label}",
        ))
    ax.legend(
        handles=handles, loc="lower right",
        fontsize=5.5, ncol=3,
        framealpha=0.9, edgecolor="#BDC3C7",
        title="Legend", title_fontsize=6.5,
    )


# ── public API ────────────────────────────────────────────────────────────────

def render_farm(
    tractors,
    tasks,
    chargers,
    model,
    title: str = "Farm State",
    map_w: float = _DEFAULT_MAP_W,
    map_h: float = _DEFAULT_MAP_H,
    output=None,
    dpi: int = 150,
    events=None,
    injected_task_ids=None,
) -> plt.Figure:
    """Render a top-down farm map snapshot and return the matplotlib Figure.

    Parameters
    ----------
    tractors, tasks, chargers, model
        Objects from ``SimulationConfig`` after ``Simulator.run()``.
    title
        Figure title string.
    map_w, map_h
        Farm dimensions in metres (default 800 × 500).
    output
        File path to save the PNG (or None to skip saving).
    dpi
        Output resolution.
    """
    tasks_by_id    = {t.task_id: t for t in tasks}
    chargers_by_id = {c.charger_id: c for c in chargers}

    fig = plt.figure(figsize=(18, 10.5), facecolor="#F0F2F5")
    gs  = GridSpec(
        1, 2, figure=fig,
        width_ratios=[3.2, 1],
        left=0.03, right=0.98, top=0.93, bottom=0.04,
        wspace=0.03,
    )
    ax_map   = fig.add_subplot(gs[0])
    ax_fleet = fig.add_subplot(gs[1])

    # ── farm map ──────────────────────────────────────────────────────────────
    ax_map.set_xlim(0, map_w)
    ax_map.set_ylim(0, map_h)
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_xlabel("x (m)", fontsize=8)
    ax_map.set_ylabel("y (m)", fontsize=8)
    ax_map.tick_params(labelsize=7)

    _draw_farm_bg(ax_map, map_w, map_h)
    _draw_depot(ax_map, chargers)

    for i, ch in enumerate(chargers):
        _draw_charger(ax_map, ch, label_above=(i % 2 == 1))

    _draw_tasks(ax_map, tasks, injected_task_ids=injected_task_ids)

    # Transit arrows (below tractor icons)
    for tr in tractors:
        if tr.current_task_id:
            task = tasks_by_id.get(tr.current_task_id)
            if task and task.phase == "TRANSIT":
                _draw_transit_arrow(ax_map, tr, task)

    # Charger connection lines
    for tr in tractors:
        if tr.is_charging and tr.assigned_charger_id:
            ch = chargers_by_id.get(tr.assigned_charger_id)
            if ch:
                _draw_charger_line(ax_map, tr, ch)

    for tr in tractors:
        _draw_tractor(ax_map, tr, tasks_by_id, model)

    _draw_legend(ax_map)

    # ── fleet panel ───────────────────────────────────────────────────────────
    _draw_fleet_panel(ax_fleet, tractors, tasks_by_id, model, events=events)

    fig.suptitle(title, fontsize=13, fontweight="bold", color="#1A252F", y=0.98)

    if output:
        fig.savefig(
            output, dpi=dpi, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    return fig
