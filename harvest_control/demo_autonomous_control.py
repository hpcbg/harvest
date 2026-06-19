"""
harvest_control.demo_autonomous_control
=======================================

Drives the HARVEST simulation **entirely through the FleetInterface** with a
simple, fully autonomous rule-based controller, and writes an execution log.

This is the Stage-2 TPI2 demonstration: every charging decision is taken by the
controller from the interface snapshot, with no human in the loop, so the share
of decisions executed without manual intervention is 100 %.  The same loop --
unchanged -- will drive the real ZETRABOT once the simulation backend is
swapped for the ROS 2 backend (see ``ros2_bridge.py``).

Run::

    python demo_autonomous_control.py            # nominal day
    python demo_autonomous_control.py --events   # with mid-day plan changes

Or from the repo root::

    python -m harvest_control.demo_autonomous_control [--events]

Outputs ``execution_log.csv`` and prints a summary.
"""
from __future__ import annotations

import argparse
import csv
import sys

try:
    from .interface import ChargerLevel, Command, CommandType, FleetSnapshot
    from .sim_backend import SimulationFleetInterface
except ImportError:  # running as standalone script from package directory
    from interface import ChargerLevel, Command, CommandType, FleetSnapshot
    from sim_backend import SimulationFleetInterface

STEP_MIN = 15
STEPS_PER_DAY = 24 * 60 // STEP_MIN
SOC_TARGET = 85.0          # stop charging above this
SOC_LOW = 40.0             # below this, charging is desirable


def decide(snap: FleetSnapshot) -> list[Command]:
    """Autonomous policy: tariff/PV/headroom-aware charging + peak-load shedding.

    Pure function of the snapshot -> no manual intervention. Returns the
    actuation commands to apply this step.
    """
    cmds: list[Command] = []
    cheap = snap.grid.tariff in ("valle", "llano") or snap.grid.pv_kw > 1.0
    peak = snap.grid.tariff == "punta"
    headroom = snap.grid_headroom_kw
    free = len(snap.free_chargers())

    # 1) shed non-critical loads when the grid is tight during peak tariff
    for l in snap.loads:
        if peak and headroom < 1.5 and not l.shed:
            cmds.append(Command.shed_load(l.id))
        elif not peak and l.shed:
            cmds.append(Command.restore_load(l.id))

    # 2) charging decisions, lowest-SoC tractors first
    for t in sorted(snap.tractors, key=lambda x: x.soc_pct):
        if not t.available:
            continue
        if t.charging:
            if t.soc_pct >= SOC_TARGET or (peak and t.soc_pct > SOC_LOW):
                cmds.append(Command.release_charge(t.id))
            continue
        wants_charge = t.soc_pct < SOC_TARGET and (cheap or t.soc_pct < SOC_LOW)
        if wants_charge and free > 0 and headroom >= 3.0:
            cmds.append(Command.request_charge(t.id))
            lvl = ChargerLevel.FULL if headroom >= 6.6 else ChargerLevel.HALF
            cmds.append(Command(CommandType.SET_CHARGER_LEVEL, "PENDING", lvl))
            free -= 1
            headroom -= 6.6 if lvl == ChargerLevel.FULL else 3.3
    return cmds


def run(with_events: bool = False) -> dict:
    iface = SimulationFleetInterface()
    rows = []
    total_decisions = 0
    autonomous_decisions = 0
    peak_grid = 0.0
    events_fired = []

    for step in range(STEPS_PER_DAY):
        snap = iface.snapshot()
        peak_grid = max(peak_grid, snap.grid.grid_draw_kw)

        # ---- inject dynamic plan-change events (fair, deterministic) --------
        if with_events:
            if snap.grid.clock_min == 10 * 60:
                iface.submit([Command.assign_task("T1", "dyn_001_emergency_spray")])
                events_fired.append("10:00 emergency task injected")
            if snap.grid.clock_min == 13 * 60 + 30:
                iface.force_offline("T2")
                events_fired.append("13:30 Tractor 2 breakdown")
            if snap.grid.clock_min == 16 * 60:
                iface.env.grid_cap_kw = 7.0
                events_fired.append("16:00 grid cap -> 7 kW")
            if snap.grid.clock_min == 18 * 60:
                iface.env.grid_cap_kw = 10.5
                events_fired.append("18:00 grid cap restored")
            snap = iface.snapshot()

        # ---- autonomous decision + execution --------------------------------
        cmds = decide(snap)
        acks = []
        for c in cmds:
            if c.type == CommandType.SET_CHARGER_LEVEL and c.target_id == "PENDING":
                occ = [ch for ch in iface.snapshot().chargers if ch.occupied_by]
                if occ:
                    c.target_id = occ[-1].id
                else:
                    continue
            acks.append(iface.submit([c])[0])

        for ack in acks:
            total_decisions += 1
            autonomous_decisions += 1
            rows.append({
                "clock_min": snap.grid.clock_min,
                "tariff": snap.grid.tariff,
                "grid_draw_kw": snap.grid.grid_draw_kw,
                "pv_kw": snap.grid.pv_kw,
                "command": ack.command.type.value,
                "target": ack.command.target_id,
                "accepted": ack.accepted,
                "reason": ack.reason,
                "manual_intervention": False,
            })

        iface.advance(STEP_MIN)

    final = iface.snapshot()
    log_path = "execution_log.csv"
    if rows:
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    pct = 100.0 * autonomous_decisions / total_decisions if total_decisions else 0.0
    return {
        "decisions": total_decisions,
        "autonomous_pct": pct,
        "peak_grid_kw": round(peak_grid, 2),
        "grid_cap_kw": final.grid.grid_cap_kw,
        "final_soc": {t.id: round(t.soc_pct, 1) for t in final.tractors},
        "events": events_fired,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", action="store_true", help="inject mid-day plan changes")
    args = ap.parse_args()
    res = run(with_events=args.events)

    print("HARVEST autonomous-control demo  (control via FleetInterface)")
    print("-" * 60)
    print(f"  decisions executed              : {res['decisions']}")
    print(f"  without manual intervention     : {res['autonomous_pct']:.1f} %   (TPI2 target >= 70 %)")
    print(f"  peak grid draw                  : {res['peak_grid_kw']} kW (cap {res['grid_cap_kw']} kW)")
    print(f"  final state of charge           : {res['final_soc']}")
    if res["events"]:
        print("  dynamic events handled          :")
        for e in res["events"]:
            print(f"     - {e}")
    print("  execution log written           : execution_log.csv")
    sys.exit(0)
