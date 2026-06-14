"""
HARVEST farmview CLI.

    python -m farmview [options]

Runs one simulation scenario and renders:
  - farm_map.png   — top-down farm snapshot (always)
  - marl_dashboard.png — MARL agent log (only when scenario uses MARL)

Examples
--------
    python -m farmview
    python -m farmview --scenario marl --show
    python -m farmview --scenario smart --output outputs/smart_map.png --dpi 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render HARVEST farm map and MARL dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",       default="config.yaml",
                   help="Path to config.yaml (default: ./config.yaml)")
    p.add_argument("--scenario",     default="full_smart",
                   help="Scenario name to run (default: full_smart)")
    p.add_argument("--output",       default="outputs/farm_map.png",
                   help="Farm map output path (default: outputs/farm_map.png)")
    p.add_argument("--marl-output",  default="outputs/marl_dashboard.png",
                   help="MARL dashboard output path (default: outputs/marl_dashboard.png)")
    p.add_argument("--dpi",          default=150, type=int,
                   help="Output resolution in DPI (default: 150)")
    p.add_argument("--show",         action="store_true",
                   help="Open matplotlib window after saving")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}\n"
                 "Run from the harvest project directory or pass --config <path>.")

    try:
        from main import Simulator, ScenarioDef, load_yaml_with_local
    except ImportError as exc:
        sys.exit(
            f"Cannot import main.py: {exc}\n"
            "Run from the harvest project directory."
        )

    try:
        from farmview import render_farm, render_marl_log
    except ImportError as exc:
        sys.exit(f"Cannot import farmview package: {exc}")

    # ── load config ───────────────────────────────────────────────────────────
    cfg = load_yaml_with_local(cfg_path)

    scenario_defs = cfg.get("scenarios", [])
    sd_raw = next((s for s in scenario_defs if s["name"] == args.scenario), None)
    if sd_raw is None:
        names = [s["name"] for s in scenario_defs]
        sys.exit(f"Scenario '{args.scenario}' not found.\nAvailable: {names}")

    sdef = ScenarioDef(
        name=sd_raw["name"],
        charging_strategy=sd_raw["charging_strategy"],
        tractor_pv_enabled=bool(sd_raw.get("tractor_pv_enabled", False)),
        load_shedding=bool(sd_raw.get("load_shedding", False)),
        use_marl=bool(sd_raw.get("use_marl", False)),
    )

    # ── run simulation ────────────────────────────────────────────────────────
    print(f"Running scenario '{sdef.name}' ...")
    sim = Simulator(cfg, sdef)
    sim.run()

    done  = sum(1 for t in sim.config.tasks if t.is_done)
    total = len(sim.config.tasks)
    print(f"Done - {done}/{total} tasks completed")

    farm_cfg = cfg.get("farm", {}).get("map", {})
    map_w = float(farm_cfg.get("width_m", 800.0))
    map_h = float(farm_cfg.get("height_m", 500.0))

    # ── farm map ──────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    farm_title = (
        f"HARVEST — {sdef.name.replace('_', ' ').title()}  |  "
        f"Tasks {done}/{total} completed  |  End-of-day snapshot"
    )
    render_farm(
        tractors=sim.config.tractors,
        tasks=sim.config.tasks,
        chargers=sim.config.chargers,
        model=sim.config.tractors_model,
        title=farm_title,
        map_w=map_w,
        map_h=map_h,
        output=out,
        dpi=args.dpi,
    )
    print(f"Farm map  -> {out}")

    # ── MARL dashboard ────────────────────────────────────────────────────────
    if sim.marl_engine is not None and sim.marl_engine.step_log:
        marl_out = Path(args.marl_output)
        marl_out.parent.mkdir(parents=True, exist_ok=True)

        marl_title = (
            f"MARL Agent Log — {sdef.name.replace('_', ' ').title()}  |  "
            f"{len(sim.marl_engine.step_log)} time steps"
        )
        render_marl_log(
            step_log=sim.marl_engine.step_log,
            output=marl_out,
            title=marl_title,
            dpi=args.dpi,
        )
        print(f"MARL dash -> {marl_out}")
    elif sdef.use_marl:
        print(
            "Warning: MARL scenario ran but step_log is empty.\n"
            "Check that marl_engine.record_step() is called in Simulator.run()."
        )

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
