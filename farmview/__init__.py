"""
HARVEST farmview — farm map and MARL dashboard visualizations.

Usage::

    from farmview import render_farm, render_marl_log

    # Farm map (end-of-simulation snapshot)
    fig = render_farm(
        tractors=sim.config.tractors,
        tasks=sim.config.tasks,
        chargers=sim.config.chargers,
        model=sim.config.tractors_model,
        output="outputs/farm_map.png",
    )

    # MARL agent dashboard (requires marl scenario + record_step() wired in)
    fig = render_marl_log(
        step_log=sim.marl_engine.step_log,
        output="outputs/marl_dashboard.png",
    )

Command-line::

    python -m farmview --scenario full_smart --output outputs/farm_map.png
    python -m farmview --scenario marl       --output outputs/farm_map.png
"""

from ._renderer import render_farm
from ._marl import render_marl_log

__all__ = ["render_farm", "render_marl_log"]
