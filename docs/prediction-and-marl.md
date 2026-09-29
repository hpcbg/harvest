# Prediction module and MARL engine

## Prediction Module

The `predictor/` package satisfies **TPI3** (AI prediction module, target ≤25% MAPE) and provides the foundation for future MARL-based pro-active scheduling.

[![Prediction overview](../images/prediction_overview.png)](../images/prediction_overview.png)

> **Regenerate this image** after changing `config.yaml` (e.g. PV peak, consumers, tariffs):
> ```bash
> python generate_prediction_overview.py
> ```
> Output: `images/prediction_overview.png`

The four panels above show:

- **Top-left — PV Generation (Seasonal Variation)**: PV output varies 5× between winter (Jan ≈1 kW peak) and summer (Jun 5 kW peak). The `WeatherStub` backend models this with a seasonal cosine factor. The static config profile (dashed) is used by default.
- **Top-right — Farm Load Profile**: All 9 consumers stacked by schedule. Total load ranges from 0.5 kW overnight (fence only) to 9.7 kW at the morning irrigation + workshop peak.
- **Bottom-left — Predictor Backends**: Four faint lines show stochastic synthetic training samples (with noise). The static profile and seasonal stub are compared — the NN backend learns the bell-curve shape from these samples.
- **Bottom-right — ForecastBundle Charging Headroom**: `grid_cap + PV_forecast − load_forecast` computed for each hour. The best 2-hour charging window (12:00 in June) is highlighted. Tariff bands show cost context: valle (cheap, 00–08h), llano (medium), punta (expensive, 10–14h and 18–22h).

### Local configuration overrides

To change the backend (or any other setting) without modifying `config.yaml`, create a `config.local.yaml` file in the same folder. It is **gitignored** and deep-merged on top of `config.yaml` at every startup — no git commits needed.

```yaml
# config.local.yaml  (gitignored — safe to edit freely)
prediction:
  pv:
    backend: nn
    model_path: models/harvest_nn_hu50_ep2000_dropNone.keras
```

A ready-to-use template with all common examples is provided in `config.local.yaml.example`. Copy and rename it:

```bash
# Windows
copy config.local.yaml.example config.local.yaml

# macOS / Linux
cp config.local.yaml.example config.local.yaml
```

Delete or rename the file to revert to `config.yaml` defaults instantly.

### Backends

Select the backend in `config.local.yaml` (preferred) or `config.yaml`:

```yaml
# config.local.yaml — pick one backend:
prediction:
  pv:
    backend: static      # default — uses the static hourly profile from config
    # backend: stub      # seasonal bell curve, no dependencies, fully offline
    # backend: openmeteo # live weather forecast (free, no API key, needs internet)
    # backend: nn        # trained neural network (see training steps below)
    # model_path: models/harvest_nn_hu50_ep2000_dropNone.keras
```

| Backend | When to use | Internet | TensorFlow |
|---|---|---|---|
| `static` | Simulation, demos | No | No |
| `stub` | Offline seasonal estimate | No | No |
| `openmeteo` | Live pilot deployment | Yes | No |
| `nn` | TPI3 validation, MARL training features | No | Yes |

### Training the neural network (TPI3)

The architecture directly mirrors the paper *"Using Neural Network for Predicting the Load of Conveyor Systems"* (Tsvetanov et al.) - a single hidden-layer FFNN with ReLU activations, Xavier initialisation, and Adamax optimiser. The key additions for HARVEST are a third input feature (`month`) to capture seasonal PV variation, and two outputs: `pv_shape` (normalised 0–1 irradiance) and `farm_load_kw`.

```bash
# 1. Generate training and test datasets from synthetic data
python -m predictor.synthetic --weeks 12 --seed 42 --output train.npz
python -m predictor.synthetic --weeks 5  --seed 99 --output test.npz

# 2. Train (50 hidden units — best result in the paper)
pip install tensorflow
# bash / macOS / Linux:
python -m predictor.nn_predictor \
    --train  train.npz     \
    --test   test.npz      \
    --hidden 50            \
    --epochs 2000          \
    --out    models/

# PowerShell (Windows) — use backtick ` for line continuation:
python -m predictor.nn_predictor `
    --train  train.npz `
    --test   test.npz  `
    --hidden 50        `
    --epochs 2000      `
    --out    models/

# 3. Enable in config.yaml
#    prediction.pv.backend: nn
#    prediction.pv.model_path: models/harvest_nn_hu50_ep2000_dropNone.keras
```

The training script prints MAPE per output at the end. A 50-HU network on 12 weeks of synthetic data consistently meets the ≤25% MAPE target (TPI3). A `*_loss.csv` file is also written alongside the model for loss curve analysis.

### Using `ForecastBundle` for pro-active scheduling

```python
from predictor import build_predictors, ForecastBundle
import yaml
from datetime import date, datetime

cfg    = yaml.safe_load(open("config.yaml"))
pv, ld = build_predictors(cfg)
bundle = ForecastBundle(pv, ld, grid_max_kw=10.5)

# Available charging headroom at any future timestamp
headroom = bundle.net_available_kw(datetime(2026, 6, 1, 14))

# Best 2-hour charging window for the day (used by smart scheduler)
best_h = bundle.best_charging_window(date(2026, 6, 1), duration_hours=2)
print(f"Best charging window: {best_h}:00 – {best_h+2}:00")
```

The `ForecastBundle` is the bridge between the prediction module and the future MARL engine: each PPO agent will query it as a feature when deciding whether to charge now or wait for a better window.

## MARL Engine

The `marl/` package implements **T3.4** — a multi-agent reinforcement learning engine that replaces the centralised `Scheduler.allocate_charging()` with per-agent decisions. It is activated by the `marl` scenario or by setting `marl.enabled: true` in `config.yaml`. The simulator falls back to the rule-based scheduler transparently on any error.

### Agents

| Agent | Count | Observation space | Action space |
|---|---|---|---|
| `TractorAgent` | one per tractor | SOC, is_charging, has_task, task_urgency, deadline, tariff, net_power, pv_shape, hour | idle / request_charge |
| `ChargingStationAgent` | one per charger | is_occupied, connected_soc, net_power, tariff, pv_shape, hour | off / low (50%) / full |
| `LoadAgent` | one per deferrable consumer | priority, power_kw, net_power, tariff, hour | on / off |

All agents are currently **rule-based**. The architecture is designed so that a learned PPO policy can be dropped in by overriding `act()` on any agent class — `learn()` and `compute_step_rewards()` are already wired into every simulation step to provide the data pipeline.

### Reward

`MARLEnvironment.compute_step_rewards()` returns a scalar per agent at each 15-min step:

```
team_reward = −cost_eur × w_cost  −  grid_excess_kw × w_peak  +  tasks_done × 0.01 × w_task
tractor_reward = team_reward − |soc − 0.6| × w_battery × 0.05
```

Weights are set in `config.yaml` under `marl.reward_weights`.

### Configuration

```yaml
marl:
  enabled: true
  algorithm: rule_based     # rule_based | ppo (planned)
  agents:
    tractors:
      enabled: true
      charge_threshold_soc: 90
    charging_stations:
      enabled: true
    loads:
      enabled: true
      managed_priorities: [low, normal]   # critical/high are never shed
  reward_weights:
    energy_cost: 1.0
    peak_power: 3.0
    task_completion: 10.0
    battery_stress: 1.0
```

### MARL Visualisations

Running `python main.py` (or `python -m farmview marl`) produces three output files for the `marl` scenario.

**Power profile** (`marl_detail.png`) — grid draw, PV generation, and charging events across the day, generated by the MARL engine alongside the standard scenario comparison charts:

[![MARL power profile](../images/marl_detail.png)](../images/marl_detail.png)

**Farm map** (`marl_farm_map.png`) — top-down view of the 800 × 500 m farm at end-of-day. Shows tractor positions and statuses (charging, executing, idle), charger occupancy, task markers by type (spray, harvest, transport, …), and a fleet panel with task completion summary:

[![MARL farm map](../images/marl_farm_map.png)](../images/marl_farm_map.png)

**MARL agent dashboard** (`marl_marl_dashboard.png`) — four panels driven by per-step agent logs:

- **SOC traces** — battery state-of-charge for each tractor through the day
- **Agent heatmap** — per-step action of every tractor, charger, and load agent (colour-coded by decision state)
- **Reward decomposition** — per-step team reward broken down by cost, peak, and task components
- **Grid power** — grid draw vs cap with tariff bands overlaid

[![MARL dashboard](../images/marl_marl_dashboard.png)](../images/marl_marl_dashboard.png)

### Dynamic Plan-Change Events

Setting `dynamic_events_enabled: true` in `config.yaml` injects four mid-day disruptions into the simulation, exercising the system's ability to re-plan autonomously without human intervention:

```yaml
dynamic_events_enabled: true

dynamic_events:
  - at: '2026-06-01 10:00:00'
    type: task_inject
    label: 'Urgent spray injected'
    task: {name: 'Emergency sprayer B', priority: urgent, duration_minutes: 35, uses_pto: true}
  - at: '2026-06-01 13:30:00'
    type: tractor_offline
    label: 'Tractor 2 breakdown'
    tractor_id: tractor_2
  - at: '2026-06-01 16:00:00'
    type: grid_reduce
    label: 'Grid cap reduced to 7 kW'
    new_max_kw: 7.0
  - at: '2026-06-01 18:00:00'
    type: grid_restore
    label: 'Grid cap restored'
```

Event markers (coloured dashed verticals) appear on all dashboard panels and in the fleet log on the farm map. Dynamically injected tasks are highlighted with a purple halo. The dashboard title appends `| plan changes active` when events are enabled.

**Farm map with dynamic events** — the fleet panel on the right lists each fired event with its timestamp. The purple-haloed task marker shows the urgently injected spray run:

[![MARL farm map with dynamic events](../images/marl_farm_map_dynamic_events.png)](../images/marl_farm_map_dynamic_events.png)

**MARL dashboard with dynamic events** — vertical markers on all four panels show exactly when each disruption occurred, making it easy to read the system's response (Grid outage and restore, SOC dip after breakdown, grid draw drop after cap reduction, load restoration after cap restore):

[![MARL dashboard with dynamic events](../images/marl_marl_dashboard_dynamic_events.png)](../images/marl_marl_dashboard_dynamic_events.png)

