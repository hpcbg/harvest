# HARVEST

**Hybrid Agricultural Renewable Via Energy Storage**  
Cloud–Edge–IoT orchestration framework for energy-aware agricultural systems - intelligent coordination of electric tractors, PV generation, batteries and smart farm infrastructure.

> O-CEI 1st Open Call · Challenge P6C1 · High Performance Creators (HPC Bulgaria)

---

## What is HARVEST?

HARVEST is a CEI energy orchestration system for smart agriculture. It enables coordinated management of electric tractor fleets (ZETRABOT), renewable energy (PV), battery storage, and distributed farm IoT loads through a unified, vendor-agnostic platform.

The current repository contains the **pilot6 simulation engine** - a Python-based discrete-time simulator used to develop, validate and demonstrate charging strategies and energy management logic ahead of real hardware integration.

---

## Repository Structure

```
harvest/
├── main.py              # Core simulation engine — scheduler, PV model, KPI computation
├── task_generator.py    # Realistic farm task generation (priority waves, PTO, deadlines)
├── config.yaml          # All simulation parameters — fleet, PV, consumers, scenarios
├── server.py            # Local HTTP server bridging dashboard ↔ simulation
└── dashboard.html       # Self-contained web UI (zero external dependencies)
```

---

## Quick Start

### Option A — Command Line

```bash
# Install dependencies (once)
pip install pyyaml pandas matplotlib

# Run all scenarios
python main.py
```

Results are saved to `./outputs/`:
- `scenario_summary.csv` — KPIs for all scenarios
- `timeseries_<scenario>.csv` — 15-min power/SOC timeseries
- `task_schedule_<scenario>.csv` — per-task lifecycle log
- `*.png` — KPI comparison charts and per-scenario power profiles

The power flow of full_smart scenario is shown in the following figure:

[![Dashboard overview](./images/full_smart_detail.png)](./images/full_smart_detail.png)

### Option B — Web Dashboard (recommended for demos)

```bash
# Start the local server (opens browser automatically)
python server.py
```

Then open **http://localhost:8765** in Firefox, Chrome, or Edge.

The dashboard is fully self-contained - Chart.js is bundled inline, no internet connection required.

---

## Dashboard Usage

1. **Left panel** — adjust simulation parameters with sliders:
   - *Grid & PV*: grid cap (kW), farm PV array size, tractor roof panel wattage
   - *Fleet*: number of tractors, chargers, charger power, battery capacity
   - *Tasks*: task count (5–60), RNG seed (changes the task set)

2. **Scenarios** — click pills to include/exclude from the run. Tags show active features:
   - `PV` — tractor roof panels enabled
   - `shed` — non-critical loads suppressed during grid stress

3. **RUN SIMULATION** — sends parameters to `server.py`, which runs the real Python simulator and returns results.

4. **Results panel**:
   - 5 KPI summary cards (lowest cost, best PV self-use, tasks completed, peak grid, grid efficiency)
   - Scenario comparison table with inline bar charts
   - Energy cost and task completion charts
   - **Task status table** — collapsible, per-scenario view showing each task's phase (PENDING / TRANSIT / EXECUTING / DONE / DELAYED / INTERRUPTED), progress %, tractor assignment, and delay reason

The dashboard overview is shown in the following figure:

[![Dashboard overview](./images/dashboard-overview.png)](./images/dashboard-overview.png)

The task status by scenario is shown in the following figure:

[![Dashboard overview](./images/task-status.png)](./images/task-status.png)

---

## Simulation Model

### Scenarios

| Scenario | Charging strategy | Tractor PV | Load shedding |
|---|---|---|---|
| naive | Immediate full power | ✗ | ✗ |
| night_only | Valle tariff hours only (00–08h) | ✗ | ✗ |
| smart | PV surplus + grid headroom | ✗ | ✗ |
| smart_with_swap | Smart + battery module swaps | ✗ | ✗ |
| pv_roof | Smart + tractor roof panels | ✓ | ✗ |
| pv_roof_swap | Smart+swap + roof panels | ✓ | ✗ |
| pv_roof_shed | Smart + roof + load shedding | ✓ | ✓ |
| full_smart | All optimisations active | ✓ | ✓ |

### Farm Consumers Modelled

| Load | kW | Priority | Schedule |
|---|---|---|---|
| Electric fence | 0.2 | critical | always |
| Irrigation pump ×2 | 3.0 | high | 06–08h / 19–21h |
| Barn door motors | 0.5 | normal | 07–08h |
| Cold storage | 1.2 | high | 08–20h |
| Workshop tools | 2.5 | normal | 08–17h |
| Office HVAC | 1.5 | low | 08–18h |
| Outdoor lighting | 0.8 | normal | 20–23h |
| Security lighting | 0.3 | critical | 22–06h |

### Task Lifecycle

Tasks follow a two-phase model:

```
PENDING → TRANSIT → EXECUTING → DONE
              ↓           
         INTERRUPTED (preempted by urgent task, re-queued)
PENDING → DELAYED (window expired, extended deadline, re-queued)
```

- **TRANSIT**: tractor drives to task location at eco speed (10 km/h). Interruptible by higher-priority urgent tasks.
- **EXECUTING**: PTO engaged, active work. Not interruptible.
- **DELAYED**: original window expired but task remains in queue with a +6h extended deadline.

### Key KPIs

| KPI | Description |
|---|---|
| `total_cost_eur` | Total grid energy cost for the day |
| `pv_self_use_share_pct` | PV used ÷ total demand (note: inflated by low demand — see `grid_kwh_per_completed_task`) |
| `pv_utilisation_pct` | PV used ÷ PV generated (demand-independent solar integration metric) |
| `grid_kwh_per_completed_task` | Normalised energy efficiency per task completed |
| `task_completion_pct` | % of tasks reaching DONE status |
| `tractor_downtime_pct` | % of fleet time spent idle (not working or charging) |
| `peak_grid_kw` | Maximum instantaneous grid draw |
| `cost_per_completed_task_eur` | Total cost divided by tasks completed |

> **Night only** appears cheap because it charges at valle tariff (0.15 €/kWh) but tractors run out of battery by afternoon and complete only ~85% of tasks. Use `grid_kwh_per_completed_task` to compare true efficiency.

---

## Configuration

All parameters are in `config.yaml`. Key sections:

```yaml
simulation:
  start_time: "2026-06-01 00:00:00"
  end_time:   "2026-06-02 00:00:00"
  time_step_minutes: 15

grid:
  max_power_kw: 10.5

pv:
  farm_fixed_peak_kw: 5.0    # building/ground array

tractor_pv:
  panel_peak_w: 650          # per-tractor roof panel

task_generation:
  mode: "generated"          # static | generated
  num_tasks: 20              # scale with fleet: ~6-7 tasks per tractor per day
  seed: 42
```

---

## Project Status (Stage 2 — Development of CEI Utilities)

| Component | Status | Notes |
|---|---|---|
| T3.3 Synthetic Data Pipeline | ✅ Done | `task_generator.py` + simulation engine |
| T3.5 Predictive Scheduler | ✅ Done | Rule-based multi-scenario scheduler |
| Web Dashboard | ✅ Done | `server.py` + `dashboard.html` |
| TPI1 Predictive Scheduling | ✅ Pass | ≥14% cost reduction vs naive baseline |
| TPI2 Autonomous Decisions | ✅ Pass | 100% autonomous across all scenarios |
| TPI3 AI Prediction Module | 🔄 Partial | Rule-based; ML forecasting in progress |
| T3.1 FIWARE NGSI-LD Layer | ⬜ Pending | Digital twin adapter planned |
| T3.2 ROS2 Agro-Robotics | ⬜ Pending | ZETRABOT interface planned |
| T3.4 MARL Engine | ⬜ Pending | PPO agents to replace rule-based scheduler |
| T3.6 Edge Autonomy | ⬜ Pending | BLE Mesh sensors + Jetson Orin deployment |

**D2 Prototype deadline: 30 June 2026**

---

## Architecture (Target)

```
                    ┌─────────────────────────────────────┐
                    │         FIWARE NGSI-LD Broker        │  ← T3.1
                    │   Digital twins for all farm assets  │
                    └──────────────┬──────────────────────┘
                                   │ NGSI-LD
          ┌────────────────────────┼────────────────────┐
          │                        │                    │
   ┌──────▼──────┐        ┌────────▼────────┐   ┌──────▼──────┐
   │  ROS2 / FIROS2│        │  MARL Engine    │   │  BLE Mesh   │
   │  ZETRABOT    │        │  PPO agents     │   │  IoT sensors│
   │  interface   │        │  (edge, INT8)   │   │  (PV-powered│
   └─────────────┘        └─────────────────┘   └─────────────┘
        T3.2                    T3.4                  T3.6
          ↑
   ┌──────┴──────────────────────────────────┐
   │     pilot6 Simulation Engine (current)  │  ← THIS REPO
   │  main.py · task_generator.py · config   │
   └─────────────────────────────────────────┘
```

---

## Dependencies

```
Python ≥ 3.10
pyyaml
pandas
matplotlib
```

No cloud services, no API keys, no external network access required.

---

## Contact

**Simeon Tsvetanov** · set@hpc.bg  
High Performance Creators Ltd · Sofia, Bulgaria · [hpc.bg](https://hpc.bg)  
O-CEI Challenge P6C1 · Application ID: 691486e3b5fba953e852532f
