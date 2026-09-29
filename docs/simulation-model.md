# Simulation model

### Scenarios

| Scenario | Charging strategy | Tractor PV | Load shedding | MARL |
|---|---|---|---|---|
| naive | Immediate full power | ✗ | ✗ | ✗ |
| night_only | Valle tariff hours only (00–08h) | ✗ | ✗ | ✗ |
| smart | PV surplus + grid headroom | ✗ | ✗ | ✗ |
| smart_with_swap | Smart + battery module swaps | ✗ | ✗ | ✗ |
| pv_roof | Smart + tractor roof panels | ✓ | ✗ | ✗ |
| pv_roof_swap | Smart+swap + roof panels | ✓ | ✗ | ✗ |
| pv_roof_shed | Smart + roof + load shedding | ✓ | ✓ | ✗ |
| full_smart | All optimisations active | ✓ | ✓ | ✗ |
| marl | Per-agent decisions (MARL engine) | ✓ | ✓ | ✓ |

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
              |
         INTERRUPTED (preempted by urgent task, re-queued)
PENDING → DELAYED   (window expired, extended deadline, re-queued)
```

- **TRANSIT**: tractor drives to task location at eco speed (10 km/h). Interruptible by higher-priority urgent tasks.
- **EXECUTING**: PTO engaged, active work. Not interruptible.
- **DELAYED**: original window expired but task remains in queue with a +6h extended deadline.

### Key KPIs

| KPI | Description |
|---|---|
| `total_cost_eur` | Total grid energy cost for the day |
| `pv_self_use_share_pct` | PV used ÷ total demand (note: can be inflated by low demand) |
| `pv_utilisation_pct` | PV used ÷ PV generated (demand-independent solar integration metric) |
| `grid_kwh_per_completed_task` | Normalised energy efficiency per task completed |
| `task_completion_pct` | % of tasks reaching DONE status |
| `tractor_downtime_pct` | % of fleet time spent idle (not working or charging) |
| `peak_grid_kw` | Maximum instantaneous grid draw |
| `cost_per_completed_task_eur` | Total cost divided by tasks completed |

> **Night only** appears cheap because it charges at valle tariff (0.15 €/kWh) but tractors run out of battery by afternoon and complete only ~85% of tasks. Use `grid_kwh_per_completed_task` to compare true efficiency across scenarios.

