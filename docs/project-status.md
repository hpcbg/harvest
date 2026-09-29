# Project status and target architecture

## Project Status

### Stage 2 — Development of CEI Utilities

| TPI | Description | Target | Result | Status |
|---|---|---|---|---|
| TPI1 | Predictive scheduling module (local demo) | ≥ 10% peak / efficiency gain vs baseline | ≈ 40% peak reduction, ≈ 42% cost reduction | **ACHIEVED** |
| TPI2 | Autonomous decision execution | ≥ 70% decisions without manual intervention | 100% autonomous in all test scenarios | **ACHIEVED** |
| TPI3 | AI prediction module (energy demand) | ≤ 25% MAPE | Farm-load MAPE 22.98% | **ACHIEVED** |

### Stage 3 — Pilot Integration (target: 30 September 2026)

| TPI | Description | Target | Evidence |
|---|---|---|---|
| TPI4 | Pilot Integration | Fully operational pilot setup with ≥ 50% of system data exchanged using real data streams | System logs, data exchange records, interface monitoring outputs, validation report |
| TPI5 | Energy Optimisation | ≥ 10% reduction in energy consumption vs baseline under real pilot conditions | Energy measurements, before/after comparison, analysis report |
| TPI6 | O-CEI Marketplace Contribution | HARVEST components published (FIWARE data models, ROS2 interfaces, documentation/demo) | Uploaded assets, documentation, repository links |

**D2 Prototype deadline: 30 June 2026**

## Architecture (Target)

> **Status:** the FIWARE NGSI-LD broker layer (T3.1) and a containerised ROS 2
> interface (T3.2 northbound) are now implemented — see
> [integrations.md](integrations.md).
> The diagram below remains the Stage 3 target picture (FIROS2/ZETRABOT
> hardware topics, PPO agents, BLE mesh).

```
                    +-------------------------------------+
                    |         FIWARE NGSI-LD Broker        |  <- T3.1
                    |   Digital twins for all farm assets  |
                    +------------------+------------------+
                                       | NGSI-LD
          +----------------------------+--------------------+
          |                            |                    |
   +------+------+          +----------+--------+   +------+------+
   |  ROS2/FIROS2|          |  MARL Engine      |   |  BLE Mesh   |
   |  ZETRABOT   |          |  PPO agents       |   |  IoT sensors|
   |  interface  |          |  (edge, INT8)     |   |  (PV-powered|
   +------+------+          +----------+--------+   +-------------+
        T3.2                           | T3.4              T3.6
          |                   +--------+---------+
          |                   |  predictor/      |
          |                   |  PV + load       |
          |                   |  ForecastBundle  |
          |                   +--------+---------+
          |                            |
          +----------+   +------------+
                     |   |
              +------+---+-------+
              |  FleetInterface  |  <- harvest_control/ (THIS RELEASE)
              |  snapshot()      |     decision layer to plant boundary
              |  submit(cmds)    |     same API -> sim today, ZETRABOT Stage 3
              +--------+---------+
                       |
   +---------------------------------------+---------------------+
   |           pilot6 Simulation Engine (current)               |
   |   main.py  task_generator.py  config.yaml  dashboard.html  |
   +------------------------------------------------------------+
```

