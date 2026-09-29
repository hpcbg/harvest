# HARVEST

**Hybrid Agricultural Renewable Via Energy Storage**

> O-CEI 1st Open Call · Challenge P6C1 · High Performance Creators (HPC Bulgaria)

## What HARVEST is

HARVEST is a cooperative energy-management and decision-support framework for
electrified agriculture. It schedules agricultural tasks and renewable-aware
charging for a fleet of electric tractors (ZETRABOT), coordinates PV, grid and
farm loads, talks to heterogeneous field devices, publishes its state through
semantic interoperability standards, drives a physical simulation, and
validates its energy model against real agricultural telemetry.

[![Dashboard overview](./images/dashboard-overview.png)](./images/dashboard-overview.png)

## Headline features

- **Cooperative energy optimisation** — renewable-aware, tariff-aware smart charging and load shedding, compared across scenarios (naive → full smart → MARL).
- **Agricultural task scheduling** — PTO work, deadlines, priorities and transit, assigned by HARVEST's own scheduler.
- **Multi-agent / device architecture** — tractor, charger and load agents (MARL engine) behind one `FleetInterface`.
- **Modbus / OPC-UA DeviceIO abstraction** — one protocol seam; a farm-device simulator speaks real Modbus TCP and OPC-UA.
- **FIWARE / NGSI-LD interoperability** — fleet, tasks, simulation and telemetry mirrored to Orion-LD; inbound commands via NGSI-LD.
- **ROS 2 integration** — containerised fleet bridge on `/harvest/*` topics.
- **NVIDIA Isaac Sim digital twin** — tractors as PhysX vehicles that physically drive HARVEST's schedule (plus a GPU-free stand-in).
- **Real ZETRABOT telemetry replay** — a recorded mission replayed on its original timestamps through the canonical telemetry model.
- **KPI / model validation** — mission KPIs with explicit provenance, real-vs-model comparison, rule-based findings, JSON/CSV export.
- **AWS-ready `TelemetrySource` architecture** — the CSV replay is one implementation of the interface a live AWS source will use.
- **Docker-based reproducible deployment** — one command per stack, end-to-end validation script.
- **Interactive Operations, ROI and Diagnostics views** — simulation, long-term investment economics and live integration health.

## Quick start

Requires Docker + Compose (host-Python-only setup: [docs/getting-started.md](docs/getting-started.md)).

```bash
./run_harvest_dashboard.sh              # dashboard + Modbus/OPC-UA device simulator + FIWARE (default)
./run_harvest_dashboard.sh full         # the same plus the ROS 2 bridge
./run_harvest_dashboard.sh isaac        # plus the Isaac Sim bridge, starting real Isaac Sim on the host
./run_harvest_dashboard.sh isaac-demo   # the full HARVEST → ROS 2 → simulator loop with a GPU-free stand-in
./run_harvest_dashboard.sh replay --file telemetry/telemetria_mision_63.csv --speed 20
                                        # replay the real ZETRABOT mission (add --stack fiware for NGSI-LD)
./validate_harvest_stack.sh             # end-to-end check of whatever is running (exit code = failures)
```

Open **http://localhost:8765** — tabs **Operations**, **ROI & Investment**, **Diagnostics**.
`./run_harvest_dashboard.sh stop` tears everything down.

## Architecture

```
                 Dashboard: Operations · ROI · Diagnostics
                                   │  HTTP API (server.py)
  ┌────────────────────────────────┴─────────────────────────────────┐
  │  HARVEST core: task scheduling · renewable-aware charging ·      │
  │  MARL agents · PV/load prediction · ROI                          │
  │  FleetInterface (harvest_control) — the one seam to every fleet  │
  └────┬──────────────────┬──────────────────┬───────────────────┬───┘
       │ commands/state   │ inbound only     │ fleet API         │ fleet API
   DeviceIO           TelemetrySource     ROS 2 bridges       FIWARE sync
   Modbus / OPC-UA    CSV replay          (fleet, Isaac)      Orion-LD
   (farm simulator    AWS (scaffold)           │              NGSI-LD entities
    or real devices)       │               Isaac Sim
                    KPIs & model          digital twin
                    validation            (proxy vehicles)
```

Telemetry ingestion and control never mix: data comes in through
`TelemetrySource`, commands go out through `DeviceIO`.

## Real telemetry and KPIs

A ZETRABOT mission export (Zetrack V2 CSV: asynchronous messages, 14 message
types) is replayed through the same canonical telemetry model intended for
future live AWS ingestion. The replayed tractor joins the fleet snapshot,
Diagnostics and the FIWARE mirror.

The Diagnostics tab opens with a **mission KPI & model-validation panel**:
energy used, SOC change, mean/peak power, distance, moving and PTO time,
energy per km and per hour, consumption while PTO-active / moving / idle,
thermal maxima. Every value is labelled **MEASURED**, **DERIVED**,
**CONFIGURED**, **ESTIMATED** or **MISSING**. Next to it: HARVEST's energy model
applied to the measured operating profile, rule-based findings, calibration
proposals for review (never written to `config.yaml`), and a traceable
JSON/CSV export for publications.

For the supplied mission 63: 19.93 kWh (V×I), SOC 64 → 18 %, 3.70 km, 2.50 h
PTO-active. The committed model parameters over-predict the energy by 37 %,
mainly through `pto_power_kw`. The configured 44.8 kWh capacity is broadly
consistent with the 43.3 kWh effective-capacity estimate.
Details: [docs/telemetry.md](docs/telemetry.md).

```bash
python -m harvest_integrations.telemetry.kpi telemetry/telemetria_mision_63.csv --json kpis.json --csv-out kpis.csv
```

## Limitations

- **One mission** is available: it is not enough for robust model calibration, and generalisation is not established.
- The supplied CSV has **no GPS coordinates**, **no explicit charging-state stream** and **no charging cycle**.
- **AWS live connectivity is not implemented**, because the actual AWS service and interface have not been provided yet. Only a scaffold exists behind `TelemetrySource`.
- HARVEST does **not control the real ZETRABOT** in this deployment. Modbus/OPC-UA are optional DeviceIO capabilities, exercised against simulators.
- **Isaac Sim** uses a functional tractor/mobile-robot proxy, not an exact ZETRABOT model.
- **Effective-capacity estimates** and the `DischEnrgActualSesion` per-session interpretation both need confirmation of the signal semantics and the official battery specification.

## Documentation

| Topic | |
|---|---|
| Installation, CLI, dashboard, configuration, dependencies | [docs/getting-started.md](docs/getting-started.md) |
| Integrations: launcher modes, fleet API, DeviceIO, FIWARE, ROS 2, Isaac Sim, Diagnostics | [docs/integrations.md](docs/integrations.md) |
| Real ZETRABOT telemetry: replay, signal dictionary, KPIs, model validation, AWS | [docs/telemetry.md](docs/telemetry.md) · [AWS scaffold](harvest_integrations/aws/README.md) |
| Isaac Sim layer | [harvest_integrations/simulators/isaac/README.md](harvest_integrations/simulators/isaac/README.md) |
| Simulation model (scenarios, consumers, task lifecycle, KPIs) | [docs/simulation-model.md](docs/simulation-model.md) |
| Prediction module and MARL engine | [docs/prediction-and-marl.md](docs/prediction-and-marl.md) |
| ROI & investment analysis | [docs/roi.md](docs/roi.md) |
| Project status (TPIs) and target architecture | [docs/project-status.md](docs/project-status.md) |
| Repository structure | [docs/repository-structure.md](docs/repository-structure.md) |

Tests: `python -m unittest discover tests` (stdlib `unittest`; protocol, ROS and real-mission tests skip themselves when their prerequisites are absent).

## Contact

**Simeon Tsvetanov** · set@hpc.bg
High Performance Creators Ltd · Sofia, Bulgaria · [hpc.bg](https://hpc.bg)
O-CEI Challenge P6C1 · Application ID: 691486e3b5fba953e852532f
