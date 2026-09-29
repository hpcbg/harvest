# Repository structure

## Repository Structure

```
harvest/
├── main.py                   # Core simulation engine — scheduler, PV model, KPI computation
├── task_generator.py         # Realistic farm task generation (priority waves, PTO, deadlines)
├── config.yaml               # All simulation parameters — fleet, PV, consumers, scenarios
├── server.py                 # Local HTTP server bridging dashboard <-> simulation
├── dashboard.html            # Self-contained web UI (zero external dependencies)
├── requirements.txt          # Python dependencies
├── config.local.yaml         # Local overrides — gitignored, never committed (optional)
├── generate_prediction_overview.py  # Regenerates images/prediction_overview.png
├── predictor/                # Prediction module (TPI3)
│   ├── __init__.py           #   build_predictors() factory + public API
│   ├── base.py               #   Abstract base classes (BasePVPredictor, BaseLoadPredictor)
│   ├── static.py             #   Static profile wrapper — backward-compatible default
│   ├── synthetic.py          #   Synthetic training data generator
│   ├── nn_predictor.py       #   Neural network train + inference (Keras/TensorFlow)
│   └── weather.py            #   Open-Meteo live weather + offline seasonal stub
├── marl/                     # Multi-agent RL engine (T3.4)
│   ├── __init__.py           #   build_marl_engine() factory
│   ├── base.py               #   BaseAgent + observation dataclasses
│   ├── agents.py             #   TractorAgent, ChargingStationAgent, LoadAgent
│   └── environment.py        #   MARLEnvironment — replaces Scheduler.allocate_charging()
├── harvest_control/          # Fleet control interface (TPI2 / T3.2 bridge)
│   ├── __init__.py           #   Public re-exports
│   ├── interface.py          #   FleetInterface contract + state/command dataclasses
│   ├── sim_backend.py        #   SimulationFleetInterface + Adapters hook for pilot6
│   ├── demo_autonomous_control.py  # Runnable TPI2 demo — writes execution_log.csv
│   └── ros2_bridge.py        #   Skeleton ROS 2 node (Stage 3, guarded imports)
├── farmview/                 # Visualisation package
│   ├── __init__.py
│   ├── _colors.py
│   ├── _renderer.py          #   render_farm() — top-down farm map
│   ├── _marl.py              #   render_marl_log() — MARL agent dashboard
│   └── __main__.py           #   python -m farmview CLI
├── roi/                      # ROI & Investment analysis (long-term economics)
│   ├── __init__.py           #   run_roi_analysis() public API
│   ├── models.py             #   Typed assumptions / totals / cash-flow / result structures
│   ├── calculator.py         #   NPV, IRR, simple & discounted payback, ROI, escalation
│   ├── period_runner.py      #   Aggregates the simulator over exact/representative periods
│   ├── investments.py        #   Electric-vs-diesel, farm PV, roof PV, chargers, portfolio
│   ├── reliability.py        #   Outage expected-value model + avoided outage cost
│   ├── engine.py             #   Orchestrator: variants → investments → sensitivity
│   ├── validation.py         #   Central input validation + advisory warnings
│   ├── export.py             #   roi_summary.csv / roi_cashflows.csv / *.json writers
│   └── __main__.py           #   python -m roi CLI
├── harvest_integrations/     # External-integration layer (optional; see below)
│   ├── codec.py              #   Canonical fleet JSON schema (HTTP / FIWARE / ROS share it)
│   ├── runtime.py            #   Live fleet runtime behind /api/fleet/* (sim | devices)
│   ├── devices/              #   Protocol abstraction: DeviceIO seam + registry
│   │   ├── base.py           #     PointSpec / DeviceEndpoint / register_protocol()
│   │   ├── modbus_io.py      #     Modbus TCP backend (pymodbus)
│   │   ├── opcua_io.py       #     OPC-UA backend (asyncua.sync)
│   │   ├── fake_io.py        #     In-memory backend for tests
│   │   └── fleet_backend.py  #     DeviceFleetInterface (FleetInterface over field devices)
│   ├── simulators/           #   Farm-device simulator speaking real Modbus + OPC-UA
│   │   └── isaac/            #     Optional Isaac Sim layer: contract, motion core,
│   │                         #     GPU-free stub, robot-model registry, world,
│   │                         #     WebRTC, Isaac standalone app (see its README)
│   ├── fiware/               #   NGSI-LD client, entity mapping, sync daemon
│   ├── telemetry/            #   Real ZETRABOT telemetry (inbound only, see below)
│   │   ├── model.py          #     TelemetryMessage, ZetrabotTelemetry, TelemetrySource seam
│   │   ├── zetrack.py        #     Zetrack V2 CSV schema + signal dictionary + CsvTelemetrySource
│   │   ├── normalizer.py     #     raw messages -> canonical state (unknowns preserved)
│   │   ├── replay.py         #     original-timeline replay (speed, gap compression)
│   │   ├── service.py        #     wiring into FleetRuntime / Diagnostics / FIWARE
│   │   ├── analysis.py       #     mission analysis: regimes, energy, gaps, data quality
│   │   └── kpi.py            #     KPIs + provenance, real vs model, findings, JSON/CSV export
│   └── aws/                  #   AWS telemetry-source scaffold + README (no SDK by default)
├── telemetry/                # Supplied ZETRABOT mission export + Zetrack report (git-ignored)
├── ros2_ws/                  # ROS 2 workspace (built & run inside Docker)
│   └── src/harvest_ros/      #   fleet_bridge + isaac_bridge nodes + topics/qos contract
├── docker/                   # Dockerfile (app) + Dockerfile.ros2 (bridge)
├── docker-compose.yml        # Full stack; profiles: devices / fiware / ros2 / isaac[-demo]
├── run_harvest_dashboard.sh  # One-command startup (lite|core|devices|fiware|full|isaac…)
├── validate_harvest_stack.sh # End-to-end validation of the running stack
├── examples/                 # HTTP / NGSI-LD / DeviceIO / Isaac client examples
├── scripts/                  #   validate_stack.py, run_isaac_sim.sh (host Isaac launcher)
├── requirements-integrations.txt  # Optional deps: pymodbus (pinned), asyncua
├── docs/                     # Detailed documentation (integrations, telemetry, ROI, …)
└── tests/                    # Unit + integration tests (stdlib unittest)
    ├── test_calculator.py    #   NPV / IRR / payback / ROI primitives
    ├── test_investments.py   #   Diesel litres, PV paired-sim, degradation, double-count
    ├── test_reliability.py   #   Islanding rule, expected-value outage cost
    ├── test_period_runner.py #   Period resolution, seasonality, determinism
    ├── test_integration.py   #   Existing sim preserved + ROI engine end-to-end
    ├── test_device_io.py     #   DeviceIO seam, point maps, protocol registry
    ├── test_fleet_backend.py #   DeviceFleetInterface mapping + command acks
    ├── test_fleet_codec.py   #   Fleet JSON schema round-trips
    ├── test_fleet_api.py     #   /api/fleet/* endpoints over live HTTP
    ├── test_fiware_entities.py #  NGSI-LD mapping + sync engine (stub broker)
    ├── test_ros_contract.py  #   ROS topic contract (no ROS required)
    ├── test_isaac_contract.py #  Sim wire contract + shared motion core (no Isaac required)
    ├── test_isaac_robots.py   #  Robot registry, steering, WebRTC config, Isaac diagnostics
    ├── test_live_tasks.py     #  Live task layer: HARVEST schedules, the simulator reports
    ├── test_telemetry.py      #  Real telemetry: parsing, normalisation, replay, FIWARE
    ├── test_telemetry_kpi.py  #  Mission KPIs, provenance, real vs model, export, API
    └── test_protocol_integration.py # Live Modbus/OPC-UA loopback (auto-skips)
```

