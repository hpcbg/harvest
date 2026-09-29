# Real ZETRABOT telemetry

HARVEST ingests **real ZETRABOT telemetry** through a generic, inbound-only
layer (`harvest_integrations/telemetry`) that knows nothing about AWS, CSV
files or FIWARE.  Today the source is the supplied mission export
(`telemetry/telemetria_mision_63.csv`, git-ignored, alongside the Zetrack V2
PDF report); tomorrow it is ZETRABOT's AWS feed — swapping one for the other
changes nothing above the `TelemetrySource` seam.

```
 CsvTelemetrySource ---\
                        \
 AwsTelemetrySource -----> TelemetryNormalizer --> ZetrabotTelemetry (canonical, per tractor)
 (scaffold, aws/)                                        |
                                          +--------------+---------------+
                                          |                              |
                                   HARVEST state                 FIWARE / NGSI-LD
                          (FleetRuntime merges the tractor      (TractorTelemetry entity,
                           into every fleet snapshot,            mirrored by fiware-sync)
                           Diagnostics "Real telemetry")
                                          |
                                 replay / calibration

 HARVEST control -> FleetInterface -> DeviceIO -> Modbus / OPC-UA -> ZETRABOT   (unchanged, separate)
```

Telemetry ingestion and robot control never mix: data comes in through
`telemetry`/`aws`, commands go out through `devices`.

## Replaying the supplied mission

```bash
# host only, no Docker (lite mode) — 20 telemetry seconds per real second
./run_harvest_dashboard.sh replay --file telemetry/telemetria_mision_63.csv --speed 20

# the same, plus the Docker stack so the state is mirrored to Orion-LD as NGSI-LD
./run_harvest_dashboard.sh replay --file telemetry/telemetria_mision_63.csv --speed 20 --stack fiware

# options: --max-gap 30 (cap, in REAL seconds, on idle/overnight pauses; the
#          mission has a 2.2 h lunch break and two overnights), --loop,
#          --start/--end <ISO-8601 UTC> to replay a window, --stack <any mode>
# equivalently, for any mode: HARVEST_TELEMETRY_SOURCE=csv HARVEST_TELEMETRY_FILE=<csv> \
#                             HARVEST_TELEMETRY_SPEED=20 ./run_harvest_dashboard.sh fiware
```

Then open the dashboard's **Diagnostics** tab: the **mission KPIs & model
validation** panel (below), the **Real telemetry** service card (mission,
tractor, replay state, source type, latest telemetry timestamp, SOC,
voltage/current, derived power, cumulative discharged energy, temperatures,
PTO/drive state) and the **low-level signals** panel with the full canonical
state, each value marked with its provenance where it is not a plain
measurement; the replayed tractor also appears as a
`csv-replay` row in the Devices table and as `zetrabot_1` in
`GET /api/fleet/snapshot` (map it onto an existing fleet id with
`integrations.telemetry.tractor_ids`).  `python examples/telemetry_replay_demo.py`
watches all of this from the command line, and `./validate_harvest_stack.sh`
checks the whole path, including the KPI endpoint and CSV export (skipped
when no source is configured).

| Endpoint | Description |
|---|---|
| `GET /api/telemetry` | Source description, replay progress, canonical state per real tractor (`state` = `inactive` when no source is configured, never an error) |
| `GET /api/telemetry/kpis` | Mission KPIs with provenance, real-vs-model comparison, findings, proposals, limitations (`.json` / `.csv` = downloads) |
| `GET /api/telemetry/analysis` | The raw calibration analysis over the source's whole history |

The replay preserves timestamp ordering (the export is not sorted; the order
key is timestamp → the robot's `sequence` → row id, because `sequence`
restarts on every power cycle), keeps the asynchronous sampling of the
message types as it was (SOC every ~2 min, battery current several times a
second — every canonical field carries its own `observed_at`), identifies its
records as `csv-replay`, and **fabricates nothing**: a field is `null` until
its signal has been seen, position is always `null` (there is no GPS in the
export — the Zetrack report's route map comes from a separate, undocumented
stream), and a finished replay marks the tractor *unavailable* rather than
pretending it is still reporting.

## What the export contains (documented in `harvest_integrations/telemetry/zetrack.py`)

One CSV row per **asynchronous message**: `id, user_id, tractor_id,
mission_id, timestamp, sequence, schema_version, source_message, signals,
created_at`, where `signals` is a JSON object whose keys depend on
`source_message` (often a subset, frequently `{}`).  Mission 63 is 56 428
rows over three working days (2026-05-12/13/14), one tractor, schema 1.0,
14 message types.

| Canonical field(s) | Wire signal(s) (`source_message.signal`) | Unit |
|---|---|---|
| `soc_pct`, `battery_voltage_v`, `battery_temp_c` | `BatteryStatus1.MainBatterySOC / MainBatteryVoltage / MainBatteryTemp` | %, V, °C |
| `battery_current_a`, `discharged_energy_session_kwh` | `BatteryStatus3.BatteryCurrent / DischEnrgActualSesion` (session counter, resets per power cycle) | A, kWh |
| `aux_soc_pct`, `aux_battery_voltage_v` | `BatteryStatus2.AuxBatterySOC / AuxBatteryVoltage` | %, V |
| `speed_kmh`, `oil_temp_c`, `controller_motor_temp_c`, `lifetime_km`, `lifetime_hours` | `MiscInfo.SpeedDisplay / OilTemp / MotorTemp / LifetimeKm / LifetimeHoursActive` | km/h, °C, km, h |
| `wheel_speed_rpm{T1..T4}` | `PmsMotorSpeed.T1SpeedAbs..T4SpeedAbs` | rpm |
| `motor_current_a{T1,T3,T4}` | `PmsMotorCurrent.T1Current / T3Current / T4Current` (T2 never reported) | A |
| `motor_temp_c{T1..T4}`, `pto_temp_c`, `implement_temp_c{BHD,BHG}` | `MotorTemp.TempT1..TempT4 / TempPTO / TempBHD / TempBHG` | °C |
| `pto_speed_rpm`, `implement_speed_rpm{BHD,BHG}` | `IpmMotorSpeed.PTOSpeed / BHDSpeed / BHGSpeed` | rpm |
| `pto_current_a`, `implement_current_a{BHD,BHG}` | `IpmMotorCurrent.PTOCurrent / BHDCurrent / BHGCurrent` | A |
| `pto_active`, `drive_active`, `parked`, `lights_on`, `shovel_active`, `drive_mode`, `move_mode`, `function_leds{*}` | `FunctionStatus.PtoLed / GoLed / PLed / LightsLed / PalaLed / DriveModeLed / MoveModeLed` (+ every other `*Led` raw) | bool / enum |
| `steering_angle_deg`, `brake_pedal_pct` | `SensorsAnalogValues3.SteerLeftAngle / BrakePedal` | °, % |
| `hydraulic_pressure_bar` | `SensorsAnalogValues1.AccumPressure` | bar |
| **derived** `battery_power_kw` | voltage × current, only when both were observed within 120 s of each other | kW |
| **derived** `discharged_energy_kwh` | the session counter accumulated across its resets (a reset = a drop below half of the running session maximum) | kWh |
| raw only | `LimitsStatus` (always `{}`), `MemoryData2.CursorPositionOffset` | — |

Anything else on the wire is kept verbatim in `signals_raw` and flagged in
`unknown_signals` / `unknown_messages` (and mirrored to the broker as
`unknownSignals`), so a new signal is never silently lost.  **Not in the
export**: GPS position, charger/charging state, ambient conditions.

## Mission KPIs and model validation

The Diagnostics tab opens with a **Real ZETRABOT mission — KPIs & model
validation** panel (`harvest_integrations/telemetry/kpi.py`, served by
`GET /api/telemetry/kpis`).  It answers the operational questions directly —
energy used, SOC change, mean / peak power, distance, moving and PTO time,
energy per km and per hour, energy while PTO-active / moving / idle, thermal
maxima — and how closely the mission agrees with HARVEST's energy model.  The
low-level signal panel stays underneath it.

When no replay is running but the configured export
(`integrations.telemetry.csv.file`) exists, the panel analyses that file
*offline* through the same `TelemetrySource.history()` call and says so; with
a replay or (future) live source it analyses that source's history.

### Provenance

Every KPI carries exactly one class, shown as a badge and exported with the
value, its formula, the wire signals it came from and any quality flags:

| Class | Meaning | Examples |
|---|---|---|
| `MEASURED` | present in the telemetry | initial/final SOC, peak current, temperature maxima |
| `DERIVED` | calculated from measured values only | V×I energy, ΔSOC, mean power, energy/km, moving & PTO time, LifetimeKm distance |
| `CONFIGURED` | taken from `config.yaml` | battery capacity (44.8 kWh) |
| `ESTIMATED` | inferred under a stated assumption, incl. HARVEST model predictions | effective capacity, estimated remaining energy (SOC × *configured* capacity), session-counter energy, model energy |
| `MISSING` | the source cannot supply it | GPS / location, charging state |

A value whose inputs are absent is `MISSING` with the reason — never `0`.

### Real vs HARVEST model

HARVEST's own energy model (`tractors.model`, the parameters `main.py`'s
scheduler and simulator use) is applied to the operating profile that was
*measured*:

```
E_model = pto_power_kw x PTO-active h + driving_kwh_per_km x distance km
        + idle_kwh_per_h x (powered, stationary, PTO-off) h
```

Powered intervals are split into exclusive regimes (PTO on & moving, PTO on &
stationary, moving with PTO off, idle, and *unclassified* while the PTO/speed
state has not been observed yet); the LifetimeKm distance is allocated
between regimes in proportion to the SpeedDisplay integral.  Sign
convention: `difference = model − real`, `error % = (model − real) / |real|`.

Mission 63 (config.yaml as committed):

| Metric | Real | HARVEST model | Error |
|---|---|---|---|
| Total energy | 19.93 kWh (V×I) | 27.37 kWh | +37 % |
| SOC decrease | 46 pts | 61.1 pts | +15.1 pts |
| PTO-active consumption | 17.87 kWh | 26.38 kWh | +48 % |
| Driving with PTO off | 1.44 kWh (0.36 h — low confidence) | 0.50 kWh | −65 % |
| Idle (stationary, PTO off) | 0.62 kWh over 1.63 h | 0.49 kWh | −21 % |
| SOC change while powered off (42.9 h) | ≤ 5 pts (upper bound) | 28.7 pts | — |
| Battery capacity | 43.3 kWh (estimate; 42.4–44.3 for ±1 SOC pt) | 44.8 kWh configured | +3.4 % |
| Peak power, charging behaviour | — | — | unavailable: not predicted by the model / no data |

Measured energy with the configured capacity reproduces the SOC decrease to
−1.5 pts, so the gap is in the consumption parameters, not the capacity.

### Findings, proposals, limitations

* **Findings** are fixed rules over those numbers (agreement bands ±10 % /
  ±25 %), e.g. *"HARVEST model predicts 27.37 kWh … 37.4 % higher than the
  measured 19.93 kWh"*, *"battery-capacity assumption broadly consistent"*,
  *"GPS stream unavailable"*, *"only one mission available"*.
* **Calibration proposals** (`pto_power_kw` 10 → 7.14 kW as an upper bound,
  `idle_kwh_per_h`, `driving_kwh_per_km` at very low confidence …) are shown
  for review with their basis and confidence.  **Nothing writes
  `config.yaml`**; applying them would be an in-sample fit to the one mission.
* **Limitations** travel with every export: one mission only; no GPS; no
  charging-state stream and no charging cycle; SOC in whole percent; AWS live
  ingestion not implemented; no direct control of the real ZETRABOT;
  Modbus/OPC-UA optional; Isaac Sim uses a proxy vehicle; capacity estimates
  and the `DischEnrgActualSesion` interpretation unconfirmed.

### `DischEnrgActualSesion` — handled with care

The counter behaves like a per-power-on *session* total (8 sessions whose
maxima sum to 20.06 kWh; the Zetrack report's 6.43 kWh is the largest
single session).  That interpretation is **not confirmed** by the ZETRABOT
team, so the headline energy is the independent V×I integral (19.93 kWh,
`DERIVED`); the counter sum is shown as an `ESTIMATED` cross-check
(0.7 % apart) with the raw per-session maxima alongside.

### Data-quality rules

* sample-and-hold between asynchronous messages; intervals > 120 s
  (power-offs, lunch, overnights) contribute nothing;
* exact 0 V voltage and 0 °C temperature readings are dropouts (they arrive
  together in `BatteryStatus1`) — rejected and counted, never integrated;
* one LifetimeKm glitch (a 108 km reading among ~170 km) is discarded.

### Export (for the paper)

```bash
python -m harvest_integrations.telemetry.kpi telemetry/telemetria_mision_63.csv \
       --json mission63_kpis.json --csv-out mission63_kpis.csv   # add --wide for one row per mission
```

or from the running dashboard: **Export JSON / Export CSV / CSV (wide)**
(`GET /api/telemetry/kpis.json`, `/api/telemetry/kpis.csv[?format=wide]`).
The JSON holds the flat `summary` (`mission_id, tractor_id, distance_km,
duration_h, moving_h, pto_active_h, soc_initial_pct, soc_final_pct,
soc_delta_pct, energy_kwh, mean_power_kw, peak_power_kw, energy_per_km,
energy_per_moving_h, configured_capacity_kwh, estimated_capacity_kwh,
model_energy_kwh, energy_error_pct, …`), `summary_provenance`,
`summary_flags`, every KPI with formula and sources, the comparison table,
findings, proposals and limitations.  Both formats carry the input file's
SHA-256 and the `analysis_version`, so every number traces back to the raw
telemetry and the code that computed it.  No publication claim is generated.

### Low-level calibration dump

`python -m harvest_integrations.telemetry.analysis <csv> [--json out.json]`
prints the underlying `MissionAnalysis` result (all regimes, percentiles,
power-off gaps, per-motor temperatures, quality counters); `GET
/api/telemetry/analysis` serves the same document.

## How the real-data replay differs from the synthetic farm simulator

| | Synthetic simulator (`sim` / `devices` + farm-sim, Isaac) | Real-data replay (`csv-replay`) |
|---|---|---|
| Clock | HARVEST's farm day, 1 sim-minute per second | the robot's own timestamps (May 2026), accelerated by `--speed`, idle gaps compressed |
| Sampling | one coherent snapshot per tick | asynchronous messages, each field with its own `observed_at`; some fields stale, some never seen |
| Fields | SOC, energy, charging, position, task | SOC, V/I/power, energies, temperatures, PTO/drive/steering/brake/hydraulics — **no position, no charging state** |
| Control | commands actuate it (`request_charge`, `assign_task` …) | read-only; commands go to the fleet backend, never to the recording |
| Availability | always present | present while the replay delivers; *unavailable* once the recording ends |
| Scheduling | HARVEST's `main.Scheduler` schedules it | untouched: the replayed tractor is not in `tractors.fleet`, so the task layer ignores it unless mapped onto a fleet id |
| Diagnostics state | `simulated` | `healthy` (real data; the detail says *replay* and the source type) — never `simulated` |

## AWS: what is prepared, and what is still required

`harvest_integrations/aws` holds the adapter scaffold — `AwsTelemetrySource`
with `IotCoreTelemetrySource` (MQTT), `TimestreamTelemetrySource`,
`S3TelemetrySource` (historical files, replayed exactly like the CSV) and
`RestTelemetrySource` — plus `README.md` describing how each would be
implemented.  None is functional yet and **no AWS SDK is imported by the
default installation** (`requirements-aws.txt` lists the optional extras,
commented out).  Configuring `integrations.telemetry.source: aws` makes the
Diagnostics row read `failed` with the reason, not `healthy`.

Still required from the ZETRABOT team (details in `harvest_integrations/aws/README.md`):
which AWS service partners read from (IoT Core topic / Timestream table / S3
bucket / HTTP API) and region; authentication; the document schema
(ideally the export's `source_message` + `signals` shape) and timestamp
conventions; the dictionary for signals not in the export — GPS, charging
state, alarms, `LimitsStatus`, `MemoryData2`; message rates, retention and
ordering guarantees; the battery's nominal capacity, to confirm the
mission-derived estimate.

## Still required for direct Modbus / OPC-UA control of ZETRABOT

The control path (`FleetInterface → DeviceIO → Modbus/OPC-UA`) is exercised
today against the bundled farm simulator.  Pointing it at the real robot
needs, from ZETRABOT: the protocol it exposes (Modbus TCP register map or
OPC-UA address space, or a ROS 2 interface), the writable points that
correspond to HARVEST's commands (`request_charge`/`release_charge`,
`assign_task`/`preempt_task`, `v2l_start`/`v2l_stop`), the readable points
mirroring the telemetry above (so the two sources can be cross-checked), the
network path and credentials, and the safety interlocks (who may command what,
and what the robot does when HARVEST goes silent).

