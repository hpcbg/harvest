# AWS telemetry source (scaffold)

ZETRABOT's telemetry is collected by its manufacturer through AWS.  HARVEST
will read it from there once the ZETRABOT team documents *which* service is
exposed to partners.  Until then this package holds only the adapter
scaffold (`base.py`) and this document.  **No AWS SDK is installed or
imported by default** (`requirements-aws.txt` lists the candidates, all
commented out).

```
ZETRABOT -> AWS -> [harvest_integrations.aws adapter]
                          |  TelemetryMessage records
                          v
              harvest_integrations.telemetry.TelemetryNormalizer
                          |  ZetrabotTelemetry (canonical)
             +------------+-------------+
             v                          v
     HARVEST state (FleetRuntime)   FIWARE / NGSI-LD (TractorTelemetry)

HARVEST control -> FleetInterface -> DeviceIO -> Modbus / OPC-UA -> ZETRABOT
```

The two arrows never meet: this package is **inbound telemetry only**.
Commands to the robot go through `harvest_integrations.devices`.

## The contract every adapter fulfils

`AwsTelemetrySource` is a `TelemetrySource` (`harvest_integrations/telemetry/model.py`):

| Method | Meaning | CSV replay today |
|---|---|---|
| `history(start, end, tractor_id)` | ordered `TelemetryMessage` records for a time range | the parsed file |
| `subscribe(callback, start=None)` | deliver records to `callback` as they arrive, in order | `ReplayPlayer` on the original timestamps |
| `describe()` | non-secret connection facts for Diagnostics | file name, parse stats |

A record is a `TelemetryMessage(tractor_id, timestamp, source_message,
signals, mission_id, sequence, schema_version, record_id, received_at,
source, extra)`.  If the AWS documents keep the export's
`source_message` / `signals` split, the ZETRABOT dictionary in
`telemetry/zetrack.py` decodes them unchanged; if they are flattened, the
dictionary's by-name fallback (`zetrack.lookup`) still resolves every known
signal, and unknown ones are preserved raw by the normaliser.

Selecting an adapter (config.yaml, all values overridable with
`HARVEST_TELEMETRY_*` environment variables):

```yaml
integrations:
  telemetry:
    source: aws            # none | csv | aws
    aws:
      adapter: iot-core    # iot-core | timestream | s3 | rest
      region: eu-west-1
      # adapter-specific keys below; credentials come from the standard AWS
      # credential chain (env / profile / instance role), never from this file
```

Every adapter currently raises `NotImplementedError` with a pointer here;
the Diagnostics view shows the telemetry source as **failed** with that
reason rather than pretending anything is live.

## How each candidate adapter would be implemented

### AWS IoT Core / MQTT (`IotCoreTelemetrySource`)
* Dependency: `awsiotsdk` (or `paho-mqtt` with SigV4/WebSocket auth).
* `connect()`: open the MQTT-over-TLS connection to the account's IoT endpoint
  with the device/partner certificate or SigV4 credentials.
* `subscribe()`: subscribe to the telemetry topic(s) (e.g.
  `zetrabot/<tractor_id>/telemetry`); each payload becomes one
  `TelemetryMessage` (timestamp from the payload, not from receipt time).
  Return a `Subscription` whose `stop()` unsubscribes.
* `history()`: MQTT has no history.  Either raise (documented) or delegate to
  a store adapter (Timestream / S3) configured alongside.
* Out-of-order delivery is possible at QoS 0/1: the normaliser tolerates
  it (counted in `out_of_order`).

### Amazon Timestream (`TimestreamTelemetrySource`)
* Dependency: `boto3` (`timestream-query`).
* `history()`: one SQL query per range, ordered by time, paginated with
  `NextToken`; map the row's measure columns onto `signals`.
* `subscribe()`: poll `history(last_seen, now)` every N seconds from a
  background thread (a `ReplayPlayer` with `speed<=0` can pace delivery).

### S3 historical files (`S3TelemetrySource`)
* Dependency: `boto3` (`s3`).
* `history()`: list objects under `bucket/prefix`, download the ones whose
  key/date falls in range, parse with `telemetry.zetrack.read_csv` (or a JSON
  variant) and merge-sort by `sort_key()`.
* `subscribe()`: replay `history()` with `ReplayPlayer` — identical to the
  CSV source, which is the point of the abstraction.

### REST / API source (`RestTelemetrySource`)
* Dependency: none (`urllib`), or `requests` if preferred.
* `history()`: paged `GET` with `from`/`to` query parameters, auth via
  bearer token or API key from the environment.
* `subscribe()`: poll the "latest" endpoint or long-poll, de-duplicate on
  `record_id`.

## Information still required from the ZETRABOT team

1. **Which service** partners read from (IoT Core topic, Timestream
   database/table, S3 bucket/prefix, or an HTTP API) and its region.
2. **Authentication**: IAM role to assume / access-key pair / device
   certificate / API key, and how credentials are rotated.
3. **Document schema**: whether records keep the export's
   `{tractor_id, mission_id, timestamp, sequence, schema_version,
   source_message, signals}` shape, the timestamp zone and precision, and
   the `schema_version` change policy.
4. **Message dictionary** for anything not in the CSV: GPS
   (latitude/longitude, the route map's source), charging/charger state,
   any error/alarm messages, and the meaning of `LimitsStatus` and
   `MemoryData2`.
5. **Rates and retention**: message rates per type, history depth, and
   whether late/out-of-order delivery happens.
6. **Session semantics** of `DischEnrgActualSesion` and `sequence`
   (confirmed from the data: both reset at power-on) and the battery's
   nominal capacity, to confirm the mission-derived effective capacity
   estimate produced by `telemetry/analysis.py`.
7. **Mission identity**: how `mission_id` is assigned and whether a mission
   can span several days (mission 63 spans three).
