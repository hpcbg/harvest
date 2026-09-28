"""
harvest_integrations.telemetry
==============================

Generic telemetry ingestion for HARVEST -- independent of AWS, CSV and FIWARE.

    CsvTelemetrySource ---\\
                           \\
    AwsTelemetrySource -----> TelemetryNormalizer --> ZetrabotTelemetry (canonical)
    (scaffold, aws/)                                        |
                                            +---------------+---------------+
                                            |                               |
                                     HARVEST state                  FIWARE / NGSI-LD
                                  (FleetRuntime merges              (TractorTelemetry
                                   TractorState rows)                entity, mirrored by
                                            |                        the sync daemon)
                                   replay / calibration
                                   (replay.py, analysis.py)

* :mod:`model`      -- :class:`TelemetryMessage` (raw record), :class:`ZetrabotTelemetry`
                       (canonical state), :class:`TelemetrySource` (the seam: ``history()``
                       + ``subscribe()``), :class:`TelemetrySink`;
* :mod:`zetrack`    -- the ZETRABOT / Zetrack V2 signal dictionary and the CSV source;
* :mod:`normalizer` -- raw messages -> canonical state (partial updates, ordering,
                       unknown-field preservation, derived power / cumulative energy);
* :mod:`replay`     -- original-timeline replay with acceleration and gap compression;
* :mod:`service`    -- wiring into ``FleetRuntime`` / Diagnostics / FIWARE;
* :mod:`analysis`   -- energy-model calibration against the real mission.

Telemetry ingestion is read-only.  Robot control stays in
``harvest_integrations.devices`` (Modbus / OPC-UA) -- the two never mix.
"""
from .model import (
    SOURCE_CSV_REPLAY,
    Subscription,
    TelemetryMessage,
    TelemetrySink,
    TelemetrySource,
    ZetrabotTelemetry,
    parse_timestamp,
    sort_messages,
)
from .normalizer import TelemetryNormalizer, TelemetryRegistry
from .replay import ReplayOptions, ReplayPlayer
from .zetrack import CsvTelemetrySource, read_csv

__all__ = [
    "SOURCE_CSV_REPLAY",
    "CsvTelemetrySource",
    "ReplayOptions",
    "ReplayPlayer",
    "Subscription",
    "TelemetryMessage",
    "TelemetryNormalizer",
    "TelemetryRegistry",
    "TelemetrySink",
    "TelemetrySource",
    "ZetrabotTelemetry",
    "parse_timestamp",
    "read_csv",
    "sort_messages",
]
