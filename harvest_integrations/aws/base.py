"""
AWS telemetry-source scaffold.

ZETRABOT's telemetry reaches its operator through AWS; *which* AWS service
HARVEST will read from is not yet documented by the ZETRABOT team, so this
module deliberately implements no transport.  It fixes the shape every AWS
adapter will have -- a :class:`~harvest_integrations.telemetry.model.TelemetrySource`
producing :class:`~harvest_integrations.telemetry.model.TelemetryMessage`
records -- so that swapping the CSV replay for the live feed touches nothing
above the telemetry seam.

No AWS SDK is imported here or anywhere in the default HARVEST installation.
When the service is known, the matching adapter imports its client lazily
(inside ``connect()``), and the dependency goes into ``requirements-aws.txt``
as an optional extra.  See ``README.md`` in this directory for the four
candidate adapters and the information still required from ZETRABOT.

Telemetry only.  Commands to the robot are NOT sent through AWS from here;
HARVEST control goes ``FleetInterface -> DeviceIO -> Modbus / OPC-UA``.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Iterator, Optional

from harvest_integrations.telemetry.model import (
    SOURCE_AWS_IOT,
    SOURCE_AWS_S3,
    SOURCE_AWS_TIMESTREAM,
    SOURCE_REST,
    MessageCallback,
    Subscription,
    TelemetryMessage,
    TelemetrySource,
)

ADAPTER_KINDS = {
    "iot-core": SOURCE_AWS_IOT,
    "timestream": SOURCE_AWS_TIMESTREAM,
    "s3": SOURCE_AWS_S3,
    "rest": SOURCE_REST,
}

_PENDING = ("not implemented yet: the AWS service ZETRABOT exposes is undocumented. "
            "See harvest_integrations/aws/README.md for what is needed and how each "
            "adapter would be written; replay the CSV export meanwhile "
            "(HARVEST_TELEMETRY_SOURCE=csv).")


class AwsTelemetrySource(TelemetrySource):
    """Common base for the future AWS adapters.

    Subclasses implement :meth:`connect`, :meth:`history` and
    :meth:`subscribe`, and translate each incoming document into
    :class:`TelemetryMessage` -- ideally with the same ``source_message`` /
    ``signals`` split the CSV export uses, so the ZETRABOT dictionary in
    :mod:`harvest_integrations.telemetry.zetrack` decodes both unchanged.
    """

    kind = "aws"

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        self.options: Dict[str, Any] = dict(options or {})
        self.region = self.options.get("region")
        self.connected = False

    def connect(self) -> None:
        """Open the AWS client (lazy SDK import belongs here)."""
        raise NotImplementedError(f"{type(self).__name__}: {_PENDING}")

    def history(self, start: Optional[_dt.datetime] = None, end: Optional[_dt.datetime] = None,
                tractor_id: Optional[str] = None) -> Iterator[TelemetryMessage]:
        raise NotImplementedError(f"{type(self).__name__}.history: {_PENDING}")

    def subscribe(self, callback: MessageCallback, *,
                  start: Optional[_dt.datetime] = None) -> Subscription:
        raise NotImplementedError(f"{type(self).__name__}.subscribe: {_PENDING}")

    def describe(self) -> Dict[str, Any]:
        # Never echo credentials: only the non-secret connection facts.
        safe = {k: v for k, v in self.options.items()
                if k in ("region", "endpoint", "topic", "database", "table", "bucket",
                         "prefix", "url", "profile")}
        return {"kind": self.kind, "implemented": False, "connected": self.connected, **safe}


class IotCoreTelemetrySource(AwsTelemetrySource):
    """AWS IoT Core (MQTT) -- live stream.  ``history()`` would need a store."""
    kind = SOURCE_AWS_IOT


class TimestreamTelemetrySource(AwsTelemetrySource):
    """Amazon Timestream -- ``history()`` by SQL time range; ``subscribe()`` by polling."""
    kind = SOURCE_AWS_TIMESTREAM


class S3TelemetrySource(AwsTelemetrySource):
    """S3 objects (CSV/JSON exports) -- ``history()`` from files, replayed with ReplayPlayer."""
    kind = SOURCE_AWS_S3


class RestTelemetrySource(AwsTelemetrySource):
    """An HTTP API (API Gateway or the Zetrack backend) -- paged ``history()``, polled ``subscribe()``."""
    kind = SOURCE_REST


def build_aws_source(options: Dict[str, Any]) -> AwsTelemetrySource:
    """Instantiate the adapter named by ``integrations.telemetry.aws.adapter``.

    Every adapter currently raises ``NotImplementedError`` when used; the
    instance exists so Diagnostics can report *why* the source is not live.
    """
    adapter = str(options.get("adapter") or "").strip().lower()
    classes = {
        "iot-core": IotCoreTelemetrySource,
        "timestream": TimestreamTelemetrySource,
        "s3": S3TelemetrySource,
        "rest": RestTelemetrySource,
    }
    if adapter not in classes:
        raise ValueError(
            "integrations.telemetry.aws.adapter must be one of "
            f"{sorted(classes)} (got {adapter!r}); no adapter is functional yet -- {_PENDING}")
    return classes[adapter](options)


__all__ = ["ADAPTER_KINDS", "AwsTelemetrySource", "IotCoreTelemetrySource", "RestTelemetrySource",
           "S3TelemetrySource", "TimestreamTelemetrySource", "build_aws_source"]
