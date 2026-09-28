"""
harvest_integrations.aws
========================

Telemetry-source integration for the AWS side of ZETRABOT's data pipeline.

This is a *telemetry / data-source* integration, which is why it lives next
to ``telemetry`` and not under ``devices``: ``devices`` is the field-control
seam (Modbus / OPC-UA, commands to the robot), this package is inbound data
only.  It currently holds the adapter scaffold (:mod:`base`) and the
documentation of what is still required from the ZETRABOT team
(``README.md``).  No AWS SDK is imported by the default installation.
"""
from .base import (
    AwsTelemetrySource,
    IotCoreTelemetrySource,
    RestTelemetrySource,
    S3TelemetrySource,
    TimestreamTelemetrySource,
    build_aws_source,
)

__all__ = [
    "AwsTelemetrySource",
    "IotCoreTelemetrySource",
    "RestTelemetrySource",
    "S3TelemetrySource",
    "TimestreamTelemetrySource",
    "build_aws_source",
]
