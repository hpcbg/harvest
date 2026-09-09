"""
harvest_integrations.devices
============================

Protocol abstraction for heterogeneous farm devices, adapted from TEMPO's
``CellIO`` seam (tempo_adapters/cell_io.py).

The single boundary is :class:`DeviceIO`: ``read()`` returns a flat
``{point_name: value}`` dict, ``write(point, value)`` actuates one named point.
Which register / OPC-UA node a point lives at is *configuration*, not code, so
one client class serves every Modbus (or OPC-UA) device on the farm.

Protocol backends are looked up in a registry by name; adding a protocol means
registering a factory -- nothing above this package changes.
"""
from .base import (
    DeviceEndpoint,
    DeviceIO,
    PointSpec,
    available_protocols,
    create_device_io,
    endpoint_from_dict,
    register_protocol,
)
from .fake_io import FakeDeviceIO
from .fleet_backend import DeviceFleetInterface

__all__ = [
    "DeviceEndpoint",
    "DeviceIO",
    "PointSpec",
    "FakeDeviceIO",
    "DeviceFleetInterface",
    "available_protocols",
    "create_device_io",
    "endpoint_from_dict",
    "register_protocol",
]
