"""
Device/protocol seam.

A *device endpoint* is one addressable unit on the farm network (a charger, a
tractor BMS, a load controller, the grid meter).  Every endpoint exposes a set
of named *points*; where a point lives on the wire (holding register, OPC-UA
node, ...) is carried by :class:`PointSpec` and interpreted only by the
protocol backend.

Layering contract:

    config dict -> DeviceEndpoint -> create_device_io() -> DeviceIO
                                                            |
                        harvest_control dataclasses  <-  fleet_backend

Nothing above ``fleet_backend`` ever sees a protocol type, and no protocol
module ever imports the HARVEST domain model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
#  Point and endpoint descriptions (pure configuration data)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PointSpec:
    """One named datum on a device.

    ``address`` is protocol-specific: an ``int`` holding-register number for
    Modbus, a string node name (resolved under the device object) for OPC-UA.
    ``scale`` converts the raw wire value into engineering units
    (``value = raw * scale``); writes apply the inverse.
    """
    name: str
    address: Any
    scale: float = 1.0
    writable: bool = False


@dataclass
class DeviceEndpoint:
    """Connection description for one device, as loaded from configuration."""
    device_id: str
    kind: str                                # "tractor" | "charger" | "load" | "grid"
    protocol: str                            # registry key: "modbus" | "opcua" | "fake" | ...
    options: Dict[str, Any] = field(default_factory=dict)
    points: Dict[str, PointSpec] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  The protocol seam
# --------------------------------------------------------------------------- #
@runtime_checkable
class DeviceIO(Protocol):
    """What the rest of HARVEST needs from any field protocol.

    ``read`` returns every readable point in engineering units, or ``None``
    when the device is unreachable (callers treat that as "device offline",
    never as an exception).  ``write`` actuates a single writable point.
    """

    def read(self) -> Optional[Dict[str, float]]: ...

    def write(self, point: str, value: float) -> None: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
#  Protocol registry
# --------------------------------------------------------------------------- #
DeviceIOFactory = Callable[[DeviceEndpoint], DeviceIO]

_PROTOCOLS: Dict[str, DeviceIOFactory] = {}


def register_protocol(name: str, factory: DeviceIOFactory) -> None:
    """Register (or replace) a protocol backend under ``name``.

    Third-party code can call this to plug in additional protocols (MQTT,
    CAN/ISOBUS, vendor REST APIs, ...) without touching this package.
    """
    _PROTOCOLS[name.lower()] = factory


def available_protocols() -> list[str]:
    _ensure_builtins()
    return sorted(_PROTOCOLS)


def create_device_io(endpoint: DeviceEndpoint) -> DeviceIO:
    """Instantiate the backend for ``endpoint`` via the registry."""
    _ensure_builtins()
    try:
        factory = _PROTOCOLS[endpoint.protocol.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown device protocol {endpoint.protocol!r} for device "
            f"{endpoint.device_id!r}. Available: {', '.join(sorted(_PROTOCOLS))}"
        ) from None
    return factory(endpoint)


def _ensure_builtins() -> None:
    """Register built-in backends lazily.

    ``fake`` has no dependencies.  ``modbus``/``opcua`` register import their
    third-party client library only when first instantiated, so this module
    stays importable on a bare HARVEST install.
    """
    if "fake" not in _PROTOCOLS:
        from .fake_io import FakeDeviceIO
        register_protocol("fake", FakeDeviceIO.from_endpoint)
    if "modbus" not in _PROTOCOLS:
        register_protocol("modbus", _modbus_factory)
    if "opcua" not in _PROTOCOLS:
        register_protocol("opcua", _opcua_factory)


def _modbus_factory(endpoint: DeviceEndpoint) -> DeviceIO:
    from .modbus_io import ModbusDeviceIO
    return ModbusDeviceIO(endpoint)


def _opcua_factory(endpoint: DeviceEndpoint) -> DeviceIO:
    from .opcua_io import OpcUaDeviceIO
    return OpcUaDeviceIO(endpoint)


# --------------------------------------------------------------------------- #
#  Config parsing
# --------------------------------------------------------------------------- #
def endpoint_from_dict(raw: Dict[str, Any]) -> DeviceEndpoint:
    """Build a :class:`DeviceEndpoint` from one ``integrations.fleet.devices``
    config entry.

    Expected shape::

        id: charger_1
        kind: charger
        protocol: modbus
        host: 127.0.0.1        # protocol options stay flat in the entry
        port: 5020
        points:
          power_kw:  {address: 100, scale: 0.1}
          level:     {address: 101, writable: true}
    """
    known = {"id", "kind", "protocol", "points"}
    points: Dict[str, PointSpec] = {}
    for pname, praw in (raw.get("points") or {}).items():
        if isinstance(praw, dict):
            points[pname] = PointSpec(
                name=pname,
                address=praw.get("address"),
                scale=float(praw.get("scale", 1.0)),
                writable=bool(praw.get("writable", False)),
            )
        else:                                # shorthand: point_name: address
            points[pname] = PointSpec(name=pname, address=praw)
    return DeviceEndpoint(
        device_id=str(raw["id"]),
        kind=str(raw.get("kind", "device")),
        protocol=str(raw.get("protocol", "fake")),
        options={k: v for k, v in raw.items() if k not in known},
        points=points,
    )
