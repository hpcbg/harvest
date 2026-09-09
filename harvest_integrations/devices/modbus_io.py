"""
Modbus TCP backend for :class:`harvest_integrations.devices.DeviceIO`.

Adapted from TEMPO's ``ModbusCellIO`` (tempo_adapters/cell_io.py), generalised
so the register map comes from :class:`PointSpec` configuration instead of
hard-coded constants: values are 16-bit holding registers in engineering units
times ``1/scale`` (e.g. kW x10 when ``scale: 0.1``).

Requires ``pymodbus`` (pinned 3.6.9 in requirements-integrations.txt -- 3.14+
is mid-migration to a new datastore API, same rationale as TEMPO).
"""
from __future__ import annotations

from typing import Dict, Optional

from .base import DeviceEndpoint, PointSpec


class ModbusDeviceIO:
    """One Modbus TCP unit; every point is a holding register."""

    def __init__(self, endpoint: DeviceEndpoint):
        self._endpoint = endpoint
        self._host = str(endpoint.options.get("host", "127.0.0.1"))
        self._port = int(endpoint.options.get("port", 5020))
        self._unit = int(endpoint.options.get("unit", 1))
        self._points: Dict[str, PointSpec] = endpoint.points
        self._client = None

    # -- connection management ------------------------------------------------
    def _ensure_connected(self) -> bool:
        if self._client is None:
            from pymodbus.client import ModbusTcpClient  # lazy: optional dep
            self._client = ModbusTcpClient(self._host, port=self._port, timeout=2.0)
        if self._client.connected:
            return True
        return bool(self._client.connect())

    def _disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None

    # -- DeviceIO -------------------------------------------------------------
    def read(self) -> Optional[Dict[str, float]]:
        if not self._points:
            return {}
        try:
            if not self._ensure_connected():
                return None
            # One request per contiguous block would be an optimisation; the
            # farm poll rate (~1 Hz) makes per-point reads acceptable and keeps
            # arbitrary register layouts valid.
            out: Dict[str, float] = {}
            for spec in self._points.values():
                rr = self._client.read_holding_registers(
                    int(spec.address), count=1, slave=self._unit
                )
                if rr.isError():
                    return None
                raw = rr.registers[0]
                # Registers are unsigned; interpret as int16 so negative
                # engineering values (e.g. V2L reverse power) survive.
                if raw >= 0x8000:
                    raw -= 0x10000
                out[spec.name] = raw * spec.scale
            return out
        except Exception:
            self._disconnect()
            return None

    def write(self, point: str, value: float) -> None:
        spec = self._require_writable(point)
        if not self._ensure_connected():
            raise ConnectionError(
                f"Modbus device {self._endpoint.device_id} unreachable "
                f"at {self._host}:{self._port}"
            )
        raw = int(round(float(value) / spec.scale)) & 0xFFFF
        rq = self._client.write_register(int(spec.address), raw, slave=self._unit)
        if rq.isError():
            self._disconnect()
            raise ConnectionError(
                f"Modbus write failed: {self._endpoint.device_id}.{point}"
            )

    def close(self) -> None:
        self._disconnect()

    # -- helpers --------------------------------------------------------------
    def _require_writable(self, point: str) -> PointSpec:
        spec = self._points.get(point)
        if spec is None:
            raise KeyError(
                f"Device {self._endpoint.device_id} has no point {point!r}"
            )
        if not spec.writable:
            raise PermissionError(
                f"Point {self._endpoint.device_id}.{point} is not writable"
            )
        return spec
