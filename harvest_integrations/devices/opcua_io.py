"""
OPC-UA backend for :class:`harvest_integrations.devices.DeviceIO`.

Adapted from TEMPO's ``OpcUaCellIO`` (tempo_adapters/cell_io.py): uses the
``asyncua.sync`` facade so callers stay synchronous and no asyncio event loop
leaks into HARVEST's threads.  Point addresses are node browse-names resolved
under ``Objects/<ns>:<object_path>``, where ``object_path`` defaults to the
device id (e.g. ``Objects/harvest:tractor_1/harvest:SoC_pct``).

Where Modbus hides meaning behind scaled integers at numbered registers,
OPC-UA carries names and types in the server -- keeping both behind the same
``DeviceIO`` seam is the whole point of the abstraction.

Requires ``asyncua`` (see requirements-integrations.txt).
"""
from __future__ import annotations

from typing import Dict, Optional

from .base import DeviceEndpoint, PointSpec


class OpcUaDeviceIO:
    """One OPC-UA object (one device) on a server."""

    def __init__(self, endpoint: DeviceEndpoint):
        self._endpoint = endpoint
        self._url = str(endpoint.options.get("endpoint", "opc.tcp://127.0.0.1:4840/harvest/"))
        self._namespace = str(endpoint.options.get("namespace", "harvest"))
        self._object_path = str(endpoint.options.get("object_path", endpoint.device_id))
        self._points: Dict[str, PointSpec] = endpoint.points
        self._client = None
        self._nodes: Dict[str, object] = {}

    # -- connection management ------------------------------------------------
    def _ensure_connected(self) -> bool:
        if self._client is not None:
            return True
        try:
            from asyncua.sync import Client  # lazy: optional dep
            client = Client(self._url)
            client.connect()
            idx = client.get_namespace_index(self._namespace)
            root = client.nodes.objects
            path = [f"{idx}:{part}" for part in self._object_path.split("/")]
            obj = root.get_child(path)
            nodes: Dict[str, object] = {}
            for spec in self._points.values():
                nodes[spec.name] = obj.get_child([f"{idx}:{spec.address}"])
            self._client, self._nodes = client, nodes
            return True
        except Exception:
            self._disconnect()
            return False

    def _disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
        self._client = None
        self._nodes = {}

    # -- DeviceIO -------------------------------------------------------------
    def read(self) -> Optional[Dict[str, float]]:
        if not self._points:
            return {}
        try:
            if not self._ensure_connected():
                return None
            out: Dict[str, float] = {}
            for spec in self._points.values():
                value = self._nodes[spec.name].read_value()
                out[spec.name] = float(value) * spec.scale
            return out
        except Exception:
            # Drop the session; the next poll reconnects (TEMPO pattern).
            self._disconnect()
            return None

    def write(self, point: str, value: float) -> None:
        spec = self._points.get(point)
        if spec is None:
            raise KeyError(f"Device {self._endpoint.device_id} has no point {point!r}")
        if not spec.writable:
            raise PermissionError(
                f"Point {self._endpoint.device_id}.{point} is not writable"
            )
        if not self._ensure_connected():
            raise ConnectionError(
                f"OPC-UA device {self._endpoint.device_id} unreachable at {self._url}"
            )
        try:
            node = self._nodes[spec.name]
            # Preserve the server-declared type: write back what we read.
            from asyncua import ua
            current = node.read_data_value()
            vtype = current.Value.VariantType
            raw = float(value) / spec.scale
            if vtype in (ua.VariantType.Int16, ua.VariantType.Int32,
                         ua.VariantType.Int64, ua.VariantType.UInt16,
                         ua.VariantType.UInt32, ua.VariantType.UInt64):
                cast = int(round(raw))
            elif vtype == ua.VariantType.Boolean:
                cast = bool(round(raw))
            else:
                cast = raw
            node.write_value(ua.DataValue(ua.Variant(cast, vtype)))
        except Exception as exc:
            self._disconnect()
            raise ConnectionError(
                f"OPC-UA write failed: {self._endpoint.device_id}.{point}: {exc}"
            ) from exc

    def close(self) -> None:
        self._disconnect()
