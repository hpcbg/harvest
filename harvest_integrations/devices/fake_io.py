"""
In-memory DeviceIO used by tests and by the ``fake`` protocol.

Mirrors TEMPO's ``FakeCellIO``: no server, no network, records every write so
tests can assert on actuation without hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .base import DeviceEndpoint


@dataclass
class FakeDeviceIO:
    values: Dict[str, float] = field(default_factory=dict)
    writes: List[Tuple[str, float]] = field(default_factory=list)
    offline: bool = False

    @classmethod
    def from_endpoint(cls, endpoint: DeviceEndpoint) -> "FakeDeviceIO":
        # Initial values may be supplied inline in the endpoint options.
        initial = {
            k: float(v)
            for k, v in (endpoint.options.get("initial") or {}).items()
        }
        return cls(values=initial)

    def read(self) -> Optional[Dict[str, float]]:
        if self.offline:
            return None
        return dict(self.values)

    def write(self, point: str, value: float) -> None:
        if self.offline:
            raise ConnectionError("fake device offline")
        self.writes.append((point, float(value)))
        self.values[point] = float(value)

    def close(self) -> None:
        return None
