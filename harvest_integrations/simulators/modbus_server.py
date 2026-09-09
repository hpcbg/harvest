"""
Modbus TCP skin over :class:`FarmState` (chargers, loads, grid meter).

Register map (holding registers, FC 3/6, unit 1, ``zero_mode`` addressing)::

    0  clock_min                    3  pv_kw x10
    1  grid_draw_kw x10 (int16)     4  tariff_code   0 valle | 1 llano | 2 punta
    2  grid_cap_kw x10              5  price_eur_per_kwh x1000

    100 + i*10   charger i:  +0 power_kw x10 (int16, <0 = V2L)
                             +1 level (RW: 0 off | 1 half | 2 full)
                             +2 occupied (0 free, n = n-th tractor)

    200 + i*10   load i:     +0 power_kw x10
                             +1 shed (RW: 0/1)

The scaled-integer convention is TEMPO's (``modbus_cell_sim.py``): registers
are 16-bit, so meaning lives in this map, not on the wire -- which is exactly
what the OPC-UA skin does differently with the same state.
"""
from __future__ import annotations

from typing import Dict

from .farm_state import FarmState

GRID_BASE = 0
CHARGER_BASE = 100
LOAD_BASE = 200
BLOCK_STRIDE = 10

REG_COUNT = 400


def _s16(value: float) -> int:
    return int(round(value)) & 0xFFFF


def _from_s16(raw: int) -> int:
    return raw - 0x10000 if raw >= 0x8000 else raw


class ModbusFarmServer:
    """Owns the datastore; :meth:`sync` runs once per physics tick."""

    def __init__(self, state: FarmState):
        from pymodbus.datastore import (
            ModbusSequentialDataBlock,
            ModbusServerContext,
            ModbusSlaveContext,
        )
        self.state = state
        block = ModbusSequentialDataBlock(0, [0] * REG_COUNT)
        self._slave = ModbusSlaveContext(hr=block, zero_mode=True)
        self.context = ModbusServerContext(slaves=self._slave, single=True)
        # Shadow of the last values *we* pushed to writable registers; a
        # mismatch on the next tick means an external client wrote in between.
        self._shadow: Dict[int, int] = {}

    # -- register <-> state ---------------------------------------------------
    def sync(self) -> None:
        self._apply_external_writes()
        self._push_state()

    def _apply_external_writes(self) -> None:
        for i, charger in enumerate(self.state.chargers):
            addr = CHARGER_BASE + i * BLOCK_STRIDE + 1
            raw = self._get(addr)
            if addr in self._shadow and raw != self._shadow[addr]:
                charger.level = max(0, min(2, raw))
        for i, load in enumerate(self.state.loads):
            addr = LOAD_BASE + i * BLOCK_STRIDE + 1
            raw = self._get(addr)
            if addr in self._shadow and raw != self._shadow[addr]:
                load.shed = bool(raw)

    def _push_state(self) -> None:
        s = self.state
        self._set(GRID_BASE + 0, _s16(s.clock_min))
        self._set(GRID_BASE + 1, _s16(s.grid_draw_kw * 10))
        self._set(GRID_BASE + 2, _s16(s.grid_cap_kw * 10))
        self._set(GRID_BASE + 3, _s16(s.pv_kw * 10))
        self._set(GRID_BASE + 4, s.tariff_code)
        self._set(GRID_BASE + 5, _s16(s.price_eur_per_kwh * 1000))

        tractor_index = {t.id: n + 1 for n, t in enumerate(s.tractors)}
        for i, c in enumerate(s.chargers):
            base = CHARGER_BASE + i * BLOCK_STRIDE
            self._set(base + 0, _s16(c.power_kw * 10))
            self._set_writable(base + 1, c.level)
            self._set(base + 2, tractor_index.get(c.docked, 0))
        for i, l in enumerate(s.loads):
            base = LOAD_BASE + i * BLOCK_STRIDE
            self._set(base + 0, _s16(l.power_kw * 10))
            self._set_writable(base + 1, int(l.shed))

    # -- datastore helpers ----------------------------------------------------
    def _get(self, addr: int) -> int:
        return self._slave.getValues(3, addr, count=1)[0]

    def _set(self, addr: int, value: int) -> None:
        self._slave.setValues(3, addr, [value & 0xFFFF])

    def _set_writable(self, addr: int, value: int) -> None:
        self._set(addr, value)
        self._shadow[addr] = value & 0xFFFF

    # -- server ---------------------------------------------------------------
    async def serve(self, host: str, port: int) -> None:
        from pymodbus.server import StartAsyncTcpServer
        await StartAsyncTcpServer(context=self.context, address=(host, port))
