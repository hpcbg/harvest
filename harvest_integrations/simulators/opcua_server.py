"""
OPC-UA skin over :class:`FarmState` (tractor battery-management view).

Address space (namespace ``harvest``)::

    Objects/
      tractor_<n>/
        SoC_pct        Double        Pos_X          Double
        Energy_kWh     Double        Pos_Y          Double
        Available      Boolean       Charge_Request Int32   (writable)
        Charging       Boolean       V2L_kW         Double  (writable)
        Discharging    Boolean
        Discharge_kW   Double

Same information style as TEMPO's ``opcua_cell_sim.py``: names and types live
in the server's address space instead of a register map.
"""
from __future__ import annotations

from typing import Dict

from .farm_state import FarmState

NAMESPACE = "harvest"


class OpcUaFarmServer:
    def __init__(self, state: FarmState, endpoint: str):
        self.state = state
        self.endpoint = endpoint
        self._server = None
        self._nodes: Dict[str, Dict[str, object]] = {}

    async def start(self) -> None:
        from asyncua import Server, ua

        server = Server()
        await server.init()
        server.set_endpoint(self.endpoint)
        server.set_server_name("HARVEST farm simulator")
        idx = await server.register_namespace(NAMESPACE)
        objects = server.nodes.objects

        async def var(parent, name, value, vtype, writable=False):
            node = await parent.add_variable(idx, name, value, vtype)
            if writable:
                await node.set_writable()
            return node

        for t in self.state.tractors:
            obj = await objects.add_object(idx, t.id)
            self._nodes[t.id] = {
                "SoC_pct": await var(obj, "SoC_pct", t.soc_pct, ua.VariantType.Double),
                "Energy_kWh": await var(obj, "Energy_kWh", t.energy_kwh, ua.VariantType.Double),
                "Available": await var(obj, "Available", t.available, ua.VariantType.Boolean),
                "Charging": await var(obj, "Charging", t.charging, ua.VariantType.Boolean),
                "Discharging": await var(obj, "Discharging", t.discharging, ua.VariantType.Boolean),
                "Discharge_kW": await var(obj, "Discharge_kW", 0.0, ua.VariantType.Double),
                "Pos_X": await var(obj, "Pos_X", t.pos[0], ua.VariantType.Double),
                "Pos_Y": await var(obj, "Pos_Y", t.pos[1], ua.VariantType.Double),
                "Charge_Request": await var(obj, "Charge_Request", 0, ua.VariantType.Int32, writable=True),
                "V2L_kW": await var(obj, "V2L_kW", 0.0, ua.VariantType.Double, writable=True),
            }
        self._server = server
        await server.start()

    async def stop(self) -> None:
        if self._server is not None:
            await self._server.stop()

    async def sync(self) -> None:
        """Pull writable nodes into the state, push telemetry out."""
        for t in self.state.tractors:
            nodes = self._nodes[t.id]
            t.charge_request = bool(await nodes["Charge_Request"].read_value())
            t.v2l_kw = max(0.0, float(await nodes["V2L_kW"].read_value()))

            await nodes["SoC_pct"].write_value(round(t.soc_pct, 2))
            await nodes["Energy_kWh"].write_value(round(t.energy_kwh, 3))
            await nodes["Available"].write_value(t.available)
            await nodes["Charging"].write_value(t.charging)
            await nodes["Discharging"].write_value(t.discharging)
            await nodes["Discharge_kW"].write_value(t.v2l_kw if t.discharging else 0.0)
            await nodes["Pos_X"].write_value(t.pos[0])
            await nodes["Pos_Y"].write_value(t.pos[1])
