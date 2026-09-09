"""
Live protocol integration tests: the bundled farm simulator's Modbus and
OPC-UA servers run in-process, and the real protocol clients talk to them over
loopback TCP.  Skipped automatically when pymodbus/asyncua are not installed
(they are optional dependencies -- see requirements-integrations.txt).

Also proves cross-protocol coherence: a charge request written over OPC-UA
must show up as charger power on the Modbus side, because both skins share one
FarmState.
"""
import asyncio
import threading
import time
import unittest

try:
    import pymodbus  # noqa: F401
    import asyncua   # noqa: F401
    _PROTO_LIBS = True
except ImportError:
    _PROTO_LIBS = False

MODBUS_PORT = 15020
OPCUA_PORT = 14840
OPCUA_ENDPOINT = f"opc.tcp://127.0.0.1:{OPCUA_PORT}/harvest/"


@unittest.skipUnless(_PROTO_LIBS, "pymodbus/asyncua not installed")
class TestProtocolsEndToEnd(unittest.TestCase):
    """One simulator instance shared by every test in this class."""

    @classmethod
    def setUpClass(cls):
        from harvest_integrations.simulators.farm_state import FarmState
        from harvest_integrations.simulators.modbus_server import ModbusFarmServer
        from harvest_integrations.simulators.opcua_server import OpcUaFarmServer

        cls.state = FarmState.from_config(None)
        cls.loop = asyncio.new_event_loop()
        cls._stop = threading.Event()

        async def _run():
            modbus = ModbusFarmServer(cls.state)
            opcua = OpcUaFarmServer(cls.state, OPCUA_ENDPOINT)
            await opcua.start()
            server_task = asyncio.create_task(modbus.serve("127.0.0.1", MODBUS_PORT))
            try:
                while not cls._stop.is_set():
                    cls.state.tick(60.0)            # 1 sim minute per iteration
                    await opcua.sync()
                    modbus.sync()
                    await asyncio.sleep(0.05)
            finally:
                server_task.cancel()
                await opcua.stop()

        cls.thread = threading.Thread(
            target=lambda: cls.loop.run_until_complete(_run()), daemon=True)
        cls.thread.start()
        time.sleep(2.0)                              # let both servers bind

    @classmethod
    def tearDownClass(cls):
        cls._stop.set()
        cls.thread.join(timeout=10.0)
        cls.loop.close()

    # -- helpers --------------------------------------------------------------
    def _modbus_io(self, device_id, kind, points):
        from harvest_integrations.devices import DeviceEndpoint, create_device_io
        return create_device_io(DeviceEndpoint(
            device_id=device_id, kind=kind, protocol="modbus",
            options={"host": "127.0.0.1", "port": MODBUS_PORT}, points=points))

    def _opcua_io(self, tractor_id, points):
        from harvest_integrations.devices import DeviceEndpoint, create_device_io
        return create_device_io(DeviceEndpoint(
            device_id=tractor_id, kind="tractor", protocol="opcua",
            options={"endpoint": OPCUA_ENDPOINT, "object_path": tractor_id},
            points=points))

    # -- tests ----------------------------------------------------------------
    def test_modbus_grid_read(self):
        from harvest_integrations.devices import PointSpec
        io = self._modbus_io("grid", "grid", {
            "grid_cap_kw": PointSpec("grid_cap_kw", 2, 0.1),
            "tariff_code": PointSpec("tariff_code", 4),
            "price_eur_per_kwh": PointSpec("price_eur_per_kwh", 5, 0.001),
        })
        try:
            vals = io.read()
            self.assertIsNotNone(vals)
            self.assertAlmostEqual(vals["grid_cap_kw"], 10.5, places=1)
            self.assertIn(int(vals["tariff_code"]), (0, 1, 2))
            self.assertGreater(vals["price_eur_per_kwh"], 0.0)
        finally:
            io.close()

    def test_modbus_load_shed_write(self):
        from harvest_integrations.devices import PointSpec
        io = self._modbus_io("load_1", "load", {
            "power_kw": PointSpec("power_kw", 200, 0.1),
            "shed": PointSpec("shed", 201, writable=True),
        })
        try:
            io.write("shed", 1)
            time.sleep(0.3)                          # a few sim ticks
            self.assertTrue(self.state.loads[0].shed)
            self.assertEqual(io.read()["shed"], 1.0)
            io.write("shed", 0)
            time.sleep(0.3)
            self.assertFalse(self.state.loads[0].shed)
        finally:
            io.close()

    def test_opcua_tractor_read(self):
        from harvest_integrations.devices import PointSpec
        io = self._opcua_io("tractor_1", {
            "soc_pct": PointSpec("soc_pct", "SoC_pct"),
            "available": PointSpec("available", "Available"),
        })
        try:
            vals = io.read()
            self.assertIsNotNone(vals)
            self.assertGreater(vals["soc_pct"], 0.0)
            self.assertEqual(vals["available"], 1.0)
        finally:
            io.close()

    def test_cross_protocol_charge_flow(self):
        """OPC-UA charge request -> SoC rises and Modbus charger shows power."""
        from harvest_integrations.devices import PointSpec
        bms = self._opcua_io("tractor_1", {
            "soc_pct": PointSpec("soc_pct", "SoC_pct"),
            "charging": PointSpec("charging", "Charging"),
            "charge_request": PointSpec("charge_request", "Charge_Request", writable=True),
        })
        charger = self._modbus_io("charger_1", "charger", {
            "power_kw": PointSpec("power_kw", 100, 0.1),
            "occupied": PointSpec("occupied", 102),
        })
        try:
            soc_before = bms.read()["soc_pct"]
            bms.write("charge_request", 1)
            deadline = time.time() + 10.0
            charging, power, occupied = 0.0, 0.0, 0.0
            while time.time() < deadline:
                time.sleep(0.5)
                b, c = bms.read(), charger.read()
                if b is None or c is None:
                    continue
                charging, power, occupied = b["charging"], c["power_kw"], c["occupied"]
                if charging and power > 0:
                    break
            self.assertEqual(charging, 1.0, "tractor never started charging")
            self.assertGreater(power, 0.0, "charger power not visible over Modbus")
            self.assertEqual(occupied, 1.0, "charger occupancy not visible")
            time.sleep(1.5)                          # ~90 sim-minutes of charging
            self.assertGreater(bms.read()["soc_pct"], soc_before)
        finally:
            bms.write("charge_request", 0)
            bms.close()
            charger.close()

    def test_device_fleet_interface_against_simulator(self):
        """The full DeviceFleetInterface stack over both live protocols."""
        from harvest_control.interface import Command
        from harvest_integrations.runtime import build_device_endpoints
        from harvest_integrations.devices import DeviceFleetInterface

        cfg = {
            "tractors": {"fleet": [{"id": "tractor_1"}, {"id": "tractor_2"},
                                   {"id": "tractor_3"}]},
            "charging": {"stations": [{"id": "charger_1"}, {"id": "charger_2"}]},
            "energy_consumers": [{"id": "electric_fence"}, {"id": "cold_storage"},
                                 {"id": "workshop_tools"}],
            "integrations": {"fleet": {
                "backend": "devices",
                "modbus_host": "127.0.0.1", "modbus_port": MODBUS_PORT,
                "opcua_endpoint": OPCUA_ENDPOINT,
            }},
        }
        fleet = DeviceFleetInterface(build_device_endpoints(cfg))
        try:
            snap = fleet.snapshot()
            self.assertEqual(len(snap.tractors), 3)
            self.assertEqual(len(snap.chargers), 2)
            self.assertEqual(len(snap.loads), 3)
            self.assertGreater(snap.grid.grid_cap_kw, 0.0)
            self.assertTrue(all(t.soc_pct > 0 for t in snap.tractors))

            acks = fleet.submit([Command.shed_load("cold_storage")])
            self.assertTrue(acks[0].accepted, acks[0].reason)
            time.sleep(0.3)
            self.assertTrue(
                next(l for l in self.state.loads if l.id == "cold_storage").shed)
            fleet.submit([Command.restore_load("cold_storage")])
        finally:
            fleet.close()


if __name__ == "__main__":
    unittest.main()
