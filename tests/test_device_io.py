"""Unit tests for the device/protocol abstraction (no network, no hardware)."""
import unittest

from harvest_integrations.devices import (
    DeviceEndpoint,
    FakeDeviceIO,
    PointSpec,
    available_protocols,
    create_device_io,
    endpoint_from_dict,
    register_protocol,
)


class TestEndpointParsing(unittest.TestCase):
    def test_full_form(self):
        ep = endpoint_from_dict({
            "id": "charger_1",
            "kind": "charger",
            "protocol": "modbus",
            "host": "10.0.0.5",
            "port": 5020,
            "points": {
                "power_kw": {"address": 100, "scale": 0.1},
                "level": {"address": 101, "writable": True},
            },
        })
        self.assertEqual(ep.device_id, "charger_1")
        self.assertEqual(ep.protocol, "modbus")
        self.assertEqual(ep.options["host"], "10.0.0.5")
        self.assertEqual(ep.points["power_kw"].scale, 0.1)
        self.assertFalse(ep.points["power_kw"].writable)
        self.assertTrue(ep.points["level"].writable)

    def test_shorthand_point(self):
        ep = endpoint_from_dict({"id": "d", "points": {"soc": 7}})
        self.assertEqual(ep.points["soc"].address, 7)
        self.assertEqual(ep.points["soc"].scale, 1.0)

    def test_default_protocol_is_fake(self):
        ep = endpoint_from_dict({"id": "d"})
        self.assertEqual(ep.protocol, "fake")


class TestRegistry(unittest.TestCase):
    def test_builtins_registered(self):
        protocols = available_protocols()
        for name in ("fake", "modbus", "opcua"):
            self.assertIn(name, protocols)

    def test_unknown_protocol_raises(self):
        ep = DeviceEndpoint(device_id="x", kind="load", protocol="zigbee")
        with self.assertRaises(ValueError):
            create_device_io(ep)

    def test_custom_protocol_pluggable(self):
        created = []

        def factory(endpoint):
            io = FakeDeviceIO()
            created.append(endpoint.device_id)
            return io

        register_protocol("custom-test", factory)
        ep = DeviceEndpoint(device_id="dev9", kind="load", protocol="custom-test")
        io = create_device_io(ep)
        self.assertEqual(created, ["dev9"])
        self.assertIsInstance(io, FakeDeviceIO)

    def test_fake_from_endpoint_initial_values(self):
        ep = DeviceEndpoint(
            device_id="d", kind="load", protocol="fake",
            options={"initial": {"power_kw": 2.5}},
        )
        io = create_device_io(ep)
        self.assertEqual(io.read(), {"power_kw": 2.5})


class TestFakeDeviceIO(unittest.TestCase):
    def test_read_write_and_recording(self):
        io = FakeDeviceIO(values={"shed": 0.0})
        io.write("shed", 1)
        self.assertEqual(io.read()["shed"], 1.0)
        self.assertEqual(io.writes, [("shed", 1.0)])

    def test_offline_read_returns_none(self):
        io = FakeDeviceIO(offline=True)
        self.assertIsNone(io.read())
        with self.assertRaises(ConnectionError):
            io.write("x", 1)


if __name__ == "__main__":
    unittest.main()
