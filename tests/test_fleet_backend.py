"""DeviceFleetInterface mapping tests using the fake protocol (no network)."""
import unittest

from harvest_control.interface import ChargerLevel, Command, CommandType
from harvest_integrations.devices import (
    DeviceEndpoint,
    DeviceFleetInterface,
    FakeDeviceIO,
    register_protocol,
)

_IOS = {}


def _capture_factory(endpoint):
    io = FakeDeviceIO(values={
        k: float(v) for k, v in (endpoint.options.get("initial") or {}).items()
    })
    _IOS[endpoint.device_id] = io
    return io


register_protocol("capture", _capture_factory)


def make_backend():
    _IOS.clear()
    endpoints = [
        DeviceEndpoint("grid", "grid", "capture", options={"initial": {
            "clock_min": 600, "grid_draw_kw": 4.2, "grid_cap_kw": 10.5,
            "pv_kw": 3.0, "tariff_code": 2, "price_eur_per_kwh": 0.20,
        }}),
        DeviceEndpoint("tractor_1", "tractor", "capture", options={"initial": {
            "soc_pct": 55.0, "energy_kwh": 24.6, "available": 1, "charging": 1,
            "discharging": 0, "discharge_kw": 0.0, "pos_x": 50.0, "pos_y": 60.0,
        }}),
        DeviceEndpoint("tractor_2", "tractor", "capture", options={"initial": {
            "soc_pct": 30.0, "energy_kwh": 13.4, "available": 0, "charging": 0,
        }}),
        DeviceEndpoint("charger_1", "charger", "capture", options={"initial": {
            "power_kw": 6.6, "level": 2, "occupied": 1,
        }}),
        DeviceEndpoint("load_1", "load", "capture",
                       options={"name": "cold storage",
                                "initial": {"power_kw": 1.2, "shed": 0}}),
    ]
    return DeviceFleetInterface(endpoints)


class TestSnapshotMapping(unittest.TestCase):
    def setUp(self):
        self.backend = make_backend()

    def test_grid_mapping(self):
        grid = self.backend.snapshot().grid
        self.assertEqual(grid.clock_min, 600)
        self.assertEqual(grid.tariff, "punta")
        self.assertAlmostEqual(grid.price_eur_per_kwh, 0.20)
        self.assertAlmostEqual(grid.pv_kw, 3.0)

    def test_tractor_mapping(self):
        snap = self.backend.snapshot()
        t1 = snap.tractor("tractor_1")
        self.assertAlmostEqual(t1.soc_pct, 55.0)
        self.assertTrue(t1.available)
        self.assertTrue(t1.charging)
        self.assertEqual(t1.position, (50.0, 60.0))
        self.assertIsNone(t1.current_task)          # tasks never come off the wire
        t2 = snap.tractor("tractor_2")
        self.assertFalse(t2.available)

    def test_charger_occupancy_resolves_tractor_id(self):
        charger = self.backend.snapshot().chargers[0]
        self.assertEqual(charger.level, ChargerLevel.FULL)
        self.assertEqual(charger.occupied_by, "tractor_1")   # occupied=1 -> first tractor

    def test_load_mapping(self):
        load = self.backend.snapshot().loads[0]
        self.assertEqual(load.name, "cold storage")
        self.assertFalse(load.shed)
        self.assertAlmostEqual(load.power_kw, 1.2)

    def test_offline_device_marks_tractor_unavailable(self):
        self.backend.snapshot()                     # prime the last-good cache
        _IOS["tractor_1"].offline = True
        t1 = self.backend.snapshot().tractor("tractor_1")
        self.assertFalse(t1.available)
        # Last-good values are retained rather than zeroed.
        self.assertAlmostEqual(t1.soc_pct, 55.0)

    def test_is_real_time(self):
        self.assertTrue(self.backend.is_real_time())


class TestCommandMapping(unittest.TestCase):
    def setUp(self):
        self.backend = make_backend()

    def test_charger_level_write(self):
        acks = self.backend.submit([Command.set_charger_level("charger_1", ChargerLevel.HALF)])
        self.assertTrue(acks[0].accepted)
        self.assertIn(("level", 1.0), _IOS["charger_1"].writes)

    def test_shed_and_restore(self):
        self.backend.submit([Command.shed_load("load_1")])
        self.backend.submit([Command.restore_load("load_1")])
        self.assertEqual(_IOS["load_1"].writes, [("shed", 1.0), ("shed", 0.0)])

    def test_charge_request_and_v2l(self):
        self.backend.submit([Command.request_charge("tractor_1")])
        self.backend.submit([Command.v2l_start("tractor_1", 3.3)])
        self.backend.submit([Command.v2l_stop("tractor_1")])
        self.assertEqual(
            _IOS["tractor_1"].writes,
            [("charge_request", 1.0), ("v2l_kw", 3.3), ("v2l_kw", 0.0)])

    def test_task_commands_rejected_with_reason(self):
        acks = self.backend.submit([Command.assign_task("tractor_1", "task_9")])
        self.assertFalse(acks[0].accepted)
        self.assertIn("not supported", acks[0].reason)

    def test_unknown_target_rejected(self):
        acks = self.backend.submit([Command.shed_load("no_such_load")])
        self.assertFalse(acks[0].accepted)

    def test_write_failure_rejected_not_raised(self):
        _IOS["load_1"].offline = True
        acks = self.backend.submit([Command.shed_load("load_1")])
        self.assertFalse(acks[0].accepted)


if __name__ == "__main__":
    unittest.main()
