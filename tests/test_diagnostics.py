"""Diagnostics collector + cross-protocol demo tests (no Docker required).

The FIWARE-dependent paths are pointed at a dead port so results do not depend
on whether a broker happens to be running on this machine.
"""
import os
import time
import unittest

from harvest_integrations import diagnostics as diag
from harvest_integrations.runtime import FleetRuntime

DEAD_BROKER = "http://127.0.0.1:59999"


class _EnvGuard(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("ORION_URL", "HARVEST_MONGO_HOST")}
        os.environ["ORION_URL"] = DEAD_BROKER
        os.environ["HARVEST_MONGO_HOST"] = "127.0.0.1:59998"
        diag._cache._data.clear()
        self.runtime = FleetRuntime({})
        self.clients = diag.ClientRegistry()

    def tearDown(self):
        self.runtime.close()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        diag._cache._data.clear()


class TestClientRegistry(unittest.TestCase):
    def test_mark_and_age(self):
        reg = diag.ClientRegistry()
        self.assertIsNone(reg.age_s("ros2-bridge"))
        reg.mark("ros2-bridge")
        age = reg.age_s("ros2-bridge")
        self.assertIsNotNone(age)
        self.assertLess(age, 2.0)
        reg.mark(None)                       # no header — must be a no-op
        self.assertIsNone(reg.age_s("fiware-sync"))


class TestCollector(_EnvGuard):
    def _services(self):
        doc = diag.collect(self.runtime, self.clients)
        return {s["name"]: s for s in doc["services"]}, doc

    def test_service_states_sim_backend_nothing_running(self):
        services, doc = self._services()
        self.assertEqual(services["HARVEST API"]["state"], diag.HEALTHY)
        # Deliberate simulation reads as simulated, never as failure.
        self.assertEqual(services["Fleet backend"]["state"], diag.SIMULATED)
        # Optional components that were never started read as inactive.
        for name in ("Modbus adapter", "OPC-UA adapter", "Orion-LD broker",
                     "MongoDB", "FIWARE sync", "ROS 2 bridge", "Isaac Sim"):
            self.assertEqual(services[name]["state"], diag.INACTIVE, name)
        # Device rows exist even in sim mode, tagged with protocol 'sim'.
        self.assertGreater(len(doc["devices"]), 0)
        self.assertTrue(all(d["protocol"] == "sim" for d in doc["devices"]))
        self.assertTrue(all(d["reachable"] for d in doc["devices"]))

    def test_ros_bridge_heartbeat_flips_state(self):
        self.clients.mark("ros2-bridge")
        services, _ = self._services()
        self.assertEqual(services["ROS 2 bridge"]["state"], diag.HEALTHY)

    def test_inactive_details_point_to_launcher_modes(self):
        services, _ = self._services()
        self.assertIn("run_harvest_dashboard.sh full",
                      services["ROS 2 bridge"]["detail"])
        self.assertIn("run_harvest_dashboard.sh devices",
                      services["Modbus adapter"]["detail"])

    def test_document_shape(self):
        _, doc = self._services()
        self.assertIn("ts", doc)
        self.assertEqual(doc["fleet"]["backend"], "sim")
        self.assertIn("state", doc["demo"])
        row = doc["devices"][0]
        for key in ("id", "kind", "protocol", "endpoint", "reachable",
                    "last_read_ts", "values"):
            self.assertIn(key, row)


class TestIsaacRow(_EnvGuard):
    """The Isaac Sim service row must stay honest in every state."""

    def _row(self):
        doc = diag.collect(self.runtime, self.clients)
        return next(s for s in doc["services"] if s["name"] == "Isaac Sim")

    def _report(self, **sim):
        self.clients.report("isaac-bridge",
                            {"scene_ready": True, "simulator": sim})

    def test_never_enabled_is_inactive_with_instructions(self):
        row = self._row()
        self.assertEqual(row["state"], diag.INACTIVE)
        self.assertIn("run_harvest_dashboard.sh isaac", row["detail"])

    def test_bridge_waiting_for_simulator_is_not_an_error(self):
        self._report(connected=False, ever_connected=False)
        row = self._row()
        self.assertEqual(row["state"], diag.INACTIVE)
        self.assertIn("waiting for a simulator", row["detail"])
        self.assertIn("run_isaac_sim.sh", row["detail"])

    def test_stub_reads_as_simulated_never_healthy(self):
        self._report(connected=True, ever_connected=True, kind="stub",
                     state="running", entities_synced=5, telemetry_age_s=1.0,
                     scene_acknowledged=True)
        row = self._row()
        self.assertEqual(row["state"], diag.SIMULATED)
        self.assertIn("5 entities synced", row["detail"])

    def test_real_isaac_reads_as_healthy(self):
        self._report(connected=True, ever_connected=True, kind="isaac",
                     state="running", entities_synced=5, telemetry_age_s=2.0,
                     scene_acknowledged=True)
        row = self._row()
        self.assertEqual(row["state"], diag.HEALTHY)
        self.assertIn("isaac connected", row["detail"])
        self.assertEqual(row["simulator"]["entities_synced"], 5)

    def test_lost_simulator_is_failed(self):
        self._report(connected=False, ever_connected=True, telemetry_age_s=45.0)
        row = self._row()
        self.assertEqual(row["state"], diag.FAILED)
        self.assertIn("connection lost", row["detail"])

    def test_simulator_error_is_failed(self):
        self._report(connected=True, ever_connected=True, kind="isaac",
                     state="error", entities_synced=5, telemetry_age_s=1.0,
                     scene_acknowledged=True)
        self.assertEqual(self._row()["state"], diag.FAILED)


class TestDeviceBackendRows(unittest.TestCase):
    def test_diagnostics_reflect_reachability(self):
        from harvest_integrations.devices import (
            DeviceEndpoint, DeviceFleetInterface, FakeDeviceIO, register_protocol)
        ios = {}

        def factory(ep):
            ios[ep.device_id] = FakeDeviceIO(values={"power_kw": 1.0})
            return ios[ep.device_id]

        register_protocol("diagtest", factory)
        fleet = DeviceFleetInterface([
            DeviceEndpoint("load_a", "load", "diagtest"),
            DeviceEndpoint("load_b", "load", "diagtest"),
        ])
        rows = {r["id"]: r for r in fleet.diagnostics()}
        self.assertIsNone(rows["load_a"]["reachable"])   # nothing read yet

        fleet.snapshot()
        ios["load_b"].offline = True
        fleet.snapshot()
        rows = {r["id"]: r for r in fleet.diagnostics()}
        self.assertTrue(rows["load_a"]["reachable"])
        self.assertFalse(rows["load_b"]["reachable"])
        self.assertEqual(rows["load_b"]["values"], {"power_kw": 1.0})  # last-good
        self.assertEqual(rows["load_a"]["protocol"], "diagtest")


class TestCrossProtocolDemo(_EnvGuard):
    def test_demo_passes_on_sim_backend(self):
        demo = diag.CrossProtocolDemo()
        self.assertTrue(demo.start(self.runtime))
        self.assertFalse(demo.start(self.runtime))       # no concurrent runs
        deadline = time.time() + 60
        while demo.status()["state"] == "running" and time.time() < deadline:
            time.sleep(0.5)
        status = demo.status()
        self.assertEqual(status["state"], "passed", status)
        titles = [s["title"] for s in status["steps"]]
        self.assertEqual(titles[:3],
                         ["Baseline read", "Charge command", "Cross-protocol effect"])
        by_title = {s["title"]: s for s in status["steps"]}
        # Broker is dead in this test: reflection must SKIP, not fail.
        self.assertEqual(by_title["FIWARE reflection"]["status"], "skip")
        self.assertEqual(by_title["Release"]["status"], "ok")
        # A second run is allowed once the first finished.
        self.assertTrue(demo.start(self.runtime))
        while demo.status()["state"] == "running" and time.time() < deadline:
            time.sleep(0.5)
        self.assertEqual(demo.status()["state"], "passed")


if __name__ == "__main__":
    unittest.main()
