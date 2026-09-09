"""HTTP tests for the live-fleet endpoints added to server.py.

Runs the real Handler on an ephemeral port with the default ``sim`` backend
(the reference simulation: tractors T1..T3, chargers CH1/CH2, loads LD1..LD4).
Also regression-guards that the pre-existing endpoints still respond.
"""
import json
import threading
import unittest
import urllib.request
from http.server import HTTPServer

import server


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


class TestFleetApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = HTTPServer(("127.0.0.1", 0), server.Handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.thread.join(timeout=5)
        if server._FLEET_RUNTIME is not None:
            server._FLEET_RUNTIME.close()
            server._FLEET_RUNTIME = None

    def test_health_still_works(self):
        status, body = _get(f"{self.base}/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_snapshot_endpoint(self):
        status, body = _get(f"{self.base}/api/fleet/snapshot")
        self.assertEqual(status, 200)
        self.assertEqual(body["schema"], "harvest-fleet/1.0")
        self.assertEqual(len(body["tractors"]), 3)
        self.assertEqual(len(body["chargers"]), 2)
        self.assertGreater(body["grid"]["grid_cap_kw"], 0)

    def test_status_endpoint(self):
        status, body = _get(f"{self.base}/api/fleet/status")
        self.assertEqual(status, 200)
        self.assertEqual(body["backend"], "sim")
        self.assertFalse(body["real_time"])

    def test_command_round_trip(self):
        status, body = _post(f"{self.base}/api/fleet/command", {
            "commands": [{"type": "shed_load", "target_id": "LD1"}]})
        self.assertEqual(status, 200)
        self.assertTrue(body["accepted"], body)

        _, snap = _get(f"{self.base}/api/fleet/snapshot")
        ld1 = next(l for l in snap["loads"] if l["id"] == "LD1")
        self.assertTrue(ld1["shed"])

        _post(f"{self.base}/api/fleet/command", {
            "commands": [{"type": "restore_load", "target_id": "LD1"}]})

    def test_rejected_command_reported_in_ack(self):
        # LD3 (barn_doors) is critical in the reference sim -- shed refused.
        status, body = _post(f"{self.base}/api/fleet/command", {
            "commands": [{"type": "shed_load", "target_id": "LD3"}]})
        self.assertEqual(status, 200)
        self.assertFalse(body["accepted"])
        self.assertIn("critical", body["acks"][0]["reason"])

    def test_invalid_payloads_rejected(self):
        status, body = _post(f"{self.base}/api/fleet/command",
                             {"commands": [{"type": "warp", "target_id": "T1"}]})
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        status, _ = _post(f"{self.base}/api/fleet/command", {"commands": "x"})
        self.assertEqual(status, 400)

    def test_diagnostics_endpoint(self):
        status, body = _get(f"{self.base}/api/diagnostics")
        self.assertEqual(status, 200)
        names = [s["name"] for s in body["services"]]
        for expected in ("HARVEST API", "Fleet backend", "Orion-LD broker",
                         "ROS 2 bridge"):
            self.assertIn(expected, names)
        self.assertGreater(len(body["devices"]), 0)
        self.assertIn(body["demo"]["state"],
                      ("idle", "running", "passed", "failed"))

    def test_demo_status_endpoint(self):
        status, body = _get(f"{self.base}/api/diagnostics/demo")
        self.assertEqual(status, 200)
        self.assertIn("state", body)

    def test_integrations_status_roundtrip(self):
        # The generic push channel the isaac bridge uses: a daemon POSTs its
        # status blob; readers see it with an age; diagnostics interprets it.
        status, body = _post(f"{self.base}/api/integrations/status", {
            "client": "isaac-bridge",
            "status": {"simulator": {"connected": True, "kind": "stub",
                                     "state": "running", "entities_synced": 3,
                                     "telemetry_age_s": 0.5,
                                     "ever_connected": True,
                                     "scene_acknowledged": True}}})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

        status, body = _get(f"{self.base}/api/integrations/status")
        self.assertEqual(status, 200)
        info = body["clients"]["isaac-bridge"]
        self.assertLess(info["age_s"], 5.0)
        self.assertEqual(info["status"]["simulator"]["kind"], "stub")

        # And the diagnostics view reflects it as deliberate simulation.
        status, diag_body = _get(f"{self.base}/api/diagnostics")
        row = next(s for s in diag_body["services"] if s["name"] == "Isaac Sim")
        self.assertEqual(row["state"], "simulated")

    def test_integrations_status_requires_client_label(self):
        status, body = _post(f"{self.base}/api/integrations/status",
                             {"status": {}})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_simulation_advances_in_real_time(self):
        import time
        _, before = _get(f"{self.base}/api/fleet/snapshot")
        time.sleep(2.5)
        _, after = _get(f"{self.base}/api/fleet/snapshot")
        self.assertGreater(after["grid"]["clock_min"], before["grid"]["clock_min"])


if __name__ == "__main__":
    unittest.main()
