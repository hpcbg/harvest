"""Isaac robot-model registry, streaming configuration and Diagnostics facts.

None of this needs Isaac Sim, a GPU or ROS: the registry, the streaming
configuration and the Diagnostics rendering are deliberately pure stdlib
(+PyYAML) so the parts that decide WHICH robot represents a tractor and HOW an
operator watches it can be tested by the ordinary suite.  Everything that needs
PhysX lives behind ``TractorRobot.build``/``attach`` and is covered by
``./scripts/run_isaac_sim.sh --self-test-drive`` instead.
"""
import os
import unittest
from unittest import mock

from harvest_integrations import diagnostics
from harvest_integrations.simulators.isaac import robots, streaming


class TestRegistry(unittest.TestCase):
    """The tracked registry is the only robot list, so it must stay valid."""

    def test_shipped_registry_loads_and_validates(self):
        registry = robots.load_registry()
        self.assertTrue(registry["schema"].startswith("harvest-isaac-robots/"))
        self.assertIn(registry["default_model"], registry["models"])

    def test_default_model_is_enabled_and_procedural(self):
        # The default must work on a machine with nothing but Isaac Sim: no
        # asset download, no Nucleus, no network.
        model = robots.resolve_model()
        self.assertTrue(model.enabled)
        self.assertEqual(model.provider, "procedural")
        self.assertTrue(model.left_joints and model.right_joints)

    def test_env_selects_a_model(self):
        with mock.patch.dict(os.environ, {robots.MODEL_ENV: "zetrabot_usd"}):
            # Disabled in the shipped registry, and refused BY NAME rather than
            # falling back to the default -- a silent fallback would run the
            # proxy while the operator believed they had the real model.
            with self.assertRaises(robots.RobotModelError) as caught:
                robots.resolve_model()
            self.assertIn("zetrabot_usd", str(caught.exception))
            self.assertIn("disabled", str(caught.exception))

    def test_unknown_model_names_what_exists(self):
        with self.assertRaises(robots.RobotModelError) as caught:
            robots.resolve_model("no_such_tractor")
        message = str(caught.exception)
        self.assertIn("no_such_tractor", message)
        self.assertIn("proxy_utility_tractor", message)

    def test_zetrabot_seam_exists(self):
        """The registry must keep a `usd` entry for the real model.

        This is the seam the whole isolation claim rests on: a real ZETRABOT USD
        replaces the proxy by editing this file, with no HARVEST change.  A
        refactor that quietly drops it would remove the seam and nothing else
        would fail.
        """
        registry = robots.load_registry()
        model = registry["models"].get("zetrabot_usd")
        self.assertIsNotNone(model, "the zetrabot_usd registry entry is gone")
        self.assertEqual(model.provider, "usd")
        self.assertTrue(model.usd.get("asset_path_candidates"))

    def test_describe_carries_no_paths(self):
        # What Diagnostics and FIWARE see: identity and size, never a host path.
        described = robots.resolve_model().describe()
        self.assertNotIn("usd", described)
        for value in described.values():
            self.assertNotIn("/data/", str(value))

    def test_bad_registry_is_refused_not_guessed(self):
        import tempfile
        import pathlib

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bad.yaml"
            path.write_text(
                "schema: harvest-isaac-robots/1.0\n"
                "default_model: ghost\n"
                "models:\n"
                "  - id: real\n    provider: procedural\n"
                "    drive: {kind: skid_steer, left_joints: [l], right_joints: [r]}\n")
            with self.assertRaises(robots.RobotModelError):
                robots.load_registry(path)


class TestDriveMixing(unittest.TestCase):
    """Skid-steer mixing, checked without PhysX.

    ``_apply`` needs an articulation, so the arithmetic is verified directly:
    it is the one piece of the vehicle where a sign or a units error produces a
    tractor that drives away from its charger.
    """

    def setUp(self):
        self.model = robots.resolve_model()

    def _wheels(self, linear, yaw_dps):
        import math
        radius = self.model.wheel_radius
        yaw = math.radians(yaw_dps)
        return ((linear - yaw * self.model.track / 2.0) / radius,
                (linear + yaw * self.model.track / 2.0) / radius)

    def test_straight_ahead_turns_both_sides_equally(self):
        left, right = self._wheels(self.model.max_speed_mps, 0.0)
        self.assertAlmostEqual(left, right)
        # rad/s, SI: the tensor articulation API takes radians for revolute DOFs.
        self.assertAlmostEqual(
            left, self.model.max_speed_mps / self.model.wheel_radius, places=6)

    def test_turning_left_slows_the_left_side(self):
        left, right = self._wheels(1.0, +30.0)
        self.assertLess(left, right)

    def test_spin_on_the_spot_is_symmetric(self):
        left, right = self._wheels(0.0, 45.0)
        self.assertAlmostEqual(left, -right)


class TestSteering(unittest.TestCase):
    """The controller's decisions, with the physics stubbed out."""

    def setUp(self):
        self.robot = robots.TractorRobot("tractor_1", robots.resolve_model())
        # No articulation: drive_towards must still DECIDE, and _apply must be a
        # no-op rather than an exception.  This is the state during a rebuild.
        self.robot._pose = (0.0, 0.0)
        self.robot._heading_deg = 0.0

    def test_no_target_is_idle(self):
        command = self.robot.drive_towards(None, arrival_radius_m=1.5,
                                          speed_mps=3.0)
        self.assertEqual(command.reason, "no target")
        self.assertEqual(command.linear_mps, 0.0)

    def test_inside_the_arrival_radius_it_brakes(self):
        command = self.robot.drive_towards((1.0, 0.0), arrival_radius_m=1.5,
                                          speed_mps=3.0)
        self.assertEqual(command.reason, "arrived")
        self.assertEqual(command.linear_mps, 0.0)
        self.assertEqual(command.yaw_rate_dps, 0.0)

    def test_badly_misaligned_it_turns_before_driving(self):
        command = self.robot.drive_towards((0.0, 30.0), arrival_radius_m=1.5,
                                          speed_mps=3.0)
        self.assertEqual(command.reason, "turning")
        self.assertEqual(command.linear_mps, 0.0)
        self.assertGreater(command.yaw_rate_dps, 0.0)

    def test_pointed_at_the_target_it_drives(self):
        command = self.robot.drive_towards((30.0, 0.0), arrival_radius_m=1.5,
                                          speed_mps=3.0)
        self.assertEqual(command.reason, "driving")
        self.assertGreater(command.linear_mps, 0.0)
        self.assertLessEqual(command.linear_mps, self.model_speed())
        self.assertAlmostEqual(command.yaw_rate_dps, 0.0, places=6)

    def test_speed_is_capped_by_the_model(self):
        command = self.robot.drive_towards((300.0, 0.0), arrival_radius_m=1.5,
                                          speed_mps=99.0)
        self.assertLessEqual(command.linear_mps, self.model_speed())

    def test_yaw_rate_is_capped_by_the_model(self):
        self.robot._heading_deg = 180.0
        command = self.robot.drive_towards((30.0, 0.1), arrival_radius_m=1.5,
                                          speed_mps=3.0)
        self.assertLessEqual(abs(command.yaw_rate_dps),
                             self.robot.model.max_yaw_rate_dps)

    def model_speed(self):
        return self.robot.model.max_speed_mps


class TestStreamingConfig(unittest.TestCase):
    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(streaming.StreamingConfig.from_env().enabled)

    def test_webrtc_env_matches_the_wisepack_convention(self):
        with mock.patch.dict(os.environ, {
                "HARVEST_ISAAC_STREAMING": "1",
                "HARVEST_ISAAC_SIGNAL_PORT": "49111"}, clear=True):
            config = streaming.StreamingConfig.from_env()
            self.assertTrue(config.enabled)
            self.assertEqual(config.signal_port, 49111)
            self.assertEqual(config.resolved_viewer_url(),
                             "http://127.0.0.1:49111")

    def test_advertised_host_is_not_the_bind_address(self):
        """The distinction is load-bearing, so it is asserted.

        Kit binds every interface whatever HARVEST advertises; a UI that showed
        the advertised address AS the bind address would give an operator a
        false sense of who can reach an unauthenticated stream.
        """
        with mock.patch.dict(os.environ, {
                "HARVEST_ISAAC_STREAMING": "1",
                "HARVEST_ISAAC_STREAM_HOST": "10.0.0.5"}, clear=True):
            config = streaming.StreamingConfig.from_env()
            self.assertEqual(config.host, "10.0.0.5")
            self.assertEqual(config.bind_address, "0.0.0.0")
            self.assertTrue(config.host_explicit)
            self.assertIn("10.0.0.5", config.to_dict()["advertised_host"])
            self.assertNotEqual(config.to_dict()["advertised_host"],
                                config.to_dict()["bind_address"])

    def test_explicit_url_wins(self):
        with mock.patch.dict(os.environ, {
                "HARVEST_ISAAC_STREAMING": "1",
                "HARVEST_ISAAC_STREAM_URL": "https://proxy.example/isaac"},
                clear=True):
            self.assertEqual(streaming.StreamingConfig.from_env().resolved_viewer_url(),
                             "https://proxy.example/isaac")

    def test_identical_ports_are_refused(self):
        with mock.patch.dict(os.environ, {
                "HARVEST_ISAAC_STREAMING": "1",
                "HARVEST_ISAAC_SIGNAL_PORT": "47998",
                "HARVEST_ISAAC_STREAM_PORT": "47998"}, clear=True):
            with self.assertRaises(ValueError):
                streaming.StreamingConfig.from_env()

    def test_launch_config_pairs_headless_with_visible_ui(self):
        # The combination from NVIDIA's own livestream example: a hidden UI
        # streams an empty viewport.
        config = streaming.StreamingConfig(enabled=True)
        launch = streaming.launch_config(config, headless=True)
        self.assertTrue(launch["headless"])
        self.assertFalse(launch["hide_ui"])
        # Without streaming the extra window settings must not appear at all.
        self.assertNotIn("hide_ui", streaming.launch_config(
            streaming.StreamingConfig(enabled=False), headless=True))


class TestIsaacDiagnosticsFacts(unittest.TestCase):
    """The Diagnostics lines, which must report and never infer."""

    SIM = {
        "kind": "isaac", "state": "running", "telemetry_age_s": 0.4,
        "entities_synced": 5, "tractors_synced": 3, "tractors_driveable": 3,
        "sim_time_s": 128.0, "physics": "PhysX articulation",
        "robot_model": {"id": "proxy_utility_tractor", "provider": "procedural",
                        "length_m": 2.6, "width_m": 1.35},
        "visualization": {"state": "serving", "enabled": True,
                          "viewer_url": "http://127.0.0.1:49100",
                          "signal_port": 49100, "stream_port": 47998},
        "entities": {
            "tractor_3": {"kind": "tractor", "pose": [61.2, 44.9],
                          "heading_deg": -161.0, "speed_mps": 2.4,
                          "physical_state": "moving",
                          "docked_charger": "charger_1", "docked": False,
                          "distance_to_charger_m": 12.4},
            "charger_1": {"kind": "charger", "pose": [40.0, 40.0],
                          "active": True, "power_kw": 6.6},
        },
    }
    ROS = {"domain_id": "42", "rmw": "rmw_fastrtps_cpp", "transport": "UDPv4"}

    def facts(self, sim=None, ros=None):
        return diagnostics._isaac_facts(sim if sim is not None else self.SIM,
                                        ros if ros is not None else self.ROS)

    def test_reports_webrtc_endpoint_and_client(self):
        joined = "\n".join(self.facts())
        self.assertIn("http://127.0.0.1:49100", joined)
        self.assertIn("WebRTC Streaming Client", joined)
        self.assertIn("47998/UDP", joined)

    def test_reports_dds_domain_and_transport(self):
        joined = "\n".join(self.facts())
        self.assertIn("domain 42", joined)
        self.assertIn("UDPv4", joined)

    def test_reports_tractor_pose_state_and_docking(self):
        line = [f for f in self.facts() if f.startswith("tractor_3:")][0]
        self.assertIn("moving", line)
        self.assertIn("61.2", line)
        self.assertIn("charger_1", line)
        self.assertIn("approaching", line)

    def test_docked_tractor_says_docked(self):
        sim = dict(self.SIM)
        sim["entities"] = dict(self.SIM["entities"])
        sim["entities"]["tractor_3"] = {**self.SIM["entities"]["tractor_3"],
                                        "docked": True,
                                        "physical_state": "charging"}
        line = [f for f in self.facts(sim) if f.startswith("tractor_3:")][0]
        self.assertIn("docked", line)
        self.assertIn("charging", line)

    def test_failed_stream_is_reported_as_failed(self):
        sim = {**self.SIM, "visualization": {"state": "failed",
                                             "detail": "port 49100 busy"}}
        joined = "\n".join(self.facts(sim))
        self.assertIn("FAILED", joined)
        self.assertIn("port 49100 busy", joined)

    def test_chargers_do_not_get_their_own_lines(self):
        # An operational view: three tractors, not five entities.
        self.assertFalse([f for f in self.facts() if f.startswith("charger_1:")])

    def test_stays_short(self):
        # The page must not become a simulation monitor.
        self.assertLessEqual(len(self.facts()), 12)

    def test_empty_simulator_view_produces_no_noise(self):
        self.assertEqual(diagnostics._isaac_facts({}, {}), [])


if __name__ == "__main__":
    unittest.main()
