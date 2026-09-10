"""Isaac Sim integration contract + kinematics tests (no ROS, no Isaac).

The wire contract and the motion core are pure stdlib by design, so the whole
HARVEST->simulator sync logic — the code both the isaac-demo stub and the real
Isaac app run — is exercised here without a GPU, rclpy or Docker.
"""
import pathlib
import sys
import unittest

from harvest_integrations.simulators.isaac import contract
from harvest_integrations.simulators.isaac.motion import FieldKinematics

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "harvest_ros"))
from harvest_ros import topics  # noqa: E402

CFG = {
    "tractors": {"fleet": [
        {"id": "tractor_1", "initial_location": {"x": 10, "y": 10}},
        {"id": "tractor_2", "initial_location": {"x": 20, "y": 10}},
        {"id": "tractor_off", "enabled": False},
    ]},
    "charging": {"stations": [
        {"id": "charger_1", "location": {"x": 40, "y": 40}},
    ]},
}


def _snapshot(occupied_by=None, charging=False):
    return {
        "schema": "harvest-fleet/1.0",
        "tractors": [
            {"id": "tractor_1", "soc_pct": 30.0, "charging": charging,
             "position": [10, 10]},
            {"id": "tractor_2", "soc_pct": 80.0, "charging": False,
             "position": [22, 11]},
        ],
        "chargers": [
            {"id": "charger_1", "power_kw": 6.6 if occupied_by else 0.0,
             "occupied_by": occupied_by},
        ],
    }


class TestTopicContract(unittest.TestCase):
    def test_sim_topics_match_ros_package(self):
        # One string per end (Isaac's interpreter cannot import the colcon
        # package); this is the drift guard.
        self.assertEqual(contract.SIM_COMMAND_TOPIC, topics.SIM_COMMAND)
        self.assertEqual(contract.SIM_TELEMETRY_TOPIC, topics.SIM_TELEMETRY)

    def test_no_status_leaf(self):
        # Orion-LD's DDS bridge drops '/status'-leaf topics (WISEPACK finding).
        for name in (contract.SIM_COMMAND_TOPIC, contract.SIM_TELEMETRY_TOPIC):
            self.assertTrue(name.startswith("/harvest/"), name)
            self.assertFalse(name.endswith("/status"), name)


class TestSchema(unittest.TestCase):
    def test_compatible_versions(self):
        self.assertTrue(contract.schema_compatible("harvest-sim/1.0"))
        self.assertTrue(contract.schema_compatible("harvest-sim/1.7"))

    def test_refused_versions(self):
        for bad in ("harvest-sim/2.0", "harvest-fleet/1.0", "", None, 7):
            self.assertFalse(contract.schema_compatible(bad), bad)

    def test_parse_refuses_bad_messages(self):
        self.assertIsNone(contract.parse_message("not json"))
        self.assertIsNone(contract.parse_message('{"schema": "harvest-sim/2.0"}'))
        self.assertIsNotNone(contract.parse_message('{"schema": "harvest-sim/1.0"}'))


class TestSceneDerivation(unittest.TestCase):
    def test_scene_from_config(self):
        scene = contract.scene_from_config(CFG)
        by_id = {e["id"]: e for e in scene["entities"]}
        self.assertEqual(set(by_id), {"tractor_1", "tractor_2", "charger_1"},
                         "disabled tractors must not enter the scene")
        self.assertEqual(by_id["tractor_1"]["home"], [10.0, 10.0])
        self.assertEqual(by_id["charger_1"]["pose"], [40.0, 40.0])
        self.assertGreater(scene["field"]["width"], 40.0)

    def test_fingerprint_deterministic_and_sensitive(self):
        a = contract.scene_fingerprint(contract.scene_from_config(CFG))
        b = contract.scene_fingerprint(contract.scene_from_config(CFG))
        self.assertEqual(a, b)
        moved = contract.scene_from_config(CFG)
        moved["entities"][0]["home"] = [11.0, 10.0]
        self.assertNotEqual(a, contract.scene_fingerprint(moved))

    def test_goals_idle_tractor_targets_its_position(self):
        scene = contract.scene_from_config(CFG)
        derived = contract.goals_from_snapshot(scene, _snapshot())
        self.assertEqual(derived["goals"]["tractor_2"]["target"], [22.0, 11.0])
        self.assertIsNone(derived["goals"]["tractor_2"]["docked_charger"])
        self.assertFalse(derived["chargers"]["charger_1"]["active"])

    def test_goals_assigned_tractor_targets_charger(self):
        # HARVEST's semantic model docks instantly; the physical goal is the
        # charger's field pose — that's the behaviour Isaac executes.
        scene = contract.scene_from_config(CFG)
        derived = contract.goals_from_snapshot(
            scene, _snapshot(occupied_by="tractor_1", charging=True))
        goal = derived["goals"]["tractor_1"]
        self.assertEqual(goal["target"], [40.0, 40.0])
        self.assertEqual(goal["docked_charger"], "charger_1")
        self.assertTrue(goal["charging"])
        self.assertTrue(derived["chargers"]["charger_1"]["active"])


class TestFieldKinematics(unittest.TestCase):
    def _command(self, occupied_by=None, charging=False, include_scene=True):
        scene = contract.scene_from_config(CFG)
        derived = contract.goals_from_snapshot(
            scene, _snapshot(occupied_by=occupied_by, charging=charging))
        return contract.command_message(scene, derived, include_scene)

    def test_scene_applied_once_and_acknowledged(self):
        kin = FieldKinematics()
        cmd = self._command()
        self.assertTrue(kin.apply_command(cmd))
        self.assertFalse(kin.apply_command(cmd), "same fingerprint: no rebuild")
        self.assertEqual(kin.scene_fingerprint, cmd["scene_fingerprint"])
        self.assertEqual(len(kin.entities), 3)

    def test_drive_to_charger_and_dock(self):
        kin = FieldKinematics(speed_mps=5.0)
        kin.apply_command(self._command(occupied_by="tractor_1", charging=True))
        tractor = kin.entities["tractor_1"]
        self.assertFalse(kin._docked(tractor), "starts 42m away, not docked")
        for _ in range(200):                       # 20 simulated seconds
            kin.step(0.1)
        telem = kin.telemetry_entities()["tractor_1"]
        self.assertTrue(telem["docked"])
        self.assertFalse(telem["moving"])
        self.assertAlmostEqual(telem["pose"][0], 40.0, delta=0.1)
        self.assertEqual(telem["distance_to_target_m"], 0.0)

    def test_partial_drive_reports_distance(self):
        kin = FieldKinematics(speed_mps=5.0)
        kin.apply_command(self._command(occupied_by="tractor_1"))
        kin.step(1.0)                              # 5 m of a ~42 m trip
        telem = kin.telemetry_entities()["tractor_1"]
        self.assertTrue(telem["moving"])
        self.assertFalse(telem["docked"])
        self.assertGreater(telem["distance_to_target_m"], 30.0)

    def test_scene_refresh_does_not_teleport(self):
        kin = FieldKinematics(speed_mps=5.0)
        kin.apply_command(self._command(occupied_by="tractor_1"))
        kin.step(2.0)
        pose_before = list(kin.entities["tractor_1"].pose)
        moved = contract.scene_from_config(CFG)
        moved["field"]["width"] += 1.0             # different fingerprint
        derived = contract.goals_from_snapshot(moved, _snapshot())
        kin.apply_command(contract.command_message(moved, derived, True))
        self.assertEqual(kin.entities["tractor_1"].pose, pose_before)

    def test_telemetry_message_shape(self):
        kin = FieldKinematics()
        kin.apply_command(self._command())
        msg = contract.telemetry_message(
            simulator={"kind": "stub", "state": "running"},
            entities=kin.telemetry_entities(),
            fingerprint=kin.scene_fingerprint, stats=kin.stats())
        self.assertEqual(msg["schema"], contract.SCHEMA_VERSION)
        self.assertEqual(msg["stats"]["entities_synced"], 3)
        self.assertIn("charger_1", msg["entities"])
        parsed = contract.parse_message(__import__("json").dumps(msg))
        self.assertIsNotNone(parsed)



class TestPhysicalStateVocabulary(unittest.TestCase):
    """One vocabulary for both simulators, defined in the contract.

    Diagnostics, the NGSI-LD mirror and the Isaac backend all render this word,
    so a second definition anywhere would let a stand-in run and a real run
    describe the same situation differently.
    """

    def test_words_are_the_declared_set(self):
        for moving in (False, True):
            for docked in (False, True):
                for charging in (False, True):
                    self.assertIn(
                        contract.physical_state(moving=moving, docked=docked,
                                                charging=charging),
                        contract.PHYSICAL_STATES)

    def test_charging_requires_being_docked(self):
        # HARVEST may report a tractor as charging the instant it assigns a
        # charger; physically it is still driving there, and saying "charging"
        # then would hide exactly the thing Isaac exists to show.
        self.assertEqual(
            contract.physical_state(moving=True, docked=False, charging=True),
            "moving")
        self.assertEqual(
            contract.physical_state(moving=False, docked=True, charging=True),
            "charging")

    def test_docked_beats_parked(self):
        self.assertEqual(
            contract.physical_state(moving=False, docked=True, charging=False),
            "docked")
        self.assertEqual(
            contract.physical_state(moving=False, docked=False, charging=False),
            "parked")

    def test_stand_in_telemetry_carries_the_state(self):
        """The GPU-free stub must report it too, or isaac-demo shows '?'."""
        kin = FieldKinematics()
        scene = contract.scene_from_config(CFG)
        kin.apply_command(contract.command_message(
            scene, contract.goals_from_snapshot(scene, _snapshot()),
            include_scene=True))
        rows = kin.telemetry_entities()
        for eid, row in rows.items():
            if row["kind"] == "tractor":
                self.assertIn(row["physical_state"], contract.PHYSICAL_STATES,
                              f"{eid} reported {row.get('physical_state')!r}")

if __name__ == "__main__":
    unittest.main()
