"""Round-trip tests for the fleet JSON codec shared by HTTP/FIWARE/ROS."""
import unittest

from harvest_control.interface import ChargerLevel, Command, CommandAck, CommandType
from harvest_control.sim_backend import SimulationFleetInterface
from harvest_integrations import codec


class TestSnapshotCodec(unittest.TestCase):
    def test_round_trip_preserves_fleet(self):
        snap = SimulationFleetInterface().snapshot()
        wire = codec.snapshot_to_dict(snap)
        back = codec.snapshot_from_dict(wire)

        self.assertEqual(wire["schema"], codec.SCHEMA_VERSION)
        self.assertEqual([t.id for t in back.tractors], [t.id for t in snap.tractors])
        self.assertEqual([c.id for c in back.chargers], [c.id for c in snap.chargers])
        self.assertEqual([l.id for l in back.loads], [l.id for l in snap.loads])
        self.assertEqual(back.grid.tariff, snap.grid.tariff)
        self.assertAlmostEqual(back.grid.grid_cap_kw, snap.grid.grid_cap_kw, places=3)
        self.assertAlmostEqual(
            back.tractors[0].soc_pct, snap.tractors[0].soc_pct, places=2)
        self.assertEqual(back.chargers[0].level, snap.chargers[0].level)

    def test_snapshot_is_json_serialisable(self):
        import json
        snap = SimulationFleetInterface().snapshot()
        json.dumps(codec.snapshot_to_dict(snap))   # must not raise


class TestCommandCodec(unittest.TestCase):
    def test_command_round_trip(self):
        cmd = Command.set_charger_level("CH1", ChargerLevel.HALF)
        back = codec.command_from_dict(codec.command_to_dict(cmd))
        self.assertEqual(back.type, CommandType.SET_CHARGER_LEVEL)
        self.assertEqual(back.target_id, "CH1")
        self.assertEqual(back.value, ChargerLevel.HALF)

    def test_commands_from_payload_forms(self):
        raw = {"type": "shed_load", "target_id": "LD1"}
        for payload in ({"commands": [raw]}, [raw]):
            cmds = codec.commands_from_payload(payload)
            self.assertEqual(len(cmds), 1)
            self.assertEqual(cmds[0].type, CommandType.SHED_LOAD)

    def test_bad_payload_raises(self):
        with self.assertRaises(ValueError):
            codec.commands_from_payload({"commands": "not-a-list"})
        with self.assertRaises(ValueError):
            codec.command_from_dict({"type": "warp_drive", "target_id": "T1"})

    def test_acks_payload(self):
        cmd = Command.shed_load("LD1")
        payload = codec.acks_to_payload([
            CommandAck(cmd, True), CommandAck(cmd, False, "critical load"),
        ])
        self.assertFalse(payload["accepted"])
        self.assertEqual(payload["acks"][1]["reason"], "critical load")
        self.assertEqual(payload["acks"][0]["command"]["type"], "shed_load")


if __name__ == "__main__":
    unittest.main()
