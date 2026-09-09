"""NGSI-LD entity mapping and sync-engine tests (no broker required)."""
import json
import unittest

from harvest_control.sim_backend import SimulationFleetInterface
from harvest_integrations.fiware.entities import (
    COMMAND_ENTITY_ID,
    GRID_ENTITY_ID,
    command_entity,
    parse_command_value,
    snapshot_to_entities,
)
from harvest_integrations.fiware.sync import ContextSync


class TestEntityMapping(unittest.TestCase):
    def setUp(self):
        self.snap = SimulationFleetInterface().snapshot()
        self.entities = snapshot_to_entities(self.snap)

    def _by_id(self, entity_id):
        return next(e for e in self.entities if e["id"] == entity_id)

    def test_entity_count_and_urns(self):
        expected = 1 + len(self.snap.tractors) + len(self.snap.chargers) + len(self.snap.loads)
        self.assertEqual(len(self.entities), expected)
        for e in self.entities:
            self.assertTrue(e["id"].startswith("urn:ngsi-ld:"), e["id"])
            self.assertIn("type", e)

    def test_grid_entity(self):
        grid = self._by_id(GRID_ENTITY_ID)
        self.assertEqual(grid["type"], "FarmEnergySystem")
        self.assertEqual(grid["gridCapKw"]["value"], self.snap.grid.grid_cap_kw)
        self.assertEqual(grid["gridCapKw"]["unitCode"], "KWT")
        self.assertEqual(grid["tariff"]["value"], self.snap.grid.tariff)

    def test_tractor_entity(self):
        t = self.snap.tractors[0]
        entity = self._by_id(f"urn:ngsi-ld:ElectricTractor:{t.id}")
        self.assertEqual(entity["type"], "ElectricTractor")
        self.assertEqual(entity["socPct"]["value"], round(t.soc_pct, 2))
        self.assertEqual(entity["socPct"]["unitCode"], "P1")
        self.assertIn("observedAt", entity["socPct"])
        self.assertEqual(entity["currentTask"]["value"], "none")

    def test_charger_and_load_entities(self):
        c = self.snap.chargers[0]
        charger = self._by_id(f"urn:ngsi-ld:ChargingStation:{c.id}")
        self.assertEqual(charger["level"]["value"], c.level.value)
        l = self.snap.loads[0]
        load = self._by_id(f"urn:ngsi-ld:EnergyConsumer:{l.id}")
        self.assertEqual(load["name"]["value"], l.name)
        self.assertEqual(load["shed"]["value"], l.shed)

    def test_entities_json_serialisable(self):
        json.dumps(self.entities)


class TestCommandParsing(unittest.TestCase):
    def test_valid_json_string(self):
        raw = json.dumps({"nonce": "n1", "commands": [
            {"type": "shed_load", "target_id": "LD1"}]})
        parsed = parse_command_value(raw)
        self.assertEqual(parsed["nonce"], "n1")
        self.assertEqual(len(parsed["commands"]), 1)

    def test_dict_value_accepted(self):
        parsed = parse_command_value({"nonce": "n2", "commands": []})
        self.assertEqual(parsed["nonce"], "n2")

    def test_invalid_values_return_none(self):
        for bad in ("", None, "not json", json.dumps({"no_commands": 1}), 42):
            self.assertIsNone(parse_command_value(bad), bad)

    def test_command_entity_shape(self):
        entity = command_entity()
        self.assertEqual(entity["id"], COMMAND_ENTITY_ID)
        for attr in ("command", "lastNonce", "lastResult"):
            self.assertEqual(entity[attr]["type"], "Property")


# --------------------------------------------------------------------------- #
#  Sync engine with stub broker / stub HARVEST API
# --------------------------------------------------------------------------- #
class _StubBroker:
    def __init__(self):
        self.upserts = []
        self.patches = []
        self.entities = {}

    def upsert_entities(self, entities):
        self.upserts.append(entities)
        for e in entities:
            self.entities[e["id"]] = e

    def get_entity(self, entity_id):
        return self.entities.get(entity_id)

    def patch_attrs(self, entity_id, attrs):
        self.patches.append((entity_id, attrs))


class _StubHarvest:
    def __init__(self):
        from harvest_integrations import codec
        self.iface = SimulationFleetInterface()
        self.codec = codec
        self.submitted = []

    def snapshot(self):
        return self.codec.snapshot_to_dict(self.iface.snapshot())

    def submit_commands(self, payload):
        self.submitted.append(payload)
        return {"accepted": True, "acks": []}


class TestContextSync(unittest.TestCase):
    def setUp(self):
        self.harvest = _StubHarvest()
        self.broker = _StubBroker()
        self.sync = ContextSync(self.harvest, self.broker)

    def test_sync_once_upserts_all_entities(self):
        n = self.sync.sync_once()
        self.assertGreater(n, 0)
        self.assertEqual(len(self.broker.upserts), 1)
        self.assertEqual(len(self.broker.upserts[0]), n)

    def test_command_executed_once_per_nonce(self):
        cmd = {"nonce": "abc", "commands": [{"type": "shed_load", "target_id": "LD1"}]}
        self.assertTrue(self.sync.process_command(cmd))
        self.assertFalse(self.sync.process_command(cmd))      # replay suppressed
        self.assertEqual(len(self.harvest.submitted), 1)
        # Result and nonce written back to the broker.
        entity_id, attrs = self.broker.patches[-1]
        self.assertEqual(entity_id, COMMAND_ENTITY_ID)
        self.assertEqual(attrs["lastNonce"]["value"], "abc")
        self.assertIn("accepted", attrs["lastResult"]["value"])

    def test_new_nonce_executes_again(self):
        self.sync.process_command({"nonce": "n1", "commands": []})
        self.sync.process_command({"nonce": "n2", "commands": []})
        self.assertEqual(len(self.harvest.submitted), 2)

    def test_ensure_command_entity_resumes_nonce(self):
        self.broker.entities[COMMAND_ENTITY_ID] = {
            "id": COMMAND_ENTITY_ID, "lastNonce": "resumed"}
        self.sync.ensure_command_entity()
        self.assertEqual(self.sync.last_nonce, "resumed")
        self.assertFalse(self.sync.process_command(
            {"nonce": "resumed", "commands": []}))


if __name__ == "__main__":
    unittest.main()
