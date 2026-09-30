"""The optional scenario profile (``HARVEST_CONFIG_PROFILE``).

Two promises are tested: with the variable unset nothing changes, and the
shipped Isaac fast-demo profile produces the farm its header describes --
decided by HARVEST's own scheduler, not by anything demo-specific.
"""
from __future__ import annotations

import math
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import yaml

import main

REPO = Path(__file__).resolve().parents[1]
FAST_DEMO = REPO / "examples" / "isaac_fast_demo.yaml"


def _write(path: Path, data) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


class ProfileLoadingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        _write(self.dir / "config.yaml",
               {"farm": {"map": {"width_m": 800, "height_m": 500}},
                "tractors": {"fleet": [{"id": "tractor_1"}, {"id": "tractor_2"}]},
                "task_generation": {"mode": "generated", "num_tasks": 20}})
        _write(self.dir / "demo.yaml",
               {"farm": {"map": {"width_m": 100}},
                "tractors": {"fleet": [{"id": "only"}]},
                "task_generation": {"mode": "static"}})

    def tearDown(self):
        self._tmp.cleanup()

    def _load(self, **env):
        with mock.patch.dict(os.environ, env, clear=False):
            if not env:
                os.environ.pop(main.CONFIG_PROFILE_ENV, None)
            return main.load_yaml_with_local(self.dir / "config.yaml")

    def test_unset_means_the_default_config_untouched(self):
        self.assertEqual(self._load(), main.load_yaml(self.dir / "config.yaml"))

    def test_blank_is_the_same_as_unset(self):
        self.assertEqual(self._load(HARVEST_CONFIG_PROFILE="  "),
                         main.load_yaml(self.dir / "config.yaml"))

    def test_profile_merges_dicts_and_replaces_lists(self):
        cfg = self._load(HARVEST_CONFIG_PROFILE="demo.yaml")
        self.assertEqual(cfg["farm"]["map"], {"width_m": 100, "height_m": 500})
        self.assertEqual(cfg["tractors"]["fleet"], [{"id": "only"}])
        self.assertEqual(cfg["task_generation"],
                         {"mode": "static", "num_tasks": 20})

    def test_profile_is_applied_after_local_overrides(self):
        _write(self.dir / "config.local.yaml",
               {"farm": {"map": {"width_m": 300, "height_m": 200}}})
        cfg = self._load(HARVEST_CONFIG_PROFILE="demo.yaml")
        self.assertEqual(cfg["farm"]["map"], {"width_m": 100, "height_m": 200})

    def test_a_named_but_missing_profile_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            self._load(HARVEST_CONFIG_PROFILE="no_such_profile.yaml")


class IsaacFastDemoProfileTest(unittest.TestCase):
    """The shipped profile, built exactly as the live task service builds it."""

    @classmethod
    def setUpClass(cls):
        base = main.load_yaml(REPO / "config.yaml")
        cls.base = base
        with mock.patch.dict(os.environ,
                             {main.CONFIG_PROFILE_ENV: str(FAST_DEMO)}):
            cls.raw = main.apply_config_profile(base, REPO)
        scenario = main.ScenarioDef(
            name="live", charging_strategy="smart",
            tractor_pv_enabled=True, load_shedding=True, use_marl=False)
        cls.config = main.build_simulation_config(cls.raw, scenario)

    def test_the_default_scenario_is_not_modified(self):
        self.assertEqual(self.base["task_generation"]["mode"], "generated")
        self.assertEqual(self.base, main.load_yaml(REPO / "config.yaml"))

    def test_same_fleet_smaller_farm(self):
        ids = lambda items: [i["id"] for i in items]          # noqa: E731
        self.assertEqual(ids(self.raw["tractors"]["fleet"]),
                         ids(self.base["tractors"]["fleet"]))
        self.assertEqual(ids(self.raw["charging"]["stations"]),
                         ids(self.base["charging"]["stations"]))
        self.assertEqual(
            [t["initial_soc_percent"] for t in self.raw["tractors"]["fleet"]],
            [t["initial_soc_percent"] for t in self.base["tractors"]["fleet"]])
        self.assertLess(self.raw["farm"]["map"]["width_m"],
                        self.base["farm"]["map"]["width_m"])
        self.assertEqual(len(self.config.tasks), 6)

    def test_every_task_lies_on_its_tractors_track_to_the_chargers(self):
        """A tractor is built facing the charging area, and the proxy only
        drives fast when it is pointed at its target -- so each task must lie
        on the straight line from a tractor's start to the chargers."""
        chargers = [c.location for c in self.config.chargers]
        focus = (sum(c[0] for c in chargers) / len(chargers),
                 sum(c[1] for c in chargers) / len(chargers))

        def bearing(a, b):
            return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

        tracks = [bearing(tr.location, focus) for tr in self.config.tractors]
        for task in self.config.tasks:
            self.assertLess(
                min(abs(bearing(task.location, focus) - t) for t in tracks), 0.5)

    def test_harvests_scheduler_keeps_each_tractor_on_its_track(self):
        chargers = [c.location for c in self.config.chargers]
        focus = (sum(c[0] for c in chargers) / len(chargers),
                 sum(c[1] for c in chargers) / len(chargers))

        def bearing(a, b):
            return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

        scheduler = main.Scheduler(self.config)
        scheduler.assign_tasks(datetime(2026, 6, 1, 8, 30))
        self.assertTrue(all(tr.current_task_id is None
                            for tr in self.config.tractors))
        scheduler.assign_tasks(datetime(2026, 6, 1, 9, 0))
        tasks = {t.task_id: t for t in self.config.tasks}
        for tractor in self.config.tractors:
            task = tasks[tractor.current_task_id]
            # Straight ahead: the task's bearing is the tractor's own heading.
            self.assertLess(abs(bearing(tractor.location, task.location)
                                - bearing(tractor.location, focus)), 0.5)
        # Exactly one task each: the second column waits for a free tractor.
        self.assertEqual(
            sum(1 for t in self.config.tasks if t.assigned_tractor_id), 3)


if __name__ == "__main__":
    unittest.main()
