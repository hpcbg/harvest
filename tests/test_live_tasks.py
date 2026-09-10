"""HARVEST's live task layer: assignment, physical execution and reporting.

No Isaac, no GPU, no ROS, no Docker.  The whole point of these tests is the
division of responsibility the Isaac demonstrator rests on:

* HARVEST creates tasks and decides assignments (``main.Scheduler``);
* the simulator reports physical evidence and nothing else;
* a report can move a task forward but can never create, reassign or invent one.
"""
import unittest

from harvest_control.interface import (ChargerLevel, ChargerState, CommandAck,
                                       CommandType, FleetSnapshot, GridState,
                                       TractorState)
from harvest_integrations import diagnostics
from harvest_integrations.simulators.isaac import contract
from harvest_integrations.tasks import LiveTaskService
from main import load_yaml_with_local

CFG = load_yaml_with_local("config.yaml")


class FakeRuntime:
    """A fleet that does what it is told and records what it was told."""

    def __init__(self, clock_min: float = 8 * 60):
        self.clock_min = clock_min
        self.submitted = []
        self.charging = set()
        self.occupied = {}
        self.soc = {"tractor_1": 80.0, "tractor_2": 55.0, "tractor_3": 30.0}
        self.pos = {"tractor_1": (50.0, 50.0), "tractor_2": (60.0, 50.0),
                    "tractor_3": (70.0, 50.0)}

    def snapshot(self):
        return FleetSnapshot(
            grid=GridState(self.clock_min, 2.0, 10.5, 1.0, "valle", 0.1),
            tractors=[TractorState(tid, self.soc[tid], 30.0, True,
                                   tid in self.charging, None, self.pos[tid],
                                   False, 0.0) for tid in self.soc],
            chargers=[ChargerState(cid, ChargerLevel("off"), 0.0,
                                   self.occupied.get(cid))
                      for cid in ("charger_1", "charger_2")],
            loads=[])

    def submit(self, commands):
        self.submitted.extend(commands)
        return [CommandAck(c, True, "") for c in commands]


def _service(runtime, **overrides):
    cfg = {**CFG}
    if overrides:
        cfg = {**CFG, "integrations": {**(CFG.get("integrations") or {}),
                                       "tasks": overrides}}
    # tick_period_s far in the future: the tests tick deliberately, so nothing
    # races with the assertions.
    return LiveTaskService(cfg, runtime, tick_period_s=3600.0)


class TestHarvestOwnsScheduling(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.service = _service(self.runtime)
        self.addCleanup(self.service.stop)

    def test_tasks_come_from_harvests_own_generator(self):
        document = self.service.document()
        self.assertEqual(document["scheduler"], "main.Scheduler (HARVEST)")
        self.assertGreater(len(document["tasks"]), 0)
        for row in document["tasks"]:
            self.assertIn(row["state"], contract.TASK_STATES)
            self.assertEqual(len(row["location"]), 2)
            self.assertIn(row["priority"], ("urgent", "normal", "flexible"))

    def test_assignment_is_actuated_through_the_fleet_interface(self):
        """An assignment is not real until it goes through FleetInterface.

        The device/semantic fleet is the system of record for what a tractor is
        doing, so the task layer must actuate like any other client rather than
        keeping a private opinion.
        """
        self.service.tick()
        assignments = self.service.document()["assignments"]
        self.assertTrue(assignments, "nothing was assigned at 08:00")
        submitted = [(c.type, c.target_id, c.value)
                     for c in self.runtime.submitted]
        for tractor_id, task_id in assignments.items():
            self.assertIn((CommandType.ASSIGN_TASK, tractor_id, task_id),
                          submitted)

    def test_no_redundant_commands_for_idle_tractors(self):
        self.service.tick()
        before = len(self.runtime.submitted)
        self.service.tick()
        self.assertEqual(len(self.runtime.submitted), before,
                         "a tick with no change must submit nothing")

    def test_tasks_are_not_due_before_their_window(self):
        runtime = FakeRuntime(clock_min=1 * 60)      # 01:00
        service = _service(runtime)
        self.addCleanup(service.stop)
        service.tick()
        self.assertEqual(service.document()["counts"]["assigned"], 0)
        self.assertFalse(runtime.submitted)

    def test_low_soc_charging_tractor_is_not_given_work(self):
        """HARVEST's existing rule, which the task layer must not override.

        ``main.Scheduler`` treats charging tractors as a LAST-RESORT pool and
        only considers those at 40 % or better; below that a charging tractor is
        left alone.  Asserted here because "charging interacts with task
        execution according to HARVEST's existing logic" is a requirement of the
        demonstrator, and the easy mistake would be for the task layer to grab
        any idle-looking tractor.
        """
        self.runtime.charging.update({"tractor_1", "tractor_2", "tractor_3"})
        self.runtime.soc.update({"tractor_1": 20.0, "tractor_2": 25.0,
                                 "tractor_3": 15.0})
        self.service.tick()
        self.assertEqual(self.service.document()["assignments"], {})
        self.assertFalse(self.runtime.submitted)

    def test_well_charged_tractor_may_be_taken_off_the_charger(self):
        """The other half of the same rule, and it is HARVEST's call.

        A charging tractor with enough SOC is in the last-resort pool, so work
        can be given to it and the scheduler releases it from the charger.  The
        task layer neither forces nor prevents this -- it just does not get in
        the way.
        """
        self.runtime.charging.update({"tractor_1", "tractor_2", "tractor_3"})
        self.service.tick()          # default SOCs: 80 / 55 / 30
        assignments = self.service.document()["assignments"]
        self.assertTrue(assignments, "the last-resort pool was never used")
        for tractor_id in assignments:
            self.assertGreaterEqual(self.runtime.soc[tractor_id], 40.0)


class TestPhysicalExecution(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.service = _service(self.runtime)
        self.addCleanup(self.service.stop)
        self.service.tick()
        self.assignments = dict(self.service.document()["assignments"])
        self.assertTrue(self.assignments)
        self.tractor, self.task_id = next(iter(self.assignments.items()))

    def _location(self, task_id):
        row = next(r for r in self.service.document()["tasks"]
                   if r["id"] == task_id)
        return row["location"]

    def _report(self, **fields):
        base = {"pose": self._location(self.task_id), "task_id": self.task_id,
                "at_task": False, "transit_progress_pct": 0.0,
                "task_progress_pct": 0.0, "task_complete": False}
        base.update(fields)
        self.service.report_physical(
            {"kind": "isaac", "tractors": {self.tractor: base}})

    def test_arrival_moves_the_task_to_active(self):
        self.assertEqual(self._state(self.task_id), "assigned")
        self._report(at_task=True, transit_progress_pct=100.0)
        self.service.tick()
        self.assertEqual(self._state(self.task_id), "active")
        self.assertEqual(self.service.document()["execution"], "physical")

    def test_completion_is_declared_by_harvest_from_physical_evidence(self):
        self._report(at_task=True)
        self.service.tick()
        self._report(at_task=True, task_progress_pct=100.0, task_complete=True)
        self.service.tick()
        row = next(r for r in self.service.document()["tasks"]
                   if r["id"] == self.task_id)
        self.assertEqual(row["state"], "completed")
        self.assertEqual(row["completed_by"], "physical")
        # ...and the tractor is free for HARVEST's next decision.
        self.assertNotIn(self.tractor, self.service.document()["assignments"])

    def test_a_report_about_an_unassigned_task_is_ignored(self):
        """Evidence, not instruction: the simulator cannot invent work."""
        other = next(r["id"] for r in self.service.document()["tasks"]
                     if r["id"] != self.task_id)
        self.service.report_physical({"kind": "isaac", "tractors": {
            self.tractor: {"pose": [0.0, 0.0], "task_id": other,
                           "at_task": True, "task_progress_pct": 100.0,
                           "task_complete": True}}})
        self.service.tick()
        self.assertEqual(self._state(other), "pending")

    def test_a_report_cannot_reassign_a_tractor(self):
        before = dict(self.service.document()["assignments"])
        self.service.report_physical({"kind": "isaac", "tractors": {
            "tractor_1": {"pose": [1.0, 2.0], "task_id": "task_999",
                          "at_task": True, "task_complete": True}}})
        self.service.tick()
        after = self.service.document()["assignments"]
        self.assertEqual(after.get(self.tractor), before.get(self.tractor))

    def test_stale_physical_reports_fall_back_to_the_clock(self):
        """A simulator that dies must not freeze the farm's work for ever."""
        import harvest_integrations.tasks as tasks_module

        self._report(at_task=True)
        self.service.tick()
        self.assertEqual(self.service.document()["execution"], "physical")
        original = tasks_module.PHYSICAL_STALE_S
        tasks_module.PHYSICAL_STALE_S = -1.0
        try:
            self.service.tick()
            self.assertEqual(self.service.document()["execution"], "clock")
        finally:
            tasks_module.PHYSICAL_STALE_S = original

    def test_malformed_report_is_refused(self):
        self.assertFalse(
            self.service.report_physical({"nonsense": True})["accepted"])

    def _state(self, task_id):
        return next(r["state"] for r in self.service.document()["tasks"]
                    if r["id"] == task_id)


class TestDayRollover(unittest.TestCase):
    """A long-running stack must keep having work, without cancelling any."""

    def test_day_rolls_on_the_midnight_wrap(self):
        runtime = FakeRuntime(clock_min=23 * 60 + 55)
        # A fleet that cannot be given work, so the roll is not deferred by
        # something in hand -- that behaviour has its own test below.
        runtime.charging.update(runtime.soc)
        runtime.soc.update({tid: 15.0 for tid in runtime.soc})
        service = _service(runtime)
        self.addCleanup(service.stop)
        service.tick()
        spent = service.document()
        # Late in the farm day the schedule is genuinely spent, and saying so is
        # the truth about it -- the service must NOT invent a new day here.
        self.assertEqual(spent["days_rolled"], 0)
        self.assertEqual(spent["counts"]["pending"], 0)

        runtime.clock_min = 5.0                      # the clock came round
        service.tick()
        fresh = service.document()
        self.assertEqual(fresh["days_rolled"], 1)
        self.assertGreater(fresh["counts"]["pending"], 0)
        self.assertEqual(fresh["day"], "2026-06-02")

    def test_a_deferred_roll_happens_once_the_work_finishes(self):
        """The wrap happens once; the roll must not be lost with it."""
        runtime = FakeRuntime(clock_min=23 * 60 + 55)
        service = _service(runtime)
        self.addCleanup(service.stop)
        service.tick()
        runtime.clock_min = 5.0                      # wraps, but work is in hand
        service.tick()
        self.assertEqual(service.document()["days_rolled"], 0)
        self.assertTrue(service._pending_roll)

        # The fleet gives up its work (flat batteries, on charge); the remembered
        # roll then goes through without needing another wrap.
        for tractor in service._tractors.values():
            tractor.current_task_id = None
        for task in service._tasks.values():
            if task.phase in ("TRANSIT", "EXECUTING"):
                task.phase, task.assigned_tractor_id = "DELAYED", None
        runtime.charging.update(runtime.soc)
        runtime.soc.update({tid: 15.0 for tid in runtime.soc})
        service.tick()
        self.assertEqual(service.document()["days_rolled"], 1)

    def test_a_spent_day_does_not_roll_repeatedly(self):
        """The failure this guards against thrashed once per tick.

        Rolling because the deadlines had passed moved the DATE forward while
        keeping the same time of day, so the new day was spent as well.
        """
        runtime = FakeRuntime(clock_min=23 * 60 + 55)
        service = _service(runtime)
        self.addCleanup(service.stop)
        for _ in range(5):
            service.tick()
        self.assertEqual(service.document()["days_rolled"], 0)

    def test_a_roll_does_not_cancel_work_in_progress(self):
        runtime = FakeRuntime()                     # 08:00, work gets assigned
        service = _service(runtime)
        self.addCleanup(service.stop)
        service.tick()
        assignments = dict(service.document()["assignments"])
        self.assertTrue(assignments)
        tractor, task_id = next(iter(assignments.items()))
        location = next(r["location"] for r in service.document()["tasks"]
                        if r["id"] == task_id)
        service.report_physical({"kind": "isaac", "tractors": {
            tractor: {"pose": location, "task_id": task_id, "at_task": True,
                      "task_progress_pct": 40.0, "task_complete": False}}})
        service.tick()

        runtime.clock_min = 23 * 60 + 58            # the day is over
        service.tick()
        document = service.document()
        self.assertEqual(document["days_rolled"], 0,
                         "the roll cancelled a task that was being worked")
        self.assertEqual(document["assignments"].get(tractor), task_id)


class TestSimulatorGoals(unittest.TestCase):
    """What the simulator is told -- consequences, never decisions."""

    def setUp(self):
        self.runtime = FakeRuntime()
        self.service = _service(self.runtime, max_visualised_tasks=5)
        self.addCleanup(self.service.stop)
        self.service.tick()

    def test_goals_are_capped_so_the_scene_stays_readable(self):
        goals = self.service.task_goals()
        self.assertLessEqual(len(goals["tasks"]), 5)

    def test_active_work_is_visualised_first(self):
        # With a cap, the tasks that matter now must be the ones that survive it.
        goals = self.service.task_goals()
        states = [t["state"] for t in goals["tasks"].values()]
        self.assertIn("assigned", states)

    def test_only_current_work_is_emphasised(self):
        for task in self.service.task_goals()["tasks"].values():
            self.assertEqual(task["emphasis"],
                             task["state"] in ("assigned", "active"))

    def test_goals_carry_harvests_numbers_not_the_simulators(self):
        for task in self.service.task_goals()["tasks"].values():
            self.assertGreater(task["work_seconds"], 0.0)
            self.assertGreater(task["work_radius_m"], 0.0)
            self.assertIn("priority", task)

    def test_contract_maps_assignment_to_a_drive_target(self):
        goals = self.service.task_goals()
        tractor, task_id = next(iter(goals["assignments"].items()))
        scene = contract.scene_from_config(CFG)
        snapshot = {"tractors": [{"id": tractor, "soc_pct": 50.0,
                                  "position": [50.0, 50.0]}],
                    "chargers": []}
        derived = contract.goals_from_snapshot(scene, snapshot, goals)
        goal = derived["goals"][tractor]
        self.assertEqual(goal["activity"], "task")
        self.assertEqual(goal["task_id"], task_id)
        self.assertEqual(goal["target"], goals["tasks"][task_id]["location"])


class TestTasksDiagnosticsRow(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.service = _service(self.runtime)
        self.addCleanup(self.service.stop)
        self.service.tick()

    def test_absent_service_reads_as_inactive_not_failed(self):
        name, state, detail, _ = diagnostics._tasks_row(None)
        self.assertEqual(name, "Farm tasks")
        self.assertEqual(state, diagnostics.INACTIVE)
        self.assertIn("optional", detail)

    def test_row_states_who_schedules_and_who_executes(self):
        _, state, detail, extra = diagnostics._tasks_row(self.service)
        self.assertEqual(state, diagnostics.HEALTHY)
        joined = "\n".join(extra["facts"])
        self.assertIn("main.Scheduler", joined)
        self.assertIn("HARVEST's clock", joined)      # no simulator connected
        self.assertIn("executed by clock", detail)

    def test_summary_lines_match_the_requested_shape(self):
        # tractor_2 -> task_018, travelling, 248 m remaining
        lines = self.service.summary_lines()
        self.assertTrue(lines)
        travelling = [l for l in lines if "travelling" in l]
        self.assertTrue(travelling, lines)
        self.assertRegex(travelling[0],
                         r"^tractor_\d+ -> task_\d+, travelling, \d+ m remaining$")

    def test_working_line_shows_percent(self):
        assignments = dict(self.service.document()["assignments"])
        tractor, task_id = next(iter(assignments.items()))
        location = next(r["location"] for r in self.service.document()["tasks"]
                        if r["id"] == task_id)
        self.service.report_physical({"kind": "isaac", "tractors": {
            tractor: {"pose": location, "task_id": task_id, "at_task": True,
                      "task_progress_pct": 62.0, "task_complete": False}}})
        self.service.tick()
        line = [l for l in self.service.summary_lines() if "working" in l]
        self.assertTrue(line, self.service.summary_lines())
        self.assertRegex(line[0], r"^tractor_\d+ -> task_\d+, working, \d+%$")

    def test_charging_line_names_the_charger(self):
        self.runtime.charging.add("tractor_3")
        self.runtime.occupied["charger_1"] = "tractor_3"
        self.service.tick()
        line = [l for l in self.service.summary_lines() if "charging" in l]
        self.assertEqual(line, ["tractor_3 -> charger_1, charging"])


class TestChargingVersusTasks(unittest.TestCase):
    """Where charging and work meet, at the contract boundary."""

    def setUp(self):
        self.scene = contract.scene_from_config(CFG)
        self.charger = next(e for e in self.scene["entities"]
                            if e["kind"] == "charger")
        self.task_goals = {
            "tasks": {"task_1": {"location": [300.0, 200.0],
                                 "work_seconds": 30.0, "work_radius_m": 6.0,
                                 "state": "assigned"}},
            "assignments": {"tractor_1": "task_1"}, "work_radius_m": 6.0}

    def _goal(self, *, charging, occupied):
        snapshot = {
            "tractors": [{"id": "tractor_1", "soc_pct": 50.0,
                          "charging": charging, "position": [50.0, 50.0]}],
            "chargers": [{"id": self.charger["id"], "power_kw": 6.6 if charging else 0.0,
                          "occupied_by": "tractor_1" if occupied else None}]}
        return contract.goals_from_snapshot(
            self.scene, snapshot, self.task_goals)["goals"]["tractor_1"]

    def test_active_charging_overrides_the_task(self):
        goal = self._goal(charging=True, occupied=True)
        self.assertEqual(goal["activity"], "charging")
        self.assertIsNone(goal["task_id"])

    def test_finished_charge_does_not_override_the_task(self):
        """A tractor still occupying a bay is not charging.

        With occupancy alone winning, a full tractor sat on the pad while
        HARVEST believed it was driving to its task -- measured on the live
        stack, which is why this test exists.
        """
        goal = self._goal(charging=False, occupied=True)
        self.assertEqual(goal["activity"], "task")
        self.assertEqual(goal["task_id"], "task_1")
        self.assertEqual(goal["target"], [300.0, 200.0])

    def test_charger_still_wins_when_there_is_no_work(self):
        goals = contract.goals_from_snapshot(
            self.scene,
            {"tractors": [{"id": "tractor_1", "soc_pct": 50.0,
                           "charging": False, "position": [50.0, 50.0]}],
             "chargers": [{"id": self.charger["id"], "power_kw": 0.0,
                           "occupied_by": "tractor_1"}]},
            {"tasks": {}, "assignments": {}})["goals"]["tractor_1"]
        self.assertEqual(goals["activity"], "charging")


class TestTaskBoardMirror(unittest.TestCase):
    """The NGSI-LD mirror: the semantic answer, not a progress stream."""

    def setUp(self):
        self.runtime = FakeRuntime()
        self.service = _service(self.runtime)
        self.addCleanup(self.service.stop)
        self.service.tick()

    def test_board_carries_counts_assignments_and_who_decided(self):
        from harvest_integrations.fiware.entities import (TASK_BOARD_ENTITY_ID,
                                                         task_board_entity)

        entity = task_board_entity(self.service.document())
        self.assertEqual(entity["id"], TASK_BOARD_ENTITY_ID)
        self.assertEqual(entity["type"], "FarmTaskBoard")
        self.assertIn("main.Scheduler", entity["scheduler"]["value"])
        self.assertEqual(entity["executedBy"]["value"], "clock")
        self.assertGreater(entity["tasksTotal"]["value"], 0)
        self.assertTrue(entity["assignments"]["value"])

    def test_board_does_not_mirror_per_task_progress(self):
        """A context broker is not a telemetry bus.

        Progress belongs to /api/tasks and Diagnostics; if it ever appears here
        the broker gets rewritten several times a second to say almost nothing.
        """
        from harvest_integrations.fiware.entities import task_board_entity

        entity = task_board_entity(self.service.document())
        serialised = str(entity)
        self.assertNotIn("progress_pct", serialised)
        self.assertNotIn("transit", serialised)


if __name__ == "__main__":
    unittest.main()
