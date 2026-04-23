from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random


@dataclass
class GeneratedTask:
    task_id: str
    name: str
    location: Dict[str, float]
    earliest_start: str
    latest_finish: str
    duration_minutes: int
    distance_km: float
    priority: str
    can_wait: bool
    uses_pto: bool
    pto_power_kw: float


def _rand_location(rng: random.Random, width_m: float, height_m: float) -> Dict[str, float]:
    return {
        "x": round(rng.uniform(50, width_m - 50), 2),
        "y": round(rng.uniform(50, height_m - 50), 2),
    }


def _make_task(
    task_id: int,
    name: str,
    start: datetime,
    duration_min: int,
    deadline_slack_min: int,
    distance_km: float,
    priority: str,
    can_wait: bool,
    uses_pto: bool,
    pto_power_kw: float,
    location: Dict[str, float],
) -> GeneratedTask:
    latest_finish = start + timedelta(minutes=duration_min + deadline_slack_min)
    return GeneratedTask(
        task_id=f"task_{task_id:03d}",
        name=name,
        location=location,
        earliest_start=start.isoformat(sep=" "),
        latest_finish=latest_finish.isoformat(sep=" "),
        duration_minutes=duration_min,
        distance_km=round(distance_km, 2),
        priority=priority,
        can_wait=can_wait,
        uses_pto=uses_pto,
        pto_power_kw=round(pto_power_kw, 2),
    )


def generate_tasks(
    start_date: str,
    num_tasks: int = 12,
    seed: int = 42,
    map_width_m: float = 800,
    map_height_m: float = 500,
) -> List[Dict[str, Any]]:
    """
    Generate tasks with a realistic daily farm pattern:
    - morning wave
    - midday wave
    - afternoon wave
    - mix of urgent/normal/flexible
    - PTO and non-PTO tasks
    """

    rng = random.Random(seed)
    day0 = datetime.fromisoformat(start_date)

    task_templates = [
        ("Mowing block", True, (6.0, 9.0), (60, 120)),
        ("Pruning support", True, (5.0, 8.0), (45, 90)),
        ("Transport boxes", False, (3.0, 6.0), (30, 75)),
        ("Inspection route", False, (2.0, 5.0), (20, 60)),
        ("Sprayer assistance", True, (7.0, 10.0), (60, 120)),
        ("Tool delivery", False, (1.5, 4.0), (20, 45)),
    ]

    waves = [
        {"start_hour": 8, "end_hour": 10, "weight": 0.35},
        {"start_hour": 11, "end_hour": 14, "weight": 0.30},
        {"start_hour": 15, "end_hour": 19, "weight": 0.35},
    ]

    priorities = [
        ("urgent", 0.25),
        ("normal", 0.45),
        ("flexible", 0.30),
    ]

    def weighted_choice(items):
        r = rng.random()
        cumulative = 0.0
        for value, prob in items:
            cumulative += prob
            if r <= cumulative:
                return value
        return items[-1][0]

    def choose_wave():
        r = rng.random()
        cumulative = 0.0
        for w in waves:
            cumulative += w["weight"]
            if r <= cumulative:
                return w
        return waves[-1]

    tasks: List[GeneratedTask] = []

    for i in range(1, num_tasks + 1):
        tmpl_name, uses_pto, pto_range, dur_range = rng.choice(task_templates)
        wave = choose_wave()

        hour = rng.randint(wave["start_hour"], wave["end_hour"] - 1)
        minute = rng.choice([0, 15, 30, 45])
        start = day0.replace(hour=hour, minute=minute, second=0, microsecond=0)

        duration_min = rng.randint(dur_range[0], dur_range[1])
        distance_km = rng.uniform(1.5, 8.0)
        priority = weighted_choice(priorities)

        # more restrictive deadlines for urgent tasks
        if priority == "urgent":
            deadline_slack = rng.randint(20, 60)
            can_wait = False
        elif priority == "normal":
            deadline_slack = rng.randint(60, 180)
            can_wait = True
        else:
            deadline_slack = rng.randint(180, 300)
            can_wait = True

        pto_power_kw = rng.uniform(*pto_range) if uses_pto else 0.0

        # slightly richer names
        name = f"{tmpl_name} {rng.choice(['A', 'B', 'C', 'D'])}"

        task = _make_task(
            task_id=i,
            name=name,
            start=start,
            duration_min=duration_min,
            deadline_slack_min=deadline_slack,
            distance_km=distance_km,
            priority=priority,
            can_wait=can_wait,
            uses_pto=uses_pto,
            pto_power_kw=pto_power_kw,
            location=_rand_location(rng, map_width_m, map_height_m),
        )
        tasks.append(task)

    # sort by earliest start
    tasks.sort(key=lambda t: t.earliest_start)

    return [t.__dict__ for t in tasks]