"""
Replay of historical telemetry on its original timeline.

:class:`ReplayPlayer` takes an ordered list of :class:`TelemetryMessage` and
delivers them to a callback from a background thread, waiting between
consecutive messages for the *original* interval divided by ``speed``.  It
is how :class:`~.zetrack.CsvTelemetrySource.subscribe` works, and it is
reusable for any historical source (an S3 archive, a Timestream query) that
wants to be fed into HARVEST as if it were live.

Timeline rules:

* ordering is the source's (timestamp, sequence, id) order and is never
  changed by the player;
* ``speed`` scales every interval (20 = twenty telemetry seconds per real
  second); ``speed <= 0`` delivers as fast as the consumer accepts;
* ``max_gap_s`` caps the *real-time* wait between two messages.  The
  supplied mission has a 2.2-hour lunch break and overnight gaps of 15+ hours
  between its three days; with the cap those become a short pause instead of
  a stalled dashboard.  Every compressed gap is counted and reported;
* ``loop`` restarts from the beginning when the end is reached (the
  normaliser sees the restart as out-of-order messages, which is honest --
  the replay position is also reported so nobody mistakes a loop for new
  data).
"""
from __future__ import annotations

import datetime as _dt
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .model import MessageCallback, Subscription, TelemetryMessage, iso_utc


@dataclass
class ReplayOptions:
    speed: float = 1.0
    max_gap_s: Optional[float] = 30.0
    loop: bool = False
    start: Optional[_dt.datetime] = None
    end: Optional[_dt.datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"speed": self.speed, "max_gap_s": self.max_gap_s, "loop": self.loop,
                "start": iso_utc(self.start), "end": iso_utc(self.end)}


@dataclass
class ReplayProgress:
    state: str = "pending"          # pending | running | finished | stopped | failed
    index: int = 0                  # messages delivered in the current pass
    total: int = 0
    passes: int = 0
    position: Optional[_dt.datetime] = None      # telemetry time of the last delivered message
    first: Optional[_dt.datetime] = None
    last: Optional[_dt.datetime] = None
    started_wall: Optional[float] = None
    finished_wall: Optional[float] = None
    gaps_compressed: int = 0
    gap_time_skipped_s: float = 0.0
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        now = time.time()
        elapsed = None
        if self.started_wall is not None:
            elapsed = (self.finished_wall or now) - self.started_wall
        replay_elapsed = None
        if self.first is not None and self.position is not None:
            replay_elapsed = (self.position - self.first).total_seconds()
        span = None
        if self.first is not None and self.last is not None:
            span = (self.last - self.first).total_seconds()
        return {
            "state": self.state,
            "index": self.index,
            "total": self.total,
            "passes": self.passes,
            "progress_pct": round(100.0 * self.index / self.total, 2) if self.total else 0.0,
            "position": iso_utc(self.position),
            "first_timestamp": iso_utc(self.first),
            "last_timestamp": iso_utc(self.last),
            "telemetry_span_s": span,
            "replay_elapsed_s": replay_elapsed,
            "real_elapsed_s": round(elapsed, 3) if elapsed is not None else None,
            "gaps_compressed": self.gaps_compressed,
            "gap_time_skipped_s": round(self.gap_time_skipped_s, 3),
            "error": self.error,
            **self.extra,
        }


class ReplayPlayer(Subscription):
    """Background delivery of an ordered message list on its own timeline."""

    def __init__(self, messages: Sequence[TelemetryMessage], callback: MessageCallback,
                 options: Optional[ReplayOptions] = None, label: str = "replay"):
        self._messages: List[TelemetryMessage] = list(messages)
        self._callback = callback
        self.options = options or ReplayOptions()
        self.label = label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.progress = ReplayProgress(total=len(self._messages))
        if self._messages:
            self.progress.first = self._messages[0].timestamp
            self.progress.last = self._messages[-1].timestamp
        self.progress.extra["speed"] = self.options.speed
        self.progress.extra["max_gap_s"] = self.options.max_gap_s
        self.progress.extra["loop"] = self.options.loop
        self.progress.extra["label"] = label

    # -- Subscription ---------------------------------------------------------
    def start(self) -> "ReplayPlayer":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name=f"telemetry-replay:{self.label}",
                                        daemon=True)
        with self._lock:
            self.progress.state = "running"
            self.progress.started_wall = time.time()
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            doc = self.progress.to_dict()
        doc["alive"] = self.is_alive()
        return doc

    # -- the loop -------------------------------------------------------------
    def _wait_for(self, previous: Optional[_dt.datetime], current: _dt.datetime) -> None:
        if previous is None or self.options.speed <= 0:
            return
        delta = (current - previous).total_seconds()
        if delta <= 0:
            return
        wait = delta / self.options.speed
        cap = self.options.max_gap_s
        if cap is not None and wait > cap:
            with self._lock:
                self.progress.gaps_compressed += 1
                self.progress.gap_time_skipped_s += (wait - cap) * self.options.speed
            wait = cap
        if wait > 0:
            self._stop.wait(wait)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                previous: Optional[_dt.datetime] = None
                with self._lock:
                    self.progress.index = 0
                    self.progress.passes += 1
                for msg in self._messages:
                    if self._stop.is_set():
                        break
                    self._wait_for(previous, msg.timestamp)
                    if self._stop.is_set():
                        break
                    self._callback(msg)
                    previous = msg.timestamp
                    with self._lock:
                        self.progress.index += 1
                        self.progress.position = msg.timestamp
                if not self.options.loop:
                    break
        except Exception as exc:                                   # noqa: BLE001
            with self._lock:
                self.progress.state = "failed"
                self.progress.error = f"{type(exc).__name__}: {exc}"
                self.progress.finished_wall = time.time()
            return
        with self._lock:
            self.progress.state = "stopped" if self._stop.is_set() else "finished"
            self.progress.finished_wall = time.time()


def replay_messages(messages: Sequence[TelemetryMessage], callback: MessageCallback,
                    options: Optional[ReplayOptions] = None) -> ReplayProgress:
    """Synchronous helper: replay to completion on the calling thread."""
    player = ReplayPlayer(messages, callback, options)
    player.start()
    player.join()
    return player.progress


__all__ = ["ReplayOptions", "ReplayPlayer", "ReplayProgress", "replay_messages"]
