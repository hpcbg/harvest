"""
Energy-model calibration: the real mission versus HARVEST's simulated model.

    python -m harvest_integrations.telemetry.analysis telemetry/telemetria_mision_63.csv
    python -m harvest_integrations.telemetry.analysis <csv> --json out.json --config config.yaml

Everything computed here is derived ONLY from signals present in the export
(see :mod:`.zetrack`).  Where a figure needs an assumption (an interpolation
rule, a gap cap, a nominal capacity from config.yaml) the assumption is
stated next to the number.  Nothing here is a confirmed vehicle
specification: the *estimated effective capacity* in particular is what the
mission data implies, and must be confirmed with the ZETRABOT team before it
replaces ``tractors.model.battery_capacity_kwh``.

Method notes
------------
* **Time base** -- signals are sampled asynchronously.  Integrals use
  sample-and-hold: between two consecutive messages the last observed value
  of every signal is held.  Intervals longer than ``gap_cap_s`` (default
  120 s: a power-off, the lunch break, the overnight) contribute nothing, so
  a stale reading never gets integrated across a gap.
* **Discharged energy** -- ``DischEnrgActualSesion`` is a per-power-on
  session counter.  The mission total is the sum of the maxima of each
  session (a reset = a drop below half of the running session maximum).  The
  Zetrack report quotes 6.43 kWh, which is the largest single session.
* **Power** -- ``MainBatteryVoltage x BatteryCurrent`` (held), positive =
  discharge.  Its time integral is an *independent* check on the counter.
* **PTO / motion / idle** -- the ``PtoLed`` and ``GoLed`` indicators (held)
  split the timeline into PTO-active, drive-active and idle periods; the
  power integral is attributed to each.
* **Distance** -- the export has no GPS; the only distance signal is the
  ``LifetimeKm`` counter, which contains an outlier in this mission (a 108 km
  reading among ~170 km readings).  The delta is computed after discarding
  readings more than 25 % away from the median and is flagged accordingly.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .model import TelemetryMessage, iso_utc, sort_messages
from .normalizer import _SessionAccumulator, _num

DEFAULT_GAP_CAP_S = 120.0
ESTIMATE_LABEL = ("ESTIMATE derived from one mission (discharged energy / SOC change); "
                  "requires confirmation from the ZETRABOT team before use as a specification")


def _pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    data = sorted(values)
    pos = (len(data) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return data[lo]
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def _r(value: Optional[float], nd: int = 3) -> Optional[float]:
    return None if value is None else round(value, nd)


class _Bucket:
    """Energy / time accumulator for one operating regime."""

    def __init__(self) -> None:
        self.hours = 0.0
        self.kwh = 0.0

    def add(self, dt_h: float, kw: Optional[float]) -> None:
        self.hours += dt_h
        if kw is not None:
            self.kwh += kw * dt_h

    def to_dict(self) -> Dict[str, Any]:
        mean = self.kwh / self.hours if self.hours > 0 else None
        return {"hours": _r(self.hours), "energy_kwh": _r(self.kwh), "mean_power_kw": _r(mean)}


class MissionAnalysis:
    """Compute the calibration summary for one mission (one tractor)."""

    def __init__(self, messages: Iterable[TelemetryMessage], *, gap_cap_s: float = DEFAULT_GAP_CAP_S,
                 harvest_model: Optional[Dict[str, Any]] = None, tractor_id: Optional[str] = None):
        msgs = sort_messages(messages)
        if tractor_id is not None:
            msgs = [m for m in msgs if m.tractor_id == str(tractor_id)]
        self.messages = msgs
        self.gap_cap_s = gap_cap_s
        self.harvest_model = dict(harvest_model or {})
        self.result: Dict[str, Any] = {}

    # -- entry point ----------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        msgs = self.messages
        if not msgs:
            self.result = {"error": "no telemetry messages"}
            return self.result

        soc: List[Tuple[_dt.datetime, float]] = []
        voltage: List[float] = []
        current: List[float] = []
        power_samples: List[float] = []
        battery_temp: List[float] = []
        motor_temps: Dict[str, List[float]] = {}
        oil_temp: List[float] = []
        speeds: List[float] = []
        lifetime_km: List[float] = []
        session = _SessionAccumulator()
        session_values: List[Tuple[_dt.datetime, float]] = []
        message_counts: Dict[str, int] = {}
        tractors, missions = set(), set()

        # held state
        v: Optional[float] = None
        i: Optional[float] = None
        pto: Optional[bool] = None
        go: Optional[bool] = None
        speed: Optional[float] = None
        prev_ts: Optional[_dt.datetime] = None

        total = _Bucket()
        pto_b, drive_b, moving_b, idle_b = _Bucket(), _Bucket(), _Bucket(), _Bucket()
        active_h = 0.0
        gap_count = 0
        gap_h = 0.0
        peak_power = None
        peak_power_ts = None
        peak_current = None
        regen_samples = 0
        sessions_energy: List[float] = []

        for msg in msgs:
            tractors.add(msg.tractor_id)
            if msg.mission_id:
                missions.add(msg.mission_id)
            message_counts[msg.source_message] = message_counts.get(msg.source_message, 0) + 1

            # -- integrate the interval that just ended (sample-and-hold) --
            if prev_ts is not None:
                dt_s = (msg.timestamp - prev_ts).total_seconds()
                if 0 < dt_s <= self.gap_cap_s:
                    dt_h = dt_s / 3600.0
                    active_h += dt_h
                    kw = (v * i / 1000.0) if (v is not None and i is not None) else None
                    total.add(dt_h, kw)
                    if pto:
                        pto_b.add(dt_h, kw)
                    if go:
                        drive_b.add(dt_h, kw)
                    if speed is not None and speed > 0:
                        moving_b.add(dt_h, kw)
                    if not pto and not go and not (speed and speed > 0):
                        idle_b.add(dt_h, kw)
                elif dt_s > self.gap_cap_s:
                    gap_count += 1
                    gap_h += dt_s / 3600.0
            prev_ts = msg.timestamp

            # -- update held state from this message --
            s = msg.signals
            if "MainBatterySOC" in s and _num(s["MainBatterySOC"]) is not None:
                soc.append((msg.timestamp, _num(s["MainBatterySOC"])))
            if "MainBatteryVoltage" in s and _num(s["MainBatteryVoltage"]) is not None:
                v = _num(s["MainBatteryVoltage"]); voltage.append(v)
            if "BatteryCurrent" in s and _num(s["BatteryCurrent"]) is not None:
                i = _num(s["BatteryCurrent"]); current.append(i)
                if i < 0:
                    regen_samples += 1
                if peak_current is None or i > peak_current:
                    peak_current = i
                if v is not None:
                    p = v * i / 1000.0
                    power_samples.append(p)
                    if peak_power is None or p > peak_power:
                        peak_power, peak_power_ts = p, msg.timestamp
            if "DischEnrgActualSesion" in s and _num(s["DischEnrgActualSesion"]) is not None:
                val = _num(s["DischEnrgActualSesion"])
                before = session.resets
                session.push(val)
                if session.resets > before:
                    sessions_energy.append(session.completed_kwh - sum(sessions_energy))
                session_values.append((msg.timestamp, val))
            if "MainBatteryTemp" in s and _num(s["MainBatteryTemp"]) is not None:
                battery_temp.append(_num(s["MainBatteryTemp"]))
            if msg.source_message == "MotorTemp":
                for key, val in s.items():
                    num = _num(val)
                    if num is not None:
                        motor_temps.setdefault(key, []).append(num)
            if "OilTemp" in s and _num(s["OilTemp"]) is not None:
                oil_temp.append(_num(s["OilTemp"]))
            if "SpeedDisplay" in s and _num(s["SpeedDisplay"]) is not None:
                speed = _num(s["SpeedDisplay"]); speeds.append(speed)
            if "LifetimeKm" in s and _num(s["LifetimeKm"]) is not None:
                lifetime_km.append(_num(s["LifetimeKm"]))
            if "PtoLed" in s and _num(s["PtoLed"]) is not None:
                pto = bool(round(_num(s["PtoLed"])))
            if "GoLed" in s and _num(s["GoLed"]) is not None:
                go = bool(round(_num(s["GoLed"])))

        # close the last session
        if session.session_max is not None:
            sessions_energy.append(session.session_max)
        discharged_total = sum(sessions_energy)

        first_ts, last_ts = msgs[0].timestamp, msgs[-1].timestamp
        span_h = (last_ts - first_ts).total_seconds() / 3600.0

        soc_first = soc[0][1] if soc else None
        soc_last = soc[-1][1] if soc else None
        soc_delta = (soc_first - soc_last) if soc else None

        capacity_est = None
        if soc_delta and soc_delta > 0 and discharged_total > 0:
            capacity_est = discharged_total / (soc_delta / 100.0)

        distance = self._distance(lifetime_km)

        result: Dict[str, Any] = {
            "mission": {
                "mission_ids": sorted(missions),
                "tractor_ids": sorted(tractors),
                "messages": len(msgs),
                "message_counts": dict(sorted(message_counts.items())),
                "first_timestamp": iso_utc(first_ts),
                "last_timestamp": iso_utc(last_ts),
                "wall_span_h": _r(span_h),
                "active_h": _r(active_h),
                "gaps_over_cap": gap_count,
                "gap_hours": _r(gap_h),
                "gap_cap_s": self.gap_cap_s,
                "power_sessions": len(sessions_energy),
            },
            "soc": {
                "first_pct": soc_first, "last_pct": soc_last,
                "delta_pct": soc_delta,
                "min_pct": min(x[1] for x in soc) if soc else None,
                "max_pct": max(x[1] for x in soc) if soc else None,
                "samples": len(soc),
            },
            "discharged_energy": {
                "session_maxima_kwh": [_r(x) for x in sessions_energy],
                "sum_of_sessions_kwh": _r(discharged_total),
                "largest_session_kwh": _r(max(sessions_energy)) if sessions_energy else None,
                "samples": len(session_values),
                "method": "sum of per-power-on-session maxima of DischEnrgActualSesion "
                          "(reset = drop below half the running session maximum)",
            },
            "power": {
                "integrated_kwh": _r(total.kwh),
                "mean_kw_active": _r(total.kwh / total.hours) if total.hours else None,
                "peak_kw": _r(peak_power),
                "peak_at": iso_utc(peak_power_ts),
                "peak_current_a": _r(peak_current),
                "regen_samples": regen_samples,
                "samples": len(power_samples),
                "percentiles_kw": {f"p{int(q * 100)}": _r(_pct(power_samples, q))
                                   for q in (0.5, 0.75, 0.9, 0.99)},
                "mean_sample_kw": _r(statistics.fmean(power_samples)) if power_samples else None,
                "voltage_v": {"min": _r(min(voltage)), "max": _r(max(voltage)),
                              "median": _r(statistics.median(voltage))} if voltage else None,
                "current_a": {"min": _r(min(current)), "max": _r(max(current)),
                              "median": _r(statistics.median(current)),
                              "mean": _r(statistics.fmean(current))} if current else None,
                "method": "sample-and-hold V x I integrated over intervals <= gap_cap_s",
            },
            "energy_rate": {
                "kwh_per_active_hour": _r(discharged_total / active_h) if active_h else None,
                "kwh_per_wall_hour": _r(discharged_total / span_h) if span_h else None,
                "basis": "sum_of_sessions_kwh",
            },
            "regimes": {
                "pto_active": pto_b.to_dict(),
                "drive_active": drive_b.to_dict(),
                "moving_speed_gt_0": moving_b.to_dict(),
                "idle": idle_b.to_dict(),
                "all_active": total.to_dict(),
                "note": "regimes overlap (PTO usually runs while driving); energy is the "
                        "V x I integral attributed by the held PtoLed / GoLed / SpeedDisplay state",
            },
            "temperatures_c": {
                "battery": self._stats(battery_temp),
                "oil": self._stats(oil_temp),
                "motors": {k: self._stats(vals) for k, vals in sorted(motor_temps.items())},
            },
            "speed_kmh": self._stats(speeds),
            "distance": distance,
            "capacity_estimate": {
                "effective_capacity_kwh": _r(capacity_est, 2),
                "status": ESTIMATE_LABEL,
                "inputs": {"discharged_kwh": _r(discharged_total), "soc_delta_pct": soc_delta},
                "nominal_config_kwh": self.harvest_model.get("battery_capacity_kwh"),
                "ratio_to_nominal": (_r(capacity_est / float(self.harvest_model["battery_capacity_kwh"]), 3)
                                     if capacity_est and self.harvest_model.get("battery_capacity_kwh") else None),
            },
            "not_available": [
                "GPS latitude/longitude (route map exists only in the Zetrack PDF report)",
                "charger / charging power (no charging occurred and no charger signal exists)",
                "ambient temperature, PV, grid",
            ],
        }
        result["harvest_model_comparison"] = self._compare(result, pto_b, drive_b, moving_b, idle_b, distance)
        self.result = result
        return result

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _stats(values: Sequence[float]) -> Optional[Dict[str, Any]]:
        if not values:
            return None
        return {"min": _r(min(values)), "max": _r(max(values)),
                "mean": _r(statistics.fmean(values)), "samples": len(values)}

    @staticmethod
    def _distance(lifetime_km: Sequence[float]) -> Dict[str, Any]:
        if len(lifetime_km) < 2:
            return {"delta_km": None, "note": "LifetimeKm has fewer than two samples"}
        med = statistics.median(lifetime_km)
        clean = [x for x in lifetime_km if abs(x - med) <= 0.25 * med]
        outliers = len(lifetime_km) - len(clean)
        delta = (clean[-1] - clean[0]) if len(clean) >= 2 else None
        return {
            "delta_km": _r(delta),
            "first_km": _r(clean[0]) if clean else None,
            "last_km": _r(clean[-1]) if clean else None,
            "outliers_discarded": outliers,
            "samples": len(lifetime_km),
            "note": "from the LifetimeKm counter (no GPS in the export); readings more "
                    "than 25 % from the median discarded as counter glitches",
        }

    def _compare(self, result: Dict[str, Any], pto_b: _Bucket, drive_b: _Bucket, moving_b: _Bucket,
                 idle_b: _Bucket, distance: Dict[str, Any]) -> List[Dict[str, Any]]:
        model = self.harvest_model
        rows: List[Dict[str, Any]] = []

        def row(param, harvest, measured, unit, note):
            ratio = None
            if isinstance(harvest, (int, float)) and isinstance(measured, (int, float)) and harvest:
                ratio = _r(measured / harvest, 3)
            rows.append({"parameter": param, "harvest_config": harvest, "measured": _r(measured) if isinstance(measured, float) else measured,
                         "unit": unit, "measured_over_config": ratio, "note": note})

        row("battery_capacity_kwh", model.get("battery_capacity_kwh"),
            result["capacity_estimate"]["effective_capacity_kwh"], "kWh",
            "effective capacity implied by this mission -- " + ESTIMATE_LABEL)
        row("pto_power_kw", model.get("pto_power_kw"),
            pto_b.kwh / pto_b.hours if pto_b.hours else None, "kW",
            "mean battery power while PtoLed=1 (includes traction while working)")
        row("idle_kwh_per_h", model.get("idle_kwh_per_h"),
            idle_b.kwh / idle_b.hours if idle_b.hours else None, "kW",
            "mean battery power while powered on with PTO off, drive off, speed 0")
        drive_kwh_per_km = None
        if distance.get("delta_km") and moving_b.kwh:
            drive_kwh_per_km = moving_b.kwh / distance["delta_km"]
        row("driving_kwh_per_km", model.get("driving_kwh_per_km"), drive_kwh_per_km, "kWh/km",
            "V x I energy while SpeedDisplay > 0 over the LifetimeKm delta -- includes PTO "
            "work done while moving, so an upper bound on pure driving consumption")
        speed = result.get("speed_kmh") or {}
        row("eco_speed_kmh", model.get("eco_speed_kmh"), speed.get("mean"), "km/h",
            "mean of SpeedDisplay samples (0 while stationary); max observed "
            f"{speed.get('max')} km/h vs config max_speed_kmh {model.get('max_speed_kmh')}")
        return rows

    # -- rendering ------------------------------------------------------------
    def text(self) -> str:
        r = self.result or self.run()
        if "error" in r:
            return r["error"]
        m, s, d, p, e, reg, cap = (r["mission"], r["soc"], r["discharged_energy"], r["power"],
                                   r["energy_rate"], r["regimes"], r["capacity_estimate"])
        lines = [
            f"Mission {', '.join(m['mission_ids']) or '?'}  tractor {', '.join(m['tractor_ids'])}",
            f"  {m['messages']} messages, {m['first_timestamp']} .. {m['last_timestamp']}",
            f"  wall span {m['wall_span_h']} h, powered/active {m['active_h']} h, "
            f"{m['gaps_over_cap']} gaps > {m['gap_cap_s']:.0f}s ({m['gap_hours']} h), "
            f"{m['power_sessions']} power-on sessions",
            "",
            f"SOC              {s['first_pct']} % -> {s['last_pct']} %  (delta {s['delta_pct']} %, "
            f"{s['samples']} samples)",
            f"Discharged       {d['sum_of_sessions_kwh']} kWh  = sum of {len(d['session_maxima_kwh'])} "
            f"session maxima {d['session_maxima_kwh']}",
            f"                 largest single session {d['largest_session_kwh']} kWh "
            "(the figure the Zetrack report quotes)",
            f"V x I integral   {p['integrated_kwh']} kWh  (independent check; "
            f"mean {p['mean_kw_active']} kW while powered, peak {p['peak_kw']} kW at {p['peak_at']}, "
            f"peak current {p['peak_current_a']} A)",
            f"Power percentiles kW  {p['percentiles_kw']}",
            f"Energy rate      {e['kwh_per_active_hour']} kWh per powered hour, "
            f"{e['kwh_per_wall_hour']} kWh per wall-clock hour",
            "",
            "Regimes (V x I energy attributed by held PTO / drive / speed state):",
        ]
        for name in ("pto_active", "drive_active", "moving_speed_gt_0", "idle", "all_active"):
            b = reg[name]
            lines.append(f"  {name:18s} {b['hours']!s:>8} h  {b['energy_kwh']!s:>8} kWh  "
                         f"mean {b['mean_power_kw']} kW")
        t = r["temperatures_c"]
        lines += ["", "Temperatures (degC):",
                  f"  battery {t['battery']}", f"  oil     {t['oil']}"]
        for k, val in (t["motors"] or {}).items():
            lines.append(f"  {k:8s}{val}")
        lines += ["", f"Distance: {r['distance']}", "",
                  f"Effective capacity ESTIMATE: {cap['effective_capacity_kwh']} kWh "
                  f"(config nominal {cap['nominal_config_kwh']} kWh, ratio {cap['ratio_to_nominal']})",
                  f"  {cap['status']}", "",
                  "HARVEST model comparison:"]
        for c in r["harvest_model_comparison"]:
            lines.append(f"  {c['parameter']:22s} config {c['harvest_config']!s:>8} {c['unit']:7s} "
                         f"measured {c['measured']!s:>9}  ratio {c['measured_over_config']}")
            lines.append(f"      {c['note']}")
        lines += ["", "Not available in this telemetry: " + "; ".join(r["not_available"])]
        return "\n".join(lines)


def analyse_csv(path: Any, config_path: Optional[Any] = None, gap_cap_s: float = DEFAULT_GAP_CAP_S,
                tractor_id: Optional[str] = None) -> MissionAnalysis:
    from .zetrack import read_csv
    model: Dict[str, Any] = {}
    if config_path:
        try:
            import yaml
            with open(config_path, encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle) or {}
            model = (cfg.get("tractors") or {}).get("model") or {}
        except Exception as exc:                                   # noqa: BLE001
            print(f"[analysis] config not loaded ({exc}); comparison rows will lack config values",
                  file=sys.stderr)
    analysis = MissionAnalysis(read_csv(path), gap_cap_s=gap_cap_s, harvest_model=model,
                               tractor_id=tractor_id)
    analysis.run()
    return analysis


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ZETRABOT mission vs HARVEST energy model")
    parser.add_argument("csv", help="Zetrack telemetry export (CSV)")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "config.yaml"),
                        help="HARVEST config.yaml for the model comparison (default: repo config)")
    parser.add_argument("--json", help="write the full result as JSON to this path ('-' = stdout)")
    parser.add_argument("--gap-cap", type=float, default=DEFAULT_GAP_CAP_S,
                        help="intervals longer than this (s) are treated as power-off gaps")
    parser.add_argument("--tractor", help="restrict to one tractor_id")
    args = parser.parse_args(argv)
    analysis = analyse_csv(args.csv, args.config, args.gap_cap, args.tractor)
    if args.json == "-":
        json.dump(analysis.result, sys.stdout, indent=2)
        print()
    else:
        print(analysis.text())
        if args.json:
            Path(args.json).write_text(json.dumps(analysis.result, indent=2), encoding="utf-8")
            print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
