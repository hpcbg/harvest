"""
Mission KPIs and model validation for a real ZETRABOT mission.

    python -m harvest_integrations.telemetry.kpi telemetry/telemetria_mision_63.csv
    python -m harvest_integrations.telemetry.kpi <csv> --json kpis.json --csv kpis.csv [--wide]

Built on top of :class:`~.analysis.MissionAnalysis` (which computes the
numbers); this module *classifies*, *compares* and *exports* them:

* every KPI carries a **provenance** class -- MEASURED (a value present in
  the telemetry), DERIVED (calculated from measured values only), CONFIGURED
  (taken from HARVEST's config.yaml), ESTIMATED (inferred, subject to stated
  assumptions -- including HARVEST model predictions) or MISSING (the source
  cannot supply it) -- plus its formula, the wire signals it came from and
  any quality flags;
* the **real vs HARVEST model** table applies HARVEST's own energy model
  (``tractors.model`` in config.yaml, the same parameters ``main.py``'s
  scheduler and simulator use) to the operating profile that was *measured*,
  and compares like with like.  A metric the model cannot predict is marked
  unavailable instead of being forced;
* **findings** are fixed rules over those numbers (thresholds below), not
  generated prose.  **Calibration proposals** are suggestions for review:
  nothing here ever writes config.yaml;
* **limitations** of the data and of the current deployment travel with
  every export, so a number is never separated from its caveats.

Sign convention everywhere: ``difference = model - real`` and
``error_pct = (model - real) / |real| x 100`` -- positive means the HARVEST
model predicts MORE than was measured.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .analysis import ANALYSIS_VERSION, DEFAULT_GAP_CAP_S, MissionAnalysis
from .model import TelemetryMessage

SCHEMA = "harvest-mission-kpis/1.0"

MEASURED = "MEASURED"
DERIVED = "DERIVED"
CONFIGURED = "CONFIGURED"
ESTIMATED = "ESTIMATED"
MISSING = "MISSING"
PROVENANCE = {
    MEASURED: "directly present in the ZETRABOT telemetry",
    DERIVED: "calculated from measured values only (formula stated)",
    CONFIGURED: "taken from HARVEST config.yaml",
    ESTIMATED: "inferred from measurements and/or the HARVEST model; subject to the stated assumptions",
    MISSING: "required information not present in the supplied source",
}

# Finding thresholds (|model - real| / real).  Fixed so a statement can be
# reproduced; they classify the wording, they do not hide the number.
AGREE_PCT = 10.0
MODERATE_PCT = 25.0
# Below these, a per-regime figure is reported but flagged low-confidence.
MIN_REGIME_H = 0.5
MIN_REGIME_KM = 1.0
# MainBatterySOC is reported in whole percent.
SOC_RESOLUTION_PCT = 1.0

_SIG_V = "BatteryStatus1.MainBatteryVoltage"
_SIG_I = "BatteryStatus3.BatteryCurrent"
_SIG_SOC = "BatteryStatus1.MainBatterySOC"
_SIG_KM = "MiscInfo.LifetimeKm"
_SIG_SPEED = "MiscInfo.SpeedDisplay"
_SIG_PTO = "FunctionStatus.PtoLed"
_SIG_GO = "FunctionStatus.GoLed"
_SIG_DISCH = "BatteryStatus3.DischEnrgActualSesion"
_CFG = "config.yaml tractors.model."

# Deployment-level limitations: true of the current HARVEST deployment
# whatever the data says.  Data-level ones are computed in _limitations().
DEPLOYMENT_LIMITATIONS = (
    ("aws_not_implemented",
     "Live AWS ingestion is not implemented: the actual ZETRABOT AWS service, authentication, "
     "topic/table/API structure and history/live interfaces have not been provided. CSV replay "
     "is the only functional source; the AWS adapters are scaffolds behind the same "
     "TelemetrySource interface."),
    ("no_direct_control",
     "HARVEST does not control the real ZETRABOT in this deployment: telemetry is inbound only."),
    ("deviceio_optional",
     "Modbus / OPC-UA are optional DeviceIO capabilities exercised against simulators; no "
     "ZETRABOT register or node map has been provided."),
    ("isaac_proxy",
     "Isaac Sim drives a functional tractor / mobile-robot proxy, not an exact ZETRABOT model."),
    ("capacity_semantics_unconfirmed",
     "Effective-capacity estimates assume MainBatterySOC refers to usable capacity; they require "
     "the official battery specification and confirmation of the signal semantics."),
    ("disch_counter_semantics_unconfirmed",
     "DischEnrgActualSesion is interpreted as a per-power-on session counter (resets detected "
     "as drops below half the running maximum). This is NOT confirmed by the ZETRABOT team; the "
     "raw session maxima are kept and the independent V x I integral is the headline energy."),
)


def _r(value: Optional[float], nd: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), nd)


def _div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """Division that answers ``None`` instead of raising or inventing a value."""
    if num is None or den is None or den == 0:
        return None
    return num / den


def _err_pct(model: Optional[float], real: Optional[float]) -> Optional[float]:
    if model is None or real is None or real == 0:
        return None
    return (model - real) / abs(real) * 100.0


def sha256_file(path: Any) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


# --------------------------------------------------------------------------- #
#  The builder
# --------------------------------------------------------------------------- #
class _Kpis:
    def __init__(self) -> None:
        self.items: Dict[str, Dict[str, Any]] = {}

    def add(self, key: str, label: str, group: str, value: Any, unit: str, provenance: str,
            method: str, sources: Sequence[str] = (), flags: Sequence[str] = (),
            detail: Optional[str] = None, nd: int = 3, missing: Optional[str] = None) -> None:
        if isinstance(value, float):
            value = round(value, nd)
        flags = list(flags)
        if value is None and provenance != MISSING:
            # A KPI whose inputs are absent is MISSING -- never a zero.
            provenance = MISSING
            flags.append(missing or "inputs not available in this telemetry")
        self.items[key] = {
            "key": key, "label": label, "group": group, "value": value, "unit": unit,
            "provenance": provenance, "method": method, "sources": list(sources),
            "flags": flags, "detail": detail,
        }

    def value(self, key: str) -> Any:
        item = self.items.get(key)
        return None if item is None else item["value"]


GROUPS = (
    ("mission", "Mission"),
    ("energy", "Energy"),
    ("validation", "Model validation"),
    ("operational", "Thermal / operational"),
)


def build_mission_kpis(result: Dict[str, Any], model: Optional[Dict[str, Any]] = None, *,
                       source: Optional[Dict[str, Any]] = None,
                       config_file: Optional[str] = None) -> Dict[str, Any]:
    """KPI / validation document from a :meth:`MissionAnalysis.run` result.

    ``model`` is ``tractors.model`` from config.yaml; ``source`` describes
    where the telemetry came from (file, sha256, kind) for traceability.
    """
    model = dict(model or {})
    generated = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {
        "schema": SCHEMA,
        "generated_at": generated,
        "analysis_version": result.get("analysis_version", ANALYSIS_VERSION),
        "provenance_classes": PROVENANCE,
        "input": dict(source or {}),
        "config": {"file": config_file, "tractors_model": model},
    }
    if not result or "error" in result:
        reason = (result or {}).get("error", "no analysis result")
        return {**base, "state": "empty", "error": reason, "groups": [], "kpis": {},
                "comparison": {"rows": []}, "findings": [], "calibration_proposals": [],
                "limitations": _limitations({}, []), "summary": {}}

    m = result["mission"]
    soc = result["soc"]
    power = result["power"]
    disc = result["discharged_energy"]
    reg = result["regimes"]
    ex = result.get("exclusive_regimes") or {}
    motion = result.get("motion") or {}
    dist = result.get("distance") or {}
    temps = result.get("temperatures_c") or {}
    quality = result.get("quality") or {}
    k = _Kpis()

    # ---- MISSION ----------------------------------------------------------
    missions = m.get("mission_ids") or []
    tractors = m.get("tractor_ids") or []
    k.add("mission_id", "Mission", "mission", ", ".join(missions) or None, "", MEASURED,
          "mission_id column of the telemetry records", ["mission_id"],
          flags=(["several missions in one analysis"] if len(missions) > 1 else []),
          missing="no mission_id in the records")
    k.add("tractor_id", "Tractor", "mission", ", ".join(tractors) or None, "", MEASURED,
          "tractor_id column of the telemetry records", ["tractor_id"],
          flags=(["several tractors in one analysis"] if len(tractors) > 1 else []))
    k.add("wall_span_h", "Calendar span", "mission", m.get("wall_span_h"), "h", DERIVED,
          "last timestamp - first timestamp", ["timestamp"], nd=2,
          detail=f"{m.get('first_timestamp')} .. {m.get('last_timestamp')}")
    active_h = m.get("active_h")
    k.add("duration_h", "Operating time (powered on)", "mission", active_h, "h", DERIVED,
          f"sum of inter-message intervals <= {m.get('gap_cap_s'):.0f} s (longer gaps = powered off)",
          ["timestamp"], nd=2,
          detail=f"{m.get('gaps_over_cap')} power-off gaps, {m.get('gap_hours')} h excluded")

    distance_km, distance_prov, distance_flags = None, MISSING, []
    if dist.get("delta_km") is not None:
        distance_km, distance_prov = dist["delta_km"], DERIVED
        distance_method = "LifetimeKm(last) - LifetimeKm(first) after dropping readings >25 % from the median"
        distance_sources = [_SIG_KM]
        distance_flags.append("counter resolution 0.1 km")
        if dist.get("outliers_discarded"):
            distance_flags.append(f"{dist['outliers_discarded']} counter outlier(s) discarded")
    elif motion.get("speed_integral_km"):
        distance_km, distance_prov = motion["speed_integral_km"], ESTIMATED
        distance_method = "time integral of the held SpeedDisplay (no LifetimeKm available)"
        distance_sources = [_SIG_SPEED]
        distance_flags.append("SpeedDisplay is an integer display value")
    else:
        distance_method = "needs LifetimeKm or SpeedDisplay"
        distance_sources = []
    distance_flags.append("no GPS in the export")
    k.add("distance_km", "Distance", "mission", distance_km, "km", distance_prov, distance_method,
          distance_sources, distance_flags, nd=2,
          detail=(f"speed integral {motion.get('speed_integral_km')} km (cross-check)"
                  if distance_prov == DERIVED else None),
          missing="no distance signal (LifetimeKm / SpeedDisplay) in the telemetry")
    moving_h = motion.get("moving_h") if motion.get("speed_observed") else None
    k.add("moving_h", "Moving time", "mission", moving_h, "h", DERIVED,
          "time with held SpeedDisplay > 0", [_SIG_SPEED], nd=2,
          missing="SpeedDisplay never observed")
    pto_h = reg["pto_active"]["hours"] if motion.get("pto_observed") else None
    k.add("pto_active_h", "PTO-active time", "mission", pto_h, "h", DERIVED,
          "time with held PtoLed = 1", [_SIG_PTO], nd=2, missing="PtoLed never observed")
    k.add("position", "GPS / location", "mission", None, "", MISSING,
          "no latitude / longitude signal in the supplied export", [],
          missing="the Zetrack PDF route map comes from a stream not in the export")

    # ---- ENERGY -----------------------------------------------------------
    k.add("soc_initial_pct", "Initial SOC", "energy", soc.get("first_pct"), "%", MEASURED,
          "first MainBatterySOC sample", [_SIG_SOC], nd=1, missing="no SOC samples")
    k.add("soc_final_pct", "Final SOC", "energy", soc.get("last_pct"), "%", MEASURED,
          "last MainBatterySOC sample", [_SIG_SOC], nd=1, missing="no SOC samples")
    soc_delta = soc.get("delta_pct")
    k.add("soc_delta_pct", "SOC decrease", "energy", soc_delta, "pts", DERIVED,
          "initial SOC - final SOC", [_SIG_SOC],
          [f"SOC reported in whole percent (+/-{SOC_RESOLUTION_PCT:.0f} pt per reading)"], nd=1,
          detail=(f"{soc.get('first_pct')} % -> {soc.get('last_pct')} % ({soc.get('samples')} samples)"
                  if soc_delta is not None else None),
          missing="fewer than one SOC sample")
    energy = power.get("integrated_kwh") if power.get("samples") else None
    k.add("energy_kwh", "Discharged energy", "energy", energy, "kWh", DERIVED,
          f"integral of V x I dt (sample-and-hold, intervals <= {m.get('gap_cap_s'):.0f} s; "
          "0 V dropouts rejected)", [_SIG_V, _SIG_I],
          ([f"{quality['voltage_dropouts_rejected']} 0 V dropouts rejected"]
           if quality.get("voltage_dropouts_rejected") else []),
          missing="needs both MainBatteryVoltage and BatteryCurrent")
    counter = disc.get("sum_of_sessions_kwh") if disc.get("samples") else None
    k.add("energy_counter_kwh", "Discharged energy (session counter)", "energy", counter, "kWh",
          ESTIMATED, "sum of per-session maxima of DischEnrgActualSesion", [_SIG_DISCH],
          ["session/reset semantics NOT confirmed by the ZETRABOT team"],
          detail=f"raw session maxima {disc.get('session_maxima_kwh')} kWh",
          missing="DischEnrgActualSesion not in the telemetry")
    k.add("mean_power_kw", "Mean battery power", "energy", _div(energy, active_h), "kW", DERIVED,
          "discharged energy / operating time (= energy per operating hour)", [_SIG_V, _SIG_I],
          nd=2, missing="needs energy and a non-zero operating time")
    k.add("peak_power_kw", "Peak battery power", "energy", power.get("peak_kw"), "kW", DERIVED,
          "max of V x I over the samples", [_SIG_V, _SIG_I],
          ["single sample; see p99 for a robust peak"], nd=2,
          detail=f"p99 {power.get('percentiles_kw', {}).get('p99')} kW at the 99th percentile",
          missing="needs voltage and current")
    k.add("p99_power_kw", "99th-percentile power", "energy",
          power.get("percentiles_kw", {}).get("p99"), "kW", DERIVED,
          "99th percentile of V x I samples", [_SIG_V, _SIG_I], nd=2)
    k.add("energy_per_km", "Energy per km", "energy", _div(energy, distance_km), "kWh/km",
          DERIVED if distance_prov == DERIVED else ESTIMATED,
          "discharged energy / distance (all work, incl. PTO)", [_SIG_V, _SIG_I] + distance_sources,
          nd=2, missing="needs energy and a non-zero distance")
    k.add("energy_per_operating_h", "Energy per operating hour", "energy",
          _div(energy, active_h), "kWh/h", DERIVED, "discharged energy / operating time",
          [_SIG_V, _SIG_I], nd=2, missing="needs a non-zero operating time")
    k.add("energy_per_moving_h", "Energy per moving hour", "energy", _div(energy, moving_h),
          "kWh/h", DERIVED, "discharged energy / moving time", [_SIG_V, _SIG_I, _SIG_SPEED], nd=2,
          missing="needs a non-zero moving time")
    pto_e = reg["pto_active"]["energy_kwh"] if pto_h else None
    k.add("energy_pto_kwh", "Energy while PTO active", "energy", pto_e, "kWh", DERIVED,
          "V x I integral while PtoLed = 1 (includes traction while working)",
          [_SIG_V, _SIG_I, _SIG_PTO], nd=2,
          detail=(f"mean {reg['pto_active']['mean_power_kw']} kW" if pto_e is not None else None),
          missing="needs PtoLed and V x I")
    mov_e = reg["moving_speed_gt_0"]["energy_kwh"] if moving_h else None
    k.add("energy_moving_kwh", "Energy while moving", "energy", mov_e, "kWh", DERIVED,
          "V x I integral while SpeedDisplay > 0 (incl. PTO work done while moving)",
          [_SIG_V, _SIG_I, _SIG_SPEED], nd=2,
          detail=(f"mean {reg['moving_speed_gt_0']['mean_power_kw']} kW" if mov_e is not None else None),
          missing="needs SpeedDisplay and V x I")
    idle = ex.get("stationary_no_pto") or {}
    idle_ok = motion.get("speed_observed") and motion.get("pto_observed") and idle.get("hours")
    idle_flags = []
    if idle_ok and idle["hours"] < MIN_REGIME_H:
        idle_flags.append(f"low confidence: only {idle['hours']} h")
    k.add("energy_idle_kwh", "Energy while idle", "energy", idle.get("energy_kwh") if idle_ok else None,
          "kWh", DERIVED, "V x I integral while powered on, stationary, PTO off",
          [_SIG_V, _SIG_I, _SIG_SPEED, _SIG_PTO], idle_flags, nd=3,
          detail=(f"{idle.get('hours')} h at mean {idle.get('mean_power_kw')} kW; "
                  f"{(idle.get('of_which_drive_engaged') or {}).get('hours')} h of it with drive engaged"
                  if idle_ok else None),
          missing="needs SpeedDisplay, PtoLed and powered stationary time")
    capacity = model.get("battery_capacity_kwh")
    capacity = float(capacity) if capacity else None
    remaining = _div(soc.get("last_pct"), 100.0)
    remaining = remaining * capacity if remaining is not None and capacity else None
    k.add("estimated_remaining_energy_kwh", "Estimated remaining energy", "energy", remaining,
          "kWh", ESTIMATED, f"final SOC x configured capacity ({capacity} kWh)",
          [_SIG_SOC, _CFG + "battery_capacity_kwh"],
          ["not a measured value: depends on the configured capacity"], nd=2,
          detail=f"SOC x configured capacity ({capacity} kWh)",
          missing="needs a SOC sample and a configured capacity")

    # ---- MODEL VALIDATION -------------------------------------------------
    k.add("configured_capacity_kwh", "Configured battery capacity", "validation", capacity, "kWh",
          CONFIGURED, "tractors.model.battery_capacity_kwh", [_CFG + "battery_capacity_kwh"],
          nd=2, missing="battery_capacity_kwh not configured")
    cap_est = _div(energy, _div(soc_delta, 100.0)) if soc_delta and soc_delta > 0 else None
    cap_lo = cap_hi = None
    if cap_est is not None:
        cap_lo = energy / ((soc_delta + SOC_RESOLUTION_PCT) / 100.0)
        cap_hi = (energy / ((soc_delta - SOC_RESOLUTION_PCT) / 100.0)
                  if soc_delta > SOC_RESOLUTION_PCT else None)
    cap_counter = _div(counter, _div(soc_delta, 100.0)) if soc_delta and soc_delta > 0 else None
    k.add("estimated_capacity_kwh", "Effective capacity (estimate)", "validation", cap_est, "kWh",
          ESTIMATED, "discharged energy (V x I) / (SOC decrease / 100)", [_SIG_V, _SIG_I, _SIG_SOC],
          ["one mission only", "requires official battery specification and SOC-semantics confirmation"],
          nd=2,
          detail=(f"range {_r(cap_lo, 1)}-{_r(cap_hi, 1)} kWh for +/-1 SOC pt; "
                  f"session counter gives {_r(cap_counter, 2)} kWh") if cap_est else None,
          missing=("SOC did not decrease -- capacity cannot be inferred"
                   if soc_delta is not None else "needs SOC samples and energy"))

    # HARVEST model applied to the measured operating profile.
    model_doc = _model_energy(ex, model, distance_prov != MISSING)
    model_e = model_doc["model_kwh"]
    real_cmp = model_doc["real_kwh"]
    k.add("model_energy_kwh", "HARVEST model energy", "validation", model_e, "kWh", ESTIMATED,
          model_doc["formula"],
          [_CFG + p for p in ("pto_power_kw", "driving_kwh_per_km", "idle_kwh_per_h")] +
          [_SIG_PTO, _SIG_SPEED, _SIG_KM],
          model_doc["flags"], nd=2, missing=model_doc.get("missing"),
          detail="HARVEST energy model applied to the measured PTO / driving / idle profile")
    diff = (model_e - real_cmp) if model_e is not None and real_cmp is not None else None
    k.add("energy_error_kwh", "Model - measured energy", "validation", diff, "kWh", DERIVED,
          "HARVEST model energy - measured energy (same intervals)", [], nd=2,
          missing="needs both the model and the measured energy")
    k.add("energy_error_pct", "Energy error", "validation", _err_pct(model_e, real_cmp), "%",
          DERIVED, "(model - measured) / measured x 100; + = model predicts more", [], nd=1,
          missing="needs both the model and a non-zero measured energy")
    model_soc = _div(model_e, capacity)
    model_soc = model_soc * 100.0 if model_soc is not None else None
    k.add("model_soc_delta_pct", "Model SOC decrease", "validation", model_soc, "pts", ESTIMATED,
          "HARVEST model energy / configured capacity x 100",
          [_CFG + "battery_capacity_kwh"], nd=1, missing="needs the model energy and a capacity")
    soc_err = (model_soc - soc_delta) if model_soc is not None and soc_delta is not None else None
    k.add("soc_error_pct_points", "SOC prediction error", "validation", soc_err, "pts", DERIVED,
          "model SOC decrease - measured SOC decrease", [_SIG_SOC],
          [f"+/-{SOC_RESOLUTION_PCT:.0f} pt SOC resolution"], nd=1,
          missing="needs the model SOC decrease and measured SOC")
    soc_from_energy = _div(energy, capacity)
    soc_from_energy = soc_from_energy * 100.0 if soc_from_energy is not None else None
    cap_err = (soc_from_energy - soc_delta) if soc_from_energy is not None and soc_delta is not None else None
    k.add("soc_capacity_check_pts", "SOC check (measured energy / configured capacity)",
          "validation", cap_err, "pts", DERIVED,
          "measured energy / configured capacity x 100 - measured SOC decrease "
          "(isolates the capacity assumption from the consumption model)",
          [_SIG_V, _SIG_I, _SIG_SOC, _CFG + "battery_capacity_kwh"], nd=1,
          detail=(f"predicts {_r(soc_from_energy, 1)} pts vs {soc_delta} measured"
                  if soc_from_energy is not None else None),
          missing="needs energy, SOC and a capacity")
    agreement = _err_pct(counter, energy)
    k.add("counter_vs_vxi_pct", "Session counter vs V x I", "validation", agreement, "%", DERIVED,
          "(session-counter energy - V x I energy) / V x I energy x 100", [_SIG_DISCH, _SIG_V, _SIG_I],
          ["cross-check only: counter semantics unconfirmed"], nd=2,
          missing="needs both energy reconstructions")

    # ---- THERMAL / OPERATIONAL -------------------------------------------
    bat = temps.get("battery") or {}
    k.add("battery_temp_max_c", "Battery temperature max", "operational", bat.get("max"), "degC",
          MEASURED, "max of MainBatteryTemp", ["BatteryStatus1.MainBatteryTemp"],
          _zero_flag(quality, "MainBatteryTemp"), nd=1,
          detail=f"mean {bat.get('mean')} degC" if bat else None)
    k.add("battery_temp_mean_c", "Battery temperature mean", "operational", bat.get("mean"), "degC",
          DERIVED, "sample mean of MainBatteryTemp", ["BatteryStatus1.MainBatteryTemp"],
          _zero_flag(quality, "MainBatteryTemp"), nd=1)
    motors = temps.get("motors") or {}
    traction = {n: v for n, v in motors.items() if n.startswith("TempT") and v}
    hottest = max(traction.items(), key=lambda kv: kv[1]["max"]) if traction else None
    k.add("motor_temp_max_c", "Traction motor temperature max", "operational",
          hottest[1]["max"] if hottest else None, "degC", MEASURED,
          "max over TempT1..TempT4", ["MotorTemp.TempT1..TempT4"],
          _zero_flag(quality, *(hottest[0],) if hottest else ()), nd=1,
          detail=(f"{hottest[0].replace('Temp', '')}; " + ", ".join(
              f"{n.replace('Temp', '')} {v['max']:.0f}" for n, v in sorted(traction.items())))
          if hottest else None)
    pto_t = motors.get("TempPTO") or {}
    k.add("pto_motor_temp_max_c", "PTO motor temperature max", "operational", pto_t.get("max"),
          "degC", MEASURED, "max of TempPTO", ["MotorTemp.TempPTO"], _zero_flag(quality, "TempPTO"),
          nd=1)
    oil = temps.get("oil") or {}
    k.add("oil_temp_max_c", "Oil temperature max", "operational", oil.get("max"), "degC", MEASURED,
          "max of OilTemp", ["MiscInfo.OilTemp"], _zero_flag(quality, "OilTemp"), nd=1)
    k.add("pto_utilisation_pct", "PTO utilisation", "operational",
          (_div(pto_h, active_h) or 0) * 100 if _div(pto_h, active_h) is not None else None, "%",
          DERIVED, "PTO-active time / operating time x 100", [_SIG_PTO], nd=1,
          missing="needs PtoLed and a non-zero operating time")
    k.add("peak_current_a", "Peak battery current", "operational", power.get("peak_current_a"), "A",
          MEASURED, "max of BatteryCurrent", [_SIG_I], nd=1)
    k.add("mean_moving_speed_kmh", "Mean speed while moving", "operational",
          motion.get("mean_moving_speed_kmh"), "km/h", DERIVED, "distance / moving time",
          [_SIG_KM, _SIG_SPEED], nd=2,
          detail=f"max displayed {(result.get('speed_kmh') or {}).get('max')} km/h")
    shares = {}
    for name in ("pto_moving", "pto_stationary", "moving_no_pto", "stationary_no_pto", "unclassified"):
        hours = (ex.get(name) or {}).get("hours")
        share = _div(hours, active_h)
        shares[name] = _r(share * 100, 1) if share is not None else None
    k.add("operating_state_shares_pct", "Operating-state shares", "operational",
          shares if active_h else None, "%", DERIVED,
          "exclusive regime time / operating time x 100", [_SIG_PTO, _SIG_SPEED],
          detail=" · ".join(f"{n.replace('_', ' ')} {v}%" for n, v in shares.items() if v))
    k.add("charging_state", "Charging state", "operational", None, "", MISSING,
          "no charger / charging-state signal in the export", [],
          missing="no charging-state stream; SOC never rose"
          if not (result.get("charging_evidence") or {}).get("soc_increase_events")
          else "no charging-state stream")

    comparison = _comparison(k, result, model, model_doc, capacity)
    findings = _findings(k, result, model, model_doc, comparison, capacity, (cap_lo, cap_hi))
    proposals = _proposals(k, result, model, model_doc, (cap_lo, cap_hi))
    limitations = _limitations(result, missions)
    groups = [{"id": gid, "title": title,
               "kpis": [key for key, item in k.items.items() if item["group"] == gid]}
              for gid, title in GROUPS]
    summary_keys = [
        "mission_id", "tractor_id", "distance_km", "duration_h", "wall_span_h", "moving_h",
        "pto_active_h", "soc_initial_pct", "soc_final_pct", "soc_delta_pct", "energy_kwh",
        "energy_counter_kwh", "mean_power_kw", "peak_power_kw", "p99_power_kw", "energy_per_km",
        "energy_per_operating_h", "energy_per_moving_h", "energy_pto_kwh", "energy_moving_kwh",
        "energy_idle_kwh", "configured_capacity_kwh", "estimated_capacity_kwh", "model_energy_kwh",
        "energy_error_kwh", "energy_error_pct", "model_soc_delta_pct", "soc_error_pct_points",
        "battery_temp_max_c", "motor_temp_max_c", "pto_utilisation_pct", "peak_current_a",
    ]
    summary = {key: k.value(key) for key in summary_keys}
    summary["estimated_capacity_low_kwh"] = _r(cap_lo, 2)
    summary["estimated_capacity_high_kwh"] = _r(cap_hi, 2)
    return {
        **base,
        "state": "ok",
        "groups": groups,
        "kpis": k.items,
        "comparison": comparison,
        "findings": findings,
        "calibration_proposals": proposals,
        "calibration_note": CALIBRATION_NOTE,
        "limitations": limitations,
        "summary": summary,
        "summary_provenance": {key: k.items[key]["provenance"] for key in summary_keys},
        "summary_flags": {key: k.items[key]["flags"] for key in summary_keys if k.items[key]["flags"]},
    }


def _zero_flag(quality: Dict[str, Any], *names: str) -> List[str]:
    zeros = (quality.get("zero_temperature_readings_excluded") or {})
    count = sum(zeros.get(n, 0) for n in names)
    return [f"{count} exact-0 degC dropout reading(s) excluded"] if count else []


# --------------------------------------------------------------------------- #
#  HARVEST model applied to the measured profile
# --------------------------------------------------------------------------- #
def _model_energy(ex: Dict[str, Any], model: Dict[str, Any], have_distance: bool) -> Dict[str, Any]:
    """HARVEST's energy model (main.py) evaluated on the measured regimes.

    HARVEST charges ``driving_kwh_per_km`` per km driven, ``pto_power_kw``
    per hour of PTO work and ``idle_kwh_per_h`` per hour without a task
    (``Scheduler._task_energy`` and ``Simulator`` in main.py).  Applied to
    the exclusive regimes of the real mission::

        E_model = pto_power_kw     x (PTO-active h)
                + driving_kwh_per_km x (distance km)
                + idle_kwh_per_h   x (stationary, PTO-off h)

    Distance per regime = the LifetimeKm distance allocated by the speed
    integral (ESTIMATED).  Unclassified time (PTO / speed state not yet
    observed) is left out of BOTH sides of the comparison.
    """
    pto_kw = model.get("pto_power_kw")
    drive = model.get("driving_kwh_per_km")
    idle_kw = model.get("idle_kwh_per_h")
    formula = ("pto_power_kw x PTO-active h + driving_kwh_per_km x distance km "
               "+ idle_kwh_per_h x stationary PTO-off h")
    out: Dict[str, Any] = {"formula": formula, "flags": [], "per_regime": {},
                           "model_kwh": None, "real_kwh": None}
    names = ("pto_moving", "pto_stationary", "moving_no_pto", "stationary_no_pto")
    if not ex or any(n not in ex for n in names):
        out["missing"] = "operating regimes unavailable (needs PtoLed and SpeedDisplay)"
        return out
    if sum(ex[n].get("hours") or 0.0 for n in names) <= 0:
        out["missing"] = "no classified operating time (PTO / speed state never observed while powered)"
        return out
    absent = [p for p, v in (("pto_power_kw", pto_kw), ("driving_kwh_per_km", drive),
                             ("idle_kwh_per_h", idle_kw)) if v is None]
    if absent:
        out["missing"] = "model parameters not configured: " + ", ".join(absent)
        return out
    pto_kw, drive, idle_kw = float(pto_kw), float(drive), float(idle_kw)

    def km(name: str) -> Optional[float]:
        reg = ex[name]
        if reg.get("allocated_distance_km") is not None:
            return reg["allocated_distance_km"]
        return reg.get("speed_integral_km")

    if not have_distance and (ex["pto_moving"]["hours"] or ex["moving_no_pto"]["hours"]):
        out["missing"] = "no distance signal: the driving term cannot be evaluated"
        return out
    per = {
        "pto_moving": pto_kw * ex["pto_moving"]["hours"] + drive * (km("pto_moving") or 0.0),
        "pto_stationary": pto_kw * ex["pto_stationary"]["hours"],
        "moving_no_pto": drive * (km("moving_no_pto") or 0.0),
        "stationary_no_pto": idle_kw * ex["stationary_no_pto"]["hours"],
    }
    out["per_regime"] = {n: _r(v) for n, v in per.items()}
    out["model_kwh"] = sum(per.values())
    out["real_kwh"] = sum(ex[n]["energy_kwh"] for n in names)
    out["real_per_regime"] = {n: ex[n]["energy_kwh"] for n in names}
    out["distance_km"] = {n: km(n) for n in ("pto_moving", "moving_no_pto")}
    unclassified = (ex.get("unclassified") or {}).get("hours") or 0.0
    if unclassified > 0:
        out["flags"].append(f"{unclassified} h unclassified (state not yet observed) left out of both sides")
    out["flags"].append("regime distance allocated from the speed integral")
    return out


# --------------------------------------------------------------------------- #
#  Real vs model table
# --------------------------------------------------------------------------- #
def _row(metric: str, unit: str, real: Optional[float], real_prov: str, model: Optional[float],
         status: str = "compared", note: str = "", nd: int = 2, flags: Sequence[str] = (),
         model_prov: str = ESTIMATED) -> Dict[str, Any]:
    diff = (model - real) if model is not None and real is not None else None
    err = _err_pct(model, real)
    if status == "compared" and (real is None or model is None):
        status = "unavailable"
    return {"metric": metric, "unit": unit, "real": _r(real, nd), "real_provenance": real_prov,
            "model": _r(model, nd), "model_provenance": model_prov if model is not None else MISSING,
            "difference": _r(diff, nd), "error_pct": _r(err, 1), "status": status, "note": note,
            "flags": list(flags)}


def _comparison(k: _Kpis, result: Dict[str, Any], model: Dict[str, Any],
                md: Dict[str, Any], capacity: Optional[float]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    active_h = k.value("duration_h")
    moving_h = k.value("moving_h")
    distance = k.value("distance_km")
    real_e, model_e = md.get("real_kwh"), md.get("model_kwh")
    rows.append(_row("Total energy", "kWh", real_e, DERIVED, model_e,
                     note="V x I integral vs HARVEST model on the same powered intervals",
                     flags=md.get("flags", [])))
    model_soc = k.value("model_soc_delta_pct")
    rows.append(_row("SOC decrease", "pts", k.value("soc_delta_pct"), DERIVED, model_soc, nd=1,
                     note="model energy / configured capacity"))
    rows.append(_row("Mean power (powered on)", "kW", _div(real_e, active_h), DERIVED,
                     _div(model_e, active_h), note="total energy / operating time (not independent of total energy)"))
    rows.append(_row("Peak power", "kW", k.value("peak_power_kw"), DERIVED, None, status="unavailable",
                     note=("HARVEST's energy model is an average-power model and does not predict peaks; "
                           f"configured max_power_kw {model.get('max_power_kw')} kW is a limit, not a prediction")))
    rows.append(_row("Energy per km", "kWh/km", _div(real_e, distance), DERIVED, _div(model_e, distance),
                     note="all work (PTO + driving + idle) per km travelled"))
    rows.append(_row("Energy per moving hour", "kWh/h", _div(real_e, moving_h), DERIVED,
                     _div(model_e, moving_h), note="total energy / moving time"))
    per, real_per = md.get("per_regime") or {}, md.get("real_per_regime") or {}
    ex = result.get("exclusive_regimes") or {}

    def regime_flags(*names: str) -> List[str]:
        hours = sum((ex.get(n) or {}).get("hours") or 0.0 for n in names)
        return [f"low confidence: only {hours:.2f} h in this regime"] if hours and hours < MIN_REGIME_H else []

    if per:
        rows.append(_row("PTO-active consumption", "kWh",
                         (real_per.get("pto_moving") or 0) + (real_per.get("pto_stationary") or 0), DERIVED,
                         per["pto_moving"] + per["pto_stationary"],
                         note="model: pto_power_kw x PTO h + driving_kwh_per_km x km driven with PTO on",
                         flags=regime_flags("pto_moving", "pto_stationary") + ["distance allocated from the speed integral"]))
        rows.append(_row("Driving without PTO", "kWh", real_per.get("moving_no_pto"), DERIVED,
                         per["moving_no_pto"], note="model: driving_kwh_per_km x km driven with PTO off",
                         flags=regime_flags("moving_no_pto") + ["distance allocated from the speed integral"]))
        rows.append(_row("Idle (stationary, PTO off)", "kWh", real_per.get("stationary_no_pto"), DERIVED,
                         per["stationary_no_pto"], nd=3, note="model: idle_kwh_per_h x idle h",
                         flags=regime_flags("stationary_no_pto")))
    else:
        for metric in ("PTO-active consumption", "Driving without PTO", "Idle (stationary, PTO off)"):
            rows.append(_row(metric, "kWh", None, MISSING, None, status="unavailable",
                             note=md.get("missing", "regimes unavailable")))
    gaps = [g for g in result.get("power_off_gaps") or []
            if g.get("soc_before_pct") is not None and g.get("soc_after_pct") is not None]
    idle_kw = model.get("idle_kwh_per_h")
    if gaps and capacity and idle_kw is not None:
        gap_h = sum(g["hours"] for g in gaps)
        real_drop = sum(g["soc_before_pct"] - g["soc_after_pct"] for g in gaps)
        rows.append(_row("SOC change while powered off", "pts", real_drop, DERIVED,
                         float(idle_kw) * gap_h / capacity * 100.0, nd=1,
                         note=(f"{len(gaps)} gaps, {gap_h:.1f} h; HARVEST applies idle_kwh_per_h to every "
                               "tractor without a task. The measured drop is an UPPER BOUND on parked drain: "
                               "it also contains SOC reporting lag / BMS re-estimation at power-on"),
                         flags=["upper bound", f"+/-{SOC_RESOLUTION_PCT:.0f} pt per SOC reading"]))
    else:
        rows.append(_row("SOC change while powered off", "pts", None, MISSING, None, status="unavailable",
                         note="no power-off gap with SOC on both sides"))
    rows.append(_row("Battery capacity", "kWh", k.value("estimated_capacity_kwh"), ESTIMATED, capacity,
                     note="configured capacity vs effective capacity implied by energy / SOC decrease",
                     flags=["real side is an ESTIMATE from one mission"], model_prov=CONFIGURED))
    rows.append(_row("Charging behaviour", "", None, MISSING, None, status="unavailable",
                     note="no charging-state stream and no charging cycle in the data"))
    return {
        "rows": rows,
        "formula": md.get("formula"),
        "sign_convention": "difference = model - real; error_pct = (model - real) / |real| x 100",
        "per_regime_model_kwh": md.get("per_regime"),
        "per_regime_real_kwh": md.get("real_per_regime"),
    }


# --------------------------------------------------------------------------- #
#  Rule-based findings
# --------------------------------------------------------------------------- #
def _band(err: float) -> str:
    a = abs(err)
    return "ok" if a <= AGREE_PCT else ("info" if a <= MODERATE_PCT else "warning")


def _more_less(err: float) -> str:
    return "higher" if err > 0 else "lower"


def _findings(k: _Kpis, result: Dict[str, Any], model: Dict[str, Any], md: Dict[str, Any],
              comparison: Dict[str, Any], capacity: Optional[float],
              cap_range: tuple) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def add(fid: str, level: str, text: str, **evidence: Any) -> None:
        out.append({"id": fid, "level": level, "text": text, "evidence": evidence})

    rows = {r["metric"]: r for r in comparison["rows"]}
    total = rows.get("Total energy") or {}
    if total.get("error_pct") is not None:
        err = total["error_pct"]
        add("model_energy", _band(err),
            f"HARVEST model predicts {total['model']} kWh for the measured operating profile, "
            f"{abs(err):.1f} % {_more_less(err)} than the measured {total['real']} kWh.",
            model_kwh=total["model"], real_kwh=total["real"], error_pct=err)
    else:
        add("model_energy", "info", "Model-vs-measured energy not computable: " +
            (md.get("missing") or "inputs missing") + ".")

    cap_est = k.value("estimated_capacity_kwh")
    lo, hi = cap_range
    if cap_est is not None and capacity:
        dev = (capacity - cap_est) / cap_est * 100.0
        if lo is not None and hi is not None and lo <= capacity <= hi:
            add("capacity", "ok",
                f"Configured battery capacity ({capacity} kWh) is consistent with the effective capacity "
                f"implied by this mission ({cap_est:.1f} kWh; {lo:.1f}-{hi:.1f} kWh for +/-1 SOC point). "
                "Requires confirmation from the ZETRABOT battery specification.",
                configured=capacity, estimated=cap_est)
        elif abs(dev) <= AGREE_PCT:
            add("capacity", "ok",
                f"Battery-capacity assumption broadly consistent with the observed mission: configured "
                f"{capacity} kWh is {abs(dev):.1f} % {'above' if dev > 0 else 'below'} the effective capacity "
                f"implied by this mission ({cap_est:.1f} kWh), just outside the +/-1 SOC-point band "
                f"({_r(lo, 1)}-{_r(hi, 1)} kWh). No change warranted on one mission; the official battery "
                "specification is still required.",
                configured=capacity, estimated=cap_est, deviation_pct=_r(dev, 1))
        else:
            add("capacity", "warning",
                f"Configured battery capacity ({capacity} kWh) is {abs(dev):.1f} % "
                f"{'above' if dev > 0 else 'below'} the effective capacity implied by this mission "
                f"({cap_est:.1f} kWh; +/-1 SOC point gives "
                f"{_r(lo, 1)}-{_r(hi, 1)} kWh). One mission; requires confirmation before any change.",
                configured=capacity, estimated=cap_est, deviation_pct=_r(dev, 1))
    else:
        add("capacity", "info", "Battery capacity cannot be checked: no SOC decrease with energy data.")

    soc_err = k.value("soc_error_pct_points")
    if soc_err is not None:
        add("soc_prediction", "ok" if abs(soc_err) <= 2 * SOC_RESOLUTION_PCT else "warning",
            f"Model SOC decrease is {abs(soc_err):.1f} points {_more_less(soc_err)} than measured "
            f"({k.value('model_soc_delta_pct')} vs {k.value('soc_delta_pct')} pts). "
            f"With the measured energy and the configured capacity the SOC error is "
            f"{k.value('soc_capacity_check_pts')} pts -- the gap is the consumption model, not the capacity."
            if k.value("soc_capacity_check_pts") is not None and abs(k.value("soc_capacity_check_pts")) < abs(soc_err)
            else f"Model SOC decrease is {abs(soc_err):.1f} points {_more_less(soc_err)} than measured.",
            error_pts=soc_err)

    ex = result.get("exclusive_regimes") or {}
    pto_real = (result.get("regimes") or {}).get("pto_active", {}).get("mean_power_kw")
    pto_cfg = model.get("pto_power_kw")
    if pto_real and pto_cfg:
        dev = (float(pto_cfg) - pto_real) / pto_real * 100.0
        add("pto_power", _band(dev),
            f"Configured PTO power ({pto_cfg} kW) is {abs(dev):.0f} % {'above' if dev > 0 else 'below'} the "
            f"measured mean battery power while the PTO was active ({pto_real} kW), which itself includes "
            "traction -- i.e. an upper bound on PTO power alone." if dev > 0 else
            f"Configured PTO power ({pto_cfg} kW) is below the measured mean battery power while the PTO "
            f"was active ({pto_real} kW, which includes traction).",
            configured=pto_cfg, measured_upper_bound=pto_real)

    drive_cfg = model.get("driving_kwh_per_km")
    mov = ex.get("moving_no_pto") or {}
    km = mov.get("allocated_distance_km") or mov.get("speed_integral_km")
    if drive_cfg and mov.get("hours") and km:
        rate = mov["energy_kwh"] / km
        low = mov["hours"] < MIN_REGIME_H or km < MIN_REGIME_KM
        add("driving", "info" if low else _band((float(drive_cfg) - rate) / rate * 100),
            f"Driving with the PTO off used {rate:.2f} kWh/km over {km:.2f} km ({mov['hours']:.2f} h) vs "
            f"configured driving_kwh_per_km {drive_cfg}" + (" -- too little data for a conclusion." if low else "."),
            measured_kwh_per_km=_r(rate), km=km, hours=mov["hours"], low_confidence=low)

    idle = ex.get("stationary_no_pto") or {}
    idle_cfg = model.get("idle_kwh_per_h")
    if idle_cfg is not None and idle.get("hours"):
        add("idle", "info",
            f"Powered-on idle (stationary, PTO off) averaged {idle['mean_power_kw']} kW over {idle['hours']} h "
            f"vs configured idle_kwh_per_h {idle_cfg}; "
            f"{(idle.get('of_which_drive_engaged') or {}).get('mean_power_kw')} kW with drive engaged, "
            "near zero with drive off.", measured_kw=idle["mean_power_kw"], hours=idle["hours"])
    off = rows.get("SOC change while powered off") or {}
    if off.get("status") == "compared":
        add("powered_off_drain", "warning" if (off["model"] or 0) > 2 * max(off["real"] or 0, 1) else "info",
            f"While powered off the SOC fell by at most {off['real']} points; HARVEST's idle drain would "
            f"predict {off['model']} points for the same time ({off['note'].split(';')[0]}). "
            "The data do not support applying idle_kwh_per_h to a switched-off tractor.",
            measured_pts=off["real"], model_pts=off["model"])

    agree = k.value("counter_vs_vxi_pct")
    if agree is not None:
        add("session_counter", "ok" if abs(agree) <= 5 else "warning",
            f"Sum of DischEnrgActualSesion session maxima ({k.value('energy_counter_kwh')} kWh) and the "
            f"independent V x I integral ({k.value('energy_kwh')} kWh) differ by {abs(agree):.1f} %: "
            "consistent with the per-session interpretation, which remains UNCONFIRMED by the ZETRABOT team.",
            difference_pct=agree)

    peak = k.value("peak_power_kw")
    pmax = model.get("max_power_kw")
    if peak is not None and pmax:
        add("peak_power", "ok" if peak <= float(pmax) else "warning",
            f"Peak battery power {peak} kW ({peak / float(pmax) * 100:.0f} % of configured max_power_kw "
            f"{pmax} kW); p99 {k.value('p99_power_kw')} kW.", peak_kw=peak, max_power_kw=pmax)

    speed = k.value("mean_moving_speed_kmh")
    eco = model.get("eco_speed_kmh")
    if speed is not None and eco:
        add("speed", "info",
            f"Mean speed while moving was {speed} km/h (max displayed "
            f"{(result.get('speed_kmh') or {}).get('max')} km/h); HARVEST uses eco_speed_kmh {eco} km/h "
            "for transit times, so field-work transit durations are not represented by this mission.",
            measured_kmh=speed, eco_speed_kmh=eco)

    if not result.get("position_signals_seen"):
        add("gps", "limitation", "GPS stream unavailable: no position signal in the supplied telemetry; "
            "distance comes from the LifetimeKm counter.")
    ce = result.get("charging_evidence") or {}
    if not ce.get("charging_signals_seen"):
        add("charging", "limitation",
            "Insufficient data to validate charging behaviour: no charging-state signal"
            + ("" if ce.get("soc_increase_events") else " and the SOC never increased (no charging cycle)") + ".")
    missions = (result.get("mission") or {}).get("mission_ids") or []
    if len(missions) <= 1:
        add("single_mission", "limitation",
            "Only one mission available: generalisation not established; parameters fitted to it "
            "would be in-sample.")
    return out


# --------------------------------------------------------------------------- #
#  Calibration proposals (never applied)
# --------------------------------------------------------------------------- #
def _proposals(k: _Kpis, result: Dict[str, Any], model: Dict[str, Any], md: Dict[str, Any],
               cap_range: tuple) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    status = "proposal for review -- NOT applied to config.yaml"

    def add(param: str, current: Any, proposed: Optional[float], unit: str, basis: str,
            confidence: str, nd: int = 2) -> None:
        if proposed is None:
            return
        out.append({"parameter": f"tractors.model.{param}", "current": current,
                    "proposed": _r(proposed, nd), "unit": unit, "basis": basis,
                    "confidence": confidence, "status": status})

    lo, hi = cap_range
    cap = model.get("battery_capacity_kwh")
    est = k.value("estimated_capacity_kwh")
    material = est is not None and cap and abs(float(cap) - est) / est * 100.0 > AGREE_PCT
    if material and not (lo is not None and hi is not None and lo <= float(cap) <= hi):
        add("battery_capacity_kwh", cap, est, "kWh",
            f"effective capacity implied by this mission (range {_r(lo, 1)}-{_r(hi, 1)} kWh); "
            "replace only after the official battery specification confirms SOC semantics", "low (1 mission)")
    pto = (result.get("regimes") or {}).get("pto_active", {})
    if pto.get("hours") and model.get("pto_power_kw") is not None:
        add("pto_power_kw", model.get("pto_power_kw"), pto.get("mean_power_kw"), "kW",
            f"mean battery power while PtoLed = 1 over {pto['hours']} h (includes traction: upper bound)",
            "low (1 mission)" if pto["hours"] >= MIN_REGIME_H else "very low (<0.5 h)")
    ex = result.get("exclusive_regimes") or {}
    mov = ex.get("moving_no_pto") or {}
    km = mov.get("allocated_distance_km") or mov.get("speed_integral_km")
    if mov.get("hours") and km and model.get("driving_kwh_per_km") is not None:
        low = mov["hours"] < MIN_REGIME_H or km < MIN_REGIME_KM
        add("driving_kwh_per_km", model.get("driving_kwh_per_km"), mov["energy_kwh"] / km, "kWh/km",
            f"V x I energy / allocated km while moving with PTO off ({mov['hours']} h, {km} km)",
            "very low (<0.5 h or <1 km)" if low else "low (1 mission)")
    idle = ex.get("stationary_no_pto") or {}
    if idle.get("hours") and model.get("idle_kwh_per_h") is not None:
        add("idle_kwh_per_h", model.get("idle_kwh_per_h"), idle.get("mean_power_kw"), "kW",
            f"mean V x I while powered on, stationary, PTO off ({idle['hours']} h); applies to powered-on "
            "standby only", "low (1 mission)" if idle["hours"] >= MIN_REGIME_H else "very low (<0.5 h)", nd=3)
    return out


CALIBRATION_NOTE = ("Proposals are for review only and are never written to config.yaml. Applying them "
                    "would be an in-sample fit to the same mission -- not a validation; hold out further "
                    "missions first.")


# --------------------------------------------------------------------------- #
#  Limitations
# --------------------------------------------------------------------------- #
def _limitations(result: Dict[str, Any], missions: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if result:
        if len(missions) <= 1:
            out.append({"id": "single_mission", "scope": "data",
                        "text": "One mission is insufficient for robust model calibration; "
                                "generalisation is not established."})
        if not result.get("position_signals_seen"):
            out.append({"id": "no_gps", "scope": "data",
                        "text": "No GPS coordinates in the supplied CSV (distance from the LifetimeKm counter)."})
        ce = result.get("charging_evidence") or {}
        if not ce.get("charging_signals_seen"):
            out.append({"id": "no_charging_state", "scope": "data",
                        "text": "No explicit charging-state stream in the supplied CSV."})
        if not ce.get("soc_increase_events"):
            out.append({"id": "no_charging_cycle", "scope": "data",
                        "text": "No charging cycle in the data (SOC never increased)."})
        out.append({"id": "soc_resolution", "scope": "data",
                    "text": "MainBatterySOC is reported in whole percent (+/-1 point per reading)."})
    for lid, text in DEPLOYMENT_LIMITATIONS:
        out.append({"id": lid, "scope": "deployment", "text": text})
    return out


# --------------------------------------------------------------------------- #
#  Export
# --------------------------------------------------------------------------- #
LONG_COLUMNS = ("mission_id", "tractor_id", "key", "label", "group", "value", "unit", "provenance",
                "method", "sources", "flags", "analysis_version", "input_sha256")


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else value


def export_json(doc: Dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False)


def export_csv(doc: Dict[str, Any], wide: bool = False) -> str:
    """CSV export.

    *long* (default): one row per KPI with value, unit, provenance, formula,
    source signals and quality flags -- every number traceable on its own
    row.  *wide*: one row per mission with the summary keys, each followed by
    a ``<key>__provenance`` column (for multi-mission tables).
    """
    buf = io.StringIO()
    summary = doc.get("summary") or {}
    sha = (doc.get("input") or {}).get("sha256") or ""
    version = doc.get("analysis_version") or ""
    if wide:
        prov = doc.get("summary_provenance") or {}
        header = []
        for key in summary:
            header.append(key)
            if key in prov:
                header.append(f"{key}__provenance")
        header += ["analysis_version", "input_sha256"]
        writer = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        row = {}
        for key, value in summary.items():
            row[key] = _cell(value)
            if key in prov:
                row[f"{key}__provenance"] = prov[key]
        row["analysis_version"], row["input_sha256"] = version, sha
        writer.writerow(row)
        return buf.getvalue()
    writer = csv.DictWriter(buf, fieldnames=LONG_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for item in (doc.get("kpis") or {}).values():
        writer.writerow({
            "mission_id": _cell(summary.get("mission_id")), "tractor_id": _cell(summary.get("tractor_id")),
            "key": item["key"], "label": item["label"], "group": item["group"],
            "value": _cell(item["value"]), "unit": item["unit"], "provenance": item["provenance"],
            "method": item["method"], "sources": ";".join(item["sources"]),
            "flags": ";".join(item["flags"]), "analysis_version": version, "input_sha256": sha,
        })
    return buf.getvalue()


# --------------------------------------------------------------------------- #
#  Convenience entry points
# --------------------------------------------------------------------------- #
def load_model(config_path: Optional[Any]) -> Dict[str, Any]:
    if not config_path:
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        return dict((cfg.get("tractors") or {}).get("model") or {})
    except Exception as exc:                                   # noqa: BLE001
        print(f"[kpi] config not loaded ({exc}); model rows will be unavailable", file=sys.stderr)
        return {}


def mission_kpis(messages: Iterable[TelemetryMessage], model: Optional[Dict[str, Any]] = None, *,
                 source: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None,
                 gap_cap_s: float = DEFAULT_GAP_CAP_S, tractor_id: Optional[str] = None) -> Dict[str, Any]:
    analysis = MissionAnalysis(messages, gap_cap_s=gap_cap_s, harvest_model=model, tractor_id=tractor_id)
    return build_mission_kpis(analysis.run(), model, source=source, config_file=config_file)


def kpis_from_csv(path: Any, config_path: Optional[Any] = None, gap_cap_s: float = DEFAULT_GAP_CAP_S,
                  tractor_id: Optional[str] = None) -> Dict[str, Any]:
    from .zetrack import CsvParseStats, read_csv
    stats = CsvParseStats()
    messages = read_csv(path, stats)
    source = {"kind": "csv-file", "file": str(path), "sha256": sha256_file(path),
              "messages": len(messages), "parse": stats.to_dict()}
    return mission_kpis(messages, load_model(config_path), source=source,
                        config_file=str(config_path) if config_path else None,
                        gap_cap_s=gap_cap_s, tractor_id=tractor_id)


def text(doc: Dict[str, Any]) -> str:
    if doc.get("state") != "ok":
        return f"no KPIs: {doc.get('error')}"
    lines: List[str] = []
    for group in doc["groups"]:
        lines.append(group["title"].upper())
        for key in group["kpis"]:
            item = doc["kpis"][key]
            value = item["value"]
            shown = "n/a" if value is None else (
                json.dumps(value) if isinstance(value, dict) else f"{value} {item['unit']}".strip())
            lines.append(f"  {item['label']:<46s} {shown:<22s} [{item['provenance']}]")
        lines.append("")
    lines.append("REAL vs HARVEST MODEL   (difference = model - real)")
    for row in doc["comparison"]["rows"]:
        lines.append(f"  {row['metric']:<30s} real {row['real']!s:>8}  model {row['model']!s:>8}  "
                     f"diff {row['difference']!s:>8}  err {row['error_pct']!s:>6} %  {row['status']}")
    lines += ["", "FINDINGS"]
    lines += [f"  [{f['level']}] {f['text']}" for f in doc["findings"]]
    lines += ["", "CALIBRATION PROPOSALS (not applied) -- " + doc.get("calibration_note", "")]
    lines += [f"  {p['parameter']}: {p['current']} -> {p['proposed']} {p['unit']}  ({p['confidence']}) {p['basis']}"
              for p in doc["calibration_proposals"]]
    lines += ["", "LIMITATIONS"]
    lines += [f"  [{lim['scope']}] {lim['text']}" for lim in doc["limitations"]]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ZETRABOT mission KPIs and HARVEST model validation")
    parser.add_argument("csv", help="Zetrack telemetry export (CSV)")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "config.yaml"),
                        help="HARVEST config.yaml (tractors.model) -- default: repo config")
    parser.add_argument("--json", help="write the KPI document as JSON ('-' = stdout)")
    parser.add_argument("--csv-out", "--csv-file", dest="csv_out", help="write the KPIs as CSV")
    parser.add_argument("--wide", action="store_true", help="CSV: one row per mission instead of one per KPI")
    parser.add_argument("--gap-cap", type=float, default=DEFAULT_GAP_CAP_S)
    parser.add_argument("--tractor", help="restrict to one tractor_id")
    args = parser.parse_args(argv)
    doc = kpis_from_csv(args.csv, args.config, args.gap_cap, args.tractor)
    if args.json == "-":
        print(export_json(doc))
        return 0
    print(text(doc))
    if args.json:
        Path(args.json).write_text(export_json(doc), encoding="utf-8")
        print(f"\nJSON written to {args.json}")
    if args.csv_out:
        Path(args.csv_out).write_text(export_csv(doc, wide=args.wide), encoding="utf-8")
        print(f"CSV written to {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
