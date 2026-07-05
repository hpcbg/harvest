"""
HARVEST local server
====================
Serves dashboard.html and exposes a /simulate endpoint that runs
the real Python simulation engine without any external API calls.

Usage:
    python server.py

Then open http://localhost:8765 in your browser.
"""

from __future__ import annotations

import copy
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List

# ── Import the simulation engine from the same directory ──────────────────────
try:
    from main import Simulator, ScenarioDef, load_yaml, load_yaml_with_local
except ImportError as exc:
    sys.exit(
        f"Could not import main.py: {exc}\n"
        "Make sure server.py is in the same folder as main.py, "
        "task_generator.py, and config.yaml."
    )

# ROI engine is optional — the dashboard Operations view works without it.
try:
    from roi import run_roi_analysis, ROIValidationError
    from roi.engine import config_hash
    _ROI_AVAILABLE = True
except ImportError:
    _ROI_AVAILABLE = False

    class ROIValidationError(Exception):
        pass

    def config_hash(cfg):
        return ""

import time
import uuid

# ── Operations-run store ──────────────────────────────────────────────────────
# Single-user local dashboard: retaining only the latest successful Operations run
# in memory is sufficient.  ROI must be based strictly on this run.
_LAST_OPS_RUN: Dict[str, Any] = {}
_OPS_RUN_TTL_SECONDS = 6 * 3600   # expire after 6 hours of inactivity


def _store_ops_run(params, scenario_defs, cfg, results) -> Dict[str, str]:
    """Record the latest successful Operations run and return its identifier."""
    run_id = uuid.uuid4().hex[:12]
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    one_day = {}
    for sd, res in zip(scenario_defs, results):
        sid = sd.get("id") or sd.get("name") or sd.get("label")
        one_day[sid] = {
            "completed_tasks": res.get("completed_tasks"),
            "total_tasks": res.get("total_tasks"),
            "total_cost_eur": res.get("total_cost_eur"),
            "task_completion_pct": res.get("task_completion_pct"),
        }
    _LAST_OPS_RUN.clear()
    _LAST_OPS_RUN.update({
        "run_id": run_id,
        "timestamp": timestamp,
        "created_at": time.time(),
        "params": params,
        "scenarios": scenario_defs,
        "cfg": cfg,
        "hash": config_hash(cfg),
        "one_day": one_day,
    })
    return {"operations_run_id": run_id, "operations_timestamp": timestamp}


def _get_ops_run(run_id: str):
    run = _LAST_OPS_RUN.get("run_id")
    if not run or run != run_id:
        return None
    if (time.time() - _LAST_OPS_RUN.get("created_at", 0)) > _OPS_RUN_TTL_SECONDS:
        return None
    return _LAST_OPS_RUN

BASE_DIR   = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.yaml"
DASHBOARD   = BASE_DIR / "dashboard.html"

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_config_override(base: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-copy base config and apply UI parameter overrides.
    params keys mirror the slider ids in dashboard.html.
    """
    cfg = copy.deepcopy(base)

    # Grid
    if "grid_kw" in params:
        cfg["grid"]["max_power_kw"] = float(params["grid_kw"])

    # PV
    if "farm_pv_kw" in params:
        cfg["pv"]["farm_fixed_peak_kw"] = float(params["farm_pv_kw"])
    if "panel_w" in params:
        cfg["tractor_pv"]["panel_peak_w"] = float(params["panel_w"])

    # Fleet – rebuild fleet list preserving structure
    if "tractors" in params:
        n = int(params["tractors"])
        base_fleet = base["tractors"]["fleet"]
        new_fleet = []
        for i in range(n):
            if i < len(base_fleet):
                entry = copy.deepcopy(base_fleet[i])
            else:
                # Clone last tractor, give it a new id and shifted position
                entry = copy.deepcopy(base_fleet[-1])
                entry["id"] = f"tractor_{i+1}"
                entry["initial_location"] = {
                    "x": 50 + i * 10,
                    "y": 50,
                }
                entry["initial_soc_percent"] = 70
            new_fleet.append(entry)
        cfg["tractors"]["fleet"] = new_fleet

    if "chargers" in params or "charger_kw" in params:
        n_ch   = int(params.get("chargers", len(base["charging"]["stations"])))
        ch_kw  = float(params.get("charger_kw", base["charging"]["stations"][0]["max_power_kw"]))
        base_stations = base["charging"]["stations"]
        new_stations = []
        for i in range(n_ch):
            if i < len(base_stations):
                entry = copy.deepcopy(base_stations[i])
            else:
                entry = {
                    "id": f"charger_{i+1}",
                    "location": {"x": 40 + i * 5, "y": 40},
                    "max_power_kw": ch_kw,
                }
            entry["max_power_kw"] = ch_kw
            new_stations.append(entry)
        cfg["charging"]["stations"] = new_stations

    if "battery_kwh" in params:
        cfg["tractors"]["model"]["battery_capacity_kwh"] = float(params["battery_kwh"])
        # Keep swappable at 50 % of total
        cfg["tractors"]["model"]["swappable_capacity_kwh"] = round(float(params["battery_kwh"]) * 0.5, 1)

    # Tasks
    if "num_tasks" in params:
        cfg["task_generation"]["num_tasks"] = int(params["num_tasks"])
    if "seed" in params:
        cfg["task_generation"]["seed"] = int(params["seed"])

    # Always use generated mode when called from dashboard
    cfg["task_generation"]["mode"] = "generated"

    return cfg


def run_scenarios(
    cfg: Dict[str, Any],
    scenario_defs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run requested scenarios and return list of summary dicts."""
    results = []
    for sd in scenario_defs:
        sdef = ScenarioDef(
            name=sd.get("name") or sd.get("label") or sd.get("id", "scenario"),
            charging_strategy=sd["charging_strategy"],
            tractor_pv_enabled=bool(sd.get("tractor_pv_enabled", False)),
            load_shedding=bool(sd.get("load_shedding", False)),
            use_marl=bool(sd.get("use_marl", False)),
        )
        sim = Simulator(cfg, sdef)
        sim.run()
        summary = sim.summarize()
        summary["tractor_pv"]    = bool(summary["tractor_pv"])
        summary["load_shedding"] = bool(summary["load_shedding"])

        # Include per-task status for live task table in dashboard
        task_df = sim.task_schedule_dataframe()
        task_rows = []
        for _, row in task_df.iterrows():
            task_rows.append({
                "task_id":              str(row["task_id"]),
                "name":                 str(row["name"]),
                "priority":             str(row["priority"]),
                "phase":                str(row["phase"]),
                "status_label":         str(row["status_label"]),
                "progress_pct":         float(row["progress_pct"]),
                "transit_progress_pct": float(row["transit_progress_pct"]),
                "is_delayed":           bool(row["is_delayed"]),
                "delay_reason":         str(row["delay_reason"]),
                "interruption_count":   int(row["interruption_count"]),
                "assigned_tractor":     str(row["assigned_tractor"]) if row["assigned_tractor"] else None,
                "earliest_start":       str(row["earliest_start"])[:16],
                "latest_finish":        str(row["latest_finish"])[:16],
                "duration_minutes":     int(row["duration_minutes"]),
                "uses_pto":             bool(row["uses_pto"]),
            })
        summary["tasks"] = task_rows
        results.append(summary)
    return results


# ── Request handler ───────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # silence default access log spam
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, mime: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/dashboard.html"):
            if DASHBOARD.exists():
                self._send_file(DASHBOARD, "text/html; charset=utf-8")
            else:
                self._send_json({"error": "dashboard.html not found"}, 404)

        elif path == "/chart.js":
            chart_file = BASE_DIR / "chart.umd.js"
            if chart_file.exists():
                self._send_file(chart_file, "application/javascript; charset=utf-8")
            else:
                self._send_json({"error": "chart.umd.js not found — copy it next to server.py"}, 404)

        elif path in ("/config", "/api/config"):
            # Return current merged config as JSON so the dashboard can read
            # defaults (including roi: demonstration assumptions). Read-only.
            try:
                cfg = load_yaml_with_local(CONFIG_FILE)
                self._send_json(cfg)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/health":
            self._send_json({"status": "ok", "config": str(CONFIG_FILE)})

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]

        if path == "/simulate":
            self._handle_simulate()
        elif path in ("/api/roi", "/roi"):
            self._handle_roi()
        else:
            self._send_json({"error": "unknown endpoint"}, 404)

    def _read_payload(self):
        length  = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def _handle_simulate(self):
        try:
            payload = self._read_payload()
        except Exception as e:
            self._send_json({"error": f"bad request: {e}"}, 400)
            return

        params         = payload.get("params", {})
        scenario_defs  = payload.get("scenarios", [])

        if not scenario_defs:
            self._send_json({"error": "no scenarios provided"}, 400)
            return

        try:
            base_cfg = load_yaml_with_local(CONFIG_FILE)
            cfg      = build_config_override(base_cfg, params)
            results  = run_scenarios(cfg, scenario_defs)
            # Record this successful run as the source of truth for ROI.
            ops = _store_ops_run(params, scenario_defs, cfg, results)
            self._send_json({"results": results, **ops})
        except Exception as e:
            import traceback
            self._send_json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def _handle_roi(self):
        """POST /api/roi — long-term ROI, based strictly on the latest Operations run."""
        if not _ROI_AVAILABLE:
            self._send_json({"error": "ROI module not available on the server."}, 500)
            return
        try:
            payload = self._read_payload()
        except Exception as e:
            self._send_json({"error": f"bad request: {e}"}, 400)
            return

        run_id = payload.get("operations_run_id")
        run = _get_ops_run(run_id) if run_id else None
        if run is None:
            # Missing / unknown / expired identifier → 409 Conflict, clean message.
            self._send_json(
                {"error": "Run the Operations simulation before running ROI analysis."}, 409)
            return

        client_hash = payload.get("operations_hash")
        if client_hash and client_hash != run["hash"]:
            self._send_json(
                {"error": "Operations settings have changed. Run the Operations "
                          "simulation again before calculating ROI."}, 409)
            return

        # Operational parameters come from the SAVED run, never from ROI fields.
        request = dict(payload)
        request["operations"] = {
            "run_id": run["run_id"],
            "timestamp": run["timestamp"],
            "params": run["params"],
            "scenarios": run["scenarios"],
            "one_day": run["one_day"],
        }
        try:
            report = run_roi_analysis(run["cfg"], request)
            self._send_json(report)
        except ROIValidationError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            import traceback
            print("ROI error:\n" + traceback.format_exc(), file=sys.stderr)
            self._send_json({"error": f"ROI analysis failed: {e}"}, 500)


# ── Entry point ───────────────────────────────────────────────────────────────

PORT = 8765

def main() -> None:
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    url    = f"http://localhost:{PORT}"

    cfg_str = str(CONFIG_FILE)
    if len(cfg_str) > 36: cfg_str = "..." + cfg_str[-33:]
    print(f"""
╔══════════════════════════════════════════════╗
║   HARVEST  local simulation server           ║
╠══════════════════════════════════════════════╣
║   URL   : {url:<36}║
║   Config: {cfg_str:<36}║
║   Stop  : Ctrl+C                             ║
╚══════════════════════════════════════════════╝
""")

    # Open browser after 600 ms so the server is ready
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
