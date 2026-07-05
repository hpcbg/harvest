"""
roi.__main__
============
Command-line entry point for an ROI analysis without the dashboard.

Example
-------
    python -m roi \
        --config config.yaml \
        --start 2026-01-01 \
        --end 2026-12-31 \
        --period-mode auto \
        --horizon 10

Writes (to ``outputs/roi/`` by default):
    roi_summary.csv    roi_cashflows.csv    roi_assumptions.json    roi_report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from main import load_yaml_with_local

from . import export
from .engine import run_roi_analysis
from .validation import ROIValidationError


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m roi",
        description="HARVEST ROI & Investment analysis (headless).",
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--start", default=None, help="Operational period start (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="Operational period end (YYYY-MM-DD)")
    p.add_argument("--period-mode", default=None,
                   choices=["exact", "representative_month", "auto"],
                   help="Period calculation method")
    p.add_argument("--horizon", type=int, default=None, help="Financial horizon (years)")
    p.add_argument("--scenario", default=None, help="Operating scenario to use for ROI runs")
    p.add_argument("--investments", default=None,
                   help="Comma-separated ids "
                        "(electric_fleet,farm_pv,tractor_roof_pv,backup_islanding,portfolio)")
    p.add_argument("--out", default="outputs/roi", help="Output directory")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    config = load_yaml_with_local(args.config)

    request: Dict[str, Any] = {}
    if args.start:
        request["start"] = args.start
    if args.end:
        request["end"] = args.end
    if args.period_mode:
        request["period_mode"] = args.period_mode
    if args.horizon is not None:
        request["horizon"] = args.horizon
    if args.scenario:
        request["scenario"] = args.scenario
    if args.investments:
        request["investments"] = [s.strip() for s in args.investments.split(",") if s.strip()]
    else:
        request["investments"] = [
            "electric_fleet", "farm_pv", "tractor_roof_pv", "backup_islanding", "portfolio",
        ]

    print("Running HARVEST ROI analysis ...")
    try:
        report = run_roi_analysis(config, request)
    except ROIValidationError as e:
        print(f"  Invalid request: {e}", file=sys.stderr)
        return 2

    cur = report["meta"].get("currency", "EUR")
    meta = report["meta"]
    print(f"  Period      : {meta['start_date']} -> {meta['end_date']} "
          f"({meta['period_mode']}, {meta['simulations_run']} sims, "
          f"{meta['days_represented']} days represented)")
    print(f"  Horizon     : {meta['financial_horizon_years']} yr @ "
          f"{meta['discount_rate_pct']}% discount")

    def _line(prefix: str, r: Dict[str, Any]) -> None:
        m = r["metrics"]
        print(f"  {prefix} {r['name']:<44} "
              f"CAPEX {cur} {m.get('net_capex_eur')}  "
              f"NPV {cur} {m.get('npv_eur')}  "
              f"payback {m.get('simple_payback_years')} yr  "
              f"ROI {m.get('roi_pct')}%")

    for r in report.get("investments", []):
        _line("-", r)
    if report.get("portfolio"):
        _line("=", report["portfolio"])

    out_dir = Path(args.out)
    written = export.write_all(report, out_dir)
    print(f"  Files written to {out_dir}/:")
    for p in written:
        print(f"    {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
