"""
roi.export
=========
Serialisers that turn an ROI response dict into the documented output files:

* ``roi_summary.csv``     – one row per investment plus the combined portfolio.
* ``roi_cashflows.csv``   – one row per investment per year.
* ``roi_assumptions.json``– the resolved assumptions actually used.
* ``roi_report.json``     – the full response (meta, operational, investments…).

Values are rounded only here (for reporting); the API keeps full precision.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

SUMMARY_FIELDS = [
    "id", "name", "net_capex_eur", "annual_net_benefit_eur",
    "annual_operating_savings_eur", "annual_avoided_grid_cost_eur",
    "annual_avoided_outage_cost_eur", "roi_pct", "simple_payback_years",
    "discounted_payback_years", "npv_eur", "irr_pct",
]


def _investments_and_portfolio(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = list(report.get("investments", []))
    if report.get("portfolio"):
        rows.append(report["portfolio"])
    return rows


def write_summary_csv(report: Dict[str, Any], path: Path) -> None:
    rows = _investments_and_portfolio(report)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            m = r.get("metrics", {})
            w.writerow({
                "id": r.get("id"),
                "name": r.get("name"),
                "net_capex_eur": m.get("net_capex_eur"),
                "annual_net_benefit_eur": m.get("annual_net_benefit_eur"),
                "annual_operating_savings_eur": m.get("annual_operating_savings_eur"),
                "annual_avoided_grid_cost_eur": m.get("annual_avoided_grid_cost_eur"),
                "annual_avoided_outage_cost_eur": m.get("annual_avoided_outage_cost_eur"),
                "roi_pct": m.get("roi_pct"),
                "simple_payback_years": m.get("simple_payback_years"),
                "discounted_payback_years": m.get("discounted_payback_years"),
                "npv_eur": m.get("npv_eur"),
                "irr_pct": m.get("irr_pct"),
            })


def write_cashflows_csv(report: Dict[str, Any], path: Path) -> None:
    rows = _investments_and_portfolio(report)
    fields = [
        "investment_id", "year", "baseline_operating_cost", "candidate_operating_cost",
        "fuel_cost", "electricity_cost", "maintenance", "outage_loss", "revenue",
        "replacement_cost", "net_cash_flow", "discount_factor", "discounted_cash_flow",
        "cumulative_cash_flow", "cumulative_discounted_cash_flow",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            for cf in r.get("annual_cashflows", []):
                out = {"investment_id": r.get("id")}
                out.update(cf)
                w.writerow(out)


def write_assumptions_json(report: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.get("assumptions", {}), f, indent=2, ensure_ascii=False)


def write_report_json(report: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def write_all(report: Dict[str, Any], out_dir: Path) -> List[Path]:
    """Write all four documented files; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "roi_summary.csv": write_summary_csv,
        "roi_cashflows.csv": write_cashflows_csv,
        "roi_assumptions.json": write_assumptions_json,
        "roi_report.json": write_report_json,
    }
    written: List[Path] = []
    for name, fn in paths.items():
        p = out_dir / name
        fn(report, p)
        written.append(p)
    return written
