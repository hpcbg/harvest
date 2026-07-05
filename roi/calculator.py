"""
roi.calculator
==============
Pure, reusable financial primitives: NPV, IRR, simple / discounted payback,
ROI, discount factors, escalation.

All functions operate on plain Python numbers and lists so they are trivially
unit-testable and carry no dependency on the HARVEST simulator, the HTTP server
or the dashboard.  Nothing here formats currency — callers round only for
display or CSV export.

Conventions
-----------
A *cash-flow series* is a list ``flows`` where ``flows[0]`` is the year-0 net
cash flow (typically ``-net_capex``) and ``flows[y]`` is the net cash flow in
year ``y``.  Positive numbers are inflows (savings / revenue), negative numbers
are outflows (costs / replacements).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

# Value returned when a metric is mathematically undefined (e.g. payback that
# never happens, IRR of an all-positive series).  Callers surface this as "N/A".
NA: Optional[float] = None


def escalate(base_value: float, annual_rate_pct: float, year: int) -> float:
    """Compound-escalate ``base_value`` to ``year`` (year 1 == base year).

    Year 1 is treated as the first operating year with no escalation applied,
    so escalation uses an exponent of ``year - 1``.
    """
    factor = (1.0 + annual_rate_pct / 100.0) ** max(0, year - 1)
    return base_value * factor


def degrade(base_value: float, annual_degradation_pct: float, year: int) -> float:
    """Compound-degrade a value (e.g. PV yield) by year (year 1 == full output)."""
    factor = (1.0 - annual_degradation_pct / 100.0) ** max(0, year - 1)
    return base_value * max(0.0, factor)


def discount_factor(rate_pct: float, year: int) -> float:
    """1 / (1 + r)^year."""
    return 1.0 / ((1.0 + rate_pct / 100.0) ** year)


def npv(rate_pct: float, flows: Sequence[float]) -> float:
    """Net present value of a cash-flow series (``flows[0]`` is year 0)."""
    r = rate_pct / 100.0
    return sum(cf / ((1.0 + r) ** y) for y, cf in enumerate(flows))


def _npv_at_fraction(rate_fraction: float, flows: Sequence[float]) -> float:
    return sum(cf / ((1.0 + rate_fraction) ** y) for y, cf in enumerate(flows))


def irr(
    flows: Sequence[float],
    low: float = -0.9,
    high: float = 10.0,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> Optional[float]:
    """Internal rate of return as a percentage, via bisection.

    Returns ``None`` ("not available") when the cash flows do not admit a valid
    IRR — for example an all-positive or all-negative series, or one whose NPV
    does not change sign over the search interval ``[low, high]`` (default
    -90 % … +1000 %).  Bisection is used deliberately: it needs no external
    dependency and cannot diverge the way Newton's method can.
    """
    flows = list(flows)
    if len(flows) < 2:
        return NA
    if all(cf >= 0 for cf in flows) or all(cf <= 0 for cf in flows):
        return NA

    f_low = _npv_at_fraction(low, flows)
    f_high = _npv_at_fraction(high, flows)
    if math.isnan(f_low) or math.isnan(f_high):
        return NA
    if f_low * f_high > 0:
        # No sign change in the interval → no bracketable root.
        return NA

    lo, hi = low, high
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = _npv_at_fraction(mid, flows)
        if abs(f_mid) < tol:
            return mid * 100.0
        if f_low * f_mid < 0:
            hi = mid
            f_high = f_mid
        else:
            lo = mid
            f_low = f_mid
    return ((lo + hi) / 2.0) * 100.0


def _first_crossing(cumulative: Sequence[float]) -> Optional[float]:
    """Fractional index at which a cumulative series first becomes non-negative.

    ``cumulative[0]`` corresponds to year 0.  Linear interpolation is used
    between the last-negative and first-non-negative year.  Returns ``None``
    when the series never reaches zero.
    """
    if not cumulative:
        return NA
    if cumulative[0] >= 0:
        return 0.0
    for y in range(1, len(cumulative)):
        prev, cur = cumulative[y - 1], cumulative[y]
        if cur >= 0:
            step = cur - prev
            if abs(step) < 1e-12:
                return float(y)
            frac = -prev / step
            return (y - 1) + frac
    return NA


def simple_payback(flows: Sequence[float]) -> Optional[float]:
    """Simple (undiscounted) payback in fractional years, or ``None``."""
    cumulative: List[float] = []
    running = 0.0
    for cf in flows:
        running += cf
        cumulative.append(running)
    return _first_crossing(cumulative)


def discounted_payback(flows: Sequence[float], rate_pct: float) -> Optional[float]:
    """Discounted payback in fractional years, or ``None``."""
    r = rate_pct / 100.0
    cumulative: List[float] = []
    running = 0.0
    for y, cf in enumerate(flows):
        running += cf / ((1.0 + r) ** y)
        cumulative.append(running)
    return _first_crossing(cumulative)


def roi_pct(net_capex: float, cumulative_undiscounted_net_benefit: float) -> Optional[float]:
    """Simple ROI over the horizon as a percentage.

    ``(cumulative_net_benefit - net_capex) / net_capex * 100``.  Returns
    ``None`` when ``net_capex`` is zero or negative (division would be
    undefined / misleading), so the caller shows "N/A".
    """
    if net_capex <= 0:
        return NA
    return (cumulative_undiscounted_net_benefit - net_capex) / net_capex * 100.0


def sanitize(value: Optional[float], decimals: Optional[int] = None) -> Optional[float]:
    """Coerce NaN / Infinity to ``None`` so JSON stays valid; optionally round."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if decimals is not None:
        return round(float(value), decimals)
    return float(value)
