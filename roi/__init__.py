"""
HARVEST ROI & Investment Analysis package
=========================================
Long-term return-on-investment and reliability analysis for the main HARVEST
investments (electric fleet + chargers, fixed farm PV, tractor-roof PV, and
optional islanding/backup), layered on top of the existing pilot6 simulator.

Public API
----------
* :func:`roi.engine.run_roi_analysis` – run a full analysis, return JSON-safe dict.
* :mod:`roi.calculator`               – pure financial primitives (NPV/IRR/payback).
* :mod:`roi.models`                   – typed assumptions / result structures.
* :class:`roi.validation.ROIValidationError` – raised on invalid requests.

Energy sharing with neighbouring farms or households is explicitly out of scope
(see README → "Future work").
"""

from .engine import run_roi_analysis
from .models import ROIAssumptions
from .validation import ROIValidationError

__all__ = ["run_roi_analysis", "ROIAssumptions", "ROIValidationError"]
