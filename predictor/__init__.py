"""
predictor/__init__.py
=====================
HARVEST Prediction Module

Public API
----------
    from predictor import build_predictors, ForecastBundle
    from predictor import BasePVPredictor, BaseLoadPredictor

    pv_pred, load_pred = build_predictors(cfg)
    bundle = ForecastBundle(pv_pred, load_pred, grid_max_kw=10.5)

    # In the simulator step:
    shape   = pv_pred.predict_shape(now)
    load_kw = load_pred.predict_load_kw(now)

    # For pro-active scheduling:
    headroom = bundle.net_available_kw(now + timedelta(hours=1))
    best_h   = bundle.best_charging_window(today, duration_hours=2)
"""

from .base        import BasePVPredictor, BaseLoadPredictor, ForecastBundle
from .static      import StaticPVPredictor, StaticLoadPredictor
from .weather     import OpenMeteoPVPredictor, WeatherStubPredictor

__all__ = [
    "BasePVPredictor",
    "BaseLoadPredictor",
    "ForecastBundle",
    "StaticPVPredictor",
    "StaticLoadPredictor",
    "OpenMeteoPVPredictor",
    "WeatherStubPredictor",
    "build_predictors",
]


def build_predictors(
    cfg: dict,
    model_path=None,
) -> tuple:
    """
    Factory function.  Reads ``cfg["prediction"]`` and instantiates the
    appropriate PV and load predictors.

    Backend selection (``cfg.prediction.pv.backend``):

    ==================  ================================================
    ``"static"``        StaticPVPredictor (default, uses config profile)
    ``"nn"``            NNPVPredictor (requires trained model)
    ``"openmeteo"``     OpenMeteoPVPredictor (live weather, needs internet)
    ``"stub"``          WeatherStubPredictor (seasonal bell curve)
    ==================  ================================================

    Load predictor is always built from the consumer list in ``cfg``.
    If a trained model is available (``nn`` backend) the NN is used for
    both PV and load simultaneously.

    Returns
    -------
    (pv_predictor, load_predictor) : tuple[BasePVPredictor, BaseLoadPredictor]
    """
    from pathlib import Path

    pv_cfg  = cfg.get("pv", {})
    pr_cfg  = cfg.get("prediction", {})
    pv_pcfg = pr_cfg.get("pv", {})
    consumers = cfg.get("energy_consumers", [])

    farm_peak_kw = float(pv_cfg.get("farm_fixed_peak_kw", 5.0))
    backend      = pv_pcfg.get("backend", "static").lower()

    # ── Load predictor (always from consumer list) ────────────
    load_pred = StaticLoadPredictor(consumers)

    # ── PV predictor ──────────────────────────────────────────
    if backend == "nn":
        from .nn_predictor import NNPVPredictor, NNLoadPredictor
        mp = model_path or pv_pcfg.get("model_path")
        if mp is None:
            raise ValueError(
                "prediction.pv.backend='nn' requires prediction.pv.model_path"
            )
        pv_pred   = NNPVPredictor(Path(mp), farm_peak_kw)
        load_pred = NNLoadPredictor(Path(mp))   # same model, different output index

    elif backend == "openmeteo":
        lat = float(pv_pcfg.get("latitude",  41.69))
        lon = float(pv_pcfg.get("longitude", 23.31))
        pv_pred = OpenMeteoPVPredictor(
            farm_fixed_peak_kw = farm_peak_kw,
            latitude           = lat,
            longitude          = lon,
            cache_file         = Path(pv_pcfg.get("cache_file", "pv_weather_cache.json")),
        )

    elif backend == "stub":
        pv_pred = WeatherStubPredictor(
            farm_fixed_peak_kw = farm_peak_kw,
            peak_hour          = float(pv_pcfg.get("peak_hour", 12.0)),
            width_h            = float(pv_pcfg.get("width_h",   2.5)),
        )

    else:  # "static" or anything unrecognised → backward-compatible default
        profile = {int(k): float(v) for k, v in pv_cfg.get("profile", {}).items()}
        pv_pred = StaticPVPredictor(profile, farm_peak_kw)

    return pv_pred, load_pred
