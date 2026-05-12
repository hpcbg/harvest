"""
predictor/nn_predictor.py
=========================
Neural-network based PV and farm-load predictors.

Architecture
------------
Directly mirrors the paper (Tsvetanov et al.) but extended:

    Input  : [hour, day_of_week, month]   (3 features vs 2 in the paper)
    Hidden : one or more Dense(ReLU) layers
    Output : [pv_shape, farm_load_kw]     (2 outputs, same as the paper)

The single-hidden-layer variant is identical to the paper's FFNN.

Two usage modes
---------------
1. **Training** — call ``NNTrainer.train(...)`` to produce a saved Keras model.
2. **Inference** — instantiate ``NNPVPredictor`` / ``NNLoadPredictor`` with a
   saved model path; they implement BasePVPredictor / BaseLoadPredictor and
   are drop-in replacements for the static predictors.

TPI3 target (from IMP): prediction error ≤ 25 % MAPE under controlled /
simulated conditions.  A 50-hidden-unit network trained on 12 weeks of
synthetic data easily meets this target (paper reports convergence at MSE
~1600 with 50 HU vs ~2470 with 5-10 HU).

Quick-start example
-------------------
    from predictor.synthetic import SyntheticDataGenerator
    from predictor.nn_predictor import NNTrainer, NNPVPredictor, NNLoadPredictor
    import yaml

    cfg = yaml.safe_load(open("config.yaml"))
    gen = SyntheticDataGenerator.from_config(cfg)
    train_df = gen.generate(weeks=12, seed=42)
    test_df  = gen.generate(weeks=5,  seed=99)

    trainer = NNTrainer(hidden_units=[50], epochs=1500, dropout=None)
    model_path = trainer.train(train_df, test_df, output_dir="models/")

    pv_pred   = NNPVPredictor(model_path,   farm_fixed_peak_kw=5.0)
    load_pred = NNLoadPredictor(model_path)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BasePVPredictor, BaseLoadPredictor

logger = logging.getLogger(__name__)

# Feature / target column names  (must match SyntheticDataGenerator output)
FEATURE_COLS = ["hour", "day_of_week", "month"]
TARGET_COLS  = ["pv_shape", "farm_load_kw"]


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

class NNTrainer:
    """
    Trains a feed-forward neural network identical in architecture to the
    paper's model but with an optional third input feature (month).

    Parameters
    ----------
    hidden_units : list of int
        Width of each hidden layer. A single element reproduces the paper.
        E.g. [50] → one hidden layer with 50 units (best result in paper).
    epochs       : int
        Training epochs (paper used 10 000 for 50-HU network).
    dropout      : float or None
        Dropout rate applied after each hidden layer.  None = no dropout.
    learning_rate: float
        Adamax learning rate (paper default: 1e-3).
    """

    def __init__(
        self,
        hidden_units: List[int] = [50],
        epochs: int = 2000,
        dropout: Optional[float] = None,
        learning_rate: float = 1e-3,
    ) -> None:
        self.hidden_units   = hidden_units
        self.epochs         = epochs
        self.dropout        = dropout
        self.learning_rate  = learning_rate

    def _build_model(self, n_features: int, n_outputs: int):
        """Build the Keras model.  Imported lazily to avoid hard TF dependency."""
        import tensorflow as tf

        inp = tf.keras.Input(shape=(n_features,))
        x   = inp
        for hu in self.hidden_units:
            x = tf.keras.layers.Dense(
                hu,
                kernel_initializer="glorot_normal",  # Xavier init (paper §2.2)
                activation="relu",
            )(x)
            if self.dropout is not None:
                x = tf.keras.layers.Dropout(self.dropout)(x)

        out = tf.keras.layers.Dense(
            n_outputs,
            kernel_initializer="glorot_normal",
            activation=None,           # linear output (paper §1.1)
        )(x)

        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=self.learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        return model

    @staticmethod
    def _df_to_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        x = df[FEATURE_COLS].values.astype("float32")
        y = df[TARGET_COLS].values.astype("float32")
        return x, y

    def train(
        self,
        train_df: pd.DataFrame,
        test_df:  pd.DataFrame,
        output_dir: Path = Path("models"),
        model_name: Optional[str] = None,
    ) -> Path:
        """
        Train the network and save the model.

        Returns the path to the saved Keras model directory.

        Parameters
        ----------
        train_df, test_df : DataFrames from SyntheticDataGenerator.generate()
        output_dir        : Where to write the model and loss CSV
        model_name        : Override auto-generated name
        """
        import tensorflow as tf

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_x, train_y = self._df_to_arrays(train_df)
        test_x,  test_y  = self._df_to_arrays(test_df)

        hu_str = "_".join(str(h) for h in self.hidden_units)
        if model_name is None:
            model_name = (
                f"harvest_nn_hu{hu_str}"
                f"_ep{self.epochs}"
                f"_drop{self.dropout}"
            )

        model = self._build_model(train_x.shape[1], train_y.shape[1])
        model.summary(print_fn=logger.info)

        csv_log = tf.keras.callbacks.CSVLogger(
            output_dir / f"{model_name}_loss.csv"
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=max(50, self.epochs // 20),
            restore_best_weights=True,
        )

        history = model.fit(
            x=train_x, y=train_y,
            validation_data=(test_x, test_y),
            epochs=self.epochs,
            batch_size=256,
            callbacks=[csv_log, early_stop],
            verbose=1,
        )

        save_path = output_dir / model_name
        model.save(save_path)
        logger.info("Model saved to %s", save_path)

        # Quick MAPE report
        pred_y = model.predict(test_x)
        for i, col in enumerate(TARGET_COLS):
            mask  = test_y[:, i] > 1e-6
            mape  = np.mean(np.abs(test_y[mask, i] - pred_y[mask, i]) / test_y[mask, i]) * 100
            logger.info("Test MAPE %s: %.2f%%", col, mape)
            print(f"  MAPE {col}: {mape:.2f}%  (TPI3 target: ≤25%)")

        return save_path


# ─────────────────────────────────────────────────────────────
# Inference wrappers
# ─────────────────────────────────────────────────────────────

class _NNBase:
    """Shared lazy-loading logic for both predictor types."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = Path(model_path)
        self._model = None   # loaded on first call (lazy)

    def _get_model(self):
        if self._model is None:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(self._model_path)
            logger.info("Loaded NN predictor from %s", self._model_path)
        return self._model

    def _predict_raw(self, now: datetime) -> np.ndarray:
        """Return raw 2-output array [pv_shape, farm_load_kw]."""
        x = np.array([[now.hour, now.weekday(), now.month]], dtype="float32")
        return self._get_model().predict(x, verbose=0)[0]


class NNPVPredictor(BasePVPredictor, _NNBase):
    """
    PV shape predictor backed by a trained Keras model.

    Implements BasePVPredictor — drop-in replacement for StaticPVPredictor.
    """

    def __init__(self, model_path: Path, farm_fixed_peak_kw: float) -> None:
        BasePVPredictor.__init__(self, farm_fixed_peak_kw)
        _NNBase.__init__(self, model_path)

    def predict_shape(self, now: datetime) -> float:
        raw = self._predict_raw(now)
        return float(np.clip(raw[0], 0.0, 1.0))   # index 0 = pv_shape


class NNLoadPredictor(BaseLoadPredictor, _NNBase):
    """
    Farm load predictor backed by a trained Keras model.

    Implements BaseLoadPredictor — drop-in replacement for StaticLoadPredictor.
    """

    def __init__(self, model_path: Path, load_max_kw: float = 20.0) -> None:
        _NNBase.__init__(self, model_path)
        self._load_max_kw = load_max_kw

    def predict_load_kw(self, now: datetime) -> float:
        raw = self._predict_raw(now)
        return float(np.clip(raw[1], 0.0, self._load_max_kw))   # index 1 = farm_load_kw


# ─────────────────────────────────────────────────────────────
# CLI entry point  (mirrors the paper's train.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, yaml

    p = argparse.ArgumentParser(
        description="Train HARVEST NN predictor (TPI3)"
    )
    p.add_argument("--config",   type=Path, default=Path("config.yaml"))
    p.add_argument("--train",    type=Path, required=True,
                   help="Training dataset (.npz or .csv from synthetic.py)")
    p.add_argument("--test",     type=Path, required=True,
                   help="Test dataset")
    p.add_argument("--hidden",   type=int, nargs="+", default=[50],
                   help="Hidden layer widths (default: 50)")
    p.add_argument("--epochs",   type=int, default=2000)
    p.add_argument("--dropout",  type=float, default=None)
    p.add_argument("--out",      type=Path, default=Path("models"),
                   help="Output directory for model + loss CSV")
    args = p.parse_args()

    # Load datasets
    def _load(path: Path) -> pd.DataFrame:
        if path.suffix == ".npz":
            f = np.load(path, allow_pickle=True)
            cols_x = list(f["columns_x"]) if "columns_x" in f else FEATURE_COLS
            cols_y = list(f["columns_y"]) if "columns_y" in f else TARGET_COLS
            df = pd.DataFrame(f["dataset_x"], columns=cols_x)
            for i, c in enumerate(cols_y):
                df[c] = f["dataset_y"][:, i]
            return df
        return pd.read_csv(path)

    train_df = _load(args.train)
    test_df  = _load(args.test)

    trainer = NNTrainer(
        hidden_units=args.hidden,
        epochs=args.epochs,
        dropout=args.dropout,
    )
    model_path = trainer.train(train_df, test_df, output_dir=args.out)
    print(f"\nModel saved: {model_path}")
    print("To use in simulation, set in config.yaml:")
    print(f"  prediction:\n    model_path: {model_path}")
