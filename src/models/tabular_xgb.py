"""Tabular XGBoost regressor wrapper for OCPM Remaining Time Prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.models.base import BaseOCPMModel

LOGGER = logging.getLogger(__name__)


class TabularXGBModel(BaseOCPMModel):
    """XGBRegressor wrapper adhering to ``BaseOCPMModel``."""

    def __init__(self, config: dict[str, Any], random_seed: int | None = None) -> None:
        """Initialize the XGBoost model from ``config.yaml``.

        Args:
            config: Loaded configuration dictionary.
            random_seed: Optional random seed override.
        """
        super().__init__(config=config, random_seed=random_seed)
        xgb_cfg = config.get("models", {}).get("xgboost", {})
        n_estimators = (
            int(config.get("debug", {}).get("xgb_n_estimators", 10))
            if self.debug_mode
            else int(xgb_cfg.get("n_estimators", 300))
        )
        self._early_stopping_rounds = int(xgb_cfg.get("early_stopping_rounds", 20))
        self._params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": int(xgb_cfg.get("max_depth", 6)),
            "learning_rate": float(xgb_cfg.get("learning_rate", 0.05)),
            "subsample": float(xgb_cfg.get("subsample", 0.8)),
            "colsample_bytree": float(xgb_cfg.get("colsample_bytree", 0.8)),
            "objective": str(xgb_cfg.get("objective", "reg:squarederror")),
            "n_jobs": int(xgb_cfg.get("n_jobs", -1)),
            "random_state": self.random_seed,
            "verbosity": 0,
        }
        self._model: XGBRegressor | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> TabularXGBModel:
        """Fit ``XGBRegressor`` on preprocessed training features.

        Args:
            X_train: Preprocessed training feature DataFrame.
            y_train: Training targets.
            X_val: Optional validation feature DataFrame for early stopping.
            y_val: Optional validation targets.
            **kwargs: Unused extra arguments for interface compatibility.

        Returns:
            Fitted ``TabularXGBModel``.
        """
        params = dict(self._params)
        use_es = X_val is not None and y_val is not None and len(X_val) > 0
        if use_es:
            params["early_stopping_rounds"] = self._early_stopping_rounds

        self._model = XGBRegressor(**params)
        y_tr = np.asarray(y_train, dtype=np.float64)

        if use_es:
            y_va = np.asarray(y_val, dtype=np.float64)
            self._model.fit(
                X_train,
                y_tr,
                eval_set=[(X_val, y_va)],
                verbose=False,
            )
        else:
            self._model.fit(X_train, y_tr)

        return self

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Generate event-level predictions aligned with ``X``.

        Args:
            X: Preprocessed feature DataFrame.
            **kwargs: Unused extra arguments for interface compatibility.

        Returns:
            1D float64 NumPy array of predictions.
        """
        if self._model is None:
            raise RuntimeError("TabularXGBModel must be fitted before calling predict().")
        preds = self._model.predict(X)
        return np.asarray(preds, dtype=np.float64)

