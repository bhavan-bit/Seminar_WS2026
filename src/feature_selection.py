"""Stage 2b: Nested Tier 3 collinearity filtering and validation-only SHAP-RFE.

Enforces two critical anti-leakage invariants:
    - Collinearity filtering (Pearson ``|r| > 0.85``) is computed exclusively on
      ``X_sub_train``.
    - ``shap.TreeExplainer`` feature importances and MAE stopping checks are
      evaluated exclusively on ``X_sub_val``, never on ``X_sub_train``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

LOGGER = logging.getLogger(__name__)


@dataclass
class ShapRfeResult:
    """Structured output from nested Tier 3 collinearity filter + SHAP-RFE."""

    best_features: list[str]
    best_val_mae: float
    dropped_collinear: list[str]
    initial_feature_count: int
    trace: list[dict[str, Any]] = field(default_factory=list)
    collinear_drop_pairs: list[dict[str, Any]] = field(default_factory=list)
    initial_shap_importances: dict[str, float] = field(default_factory=dict)
    best_shap_importances: dict[str, float] = field(default_factory=dict)
    best_val_shap_values: pd.DataFrame | None = field(default=None, repr=False)

    def to_log_dict(self) -> dict[str, Any]:
        """Serialize fold feature-selection results and SHAP rankings for JSON logging."""
        return {
            "initial_feature_count": self.initial_feature_count,
            "post_collinearity_count": self.initial_feature_count - len(self.dropped_collinear),
            "selected_feature_count": len(self.best_features),
            "best_val_mae": float(self.best_val_mae),
            "dropped_collinear": list(self.dropped_collinear),
            "collinear_drop_pairs": list(self.collinear_drop_pairs),
            "selected_features": list(self.best_features),
            "best_shap_importances": dict(self.best_shap_importances),
            "initial_shap_importances": dict(self.initial_shap_importances),
            "trace": list(self.trace),
        }


def filter_collinear_features_with_details(
    X_sub_train: pd.DataFrame,
    threshold: float = 0.85,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Drop redundant collinear features on ``X_sub_train`` and record exact pair diagnostics.

    Iteratively identifies the most correlated pair ``(a, b)`` with ``|r| > threshold``
    and deterministically drops the feature with higher mean absolute correlation
    to all other remaining features (tie-breaking lexicographically).

    Args:
        X_sub_train: Preprocessed numeric Sub-Train feature DataFrame.
        threshold: Maximum allowed absolute Pearson correlation (default ``0.85``).

    Returns:
        Tuple of ``(kept_features, dropped_features, drop_pair_records)``.
    """
    remaining: list[str] = list(X_sub_train.columns)
    dropped: list[str] = []
    drop_pairs: list[dict[str, Any]] = []

    if len(remaining) <= 1:
        return remaining, dropped, drop_pairs

    corr_df = X_sub_train.corr(method="pearson").abs().fillna(0.0)
    np.fill_diagonal(corr_df.values, 0.0)

    while len(remaining) > 1:
        sub_corr = corr_df.loc[remaining, remaining]
        mat = sub_corr.to_numpy()
        max_corr = float(np.max(mat))
        if max_corr <= threshold:
            break

        rows, cols = np.where(np.isclose(mat, max_corr, atol=1e-12))
        pairs: list[tuple[str, str]] = []
        for r_idx, c_idx in zip(rows, cols):
            if r_idx < c_idx:
                pairs.append((remaining[int(r_idx)], remaining[int(c_idx)]))
        if not pairs:
            break
        pairs.sort()
        feat_a, feat_b = pairs[0]

        mean_a = float(sub_corr.loc[feat_a].sum() / max(len(remaining) - 1, 1))
        mean_b = float(sub_corr.loc[feat_b].sum() / max(len(remaining) - 1, 1))

        if mean_a > mean_b + 1e-12:
            to_drop, kept_partner = feat_a, feat_b
            drop_mean, kept_mean = mean_a, mean_b
        elif mean_b > mean_a + 1e-12:
            to_drop, kept_partner = feat_b, feat_a
            drop_mean, kept_mean = mean_b, mean_a
        else:
            to_drop = max(feat_a, feat_b)
            kept_partner = min(feat_a, feat_b)
            drop_mean = mean_a if to_drop == feat_a else mean_b
            kept_mean = mean_b if to_drop == feat_a else mean_a

        remaining.remove(to_drop)
        dropped.append(to_drop)
        drop_pairs.append(
            {
                "dropped_feature": to_drop,
                "correlated_with": kept_partner,
                "abs_pearson_r": round(max_corr, 6),
                "dropped_mean_corr": round(drop_mean, 6),
                "kept_mean_corr": round(kept_mean, 6),
            }
        )

    LOGGER.info(
        "Collinearity filter (|r| > %.2f) on Sub-Train: kept %d / %d features (dropped %d).",
        threshold,
        len(remaining),
        len(X_sub_train.columns),
        len(dropped),
    )
    return remaining, dropped, drop_pairs


def filter_collinear_features(
    X_sub_train: pd.DataFrame,
    threshold: float = 0.85,
) -> tuple[list[str], list[str]]:
    """Drop redundant collinear features using Pearson correlation on ``X_sub_train`` only.

    Args:
        X_sub_train: Preprocessed numeric Sub-Train feature DataFrame.
        threshold: Maximum allowed absolute Pearson correlation (default ``0.85``).

    Returns:
        Tuple of ``(kept_features, dropped_features)``.
    """
    kept, dropped, _ = filter_collinear_features_with_details(
        X_sub_train, threshold=threshold
    )
    return kept, dropped


def _build_rfe_xgb_regressor(config: dict[str, Any], random_seed: int) -> XGBRegressor:
    """Create a fresh ``XGBRegressor`` configured for SHAP-RFE."""
    xgb_cfg = config.get("models", {}).get("xgboost", {})
    debug_mode = bool(config.get("debug_mode", False))
    n_estimators = (
        int(config.get("debug", {}).get("xgb_n_estimators", 10))
        if debug_mode
        else int(xgb_cfg.get("n_estimators", 300))
    )
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=int(xgb_cfg.get("max_depth", 6)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        objective=str(xgb_cfg.get("objective", "reg:squarederror")),
        n_jobs=int(xgb_cfg.get("n_jobs", -1)),
        random_state=random_seed,
        verbosity=0,
    )


def run_shap_rfe(
    X_sub_train: pd.DataFrame,
    y_sub_train: np.ndarray | pd.Series,
    X_sub_val: pd.DataFrame,
    y_sub_val: np.ndarray | pd.Series,
    config: dict[str, Any],
    random_seed: int | None = None,
) -> ShapRfeResult:
    """Run collinearity filtering followed by validation-only iterative SHAP-RFE.

    Records full out-of-sample validation SHAP importances (both initial iteration
    and best selected subset) as well as the event-level validation SHAP matrix
    for the best iteration.

    Args:
        X_sub_train: Preprocessed Sub-Train feature DataFrame.
        y_sub_train: Sub-Train target values.
        X_sub_val: Preprocessed Sub-Val feature DataFrame.
        y_sub_val: Sub-Val target values.
        config: Loaded configuration dictionary.
        random_seed: Optional seed override; defaults to ``config["random_seed"]``.

    Returns:
        ``ShapRfeResult`` containing the best feature subset, validation SHAP
        importances/matrix, collinearity drop pairs, and full iteration trace.
    """
    if X_sub_train.empty or X_sub_val.empty:
        raise ValueError("X_sub_train and X_sub_val must both be non-empty for SHAP-RFE.")

    seed = int(config["random_seed"] if random_seed is None else random_seed)
    rfe_cfg = config.get("shap_rfe", {})
    corr_threshold = float(rfe_cfg.get("collinearity_threshold", 0.85))
    drop_fraction = float(rfe_cfg.get("drop_fraction", 0.10))
    min_drop_count = int(rfe_cfg.get("min_drop_count", 1))
    patience = int(rfe_cfg.get("patience", 2))
    min_features = int(rfe_cfg.get("min_features", 1))

    y_tr = np.asarray(y_sub_train, dtype=np.float64)
    y_va = np.asarray(y_sub_val, dtype=np.float64)

    initial_count = int(X_sub_train.shape[1])
    current_features, dropped_collinear, collinear_drop_pairs = (
        filter_collinear_features_with_details(X_sub_train, threshold=corr_threshold)
    )

    if not current_features:
        raise ValueError("No features remained after collinearity filtering.")

    best_val_mae = float("inf")
    best_features = list(current_features)
    initial_shap_importances: dict[str, float] = {}
    best_shap_importances: dict[str, float] = {}
    best_val_shap_df: pd.DataFrame | None = None
    degradation_streak = 0
    trace: list[dict[str, Any]] = []
    iteration = 0

    while len(current_features) >= min_features:
        X_tr_curr = X_sub_train[current_features]
        X_va_curr = X_sub_val[current_features]

        model = _build_rfe_xgb_regressor(config, random_seed=seed + iteration)
        model.fit(X_tr_curr, y_tr)

        val_preds = model.predict(X_va_curr)
        val_mae = float(mean_absolute_error(y_va, val_preds))

        # Validation-only SHAP (Invariant 4): NEVER evaluate TreeExplainer on X_sub_train
        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(explainer.shap_values(X_va_curr), dtype=np.float64)
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(-1, 1)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        sorted_desc = sorted(
            zip(current_features, mean_abs_shap.tolist()),
            key=lambda item: (-item[1], item[0]),
        )
        curr_importance_map = {feat: float(score) for feat, score in sorted_desc}

        if iteration == 0:
            initial_shap_importances = dict(curr_importance_map)

        if val_mae <= best_val_mae:
            best_val_mae = val_mae
            best_features = list(current_features)
            best_shap_importances = dict(curr_importance_map)
            best_val_shap_df = pd.DataFrame(
                shap_values,
                columns=list(current_features),
                index=X_va_curr.index,
            )
            degradation_streak = 0
        else:
            degradation_streak += 1

        step_record: dict[str, Any] = {
            "iteration": iteration,
            "num_features": len(current_features),
            "val_mae": val_mae,
            "best_val_mae": best_val_mae,
            "degradation_streak": degradation_streak,
            "top_5_features": [
                {"feature": f, "mean_abs_shap": round(s, 6)}
                for f, s in sorted_desc[:5]
            ],
            "dropped_next": [],
        }

        LOGGER.debug(
            "SHAP-RFE iter %d | n_feat=%d | val_mae=%.6f | best=%.6f | streak=%d",
            iteration,
            len(current_features),
            val_mae,
            best_val_mae,
            degradation_streak,
        )

        if degradation_streak >= patience:
            trace.append(step_record)
            LOGGER.info(
                "SHAP-RFE stopped at iter %d after %d consecutive degradations. "
                "Best subset: %d features (val_mae=%.6f).",
                iteration,
                degradation_streak,
                len(best_features),
                best_val_mae,
            )
            break

        if len(current_features) <= min_features:
            trace.append(step_record)
            break

        n_drop = max(min_drop_count, int(np.floor(len(current_features) * drop_fraction)))
        n_drop = min(n_drop, len(current_features) - min_features)
        if n_drop <= 0:
            trace.append(step_record)
            break

        ranked_asc = sorted(
            zip(current_features, mean_abs_shap.tolist()),
            key=lambda item: (item[1], item[0]),
        )
        to_drop = [feat for feat, _score in ranked_asc[:n_drop]]
        step_record["dropped_next"] = to_drop
        trace.append(step_record)

        drop_set = set(to_drop)
        current_features = [f for f in current_features if f not in drop_set]
        iteration += 1

    return ShapRfeResult(
        best_features=best_features,
        best_val_mae=best_val_mae,
        dropped_collinear=dropped_collinear,
        initial_feature_count=initial_count,
        trace=trace,
        collinear_drop_pairs=collinear_drop_pairs,
        initial_shap_importances=initial_shap_importances,
        best_shap_importances=best_shap_importances,
        best_val_shap_values=best_val_shap_df,
    )
