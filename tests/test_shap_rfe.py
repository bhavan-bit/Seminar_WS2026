"""Unit tests for Stage 2a collinearity filtering and validation-only SHAP-RFE.

Verifies:
    - Pearson collinearity filter (``|r| > 0.85``) runs strictly on ``X_sub_train``.
    - ``shap.TreeExplainer.shap_values`` is invoked EXCLUSIVELY on ``X_sub_val``,
      never on ``X_sub_train`` (Invariant 4).
    - Iterative bottom-10% (min 1) feature pruning stops after 2 consecutive
      validation MAE degradations and returns the exact best feature subset.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import shap

from src import load_config
from src.feature_selection import filter_collinear_features, run_shap_rfe


def _make_rfe_dataset(
    n_train: int = 120,
    n_val: int = 40,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Create a minimal inline Sub-Train / Sub-Val dataset with signal, collinear, and noise columns."""
    rng = np.random.default_rng(seed)
    n_total = n_train + n_val

    signal_1 = rng.normal(0.0, 1.0, size=n_total)
    signal_2 = rng.normal(0.0, 1.0, size=n_total)
    # Near-exact duplicate of signal_1 (|r| > 0.99)
    collinear_dup = signal_1 * 2.0 + rng.normal(0.0, 0.01, size=n_total)

    data: dict[str, np.ndarray] = {
        "signal_1": signal_1,
        "signal_2": signal_2,
        "collinear_dup": collinear_dup,
    }
    for i in range(8):
        data[f"noise_{i}"] = rng.normal(0.0, 1.0, size=n_total)

    df = pd.DataFrame(data)
    y = 5.0 * signal_1 - 3.0 * signal_2 + rng.normal(0.0, 0.2, size=n_total)

    # Distinct, non-overlapping DataFrame indices to detect any Train/Val mix-up
    X_sub_train = df.iloc[:n_train].copy()
    X_sub_train.index = pd.RangeIndex(start=0, stop=n_train)
    y_sub_train = y[:n_train]

    X_sub_val = df.iloc[n_train:].copy()
    X_sub_val.index = pd.RangeIndex(start=1000, stop=1000 + n_val)
    y_sub_val = y[n_train:]

    return X_sub_train, y_sub_train, X_sub_val, y_sub_val


def test_collinearity_filter_drops_redundant_pair_deterministically() -> None:
    """Verify pairs with |r| > 0.85 are reduced so no remaining pair exceeds threshold."""
    X_sub_train, _, _, _ = _make_rfe_dataset()

    kept, dropped = filter_collinear_features(X_sub_train, threshold=0.85)

    # Exactly one of (signal_1, collinear_dup) must be dropped
    assert len(dropped) == 1
    assert dropped[0] in {"signal_1", "collinear_dup"}
    assert len({"signal_1", "collinear_dup"}.intersection(set(kept))) == 1

    # All remaining pairwise off-diagonal correlations must be <= 0.85
    corr = X_sub_train[kept].corr().abs().fillna(0.0)
    np.fill_diagonal(corr.values, 0.0)
    assert float(corr.to_numpy().max()) <= 0.85


def test_shap_values_computed_exclusively_on_sub_val() -> None:
    """Invariant 4: TreeExplainer.shap_values MUST only ever receive X_sub_val."""
    cfg = copy.deepcopy(load_config("config.yaml"))
    cfg["debug_mode"] = True
    cfg["debug"]["xgb_n_estimators"] = 10

    X_sub_train, y_sub_train, X_sub_val, y_sub_val = _make_rfe_dataset()
    train_indices = set(X_sub_train.index.tolist())
    val_indices = set(X_sub_val.index.tolist())

    recorded_eval_indices: list[set[int]] = []
    original_shap_values = shap.TreeExplainer.shap_values

    def spy_shap_values(self: Any, X: Any, *args: Any, **kwargs: Any) -> Any:
        assert isinstance(X, pd.DataFrame), "Expected DataFrame passed to TreeExplainer.shap_values"
        eval_idx = set(X.index.tolist())
        recorded_eval_indices.append(eval_idx)
        # Strict check: NEVER evaluated on Sub-Train
        assert eval_idx == val_indices
        assert eval_idx.isdisjoint(train_indices)
        return original_shap_values(self, X, *args, **kwargs)

    with patch.object(shap.TreeExplainer, "shap_values", new=spy_shap_values):
        result = run_shap_rfe(
            X_sub_train=X_sub_train,
            y_sub_train=y_sub_train,
            X_sub_val=X_sub_val,
            y_sub_val=y_sub_val,
            config=cfg,
        )

    assert len(recorded_eval_indices) >= 1
    assert len(result.best_features) >= 1
    assert np.isfinite(result.best_val_mae)


def test_shap_rfe_stopping_criterion_after_two_consecutive_degradations() -> None:
    """Verify loop stops after 2 consecutive MAE degradations and returns best subset."""
    cfg = copy.deepcopy(load_config("config.yaml"))
    cfg["debug_mode"] = True
    cfg["shap_rfe"]["patience"] = 2
    cfg["shap_rfe"]["drop_fraction"] = 0.10
    cfg["shap_rfe"]["min_drop_count"] = 1
    cfg["shap_rfe"]["collinearity_threshold"] = 1.0

    X_sub_train, y_sub_train, X_sub_val, y_sub_val = _make_rfe_dataset()
    # Scripted validation MAE sequence (with collinearity filter disabled at threshold=1.0):
    # iter 0 (11 feats): 10.0 (best=10.0, streak=0) -> drop 1
    # iter 1 (10 feats):  8.0 (best= 8.0, streak=0) -> BEST SUBSET! -> drop 1
    # iter 2 ( 9 feats):  8.5 (degrade, streak=1)   -> drop 1
    # iter 3 ( 8 feats):  9.0 (degrade, streak=2)   -> STOP! Should return iter 1's 10 features
    # iter 4 ( 7 feats):  1.0 (must never be reached!)
    scripted_maes = iter([10.0, 8.0, 8.5, 9.0, 1.0])

    with patch("src.feature_selection.mean_absolute_error", side_effect=lambda *_a, **_kw: next(scripted_maes)):
        result = run_shap_rfe(
            X_sub_train=X_sub_train,
            y_sub_train=y_sub_train,
            X_sub_val=X_sub_val,
            y_sub_val=y_sub_val,
            config=cfg,
        )

    assert len(result.trace) == 4
    assert result.trace[-1]["degradation_streak"] == 2
    assert result.best_val_mae == 8.0
    # Started with 11 features -> iter 0 dropped 1 -> iter 1 had 10 features (best)
    assert len(result.best_features) == 10

