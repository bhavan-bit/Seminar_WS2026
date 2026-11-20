"""Stage 2c: 5-Fold Case-Level Cross-Validation orchestrator over ``ocpa`` ``Feature_Storage``.

Splits the 20 ``Feature_Graph`` objects using ``KFold(n_splits=5, shuffle=True, random_state=seed)``
(16 Train graphs / 4 Test graphs per fold) and an inner 12 Sub-Train / 4 Sub-Val split,
performing inside every fold:
    1. Slicing ``Feature_Storage`` strictly by fold graph indices via ``ocpa_encoding.py``.
    2. Fitting fold-scoped normalization strictly on training graphs (never globally).
    3. Running nested Tier 3 collinearity filtering + validation-only SHAP-RFE.
    4. Encoding tabular (``ocpa`` ``construct_table``), sequential (``ocpa`` ``construct_sequence``),
       and PyG graph inputs per tier.
    5. Training XGBoost, LSTM, and GNN across Tiers 1–3 and aggregating holdout OOF predictions.

Strict Isolation Invariant:
    This module never loads or references Stage 1b structural entanglement artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ocpa.algo.predictive_monitoring.obj import Feature_Storage
from sklearn.model_selection import KFold, ShuffleSplit

from src import configure_logging, load_config, set_global_seed
from src.feature_selection import run_shap_rfe
from src.models.graph_gnn import GraphGNNModel
from src.models.seq_lstm import SequenceLSTMModel
from src.models.tabular_xgb import TabularXGBModel
from src.ocpa_encoding import (
    OcpaFeatureScaler,
    encode_pyg_graphs,
    encode_sequential,
    encode_tabular,
    filter_feature_keys_for_tier,
    get_case_ids,
    slice_feature_storage,
)

LOGGER = logging.getLogger(__name__)

OOF_PREDICTION_COLUMNS: list[str] = [
    "Pred_XGB_T1",
    "Pred_XGB_T2",
    "Pred_XGB_T3",
    "Pred_LSTM_T1",
    "Pred_LSTM_T2",
    "Pred_LSTM_T3",
    "Pred_GNN_T1",
    "Pred_GNN_T2",
    "Pred_GNN_T3",
]


def _assert_disjoint_cases(
    fs_left: Feature_Storage,
    fs_right: Feature_Storage,
    label_left: str,
    label_right: str,
) -> None:
    """Assert zero ``Case_ID`` / ``Feature_Graph`` overlap between two splits (Invariant 1)."""
    left_ids = set(get_case_ids(fs_left))
    right_ids = set(get_case_ids(fs_right))
    overlap = left_ids.intersection(right_ids)
    if overlap:
        raise AssertionError(
            f"Intra-case leakage detected between {label_left} and {label_right}: "
            f"overlapping Case_IDs={sorted(overlap)}"
        )


def _align_predictions_to_base(
    base_meta: pd.DataFrame,
    pred_meta: pd.DataFrame,
    preds: np.ndarray,
    case_col: str,
    event_col: str,
) -> np.ndarray:
    """Align model predictions onto ``base_meta``'s exact ``(Case_ID, Event_ID)`` row order."""
    pred_lookup = {
        (int(cid), eid): float(p)
        for cid, eid, p in zip(
            pred_meta[case_col].tolist(),
            pred_meta[event_col].tolist(),
            preds.tolist(),
        )
    }
    aligned = np.array(
        [
            pred_lookup[(int(cid), eid)]
            for cid, eid in zip(
                base_meta[case_col].tolist(),
                base_meta[event_col].tolist(),
            )
        ],
        dtype=np.float64,
    )
    if len(aligned) != len(base_meta):
        raise ValueError("Prediction alignment length mismatch.")
    return aligned


def generate_graph_cv_splits(
    num_graphs: int,
    n_splits: int = 5,
    inner_test_size: float = 0.25,
    random_seed: int = 42,
) -> list[dict[str, np.ndarray]]:
    """Generate outer ``KFold`` and inner ``ShuffleSplit`` graph index splits.

    Args:
        num_graphs: Total number of ``Feature_Graph`` objects (e.g., 20).
        n_splits: Number of outer folds (default 5).
        inner_test_size: Fraction of outer train graphs allocated to Sub-Val (default 0.25).
        random_seed: Random seed for outer ``KFold`` and inner ``ShuffleSplit``.

    Returns:
        List of dicts per fold containing ``fold_id``, ``train_idx``, ``test_idx``,
        ``sub_train_idx``, and ``sub_val_idx``.
    """
    all_indices = np.arange(num_graphs, dtype=np.int64)
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds: list[dict[str, np.ndarray]] = []

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(all_indices)):
        inner_splitter = ShuffleSplit(
            n_splits=1,
            test_size=inner_test_size,
            random_state=random_seed,
        )
        sub_tr_rel, sub_va_rel = next(inner_splitter.split(train_idx))
        sub_train_idx = train_idx[sub_tr_rel]
        sub_val_idx = train_idx[sub_va_rel]

        if set(train_idx.tolist()) & set(test_idx.tolist()):
            raise AssertionError("Outer Train/Test graph index overlap detected.")
        if set(sub_train_idx.tolist()) & set(sub_val_idx.tolist()):
            raise AssertionError("Inner Sub-Train/Sub-Val graph index overlap detected.")

        folds.append(
            {
                "fold_id": np.int64(fold_id),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "sub_train_idx": sub_train_idx,
                "sub_val_idx": sub_val_idx,
            }
        )

    return folds


def run_cross_validation_in_memory(
    feature_storage: Feature_Storage,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run leak-free 5-fold cross-validation across all 9 model-tier combinations.

    Args:
        feature_storage: Unnormalized Stage 1a ``ocpa`` ``Feature_Storage``.
        config: Loaded configuration dictionary.

    Returns:
        Tuple of ``(oof_predictions_df, selected_features_per_fold)``.
    """
    seed = int(config["random_seed"])
    set_global_seed(seed)

    if feature_storage.scaler is not None:
        raise ValueError(
            "Input Feature_Storage appears pre-normalized. Stage 2 requires raw unnormalized Feature_Storage."
        )

    case_col = config["identifiers"]["case_id"]
    event_col = config["identifiers"]["event_id"]
    target_col = config["target"]["column"]
    n_splits = int(config["cv"]["n_splits"])
    inner_test_size = float(config["cv"]["inner_test_size"])
    window_size = int(config.get("models", {}).get("lstm", {}).get("window_size", 20))

    tier_keys: dict[str, list[str]] = {
        "T1": filter_feature_keys_for_tier(feature_storage, config, "tier_1"),
        "T2": filter_feature_keys_for_tier(feature_storage, config, "tier_2"),
        "T3": filter_feature_keys_for_tier(feature_storage, config, "tier_3"),
    }
    for label, keys in tier_keys.items():
        if not keys:
            raise ValueError(f"Resolved zero feature keys for {label}.")
        LOGGER.info("Tier %s feature key count: %d", label, len(keys))

    num_graphs = len(feature_storage.feature_graphs)
    total_input_events = sum(len(g.nodes) for g in feature_storage.feature_graphs)
    folds = generate_graph_cv_splits(
        num_graphs=num_graphs,
        n_splits=n_splits,
        inner_test_size=inner_test_size,
        random_seed=seed,
    )

    oof_frames: list[pd.DataFrame] = []
    selected_features_log: dict[str, Any] = {}

    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_fs_raw = slice_feature_storage(feature_storage, fold["train_idx"])
        test_fs_raw = slice_feature_storage(feature_storage, fold["test_idx"])
        sub_train_fs_raw = slice_feature_storage(feature_storage, fold["sub_train_idx"])
        sub_val_fs_raw = slice_feature_storage(feature_storage, fold["sub_val_idx"])

        _assert_disjoint_cases(train_fs_raw, test_fs_raw, "Train", "Test")
        _assert_disjoint_cases(sub_train_fs_raw, sub_val_fs_raw, "Sub-Train", "Sub-Val")

        LOGGER.info(
            "=== Fold %d/%d | Train=%d cases | Sub-Train=%d cases | Sub-Val=%d cases | Test=%d cases ===",
            fold_id + 1,
            n_splits,
            len(train_fs_raw.feature_graphs),
            len(sub_train_fs_raw.feature_graphs),
            len(sub_val_fs_raw.feature_graphs),
            len(test_fs_raw.feature_graphs),
        )

        # ------------------------------------------------------------------
        # Step 1: Inner-fold scaling (fit strictly on 12 Sub-Train graphs)
        # ------------------------------------------------------------------
        inner_scaler = OcpaFeatureScaler(target_key=target_col)
        sub_train_fs_scaled = inner_scaler.fit_transform(sub_train_fs_raw)
        sub_val_fs_scaled = inner_scaler.transform(sub_val_fs_raw)
        test_fs_inner_scaled = inner_scaler.transform(test_fs_raw)

        # ------------------------------------------------------------------
        # Step 2: Nested Tier 3 SHAP-RFE strictly on (Sub-Train, Sub-Val)
        # ------------------------------------------------------------------
        X_sub_tr_t3, y_sub_tr_t3, _ = encode_tabular(
            sub_train_fs_scaled,
            feature_keys=tier_keys["T3"],
            target_key=target_col,
            case_col=case_col,
            event_col=event_col,
        )
        X_sub_va_t3, y_sub_va_t3, _ = encode_tabular(
            sub_val_fs_scaled,
            feature_keys=tier_keys["T3"],
            target_key=target_col,
            case_col=case_col,
            event_col=event_col,
        )

        rfe_result = run_shap_rfe(
            X_sub_train=X_sub_tr_t3,
            y_sub_train=y_sub_tr_t3,
            X_sub_val=X_sub_va_t3,
            y_sub_val=y_sub_va_t3,
            config=config,
            random_seed=seed + fold_id * 100,
        )
        t3_selected_keys = rfe_result.best_features
        selected_features_log[f"fold_{fold_id}"] = {
            "train_cases": get_case_ids(train_fs_raw),
            "sub_train_cases": get_case_ids(sub_train_fs_raw),
            "sub_val_cases": get_case_ids(sub_val_fs_raw),
            "test_cases": get_case_ids(test_fs_raw),
            **rfe_result.to_log_dict(),
        }

        # ------------------------------------------------------------------
        # Step 3: Outer-fold scaling (fit strictly on 16 Train graphs) for XGBoost
        # ------------------------------------------------------------------
        outer_scaler = OcpaFeatureScaler(target_key=target_col)
        train_fs_outer_scaled = outer_scaler.fit_transform(train_fs_raw)
        test_fs_outer_scaled = outer_scaler.transform(test_fs_raw)

        # Canonical holdout test metadata & targets
        _, y_test_base, base_test_meta = encode_tabular(
            test_fs_outer_scaled,
            feature_keys=tier_keys["T1"],
            target_key=target_col,
            case_col=case_col,
            event_col=event_col,
        )
        fold_preds: dict[str, Any] = {
            case_col: base_test_meta[case_col].to_numpy(),
            event_col: base_test_meta[event_col].to_numpy(),
            "Fold_ID": np.full(len(base_test_meta), fold_id, dtype=np.int64),
            target_col: y_test_base,
        }

        # ------------------------------------------------------------------
        # Step 4: Train & evaluate XGBoost, LSTM, and GNN across Tiers 1–3
        # ------------------------------------------------------------------
        active_tier_map = {
            "T1": tier_keys["T1"],
            "T2": tier_keys["T2"],
            "T3": t3_selected_keys,
        }

        for tier_idx, (tier_code, active_keys) in enumerate(active_tier_map.items()):
            tier_seed = seed + fold_id * 100 + tier_idx * 10

            # 1. XGBoost (ocpa tabular encoding on 16 outer Train graphs)
            X_tr_tab, y_tr_tab, _ = encode_tabular(
                train_fs_outer_scaled,
                feature_keys=active_keys,
                target_key=target_col,
                case_col=case_col,
                event_col=event_col,
            )
            X_te_tab, _, te_tab_meta = encode_tabular(
                test_fs_outer_scaled,
                feature_keys=active_keys,
                target_key=target_col,
                case_col=case_col,
                event_col=event_col,
            )
            xgb_model = TabularXGBModel(config=config, random_seed=tier_seed)
            xgb_model.fit(X_tr_tab, y_tr_tab)
            xgb_preds = xgb_model.predict(X_te_tab)
            fold_preds[f"Pred_XGB_{tier_code}"] = _align_predictions_to_base(
                base_test_meta, te_tab_meta, xgb_preds, case_col, event_col
            )

            # 2. LSTM (ocpa sequential encoding on 12 Sub-Train / 4 Sub-Val graphs)
            X_sub_tr_seq, y_sub_tr_seq, _ = encode_sequential(
                sub_train_fs_scaled,
                feature_keys=active_keys,
                target_key=target_col,
                window_size=window_size,
                case_col=case_col,
                event_col=event_col,
            )
            X_sub_va_seq, y_sub_va_seq, _ = encode_sequential(
                sub_val_fs_scaled,
                feature_keys=active_keys,
                target_key=target_col,
                window_size=window_size,
                case_col=case_col,
                event_col=event_col,
            )
            X_te_seq, _, te_seq_meta = encode_sequential(
                test_fs_inner_scaled,
                feature_keys=active_keys,
                target_key=target_col,
                window_size=window_size,
                case_col=case_col,
                event_col=event_col,
            )
            lstm_model = SequenceLSTMModel(config=config, random_seed=tier_seed)
            lstm_model.fit(
                X_sub_tr_seq,
                y_sub_tr_seq,
                X_val=X_sub_va_seq,
                y_val=y_sub_va_seq,
            )
            lstm_preds = lstm_model.predict(X_te_seq)
            fold_preds[f"Pred_LSTM_{tier_code}"] = _align_predictions_to_base(
                base_test_meta, te_seq_meta, lstm_preds, case_col, event_col
            )

            # 3. GNN (PyG graph encoding on 12 Sub-Train / 4 Sub-Val graphs)
            tr_pyg, _ = encode_pyg_graphs(
                sub_train_fs_scaled,
                feature_keys=active_keys,
                target_key=target_col,
                case_col=case_col,
                event_col=event_col,
            )
            va_pyg, _ = encode_pyg_graphs(
                sub_val_fs_scaled,
                feature_keys=active_keys,
                target_key=target_col,
                case_col=case_col,
                event_col=event_col,
            )
            te_pyg, te_pyg_meta = encode_pyg_graphs(
                test_fs_inner_scaled,
                feature_keys=active_keys,
                target_key=target_col,
                case_col=case_col,
                event_col=event_col,
            )
            gnn_model = GraphGNNModel(config=config, random_seed=tier_seed)
            gnn_model.fit(tr_pyg, X_val=va_pyg)
            gnn_preds = gnn_model.predict(te_pyg)
            fold_preds[f"Pred_GNN_{tier_code}"] = _align_predictions_to_base(
                base_test_meta, te_pyg_meta, gnn_preds, case_col, event_col
            )

        oof_frames.append(pd.DataFrame(fold_preds))

    oof_df = pd.concat(oof_frames, ignore_index=True)
    ordered_cols = [case_col, event_col, "Fold_ID", target_col, *OOF_PREDICTION_COLUMNS]
    oof_df = oof_df[ordered_cols].sort_values([case_col, event_col]).reset_index(drop=True)

    if len(oof_df) != total_input_events:
        raise AssertionError(
            f"OOF prediction row count ({len(oof_df)}) != input event count ({total_input_events})."
        )

    LOGGER.info(
        "Completed %d-fold CV: generated OOF predictions for %d events across %d cases.",
        n_splits,
        len(oof_df),
        oof_df[case_col].nunique(),
    )
    return oof_df, selected_features_log


def run_cross_validation(
    config_path: str | Path = "config.yaml",
    debug_override: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run Stage 2c end-to-end and persist OOF predictions and per-fold feature logs.

    Args:
        config_path: Path to ``config.yaml``.
        debug_override: Optional boolean override for ``config["debug_mode"]``.

    Returns:
        Tuple of ``(oof_df, selected_features_log)``.
    """
    config = load_config(config_path)
    if debug_override is not None:
        config["debug_mode"] = debug_override
    set_global_seed(int(config["random_seed"]))

    fs_path = Path(config["paths"]["feature_storage"])
    preds_out = Path(config["paths"]["test_predictions"])
    sel_out = Path(config["paths"]["selected_features_per_fold"])

    if not fs_path.exists():
        raise FileNotFoundError(
            f"Feature_Storage not found at {fs_path}. Run Stage 1a first."
        )

    with fs_path.open("rb") as handle:
        feature_storage: Feature_Storage = pickle.load(handle)

    oof_df, selected_features_log = run_cross_validation_in_memory(
        feature_storage=feature_storage,
        config=config,
    )

    preds_out.parent.mkdir(parents=True, exist_ok=True)
    sel_out.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(preds_out, index=False)
    with sel_out.open("w", encoding="utf-8") as handle:
        json.dump(selected_features_log, handle, indent=2)

    LOGGER.info("Saved OOF predictions to %s", preds_out)
    LOGGER.info("Saved per-fold SHAP-RFE feature log to %s", sel_out)
    return oof_df, selected_features_log


def main() -> None:
    """CLI entry point for Stage 2c cross-validation."""
    parser = argparse.ArgumentParser(description="Stage 2c: 5-Fold Case-Level Cross-Validation")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="Override config.yaml to enable debug_mode",
    )
    args = parser.parse_args()
    configure_logging()
    run_cross_validation(config_path=args.config, debug_override=args.debug)


if __name__ == "__main__":
    main()
