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
import gc
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ocpa.algo.predictive_monitoring.obj import Feature_Storage
from sklearn.model_selection import KFold, ShuffleSplit

from src import configure_logging, load_config, resolve_tier_columns, set_global_seed
from src.feature_selection import run_shap_rfe
from src.models.graph_gnn import GraphGNNModel
from src.models.seq_lstm import SequenceLSTMModel
from src.models.tabular_xgb import TabularXGBModel
from src.ocpa_encoding import (
    CausalPrefixSequenceDataset,
    CompactFoldScaler,
    CompactOcpaDataset,
    build_pyg_graphs_from_compact,
    filter_feature_keys_for_tier,
    unpack_feature_storage_compact,
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
    cv_groups: list[Any] | None = None,
) -> list[dict[str, np.ndarray]]:
    """Generate outer ``KFold`` and inner ``ShuffleSplit`` graph index splits.

    When ``cv_groups`` is provided (e.g., ``Match`` IDs linking paired player
    ``Session``s), both the outer 5-fold split and inner Sub-Train/Sub-Val split
    operate strictly at the unique-group level so paired cases from the same
    group never split across Train/Test or Sub-Train/Sub-Val.

    Args:
        num_graphs: Total number of ``Feature_Graph`` objects (e.g., 20).
        n_splits: Number of outer folds (default 5).
        inner_test_size: Fraction of outer train graphs/groups allocated to Sub-Val.
        random_seed: Random seed for outer ``KFold`` and inner ``ShuffleSplit``.
        cv_groups: Optional sequence of length ``num_graphs`` giving the parent
            CV group ID (e.g., Match ID) per ``Feature_Graph``.

    Returns:
        List of dicts per fold containing ``fold_id``, ``train_idx``, ``test_idx``,
        ``sub_train_idx``, and ``sub_val_idx``.
    """
    folds: list[dict[str, np.ndarray]] = []

    if cv_groups is not None and len(cv_groups) == num_graphs and len(set(cv_groups)) >= n_splits:
        unique_groups = np.array(sorted(set(cv_groups)), dtype=object)
        group_to_indices: dict[Any, list[int]] = {}
        for idx, grp in enumerate(cv_groups):
            group_to_indices.setdefault(grp, []).append(idx)

        outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        for fold_id, (tr_g_idx, te_g_idx) in enumerate(outer_cv.split(unique_groups)):
            tr_groups = unique_groups[tr_g_idx]
            te_groups = unique_groups[te_g_idx]

            inner_splitter = ShuffleSplit(
                n_splits=1,
                test_size=inner_test_size,
                random_state=random_seed,
            )
            sub_tr_g_rel, sub_va_g_rel = next(inner_splitter.split(tr_groups))
            sub_tr_groups = tr_groups[sub_tr_g_rel]
            sub_va_groups = tr_groups[sub_va_g_rel]

            if set(tr_groups.tolist()) & set(te_groups.tolist()):
                raise AssertionError("Outer CV group overlap detected.")
            if set(sub_tr_groups.tolist()) & set(sub_va_groups.tolist()):
                raise AssertionError("Inner CV group overlap detected.")

            train_idx = np.array(
                sorted(i for g in tr_groups for i in group_to_indices[g]), dtype=np.int64
            )
            test_idx = np.array(
                sorted(i for g in te_groups for i in group_to_indices[g]), dtype=np.int64
            )
            sub_train_idx = np.array(
                sorted(i for g in sub_tr_groups for i in group_to_indices[g]), dtype=np.int64
            )
            sub_val_idx = np.array(
                sorted(i for g in sub_va_groups for i in group_to_indices[g]), dtype=np.int64
            )

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

    all_indices = np.arange(num_graphs, dtype=np.int64)
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

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


def _export_stage2_diagnostics(
    diagnostics_dir: Path,
    oof_df: pd.DataFrame,
    selected_features_log: dict[str, Any],
    training_history_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    val_shap_frames: dict[int, pd.DataFrame],
    case_col: str,
    target_col: str,
    n_splits: int,
) -> None:
    """Export intermediate Stage 2 diagnostic tables and validation SHAP matrices."""
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-fold validation SHAP feature importances & selection status
    imp_rows: list[dict[str, Any]] = []
    rfe_trace_rows: list[dict[str, Any]] = []
    collinear_rows: list[dict[str, Any]] = []

    for fold_key in sorted(selected_features_log.keys()):
        fold_id = int(fold_key.split("_")[1])
        info = selected_features_log[fold_key]
        selected_set = set(info.get("selected_features", []))
        init_imp: dict[str, float] = info.get("initial_shap_importances", {})
        best_imp: dict[str, float] = info.get("best_shap_importances", {})
        dropped_collinear_set = set(info.get("dropped_collinear", []))

        best_rank_map = {
            feat: rank
            for rank, (feat, _val) in enumerate(
                sorted(best_imp.items(), key=lambda kv: (-kv[1], kv[0])), start=1
            )
        }
        all_seen = sorted(set(init_imp.keys()) | set(best_imp.keys()) | dropped_collinear_set)
        for feat in all_seen:
            status = (
                "Dropped_Collinear"
                if feat in dropped_collinear_set
                else ("Selected" if feat in selected_set else "Pruned_SHAP_RFE")
            )
            imp_rows.append(
                {
                    "Fold_ID": fold_id,
                    "Feature": feat,
                    "Status": status,
                    "Selected_In_Best": feat in selected_set,
                    "Initial_Mean_Abs_SHAP": float(init_imp[feat]) if feat in init_imp else float("nan"),
                    "Best_Mean_Abs_SHAP": float(best_imp[feat]) if feat in best_imp else float("nan"),
                    "Rank_In_Best": best_rank_map.get(feat, pd.NA),
                }
            )

        for step in info.get("trace", []):
            dropped_next = step.get("dropped_next", [])
            rfe_trace_rows.append(
                {
                    "Fold_ID": fold_id,
                    "Iteration": int(step["iteration"]),
                    "Num_Features": int(step["num_features"]),
                    "Val_MAE": float(step["val_mae"]),
                    "Best_Val_MAE": float(step["best_val_mae"]),
                    "Degradation_Streak": int(step["degradation_streak"]),
                    "Dropped_Count": len(dropped_next),
                    "Dropped_Features": ";".join(dropped_next),
                }
            )

        for pair in info.get("collinear_drop_pairs", []):
            collinear_rows.append({"Fold_ID": fold_id, **pair})

    imp_df = pd.DataFrame(imp_rows)
    if not imp_df.empty:
        imp_df = imp_df.sort_values(
            ["Fold_ID", "Selected_In_Best", "Best_Mean_Abs_SHAP", "Initial_Mean_Abs_SHAP"],
            ascending=[True, False, False, False],
        )
        imp_df.to_csv(diagnostics_dir / "shap_feature_importance_per_fold.csv", index=False)

        # 2. Cross-fold SHAP stability summary
        stability = (
            imp_df.groupby("Feature", as_index=False)
            .agg(
                Selection_Count=("Selected_In_Best", "sum"),
                Collinear_Drop_Count=("Status", lambda s: int((s == "Dropped_Collinear").sum())),
                Mean_Best_Abs_SHAP=("Best_Mean_Abs_SHAP", "mean"),
                Std_Best_Abs_SHAP=("Best_Mean_Abs_SHAP", "std"),
                Mean_Initial_Abs_SHAP=("Initial_Mean_Abs_SHAP", "mean"),
            )
            .assign(Selection_Rate=lambda d: d["Selection_Count"] / float(n_splits))
            .sort_values(
                ["Selection_Count", "Mean_Best_Abs_SHAP", "Mean_Initial_Abs_SHAP"],
                ascending=[False, False, False],
            )
        )
        stability = stability[
            [
                "Feature",
                "Selection_Count",
                "Selection_Rate",
                "Collinear_Drop_Count",
                "Mean_Best_Abs_SHAP",
                "Std_Best_Abs_SHAP",
                "Mean_Initial_Abs_SHAP",
            ]
        ]
        stability.to_csv(diagnostics_dir / "shap_feature_stability_summary.csv", index=False)

    pd.DataFrame(rfe_trace_rows).to_csv(diagnostics_dir / "shap_rfe_trace.csv", index=False)
    pd.DataFrame(collinear_rows).to_csv(diagnostics_dir / "collinear_dropped_pairs.csv", index=False)

    # 3. Event-level out-of-sample validation SHAP matrices per fold
    for fold_id, shap_df in sorted(val_shap_frames.items()):
        shap_df.to_parquet(
            diagnostics_dir / f"shap_values_val_fold_{fold_id}.parquet",
            index=False,
        )

    # 4. Per-epoch LSTM & GNN training curves
    pd.DataFrame(training_history_rows).to_csv(
        diagnostics_dir / "training_history.csv", index=False
    )

    # 5. Per-fold holdout Test MAE across all 9 model-tier combinations
    model_tier_specs = [
        ("Tier 1", "XGBoost", "Pred_XGB_T1"),
        ("Tier 2", "XGBoost", "Pred_XGB_T2"),
        ("Tier 3", "XGBoost", "Pred_XGB_T3"),
        ("Tier 1", "LSTM", "Pred_LSTM_T1"),
        ("Tier 2", "LSTM", "Pred_LSTM_T2"),
        ("Tier 3", "LSTM", "Pred_LSTM_T3"),
        ("Tier 1", "GNN", "Pred_GNN_T1"),
        ("Tier 2", "GNN", "Pred_GNN_T2"),
        ("Tier 3", "GNN", "Pred_GNN_T3"),
    ]
    fold_metric_rows: list[dict[str, Any]] = []
    for fold_id, fgrp in oof_df.groupby("Fold_ID", sort=True):
        f_info = selected_features_log.get(f"fold_{int(fold_id)}", {})
        test_groups_str = ";".join(str(x) for x in f_info.get("test_cv_groups", []))
        test_cases_str = ";".join(str(x) for x in f_info.get("test_cases", []))
        y_true = fgrp[target_col].to_numpy(dtype=np.float64)
        for tier_label, model_label, pred_col in model_tier_specs:
            abs_err = np.abs(y_true - fgrp[pred_col].to_numpy(dtype=np.float64))
            ev_mae = float(np.mean(abs_err))
            case_mae = float(
                pd.DataFrame({case_col: fgrp[case_col].to_numpy(), "ae": abs_err})
                .groupby(case_col)["ae"]
                .mean()
                .mean()
            )
            fold_metric_rows.append(
                {
                    "Fold_ID": int(fold_id),
                    "Test_CV_Groups": test_groups_str,
                    "Test_Cases": test_cases_str,
                    "Num_Events": int(len(fgrp)),
                    "Tier": tier_label,
                    "Model": model_label,
                    "Event_Level_MAE": ev_mae,
                    "Case_Normalized_MAE": case_mae,
                }
            )
    pd.DataFrame(fold_metric_rows).to_csv(diagnostics_dir / "fold_metrics.csv", index=False)

    # 6. Per-case MAE across all 20 cases
    case_rows: list[dict[str, Any]] = []
    for case_id, cgrp in oof_df.groupby(case_col, sort=True):
        y_true = cgrp[target_col].to_numpy(dtype=np.float64)
        row: dict[str, Any] = {
            case_col: int(case_id),
            "Fold_ID": int(cgrp["Fold_ID"].iloc[0]),
            "Num_Events": int(len(cgrp)),
            "Mean_y_true": float(np.mean(y_true)),
            "Max_y_true": float(np.max(y_true)),
        }
        for pred_col in OOF_PREDICTION_COLUMNS:
            row[f"{pred_col}_MAE"] = float(
                np.mean(np.abs(y_true - cgrp[pred_col].to_numpy(dtype=np.float64)))
            )
        case_rows.append(row)
    pd.DataFrame(case_rows).to_csv(diagnostics_dir / "per_case_mae.csv", index=False)

    # 7. Per-fold & aggregated model runtimes (Encode / Train / Predict / SHAP-RFE)
    if runtime_rows:
        rt_df = pd.DataFrame(runtime_rows)
        rt_df.to_csv(diagnostics_dir / "model_runtimes_per_fold.csv", index=False)

        rt_summary = (
            rt_df.groupby(["Tier", "Model"], as_index=False, sort=False)
            .agg(
                Mean_Num_Features=("Num_Features", "mean"),
                Mean_Epochs=("Epochs_Run", "mean"),
                Mean_Encode_Sec=("Encode_Seconds", "mean"),
                Mean_Train_Sec=("Train_Seconds", "mean"),
                Mean_Predict_Sec=("Predict_Seconds", "mean"),
                Mean_Total_Sec=("Total_Seconds", "mean"),
                Std_Total_Sec=("Total_Seconds", "std"),
                Total_5Fold_Sec=("Total_Seconds", "sum"),
                Total_SHAP_RFE_Sec=("SHAP_RFE_Seconds", "sum"),
            )
        )
        rt_summary["Total_5Fold_With_SHAP_RFE_Sec"] = (
            rt_summary["Total_5Fold_Sec"] + rt_summary["Total_SHAP_RFE_Sec"]
        )
        tier_order = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}
        model_order = {"XGBoost": 1, "LSTM": 2, "GNN": 3}
        rt_summary["_t"] = rt_summary["Tier"].map(tier_order)
        rt_summary["_m"] = rt_summary["Model"].map(model_order)
        rt_summary = rt_summary.sort_values(["_t", "_m"]).drop(columns=["_t", "_m"]).reset_index(drop=True)
        rt_summary.to_csv(diagnostics_dir / "model_runtimes_summary.csv", index=False)

    LOGGER.info("Saved Stage 2 intermediate diagnostics & validation SHAP artifacts to %s", diagnostics_dir)


def run_cross_validation_in_memory(
    feature_storage: Feature_Storage | CompactOcpaDataset,
    config: dict[str, Any],
    diagnostics_out_dir: str | Path | None = None,
    consume_storage: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run leak-free 5-fold cross-validation across all 9 model-tier combinations.

    Accepts either an unnormalized ``ocpa`` ``Feature_Storage`` or a pre-unpacked
    unnormalized ``CompactOcpaDataset`` (avoiding multi-GB Python dict overhead on
    memory-constrained hosts) and performs all fold-scoped scaling, nested
    SHAP-RFE, and model training strictly within each fold's training splits.

    Args:
        feature_storage: Unnormalized Stage 1a ``Feature_Storage`` or ``CompactOcpaDataset``.
        config: Loaded configuration dictionary.
        diagnostics_out_dir: Optional directory for intermediate CSV/SHAP diagnostics.
        consume_storage: If ``True``, releases ``Feature_Storage`` node dicts after
            compact unpacking to keep peak RAM low.

    Returns:
        Tuple of ``(oof_predictions_df, selected_features_per_fold)``.
    """
    seed = int(config["random_seed"])
    set_global_seed(seed)

    case_col = config["identifiers"]["case_id"]
    event_col = config["identifiers"]["event_id"]
    target_col = config["target"]["column"]
    n_splits = int(config["cv"]["n_splits"])
    inner_test_size = float(config["cv"]["inner_test_size"])
    window_size = int(config.get("models", {}).get("lstm", {}).get("window_size", 20))

    if isinstance(feature_storage, CompactOcpaDataset):
        compact = feature_storage
        available_keys = list(compact.feature_keys)
    else:
        if feature_storage.scaler is not None:
            raise ValueError(
                "Input Feature_Storage appears pre-normalized. Stage 2 requires raw unnormalized Feature_Storage."
            )
        available_keys = list(feature_storage.event_features)
        compact = unpack_feature_storage_compact(
            feature_storage=feature_storage,
            target_key=target_col,
            consume_storage=consume_storage,
        )

    tier_keys: dict[str, list[str]] = {
        "T1": resolve_tier_columns(available_keys, config, "tier_1"),
        "T2": resolve_tier_columns(available_keys, config, "tier_2"),
        "T3": resolve_tier_columns(available_keys, config, "tier_3"),
    }
    for label, keys in tier_keys.items():
        if not keys:
            raise ValueError(f"Resolved zero feature keys for {label}.")
        LOGGER.info("Tier %s feature key count: %d", label, len(keys))

    LOGGER.info(
        "Loaded compact float32 matrix: shape=%s (%.1f MB RAM).",
        compact.X_raw.shape,
        compact.X_raw.nbytes / (1024 * 1024),
    )

    num_graphs = len(compact.case_ids_per_graph)
    total_input_events = int(compact.X_raw.shape[0])

    raw_cv_groups = compact.cv_groups_per_graph
    cv_groups = raw_cv_groups if all(v is not None for v in raw_cv_groups) else None
    if cv_groups is not None:
        LOGGER.info(
            "Grouping %d-fold CV strictly by parent CV group (%d unique groups across %d cases).",
            n_splits,
            len(set(cv_groups)),
            num_graphs,
        )

    folds = generate_graph_cv_splits(
        num_graphs=num_graphs,
        n_splits=n_splits,
        inner_test_size=inner_test_size,
        random_seed=seed,
        cv_groups=cv_groups,
    )

    oof_frames: list[pd.DataFrame] = []
    selected_features_log: dict[str, Any] = {}
    training_history_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    val_shap_frames: dict[int, pd.DataFrame] = {}

    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_idx = fold["train_idx"].tolist()
        test_idx = fold["test_idx"].tolist()
        sub_train_idx = fold["sub_train_idx"].tolist()
        sub_val_idx = fold["sub_val_idx"].tolist()

        train_cases = sorted(compact.case_ids_per_graph[i] for i in train_idx)
        test_cases = sorted(compact.case_ids_per_graph[i] for i in test_idx)
        sub_train_cases = sorted(compact.case_ids_per_graph[i] for i in sub_train_idx)
        sub_val_cases = sorted(compact.case_ids_per_graph[i] for i in sub_val_idx)

        if set(train_cases) & set(test_cases):
            raise AssertionError(
                f"Intra-case leakage between Train and Test: {set(train_cases) & set(test_cases)}"
            )
        if set(sub_train_cases) & set(sub_val_cases):
            raise AssertionError(
                f"Intra-case leakage between Sub-Train and Sub-Val: {set(sub_train_cases) & set(sub_val_cases)}"
            )
        if cv_groups is not None:
            tr_grp_set = {cv_groups[i] for i in train_idx}
            te_grp_set = {cv_groups[i] for i in test_idx}
            sub_tr_grp_set = {cv_groups[i] for i in sub_train_idx}
            sub_va_grp_set = {cv_groups[i] for i in sub_val_idx}
            if tr_grp_set & te_grp_set:
                raise AssertionError(f"Parent CV group leakage between Train and Test: {tr_grp_set & te_grp_set}")
            if sub_tr_grp_set & sub_va_grp_set:
                raise AssertionError(
                    f"Parent CV group leakage between Sub-Train and Sub-Val: {sub_tr_grp_set & sub_va_grp_set}"
                )

        LOGGER.info(
            "=== Fold %d/%d | Train=%d cases | Sub-Train=%d cases | Sub-Val=%d cases | Test=%d cases ===",
            fold_id + 1,
            n_splits,
            len(train_cases),
            len(sub_train_cases),
            len(sub_val_cases),
            len(test_cases),
        )

        # ------------------------------------------------------------------
        # Step 1: Inner-fold scaling (fit strictly on 12 Sub-Train graphs)
        # ------------------------------------------------------------------
        inner_scaler = CompactFoldScaler().fit(compact, sub_train_idx)

        # ------------------------------------------------------------------
        # Step 2: Nested Tier 3 SHAP-RFE strictly on (Sub-Train, Sub-Val)
        # ------------------------------------------------------------------
        X_sub_tr_arr, y_sub_tr_t3, _, _, _ = inner_scaler.transform_split(
            compact, sub_train_idx, tier_keys["T3"], case_col=case_col, event_col=event_col
        )
        X_sub_va_arr, y_sub_va_t3, meta_sub_va_t3, _, _ = inner_scaler.transform_split(
            compact, sub_val_idx, tier_keys["T3"], case_col=case_col, event_col=event_col
        )
        X_sub_tr_t3 = pd.DataFrame(X_sub_tr_arr, columns=tier_keys["T3"])
        X_sub_va_t3 = pd.DataFrame(X_sub_va_arr, columns=tier_keys["T3"])

        t_rfe_start = time.perf_counter()
        rfe_result = run_shap_rfe(
            X_sub_train=X_sub_tr_t3,
            y_sub_train=y_sub_tr_t3,
            X_sub_val=X_sub_va_t3,
            y_sub_val=y_sub_va_t3,
            config=config,
            random_seed=seed + fold_id * 100,
        )
        fold_shap_rfe_sec = time.perf_counter() - t_rfe_start
        t3_selected_keys = rfe_result.best_features

        if rfe_result.best_val_shap_values is not None:
            val_shap_df = pd.concat(
                [
                    meta_sub_va_t3.reset_index(drop=True),
                    pd.Series(y_sub_va_t3, name=target_col),
                    rfe_result.best_val_shap_values.reset_index(drop=True),
                ],
                axis=1,
            )
            val_shap_frames[fold_id] = val_shap_df

        selected_features_log[f"fold_{fold_id}"] = {
            "train_cases": train_cases,
            "sub_train_cases": sub_train_cases,
            "sub_val_cases": sub_val_cases,
            "test_cases": test_cases,
            "shap_rfe_seconds": round(fold_shap_rfe_sec, 4),
            **(
                {
                    "train_cv_groups": sorted({str(cv_groups[i]) for i in train_idx}),
                    "sub_train_cv_groups": sorted({str(cv_groups[i]) for i in sub_train_idx}),
                    "sub_val_cv_groups": sorted({str(cv_groups[i]) for i in sub_val_idx}),
                    "test_cv_groups": sorted({str(cv_groups[i]) for i in test_idx}),
                }
                if cv_groups is not None
                else {}
            ),
            **rfe_result.to_log_dict(),
        }
        del X_sub_tr_arr, X_sub_va_arr, X_sub_tr_t3, X_sub_va_t3
        gc.collect()

        # ------------------------------------------------------------------
        # Step 3: Canonical holdout test metadata & targets
        # ------------------------------------------------------------------
        _, y_test_base, base_test_meta, _, _ = inner_scaler.transform_split(
            compact, test_idx, tier_keys["T1"][:1], case_col=case_col, event_col=event_col
        )
        fold_preds: dict[str, Any] = {
            case_col: base_test_meta[case_col].to_numpy(),
            event_col: base_test_meta[event_col].to_numpy(),
            "Fold_ID": np.full(len(base_test_meta), fold_id, dtype=np.int64),
            target_col: y_test_base,
        }

        active_tier_map = {
            "T1": tier_keys["T1"],
            "T2": tier_keys["T2"],
            "T3": t3_selected_keys,
        }
        tier_display = {"T1": "Tier 1", "T2": "Tier 2", "T3": "Tier 3"}

        # Outer-fold scaler (fit strictly on all 16 Train graphs) for XGBoost
        outer_scaler = CompactFoldScaler().fit(compact, train_idx)

        # ------------------------------------------------------------------
        # Step 4: Train & evaluate XGBoost, LSTM (lazy Dataset), and GNN
        # ------------------------------------------------------------------
        for tier_idx, (tier_code, active_keys) in enumerate(active_tier_map.items()):
            tier_seed = seed + fold_id * 100 + tier_idx * 10
            tier_label = tier_display[tier_code]
            shap_sec = fold_shap_rfe_sec if tier_code == "T3" else 0.0

            # 1. XGBoost (outer 16 Train graphs)
            t0 = time.perf_counter()
            X_tr_arr, y_tr_tab, _, _, _ = outer_scaler.transform_split(
                compact, train_idx, active_keys, case_col=case_col, event_col=event_col
            )
            X_te_arr, _, te_tab_meta, _, _ = outer_scaler.transform_split(
                compact, test_idx, active_keys, case_col=case_col, event_col=event_col
            )
            X_tr_tab = pd.DataFrame(X_tr_arr, columns=active_keys)
            X_te_tab = pd.DataFrame(X_te_arr, columns=active_keys)
            t_enc = time.perf_counter() - t0

            xgb_model = TabularXGBModel(config=config, random_seed=tier_seed)
            t0 = time.perf_counter()
            xgb_model.fit(X_tr_tab, y_tr_tab)
            t_train = time.perf_counter() - t0

            t0 = time.perf_counter()
            xgb_preds = xgb_model.predict(X_te_tab)
            t_pred = time.perf_counter() - t0

            fold_preds[f"Pred_XGB_{tier_code}"] = _align_predictions_to_base(
                base_test_meta, te_tab_meta, xgb_preds, case_col, event_col
            )
            runtime_rows.append(
                {
                    "Fold_ID": fold_id,
                    "Tier": tier_label,
                    "Model": "XGBoost",
                    "Num_Features": len(active_keys),
                    "Epochs_Run": float("nan"),
                    "Encode_Seconds": t_enc,
                    "Train_Seconds": t_train,
                    "Predict_Seconds": t_pred,
                    "Total_Seconds": t_enc + t_train + t_pred,
                    "SHAP_RFE_Seconds": shap_sec,
                }
            )
            del X_tr_arr, X_te_arr, X_tr_tab, X_te_tab, xgb_model

            # Inner-scaled compact split shared by LSTM and GNN for this tier
            t0 = time.perf_counter()
            X_st, y_st, _, st_slices, st_edges = inner_scaler.transform_split(
                compact, sub_train_idx, active_keys, case_col=case_col, event_col=event_col
            )
            X_sv, y_sv, _, sv_slices, sv_edges = inner_scaler.transform_split(
                compact, sub_val_idx, active_keys, case_col=case_col, event_col=event_col
            )
            X_te_in, y_te_in, te_in_meta, te_slices, te_edges = inner_scaler.transform_split(
                compact, test_idx, active_keys, case_col=case_col, event_col=event_col
            )
            t_inner_scale = time.perf_counter() - t0

            # 2. LSTM (memory-safe lazy causal prefix Dataset; zero 3D tensor bloat)
            t0 = time.perf_counter()
            ds_tr = CausalPrefixSequenceDataset(X_st, st_slices, window_size=window_size, y=y_st)
            ds_va = CausalPrefixSequenceDataset(X_sv, sv_slices, window_size=window_size, y=y_sv)
            ds_te = CausalPrefixSequenceDataset(X_te_in, te_slices, window_size=window_size, y=None)
            t_enc_lstm = t_inner_scale + (time.perf_counter() - t0)

            lstm_model = SequenceLSTMModel(config=config, random_seed=tier_seed)
            t0 = time.perf_counter()
            lstm_model.fit(ds_tr, X_val=ds_va)
            t_train = time.perf_counter() - t0
            for h_row in lstm_model.training_history:
                training_history_rows.append(
                    {"Fold_ID": fold_id, "Tier": tier_code, "Model": "LSTM", **h_row}
                )

            t0 = time.perf_counter()
            lstm_preds = lstm_model.predict(ds_te)
            t_pred = time.perf_counter() - t0

            fold_preds[f"Pred_LSTM_{tier_code}"] = _align_predictions_to_base(
                base_test_meta, te_in_meta, lstm_preds, case_col, event_col
            )
            runtime_rows.append(
                {
                    "Fold_ID": fold_id,
                    "Tier": tier_label,
                    "Model": "LSTM",
                    "Num_Features": len(active_keys),
                    "Epochs_Run": len(lstm_model.training_history),
                    "Encode_Seconds": t_enc_lstm,
                    "Train_Seconds": t_train,
                    "Predict_Seconds": t_pred,
                    "Total_Seconds": t_enc_lstm + t_train + t_pred,
                    "SHAP_RFE_Seconds": shap_sec,
                }
            )
            del ds_tr, ds_va, ds_te, lstm_model

            # 3. GNN (PyG causal Data graphs)
            t0 = time.perf_counter()
            tr_pyg = build_pyg_graphs_from_compact(X_st, y_st, st_slices, st_edges)
            va_pyg = build_pyg_graphs_from_compact(X_sv, y_sv, sv_slices, sv_edges)
            te_pyg = build_pyg_graphs_from_compact(X_te_in, y_te_in, te_slices, te_edges)
            t_enc_gnn = t_inner_scale + (time.perf_counter() - t0)

            gnn_model = GraphGNNModel(config=config, random_seed=tier_seed)
            t0 = time.perf_counter()
            gnn_model.fit(tr_pyg, X_val=va_pyg)
            t_train = time.perf_counter() - t0
            for h_row in gnn_model.training_history:
                training_history_rows.append(
                    {"Fold_ID": fold_id, "Tier": tier_code, "Model": "GNN", **h_row}
                )

            t0 = time.perf_counter()
            gnn_preds = gnn_model.predict(te_pyg)
            t_pred = time.perf_counter() - t0

            fold_preds[f"Pred_GNN_{tier_code}"] = _align_predictions_to_base(
                base_test_meta, te_in_meta, gnn_preds, case_col, event_col
            )
            runtime_rows.append(
                {
                    "Fold_ID": fold_id,
                    "Tier": tier_label,
                    "Model": "GNN",
                    "Num_Features": len(active_keys),
                    "Epochs_Run": len(gnn_model.training_history),
                    "Encode_Seconds": t_enc_gnn,
                    "Train_Seconds": t_train,
                    "Predict_Seconds": t_pred,
                    "Total_Seconds": t_enc_gnn + t_train + t_pred,
                    "SHAP_RFE_Seconds": shap_sec,
                }
            )
            del X_st, X_sv, X_te_in, tr_pyg, va_pyg, te_pyg, gnn_model
            gc.collect()

        oof_frames.append(pd.DataFrame(fold_preds))

    oof_df = pd.concat(oof_frames, ignore_index=True)
    ordered_cols = [case_col, event_col, "Fold_ID", target_col, *OOF_PREDICTION_COLUMNS]
    oof_df = oof_df[ordered_cols].sort_values([case_col, event_col]).reset_index(drop=True)

    if len(oof_df) != total_input_events:
        raise AssertionError(
            f"OOF prediction row count ({len(oof_df)}) != input event count ({total_input_events})."
        )

    if diagnostics_out_dir is not None:
        _export_stage2_diagnostics(
            diagnostics_dir=Path(diagnostics_out_dir),
            oof_df=oof_df,
            selected_features_log=selected_features_log,
            training_history_rows=training_history_rows,
            runtime_rows=runtime_rows,
            val_shap_frames=val_shap_frames,
            case_col=case_col,
            target_col=target_col,
            n_splits=n_splits,
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
    diag_dir = Path(config["paths"].get("diagnostics_dir", "data/results/diagnostics"))

    compact_path = fs_path.with_name("feature_storage_compact.pt")
    use_compact_cache = compact_path.exists() and (
        not fs_path.exists() or compact_path.stat().st_mtime >= fs_path.stat().st_mtime
    )
    if use_compact_cache:
        LOGGER.info("Loading compact Stage 1a cache from %s (low-RAM mode).", compact_path)
        stage1_input: Feature_Storage | CompactOcpaDataset = torch.load(
            compact_path, weights_only=False
        )
    else:
        if not fs_path.exists():
            raise FileNotFoundError(
                f"Feature_Storage not found at {fs_path}. Run Stage 1a first."
            )
        with fs_path.open("rb") as handle:
            stage1_input = pickle.load(handle)

    oof_df, selected_features_log = run_cross_validation_in_memory(
        feature_storage=stage1_input,
        config=config,
        diagnostics_out_dir=diag_dir,
        consume_storage=True,
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
