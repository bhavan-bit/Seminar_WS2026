"""Anti-leakage verification tests for Stage 2 encoding, cross-validation, and Stage 3 evaluation.

Covers Section 6 Invariants 1, 2, 3, and 5:
    - Zero Case_ID / Feature_Graph overlap across outer (16/4) and inner (12/4) splits.
    - Zero executable references to entanglement_scores / entanglement_metrics in Stage 2.
    - Zero calls to ocpa's leaky ``extract_normalized_train_test_split()`` anywhere in ``src/``.
    - Strictly train-only normalization in ``OcpaFeatureScaler`` (raw Feature_Storage immutable;
      test outliers never influence train statistics; target y_true stays unscaled).
    - Exact (Case_ID, Event_ID) alignment across tabular, sequential, and PyG graph encodings.
    - End-to-end 5-fold CV + Stage 3 evaluation schema and reproducibility on a minimal
      inline 20-case ``ocpa`` ``Feature_Storage`` fixture.
"""

from __future__ import annotations

import ast
import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import load_config
from src.cross_validation import (
    OOF_PREDICTION_COLUMNS,
    generate_graph_cv_splits,
    run_cross_validation_in_memory,
)
from src.entanglement_metrics import compute_entanglement_scores
from src.evaluation import (
    build_matrix1_global_mae,
    build_matrix2_regime_mae,
    compute_case_normalized_mae,
    compute_event_level_mae,
    split_matrix2_by_tier,
)
from src.feature_extraction import build_ocel_from_dataframe, extract_feature_storage
from src.ocpa_encoding import (
    OcpaFeatureScaler,
    encode_pyg_graphs,
    encode_sequential,
    encode_tabular,
    filter_feature_keys_for_tier,
    slice_feature_storage,
)


def _make_minimal_20case_feature_storage(seed: int = 42) -> tuple[Any, dict[str, Any]]:
    """Build a minimal inline 20-case native ``ocpa`` ``Feature_Storage`` fixture."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2025, 1, 1, 8, 0, 0)
    rows: list[dict[str, Any]] = []
    ev_id = 0

    for case_idx in range(20):
        n_events = 4 + (case_idx % 3)  # 4..6 events per case
        case_start = t0 + timedelta(days=case_idx * 2)
        order_id = f"o_{case_idx}"
        item_a = f"i_{case_idx}_a"
        item_b = f"i_{case_idx}_b"

        for step in range(n_events):
            ts = case_start + timedelta(hours=step * 2)
            start_ts = ts - timedelta(minutes=30 if step > 0 else 0)
            if step == 0:
                act = "create_order"
                orders = [order_id]
                items = [item_a, item_b] if case_idx % 2 == 0 else [item_a]
            elif step == n_events - 1:
                act = "ship_order"
                orders = [order_id]
                items = [item_a, item_b] if case_idx % 2 == 0 else [item_a]
            elif step % 2 == 1:
                act = "pick_item"
                orders = [] if case_idx % 3 == 0 else [order_id]
                items = [item_a]
            else:
                act = "pack_item"
                orders = [] if case_idx % 2 == 0 else [order_id]
                items = [item_b] if case_idx % 2 == 0 else [item_a]

            amount = float(rng.normal(100.0 + case_idx * 5.0, 10.0))
            if case_idx == 0 and step == 1:
                amount = float("nan")  # deliberate missing numeric attribute

            rows.append(
                {
                    "event_id": ev_id,
                    "event_activity": act,
                    "event_timestamp": ts,
                    "event_start_timestamp": start_ts,
                    "order": orders,
                    "item": items,
                    "event_amount": amount,
                }
            )
            ev_id += 1

    df = pd.DataFrame(rows)
    parameters = {
        "obj_names": ["order", "item"],
        "val_names": ["event_amount"],
        "act_name": "event_activity",
        "time_name": "event_timestamp",
        "sep": ",",
    }
    cfg = copy.deepcopy(load_config("config.yaml"))
    cfg["ocpa"]["object_types"] = ["order", "item"]
    cfg["ocpa"]["execution_extraction"] = "connected_components"
    cfg["ocpa"]["leading_type"] = None
    cfg["ocpa"]["cv_group_object_type"] = None
    cfg["ocpa"]["target_object_type"] = "item"
    cfg["debug_mode"] = True
    cfg["debug"]["max_cases"] = 20
    cfg["debug"]["max_events_per_case"] = 20
    cfg["debug"]["xgb_n_estimators"] = 5
    cfg["debug"]["lstm_epochs"] = 2
    cfg["debug"]["gnn_epochs"] = 2

    ocel = build_ocel_from_dataframe(df, parameters)
    fs, _ = extract_feature_storage(ocel, cfg)
    return fs, cfg


def test_zero_case_id_overlap_across_all_cv_splits() -> None:
    """Invariant 1: Zero graph/Case_ID overlap in outer (16/4) and inner (12/4) splits."""
    folds = generate_graph_cv_splits(
        num_graphs=20,
        n_splits=5,
        inner_test_size=0.25,
        random_seed=42,
    )
    assert len(folds) == 5
    seen_test: list[int] = []

    for fold in folds:
        tr = set(fold["train_idx"].tolist())
        te = set(fold["test_idx"].tolist())
        sub_tr = set(fold["sub_train_idx"].tolist())
        sub_va = set(fold["sub_val_idx"].tolist())

        assert len(tr) == 16
        assert len(te) == 4
        assert len(sub_tr) == 12
        assert len(sub_va) == 4

        assert tr.isdisjoint(te)
        assert sub_tr.isdisjoint(sub_va)
        assert sub_tr.union(sub_va) == tr
        assert tr.union(te) == set(range(20))
        seen_test.extend(sorted(te))

    assert sorted(seen_test) == list(range(20))

    # Also verify parent-group (Match-level) CV grouping: 10 Matches x 2 Sessions -> 16/4 outer, 12/4 inner
    match_groups = [f"M{i // 2}" for i in range(20)]
    grouped_folds = generate_graph_cv_splits(
        num_graphs=20,
        n_splits=5,
        inner_test_size=0.25,
        random_seed=42,
        cv_groups=match_groups,
    )
    assert len(grouped_folds) == 5
    for fold in grouped_folds:
        tr_matches = {match_groups[i] for i in fold["train_idx"]}
        te_matches = {match_groups[i] for i in fold["test_idx"]}
        sub_tr_matches = {match_groups[i] for i in fold["sub_train_idx"]}
        sub_va_matches = {match_groups[i] for i in fold["sub_val_idx"]}
        assert len(fold["train_idx"]) == 16 and len(fold["test_idx"]) == 4
        assert len(fold["sub_train_idx"]) == 12 and len(fold["sub_val_idx"]) == 4
        assert tr_matches.isdisjoint(te_matches)
        assert sub_tr_matches.isdisjoint(sub_va_matches)


def test_zero_cross_stage_and_banned_ocpa_split_references() -> None:
    """Invariant 2 & 3: No Stage 2 entanglement references and no extract_normalized_train_test_split."""
    forbidden_entanglement = ("entanglement_scores", "entanglement_metrics", "Avg_Entanglement_Score")
    stage2_files = [
        Path("src/ocpa_encoding.py"),
        Path("src/feature_selection.py"),
        Path("src/cross_validation.py"),
    ]
    all_src_files = sorted(Path("src").rglob("*.py"))

    def _collect_docstring_ids(tree: ast.AST) -> set[int]:
        doc_ids: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    doc_ids.add(id(node.body[0].value))
        return doc_ids

    for path in stage2_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        doc_ids = _collect_docstring_ids(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for token in forbidden_entanglement:
                        assert token not in alias.name, f"Forbidden import {alias.name} in {path}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for token in forbidden_entanglement:
                    assert token not in mod, f"Forbidden import-from {mod} in {path}"
                for alias in node.names:
                    for token in forbidden_entanglement:
                        assert token not in alias.name, f"Forbidden import {alias.name} in {path}"
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) not in doc_ids:
                    for token in forbidden_entanglement:
                        assert token not in node.value, f"Forbidden string '{node.value}' in {path}"
            elif isinstance(node, ast.Name):
                for token in forbidden_entanglement:
                    assert token not in node.id, f"Forbidden identifier '{node.id}' in {path}"
            elif isinstance(node, ast.Attribute):
                for token in forbidden_entanglement:
                    assert token not in node.attr, f"Forbidden attribute '{node.attr}' in {path}"

    # Ensure extract_normalized_train_test_split is NEVER called anywhere in src/
    for path in all_src_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        doc_ids = _collect_docstring_ids(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "extract_normalized_train_test_split", (
                    f"Banned ocpa helper called in {path}"
                )
            elif isinstance(node, ast.Name):
                assert node.id != "extract_normalized_train_test_split", (
                    f"Banned ocpa helper referenced in {path}"
                )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_ids:
                assert "extract_normalized_train_test_split" not in node.value


def test_ocpa_scaler_strict_train_only_fitting_and_encoding_alignment() -> None:
    """Invariant 3: Scaling fits strictly on Train Feature_Storage and all 3 encoders align 1-to-1."""
    fs, cfg = _make_minimal_20case_feature_storage(seed=42)

    train_fs = slice_feature_storage(fs, list(range(16)))
    test_fs = slice_feature_storage(fs, list(range(16, 20)))

    # Inject an extreme outlier into test_fs and record raw target values
    test_node = test_fs.feature_graphs[0].nodes[0]
    raw_test_y = test_node.attributes["y_true"]
    test_node.attributes["D3_current_event_attr"] = 1_000_000.0

    scaler = OcpaFeatureScaler(target_key="y_true")
    train_scaled = scaler.fit_transform(train_fs)
    test_scaled = scaler.transform(test_fs)

    # Raw Feature_Storage must never be mutated
    assert test_fs.feature_graphs[0].nodes[0].attributes["D3_current_event_attr"] == 1_000_000.0

    # Target y_true must remain unscaled
    assert np.isclose(test_scaled.feature_graphs[0].nodes[0].attributes["y_true"], raw_test_y)

    # Train scaled column must have ~0 mean, unaffected by the 1,000,000 test outlier
    t1_keys = filter_feature_keys_for_tier(train_scaled, cfg, "tier_1")
    X_tr, y_tr, meta_tr = encode_tabular(train_scaled, t1_keys)
    X_te, y_te, meta_te = encode_tabular(test_scaled, t1_keys)

    assert np.isclose(float(X_tr["D3_current_event_attr"].mean()), 0.0, atol=1e-5)
    assert not X_tr.isna().any().any()
    assert not X_te.isna().any().any()
    assert float(X_te["D3_current_event_attr"].max()) > 1000.0

    # Tabular, Sequential, and PyG graph encodings must produce identical event sets & targets
    X_seq, y_seq, meta_seq = encode_sequential(test_scaled, t1_keys, window_size=5)
    pyg_graphs, meta_pyg = encode_pyg_graphs(test_scaled, t1_keys)

    assert X_seq.shape == (len(X_te), 5, len(t1_keys))
    assert len(pyg_graphs) == 4
    assert sum(g.num_nodes for g in pyg_graphs) == len(X_te)

    tab_pairs = list(zip(meta_te["Case_ID"], meta_te["Event_ID"]))
    seq_pairs = list(zip(meta_seq["Case_ID"], meta_seq["Event_ID"]))
    pyg_pairs = list(zip(meta_pyg["Case_ID"], meta_pyg["Event_ID"]))
    assert tab_pairs == seq_pairs == pyg_pairs
    assert np.allclose(y_te, y_seq)


def test_full_5fold_cv_and_stage3_evaluation_reproducibility() -> None:
    """Verify full Stage 2 (9 models x 5 folds) + Stage 3 (Matrix 1 & 2) end-to-end."""
    # Verify Case-Normalized MAE weights short and long cases equally
    unbalanced_df = pd.DataFrame(
        {
            "Case_ID": [0, 1, 1, 1, 1],
            "y_true": [0.0, 0.0, 0.0, 0.0, 0.0],
            "pred": [10.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    assert np.isclose(compute_event_level_mae(unbalanced_df["y_true"], unbalanced_df["pred"]), 3.6)
    assert np.isclose(compute_case_normalized_mae(unbalanced_df, "Case_ID", "y_true", "pred"), 6.0)

    fs, cfg = _make_minimal_20case_feature_storage(seed=42)
    oof_1, sel_1 = run_cross_validation_in_memory(fs, cfg)
    oof_2, sel_2 = run_cross_validation_in_memory(fs, cfg)

    expected_cols = ["Case_ID", "Event_ID", "Fold_ID", "y_true", *OOF_PREDICTION_COLUMNS]
    assert oof_1.columns.tolist() == expected_cols
    assert oof_1["Case_ID"].nunique() == 20
    assert set(oof_1["Fold_ID"].unique()) == {0, 1, 2, 3, 4}
    assert not oof_1.isna().any().any()

    # Reproducibility (Invariant 5)
    assert np.allclose(
        oof_1[OOF_PREDICTION_COLUMNS].to_numpy(),
        oof_2[OOF_PREDICTION_COLUMNS].to_numpy(),
        atol=1e-5,
    )
    for fold_key in sel_1:
        assert sel_1[fold_key]["selected_features"] == sel_2[fold_key]["selected_features"]
        assert set(sel_1[fold_key]["train_cases"]).isdisjoint(set(sel_1[fold_key]["test_cases"]))
        assert set(sel_1[fold_key]["sub_train_cases"]).isdisjoint(
            set(sel_1[fold_key]["sub_val_cases"])
        )

    # Stage 3 Evaluation on OOF + isolated Stage 1b entanglement scores
    entanglement_df = compute_entanglement_scores(fs, normalization="minmax")
    m1 = build_matrix1_global_mae(oof_1)
    assert m1.columns.tolist() == ["Metric", "Tier", "XGBoost", "LSTM", "GNN"]
    assert len(m1) == 6
    assert not m1[["XGBoost", "LSTM", "GNN"]].isna().any().any()

    m2 = build_matrix2_regime_mae(oof_1, entanglement_df)
    assert m2.columns.tolist() == [
        "Metric",
        "Regime",
        "Tier",
        "Num_Cases",
        "Num_Events",
        "XGBoost",
        "LSTM",
        "GNN",
    ]
    assert len(m2) == 18
    ev_t1 = m2[(m2["Metric"] == "Event_Level_MAE") & (m2["Tier"] == "Tier 1")]
    assert int(ev_t1["Num_Cases"].sum()) == 20
    assert not m2[["XGBoost", "LSTM", "GNN"]].isna().any().any()

    per_tier = split_matrix2_by_tier(m2)
    assert set(per_tier.keys()) == {"Tier 1", "Tier 2", "Tier 3"}
    for tier_df in per_tier.values():
        assert tier_df.columns.tolist() == [
            "Metric",
            "Regime",
            "Num_Cases",
            "Num_Events",
            "XGBoost",
            "LSTM",
            "GNN",
        ]
        assert len(tier_df) == 6
