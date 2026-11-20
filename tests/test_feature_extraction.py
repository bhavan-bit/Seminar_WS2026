"""Unit tests for Stage 1a (feature_extraction.py) and Stage 1b (entanglement_metrics.py).

Uses a minimal inline DataFrame converted to a native ``ocpa`` ``OCEL`` object
to verify ``Feature_Storage`` extraction, ``case_index`` alignment, native
``ocpa`` tabular/sequential compatibility, unnormalized raw storage, NaN
preservation (no global imputation), ``debug_mode`` capping, pickling round-trip,
and entanglement score normalization.
"""

from __future__ import annotations

import copy
import pickle
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import ocpa.algo.predictive_monitoring.sequential as pm_seq
import ocpa.algo.predictive_monitoring.tabular as pm_tab
import pandas as pd
import pytest

from src import load_config, resolve_tier_columns
from src.entanglement_metrics import compute_entanglement_scores
from src.feature_extraction import (
    build_ocel_from_dataframe,
    extract_feature_storage,
    validate_feature_storage,
)


def _make_minimal_ocel_fixture() -> Any:
    """Create a tiny inline 3-case OCEL fixture with two object types and a NaN attribute."""
    t0 = datetime(2025, 1, 1, 8, 0, 0)
    rows = [
        # Case 0: convergence case (order o1, items i1 & i2)
        {
            "event_id": 0,
            "event_activity": "create_order",
            "event_timestamp": t0 + timedelta(hours=0),
            "event_start_timestamp": t0 + timedelta(hours=0),
            "order": ["o1"],
            "item": ["i1", "i2"],
            "event_amount": 100.0,
        },
        {
            "event_id": 1,
            "event_activity": "pick_item",
            "event_timestamp": t0 + timedelta(hours=2),
            "event_start_timestamp": t0 + timedelta(hours=1),
            "order": [],
            "item": ["i1"],
            "event_amount": np.nan,  # deliberate missing value
        },
        {
            "event_id": 2,
            "event_activity": "pick_item",
            "event_timestamp": t0 + timedelta(hours=3),
            "event_start_timestamp": t0 + timedelta(hours=2),
            "order": [],
            "item": ["i2"],
            "event_amount": 50.0,
        },
        {
            "event_id": 3,
            "event_activity": "ship_order",
            "event_timestamp": t0 + timedelta(hours=6),
            "event_start_timestamp": t0 + timedelta(hours=5),
            "order": ["o1"],
            "item": ["i1", "i2"],
            "event_amount": 150.0,
        },
        # Case 1: linear case (order o2, item i3)
        {
            "event_id": 4,
            "event_activity": "create_order",
            "event_timestamp": t0 + timedelta(hours=10),
            "event_start_timestamp": t0 + timedelta(hours=10),
            "order": ["o2"],
            "item": ["i3"],
            "event_amount": 80.0,
        },
        {
            "event_id": 5,
            "event_activity": "pick_item",
            "event_timestamp": t0 + timedelta(hours=12),
            "event_start_timestamp": t0 + timedelta(hours=11),
            "order": ["o2"],
            "item": ["i3"],
            "event_amount": 80.0,
        },
        {
            "event_id": 6,
            "event_activity": "ship_order",
            "event_timestamp": t0 + timedelta(hours=15),
            "event_start_timestamp": t0 + timedelta(hours=14),
            "order": ["o2"],
            "item": [],
            "event_amount": 80.0,
        },
        # Case 2: simple 2-event case (order o3)
        {
            "event_id": 7,
            "event_activity": "create_order",
            "event_timestamp": t0 + timedelta(hours=20),
            "event_start_timestamp": t0 + timedelta(hours=20),
            "order": ["o3"],
            "item": [],
            "event_amount": 40.0,
        },
        {
            "event_id": 8,
            "event_activity": "ship_order",
            "event_timestamp": t0 + timedelta(hours=24),
            "event_start_timestamp": t0 + timedelta(hours=23),
            "order": ["o3"],
            "item": [],
            "event_amount": 40.0,
        },
    ]
    df = pd.DataFrame(rows)
    parameters = {
        "obj_names": ["order", "item"],
        "val_names": ["event_amount"],
        "act_name": "event_activity",
        "time_name": "event_timestamp",
        "sep": ",",
    }
    return build_ocel_from_dataframe(df, parameters)


@pytest.fixture()
def base_config() -> dict[str, Any]:
    """Load repository config.yaml."""
    return load_config("config.yaml")


def test_feature_storage_extraction_schema_and_ocpa_compatibility(
    base_config: dict[str, Any],
) -> None:
    """Verify Feature_Storage schema, case_index alignment, pickleability, and native ocpa encoding."""
    cfg = copy.deepcopy(base_config)
    cfg["debug_mode"] = False
    cfg["target"]["unit"] = "hours"
    ocel = _make_minimal_ocel_fixture()

    fs, case_index = extract_feature_storage(ocel, cfg)

    # Anti-leakage Stage 1 checks: unnormalized and unsplit
    assert fs.scaler is None
    assert fs.training_indices is None
    assert fs.test_indices is None
    validate_feature_storage(fs, case_index, cfg)

    # Exact case and event counts
    assert case_index.columns.tolist() == ["Graph_Index", "Case_ID", "Num_Events", "Num_Edges"]
    assert len(case_index) == 3
    assert int(case_index["Num_Events"].sum()) == 9

    # Native ocpa tabular and sequential encoders work directly on Feature_Storage
    tab_df = pm_tab.construct_table(fs, index_list="all")
    seqs = pm_seq.construct_sequence(fs, index_list="all")
    assert len(tab_df) == 9
    assert len(seqs) == 3
    assert sum(len(s) for s in seqs) == 9

    # Tier resolution matches expected features
    t1 = resolve_tier_columns(fs.event_features, cfg, "tier_1")
    t2 = resolve_tier_columns(fs.event_features, cfg, "tier_2")
    t3 = resolve_tier_columns(fs.event_features, cfg, "tier_3")

    assert {"D3_current_event_attr", "P9_service_time", "P2_elapsed_time"}.issubset(set(t1))
    assert any(c.startswith("C5_activity_") for c in t1)
    assert {"D1_avg_prev_event_attr", "P2_elapsed_time", "O3_prev_target_obj_count"}.issubset(set(t2))
    assert any(c.startswith("C2_preceding_act_") for c in t2)
    assert set(t1).issubset(set(t3))
    assert set(t2).issubset(set(t3))
    assert {
        "P5_synchronization_time",
        "P7_pooling_time",
        "P10_waiting_time",
        "O2_prev_object_count",
        "O5_event_object_count",
    }.issubset(set(t3))

    # Remaining time on Case 0 (0h -> 6h) matches exact ground truth in hours
    fg0 = next(g for g in fs.feature_graphs if any(n.event_id == 0 for n in g.nodes))
    y_vals = [n.attributes["y_true"] for n in fg0.nodes]
    assert np.allclose(y_vals, [6.0, 4.0, 3.0, 0.0])

    # Pickle round-trip preserves full Feature_Storage cleanly
    reloaded = pickle.loads(pickle.dumps(fs))
    assert len(reloaded.feature_graphs) == 3
    assert reloaded.event_features == fs.event_features


def test_missing_values_preserved_without_global_imputation(
    base_config: dict[str, Any],
) -> None:
    """Ensure Stage 1 preserves NaN raw numeric attributes for fold-scoped Stage 2 imputation."""
    cfg = copy.deepcopy(base_config)
    cfg["debug_mode"] = False
    ocel = _make_minimal_ocel_fixture()

    fs, _ = extract_feature_storage(ocel, cfg)

    ev1_node = next(n for g in fs.feature_graphs for n in g.nodes if n.event_id == 1)
    assert np.isnan(ev1_node.attributes["D3_current_event_attr"])
    assert np.isfinite(ev1_node.attributes["y_true"])
    assert ev1_node.attributes["y_true"] >= 0.0


def test_debug_mode_caps_cases_and_events(base_config: dict[str, Any]) -> None:
    """Verify debug_mode limits both cases and events per case cleanly."""
    cfg = copy.deepcopy(base_config)
    cfg["debug_mode"] = True
    cfg["debug"]["max_cases"] = 2
    cfg["debug"]["max_events_per_case"] = 2
    ocel = _make_minimal_ocel_fixture()

    fs, case_index = extract_feature_storage(ocel, cfg)

    assert len(fs.feature_graphs) == 2
    assert len(case_index) == 2
    assert (case_index["Num_Events"] <= 2).all()
    validate_feature_storage(fs, case_index, cfg)


def test_entanglement_metrics_on_feature_storage(base_config: dict[str, Any]) -> None:
    """Verify Stage 1b consumes Feature_Storage and exports normalized scores."""
    cfg = copy.deepcopy(base_config)
    cfg["debug_mode"] = False
    ocel = _make_minimal_ocel_fixture()

    fs, _ = extract_feature_storage(ocel, cfg)
    scores_df = compute_entanglement_scores(fs, normalization="minmax")

    expected_cols = [
        "Case_ID",
        "Metric_1",
        "Metric_2",
        "Metric_3",
        "Metric_4",
        "Avg_Entanglement_Score",
    ]
    assert scores_df.columns.tolist() == expected_cols
    assert len(scores_df) == 3
    assert not scores_df.isna().any().any()

    metric_cols = ["Metric_1", "Metric_2", "Metric_3", "Metric_4"]
    for col in metric_cols:
        assert (scores_df[col] >= 0.0).all() and (scores_df[col] <= 1.0).all()

    assert np.allclose(
        scores_df["Avg_Entanglement_Score"].to_numpy(),
        scores_df[metric_cols].mean(axis=1).to_numpy(),
    )
