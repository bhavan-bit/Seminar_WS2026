"""Stage 2a: Fold-scoped ``ocpa`` ``Feature_Storage`` slicing, scaling, and encoding.

Provides strictly leak-free operations over ``ocpa`` ``Feature_Storage`` objects:
    1. Slicing ``Feature_Storage`` by ``Feature_Graph`` indices into fold-scoped
       ``train_fs`` / ``test_fs`` and ``sub_train_fs`` / ``sub_val_fs`` copies.
    2. Fitting per-feature median/mean/std normalization strictly on the training
       ``Feature_Storage`` (NEVER calling ``extract_normalized_train_test_split()``
       and NEVER normalizing globally across all cases).
    3. Filtering active feature keys to Tier 1, Tier 2, or fold-selected Tier 3 keys.
    4. Producing model-specific inputs directly via ``ocpa``:
       - Tabular (XGBoost): ``ocpa.algo.predictive_monitoring.tabular.construct_table``
       - Sequential (LSTM): ``ocpa.algo.predictive_monitoring.sequential.construct_sequence``
       - Graph (GNN): PyTorch Geometric ``Data`` objects from ``fs.feature_graphs``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Sequence

import numpy as np
import ocpa.algo.predictive_monitoring.sequential as pm_sequential
import ocpa.algo.predictive_monitoring.tabular as pm_tabular
import pandas as pd
import torch
from ocpa.algo.predictive_monitoring.obj import Feature_Storage
from torch_geometric.data import Data

from src import resolve_tier_columns

LOGGER = logging.getLogger(__name__)


def _new_empty_feature_storage(template: Feature_Storage) -> Feature_Storage:
    """Create a fresh ``Feature_Storage`` shell sharing feature metadata with ``template``."""
    fs = Feature_Storage.__new__(Feature_Storage)
    fs._event_features = list(template.event_features)
    fs._edge_features = list(getattr(template, "_edge_features", []))
    fs._case_features = list(template.execution_features)
    fs._feature_graphs = []
    fs._scaler = None
    fs._training_indices = None
    fs._test_indices = None
    return fs


def slice_feature_storage(
    feature_storage: Feature_Storage,
    graph_indices: Sequence[int],
    deep_copy: bool = True,
) -> Feature_Storage:
    """Slice a ``Feature_Storage`` by positional ``Feature_Graph`` indices.

    Args:
        feature_storage: Source ``ocpa`` ``Feature_Storage``.
        graph_indices: Positional indices into ``feature_storage.feature_graphs``.
        deep_copy: If ``True`` (default), deep-copy graphs so mutations inside a
            fold can never contaminate the original ``Feature_Storage``.

    Returns:
        New ``Feature_Storage`` containing only the requested ``Feature_Graph``s.
    """
    sliced = _new_empty_feature_storage(feature_storage)
    graphs = feature_storage.feature_graphs
    selected = [graphs[int(idx)] for idx in graph_indices]
    sliced._feature_graphs = copy.deepcopy(selected) if deep_copy else list(selected)
    return sliced


def get_case_ids(feature_storage: Feature_Storage) -> list[int]:
    """Return sorted list of ``Case_ID`` (``pexec_id``) values in ``feature_storage``."""
    return sorted(int(g.pexec_id) for g in feature_storage.feature_graphs)


def filter_feature_keys_for_tier(
    feature_storage: Feature_Storage,
    config: dict[str, Any],
    tier_name: str,
) -> list[str]:
    """Resolve active feature keys in ``feature_storage`` for a given feature tier.

    Args:
        feature_storage: ``Feature_Storage`` with canonical string ``event_features``.
        config: Loaded configuration dictionary.
        tier_name: ``"tier_1"``, ``"tier_2"``, or ``"tier_3"``.

    Returns:
        Ordered list of feature keys belonging to ``tier_name``.
    """
    return resolve_tier_columns(list(feature_storage.event_features), config, tier_name)


class OcpaFeatureScaler:
    """Fold-scoped numeric imputer + z-score scaler fit strictly on training ``Feature_Storage``.

    Never scales the target key (``y_true``), never mutates input ``Feature_Storage``
    objects, and handles missing values and zero-variance features deterministically.
    """

    def __init__(self, target_key: str = "y_true") -> None:
        """Initialize the fold-scoped scaler.

        Args:
            target_key: Target attribute key excluded from feature scaling.
        """
        self.target_key = target_key
        self._medians: dict[str, float] = {}
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        self._fitted_keys: list[str] = []

    @property
    def fitted_keys(self) -> list[str]:
        """Return the list of feature keys fitted by this scaler."""
        return list(self._fitted_keys)

    def fit(
        self,
        train_fs: Feature_Storage,
        feature_keys: Sequence[str] | None = None,
    ) -> OcpaFeatureScaler:
        """Fit per-feature median, mean, and std strictly on ``train_fs``.

        Args:
            train_fs: Training-split ``Feature_Storage``.
            feature_keys: Optional subset of feature keys to scale. Defaults to
                all keys in ``train_fs.event_features`` excluding ``target_key``.

        Returns:
            Fitted ``OcpaFeatureScaler``.
        """
        keys = (
            [k for k in feature_keys if k != self.target_key]
            if feature_keys is not None
            else [k for k in train_fs.event_features if k != self.target_key]
        )
        if not keys:
            raise ValueError("OcpaFeatureScaler.fit requires at least one feature key.")

        train_table = pm_tabular.construct_table(train_fs, index_list="all")
        if train_table.empty:
            raise ValueError("Training Feature_Storage produced an empty table during fit().")

        self._medians.clear()
        self._means.clear()
        self._stds.clear()
        self._fitted_keys = list(keys)

        for key in keys:
            raw_vals = pd.to_numeric(train_table[key], errors="coerce").to_numpy(dtype=np.float64)
            finite_mask = np.isfinite(raw_vals)
            median = float(np.median(raw_vals[finite_mask])) if np.any(finite_mask) else 0.0
            imputed = np.where(finite_mask, raw_vals, median)
            mean = float(np.mean(imputed))
            std = float(np.std(imputed, ddof=0))
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            self._medians[key] = median
            self._means[key] = mean
            self._stds[key] = std

        return self

    def transform(self, fs: Feature_Storage) -> Feature_Storage:
        """Return a normalized deep copy of ``fs`` using training-split parameters.

        Args:
            fs: ``Feature_Storage`` (Train, Sub-Val, or Holdout Test).

        Returns:
            Normalized ``Feature_Storage`` copy with ``y_true`` left unscaled.
        """
        if not self._fitted_keys:
            raise RuntimeError("OcpaFeatureScaler must be fitted before calling transform().")

        scaled_fs = _new_empty_feature_storage(fs)
        scaled_fs._feature_graphs = copy.deepcopy(fs.feature_graphs)

        for fg in scaled_fs.feature_graphs:
            for node in fg.nodes:
                attrs = node.attributes
                for key in self._fitted_keys:
                    raw = attrs.get(key, float("nan"))
                    try:
                        fval = float(raw)
                    except (TypeError, ValueError):
                        fval = float("nan")
                    if not np.isfinite(fval):
                        fval = self._medians[key]
                    attrs[key] = float((fval - self._means[key]) / self._stds[key])

        return scaled_fs

    def fit_transform(
        self,
        train_fs: Feature_Storage,
        feature_keys: Sequence[str] | None = None,
    ) -> Feature_Storage:
        """Fit strictly on ``train_fs`` and return its normalized copy."""
        return self.fit(train_fs, feature_keys=feature_keys).transform(train_fs)


def _selected_graphs(
    fs: Feature_Storage,
    index_list: str | Sequence[int] = "all",
) -> list[Feature_Storage.Feature_Graph]:
    """Filter ``fs.feature_graphs`` identically to ``ocpa``'s ``index_list`` convention."""
    if index_list == "all":
        return list(fs.feature_graphs)
    allowed = set(index_list)
    return [g for g in fs.feature_graphs if g.pexec_id in allowed]


def encode_tabular(
    fs: Feature_Storage,
    feature_keys: Sequence[str],
    target_key: str = "y_true",
    index_list: str | list[int] = "all",
    case_col: str = "Case_ID",
    event_col: str = "Event_ID",
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Construct tabular features via ``ocpa.algo.predictive_monitoring.tabular.construct_table``.

    Args:
        fs: Fold-scoped (normalized) ``Feature_Storage``.
        feature_keys: Active feature keys for the current tier.
        target_key: Target attribute key (default ``"y_true"``).
        index_list: ``"all"`` or list of ``pexec_id`` values passed directly to ``ocpa``.
        case_col: Case identifier column name in returned metadata.
        event_col: Event identifier column name in returned metadata.

    Returns:
        Tuple ``(X_df, y_arr, meta_df)`` with strict 1-to-1 row alignment.
    """
    keys = list(feature_keys)
    table = pm_tabular.construct_table(fs, index_list=index_list)

    meta_rows: list[tuple[int, Any]] = []
    for g in _selected_graphs(fs, index_list=index_list):
        cid = int(g.pexec_id)
        for node in g.nodes:
            meta_rows.append((cid, node.event_id))

    if len(table) != len(meta_rows):
        raise ValueError(
            f"Tabular encoding alignment mismatch: table has {len(table)} rows, "
            f"metadata has {len(meta_rows)} rows."
        )

    X_df = table[keys].astype(np.float32).reset_index(drop=True)
    y_arr = table[target_key].to_numpy(dtype=np.float64)
    meta_df = pd.DataFrame(meta_rows, columns=[case_col, event_col])
    return X_df, y_arr, meta_df


def encode_sequential(
    fs: Feature_Storage,
    feature_keys: Sequence[str],
    target_key: str = "y_true",
    window_size: int = 20,
    index_list: str | list[int] = "all",
    case_col: str = "Case_ID",
    event_col: str = "Event_ID",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Construct causal prefix sequences via ``ocpa.algo.predictive_monitoring.sequential.construct_sequence``.

    Each event in each case receives a left-padded causal sliding window of shape
    ``(window_size, len(feature_keys))`` built strictly from that case's prefix
    up to and including the current event (zero cross-case or future-event leakage).

    Args:
        fs: Fold-scoped (normalized) ``Feature_Storage``.
        feature_keys: Active feature keys for the current tier.
        target_key: Target attribute key (default ``"y_true"``).
        window_size: Sliding-window prefix length ``W``.
        index_list: ``"all"`` or list of ``pexec_id`` values passed directly to ``ocpa``.
        case_col: Case identifier column name in returned metadata.
        event_col: Event identifier column name in returned metadata.

    Returns:
        Tuple ``(X_seq, y_arr, meta_df)`` where ``X_seq`` has shape
        ``(N_events, window_size, len(feature_keys))``.
    """
    keys = list(feature_keys)
    sequences = pm_sequential.construct_sequence(fs, index_list=index_list)
    graphs = _selected_graphs(fs, index_list=index_list)

    if len(sequences) != len(graphs):
        raise ValueError(
            f"Sequential encoding mismatch: {len(sequences)} sequences for {len(graphs)} graphs."
        )

    total_events = sum(len(seq) for seq in sequences)
    n_features = len(keys)
    X_seq = np.zeros((total_events, window_size, n_features), dtype=np.float32)
    y_arr = np.zeros(total_events, dtype=np.float64)
    meta_rows: list[tuple[int, Any]] = []

    cursor = 0
    for g, seq in zip(graphs, sequences):
        cid = int(g.pexec_id)
        # Match ocpa sequential.construct_sequence exact sorted event_id order
        sorted_eids = sorted(n.event_id for n in g.nodes)
        if len(sorted_eids) != len(seq):
            raise ValueError(
                f"Sequence length mismatch on Case_ID={cid}: {len(seq)} vs {len(sorted_eids)}."
            )
        if not seq:
            continue

        seq_matrix = np.array(
            [[float(step_attrs[k]) for k in keys] for step_attrs in seq],
            dtype=np.float32,
        )
        seq_targets = np.array(
            [float(step_attrs[target_key]) for step_attrs in seq],
            dtype=np.float64,
        )

        for local_idx, eid in enumerate(sorted_eids):
            start = max(0, local_idx - window_size + 1)
            window = seq_matrix[start : local_idx + 1]
            win_len = window.shape[0]
            X_seq[cursor, window_size - win_len :, :] = window
            y_arr[cursor] = seq_targets[local_idx]
            meta_rows.append((cid, eid))
            cursor += 1

    meta_df = pd.DataFrame(meta_rows, columns=[case_col, event_col])
    return X_seq, y_arr, meta_df


def encode_pyg_graphs(
    fs: Feature_Storage,
    feature_keys: Sequence[str],
    target_key: str = "y_true",
    index_list: str | list[int] = "all",
    case_col: str = "Case_ID",
    event_col: str = "Event_ID",
) -> tuple[list[Data], pd.DataFrame]:
    """Convert ``fs.feature_graphs`` into PyTorch Geometric ``Data`` objects.

    Each ``Feature_Graph`` becomes one PyG ``Data(x, edge_index, y)`` with node
    features filtered strictly to ``feature_keys`` and strictly causal directed
    edges (``src_idx < dst_idx``).

    Args:
        fs: Fold-scoped (normalized) ``Feature_Storage``.
        feature_keys: Active feature keys for the current tier.
        target_key: Target attribute key (default ``"y_true"``).
        index_list: ``"all"`` or list of ``pexec_id`` values.
        case_col: Case identifier column name in returned metadata.
        event_col: Event identifier column name in returned metadata.

    Returns:
        Tuple ``(pyg_graphs, meta_df)`` where ``meta_df`` aligns 1-to-1 with
        the concatenated nodes across ``pyg_graphs``.
    """
    keys = list(feature_keys)
    pyg_graphs: list[Data] = []
    meta_rows: list[tuple[int, Any]] = []

    for g in _selected_graphs(fs, index_list=index_list):
        cid = int(g.pexec_id)
        nodes = list(g.nodes)
        if not nodes:
            continue

        pos_by_eid = {node.event_id: idx for idx, node in enumerate(nodes)}
        x_np = np.array(
            [[float(node.attributes[k]) for k in keys] for node in nodes],
            dtype=np.float32,
        )
        y_np = np.array(
            [float(node.attributes[target_key]) for node in nodes],
            dtype=np.float32,
        )

        src_list: list[int] = []
        dst_list: list[int] = []
        for edge in g.edges:
            if edge.source in pos_by_eid and edge.target in pos_by_eid:
                s_idx = pos_by_eid[edge.source]
                d_idx = pos_by_eid[edge.target]
                if s_idx < d_idx:
                    src_list.append(s_idx)
                    dst_list.append(d_idx)

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(
            x=torch.from_numpy(x_np),
            edge_index=edge_index,
            y=torch.from_numpy(y_np),
        )
        pyg_graphs.append(data)

        for node in nodes:
            meta_rows.append((cid, node.event_id))

    meta_df = pd.DataFrame(meta_rows, columns=[case_col, event_col])
    return pyg_graphs, meta_df

