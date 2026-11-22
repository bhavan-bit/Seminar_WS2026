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
from dataclasses import dataclass
import gc
import logging
from typing import Any, Sequence

import numpy as np
import ocpa.algo.predictive_monitoring.sequential as pm_sequential
import ocpa.algo.predictive_monitoring.tabular as pm_tabular
import pandas as pd
import torch
from ocpa.algo.predictive_monitoring.obj import Feature_Storage
from torch.utils.data import Dataset
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
        """Return a normalized copy of ``fs`` using training-split parameters.

        Allocates fresh node attribute dictionaries while sharing immutable graph
        topology references (edges, node object lists) to minimize memory
        overhead without ever mutating ``fs``.

        Args:
            fs: ``Feature_Storage`` (Train, Sub-Val, or Holdout Test).

        Returns:
            Normalized ``Feature_Storage`` copy with ``y_true`` left unscaled.
        """
        if not self._fitted_keys:
            raise RuntimeError("OcpaFeatureScaler must be fitted before calling transform().")

        scaled_fs = _new_empty_feature_storage(fs)
        new_graphs: list[Feature_Storage.Feature_Graph] = []

        medians = self._medians
        means = self._means
        stds = self._stds
        fitted_keys = self._fitted_keys
        target_key = self.target_key

        for fg in fs.feature_graphs:
            new_fg = Feature_Storage.Feature_Graph.__new__(Feature_Storage.Feature_Graph)
            new_fg._pexec_id = fg.pexec_id
            new_fg._edges = fg.edges
            new_fg._edge_mapping = getattr(fg, "_edge_mapping", {})
            new_fg._objects = fg.objects
            new_fg._attributes = dict(fg.attributes)

            new_nodes: list[Feature_Storage.Feature_Graph.Node] = []
            for node in fg.nodes:
                src_attrs = node.attributes
                new_attrs: dict[str, float] = {}
                if target_key in src_attrs:
                    new_attrs[target_key] = src_attrs[target_key]
                for key in fitted_keys:
                    raw = src_attrs.get(key, float("nan"))
                    try:
                        fval = float(raw)
                    except (TypeError, ValueError):
                        fval = float("nan")
                    if not np.isfinite(fval):
                        fval = medians[key]
                    new_attrs[key] = (fval - means[key]) / stds[key]

                new_node = Feature_Storage.Feature_Graph.Node.__new__(
                    Feature_Storage.Feature_Graph.Node
                )
                new_node._event = node.event_id
                new_node._objects = node.objects
                new_node._pexec_id = node.pexec_id
                new_node._attributes = new_attrs
                new_nodes.append(new_node)

            new_fg._nodes = new_nodes
            new_fg._node_mapping = {n.event_id: n for n in new_nodes}
            new_graphs.append(new_fg)

        scaled_fs._feature_graphs = new_graphs
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


# ==============================================================================
# Memory-Efficient Compact Representation for Large Full-Dataset Runs
# Avoids duplicating millions of boxed Python float dicts across CV folds while
# guaranteeing bit-identical fold-scoped normalization and causal encoding.
# ==============================================================================


class CausalPrefixSequenceDataset(Dataset):
    """Lazy PyTorch Dataset yielding left-padded causal prefix windows ``(W, F)``.

    Avoids materializing a dense ``(N_events, window_size, F)`` 3D tensor in RAM
    by slicing each event's intra-case prefix directly from a compact 2D
    ``float32`` feature array on the fly.
    """

    def __init__(
        self,
        X_2d: np.ndarray,
        case_slices: list[tuple[int, int]],
        window_size: int,
        y: np.ndarray | None = None,
    ) -> None:
        """Initialize the lazy prefix sequence dataset.

        Args:
            X_2d: Compact 2D ``float32`` feature array ``(N_events, F)`` where
                each case occupies a contiguous chronological slice.
            case_slices: List of ``(start, end)`` row slices, one per case.
            window_size: Causal sliding-window length ``W``.
            y: Optional 1D target array of length ``N_events``.
        """
        self.X_2d = np.ascontiguousarray(X_2d, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.float32) if y is not None else None
        self.window_size = int(window_size)
        self.n_features = int(self.X_2d.shape[1])

        # Precompute (row_index, case_start) for O(1) __getitem__
        row_indices: list[int] = []
        case_starts: list[int] = []
        for c_start, c_end in case_slices:
            for r in range(c_start, c_end):
                row_indices.append(r)
                case_starts.append(c_start)
        self._row_indices = np.array(row_indices, dtype=np.int64)
        self._case_starts = np.array(case_starts, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self._row_indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        row_idx = int(self._row_indices[idx])
        c_start = int(self._case_starts[idx])
        win_start = max(c_start, row_idx - self.window_size + 1)
        seq = self.X_2d[win_start : row_idx + 1]
        seq_len = seq.shape[0]
        window = np.zeros((self.window_size, self.n_features), dtype=np.float32)
        window[self.window_size - seq_len :, :] = seq
        x_t = torch.from_numpy(window)
        if self.y is not None:
            return x_t, torch.tensor(self.y[row_idx], dtype=torch.float32)
        return (x_t,)


@dataclass
class CompactOcpaDataset:
    """Compact float32 representation unpacked once from raw ``Feature_Storage``."""

    X_raw: np.ndarray
    y: np.ndarray
    feature_keys: list[str]
    key_to_col: dict[str, int]
    case_ids_per_graph: list[int]
    cv_groups_per_graph: list[Any]
    graph_slices: list[tuple[int, int]]
    graph_edge_indices: list[torch.Tensor]
    meta_case_ids: np.ndarray
    meta_event_ids: list[Any]


def unpack_feature_storage_compact(
    feature_storage: Feature_Storage,
    target_key: str = "y_true",
    consume_storage: bool = False,
) -> CompactOcpaDataset:
    """Unpack an unnormalized ``Feature_Storage`` into a single compact ``float32`` matrix.

    Preserves exact graph ordering, chronological node ordering, causal edges,
    and raw unnormalized values (including NaNs) while shrinking memory usage
    by >15x compared to Python ``dict`` objects.

    Args:
        feature_storage: Raw Stage 1a ``ocpa`` ``Feature_Storage``.
        target_key: Target attribute key (default ``"y_true"``).
        consume_storage: If ``True``, releases ``Feature_Graph`` node dicts as
            they are unpacked to minimize peak RAM on memory-constrained hosts.

    Returns:
        ``CompactOcpaDataset`` ready for leak-free fold-scoped scaling and encoding.
    """
    feature_keys = [k for k in feature_storage.event_features if k != target_key]
    key_to_col = {k: idx for idx, k in enumerate(feature_keys)}
    n_features = len(feature_keys)
    graphs = feature_storage.feature_graphs
    total_events = sum(len(g.nodes) for g in graphs)

    X_raw = np.empty((total_events, n_features), dtype=np.float32)
    y = np.empty(total_events, dtype=np.float64)
    meta_case_ids = np.empty(total_events, dtype=np.int64)
    meta_event_ids: list[Any] = [None] * total_events

    case_ids_per_graph: list[int] = []
    cv_groups_per_graph: list[Any] = []
    graph_slices: list[tuple[int, int]] = []
    graph_edge_indices: list[torch.Tensor] = []

    cursor = 0
    for fg in graphs:
        cid = int(fg.pexec_id)
        case_ids_per_graph.append(cid)
        cv_groups_per_graph.append(fg.attributes.get("cv_group_id"))

        nodes = list(fg.nodes)
        start = cursor
        pos_by_eid: dict[Any, int] = {}

        for local_i, node in enumerate(nodes):
            pos_by_eid[node.event_id] = local_i
            attrs = node.attributes
            y[cursor] = float(attrs[target_key])
            meta_case_ids[cursor] = cid
            meta_event_ids[cursor] = node.event_id
            for col_i, key in enumerate(feature_keys):
                raw = attrs.get(key, float("nan"))
                try:
                    X_raw[cursor, col_i] = float(raw)
                except (TypeError, ValueError):
                    X_raw[cursor, col_i] = np.nan
            cursor += 1

        end = cursor
        graph_slices.append((start, end))

        src_list: list[int] = []
        dst_list: list[int] = []
        for edge in fg.edges:
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
        graph_edge_indices.append(edge_index)

        if consume_storage:
            fg._nodes = []
            fg._node_mapping = {}
            fg._edges = []
            fg._edge_mapping = {}
            fg._objects = {}

    if consume_storage:
        feature_storage._feature_graphs = []
        gc.collect()

    return CompactOcpaDataset(
        X_raw=X_raw,
        y=y,
        feature_keys=feature_keys,
        key_to_col=key_to_col,
        case_ids_per_graph=case_ids_per_graph,
        cv_groups_per_graph=cv_groups_per_graph,
        graph_slices=graph_slices,
        graph_edge_indices=graph_edge_indices,
        meta_case_ids=meta_case_ids,
        meta_event_ids=meta_event_ids,
    )


class CompactFoldScaler:
    """Vectorized fold-scoped median imputer + z-score scaler over ``CompactOcpaDataset``.

    Mathematically identical to ``OcpaFeatureScaler`` while operating directly
    on ``float32`` arrays with minimal RAM overhead.
    """

    def __init__(self) -> None:
        self._medians: np.ndarray | None = None
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None

    def fit(self, dataset: CompactOcpaDataset, train_graph_indices: Sequence[int]) -> CompactFoldScaler:
        """Fit per-column median, mean, and std strictly on ``train_graph_indices``."""
        rows = np.concatenate(
            [
                np.arange(dataset.graph_slices[int(g)][0], dataset.graph_slices[int(g)][1], dtype=np.int64)
                for g in train_graph_indices
            ]
        )
        X_tr = dataset.X_raw[rows].astype(np.float64, copy=False)
        n_cols = X_tr.shape[1]
        medians = np.zeros(n_cols, dtype=np.float64)
        means = np.zeros(n_cols, dtype=np.float64)
        stds = np.ones(n_cols, dtype=np.float64)

        for c in range(n_cols):
            col = X_tr[:, c]
            finite = np.isfinite(col)
            med = float(np.median(col[finite])) if np.any(finite) else 0.0
            imputed = np.where(finite, col, med)
            mean = float(np.mean(imputed))
            std = float(np.std(imputed, ddof=0))
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            medians[c] = med
            means[c] = mean
            stds[c] = std

        self._medians = medians
        self._means = means
        self._stds = stds
        return self

    def transform_split(
        self,
        dataset: CompactOcpaDataset,
        graph_indices: Sequence[int],
        feature_keys: Sequence[str],
        case_col: str = "Case_ID",
        event_col: str = "Event_ID",
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[tuple[int, int]], list[torch.Tensor]]:
        """Extract and scale only the requested ``graph_indices`` and ``feature_keys``."""
        if self._medians is None or self._means is None or self._stds is None:
            raise RuntimeError("CompactFoldScaler must be fitted before transform_split().")

        col_idx = np.array([dataset.key_to_col[k] for k in feature_keys], dtype=np.int64)
        medians = self._medians[col_idx]
        means = self._means[col_idx]
        stds = self._stds[col_idx]

        total_rows = sum(
            dataset.graph_slices[int(g)][1] - dataset.graph_slices[int(g)][0]
            for g in graph_indices
        )
        X_out = np.empty((total_rows, len(col_idx)), dtype=np.float32)
        y_out = np.empty(total_rows, dtype=np.float64)
        meta_cids = np.empty(total_rows, dtype=np.int64)
        meta_eids: list[Any] = []
        local_slices: list[tuple[int, int]] = []
        local_edges: list[torch.Tensor] = []

        cursor = 0
        for g_idx in graph_indices:
            g = int(g_idx)
            s, e = dataset.graph_slices[g]
            n = e - s
            if n == 0:
                continue
            block = dataset.X_raw[s:e, :][:, col_idx].astype(np.float64, copy=True)
            invalid = ~np.isfinite(block)
            if np.any(invalid):
                r_inv, c_inv = np.where(invalid)
                block[r_inv, c_inv] = medians[c_inv]
            block -= means
            block /= stds
            X_out[cursor : cursor + n] = block.astype(np.float32)
            y_out[cursor : cursor + n] = dataset.y[s:e]
            meta_cids[cursor : cursor + n] = dataset.meta_case_ids[s:e]
            meta_eids.extend(dataset.meta_event_ids[s:e])
            local_slices.append((cursor, cursor + n))
            local_edges.append(dataset.graph_edge_indices[g])
            cursor += n

        meta_df = pd.DataFrame({case_col: meta_cids, event_col: meta_eids})
        return X_out, y_out, meta_df, local_slices, local_edges


def build_pyg_graphs_from_compact(
    X_2d: np.ndarray,
    y: np.ndarray,
    local_slices: list[tuple[int, int]],
    local_edges: list[torch.Tensor],
) -> list[Data]:
    """Construct PyG ``Data`` objects directly from scaled compact arrays."""
    graphs: list[Data] = []
    for (start, end), edge_index in zip(local_slices, local_edges):
        if end <= start:
            continue
        graphs.append(
            Data(
                x=torch.from_numpy(np.ascontiguousarray(X_2d[start:end], dtype=np.float32)),
                edge_index=edge_index,
                y=torch.from_numpy(np.ascontiguousarray(y[start:end], dtype=np.float32)),
            )
        )
    return graphs

