"""Stage 1b: Case-level structural-complexity / entanglement metrics for directed OCEL event graphs.

Computes four structural-complexity metrics per process execution graph directly
from the Stage 1a ``ocpa`` ``Feature_Storage`` (``data/preprocessed/feature_storage.pkl``),
normalizes each metric across cases to a comparable scale, and exports
``data/preprocessed/entanglement_scores.parquet``.

Metric mapping:
    - ``Metric_1``: ``branching_join_per_node`` — $B(G)$
    - ``Metric_2``: ``cycle_rank_per_node`` — $R(G)$
    - ``Metric_3``: ``directed_density`` — $D(G)$
    - ``Metric_4``: ``multi_object_frequency`` — $M(G)$

Strict Isolation Invariant:
    ``entanglement_scores.parquet`` MUST NEVER be imported or read during Stage 2
    (``ocpa_encoding.py``, ``feature_selection.py``, or ``cross_validation.py``).
    It is consumed exclusively by Stage 3 (``evaluation.py``).
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Union

import networkx as nx
import numpy as np
import pandas as pd
from ocpa.algo.predictive_monitoring.obj import Feature_Storage

from src import configure_logging, load_config

LOGGER = logging.getLogger(__name__)

MetricValue = Union[int, float, bool, str]
MetricRow = Dict[str, MetricValue]

METRIC_COLUMN_MAP: dict[str, str] = {
    "Metric_1": "branching_join_per_node",
    "Metric_2": "cycle_rank_per_node",
    "Metric_3": "directed_density",
    "Metric_4": "multi_object_frequency",
}


def compute_structural_metrics(
    graph: nx.DiGraph,
    event_objects_by_id: Mapping[Hashable, Sequence[Hashable]],
    graph_id: Optional[Hashable] = None,
) -> MetricRow:
    """Compute four structural-complexity metrics for one event graph.

    Args:
        graph:
            Directed graph whose nodes are event IDs. Edges must follow the
            graph-construction rule agreed for the experiment. Self-loops are
            rejected. Parallel edges are collapsed to one event-pair edge.
        event_objects_by_id:
            Mapping from each event ID to the distinct OCEL object IDs
            associated with that event. Every graph node must be present.
        graph_id:
            Stable identifier for the prediction instance/prefix. It is
            returned unchanged so the row can later be joined to predictions.

    Returns:
        A flat dictionary containing graph diagnostics and four metrics:
        branching_join_per_node, cycle_rank_per_node, directed_density, and
        multi_object_frequency.

    Metric definitions:
        B(G) = [sum_v max(0, indeg(v)-1) + sum_v max(0, outdeg(v)-1)] / n

        R(G) = [m_u - n + c] / n, where m_u and c are measured on the
        underlying simple undirected graph. The numerator is the cycle rank
        (first Betti number).

        D(G) = m / [n(n-1)], standard directed density without self-loops.

        M(G) = number of event nodes associated with >1 distinct OCEL object
               divided by n.

    Notes:
        * B(G) and R(G) are not bounded by 1.
        * Standard directed density for a DAG with a fixed global time order
          cannot exceed 0.5. This function does not rescale it.
        * R(G) uses an undirected projection. A non-zero value does not
          establish that the original directed graph has directed cycles.
        * M(G) comes from OCEL event-object associations, not graph degree.
    """
    if not graph.is_directed():
        raise TypeError("graph must be directed (for example, nx.DiGraph).")

    # This experiment treats an event-pair relation as a simple edge.
    # Collapsing parallel edges prevents repeated object associations from
    # artificially inflating density or topological degree.
    simple_graph = nx.DiGraph(graph)

    if nx.number_of_selfloops(simple_graph) > 0:
        raise ValueError("Self-loops are not allowed in the event graph.")

    nodes = list(simple_graph.nodes())
    n_nodes = len(nodes)
    if n_nodes == 0:
        raise ValueError("Cannot compute structural metrics for an empty graph.")

    missing_event_ids = [event_id for event_id in nodes if event_id not in event_objects_by_id]
    if missing_event_ids:
        preview = missing_event_ids[:5]
        suffix = "..." if len(missing_event_ids) > 5 else ""
        raise KeyError(
            f"Missing OCEL event-to-object associations for graph nodes {preview}{suffix}."
        )

    n_directed_edges = simple_graph.number_of_edges()

    # Metric 1: excess in-degree and out-degree beyond one predecessor and
    # one successor per event node.
    in_excess = sum(max(0, degree - 1) for _, degree in simple_graph.in_degree())
    out_excess = sum(max(0, degree - 1) for _, degree in simple_graph.out_degree())
    branching_join_excess = in_excess + out_excess
    branching_join_per_node = branching_join_excess / float(n_nodes)

    # Metric 2: cycle rank on the underlying simple undirected projection.
    undirected_graph = simple_graph.to_undirected(as_view=False)
    n_undirected_edges = undirected_graph.number_of_edges()
    n_components = nx.number_connected_components(undirected_graph)
    cycle_rank = n_undirected_edges - n_nodes + n_components
    cycle_rank_per_node = cycle_rank / float(n_nodes)

    # Metric 3: standard simple directed graph density.
    possible_directed_edges = n_nodes * (n_nodes - 1)
    directed_density = (
        n_directed_edges / float(possible_directed_edges) if possible_directed_edges > 0 else 0.0
    )

    # Metric 4: fraction of events associated with more than one distinct
    # OCEL object. De-duplicate object IDs defensively.
    multi_object_event_count = 0
    for event_id in nodes:
        object_ids = event_objects_by_id[event_id]
        distinct_object_ids = set(object_ids) if object_ids is not None else set()
        if len(distinct_object_ids) > 1:
            multi_object_event_count += 1
    multi_object_frequency = multi_object_event_count / float(n_nodes)

    return {
        "graph_id": graph_id if graph_id is not None else "",
        "n_nodes": n_nodes,
        "n_directed_edges": n_directed_edges,
        "n_undirected_edges": n_undirected_edges,
        "n_weak_components": n_components,
        "is_dag": nx.is_directed_acyclic_graph(simple_graph),
        "in_degree_excess": in_excess,
        "out_degree_excess": out_excess,
        "branching_join_excess": branching_join_excess,
        "branching_join_per_node": branching_join_per_node,
        "cycle_rank": cycle_rank,
        "cycle_rank_per_node": cycle_rank_per_node,
        "directed_density": directed_density,
        "multi_object_event_count": multi_object_event_count,
        "multi_object_frequency": multi_object_frequency,
    }


def feature_graph_to_networkx(
    feature_graph: Feature_Storage.Feature_Graph,
) -> tuple[nx.DiGraph, dict[Hashable, list[Hashable]]]:
    """Convert an ``ocpa`` ``Feature_Graph`` into a NetworkX ``DiGraph`` and object map.

    Args:
        feature_graph: Single-case ``ocpa`` ``Feature_Storage.Feature_Graph``.

    Returns:
        Tuple of ``(graph, event_objects_by_id)`` ready for
        ``compute_structural_metrics(...)``.
    """
    graph = nx.DiGraph()
    event_objects_by_id: dict[Hashable, list[Hashable]] = {}

    for node in feature_graph.nodes:
        eid = node.event_id
        graph.add_node(eid)
        raw_objs = node.objects if node.objects is not None else []
        event_objects_by_id[eid] = list(raw_objs)

    for edge in feature_graph.edges:
        graph.add_edge(edge.source, edge.target)

    return graph, event_objects_by_id


def compute_raw_structural_metrics_table(
    feature_storage: Feature_Storage,
    case_id_col: str = "Case_ID",
) -> pd.DataFrame:
    """Compute unnormalized structural-complexity diagnostics for all cases in ``Feature_Storage``.

    Args:
        feature_storage: Stage 1a ``ocpa`` ``Feature_Storage``.
        case_id_col: Case identifier column name (default ``"Case_ID"``).

    Returns:
        DataFrame with ``Case_ID`` and all diagnostic fields from
        ``compute_structural_metrics``.
    """
    if not feature_storage.feature_graphs:
        raise ValueError("Feature_Storage contains no Feature_Graphs.")

    rows: list[dict[str, Any]] = []
    for fg in sorted(feature_storage.feature_graphs, key=lambda g: g.pexec_id):
        case_id = int(fg.pexec_id)
        nx_graph, event_objects_by_id = feature_graph_to_networkx(fg)
        metrics_row = compute_structural_metrics(
            graph=nx_graph,
            event_objects_by_id=event_objects_by_id,
            graph_id=case_id,
        )
        rows.append({case_id_col: case_id, **metrics_row})

    return pd.DataFrame(rows)


def normalize_metric_series(series: pd.Series, method: str = "minmax") -> pd.Series:
    """Normalize a case-level metric series across cases.

    Args:
        series: Raw numeric metric values indexed by case.
        method: ``"minmax"`` (scales to ``[0, 1]``) or ``"zscore"`` (zero mean,
            unit standard deviation).

    Returns:
        Normalized ``pd.Series`` of floats.

    Raises:
        ValueError: If ``method`` is not supported.
    """
    values = series.astype(float)
    if method == "minmax":
        vmin = float(values.min())
        vmax = float(values.max())
        denom = vmax - vmin
        if denom == 0.0 or not np.isfinite(denom):
            return pd.Series(np.zeros(len(values), dtype=float), index=series.index)
        return (values - vmin) / denom

    if method == "zscore":
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std == 0.0 or not np.isfinite(std):
            return pd.Series(np.zeros(len(values), dtype=float), index=series.index)
        return (values - mean) / std

    raise ValueError(f"Unsupported normalization method '{method}'. Use 'minmax' or 'zscore'.")


def compute_entanglement_scores(
    feature_storage: Feature_Storage,
    case_id_col: str = "Case_ID",
    normalization: str = "minmax",
) -> pd.DataFrame:
    """Compute normalized entanglement metrics and ``Avg_Entanglement_Score`` per case.

    Maps the four structural-complexity metrics from ``compute_structural_metrics``
    to ``[Metric_1, Metric_2, Metric_3, Metric_4]``, normalizes each across cases,
    and averages them into ``Avg_Entanglement_Score``.

    Args:
        feature_storage: Stage 1a ``ocpa`` ``Feature_Storage``.
        case_id_col: Case identifier column name (default ``"Case_ID"``).
        normalization: Normalization scheme (``"minmax"`` or ``"zscore"``).

    Returns:
        DataFrame with exact columns
        ``[Case_ID, Metric_1, Metric_2, Metric_3, Metric_4, Avg_Entanglement_Score]``.
    """
    raw_df = compute_raw_structural_metrics_table(
        feature_storage=feature_storage,
        case_id_col=case_id_col,
    )

    out = pd.DataFrame({case_id_col: raw_df[case_id_col]})
    metric_cols = list(METRIC_COLUMN_MAP.keys())

    for out_col, raw_col in METRIC_COLUMN_MAP.items():
        vals = pd.to_numeric(raw_df[raw_col], errors="coerce").astype(float)
        if not np.isfinite(vals).all():
            LOGGER.warning(
                "Non-finite values detected in %s (%s); coercing to 0.0 before normalization.",
                out_col,
                raw_col,
            )
            vals = vals.where(np.isfinite(vals), 0.0)
        out[out_col] = normalize_metric_series(vals, method=normalization)

    out["Avg_Entanglement_Score"] = out[metric_cols].mean(axis=1)
    return out[[case_id_col, *metric_cols, "Avg_Entanglement_Score"]]


def run_entanglement_metrics(config_path: str | Path = "config.yaml") -> pd.DataFrame:
    """Run Stage 1b end-to-end and export ``entanglement_scores.parquet``.

    Args:
        config_path: Path to ``config.yaml``.

    Returns:
        Computed case-level entanglement DataFrame.
    """
    config = load_config(config_path)
    fs_path = Path(config["paths"]["feature_storage"])
    output_path = Path(config["paths"]["entanglement_scores"])
    case_id_col = config["identifiers"]["case_id"]
    normalization = str(config.get("entanglement", {}).get("normalization", "minmax"))

    if not fs_path.exists():
        raise FileNotFoundError(
            f"Feature_Storage artifact not found at {fs_path}. Run Stage 1a first."
        )

    with fs_path.open("rb") as handle:
        feature_storage: Feature_Storage = pickle.load(handle)

    raw_df = compute_raw_structural_metrics_table(
        feature_storage=feature_storage,
        case_id_col=case_id_col,
    )
    scores_df = compute_entanglement_scores(
        feature_storage=feature_storage,
        case_id_col=case_id_col,
        normalization=normalization,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(output_path, index=False)

    # Export full raw + normalized diagnostics CSV for inspection (Stage 1b write-only)
    diag_df = raw_df.merge(scores_df, on=case_id_col, how="left")
    diag_path = output_path.parent / "entanglement_raw_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False)

    LOGGER.info(
        "Saved entanglement scores for %d cases to %s (normalization=%s).",
        len(scores_df),
        output_path,
        normalization,
    )
    LOGGER.info("Saved raw + normalized entanglement diagnostics to %s.", diag_path)
    return scores_df


def main() -> None:
    """CLI entry point for Stage 1b entanglement metric computation."""
    parser = argparse.ArgumentParser(description="Stage 1b: Structural Entanglement Metrics")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    configure_logging()
    run_entanglement_metrics(config_path=args.config)


if __name__ == "__main__":
    main()
