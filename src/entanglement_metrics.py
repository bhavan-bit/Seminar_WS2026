"""Stage 1b: Case-level structural entanglement metrics.

Computes 4 modular case-level structural entanglement metrics directly from the
Stage 1a ``ocpa`` ``Feature_Storage`` (``data/preprocessed/feature_storage.pkl``),
normalizes each metric across cases to a comparable scale, and exports
``data/preprocessed/entanglement_scores.parquet``.

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
from typing import Any, Callable

import numpy as np
import pandas as pd
from ocpa.algo.predictive_monitoring.obj import Feature_Storage

from src import configure_logging, load_config

LOGGER = logging.getLogger(__name__)

MetricFn = Callable[[Feature_Storage.Feature_Graph], float]


# ==============================================================================
# Modular Placeholder Metric Functions
# Replace the body of each function below with your exact structural formula.
# Each function receives one ``ocpa`` ``Feature_Storage.Feature_Graph``:
#   - feature_graph.pexec_id: int (Case_ID)
#   - feature_graph.nodes: list[Node] with .event_id, .objects (list[(ot, oid)])
#   - feature_graph.edges: list[Edge] with .source, .target, .objects
# ==============================================================================


def compute_metric_1(feature_graph: Feature_Storage.Feature_Graph) -> float:
    """Placeholder Metric 1: Average directed edge density per event node.

    Replace this body with the exact formula for Entanglement Metric 1.

    Args:
        feature_graph: Single-case ``ocpa`` ``Feature_Graph``.

    Returns:
        Raw scalar metric value for this case.
    """
    num_nodes = len(feature_graph.nodes)
    num_edges = len(feature_graph.edges)
    if num_nodes <= 0:
        return 0.0
    return float(num_edges / num_nodes)


def compute_metric_2(feature_graph: Feature_Storage.Feature_Graph) -> float:
    """Placeholder Metric 2: Mean object multiplicity per event (mean |O_e|).

    Replace this body with the exact formula for Entanglement Metric 2.

    Args:
        feature_graph: Single-case ``ocpa`` ``Feature_Graph``.

    Returns:
        Raw scalar metric value for this case.
    """
    if not feature_graph.nodes:
        return 0.0
    return float(np.mean([len(node.objects or []) for node in feature_graph.nodes]))


def compute_metric_3(feature_graph: Feature_Storage.Feature_Graph) -> float:
    """Placeholder Metric 3: Cross-type interaction rate (mean distinct object types per event).

    Replace this body with the exact formula for Entanglement Metric 3.

    Args:
        feature_graph: Single-case ``ocpa`` ``Feature_Graph``.

    Returns:
        Raw scalar metric value for this case.
    """
    if not feature_graph.nodes:
        return 0.0
    distinct_types = [
        len({str(ot) for (ot, _oid) in (node.objects or [])})
        for node in feature_graph.nodes
    ]
    return float(np.mean(distinct_types))


def compute_metric_4(feature_graph: Feature_Storage.Feature_Graph) -> float:
    """Placeholder Metric 4: Convergence/divergence node ratio (in-degree > 1 or out-degree > 1).

    Replace this body with the exact formula for Entanglement Metric 4.

    Args:
        feature_graph: Single-case ``ocpa`` ``Feature_Graph``.

    Returns:
        Raw scalar metric value for this case.
    """
    num_nodes = len(feature_graph.nodes)
    if num_nodes <= 0 or not feature_graph.edges:
        return 0.0

    in_deg: dict[Any, int] = {node.event_id: 0 for node in feature_graph.nodes}
    out_deg: dict[Any, int] = {node.event_id: 0 for node in feature_graph.nodes}
    for edge in feature_graph.edges:
        if edge.source in out_deg:
            out_deg[edge.source] += 1
        if edge.target in in_deg:
            in_deg[edge.target] += 1

    entangled = sum(
        1 for eid in in_deg if in_deg[eid] > 1 or out_deg[eid] > 1
    )
    return float(entangled / num_nodes)


METRIC_FUNCTIONS: dict[str, MetricFn] = {
    "Metric_1": compute_metric_1,
    "Metric_2": compute_metric_2,
    "Metric_3": compute_metric_3,
    "Metric_4": compute_metric_4,
}


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
    metric_functions: dict[str, MetricFn] | None = None,
) -> pd.DataFrame:
    """Compute normalized entanglement metrics and ``Avg_Entanglement_Score`` per case.

    Args:
        feature_storage: Stage 1a ``ocpa`` ``Feature_Storage``.
        case_id_col: Case identifier column name (default ``"Case_ID"``).
        normalization: Normalization scheme (``"minmax"`` or ``"zscore"``).
        metric_functions: Optional override mapping metric names to callables
            ``(Feature_Graph) -> float``. Defaults to ``METRIC_FUNCTIONS``.

    Returns:
        DataFrame with exact columns
        ``[Case_ID, Metric_1, Metric_2, Metric_3, Metric_4, Avg_Entanglement_Score]``.
    """
    if not feature_storage.feature_graphs:
        raise ValueError("Feature_Storage contains no Feature_Graphs.")

    funcs = metric_functions or METRIC_FUNCTIONS
    metric_names = list(funcs.keys())

    records: list[dict[str, Any]] = []
    for fg in sorted(feature_storage.feature_graphs, key=lambda g: g.pexec_id):
        case_id = int(fg.pexec_id)
        row: dict[str, Any] = {case_id_col: case_id}
        for name, fn in funcs.items():
            raw_val = float(fn(fg))
            if not np.isfinite(raw_val):
                LOGGER.warning(
                    "Non-finite value for %s on Case_ID=%s; coercing to 0.0.",
                    name,
                    case_id,
                )
                raw_val = 0.0
            row[name] = raw_val
        records.append(row)

    df = pd.DataFrame(records)
    for name in metric_names:
        df[name] = normalize_metric_series(df[name], method=normalization)

    df["Avg_Entanglement_Score"] = df[metric_names].mean(axis=1)
    ordered_cols = [case_id_col, *metric_names, "Avg_Entanglement_Score"]
    return df[ordered_cols]


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

    scores_df = compute_entanglement_scores(
        feature_storage=feature_storage,
        case_id_col=case_id_col,
        normalization=normalization,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(output_path, index=False)
    LOGGER.info(
        "Saved entanglement scores for %d cases to %s (normalization=%s).",
        len(scores_df),
        output_path,
        normalization,
    )
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
