"""Stage 3: Post-hoc evaluation and structural entanglement regime analysis.

Consumes the unified Out-of-Fold (OOF) predictions from Stage 2
(``data/results/test_predictions.parquet``) and the isolated case-level
entanglement scores from Stage 1b (``data/preprocessed/entanglement_scores.parquet``)
to compute both **Event-Level MAE** and **Case-Normalized MAE** for:
    - Part 1: Global 3x3 Representation vs. Architecture Matrix
      (``data/results/matrix1_global_mae.csv``).
    - Part 2: 3x9 Structural Entanglement Regime Matrix
      (``data/results/matrix2_regime_mae.csv``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import configure_logging, load_config

LOGGER = logging.getLogger(__name__)

PREDICTION_COLUMNS: list[str] = [
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

TIER_MAP: list[tuple[str, str]] = [
    ("Tier 1", "T1"),
    ("Tier 2", "T2"),
    ("Tier 3", "T3"),
]

MODEL_MAP: list[tuple[str, str]] = [
    ("XGBoost", "XGB"),
    ("LSTM", "LSTM"),
    ("GNN", "GNN"),
]


def compute_event_level_mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Compute Event-Level MAE: mean(|y_true - y_pred|) across all events.

    Args:
        y_true: Ground-truth event targets.
        y_pred: Event predictions.

    Returns:
        Scalar Event-Level MAE.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def compute_case_normalized_mae(
    df: pd.DataFrame,
    case_col: str,
    target_col: str,
    pred_col: str,
) -> float:
    """Compute Case-Normalized MAE: unweighted mean of per-case event MAEs.

    Computes MAE within each ``Case_ID`` first, then averages equally across
    all cases in ``df`` so unusually long cases do not dominate shorter ones.

    Args:
        df: Prediction DataFrame containing ``case_col``, ``target_col``, and ``pred_col``.
        case_col: Case identifier column name.
        target_col: Ground-truth column name.
        pred_col: Prediction column name.

    Returns:
        Scalar Case-Normalized MAE.
    """
    if df.empty:
        return float("nan")
    abs_err = np.abs(
        df[target_col].to_numpy(dtype=np.float64) - df[pred_col].to_numpy(dtype=np.float64)
    )
    per_case_mae = (
        pd.DataFrame({case_col: df[case_col].to_numpy(), "abs_err": abs_err})
        .groupby(case_col, sort=False)["abs_err"]
        .mean()
    )
    return float(per_case_mae.mean())


def build_matrix1_global_mae(
    oof_df: pd.DataFrame,
    case_col: str = "Case_ID",
    target_col: str = "y_true",
) -> pd.DataFrame:
    """Build Part 1 Global 3x3 MAE matrix for both Event-Level and Case-Normalized MAE.

    Args:
        oof_df: Combined Out-of-Fold predictions DataFrame across all cases.
        case_col: Case identifier column name.
        target_col: Ground-truth target column name.

    Returns:
        DataFrame with columns ``[Metric, Tier, XGBoost, LSTM, GNN]`` containing
        both 3x3 blocks (``Event_Level_MAE`` and ``Case_Normalized_MAE``).
    """
    rows: list[dict[str, Any]] = []

    for metric_name in ("Event_Level_MAE", "Case_Normalized_MAE"):
        for tier_label, tier_code in TIER_MAP:
            row: dict[str, Any] = {"Metric": metric_name, "Tier": tier_label}
            for model_label, model_code in MODEL_MAP:
                pred_col = f"Pred_{model_code}_{tier_code}"
                if metric_name == "Event_Level_MAE":
                    row[model_label] = compute_event_level_mae(
                        oof_df[target_col], oof_df[pred_col]
                    )
                else:
                    row[model_label] = compute_case_normalized_mae(
                        oof_df, case_col=case_col, target_col=target_col, pred_col=pred_col
                    )
            rows.append(row)

    return pd.DataFrame(rows)[["Metric", "Tier", "XGBoost", "LSTM", "GNN"]]


def assign_entanglement_regimes(
    entanglement_df: pd.DataFrame,
    case_col: str = "Case_ID",
    num_regimes: int = 3,
    regime_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Assign cases into balanced entanglement regimes via ``pd.qcut(..., q=3)``.

    Args:
        entanglement_df: Case-level entanglement table containing ``Case_ID`` and
            ``Avg_Entanglement_Score``.
        case_col: Case identifier column name.
        num_regimes: Number of quantile bins (default 3).
        regime_labels: Labels ordered from lowest to highest entanglement.

    Returns:
        Copy of ``entanglement_df`` with an added ``Regime`` column.
    """
    labels = regime_labels or ["Regime_1_Low", "Regime_2_Medium", "Regime_3_High"]
    if len(labels) != num_regimes:
        raise ValueError(
            f"Length of regime_labels ({len(labels)}) must match num_regimes ({num_regimes})."
        )
    if len(entanglement_df) < num_regimes:
        raise ValueError(
            f"Need at least {num_regimes} cases to form {num_regimes} entanglement regimes, "
            f"got {len(entanglement_df)}."
        )

    out = entanglement_df.copy()
    scores = out["Avg_Entanglement_Score"].astype(float)
    try:
        out["Regime"] = pd.qcut(scores, q=num_regimes, labels=labels)
    except ValueError:
        LOGGER.warning(
            "Duplicate quantile edges in Avg_Entanglement_Score; breaking ties via "
            "deterministic rank(method='first') before pd.qcut."
        )
        ranked = scores.rank(method="first")
        out["Regime"] = pd.qcut(ranked, q=num_regimes, labels=labels)

    out["Regime"] = out["Regime"].astype(str)
    return out


def build_matrix2_regime_mae(
    oof_df: pd.DataFrame,
    entanglement_df: pd.DataFrame,
    case_col: str = "Case_ID",
    target_col: str = "y_true",
    num_regimes: int = 3,
    regime_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Build Part 2 Structural Entanglement Regime MAE matrix in Part 1 table style.

    Left-joins quantile-bucketed case entanglement regimes onto OOF predictions
    by ``Case_ID`` and computes both Event-Level MAE and Case-Normalized MAE
    per ``(Regime, Tier)`` with columns ``[XGBoost, LSTM, GNN]``.

    Args:
        oof_df: Combined Out-of-Fold predictions DataFrame.
        entanglement_df: Case-level entanglement scores DataFrame from Stage 1b.
        case_col: Case identifier column name.
        target_col: Ground-truth target column name.
        num_regimes: Number of balanced quantile regimes (default 3).
        regime_labels: Ordered regime labels.

    Returns:
        DataFrame with columns
        ``[Metric, Regime, Tier, Num_Cases, Num_Events, XGBoost, LSTM, GNN]``.
    """
    labels = regime_labels or ["Regime_1_Low", "Regime_2_Medium", "Regime_3_High"]
    bucketed = assign_entanglement_regimes(
        entanglement_df=entanglement_df,
        case_col=case_col,
        num_regimes=num_regimes,
        regime_labels=labels,
    )

    merged = oof_df.merge(
        bucketed[[case_col, "Avg_Entanglement_Score", "Regime"]],
        on=case_col,
        how="left",
        validate="many_to_one",
    )
    if merged["Regime"].isna().any():
        missing_cases = sorted(merged.loc[merged["Regime"].isna(), case_col].unique().tolist())
        raise ValueError(
            f"Missing entanglement regime for Case_IDs after left-join: {missing_cases}"
        )

    rows: list[dict[str, Any]] = []
    for metric_name in ("Event_Level_MAE", "Case_Normalized_MAE"):
        for regime in labels:
            sub = merged[merged["Regime"] == regime]
            num_cases = int(sub[case_col].nunique())
            num_events = int(len(sub))
            for tier_label, tier_code in TIER_MAP:
                row: dict[str, Any] = {
                    "Metric": metric_name,
                    "Regime": regime,
                    "Tier": tier_label,
                    "Num_Cases": num_cases,
                    "Num_Events": num_events,
                }
                for model_label, model_code in MODEL_MAP:
                    pred_col = f"Pred_{model_code}_{tier_code}"
                    if metric_name == "Event_Level_MAE":
                        row[model_label] = compute_event_level_mae(
                            sub[target_col], sub[pred_col]
                        )
                    else:
                        row[model_label] = compute_case_normalized_mae(
                            sub,
                            case_col=case_col,
                            target_col=target_col,
                            pred_col=pred_col,
                        )
                rows.append(row)

    ordered_cols = [
        "Metric",
        "Regime",
        "Tier",
        "Num_Cases",
        "Num_Events",
        "XGBoost",
        "LSTM",
        "GNN",
    ]
    return pd.DataFrame(rows)[ordered_cols]


def split_matrix2_by_tier(matrix2_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the Part 2 regime table into three 3x3 tables, one per feature tier.

    Each returned DataFrame has columns
    ``[Metric, Regime, Num_Cases, Num_Events, XGBoost, LSTM, GNN]``
    with 3 regime rows per metric (``Event_Level_MAE`` and ``Case_Normalized_MAE``).

    Args:
        matrix2_df: Combined Part 2 DataFrame returned by ``build_matrix2_regime_mae``.

    Returns:
        Dictionary mapping ``"Tier 1"``, ``"Tier 2"``, ``"Tier 3"`` to their
        respective 3x3 regime-by-model DataFrames.
    """
    keep_cols = ["Metric", "Regime", "Num_Cases", "Num_Events", "XGBoost", "LSTM", "GNN"]
    per_tier: dict[str, pd.DataFrame] = {}
    for tier_label, _tier_code in TIER_MAP:
        sub = matrix2_df[matrix2_df["Tier"] == tier_label][keep_cols].reset_index(drop=True)
        per_tier[tier_label] = sub
    return per_tier


def run_evaluation(config_path: str | Path = "config.yaml") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Stage 3 evaluation end-to-end and export Matrix 1 and Matrix 2 CSVs.

    Exports:
        - ``matrix1_global_mae.csv`` (Part 1 global 3x3 matrix)
        - ``matrix2_regime_mae.csv`` (Part 2 combined regime x tier matrix)
        - ``matrix2_regime_mae_tier1.csv``, ``matrix2_regime_mae_tier2.csv``,
          ``matrix2_regime_mae_tier3.csv`` (three separate 3x3 Regime x Model tables)

    Args:
        config_path: Path to ``config.yaml``.

    Returns:
        Tuple of ``(matrix1_df, matrix2_df)``.
    """
    config = load_config(config_path)
    case_col = config["identifiers"]["case_id"]
    target_col = config["target"]["column"]

    preds_path = Path(config["paths"]["test_predictions"])
    entang_path = Path(config["paths"]["entanglement_scores"])
    m1_out = Path(config["paths"]["matrix1_global_mae"])
    m2_out = Path(config["paths"]["matrix2_regime_mae"])

    if not preds_path.exists():
        raise FileNotFoundError(
            f"OOF predictions not found at {preds_path}. Run Stage 2 first."
        )
    if not entang_path.exists():
        raise FileNotFoundError(
            f"Entanglement scores not found at {entang_path}. Run Stage 1b first."
        )

    oof_df = pd.read_parquet(preds_path)
    entanglement_df = pd.read_parquet(entang_path)

    ent_cfg = config.get("entanglement", {})
    num_regimes = int(ent_cfg.get("num_regimes", 3))
    regime_labels = list(
        ent_cfg.get("regime_labels", ["Regime_1_Low", "Regime_2_Medium", "Regime_3_High"])
    )

    matrix1_df = build_matrix1_global_mae(
        oof_df=oof_df,
        case_col=case_col,
        target_col=target_col,
    )
    matrix2_df = build_matrix2_regime_mae(
        oof_df=oof_df,
        entanglement_df=entanglement_df,
        case_col=case_col,
        target_col=target_col,
        num_regimes=num_regimes,
        regime_labels=regime_labels,
    )
    matrix2_per_tier = split_matrix2_by_tier(matrix2_df)

    m1_out.parent.mkdir(parents=True, exist_ok=True)
    m2_out.parent.mkdir(parents=True, exist_ok=True)
    matrix1_df.to_csv(m1_out, index=False)
    matrix2_df.to_csv(m2_out, index=False)

    for idx, (tier_label, tier_df) in enumerate(matrix2_per_tier.items(), start=1):
        tier_path = m2_out.with_name(f"{m2_out.stem}_tier{idx}{m2_out.suffix}")
        tier_df.to_csv(tier_path, index=False)
        LOGGER.info("Saved Part 2 %s 3x3 MAE matrix to %s", tier_label, tier_path)

    LOGGER.info("Saved Part 1 Global 3x3 MAE matrix to %s", m1_out)
    LOGGER.info("Saved Part 2 combined Regime MAE matrix to %s", m2_out)
    return matrix1_df, matrix2_df


def main() -> None:
    """CLI entry point for Stage 3 evaluation."""
    parser = argparse.ArgumentParser(description="Stage 3: Global & Regime MAE Evaluation")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    configure_logging()
    run_evaluation(config_path=args.config)


if __name__ == "__main__":
    main()

