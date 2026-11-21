"""Post-hoc visual diagnostic report generator for Stages 1–3.

Reads the intermediate CSV/Parquet/JSON artifacts produced by Stages 1–3 and
generates publication-ready PNG charts under ``data/results/diagnostics/plots/``
plus a single human-readable HTML dashboard at
``data/results/diagnostics/report.html``.

All outputs live inside ``data/results/`` and are excluded from git via
``.gitignore``.
"""

from __future__ import annotations

import argparse
import html
import logging
import os
from pathlib import Path
from typing import Any

# Ensure headless rendering and writable matplotlib cache before importing pyplot
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import configure_logging, load_config

LOGGER = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9.5,
    }
)


def _save_fig(fig: plt.Figure, out_path: Path) -> Path:
    """Save and close a matplotlib figure cleanly."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.info("Saved diagnostic plot: %s", out_path)
    return out_path


def _plot_heatmap(
    ax: plt.Axes,
    df_matrix: pd.DataFrame,
    title: str,
    cmap: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Render an annotated numeric heatmap on ``ax``."""
    vals = df_matrix.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(df_matrix.columns)))
    ax.set_xticklabels(df_matrix.columns, rotation=0)
    ax.set_yticks(range(len(df_matrix.index)))
    ax.set_yticklabels(df_matrix.index)
    ax.set_title(title, fontweight="bold", pad=8)
    ax.grid(False)

    mid = (np.nanmin(vals) + np.nanmax(vals)) / 2.0 if vals.size else 0.0
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            val = vals[r, c]
            txt = f"{val:.3f}" if np.isfinite(val) else "NaN"
            color = "white" if np.isfinite(val) and val > mid else "#111111"
            ax.text(c, r, txt, ha="center", va="center", color=color, fontweight="bold", fontsize=9.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_global_mae_matrix(results_dir: Path, plots_dir: Path, unit: str) -> Path | None:
    """Plot Part 1 Global 3x3 MAE matrix (Event-Level & Case-Normalized)."""
    m1_path = results_dir / "matrix1_global_mae.csv"
    if not m1_path.exists():
        return None
    df = pd.read_csv(m1_path)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    all_vals = df[["XGBoost", "LSTM", "GNN"]].to_numpy(dtype=float)
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    for ax, metric in zip(axes, ("Event_Level_MAE", "Case_Normalized_MAE")):
        sub = df[df["Metric"] == metric].set_index("Tier")[["XGBoost", "LSTM", "GNN"]]
        _plot_heatmap(
            ax,
            sub,
            title=f"{metric.replace('_', ' ')} ({unit})",
            vmin=vmin,
            vmax=vmax,
        )

    fig.suptitle("Part 1 — Global 3x3 Representation vs. Architecture MAE", fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save_fig(fig, plots_dir / "01_global_mae_matrix.png")


def plot_regime_mae_by_tier(results_dir: Path, plots_dir: Path, unit: str) -> Path | None:
    """Plot Part 2 three separate 3x3 Regime x Architecture heatmaps (one per Tier)."""
    tier_paths = [results_dir / f"matrix2_regime_mae_tier{i}.csv" for i in (1, 2, 3)]
    if not all(p.exists() for p in tier_paths):
        return None

    tier_dfs = [pd.read_csv(p) for p in tier_paths]
    all_vals = np.concatenate(
        [d[["XGBoost", "LSTM", "GNN"]].to_numpy(dtype=float).ravel() for d in tier_dfs]
    )
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.8))
    for row_idx, metric in enumerate(("Event_Level_MAE", "Case_Normalized_MAE")):
        for col_idx, (tier_num, tdf) in enumerate(zip((1, 2, 3), tier_dfs)):
            sub = tdf[tdf["Metric"] == metric].copy()
            sub["Label"] = [
                f"{r.replace('Regime_', 'R').replace('_', ' ')} (n={int(nc)})"
                for r, nc in zip(sub["Regime"], sub["Num_Cases"])
            ]
            mat = sub.set_index("Label")[["XGBoost", "LSTM", "GNN"]]
            _plot_heatmap(
                axes[row_idx, col_idx],
                mat,
                title=f"Tier {tier_num} — {metric.replace('_', ' ')}",
                vmin=vmin,
                vmax=vmax,
            )

    fig.suptitle(
        f"Part 2 — Structural Entanglement Regime Breakdown (3x3 per Tier, MAE in {unit})",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return _save_fig(fig, plots_dir / "02_regime_mae_by_tier.png")


def plot_shap_top_features(diag_dir: Path, plots_dir: Path, top_k: int = 20) -> Path | None:
    """Plot top-K Tier 3 features by cross-fold validation mean |SHAP| and selection count."""
    stab_path = diag_dir / "shap_feature_stability_summary.csv"
    if not stab_path.exists():
        return None
    df = pd.read_csv(stab_path)
    if df.empty:
        return None

    df["Plot_SHAP"] = df["Mean_Best_Abs_SHAP"].fillna(df["Mean_Initial_Abs_SHAP"]).fillna(0.0)
    df["Plot_Std"] = df["Std_Best_Abs_SHAP"].fillna(0.0)
    top = df.sort_values(["Plot_SHAP", "Selection_Count"], ascending=[False, False]).head(top_k)
    top = top.iloc[::-1]  # largest on top for horizontal bar chart

    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.34 * len(top))))
    colors = [
        "#1f77b4" if rate >= 0.8 else ("#ff7f0e" if rate >= 0.4 else "#999999")
        for rate in top["Selection_Rate"]
    ]
    y_pos = np.arange(len(top))
    ax.barh(
        y_pos,
        top["Plot_SHAP"],
        xerr=top["Plot_Std"],
        color=colors,
        alpha=0.9,
        capsize=3,
        ecolor="#444444",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["Feature"], fontsize=8.5)
    ax.set_xlabel("Mean |SHAP| on Validation Split (Sub-Val) ± 1 SD across Folds")
    ax.set_title(
        f"Top {len(top)} Tier 3 Features by Out-of-Sample Validation SHAP Importance",
        fontweight="bold",
    )

    max_x = float((top["Plot_SHAP"] + top["Plot_Std"]).max()) if len(top) else 1.0
    ax.set_xlim(0.0, max(max_x * 1.18, 1e-4))
    for i, (_, row) in enumerate(top.iterrows()):
        cnt = int(row["Selection_Count"])
        rate = float(row["Selection_Rate"])
        n_folds = int(round(cnt / rate)) if rate > 0 else 5
        x_text = float(row["Plot_SHAP"] + row["Plot_Std"]) + max_x * 0.015
        ax.text(x_text, i, f"{cnt}/{n_folds} folds", va="center", fontsize=8, color="#333333")

    fig.tight_layout()
    return _save_fig(fig, plots_dir / "03_shap_top_features.png")


def plot_shap_rfe_curves(diag_dir: Path, plots_dir: Path, unit: str) -> Path | None:
    """Plot SHAP-RFE validation MAE trajectory per fold with best-subset markers."""
    trace_path = diag_dir / "shap_rfe_trace.csv"
    if not trace_path.exists():
        return None
    df = pd.read_csv(trace_path)
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    cmap = plt.get_cmap("tab10")
    for idx, (fold_id, grp) in enumerate(df.groupby("Fold_ID", sort=True)):
        grp = grp.sort_values("Iteration")
        color = cmap(idx % 10)
        ax.plot(
            grp["Num_Features"],
            grp["Val_MAE"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"Fold {int(fold_id)}",
        )
        best_row = grp.loc[grp["Val_MAE"].idxmin()]
        ax.scatter(
            [best_row["Num_Features"]],
            [best_row["Val_MAE"]],
            marker="*",
            s=180,
            color=color,
            edgecolors="black",
            zorder=5,
        )

    ax.invert_xaxis()  # Pruning progresses left -> right (more features -> fewer features)
    ax.set_xlabel("Remaining Feature Count (Pruning Direction →)")
    ax.set_ylabel(f"Sub-Val MAE ({unit})")
    ax.set_title("Nested Tier 3 SHAP-RFE Curves per Fold (★ = Selected Best Subset)", fontweight="bold")
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    return _save_fig(fig, plots_dir / "04_shap_rfe_curves.png")


def plot_training_curves(diag_dir: Path, plots_dir: Path, unit: str) -> Path | None:
    """Plot LSTM & GNN per-epoch Train vs. Sub-Val MAE learning curves."""
    hist_path = diag_dir / "training_history.csv"
    if not hist_path.exists():
        return None
    df = pd.read_csv(hist_path)
    if df.empty:
        return None

    models = ["LSTM", "GNN"]
    tiers = ["T1", "T2", "T3"]
    fig, axes = plt.subplots(len(models), len(tiers), figsize=(14.0, 6.8), sharex=True)
    cmap = plt.get_cmap("tab10")

    for r_idx, model_name in enumerate(models):
        for c_idx, tier_code in enumerate(tiers):
            ax = axes[r_idx, c_idx]
            sub = df[(df["Model"] == model_name) & (df["Tier"] == tier_code)]
            for f_idx, (fold_id, fgrp) in enumerate(sub.groupby("Fold_ID", sort=True)):
                fgrp = fgrp.sort_values("Epoch")
                color = cmap(f_idx % 10)
                ax.plot(
                    fgrp["Epoch"],
                    fgrp["Train_MAE"],
                    linestyle="--",
                    alpha=0.55,
                    color=color,
                )
                ax.plot(
                    fgrp["Epoch"],
                    fgrp["Val_MAE"],
                    linestyle="-",
                    marker=".",
                    linewidth=1.7,
                    color=color,
                    label=f"F{int(fold_id)}" if (r_idx == 0 and c_idx == 0) else None,
                )
            ax.set_title(f"{model_name} — {tier_code}", fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(f"MAE ({unit})")
            if r_idx == len(models) - 1:
                ax.set_xlabel("Epoch (dashed=Train, solid=Sub-Val)")

    axes[0, 0].legend(title="Fold", ncol=5, fontsize=8, loc="upper right")
    fig.suptitle("Neural Model Convergence Curves (Train vs. Sub-Val MAE)", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save_fig(fig, plots_dir / "05_training_curves.png")


def plot_fold_and_case_mae(diag_dir: Path, plots_dir: Path, unit: str) -> Path | None:
    """Plot per-fold holdout MAE and per-case MAE across all 20 sessions."""
    fold_path = diag_dir / "fold_metrics.csv"
    case_path = diag_dir / "per_case_mae.csv"
    if not fold_path.exists() or not case_path.exists():
        return None

    fm = pd.read_csv(fold_path)
    cm = pd.read_csv(case_path)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.8))

    # Left: Per-fold Tier 3 Holdout MAE by Model
    t3_fm = fm[fm["Tier"] == "Tier 3"].pivot(index="Fold_ID", columns="Model", values="Event_Level_MAE")
    t3_fm = t3_fm[["XGBoost", "LSTM", "GNN"]]
    t3_fm.plot(kind="bar", ax=axes[0], width=0.72, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[0].set_title("Per-Fold Holdout MAE (Tier 3)", fontweight="bold")
    axes[0].set_xlabel("Outer Fold ID")
    axes[0].set_ylabel(f"Event-Level MAE ({unit})")
    axes[0].tick_params(axis="x", rotation=0)

    # Right: Per-case Tier 3 MAE across all cases
    cm_sorted = cm.sort_values("Case_ID")
    x = np.arange(len(cm_sorted))
    axes[1].plot(x, cm_sorted["Pred_XGB_T3_MAE"], marker="o", label="XGBoost T3", color="#1f77b4")
    axes[1].plot(x, cm_sorted["Pred_LSTM_T3_MAE"], marker="s", label="LSTM T3", color="#ff7f0e")
    axes[1].plot(x, cm_sorted["Pred_GNN_T3_MAE"], marker="^", label="GNN T3", color="#2ca02c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(int(c)) for c in cm_sorted["Case_ID"]], fontsize=8)
    axes[1].set_xlabel("Case_ID")
    axes[1].set_ylabel(f"Case MAE ({unit})")
    axes[1].set_title("Per-Case Holdout MAE (Tier 3 across All Cases)", fontweight="bold")
    axes[1].legend(frameon=True)

    fig.tight_layout()
    return _save_fig(fig, plots_dir / "06_fold_and_case_mae.png")


def plot_entanglement_regimes(diag_dir: Path, plots_dir: Path) -> Path | None:
    """Plot all cases ranked by Avg_Entanglement_Score colored by Regime + 4 metric components."""
    reg_path = diag_dir / "case_regime_assignments.csv"
    if not reg_path.exists():
        return None
    df = pd.read_csv(reg_path).sort_values(["Avg_Entanglement_Score", "Case_ID"]).reset_index(drop=True)

    regime_colors = {
        "Regime_1_Low": "#2b83ba",
        "Regime_2_Medium": "#fdae61",
        "Regime_3_High": "#d7191c",
    }
    labels = [
        f"C{int(cid)} ({str(obj)[-7:]})" if "Case_Object_ID" in df.columns and pd.notna(obj) else f"C{int(cid)}"
        for cid, obj in zip(
            df["Case_ID"],
            df["Case_Object_ID"] if "Case_Object_ID" in df.columns else [None] * len(df),
        )
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2))
    y_pos = np.arange(len(df))
    bar_colors = [regime_colors.get(r, "#888888") for r in df["Regime"]]

    axes[0].barh(y_pos, df["Avg_Entanglement_Score"], color=bar_colors, alpha=0.9)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Avg_Entanglement_Score (Mean of Normalized Metrics 1–4)")
    axes[0].set_title("Cases Ranked by Structural Entanglement Regime", fontweight="bold")
    # Custom legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=col, label=name.replace("_", " "))
        for name, col in regime_colors.items()
    ]
    axes[0].legend(handles=handles, loc="lower right", frameon=True)

    # Right: Normalized component metrics heatmap per case
    metric_cols = ["Metric_1", "Metric_2", "Metric_3", "Metric_4"]
    comp_df = df.set_index(pd.Index(labels))[metric_cols]
    comp_df.columns = ["M1: B(G)\nBranch/Join", "M2: R(G)\nCycle Rank", "M3: D(G)\nDensity", "M4: M(G)\nMulti-Obj"]
    im = axes[1].imshow(comp_df.to_numpy(dtype=float), cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(comp_df.columns, fontsize=8.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_title("Normalized Entanglement Components per Case", fontweight="bold")
    axes[1].grid(False)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return _save_fig(fig, plots_dir / "07_entanglement_regimes.png")


def plot_collinearity_drops(diag_dir: Path, plots_dir: Path, top_k: int = 20) -> Path | None:
    """Plot the most frequently dropped collinear features across folds."""
    cdp_path = diag_dir / "collinear_dropped_pairs.csv"
    if not cdp_path.exists():
        return None
    df = pd.read_csv(cdp_path)
    if df.empty:
        return None

    summary = (
        df.groupby(["dropped_feature", "correlated_with"], as_index=False)
        .agg(Folds_Dropped=("Fold_ID", "nunique"), Mean_Abs_R=("abs_pearson_r", "mean"))
        .sort_values(["Folds_Dropped", "Mean_Abs_R"], ascending=[False, False])
        .head(top_k)
        .iloc[::-1]
    )
    pair_labels = [
        f"{d}  ↔  {k}" for d, k in zip(summary["dropped_feature"], summary["correlated_with"])
    ]

    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.34 * len(summary))))
    y_pos = np.arange(len(summary))
    ax.barh(y_pos, summary["Folds_Dropped"], color="#8c564b", alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_labels, fontsize=8)
    ax.set_xlabel("Number of Folds Dropped (|r| > 0.85 on Sub-Train)")
    ax.set_title("Most Frequent Collinear Feature Drops (Dropped ↔ Kept Partner)", fontweight="bold")
    ax.set_xlim(0, max(5.8, float(summary["Folds_Dropped"].max()) + 1.0))

    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(
            float(row["Folds_Dropped"]) + 0.08,
            i,
            f"|r|={row['Mean_Abs_R']:.3f}",
            va="center",
            fontsize=8,
            color="#333333",
        )

    fig.tight_layout()
    return _save_fig(fig, plots_dir / "08_collinearity_drops.png")


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    """Render a pandas DataFrame into a clean styled HTML table."""
    clipped = df.head(max_rows).copy()
    for col in clipped.select_dtypes(include=[np.number]).columns:
        if pd.api.types.is_float_dtype(clipped[col]):
            clipped[col] = clipped[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    return clipped.to_html(index=False, classes="diag-table", border=0, escape=True)


def build_html_report(
    config: dict[str, Any],
    results_dir: Path,
    diag_dir: Path,
    prep_dir: Path,
    plot_paths: list[tuple[str, Path]],
) -> Path:
    """Compile all generated plots and key summary tables into ``report.html``."""
    unit = str(config.get("target", {}).get("unit", "minutes"))
    debug_flag = bool(config.get("debug_mode", False))

    sections: list[str] = []

    # Summary tables
    m1_path = results_dir / "matrix1_global_mae.csv"
    if m1_path.exists():
        sections.append(
            f"<section><h2>Part 1 — Global 3×3 MAE Matrix ({html.escape(unit)})</h2>"
            f"{_df_to_html_table(pd.read_csv(m1_path))}</section>"
        )

    stab_path = diag_dir / "shap_feature_stability_summary.csv"
    if stab_path.exists():
        sections.append(
            "<section><h2>Top 20 Selected Tier 3 Features (Cross-Fold Validation SHAP)</h2>"
            f"{_df_to_html_table(pd.read_csv(stab_path), max_rows=20)}</section>"
        )

    raw_ent_path = prep_dir / "entanglement_raw_diagnostics.csv"
    if raw_ent_path.exists():
        cols_keep = [
            c
            for c in [
                "Case_ID",
                "n_nodes",
                "n_directed_edges",
                "branching_join_per_node",
                "cycle_rank_per_node",
                "directed_density",
                "multi_object_frequency",
                "Avg_Entanglement_Score",
            ]
            if c in pd.read_csv(raw_ent_path, nrows=1).columns
        ]
        sections.append(
            "<section><h2>Case Structural Complexity Diagnostics (Raw B, R, D, M + Score)</h2>"
            f"{_df_to_html_table(pd.read_csv(raw_ent_path)[cols_keep], max_rows=30)}</section>"
        )

    # Embedded relative plot images
    plot_blocks: list[str] = []
    for title, ppath in plot_paths:
        rel = ppath.relative_to(diag_dir).as_posix()
        plot_blocks.append(
            f"<div class='plot-card'><h3>{html.escape(title)}</h3>"
            f"<img src='{html.escape(rel)}' alt='{html.escape(title)}'/></div>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>OCPM Benchmark — Diagnostic Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px auto; max-width: 1280px; color: #1f2937; background: #f9fafb; padding: 0 20px; }}
    header {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 12px; margin-bottom: 24px; }}
    h1 {{ margin: 0 0 6px 0; font-size: 1.75rem; }}
    .meta {{ color: #6b7280; font-size: 0.92rem; }}
    section {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px 20px; margin-bottom: 22px; overflow-x: auto; }}
    h2 {{ margin-top: 0; font-size: 1.2rem; color: #111827; }}
    .plot-card {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px 20px; margin-bottom: 22px; }}
    .plot-card h3 {{ margin: 0 0 12px 0; font-size: 1.05rem; }}
    .plot-card img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
    table.diag-table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
    table.diag-table th, table.diag-table td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 10px; text-align: left; white-space: nowrap; }}
    table.diag-table th {{ background: #f3f4f6; font-weight: 600; }}
    table.diag-table tr:hover {{ background: #f9fafb; }}
  </style>
</head>
<body>
  <header>
    <h1>OCPM Predictive Process Monitoring — Diagnostic Report</h1>
    <div class="meta">Target Unit: <b>{html.escape(unit)}</b> &nbsp;|&nbsp; Case Notion: <b>{html.escape(str(config.get("ocpa", {}).get("leading_type")))}</b> &nbsp;|&nbsp; CV Group: <b>{html.escape(str(config.get("ocpa", {}).get("cv_group_object_type")))}</b> &nbsp;|&nbsp; Config debug_mode: <b>{debug_flag}</b></div>
  </header>
  {"".join(plot_blocks)}
  {"".join(sections)}
</body>
</html>
"""
    report_path = diag_dir / "report.html"
    report_path.write_text(html_doc, encoding="utf-8")
    LOGGER.info("Saved HTML diagnostic dashboard to %s", report_path)
    return report_path


def generate_all_diagnostic_plots(
    config_path: str | Path = "config.yaml",
) -> dict[str, Path]:
    """Generate all diagnostic PNG charts and the HTML summary report.

    Args:
        config_path: Path to ``config.yaml``.

    Returns:
        Dictionary mapping artifact keys to saved ``Path`` objects.
    """
    config = load_config(config_path)
    results_dir = Path(config["paths"]["results_dir"])
    prep_dir = Path(config["paths"]["preprocessed_dir"])
    diag_dir = Path(config["paths"].get("diagnostics_dir", results_dir / "diagnostics"))
    plots_dir = diag_dir / "plots"
    unit = str(config.get("target", {}).get("unit", "minutes"))

    generators: list[tuple[str, str, Path | None]] = [
        ("global_mae", "Part 1: Global 3x3 MAE Matrix", plot_global_mae_matrix(results_dir, plots_dir, unit)),
        ("regime_mae", "Part 2: Entanglement Regime 3x3 Matrices per Tier", plot_regime_mae_by_tier(results_dir, plots_dir, unit)),
        ("shap_top", "Top Tier 3 Features (Validation SHAP Importance)", plot_shap_top_features(diag_dir, plots_dir)),
        ("shap_rfe", "SHAP-RFE Validation MAE Curves per Fold", plot_shap_rfe_curves(diag_dir, plots_dir, unit)),
        ("training_curves", "LSTM & GNN Learning Curves (Train vs. Sub-Val)", plot_training_curves(diag_dir, plots_dir, unit)),
        ("fold_case_mae", "Per-Fold & Per-Case Holdout MAE (Tier 3)", plot_fold_and_case_mae(diag_dir, plots_dir, unit)),
        ("entanglement", "Structural Entanglement Regimes & Metrics", plot_entanglement_regimes(diag_dir, plots_dir)),
        ("collinearity", "Collinear Feature Drops on Sub-Train", plot_collinearity_drops(diag_dir, plots_dir)),
    ]

    outputs: dict[str, Path] = {}
    rendered_for_html: list[tuple[str, Path]] = []
    for key, title, path in generators:
        if path is not None:
            outputs[key] = path
            rendered_for_html.append((title, path))

    report_path = build_html_report(
        config=config,
        results_dir=results_dir,
        diag_dir=diag_dir,
        prep_dir=prep_dir,
        plot_paths=rendered_for_html,
    )
    outputs["report_html"] = report_path
    LOGGER.info("Generated %d diagnostic plots + HTML report in %s", len(rendered_for_html), diag_dir)
    return outputs


def main() -> None:
    """CLI entry point for post-hoc diagnostic visualization."""
    parser = argparse.ArgumentParser(description="Generate Diagnostic Plots & HTML Report")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    configure_logging()
    generate_all_diagnostic_plots(config_path=args.config)


if __name__ == "__main__":
    main()

