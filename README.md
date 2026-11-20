# OCPM Predictive Process Monitoring Benchmark

An end-to-end, leak-free experimental benchmark for **Object-Centric Predictive Process Monitoring (OCPM)**, targeting event-level **Remaining Time Prediction ($P3$)** on Object-Centric Event Logs (OCEL).

Built natively around [`ocpa`](https://github.com/ocpm/ocpa) for process execution graph extraction and feature encoding, the pipeline evaluates **XGBoost**, **PyTorch Prefix LSTM**, and **PyTorch Geometric (PyG) Causal GNN** across three progressively richer topological feature tiers using **5-Fold Case-Level Cross-Validation** to produce unified Out-of-Fold (OOF) predictions across all 20 cases.

---

## Research Objectives

From a single leak-free 5-fold cross-validation pass (`test_predictions.parquet`), the benchmark answers two decoupled research questions:

1. **Part 1 — Feature Representation vs. Model Architecture (Global $3 \times 3$ Matrix):**
   Evaluates how progressive graph-context feature engineering (**Tiers 1–3**) impacts tabular (**XGBoost**), sequential (**LSTM**), and graph (**GNN**) architectures across all cases (`matrix1_global_mae.csv`).
2. **Part 2 — Structural Entanglement Breakdown (Conditional $3 \times 9$ Matrix):**
   Slices the exact same OOF predictions into balanced **Low / Medium / High Structural Entanglement Regimes** (`pd.qcut(..., q=3)`) to identify the complexity regimes where explicit graph-context tabular features match or fall behind native GNN message passing (`matrix2_regime_mae.csv`).

---

## Repository Structure

```text
.
├── PROJECT_SPEC.md                 # Architectural specification & anti-leakage invariants
├── config.yaml                     # Feature tier definitions, paths, debug_mode, & hyperparameters
├── requirements.txt                # Pinned Python dependencies
├── data/
│   ├── raw/                        # Input OCEL file (.jsonocel / .xmlocel / .sqlite / .csv) — gitignored
│   ├── preprocessed/               # Stage 1 outputs (feature_storage.pkl, case_index.parquet, entanglement_scores.parquet) — gitignored
│   └── results/                    # Stage 2 OOF predictions & Stage 3 evaluation matrices — gitignored
├── src/
│   ├── feature_extraction.py       # Stage 1a: OCEL parsing & native ocpa Feature_Storage extraction
│   ├── entanglement_metrics.py     # Stage 1b: 4 modular case-level structural entanglement metrics
│   ├── ocpa_encoding.py            # Stage 2a: Fold-scoped Feature_Storage slicing, scaling, & tabular/seq/PyG encoding
│   ├── feature_selection.py        # Stage 2b: Sub-Train collinearity filter (|r| > 0.85) + validation-only SHAP-RFE
│   ├── models/
│   │   ├── base.py                 # Common model interface (fit, predict)
│   │   ├── tabular_xgb.py          # XGBRegressor wrapper
│   │   ├── seq_lstm.py             # PyTorch causal prefix/sliding-window LSTM
│   │   └── graph_gnn.py            # PyTorch Geometric (PyG) causal SAGEConv GNN
│   ├── cross_validation.py         # Stage 2c: 5-Fold KFold orchestrator over Feature_Graphs (16/4 outer, 12/4 inner)
│   └── evaluation.py               # Stage 3: Event-Level & Case-Normalized MAE (Matrix 1 & Matrix 2)
└── tests/
    ├── test_feature_extraction.py  # Stage 1 schema, Feature_Storage alignment, NaNs, & debug capping
    ├── test_no_leakage.py          # Verifies zero Case_ID overlap, fold-scoped scaling, & Stage 2 isolation
    └── test_shap_rfe.py            # Verifies out-of-sample SHAP evaluation & 2-step degradation stopping
```

---

## Feature Tiers

| Tier | Scope | Features |
| :--- | :--- | :--- |
| **Tier 1** | **0-Hop Node-Local** | Current Activity ($C5$), Current Event Attribute/Amount ($D3$), Event Service Time ($P9$), Elapsed Time ($P2$). |
| **Tier 2** | **Adams et al. (2022) Baseline** | Preceding Activities ($C2$, 1-hop preset), Average Previous Attribute/Amount ($D1$), Elapsed Time ($P2$), Previous Target-Object Count ($O3$). |
| **Tier 3** | **Full OCEL Graph Context + Nested SHAP-RFE** | All Tier 1 & Tier 2 features plus Synchronization Time ($P5$), Pooling Time ($P7$), Lagging Time ($P8$), Waiting Time ($P10$), Flow/Sojourn Time ($P11$/$P12$), Object Counts ($O2$/$O5$), Previous Activity Counts ($C3$), and remaining `ocpa` prefix features — pruned per fold via nested collinearity filtering + validation-only SHAP-RFE. |

---

## Anti-Leakage Guarantees

1. **Zero Intra-Case Leakage:** Cross-validation splits strictly by `Feature_Graph` / `Case_ID` (16 Train / 4 Test outer; 12 Sub-Train / 4 Sub-Val inner).
2. **Zero Cross-Stage Leakage:** `entanglement_scores.parquet` is strictly isolated from Stage 2 (`ocpa_encoding.py`, `feature_selection.py`, `cross_validation.py`) and read only in Stage 3 (`evaluation.py`).
3. **Zero Scaling or Selection Leakage:** Stage 1 persists raw unnormalized `Feature_Storage`. All normalization (`OcpaFeatureScaler`), collinearity filtering, and SHAP-RFE are fit exclusively inside each fold on training splits (never calling `ocpa`'s `extract_normalized_train_test_split()`).
4. **Validation-Only SHAP:** `shap.TreeExplainer` importances in SHAP-RFE are evaluated strictly on `X_sub_val`, never on `X_sub_train`.
5. **Reproducibility:** Global random seeds (`random`, `numpy`, `torch`, `xgboost`) are pinned from `config.yaml`.

---

## Setup

Python **3.10** is recommended for full compatibility with `ocpa==1.3.3` and `pm4py==2.2.32`.

```bash
# Create and activate a local Python 3.10 environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

*(If using Conda: `conda create --prefix ./.venv python=3.10 pip -y && ./.venv/bin/pip install -r requirements.txt`.)*

---

## Usage

Place your raw OCEL dataset (`.jsonocel`, `.xmlocel`, `.sqlite`, or `.csv`) inside `data/raw/` (all data files are ignored by `.gitignore`), and configure paths or hyperparameters in `config.yaml`.

### Run Full Pipeline

```bash
# Stage 1a: Extract unnormalized ocpa Feature_Storage & case index
python -m src.feature_extraction

# Stage 1b: Compute 4 case-level structural entanglement metrics
python -m src.entanglement_metrics

# Stage 2: 5-Fold Case-Level CV (9 model-tier pairs + nested Tier 3 SHAP-RFE)
python -m src.cross_validation

# Stage 3: Compute Global 3x3 & Regime 3x9 MAE matrices (Event-Level & Case-Normalized)
python -m src.evaluation
```

### Fast Smoke Test (`debug_mode`)

Set `debug_mode: true` in `config.yaml` or pass `--debug` to cap cases/events and run lightweight models (`XGBRegressor(n_estimators=10)`, `epochs=2` for LSTM/GNN):

```bash
python -m src.feature_extraction --debug
python -m src.entanglement_metrics
python -m src.cross_validation --debug
python -m src.evaluation
```

### Customizing Entanglement Metrics

Replace the 4 modular placeholder functions (`compute_metric_1`–`compute_metric_4`) in `src/entanglement_metrics.py` with your exact structural formulas; normalization and `Avg_Entanglement_Score` export are handled automatically.

---

## Running Tests

```bash
pytest -v
```

All tests verify Stage 1 `Feature_Storage` schema/alignment, fold-scoped normalization without leakage, validation-only SHAP-RFE stopping behavior, Stage 2 isolation from entanglement metrics, and Stage 3 evaluation matrix computation.
