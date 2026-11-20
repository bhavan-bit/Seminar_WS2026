# OCPM Predictive Process Monitoring Benchmark — Technical Specification

## 1. Project Overview & Research Objectives

This repository implements an end-to-end, leak-free experimental pipeline for **Object-Centric Predictive Process Monitoring (OCPM)**, targeting **Remaining Time Prediction ($P3$)** at the event level on an Object-Centric Event Log (OCEL).

The dataset consists of **20 macro process executions (cases)**, each containing a large number of events, activities, and interacting object instances. Because the case count is small ($N = 20$) while the event graph per case is dense, the pipeline uses **5-Fold Case-Level Cross-Validation** to produce **Out-of-Fold (OOF) predictions** across all 20 cases in a single training pass.

The pipeline answers two decoupled research questions from the same unified OOF prediction table:

1. **Part 1 — Feature Representation vs. Model Architecture (Global $3 \times 3$ Matrix):**
   Evaluates how progressive graph-context feature engineering (Tiers 1–3) affects tabular (**XGBoost**), sequential (**LSTM**), and graph (**GNN**) models globally across all 20 cases.
2. **Part 2 — Structural Entanglement Breakdown (Conditional $3 \times 9$ Matrix):**
   Slices the exact same OOF predictions by ground-truth case complexity (**Low / Medium / High Entanglement Regimes**) to identify the structural boundary conditions under which tabular graph features match or fall behind native GNN message passing.

---

## 2. Repository Structure

```text
ocpm-ppm-benchmark/
├── PROJECT_SPEC.md                 # This architectural specification
├── config.yaml                     # Feature tier definitions, paths, debug_mode, & hyperparameters
├── requirements.txt                # Pinned Python dependencies
├── data/
│   ├── raw/                        # Input OCEL file (.jsonocel / .xmlocel / .sqlite)
│   ├── preprocessed/               # Stage 1 outputs (feature_storage.pkl, case_index.parquet, entanglement_scores.parquet)
│   └── results/                    # Stage 2 OOF predictions & Stage 3 evaluation matrices
├── src/
│   ├── feature_extraction.py       # Stage 1a: OCEL parsing & native ocpa feature extraction
│   ├── entanglement_metrics.py     # Stage 1b: 4 case-level structural entanglement metrics
│   ├── ocpa_encoding.py            # Stage 2a: Fold-scoped Feature_Storage slicing, scaling, & tabular/seq/graph encoding
│   ├── feature_selection.py        # Stage 2b: Collinearity filter + validation SHAP-RFE
│   ├── models/
│   │   ├── base.py                 # Common model interface (fit, predict)
│   │   ├── tabular_xgb.py          # XGBRegressor wrapper
│   │   ├── seq_lstm.py             # PyTorch prefix/sliding-window LSTM
│   │   └── graph_gnn.py            # PyTorch Geometric (PyG) GNN model
│   ├── cross_validation.py         # Stage 2c: 5-Fold GroupKFold orchestrator
│   └── evaluation.py               # Stage 3: Matrix 1 (3x3) & Matrix 2 (3x9) analytics
└── tests/
    ├── test_feature_extraction.py  # Unit tests for extraction schema, graph alignment, & NaNs
    ├── test_no_leakage.py          # Verifies Case_ID disjointness & fold-scoped fitting
    └── test_shap_rfe.py            # Verifies out-of-sample SHAP & stopping criterion
```

---

## 3. End-to-End Pipeline Architecture

```text
[Raw OCEL Dataset (20 Cases)]
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Preprocessing & Feature Extraction (Run Once)                  │
│ • Parse OCEL & extract all raw features (Tiers 1, 2, 3) + target via ocpa│
│ • Save unnormalized Feature_Storage (20 Feature_Graph objects)          │
│ • Compute 4 case-level Entanglement Metrics (stored separately)         │
│ • NO global scaling, NO global encoding, NO feature selection           │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: 5-Fold Case-Level Cross-Validation & Training (Run Once)       │
│ • Split 20 Feature_Graphs across 5 folds (16 Train graphs, 4 Test)      │
│   ├──► Slice Feature_Storage by fold; fit scaling strictly on Train     │
│   ├──► Inner Split: 12 Sub-Train graphs / 4 Sub-Val graphs              │
│   ├──► Run SHAP-RFE on (Sub-Train, Sub-Val) ──► Tier 3 selected keys    │
│   ├──► Encode via ocpa (tabular / sequential / PyG graph) per tier      │
│   ├──► Train XGBoost, LSTM, and GNN across Tier 1, Tier 2, Tier 3       │
│   └──► Predict on 4 Holdout Test graphs ──► Append to OOF results       │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Post-Hoc Evaluation & Regime Analysis                          │
│ • Load combined OOF predictions across all 20 cases                     │
│   ├──► Part 1: Global MAE (Event-Level & Case-Normalized) ──► 3x3 Table │
│   └──► Part 2: Merge Entanglement Scores + pd.qcut(q=3)  ──► 3x9 Table  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Feature Tier Definitions (Topological Receptive Field)

For each process execution graph $G_p = (E', K)$ and target event node $e \in E'$ with incoming preset $\bullet e = \{u \in E' \mid (u, e) \in K\}$ and historical prefix $E'_{<t}$, all features are extracted natively via `ocpa`:

| Tier | Formal Scope | Included Features | Experimental Role |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **0-Hop Node-Local** ($\mathcal{R}_0(e) = \{e\}$) | Current Activity ($C5$), Current Event Attribute/Amount ($D3$), Event Service Time ($P9$), Elapsed Time ($P2$). | Zero edge/neighbor context. Isolates whether sequence (LSTM) or graph (GNN) architectures are strictly required when tabular models are blind to topology. |
| **Tier 2** | **Adams et al. (2022) Baseline** | Preceding Activities ($C2$, 1-hop preset), Average Previous Attribute/Amount ($D1$), Elapsed Time ($P2$), Previous Target-Object Count ($O3$). | Replicates the 4-feature baseline from Section 6 of Adams et al. (2022). Isolates algorithm upgrade (e.g., XGBoost vs. OLS/GNN) under identical feature constraints. |
| **Tier 3** | **Full OCEL Graph Context + SHAP-RFE** | All Tier 2 features + Synchronization Time ($P5$), Pooling Time ($P7$), Waiting Time ($P10$), Event Object Count ($O5$, $|O_e|$), Previous Object Count ($O2$), Previous Activity Counts ($C3$), and remaining `ocpa` prefix/graph features — **pruned per fold via nested SHAP-RFE**. | Tests whether explicit graph-summary features with automated collinearity/noise removal allow tabular models to match or outperform GNNs. |

---

## 5. Module Specifications & Data Contracts

### Stage 1a: `src/feature_extraction.py`
* **Input:** Raw OCEL file in `data/raw/`.
* **Logic:**
  1. Import the OCEL using `ocpa.objects.log.importer` according to `config.yaml`.
  2. Extract the full `Feature_Storage` object once via `ocpa.algo.predictive_monitoring.factory.apply(...)` containing all Tier 1, Tier 2, and Tier 3 features plus the target label (`EVENT_REMAINING_TIME`).
  3. **Do NOT normalize or split inside Stage 1.** Keep the raw `Feature_Storage` (20 `Feature_Graph` objects) intact so fold-level scaling happens strictly inside Stage 2.
* **Output:**
  * `data/preprocessed/feature_storage.pkl`: Raw `ocpa` `Feature_Storage` containing all 20 `Feature_Graph` objects.
  * `data/preprocessed/case_index.parquet`: Metadata mapping each `Feature_Graph` index to `Case_ID` and event count.

### Stage 1b: `src/entanglement_metrics.py`
* **Input:** Execution graphs / `Feature_Storage` from Stage 1a.
* **Logic:**
  1. Compute the 4 chosen structural entanglement metrics per `Case_ID` (normalized to a comparable scale across the 20 cases before averaging).
  2. Compute `Avg_Entanglement_Score` = mean of the 4 normalized metrics.
  3. **Strict Isolation:** This file MUST NOT be read during Stage 2.
* **Output:** `data/preprocessed/entanglement_scores.parquet` with columns `[Case_ID, Metric_1, Metric_2, Metric_3, Metric_4, Avg_Entanglement_Score]`.

### Stage 2a: `src/ocpa_encoding.py`
* **Logic:**
  1. Slice `Feature_Storage` by `Feature_Graph` indices into fold-scoped `train_fs` / `test_fs` (and `sub_train_fs` / `sub_val_fs`).
  2. Fit normalization parameters (mean/std per numeric feature key) **strictly on the training `Feature_Storage`**, then apply to validation and test `Feature_Storage` objects. Never use `extract_normalized_train_test_split()` or normalize globally across all 20 cases.
  3. Filter active feature keys to the requested tier (`Tier 1`, `Tier 2`, or fold-selected `Tier 3` keys).
  4. Generate model-specific representations directly from `ocpa`:
     * **Tabular (XGBoost):** `ocpa.algo.predictive_monitoring.tabular.construct_table(fs, index_list=...)`
     * **Sequential (LSTM):** `ocpa.algo.predictive_monitoring.sequential.construct_sequence(fs, index_list=...)`
     * **Graph-Based (GNN):** Convert `fs.feature_graphs` (nodes, directed edges, and filtered node feature vectors) into PyTorch Geometric `Data` objects.

### Stage 2b: `src/feature_selection.py` (Nested Tier 3 SHAP-RFE)
* **Inputs:** `X_sub_train, y_sub_train, X_sub_val, y_sub_val` (derived from the tabular encoding of `sub_train_fs` and `sub_val_fs` in the current outer fold).
* **Algorithm:**
  1. **Collinearity Filter (Sub-Train only):** Compute Pearson correlation matrix on `X_sub_train`. For any feature pair with $|r| > 0.85$, drop one feature deterministically.
  2. **Out-of-Sample SHAP-RFE Loop:**
     * Train `XGBRegressor` on `(X_sub_train, y_sub_train)`.
     * Compute `shap.TreeExplainer(model).shap_values(X_sub_val)` **strictly on `X_sub_val`** (never on `X_sub_train`).
     * Compute `val_mae = mean_absolute_error(y_sub_val, model.predict(X_sub_val))`.
     * Rank features by mean absolute validation SHAP value.
     * Drop the bottom 10% of features (minimum 1 feature per iteration) and retrain from scratch on the remaining columns.
  3. **Stopping Criterion:** Track `best_val_mae` and `best_features`. Stop when `val_mae` degrades (increases) for **2 consecutive iterations** (or when minimum feature count is reached).
* **Output:** `best_features: list[str]` + logged feature selection trace for the fold.

### Stage 2c: `src/cross_validation.py`
* **Splitting Strategy:** `KFold(n_splits=5, shuffle=True, random_state=seed)` over the 20 `Feature_Graph` indices (16 Train graphs / 4 Test graphs per fold), with an inner split of 12 Sub-Train graphs / 4 Sub-Val graphs from the 16 Train graphs.
* **Model Training Matrix (9 Combinations per Fold):**
  * Models: `XGBoost` (`tabular_xgb.py`), `LSTM` (`seq_lstm.py`), `GNN` (`graph_gnn.py`).
  * Tiers: `Tier 1`, `Tier 2`, `Tier 3 (fold-selected keys)`.
* **Outputs:**
  * `data/results/test_predictions.parquet`: `[Case_ID, Event_ID, Fold_ID, y_true, Pred_XGB_T1, Pred_XGB_T2, Pred_XGB_T3, Pred_LSTM_T1, Pred_LSTM_T2, Pred_LSTM_T3, Pred_GNN_T1, Pred_GNN_T2, Pred_GNN_T3]`
  * `data/results/selected_features_per_fold.json`

### Stage 3: `src/evaluation.py`
* **Inputs:** `data/results/test_predictions.parquet` and `data/preprocessed/entanglement_scores.parquet`.
* **Metrics (Computed for BOTH matrices):**
  1. **Event-Level MAE:** Mean of $|y_{\text{true}} - \hat{y}|$ across all events in the group.
  2. **Case-Normalized MAE:** Compute MAE per `Case_ID` first, then take the unweighted mean across cases in the group.
* **Outputs:**
  1. **Part 1 — Global Baseline Matrix ($3 \times 3$):** `data/results/matrix1_global_mae.csv` (Rows = `[Tier 1, Tier 2, Tier 3]`, Columns = `[XGBoost, LSTM, GNN]`).
  2. **Part 2 — Entanglement Regime Matrix ($3 \times 9$):** Bucket the 20 cases via `pd.qcut(Avg_Entanglement_Score, q=3, labels=["Regime_1_Low", "Regime_2_Medium", "Regime_3_High"])`, left-join onto `test_predictions.parquet` on `Case_ID`, group by `Regime`, and save to `data/results/matrix2_regime_mae.csv`.

---

## 6. Non-Negotiable Engineering Invariants

1. **Zero Intra-Case Leakage:** No `Case_ID` / `Feature_Graph` may ever appear in both Train and Test (or Sub-Train and Sub-Val) within the same fold.
2. **Zero Cross-Stage Leakage:** `entanglement_scores.parquet` is never imported or referenced in `cross_validation.py`, `ocpa_encoding.py`, or `feature_selection.py`.
3. **Zero Selection & Scaling Leakage:** Collinearity filtering, feature normalization, and SHAP-RFE run strictly inside the cross-validation loop on training data only.
4. **Validation-Only SHAP:** `shap.TreeExplainer` values used to rank features in SHAP-RFE MUST be computed on `X_sub_val`, never `X_sub_train`.
5. **Reproducibility:** Fix all random seeds (`numpy`, `random`, `torch`, `xgboost`) from `config.yaml`.