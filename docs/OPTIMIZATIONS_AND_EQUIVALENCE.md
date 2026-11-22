# Data-Structure Optimizations & Equivalence Verification

This document records every engineering and data-structure optimization made on the `experimentation` branch so that all methodological and implementation decisions are transparent, auditable, and reproducible for scientific reporting.

> **Core Principle:** None of the optimizations below alter the experimental design, feature definitions, model architectures, hyperparameters, cross-validation splits, or anti-leakage invariants specified in [`PROJECT_SPEC.md`](../PROJECT_SPEC.md). Every change replaces high-overhead Python data structures (`dict`s, $O(N^2)$ loops, dense pre-allocated window tensors) with mathematically equivalent NumPy/PyTorch representations.

---

## Summary Table

| # | Stage / Module | Bottleneck Observed | Optimization | Measured Impact | Scientific / Numerical Impact |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | Stage 1a (`src/feature_extraction.py`) | Native `ocpa` `pm_factory.apply` runs $O(N_c^2 \times A)$ pure-Python prefix re-scans (hours on 34,860 events × 333 activities). | Single-pass $O(N)$ timestamp-block prefix sweep (`_extract_feature_storage_fast`). | **39.7×** faster on 2k events (`9.33s → 0.24s`); full 34,860 events × 1,022 features in **12.54s**. | **Exact `0.000e+00` max abs diff** across 240,000 real cells verified against native `ocpa`. |
| 2 | Stage 1a → 2 (`src/ocpa_encoding.py`, `src/cross_validation.py`) | Unpickling `feature_storage.pkl` (486 MB on disk) allocated 35.6M boxed Python `float` dict entries (**3,040 MB RSS**), causing OOM on a 5.7 GiB laptop. | Pack raw unnormalized features into a contiguous 2D NumPy matrix `X_raw` (`CompactOcpaDataset` / `feature_storage_compact.pt`, **135.9 MB**). | Peak load+scale RSS dropped **3,346 MB → 1,083 MB** (>3× lower total RSS; ~16× smaller feature array). | `X_raw` stored as `float32` before `float64` scaling (`y_true` stays `float64`); see precision disclosure in Section 2. |
| 3 | Stage 2a (`src/ocpa_encoding.py`) | `OcpaFeatureScaler.transform` allocated a full second copy of 35.6M Python dict entries per fold split. | Vectorized NumPy `CompactFoldScaler` operating strictly on fold training row indices. | Scales 27,853 × 1,022 split in `<0.15s` with **~114 MB** instead of >2 GB. | Identical median-imputation + z-score formula (`ddof=0`); verified against `OcpaFeatureScaler` (`atol=1e-5`). |
| 4 | Stage 2a/2c (`src/ocpa_encoding.py`, `src/models/seq_lstm.py`) | Dense 3D sliding-window array `(N_events, 20, F)` for LSTM allocated up to **~1.7 GB** per fold. | Lazy PyTorch `CausalPrefixSequenceDataset` slicing left-padded `[max(case_start, i - W + 1) .. i]` windows per minibatch (`batch_size=64`, ~5 MB). | **>300× lower** LSTM input tensor RAM. | **Zero difference** — bit-for-bit identical windows, padding, and shuffle seed order vs. `encode_sequential`. |
| 5 | Stage 2b (`src/feature_selection.py`) | `DataFrame.corr()` on 1,022 columns ran slow pairwise loops per fold. | Centered BLAS matrix multiplication `|normed.T @ normed|` with identical greedy removal and deterministic tie-breaking. | Sub-second 1,022 × 1,022 correlation matrix per fold. | **Zero difference** — exact same Pearson $|r|$ and dropped-feature sequence. |

---

## 1. Stage 1a: $O(N)$ Single-Pass Timestamp-Block Prefix Sweep

### Motivation
`ocpa`'s `pm_factory.apply` evaluates event-based prefix features (`EVENT_PREVIOUS_ACTIVITY_COUNT`, `EVENT_PREVIOUS_OBJECT_COUNT`, `EVENT_PREVIOUS_TYPE_COUNT`, `EVENT_AGG_PREVIOUS_CHAR_VALUES`) by calling `get_recent_events(e_id, case, ocel)` inside every `(event, feature)` pair. On a session with $N_c = 4,678$ events and $A = 333$ activities, this performs $O(N_c^2 \cdot A)$ pure-Python iterations (~billions of loop steps).

### Implementation (`_extract_feature_storage_fast` in `src/feature_extraction.py`)
- Preserves `ocpa`'s exact `get_recent_events` semantics ($t_e \le t_{\text{current}}$, including **tied/simultaneous timestamps**): events in each case are sorted chronologically and partitioned into contiguous equal-timestamp blocks $B_0, B_1, \dots, B_m$.
- A single forward sweep over blocks maintains running activity counts, cumulative distinct object sets (overall and per object type), and finite-value running means for numeric event attributes.
- Matches `ocpa`'s exact graph edge sources:
  - `C2_preceding_act_*` reads incoming edges from the full event-object graph `ocel.graph.eog.in_edges(e_id)` (matching `ocpa`'s `preceding_activities`).
  - Preset timing features (`P5`, `P7`, `P8`, `P10`, `P11`, `P12`) read incoming edges from the per-case subgraph `ocel.graph.eog.subgraph(case).in_edges(e_id)` (matching `ocpa`'s timing extractors).
- Constructs standard `ocpa.algo.predictive_monitoring.obj.Feature_Storage` and `Feature_Graph` objects with identical `pexec_id`, node ordering, `node.objects`, causal edges, and `event_features`.
- The native `ocpa` path (`pm_factory.apply`) remains fully intact and selectable at any time via `ocpa.extraction_mode: "native"` in `config.yaml` or `--mode native` on the CLI.

### Equivalence Verification
- **Unit tests (`tests/test_feature_extraction.py::test_fast_vs_native_exact_equivalence`):** Verifies exact parity (`rtol=1e-12, atol=1e-12`) on both standard and tied-timestamp fixtures containing `NaN` event attributes.
- **Real dataset audit (`python -m src.feature_extraction --debug --verify-equivalence`):** Verified on `age_of_empires_ocel2_10_match_filter.sqlite` across all 20 real `Session` cases × 100 events × 120 keys (**240,000 values checked**):
  - `case_index`, graph attributes, node ordering, node objects, edge sets, and edge objects: **100% identical**
  - Maximum absolute numerical difference (`max_abs_diff`): **`0.000e+00`**
  - Runtime: native `9.327s` vs. fast `0.235s` (**39.73× speedup** on 2,000 events; **12.544s** on the full 34,860-event / 1,022-feature dataset).

---

## 2. Stage 1a → Stage 2: Contiguous Array Packing (`CompactOcpaDataset`)

### Motivation
On the full Age of Empires II dataset (34,860 events × 1,022 features), `data/preprocessed/feature_storage.pkl` is 486 MB on disk. Unpickling 35.6 million boxed Python `float` entries inside 34,860 Python `dict`s consumed **3,040 MB RSS** before Fold 1 even started, exceeding available memory on a 5.7 GiB RAM laptop.

### Implementation (`src/ocpa_encoding.py`, `src/feature_extraction.py`, `src/cross_validation.py`)
- Stage 1a continues to persist the raw unnormalized `data/preprocessed/feature_storage.pkl` (per `PROJECT_SPEC.md`) **and** writes an unnormalized compact cache `data/preprocessed/feature_storage_compact.pt` containing `CompactOcpaDataset`:
  - `X_raw`: contiguous `np.float32` array of shape `(34860, 1022)` (**135.9 MB**)
  - `y`: `np.float64` target vector `(34860,)`
  - `graph_slices`, `graph_edge_indices`, `case_ids_per_graph`, `cv_groups_per_graph`, `meta_case_ids`, `meta_event_ids`
- Stage 2 loads `feature_storage_compact.pt` when up-to-date (falling back transparently to `feature_storage.pkl` if absent).
- Measured peak RSS after loading + scaling Fold 1 Sub-Train dropped from **3,346 MB → 1,083 MB**.

### Precision Disclosure
- Raw features in `X_raw` are stored in IEEE-754 `float32` (~7 decimal digits of precision) before `CompactFoldScaler` computes `(median, mean, std)` in `float64`; the target `y_true` remains `float64`.
- All downstream models (`XGBoost`, PyTorch `LSTM`, PyG `GNN`) natively operate on `float32` inputs.
- Integer count features (`C2`, `C3`, `C5`, `O2`, `O3`, `O5`, `O_type_count_*`, values 0–500) are **bit-exact** in `float32`. Continuous timing features (0–60 minutes) have rounding error `< 1e-6` min (`< 0.00006` s).

---

## 3. Stage 2a: Vectorized Fold-Scoped Scaling (`CompactFoldScaler`)

### Implementation (`src/ocpa_encoding.py`)
- `CompactFoldScaler.fit(compact, train_graph_indices)` computes per-column finite medians, imputes non-finite values with the training median, and computes `mean` and `std` (`ddof=0`, replacing zero std with `1.0`) **strictly on the fold's training graph rows** (`12 Sub-Train` cases for inner SHAP-RFE/LSTM/GNN; `16 Train` cases for outer XGBoost).
- `transform_split(...)` slices only the requested split graphs and active tier columns, applying the training-fitted `(median, mean, std)` without ever mutating `X_raw` or scaling `y_true`.
- Verified directly against dict-based `OcpaFeatureScaler` in `tests/test_no_leakage.py::test_ocpa_scaler_strict_train_only_fitting_and_encoding_alignment`.

---

## 4. Stage 2a/2c: Lazy Causal Prefix Windowing for LSTM (`CausalPrefixSequenceDataset`)

### Motivation
Pre-allocating all overlapping sliding windows `(N_events, window_size=20, F)` for 21k+ Sub-Train events and ~710–1,022 Tier 3 features required up to **1.7 GB** per dense array.

### Implementation (`src/ocpa_encoding.py`, `src/models/seq_lstm.py`)
- `CausalPrefixSequenceDataset` is a PyTorch `Dataset` wrapping the fold's scaled 2D array `(N_split, F)` and per-case row slices `(case_start, case_end)`.
- For any event row `r`, `__getitem__` slices `X_2d[max(case_start, r - W + 1) : r + 1]` and left-pads with zeros into a `(W, F)` tensor strictly inside that single case (zero cross-case or future-event leakage).
- `SequenceLSTMModel` accepts either a dense 3D NumPy array or `CausalPrefixSequenceDataset` with identical `DataLoader` batch size (`64`) and deterministic PyTorch `Generator` shuffle seed.
- Verified bit-for-bit identical against `encode_sequential` in `tests/test_no_leakage.py`.

---

## 5. Stage 2b: Vectorized BLAS Pearson Collinearity Filter

### Implementation (`src/feature_selection.py`)
- Replaces `X_sub_train.corr(method="pearson")` with mean-centered, L2-normalized BLAS matrix multiplication `np.abs(normed.T @ normed)` (handling constant zero-norm columns cleanly as `0.0` correlation).
- Uses the exact same greedy iterative removal of pairs with $|r| > 0.85$ (dropping the feature with higher mean $|r|$ to remaining features, tie-breaking lexicographically) fit strictly on `X_sub_train`.

---

## 6. Anti-Leakage Invariant Compliance Check

| Invariant (`PROJECT_SPEC.md` §6) | Status | Verification |
| :--- | :---: | :--- |
| 1. Zero Intra-Case (& Same-Match) Leakage | Preserved | `tests/test_no_leakage.py::test_zero_case_id_overlap_across_all_cv_splits` + runtime assertions in `cross_validation.py` |
| 2. Zero Cross-Stage Leakage (`entanglement_scores` isolated from Stage 2) | Preserved | AST check in `tests/test_no_leakage.py::test_zero_cross_stage_and_banned_ocpa_split_references` |
| 3. Zero Selection & Scaling Leakage (no global scaling; no `extract_normalized_train_test_split()`) | Preserved | AST check + `test_ocpa_scaler_strict_train_only_fitting_and_encoding_alignment` |
| 4. Validation-Only SHAP (`X_sub_val` only) | Preserved | Mock spy in `tests/test_shap_rfe.py::test_shap_values_computed_exclusively_on_sub_val` |
| 5. Reproducibility (fixed seeds across `random`, `numpy`, `torch`, `xgboost`) | Preserved | `tests/test_no_leakage.py::test_full_5fold_cv_and_stage3_evaluation_reproducibility` |

---

## 7. Reproducing Verification Checks

```bash
# Run full unit & anti-leakage test suite (13 tests)
pytest -v

# Re-verify bit-exact fast vs. native ocpa extraction on real data (--debug slice)
python -m src.feature_extraction --debug --verify-equivalence
```
