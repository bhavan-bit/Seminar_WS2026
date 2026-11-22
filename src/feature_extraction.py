"""Stage 1a: OCEL parsing and native ``ocpa`` ``Feature_Storage`` extraction.

Imports an OCEL dataset via ``ocpa.objects.log.importer``, extracts all raw
Tier 1, Tier 2, and Tier 3 event features plus ``EVENT_REMAINING_TIME`` once
via ``ocpa.algo.predictive_monitoring.factory.apply(...)``, and persists the
unnormalized ``Feature_Storage`` alongside ``case_index.parquet``.

Anti-Leakage Guarantee:
    No global normalization, no global imputation, no train/test splitting,
    and no feature selection occur inside Stage 1. All fold-scoped scaling and
    encoding are deferred to Stage 2 (``src/ocpa_encoding.py``).

Outputs:
    - ``data/preprocessed/feature_storage.pkl``
    - ``data/preprocessed/case_index.parquet``
"""

from __future__ import annotations

import argparse
from collections import Counter
import logging
import pickle
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import ocpa.algo.predictive_monitoring.factory as pm_factory
import ocpa.algo.predictive_monitoring.tabular as pm_tabular
import ocpa.objects.log.converter.versions.df_to_ocel as df_to_ocel
import ocpa.objects.log.importer.csv.factory as csv_import
import ocpa.objects.log.importer.ocel.factory as ocel_import
import ocpa.objects.log.variants.util.table as table_utils
import pandas as pd
from ocpa.algo.predictive_monitoring.obj import Feature_Storage
from ocpa.objects.log.ocel import OCEL
from ocpa.objects.log.variants.graph import EventGraph
from ocpa.objects.log.variants.table import Table
from ocpa.util.constants import CONN_COMP, LEAD_TYPE

from src import configure_logging, load_config, resolve_tier_columns, set_global_seed

LOGGER = logging.getLogger(__name__)

TIME_UNIT_DIVISORS: dict[str, float] = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
}

SUPPORTED_RAW_EXTENSIONS: tuple[str, ...] = (
    ".jsonocel",
    ".xmlocel",
    ".sqlite",
    ".csv",
)


def _sanitize_name(raw_name: Any) -> str:
    """Convert an activity or object-type label into a clean feature key suffix."""
    text = str(raw_name).strip()
    sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
    return sanitized or "unknown"


def _safe_nanmean(values: list[Any]) -> float:
    """Compute a NaN-safe mean over prefix numeric values."""
    numeric: list[float] = []
    for val in values:
        try:
            fval = float(val)
            if np.isfinite(fval):
                numeric.append(fval)
        except (TypeError, ValueError):
            continue
    if not numeric:
        return float("nan")
    return float(np.mean(numeric))


def _safe_lagging_time(node: Any, ocel: OCEL, object_type: str) -> float:
    """Compute ``ocpa`` lagging time safely when an object type has an empty preset."""
    cases = ocel.process_execution_mappings.get(node.event_id, [node.pexec_id])
    res: list[float] = []
    for case in cases:
        subgraph = ocel.graph.eog.subgraph(ocel.process_executions[case])
        preset = [src for (src, _dst) in subgraph.in_edges(node.event_id)]
        if not preset:
            res.append(0.0)
            continue

        ot_maxs: list[pd.Timestamp] = []
        for ot in ocel.object_types:
            ot_ts = [
                ocel.get_value(e, "event_timestamp")
                for e in preset
                if ocel.get_value(e, ot)
            ]
            if ot_ts:
                ot_maxs.append(max(ot_ts))

        target_ts = [
            ocel.get_value(e, "event_timestamp")
            for e in preset
            if ocel.get_value(e, object_type)
        ]
        if not target_ts or not ot_maxs:
            res.append(0.0)
        else:
            duration = (max(target_ts) - min(ot_maxs)).total_seconds()
            res.append(float(max(duration, 0.0)))

    return float(np.mean(res)) if res else 0.0


def discover_raw_ocel_path(config: dict[str, Any]) -> Path:
    """Locate the raw OCEL file in ``data/raw/``.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        Path to the discovered OCEL file.

    Raises:
        FileNotFoundError: If no supported OCEL file is found.
    """
    raw_dir = Path(config["paths"]["raw_data_dir"])
    configured_file = config["paths"].get("raw_ocel_file")
    if configured_file:
        candidate = Path(configured_file)
        if not candidate.is_absolute():
            candidate = raw_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Configured raw OCEL file not found: {candidate}")
        return candidate

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    candidates = sorted(
        p
        for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_RAW_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(
            f"No OCEL file with extension {SUPPORTED_RAW_EXTENSIONS} found in {raw_dir}"
        )
    if len(candidates) > 1:
        LOGGER.warning(
            "Multiple OCEL files found in %s; selecting %s. Set paths.raw_ocel_file to override.",
            raw_dir,
            candidates[0].name,
        )
    return candidates[0]


def build_ocel_from_dataframe(df: pd.DataFrame, parameters: dict[str, Any]) -> OCEL:
    """Build a native ``ocpa`` ``OCEL`` object from an in-memory DataFrame.

    Ensures events are sorted chronologically and assigned monotonic integer
    ``event_id`` indices matching ``ocpa``'s ``sequential.construct_sequence``
    ordering convention.

    Args:
        df: Event DataFrame containing ``event_activity``, ``event_timestamp``,
            object-type list columns, and optional event attribute columns.
        parameters: ``ocpa`` table/execution parameters including ``obj_names``.

    Returns:
        Native ``ocpa.objects.log.ocel.OCEL`` instance.
    """
    work_df = df.copy()
    if "event_id" not in work_df.columns:
        work_df["event_id"] = range(len(work_df))
    work_df["event_timestamp"] = pd.to_datetime(work_df["event_timestamp"])
    work_df = work_df.sort_values(["event_timestamp", "event_id"]).reset_index(drop=True)
    work_df["event_id"] = range(len(work_df))

    table_log = Table(work_df, parameters=parameters)
    obj_log = df_to_ocel.apply(work_df)
    event_graph = EventGraph(table_utils.eog_from_log(table_log))
    return OCEL(table_log, obj_log, event_graph, parameters)


def _resolve_object_type_subset(
    available_types: list[str],
    configured_types: list[str] | None,
) -> list[str]:
    """Resolve configured object types against available OCEL types (case-insensitive)."""
    if not configured_types:
        return sorted(available_types)
    lookup = {ot.lower(): ot for ot in available_types}
    resolved: list[str] = []
    missing: list[str] = []
    for req in configured_types:
        matched = lookup.get(str(req).strip().lower())
        if matched is None:
            missing.append(str(req))
        elif matched not in resolved:
            resolved.append(matched)
    if missing:
        raise ValueError(
            f"Configured object_types {missing} not found in OCEL object types {sorted(available_types)}."
        )
    return resolved


def _load_sqlite_ocel_to_ocpa(
    filepath: Path,
    parameters: dict[str, Any],
    allowed_object_types: list[str] | None = None,
) -> OCEL:
    """Parse an OCEL 2.0 SQLite database into a native ``ocpa`` ``OCEL`` object."""
    conn = sqlite3.connect(str(filepath))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        if "event" not in tables or "event_object" not in tables or "object" not in tables:
            raise ValueError(
                f"SQLite file {filepath} does not contain standard OCEL 2.0 tables "
                "(event, object, event_object)."
            )

        ev_map = (
            pd.read_sql_query("SELECT * FROM event_map_type", conn)
            if "event_map_type" in tables
            else None
        )
        events_df = pd.read_sql_query("SELECT * FROM event", conn)
        objects_df = pd.read_sql_query("SELECT * FROM object", conn)
        e2o_df = pd.read_sql_query("SELECT * FROM event_object", conn)

        ev_attr_frames: list[pd.DataFrame] = []
        if ev_map is not None and not ev_map.empty:
            for _, row in ev_map.iterrows():
                tbl = f"event_{row['ocel_type_map']}"
                if tbl in tables:
                    sub = pd.read_sql_query(f'SELECT * FROM "{tbl}"', conn)
                    sub["ocel_type"] = row["ocel_type"]
                    ev_attr_frames.append(sub)
        else:
            for tbl in sorted(tables):
                if tbl.startswith("event_") and tbl not in {"event_object", "event_map_type"}:
                    sub = pd.read_sql_query(f'SELECT * FROM "{tbl}"', conn)
                    sub["ocel_type"] = tbl[len("event_") :]
                    ev_attr_frames.append(sub)

        if ev_attr_frames:
            ev_details = pd.concat(ev_attr_frames, ignore_index=True)
            events_df = events_df.merge(
                ev_details,
                on=["ocel_id", "ocel_type"],
                how="left",
                suffixes=("", "_detail"),
            )

        if "ocel_time" not in events_df.columns:
            raise ValueError("OCEL 2.0 SQLite event tables missing 'ocel_time' column.")

        all_object_types = sorted(objects_df["ocel_type"].dropna().unique().tolist())
        object_types = _resolve_object_type_subset(all_object_types, allowed_object_types)
        allowed_set = set(object_types)

        filtered_objects = objects_df[objects_df["ocel_type"].isin(allowed_set)]
        obj_type_lookup = dict(zip(filtered_objects["ocel_id"], filtered_objects["ocel_type"]))

        e2o_map: dict[Any, dict[str, list[str]]] = {}
        for ev_id, obj_id in zip(e2o_df["ocel_event_id"], e2o_df["ocel_object_id"]):
            ot = obj_type_lookup.get(obj_id)
            if ot is None:
                continue
            e2o_map.setdefault(ev_id, {}).setdefault(str(ot), []).append(str(obj_id))

        base_df = pd.DataFrame(
            {
                "event_id": range(len(events_df)),
                "event_activity": events_df["ocel_type"].astype(str),
                "event_timestamp": pd.to_datetime(
                    events_df["ocel_time"], utc=True, errors="coerce"
                ),
            }
        )
        for ot in object_types:
            base_df[ot] = [
                e2o_map.get(raw_id, {}).get(ot, []) for raw_id in events_df["ocel_id"]
            ]

        reserved = {"ocel_id", "ocel_type", "ocel_time"}
        val_names: list[str] = []
        for col in events_df.columns:
            if col in reserved or col.endswith("_detail"):
                continue
            target_col = col if col.startswith("event_") else f"event_{col}"
            base_df[target_col] = events_df[col].values
            val_names.append(target_col)

        table_params = {
            "obj_names": object_types,
            "val_names": val_names,
            "act_name": "event_activity",
            "time_name": "event_timestamp",
            "sep": ",",
        }
        table_params.update(parameters)
        LOGGER.info(
            "Loaded SQLite OCEL: %d events, filtered object_types=%s (from %d total types).",
            len(base_df),
            object_types,
            len(all_object_types),
        )
        return build_ocel_from_dataframe(base_df, table_params)
    finally:
        conn.close()


def _apply_strict_leading_type_executions(ocel: OCEL, leading_type: str) -> OCEL:
    """Partition events strictly by ``leading_type`` without cross-case duplication.

    Each distinct object of ``leading_type`` forms one process execution containing
    exclusively the events directly linked to that leading object. All configured
    object types remain attached to those events for ``ocpa`` feature extraction
    and entanglement calculation, while parent objects shared across cases (e.g.,
    ``Match`` shared by two ``Session``s) do NOT pull opponent events into a case.

    Args:
        ocel: Imported ``ocpa`` ``OCEL`` object.
        leading_type: Object type name defining the case notion (e.g., ``"Session"``).

    Returns:
        The ``OCEL`` instance with strict per-object process executions assigned.
    """
    resolved = _resolve_object_type_subset(list(ocel.object_types), [leading_type])[0]
    log_df = ocel.log.log

    case_events_by_obj: dict[str, list[Any]] = {}
    case_objs_by_obj: dict[str, set[tuple[str, str]]] = {}

    for row in log_df.itertuples(index=False):
        eid = getattr(row, "event_id")
        lead_objs = getattr(row, resolved)
        if not lead_objs:
            continue
        row_all_objs = [
            (ot, str(oid))
            for ot in ocel.object_types
            for oid in getattr(row, ot)
        ]
        for lead_oid in lead_objs:
            key = str(lead_oid)
            case_events_by_obj.setdefault(key, []).append(eid)
            case_objs_by_obj.setdefault(key, set()).update(row_all_objs)

    if not case_events_by_obj:
        raise ValueError(
            f"No events associated with leading_type='{resolved}' were found in OCEL."
        )

    sorted_case_keys = sorted(case_events_by_obj.keys())
    executions: list[list[Any]] = []
    exec_objects: list[list[tuple[str, str]]] = []
    exec_mappings: dict[Any, list[int]] = {}

    for case_idx, case_key in enumerate(sorted_case_keys):
        eids = case_events_by_obj[case_key]
        executions.append(eids)
        exec_objects.append(sorted(case_objs_by_obj[case_key]))
        for eid in eids:
            exec_mappings.setdefault(eid, []).append(case_idx)

    ocel._process_executions = executions
    ocel._process_execution_objects = exec_objects
    ocel._process_execution_mappings = exec_mappings
    ocel.parameters["strict_case_object_ids"] = sorted_case_keys
    LOGGER.info(
        "Constructed %d strict '%s' process executions (%d total event memberships).",
        len(executions),
        resolved,
        sum(len(e) for e in executions),
    )
    return ocel


def load_ocel(filepath: str | Path, config: dict[str, Any]) -> OCEL:
    """Import an OCEL file via ``ocpa.objects.log.importer``.

    Args:
        filepath: Path to ``.jsonocel``, ``.xmlocel``, ``.csv``, or ``.sqlite`` file.
        config: Loaded configuration dictionary.

    Returns:
        Native ``ocpa`` ``OCEL`` object.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"OCEL input file does not exist: {path}")

    ocpa_cfg = config.get("ocpa", {})
    exec_method = str(ocpa_cfg.get("execution_extraction", "connected_components"))
    allowed_object_types = ocpa_cfg.get("object_types")
    leading_type = ocpa_cfg.get("leading_type")

    parameters: dict[str, Any] = {
        "execution_extraction": LEAD_TYPE if exec_method == "leading_type" else CONN_COMP,
    }
    if exec_method == "leading_type" and leading_type:
        parameters["leading_type"] = leading_type

    ext = path.suffix.lower()
    LOGGER.info("Importing OCEL from %s (format=%s, extraction=%s)", path, ext, exec_method)

    if ext == ".jsonocel":
        ocel = ocel_import.apply(str(path), variant=ocel_import.OCEL_JSON, parameters=parameters)
    elif ext == ".xmlocel":
        ocel = ocel_import.apply(str(path), variant=ocel_import.OCEL_XML, parameters=parameters)
    elif ext == ".csv":
        ocel = csv_import.apply(str(path), parameters=parameters)
    elif ext == ".sqlite":
        ocel = _load_sqlite_ocel_to_ocpa(
            path,
            parameters=parameters,
            allowed_object_types=allowed_object_types,
        )
    else:
        raise ValueError(
            f"Unsupported OCEL extension '{ext}'. Supported: {SUPPORTED_RAW_EXTENSIONS}"
        )

    if exec_method == "strict_leading_type":
        if not leading_type:
            raise ValueError(
                "ocpa.leading_type must be set when execution_extraction='strict_leading_type'."
            )
        ocel = _apply_strict_leading_type_executions(ocel, leading_type=str(leading_type))

    return ocel


def _apply_debug_caps(
    ocel: OCEL,
    max_cases: int | None,
    max_events_per_case: int | None,
) -> OCEL:
    """Cap cases and events per case in ``debug_mode`` without splitting process executions."""
    raw_executions = ocel.process_executions
    raw_exec_objects = ocel.process_execution_objects
    log_df = ocel.log.log
    ts_by_eid = dict(zip(log_df["event_id"], log_df["event_timestamp"]))

    selected_case_indices = list(range(len(raw_executions)))
    if max_cases is not None and len(selected_case_indices) > max_cases:
        selected_case_indices = selected_case_indices[:max_cases]

    capped_executions: list[list[Any]] = []
    capped_objects: list[Any] = []
    capped_mappings: dict[Any, list[int]] = {}
    kept_eids: set[Any] = set()

    for new_case_idx, orig_case_idx in enumerate(selected_case_indices):
        ev_ids = raw_executions[orig_case_idx]
        sorted_eids = sorted(ev_ids, key=lambda eid: (ts_by_eid[eid], eid))
        if max_events_per_case is not None:
            sorted_eids = sorted_eids[:max_events_per_case]
        capped_executions.append(sorted_eids)
        capped_objects.append(raw_exec_objects[orig_case_idx])
        for eid in sorted_eids:
            kept_eids.add(eid)
            capped_mappings.setdefault(eid, []).append(new_case_idx)

    LOGGER.info(
        "Debug mode active: capped to %d cases and <= %s events/case (%d total events).",
        len(capped_executions),
        str(max_events_per_case),
        len(kept_eids),
    )

    capped_params = dict(ocel.parameters)
    if "strict_case_object_ids" in capped_params:
        orig_oids = capped_params["strict_case_object_ids"]
        capped_params["strict_case_object_ids"] = [orig_oids[i] for i in selected_case_indices]

    filtered_df = log_df[log_df["event_id"].isin(kept_eids)].copy().reset_index(drop=True)
    new_table = Table(filtered_df, parameters=capped_params)
    sub_eog = ocel.graph.eog.subgraph(kept_eids).copy()
    new_graph = EventGraph(sub_eog)
    capped_ocel = OCEL(new_table, ocel.obj, new_graph, capped_params)
    capped_ocel._process_executions = capped_executions
    capped_ocel._process_execution_mappings = capped_mappings
    capped_ocel._process_execution_objects = capped_objects
    return capped_ocel


def _detect_start_timestamp_column(ocel: OCEL, config: dict[str, Any]) -> str | None:
    """Detect an event start-timestamp column if present in ``ocel.log.log``."""
    configured = config.get("ocpa", {}).get("start_timestamp_column")
    cols = set(ocel.log.log.columns)
    if configured and configured in cols:
        return str(configured)
    for candidate in ("event_start_timestamp", "start_timestamp"):
        if candidate in cols:
            return candidate
    return None


def _detect_numeric_event_attributes(ocel: OCEL, config: dict[str, Any]) -> list[str]:
    """Identify numeric event-level attributes present in ``ocel.log.log``."""
    excluded = {
        "event_id",
        "event_activity",
        "event_timestamp",
        "event_start_timestamp",
        "event_objects",
        *ocel.object_types,
    }
    numeric_cols: list[str] = []
    for col in ocel.log.log.columns:
        if col in excluded:
            continue
        series = ocel.log.log[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_cols.append(col)

    preferred = config.get("ocpa", {}).get("event_value_attribute")
    if preferred:
        for cand in (preferred, f"event_{preferred}"):
            if cand in numeric_cols:
                numeric_cols.remove(cand)
                numeric_cols.insert(0, cand)
                break
    return numeric_cols


def _detect_target_object_type(ocel: OCEL, config: dict[str, Any]) -> str:
    """Select the target object type for O3 and P7 features."""
    configured = config.get("ocpa", {}).get("target_object_type")
    if configured and configured in ocel.object_types:
        return str(configured)
    if not ocel.object_types:
        raise ValueError("OCEL contains no object types.")
    return str(sorted(ocel.object_types)[0])


def _extract_feature_storage_fast(
    ocel: OCEL,
    config: dict[str, Any],
    activities: list[str],
    object_types: list[str],
    target_ot: str,
    start_ts_col: str | None,
    numeric_attrs: list[str],
    primary_attr: str | None,
    canonical_feature_keys: list[str],
    time_divisor: float,
) -> tuple[Feature_Storage, pd.DataFrame]:
    """O(N) single-pass prefix-sweep extraction numerically equivalent to native ``ocpa``.

    Produces a native ``ocpa.algo.predictive_monitoring.obj.Feature_Storage`` with
    exact bit-level parity to ``ocpa.algo.predictive_monitoring.factory.apply(...)``
    while avoiding ``ocpa``'s O(N^2 * A) pure-Python prefix re-scanning.
    """
    case_col = config["identifiers"]["case_id"]
    target_col = config["target"]["column"]

    log_df = ocel.log.log
    event_ids_all = log_df["event_id"].tolist()
    ts_list_all = log_df["event_timestamp"].tolist()
    act_list_all = log_df["event_activity"].tolist()

    ts_lookup: dict[Any, pd.Timestamp] = dict(zip(event_ids_all, ts_list_all))
    ts_sec_lookup: dict[Any, float] = {
        eid: pd.Timestamp(ts).value / 1e9 for eid, ts in ts_lookup.items()
    }
    act_lookup: dict[Any, Any] = dict(zip(event_ids_all, act_list_all))

    start_ts_sec_lookup: dict[Any, float] | None = None
    if start_ts_col is not None:
        start_ts_sec_lookup = {
            eid: pd.Timestamp(ts).value / 1e9
            for eid, ts in zip(event_ids_all, log_df[start_ts_col].tolist())
        }

    # Per-object-type object lists and combined event_objects per event_id
    # Use ocel.object_types order to match native ocpa's exact node.objects list order
    ocel_ot_order = [str(ot) for ot in ocel.object_types]
    ot_lists: dict[str, dict[Any, list[Any]]] = {
        ot: dict(zip(event_ids_all, log_df[ot].tolist())) for ot in ocel_ot_order
    }
    event_objects_lookup: dict[Any, list[tuple[str, Any]]] = {}
    event_objects_set_lookup: dict[Any, set[tuple[str, Any]]] = {}
    for eid in event_ids_all:
        objs = [(ot, o) for ot in ocel_ot_order for o in ot_lists[ot][eid]]
        event_objects_lookup[eid] = objs
        event_objects_set_lookup[eid] = set(objs)

    # Numeric event attribute lookups
    attr_lookup: dict[str, dict[Any, float]] = {}
    for attr in numeric_attrs:
        vals: dict[Any, float] = {}
        for eid, raw_v in zip(event_ids_all, log_df[attr].tolist()):
            try:
                fv = float(raw_v)
                vals[eid] = fv if np.isfinite(fv) else float("nan")
            except (TypeError, ValueError):
                vals[eid] = float("nan")
        attr_lookup[attr] = vals

    # Pre-index full EOG incoming predecessor activities (matches ocpa preceding_activities)
    eog = ocel.graph.eog
    eog_pred_act_counts: dict[Any, Counter[str]] = {}
    for eid in event_ids_all:
        cnt: Counter[str] = Counter()
        for src, _dst in eog.in_edges(eid):
            src_act = act_lookup.get(src)
            if src_act is not None and not pd.isna(src_act):
                cnt[str(src_act)] += 1
        eog_pred_act_counts[eid] = cnt

    # Precompute activity key strings once
    act_c5_keys = [(act, f"C5_activity_{_sanitize_name(act)}") for act in activities]
    act_c2_keys = [(act, f"C2_preceding_act_{_sanitize_name(act)}") for act in activities]
    act_c3_keys = [(act, f"C3_prev_act_count_{_sanitize_name(act)}") for act in activities]
    act_to_idx = {act: idx for idx, act in enumerate(activities)}
    num_acts = len(activities)

    ot_type_count_keys = [(ot, f"O_type_count_{_sanitize_name(ot)}") for ot in object_types]
    ot_prev_keys = [
        (
            ot,
            "O3_prev_target_obj_count"
            if ot == target_ot
            else f"ocpa_extra_prev_type_count_{_sanitize_name(ot)}",
        )
        for ot in object_types
    ]
    ot_pool_keys = [
        (
            ot,
            "P7_pooling_time"
            if ot == target_ot
            else f"ocpa_extra_pooling_time_{_sanitize_name(ot)}",
        )
        for ot in object_types
    ]
    extra_attrs = numeric_attrs[1:]
    extra_attr_keys = [
        (
            attr,
            f"ocpa_extra_char_{_sanitize_name(attr)}",
            f"ocpa_extra_avg_prev_{_sanitize_name(attr)}",
        )
        for attr in extra_attrs
    ]

    process_executions = ocel.process_executions
    exec_mappings = ocel.process_execution_mappings
    num_cases = len(process_executions)

    # Check if every event belongs to exactly 1 case (fast path with zero cross-case averaging)
    all_single_case = all(len(c_list) == 1 for c_list in exec_mappings.values())

    # Precompute per-case timestamp-block prefix metrics & per-case preset timing metrics in O(N)
    # Structure per case c_id: maps eid -> precomputed scalar/vector values
    case_ordered_eids: list[list[Any]] = []
    case_remaining_sec: list[dict[Any, float]] = []
    case_elapsed_sec: list[dict[Any, float]] = []
    case_prev_obj_cnt: list[dict[Any, int]] = []
    case_prev_type_cnt: list[dict[str, dict[Any, int]]] = []
    case_prev_act_arr: list[dict[Any, np.ndarray]] = []
    case_avg_prev_attr: list[dict[str, dict[Any, float]]] = []
    case_preset_timings: list[dict[Any, dict[str, float]]] = []

    for c_id in range(num_cases):
        raw_eids = list(process_executions[c_id])
        ordered_eids = sorted(raw_eids, key=lambda eid: (ts_lookup[eid], eid))
        case_ordered_eids.append(ordered_eids)

        if not ordered_eids:
            case_remaining_sec.append({})
            case_elapsed_sec.append({})
            case_prev_obj_cnt.append({})
            case_prev_type_cnt.append({ot: {} for ot in object_types})
            case_prev_act_arr.append({})
            case_avg_prev_attr.append({attr: {} for attr in numeric_attrs})
            case_preset_timings.append({})
            continue

        min_ts_sec = ts_sec_lookup[ordered_eids[0]]
        max_ts_sec = ts_sec_lookup[ordered_eids[-1]]

        rem_map: dict[Any, float] = {}
        elap_map: dict[Any, float] = {}
        prev_obj_map: dict[Any, int] = {}
        prev_type_map: dict[str, dict[Any, int]] = {ot: {} for ot in object_types}
        prev_act_map: dict[Any, np.ndarray] = {}
        avg_attr_map: dict[str, dict[Any, float]] = {attr: {} for attr in numeric_attrs}

        seen_objs: set[tuple[str, Any]] = set()
        seen_by_ot: dict[str, set[tuple[str, Any]]] = {ot: set() for ot in object_types}
        running_act_counts = np.zeros(num_acts, dtype=np.int64)
        running_attr_sum: dict[str, float] = {attr: 0.0 for attr in numeric_attrs}
        running_attr_cnt: dict[str, int] = {attr: 0 for attr in numeric_attrs}

        # Sweep contiguous equal-timestamp blocks to match ocpa's get_recent_events (ts <= ts_e)
        n_ev = len(ordered_eids)
        idx = 0
        while idx < n_ev:
            block_ts = ts_lookup[ordered_eids[idx]]
            block_end = idx + 1
            while block_end < n_ev and ts_lookup[ordered_eids[block_end]] == block_ts:
                block_end += 1
            block_eids = ordered_eids[idx:block_end]

            # Accumulate all events in this equal-timestamp block first
            for eid in block_eids:
                objs = event_objects_lookup[eid]
                seen_objs.update(objs)
                for ot_o in objs:
                    seen_by_ot[ot_o[0]].add(ot_o)
                act_val = act_lookup[eid]
                if act_val is not None and not pd.isna(act_val):
                    a_idx = act_to_idx.get(str(act_val))
                    if a_idx is not None:
                        running_act_counts[a_idx] += 1
                for attr in numeric_attrs:
                    fv = attr_lookup[attr][eid]
                    if np.isfinite(fv):
                        running_attr_sum[attr] += fv
                        running_attr_cnt[attr] += 1

            block_ts_sec = ts_sec_lookup[block_eids[0]]
            block_elapsed = block_ts_sec - min_ts_sec
            block_remaining = max_ts_sec - block_ts_sec
            block_obj_total = len(seen_objs)
            block_ot_totals = {ot: len(seen_by_ot[ot]) for ot in object_types}
            block_act_snapshot = running_act_counts.copy()
            block_attr_means = {
                attr: (
                    running_attr_sum[attr] / running_attr_cnt[attr]
                    if running_attr_cnt[attr] > 0
                    else float("nan")
                )
                for attr in numeric_attrs
            }

            for eid in block_eids:
                rem_map[eid] = block_remaining
                elap_map[eid] = block_elapsed
                prev_obj_map[eid] = block_obj_total
                for ot in object_types:
                    prev_type_map[ot][eid] = block_ot_totals[ot]
                prev_act_map[eid] = block_act_snapshot
                for attr in numeric_attrs:
                    avg_attr_map[attr][eid] = block_attr_means[attr]

            idx = block_end

        case_remaining_sec.append(rem_map)
        case_elapsed_sec.append(elap_map)
        case_prev_obj_cnt.append(prev_obj_map)
        case_prev_type_cnt.append(prev_type_map)
        case_prev_act_arr.append(prev_act_map)
        case_avg_prev_attr.append(avg_attr_map)

        # Case-subgraph preset timing features in O(N_c + |E_c|)
        case_subgraph = eog.subgraph(ordered_eids)
        preset_map: dict[Any, dict[str, float]] = {}
        for eid in ordered_eids:
            preset = [src for (src, _dst) in case_subgraph.in_edges(eid)]
            if not preset:
                tdict: dict[str, float] = {
                    "P5_synchronization_time": 0.0,
                    "P8_lagging_time": 0.0,
                    "P10_waiting_time": 0.0,
                    "P11_flow_time": 0.0,
                    "P12_sojourn_time": 0.0,
                }
                for _ot, pkey in ot_pool_keys:
                    tdict[pkey] = 0.0
                preset_map[eid] = tdict
                continue

            end_ts_secs = [ts_sec_lookup[u] for u in preset]
            max_end = max(end_ts_secs)
            min_end = min(end_ts_secs)
            curr_ts = ts_sec_lookup[eid]

            sojourn = curr_ts - max_end
            flow = curr_ts - min_end
            sync = max_end - min_end
            waiting = (
                (start_ts_sec_lookup[eid] - max_end)
                if start_ts_sec_lookup is not None
                else sojourn
            )

            ot_maxs: list[float] = []
            target_ot_ts: list[float] = []
            tdict = {
                "P5_synchronization_time": sync,
                "P10_waiting_time": waiting,
                "P11_flow_time": flow,
                "P12_sojourn_time": sojourn,
            }
            for ot, pkey in ot_pool_keys:
                ot_ts = [ts_sec_lookup[u] for u in preset if ot_lists[ot][u]]
                if ot_ts:
                    ot_max = max(ot_ts)
                    ot_maxs.append(ot_max)
                    tdict[pkey] = ot_max - min(ot_ts)
                    if ot == target_ot:
                        target_ot_ts = ot_ts
                else:
                    tdict[pkey] = 0.0

            if target_ot_ts and ot_maxs:
                tdict["P8_lagging_time"] = max(0.0, max(target_ot_ts) - min(ot_maxs))
            else:
                tdict["P8_lagging_time"] = 0.0

            preset_map[eid] = tdict

        case_preset_timings.append(preset_map)

    # Assemble native ocpa Feature_Storage & Feature_Graphs matching native structure 1-to-1
    cv_group_ot = config.get("ocpa", {}).get("cv_group_object_type")
    resolved_cv_ot = (
        _resolve_object_type_subset(list(ocel.object_types), [str(cv_group_ot)])[0]
        if cv_group_ot
        else None
    )
    strict_case_oids: list[str] | None = ocel.parameters.get("strict_case_object_ids")

    feature_storage = Feature_Storage(
        event_features=list(canonical_feature_keys),
        execution_features=[],
        ocel=ocel,
    )
    sorted_graphs: list[Feature_Storage.Feature_Graph] = []
    case_index_rows: list[dict[str, Any]] = []

    for graph_idx, c_id in enumerate(range(num_cases)):
        ordered_eids = case_ordered_eids[c_id]
        pos_by_eid = {eid: i for i, eid in enumerate(ordered_eids)}
        case_subgraph = eog.subgraph(ordered_eids)

        fg = Feature_Storage.Feature_Graph.__new__(Feature_Storage.Feature_Graph)
        fg._pexec_id = c_id
        nodes = [
            Feature_Storage.Feature_Graph.Node(eid, event_objects_lookup[eid], c_id)
            for eid in ordered_eids
        ]
        fg._nodes = nodes
        fg._node_mapping = {n.event_id: n for n in nodes}

        causal_edges: list[Feature_Storage.Feature_Graph.Edge] = []
        edge_objects_map: dict[tuple[Any, Any], set[tuple[str, Any]]] = {}
        for src, dst in case_subgraph.edges:
            if src in pos_by_eid and dst in pos_by_eid and pos_by_eid[src] < pos_by_eid[dst]:
                shared = event_objects_set_lookup[src].intersection(event_objects_set_lookup[dst])
                edge_objects_map[(src, dst)] = shared
                causal_edges.append(
                    Feature_Storage.Feature_Graph.Edge(src, dst, objects=shared)
                )
        fg._objects = edge_objects_map
        fg._edges = causal_edges
        fg._edge_mapping = {(e.source, e.target): e for e in causal_edges}
        fg._attributes = {}

        cv_group_id: str | None = None
        if resolved_cv_ot is not None:
            group_oids = sorted(
                {
                    str(oid)
                    for node in nodes
                    for (ot, oid) in (node.objects or [])
                    if str(ot) == resolved_cv_ot
                }
            )
            if group_oids:
                cv_group_id = group_oids[0]
                fg.add_attribute("cv_group_id", cv_group_id)

        case_obj_id = (
            strict_case_oids[c_id]
            if strict_case_oids and c_id < len(strict_case_oids)
            else str(c_id)
        )
        fg.add_attribute("case_object_id", case_obj_id)

        rem_c = case_remaining_sec[c_id]
        elap_c = case_elapsed_sec[c_id]
        prev_obj_c = case_prev_obj_cnt[c_id]
        prev_type_c = case_prev_type_cnt[c_id]
        prev_act_c = case_prev_act_arr[c_id]
        avg_attr_c = case_avg_prev_attr[c_id]
        preset_c = case_preset_timings[c_id]

        for node in nodes:
            eid = node.event_id
            act_val = act_lookup[eid]
            act_str = None if (act_val is None or pd.isna(act_val)) else str(act_val)
            pred_cnts = eog_pred_act_counts[eid]
            cases_for_ev = [c_id] if all_single_case else exec_mappings.get(eid, [c_id])
            inv_n_cases = 1.0 / len(cases_for_ev)

            attrs: dict[str, float] = {}

            # Target (strictly node.pexec_id, matching ocpa remaining_time)
            y_val = rem_c[eid] / time_divisor
            attrs[target_col] = max(0.0, float(y_val)) if np.isfinite(y_val) else 0.0

            # C5: Current Activity
            for act, k_c5 in act_c5_keys:
                attrs[k_c5] = 1.0 if act_str == act else 0.0

            # C2: Preceding Activities (from full EOG in-edges)
            for act, k_c2 in act_c2_keys:
                attrs[k_c2] = float(pred_cnts.get(act, 0))

            # C3: Previous Activity Counts
            if len(cases_for_ev) == 1:
                c0 = cases_for_ev[0]
                act_vec = prev_act_c[eid] if c0 == c_id else case_prev_act_arr[c0][eid]
                for a_i, (_act, k_c3) in enumerate(act_c3_keys):
                    attrs[k_c3] = float(act_vec[a_i])
            else:
                summed = np.zeros(num_acts, dtype=np.float64)
                for cc in cases_for_ev:
                    summed += case_prev_act_arr[cc][eid]
                summed *= inv_n_cases
                for a_i, (_act, k_c3) in enumerate(act_c3_keys):
                    attrs[k_c3] = float(summed[a_i])

            # P2: Elapsed time & P9: Service time
            if len(cases_for_ev) == 1:
                c0 = cases_for_ev[0]
                elap_s = elap_c[eid] if c0 == c_id else case_elapsed_sec[c0][eid]
            else:
                elap_s = sum(case_elapsed_sec[cc][eid] for cc in cases_for_ev) * inv_n_cases
            attrs["P2_elapsed_time"] = float(elap_s / time_divisor)

            if start_ts_sec_lookup is not None:
                attrs["P9_service_time"] = float(
                    (ts_sec_lookup[eid] - start_ts_sec_lookup[eid]) / time_divisor
                )
            else:
                attrs["P9_service_time"] = 0.0

            # Preset timing features: P5, P7, P8, P10, P11, P12, extra pooling
            if len(cases_for_ev) == 1:
                c0 = cases_for_ev[0]
                pmap = preset_c[eid] if c0 == c_id else case_preset_timings[c0][eid]
                for tkey, tsec in pmap.items():
                    attrs[tkey] = float(tsec / time_divisor)
            else:
                for tkey in preset_c[eid]:
                    avg_s = (
                        sum(case_preset_timings[cc][eid][tkey] for cc in cases_for_ev)
                        * inv_n_cases
                    )
                    attrs[tkey] = float(avg_s / time_divisor)

            # Object count features: O5, O2, O_type_count_*, O3, extra prev_type_count
            attrs["O5_event_object_count"] = float(len(event_objects_lookup[eid]))
            if len(cases_for_ev) == 1:
                c0 = cases_for_ev[0]
                attrs["O2_prev_object_count"] = float(
                    prev_obj_c[eid] if c0 == c_id else case_prev_obj_cnt[c0][eid]
                )
                for ot, k_prev in ot_prev_keys:
                    attrs[k_prev] = float(
                        prev_type_c[ot][eid] if c0 == c_id else case_prev_type_cnt[c0][ot][eid]
                    )
            else:
                attrs["O2_prev_object_count"] = float(
                    sum(case_prev_obj_cnt[cc][eid] for cc in cases_for_ev) * inv_n_cases
                )
                for ot, k_prev in ot_prev_keys:
                    attrs[k_prev] = float(
                        sum(case_prev_type_cnt[cc][ot][eid] for cc in cases_for_ev) * inv_n_cases
                    )

            for ot, k_tc in ot_type_count_keys:
                attrs[k_tc] = float(len(ot_lists[ot][eid]))

            # Event attributes: D3, D1, and any extra numeric attributes
            if primary_attr is not None:
                attrs["D3_current_event_attr"] = float(attr_lookup[primary_attr][eid])
                if len(cases_for_ev) == 1:
                    c0 = cases_for_ev[0]
                    attrs["D1_avg_prev_event_attr"] = float(
                        avg_attr_c[primary_attr][eid]
                        if c0 == c_id
                        else case_avg_prev_attr[c0][primary_attr][eid]
                    )
                else:
                    vals = [case_avg_prev_attr[cc][primary_attr][eid] for cc in cases_for_ev]
                    attrs["D1_avg_prev_event_attr"] = float(sum(vals) * inv_n_cases)
            else:
                attrs["D3_current_event_attr"] = 0.0
                attrs["D1_avg_prev_event_attr"] = 0.0

            for attr, k_ch, k_avg in extra_attr_keys:
                attrs[k_ch] = float(attr_lookup[attr][eid])
                if len(cases_for_ev) == 1:
                    c0 = cases_for_ev[0]
                    attrs[k_avg] = float(
                        avg_attr_c[attr][eid] if c0 == c_id else case_avg_prev_attr[c0][attr][eid]
                    )
                else:
                    vals = [case_avg_prev_attr[cc][attr][eid] for cc in cases_for_ev]
                    attrs[k_avg] = float(sum(vals) * inv_n_cases)

            node._attributes = attrs

        sorted_graphs.append(fg)
        case_row: dict[str, Any] = {
            "Graph_Index": int(graph_idx),
            case_col: int(c_id),
            "Num_Events": int(len(fg.nodes)),
            "Num_Edges": int(len(fg.edges)),
        }
        if strict_case_oids is not None:
            case_row["Case_Object_ID"] = case_obj_id
        if cv_group_id is not None:
            case_row["CV_Group_ID"] = cv_group_id
        case_index_rows.append(case_row)

    feature_storage.feature_graphs = sorted_graphs
    feature_storage.event_features = list(canonical_feature_keys)
    feature_storage._scaler = None
    feature_storage._training_indices = None
    feature_storage._test_indices = None

    case_index_df = pd.DataFrame(case_index_rows)
    if case_index_df.empty:
        raise ValueError("Feature extraction produced zero Feature_Graphs.")
    validate_feature_storage(feature_storage, case_index_df, config)
    return feature_storage, case_index_df


def extract_feature_storage(
    ocel: OCEL,
    config: dict[str, Any],
    extraction_mode: str | None = None,
) -> tuple[Feature_Storage, pd.DataFrame]:
    """Extract an unnormalized ``ocpa`` ``Feature_Storage`` and case index metadata.

    Supports two numerically equivalent engines selectable via
    ``config["ocpa"]["extraction_mode"]`` or ``extraction_mode``:
        - ``"fast"`` (default): O(N) single-pass timestamp-block prefix sweep.
        - ``"native"``: ``ocpa.algo.predictive_monitoring.factory.apply(...)``.

    NEVER normalizes or splits inside Stage 1.

    Args:
        ocel: Imported ``ocpa`` ``OCEL`` instance.
        config: Loaded configuration dictionary.
        extraction_mode: Optional override (``"fast"`` or ``"native"``).

    Returns:
        Tuple of ``(feature_storage, case_index_df)``.
    """
    if config.get("debug_mode", False):
        dbg = config.get("debug", {})
        ocel = _apply_debug_caps(
            ocel,
            max_cases=dbg.get("max_cases"),
            max_events_per_case=dbg.get("max_events_per_case"),
        )

    mode = str(
        extraction_mode
        if extraction_mode is not None
        else config.get("ocpa", {}).get("extraction_mode", "fast")
    ).lower()
    if mode not in {"fast", "native"}:
        raise ValueError(f"Unsupported extraction_mode '{mode}'. Choose 'fast' or 'native'.")

    case_col = config["identifiers"]["case_id"]
    target_col = config["target"]["column"]
    target_unit = str(config["target"].get("unit", "hours")).lower()
    if target_unit not in TIME_UNIT_DIVISORS:
        raise ValueError(
            f"Unsupported target.unit '{target_unit}'. Choose from {list(TIME_UNIT_DIVISORS)}."
        )
    time_divisor = TIME_UNIT_DIVISORS[target_unit]

    activities = sorted(
        str(a) for a in ocel.log.log["event_activity"].dropna().unique().tolist()
    )
    object_types = sorted(str(ot) for ot in ocel.object_types)
    target_ot = _detect_target_object_type(ocel, config)
    start_ts_col = _detect_start_timestamp_column(ocel, config)
    numeric_attrs = _detect_numeric_event_attributes(ocel, config)
    primary_attr = numeric_attrs[0] if numeric_attrs else None

    LOGGER.info(
        "Extracting ocpa Feature_Storage (mode=%s) for %d cases | %d activities | "
        "object_types=%s | target_ot=%s | primary_attr=%s | unit=%s",
        mode,
        len(ocel.process_executions),
        len(activities),
        object_types,
        target_ot,
        primary_attr,
        target_unit,
    )

    feature_specs: list[tuple[Any, tuple[Any, ...]]] = []
    spec_to_key: dict[tuple[Any, tuple[Any, ...]], str] = {}
    time_valued_keys: set[str] = {target_col}

    def _register(spec: tuple[Any, tuple[Any, ...]], key_name: str, is_time: bool = False) -> None:
        if spec not in spec_to_key:
            feature_specs.append(spec)
            spec_to_key[spec] = key_name
            if is_time:
                time_valued_keys.add(key_name)

    # Target: Remaining Time (P3)
    _register((pm_factory.EVENT_REMAINING_TIME, ()), target_col, is_time=True)

    # Tier 1: C5 (Current Activity via ocpa EVENT_ACTIVITY), D3, P9, P2
    for act in activities:
        _register(
            (pm_factory.EVENT_ACTIVITY, (act,)),
            f"C5_activity_{_sanitize_name(act)}",
        )
    _register((pm_factory.EVENT_ELAPSED_TIME, ()), "P2_elapsed_time", is_time=True)
    if start_ts_col is not None:
        _register((pm_factory.EVENT_SERVICE_TIME, (start_ts_col,)), "P9_service_time", is_time=True)
        _register((pm_factory.EVENT_WAITING_TIME, (start_ts_col,)), "P10_waiting_time", is_time=True)
    if primary_attr is not None:
        _register((pm_factory.EVENT_CHAR_VALUE, (primary_attr,)), "D3_current_event_attr")
        _register(
            (pm_factory.EVENT_AGG_PREVIOUS_CHAR_VALUES, (primary_attr, _safe_nanmean)),
            "D1_avg_prev_event_attr",
        )

    # Tier 2: C2 (Preceding Activities, 1-hop preset), D1, P2, O3
    for act in activities:
        _register(
            (pm_factory.EVENT_PRECEDING_ACTIVITES, (act,)),
            f"C2_preceding_act_{_sanitize_name(act)}",
        )
    _register((pm_factory.EVENT_PREVIOUS_TYPE_COUNT, (target_ot,)), "O3_prev_target_obj_count")

    # Tier 3: Additional graph & prefix features
    _register((pm_factory.EVENT_SYNCHRONIZATION_TIME, ()), "P5_synchronization_time", is_time=True)
    _register((pm_factory.EVENT_POOLING_TIME, (target_ot,)), "P7_pooling_time", is_time=True)
    _register((pm_factory.EVENT_FLOW_TIME, ()), "P11_flow_time", is_time=True)
    _register((pm_factory.EVENT_SOJOURN_TIME, ()), "P12_sojourn_time", is_time=True)
    _register((pm_factory.EVENT_PREVIOUS_OBJECT_COUNT, ()), "O2_prev_object_count")
    _register((pm_factory.EVENT_NUM_OF_OBJECTS, ()), "O5_event_object_count")

    for act in activities:
        _register(
            (pm_factory.EVENT_PREVIOUS_ACTIVITY_COUNT, (act,)),
            f"C3_prev_act_count_{_sanitize_name(act)}",
        )

    for ot in object_types:
        sanitized_ot = _sanitize_name(ot)
        _register((pm_factory.EVENT_TYPE_COUNT, (ot,)), f"O_type_count_{sanitized_ot}")
        if ot != target_ot:
            _register(
                (pm_factory.EVENT_PREVIOUS_TYPE_COUNT, (ot,)),
                f"ocpa_extra_prev_type_count_{sanitized_ot}",
            )
            _register(
                (pm_factory.EVENT_POOLING_TIME, (ot,)),
                f"ocpa_extra_pooling_time_{sanitized_ot}",
                is_time=True,
            )

    for extra_attr in numeric_attrs[1:]:
        sanitized_attr = _sanitize_name(extra_attr)
        _register(
            (pm_factory.EVENT_CHAR_VALUE, (extra_attr,)),
            f"ocpa_extra_char_{sanitized_attr}",
        )
        _register(
            (pm_factory.EVENT_AGG_PREVIOUS_CHAR_VALUES, (extra_attr, _safe_nanmean)),
            f"ocpa_extra_avg_prev_{sanitized_attr}",
        )

    canonical_feature_keys: list[str] = [
        spec_to_key[spec] for spec in feature_specs if spec_to_key[spec] != target_col
    ]
    for fallback_key in (
        "P8_lagging_time",
        "P9_service_time",
        "P10_waiting_time",
        "D3_current_event_attr",
        "D1_avg_prev_event_attr",
    ):
        if fallback_key not in canonical_feature_keys:
            canonical_feature_keys.append(fallback_key)

    t_start = time.perf_counter()
    if mode == "fast":
        feature_storage, case_index_df = _extract_feature_storage_fast(
            ocel=ocel,
            config=config,
            activities=activities,
            object_types=object_types,
            target_ot=target_ot,
            start_ts_col=start_ts_col,
            numeric_attrs=numeric_attrs,
            primary_attr=primary_attr,
            canonical_feature_keys=canonical_feature_keys,
            time_divisor=time_divisor,
        )
        elapsed = time.perf_counter() - t_start
        LOGGER.info(
            "Extracted unnormalized Feature_Storage (mode=fast) in %.3fs: %d cases, %d events, %d feature keys.",
            elapsed,
            len(case_index_df),
            int(case_index_df["Num_Events"].sum()),
            len(feature_storage.event_features),
        )
        return feature_storage, case_index_df

    workers = int(config.get("ocpa", {}).get("workers", 1))
    feature_storage = pm_factory.apply(
        ocel,
        event_based_features=feature_specs,
        workers=max(1, workers),
    )

    # Sort Feature_Graphs deterministically by pexec_id and canonicalize node attributes
    log_df = ocel.log.log
    ts_lookup = dict(zip(log_df["event_id"], log_df["event_timestamp"]))
    cv_group_ot = config.get("ocpa", {}).get("cv_group_object_type")
    if cv_group_ot:
        resolved_cv_ot = _resolve_object_type_subset(list(ocel.object_types), [str(cv_group_ot)])[0]
    else:
        resolved_cv_ot = None
    strict_case_oids: list[str] | None = ocel.parameters.get("strict_case_object_ids")

    sorted_graphs = sorted(feature_storage.feature_graphs, key=lambda g: g.pexec_id)
    case_index_rows: list[dict[str, Any]] = []

    for graph_idx, fg in enumerate(sorted_graphs):
        # Enforce chronological node order matching ocpa sequential encoding
        ordered_nodes = sorted(
            fg.nodes,
            key=lambda n: (ts_lookup[n.event_id], n.event_id),
        )
        fg._nodes = ordered_nodes
        fg._node_mapping = {n.event_id: n for n in ordered_nodes}
        pos_by_eid = {n.event_id: i for i, n in enumerate(ordered_nodes)}

        # Keep strictly causal directed edges (src < dst in chronological order)
        causal_edges = [
            edge
            for edge in fg.edges
            if edge.source in pos_by_eid
            and edge.target in pos_by_eid
            and pos_by_eid[edge.source] < pos_by_eid[edge.target]
        ]
        fg._edges = causal_edges
        fg._edge_mapping = {(e.source, e.target): e for e in causal_edges}

        # Attach optional CV group metadata (e.g., Match ID) directly on Feature_Graph
        cv_group_id: str | None = None
        if resolved_cv_ot is not None:
            group_oids = sorted(
                {
                    str(oid)
                    for node in ordered_nodes
                    for (ot, oid) in (node.objects or [])
                    if str(ot) == resolved_cv_ot
                }
            )
            if group_oids:
                cv_group_id = group_oids[0]
                fg.add_attribute("cv_group_id", cv_group_id)

        case_obj_id = (
            strict_case_oids[int(fg.pexec_id)]
            if strict_case_oids and int(fg.pexec_id) < len(strict_case_oids)
            else str(fg.pexec_id)
        )
        fg.add_attribute("case_object_id", case_obj_id)

        for node in ordered_nodes:
            raw_attrs = dict(node.attributes)
            clean_attrs: dict[str, float] = {}

            for spec, key_name in spec_to_key.items():
                val = raw_attrs.get(spec, float("nan"))
                try:
                    fval = float(val)
                    if not np.isfinite(fval):
                        fval = float("nan")
                except (TypeError, ValueError):
                    fval = float("nan")
                if key_name in time_valued_keys and np.isfinite(fval):
                    fval = fval / time_divisor
                clean_attrs[key_name] = fval

            # Safe P8 lagging time (guards empty object-type presets in ocpa==1.3.3)
            clean_attrs["P8_lagging_time"] = (
                _safe_lagging_time(node, ocel, target_ot) / time_divisor
            )

            # Deterministic fallbacks only when optional source columns do not exist
            if "P9_service_time" not in clean_attrs:
                clean_attrs["P9_service_time"] = 0.0
            if "P10_waiting_time" not in clean_attrs:
                clean_attrs["P10_waiting_time"] = clean_attrs.get("P12_sojourn_time", 0.0)
            if "D3_current_event_attr" not in clean_attrs:
                clean_attrs["D3_current_event_attr"] = 0.0
            if "D1_avg_prev_event_attr" not in clean_attrs:
                clean_attrs["D1_avg_prev_event_attr"] = 0.0

            # Remaining time target must be finite and non-negative
            y_val = clean_attrs.get(target_col, 0.0)
            clean_attrs[target_col] = max(0.0, float(y_val)) if np.isfinite(y_val) else 0.0

            node._attributes = clean_attrs

        case_row: dict[str, Any] = {
            "Graph_Index": int(graph_idx),
            case_col: int(fg.pexec_id),
            "Num_Events": int(len(fg.nodes)),
            "Num_Edges": int(len(fg.edges)),
        }
        if strict_case_oids is not None:
            case_row["Case_Object_ID"] = case_obj_id
        if cv_group_id is not None:
            case_row["CV_Group_ID"] = cv_group_id
        case_index_rows.append(case_row)

    feature_storage.feature_graphs = sorted_graphs
    feature_storage.event_features = list(canonical_feature_keys)
    # Explicitly guarantee no Stage 1 normalization or split state exists
    feature_storage._scaler = None
    feature_storage._training_indices = None
    feature_storage._test_indices = None

    case_index_df = pd.DataFrame(case_index_rows)
    if case_index_df.empty:
        raise ValueError("Feature extraction produced zero Feature_Graphs.")

    validate_feature_storage(feature_storage, case_index_df, config)
    elapsed = time.perf_counter() - t_start
    LOGGER.info(
        "Extracted unnormalized Feature_Storage (mode=native) in %.3fs: %d cases, %d events, %d feature keys.",
        elapsed,
        len(case_index_df),
        int(case_index_df["Num_Events"].sum()),
        len(feature_storage.event_features),
    )
    return feature_storage, case_index_df


def assert_feature_storages_equivalent(
    fs_native: Feature_Storage,
    case_index_native: pd.DataFrame,
    fs_fast: Feature_Storage,
    case_index_fast: pd.DataFrame,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Assert exact structural and numerical equivalence between native and fast extraction.

    Raises:
        AssertionError: On any mismatch in case index, graph topology, node objects,
            edge sets, feature keys, NaN masks, or floating-point values beyond tolerance.

    Returns:
        Summary dictionary with total values compared and maximum absolute difference.
    """
    if not case_index_native.equals(case_index_fast):
        raise AssertionError("case_index DataFrames differ between native and fast extraction.")
    if fs_native.event_features != fs_fast.event_features:
        raise AssertionError("event_features ordering/keys differ between native and fast extraction.")
    if len(fs_native.feature_graphs) != len(fs_fast.feature_graphs):
        raise AssertionError("Number of Feature_Graphs differs between native and fast extraction.")

    max_abs_diff = 0.0
    total_values_checked = 0
    total_nans_matched = 0

    for fg_n, fg_f in zip(fs_native.feature_graphs, fs_fast.feature_graphs):
        if fg_n.pexec_id != fg_f.pexec_id:
            raise AssertionError(f"pexec_id mismatch: {fg_n.pexec_id} vs {fg_f.pexec_id}")
        if fg_n.attributes != fg_f.attributes:
            raise AssertionError(
                f"Graph attributes mismatch on Case_ID={fg_n.pexec_id}: {fg_n.attributes} vs {fg_f.attributes}"
            )
        if len(fg_n.nodes) != len(fg_f.nodes):
            raise AssertionError(f"Node count mismatch on Case_ID={fg_n.pexec_id}")

        edges_n = sorted((e.source, e.target, frozenset(e.objects)) for e in fg_n.edges)
        edges_f = sorted((e.source, e.target, frozenset(e.objects)) for e in fg_f.edges)
        if edges_n != edges_f:
            raise AssertionError(f"Edge topology/objects mismatch on Case_ID={fg_n.pexec_id}")

        for node_n, node_f in zip(fg_n.nodes, fg_f.nodes):
            if node_n.event_id != node_f.event_id:
                raise AssertionError(
                    f"Event_ID ordering mismatch on Case_ID={fg_n.pexec_id}: "
                    f"{node_n.event_id} vs {node_f.event_id}"
                )
            if list(node_n.objects) != list(node_f.objects):
                raise AssertionError(
                    f"Node objects mismatch on Case_ID={fg_n.pexec_id}, Event_ID={node_n.event_id}"
                )
            keys_n = set(node_n.attributes.keys())
            keys_f = set(node_f.attributes.keys())
            if keys_n != keys_f:
                raise AssertionError(
                    f"Attribute key mismatch on Case_ID={fg_n.pexec_id}, Event_ID={node_n.event_id}: "
                    f"diff={keys_n ^ keys_f}"
                )
            for k in sorted(keys_n):
                vn = float(node_n.attributes[k])
                vf = float(node_f.attributes[k])
                total_values_checked += 1
                if np.isnan(vn) and np.isnan(vf):
                    total_nans_matched += 1
                    continue
                if np.isnan(vn) != np.isnan(vf):
                    raise AssertionError(
                        f"NaN mismatch on Case_ID={fg_n.pexec_id}, Event_ID={node_n.event_id}, "
                        f"key={k}: native={vn}, fast={vf}"
                    )
                diff = abs(vn - vf)
                if diff > max_abs_diff:
                    max_abs_diff = diff
                if not np.isclose(vn, vf, rtol=rtol, atol=atol):
                    raise AssertionError(
                        f"Value mismatch on Case_ID={fg_n.pexec_id}, Event_ID={node_n.event_id}, "
                        f"key={k}: native={vn}, fast={vf}, abs_diff={diff}"
                    )

    return {
        "cases_verified": len(fs_native.feature_graphs),
        "feature_keys_verified": len(fs_native.event_features),
        "total_values_checked": total_values_checked,
        "total_nans_matched": total_nans_matched,
        "max_abs_diff": max_abs_diff,
    }


def verify_native_vs_fast_equivalence(
    ocel: OCEL,
    config: dict[str, Any],
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Run both native and fast extractors on ``ocel`` and verify exact equivalence."""
    t0 = time.perf_counter()
    fs_native, idx_native = extract_feature_storage(ocel, config, extraction_mode="native")
    native_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    fs_fast, idx_fast = extract_feature_storage(ocel, config, extraction_mode="fast")
    fast_sec = time.perf_counter() - t0

    stats = assert_feature_storages_equivalent(
        fs_native=fs_native,
        case_index_native=idx_native,
        fs_fast=fs_fast,
        case_index_fast=idx_fast,
        rtol=rtol,
        atol=atol,
    )
    stats["native_seconds"] = round(native_sec, 4)
    stats["fast_seconds"] = round(fast_sec, 4)
    stats["speedup"] = round(native_sec / max(fast_sec, 1e-9), 2)
    LOGGER.info(
        "EXACT EQUIVALENCE VERIFIED: %d values checked (%d NaNs matched), "
        "max_abs_diff=%.3e | native=%.3fs vs fast=%.3fs (%.2fx speedup).",
        stats["total_values_checked"],
        stats["total_nans_matched"],
        stats["max_abs_diff"],
        stats["native_seconds"],
        stats["fast_seconds"],
        stats["speedup"],
    )
    return stats


def validate_feature_storage(
    feature_storage: Feature_Storage,
    case_index_df: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Validate Stage 1a ``Feature_Storage`` integrity, graph alignment, and anti-leakage."""
    case_col = config["identifiers"]["case_id"]
    target_col = config["target"]["column"]

    if feature_storage.scaler is not None:
        raise ValueError("Stage 1 Feature_Storage must NOT be normalized (scaler must be None).")
    if feature_storage.training_indices is not None or feature_storage.test_indices is not None:
        raise ValueError("Stage 1 Feature_Storage must NOT contain train/test split indices.")

    if len(feature_storage.feature_graphs) != len(case_index_df):
        raise ValueError(
            f"Feature_Storage graph count ({len(feature_storage.feature_graphs)}) "
            f"!= case_index row count ({len(case_index_df)})."
        )

    expected_keys = set(feature_storage.event_features) | {target_col}
    for row, fg in zip(case_index_df.itertuples(index=False), feature_storage.feature_graphs):
        if int(getattr(row, case_col)) != int(fg.pexec_id):
            raise ValueError(
                f"Case_ID mismatch between case_index ({getattr(row, case_col)}) "
                f"and Feature_Graph.pexec_id ({fg.pexec_id})."
            )
        if int(row.Num_Events) != len(fg.nodes):
            raise ValueError(f"Event count mismatch on Case_ID={fg.pexec_id}.")

        pos_by_eid = {n.event_id: i for i, n in enumerate(fg.nodes)}
        for edge in fg.edges:
            if edge.source not in pos_by_eid or edge.target not in pos_by_eid:
                raise ValueError(f"Dangling edge in Case_ID={fg.pexec_id}.")
            if pos_by_eid[edge.source] >= pos_by_eid[edge.target]:
                raise ValueError(f"Non-causal edge in Case_ID={fg.pexec_id}.")

        for node in fg.nodes:
            missing = expected_keys - set(node.attributes.keys())
            if missing:
                raise ValueError(
                    f"Node {node.event_id} in Case_ID={fg.pexec_id} missing keys: {sorted(missing)}"
                )
            y_val = node.attributes[target_col]
            if not np.isfinite(y_val) or y_val < 0.0:
                raise ValueError(
                    f"Invalid target value {y_val} on Case_ID={fg.pexec_id}, Event_ID={node.event_id}."
                )


def export_stage1_diagnostics(
    feature_storage: Feature_Storage,
    case_index_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Export human-readable Stage 1a diagnostic summaries (write-only; never read by Stage 2).

    Saves ``case_index.csv`` and ``stage1_feature_summary.csv`` under
    ``config["paths"]["preprocessed_dir"]``.
    """
    prep_dir = Path(config["paths"]["preprocessed_dir"])
    prep_dir.mkdir(parents=True, exist_ok=True)
    case_index_df.to_csv(prep_dir / "case_index.csv", index=False)

    target_col = config["target"]["column"]
    raw_table = pm_tabular.construct_table(feature_storage, index_list="all")
    all_keys = list(feature_storage.event_features)

    t1_set = set(resolve_tier_columns(all_keys, config, "tier_1"))
    t2_set = set(resolve_tier_columns(all_keys, config, "tier_2"))
    t3_set = set(resolve_tier_columns(all_keys, config, "tier_3"))

    summary_rows: list[dict[str, Any]] = []
    for col in [target_col, *all_keys]:
        vals = pd.to_numeric(raw_table[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        n_total = int(vals.size)
        n_valid = int(finite.size)
        n_nan = n_total - n_valid
        vmin = float(np.min(finite)) if n_valid else float("nan")
        vmean = float(np.mean(finite)) if n_valid else float("nan")
        vstd = float(np.std(finite, ddof=0)) if n_valid else float("nan")
        vmax = float(np.max(finite)) if n_valid else float("nan")
        summary_rows.append(
            {
                "Feature": col,
                "Role": "Target" if col == target_col else "Feature",
                "In_Tier_1": col in t1_set,
                "In_Tier_2": col in t2_set,
                "In_Tier_3": col in t3_set,
                "Non_Null_Count": n_valid,
                "NaN_Count": n_nan,
                "Min": vmin,
                "Mean": vmean,
                "Std": vstd,
                "Max": vmax,
                "Is_Constant": bool(n_valid > 0 and vstd == 0.0),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = prep_dir / "stage1_feature_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info(
        "Saved Stage 1a diagnostics to %s and %s (%d constant features).",
        prep_dir / "case_index.csv",
        summary_path,
        int(summary_df.loc[summary_df["Role"] == "Feature", "Is_Constant"].sum()),
    )
    return summary_df


def run_feature_extraction(
    config_path: str | Path = "config.yaml",
    debug_override: bool | None = None,
    extraction_mode: str | None = None,
    verify_equivalence: bool = False,
) -> tuple[Feature_Storage, pd.DataFrame]:
    """Run Stage 1a end-to-end and save ``feature_storage.pkl`` and ``case_index.parquet``.

    Args:
        config_path: Path to ``config.yaml``.
        debug_override: Optional boolean override for ``config["debug_mode"]``.
        extraction_mode: Optional override (``"fast"`` or ``"native"``).
        verify_equivalence: If ``True``, runs both native and fast extractors and
            asserts exact equivalence before saving.

    Returns:
        Tuple of ``(feature_storage, case_index_df)``.
    """
    config = load_config(config_path)
    if debug_override is not None:
        config["debug_mode"] = debug_override
    set_global_seed(int(config["random_seed"]))

    raw_path = discover_raw_ocel_path(config)
    ocel = load_ocel(raw_path, config)

    if verify_equivalence:
        verify_native_vs_fast_equivalence(ocel, config)

    feature_storage, case_index_df = extract_feature_storage(
        ocel, config, extraction_mode=extraction_mode
    )

    fs_out = Path(config["paths"]["feature_storage"])
    idx_out = Path(config["paths"]["case_index"])
    fs_out.parent.mkdir(parents=True, exist_ok=True)
    idx_out.parent.mkdir(parents=True, exist_ok=True)

    with fs_out.open("wb") as handle:
        pickle.dump(feature_storage, handle)
    case_index_df.to_parquet(idx_out, index=False)
    export_stage1_diagnostics(feature_storage, case_index_df, config)

    # Also persist a compact float32 cache (~15x lower RAM on load) for Stage 2
    import torch
    from src.ocpa_encoding import unpack_feature_storage_compact

    compact_out = fs_out.with_name("feature_storage_compact.pt")
    compact = unpack_feature_storage_compact(
        feature_storage,
        target_key=config["target"]["column"],
        consume_storage=False,
    )
    torch.save(compact, compact_out)

    LOGGER.info("Saved unnormalized Feature_Storage to %s", fs_out)
    LOGGER.info("Saved compact unnormalized Stage 1a cache to %s", compact_out)
    LOGGER.info("Saved case index metadata to %s", idx_out)
    return feature_storage, case_index_df


def main() -> None:
    """CLI entry point for Stage 1a feature extraction."""
    parser = argparse.ArgumentParser(description="Stage 1a: OCEL Feature Extraction via ocpa")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="Override config.yaml to enable debug_mode",
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "native"),
        default=None,
        help="Extraction engine override ('fast' O(N) prefix sweep or 'native' ocpa)",
    )
    parser.add_argument(
        "--verify-equivalence",
        action="store_true",
        help="Run both native and fast extractors and assert exact numerical equivalence",
    )
    args = parser.parse_args()
    configure_logging()
    run_feature_extraction(
        config_path=args.config,
        debug_override=args.debug,
        extraction_mode=args.mode,
        verify_equivalence=args.verify_equivalence,
    )


if __name__ == "__main__":
    main()
