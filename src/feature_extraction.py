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
import logging
import pickle
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import ocpa.algo.predictive_monitoring.factory as pm_factory
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

from src import configure_logging, load_config, set_global_seed

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


def extract_feature_storage(
    ocel: OCEL,
    config: dict[str, Any],
) -> tuple[Feature_Storage, pd.DataFrame]:
    """Extract an unnormalized ``ocpa`` ``Feature_Storage`` and case index metadata.

    Calls ``ocpa.algo.predictive_monitoring.factory.apply(...)`` once to compute
    all Tier 1, Tier 2, and Tier 3 features plus ``EVENT_REMAINING_TIME``,
    converts feature tuple keys to canonical string keys for clean pickling and
    downstream tier filtering, enforces chronological node order and causal
    directed edges within every ``Feature_Graph``, and NEVER normalizes or splits.

    Args:
        ocel: Imported ``ocpa`` ``OCEL`` instance.
        config: Loaded configuration dictionary.

    Returns:
        Tuple of ``(feature_storage, case_index_df)`` where ``case_index_df``
        has columns ``[Graph_Index, Case_ID, Num_Events, Num_Edges]``.
    """
    if config.get("debug_mode", False):
        dbg = config.get("debug", {})
        ocel = _apply_debug_caps(
            ocel,
            max_cases=dbg.get("max_cases"),
            max_events_per_case=dbg.get("max_events_per_case"),
        )

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
        "Extracting ocpa Feature_Storage for %d cases | %d activities | "
        "object_types=%s | target_ot=%s | primary_attr=%s | unit=%s",
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
    LOGGER.info(
        "Extracted unnormalized Feature_Storage: %d cases, %d total events, %d feature keys.",
        len(case_index_df),
        int(case_index_df["Num_Events"].sum()),
        len(feature_storage.event_features),
    )
    return feature_storage, case_index_df


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


def run_feature_extraction(
    config_path: str | Path = "config.yaml",
    debug_override: bool | None = None,
) -> tuple[Feature_Storage, pd.DataFrame]:
    """Run Stage 1a end-to-end and save ``feature_storage.pkl`` and ``case_index.parquet``.

    Args:
        config_path: Path to ``config.yaml``.
        debug_override: Optional boolean override for ``config["debug_mode"]``.

    Returns:
        Tuple of ``(feature_storage, case_index_df)``.
    """
    config = load_config(config_path)
    if debug_override is not None:
        config["debug_mode"] = debug_override
    set_global_seed(int(config["random_seed"]))

    raw_path = discover_raw_ocel_path(config)
    ocel = load_ocel(raw_path, config)
    feature_storage, case_index_df = extract_feature_storage(ocel, config)

    fs_out = Path(config["paths"]["feature_storage"])
    idx_out = Path(config["paths"]["case_index"])
    fs_out.parent.mkdir(parents=True, exist_ok=True)
    idx_out.parent.mkdir(parents=True, exist_ok=True)

    with fs_out.open("wb") as handle:
        pickle.dump(feature_storage, handle)
    case_index_df.to_parquet(idx_out, index=False)

    LOGGER.info("Saved unnormalized Feature_Storage to %s", fs_out)
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
    args = parser.parse_args()
    configure_logging()
    run_feature_extraction(config_path=args.config, debug_override=args.debug)


if __name__ == "__main__":
    main()
