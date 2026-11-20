"""OCPM Predictive Process Monitoring Benchmark package.

Provides shared configuration loading, deterministic seeding, and tier-column
resolution utilities used across Stage 1, Stage 2, and Stage 3 modules.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a consistent timestamped format.

    Args:
        level: Logging level (e.g., ``logging.INFO`` or ``logging.DEBUG``).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load and validate the benchmark YAML configuration file.

    Args:
        config_path: Path to ``config.yaml``.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If the YAML file does not parse to a mapping.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file {path} must contain a YAML mapping.")
    return cfg


def set_global_seed(seed: int) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer random seed from ``config.yaml``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        LOGGER.debug("PyTorch not available while setting global seed.")


def resolve_tier_columns(
    available_columns: list[str],
    config: dict[str, Any],
    tier_name: str,
) -> list[str]:
    """Resolve the ordered list of columns belonging to a feature tier.

    Matches both explicit column names and configured dynamic column prefixes
    (e.g., ``C2_preceding_act_``, ``C3_prev_act_count_``) present in the
    extracted feature table.

    Args:
        available_columns: Columns present in ``extracted_features.parquet``.
        config: Loaded configuration dictionary.
        tier_name: One of ``"tier_1"``, ``"tier_2"``, or ``"tier_3"``.

    Returns:
        Deduplicated list of column names preserving deterministic order.

    Raises:
        KeyError: If ``tier_name`` is not configured under ``config["features"]``.
    """
    tier_cfg = config["features"][tier_name]
    explicit_cols: list[str] = list(tier_cfg.get("columns", []))
    prefixes: list[str] = list(tier_cfg.get("prefixes", []))

    available_set = set(available_columns)
    resolved: list[str] = []
    seen: set[str] = set()

    for col in explicit_cols:
        if col in available_set and col not in seen:
            resolved.append(col)
            seen.add(col)

    for prefix in prefixes:
        matching = sorted(col for col in available_columns if col.startswith(prefix))
        for col in matching:
            if col not in seen:
                resolved.append(col)
                seen.add(col)

    return resolved
