"""Common abstract model interface for all benchmark architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseOCPMModel(ABC):
    """Unified interface implemented by Tabular (XGBoost), Sequence (LSTM), and Graph (GNN) models."""

    def __init__(self, config: dict[str, Any], random_seed: int | None = None) -> None:
        """Initialize the model from repository configuration.

        Args:
            config: Loaded configuration dictionary from ``config.yaml``.
            random_seed: Optional seed override; defaults to ``config["random_seed"]``.
        """
        self.config = config
        self.random_seed = int(
            config["random_seed"] if random_seed is None else random_seed
        )
        self.debug_mode = bool(config.get("debug_mode", False))

    @abstractmethod
    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray | None = None,
        X_val: Any | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> BaseOCPMModel:
        """Fit the predictive model on fold-encoded training inputs.

        Args:
            X_train: Encoded training input (DataFrame for XGBoost, 3D array for
                LSTM, or list of PyG ``Data`` graphs for GNN).
            y_train: Target remaining times (optional for PyG graphs carrying ``data.y``).
            X_val: Optional encoded validation input for early stopping.
            y_val: Optional validation targets.
            **kwargs: Optional architecture-specific keyword arguments.

        Returns:
            Fitted model instance.
        """

    @abstractmethod
    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        """Predict event-level remaining times for encoded evaluation inputs.

        Args:
            X: Encoded evaluation input matching the architecture's format.
            **kwargs: Optional architecture-specific keyword arguments.

        Returns:
            1D float64 NumPy array of predictions.
        """
