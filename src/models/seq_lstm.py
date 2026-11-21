"""PyTorch causal prefix/sliding-window LSTM for OCPM Remaining Time Prediction.

Consumes 3D prefix sequence tensors ``(N, window_size, F)`` produced directly by
``src.ocpa_encoding.encode_sequential(...)`` (backed by ``ocpa``'s
``sequential.construct_sequence``) and trains with early stopping and
best-checkpoint restoration on Sub-Val.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import set_global_seed
from src.models.base import BaseOCPMModel

LOGGER = logging.getLogger(__name__)


class _PrefixLSTMNet(nn.Module):
    """Multi-layer LSTM regressor mapping causal event prefixes to scalar targets."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over padded prefix batch ``(B, W, F)``."""
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


class SequenceLSTMModel(BaseOCPMModel):
    """PyTorch prefix LSTM regressor with early stopping on Sub-Val."""

    def __init__(self, config: dict[str, Any], random_seed: int | None = None) -> None:
        """Initialize LSTM hyperparameters from ``config.yaml``.

        Args:
            config: Loaded configuration dictionary.
            random_seed: Optional random seed override.
        """
        super().__init__(config=config, random_seed=random_seed)
        lstm_cfg = config.get("models", {}).get("lstm", {})
        self.hidden_dim = int(lstm_cfg.get("hidden_dim", 64))
        self.num_layers = int(lstm_cfg.get("num_layers", 2))
        self.dropout = float(lstm_cfg.get("dropout", 0.2))
        self.batch_size = int(lstm_cfg.get("batch_size", 64))
        self.learning_rate = float(lstm_cfg.get("learning_rate", 1e-3))
        self.epochs = (
            int(config.get("debug", {}).get("lstm_epochs", 2))
            if self.debug_mode
            else int(lstm_cfg.get("epochs", 50))
        )
        self.patience = int(lstm_cfg.get("patience", 5))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net: _PrefixLSTMNet | None = None
        self.training_history: list[dict[str, Any]] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> SequenceLSTMModel:
        """Train the prefix LSTM on Sub-Train sequences with early stopping on Sub-Val.

        Args:
            X_train: 3D float array of shape ``(N_train, W, F)`` from ``encode_sequential``.
            y_train: 1D target array of shape ``(N_train,)``.
            X_val: Optional 3D validation array ``(N_val, W, F)``.
            y_val: Optional 1D validation target array ``(N_val,)``.
            **kwargs: Unused extra arguments for interface compatibility.

        Returns:
            Fitted ``SequenceLSTMModel`` restored to best validation checkpoint.
        """
        if y_train is None:
            raise ValueError("SequenceLSTMModel.fit requires y_train.")
        set_global_seed(self.random_seed)

        x_tr_t = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
        y_tr_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
        train_ds = TensorDataset(x_tr_t, y_tr_t)
        generator = torch.Generator().manual_seed(self.random_seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        val_loader: DataLoader | None = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            x_va_t = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
            y_va_t = torch.from_numpy(np.asarray(y_val, dtype=np.float32))
            val_loader = DataLoader(
                TensorDataset(x_va_t, y_va_t),
                batch_size=self.batch_size,
                shuffle=False,
            )

        input_dim = int(x_tr_t.shape[-1])
        self._net = _PrefixLSTMNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.L1Loss()

        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        no_improve = 0
        self.training_history = []

        for epoch in range(self.epochs):
            self._net.train()
            train_abs_err = 0.0
            train_count = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self._net(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                train_abs_err += float(torch.abs(preds.detach() - batch_y).sum().item())
                train_count += int(batch_y.numel())
            train_mae = train_abs_err / max(train_count, 1)

            val_mae: float | None = None
            is_best = False
            if val_loader is not None:
                self._net.eval()
                val_abs_err = 0.0
                val_count = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        preds = self._net(batch_x)
                        val_abs_err += float(torch.abs(preds - batch_y).sum().item())
                        val_count += int(batch_y.numel())
                val_mae = val_abs_err / max(val_count, 1)

                if val_mae < best_val_loss:
                    best_val_loss = val_mae
                    best_state = copy.deepcopy(self._net.state_dict())
                    no_improve = 0
                    is_best = True
                else:
                    no_improve += 1

            self.training_history.append(
                {
                    "Epoch": epoch + 1,
                    "Train_MAE": float(train_mae),
                    "Val_MAE": float(val_mae) if val_mae is not None else float("nan"),
                    "Is_Best_Epoch": is_best,
                }
            )

            if val_loader is not None and no_improve >= self.patience:
                LOGGER.debug(
                    "LSTM early stopped at epoch %d (best val MAE=%.6f).",
                    epoch + 1,
                    best_val_loss,
                )
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict event-level remaining times for 3D prefix sequences ``X``.

        Args:
            X: 3D float array of shape ``(N, W, F)`` from ``encode_sequential``.
            **kwargs: Unused extra arguments for interface compatibility.

        Returns:
            1D float64 NumPy array of length ``N``.
        """
        if self._net is None:
            raise RuntimeError("SequenceLSTMModel must be fitted before calling predict().")

        x_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        loader = DataLoader(TensorDataset(x_t), batch_size=self.batch_size, shuffle=False)

        self._net.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                chunks.append(self._net(batch_x).cpu().numpy())

        if not chunks:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(chunks, axis=0).astype(np.float64)
