"""PyTorch Geometric (PyG) causal GNN for OCPM Remaining Time Prediction.

Consumes PyG ``Data`` graph lists produced directly by
``src.ocpa_encoding.encode_pyg_graphs(...)`` from ``ocpa`` ``Feature_Storage``
graphs, using causal directed message passing (``SAGEConv`` by default) with
early stopping and best-checkpoint restoration on Sub-Val.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from src import set_global_seed
from src.models.base import BaseOCPMModel

LOGGER = logging.getLogger(__name__)


class _CausalOCPMGNN(nn.Module):
    """Multi-layer PyTorch Geometric node-level regression network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        architecture: str = "sage",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        arch = architecture.lower()
        in_dim = input_dim
        for _ in range(max(1, num_layers)):
            if arch == "sage":
                # SAGEConv aggregates strictly incoming causal predecessors (source_to_target)
                self.convs.append(SAGEConv(in_dim, hidden_dim, aggr="mean"))
            elif arch == "gcn":
                self.convs.append(GCNConv(in_dim, hidden_dim, add_self_loops=True))
            elif arch == "gat":
                self.convs.append(GATConv(in_dim, hidden_dim, heads=1, concat=False))
            else:
                raise ValueError(
                    f"Unsupported GNN architecture '{architecture}'. Choose 'sage', 'gcn', or 'gat'."
                )
            in_dim = hidden_dim

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute node-level predictions from node features and directed edges."""
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            if self.dropout > 0.0:
                h = torch.dropout(h, p=self.dropout, train=self.training)
        return self.head(h).squeeze(-1)


class GraphGNNModel(BaseOCPMModel):
    """PyTorch Geometric GNN regressor with early stopping on Sub-Val."""

    def __init__(self, config: dict[str, Any], random_seed: int | None = None) -> None:
        """Initialize GNN hyperparameters from ``config.yaml``.

        Args:
            config: Loaded configuration dictionary.
            random_seed: Optional random seed override.
        """
        super().__init__(config=config, random_seed=random_seed)
        gnn_cfg = config.get("models", {}).get("gnn", {})
        self.architecture = str(gnn_cfg.get("architecture", "sage"))
        self.hidden_dim = int(gnn_cfg.get("hidden_dim", 64))
        self.num_layers = int(gnn_cfg.get("num_layers", 2))
        self.dropout = float(gnn_cfg.get("dropout", 0.2))
        self.learning_rate = float(gnn_cfg.get("learning_rate", 1e-3))
        self.batch_size = int(gnn_cfg.get("batch_size", 4))
        self.epochs = (
            int(config.get("debug", {}).get("gnn_epochs", 2))
            if self.debug_mode
            else int(gnn_cfg.get("epochs", 50))
        )
        self.patience = int(gnn_cfg.get("patience", 5))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net: _CausalOCPMGNN | None = None

    def fit(
        self,
        X_train: list[Data],
        y_train: np.ndarray | None = None,
        X_val: list[Data] | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> GraphGNNModel:
        """Train the PyG GNN on Sub-Train graphs with early stopping on Sub-Val graphs.

        Args:
            X_train: List of PyG ``Data(x, edge_index, y)`` graphs from ``encode_pyg_graphs``.
            y_train: Optional unused argument (node targets are stored in ``data.y``).
            X_val: Optional list of validation PyG ``Data`` graphs for early stopping.
            y_val: Optional unused argument.
            **kwargs: Unused extra arguments for interface compatibility.

        Returns:
            Fitted ``GraphGNNModel`` restored to best validation checkpoint.
        """
        if not X_train:
            raise ValueError("GraphGNNModel.fit requires at least one training PyG Data graph.")
        set_global_seed(self.random_seed)

        train_loader = PyGDataLoader(
            X_train,
            batch_size=min(self.batch_size, len(X_train)),
            shuffle=True,
        )
        val_loader: PyGDataLoader | None = None
        if X_val is not None and len(X_val) > 0:
            val_loader = PyGDataLoader(
                X_val,
                batch_size=min(self.batch_size, len(X_val)),
                shuffle=False,
            )

        input_dim = int(X_train[0].x.shape[-1])
        self._net = _CausalOCPMGNN(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            architecture=self.architecture,
        ).to(self.device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.L1Loss()

        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        no_improve = 0

        for epoch in range(self.epochs):
            self._net.train()
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                preds = self._net(batch.x, batch.edge_index)
                loss = criterion(preds, batch.y)
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                self._net.eval()
                val_abs_err = 0.0
                val_count = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        preds = self._net(batch.x, batch.edge_index)
                        val_abs_err += float(torch.abs(preds - batch.y).sum().item())
                        val_count += int(batch.y.numel())
                val_mae = val_abs_err / max(val_count, 1)

                if val_mae < best_val_loss:
                    best_val_loss = val_mae
                    best_state = copy.deepcopy(self._net.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        LOGGER.debug(
                            "GNN early stopped at epoch %d (best val MAE=%.6f).",
                            epoch + 1,
                            best_val_loss,
                        )
                        break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict(self, X: list[Data], **kwargs: Any) -> np.ndarray:
        """Predict node-level remaining times concatenated across PyG ``Data`` graphs.

        Args:
            X: List of PyG ``Data`` objects from ``encode_pyg_graphs``.
            **kwargs: Unused extra arguments for interface compatibility.

        Returns:
            1D float64 NumPy array aligned with the node concatenation order of ``X``.
        """
        if self._net is None:
            raise RuntimeError("GraphGNNModel must be fitted before calling predict().")

        self._net.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for data in X:
                graph = data.to(self.device)
                preds = self._net(graph.x, graph.edge_index).cpu().numpy()
                chunks.append(preds)

        if not chunks:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(chunks, axis=0).astype(np.float64)
