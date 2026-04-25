"""
model.py — GraphSAGE model for root-cause classification (SPEC-T2 §7)

Architecture:
    2-layer GraphSAGE, 64 hidden, dual-head:
    - Classification head: N-way (root-cause service prediction)
    - Embedding head: 64-dim (input to serializer for blurb generation)

Input features: 24 dims per node (21 ServiceMetrics fields + 3 status one-hot).
Training device: CPU (per SPEC-T2 §7.4 — keeps GPU free for LLM).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from .adjacency import NUM_SERVICES


class GraphSAGEModel(nn.Module):
    """GraphSAGE with dual classification + embedding heads."""

    def __init__(
        self,
        in_channels: int = 24,
        hidden: int = 64,
        num_classes: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes is None:
            num_classes = NUM_SERVICES

        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

        # Classification head: hidden -> num_classes
        self.classifier = nn.Linear(hidden, num_classes)

        self._num_classes = num_classes
        self._hidden = hidden

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge index [2, num_edges]

        Returns:
            logits: Classification logits [num_nodes, num_classes]
            embeddings: Node embeddings [num_nodes, hidden]
        """
        # Layer 1
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)

        # Layer 2
        h = self.conv2(h, edge_index)
        embeddings = h  # 64-dim embeddings

        # Classification head
        logits = self.classifier(h)

        return logits, embeddings

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def hidden_dim(self) -> int:
        return self._hidden
