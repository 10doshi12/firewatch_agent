"""
adjacency.py — FirewatchEnv service topology for GNN training

Sources from firewatch_env.config: ALL_SERVICES and FULL_DEPENDENCY_GRAPH.
The adjacency is static — it does not change across batches.
Node order is sorted alphabetically for determinism.

Exports:
    SERVICE_NAMES: tuple[str, ...]  — sorted canonical service names
    SERVICE_TO_IDX: dict[str, int]  — service name → node index
    NUM_SERVICES: int               — total number of services
    EDGE_INDEX: torch.LongTensor    — [2, num_edges] COO format
    ADJACENCY_DICT: dict[str, list[str]] — raw adjacency from config
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Import FirewatchEnv config — handle both installed and path-based access
# ---------------------------------------------------------------------------

_env_root = Path(__file__).resolve().parent.parent.parent / "firewatch_env"
if str(_env_root.parent) not in sys.path:
    sys.path.insert(0, str(_env_root.parent))

try:
    from firewatch_env.config import ALL_SERVICES, FULL_DEPENDENCY_GRAPH
except ImportError:
    # Fallback: import directly from the file
    sys.path.insert(0, str(_env_root))
    from config import ALL_SERVICES, FULL_DEPENDENCY_GRAPH  # type: ignore[import]

# ---------------------------------------------------------------------------
# Canonical service ordering (sorted alphabetically for determinism)
# ---------------------------------------------------------------------------

SERVICE_NAMES: tuple[str, ...] = tuple(sorted(ALL_SERVICES))
SERVICE_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(SERVICE_NAMES)}
NUM_SERVICES: int = len(SERVICE_NAMES)

# ---------------------------------------------------------------------------
# Adjacency dict (raw)
# ---------------------------------------------------------------------------

ADJACENCY_DICT: dict[str, list[str]] = dict(FULL_DEPENDENCY_GRAPH)

# ---------------------------------------------------------------------------
# Edge index in COO format for PyTorch Geometric
# ---------------------------------------------------------------------------


def _build_edge_index() -> torch.LongTensor:
    """Build bidirectional edge index from the dependency graph."""
    src_list: list[int] = []
    dst_list: list[int] = []

    for service, deps in FULL_DEPENDENCY_GRAPH.items():
        if service not in SERVICE_TO_IDX:
            continue
        src_idx = SERVICE_TO_IDX[service]
        for dep in deps:
            if dep not in SERVICE_TO_IDX:
                continue
            dst_idx = SERVICE_TO_IDX[dep]
            # Forward edge: caller -> callee
            src_list.append(src_idx)
            dst_list.append(dst_idx)
            # Reverse edge: callee -> caller (message passing needs both)
            src_list.append(dst_idx)
            dst_list.append(src_idx)

    # Add self-loops for every node (GraphSAGE benefits from self-loops)
    for idx in range(NUM_SERVICES):
        src_list.append(idx)
        dst_list.append(idx)

    return torch.tensor([src_list, dst_list], dtype=torch.long)


EDGE_INDEX: torch.LongTensor = _build_edge_index()
