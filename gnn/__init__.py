"""
gnn/ — GNN subpackage for FirewatchEnv training pipeline (SPEC-T2)

GraphSAGE model for service-graph root-cause classification.
Adjacency sourced from FirewatchEnv's comprehensive service topology.
"""

from .adjacency import (
    SERVICE_NAMES,
    SERVICE_TO_IDX,
    NUM_SERVICES,
    EDGE_INDEX,
    ADJACENCY_DICT,
)
from .model import GraphSAGEModel
from .serializer import serialize_blurb

__all__ = [
    "SERVICE_NAMES",
    "SERVICE_TO_IDX",
    "NUM_SERVICES",
    "EDGE_INDEX",
    "ADJACENCY_DICT",
    "GraphSAGEModel",
    "serialize_blurb",
]
