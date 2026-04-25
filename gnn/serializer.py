"""
serializer.py — GNN blurb generator for LLM prompt inclusion (SPEC-T2 §8)

Produces a text blurb per example from GNN classification logits and node embeddings.

Format:
    [Graph analysis]
    Root-cause probabilities: api-gateway=0.04, auth-service=0.81, ...
    Top-3 suspect services (by probability): auth-service (0.81), checkout-service (0.09), payment-service (0.05)
    Downstream blast-radius from top suspect: [checkout-service, payment-service, db-proxy]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .adjacency import ADJACENCY_DICT, SERVICE_NAMES


def _get_downstream(service: str, depth: int = 2) -> list[str]:
    """BFS to find downstream services from the dependency graph."""
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(service, 0)]
    result: list[str] = []

    while queue:
        current, d = queue.pop(0)
        if current in visited or d > depth:
            continue
        visited.add(current)
        if current != service:
            result.append(current)
        if d < depth:
            for dep in ADJACENCY_DICT.get(current, []):
                if dep not in visited:
                    queue.append((dep, d + 1))

    return result


def serialize_blurb(
    logits: torch.Tensor,
    service_names: tuple[str, ...] | None = None,
) -> str:
    """
    Generate a text blurb from GNN classification logits.

    Args:
        logits: Raw logits [num_services] — one value per service.
        service_names: Ordered service names matching logit indices.
                      Defaults to the canonical SERVICE_NAMES.

    Returns:
        Multi-line text blurb for LLM prompt inclusion.
    """
    if service_names is None:
        service_names = SERVICE_NAMES

    # Softmax to get probabilities
    probs = F.softmax(logits.detach().float(), dim=-1)

    # Build probability string
    prob_parts: list[str] = []
    for name, p in zip(service_names, probs.tolist()):
        prob_parts.append(f"{name}={p:.2f}")
    prob_line = ", ".join(prob_parts)

    # Top-3 suspects
    top_k = min(3, len(service_names))
    top_indices = torch.topk(probs, top_k).indices.tolist()
    top_parts: list[str] = []
    for idx in top_indices:
        top_parts.append(f"{service_names[idx]} ({probs[idx]:.2f})")
    top_line = ", ".join(top_parts)

    # Downstream blast-radius from top suspect
    top_service = service_names[top_indices[0]]
    downstream = _get_downstream(top_service)
    if downstream:
        blast_line = "[" + ", ".join(downstream) + "]"
    else:
        blast_line = "[none — leaf service]"

    blurb = (
        "[Graph analysis]\n"
        f"Root-cause probabilities: {prob_line}\n"
        f"Top-3 suspect services (by probability): {top_line}\n"
        f"Downstream blast-radius from top suspect: {blast_line}"
    )
    return blurb
