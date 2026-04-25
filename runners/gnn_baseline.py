"""
gnn_baseline.py — Production-time GNN wrapper with graceful fallback.

Two ranking sources, selected at construction time:

  use_trained_gnn:
      Loads the existing GraphSAGEModel from ``gnn/`` (with optional
      checkpoint state_dict and Welford normalization stats). This is the
      *trained* path used after SFT + GNN training. Returns a ranked list
      AND a serialized text blurb that gets injected into the LLM prompt.

  untrained:
      Constructs the GraphSAGE model with random weights. The output
      ranking is not informative and on purpose: this is the honest
      starting baseline. Pairing this with the LLM should perform near
      the no-graph LLM baseline. The number is what the GRPO reward
      curve has to beat.

If torch / torch_geometric are unavailable, falls back to a deterministic
heuristic ranker (error_rate + latency + memory + downstream-blast-radius)
so that the runner stays usable on a stock python install. The blurb
format is identical so the LLM prompt is shape-preserving across
backends.

Returned blurb is a single string suitable for direct concatenation into
the user message — it never contains rewards, scores, or task hints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GnnRankItem:
    service: str
    score: float
    error_rate: float
    latency_p99: float
    memory_utilization: float
    downstream_blast_radius: int


def _safe_float(value, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return float(value)
    return default


def _downstream(name: str, dep_graph: dict) -> set[str]:
    if not dep_graph:
        return set()
    visited: set[str] = set()
    stack = list(dep_graph.get(name, []) or [])
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(dep_graph.get(node, []) or [])
    return visited


def _heuristic_rank(obs: dict, top_k: int = 5) -> list[GnnRankItem]:
    services = obs.get("services") or {}
    dep_graph = obs.get("dependency_graph") or {}
    items: list[GnnRankItem] = []
    for name, metrics in services.items():
        if not isinstance(metrics, dict):
            continue
        err = _safe_float(metrics.get("http_server_error_rate"))
        lat = _safe_float(metrics.get("http_server_request_duration_p99"))
        mem = _safe_float(metrics.get("process_memory_utilization"))
        load = _safe_float(metrics.get("http_server_active_requests"))
        downstream = _downstream(name, dep_graph)
        score = (
            (2.0 * err)
            + min(lat, 5.0)
            + mem
            + min(load / 500.0, 1.0)
            + 0.05 * len(downstream)
        )
        items.append(
            GnnRankItem(
                service=str(name),
                score=round(float(score), 4),
                error_rate=err,
                latency_p99=lat,
                memory_utilization=mem,
                downstream_blast_radius=len(downstream),
            )
        )
    items.sort(key=lambda item: item.score, reverse=True)
    return items[:top_k]


def _format_blurb(items: list[GnnRankItem], dep_graph: dict, top_k: int) -> str:
    if not items:
        return "[Graph analysis] No active services."
    suspect_lines = ", ".join(
        f"{item.service}={item.score:.2f}" for item in items[:top_k]
    )
    top = items[0]
    downstream = sorted(_downstream(top.service, dep_graph))
    radius = ", ".join(downstream) if downstream else "none — leaf"
    return (
        "[Graph analysis]\n"
        f"Top suspects (graph score, untrained-or-trained-GNN agnostic): {suspect_lines}\n"
        f"Top suspect: {top.service}  "
        f"(error_rate={top.error_rate:.2f}, latency_p99={top.latency_p99:.2f}s, "
        f"mem={top.memory_utilization:.2f})\n"
        f"Downstream blast-radius from top suspect: [{radius}]"
    )


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class GnnBaseline:
    """Wrapper that produces a per-service ranking + text blurb."""

    def __init__(
        self,
        mode: str = "heuristic",
        checkpoint_path: Path | str | None = None,
        normalization_path: Path | str | None = None,
        top_k: int = 5,
    ) -> None:
        valid = {"heuristic", "untrained", "from_checkpoint"}
        if mode not in valid:
            raise ValueError(f"mode must be one of {sorted(valid)}; got {mode!r}")
        self.mode = mode
        self.top_k = top_k
        self._gnn_model = None
        self._normalizer = None
        self._edge_index = None
        self._extract_features = None
        self._service_names: tuple[str, ...] = ()

        if mode in {"untrained", "from_checkpoint"}:
            self._try_load_gnn(checkpoint_path, normalization_path)

    def _try_load_gnn(
        self,
        checkpoint_path: Path | str | None,
        normalization_path: Path | str | None,
    ) -> None:
        try:
            import torch
            from gnn.adjacency import EDGE_INDEX, NUM_SERVICES, SERVICE_NAMES
            from gnn.model import GraphSAGEModel
            from gnn.train_gnn import (
                NUM_FEATURES,
                WelfordNormalizer,
                extract_node_features,
            )
        except Exception:
            self.mode = "heuristic"
            return

        model = GraphSAGEModel(in_channels=NUM_FEATURES, num_classes=NUM_SERVICES)
        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.exists():
                state_dict = torch.load(
                    str(ckpt_path), map_location="cpu", weights_only=True
                )
                model.load_state_dict(state_dict, strict=False)
        model.eval()
        self._gnn_model = model

        normalizer = WelfordNormalizer(NUM_FEATURES)
        if normalization_path is not None:
            norm_path = Path(normalization_path)
            if norm_path.exists():
                import json as _json

                normalizer = WelfordNormalizer.from_dict(
                    _json.loads(norm_path.read_text())
                )
        self._normalizer = normalizer
        self._edge_index = EDGE_INDEX
        self._extract_features = extract_node_features
        self._service_names = SERVICE_NAMES

    def rank(self, obs: dict) -> list[GnnRankItem]:
        """Return the top-K ranked services for this observation."""
        if self._gnn_model is None:
            return _heuristic_rank(obs, top_k=self.top_k)

        import torch

        example = {"observation": {"service_metrics": obs.get("services", {})}}
        features = self._extract_features(example, self._normalizer)
        with torch.no_grad():
            logits, _embeddings = self._gnn_model(features, self._edge_index)
            graph_logits = logits.mean(dim=0)
            probabilities = torch.softmax(graph_logits, dim=-1).tolist()

        services = obs.get("services") or {}
        dep_graph = obs.get("dependency_graph") or {}
        ranked: list[GnnRankItem] = []
        for name, prob in zip(self._service_names, probabilities):
            metrics = services.get(name) or {}
            ranked.append(
                GnnRankItem(
                    service=str(name),
                    score=round(float(prob), 4),
                    error_rate=_safe_float(metrics.get("http_server_error_rate")),
                    latency_p99=_safe_float(
                        metrics.get("http_server_request_duration_p99")
                    ),
                    memory_utilization=_safe_float(
                        metrics.get("process_memory_utilization")
                    ),
                    downstream_blast_radius=len(_downstream(str(name), dep_graph)),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: self.top_k]

    def blurb(self, obs: dict, ranked: list[GnnRankItem] | None = None) -> str:
        """Return the text blurb for inclusion in the prompt."""
        items = ranked if ranked is not None else self.rank(obs)
        return _format_blurb(items, obs.get("dependency_graph") or {}, self.top_k)
