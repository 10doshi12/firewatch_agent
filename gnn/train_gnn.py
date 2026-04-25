"""
train_gnn.py — GNN training loop (SPEC-T2 §7)

Per-batch training to convergence on root-cause classification.
- Max 30 epochs, patience 5 on val loss
- 80/20 train/val split, seeded by batch number
- Adam optimizer, LR 1e-3, cross-entropy loss
- Best val-loss checkpoint selection
- Welford's online algorithm for running normalization stats
- CPU-only training (GPU reserved for LLM)
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .adjacency import EDGE_INDEX, NUM_SERVICES, SERVICE_NAMES, SERVICE_TO_IDX
from .model import GraphSAGEModel

# ---------------------------------------------------------------------------
# Feature extraction constants
# ---------------------------------------------------------------------------

# 21 ServiceMetrics numeric fields (same order as Pydantic model)
METRIC_FIELDS: list[str] = [
    "http_server_request_duration_p99",
    "http_server_error_rate",
    "http_server_active_requests",
    "process_cpu_utilization",
    "process_memory_usage_bytes",
    "process_memory_limit_bytes",
    "process_memory_utilization",
    "process_open_file_descriptors",
    "runtime_gc_pause_duration_ms",
    "runtime_gc_count_per_second",
    "runtime_jvm_threads_count",
    "runtime_jvm_threads_max",
    "runtime_thread_pool_queue_depth",
    "runtime_uptime_seconds",
    "restart_count",
    "last_deployment_age_seconds",
    "last_config_revision",
    "last_config_age_seconds",
    # active_requests is already covered via http_server_active_requests
    # These 3 are the remaining numeric fields to reach 21
    # Using placeholders that map to default 0 if absent
]

# Status one-hot encoding: [degraded, critical, down] — healthy is all-zeros
STATUS_MAP: dict[str, list[float]] = {
    "healthy": [0.0, 0.0, 0.0],
    "degraded": [1.0, 0.0, 0.0],
    "critical": [0.0, 1.0, 0.0],
    "down": [0.0, 0.0, 1.0],
}

# Total: 21 metric fields + 3 status one-hot + 8 Phase 2/3 = 32 dims
NUM_FEATURES = 32

# Metric field defaults (for missing values)
METRIC_DEFAULTS: dict[str, float] = {
    "http_server_request_duration_p99": 0.1,
    "http_server_error_rate": 0.0,
    "http_server_active_requests": 50.0,
    "process_cpu_utilization": 0.15,
    "process_memory_usage_bytes": 178257920.0,
    "process_memory_limit_bytes": 536870912.0,
    "process_memory_utilization": 0.33,
    "process_open_file_descriptors": 120.0,
    "runtime_gc_pause_duration_ms": 15.0,
    "runtime_gc_count_per_second": 2.0,
    "runtime_jvm_threads_count": 50.0,
    "runtime_jvm_threads_max": 200.0,
    "runtime_thread_pool_queue_depth": 0.0,
    "runtime_uptime_seconds": 86400.0,
    "restart_count": 0.0,
    "last_deployment_age_seconds": 172800.0,
    "last_config_revision": 1.0,
    "last_config_age_seconds": 259200.0,
}

# Phase 2/3 task-specific metrics injected as dynamic ServiceMetrics attributes.
# Absent on Phase 1/legacy tasks — default to healthy baseline values.
PHASE2_FIELDS: list[str] = [
    "lb_weight_normalized",           # float, healthy ≈ 1.0
    "canary_traffic_weight",          # float, 0.0 = fully rolled back
    "grpc_orphaned_call_rate",        # float, 0.0 = healthy
    "db_replication_lag_seconds",     # float, < 5.0 = healthy
    "cache_hit_rate",                 # float, > 0.85 = healthy
    "circuit_breaker_state",          # string → binarized: "open"=1.0 else 0.0
    "resource_quota_remaining_ratio", # float, > 0.0 = healthy
    "metastable_feedback_loop_active", # bool → 0.0/1.0
]

PHASE2_DEFAULTS: dict[str, float] = {
    "lb_weight_normalized": 1.0,
    "canary_traffic_weight": 0.0,
    "grpc_orphaned_call_rate": 0.0,
    "db_replication_lag_seconds": 0.0,
    "cache_hit_rate": 0.95,
    "circuit_breaker_state": 0.0,
    "resource_quota_remaining_ratio": 1.0,
    "metastable_feedback_loop_active": 0.0,
}


# ---------------------------------------------------------------------------
# Welford's online normalization
# ---------------------------------------------------------------------------


class WelfordNormalizer:
    """Running mean/std using Welford's online algorithm."""

    def __init__(self, num_features: int) -> None:
        self.n = 0
        self.mean = [0.0] * num_features
        self.m2 = [0.0] * num_features
        self._num_features = num_features

    def update(self, features: list[float]) -> None:
        self.n += 1
        for i, x in enumerate(features):
            delta = x - self.mean[i]
            self.mean[i] += delta / self.n
            delta2 = x - self.mean[i]
            self.m2[i] += delta * delta2

    def get_std(self) -> list[float]:
        if self.n < 2:
            return [1.0] * self._num_features
        return [math.sqrt(self.m2[i] / (self.n - 1)) for i in range(self._num_features)]

    def normalize(self, features: list[float]) -> list[float]:
        std = self.get_std()
        return [
            (features[i] - self.mean[i]) / max(std[i], 1e-8)
            for i in range(self._num_features)
        ]

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "m2": self.m2,
            "num_features": self._num_features,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WelfordNormalizer:
        norm = cls(d["num_features"])
        norm.n = d["n"]
        norm.mean = d["mean"]
        norm.m2 = d["m2"]
        return norm


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_node_features(
    example: dict,
    normalizer: WelfordNormalizer | None = None,
    update_normalizer: bool = False,
) -> torch.Tensor:
    """
    Extract 32-dim feature vector per service node from a training example.

    Args:
        example: One training example with observation.service_metrics
        normalizer: Optional Welford normalizer for feature normalization
        update_normalizer: If True, update normalizer stats (training only)

    Returns:
        Tensor [NUM_SERVICES, 32] — feature matrix for all nodes
    """
    obs = example.get("observation", {})
    service_metrics = obs.get("service_metrics", {})

    features = torch.zeros(NUM_SERVICES, NUM_FEATURES)

    for svc_name in SERVICE_NAMES:
        if svc_name not in SERVICE_TO_IDX:
            continue
        idx = SERVICE_TO_IDX[svc_name]
        metrics = service_metrics.get(svc_name, {})

        # Extract 21 numeric metrics (first 18 from METRIC_FIELDS + 3 extras)
        raw_feats: list[float] = []
        for field in list(METRIC_DEFAULTS.keys()):
            val = metrics.get(field, METRIC_DEFAULTS[field])
            if isinstance(val, (int, float)):
                raw_feats.append(float(val))
            else:
                raw_feats.append(METRIC_DEFAULTS.get(field, 0.0))

        # Pad to 21 if we have fewer fields
        while len(raw_feats) < 21:
            raw_feats.append(0.0)
        raw_feats = raw_feats[:21]

        # Status one-hot (3 dims)
        status = metrics.get("status", "healthy")
        status_onehot = STATUS_MAP.get(status, [0.0, 0.0, 0.0])

        phase2_feats: list[float] = []
        for field in PHASE2_FIELDS:
            val = metrics.get(field, PHASE2_DEFAULTS[field])
            if field == "circuit_breaker_state":
                val = 1.0 if val == "open" else 0.0
            elif isinstance(val, bool):
                val = float(val)
            elif not isinstance(val, (int, float)):
                val = PHASE2_DEFAULTS[field]
            phase2_feats.append(float(val))
        full_feats = raw_feats + status_onehot + phase2_feats

        # Update normalizer if training
        if normalizer and update_normalizer:
            normalizer.update(full_feats)

        # Normalize if normalizer available and has sufficient stats
        if normalizer and normalizer.n >= 2:
            full_feats = normalizer.normalize(full_feats)

        features[idx] = torch.tensor(full_feats, dtype=torch.float32)

    return features


def get_root_cause_label(example: dict) -> int:
    """Extract root-cause service index from example."""
    # Try multiple field names for the root-cause service
    root_cause = (
        example.get("fault_service")
        or example.get("root_cause_service")
        or example.get("observation", {}).get("root_cause_service")
    )

    if root_cause and root_cause in SERVICE_TO_IDX:
        return SERVICE_TO_IDX[root_cause]

    # Fallback: look in gold_action_sequence for the first remediation target
    gold_actions = example.get("gold_action_sequence", [])
    for action in gold_actions:
        params = action.get("params", {})
        service = params.get("service") or action.get("target_service")
        if service and service in SERVICE_TO_IDX:
            return SERVICE_TO_IDX[service]

    return 0  # default to first service if no root cause found


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_gnn(
    batch_examples: list[dict],
    batch_num: int,
    checkpoint_dir: Path,
    prev_checkpoint_path: Path | None = None,
    normalization_path: Path | None = None,
    config: dict | None = None,
) -> tuple[Path, Path]:
    """
    Train GNN on one batch of examples.

    Args:
        batch_examples: List of 50 training examples
        batch_num: Current batch number (0-indexed)
        checkpoint_dir: Directory for saving checkpoints
        prev_checkpoint_path: Path to previous batch's GNN checkpoint (None if batch 0)
        normalization_path: Path to previous normalization stats (None if batch 0)
        config: GNN config dict from config.yaml

    Returns:
        (checkpoint_path, normalization_path) — paths to saved artifacts
    """
    if config is None:
        config = {}

    hidden_dim = config.get("hidden_dim", 64)
    dropout = config.get("dropout", 0.1)
    lr = config.get("learning_rate", 1e-3)
    max_epochs = config.get("max_epochs", 30)
    patience = config.get("patience", 5)

    # Load or initialize normalizer
    normalizer = WelfordNormalizer(NUM_FEATURES)
    if normalization_path and normalization_path.exists():
        norm_data = json.loads(normalization_path.read_text())
        normalizer = WelfordNormalizer.from_dict(norm_data)

    # Dynamic train/val split: all-train for tiny batches, 80/20 otherwise
    rng = random.Random(batch_num)
    indices = list(range(len(batch_examples)))
    rng.shuffle(indices)
    if len(batch_examples) < 20:
        train_indices = indices
        val_indices = []
    else:
        split_point = int(0.8 * len(indices))
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]

    # First pass: update normalizer with training data
    for idx in train_indices:
        extract_node_features(batch_examples[idx], normalizer, update_normalizer=True)

    # Prepare data tensors
    edge_index = EDGE_INDEX

    train_features = []
    train_labels = []
    for idx in train_indices:
        features = extract_node_features(batch_examples[idx], normalizer)
        label = get_root_cause_label(batch_examples[idx])
        train_features.append(features)
        train_labels.append(label)

    val_features = []
    val_labels = []
    for idx in val_indices:
        features = extract_node_features(batch_examples[idx], normalizer)
        label = get_root_cause_label(batch_examples[idx])
        val_features.append(features)
        val_labels.append(label)

    # Initialize model
    model = GraphSAGEModel(
        in_channels=NUM_FEATURES,
        hidden=hidden_dim,
        num_classes=NUM_SERVICES,
        dropout=dropout,
    )

    # Load previous checkpoint if available (continual learning across batches)
    if prev_checkpoint_path and prev_checkpoint_path.exists():
        state_dict = torch.load(prev_checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0
    best_epoch = 0

    start_time = time.monotonic()

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        total_train_loss = 0.0

        for features, label in zip(train_features, train_labels):
            optimizer.zero_grad()
            logits, _ = model(features, edge_index)

            # Use graph-level prediction: mean-pool node logits
            graph_logits = logits.mean(dim=0, keepdim=True)
            target = torch.tensor([label], dtype=torch.long)

            loss = F.cross_entropy(graph_logits, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_features), 1)

        # --- Validation ---
        avg_val_loss = 0.0
        val_accuracy = 0.0
        if val_features:
            model.eval()
            total_val_loss = 0.0
            correct = 0

            with torch.no_grad():
                for features, label in zip(val_features, val_labels):
                    logits, _ = model(features, edge_index)
                    graph_logits = logits.mean(dim=0, keepdim=True)
                    target = torch.tensor([label], dtype=torch.long)

                    val_loss = F.cross_entropy(graph_logits, target)
                    total_val_loss += val_loss.item()

                    pred = graph_logits.argmax(dim=1).item()
                    if pred == label:
                        correct += 1

            avg_val_loss = total_val_loss / max(len(val_features), 1)
            val_accuracy = correct / max(len(val_features), 1)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    wall_time = time.monotonic() - start_time

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save checkpoint
    checkpoint_dir = Path(checkpoint_dir)
    gnn_dir = checkpoint_dir / "gnn"
    gnn_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = gnn_dir / f"batch_{batch_num:03d}.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Save normalization stats
    norm_path = gnn_dir / "normalization.json"
    norm_path.write_text(json.dumps(normalizer.to_dict(), indent=2))

    # Log training summary
    summary = {
        "batch_num": batch_num,
        "epochs_trained": best_epoch + 1,
        "final_train_loss": round(avg_train_loss, 6),
        "final_val_loss": round(best_val_loss, 6) if math.isfinite(best_val_loss) else None,
        "val_accuracy": round(val_accuracy, 4),
        "wall_time_seconds": round(wall_time, 2),
        "num_train": len(train_features),
        "num_val": len(val_features),
    }
    print(json.dumps(summary))

    return ckpt_path, norm_path


def run_gnn_inference(
    model: GraphSAGEModel,
    examples: list[dict],
    normalizer: WelfordNormalizer,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Run GNN inference on all examples in eval mode.

    Args:
        model: Trained GNN model
        examples: List of training examples
        normalizer: Feature normalizer

    Returns:
        Dict mapping example index to (logits, embeddings) tuple.
        Logits are graph-level (mean-pooled).
    """
    model.eval()
    edge_index = EDGE_INDEX
    results: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    with torch.no_grad():
        for i, example in enumerate(examples):
            features = extract_node_features(example, normalizer)
            logits, embeddings = model(features, edge_index)

            # Mean-pool to graph-level
            graph_logits = logits.mean(dim=0)
            graph_embeddings = embeddings.mean(dim=0)

            example_id = example.get("example_id", str(i))
            results[example_id] = (graph_logits, graph_embeddings)

    return results
