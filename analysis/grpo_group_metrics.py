"""GRPO metrics.jsonl helpers: TRL batch grouping + learnability summaries."""

from __future__ import annotations

import statistics
from typing import Any


def _float_reward(record: dict[str, Any]) -> float | None:
    raw = record.get("reward")
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def action_type_is_valid_firewatch(action_type: object) -> bool:
    """
    True if action_type lands in the Firewatch verb set after rollout aliases.

    Mirrors grpo.rollout._normalize_action verb resolution only (no target checks),
    without calling _normalize_action (avoids warning spam during batch scans).
    """
    try:
        from grpo.rollout import _ACTION_ALIASES, _VALID_ACTIONS
    except Exception:
        return False
    if not isinstance(action_type, str) or not action_type.strip():
        return False
    normalized = str(action_type).strip().lower().replace("-", "_")
    normalized = _ACTION_ALIASES.get(normalized, normalized)
    return normalized in _VALID_ACTIONS


def group_reward_eval_records(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """
    Split reward_eval lines into TRL rollout batches.

    Batches are delimited by completion_idx == 0 (except the first batch) and
    by grpo_complete markers (run boundaries), which flush the current batch.
    """
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for record in records:
        event = record.get("event")
        if event == "grpo_complete":
            if current:
                groups.append(current)
                current = []
            continue
        if event != "reward_eval":
            continue

        completion_idx = record.get("completion_idx")
        if isinstance(completion_idx, int) and completion_idx == 0 and current:
            groups.append(current)
            current = []
        current.append(record)

    if current:
        groups.append(current)

    return groups


def grpo_complete_group_cut_indices(records: list[dict[str, Any]]) -> list[int]:
    """
    For each grpo_complete event, return how many complete groups had been
    closed *before* that marker (useful for vertical lines on batch-index plots).
    """
    cuts: list[int] = []
    groups_closed = 0
    current: list[dict[str, Any]] = []

    for record in records:
        event = record.get("event")
        if event == "grpo_complete":
            if current:
                groups_closed += 1
                current = []
            cuts.append(groups_closed)
            continue
        if event != "reward_eval":
            continue
        completion_idx = record.get("completion_idx")
        if isinstance(completion_idx, int) and completion_idx == 0 and current:
            groups_closed += 1
            current = []
        current.append(record)

    return cuts


def summarize_grpo_group_batches(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate learnability stats for Markdown / README."""
    groups = group_reward_eval_records(records)
    if not groups:
        return {}

    stds: list[float] = []
    valid_rates: list[float] = []
    seq_stds: list[float] = []
    seq_valid: list[float] = []

    for batch in groups:
        rewards = [r for r in (_float_reward(row) for row in batch) if r is not None]
        std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        stds.append(std)

        raw_actions = [row.get("action_type") for row in batch]
        denom = len(raw_actions) or 1
        valid = sum(1 for a in raw_actions if action_type_is_valid_firewatch(a)) / denom
        valid_rates.append(valid)

        if any(row.get("sequence_mode") for row in batch):
            seq_stds.append(std)
            seq_valid.append(valid)

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    pos_std = sum(1 for s in stds if s > 1e-6)

    return {
        "group_count": len(groups),
        "mean_within_batch_std": _mean(stds),
        "pct_batches_with_positive_std": pos_std / len(stds) if stds else 0.0,
        "mean_valid_action_rate": _mean(valid_rates),
        "sequence_group_count": len(seq_stds),
        "sequence_mean_within_batch_std": _mean(seq_stds),
        "sequence_mean_valid_action_rate": _mean(seq_valid),
    }
