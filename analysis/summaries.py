"""Aggregate Firewatch analysis records into report-friendly dictionaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Iterable


def summarize_sft_examples(examples: list[dict]) -> dict:
    """Summarize SFT task coverage and gold-action distribution."""
    tier_counts: Counter[str] = Counter()
    fault_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    for example in examples:
        tier = example.get("tier")
        if isinstance(tier, str) and tier:
            tier_counts[tier] += 1

        fault_type = example.get("fault_type")
        if isinstance(fault_type, str) and fault_type:
            fault_counts[fault_type] += 1

        for action in _iter_actions(example.get("gold_action_sequence")):
            action_name = _action_name(action)
            if action_name:
                action_counts[action_name] += 1
            target = _action_target(action)
            if target:
                target_counts[target] += 1

    return {
        "example_count": len(examples),
        "tier_counts": dict(tier_counts),
        "fault_counts": dict(fault_counts),
        "action_counts": dict(action_counts),
        "target_counts": dict(target_counts),
    }


def summarize_inference_runs(runs: list[dict]) -> dict:
    """Summarize inference trajectory success and decision-source mix."""
    all_episodes = [episode for run in runs for episode in run.get("episodes", [])]
    all_steps = [step for run in runs for step in run.get("steps", [])]

    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    for episode in all_episodes:
        difficulty = episode.get("difficulty")
        if isinstance(difficulty, str) and difficulty:
            by_difficulty[difficulty].append(episode)

    success_by_difficulty = {
        difficulty: _success_rate(episodes)
        for difficulty, episodes in sorted(by_difficulty.items())
    }

    decision_source_counts: Counter[str] = Counter()
    rewards: list[float] = []
    for step in all_steps:
        source = step.get("source")
        if isinstance(source, str) and source:
            decision_source_counts[source] += 1
        reward = _float_or_none(step.get("reward"))
        if reward is not None:
            rewards.append(reward)

    return {
        "run_count": len(runs),
        "episode_count": len(all_episodes),
        "step_count": len(all_steps),
        "success_by_difficulty": success_by_difficulty,
        "decision_source_counts": dict(decision_source_counts),
        "mean_step_reward": mean(rewards) if rewards else 0.0,
    }


def summarize_grpo_metrics(records: list[dict]) -> dict:
    """Summarize GRPO reward-evaluation records."""
    reward_records = [record for record in records if record.get("event") == "reward_eval"]
    rewards = [
        reward
        for reward in (_float_or_none(record.get("reward")) for record in reward_records)
        if reward is not None
    ]
    action_counts: Counter[str] = Counter()
    for record in reward_records:
        action_type = record.get("action_type")
        if isinstance(action_type, str) and action_type:
            action_counts[action_type] += 1

    return {
        "record_count": len(records),
        "reward_eval_count": len(reward_records),
        "mean_reward": mean(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "action_counts": dict(action_counts),
    }


def _iter_actions(value: object) -> Iterable[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _action_name(action: dict) -> str | None:
    value = action.get("action") or action.get("action_type")
    return value if isinstance(value, str) and value else None


def _action_target(action: dict) -> str | None:
    target = action.get("target_service")
    if isinstance(target, str) and target:
        return target

    params = action.get("params")
    if isinstance(params, dict):
        service = params.get("service") or params.get("target_service")
        if isinstance(service, str) and service:
            return service
    return None


def _success_rate(episodes: list[dict]) -> float:
    if not episodes:
        return 0.0
    return sum(1 for episode in episodes if bool(episode.get("success"))) / len(episodes)


def _float_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None
