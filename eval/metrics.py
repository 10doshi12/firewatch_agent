"""
metrics.py — Metric definitions and aggregation (SPEC-T4 v2 §8)

Per-episode metrics:
  - cumulative_reward: sum of per-step rewards
  - episode_length: steps taken (1–15)
  - success: True if final step done=True AND final action was declare_resolved
  - wrong_actions: count of remediation actions on services with error_rate < 0.10
  - task: which task this was (easy/medium/hard)

Aggregation:
  - Per-task: mean reward, success rate, mean episode length, mean wrong actions
  - Overall: overall success rate, overall mean reward, total wrong actions
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class EpisodeMetrics:
    """Metrics collected from a single evaluation episode."""

    cumulative_reward: float
    episode_length: int
    success: bool
    wrong_actions: int
    task: str  # "easy", "medium", or "hard"


@dataclass
class TaskAggregate:
    """Aggregated metrics for one task difficulty across all episodes."""

    mean_reward: float
    success_rate: float
    mean_episode_length: float
    mean_wrong_actions: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OverallAggregate:
    """Aggregated metrics across all episodes and tasks."""

    overall_success_rate: float
    overall_mean_reward: float
    total_wrong_actions: int

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate_by_task(
    episodes: list[EpisodeMetrics],
) -> dict[str, TaskAggregate]:
    """
    Aggregate per-episode metrics by task difficulty.

    Returns:
        Dict mapping task name ("easy", "medium", "hard") to TaskAggregate.
    """
    by_task: dict[str, list[EpisodeMetrics]] = {}
    for ep in episodes:
        by_task.setdefault(ep.task, []).append(ep)

    result: dict[str, TaskAggregate] = {}
    for task_name, task_episodes in sorted(by_task.items()):
        n = len(task_episodes)
        if n == 0:
            continue

        result[task_name] = TaskAggregate(
            mean_reward=sum(ep.cumulative_reward for ep in task_episodes) / n,
            success_rate=sum(1 for ep in task_episodes if ep.success) / n,
            mean_episode_length=sum(ep.episode_length for ep in task_episodes) / n,
            mean_wrong_actions=sum(ep.wrong_actions for ep in task_episodes) / n,
        )

    return result


def aggregate_overall(episodes: list[EpisodeMetrics]) -> OverallAggregate:
    """Aggregate metrics across all episodes."""
    n = len(episodes)
    if n == 0:
        return OverallAggregate(
            overall_success_rate=0.0,
            overall_mean_reward=0.0,
            total_wrong_actions=0,
        )

    return OverallAggregate(
        overall_success_rate=sum(1 for ep in episodes if ep.success) / n,
        overall_mean_reward=sum(ep.cumulative_reward for ep in episodes) / n,
        total_wrong_actions=sum(ep.wrong_actions for ep in episodes),
    )
