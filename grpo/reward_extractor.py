"""
reward_extractor.py — Episode reward aggregation (SPEC-T3 §6)

Sums per-step rewards from a rollout trajectory into a single scalar
for GRPO advantage estimation. Includes the -5.0 terminal penalty
when the 15-step cap is exhausted.
"""

from __future__ import annotations


def extract_episode_reward(trajectory: list[dict]) -> float:
    """
    Sum per-step rewards from a trajectory.

    Each trajectory entry has a 'reward' key (float).
    The -5.0 cap-exhaust penalty (if applied) is already included
    in the final step's reward by the rollout function.

    Args:
        trajectory: List of step dicts, each containing at minimum
                    {'prompt': str, 'completion': str, 'reward': float, 'done': bool}

    Returns:
        Total episode reward (scalar).
    """
    if not trajectory:
        return 0.0

    return sum(step.get("reward", 0.0) for step in trajectory)


def is_cap_exhausted(trajectory: list[dict]) -> bool:
    """Check whether the trajectory ended due to the 15-step cap."""
    if not trajectory:
        return False
    last = trajectory[-1]
    return last.get("cap_exhausted", False)
