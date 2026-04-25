"""
eval/ — FirewatchAgent baseline evaluation module (SPEC-T4 v2)

Evaluates trained models against the HF Space sim, records metrics
to a chronological baselines log on HuggingFace Hub.

Exports:
  run_baseline()    — Main entry point (auto-invoke from SFT or manual CLI)
  EpisodeMetrics    — Per-episode metric container
  TaskAggregate     — Per-task aggregated metrics
  OverallAggregate  — Cross-task aggregated metrics
"""

from .metrics import EpisodeMetrics, OverallAggregate, TaskAggregate
from .baseline import run_baseline

__all__ = [
    "run_baseline",
    "EpisodeMetrics",
    "TaskAggregate",
    "OverallAggregate",
]
