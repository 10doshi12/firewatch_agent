from __future__ import annotations

import sys
from pathlib import Path


_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))


def test_group_reward_eval_records_splits_on_completion_zero_and_grpo_complete() -> None:
    from analysis.grpo_group_metrics import group_reward_eval_records

    records = [
        {"event": "reward_eval", "completion_idx": 0, "reward": 1.0, "action_type": "fetch_logs"},
        {"event": "reward_eval", "completion_idx": 1, "reward": -1.0, "action_type": "diagnose"},
        {"event": "reward_eval", "completion_idx": 0, "reward": 0.5, "action_type": "trace_dependencies"},
        {"event": "reward_eval", "completion_idx": 1, "reward": 0.25, "action_type": "fetch_logs"},
        {"event": "grpo_complete"},
        {"event": "reward_eval", "completion_idx": 0, "reward": 2.0, "action_type": "declare_resolved"},
    ]

    groups = group_reward_eval_records(records)

    assert len(groups) == 3
    assert [len(g) for g in groups] == [2, 2, 1]


def test_summarize_grpo_group_batches_reports_std_and_valid_rate() -> None:
    from analysis.grpo_group_metrics import summarize_grpo_group_batches

    records = [
        {"event": "reward_eval", "completion_idx": 0, "reward": 1.0, "action_type": "fetch_logs"},
        {"event": "reward_eval", "completion_idx": 1, "reward": -1.0, "action_type": "diagnose"},
    ]

    summary = summarize_grpo_group_batches(records)

    assert summary["group_count"] == 1
    assert summary["mean_within_batch_std"] > 0
    assert 0.0 <= summary["mean_valid_action_rate"] <= 1.0
