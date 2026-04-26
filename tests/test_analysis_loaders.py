from __future__ import annotations

import json
import sys
from pathlib import Path


_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))


def _write_jsonl(path: Path, records: list[dict], malformed: bool = False) -> None:
    lines = [json.dumps(record) for record in records]
    if malformed:
        lines.insert(1, "{not-json")
        lines.insert(2, "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def test_load_jsonl_skips_blank_and_malformed_lines(tmp_path: Path) -> None:
    from analysis.loaders import load_jsonl

    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [{"event": "one"}, {"event": "two"}], malformed=True)

    records = load_jsonl(path)

    assert records == [{"event": "one"}, {"event": "two"}]


def test_load_sft_examples_reads_sorted_reviewed_batches(tmp_path: Path) -> None:
    from analysis.loaders import load_sft_examples

    sft_dir = tmp_path / "reviewed"
    _write_jsonl(
        sft_dir / "batch_001.jsonl",
        [{"task_seed_id": "later", "tier": "hard", "fault_type": "bad_deploy"}],
    )
    _write_jsonl(
        sft_dir / "batch_000.jsonl",
        [{"task_seed_id": "first", "tier": "easy", "fault_type": "oom"}],
    )

    examples = load_sft_examples(sft_dir)

    assert [example["task_seed_id"] for example in examples] == ["first", "later"]


def test_summarize_sft_examples_counts_actions_and_faults() -> None:
    from analysis.summaries import summarize_sft_examples

    summary = summarize_sft_examples(
        [
            {
                "task_seed_id": "task_easy_oom_baseline",
                "tier": "easy",
                "fault_type": "oom",
                "gold_action_sequence": [
                    {"action": "fetch_logs", "params": {"service": "auth-service"}},
                    {"action_type": "scale_replicas", "target_service": "auth-service"},
                ],
            },
            {
                "task_seed_id": "task_medium_config",
                "tier": "medium",
                "fault_type": "config_drift",
                "gold_action_sequence": [
                    {"action": "revert_config", "params": {"service": "api-gateway"}}
                ],
            },
        ]
    )

    assert summary["example_count"] == 2
    assert summary["tier_counts"] == {"easy": 1, "medium": 1}
    assert summary["fault_counts"] == {"oom": 1, "config_drift": 1}
    assert summary["action_counts"]["fetch_logs"] == 1
    assert summary["action_counts"]["scale_replicas"] == 1
    assert summary["target_counts"]["auth-service"] == 2


def test_load_inference_runs_collects_metadata_episodes_and_steps(tmp_path: Path) -> None:
    from analysis.loaders import load_inference_runs

    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"run_id": "run-a", "backend": "echo"}))
    _write_jsonl(
        run_dir / "episodes.jsonl",
        [{"run_id": "run-a", "difficulty": "easy", "success": True, "episode_score": 0.7}],
    )
    _write_jsonl(
        run_dir / "steps.jsonl",
        [{"run_id": "run-a", "difficulty": "easy", "reward": 0.5, "source": "llm"}],
    )

    runs = load_inference_runs(tmp_path / "runs")

    assert len(runs) == 1
    assert runs[0]["metadata"]["backend"] == "echo"
    assert runs[0]["episodes"][0]["success"] is True
    assert runs[0]["steps"][0]["reward"] == 0.5


def test_summarize_inference_runs_counts_success_and_sources() -> None:
    from analysis.summaries import summarize_inference_runs

    summary = summarize_inference_runs(
        [
            {
                "metadata": {"run_id": "run-a"},
                "episodes": [
                    {"difficulty": "easy", "success": True, "episode_score": 0.8},
                    {"difficulty": "easy", "success": False, "episode_score": 0.2},
                    {"difficulty": "hard", "success": True, "episode_score": 0.9},
                ],
                "steps": [
                    {"source": "llm", "reward": 0.5, "cumulative_reward": 0.5},
                    {"source": "llm_parse_error", "reward": -0.5, "cumulative_reward": 0.0},
                ],
            }
        ]
    )

    assert summary["episode_count"] == 3
    assert summary["success_by_difficulty"]["easy"] == 0.5
    assert summary["success_by_difficulty"]["hard"] == 1.0
    assert summary["decision_source_counts"] == {"llm": 1, "llm_parse_error": 1}


def test_summarize_grpo_metrics_reports_reward_trend() -> None:
    from analysis.summaries import summarize_grpo_metrics

    summary = summarize_grpo_metrics(
        [
            {"event": "reward_eval", "reward": 0.25, "action_type": "fetch_logs"},
            {"event": "reward_eval", "reward": -1.0, "action_type": "restart_service"},
            {"event": "grpo_complete", "wall_time_seconds": 12.5},
        ]
    )

    assert summary["reward_eval_count"] == 2
    assert summary["mean_reward"] == -0.375
    assert summary["min_reward"] == -1.0
    assert summary["max_reward"] == 0.25
    assert summary["action_counts"] == {"fetch_logs": 1, "restart_service": 1}
