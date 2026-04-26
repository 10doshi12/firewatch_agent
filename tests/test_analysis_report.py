from __future__ import annotations

import json
import sys
from pathlib import Path


_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))


def test_write_report_includes_key_sections(tmp_path: Path) -> None:
    from analysis.report import write_report

    report_path = write_report(
        output_dir=tmp_path,
        summaries={
            "sft": {"example_count": 2},
            "inference": {"episode_count": 3},
            "grpo": {
                "reward_eval_count": 4,
                "mean_reward": 0.25,
                "positive_reward_rate": 0.5,
                "action_counts": {"fetch_logs": 3},
            },
            "baseline": {
                "record_count": 2,
                "grpo_reward_delta": 0.4,
                "grpo_success_delta": 0.1,
            },
        },
        plot_paths=[tmp_path / "plots" / "grpo_reward_eval.png"],
    )

    content = report_path.read_text()

    assert "# Firewatch Analysis Report" in content
    assert "SFT examples analyzed: 2" in content
    assert "Inference episodes analyzed: 3" in content
    assert "GRPO reward evaluations analyzed: 4" in content
    assert "GRPO pre/post mean reward delta: +0.400" in content
    assert "`fetch_logs`: 3" in content
    assert "GRPO rewards are single-step evaluations" in content
    assert "plots/grpo_reward_eval.png" in content


def test_generate_plots_writes_png_files(tmp_path: Path) -> None:
    from analysis.plots import generate_plots

    plot_paths = generate_plots(
        output_dir=tmp_path,
        sft_summary={
            "action_counts": {"fetch_logs": 2, "scale_replicas": 1},
            "fault_counts": {"oom": 1},
            "tier_counts": {"easy": 1},
        },
        inference_summary={
            "success_by_difficulty": {"easy": 1.0, "hard": 0.5},
            "decision_source_counts": {"llm": 3, "fallback": 1},
        },
        grpo_records=[
            {"event": "reward_eval", "reward": 0.2, "action_type": "fetch_logs"},
            {"event": "reward_eval", "reward": -0.1, "action_type": "restart_service"},
        ],
        baseline_records=[
            {
                "trigger": "run-a",
                "overall": {"overall_success_rate": 0.5, "overall_mean_reward": 0.1},
            },
            {
                "trigger": "run-b",
                "overall": {"overall_success_rate": 0.75, "overall_mean_reward": 0.3},
            },
        ],
    )

    assert {path.name for path in plot_paths} == {
        "sft_action_distribution.png",
        "sft_task_fault_coverage.png",
        "inference_success_by_difficulty.png",
        "decision_source_mix.png",
        "grpo_reward_eval.png",
        "grpo_action_distribution.png",
        "grpo_reward_by_action.png",
        "baseline_progression.png",
        "grpo_pre_post_delta.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in plot_paths)


def test_analysis_cli_writes_summary_report_and_plots(tmp_path: Path, monkeypatch) -> None:
    from analysis.analyze import main

    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    (sft_dir / "batch_000.jsonl").write_text(
        json.dumps(
            {
                "task_seed_id": "task_easy_oom_baseline",
                "tier": "easy",
                "fault_type": "oom",
                "gold_action_sequence": [{"action": "fetch_logs", "params": {"service": "auth-service"}}],
            }
        )
        + "\n"
    )

    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"run_id": "run-a"}))
    (run_dir / "episodes.jsonl").write_text(
        json.dumps({"difficulty": "easy", "success": True, "episode_score": 0.7}) + "\n"
    )
    (run_dir / "steps.jsonl").write_text(
        json.dumps({"source": "llm", "reward": 0.5, "cumulative_reward": 0.5}) + "\n"
    )

    grpo_log = tmp_path / "metrics.jsonl"
    grpo_log.write_text(json.dumps({"event": "reward_eval", "reward": 0.5, "action_type": "fetch_logs"}) + "\n")
    baseline_log = tmp_path / "baselines.jsonl"
    baseline_log.write_text(
        json.dumps({
            "model_variant": "grpo-pre",
            "overall": {"overall_success_rate": 0.0, "overall_mean_reward": -2.0},
        })
        + "\n"
        + json.dumps({
            "model_variant": "grpo-post",
            "overall": {"overall_success_rate": 0.2, "overall_mean_reward": -1.0},
        })
        + "\n"
    )

    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analysis.analyze",
            "--sft-dir",
            str(sft_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--grpo-log",
            str(grpo_log),
            "--baseline-log",
            str(baseline_log),
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "plots" / "grpo_reward_eval.png").exists()
    assert (output_dir / "plots" / "grpo_pre_post_delta.png").exists()
