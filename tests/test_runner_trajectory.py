"""Tests for runners.trajectory — JSONL schema and reward capture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from runners.trajectory import SCHEMA_VERSION, TrajectoryLogger  # noqa: E402


def _make_logger(tmp_path: Path) -> TrajectoryLogger:
    return TrajectoryLogger(
        runs_root=tmp_path,
        run_id="unit-test",
        backend="echo",
        model="echo-stub",
        gnn_mode="heuristic",
        policy_inform_agent=False,
    )


def test_logger_creates_run_directory_and_metadata(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    assert logger.run_dir.exists()
    assert logger.metadata_path.exists()
    meta = json.loads(logger.metadata_path.read_text())
    assert meta["run_id"] == "unit-test"
    assert meta["backend"] == "echo"
    assert meta["schema_version"] == SCHEMA_VERSION


def test_step_record_captures_reward_action_and_prompt(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    logger.log_step(
        task_id="task_easy_oom_baseline",
        difficulty="easy",
        seed=42,
        step=1,
        observation={
            "sim_tick": 1,
            "slo_budget_remaining_pct": 95.0,
            "bad_customer_minutes": 0.0,
            "services": {"auth-service": {"http_server_error_rate": 0.3}},
        },
        action={"action_type": "fetch_logs", "target_service": "auth-service"},
        reward=0.25,
        cumulative_reward=0.25,
        done=False,
        info={"action_feedback": "ok"},
        prompt="prompt text without rewards",
        raw_response='{"action_type":"fetch_logs"}',
        source="llm",
    )

    lines = logger.steps_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["task_id"] == "task_easy_oom_baseline"
    assert record["seed"] == 42
    assert record["reward"] == 0.25
    assert record["cumulative_reward"] == 0.25
    assert record["action"]["action_type"] == "fetch_logs"
    assert record["prompt"] == "prompt text without rewards"
    assert record["source"] == "llm"
    assert record["services_snapshot"]["auth-service"]["http_server_error_rate"] == 0.3
    assert record["sim_tick"] == 1


def test_episode_record_aggregates_rewards(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    logger.log_episode(
        task_id="task_medium_cascade_memleak",
        difficulty="medium",
        seed=295,
        steps=4,
        cumulative_reward=2.1,
        rewards=[0.5, 0.6, 0.5, 0.5],
        episode_score=0.84,
        success=True,
        success_threshold=0.5,
        final_action={"action_type": "declare_resolved"},
        decision_sources={"llm": 3, "fallback": 1},
        wall_time_seconds=12.34,
    )

    line = logger.episodes_path.read_text().strip().splitlines()[0]
    record = json.loads(line)
    assert record["task_id"] == "task_medium_cascade_memleak"
    assert record["cumulative_reward"] == 2.1
    assert record["rewards"] == [0.5, 0.6, 0.5, 0.5]
    assert record["episode_score"] == 0.84
    assert record["success"] is True
    assert record["decision_sources"] == {"llm": 3, "fallback": 1}


def test_logger_appends_multiple_records(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    for step in range(1, 4):
        logger.log_step(
            task_id="t",
            difficulty="easy",
            seed=1,
            step=step,
            observation={"services": {}, "sim_tick": step},
            action={"action_type": "fetch_logs", "target_service": "x"},
            reward=float(step),
            cumulative_reward=float(step),
            done=False,
            info={},
            prompt="p",
            raw_response="r",
            source="llm",
        )
    assert len(logger.steps_path.read_text().strip().splitlines()) == 3
