"""
trajectory.py — JSONL trajectory logger for offline analysis & SFT prep.

Two files per run, written under runs/<timestamp>/:

  steps.jsonl     — one record per env step
  episodes.jsonl  — one record per completed episode

The schema is stable (version 1). It is the single source of truth for
all downstream analysis: success-rate plots, reward distributions,
prompt diffing across model versions, and SFT data extraction.

Design notes:
  * Rewards and scores ARE recorded here. The "production agent must not
    see rewards" rule applies to the *prompt*, not to the data layer.
    The agent's prompt is a plain string captured here verbatim — if a
    prompt accidentally contains a reward, the test in
    tests/test_runner_honest_prompt.py will fail.
  * Action `source` ("llm", "fallback", "llm_unavailable",
    "llm_parse_error", "llm_invalid_action") lets us measure how often
    the LLM is the actual decision-maker.
  * Per-step `policy_inform_agent` records whether the rewards-in-
    history ablation was active for that run.
  * One file is appended per step as it happens (no batching) so the log
    survives a kill -9 of the runner process.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1


@dataclass
class StepRecord:
    schema_version: int
    run_id: str
    task_id: str
    difficulty: str
    seed: int
    step: int
    sim_tick: int
    slo_budget_remaining_pct: float
    bad_customer_minutes: float
    action: dict
    reward: float
    cumulative_reward: float
    done: bool
    info: dict
    services_snapshot: dict
    prompt: str
    raw_response: str
    source: str
    policy_inform_agent: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class EpisodeRecord:
    schema_version: int
    run_id: str
    task_id: str
    difficulty: str
    seed: int
    steps: int
    cumulative_reward: float
    rewards: list[float]
    episode_score: Optional[float]
    success: bool
    success_threshold: float
    final_action: Optional[dict]
    decision_sources: dict
    wall_time_seconds: float
    backend: str
    model: str
    gnn_mode: str
    policy_inform_agent: bool
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"


class TrajectoryLogger:
    """Append-only JSONL writer. One logger instance per inference run."""

    def __init__(
        self,
        runs_root: Path | str,
        run_id: Optional[str] = None,
        backend: str = "unknown",
        model: str = "unknown",
        gnn_mode: str = "heuristic",
        policy_inform_agent: bool = False,
    ) -> None:
        self.runs_root = Path(runs_root)
        self.run_id = run_id or make_run_id()
        self.backend = backend
        self.model = model
        self.gnn_mode = gnn_mode
        self.policy_inform_agent = policy_inform_agent

        self.run_dir = self.runs_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.run_dir / "steps.jsonl"
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.metadata_path = self.run_dir / "metadata.json"

        self._write_metadata()

    def _write_metadata(self) -> None:
        meta = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "backend": self.backend,
            "model": self.model,
            "gnn_mode": self.gnn_mode,
            "policy_inform_agent": self.policy_inform_agent,
        }
        self.metadata_path.write_text(json.dumps(meta, indent=2))

    def log_step(
        self,
        *,
        task_id: str,
        difficulty: str,
        seed: int,
        step: int,
        observation: dict,
        action: dict,
        reward: float,
        cumulative_reward: float,
        done: bool,
        info: dict,
        prompt: str,
        raw_response: str,
        source: str,
    ) -> None:
        record = StepRecord(
            schema_version=SCHEMA_VERSION,
            run_id=self.run_id,
            task_id=task_id,
            difficulty=difficulty,
            seed=seed,
            step=step,
            sim_tick=int(observation.get("sim_tick", 0) or 0),
            slo_budget_remaining_pct=float(
                observation.get("slo_budget_remaining_pct", 100.0) or 100.0
            ),
            bad_customer_minutes=float(observation.get("bad_customer_minutes", 0.0) or 0.0),
            action=action,
            reward=float(reward),
            cumulative_reward=float(cumulative_reward),
            done=bool(done),
            info=_sanitize(info),
            services_snapshot=_sanitize(observation.get("services") or {}),
            prompt=prompt,
            raw_response=raw_response,
            source=source,
            policy_inform_agent=self.policy_inform_agent,
        )
        self._append(self.steps_path, record.__dict__)

    def log_episode(
        self,
        *,
        task_id: str,
        difficulty: str,
        seed: int,
        steps: int,
        cumulative_reward: float,
        rewards: list[float],
        episode_score: Optional[float],
        success: bool,
        success_threshold: float,
        final_action: Optional[dict],
        decision_sources: dict,
        wall_time_seconds: float,
    ) -> None:
        record = EpisodeRecord(
            schema_version=SCHEMA_VERSION,
            run_id=self.run_id,
            task_id=task_id,
            difficulty=difficulty,
            seed=seed,
            steps=steps,
            cumulative_reward=cumulative_reward,
            rewards=list(rewards),
            episode_score=episode_score,
            success=bool(success),
            success_threshold=success_threshold,
            final_action=final_action,
            decision_sources=dict(decision_sources),
            wall_time_seconds=wall_time_seconds,
            backend=self.backend,
            model=self.model,
            gnn_mode=self.gnn_mode,
            policy_inform_agent=self.policy_inform_agent,
        )
        self._append(self.episodes_path, record.__dict__)

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":"), default=_default))
            fh.write("\n")


def _default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _sanitize(payload: Any) -> Any:
    """Make a JSON-serialisable deep copy; preserves dict/list shapes."""
    if isinstance(payload, dict):
        return {str(key): _sanitize(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_sanitize(item) for item in payload]
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    if hasattr(payload, "model_dump"):
        return _sanitize(payload.model_dump())
    return str(payload)
