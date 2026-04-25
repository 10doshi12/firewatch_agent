"""
inference.py — Local production-baseline runner for FirewatchEnv.

Run from inside firewatch_agent/:

    uv run python -m runners.inference --test-run
    uv run python -m runners.inference --backend ollama --model qwen2.5:14b-instruct
    uv run python -m runners.inference --gnn from_checkpoint \\
        --gnn-ckpt ../firewatch_agent_checkpoints/gnn/batch_010.pt \\
        --gnn-norm ../firewatch_agent_checkpoints/gnn/normalization.json
    uv run python -m runners.inference --inform-agent  # ablation: rewards in prompt history

Two output streams:

  1. STDOUT — the existing scoring contract:
        [START] task=<id> env=firewatch-env model=<name>
        [STEP]  step=<n> action=<type>:<target> reward=<float> done=<bool> error=<msg>
        [END]   success=<bool> steps=<n> score=<float> rewards=<csv>

  2. runs/<run-id>/ — trajectory artefacts for offline analysis:
        metadata.json
        steps.jsonl     (one record per env step)
        episodes.jsonl  (one record per completed episode)

Defaults are honest-baseline: GNN heuristic + LLM via OpenRouter +
SUCCESS_SCORE_THRESHOLD=0.5 + no rewards in prompt. Override with flags.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup so that running this as `uv run python -m runners.inference`
# from firewatch_agent/ AND running it as `python firewatch_agent/runners/
# inference.py` from the repo root both work.
# ---------------------------------------------------------------------------

_RUNNERS_DIR = Path(__file__).resolve().parent
_AGENT_ROOT = _RUNNERS_DIR.parent
_REPO_ROOT = _AGENT_ROOT.parent
for path in (_AGENT_ROOT, _REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from dotenv import load_dotenv
except ImportError as exc:  # pragma: no cover - hard dep, but be explicit
    raise SystemExit(
        "[FATAL] python-dotenv is required. Run `uv sync` from firewatch_agent/."
    ) from exc

# Load firewatch_env first (canonical secrets), then the agent-local override.
# override=True means the agent-local .env wins for keys that exist in both.
load_dotenv(_REPO_ROOT / "firewatch_env" / ".env", override=False)
load_dotenv(_AGENT_ROOT / ".env", override=True)

from runners.gnn_baseline import GnnBaseline  # noqa: E402
from runners.honest_prompt import HONEST_SYSTEM_MESSAGE  # noqa: E402
from runners.http_sim_client import HttpSimClient, resolve_sim_url  # noqa: E402
from runners.llm_client import LLMClient, llm_config_from_env  # noqa: E402
from runners.policy import FirewatchPolicy, PolicyState  # noqa: E402
from runners.trajectory import TrajectoryLogger, make_run_id  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    seed: int
    max_ticks: int


def _load_env_tasks() -> list[TaskSpec]:
    """Pull TASKS from firewatch_env.config; fall back to a 3-task default."""
    try:
        from firewatch_env.config import TASKS  # type: ignore[import]
    except Exception:
        try:
            sys.path.insert(0, str(_REPO_ROOT / "firewatch_env"))
            from config import TASKS  # type: ignore[import]
        except Exception:
            return [
                TaskSpec("task_easy_oom_baseline", "easy", 42, 20),
                TaskSpec("task_medium_cascade_memleak", "medium", 295, 30),
                TaskSpec("task_hard_config_drift_noise", "hard", 2560, 40),
            ]

    out: list[TaskSpec] = []
    for task in TASKS.values():
        out.append(
            TaskSpec(
                task_id=task.task_id,
                difficulty=task.difficulty,
                seed=task.grader_seed,
                max_ticks=task.max_ticks,
            )
        )
    return out


def select_tasks(test_run: bool) -> list[TaskSpec]:
    specs = _load_env_tasks()
    if not test_run:
        return specs
    selected: list[TaskSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.difficulty in {"easy", "medium", "hard"} and spec.difficulty not in seen:
            selected.append(spec)
            seen.add(spec.difficulty)
        if len(selected) == 3:
            break
    return selected or specs[:3]


# ---------------------------------------------------------------------------
# Stdout contract — keep exactly the same shape as the legacy runner so
# the evaluator can read both runners with one parser.
# ---------------------------------------------------------------------------


def _print_start(task_id: str, model: str) -> None:
    print(f"[START] task={task_id} env=firewatch-env model={model}", flush=True)


def _print_step(
    step: int, action: dict, reward: float, done: bool, error: Optional[str]
) -> None:
    atype = action.get("action_type", "?")
    target = action.get("target_service")
    action_str = f"{atype}:{target}" if target else atype
    err = error if error else "null"
    done_val = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={done_val} error={err}",
        flush=True,
    )


def _print_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
    sources: Optional[dict] = None,
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    parts = [
        f"[END] success={success_val}",
        f"steps={steps}",
        f"score={score:.2f}",
        f"rewards={rewards_str}",
    ]
    if sources:
        sources_str = ",".join(f"{k}:{v}" for k, v in sorted(sources.items()))
        parts.append(f"sources={sources_str}")
    print(" ".join(parts), flush=True)


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------


@dataclass
class EpisodeOutcome:
    steps: int
    cumulative_reward: float
    rewards: list[float]
    episode_score: Optional[float]
    success: bool
    decision_sources: dict
    final_action: Optional[dict]


def _episode_score_from_info(info: dict) -> Optional[float]:
    if not isinstance(info, dict):
        return None
    score = info.get("episode_score")
    if isinstance(score, (int, float)):
        return float(score)
    grader = info.get("grader_result") or {}
    if isinstance(grader, dict):
        score = grader.get("score") or grader.get("total_score")
        if isinstance(score, (int, float)):
            return float(score)
    return None


def _run_episode(
    *,
    sim: HttpSimClient,
    policy: FirewatchPolicy,
    spec: TaskSpec,
    max_steps: int,
    success_threshold: float,
    trajectory: TrajectoryLogger,
) -> EpisodeOutcome:
    state = PolicyState()
    sources: Counter = Counter()
    rewards: list[float] = []
    cumulative = 0.0
    last_action: Optional[dict] = None
    last_info: dict = {}
    last_reward: Optional[float] = None

    reset_result = sim.reset(
        difficulty=spec.difficulty, seed=spec.seed, task_id=spec.task_id
    )
    observation = reset_result.observation
    last_info = reset_result.info

    done = False
    steps_taken = 0
    for step_index in range(1, max_steps + 1):
        steps_taken = step_index
        decision = policy.decide(
            obs=observation,
            state=state,
            seed=spec.seed,
            last_reward=last_reward,
        )
        action = decision.action
        sources[decision.source] += 1

        try:
            step_result = sim.step(action)
            error_msg: Optional[str] = None
        except Exception as exc:
            error_msg = str(exc)[:120]
            step_result = type(reset_result)(
                observation=observation, reward=0.0, done=True, info={"error": error_msg}
            )

        next_observation = step_result.observation
        reward = step_result.reward
        done = step_result.done
        info = step_result.info

        cumulative += reward
        rewards.append(reward)
        last_action = action
        last_info = info
        last_reward = reward

        _print_step(step_index, action, reward, done, error_msg)

        trajectory.log_step(
            task_id=spec.task_id,
            difficulty=spec.difficulty,
            seed=spec.seed,
            step=step_index,
            observation=observation,
            action=action,
            reward=reward,
            cumulative_reward=cumulative,
            done=done,
            info=info if isinstance(info, dict) else {},
            prompt=decision.prompt,
            raw_response=decision.raw_response,
            source=decision.source,
        )

        FirewatchPolicy.update_state_after_step(state, action, info, next_observation)
        observation = next_observation
        if done:
            break

    episode_score = _episode_score_from_info(last_info)
    if episode_score is None:
        episode_score = cumulative
    success = episode_score >= success_threshold

    return EpisodeOutcome(
        steps=steps_taken,
        cumulative_reward=cumulative,
        rewards=rewards,
        episode_score=episode_score,
        success=success,
        decision_sources=dict(sources),
        final_action=last_action,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local production baseline runner for FirewatchEnv.",
    )
    parser.add_argument("--test-run", action="store_true", help="Run only one easy / medium / hard task.")
    parser.add_argument("--backend", choices=["openai", "ollama", "echo"], default=None,
                        help="Override LLM backend (defaults to LLM_BACKEND env var).")
    parser.add_argument("--model", default=None, help="Override MODEL_NAME env var.")
    parser.add_argument("--sim-url", default=None, help="Sim URL. Auto-detected if omitted.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override per-task max_ticks.")
    parser.add_argument("--success-threshold", type=float, default=0.5,
                        help="Episode score threshold for success= true (default 0.5).")
    parser.add_argument("--gnn", choices=["heuristic", "untrained", "from_checkpoint"], default="heuristic",
                        help="Graph baseline mode (default: heuristic).")
    parser.add_argument("--gnn-ckpt", default=None, help="Path to GNN .pt checkpoint (with --gnn from_checkpoint).")
    parser.add_argument("--gnn-norm", default=None, help="Path to GNN normalization.json.")
    parser.add_argument("--inform-agent", action="store_true",
                        help="Ablation: include last-step reward in prompt history.")
    parser.add_argument("--runs-dir", default=str(_AGENT_ROOT / "runs"),
                        help="Where to write per-run trajectory logs.")
    parser.add_argument("--run-id", default=None, help="Override the auto-generated run id.")
    parser.add_argument("--log-level", default="WARNING")
    return parser


def _override_env(backend: Optional[str], model: Optional[str]) -> None:
    if backend:
        os.environ["LLM_BACKEND"] = backend
    if model:
        os.environ["MODEL_NAME"] = model


def _build_policy(args: argparse.Namespace) -> FirewatchPolicy:
    llm_config = llm_config_from_env()
    llm = LLMClient(llm_config)
    llm.assert_ready()
    gnn = GnnBaseline(
        mode=args.gnn,
        checkpoint_path=args.gnn_ckpt,
        normalization_path=args.gnn_norm,
    )
    return FirewatchPolicy(
        llm_client=llm,
        gnn=gnn,
        system_message=HONEST_SYSTEM_MESSAGE,
        inform_agent=args.inform_agent,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    _override_env(args.backend, args.model)
    policy = _build_policy(args)

    sim_url = resolve_sim_url(args.sim_url)
    sim = HttpSimClient(sim_url)
    if not sim.is_healthy():
        print(
            f"[FATAL] Sim unreachable at {sim_url}. Start it with "
            f"`cd firewatch_env && uv run server --host 0.0.0.0 --port 8000`.",
            file=sys.stderr,
        )
        return 2

    trajectory = TrajectoryLogger(
        runs_root=Path(args.runs_dir),
        run_id=args.run_id or make_run_id("baseline"),
        backend=policy.llm.config.backend,
        model=policy.llm.config.model,
        gnn_mode=policy.gnn.mode,
        policy_inform_agent=args.inform_agent,
    )

    tasks = select_tasks(test_run=args.test_run)
    print(
        f"# baseline runner | sim={sim_url} | backend={policy.llm.config.backend} | "
        f"model={policy.llm.config.model} | gnn={policy.gnn.mode} | "
        f"inform_agent={args.inform_agent} | run_id={trajectory.run_id}",
        flush=True,
    )

    interrupted = False
    for spec in tasks:
        if interrupted:
            _print_start(spec.task_id, policy.llm.config.model)
            _print_end(False, 0, 0.0, [])
            continue

        max_steps = args.max_steps or spec.max_ticks
        _print_start(spec.task_id, policy.llm.config.model)
        episode_started_at = time.monotonic()

        try:
            outcome = _run_episode(
                sim=sim,
                policy=policy,
                spec=spec,
                max_steps=max_steps,
                success_threshold=args.success_threshold,
                trajectory=trajectory,
            )
        except KeyboardInterrupt:
            interrupted = True
            _print_end(False, 0, 0.0, [])
            continue
        except Exception as exc:
            logger.exception("episode crashed")
            _print_end(False, 0, 0.0, [])
            trajectory.log_episode(
                task_id=spec.task_id,
                difficulty=spec.difficulty,
                seed=spec.seed,
                steps=0,
                cumulative_reward=0.0,
                rewards=[],
                episode_score=None,
                success=False,
                success_threshold=args.success_threshold,
                final_action=None,
                decision_sources={"crash": 1},
                wall_time_seconds=time.monotonic() - episode_started_at,
            )
            continue

        _print_end(
            success=outcome.success,
            steps=outcome.steps,
            score=outcome.episode_score or 0.0,
            rewards=outcome.rewards,
            sources=outcome.decision_sources,
        )
        trajectory.log_episode(
            task_id=spec.task_id,
            difficulty=spec.difficulty,
            seed=spec.seed,
            steps=outcome.steps,
            cumulative_reward=outcome.cumulative_reward,
            rewards=outcome.rewards,
            episode_score=outcome.episode_score,
            success=outcome.success,
            success_threshold=args.success_threshold,
            final_action=outcome.final_action,
            decision_sources=outcome.decision_sources,
            wall_time_seconds=time.monotonic() - episode_started_at,
        )

    print(f"# trajectory dir: {trajectory.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
