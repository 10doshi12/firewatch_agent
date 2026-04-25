"""
runner.py — Episode runner for baseline evaluation (SPEC-T4 v2 §7)

Runs single-episode rollouts using the current model with greedy generation
(do_sample=False, temperature=0.0) for deterministic measurement.

Reuses shared infrastructure:
  - grpo/sim_client.SimClient for WebSocket connectivity
  - grpo/rollout._parse_action for action extraction
  - sft/prompt for observation serialization and system message
  - gnn/ for frozen GNN inference

Key differences from GRPO rollout:
  - Greedy generation (not sampled)
  - Tracks wrong_actions and success metrics
  - No terminal penalty for cap exhaustion (just records episode_length)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Resolve project paths
# ---------------------------------------------------------------------------

_AGENT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _AGENT_ROOT.parent

if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gnn.adjacency import EDGE_INDEX, NUM_SERVICES  # noqa: E402
from gnn.serializer import serialize_blurb  # noqa: E402
from gnn.train_gnn import NUM_FEATURES, WelfordNormalizer, extract_node_features  # noqa: E402
from sft.prompt import SYSTEM_MESSAGE, _serialize_observation  # noqa: E402

from .metrics import EpisodeMetrics  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS_PER_EPISODE = 15
TASKS = ["easy", "medium", "hard"]

# Remediation action prefixes — actions that modify service state
# Used for wrong-action detection
REMEDIATION_ACTIONS = frozenset({
    "restart_service",
    "rollback_deploy",
    "revert_config",
    "scale_replicas",
    "circuit_break",
    "traffic_shift",
    "enable_connection_throttle",
    "extend_timeout",
    "optimize_query",
    "rebalance_load",
    "rotate_tls_certificate",
    "restart_pipeline_job",
    "flush_pipeline_stage",
    "rollback_proxy_version",
    "upgrade_proxy_version",
})


# ---------------------------------------------------------------------------
# Action parsing (reuses logic from grpo/rollout.py)
# ---------------------------------------------------------------------------

_ACTION_JSON_RE = re.compile(r'\{[^{}]*"action(?:_type)?"[^{}]*\}')


def _parse_action(completion: str) -> dict:
    """
    Extract an action dict from LLM completion text.

    Looks for JSON objects containing 'action' or 'action_type' key.
    Falls back to no_op (declare_resolved) on parse failure.
    """
    # Try to find JSON-like action in the completion
    matches = _ACTION_JSON_RE.findall(completion)
    for match in matches:
        try:
            parsed = json.loads(match)
            action_type = parsed.get("action_type") or parsed.get("action")
            if action_type:
                result = {"action_type": action_type}
                target = parsed.get("target_service") or parsed.get("service")
                if target:
                    result["target_service"] = target
                params = parsed.get("parameters") or parsed.get("params")
                if params and isinstance(params, dict):
                    result["parameters"] = params
                return result
        except (json.JSONDecodeError, AttributeError):
            continue

    # Try parsing the whole completion as JSON
    try:
        parsed = json.loads(completion.strip())
        if isinstance(parsed, dict):
            action_type = parsed.get("action_type") or parsed.get("action")
            if action_type:
                result = {"action_type": action_type}
                target = parsed.get("target_service") or parsed.get("service")
                if target:
                    result["target_service"] = target
                params = parsed.get("parameters") or parsed.get("params")
                if params and isinstance(params, dict):
                    result["parameters"] = params
                return result
    except (json.JSONDecodeError, AttributeError):
        pass

    # Look for Step N: {...} pattern (matches SFT training format)
    step_pattern = re.findall(r'Step\s+\d+:\s*(\{[^{}]+\})', completion)
    for match in step_pattern:
        try:
            parsed = json.loads(match)
            action_type = parsed.get("action_type") or parsed.get("action")
            if action_type:
                result = {"action_type": action_type}
                target = parsed.get("target_service") or parsed.get("service")
                if target:
                    result["target_service"] = target
                params = parsed.get("parameters") or parsed.get("params")
                if params and isinstance(params, dict):
                    result["parameters"] = params
                return result
        except (json.JSONDecodeError, AttributeError):
            continue

    # Fallback: no_op action
    logger.warning(
        "Failed to parse action from completion, using no_op: %s",
        completion[:200],
    )
    return {"action_type": "declare_resolved"}


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _observation_to_dict(observation) -> dict:
    """Convert observation to dict suitable for prompt formatting."""
    if hasattr(observation, "model_dump"):
        obs_data = observation.model_dump()
    elif hasattr(observation, "dict"):
        obs_data = observation.dict()
    elif isinstance(observation, dict):
        obs_data = observation
    else:
        obs_data = {}

    result: dict = {}

    alerts = obs_data.get("active_alerts", [])
    if alerts:
        result["alerts"] = alerts

    services = obs_data.get("services", {})
    if services:
        result["service_metrics"] = services

    budget = obs_data.get("slo_budget_remaining_pct")
    if budget is not None:
        result["budget"] = budget

    tick = obs_data.get("sim_tick")
    if tick is not None:
        result["tick"] = tick

    action_history = obs_data.get("action_history", [])
    if action_history:
        result["action_history"] = action_history

    return result


def _format_eval_prompt(observation_dict: dict, gnn_blurb: str | None) -> str:
    """
    Build the user-message prompt from a sim observation + optional GNN blurb.

    Same structure as GRPO rollout prompts — the model sees the same format
    it was trained on.
    """
    parts: list[str] = []
    parts.append(
        "An incident has been detected in the production system. "
        "Here is the current state:"
    )
    parts.append("")
    parts.append(_serialize_observation(observation_dict))

    if gnn_blurb:
        parts.append("")
        parts.append(gnn_blurb)

    parts.append("")
    parts.append(
        "Analyze the situation and provide the next action to diagnose or resolve "
        "this incident. Respond with a single JSON action object containing "
        "'action_type', 'target_service', and optional 'parameters'."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# GNN inference helper (CPU, frozen)
# ---------------------------------------------------------------------------


def _run_gnn_for_observation(
    gnn_model: torch.nn.Module,
    observation_dict: dict,
    normalizer: WelfordNormalizer,
) -> str:
    """
    Run frozen GNN on a single observation and return the text blurb.
    The GNN stays on CPU. No gradients.
    """
    example = {"observation": observation_dict}
    features = extract_node_features(example, normalizer)

    with torch.no_grad():
        logits, _ = gnn_model(features, EDGE_INDEX)
        graph_logits = logits.mean(dim=0)

    return serialize_blurb(graph_logits)


# ---------------------------------------------------------------------------
# Wrong-action detection
# ---------------------------------------------------------------------------


def _is_wrong_action(action_dict: dict, observation_dict: dict) -> bool:
    """
    Check if a remediation action was applied to a service with error_rate < 0.10.

    Per README Section 2: remediation actions on healthy services
    (error_rate < 0.10) are counted as wrong actions.
    """
    action_type = action_dict.get("action_type", "")
    if action_type not in REMEDIATION_ACTIONS:
        return False

    target_service = action_dict.get("target_service")
    if not target_service:
        return False

    service_metrics = observation_dict.get("service_metrics", {})
    svc_data = service_metrics.get(target_service, {})

    error_rate = svc_data.get("http_server_error_rate", 0.0)
    if isinstance(error_rate, (int, float)) and error_rate < 0.10:
        return True

    return False


# ---------------------------------------------------------------------------
# Single-episode rollout
# ---------------------------------------------------------------------------


def run_episode(
    env_client,
    model,
    tokenizer,
    gnn_model: torch.nn.Module | None,
    normalizer: WelfordNormalizer | None,
    seed: int,
    difficulty: str,
    use_gnn: bool = True,
) -> EpisodeMetrics:
    """
    Run one evaluation episode with greedy generation.

    Args:
        env_client: Connected SimClient instance.
        model: LLM in eval mode (on GPU).
        tokenizer: Tokenizer for the LLM.
        gnn_model: Frozen GNN model (on CPU, eval mode). None if base variant.
        normalizer: Feature normalizer for GNN. None if base variant.
        seed: Seed for env reset (deterministic).
        difficulty: Task difficulty ("easy", "medium", "hard").
        use_gnn: Whether to include GNN blurb in prompts.

    Returns:
        EpisodeMetrics for this episode.
    """
    cumulative_reward = 0.0
    wrong_actions = 0
    last_action_type = ""

    # Reset the environment
    result = env_client.reset(seed=seed, difficulty=difficulty)
    observation = result.observation.raw
    done = result.done

    step_count = 0

    for step_num in range(MAX_STEPS_PER_EPISODE):
        if done:
            break

        step_count += 1

        # 1. Convert observation for prompt
        obs_dict = _observation_to_dict(observation)

        # 2. Run frozen GNN for blurb (skip if base variant)
        gnn_blurb = None
        if use_gnn and gnn_model is not None and normalizer is not None:
            gnn_blurb = _run_gnn_for_observation(gnn_model, obs_dict, normalizer)

        # 3. Build prompt
        user_message = _format_eval_prompt(obs_dict, gnn_blurb)
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ]

        # 4. Tokenize and generate (GREEDY — deterministic)
        try:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            input_text = (
                f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
                f"<|im_start|>user\n{user_message}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = output_ids[0, input_ids.shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # 5. Parse action
        action_dict = _parse_action(completion)
        last_action_type = action_dict.get("action_type", "")

        # 6. Check for wrong action before stepping
        if _is_wrong_action(action_dict, obs_dict):
            wrong_actions += 1

        # 7. Step the sim
        result = env_client.step(action_dict)
        step_reward = result.reward if result.reward is not None else 0.0
        cumulative_reward += step_reward
        done = result.done

        # Next observation
        if not done:
            observation = result.observation.raw

    # Episode length is at least 1 (we always take at least one step)
    episode_length = max(step_count, 1)

    # Success: episode ended naturally (done=True) AND final action was declare_resolved
    success = done and last_action_type == "declare_resolved"

    return EpisodeMetrics(
        cumulative_reward=cumulative_reward,
        episode_length=episode_length,
        success=success,
        wrong_actions=wrong_actions,
        task=difficulty,
    )


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------


def run_evaluation(
    env_client,
    model,
    tokenizer,
    gnn_model: torch.nn.Module | None,
    normalizer: WelfordNormalizer | None,
    num_episodes_per_task: int = 20,
    use_gnn: bool = True,
) -> list[EpisodeMetrics]:
    """
    Run the full evaluation suite: 3 tasks × num_episodes_per_task episodes.

    All episodes run sequentially (Constraint 4 — no concurrency against the sim).

    Seed formula: T * 1000 + E
      T = task index (0=easy, 1=medium, 2=hard)
      E = episode index (0..num_episodes_per_task-1)

    Args:
        env_client: Connected SimClient.
        model: LLM in eval mode.
        tokenizer: Tokenizer for LLM.
        gnn_model: Frozen GNN (CPU, eval mode). None if base variant.
        normalizer: Feature normalizer. None if base variant.
        num_episodes_per_task: Episodes per task (default 20).
        use_gnn: Whether to run GNN for blurbs.

    Returns:
        List of EpisodeMetrics for all episodes.
    """
    all_episodes: list[EpisodeMetrics] = []
    total = len(TASKS) * num_episodes_per_task

    for task_idx, task_name in enumerate(TASKS):
        logger.info("Starting evaluation for task '%s' (%d episodes)", task_name, num_episodes_per_task)

        for ep_idx in range(num_episodes_per_task):
            seed = task_idx * 1000 + ep_idx
            episode_num = task_idx * num_episodes_per_task + ep_idx + 1

            logger.info(
                "Episode %d/%d: task=%s seed=%d",
                episode_num, total, task_name, seed,
            )

            metrics = run_episode(
                env_client=env_client,
                model=model,
                tokenizer=tokenizer,
                gnn_model=gnn_model,
                normalizer=normalizer,
                seed=seed,
                difficulty=task_name,
                use_gnn=use_gnn,
            )

            all_episodes.append(metrics)

            logger.info(
                "  -> reward=%.3f length=%d success=%s wrong_actions=%d",
                metrics.cumulative_reward,
                metrics.episode_length,
                metrics.success,
                metrics.wrong_actions,
            )

    return all_episodes
