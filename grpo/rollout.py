"""
rollout.py — Synchronous rollout against FirewatchEnv sim (SPEC-T3 §4)

One rollout = one complete episode: reset → observe → act → step loop
until done=True or the 15-step cap is hit.

Constraint 4: All rollouts are strictly sequential — no asyncio,
no threading, no multiprocessing. The sim is single-threaded.

Supports both remote HF Space and local FastAPI server — the sim URL
is configured via config.yaml `sim_env_url` (e.g. http://localhost:8000
for local, or the HF Space URL for remote).
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS_PER_ROLLOUT = 15
CAP_EXHAUST_PENALTY = -5.0


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

_ACTION_JSON_RE = re.compile(r'\{[^{}]*"action(?:_type)?"[^{}]*\}')


def _parse_action(completion: str) -> dict:
    """
    Extract an action dict from LLM completion text.

    Looks for JSON objects containing 'action' or 'action_type' key.
    Falls back to no_op (declare_resolved) on parse failure.

    Returns:
        Dict with 'action_type', optional 'target_service', optional 'parameters'.
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
    logger.warning("Failed to parse action from completion, using no_op: %s", completion[:200])
    return {"action_type": "declare_resolved"}


# ---------------------------------------------------------------------------
# Observation → prompt formatting
# ---------------------------------------------------------------------------


def _format_rollout_prompt(observation_dict: dict, gnn_blurb: str | None) -> str:
    """
    Build the user-message prompt from a sim observation + optional GNN blurb.

    Reuses the SFT prompt components but constructs a simpler format
    for the rollout (no assistant content — that's generated by the model).
    """
    parts: list[str] = []
    parts.append("An incident has been detected in the production system. Here is the current state:")
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


def _observation_to_dict(observation) -> dict:
    """
    Convert a SystemObservation object to a dict suitable for prompt formatting.

    Handles both Pydantic models (with .model_dump()) and plain dicts.
    """
    if hasattr(observation, "model_dump"):
        obs_data = observation.model_dump()
    elif hasattr(observation, "dict"):
        obs_data = observation.dict()
    elif isinstance(observation, dict):
        obs_data = observation
    else:
        obs_data = {}

    # Map SystemObservation fields to the format expected by _serialize_observation
    result: dict = {}

    # Alerts
    alerts = obs_data.get("active_alerts", [])
    if alerts:
        result["alerts"] = alerts

    # Service metrics
    services = obs_data.get("services", {})
    if services:
        result["service_metrics"] = services

    # Budget and tick
    budget = obs_data.get("slo_budget_remaining_pct")
    if budget is not None:
        result["budget"] = budget

    tick = obs_data.get("sim_tick")
    if tick is not None:
        result["tick"] = tick

    # Action history
    action_history = obs_data.get("action_history", [])
    if action_history:
        result["action_history"] = action_history

    return result


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
    # Build a pseudo-example for extract_node_features
    example = {"observation": observation_dict}
    features = extract_node_features(example, normalizer)

    with torch.no_grad():
        logits, _ = gnn_model(features, EDGE_INDEX)
        graph_logits = logits.mean(dim=0)

    return serialize_blurb(graph_logits)


# ---------------------------------------------------------------------------
# Core rollout function
# ---------------------------------------------------------------------------


def rollout(
    env_client,
    model,
    tokenizer,
    gnn_model: torch.nn.Module,
    normalizer: WelfordNormalizer,
    seed: int,
) -> list[dict]:
    """
    Execute one complete episode against the sim.

    Synchronous and sequential per SPEC-T3 Constraint 4.

    Args:
        env_client: SimClient instance (connected to sim via WebSocket).
        model: Current policy LLM (on GPU).
        tokenizer: Tokenizer for the LLM.
        gnn_model: Frozen GraphSAGE model (on CPU, eval mode).
        normalizer: Feature normalizer for GNN.
        seed: Seed for env_client.reset().

    Returns:
        Trajectory: list of step dicts, each containing:
            - prompt: str (the formatted prompt sent to the LLM)
            - completion: str (the LLM's raw output)
            - reward: float (per-step reward from sim)
            - done: bool (whether episode ended)
            - cap_exhausted: bool (True if 15-step cap was hit)
    """
    trajectory: list[dict] = []

    # Reset the environment
    result = env_client.reset(seed=seed)
    observation = result.observation.raw  # SimObservation.raw is the dict

    for step_num in range(MAX_STEPS_PER_ROLLOUT):
        # 1. Convert observation to dict for prompt formatting
        obs_dict = _observation_to_dict(observation)

        # 2. Run frozen GNN to get diagnostic blurb
        gnn_blurb = _run_gnn_for_observation(gnn_model, obs_dict, normalizer)

        # 3. Build prompt
        user_message = _format_rollout_prompt(obs_dict, gnn_blurb)

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ]

        # 4. Tokenize and generate
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
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0, input_ids.shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # 5. Parse action from completion
        action_dict = _parse_action(completion)

        # 6. Step the sim with the action dict (no firewatch_env dependency)
        result = env_client.step(action_dict)
        step_reward = result.reward if result.reward is not None else 0.0
        done = result.done

        # 7. Check for cap exhaustion
        cap_exhausted = False
        if step_num == MAX_STEPS_PER_ROLLOUT - 1 and not done:
            step_reward += CAP_EXHAUST_PENALTY
            cap_exhausted = True
            done = True
            logger.info("Rollout hit 15-step cap at seed=%d, penalty applied", seed)

        # 8. Record step
        trajectory.append({
            "prompt": user_message,
            "completion": completion,
            "reward": step_reward,
            "done": done,
            "cap_exhausted": cap_exhausted,
        })

        if done:
            break

        # Next observation
        observation = result.observation.raw

    return trajectory
