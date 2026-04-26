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

_VALID_ACTIONS: frozenset[str] = frozenset({
    "fetch_logs",
    "get_metrics_detail",
    "trace_dependencies",
    "strace_process",
    "profiler_dump",
    "check_gc_pressure",
    "trace_distributed_request",
    "inspect_thread_pool",
    "inspect_commit_diff",
    "inspect_network_policy",
    "inspect_quota_usage",
    "inspect_consensus_state",
    "inspect_cluster_topology",
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
    "adjust_probe_timing",
    "set_log_level",
    "disable_retries",
    "configure_retry_backoff",
    "rollback_canary",
    "promote_canary",
    "redirect_reads_to_primary",
    "force_replica_resync",
    "evict_cache_by_pattern",
    "increase_cache_memory",
    "complete_traffic_switch",
    "deregister_stale_instances",
    "enable_deadline_propagation",
    "revert_network_policy",
    "disable_fallback_mode",
    "request_quota_increase",
    "force_leader_election",
    "isolate_minority_nodes",
    "redirect_config_reads_to_majority",
    "flush_diverged_keys",
    "force_cluster_resync",
    "enable_cache_warming",
    "rate_limit_cache_misses",
    "rebalance_az_traffic",
    "scale_az_capacity",
    "thread_dump",
    "inspect_mtls_status",
    "inspect_pipeline_topology",
    "inject_missing_env_var",
    "restart_thread_pool",
    "update_service_endpoint",
    "force_ntp_sync",
    "increase_cpu_limit",
    "grant_rbac_permission",
    "increase_max_streams",
    "rotate_tls_certificate",
    "rollback_deployment_rollout",
    "evict_noisy_pod",
    "pre_warm_service",
    "stagger_connection_pool_reconnect",
    "drain_availability_zone",
    "force_cert_rotation",
    "restart_pipeline_job",
    "flush_pipeline_stage",
    "scale_pipeline_workers",
    "rollback_proxy_upgrade",
    "force_complete_proxy_upgrade",
    "declare_resolved",
    "escalate",
})

_ACTION_ALIASES: dict[str, str] = {
    "collect_logs": "fetch_logs",
    "query_logs": "fetch_logs",
    "debug_logs": "fetch_logs",
    "logs": "fetch_logs",
    "check_health": "get_metrics_detail",
    "check_thresholds": "get_metrics_detail",
    "check_instance_status": "get_metrics_detail",
    "check_replication_lag": "get_metrics_detail",
    "performance_profiling": "profiler_dump",
    "monitor": "get_metrics_detail",
    "verify_status": "get_metrics_detail",
    "diagnose": "get_metrics_detail",
    "redeploy": "rollback_deploy",
    "rollback_deployment": "rollback_deploy",
    "rollback": "rollback_deploy",
    "restart": "restart_service",
    "scale": "scale_replicas",
    "resolve": "declare_resolved",
}

_SERVICE_ALIASES: dict[str, str] = {
    "user-db-primary": "db-proxy",
    "user-db-replica": "db-proxy",
    "user_db_primary": "db-proxy",
    "user_db_replica": "db-proxy",
    "database": "db-proxy",
    "db": "db-proxy",
}

_CORE_ACTIONS_FOR_PROMPT: tuple[str, ...] = (
    "fetch_logs",
    "get_metrics_detail",
    "trace_dependencies",
    "restart_service",
    "rollback_deploy",
    "revert_config",
    "scale_replicas",
    "circuit_break",
    "traffic_shift",
    "declare_resolved",
    "escalate",
)


def _iter_json_candidates(text: str) -> list[str]:
    """Return balanced JSON object candidates from arbitrary model text."""
    candidates: list[str] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            current = text[idx]
            if escaped:
                escaped = False
                continue
            if current == "\\":
                escaped = True
                continue
            if current == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:idx + 1])
                    break
    return candidates


def _normalize_action(parsed: dict) -> dict | None:
    action_type = parsed.get("action_type") or parsed.get("action")
    if not action_type:
        return None

    normalized = str(action_type).strip().lower()
    normalized = normalized.replace("-", "_")
    normalized = _ACTION_ALIASES.get(normalized, normalized)
    if normalized not in _VALID_ACTIONS:
        logger.warning("Invalid action_type from completion: %s", action_type)
        return None

    result = {"action_type": normalized}
    target = parsed.get("target_service") or parsed.get("service")
    if isinstance(target, str) and target.strip():
        raw_target = target.strip()
        result["target_service"] = _SERVICE_ALIASES.get(raw_target, raw_target)
    params = parsed.get("parameters") or parsed.get("params")
    if isinstance(params, dict):
        result["parameters"] = params
    return result


def _parse_action(completion: str) -> dict:
    """
    Extract an action dict from LLM completion text.

    Looks for JSON objects containing 'action' or 'action_type' key.
    Falls back to no_op (declare_resolved) on parse failure.

    Returns:
        Dict with 'action_type', optional 'target_service', optional 'parameters'.
    """
    # Try to find balanced JSON action objects in the completion. Regex failed
    # on nested "parameters" objects and caused valid actions to fall back to
    # declare_resolved.
    for match in _iter_json_candidates(completion):
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                result = _normalize_action(parsed)
                if result is not None:
                    return result
        except (json.JSONDecodeError, AttributeError):
            continue

    # Try parsing the whole completion as JSON
    try:
        parsed = json.loads(completion.strip())
        if isinstance(parsed, dict):
            result = _normalize_action(parsed)
            if result is not None:
                return result
    except (json.JSONDecodeError, AttributeError):
        pass

    # Look for Step N: {...} pattern (matches SFT training format)
    step_pattern = re.findall(r'Step\s+\d+:\s*(\{[^{}]+\})', completion)
    for match in step_pattern:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                result = _normalize_action(parsed)
                if result is not None:
                    return result
        except (json.JSONDecodeError, AttributeError):
            continue

    # Fallback: no_op action
    logger.warning("Failed to parse action from completion, using no_op: %s", completion[:200])
    return {"action_type": "declare_resolved"}


def parse_action_sequence(completion: str, max_actions: int = 5) -> list[dict]:
    """Parse a short action sequence from model output.

    Accepts either {"actions": [...]} or a single action object. Invalid actions
    are skipped; if none remain, falls back to the single-action parser.
    """
    for match in _iter_json_candidates(completion):
        try:
            parsed = json.loads(match)
        except (json.JSONDecodeError, AttributeError):
            continue
        if not isinstance(parsed, dict):
            continue

        raw_actions = parsed.get("actions")
        if isinstance(raw_actions, list):
            actions: list[dict] = []
            for raw_action in raw_actions:
                if not isinstance(raw_action, dict):
                    continue
                normalized = _normalize_action(raw_action)
                if normalized is not None:
                    actions.append(normalized)
                if len(actions) >= max_actions:
                    break
            if actions:
                return actions

        normalized = _normalize_action(parsed)
        if normalized is not None:
            return [normalized]

    return [_parse_action(completion)]


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
    sequence_mode = os.environ.get("GRPO_SEQUENCE_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    if sequence_mode:
        parts.append(
            "Return exactly one JSON object and nothing else. Do not include prose, "
            "Markdown, examples, or code fences. The JSON must contain an 'actions' "
            "array with 2-5 action objects. Each action must have 'action_type', "
            "'target_service' when applicable, and optional 'parameters'. Prefer a "
            "short incident-response sequence: investigate, remediate, then "
            "declare_resolved only after remediation. Use only these common "
            f"action_type values unless a task-specific metric clearly requires "
            f"another valid Firewatch action: {', '.join(_CORE_ACTIONS_FOR_PROMPT)}. "
            "Example format: "
            '{"actions":[{"action_type":"fetch_logs","target_service":"auth-service"},'
            '{"action_type":"scale_replicas","target_service":"auth-service"},'
            '{"action_type":"declare_resolved"}]}'
        )
    else:
        parts.append(
            "Return exactly one JSON object and nothing else. Do not include prose, "
            "Markdown, examples, code fences, or multiple actions. The JSON must have "
            "'action_type', 'target_service', and optional 'parameters'. Use only these "
            f"common action_type values unless a task-specific metric clearly requires "
            f"another valid Firewatch action: {', '.join(_CORE_ACTIONS_FOR_PROMPT)}. "
            "Use fetch_logs instead of collect_logs/query_logs, get_metrics_detail "
            "instead of check_health/monitor/diagnose, rollback_deploy instead of "
            "redeploy, restart_service instead of restart, and scale_replicas instead "
            "of scale. Example format: "
            '{"action_type":"fetch_logs","target_service":"auth-service"}'
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
                # Matches grpo.max_completion_length in config.yaml. This stays
                # above the old truncating 256-token cap, but below the 3072-token
                # setting that made rambling generations too slow to iterate.
                max_new_tokens=1024,
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
