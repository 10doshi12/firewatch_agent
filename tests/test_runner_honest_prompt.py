"""
Tests for runners.honest_prompt — guards against the four leakage vectors
identified in the baseline audit.
"""

from __future__ import annotations

import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from runners.honest_prompt import (  # noqa: E402
    GENERIC_ACTION_MENU,
    HONEST_SYSTEM_MESSAGE,
    active_services,
    build_user_prompt,
    episode_services,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _basic_obs() -> dict:
    return {
        "sim_tick": 4,
        "slo_budget_remaining_pct": 80.0,
        "bad_customer_minutes": 1.5,
        "services": {
            "auth-service": {
                "http_server_error_rate": 0.32,
                "http_server_request_duration_p99": 1.2,
                "process_memory_utilization": 0.95,
                "http_server_active_requests": 250,
                "status": "critical",
            },
            "checkout-service": {
                "http_server_error_rate": 0.18,
                "http_server_request_duration_p99": 0.8,
                "process_memory_utilization": 0.45,
                "http_server_active_requests": 150,
                "status": "degraded",
                "canary_traffic_weight": 0.20,
            },
            "notifications": {
                "http_server_error_rate": 0.01,
                "http_server_request_duration_p99": 0.05,
                "process_memory_utilization": 0.20,
                "http_server_active_requests": 5,
                "status": "healthy",
            },
        },
        "dependency_graph": {
            "checkout-service": ["auth-service"],
            "auth-service": [],
        },
        "active_alerts": [
            {
                "severity": "critical",
                "alertname": "HighErrorRate",
                "service_name": "auth-service",
                "description": "error_rate above 0.30",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Leakage vector 1: NO playbook / fault->remediation cheat sheet
# ---------------------------------------------------------------------------


def test_system_message_has_no_playbook_table() -> None:
    forbidden = [
        "OOMKilled",
        "rollback_deploy",  # remediations are listed by name in the action menu, not
        "revert_config",    # by mapping to log signals
        "FAULT DIAGNOSIS",
        "exit code 137",
        "ECONNREFUSED",
        "GC thrashing",
        "memory leak",
        "config revision",
    ]
    lower = HONEST_SYSTEM_MESSAGE.lower()
    for phrase in forbidden:
        assert phrase.lower() not in lower, (
            f"playbook leakage in system prompt: {phrase!r}"
        )


def test_user_prompt_has_no_playbook_table() -> None:
    prompt = build_user_prompt(_basic_obs(), history=[], fetched_logs=None)
    forbidden = [
        "FAULT DIAGNOSIS",
        "OOMKilled",
        "ECONNREFUSED",
        "GC thrashing",
        "match log signals to the right remediation",
    ]
    lower = prompt.lower()
    for phrase in forbidden:
        assert phrase.lower() not in lower, f"playbook leakage in user prompt: {phrase}"


# ---------------------------------------------------------------------------
# Leakage vector 2: NO oracle / "you MUST do X NOW" controller
# ---------------------------------------------------------------------------


def test_prompt_has_no_imperative_oracle() -> None:
    prompt = build_user_prompt(_basic_obs(), history=["fetch_logs:auth-service"], fetched_logs=None)
    forbidden_phrases = [
        "you must call",
        "must call declare_resolved now",
        "declare_resolved within",
        "incident active",
        "system is recovering",
        "system is healthy",
        "incident detected",
    ]
    lower = prompt.lower()
    for phrase in forbidden_phrases:
        assert phrase not in lower, f"oracle hint leak: {phrase!r}"


def test_prompt_has_no_decision_block() -> None:
    prompt = build_user_prompt(_basic_obs(), history=[], fetched_logs=None)
    assert "decision:" not in prompt.lower()


# ---------------------------------------------------------------------------
# Leakage vector 3: action menu must NOT branch on fault-typed dynamic fields
# ---------------------------------------------------------------------------


def test_action_menu_is_generic_regardless_of_phase2_fields() -> None:
    obs_no_phase2 = _basic_obs()
    obs_no_phase2["services"]["checkout-service"].pop("canary_traffic_weight", None)

    prompt_no_phase2 = build_user_prompt(obs_no_phase2, history=[], fetched_logs=None)
    prompt_with_phase2 = build_user_prompt(_basic_obs(), history=[], fetched_logs=None)

    fault_typed_actions = (
        "rollback_canary",
        "promote_canary",
        "rebalance_az_traffic",
        "rotate_tls_certificate",
        "force_complete_proxy_upgrade",
        "redirect_reads_to_primary",
        "force_leader_election",
        "increase_cache_memory",
    )
    for action in fault_typed_actions:
        assert action not in prompt_no_phase2, f"{action} leaked in absence of fault metric"
        assert action not in prompt_with_phase2, (
            f"{action} leaked when fault-typed metric was present — this would tell "
            "the agent the exact fault category."
        )


def test_generic_action_menu_includes_essentials_only() -> None:
    for action in (
        "fetch_logs",
        "get_metrics_detail",
        "trace_dependencies",
        "restart_service",
        "rollback_deploy",
        "revert_config",
        "scale_replicas",
        "circuit_break",
        "declare_resolved",
        "escalate",
    ):
        assert action in GENERIC_ACTION_MENU


# ---------------------------------------------------------------------------
# Leakage vector 4: NO reward / score / episode_score / task description
# ---------------------------------------------------------------------------


def test_prompt_excludes_reward_and_score_terms() -> None:
    prompt = build_user_prompt(
        _basic_obs(),
        history=["fetch_logs:auth-service", "trace_dependencies:auth-service"],
        fetched_logs={"auth-service": ["http 500 retry storm"]},
    )
    lower = prompt.lower()
    assert "reward" not in lower
    assert "episode_score" not in lower
    assert "score=" not in lower
    assert "correct path" not in lower
    assert "task_id" not in lower
    assert "task_easy_" not in lower
    assert "task_medium_" not in lower
    assert "task_hard_" not in lower


def test_history_lines_carry_no_reward_when_default() -> None:
    history = ["fetch_logs:auth-service", "restart_service:auth-service"]
    prompt = build_user_prompt(_basic_obs(), history=history, fetched_logs=None)
    for line in history:
        assert line in prompt
        # Confirm we did not splice reward into the line
        assert "reward" not in line.lower()
    assert "reward" not in prompt.lower()


# ---------------------------------------------------------------------------
# Sanity: required structural sections show up
# ---------------------------------------------------------------------------


def test_prompt_contains_required_sections() -> None:
    prompt = build_user_prompt(
        _basic_obs(),
        history=["fetch_logs:auth-service"],
        fetched_logs={"auth-service": ["http 500"]},
        gnn_blurb="[Graph analysis]\nTop suspect: auth-service",
    )
    for marker in (
        "Active service telemetry",
        "Episode dependency graph",
        "Action menu",
        "Active alerts",
        "Last actions",
        "Status summary",
        "[Graph analysis]",
        "Fetched logs",
    ):
        assert marker in prompt, f"missing section {marker!r}"


def test_active_services_filters_healthy_low_load() -> None:
    obs = _basic_obs()
    active = active_services(obs)
    assert "auth-service" in active
    assert "checkout-service" in active
    # 'notifications' is healthy + idle but has no dynamic field, should be filtered out.
    assert "notifications" not in active


# ---------------------------------------------------------------------------
# Lenient action targeting: the action menu must include every service in
# this episode, including currently-healthy ones. A real on-call can
# fetch_logs on any service in their dashboard; restricting to the
# currently-degraded set both leaks "the fault is over here" and prevents
# the agent from investigating quiet upstream root causes.
# ---------------------------------------------------------------------------


def test_episode_services_returns_all_services_regardless_of_health() -> None:
    obs = _basic_obs()
    episode = episode_services(obs)
    assert set(episode.keys()) == {
        "auth-service",
        "checkout-service",
        "notifications",
    }


def test_action_menu_lists_all_episode_services_including_healthy() -> None:
    obs = _basic_obs()
    prompt = build_user_prompt(obs, history=[], fetched_logs=None)
    menu_block = prompt.split("Action menu", 1)[1].split("\n\n", 1)[0]
    for svc in ("auth-service", "checkout-service", "notifications"):
        assert svc in menu_block, (
            f"{svc!r} must appear in the action menu so the agent can "
            "investigate quiet/healthy services and target meta actions on "
            "any service in the episode."
        )


def test_dep_graph_shows_full_topology_not_only_active() -> None:
    obs = _basic_obs()
    obs["dependency_graph"]["notifications"] = ["auth-service"]
    prompt = build_user_prompt(obs, history=[], fetched_logs=None)
    dep_block = prompt.split("Episode dependency graph", 1)[1].split("\n\n", 1)[0]
    assert "notifications" in dep_block, (
        "Healthy services must still appear in the dep graph block — "
        "filtering them out leaks 'the fault is on the loud nodes'."
    )


def test_telemetry_block_stays_focused_on_active_services() -> None:
    """The worst-first telemetry slice is still the narrow `active` set so
    the prompt isn't drowned in healthy lines, even though the action menu
    is wider."""
    obs = _basic_obs()
    prompt = build_user_prompt(obs, history=[], fetched_logs=None)
    tele_block = prompt.split("Active service telemetry", 1)[1].split(
        "\n\n", 1
    )[0]
    assert "auth-service" in tele_block
    assert "checkout-service" in tele_block
    # 'notifications' is healthy → not in the worst-first telemetry slice.
    assert "notifications" not in tele_block
