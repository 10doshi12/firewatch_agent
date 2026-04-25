"""Tests for runners.policy — composition + parsing + fallback path."""

from __future__ import annotations

import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from runners.gnn_baseline import GnnBaseline  # noqa: E402
from runners.honest_prompt import GENERIC_ACTION_MENU  # noqa: E402
from runners.llm_client import LLMClient, LLMConfig, LLMUnavailable  # noqa: E402
from runners.policy import (  # noqa: E402
    FirewatchPolicy,
    PolicyState,
    parse_action,
)


# ---------------------------------------------------------------------------
# parse_action coverage
# ---------------------------------------------------------------------------


def test_parse_action_clean_json() -> None:
    text = '{"action_type": "fetch_logs", "target_service": "auth-service"}'
    action = parse_action(text, ["auth-service", "checkout"])
    assert action == {
        "action_type": "fetch_logs",
        "target_service": "auth-service",
        "parameters": {},
    }


def test_parse_action_strips_markdown() -> None:
    text = '```json\n{"action": "restart_service", "service": "checkout"}\n```'
    action = parse_action(text, ["auth-service", "checkout"])
    assert action == {
        "action_type": "restart_service",
        "target_service": "checkout",
        "parameters": {},
    }


def test_parse_action_meta_drops_target() -> None:
    text = '{"action_type": "declare_resolved", "target_service": "auth-service"}'
    action = parse_action(text, ["auth-service"])
    assert action == {
        "action_type": "declare_resolved",
        "target_service": None,
        "parameters": {},
    }


def test_parse_action_alternate_keys_and_unknown_target() -> None:
    text = '{"action": "fetch_logs", "targets": ["unknown-service"]}'
    action = parse_action(text, ["auth-service", "checkout"])
    assert action == {
        "action_type": "fetch_logs",
        "target_service": "auth-service",
        "parameters": {},
    }


def test_parse_action_returns_none_for_garbage() -> None:
    assert parse_action("hello world", ["auth-service"]) is None
    assert parse_action("", ["auth-service"]) is None


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def _obs() -> dict:
    return {
        "sim_tick": 2,
        "slo_budget_remaining_pct": 80.0,
        "bad_customer_minutes": 1.0,
        "services": {
            "auth-service": {
                "http_server_error_rate": 0.32,
                "http_server_request_duration_p99": 1.0,
                "process_memory_utilization": 0.85,
                "http_server_active_requests": 200,
                "status": "critical",
            },
            "checkout": {
                "http_server_error_rate": 0.20,
                "http_server_request_duration_p99": 0.5,
                "process_memory_utilization": 0.40,
                "http_server_active_requests": 80,
                "status": "degraded",
            },
        },
        "dependency_graph": {"checkout": ["auth-service"], "auth-service": []},
        "active_alerts": [],
    }


def test_policy_with_echo_backend_returns_valid_action() -> None:
    policy = FirewatchPolicy(
        llm_client=LLMClient(LLMConfig(backend="echo")),
        gnn=GnnBaseline(mode="heuristic"),
    )
    decision = policy.decide(obs=_obs(), state=PolicyState())
    assert decision.source == "llm"
    assert decision.action["action_type"] in GENERIC_ACTION_MENU
    assert decision.action["target_service"] in {"auth-service", "checkout"}


def test_policy_falls_back_when_llm_unavailable() -> None:
    class BrokenClient(LLMClient):
        def complete_action(self, system_message: str, user_prompt: str, seed: int = 0) -> str:
            raise LLMUnavailable("no key")

    policy = FirewatchPolicy(
        llm_client=BrokenClient(LLMConfig(backend="openai", api_key=None)),
        gnn=GnnBaseline(mode="heuristic"),
    )
    decision = policy.decide(obs=_obs(), state=PolicyState())
    assert decision.source == "llm_unavailable"
    assert decision.action["action_type"] in GENERIC_ACTION_MENU
    # Top suspect should be auth-service (downstream blast radius from checkout).
    assert decision.action["target_service"] == "auth-service"


def test_policy_falls_back_when_action_unparseable() -> None:
    class GarbageClient(LLMClient):
        def complete_action(self, system_message: str, user_prompt: str, seed: int = 0) -> str:
            return "I think you should restart the auth service."

    policy = FirewatchPolicy(
        llm_client=GarbageClient(LLMConfig(backend="openai", api_key="x")),
        gnn=GnnBaseline(mode="heuristic"),
    )
    decision = policy.decide(obs=_obs(), state=PolicyState())
    assert decision.source == "llm_parse_error"
    assert decision.action["target_service"] == "auth-service"


def test_policy_state_updates_history_and_repeats() -> None:
    state = PolicyState()
    next_obs = {"services": {"auth-service": {"recent_logs": ["http 500 retry storm"]}}}
    FirewatchPolicy.update_state_after_step(
        state,
        action={"action_type": "fetch_logs", "target_service": "auth-service"},
        info={"action_feedback": "Fetched 5 logs"},
        next_observation=next_obs,
    )
    assert state.step == 1
    assert state.history == ["fetch_logs:auth-service"]
    assert state.fetched_logs.get("auth-service") == ["http 500 retry storm"]

    FirewatchPolicy.update_state_after_step(
        state,
        action={"action_type": "fetch_logs", "target_service": "auth-service"},
        info={},
        next_observation={"services": {}},
    )
    assert state.repeat_count == 1
