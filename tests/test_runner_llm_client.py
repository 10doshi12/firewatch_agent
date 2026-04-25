"""Tests for runners.llm_client — backend dispatch & echo stub."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from runners.llm_client import (  # noqa: E402
    LLMClient,
    LLMConfig,
    LLMUnavailable,
    llm_config_from_env,
)


def test_default_config_picks_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    cfg = llm_config_from_env()
    assert cfg.backend == "openai"
    assert cfg.base_url.startswith("http")


def test_ollama_config_targets_local_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("MODEL_NAME", "qwen2.5:14b-instruct")
    cfg = llm_config_from_env()
    assert cfg.backend == "ollama"
    assert "11434" in cfg.base_url
    assert cfg.model == "qwen2.5:14b-instruct"


def test_echo_backend_returns_deterministic_action() -> None:
    client = LLMClient(LLMConfig(backend="echo"))
    response = client.complete_action(
        system_message="sys",
        user_prompt="auth-service: error_rate=0.42 latency_p99=0.50s",
    )
    assert "fetch_logs" in response
    assert "auth-service" in response


def test_openai_backend_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(LLMConfig(backend="openai", api_key=None))
    with pytest.raises(LLMUnavailable):
        client.complete_action("sys", "user")
