"""
llm_client.py — Pluggable OpenAI-compatible LLM client.

Backends (selected by LLM_BACKEND env var or constructor):
  openai   — any OpenAI-API-compatible endpoint (OpenRouter, HuggingFace
             router, vLLM, Together). Uses API_BASE_URL + API_KEY.
  ollama   — local Ollama daemon at http://localhost:11434/v1. No API
             key required. Designed for the "small 14B local model" path.
             User runs:  ollama pull qwen2.5:14b-instruct  then sets
             LLM_BACKEND=ollama and MODEL_NAME=qwen2.5:14b-instruct.
  echo     — deterministic stub for tests. Returns a fixed JSON action so
             that policy and trajectory logger tests do not require
             network access or an OpenAI client install.

The interface is intentionally minimal: a single `complete_action(...)`
method that returns the raw model text. Action JSON parsing is the
caller's responsibility (see policy.parse_action) so that the client
stays untyped and reusable.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "openai"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None
    model: str = "google/gemma-4-31b-it"
    temperature: float = 0.3
    max_tokens: int = 256
    timeout_seconds: float = 60.0


def llm_config_from_env() -> LLMConfig:
    """Build an LLMConfig from environment variables."""
    backend = os.getenv("LLM_BACKEND", "openai").strip().lower()

    if backend == "ollama":
        return LLMConfig(
            backend="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/"),
            api_key="ollama",
            model=os.getenv("MODEL_NAME", "qwen2.5:14b-instruct"),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("MAX_TOKENS", "256")),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT", "120.0")),
        )

    if backend == "echo":
        return LLMConfig(backend="echo", model="echo-stub")

    return LLMConfig(
        backend="openai",
        base_url=os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
        api_key=os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY"),
        model=os.getenv("MODEL_NAME", "google/gemma-4-31b-it"),
        temperature=float(os.getenv("TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("MAX_TOKENS", "256")),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT", "60.0")),
    )


# ---------------------------------------------------------------------------
# Client implementation
# ---------------------------------------------------------------------------


class LLMUnavailable(RuntimeError):
    """Raised when the configured backend cannot be reached."""


class LLMClient:
    """Thin OpenAI-compatible chat client. Backend-agnostic."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or llm_config_from_env()

    # --- Public API ---------------------------------------------------------

    def complete_action(self, system_message: str, user_prompt: str, seed: int = 0) -> str:
        """Return the raw model text. Caller parses the JSON action."""
        if self.config.backend == "echo":
            return self._echo_response(user_prompt)
        return self._openai_compatible_chat(system_message, user_prompt, seed=seed)

    def assert_ready(self) -> None:
        """Raise SystemExit early if the backend will fail on every call.

        Called once at runner startup so a missing API key surfaces as a
        clear FATAL message instead of being swallowed by `policy.decide`'s
        silent fallback path on every step.
        """
        if self.config.backend == "echo":
            return
        if self.config.backend == "openai" and not self.config.api_key:
            raise SystemExit(
                "[FATAL] LLM_BACKEND=openai but no API_KEY/HF_TOKEN/OPENAI_API_KEY "
                "found in environment. Either source firewatch_env/.env (which "
                "already contains the OpenRouter key) or pass --backend echo / "
                "--backend ollama."
            )

    # --- Backends -----------------------------------------------------------

    def _echo_response(self, user_prompt: str) -> str:
        # Deterministic stub: pick the first active service mentioned in the
        # prompt and emit a fetch_logs action. Used by tests.
        target = "auth-service"
        for line in user_prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "* ")) or "error_rate=" in stripped:
                first_token = stripped.split(":", 1)[0].lstrip("-* ").strip()
                if first_token:
                    target = first_token
                    break
        return json.dumps(
            {"action_type": "fetch_logs", "target_service": target, "parameters": {}}
        )

    def _openai_compatible_chat(
        self,
        system_message: str,
        user_prompt: str,
        seed: int,
    ) -> str:
        """POST to /chat/completions on the configured base_url."""
        if not self.config.api_key:
            raise LLMUnavailable(
                f"backend={self.config.backend} requires an API key "
                f"(set API_KEY / HF_TOKEN / OPENAI_API_KEY in the environment)."
            )

        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if self.config.backend != "ollama":
            body["seed"] = seed

        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                detail = ""
            raise LLMUnavailable(f"HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise LLMUnavailable(str(exc)) from exc

        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(f"malformed LLM response: {payload!r}") from exc
