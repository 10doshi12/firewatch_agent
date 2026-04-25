"""
http_sim_client.py — Synchronous HTTP client for FirewatchEnv.

The GRPO/eval pipeline talks to the sim over WebSocket via
grpo/sim_client.py. The local production baseline talks over plain HTTP
because:

  * the local FastAPI server exposes /reset, /step, /state and /health
    out of the box (no /ws required for the simple step loop)
  * HTTP makes it trivial to spawn many parallel inference processes
    against the same server during data collection
  * no asyncio in the runtime hot path simplifies the trajectory logger

Returns plain dicts shaped exactly like the server response so callers
do not need to import any project models.

Server contract (firewatch_env/server/app.py):
    POST /reset { "difficulty": str, "seed": int, "task_id": str | null }
        -> { "observation": {...}, "info": {...}, "done": false, "reward": 0.0 }
    POST /step { "action": { "action_type": str, ... } }
        -> { "observation": {...}, "info": {...}, "reward": float, "done": bool }
    GET  /state
        -> { "observation": {...}, "info": {...} }
    GET  /health
        -> 200 OK
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class HttpStepResult:
    observation: dict
    reward: float
    done: bool
    info: dict


class HttpSimClient:
    """Tiny synchronous HTTP client. No retries, no auth — local sim only."""

    def __init__(self, base_url: str, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    # --- Connection probing -------------------------------------------------

    def is_healthy(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/health", timeout=2.0
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    # --- Core verbs ---------------------------------------------------------

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
            return json.loads(resp.read())

    def reset(
        self,
        difficulty: str,
        seed: int,
        task_id: Optional[str] = None,
    ) -> HttpStepResult:
        body: dict = {"difficulty": difficulty, "seed": seed}
        if task_id is not None:
            body["task_id"] = task_id
        payload = self._post("/reset", body)
        return _to_step_result(payload)

    def step(self, action: dict) -> HttpStepResult:
        payload = self._post("/step", {"action": action})
        return _to_step_result(payload)


def _to_step_result(payload: dict) -> HttpStepResult:
    observation = payload.get("observation") or {}
    info = payload.get("info") or {}
    reward = payload.get("reward", 0.0)
    if reward is None:
        reward = 0.0
    return HttpStepResult(
        observation=observation if isinstance(observation, dict) else {},
        reward=float(reward),
        done=bool(payload.get("done", False)),
        info=info if isinstance(info, dict) else {},
    )


# ---------------------------------------------------------------------------
# Auto-detection (mirrors the legacy resolve_server_url contract).
# Probes localhost candidates and falls back to SPACE_URL / HF Space.
# ---------------------------------------------------------------------------


DEFAULT_SPACE_URL = "https://10doshi12-firewatch-env.hf.space"


def resolve_sim_url(explicit: Optional[str] = None) -> str:
    """Return the first reachable sim URL from a fixed candidate list."""
    if explicit:
        return explicit.rstrip("/")

    env_url = os.getenv("SPACE_URL", "").strip().rstrip("/")
    candidates: list[tuple[str, float]] = [
        ("http://localhost:8000", 1.5),
        ("http://localhost:7860", 1.5),
    ]
    seen = {url for url, _ in candidates}
    if env_url and env_url not in seen:
        candidates.append((env_url, 30.0))
        seen.add(env_url)
    if DEFAULT_SPACE_URL not in seen:
        candidates.append((DEFAULT_SPACE_URL, 60.0))

    for base_url, timeout in candidates:
        try:
            with urllib.request.urlopen(
                f"{base_url}/health", timeout=timeout
            ) as resp:
                if resp.status == 200:
                    return base_url
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            continue
    return DEFAULT_SPACE_URL
