"""
sim_client.py — Lightweight WebSocket client for the FirewatchEnv simulator

Implements the OpenEnv WebSocket protocol without depending on openenv-core.
Connects to the sim's /ws endpoint (local FastAPI server or remote HF Space)
and provides synchronous reset()/step() methods.

Protocol:
    Send:    {"type": "reset", "data": {"seed": N, "difficulty": "easy"}}
    Receive: {"type": "result", "data": {"observation": {...}, "reward": null, "done": false}}

    Send:    {"type": "step", "data": {"action_type": "...", "target_service": "..."}}
    Receive: {"type": "result", "data": {"observation": {...}, "reward": 0.5, "done": false}}

Connection lifecycle: one persistent WebSocket held open for the entire
training run (SPEC-T3 §3.3). Auto-reconnect with 3 retries on connection loss.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import websockets
from websockets.asyncio.client import connect as ws_connect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types (no openenv dependency)
# ---------------------------------------------------------------------------


@dataclass
class SimObservation:
    """Parsed observation from the sim. Holds raw dict for flexibility."""
    raw: dict = field(default_factory=dict)

    @property
    def services(self) -> dict:
        return self.raw.get("services", {})

    @property
    def active_alerts(self) -> list:
        return self.raw.get("active_alerts", [])

    @property
    def slo_budget_remaining_pct(self) -> float:
        return self.raw.get("slo_budget_remaining_pct", 100.0)

    @property
    def sim_tick(self) -> int:
        return self.raw.get("sim_tick", 0)

    @property
    def action_history(self) -> list:
        return self.raw.get("action_history", [])

    @property
    def dependency_graph(self) -> dict:
        return self.raw.get("dependency_graph", {})

    @property
    def done(self) -> bool:
        return self.raw.get("done", False)

    @property
    def episode_score(self) -> float | None:
        return self.raw.get("episode_score")


@dataclass
class StepResult:
    """Result from reset() or step()."""
    observation: SimObservation
    reward: float | None
    done: bool


# ---------------------------------------------------------------------------
# Async WebSocket client (internal)
# ---------------------------------------------------------------------------


class _AsyncSimClient:
    """Async WebSocket client for the sim."""

    def __init__(
        self,
        base_url: str,
        connect_timeout: float = 30.0,
        message_timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 10.0,
    ) -> None:
        # Convert http(s):// to ws(s)://
        ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
        if not ws_url.endswith("/ws"):
            ws_url = ws_url.rstrip("/") + "/ws"
        self._ws_url = ws_url
        self._connect_timeout = connect_timeout
        self._message_timeout = message_timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._ws = None

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        if self._ws is not None:
            return

        # Bypass proxy for localhost
        ws_lower = self._ws_url.lower()
        is_local = "localhost" in ws_lower or "127.0.0.1" in ws_lower

        old_no_proxy = os.environ.get("NO_PROXY")
        if is_local:
            current = old_no_proxy or ""
            if "localhost" not in current.lower():
                os.environ["NO_PROXY"] = (
                    f"{current},localhost,127.0.0.1" if current else "localhost,127.0.0.1"
                )

        try:
            self._ws = await ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
                max_size=100 * 1024 * 1024,  # 100MB
                # Keep the connection alive across long-running GRPO steps.
                # Our client loop now runs in a daemon thread (see SimClient
                # below), so it can pong server keepalive pings autonomously.
                # 60s ping interval + 120s pong timeout tolerates a single
                # ~70s training step without triggering a 1011 close.
                ping_interval=60,
                ping_timeout=120,
            )
            logger.info("Connected to sim at %s", self._ws_url)
        except Exception as exc:
            raise ConnectionError(f"Failed to connect to {self._ws_url}: {exc}") from exc
        finally:
            if is_local:
                if old_no_proxy is None:
                    os.environ.pop("NO_PROXY", None)
                else:
                    os.environ["NO_PROXY"] = old_no_proxy

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "close"}))
            except Exception:
                pass
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _send_and_receive(self, message: dict) -> dict:
        """Send a message and wait for response, with auto-reconnect."""
        for attempt in range(self._max_retries + 1):
            try:
                if self._ws is None:
                    await self.connect()

                await self._ws.send(json.dumps(message))
                raw = await asyncio.wait_for(
                    self._ws.recv(), timeout=self._message_timeout
                )
                response = json.loads(raw)

                if response.get("type") == "error":
                    error_data = response.get("data", {})
                    raise RuntimeError(
                        f"Sim error: {error_data.get('message', 'Unknown')} "
                        f"(code: {error_data.get('code', 'UNKNOWN')})"
                    )

                return response

            except (
                websockets.exceptions.ConnectionClosed,
                ConnectionError,
                OSError,
            ) as exc:
                self._ws = None
                if attempt < self._max_retries:
                    backoff = self._retry_backoff * (attempt + 1)
                    logger.warning(
                        "Connection lost (attempt %d/%d), retrying in %.0fs: %s",
                        attempt + 1, self._max_retries, backoff, exc,
                    )
                    await asyncio.sleep(backoff)
                else:
                    raise RuntimeError(
                        f"Sim connection failed after {self._max_retries} retries: {exc}"
                    ) from exc

        raise RuntimeError("Unexpected retry exhaustion")

    async def reset(self, seed: int, difficulty: str = "easy") -> StepResult:
        """Reset the sim and return initial observation."""
        message = {
            "type": "reset",
            "data": {"seed": seed, "difficulty": difficulty},
        }
        response = await self._send_and_receive(message)
        return self._parse_result(response)

    async def step(self, action: dict) -> StepResult:
        """Execute an action and return the result."""
        message = {
            "type": "step",
            "data": action,
        }
        response = await self._send_and_receive(message)
        return self._parse_result(response)

    @staticmethod
    def _parse_result(response: dict) -> StepResult:
        """Parse a sim response into a StepResult."""
        data = response.get("data", {})
        obs_data = data.get("observation", {})

        return StepResult(
            observation=SimObservation(raw=obs_data),
            reward=data.get("reward"),
            done=data.get("done", False),
        )


# ---------------------------------------------------------------------------
# Synchronous wrapper (used by GRPO rollout — no async in the training loop)
# ---------------------------------------------------------------------------


class SimClient:
    """
    Synchronous WebSocket client for the FirewatchEnv simulator.

    Wraps the async client with a dedicated event loop thread.
    All rollout calls are sequential per SPEC-T3 Constraint 4.

    Usage:
        client = SimClient("http://localhost:8000")
        client.connect()

        result = client.reset(seed=42)
        result = client.step({"action_type": "fetch_logs", "target_service": "auth-service"})

        client.disconnect()
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout: float = 30.0,
        message_timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 10.0,
    ) -> None:
        self._async_client = _AsyncSimClient(
            base_url=base_url,
            connect_timeout=connect_timeout,
            message_timeout=message_timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._closed = False

    def _ensure_loop_thread(self) -> asyncio.AbstractEventLoop:
        """Start the asyncio loop in a daemon thread if needed.

        GRPO can spend 60-90s in GPU generation/training between simulator
        calls. A background loop keeps WebSocket keepalive pongs flowing during
        that blocked period, avoiding server-side 1011 ping timeouts.
        """
        if self._loop is not None and self._thread is not None and self._thread.is_alive():
            return self._loop

        self._started.clear()

        def run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._started.set()
            loop.run_forever()
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

        self._thread = threading.Thread(
            target=run_loop,
            name="firewatch-sim-ws-loop",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=5.0) or self._loop is None:
            raise RuntimeError("Timed out starting simulator WebSocket event loop")
        return self._loop

    def _run(self, coro):
        """Run an async coroutine on the background event loop synchronously."""
        if self._closed:
            self._closed = False
        loop = self._ensure_loop_thread()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    def connect(self) -> None:
        """Establish WebSocket connection."""
        self._run(self._async_client.connect())

    def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._loop is None:
            return
        try:
            self._run(self._async_client.disconnect())
        finally:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
            self._closed = True
            loop.call_soon_threadsafe(loop.stop)
            if thread is not None and thread.is_alive():
                thread.join(timeout=5.0)

    def reset(self, seed: int, difficulty: str = "easy") -> StepResult:
        """Reset the sim (synchronous)."""
        return self._run(self._async_client.reset(seed=seed, difficulty=difficulty))

    def step(self, action: dict) -> StepResult:
        """Execute an action (synchronous)."""
        return self._run(self._async_client.step(action))

    def __enter__(self) -> "SimClient":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    def __del__(self) -> None:
        try:
            self.disconnect()
        except Exception:
            pass
