"""
policy.py — Compose GNN baseline + LLM into a single step policy.

Decision order each step:

  1. Build a *clean* prompt via honest_prompt.build_user_prompt. The
     prompt contains telemetry, dependency graph, GNN blurb, action menu,
     fetched logs, alerts and a no-reward action history line. It does
     not contain rewards, scores, task descriptions or fault hints.

  2. Call the LLM. Parse the response into an action dict.

  3. If the LLM is unavailable OR the response can't be parsed into a
     valid action, fall back to a tiny deterministic policy that uses
     only the GNN ranking and the agent's investigation state. The
     fallback is intentionally simple — it is not a hand-written
     controller meant to "win", it is a safety net.

The PolicyState carries cross-step memory the model sees through history
plus the locally-buffered fetched logs (the env returns logs in
recent_logs but we surface them as a structured block in the prompt).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .gnn_baseline import GnnBaseline, GnnRankItem
from .honest_prompt import (
    GENERIC_ACTION_MENU,
    HONEST_SYSTEM_MESSAGE,
    META_ACTIONS,
    episode_services,
    build_user_prompt,
)
from .llm_client import LLMClient, LLMUnavailable


# ---------------------------------------------------------------------------
# Action parser — accepts the LLM's response and normalises into a dict
# matching the FirewatchAction shape the env expects.
# ---------------------------------------------------------------------------


_ACTION_JSON_RE = re.compile(r"\{[^{}]*\"action(?:_type)?\"[^{}]*\}", re.DOTALL)


def _normalize_action(raw: dict, candidate_targets: list[str]) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    action_type = raw.get("action_type") or raw.get("action")
    if not isinstance(action_type, str) or not action_type:
        return None

    target = raw.get("target_service") or raw.get("service")
    if target is None:
        targets = raw.get("targets")
        if isinstance(targets, list) and targets:
            target = targets[0]
        elif isinstance(targets, str):
            target = targets

    if action_type in META_ACTIONS:
        target = None
    elif candidate_targets and target not in candidate_targets:
        target = candidate_targets[0]

    parameters = raw.get("parameters") or raw.get("params") or {}
    if not isinstance(parameters, dict):
        parameters = {}

    return {
        "action_type": action_type,
        "target_service": target,
        "parameters": parameters,
    }


def parse_action(text: str, candidate_targets: list[str]) -> Optional[dict]:
    """Return a normalised action dict, or None if parsing fails."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
        action = _normalize_action(parsed, candidate_targets)
        if action is not None:
            return action
    except json.JSONDecodeError:
        pass

    for match in _ACTION_JSON_RE.findall(cleaned):
        try:
            parsed = json.loads(match)
        except json.JSONDecodeError:
            continue
        action = _normalize_action(parsed, candidate_targets)
        if action is not None:
            return action
    return None


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class PolicyState:
    step: int = 0
    fetched_logs: dict = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    last_action_per_service: dict = field(default_factory=dict)
    repeat_count: int = 0


@dataclass
class PolicyDecision:
    action: dict
    source: str  # "llm" | "fallback" | "llm_unavailable" | "llm_parse_error"
    raw_response: str
    prompt: str


class FirewatchPolicy:
    """LLM-first policy with deterministic GNN-driven fallback."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        gnn: GnnBaseline | None = None,
        system_message: str = HONEST_SYSTEM_MESSAGE,
        inform_agent: bool = False,
    ) -> None:
        self.llm = llm_client or LLMClient()
        self.gnn = gnn or GnnBaseline(mode="heuristic")
        self.system_message = system_message
        self.inform_agent = inform_agent

    # --- Per-step entry point ----------------------------------------------

    def decide(
        self,
        obs: dict,
        state: PolicyState,
        seed: int = 0,
        last_reward: float | None = None,
    ) -> PolicyDecision:
        ranked = self.gnn.rank(obs)
        gnn_blurb = self.gnn.blurb(obs, ranked)

        history = list(state.history)
        if self.inform_agent and last_reward is not None and history:
            # Append reward to last history entry — explicit ablation only.
            history[-1] = f"{history[-1]}  (last_step_reward={last_reward:+.2f})"

        candidate_targets = list(episode_services(obs).keys())

        prompt = build_user_prompt(
            obs=obs,
            history=history,
            fetched_logs=state.fetched_logs,
            gnn_blurb=gnn_blurb,
        )

        raw_response = ""
        source = "llm"
        try:
            raw_response = self.llm.complete_action(
                self.system_message, prompt, seed=seed
            )
        except LLMUnavailable:
            return PolicyDecision(
                action=self._fallback_action(state, ranked, candidate_targets),
                source="llm_unavailable",
                raw_response="",
                prompt=prompt,
            )

        action = parse_action(raw_response, candidate_targets)
        if action is None:
            return PolicyDecision(
                action=self._fallback_action(state, ranked, candidate_targets),
                source="llm_parse_error",
                raw_response=raw_response,
                prompt=prompt,
            )

        if action.get("action_type") not in GENERIC_ACTION_MENU:
            return PolicyDecision(
                action=self._fallback_action(state, ranked, candidate_targets),
                source="llm_invalid_action",
                raw_response=raw_response,
                prompt=prompt,
            )
        return PolicyDecision(action=action, source=source, raw_response=raw_response, prompt=prompt)

    # --- Fallback ----------------------------------------------------------

    def _fallback_action(
        self,
        state: PolicyState,
        ranked: list[GnnRankItem],
        candidate_targets: list[str],
    ) -> dict:
        if not ranked and not candidate_targets:
            return {"action_type": "declare_resolved", "target_service": None, "parameters": {}}

        # Prefer the GNN's top suspect, but only if that service is part of
        # the current episode. The untrained / heuristic GNN ranks over the
        # full static service registry, which can include services that
        # were never instantiated for the current episode.
        target = next(
            (r.service for r in ranked if r.service in candidate_targets),
            None,
        )
        if target is None:
            target = candidate_targets[0] if candidate_targets else (ranked[0].service if ranked else None)
        if target is None:
            return {"action_type": "declare_resolved", "target_service": None, "parameters": {}}
        if state.step <= 2 and target not in state.fetched_logs:
            return {
                "action_type": "fetch_logs",
                "target_service": target,
                "parameters": {},
            }
        if state.step == 3:
            return {
                "action_type": "trace_dependencies",
                "target_service": target,
                "parameters": {},
            }

        last_action = state.last_action_per_service.get(target)
        if last_action == "restart_service":
            chosen = "rollback_deploy"
        elif last_action == "rollback_deploy":
            chosen = "revert_config"
        elif last_action == "revert_config":
            chosen = "circuit_break"
        else:
            chosen = "restart_service"
        return {
            "action_type": chosen,
            "target_service": target,
            "parameters": {},
        }

    # --- Bookkeeping helpers (called by the runner) ------------------------

    @staticmethod
    def update_state_after_step(
        state: PolicyState,
        action: dict,
        info: dict,
        next_observation: dict,
    ) -> None:
        state.step += 1
        atype = action.get("action_type", "?")
        target = action.get("target_service")
        state.history.append(f"{atype}:{target}")

        if target:
            prev = state.last_action_per_service.get(target)
            if prev == atype:
                state.repeat_count += 1
            else:
                state.repeat_count = 0
            state.last_action_per_service[target] = atype

        # Surface any logs fetched by the agent into a structured per-service map.
        if atype == "fetch_logs" and target:
            services = next_observation.get("services") or {}
            metrics = services.get(target) or {}
            recent = metrics.get("recent_logs") or []
            if recent:
                state.fetched_logs[target] = list(recent)
            else:
                feedback = info.get("feedback_string") if isinstance(info, dict) else None
                if feedback:
                    state.fetched_logs[target] = [feedback]
