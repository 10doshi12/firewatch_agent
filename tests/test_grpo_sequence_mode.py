from __future__ import annotations

import sys
from pathlib import Path


_agent_root = Path(__file__).resolve().parent.parent
if str(_agent_root) not in sys.path:
    sys.path.insert(0, str(_agent_root))

_project_root = _agent_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def test_parse_action_sequence_accepts_actions_array_and_aliases() -> None:
    from grpo.rollout import parse_action_sequence

    completion = """
    Here is the plan:
    {"actions": [
      {"action_type": "collect_logs", "target_service": "auth-service"},
      {"action_type": "scale", "target_service": "auth-service"},
      {"action_type": "resolve"}
    ]}
    """

    assert parse_action_sequence(completion, max_actions=5) == [
        {"action_type": "fetch_logs", "target_service": "auth-service"},
        {"action_type": "scale_replicas", "target_service": "auth-service"},
        {"action_type": "declare_resolved"},
    ]


def test_eval_action_sequence_uses_terminal_episode_score() -> None:
    from grpo.train import eval_action_sequence

    class Observation:
        def __init__(self, episode_score: float | None = None) -> None:
            self.episode_score = episode_score

    class Result:
        def __init__(self, reward: float, done: bool, episode_score: float | None = None) -> None:
            self.reward = reward
            self.done = done
            self.observation = Observation(episode_score)

    class EnvClient:
        def __init__(self) -> None:
            self.reset_calls: list[tuple[int, str]] = []
            self.actions: list[dict] = []

        def reset(self, *, seed: int, difficulty: str = "easy") -> None:
            self.reset_calls.append((seed, difficulty))

        def step(self, action: dict) -> Result:
            self.actions.append(action)
            if action["action_type"] == "declare_resolved":
                return Result(reward=0.0, done=True, episode_score=0.72)
            return Result(reward=0.1, done=False)

    env_client = EnvClient()
    reward = eval_action_sequence(
        env_client=env_client,
        seed=123,
        difficulty="medium",
        actions=[
            {"action_type": "fetch_logs", "target_service": "auth-service"},
            {"action_type": "scale_replicas", "target_service": "auth-service"},
            {"action_type": "declare_resolved"},
        ],
        max_actions=5,
    )

    assert env_client.reset_calls == [(123, "medium")]
    assert [action["action_type"] for action in env_client.actions] == [
        "fetch_logs",
        "scale_replicas",
        "declare_resolved",
    ]
    assert reward > 0.72
