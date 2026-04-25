"""
Compare the latest baseline log entry to the previous one; optionally write
override hints to a local or Hub file (sft.learning field in config).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from shared import hf_auth, hf_io  # noqa: TID252


def _read_last_n_jsonl(path: Path, n: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    out: list[dict[str, Any]] = []
    for ln in lines[-n:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def check_regression_after_baseline(
    namespace: str,
    local_dir: Path,
    config: dict,
) -> None:
    sft = config.get("sft", {}) or {}
    if not sft.get("regression_guard", True):
        return

    min_delta_success = float(sft.get("regression_min_delta_success_rate", 0.0))
    min_delta_reward = float(sft.get("regression_min_delta_mean_reward", 0.0))

    repo_id = f"{namespace}/firewatch-sft-data"
    pulled = hf_io.pull_baselines_log(repo_id, local_dir)
    if pulled is None or not pulled.exists():
        print("[regression_guard] No baselines/metrics.jsonl on Hub yet")
        return

    cache = local_dir / "baselines" / "metrics.jsonl"
    if cache.exists():
        path = cache
    else:
        path = pulled

    entries = _read_last_n_jsonl(path, 2)
    if len(entries) < 2:
        print("[regression_guard] Need at least 2 baseline rows to compare")
        return

    prev, cur = entries[0], entries[1]
    p_succ = float(prev.get("overall", {}).get("overall_success_rate", 0.0))
    c_succ = float(cur.get("overall", {}).get("overall_success_rate", 0.0))
    p_rew = float(prev.get("overall", {}).get("overall_mean_reward", 0.0))
    c_rew = float(cur.get("overall", {}).get("overall_mean_reward", 0.0))

    d_succ = c_succ - p_succ
    d_rew = c_rew - p_rew
    print(
        f"[regression_guard] success_rate: {p_succ:.3f} -> {c_succ:.3f} (Δ {d_succ:+.3f}); "
        f"mean_reward: {p_rew:.3f} -> {c_rew:.3f} (Δ {d_rew:+.3f})"
    )

    regressed = d_succ < -min_delta_success or d_rew < -min_delta_reward
    if not regressed:
        return

    out_path = os.environ.get("SFT_REGRESSION_OVERRIDE_PATH", "").strip()
    if not out_path:
        agent_root = Path(__file__).resolve().parent.parent
        out_path = str(agent_root / "sft_regression_override.yaml")

    payload = {
        "regression_detected": True,
        "prev_success_rate": p_succ,
        "cur_success_rate": c_succ,
        "prev_mean_reward": p_rew,
        "cur_mean_reward": c_rew,
        "suggested_learning_rate_mult": 0.5,
        "suggested_max_sft_steps_next": 1,
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"[regression_guard] Wrote {out_path} — review before next run")
