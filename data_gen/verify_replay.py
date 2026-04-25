"""
verify_replay — map gold SFT actions to env /step JSON and optional sim smoke test.

SFT JSONL uses ``gold_action_sequence`` with ``action``/``action_type`` and
``params``/``target_service`` (see :func:`data_gen.check_batch._action_name`).

**Static mode (default, no network):** ensure every entry maps to a non-null
``{action_type, target_service, parameters}`` dict a Firewatch client would
send. This is cheap to run in CI or before ``sft.train``.

**Live mode (``--sim-url``):** after ``reset`` with ``task_seed_id`` and a
*deterministic* per-example seed (``replay_env_seed`` in the example, else a
hash of ``example_id``), run ``step`` for each gold action. This does *not* prove
the stored ``observation`` matches the sim — synthetic snapshots are not
logically tied to a single live episode without a recorded env seed. It only
smoke-tests that the action sequence is well-formed and often accepted
(``info.action_valid``) at least for some episode of that task.

Usage::

    uv run python -m data_gen.verify_replay --batch 0
    uv run python -m data_gen.verify_replay --jsonl /path/batch_000.jsonl
    uv run python -m data_gen.verify_replay --batch 0 --sim-url http://127.0.0.1:8000 --max-examples 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from data_gen.check_batch import _action_name, _action_target, batch_path


def candidate_service_names(example: dict) -> list[str]:
    """Service keys from the training example observation (SFT format)."""
    obs = example.get("observation")
    if not isinstance(obs, dict):
        return []
    sm = obs.get("service_metrics")
    if not isinstance(sm, dict):
        return []
    return [str(k) for k in sm.keys()]


def gold_dict_to_env_action(
    action: dict,
    candidate_targets: list[str],
) -> dict[str, Any] | None:
    """Convert one gold_action_sequence dict to a Firewatch ``/step`` action body."""
    if not isinstance(action, dict):
        return None
    atype = _action_name(action)
    if atype is None:
        return None
    raw_params = action.get("parameters") or action.get("params") or {}
    if not isinstance(raw_params, dict):
        raw_params = {}
    if atype in ("declare_resolved", "escalate"):
        return {"action_type": atype, "target_service": None, "parameters": raw_params}
    target = _action_target(action)
    if target is None and candidate_targets:
        target = candidate_targets[0]
    if target is None:
        return None
    return {"action_type": atype, "target_service": target, "parameters": raw_params}


def _replay_seed(example: dict) -> int:
    if "replay_env_seed" in example:
        v = example["replay_env_seed"]
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    eid = str(example.get("example_id") or example.get("task_seed_id") or "na")
    h = hashlib.md5(eid.encode(), usedforsecurity=False).hexdigest()
    return int(h[:8], 16) % (2**31)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_num}: invalid JSON: {exc}") from exc
            if not isinstance(ex, dict):
                raise ValueError(f"{path}: line {line_num}: expected object")
            examples.append(ex)
    return examples


def static_verify_examples(examples: list[dict], prefix: str = "") -> list[str]:
    """Return a list of error strings; empty means OK."""
    errors: list[str] = []
    for i, ex in enumerate(examples):
        label = f"{prefix}example[{i}]"
        gas = ex.get("gold_action_sequence")
        if not isinstance(gas, list) or not gas:
            errors.append(f"{label}: gold_action_sequence missing or empty")
            continue
        cands = candidate_service_names(ex)
        for j, act in enumerate(gas):
            env_act = gold_dict_to_env_action(act, cands) if isinstance(act, dict) else None
            if env_act is None:
                errors.append(
                    f"{label} action[{j}]: could not map to env action (missing type or target)"
                )
    return errors


def _live_replay_one(
    example: dict, base_url: str, timeout: float, stop_on_invalid: bool
) -> list[str]:
    from runners.http_sim_client import HttpSimClient

    errors: list[str] = []
    client = HttpSimClient(base_url, timeout_seconds=timeout)
    if not client.is_healthy():
        return [f"sim at {base_url} not healthy (/health)"]

    tier = example.get("tier")
    task_id = example.get("task_seed_id")
    if not isinstance(tier, str) or not isinstance(task_id, str):
        return ["example missing tier or task_seed_id (str)"]

    seed = _replay_seed(example)
    cands = candidate_service_names(example)
    res = client.reset(tier, seed, task_id)
    if res.info and res.info.get("error"):
        return [f"reset failed: {res.info.get('error')}"]

    gas = example.get("gold_action_sequence")
    if not isinstance(gas, list):
        return ["gold_action_sequence not a list"]

    for j, act in enumerate(gas):
        if not isinstance(act, dict):
            errors.append(f"action[{j}]: not a dict")
            break
        env_act = gold_dict_to_env_action(act, cands)
        if env_act is None:
            errors.append(f"action[{j}]: map to env action failed")
            break
        step_res = client.step(env_act)
        info = step_res.info or {}
        if info.get("action_valid") is False and stop_on_invalid:
            fb = info.get("action_feedback", "")
            errors.append(f"action[{j}]: action_valid=False feedback={fb!r}")
            break
        if step_res.done:
            break
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify gold actions map to env /step JSON (and optional sim smoke test)"
    )
    parser.add_argument("--batch", type=int, help="Zero-indexed batch 0-29 (reviewed/)")
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="Explicit path to a reviewed batch JSONL (overrides --batch)",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=0,
        help="If >0, require this many lines (0 = no count check; merged pairs use 100)",
    )
    parser.add_argument(
        "--sim-url",
        type=str,
        default="",
        help="If set, run HTTP live smoke test against this sim base URL",
    )
    parser.add_argument("--max-examples", type=int, default=5, help="Live: cap examples to replay")
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="HTTP timeout seconds for live mode"
    )
    parser.add_argument(
        "--strict-live",
        action="store_true",
        help="Live: fail on first step with action_valid=False",
    )
    args = parser.parse_args()

    if args.jsonl is not None:
        path = args.jsonl
    elif args.batch is not None:
        if not (0 <= args.batch <= 29):
            print("ERROR: --batch must be 0-29", file=sys.stderr)
            sys.exit(1)
        path = batch_path(args.batch, "reviewed")
    else:
        print("ERROR: provide --jsonl or --batch", file=sys.stderr)
        sys.exit(1)

    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        examples = _load_jsonl(path)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.expected_count > 0 and len(examples) != args.expected_count:
        print(
            f"ERROR: expected {args.expected_count} examples, got {len(examples)}",
            file=sys.stderr,
        )
        sys.exit(1)

    static_errs = static_verify_examples(examples, prefix=f"{path.name}: ")
    if static_errs:
        print(f"ERROR: static verify failed ({len(static_errs)} issues)")
        for e in static_errs[:50]:
            print(f"  {e}")
        if len(static_errs) > 50:
            print(f"  ... and {len(static_errs) - 50} more")
        sys.exit(1)
    print(f"OK: static — {path} ({len(examples)} examples) map to env actions")

    if args.sim_url:
        n = min(args.max_examples, len(examples))
        print(f"Live smoke: first {n} example(s) at {args.sim_url!r}")
        for i in range(n):
            le = _live_replay_one(
                examples[i], args.sim_url, args.timeout, args.strict_live
            )
            if le:
                print(f"ERROR: live example[{i}]: {le[0]}")
                sys.exit(1)
        print("OK: live smoke finished")


if __name__ == "__main__":
    main()
