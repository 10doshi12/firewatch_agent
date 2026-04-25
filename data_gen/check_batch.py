"""
check_batch.py — strict JSONL compliance checks for SFT/GNN batches.

Used after generation/review and by SFT preflight before expensive GPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from data_gen.validate import validate_example

SFT_DATA_DIR = Path(__file__).parent.parent.parent / "sft_data"
RAW_DIR = SFT_DATA_DIR / "raw"
REVIEWED_DIR = SFT_DATA_DIR / "reviewed"


@dataclass
class CheckResult:
    """Result from a batch compliance check."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    example_count: int = 0


def _action_name(action: dict) -> str | None:
    value = action.get("action") or action.get("action_type")
    return value if isinstance(value, str) and value else None


def _action_target(action: dict) -> str | None:
    target = action.get("target_service")
    if isinstance(target, str) and target:
        return target
    params = action.get("params", {})
    if isinstance(params, dict):
        service = params.get("service") or params.get("target_service")
        if isinstance(service, str) and service:
            return service
    return None


def _root_cause_service(example: dict) -> str | None:
    direct = (
        example.get("fault_service")
        or example.get("root_cause_service")
        or example.get("observation", {}).get("root_cause_service")
    )
    if isinstance(direct, str) and direct:
        return direct

    for action in example.get("gold_action_sequence", []):
        if isinstance(action, dict):
            target = _action_target(action)
            if target:
                return target
    return None


def _check_score_range(example: dict, index: int) -> list[str]:
    errors: list[str] = []
    score_range = example.get("expected_score_range")
    if not isinstance(score_range, dict):
        return [f"example[{index}] expected_score_range must be a dict"]

    min_score = score_range.get("min")
    max_score = score_range.get("max")
    if not isinstance(min_score, (int, float)) or not isinstance(max_score, (int, float)):
        errors.append(f"example[{index}] expected_score_range min/max must be numeric")
    elif not (0.0 <= float(min_score) <= float(max_score) <= 1.0):
        errors.append(
            f"example[{index}] expected_score_range must satisfy 0.0 <= min <= max <= 1.0"
        )
    return errors


def _check_actions(example: dict, index: int) -> list[str]:
    errors: list[str] = []
    actions = example.get("gold_action_sequence")
    if not isinstance(actions, list) or not actions:
        return [f"example[{index}] gold_action_sequence must be a non-empty list"]

    for action_index, action in enumerate(actions):
        if not isinstance(action, dict):
            errors.append(f"example[{index}] action[{action_index}] must be a dict")
            continue
        if _action_name(action) is None:
            errors.append(f"example[{index}] action[{action_index}] missing action name")
        name = _action_name(action)
        if name not in {"declare_resolved", "escalate"} and _action_target(action) is None:
            errors.append(f"example[{index}] action[{action_index}] missing target service")
    return errors


def check_examples(examples: list[dict], expected_count: int = 50) -> CheckResult:
    """Run strict compliance checks on in-memory examples."""
    errors: list[str] = []

    if len(examples) != expected_count:
        errors.append(f"expected {expected_count} examples, got {len(examples)}")

    for index, example in enumerate(examples):
        errors.extend(validate_example(example, index=index))

        if not isinstance(example.get("example_id"), str) or not example.get("example_id"):
            errors.append(f"example[{index}] missing example_id")
        if not isinstance(example.get("source_script"), str) or not example.get("source_script"):
            errors.append(f"example[{index}] missing source_script")
        if _root_cause_service(example) is None:
            errors.append(f"example[{index}] missing recoverable root-cause service")

        observation = example.get("observation", {})
        if isinstance(observation, dict):
            if not isinstance(observation.get("service_metrics"), dict):
                errors.append(f"example[{index}] observation.service_metrics must be a dict")
            if not isinstance(observation.get("logs"), dict):
                errors.append(f"example[{index}] observation.logs must be a dict")

        errors.extend(_check_score_range(example, index))
        errors.extend(_check_actions(example, index))

    return CheckResult(ok=not errors, errors=errors, example_count=len(examples))


def check_jsonl_file(jsonl_path: Path, expected_count: int = 50) -> CheckResult:
    """Load and strictly check a batch JSONL file."""
    jsonl_path = Path(jsonl_path)
    errors: list[str] = []
    examples: list[dict] = []

    if not jsonl_path.exists():
        return CheckResult(ok=False, errors=[f"batch file not found: {jsonl_path}"])

    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_num}: invalid JSON: {exc}")
                continue
            if not isinstance(loaded, dict):
                errors.append(f"line {line_num}: example must be a JSON object")
                continue
            examples.append(loaded)

    example_result = check_examples(examples, expected_count=expected_count)
    errors.extend(example_result.errors)
    return CheckResult(ok=not errors, errors=errors, example_count=len(examples))


def batch_path(batch_num: int, stage: str) -> Path:
    """Return the canonical local JSONL path for a batch/stage."""
    if stage == "raw":
        return RAW_DIR / f"batch_{batch_num:03d}.jsonl"
    if stage == "reviewed":
        return REVIEWED_DIR / f"batch_{batch_num:03d}.jsonl"
    raise ValueError(f"stage must be raw or reviewed, got: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SFT batch JSONL compliance")
    parser.add_argument("--batch", type=int, required=True, help="Zero-indexed batch number 0-29")
    parser.add_argument("--stage", choices=["raw", "reviewed"], required=True)
    parser.add_argument("--expected-count", type=int, default=50)
    args = parser.parse_args()

    if not (0 <= args.batch <= 29):
        print(f"ERROR: batch number must be 0-29, got: {args.batch}")
        sys.exit(1)

    path = batch_path(args.batch, args.stage)
    result = check_jsonl_file(path, expected_count=args.expected_count)
    if not result.ok:
        print(f"ERROR: {path} failed compliance ({len(result.errors)} errors)")
        for error in result.errors:
            print(f"  {error}")
        sys.exit(1)

    print(f"OK: {path} passed compliance ({result.example_count} examples)")


if __name__ == "__main__":
    main()
