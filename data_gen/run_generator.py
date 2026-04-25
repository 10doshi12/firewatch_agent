"""
run_generator.py — Phase A: Generator Execution

Invokes one of the 30 generator scripts, validates output, writes raw batch.
Usage: python -m firewatch_agent.data_gen.run_generator --script 01
"""

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

# Ensure firewatch_env is on path for importing TASKS
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "firewatch_env"))

from config import TASKS

# Paths
SCRIPT_DIR = Path(__file__).parent.parent / "data_gen_scripts"
SFT_DATA_DIR = Path(__file__).parent.parent.parent / "sft_data"
RAW_DIR = SFT_DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_tasks() -> list[dict]:
    """Convert TASKS dict to list of task dicts for generators."""
    return [
        {
            "task_id": tc.task_id,
            "difficulty": tc.difficulty,
            "fault_type": tc.fault_type,
            "fault_service": tc.fault_service,
            "services": list(tc.services) if tc.services else [],
            "red_herrings": list(tc.red_herrings) if tc.red_herrings else [],
            "initial_state_overrides": tc.initial_state_overrides or {},
        }
        for tc in TASKS.values()
    ]


def validate_example(example: dict, index: int) -> list[str]:
    """Validate a single example against schema. Returns list of errors."""
    errors = []
    required_fields = [
        "task_seed_id",
        "tier",
        "fault_type",
        "variation_strategy",
        "observation",
        "gold_action_sequence",
        "gold_alternatives",
        "expected_score_range",
        "suboptimal_paths",
    ]
    for field in required_fields:
        if field not in example:
            errors.append(f"example[{index}] missing required field: {field}")

    if "observation" in example:
        obs = example["observation"]
        for field in ("tick", "budget", "alerts", "service_metrics", "logs"):
            if field not in obs:
                errors.append(f"example[{index}] observation missing: {field}")
        if "alerts" in obs and not isinstance(obs["alerts"], list):
            errors.append(f"example[{index}] observation.alerts must be list")
        if "service_metrics" in obs and not isinstance(obs["service_metrics"], dict):
            errors.append(f"example[{index}] observation.service_metrics must be dict")

    if "tier" in example and example["tier"] not in ("easy", "medium", "hard"):
        errors.append(f"example[{index}] tier must be easy/medium/hard, got: {example['tier']}")

    return errors


_ACTION_CALL_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<service>[^)]*)\)$")


def _normalize_action(action: object) -> object:
    """Convert legacy string actions into structured action dictionaries."""
    if not isinstance(action, str):
        return action

    action = action.strip()
    match = _ACTION_CALL_RE.match(action)
    if match:
        service = match.group("service").strip()
        params = {"service": service} if service else {}
        return {"action": match.group("name"), "params": params}
    return {"action": action, "params": {}}


def normalize_example_contract(example: dict, source_task: dict | None = None) -> None:
    """Normalize legacy generator output into the stricter SFT/GNN contract."""
    score_range = example.get("expected_score_range")
    if (
        isinstance(score_range, list)
        and len(score_range) == 2
        and all(isinstance(value, (int, float)) for value in score_range)
    ):
        example["expected_score_range"] = {
            "min": float(score_range[0]),
            "max": float(score_range[1]),
        }

    actions = example.get("gold_action_sequence")
    if isinstance(actions, list):
        example["gold_action_sequence"] = [_normalize_action(action) for action in actions]

    if source_task and not example.get("fault_service"):
        fault_service = source_task.get("fault_service")
        if isinstance(fault_service, str) and fault_service:
            example["fault_service"] = fault_service


def resolve_script_and_batch(script: str | None, batch: int | None) -> tuple[int, int]:
    """Resolve CLI script/batch input to (script_num, batch_num)."""
    if script is None and batch is None:
        raise ValueError("either --script or --batch is required")

    script_num: int | None = None
    if script is not None:
        script_num = int(script)
        if not (1 <= script_num <= 30):
            raise ValueError(f"script number must be 01-30, got: {script}")

    batch_num: int | None = None
    if batch is not None:
        if not (0 <= batch <= 29):
            raise ValueError(f"batch number must be 0-29, got: {batch}")
        batch_num = batch

    if script_num is not None and batch_num is not None:
        expected_script = batch_num + 1
        if script_num != expected_script:
            raise ValueError(
                f"conflicting --script/--batch: script {script_num:02d} maps to "
                f"batch {script_num - 1:03d}, but --batch {batch_num} was supplied"
            )

    if script_num is None:
        assert batch_num is not None
        script_num = batch_num + 1
    else:
        batch_num = script_num - 1

    return script_num, batch_num


def main():
    parser = argparse.ArgumentParser(description="Run a generator script and produce a raw batch")
    parser.add_argument("--script", required=False, help="Script number 01-30")
    parser.add_argument("--batch", required=False, type=int, help="Zero-indexed batch number 0-29")
    args = parser.parse_args()

    try:
        script_num, batch_num = resolve_script_and_batch(args.script, args.batch)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Resolved script {script_num:02d} -> batch {batch_num:03d}")
    raw_file = RAW_DIR / f"batch_{batch_num:03d}.jsonl"

    # Refuse to overwrite existing raw batch
    if raw_file.exists():
        print(f"ERROR: raw batch already exists: {raw_file}")
        print("Delete it or use a different script number to proceed.")
        sys.exit(1)

    # Import the generator script
    script_name = f"gen_{script_num:02d}_mixed_metric_a.py"  # placeholder naming
    # Find actual script by prefix
    scripts_found = list(SCRIPT_DIR.glob(f"gen_{script_num:02d}_*.py"))
    if not scripts_found:
        print(f"ERROR: no generator script found for number {script_num:02d}")
        print(f"Looked in: {SCRIPT_DIR}")
        sys.exit(1)

    script_path = scripts_found[0]
    print(f"Loading generator: {script_path.name}")

    # Import generator module
    import importlib.util
    spec = importlib.util.spec_from_file_location("generator", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Load tasks
    tasks = load_tasks()
    tasks_by_id = {task["task_id"]: task for task in tasks}
    print(f"Loaded {len(tasks)} tasks from config")

    # Compute rng_seed = batch_number * 1000
    rng_seed = script_num * 1000
    print(f"Using rng_seed={rng_seed} (batch_num={batch_num})")

    # Generate
    print(f"Generating examples...")
    examples = module.generate(tasks, rng_seed)
    print(f"Generated {len(examples)} examples")

    for example in examples:
        task_id = example.get("task_seed_id")
        source_task = tasks_by_id.get(task_id) if isinstance(task_id, str) else None
        normalize_example_contract(example, source_task)

    # Validate count
    if len(examples) != 50:
        print(f"ERROR: generator produced {len(examples)} examples, expected 50")
        sys.exit(1)

    # Validate each example
    print("Validating examples...")
    for i, example in enumerate(examples):
        errors = validate_example(example, i)
        if errors:
            for err in errors:
                print(f"  {err}")
            print(f"ERROR: example[{i}] failed validation")
            sys.exit(1)

    print("All examples passed schema validation")

    # Assign UUIDs and source_script
    for example in examples:
        example["example_id"] = str(uuid.uuid4())
        example["source_script"] = script_path.name

    # Write raw batch
    print(f"Writing to {raw_file}")
    with open(raw_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Raw batch written: {raw_file.name} ({len(examples)} examples)")
    print("\nNext step: python -m firewatch_agent.data_gen.review --batch {batch_num}")


if __name__ == "__main__":
    main()