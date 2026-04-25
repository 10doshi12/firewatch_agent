"""
validate.py — Schema validator for SFT training examples

Used by both run_generator.py (raw-write validation) and review.py (edit validation).
Validates against SPEC-01 v2 Section 4 schema contract.
"""

from __future__ import annotations

REQUIRED_TOP_LEVEL = [
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

REQUIRED_OBSERVATION = ["tick", "budget", "alerts", "service_metrics", "logs"]

VALID_TIERS = {"easy", "medium", "hard"}


def validate_example(example: dict, index: int = 0) -> list[str]:
    """
    Validate a single training example against the schema.

    Returns a list of error strings. Empty list = valid.
    """
    errors: list[str] = []

    # Top-level required fields
    for field in REQUIRED_TOP_LEVEL:
        if field not in example:
            errors.append(f"example[{index}] missing required field: {field}")

    # Tier validation
    if "tier" in example and example["tier"] not in VALID_TIERS:
        errors.append(
            f"example[{index}] tier must be one of {VALID_TIERS}, got: {example['tier']}"
        )

    # Observation sub-fields
    if "observation" in example:
        obs = example["observation"]
        if not isinstance(obs, dict):
            errors.append(f"example[{index}] observation must be a dict")
        else:
            for field in REQUIRED_OBSERVATION:
                if field not in obs:
                    errors.append(f"example[{index}] observation missing: {field}")

            if "alerts" in obs and not isinstance(obs["alerts"], list):
                errors.append(f"example[{index}] observation.alerts must be a list")

            if "service_metrics" in obs and not isinstance(obs["service_metrics"], dict):
                errors.append(
                    f"example[{index}] observation.service_metrics must be a dict"
                )

            if "logs" in obs and not isinstance(obs["logs"], dict):
                errors.append(f"example[{index}] observation.logs must be a dict")

    # Gold action sequence
    if "gold_action_sequence" in example:
        gas = example["gold_action_sequence"]
        if not isinstance(gas, list) or len(gas) == 0:
            errors.append(
                f"example[{index}] gold_action_sequence must be a non-empty list"
            )

    # Expected score range
    if "expected_score_range" in example:
        esr = example["expected_score_range"]
        if not isinstance(esr, dict):
            errors.append(f"example[{index}] expected_score_range must be a dict")

    return errors


def validate_batch(examples: list[dict]) -> list[str]:
    """Validate an entire batch of examples. Returns all errors."""
    all_errors: list[str] = []
    for i, example in enumerate(examples):
        all_errors.extend(validate_example(example, i))
    return all_errors
