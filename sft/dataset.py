"""
dataset.py — SFT batch loader (SPEC-T2 §6)

Loads reviewed batch JSONL, validates each example, splits train/val.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from data_gen.validate import validate_example


def load_batch(jsonl_path: Path) -> list[dict]:
    """
    Load and validate a reviewed batch JSONL file.

    Args:
        jsonl_path: Path to reviewed/batch_<NNN>.jsonl

    Returns:
        List of 50 validated training examples.

    Raises:
        FileNotFoundError: If the JSONL file doesn't exist.
        ValueError: If any example fails validation.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Batch file not found: {jsonl_path}")

    examples: list[dict] = []
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {jsonl_path}: {exc}"
                ) from exc
            examples.append(example)

    # Validate every example
    for i, example in enumerate(examples):
        errors = validate_example(example, index=i)
        if errors:
            example_id = example.get("example_id", f"index-{i}")
            raise ValueError(
                f"Validation failed for example {example_id} "
                f"in {jsonl_path}:\n" + "\n".join(errors)
            )

    if len(examples) != 50:
        print(
            f"[dataset] WARNING: Expected 50 examples in {jsonl_path}, "
            f"got {len(examples)}. Proceeding with {len(examples)}."
        )

    return examples


def split_batch(
    examples: list[dict],
    batch_num: int,
    train_ratio: float = 0.8,
) -> tuple[list[dict], list[dict]]:
    """
    Deterministic 80/20 train/val split, seeded by batch number.

    Args:
        examples: List of training examples
        batch_num: Batch number used as RNG seed
        train_ratio: Fraction for training (default 0.8)

    Returns:
        (train_examples, val_examples)
    """
    rng = random.Random(batch_num)
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    split_point = int(train_ratio * len(indices))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    train_examples = [examples[i] for i in train_indices]
    val_examples = [examples[i] for i in val_indices]

    return train_examples, val_examples
