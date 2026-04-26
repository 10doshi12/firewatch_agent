"""Load Firewatch analysis inputs from JSON and JSONL artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: Path | str) -> list[dict]:
    """Read JSONL records, skipping blank and malformed lines."""
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []

    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                loaded = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                records.append(loaded)
    return records


def load_sft_examples(sft_dir: Path | str) -> list[dict]:
    """Load reviewed SFT examples from sorted batch JSONL files."""
    root = Path(sft_dir)
    examples: list[dict] = []
    for batch_path in sorted(root.glob("batch_*.jsonl")):
        examples.extend(load_jsonl(batch_path))
    return examples


def load_inference_runs(runs_dir: Path | str) -> list[dict]:
    """Load inference run metadata, episode records, and step records."""
    root = Path(runs_dir)
    if not root.exists():
        return []

    runs: list[dict] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata_path = run_dir / "metadata.json"
        metadata: dict = {}
        if metadata_path.exists():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                metadata = loaded

        episodes = load_jsonl(run_dir / "episodes.jsonl")
        steps = load_jsonl(run_dir / "steps.jsonl")
        if metadata or episodes or steps:
            runs.append(
                {
                    "run_dir": str(run_dir),
                    "metadata": metadata,
                    "episodes": episodes,
                    "steps": steps,
                }
            )
    return runs


def load_grpo_metrics(grpo_log: Path | str) -> list[dict]:
    """Load GRPO metrics JSONL records."""
    return load_jsonl(grpo_log)
