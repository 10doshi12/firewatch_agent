"""
upload.py — Phase C: Upload reviewed batch to HuggingFace

Usage: python -m firewatch_agent.data_gen.upload --batch 6
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

SFT_DATA_DIR = Path(__file__).parent.parent.parent / "sft_data"
REVIEWED_DIR = SFT_DATA_DIR / "reviewed"


def upload_batch(batch_num: int) -> None:
    """Upload a reviewed batch JSONL to HuggingFace."""
    reviewed_file = REVIEWED_DIR / f"batch_{batch_num:03d}.jsonl"

    if not reviewed_file.exists():
        print(f"ERROR: reviewed batch not found: {reviewed_file}")
        print("Run review.py first to produce the reviewed batch.")
        sys.exit(1)

    # Count accepted examples
    with open(reviewed_file) as f:
        count = sum(1 for _ in f)

    # Find the source script name from the first example
    import json
    with open(reviewed_file) as f:
        first = json.loads(f.readline())
    source_script = first.get("source_script", "unknown")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    commit_msg = (
        f"batch_{batch_num:03d} from {source_script} "
        f"({count} examples) @ {ts}"
    )

    print(f"Uploading {reviewed_file.name} ({count} examples)...")
    print(f"Commit: {commit_msg}")

    # Import shared layer for HF I/O
    from shared.hf_io import push_reviewed_batch

    push_reviewed_batch(batch_num, reviewed_file, commit_msg)
    print(f"Upload complete for batch {batch_num:03d}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a reviewed batch to HuggingFace"
    )
    parser.add_argument("--batch", required=True, type=int, help="Batch number (0-29)")
    args = parser.parse_args()

    if not (0 <= args.batch <= 29):
        print(f"ERROR: batch number must be 0-29, got: {args.batch}")
        sys.exit(1)

    upload_batch(args.batch)


if __name__ == "__main__":
    main()
