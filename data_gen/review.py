"""
review.py — Phase B: Human Review CLI

Walks the human through a raw batch, accept/reject/edit each example.
Usage: python -m firewatch_agent.data_gen.review --batch 6
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

SFT_DATA_DIR = Path(__file__).parent.parent.parent / "sft_data"
RAW_DIR = SFT_DATA_DIR / "raw"
REVIEWED_DIR = SFT_DATA_DIR / "reviewed"
REVIEW_STATE_DIR = SFT_DATA_DIR / "review_state"
REVIEW_STATE_DIR.mkdir(parents=True, exist_ok=True)
REVIEWED_DIR.mkdir(parents=True, exist_ok=True)


def load_review_state(batch_num: int) -> dict:
    """Load existing review state or create new."""
    state_file = REVIEW_STATE_DIR / f"batch_{batch_num:03d}.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {
        "batch_number": batch_num,
        "total_examples": 50,
        "cursor": 0,
        "decisions": {},  # example_id -> "accepted" | "rejected"
        "edits": {},  # example_id -> edited example dict
    }


def save_review_state(state: dict, batch_num: int) -> None:
    """Atomically write review state (temp file + rename)."""
    state_file = REVIEW_STATE_DIR / f"batch_{batch_num:03d}.json"
    tmp = state_file.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    tmp.rename(state_file)


def format_example(example: dict, index: int) -> str:
    """Format an example for human review display."""
    obs = example.get("observation", {})
    metrics = obs.get("service_metrics", {})

    lines = [
        f"=== Example {index + 1} of 50 ===",
        f"ID: {example.get('example_id', 'N/A')}",
        f"Source: {example.get('source_script', 'N/A')}",
        f"Task: {example.get('task_seed_id', 'N/A')}",
        f"Tier: {example.get('tier', 'N/A')}",
        f"Fault: {example.get('fault_type', 'N/A')}",
        f"Tick: {obs.get('tick', 'N/A')}, Budget: {obs.get('budget', 'N/A')}",
        "",
        "--- Alerts ---",
    ]
    for alert in obs.get("alerts", []):
        lines.append(f"  {alert}")

    lines.append("")
    lines.append("--- Gold Actions ---")
    for i, action in enumerate(example.get("gold_action_sequence", [])):
        lines.append(f"  {i + 1}. {action}")

    lines.append("")
    lines.append("--- Service Metrics (key services only) ---")
    # Show first 5 services
    for svc, m in list(metrics.items())[:5]:
        status = m.get("status", "?")
        err_rate = m.get("http_server_error_rate", "?")
        mem = m.get("process_memory_utilization", None)
        restart = m.get("restart_count", None)
        mem_str = f", mem={mem}" if mem is not None else ""
        restart_str = f", restarts={restart}" if restart is not None else ""
        lines.append(f"  {svc}: status={status}, error={err_rate}{mem_str}{restart_str}")

    lines.append("")
    lines.append("--- Logs (per service) ---")
    for svc, logs in obs.get("logs", {}).items():
        if logs:
            lines.append(f"  [{svc}]")
            for log in logs[:3]:
                lines.append(f"    {log}")
            if len(logs) > 3:
                lines.append(f"    ... and {len(logs) - 3} more")

    return "\n".join(lines)


def open_in_editor(example: dict, temp_file: Path) -> dict | None:
    """Open example JSON in $EDITOR, return edited dict or None if cancelled."""
    with open(temp_file, "w") as f:
        json.dump(example, f, indent=2)

    editor = os.environ.get("EDITOR", "nano")
    exit_code = os.system(f"{editor} {temp_file}")

    if exit_code != 0:
        print("Editor cancelled or failed.")
        return None

    try:
        with open(temp_file) as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("Invalid JSON in editor. Please fix and save.")
        return None


def run_review(batch_num: int) -> None:
    """Run the review loop for a batch."""
    raw_file = RAW_DIR / f"batch_{batch_num:03d}.jsonl"
    if not raw_file.exists():
        print(f"ERROR: raw batch not found: {raw_file}")
        print("Run run_generator.py first to generate the batch.")
        sys.exit(1)

    # Load raw examples
    print(f"Loading {raw_file}...")
    with open(raw_file) as f:
        raw_examples = [json.loads(line) for line in f]

    print(f"Loaded {len(raw_examples)} raw examples")

    # Load or initialize review state
    state = load_review_state(batch_num)
    cursor = state["cursor"]
    decisions = state["decisions"]
    edits = state["edits"]

    already_decided = len(decisions)
    print(f"\nReview state: cursor={cursor}, already decided={already_decided}/50")

    if cursor >= 50:
        print("All examples already reviewed! Use --replay to review again.")
        sys.exit(0)

    print("\nCommands: a=accept, r=reject, e=edit, q=quit")
    print("-" * 60)

    # Process examples from cursor forward
    while cursor < 50:
        example = raw_examples[cursor]
        example_id = example.get("example_id", f"idx_{cursor}")

        # Check if already decided
        if example_id in decisions:
            cursor += 1
            continue

        # Display example
        print(format_example(example, cursor))
        print()

        # Prompt for decision
        while True:
            print(f"[{cursor + 1}/50] Decision (a/r/e/q): ", end="", flush=True)
            try:
                line = input().strip().lower()
            except EOFError:
                print("\n(EOF - saving state)")
                break

            if line == "q":
                print("Saving state and quitting...")
                state["cursor"] = cursor
                save_review_state(state, batch_num)
                print(f"Resume later with: python -m firewatch_agent.data_gen.review --batch {batch_num}")
                return

            elif line == "a":
                decisions[example_id] = "accepted"
                print("  -> Accepted")
                break

            elif line == "r":
                decisions[example_id] = "rejected"
                print("  -> Rejected")
                break

            elif line == "e":
                # Open in editor
                with tempfile.NamedTemporaryFile(
                    suffix=".json", mode="w", delete=False
                ) as tmp:
                    temp_path = Path(tmp.name)

                edited = open_in_editor(example, temp_path)
                temp_path.unlink(missing_ok=True)

                if edited is not None:
                    # Validate edited example
                    errors = []
                    if "observation" not in edited:
                        errors.append("missing 'observation'")
                    if "gold_action_sequence" not in edited:
                        errors.append("missing 'gold_action_sequence'")

                    if errors:
                        print(f"Invalid edited example: {errors}")
                        continue

                    edits[example_id] = edited
                    decisions[example_id] = "accepted"
                    print("  -> Accepted (edited)")
                    break
                else:
                    print("  -> Edit cancelled")
                    continue

            else:
                print("Unknown command. Use a/r/e/q")

        cursor += 1

        # Save state after every decision
        state["cursor"] = cursor
        state["decisions"] = decisions
        state["edits"] = edits
        save_review_state(state, batch_num)

        print()

    # All examples reviewed — build reviewed batch
    print("\n=== Review Complete ===")
    accepted = [eid for eid, d in decisions.items() if d == "accepted"]
    rejected = [eid for eid, d in decisions.items() if d == "rejected"]
    edited_count = len(edits)

    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Edited: {edited_count}")

    # Build reviewed examples (preserve original order)
    reviewed_examples = []
    for example in raw_examples:
        eid = example.get("example_id", "")
        if eid in decisions and decisions[eid] == "accepted":
            if eid in edits:
                reviewed_examples.append(edits[eid])
            else:
                reviewed_examples.append(example)

    # Write reviewed batch
    reviewed_file = REVIEWED_DIR / f"batch_{batch_num:03d}.jsonl"
    print(f"\nWriting {len(reviewed_examples)} reviewed examples to {reviewed_file}")
    with open(reviewed_file, "w") as f:
        for example in reviewed_examples:
            f.write(json.dumps(example) + "\n")

    print(f"\nReviewed batch written: {reviewed_file.name}")
    print("Next step: python -m firewatch_agent.data_gen.upload --batch {batch_num}")


def main():
    parser = argparse.ArgumentParser(description="Review a raw batch of training examples")
    parser.add_argument("--batch", required=True, type=int, help="Batch number (0-29)")
    args = parser.parse_args()

    batch_num = args.batch
    if not (0 <= batch_num <= 29):
        print(f"ERROR: batch number must be 0-29, got: {args.batch}")
        sys.exit(1)

    run_review(batch_num)


if __name__ == "__main__":
    main()