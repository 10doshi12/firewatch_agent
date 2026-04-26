"""CLI for generating Firewatch analysis graphs and report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.loaders import (
    load_grpo_metrics,
    load_inference_runs,
    load_jsonl,
    load_sft_examples,
)
from analysis.plots import generate_plots
from analysis.report import write_report
from analysis.grpo_group_metrics import summarize_grpo_group_batches
from analysis.summaries import (
    summarize_baselines,
    summarize_grpo_metrics,
    summarize_inference_runs,
    summarize_sft_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Firewatch analysis report")
    parser.add_argument("--sft-dir", type=Path, default=Path("../sft_data/reviewed"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--grpo-log", default="auto")
    parser.add_argument("--baseline-log", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_runs/latest"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_examples = load_sft_examples(args.sft_dir)
    inference_runs = load_inference_runs(args.runs_dir)
    grpo_log = _resolve_grpo_log(args.grpo_log)
    grpo_records = load_grpo_metrics(grpo_log) if grpo_log else []
    baseline_log = _resolve_baseline_log(args.baseline_log)
    baseline_records = load_jsonl(baseline_log) if baseline_log else []

    grpo_summary = summarize_grpo_metrics(grpo_records)
    grpo_summary.update(summarize_grpo_group_batches(grpo_records))
    summaries = {
        "sft": summarize_sft_examples(sft_examples),
        "inference": summarize_inference_runs(inference_runs),
        "grpo": grpo_summary,
        "baseline": summarize_baselines(baseline_records),
    }

    plot_paths = generate_plots(
        output_dir=output_dir,
        sft_summary=summaries["sft"],
        inference_summary=summaries["inference"],
        grpo_records=grpo_records,
        baseline_records=baseline_records,
    )
    report_path = write_report(output_dir, summaries, plot_paths)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "summaries": summaries,
                "inputs": {
                    "sft_dir": str(args.sft_dir),
                    "runs_dir": str(args.runs_dir),
                    "grpo_log": str(grpo_log) if grpo_log else None,
                    "baseline_log": str(baseline_log) if baseline_log else None,
                },
                "plots": [str(path) for path in plot_paths],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"[analysis] Wrote report: {report_path}")
    print(f"[analysis] Wrote summary: {summary_path}")


def _resolve_grpo_log(value: str) -> Path | None:
    if value == "none":
        return None
    if value != "auto":
        return Path(value)
    try:
        from shared.platform import CHECKPOINTS_DIR
    except Exception:
        return None
    return CHECKPOINTS_DIR / "grpo" / "metrics.jsonl"


def _resolve_baseline_log(value: str) -> Path | None:
    if value == "none":
        return None
    if value != "auto":
        return Path(value)

    candidates: list[Path] = []
    try:
        from shared.platform import WORKING_DIR
        candidates.append(WORKING_DIR / "eval_run" / "baselines" / "metrics.jsonl")
    except Exception:
        pass
    candidates.extend([
        Path("eval_run/baselines/metrics.jsonl"),
        Path("/tmp/firewatch_agent/eval_run/baselines/metrics.jsonl"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


if __name__ == "__main__":
    main()
