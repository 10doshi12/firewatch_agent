"""Markdown report generation for Firewatch analysis outputs."""

from __future__ import annotations

from pathlib import Path


def write_report(output_dir: Path | str, summaries: dict, plot_paths: list[Path]) -> Path:
    """Write a concise Markdown report and return its path."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.md"

    sft = summaries.get("sft", {})
    inference = summaries.get("inference", {})
    grpo = summaries.get("grpo", {})
    baseline = summaries.get("baseline", {})
    reward_delta = baseline.get("grpo_reward_delta")
    success_delta = baseline.get("grpo_success_delta")

    lines = [
        "# Firewatch Analysis Report",
        "",
        "## Summary",
        f"- SFT examples analyzed: {sft.get('example_count', 0)}",
        f"- Inference episodes analyzed: {inference.get('episode_count', 0)}",
        f"- Inference steps analyzed: {inference.get('step_count', 0)}",
        f"- GRPO reward evaluations analyzed: {grpo.get('reward_eval_count', 0)}",
        f"- GRPO mean immediate reward: {grpo.get('mean_reward', 0.0):.3f}",
        f"- GRPO positive reward rate: {grpo.get('positive_reward_rate', 0.0):.2f}",
        f"- Baseline records analyzed: {baseline.get('record_count', 0)}",
    ]
    if reward_delta is not None:
        lines.append(f"- GRPO pre/post mean reward delta: {reward_delta:+.3f}")
    if success_delta is not None:
        lines.append(f"- GRPO pre/post success-rate delta: {success_delta:+.3f}")
    lines.extend([
        "",
        "## GRPO Action Mix",
    ])
    action_counts = grpo.get("action_counts", {})
    if action_counts:
        for action, count in sorted(action_counts.items(), key=lambda item: (-item[1], item[0]))[:10]:
            lines.append(f"- `{action}`: {count}")
    else:
        lines.append("- No GRPO action records found.")
    lines.extend([
        "",
        "## Caveats",
        "- GRPO rewards are single-step evaluations today, not full episode returns.",
        "- Baseline and inference success metrics should not be compared directly to GRPO reward-eval rewards.",
        "",
        "## Plots",
    ])

    for path in plot_paths:
        rel = path.relative_to(root) if path.is_relative_to(root) else path
        lines.append(f"- [{rel}]({rel})")

    report_path.write_text("\n".join(lines) + "\n")
    return report_path
