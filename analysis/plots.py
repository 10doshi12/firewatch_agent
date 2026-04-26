"""Static plot generation for Firewatch analysis reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def generate_plots(
    *,
    output_dir: Path | str,
    sft_summary: dict,
    inference_summary: dict,
    grpo_records: list[dict],
    baseline_records: list[dict] | None = None,
) -> list[Path]:
    """Generate all report PNGs and return their paths."""
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        _bar_plot(
            plots_dir / "sft_action_distribution.png",
            "SFT Gold Action Distribution",
            sft_summary.get("action_counts", {}),
            "Action",
            "Count",
        ),
        _grouped_count_plot(
            plots_dir / "sft_task_fault_coverage.png",
            "SFT Task and Fault Coverage",
            {
                f"tier:{key}": value
                for key, value in sft_summary.get("tier_counts", {}).items()
            }
            | {
                f"fault:{key}": value
                for key, value in sft_summary.get("fault_counts", {}).items()
            },
        ),
        _bar_plot(
            plots_dir / "inference_success_by_difficulty.png",
            "Inference Success by Difficulty",
            inference_summary.get("success_by_difficulty", {}),
            "Difficulty",
            "Success Rate",
            ylim=(0.0, 1.0),
        ),
        _bar_plot(
            plots_dir / "decision_source_mix.png",
            "Decision Source Mix",
            inference_summary.get("decision_source_counts", {}),
            "Source",
            "Count",
        ),
        _grpo_reward_plot(plots_dir / "grpo_reward_eval.png", grpo_records),
    ]

    if baseline_records:
        paths.append(_baseline_plot(plots_dir / "baseline_progression.png", baseline_records))
    else:
        paths.append(_placeholder_plot(plots_dir / "baseline_progression.png", "No baseline records"))
    return paths


def _bar_plot(
    path: Path,
    title: str,
    values: dict,
    xlabel: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> Path:
    labels = list(values.keys()) or ["none"]
    counts = [values[key] for key in values] or [0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, counts)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _grouped_count_plot(path: Path, title: str, values: dict) -> Path:
    return _bar_plot(path, title, values, "Category", "Count")


def _grpo_reward_plot(path: Path, records: list[dict]) -> Path:
    reward_records = [record for record in records if record.get("event") == "reward_eval"]
    rewards = [
        float(record.get("reward", 0.0))
        for record in reward_records
        if isinstance(record.get("reward"), (int, float))
    ]
    if not rewards:
        return _placeholder_plot(path, "No GRPO reward evaluations")

    indices = list(range(1, len(rewards) + 1))
    rolling = [_window_mean(rewards, index, window=10) for index in range(len(rewards))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(indices, rewards, marker="o", linewidth=1, label="reward")
    ax.plot(indices, rolling, linewidth=2, label="rolling mean")
    ax.set_title("GRPO Reward Evaluations")
    ax.set_xlabel("Reward Evaluation")
    ax.set_ylabel("Immediate Reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _baseline_plot(path: Path, records: list[dict]) -> Path:
    labels = [str(record.get("trigger", index))[:20] for index, record in enumerate(records)]
    success = [
        float(record.get("overall", {}).get("overall_success_rate", 0.0))
        for record in records
    ]
    rewards = [
        float(record.get("overall", {}).get("overall_mean_reward", 0.0))
        for record in records
    ]
    x_values = list(range(len(records)))

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(x_values, success, marker="o", label="success rate")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Success Rate")
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x_values, rewards, marker="s", label="mean reward")
    ax2.set_ylabel("Mean Reward")
    ax1.set_title("Baseline Progression")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _placeholder_plot(path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _window_mean(values: list[float], index: int, window: int) -> float:
    start = max(0, index - window + 1)
    subset = values[start : index + 1]
    return sum(subset) / len(subset)
