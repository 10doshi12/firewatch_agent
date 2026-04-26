"""Static plot generation for Firewatch analysis reports."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "firewatch_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
_XDG_CACHE_HOME = Path(tempfile.gettempdir()) / "firewatch_cache"
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

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
        _grpo_action_plot(plots_dir / "grpo_action_distribution.png", grpo_records),
        _grpo_reward_by_action_plot(plots_dir / "grpo_reward_by_action.png", grpo_records),
        _grpo_batch_learnability_plot(plots_dir / "grpo_batch_learnability.png", grpo_records),
    ]

    if baseline_records:
        paths.append(_baseline_plot(plots_dir / "baseline_progression.png", baseline_records))
        paths.append(_baseline_delta_plot(plots_dir / "grpo_pre_post_delta.png", baseline_records))
    else:
        paths.append(_placeholder_plot(plots_dir / "baseline_progression.png", "No baseline records"))
        paths.append(_placeholder_plot(plots_dir / "grpo_pre_post_delta.png", "No GRPO pre/post baseline"))
    return paths


def _bar_plot(
    path: Path,
    title: str,
    values: dict,
    xlabel: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> Path:
    items = sorted(values.items(), key=lambda item: item[1], reverse=True) if values else [("none", 0)]
    labels = [_wrap_label(str(label)) for label, _ in items]
    counts = [count for _, count in items]
    crowded = len(labels) > 10 or any(len(str(label)) > 18 for label, _ in items)

    if crowded:
        height = max(6.0, min(18.0, 0.34 * len(labels) + 2.0))
        fig, ax = plt.subplots(figsize=(11, height))
        y_values = list(range(len(labels)))
        ax.barh(y_values, counts)
        ax.set_yticks(y_values)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, counts)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _wrap_label(label: str, width: int = 24) -> str:
    if len(label) <= width:
        return label
    parts = label.replace("-", "_").split("_")
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}_{part}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = part
    if current:
        lines.append(current)
    return "\n".join(lines)


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


def _grpo_action_plot(path: Path, records: list[dict]) -> Path:
    counts: dict[str, int] = {}
    for record in records:
        if record.get("event") != "reward_eval":
            continue
        action = record.get("action_type")
        if isinstance(action, str) and action:
            counts[action] = counts.get(action, 0) + 1
    return _bar_plot(path, "GRPO Action Distribution", counts, "Action", "Count")


def _grpo_reward_by_action_plot(path: Path, records: list[dict]) -> Path:
    buckets: dict[str, list[float]] = {}
    for record in records:
        if record.get("event") != "reward_eval":
            continue
        action = record.get("action_type")
        reward = record.get("reward")
        if isinstance(action, str) and isinstance(reward, (int, float)):
            buckets.setdefault(action, []).append(float(reward))

    values = {
        action: sum(rewards) / len(rewards)
        for action, rewards in buckets.items()
        if rewards
    }
    return _bar_plot(path, "GRPO Mean Reward by Action", values, "Action", "Mean Reward")


def _grpo_batch_learnability_plot(path: Path, records: list[dict]) -> Path:
    """
    Show whether GRPO has the *precondition* for learning: reward spread inside each
    TRL rollout batch, plus valid Firewatch action names (same normalization as training).
    """
    import statistics

    from analysis.grpo_group_metrics import (
        action_type_is_valid_firewatch,
        grpo_complete_group_cut_indices,
        group_reward_eval_records,
        summarize_grpo_group_batches,
    )

    groups = group_reward_eval_records(records)
    if len(groups) < 2:
        return _placeholder_plot(path, "Not enough GRPO batches for learnability plot")

    stds: list[float] = []
    valids: list[float] = []
    for batch in groups:
        rewards = [
            float(row["reward"])
            for row in batch
            if isinstance(row.get("reward"), (int, float))
        ]
        stds.append(statistics.pstdev(rewards) if len(rewards) > 1 else 0.0)
        denom = len(batch) or 1
        valids.append(
            sum(1 for row in batch if action_type_is_valid_firewatch(row.get("action_type"))) / denom
        )

    xs = list(range(1, len(groups) + 1))
    roll_std = [_window_mean(stds, index, window=3) for index in range(len(stds))]
    roll_valid = [_window_mean(valids, index, window=3) for index in range(len(valids))]

    stats = summarize_grpo_group_batches(records)
    footer = (
        f"Across {stats.get('group_count', 0)} batches: "
        f"mean σ={stats.get('mean_within_batch_std', 0.0):.3f}, "
        f"{100.0 * float(stats.get('pct_batches_with_positive_std', 0.0)):.0f}% batches with σ>0, "
        f"mean valid-action rate={100.0 * float(stats.get('mean_valid_action_rate', 0.0)):.0f}%"
    )
    if int(stats.get("sequence_group_count", 0) or 0) > 0:
        footer += (
            f" | sequence-mode batches: "
            f"mean σ={float(stats.get('sequence_mean_within_batch_std', 0.0)):.3f}, "
            f"valid={100.0 * float(stats.get('sequence_mean_valid_action_rate', 0.0)):.0f}%"
        )

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    ax0.bar(xs, stds, color="#4C72B0", alpha=0.35, label="within-batch reward σ")
    ax0.plot(xs, roll_std, color="#DD8452", linewidth=2.4, label="rolling mean σ (window=3)")
    for cut in grpo_complete_group_cut_indices(records):
        if 0 < cut < len(groups):
            ax0.axvline(cut + 0.5, color="#333333", linestyle="--", linewidth=1.0, alpha=0.55)
    ax0.set_ylabel("Reward σ inside batch")
    ax0.set_title("GRPO learnability: reward spread per rollout batch (GRPO needs σ > 0)")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, axis="y", alpha=0.3)

    ax1.plot(xs, valids, marker="o", markersize=4, color="#55A868", linewidth=1, label="valid Firewatch verb rate")
    ax1.plot(xs, roll_valid, color="#C44E52", linewidth=2.4, label="rolling mean rate (window=3)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Valid action rate")
    ax1.set_xlabel("Rollout batch index (chronological)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
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


def _baseline_delta_plot(path: Path, records: list[dict]) -> Path:
    by_variant = {
        str(record.get("model_variant")): record
        for record in records
        if record.get("model_variant")
    }
    pre = by_variant.get("grpo-pre")
    post = by_variant.get("grpo-post")
    if not pre or not post:
        return _placeholder_plot(path, "Missing GRPO pre/post baseline")

    metrics = {
        "success_rate": (
            float(pre.get("overall", {}).get("overall_success_rate", 0.0)),
            float(post.get("overall", {}).get("overall_success_rate", 0.0)),
        ),
        "mean_reward": (
            float(pre.get("overall", {}).get("overall_mean_reward", 0.0)),
            float(post.get("overall", {}).get("overall_mean_reward", 0.0)),
        ),
    }

    labels = list(metrics)
    x_values = list(range(len(labels)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([x - width / 2 for x in x_values], [metrics[label][0] for label in labels], width, label="pre")
    ax.bar([x + width / 2 for x in x_values], [metrics[label][1] for label in labels], width, label="post")
    ax.set_title("GRPO Pre/Post Baseline Delta")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.legend()
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
