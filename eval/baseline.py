"""
baseline.py — Baseline evaluation orchestrator (SPEC-T4 v2 §3-§9)

CLI entry point and Python-importable function for evaluating trained models
against the FirewatchEnv HF Space sim.

Invocation modes:
  1. Auto-invocation from SPEC-T2 (after each SFT batch):
     run_baseline("sft-batch-5", "post_sft_batch_5", auto_triggered=True, model_in_memory=model)
  2. Manual CLI:
     python -m firewatch_agent.eval.baseline --model-variant sft-batch-5 --trigger manual_check

Phases:
  A. Pull artifacts (manual mode only)
  B. Load models (manual mode only)
  C. Run evaluation episodes (always)
  D. Compute metrics (always)
  E. Append and push baselines log (always)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

# ---------------------------------------------------------------------------
# Resolve project paths
# ---------------------------------------------------------------------------

_AGENT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _AGENT_ROOT.parent

if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared import hf_auth, hf_io  # noqa: E402
from shared.model_runtime import (  # noqa: E402
    resolve_base_model_for_inference,
    try_import_unsloth,
)
from shared.platform import CHECKPOINTS_DIR, WORKING_DIR  # noqa: E402

from gnn.adjacency import NUM_SERVICES  # noqa: E402
from gnn.model import GraphSAGEModel  # noqa: E402
from gnn.train_gnn import NUM_FEATURES, WelfordNormalizer  # noqa: E402

from grpo.sim_client import SimClient  # noqa: E402

from .metrics import (  # noqa: E402
    EpisodeMetrics,
    OverallAggregate,
    TaskAggregate,
    aggregate_by_task,
    aggregate_overall,
)
from .runner import run_evaluation  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _load_config(config_path: Path | None = None) -> dict:
    """Load training configuration from config.yaml."""
    if config_path is None:
        config_path = _AGENT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# Model variant parser
# ---------------------------------------------------------------------------

_SFT_BATCH_RE = re.compile(r"^sft-batch-(\d+)$")
_GRPO_CHECKPOINT_RE = re.compile(r"^grpo-checkpoint-(\d+)$")


def _parse_variant(variant: str) -> dict:
    """
    Parse model_variant string into structured info.

    Returns dict with keys:
        type: "base" | "sft-batch" | "sft-latest" | "grpo-checkpoint" | "grpo-latest"
        batch_num: int | None (for sft-batch)
        checkpoint_num: int | None (for grpo-checkpoint)
        use_gnn: bool (False for base, True otherwise)
        lora_repo_suffix: str (repo suffix for LoRA adapter)
        lora_subfolder: str (subfolder in repo)
    """
    if variant == "base":
        return {
            "type": "base",
            "batch_num": None,
            "checkpoint_num": None,
            "use_gnn": False,
            "lora_repo_suffix": None,
            "lora_subfolder": None,
        }

    if variant == "untrained":
        # Pre-training baseline: base LLM (no LoRA) + freshly initialized GNN.
        # Used once at step 0 before any SFT to anchor the baseline-progression chart.
        return {
            "type": "untrained",
            "batch_num": None,
            "checkpoint_num": None,
            "use_gnn": True,
            "lora_repo_suffix": None,
            "lora_subfolder": None,
        }

    m = _SFT_BATCH_RE.match(variant)
    if m:
        batch_num = int(m.group(1))
        return {
            "type": "sft-batch",
            "batch_num": batch_num,
            "checkpoint_num": None,
            "use_gnn": True,
            "lora_repo_suffix": "firewatch-agent-sft",
            "lora_subfolder": f"batch_{batch_num:03d}",
        }

    if variant == "sft-latest":
        return {
            "type": "sft-latest",
            "batch_num": None,
            "checkpoint_num": None,
            "use_gnn": True,
            "lora_repo_suffix": "firewatch-agent-sft",
            "lora_subfolder": "latest",
        }

    m = _GRPO_CHECKPOINT_RE.match(variant)
    if m:
        checkpoint_num = int(m.group(1))
        return {
            "type": "grpo-checkpoint",
            "batch_num": None,
            "checkpoint_num": checkpoint_num,
            "use_gnn": True,
            "lora_repo_suffix": "firewatch-agent-grpo",
            "lora_subfolder": f"checkpoint-{checkpoint_num}",
        }

    if variant == "grpo-latest":
        return {
            "type": "grpo-latest",
            "batch_num": None,
            "checkpoint_num": None,
            "use_gnn": True,
            "lora_repo_suffix": "firewatch-agent-grpo",
            "lora_subfolder": "latest",
        }

    raise ValueError(
        f"Unknown model_variant '{variant}'. "
        f"Expected: base, sft-batch-<N>, sft-latest, grpo-checkpoint-<N>, grpo-latest"
    )


# ---------------------------------------------------------------------------
# Phase A: Pull artifacts (manual mode only)
# ---------------------------------------------------------------------------


def _pull_artifacts(
    namespace: str,
    variant_info: dict,
    local_dir: Path,
) -> tuple[Path | None, Path | None, str | None]:
    """
    Pull LoRA adapter and GNN checkpoint from HF Hub.

    Returns:
        (lora_path, gnn_checkpoint_path, model_repo_commit)
    """
    lora_path = None
    gnn_path = None
    model_repo_commit = None

    # Pull LoRA adapter
    if variant_info["lora_repo_suffix"]:
        repo_id = f"{namespace}/{variant_info['lora_repo_suffix']}"
        subfolder = variant_info["lora_subfolder"]
        logger.info("Pulling LoRA adapter: %s/%s", repo_id, subfolder)
        lora_path = hf_io.pull_lora_adapter(repo_id, subfolder, local_dir)
        if lora_path is None:
            logger.warning("LoRA adapter not found at %s/%s", repo_id, subfolder)
        else:
            # Try to get commit hash
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_auth.get_token())
                info = api.repo_info(repo_id, repo_type="model")
                model_repo_commit = info.sha
            except Exception:
                model_repo_commit = None

    # Pull GNN checkpoint — find the highest batch number
    if variant_info["use_gnn"]:
        gnn_batch = variant_info.get("batch_num")
        if gnn_batch is not None:
            # SFT-batch: use that batch's GNN
            gnn_path = _pull_gnn_for_batch(namespace, gnn_batch, local_dir)
        else:
            # sft-latest, grpo-*: find highest available GNN batch
            gnn_path = _pull_highest_gnn(namespace, local_dir)

    return lora_path, gnn_path, model_repo_commit


def _pull_gnn_for_batch(
    namespace: str,
    batch_num: int,
    local_dir: Path,
) -> Path | None:
    """Pull GNN checkpoint for a specific batch number."""
    repo_id = f"{namespace}/firewatch-gnn"
    filename = f"gnn/batch_{batch_num:03d}.pt"

    from huggingface_hub import snapshot_download

    try:
        repo_dir = hf_io.retry_with_backoff(
            lambda: snapshot_download(
                repo_id=repo_id,
                allow_patterns=[filename, "gnn/normalization.json"],
                local_dir=local_dir,
                token=hf_auth.get_token(),
            )
        )
    except Exception as exc:
        if "404" in str(exc) or "not found" in str(exc).lower():
            logger.warning("GNN checkpoint not found for batch %d", batch_num)
            return None
        raise

    local_path = local_dir / filename
    if not local_path.exists():
        candidates = list(Path(repo_dir).rglob(f"batch_{batch_num:03d}.pt"))
        if candidates:
            local_path = candidates[0]
        else:
            return None

    return local_path


def _pull_highest_gnn(namespace: str, local_dir: Path) -> Path | None:
    """Find and pull the highest-numbered GNN checkpoint."""
    repo_id = f"{namespace}/firewatch-gnn"

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_auth.get_token())
        files = api.list_repo_files(repo_id, repo_type="model")
    except Exception:
        logger.warning("Could not list GNN repo files")
        return None

    # Find highest batch number
    batch_nums = []
    for f in files:
        m = re.match(r"gnn/batch_(\d+)\.pt", f)
        if m:
            batch_nums.append(int(m.group(1)))

    if not batch_nums:
        logger.warning("No GNN checkpoints found in %s", repo_id)
        return None

    highest = max(batch_nums)
    return _pull_gnn_for_batch(namespace, highest, local_dir)


# ---------------------------------------------------------------------------
# Phase B: Load models (manual mode only)
# ---------------------------------------------------------------------------


def _load_models(
    lora_path: Path | None,
    gnn_path: Path | None,
    config: dict,
    use_gnn: bool,
) -> tuple[object, object, GraphSAGEModel | None, WelfordNormalizer | None]:
    """
    Load base LLM + LoRA adapter + GNN in eval mode.

    Returns:
        (model, tokenizer, gnn_model, normalizer)
    """
    sft_config = config.get("sft", {})
    gnn_config = config.get("gnn", {})
    max_seq_length = sft_config.get("max_seq_length", 2048)

    FastLanguageModel, unsloth_error = try_import_unsloth()
    if FastLanguageModel is None:
        raise RuntimeError(
            f"Unsloth is required for baseline eval but failed to import: {unsloth_error}. "
            "No dense/PyTorch fallback is supported."
        )
    base_model_name = resolve_base_model_for_inference(
        sft_config,
        use_low_bit_runtime=True,
        lora_path=lora_path,
    )

    # --- Load LLM ---
    logger.info("Loading base model: %s", base_model_name)
    logger.info("Using Unsloth for model loading")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if lora_path and lora_path.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(lora_path))
        logger.info("Loaded LoRA adapter from %s", lora_path)

    FastLanguageModel.for_inference(model)

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # --- Load GNN (CPU, eval mode) ---
    gnn_model = None
    normalizer = None

    if use_gnn and gnn_path is not None and gnn_path.exists():
        gnn_model = GraphSAGEModel(
            in_channels=NUM_FEATURES,
            hidden=gnn_config.get("hidden_dim", 64),
            num_classes=NUM_SERVICES,
            dropout=gnn_config.get("dropout", 0.1),
        )
        state_dict = torch.load(gnn_path, map_location="cpu", weights_only=True)
        gnn_model.load_state_dict(state_dict)
        gnn_model.eval()
        logger.info("Loaded GNN from %s (CPU, eval mode)", gnn_path)

        # Load normalization stats
        norm_path = gnn_path.parent / "normalization.json"
        if norm_path.exists():
            normalizer = WelfordNormalizer.from_dict(
                json.loads(norm_path.read_text())
            )
            logger.info("Loaded normalizer from %s (n=%d)", norm_path, normalizer.n)
        else:
            logger.warning("Normalization stats not found at %s", norm_path)
    elif use_gnn and gnn_path is None:
        # Untrained-baseline path: instantiate a fresh GNN with random weights so
        # the runner can still produce blurbs (a deliberate ablation anchor).
        gnn_model = GraphSAGEModel(
            in_channels=NUM_FEATURES,
            hidden=gnn_config.get("hidden_dim", 64),
            num_classes=NUM_SERVICES,
            dropout=gnn_config.get("dropout", 0.1),
        )
        gnn_model.eval()
        normalizer = WelfordNormalizer(num_features=NUM_FEATURES)
        logger.info("Initialized fresh GNN (no checkpoint pulled — pretrain baseline path)")

    return model, tokenizer, gnn_model, normalizer


# ---------------------------------------------------------------------------
# Phase E: Append and push baselines log
# ---------------------------------------------------------------------------


def _build_log_line(
    trigger: str,
    auto_triggered: bool,
    model_variant: str,
    model_repo_commit: str | None,
    gnn_checkpoint_filename: str | None,
    sim_space_url: str,
    per_task: dict[str, TaskAggregate],
    overall: OverallAggregate,
    wall_time_seconds: float,
) -> dict:
    """Construct the JSONL log line per SPEC-T4 §9.2 schema."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trigger": trigger,
        "auto_triggered": auto_triggered,
        "model_variant": model_variant,
        "model_repo_commit": model_repo_commit,
        "gnn_checkpoint_filename": gnn_checkpoint_filename,
        "sim_space_url": sim_space_url,
        "per_task": {
            task_name: agg.to_dict()
            for task_name, agg in per_task.items()
        },
        "overall": overall.to_dict(),
        "wall_time_seconds": round(wall_time_seconds, 2),
    }


def _push_baselines_log(
    namespace: str,
    log_line: dict,
    local_dir: Path,
) -> None:
    """Append log line to baselines/metrics.jsonl and push to HF."""
    repo_id = f"{namespace}/firewatch-sft-data"

    # Write new line to temp file
    new_line_path = local_dir / "baselines" / "_new_line.jsonl"
    new_line_path.parent.mkdir(parents=True, exist_ok=True)
    new_line_path.write_text(json.dumps(log_line, separators=(",", ":")) + "\n")

    hf_io.append_and_push_baselines_log(repo_id, new_line_path, local_dir)
    logger.info("Baselines log updated on HF")


# ---------------------------------------------------------------------------
# Phase E (optional): Plot generation
# ---------------------------------------------------------------------------


def _generate_plot(local_dir: Path) -> Path | None:
    """
    Generate progression chart from the full baselines log.

    X-axis: baseline run index
    Left Y-axis: overall success rate
    Right Y-axis: overall mean reward

    Returns path to saved PNG, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot generation")
        return None

    log_path = local_dir / "baselines" / "metrics.jsonl"
    if not log_path.exists():
        logger.warning("No baselines log found for plotting")
        return None

    # Read all log lines
    entries: list[dict] = []
    for line in log_path.read_text().strip().split("\n"):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(entries) < 2:
        logger.info("Not enough entries for a meaningful plot (%d entries)", len(entries))
        return None

    # Extract data
    indices = list(range(len(entries)))
    success_rates = [e.get("overall", {}).get("overall_success_rate", 0.0) for e in entries]
    mean_rewards = [e.get("overall", {}).get("overall_mean_reward", 0.0) for e in entries]
    triggers = [e.get("trigger", "")[:20] for e in entries]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Success rate (left axis)
    color1 = "#2ecc71"
    ax1.set_xlabel("Baseline Run Index")
    ax1.set_ylabel("Overall Success Rate", color=color1)
    ax1.plot(indices, success_rates, color=color1, marker="o", linewidth=2, label="Success Rate")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(-0.05, 1.05)

    # Mean reward (right axis)
    ax2 = ax1.twinx()
    color2 = "#3498db"
    ax2.set_ylabel("Overall Mean Reward", color=color2)
    ax2.plot(indices, mean_rewards, color=color2, marker="s", linewidth=2, label="Mean Reward")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Trigger labels
    ax1.set_xticks(indices)
    ax1.set_xticklabels(triggers, rotation=45, ha="right", fontsize=7)

    plt.title("FirewatchEnv Baseline Progression")
    fig.tight_layout()

    png_path = local_dir / "baselines" / "progression.png"
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)

    logger.info("Progression plot saved to %s", png_path)
    return png_path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_baseline(
    model_variant: str,
    trigger: str,
    auto_triggered: bool = False,
    model_in_memory: object | None = None,
    config_path: Path | None = None,
) -> None:
    """
    Run a baseline evaluation.

    This is the primary entry point — called by SPEC-T2 auto-invocation
    or directly from CLI / notebook.

    Args:
        model_variant: What to evaluate (e.g., "base", "sft-batch-5", "grpo-latest").
        trigger: Human-readable label for the log (e.g., "post_sft_batch_5").
        auto_triggered: True if invoked by SPEC-T2 (skips Phase A+B).
        model_in_memory: Pre-loaded model object from SPEC-T2 (avoids reload).
        config_path: Path to config.yaml (default: firewatch_agent/config.yaml).
    """
    start_time = time.monotonic()
    config = _load_config(config_path)
    sft_config = config.get("sft", {})
    gnn_config = config.get("gnn", {})

    # Resolution order: env var (set by start.sh to local in-Space server)
    # > config.sft.baseline_sim_url > legacy config.sim_env_url > public Space.
    sim_url = (
        os.environ.get("FIREWATCH_SIM_URL")
        or sft_config.get("baseline_sim_url")
        or config.get("sim_env_url")
        or "https://10doshi12-firewatch-env.hf.space"
    )
    num_episodes = sft_config.get("baseline_sim_episodes", 60)

    # Parse variant
    variant_info = _parse_variant(model_variant)
    use_gnn = variant_info["use_gnn"]

    # Authenticate
    username = hf_auth.get_username()
    namespace = hf_auth.resolve_namespace(config, username)
    os.environ["HF_NAMESPACE"] = namespace

    local_dir = WORKING_DIR / "eval_run"
    local_dir.mkdir(parents=True, exist_ok=True)

    model = None
    tokenizer = None
    gnn_model = None
    normalizer = None
    model_repo_commit = None
    gnn_checkpoint_filename = None

    if auto_triggered and model_in_memory is not None:
        # ===== Auto-invocation mode: skip Phase A + B =====
        logger.info(
            "Auto-invoked baseline: variant=%s trigger=%s (skipping Phase A+B)",
            model_variant, trigger,
        )

        # Extract model and tokenizer from the in-memory object
        # The model_in_memory is expected to be a tuple (model, tokenizer)
        # or a model with a .tokenizer attribute, depending on SPEC-T2's implementation.
        if isinstance(model_in_memory, tuple) and len(model_in_memory) >= 2:
            model, tokenizer = model_in_memory[0], model_in_memory[1]
        else:
            model = model_in_memory
            # Try to get tokenizer from model
            if hasattr(model, "tokenizer"):
                tokenizer = model.tokenizer
            else:
                # Load tokenizer separately
                base_model_name = sft_config.get(
                    "base_model", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
                )
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

        model.eval()

        # For auto-invoked SFT batches, try to load GNN from checkpoints
        if use_gnn:
            m = _SFT_BATCH_RE.match(model_variant)
            if m:
                batch_num = int(m.group(1))
                gnn_ckpt = CHECKPOINTS_DIR / "gnn" / f"batch_{batch_num:03d}.pt"
                if gnn_ckpt.exists():
                    gnn_model = GraphSAGEModel(
                        in_channels=NUM_FEATURES,
                        hidden=gnn_config.get("hidden_dim", 64),
                        num_classes=NUM_SERVICES,
                        dropout=gnn_config.get("dropout", 0.1),
                    )
                    state_dict = torch.load(gnn_ckpt, map_location="cpu", weights_only=True)
                    gnn_model.load_state_dict(state_dict)
                    gnn_model.eval()
                    gnn_checkpoint_filename = f"gnn/batch_{batch_num:03d}.pt"

                    norm_path = CHECKPOINTS_DIR / "gnn" / "normalization.json"
                    if norm_path.exists():
                        normalizer = WelfordNormalizer.from_dict(
                            json.loads(norm_path.read_text())
                        )
    else:
        # ===== Manual mode: Phase A + Phase B =====
        logger.info(
            "Manual baseline: variant=%s trigger=%s",
            model_variant, trigger,
        )

        # Phase A: Pull artifacts
        if variant_info["type"] != "base":
            logger.info("=== Phase A: Pull artifacts ===")
            lora_path, gnn_path, model_repo_commit = _pull_artifacts(
                namespace, variant_info, local_dir
            )
            if gnn_path:
                gnn_checkpoint_filename = str(gnn_path.relative_to(local_dir))
        else:
            lora_path = None
            gnn_path = None
            logger.info("=== Phase A: Skipped (base variant) ===")

        # Phase B: Load models
        logger.info("=== Phase B: Load models ===")
        model, tokenizer, gnn_model, normalizer = _load_models(
            lora_path=lora_path,
            gnn_path=gnn_path,
            config=config,
            use_gnn=use_gnn,
        )

    # ===== Phase C: Run evaluation episodes =====
    logger.info("=== Phase C: Run evaluation episodes ===")
    logger.info(
        "Tasks: %s, Episodes per task: %d, Total: %d",
        ["easy", "medium", "hard"], num_episodes, 3 * num_episodes,
    )

    env_client = SimClient(sim_url)
    try:
        env_client.connect()
        logger.info("Connected to sim at %s", sim_url)

        all_episodes = run_evaluation(
            env_client=env_client,
            model=model,
            tokenizer=tokenizer,
            gnn_model=gnn_model,
            normalizer=normalizer,
            num_episodes_per_task=num_episodes,
            use_gnn=use_gnn,
        )
    finally:
        env_client.disconnect()

    # ===== Phase D: Compute metrics =====
    logger.info("=== Phase D: Compute metrics ===")
    per_task = aggregate_by_task(all_episodes)
    overall = aggregate_overall(all_episodes)

    # Print summary
    for task_name, agg in per_task.items():
        logger.info(
            "  %s: mean_reward=%.3f success_rate=%.2f mean_length=%.1f wrong_actions=%.1f",
            task_name, agg.mean_reward, agg.success_rate,
            agg.mean_episode_length, agg.mean_wrong_actions,
        )
    logger.info(
        "  OVERALL: success_rate=%.2f mean_reward=%.3f total_wrong_actions=%d",
        overall.overall_success_rate, overall.overall_mean_reward,
        overall.total_wrong_actions,
    )

    # ===== Phase E: Append and push baselines log =====
    wall_time = time.monotonic() - start_time
    logger.info("=== Phase E: Append and push baselines log ===")

    log_line = _build_log_line(
        trigger=trigger,
        auto_triggered=auto_triggered,
        model_variant=model_variant,
        model_repo_commit=model_repo_commit,
        gnn_checkpoint_filename=gnn_checkpoint_filename,
        sim_space_url=sim_url,
        per_task=per_task,
        overall=overall,
        wall_time_seconds=wall_time,
    )

    # Print the log line for visibility
    print(json.dumps(log_line, indent=2))

    _push_baselines_log(namespace, log_line, local_dir)

    # Optional: generate progression plot
    _generate_plot(local_dir)

    total_wall = time.monotonic() - start_time
    logger.info(
        "Baseline complete: variant=%s trigger=%s wall_time=%.1fs",
        model_variant, trigger, total_wall,
    )

    # Cleanup GPU memory if we loaded models manually
    if not auto_triggered:
        del model
        del tokenizer
        del gnn_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for manual baseline evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="FirewatchEnv Baseline Evaluation (SPEC-T4 v2)",
    )
    parser.add_argument(
        "--model-variant",
        required=True,
        help=(
            "Model variant to evaluate. Options: "
            "base, sft-batch-<N>, sft-latest, grpo-checkpoint-<N>, grpo-latest"
        ),
    )
    parser.add_argument(
        "--trigger",
        required=True,
        help="Human-readable label for this evaluation run (e.g., 'pre_training_baseline')",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: firewatch_agent/config.yaml)",
    )
    args = parser.parse_args()

    run_baseline(
        model_variant=args.model_variant,
        trigger=args.trigger,
        auto_triggered=False,
        model_in_memory=None,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
