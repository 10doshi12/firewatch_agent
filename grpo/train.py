"""
train.py — GRPO Training Orchestrator (SPEC-T3 §7)

CLI entry point for the GRPO reinforcement learning loop.
Takes the locked SFT LoRA from Module 2 and fine-tunes it
against the live FirewatchEnv simulator using TRL's GRPOTrainer.

Supports both remote HF Space and local FastAPI server:
  - Remote: set sim_env_url in config.yaml to the HF Space URL
  - Local:  set sim_env_url to http://localhost:8000 and run
            `uvicorn firewatch_env.server.app:app --port 8000`

5-phase pipeline:
    Phase A: Pull state (SFT LoRA, GNN checkpoint, optional GRPO resume)
    Phase B: Load models (base LLM + LoRA on GPU, GNN on CPU)
    Phase C: Configure GRPOTrainer
    Phase D: Train (rollout groups × seeds)
    Phase E: Push checkpoints to HF Hub

Usage:
    python -m firewatch_agent.grpo.train
    python -m firewatch_agent.grpo.train --config config.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
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
    resolve_optimizer_for_runtime,
    try_import_unsloth,
)
from shared.platform import CHECKPOINTS_DIR, PLATFORM, WORKING_DIR, verify_disk_space  # noqa: E402

from gnn.adjacency import NUM_SERVICES  # noqa: E402
from gnn.model import GraphSAGEModel  # noqa: E402
from gnn.train_gnn import NUM_FEATURES, WelfordNormalizer  # noqa: E402

from grpo.rollout import rollout  # noqa: E402
from grpo.reward_extractor import extract_episode_reward  # noqa: E402

logger = logging.getLogger(__name__)

# Configure JSON-line logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    """Load .env file from firewatch_agent root if present."""
    env_path = _AGENT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(config_path: Path | None = None) -> dict:
    """Load training configuration from config.yaml."""
    if config_path is None:
        config_path = _AGENT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# Phase A: Pull state from HF Hub
# ---------------------------------------------------------------------------


def pull_sft_lora(namespace: str, local_dir: Path) -> Path:
    """Pull the locked SFT LoRA adapter from latest/."""
    model_repo = f"{namespace}/firewatch-agent-sft"
    adapter_path = hf_io.pull_lora_adapter(model_repo, "latest", local_dir)
    if adapter_path is None:
        raise RuntimeError(
            f"[grpo] SFT LoRA not found at {model_repo}/latest/. "
            "GRPO requires a completed SFT phase (Module 2). Aborting."
        )
    print(f"[grpo] Pulled SFT LoRA from {model_repo}/latest/")
    return adapter_path


def pull_gnn_latest(namespace: str, local_dir: Path) -> tuple[Path, Path]:
    """
    Pull the latest GNN checkpoint and normalization stats.

    Scans the GNN repo for the highest-numbered batch checkpoint.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hf_auth.get_token())
    gnn_repo = f"{namespace}/firewatch-gnn"

    try:
        files = api.list_repo_files(gnn_repo, repo_type="model")
    except Exception as exc:
        raise RuntimeError(
            f"[grpo] GNN repo {gnn_repo} not found: {exc}. "
            "GRPO requires a trained GNN (Module 2). Aborting."
        ) from exc

    # Find highest batch number
    batch_nums: list[int] = []
    for f in files:
        if f.startswith("gnn/batch_") and f.endswith(".pt"):
            name = f.replace("gnn/batch_", "").replace(".pt", "")
            try:
                batch_nums.append(int(name))
            except ValueError:
                continue

    if not batch_nums:
        raise RuntimeError(
            f"[grpo] No GNN checkpoints found in {gnn_repo}. "
            "GRPO requires a trained GNN. Aborting."
        )

    latest_batch = max(batch_nums)
    # pull_gnn_checkpoint expects batch_num + 1 (it downloads batch_{N-1})
    gnn_path = hf_io.pull_gnn_checkpoint(latest_batch + 1, local_dir)
    if gnn_path is None:
        raise RuntimeError(f"[grpo] Failed to download GNN batch_{latest_batch:03d}.pt")

    # Normalization
    norm_path = local_dir / "gnn" / "normalization.json"
    if not norm_path.exists():
        # Try to find it
        candidates = list(local_dir.rglob("normalization.json"))
        if candidates:
            norm_path = candidates[0]
        else:
            raise RuntimeError("[grpo] normalization.json not found alongside GNN checkpoint")

    print(f"[grpo] Pulled GNN checkpoint batch_{latest_batch:03d} + normalization")
    return gnn_path, norm_path


def pull_grpo_checkpoint(namespace: str, local_dir: Path) -> Path | None:
    """Pull previous GRPO checkpoint for resume (if exists)."""
    model_repo = f"{namespace}/firewatch-agent-grpo"
    grpo_path = hf_io.pull_lora_adapter(model_repo, "latest", local_dir)
    if grpo_path:
        print(f"[grpo] Pulled GRPO checkpoint from {model_repo}/latest/ for resume")
    else:
        print("[grpo] No existing GRPO checkpoint — starting fresh from SFT LoRA")
    return grpo_path


def pull_state(
    namespace: str,
    local_dir: Path,
) -> tuple[Path, Path, Path, Path | None]:
    """
    Pull all required artifacts.

    Returns:
        (sft_lora_path, gnn_checkpoint_path, gnn_norm_path, grpo_checkpoint_path)
    """
    sft_path = pull_sft_lora(namespace, local_dir)
    gnn_path, norm_path = pull_gnn_latest(namespace, local_dir)

    # Check FORCE_FRESH_GRPO before pulling GRPO checkpoint
    force_fresh = os.environ.get("FORCE_FRESH_GRPO", "0") == "1"
    grpo_path = None
    if not force_fresh:
        grpo_path = pull_grpo_checkpoint(namespace, local_dir)
    else:
        print("[grpo] FORCE_FRESH_GRPO=1 — skipping GRPO checkpoint resume")

    return sft_path, gnn_path, norm_path, grpo_path


# ---------------------------------------------------------------------------
# Phase B: Load models
# ---------------------------------------------------------------------------


def load_llm(
    sft_lora_path: Path,
    grpo_checkpoint_path: Path | None,
    config: dict,
) -> tuple:
    """
    Load base LLM + SFT LoRA, optionally overlay GRPO checkpoint.

    Returns:
        (model, tokenizer, use_low_bit_runtime)
    """
    sft_config = config.get("sft", {})
    max_seq_length = sft_config.get("max_seq_length", 2048)

    FastLanguageModel, unsloth_error = try_import_unsloth()
    use_unsloth = FastLanguageModel is not None
    base_model = resolve_base_model_for_inference(
        sft_config,
        use_low_bit_runtime=use_unsloth,
        lora_path=sft_lora_path,
    )

    # Try Unsloth first, fall back to dense transformers + PEFT
    if use_unsloth:
        print("[grpo] Using Unsloth for model loading")
    else:
        print(f"[grpo] Unsloth unavailable ({unsloth_error}), using transformers + PEFT")

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            device_map={"": 0},
        )

        # Apply SFT LoRA
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(sft_lora_path))
        print(f"[grpo] Applied SFT LoRA from {sft_lora_path}")

        # Overlay GRPO checkpoint if resuming
        if grpo_checkpoint_path and grpo_checkpoint_path.exists():
            try:
                model = PeftModel.from_pretrained(
                    model.base_model.model, str(grpo_checkpoint_path)
                )
                print(f"[grpo] Overlaid GRPO checkpoint from {grpo_checkpoint_path}")
            except Exception as exc:
                force_fresh = os.environ.get("FORCE_FRESH_GRPO", "0") == "1"
                if force_fresh:
                    print(f"[grpo] WARNING: GRPO checkpoint load failed ({exc}), starting fresh")
                else:
                    raise RuntimeError(
                        f"[grpo] GRPO checkpoint load failed: {exc}. "
                        "Set FORCE_FRESH_GRPO=1 to start fresh."
                    ) from exc
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        load_kwargs = {
            "device_map": {"": 0} if torch.cuda.is_available() else "cpu",
        }
        chosen_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=chosen_dtype,
                **load_kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=chosen_dtype,
                **load_kwargs,
            )
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Apply SFT LoRA
        model = PeftModel.from_pretrained(model, str(sft_lora_path))
        print(f"[grpo] Applied SFT LoRA from {sft_lora_path}")

        # Overlay GRPO checkpoint if resuming
        if grpo_checkpoint_path and grpo_checkpoint_path.exists():
            try:
                model = PeftModel.from_pretrained(
                    model.base_model.model, str(grpo_checkpoint_path)
                )
                print(f"[grpo] Overlaid GRPO checkpoint from {grpo_checkpoint_path}")
            except Exception as exc:
                force_fresh = os.environ.get("FORCE_FRESH_GRPO", "0") == "1"
                if force_fresh:
                    print(f"[grpo] WARNING: GRPO checkpoint load failed ({exc}), starting fresh")
                else:
                    raise RuntimeError(
                        f"[grpo] GRPO checkpoint load failed: {exc}. "
                        "Set FORCE_FRESH_GRPO=1 to start fresh."
                    ) from exc

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, use_unsloth


def load_gnn(
    gnn_checkpoint_path: Path,
    norm_path: Path,
    config: dict,
) -> tuple[GraphSAGEModel, WelfordNormalizer]:
    """Load GNN on CPU in eval mode (frozen throughout GRPO)."""
    gnn_config = config.get("gnn", {})
    hidden_dim = gnn_config.get("hidden_dim", 64)
    dropout = gnn_config.get("dropout", 0.1)

    gnn_model = GraphSAGEModel(
        in_channels=NUM_FEATURES,
        hidden=hidden_dim,
        num_classes=NUM_SERVICES,
        dropout=dropout,
    )

    state_dict = torch.load(gnn_checkpoint_path, map_location="cpu", weights_only=True)
    gnn_model.load_state_dict(state_dict)
    gnn_model.eval()

    # Freeze all parameters
    for param in gnn_model.parameters():
        param.requires_grad = False

    normalizer = WelfordNormalizer.from_dict(
        json.loads(norm_path.read_text())
    )

    print(f"[grpo] GNN loaded on CPU (frozen, {sum(p.numel() for p in gnn_model.parameters())} params)")
    return gnn_model, normalizer


# ---------------------------------------------------------------------------
# Phase C: Build prompt dataset
# ---------------------------------------------------------------------------


EVAL_SEEDS: frozenset[int] = frozenset({42, 137, 256})


def build_prompt_dataset(config: dict) -> list[dict]:
    """
    Build the (seed, difficulty) metadata dataset for GRPO training.

    Seeds are drawn starting from grpo.base_seed (default 1000) and filtered
    to exclude the evaluator's grader seeds {42, 137, 256}.
    """
    grpo_config = config.get("grpo", {})
    base_seed = grpo_config.get("base_seed", 1000)
    prompts_per_difficulty = grpo_config.get("prompts_per_difficulty", 50)
    num_difficulties = 3  # easy, medium, hard

    dataset: list[dict] = []
    seed = base_seed
    for difficulty_idx in range(num_difficulties):
        count = 0
        while count < prompts_per_difficulty:
            if seed not in EVAL_SEEDS:
                dataset.append({
                    "seed": seed,
                    "difficulty_idx": difficulty_idx,
                    "prompt_idx": count,
                })
                count += 1
            seed += 1

    print(f"[grpo] Built prompt dataset: {len(dataset)} entries (base_seed={base_seed})")
    return dataset


# ---------------------------------------------------------------------------
# Phase D: GRPO Training with custom reward
# ---------------------------------------------------------------------------


def create_env_client(config: dict):
    """
    Create a SimClient connected to the sim via WebSocket.

    Supports both local FastAPI server and remote HF Space.
    The URL is read from config.yaml sim_env_url.

    Local:  http://localhost:8000
    Remote: https://10doshi12-firewatch-env.hf.space
    """
    sim_url = config.get("sim_env_url", "https://10doshi12-firewatch-env.hf.space")
    print(f"[grpo] Connecting to sim at: {sim_url}")

    from grpo.sim_client import SimClient
    client = SimClient(base_url=sim_url)
    client.connect()
    return client


def run_grpo_training(
    model,
    tokenizer,
    use_low_bit_runtime: bool,
    gnn_model: GraphSAGEModel,
    normalizer: WelfordNormalizer,
    env_client,
    prompt_dataset: list[dict],
    config: dict,
    namespace: str,
    resume_from: Path | None = None,
) -> Path:
    """
    Run the GRPO training loop.

    Uses TRL's GRPOTrainer with a custom reward function that performs
    sequential rollouts against the sim.

    Returns:
        Path to the final checkpoint directory.
    """
    from grpo.rollout import (  # noqa: E402
        _format_rollout_prompt,
        _observation_to_dict,
        _parse_action,
        _run_gnn_for_observation,
    )

    grpo_config = config.get("grpo", {})
    output_dir = CHECKPOINTS_DIR / "grpo"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_generations = grpo_config.get("num_generations", 8)
    learning_rate = grpo_config.get("learning_rate", 1e-5)
    num_train_epochs = grpo_config.get("num_train_epochs", 3)
    batch_size = grpo_config.get("per_device_train_batch_size", 2)
    grad_accum = grpo_config.get("gradient_accumulation_steps", 4)
    max_prompt_length = grpo_config.get("max_prompt_length", 1024)
    max_completion_length = grpo_config.get("max_completion_length", 256)
    max_grad_norm = grpo_config.get("max_grad_norm", 0.1)
    lr_scheduler = grpo_config.get("lr_scheduler_type", "cosine")
    warmup_ratio = grpo_config.get("warmup_ratio", 0.1)
    save_steps = grpo_config.get("save_steps", 50)
    base_seed = grpo_config.get("base_seed", 1000)

    # --- Build reward function ---
    # GRPO needs a reward function that takes completions and returns scalars.
    # Our reward comes from the sim, so we perform rollouts inside the reward fn.
    #
    # The GRPOTrainer generates completions internally, then calls the reward fn.
    # We intercept at the reward level: for each completion, we run a rollout
    # using the completion's first action, accumulate rewards, and return the total.

    def _eval_single_step(seed: int, action_dict: dict) -> float:
        """Reset sim with seed, take one action, return immediate reward."""
        try:
            env_client.reset(seed=seed)
            result = env_client.step(action_dict)
            return float(result.reward) if result.reward is not None else 0.0
        except Exception as exc:
            logger.warning("Single-step eval failed seed=%d: %s", seed, exc)
            return -1.0

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        """
        Reward function for GRPO.

        Parses the action from each completion and evaluates it with a single
        sim step. This creates a causal link: completion → action → reward.
        kwargs["seed"] is populated from the dataset's seed column (replicated
        once per completion in the group by TRL).
        """
        raw_seeds = kwargs.get("seed", [])
        seeds = [int(s) for s in raw_seeds] if raw_seeds else [base_seed] * len(completions)
        if len(seeds) != len(completions):
            logger.warning("seeds/completions length mismatch: %d vs %d", len(seeds), len(completions))

        rewards: list[float] = []
        for i, completion in enumerate(completions):
            seed = seeds[i] if i < len(seeds) else base_seed
            action_dict = _parse_action(completion)
            step_reward = _eval_single_step(seed, action_dict)
            rewards.append(step_reward)

            print(json.dumps({
                "event": "reward_eval",
                "completion_idx": i,
                "seed": seed,
                "action_type": action_dict.get("action_type"),
                "reward": round(step_reward, 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }))

        return rewards

    # --- Configure GRPOTrainer ---
    from trl import GRPOTrainer, GRPOConfig

    optimizer_name = resolve_optimizer_for_runtime(
        grpo_config,
        use_low_bit_runtime=use_low_bit_runtime,
    )

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_generations=num_generations,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler,
        warmup_ratio=warmup_ratio,
        optim=optimizer_name,
        logging_steps=1,
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    # Build training prompts from the dataset
    # Each prompt is a formatted initial observation placeholder
    from datasets import Dataset

    # Build real observation prompts by resetting the sim for each (seed, difficulty) pair.
    # This ensures the model trains on actual observation text matching rollout format.
    difficulties = ["easy", "medium", "hard"]
    train_prompts: list[str] = []
    train_seeds: list[int] = []
    for entry in prompt_dataset:
        seed = entry["seed"]
        difficulty = difficulties[entry["difficulty_idx"]]
        try:
            result = env_client.reset(seed=seed, difficulty=difficulty)
            obs_dict = _observation_to_dict(result.observation.raw)
            gnn_blurb = _run_gnn_for_observation(gnn_model, obs_dict, normalizer)
            prompt = _format_rollout_prompt(obs_dict, gnn_blurb)
        except Exception as exc:
            logger.warning("Failed to build prompt for seed=%d: %s", seed, exc)
            prompt = f"Diagnose and resolve the incident for seed {seed}."
        train_prompts.append(prompt)
        train_seeds.append(seed)

    train_dataset = Dataset.from_dict({"prompt": train_prompts, "seed": train_seeds})

    # --- Create trainer ---
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    # --- Register checkpoint push callback ---
    from transformers import TrainerCallback

    class CheckpointPushCallback(TrainerCallback):
        """Push checkpoints to HF Hub after each save."""

        def on_save(self, args, state, control, **cb_kwargs):
            step = state.global_step
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
            if ckpt_dir.exists():
                _push_grpo_checkpoint(namespace, step, ckpt_dir)

    trainer.add_callback(CheckpointPushCallback())

    # --- Train ---
    print(f"[grpo] Starting GRPO training ({len(train_prompts)} prompts, "
          f"{num_generations} generations/group, {num_train_epochs} epochs)")

    if resume_from and resume_from.exists():
        print(f"[grpo] Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=str(resume_from))
    else:
        trainer.train()

    # --- Save final model ---
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[grpo] Final model saved to {final_dir}")

    return final_dir


# ---------------------------------------------------------------------------
# Phase E: Push checkpoints
# ---------------------------------------------------------------------------


def _push_grpo_checkpoint(namespace: str, step: int, local_dir: Path) -> None:
    """Upload a GRPO checkpoint to HF Hub."""
    repo_id = f"{namespace}/firewatch-agent-grpo"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    from huggingface_hub import HfApi

    api = HfApi(token=hf_auth.get_token())

    # Ensure repo exists
    try:
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    except Exception:
        pass

    # Upload to checkpoint-<step>/
    try:
        hf_io.retry_with_backoff(lambda: api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=f"checkpoint-{step}/",
            commit_message=f"GRPO checkpoint step {step} @ {ts}",
        ))
        print(f"[grpo] Pushed checkpoint-{step}/ to {repo_id}")
    except Exception as exc:
        logger.warning("Failed to push checkpoint-%d: %s", step, exc)

    # Also upload to latest/
    try:
        hf_io.retry_with_backoff(lambda: api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="latest/",
            commit_message=f"GRPO latest (step {step}) @ {ts}",
        ))
        print(f"[grpo] Pushed latest/ to {repo_id}")
    except Exception as exc:
        logger.warning("Failed to push latest/: %s", exc)


def push_final_checkpoint(namespace: str, final_dir: Path) -> None:
    """Push the final GRPO model to HF Hub."""
    _push_grpo_checkpoint(namespace, step=0, local_dir=final_dir)
    print("[grpo] Final checkpoint pushed to HF Hub")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_grpo(config_path: Path | None = None) -> None:
    """
    Run the full GRPO training pipeline.

    Phases: Pull → Load → Configure → Train → Push
    """
    start_time = time.monotonic()

    # Load .env before anything else
    _load_dotenv()

    config = load_config(config_path)

    # --- Setup ---
    print(f"[grpo] Platform: {PLATFORM}")
    print(f"[grpo] Working directory: {WORKING_DIR}")

    ok, free_gb = verify_disk_space(threshold_gb=20.0)
    if not ok:
        print("[grpo] WARNING: Low disk space. Proceeding anyway.")

    # Authenticate
    username = hf_auth.get_username()
    namespace = config.get("hf_namespace") or username
    print(f"[grpo] HF namespace: {namespace}")

    local_dir = WORKING_DIR / "grpo_run"
    local_dir.mkdir(parents=True, exist_ok=True)

    # ===== Phase A: Pull state =====
    print("\n[grpo] === Phase A: Pull state ===")
    sft_path, gnn_path, norm_path, grpo_path = pull_state(namespace, local_dir)

    # ===== Phase B: Load models =====
    print("\n[grpo] === Phase B: Load models ===")
    model, tokenizer, use_low_bit_runtime = load_llm(sft_path, grpo_path, config)
    gnn_model, normalizer = load_gnn(gnn_path, norm_path, config)

    # ===== Phase C: Connect to sim + build dataset =====
    print("\n[grpo] === Phase C: Connect to sim + build dataset ===")
    env_client = create_env_client(config)
    prompt_dataset = build_prompt_dataset(config)

    # Determine resume checkpoint
    resume_from = None
    if grpo_path and grpo_path.exists():
        trainer_state = grpo_path / "trainer_state.json"
        if trainer_state.exists():
            resume_from = grpo_path
            state = json.loads(trainer_state.read_text())
            print(f"[grpo] Will resume from optimizer step {state.get('global_step', '?')}")

    # ===== Phase D: Train =====
    print("\n[grpo] === Phase D: GRPO Training ===")
    final_dir = run_grpo_training(
        model=model,
        tokenizer=tokenizer,
        use_low_bit_runtime=use_low_bit_runtime,
        gnn_model=gnn_model,
        normalizer=normalizer,
        env_client=env_client,
        prompt_dataset=prompt_dataset,
        config=config,
        namespace=namespace,
        resume_from=resume_from,
    )

    # ===== Phase E: Push final =====
    print("\n[grpo] === Phase E: Push final checkpoint ===")
    push_final_checkpoint(namespace, final_dir)

    # Cleanup
    del env_client
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Final summary
    wall_time = time.monotonic() - start_time
    summary = {
        "event": "grpo_complete",
        "final_checkpoint": str(final_dir),
        "wall_time_seconds": round(wall_time, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(f"\n[grpo] === Complete ===")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FirewatchEnv GRPO Training Pipeline (SPEC-T3)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: firewatch_agent/config.yaml)",
    )
    args = parser.parse_args()
    run_grpo(config_path=args.config)


if __name__ == "__main__":
    main()
