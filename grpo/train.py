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
from huggingface_hub import HfApi

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
    require_trainable_parameters,
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
from eval.baseline import run_baseline  # noqa: E402

logger = logging.getLogger(__name__)

# Configure JSON-line logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

GRPO_METRICS_REMOTE_PATH = "grpo/metrics.jsonl"

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------


class GrpoMetricsWriter:
    """Append GRPO metrics locally and periodically sync them to the dataset repo."""

    def __init__(
        self,
        metrics_path: Path,
        namespace: str,
        sync_every: int = 100,
        sync_enabled: bool = True,
    ) -> None:
        self.metrics_path = Path(metrics_path)
        self.namespace = namespace
        self.sync_every = max(1, int(sync_every))
        self.sync_enabled = sync_enabled
        self.records_since_sync = 0
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def repo_id(self) -> str:
        return f"{self.namespace}/firewatch-sft-data"

    @property
    def sync_dir(self) -> Path:
        return self.metrics_path.parent / "_dataset_sync"

    def append(self, record: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        self.records_since_sync += 1
        if self.sync_enabled and self.records_since_sync >= self.sync_every:
            self._sync("Periodic GRPO metrics sync")

    def sync_final(self) -> None:
        if self.sync_enabled:
            self._sync("Final GRPO metrics sync")

    def _sync(self, commit_message: str) -> None:
        try:
            hf_io.append_and_push_dataset_jsonl(
                repo_id=self.repo_id,
                remote_path=GRPO_METRICS_REMOTE_PATH,
                local_file=self.metrics_path,
                local_dir=self.sync_dir,
                commit_message=commit_message,
            )
            self.records_since_sync = 0
        except Exception as exc:
            logger.warning("GRPO metrics dataset sync failed: %s", exc)


def _grpo_metrics_path() -> Path:
    override = os.environ.get("GRPO_METRICS_PATH")
    if override:
        return Path(override)
    return CHECKPOINTS_DIR / "grpo" / "metrics.jsonl"


def _grpo_metrics_sync_every(config: dict) -> int:
    override = os.environ.get("GRPO_DATASET_SYNC_EVERY")
    if override:
        try:
            return int(override)
        except ValueError:
            logger.warning("Invalid GRPO_DATASET_SYNC_EVERY=%r; using config/default", override)
    return int(config.get("grpo", {}).get("dataset_sync_every", 100))


def _grpo_metrics_sync_enabled() -> bool:
    return os.environ.get("GRPO_DISABLE_DATASET_SYNC") != "1"


def _grpo_baseline_every_steps(config: dict) -> int:
    raw = os.environ.get("GRPO_BASELINE_EVERY_STEPS")
    if raw is None or not raw.strip():
        raw = str(config.get("grpo", {}).get("baseline_every_steps", 0))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"[grpo] GRPO_BASELINE_EVERY_STEPS must be an integer, got {raw!r}") from exc
    return max(0, value)


def run_post_grpo_baseline(
    *,
    namespace: str,
    step: int,
    env_client,
    model,
    tokenizer,
    gnn_model,
    normalizer,
    config_path: Path | None,
) -> None:
    """Evaluate the in-memory GRPO policy and append the baseline row to HF."""
    print(f"[grpo] Running post-GRPO baseline at step {step} for namespace {namespace}")
    print("[grpo] Temporarily disconnecting training sim client for baseline")
    env_client.disconnect()
    release_wait = float(os.environ.get("GRPO_BASELINE_SESSION_RELEASE_SECONDS", "1.0"))
    if release_wait > 0:
        time.sleep(release_wait)
    model.eval()
    try:
        run_baseline(
            model_variant=f"grpo-step-{step}",
            trigger=f"post_grpo_step_{step}",
            auto_triggered=True,
            model_in_memory=(model, tokenizer, gnn_model, normalizer),
            config_path=config_path,
        )
    finally:
        model.train()
        print("[grpo] Reconnecting training sim client after baseline")
        env_client.connect()


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


def trl_grpo_config_kwargs(kwargs: dict) -> dict:
    """Filter GRPOConfig kwargs for the installed TRL version."""
    import inspect

    from trl import GRPOConfig

    params = inspect.signature(GRPOConfig.__init__).parameters
    accepted = {k: v for k, v in kwargs.items() if k in params}
    ignored = sorted(set(kwargs) - set(accepted))
    if ignored:
        print(
            "[grpo] TRL GRPOConfig does not accept: "
            + ", ".join(ignored)
            + " (ignored for compatibility)"
        )
    return accepted


# ---------------------------------------------------------------------------
# Phase A: Pull state from HF Hub
# ---------------------------------------------------------------------------


def _truthy_env(name: str) -> bool:
    """Return True for common enabled env-var spellings."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _optional_positive_int_env(name: str) -> int | None:
    """Parse an optional positive integer environment variable."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"[grpo] {name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise RuntimeError(f"[grpo] {name} must be >= 0, got {value}")
    return value


def is_grpo_test_run() -> bool:
    """Whether this invocation should run the one-prompt smoke test path."""
    return _truthy_env("GRPO_TEST_RUN")


def apply_grpo_test_overrides(config: dict, test_run: bool) -> None:
    """Constrain GRPO to a tiny smoke test without changing production config."""
    if not test_run:
        return

    grpo_config = config.setdefault("grpo", {})
    grpo_config["num_generations"] = int(os.environ.get("GRPO_TEST_NUM_GENERATIONS", "2"))
    grpo_config["num_train_epochs"] = 1
    # GRPO needs at least two completions per prompt. In TRL 0.28,
    # generation_batch_size defaults to per_device_train_batch_size *
    # gradient_accumulation_steps, so the smoke-test batch must be divisible
    # by num_generations even though the prompt dataset has only one entry.
    grpo_config["per_device_train_batch_size"] = max(
        int(os.environ.get("GRPO_TEST_BATCH_SIZE", "2")),
        grpo_config["num_generations"],
    )
    grpo_config["gradient_accumulation_steps"] = 1
    grpo_config["max_prompt_length"] = int(os.environ.get("GRPO_TEST_MAX_PROMPT_LENGTH", "1024"))
    grpo_config["max_completion_length"] = int(os.environ.get("GRPO_TEST_MAX_COMPLETION_LENGTH", "128"))
    grpo_config["save_steps"] = int(os.environ.get("GRPO_TEST_SAVE_STEPS", "1"))
    grpo_config["max_steps"] = int(os.environ.get("GRPO_TEST_MAX_STEPS", "1"))


def pull_sft_lora(namespace: str, local_dir: Path) -> Path:
    """Pull the locked SFT LoRA adapter from latest/."""
    model_repo = f"{namespace}/firewatch-agent-sft"
    sft_batch = _optional_positive_int_env("GRPO_SFT_BATCH")
    subfolder = f"batch_{sft_batch:03d}" if sft_batch is not None else "latest"

    adapter_path = hf_io.pull_lora_adapter(model_repo, subfolder, local_dir)
    if adapter_path is None:
        raise RuntimeError(
            f"[grpo] SFT LoRA not found at {model_repo}/{subfolder}/. "
            "Set GRPO_SFT_BATCH to a known good batch or publish latest/. Aborting."
        )
    print(f"[grpo] Pulled SFT LoRA from {model_repo}/{subfolder}/")
    return adapter_path


def pull_gnn_latest(namespace: str, local_dir: Path) -> tuple[Path, Path]:
    """
    Pull the latest GNN checkpoint and normalization stats.

    Scans the GNN repo for the highest-numbered batch checkpoint.
    """
    gnn_repo = f"{namespace}/firewatch-gnn"
    explicit_batch = _optional_positive_int_env("GRPO_GNN_BATCH")

    if explicit_batch is not None:
        # pull_gnn_checkpoint expects batch_num + 1 (it downloads batch_{N-1})
        gnn_path = hf_io.pull_gnn_checkpoint(explicit_batch + 1, local_dir)
        if gnn_path is None:
            raise RuntimeError(f"[grpo] Failed to download GNN batch_{explicit_batch:03d}.pt")

        norm_path = local_dir / "gnn" / "normalization.json"
        if not norm_path.exists():
            candidates = list(local_dir.rglob("normalization.json"))
            if candidates:
                norm_path = candidates[0]
            else:
                raise RuntimeError("[grpo] normalization.json not found alongside GNN checkpoint")

        print(f"[grpo] Pulled pinned GNN checkpoint batch_{explicit_batch:03d} + normalization")
        return gnn_path, norm_path

    api = HfApi(token=hf_auth.get_token())

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
    if is_grpo_test_run():
        print("[grpo] GRPO_TEST_RUN=1 — skipping GRPO checkpoint resume")
    elif not force_fresh:
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
    if FastLanguageModel is None:
        raise RuntimeError(
            f"[grpo] Unsloth is required but failed to import: {unsloth_error}. "
            "No dense/PyTorch fallback is supported."
        )
    base_model = resolve_base_model_for_inference(
        sft_config,
        use_low_bit_runtime=True,
        lora_path=sft_lora_path,
    )

    print("[grpo] Using Unsloth for model loading")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map={"": 0},
    )

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(sft_lora_path), is_trainable=True)
    print(f"[grpo] Applied SFT LoRA from {sft_lora_path}")

    if grpo_checkpoint_path and grpo_checkpoint_path.exists():
        try:
            model = PeftModel.from_pretrained(
                model.base_model.model,
                str(grpo_checkpoint_path),
                is_trainable=True,
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    require_trainable_parameters(model, "GRPO policy adapter")

    return model, tokenizer, True


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
    if is_grpo_test_run():
        difficulty_names = {"easy": 0, "medium": 1, "hard": 2}
        difficulty = os.environ.get("GRPO_TEST_DIFFICULTY", "easy").strip().lower()
        if difficulty not in difficulty_names:
            raise RuntimeError(
                "[grpo] GRPO_TEST_DIFFICULTY must be one of easy, medium, hard"
            )
        seed = int(os.environ.get("GRPO_TEST_SEED", "1000"))
        dataset = [{
            "seed": seed,
            "difficulty_idx": difficulty_names[difficulty],
            "prompt_idx": 0,
        }]
        print(f"[grpo] Built GRPO test prompt dataset: {len(dataset)} entry")
        return dataset

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
    sim_url = (
        os.environ.get("FIREWATCH_SIM_URL")
        or config.get("sim_env_url")
        or "https://10doshi12-firewatch-env.hf.space"
    )
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
    config_path: Path | None,
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
    metrics_writer = GrpoMetricsWriter(
        metrics_path=_grpo_metrics_path(),
        namespace=namespace,
        sync_every=_grpo_metrics_sync_every(config),
        sync_enabled=_grpo_metrics_sync_enabled(),
    )
    baseline_every_steps = _grpo_baseline_every_steps(config)
    if baseline_every_steps > 0:
        print(f"[grpo] Post-GRPO baseline enabled every {baseline_every_steps} optimizer step(s)")

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
    max_steps = grpo_config.get("max_steps")
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
            metrics_writer.append({
                "event": "reward_eval",
                "evaluation_scope": "single_step",
                "completion_idx": i,
                "seed": seed,
                "action_type": action_dict.get("action_type"),
                "target_service": action_dict.get("target_service"),
                "reward": round(step_reward, 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        return rewards

    # --- Configure GRPOTrainer ---
    from trl import GRPOTrainer, GRPOConfig

    optimizer_name = resolve_optimizer_for_runtime(
        grpo_config,
        use_low_bit_runtime=use_low_bit_runtime,
    )

    grpo_kwargs = {
        "output_dir": str(output_dir),
        "num_generations": num_generations,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "max_grad_norm": max_grad_norm,
        "lr_scheduler_type": lr_scheduler,
        "warmup_ratio": warmup_ratio,
        "optim": optimizer_name,
        "logging_steps": 1,
        "save_steps": save_steps,
        "save_total_limit": 3,
        "report_to": "none",
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    }
    if max_steps is not None:
        grpo_kwargs["max_steps"] = max_steps

    training_args = GRPOConfig(**trl_grpo_config_kwargs(grpo_kwargs))

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

    class BaselineEvalCallback(TrainerCallback):
        """Run a lightweight baseline eval on the in-memory policy."""

        def __init__(self) -> None:
            self.completed_steps: set[int] = set()

        def on_step_end(self, args, state, control, **cb_kwargs):
            step = int(state.global_step or 0)
            if baseline_every_steps <= 0 or step <= 0:
                return
            if step % baseline_every_steps != 0 or step in self.completed_steps:
                return
            self.completed_steps.add(step)
            try:
                run_post_grpo_baseline(
                    namespace=namespace,
                    step=step,
                    env_client=env_client,
                    model=model,
                    tokenizer=tokenizer,
                    gnn_model=gnn_model,
                    normalizer=normalizer,
                    config_path=config_path,
                )
            except Exception as exc:
                logger.warning("Post-GRPO baseline failed at step %d: %s", step, exc)

    if baseline_every_steps > 0:
        trainer.add_callback(BaselineEvalCallback())

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

    api = HfApi(token=hf_auth.get_token())
    prefix = "test/" if is_grpo_test_run() else ""

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
            path_in_repo=f"{prefix}checkpoint-{step}/",
            commit_message=f"GRPO checkpoint step {step} @ {ts}",
        ))
        print(f"[grpo] Pushed {prefix}checkpoint-{step}/ to {repo_id}")
    except Exception as exc:
        logger.warning("Failed to push checkpoint-%d: %s", step, exc)

    # Also upload to latest/
    try:
        hf_io.retry_with_backoff(lambda: api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=f"{prefix}latest/",
            commit_message=f"GRPO latest (step {step}) @ {ts}",
        ))
        print(f"[grpo] Pushed {prefix}latest/ to {repo_id}")
    except Exception as exc:
        logger.warning("Failed to push latest/: %s", exc)


def push_final_checkpoint(namespace: str, final_dir: Path) -> None:
    """Push the final GRPO model to HF Hub."""
    _push_grpo_checkpoint(namespace, step=0, local_dir=final_dir)
    print("[grpo] Final checkpoint pushed to HF Hub")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_grpo(config_path: Path | None = None, test_run: bool = False) -> None:
    """
    Run the full GRPO training pipeline.

    Phases: Pull → Load → Configure → Train → Push
    """
    start_time = time.monotonic()

    # Load .env before anything else
    _load_dotenv()

    config = load_config(config_path)
    if test_run:
        os.environ["GRPO_TEST_RUN"] = "1"
    test_run_enabled = is_grpo_test_run()
    apply_grpo_test_overrides(config, test_run_enabled)

    # --- Setup ---
    print(f"[grpo] Platform: {PLATFORM}")
    print(f"[grpo] Working directory: {WORKING_DIR}")

    ok, free_gb = verify_disk_space(threshold_gb=20.0)
    if not ok:
        print("[grpo] WARNING: Low disk space. Proceeding anyway.")

    # Authenticate
    username = hf_auth.get_username()
    namespace = hf_auth.resolve_namespace(config, username)
    os.environ["HF_NAMESPACE"] = namespace
    print(f"[grpo] HF namespace: {namespace}")
    if test_run_enabled:
        print("[grpo] Test-run mode enabled")

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
        config_path=config_path,
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
    metrics_writer = GrpoMetricsWriter(
        metrics_path=_grpo_metrics_path(),
        namespace=namespace,
        sync_every=_grpo_metrics_sync_every(config),
        sync_enabled=_grpo_metrics_sync_enabled(),
    )
    metrics_writer.append(summary)
    metrics_writer.sync_final()


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
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run one tiny GRPO smoke test and write artifacts under test/.",
    )
    args = parser.parse_args()
    run_grpo(config_path=args.config, test_run=args.test_run)


if __name__ == "__main__":
    main()
