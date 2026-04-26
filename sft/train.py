"""
train.py — SFT Training Orchestrator (SPEC-T2 §4-§11)

CLI entry point for the SFT training pipeline.
One invocation processes one or more steps (see ``MAX_SFT_STEPS``). Default campaign
``paired_15``: 15 training steps from 30 reviewed data files (two files per step).
Use ``legacy_30`` for one SFT step per reviewed file (30 steps).

7-phase pipeline per training step (incremental):
    Training id ``k`` is monotonic: after each run, GNN + LoRA are pushed to Hub
    under ``batch_k``; the next run ``k+1`` **always** pulls those weights (GNN
    ``gnn/batch_{k}.pt``, LoRA ``batch_k/adapter``) as the starting point, then
    saves updated weights as ``batch_{k+1}``. Run 0 starts from base LLM + fresh LoRA.
    Phase 1: Detect next step + pull previous artifacts from HF
    Phase 2: Load reviewed JSONL(s) for this step
    Phase 3: Train GNN (init from previous GNN checkpoint when k>0)
    Phase 4: GNN inference for blurbs
    Phase 5: VRAM handoff
    Phase 6: LLM SFT (LoRA init from previous step's adapter when k>0)
    Phase 7: Push new GNN + LoRA to HF + optional baseline

Usage:
    python -m firewatch_agent.sft.train
    python -m firewatch_agent.sft.train --config config.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

# Exit codes consumed by hf_space_sft_worker/start.sh — DO NOT renumber.
# 0 = step succeeded, 1 = error, 2 = campaign complete (no more work),
# 3 = CUDA OOM (worker reduces batch / seq length on next loop).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CAMPAIGN_COMPLETE = 2
EXIT_OOM = 3

# ---------------------------------------------------------------------------
# Resolve project paths
# ---------------------------------------------------------------------------

_AGENT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _AGENT_ROOT.parent

# Ensure imports work
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared import hf_auth, hf_io  # noqa: E402
from shared.model_runtime import (  # noqa: E402
    require_trainable_parameters,
    resolve_base_model_for_training,
    resolve_optimizer_for_runtime,
    try_import_unsloth,
)
from shared.platform import CHECKPOINTS_DIR, PLATFORM, WORKING_DIR, verify_disk_space  # noqa: E402

from gnn.adjacency import NUM_SERVICES  # noqa: E402
from gnn.model import GraphSAGEModel  # noqa: E402
from gnn.serializer import serialize_blurb  # noqa: E402
from gnn.train_gnn import (  # noqa: E402
    NUM_FEATURES,
    WelfordNormalizer,
    extract_node_features,
    run_gnn_inference,
    train_gnn,
)

from sft.campaign import (  # noqa: E402
    FINAL_TRAINING_RUN_PAIRED,
    TRAINING_RUNS_PAIRED,
    data_batches_for_run,
    detect_next_sft_step,
)
from sft.dataset import load_batch  # noqa: E402
from sft.prompt import format_chat_messages, format_sft_prompt  # noqa: E402


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(config_path: Path | None = None) -> dict:
    """Load training configuration from config.yaml.

    If ``SFT_APPLY_REGRESSION_OVERRIDE=1`` and ``sft_regression_override.yaml`` exists
    (written by ``eval.regression_guard``), merge suggested learning rate into ``sft``.
    """
    if config_path is None:
        config_path = _AGENT_ROOT / "config.yaml"
    cfg: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    apply = os.environ.get("SFT_APPLY_REGRESSION_OVERRIDE", "").lower() in (
        "1",
        "true",
        "yes",
    )
    override_path = _AGENT_ROOT / "sft_regression_override.yaml"
    if apply and override_path.exists():
        with open(override_path) as f:
            over = yaml.safe_load(f) or {}
        if over.get("regression_detected") and over.get("suggested_learning_rate_mult") is not None:
            sft = cfg.setdefault("sft", {})
            base_lr = float(sft.get("learning_rate", 2e-5))
            mult = float(over["suggested_learning_rate_mult"])
            sft["learning_rate"] = base_lr * mult
            print(
                f"[train] Applied sft_regression_override.yaml: learning_rate {base_lr} -> {sft['learning_rate']}"
            )
    _apply_sft_env_overrides(cfg)
    return cfg


def _apply_sft_env_overrides(cfg: dict) -> None:
    """
    Optional runtime overrides for GPU-specific tuning (e.g. A100 Spaces).

    These env vars are intentionally narrow and SFT-scoped so we can tune
    throughput without mutating committed config files.
    """
    sft = cfg.setdefault("sft", {})
    env_map: dict[str, tuple[str, type]] = {
        "SFT_BATCH_SIZE": ("per_device_train_batch_size", int),
        "SFT_GRAD_ACCUM": ("gradient_accumulation_steps", int),
        "SFT_MAX_SEQ_LENGTH": ("max_seq_length", int),
        "SFT_EPOCHS": ("llm_epochs_per_batch", int),
        "SFT_LEARNING_RATE": ("learning_rate", float),
        "SFT_WARMUP_RATIO": ("warmup_ratio", float),
        "SFT_MAX_PROMPT_LENGTH": ("max_prompt_length", int),
        "SFT_MAX_COMPLETION_LENGTH": ("max_completion_length", int),
    }
    applied: list[str] = []
    for env_key, (cfg_key, cast) in env_map.items():
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        try:
            value = cast(raw)
        except Exception:
            print(f"[train] WARNING: ignored invalid {env_key}={raw!r}")
            continue
        sft[cfg_key] = value
        applied.append(f"{cfg_key}={value}")
    if applied:
        print("[train] Applied env SFT overrides: " + ", ".join(applied))


def trl_sft_sequence_kwargs(max_seq_length: int) -> dict[str, int]:
    """SFTConfig sequence cap: TRL uses ``max_length``; older builds used ``max_seq_length``."""
    import inspect

    from trl import SFTConfig

    params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in params:
        return {"max_length": max_seq_length}
    if "max_seq_length" in params:
        return {"max_seq_length": max_seq_length}
    return {}


def load_peft_adapter_for_training(model: object, adapter_path: Path) -> object:
    """Load a PEFT adapter for continued training, not frozen inference."""
    from peft import PeftModel

    return PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=True,
    )


# ---------------------------------------------------------------------------
# Phase 1: Detect current step and pull state
# ---------------------------------------------------------------------------


def detect_current_batch(namespace: str, campaign: str | None = None) -> int | None:
    """
    Backwards-compatible name. Pass ``campaign`` from config ``sft.campaign``
    (``paired_15`` or ``legacy_30``), or set env ``SFT_CAMPAIGN``.
    """
    return detect_next_sft_step(namespace, campaign)


def pull_state(
    namespace: str,
    batch_num: int,
    local_dir: Path,
) -> tuple[Path, Path | None, Path | None]:
    """
    Legacy: single data batch. Returns (one jsonl path, prev gnn, prev lora).

    **Incremental weights:** For step ``batch_num == N``, pulls GNN checkpoint
    produced at the end of step ``N-1`` (``gnn/batch_{N-1}.pt``) and LoRA
    ``batch_{N-1}/`` so training continues from the last saved state.
    """
    batch_path = hf_io.pull_reviewed_batch(batch_num, local_dir)
    gnn_path = hf_io.pull_gnn_checkpoint(batch_num, local_dir)
    lora_path = None
    if batch_num > 0:
        model_repo = f"{namespace}/firewatch-agent-sft"
        lora_path = hf_io.pull_lora_adapter(
            model_repo, f"batch_{batch_num - 1:03d}", local_dir
        )
    return batch_path, gnn_path, lora_path


def pull_state_paired(
    namespace: str,
    run_idx: int,
    local_dir: Path,
) -> tuple[tuple[Path, Path], Path | None, Path | None]:
    """
    Paired mode: two reviewed JSONLs per training run. Hub artifact folder index
    matches ``run_idx`` (0..14), not the data file indices.

    **Incremental weights:** Same as legacy — for run ``k`` uses GNN + LoRA from
    run ``k-1`` (pushed after run ``k-1`` completed). Run 0 has no prior weights.
    """
    a, b = data_batches_for_run(run_idx)
    path_a = hf_io.pull_reviewed_batch(a, local_dir)
    path_b = hf_io.pull_reviewed_batch(b, local_dir)
    gnn_path = hf_io.pull_gnn_checkpoint(run_idx, local_dir)
    lora_path = None
    if run_idx > 0:
        model_repo = f"{namespace}/firewatch-agent-sft"
        lora_path = hf_io.pull_lora_adapter(
            model_repo, f"batch_{run_idx - 1:03d}", local_dir
        )
    return (path_a, path_b), gnn_path, lora_path


def pull_all_reviewed_batches(namespace: str, batch_num: int, local_dir: Path) -> list[dict]:
    """Legacy GNN accumulation: data batches 0..batch_num-1."""
    all_examples: list[dict] = []
    for n in range(batch_num):
        try:
            batch_path = hf_io.pull_reviewed_batch(n, local_dir)
            all_examples.extend(load_batch(batch_path))
        except Exception as exc:
            print(f"[train] WARNING: Could not pull batch {n:03d} for GNN accumulation: {exc}")
    return all_examples


def pull_accumulated_paired(
    namespace: str,
    run_idx: int,
    local_dir: Path,
) -> list[dict]:
    """GNN accumulation for paired mode: all examples from completed runs 0..run_idx-1."""
    out: list[dict] = []
    for r in range(run_idx):
        da, db = data_batches_for_run(r)
        for b in (da, db):
            try:
                p = hf_io.pull_reviewed_batch(b, local_dir)
                out.extend(load_batch(p))
            except Exception as exc:
                print(
                    f"[train] WARNING: Could not pull batch {b:03d} for GNN accumulation: {exc}"
                )
    return out


# ---------------------------------------------------------------------------
# Phase 5: VRAM handoff
# ---------------------------------------------------------------------------


def vram_handoff(gnn_model: GraphSAGEModel | None) -> None:
    """
    Clear all GNN resources from VRAM (SPEC-T2 §9).

    Defensive — GNN trains on CPU, but any incidental CUDA allocations
    during blurb generation are released.
    """
    if gnn_model is not None:
        del gnn_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(
            f"[train] VRAM handoff complete: "
            f"{free / 1024**3:.1f} GB free / {total / 1024**3:.1f} GB total"
        )
    else:
        print("[train] VRAM handoff complete (no CUDA device)")


# ---------------------------------------------------------------------------
# Phase 6: LLM SFT Training
# ---------------------------------------------------------------------------


def load_llm_and_train(
    examples: list[dict],
    blurbs: dict[str, str],
    batch_num: int,
    prev_lora_path: Path | None,
    config: dict,
) -> Path:
    """
    Load base LLM + LoRA and run SFT training.

    If ``prev_lora_path`` is set, LoRA weights are **continued** from the previous
    training id (incremental SFT). Otherwise a new LoRA is attached to the base model.

    Returns:
        Path to saved LoRA adapter directory for this training id.
    """
    sft_config = config.get("sft", {})
    max_seq_length = sft_config.get("max_seq_length", 2048)
    lora_rank = sft_config.get("lora_rank", 16)
    lora_alpha = sft_config.get("lora_alpha", 16)
    lora_dropout = sft_config.get("lora_dropout", 0.0)
    target_modules = sft_config.get("lora_target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    epochs = sft_config.get("llm_epochs_per_batch", 3)
    batch_size = sft_config.get("per_device_train_batch_size", 2)
    grad_accum = sft_config.get("gradient_accumulation_steps", 4)
    lr = sft_config.get("learning_rate", 2e-5)
    scheduler = sft_config.get("lr_scheduler_type", "cosine")
    warmup_ratio = sft_config.get("warmup_ratio", 0.1)
    max_prompt_length = sft_config.get("max_prompt_length", 1024)
    max_completion_length = sft_config.get("max_completion_length", 256)

    output_dir = CHECKPOINTS_DIR / "sft_llm" / f"batch_{batch_num:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Unsloth is required (4-bit + LoRA); no dense PyTorch fallback ---
    FastLanguageModel, unsloth_error = try_import_unsloth()
    if FastLanguageModel is None:
        raise RuntimeError(
            f"[train] Unsloth is required for SFT but failed to import: {unsloth_error}. "
            "Install unsloth in a CUDA environment. No fallback training path is supported."
        )
    print("[train] Using Unsloth for model loading")

    base_model = resolve_base_model_for_training(
        sft_config,
        use_low_bit_runtime=True,
        prev_lora_path=prev_lora_path,
    )
    optimizer_name = resolve_optimizer_for_runtime(
        sft_config,
        use_low_bit_runtime=True,
    )
    configured_base_model = sft_config.get("base_model", base_model)
    if base_model != configured_base_model:
        print(
            "[train] Falling back from "
            f"{configured_base_model} to {base_model}"
        )
    if optimizer_name != sft_config.get("optimizer", optimizer_name):
        print(
            "[train] Falling back from optimizer "
            f"{sft_config.get('optimizer')} to {optimizer_name}"
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if prev_lora_path and prev_lora_path.exists():
        model = load_peft_adapter_for_training(model, prev_lora_path)
        print(f"[train] Loaded previous LoRA from {prev_lora_path}")
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth",
        )
    require_trainable_parameters(model, "SFT LoRA")

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Prepare training data ---
    train_texts: list[str] = []
    for example in examples:
        example_id = example.get("example_id", "")
        gnn_blurb = blurbs.get(example_id)
        prompt = format_sft_prompt(example, gnn_blurb)
        messages = format_chat_messages(prompt)

        # Apply chat template
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback: manual formatting
            text = (
                f"<|im_start|>system\n{prompt['system']}<|im_end|>\n"
                f"<|im_start|>user\n{prompt['user']}<|im_end|>\n"
                f"<|im_start|>assistant\n{prompt['assistant']}<|im_end|>"
            )
        train_texts.append(text)

    # --- Configure SFTTrainer ---
    from trl import SFTConfig, SFTTrainer

    # Disable KV cache during training when using gradient checkpointing — they conflict
    # and HF emits a warning + silently disables checkpointing if cache is left enabled.
    if hasattr(model, "config"):
        model.config.use_cache = False

    sftconfig_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=scheduler,
        warmup_ratio=warmup_ratio,
        optim=optimizer_name,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        logging_steps=1,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataset_text_field="text",
    )
    sftconfig_kwargs.update(trl_sft_sequence_kwargs(max_seq_length))
    # Unsloth handles checkpointing internally (no extra HF gradient checkpointing).

    training_args = SFTConfig(**sftconfig_kwargs)

    from datasets import Dataset

    train_dataset = Dataset.from_dict({"text": train_texts})

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    # --- Train ---
    print(f"[train] Starting LLM SFT for batch {batch_num:03d} ({len(train_texts)} examples)")
    trainer.train()

    # --- Save LoRA adapter ---
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(str(adapter_dir), save_method="lora")

    tokenizer.save_pretrained(str(adapter_dir))

    print(f"[train] LoRA adapter saved to {adapter_dir}")
    return adapter_dir


# ---------------------------------------------------------------------------
# Phase 7: Push state + auto-trigger baseline
# ---------------------------------------------------------------------------


def push_state(
    namespace: str,
    batch_num: int,
    gnn_checkpoint_path: Path,
    lora_adapter_dir: Path,
    normalization_path: Path,
    *,
    paired_campaign: bool = True,
) -> None:
    """Push trained artifacts to HuggingFace Hub."""
    # Push GNN checkpoint
    hf_io.push_gnn_checkpoint(batch_num, gnn_checkpoint_path)

    # Push normalization stats alongside
    gnn_repo = f"{namespace}/firewatch-gnn"
    from huggingface_hub import HfApi
    api = HfApi(token=hf_auth.get_token())
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        api.upload_file(
            path_or_fileobj=str(normalization_path),
            path_in_repo="gnn/normalization.json",
            repo_id=gnn_repo,
            repo_type="model",
            commit_message=f"GNN normalization stats batch_{batch_num:03d} @ {ts}",
        )
    except Exception as exc:
        print(f"[train] WARNING: Failed to push normalization.json: {exc}")

    # Push LoRA adapter
    hf_io.push_sft_lora(batch_num, lora_adapter_dir)

    print(
        f"[train] Incremental save on Hub: gnn/batch_{batch_num:03d}.pt + "
        f"LoRA at batch_{batch_num:03d}/ — the next SFT run will load these as init."
    )

    is_final = (paired_campaign and batch_num == FINAL_TRAINING_RUN_PAIRED) or (
        not paired_campaign and batch_num == 29
    )
    if is_final:
        model_repo = f"{namespace}/firewatch-agent-sft"
        try:
            api.upload_folder(
                folder_path=str(lora_adapter_dir),
                repo_id=model_repo,
                repo_type="model",
                path_in_repo="latest/",
                commit_message=f"SFT LoRA latest (final batch) @ {ts}",
            )
            print("[train] Final batch — LoRA also pushed to latest/")
        except Exception as exc:
            print(f"[train] WARNING: Failed to push latest/ LoRA: {exc}")


def auto_trigger_baseline(
    batch_num: int,
    model: object | None = None,
) -> None:
    """
    Auto-trigger Module 4 baseline evaluation after SFT batch.

    If Module 4 is not yet implemented, logs a stub message.
    """
    try:
        from eval.baseline import run_baseline  # type: ignore[import]
        run_baseline(
            model_variant=f"sft-batch-{batch_num}",
            trigger=f"post_sft_batch_{batch_num}",
            auto_triggered=True,
            model_in_memory=model,
        )
    except ImportError:
        print(
            f"[train] Baseline evaluation (Module 4) not yet implemented. "
            f"Skipping auto-trigger for batch {batch_num}."
        )
    except Exception as exc:
        print(f"[train] Baseline evaluation failed (non-fatal): {exc}")


def _pretrain_baseline_already_recorded(namespace: str, local_dir: Path) -> bool:
    """
    True iff the dataset's baselines log already contains a pretrain entry.
    Hub is the source of truth — this lets the worker idempotently re-enter
    `run_sft_batch` without re-running the untrained baseline.
    """
    try:
        repo_id = f"{namespace}/firewatch-sft-data"
        log_path = hf_io.pull_baselines_log(repo_id, local_dir, repo_type="dataset")
    except TypeError:
        # Older hf_io without repo_type kwarg — assume not recorded.
        return False
    except Exception as exc:
        print(f"[train] Could not check pretrain marker: {exc}")
        return False
    if log_path is None or not log_path.exists():
        return False
    try:
        for raw in log_path.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("model_variant") == "untrained" or entry.get("trigger") == "pretrain":
                return True
    except Exception:
        return False
    return False


def maybe_run_pretrain_baseline(namespace: str, local_dir: Path) -> None:
    """
    Run the untrained-model baseline ONCE (before training-run 0). Idempotent:
    once the record lives in the dataset's baselines/metrics.jsonl, this is a
    no-op on every subsequent worker loop / Space restart.
    """
    if os.environ.get("SKIP_PRETRAIN_BASELINE", "").lower() in ("1", "true", "yes"):
        print("[train] SKIP_PRETRAIN_BASELINE — skipping untrained baseline anchor")
        return
    if _pretrain_baseline_already_recorded(namespace, local_dir):
        return
    print("[train] === Pretrain baseline (untrained GNN + base LLM) ===")
    try:
        from eval.baseline import run_baseline  # type: ignore[import]
        run_baseline(
            model_variant="untrained",
            trigger="pretrain",
            auto_triggered=False,
        )
    except ImportError:
        print("[train] eval.baseline missing; skipping pretrain anchor")
    except Exception as exc:
        # Non-fatal — SFT continues. The next worker loop will retry.
        print(f"[train] Pretrain baseline failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def _run_single_sft_step(
    namespace: str,
    step_idx: int,
    paired: bool,
    config: dict,
    local_dir: Path,
) -> None:
    """One full 7-phase pipeline for training index ``step_idx``."""
    if step_idx > 0:
        print(
            f"[train] Incremental run {step_idx}: will init GNN from "
            f"gnn/batch_{step_idx - 1:03d}.pt and LoRA from "
            f"firewatch-agent-sft/batch_{step_idx - 1:03d}/ (previous Hub save)"
        )
    else:
        print(
            "[train] First training run: fresh GNN + new LoRA on base model "
            "(no prior step on Hub)"
        )

    if paired:
        paths, prev_gnn_path, prev_lora_path = pull_state_paired(
            namespace, step_idx, local_dir
        )
        path_a, path_b = paths
        print(f"\n[train] === Phase 2: Load paired runs (data {path_a.name} + {path_b.name}) ===")
        examples = load_batch(path_a) + load_batch(path_b)
    else:
        batch_path, prev_gnn_path, prev_lora_path = pull_state(
            namespace, step_idx, local_dir
        )
        print(f"\n[train] === Phase 2: Load batch {step_idx:03d} ===")
        examples = load_batch(batch_path)

    print(f"[train] Loaded {len(examples)} examples for SFT")

    gnn_config = config.get("gnn", {})
    norm_path = None
    if prev_gnn_path and prev_gnn_path.parent.exists():
        candidate = prev_gnn_path.parent / "normalization.json"
        if candidate.exists():
            norm_path = candidate

    if paired:
        print(
            f"\n[train] === Phase 3: Train GNN (accumulated runs 0..{step_idx - 1} + current pair) ==="
        )
        prev_acc = pull_accumulated_paired(namespace, step_idx, local_dir)
        all_gnn_examples = prev_acc + examples
    else:
        print(
            f"\n[train] === Phase 3: Train GNN (all data batches 0..{step_idx - 1:03d} + current) ==="
        )
        prev_acc = pull_all_reviewed_batches(namespace, step_idx, local_dir)
        all_gnn_examples = prev_acc + examples

    if not all_gnn_examples:
        raise RuntimeError(
            f"[train] No examples available for GNN training at step {step_idx:03d}"
        )
    print(
        f"[train] GNN training on {len(all_gnn_examples)} accumulated examples "
        f"({len(examples)} current new, {len(prev_acc)} prior accumulated)"
    )

    gnn_ckpt_path, gnn_norm_path = train_gnn(
        batch_examples=all_gnn_examples,
        batch_num=step_idx,
        checkpoint_dir=CHECKPOINTS_DIR,
        prev_checkpoint_path=prev_gnn_path,
        normalization_path=norm_path,
        config=gnn_config,
    )

    print(f"\n[train] === Phase 4: GNN inference ===")
    gnn_model = GraphSAGEModel(
        in_channels=NUM_FEATURES,
        hidden=gnn_config.get("hidden_dim", 64),
        num_classes=NUM_SERVICES,
        dropout=gnn_config.get("dropout", 0.1),
    )
    gnn_state = torch.load(gnn_ckpt_path, map_location="cpu", weights_only=True)
    gnn_model.load_state_dict(gnn_state)

    normalizer = WelfordNormalizer.from_dict(
        json.loads(gnn_norm_path.read_text())
    )

    inference_results = run_gnn_inference(gnn_model, examples, normalizer)

    blurbs: dict[str, str] = {}
    for example_id, (logits, embeddings) in inference_results.items():
        blurbs[example_id] = serialize_blurb(logits)

    print(f"[train] Generated {len(blurbs)} blurbs")

    print(f"\n[train] === Phase 5: VRAM handoff ===")
    vram_handoff(gnn_model)
    gnn_model = None  # noqa: F841

    print(f"\n[train] === Phase 6: LLM SFT Training ===")
    adapter_dir = load_llm_and_train(
        examples=examples,
        blurbs=blurbs,
        batch_num=step_idx,
        prev_lora_path=prev_lora_path,
        config=config,
    )

    print(f"\n[train] === Phase 7: Push state + baseline ===")
    push_state(
        namespace,
        step_idx,
        gnn_ckpt_path,
        adapter_dir,
        gnn_norm_path,
        paired_campaign=paired,
    )

    sft_cfg = config.get("sft", {})
    skip_bl = os.environ.get("SKIP_AUTO_BASELINE", "").lower() in ("1", "true", "yes")
    if sft_cfg.get("skip_auto_baseline"):
        skip_bl = True
    if not skip_bl:
        auto_trigger_baseline(step_idx)
        try:
            from eval.regression_guard import check_regression_after_baseline  # noqa: WPS433

            check_regression_after_baseline(
                namespace=namespace,
                local_dir=local_dir,
                config=config,
            )
        except ImportError:
            pass
        except Exception as exc:
            print(f"[train] regression_guard (non-fatal): {exc}")
    else:
        print("[train] SKIP_AUTO_BASELINE — no post-SFT baseline this step")


def run_sft_batch(config_path: Path | None = None) -> int:
    """
    Run the 7-phase SFT pipeline for one or more steps (``MAX_SFT_STEPS``, default 1).
    """
    start_time = time.monotonic()
    config = load_config(config_path)
    sft_cfg = config.get("sft", {})
    campaign = sft_cfg.get("campaign", "paired_15")
    paired = campaign.strip().lower() in ("paired_15", "paired", "15")

    print(f"[train] Platform: {PLATFORM}")
    print(f"[train] Working directory: {WORKING_DIR}")
    print(f"[train] SFT campaign: {campaign} (paired={paired})")

    ok, free_gb = verify_disk_space(threshold_gb=20.0)
    if not ok:
        print("[train] WARNING: Low disk space. Proceeding anyway.")

    username = hf_auth.get_username()
    namespace = hf_auth.resolve_namespace(config, username)
    os.environ["HF_NAMESPACE"] = namespace
    print(f"[train] HF namespace: {namespace}")

    max_steps = int(os.environ.get("MAX_SFT_STEPS", "1"))
    if sft_cfg.get("max_sft_steps_per_invocation") is not None:
        max_steps = int(sft_cfg["max_sft_steps_per_invocation"])

    local_dir = WORKING_DIR / "sft_run"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Anchor the baseline-progression chart with one untrained-model run before
    # any training. Idempotent — does nothing once the record is on the Hub.
    maybe_run_pretrain_baseline(namespace, local_dir)

    steps_completed = 0
    for _ in range(max_steps):
        print("\n[train] === Phase 1: Detect next SFT step ===")
        step_idx = detect_current_batch(namespace, campaign)
        if step_idx is None:
            if steps_completed == 0:
                print("[train] Campaign complete — no untrained runs remain.")
                return EXIT_CAMPAIGN_COMPLETE
            break

        label = (
            f"training run {step_idx}/{TRAINING_RUNS_PAIRED - 1} (paired)"
            if paired
            else f"data batch {step_idx:03d} (legacy)"
        )
        print(f"[train] Next step: {label}")

        step_start = time.monotonic()
        _run_single_sft_step(
            namespace=namespace,
            step_idx=step_idx,
            paired=paired,
            config=config,
            local_dir=local_dir,
        )
        steps_completed += 1
        print(
            f"[train] Step wall time: {time.monotonic() - step_start:.1f}s"
        )

    wall_time = time.monotonic() - start_time
    print(f"\n[train] === Invocation complete ({steps_completed} step(s)) ===")
    print(f"[train] Total wall time: {wall_time:.1f}s")
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _is_oom_error(exc: BaseException) -> bool:
    """Return True for CUDA-OOM-shaped errors regardless of exact subclass."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FirewatchEnv SFT Training Pipeline (SPEC-T2)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: firewatch_agent/config.yaml)",
    )
    args = parser.parse_args()
    try:
        code = run_sft_batch(config_path=args.config)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("[train] FATAL: interrupted")
        sys.exit(EXIT_ERROR)
    except BaseException as exc:  # noqa: BLE001 — top-level boundary
        if _is_oom_error(exc):
            print(f"[train] FATAL: CUDA OOM: {exc}")
            sys.exit(EXIT_OOM)
        print(f"[train] FATAL: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(EXIT_ERROR)
    sys.exit(code if code is not None else EXIT_OK)


if __name__ == "__main__":
    main()
