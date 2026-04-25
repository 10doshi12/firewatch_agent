"""
train.py — SFT Training Orchestrator (SPEC-T2 §4-§11)

CLI entry point for the SFT training pipeline.
One invocation processes one batch. Thirty invocations total across the campaign.

7-phase pipeline per batch:
    Phase 1: Detect current batch + pull state from HF
    Phase 2: Load the batch JSONL
    Phase 3: Train GNN on CPU to convergence
    Phase 4: Run GNN inference for blurb generation
    Phase 5: VRAM handoff (cleanup GPU memory)
    Phase 6: Load base LLM + LoRA, run TRL SFTTrainer
    Phase 7: Push new state to HF + auto-trigger baseline

Usage:
    python -m firewatch_agent.sft.train
    python -m firewatch_agent.sft.train --config config.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
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

# Ensure imports work
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared import hf_auth, hf_io  # noqa: E402
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

from sft.dataset import load_batch  # noqa: E402
from sft.prompt import format_chat_messages, format_sft_prompt  # noqa: E402


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


# ---------------------------------------------------------------------------
# Phase 1: Detect current batch and pull state
# ---------------------------------------------------------------------------


def detect_current_batch(namespace: str) -> int | None:
    """
    Determine the next batch to train by querying HuggingFace Hub.

    Logic:
        1. List reviewed batches in dataset repo
        2. List trained LoRA folders in SFT model repo
        3. Current batch = lowest reviewed batch without a trained LoRA

    Returns:
        Batch number (0-29), or None if campaign complete.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hf_auth.get_token())
    dataset_repo = f"{namespace}/firewatch-sft-data"
    model_repo = f"{namespace}/firewatch-agent-sft"

    # Get reviewed batches
    try:
        files = api.list_repo_files(dataset_repo, repo_type="dataset")
        reviewed = sorted(
            f for f in files
            if f.startswith("reviewed/batch_") and f.endswith(".jsonl")
        )
    except Exception as exc:
        if "404" in str(exc):
            print("[train] No reviewed batches found on HF — dataset repo missing")
            sys.exit(1)
        raise

    if not reviewed:
        print("[train] No reviewed batches found on HF")
        sys.exit(1)

    # Extract batch numbers from filenames
    reviewed_nums: set[int] = set()
    for f in reviewed:
        # reviewed/batch_000.jsonl -> 0
        name = f.split("/")[-1].replace("batch_", "").replace(".jsonl", "")
        try:
            reviewed_nums.add(int(name))
        except ValueError:
            continue

    # Get trained LoRA folders
    trained_nums: set[int] = set()
    try:
        files = api.list_repo_files(model_repo, repo_type="model")
        for f in files:
            parts = f.split("/")
            if len(parts) >= 2 and parts[0].startswith("batch_"):
                name = parts[0].replace("batch_", "")
                try:
                    trained_nums.add(int(name))
                except ValueError:
                    continue
    except Exception:
        pass  # Model repo may not exist yet

    # Find lowest untrained reviewed batch
    untrained = sorted(reviewed_nums - trained_nums)
    if not untrained:
        print("[train] SFT campaign complete — all reviewed batches have trained LoRAs")
        return None

    return untrained[0]


def pull_state(
    namespace: str,
    batch_num: int,
    local_dir: Path,
) -> tuple[Path, Path | None, Path | None]:
    """
    Pull artifacts for the current batch.

    Returns:
        (batch_jsonl_path, prev_gnn_checkpoint_path, prev_lora_path)
    """
    # Pull reviewed batch
    batch_path = hf_io.pull_reviewed_batch(batch_num, local_dir)

    # Pull previous GNN checkpoint (batch N-1)
    gnn_path = hf_io.pull_gnn_checkpoint(batch_num, local_dir)

    # Pull previous SFT LoRA (batch N-1)
    lora_path = None
    if batch_num > 0:
        model_repo = f"{namespace}/firewatch-agent-sft"
        subfolder = f"batch_{batch_num - 1:03d}"
        lora_path = hf_io.pull_lora_adapter(model_repo, subfolder, local_dir)

    return batch_path, gnn_path, lora_path


def pull_all_reviewed_batches(namespace: str, batch_num: int, local_dir: Path) -> list[dict]:
    """Pull previous reviewed batches 0..batch_num-1 for GNN accumulation.

    The current batch (batch_num) is already in memory as `examples` and must be
    appended at the call site to avoid a redundant HF fetch.
    """
    all_examples: list[dict] = []
    for n in range(batch_num):  # 0..batch_num-1 only
        try:
            batch_path = hf_io.pull_reviewed_batch(n, local_dir)
            all_examples.extend(load_batch(batch_path))
        except Exception as exc:
            print(f"[train] WARNING: Could not pull batch {n:03d} for GNN accumulation: {exc}")
    return all_examples


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

    Returns:
        Path to saved LoRA adapter directory.
    """
    sft_config = config.get("sft", {})
    base_model = sft_config.get("base_model", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
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
    optimizer_name = sft_config.get("optimizer", "adamw_8bit")

    output_dir = CHECKPOINTS_DIR / "sft_llm" / f"batch_{batch_num:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Try Unsloth first, fall back to transformers + peft ---
    use_unsloth = False
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
        print("[train] Using Unsloth for model loading")
    except ImportError:
        print("[train] Unsloth not available, using transformers + peft")

    if use_unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        # Apply LoRA
        if prev_lora_path and prev_lora_path.exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(prev_lora_path))
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
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, PeftModel, get_peft_model

        # Load with 4-bit quantization if bitsandbytes available
        bnb_config = None
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            print("[train] bitsandbytes 4-bit not available, loading in fp16")

        # transformers >=4.50 deprecated `torch_dtype` in favour of `dtype`; fall back for older builds.
        load_kwargs = {
            "quantization_config": bnb_config,
            "device_map": {"": 0} if torch.cuda.is_available() else "cpu",
        }
        chosen_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(base_model, dtype=chosen_dtype, **load_kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=chosen_dtype, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # NOTE: We deliberately skip peft.prepare_model_for_kbit_training here. That
        # helper iterates every parameter and casts fp16/bf16 → fp32, which doubles
        # the memory of (un-quantized) embedding/lm_head/layernorm weights. On a T4
        # with a 14B 4-bit base, the embedding cast alone OOMs (~2.9 GB delta with
        # only ~2.3 GB free). The two things that helper does that we actually need:
        #   1) freeze base params  → already done by get_peft_model
        #   2) make input embeddings produce gradients so LoRA on q/k/v gets signal
        # We do (2) directly via enable_input_require_grads(). Embeddings stay fp16;
        # bnb_4bit_compute_dtype=fp16 means the matmul still happens in fp16, and the
        # LoRA adapter weights themselves are created in fp32 by peft.
        if bnb_config is not None and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        if prev_lora_path and prev_lora_path.exists():
            model = PeftModel.from_pretrained(model, str(prev_lora_path), is_trainable=True)
            print(f"[train] Loaded previous LoRA from {prev_lora_path}")
        else:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

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
    # Gradient checkpointing is required for 14B 4-bit on a 15GB T4 — without it the
    # backward pass OOMs at batch_size=1, max_seq_length=1536. Unsloth path enables this
    # internally; here we set it via SFTConfig for the transformers+peft fallback.
    if not use_unsloth:
        sftconfig_kwargs["gradient_checkpointing"] = True
        sftconfig_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

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

    if use_unsloth:
        model.save_pretrained(str(adapter_dir), save_method="lora")
    else:
        model.save_pretrained(str(adapter_dir))

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

    # If final batch (N=29), also push to latest/
    if batch_num == 29:
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


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_sft_batch(config_path: Path | None = None) -> None:
    """
    Run the full 7-phase SFT pipeline for one batch.
    Automatically detects the next batch to train.
    """
    start_time = time.monotonic()
    config = load_config(config_path)

    # --- Setup ---
    print(f"[train] Platform: {PLATFORM}")
    print(f"[train] Working directory: {WORKING_DIR}")

    ok, free_gb = verify_disk_space(threshold_gb=20.0)
    if not ok:
        print("[train] WARNING: Low disk space. Proceeding anyway.")

    # Authenticate
    username = hf_auth.get_username()
    namespace = config.get("hf_namespace") or username
    print(f"[train] HF namespace: {namespace}")

    # ===== Phase 1: Detect current batch + pull state =====
    print("\n[train] === Phase 1: Detect batch + pull state ===")
    batch_num = detect_current_batch(namespace)
    if batch_num is None:
        return

    print(f"[train] Training batch {batch_num:03d}")
    local_dir = WORKING_DIR / "sft_run"
    local_dir.mkdir(parents=True, exist_ok=True)

    batch_path, prev_gnn_path, prev_lora_path = pull_state(namespace, batch_num, local_dir)

    # ===== Phase 2: Load the batch =====
    print(f"\n[train] === Phase 2: Load batch {batch_num:03d} ===")
    examples = load_batch(batch_path)
    print(f"[train] Loaded {len(examples)} examples")

    # ===== Phase 3: Train GNN =====
    print(f"\n[train] === Phase 3: Train GNN (all batches 0..{batch_num:03d}) ===")
    gnn_config = config.get("gnn", {})

    # Normalization from previous batch
    norm_path = None
    if prev_gnn_path and prev_gnn_path.parent.exists():
        candidate = prev_gnn_path.parent / "normalization.json"
        if candidate.exists():
            norm_path = candidate

    all_gnn_examples = pull_all_reviewed_batches(namespace, batch_num, local_dir) + examples
    if not all_gnn_examples:
        raise RuntimeError(f"[train] No examples available for GNN training at batch {batch_num:03d}")
    print(f"[train] GNN training on {len(all_gnn_examples)} accumulated examples (batches 0..{batch_num:03d})")

    gnn_ckpt_path, gnn_norm_path = train_gnn(
        batch_examples=all_gnn_examples,
        batch_num=batch_num,
        checkpoint_dir=CHECKPOINTS_DIR,
        prev_checkpoint_path=prev_gnn_path,
        normalization_path=norm_path,
        config=gnn_config,
    )

    # ===== Phase 4: GNN inference for blurb generation =====
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

    # Generate blurbs
    blurbs: dict[str, str] = {}
    for example_id, (logits, embeddings) in inference_results.items():
        blurbs[example_id] = serialize_blurb(logits)

    print(f"[train] Generated {len(blurbs)} blurbs")

    # ===== Phase 5: VRAM handoff =====
    print(f"\n[train] === Phase 5: VRAM handoff ===")
    vram_handoff(gnn_model)
    gnn_model = None  # noqa: F841

    # ===== Phase 6: LLM SFT Training =====
    print(f"\n[train] === Phase 6: LLM SFT Training ===")
    adapter_dir = load_llm_and_train(
        examples=examples,
        blurbs=blurbs,
        batch_num=batch_num,
        prev_lora_path=prev_lora_path,
        config=config,
    )

    # ===== Phase 7: Push state + auto-trigger baseline =====
    print(f"\n[train] === Phase 7: Push state + baseline ===")
    push_state(namespace, batch_num, gnn_ckpt_path, adapter_dir, gnn_norm_path)

    # Auto-trigger baseline
    auto_trigger_baseline(batch_num)

    # Final summary
    wall_time = time.monotonic() - start_time
    summary = {
        "batch_num": batch_num,
        "total_examples": len(examples),
        "gnn_checkpoint": str(gnn_ckpt_path),
        "lora_adapter": str(adapter_dir),
        "wall_time_seconds": round(wall_time, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(f"\n[train] === Complete ===")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


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
    run_sft_batch(config_path=args.config)


if __name__ == "__main__":
    main()
