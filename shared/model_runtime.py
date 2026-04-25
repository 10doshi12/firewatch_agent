"""
Helpers for choosing model/runtime settings across SFT, eval, and GRPO.

The training notebooks prefer a low-bit Unsloth stack, but cloud runtimes
occasionally drift into a state where Unsloth imports fail or the underlying
4-bit CUDA stack is unusable. These helpers centralize the fallback rules so
the pipeline either degrades deliberately or raises a clear error.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

DEFAULT_LOW_BIT_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
DEFAULT_FALLBACK_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_LOW_BIT_OPTIMIZER = "adamw_8bit"
DEFAULT_FALLBACK_OPTIMIZER = "adamw_torch"


def try_import_unsloth() -> tuple[object | None, str | None]:
    """Return ``FastLanguageModel`` or the import failure message."""
    try:
        module = importlib.import_module("unsloth")
    except Exception as exc:  # pragma: no cover - depends on external runtime
        return None, f"{type(exc).__name__}: {exc}"

    return getattr(module, "FastLanguageModel", None), None


def is_low_bit_model_name(model_name: str) -> bool:
    """Heuristic for repo-managed low-bit model ids."""
    return model_name.endswith("-bnb-4bit")


def get_config_base_model(sft_config: dict) -> str:
    return str(sft_config.get("base_model", DEFAULT_LOW_BIT_MODEL))


def get_fallback_base_model(sft_config: dict) -> str:
    return str(sft_config.get("fallback_base_model", DEFAULT_FALLBACK_MODEL))


def load_adapter_base_model(adapter_path: Path | None) -> str | None:
    """
    Read the PEFT adapter's recorded base model, if present.

    PEFT writes this into ``adapter_config.json``; later stages should trust it
    over the config file because a real run may have fallen back to a smaller
    base model.
    """
    if adapter_path is None:
        return None

    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        return None

    try:
        data = json.loads(config_path.read_text())
    except Exception:
        return None

    value = data.get("base_model_name_or_path")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def resolve_base_model_for_training(
    sft_config: dict,
    *,
    use_low_bit_runtime: bool,
    prev_lora_path: Path | None,
) -> str:
    """
    Pick the base model for a new SFT step.

    If a previous adapter exists, its recorded base model wins. This protects
    later batches when batch 0 already fell back to a smaller model.
    """
    previous_base_model = load_adapter_base_model(prev_lora_path)
    if previous_base_model:
        if not use_low_bit_runtime and is_low_bit_model_name(previous_base_model):
            raise RuntimeError(
                "Previous LoRA adapter requires the low-bit base model "
                f"{previous_base_model}, but Unsloth is unavailable in this runtime. "
                "Restore the low-bit stack or restart from a fresh batch-000 run "
                "with the configured fallback base model."
            )
        return previous_base_model

    configured_base_model = get_config_base_model(sft_config)
    if use_low_bit_runtime or not is_low_bit_model_name(configured_base_model):
        return configured_base_model
    return get_fallback_base_model(sft_config)


def resolve_base_model_for_inference(
    sft_config: dict,
    *,
    use_low_bit_runtime: bool,
    lora_path: Path | None,
) -> str:
    """
    Pick the base model for eval/GRPO loading.

    When a LoRA adapter exists we must respect its recorded base model. If that
    adapter was trained against a low-bit base and Unsloth is broken, we fail
    clearly instead of trying to attach it to the wrong dense checkpoint.
    """
    adapter_base_model = load_adapter_base_model(lora_path)
    if adapter_base_model:
        if not use_low_bit_runtime and is_low_bit_model_name(adapter_base_model):
            raise RuntimeError(
                "This LoRA adapter was trained against the low-bit base model "
                f"{adapter_base_model}, but Unsloth is unavailable in this runtime."
            )
        return adapter_base_model

    configured_base_model = get_config_base_model(sft_config)
    if use_low_bit_runtime or not is_low_bit_model_name(configured_base_model):
        return configured_base_model
    return get_fallback_base_model(sft_config)


def resolve_optimizer_for_runtime(grpo_or_sft_config: dict, *, use_low_bit_runtime: bool) -> str:
    """
    Swap the bitsandbytes optimizer out when the low-bit stack is unavailable.
    """
    requested_optimizer = str(
        grpo_or_sft_config.get("optimizer", DEFAULT_LOW_BIT_OPTIMIZER)
    )
    if use_low_bit_runtime or requested_optimizer != DEFAULT_LOW_BIT_OPTIMIZER:
        return requested_optimizer
    return str(
        grpo_or_sft_config.get("fallback_optimizer", DEFAULT_FALLBACK_OPTIMIZER)
    )
