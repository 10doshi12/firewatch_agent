"""
shared/ — FirewatchAgent shared layer (SPEC-T1)

Platform detection, HF auth, HF Hub I/O wrappers.
No dependencies on Modules 2, 3, or 4.

Exports:
  Platform:  PLATFORM, WORKING_DIR, INPUT_DIR, CHECKPOINTS_DIR, HF_CACHE_DIR, verify_disk_space()
  Auth:      get_token(), get_username(), load_token(), verify_token()
  I/O Pull:  pull_reviewed_batch(), pull_lora_adapter(), pull_gnn_checkpoint(), pull_baselines_log()
  I/O Push:  push_reviewed_batch(), push_sft_lora(), push_gnn_checkpoint(), append_and_push_baselines_log()
  Utility:   retry_with_backoff()
"""

from .platform import (
    PLATFORM,
    WORKING_DIR,
    INPUT_DIR,
    CHECKPOINTS_DIR,
    HF_CACHE_DIR,
    verify_disk_space,
)
from .hf_auth import get_token, get_username, load_token, verify_token
from .hf_io import (
    pull_reviewed_batch,
    pull_lora_adapter,
    pull_gnn_checkpoint,
    pull_baselines_log,
    push_reviewed_batch,
    push_sft_lora,
    push_gnn_checkpoint,
    append_and_push_baselines_log,
    retry_with_backoff,
)

__all__ = [
    "PLATFORM",
    "WORKING_DIR",
    "INPUT_DIR",
    "CHECKPOINTS_DIR",
    "HF_CACHE_DIR",
    "verify_disk_space",
    "get_token",
    "get_username",
    "load_token",
    "verify_token",
    "pull_reviewed_batch",
    "pull_lora_adapter",
    "pull_gnn_checkpoint",
    "pull_baselines_log",
    "push_reviewed_batch",
    "push_sft_lora",
    "push_gnn_checkpoint",
    "append_and_push_baselines_log",
    "retry_with_backoff",
]