"""
platform.py — Platform detection and canonical paths

Detects: Kaggle (KAGGLE_KERNEL_RUN_TYPE or /kaggle/working),
         Colab (COLAB_GPU or /content + google.colab),
         Local (fallback)

Exposes:
  PLATFORM: str = "kaggle" | "colab" | "local"
  WORKING_DIR: Path
  INPUT_DIR: Path
  CHECKPOINTS_DIR: Path
  HF_CACHE_DIR: Path

  verify_disk_space(threshold_gb: float = 20.0) -> tuple[bool, float]
    # Returns (enough_space, free_gb)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or Path("/kaggle/working").exists()
_COLAB = os.environ.get("COLAB_GPU") is not None or Path("/content").exists()

# Guard against ambiguous detection
if _KAGGLE and _COLAB:
    raise RuntimeError(
        "Ambiguous platform: both Kaggle and Colab indicators detected. "
        "Cannot determine platform. Set environment variables explicitly."
    )

if _KAGGLE:
    PLATFORM = "kaggle"
elif _COLAB:
    PLATFORM = "colab"
else:
    PLATFORM = "local"

# ---------------------------------------------------------------------------
# Canonical paths per platform
# ---------------------------------------------------------------------------

if PLATFORM == "kaggle":
    WORKING_DIR = Path("/kaggle/working")
    INPUT_DIR = Path("/kaggle/input")
    HF_CACHE_DIR = WORKING_DIR / ".hf_cache"
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)

elif PLATFORM == "colab":
    WORKING_DIR = Path("/content")
    INPUT_DIR = WORKING_DIR  # Colab has no separate input mount
    HF_CACHE_DIR = WORKING_DIR / ".hf_cache"
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)

else:  # local
    _base = Path(tempfile.gettempdir()) / "firewatch_agent"
    WORKING_DIR = _base
    INPUT_DIR = _base
    # HF_HOME left to default for local — use default cache location
    HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"

# checkpoints/ is always a subdirectory of working
CHECKPOINTS_DIR = WORKING_DIR / "checkpoints"

# Ensure checkpoints dir exists
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Environment hygiene — suppress HF telemetry everywhere
# ---------------------------------------------------------------------------

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Enable hf_transfer (Rust-based fast downloader) if available
try:
    import hf_transfer  # noqa: F401
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Disk space verification
# ---------------------------------------------------------------------------

def verify_disk_space(threshold_gb: float = 20.0) -> tuple[bool, float]:
    """
    Check that WORKING_DIR has at least `threshold_gb` GB free.

    Returns:
        enough_space: bool — True if threshold met
        free_gb: float — free space in GB (may be below threshold)
    """
    total, used, free = shutil.disk_usage(WORKING_DIR)
    free_gb = free / (1024 ** 3)
    enough_space = free_gb >= threshold_gb

    print(
        f"[platform] PLATFORM={PLATFORM} | WORKING_DIR={WORKING_DIR} "
        f"| free={free_gb:.1f}GB | threshold={threshold_gb}GB | "
        f"OK={enough_space}"
    )

    if not enough_space:
        print(
            f"[platform] WARNING: Low disk space in {WORKING_DIR}. "
            f"Free: {free_gb:.1f}GB, required: {threshold_gb}GB"
        )

    return enough_space, free_gb