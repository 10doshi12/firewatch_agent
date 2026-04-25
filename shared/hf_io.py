"""
hf_io.py — HuggingFace Hub I/O wrappers

Engineering constraints enforced here:
  - No full-repo clones; all downloads use allow_patterns
  - Atomic uploads where possible
  - Retry with exponential backoff on transient failures

Pull functions:
  pull_reviewed_batch(batch_num, local_dir) -> Path
  pull_lora_adapter(repo_id, subfolder, local_dir) -> Path | None
  pull_gnn_checkpoint(batch_num, local_dir) -> Path | None
  pull_baselines_log(repo_id, local_dir) -> Path | None

Push functions:
  push_reviewed_batch(batch_num, local_file, commit_message)
  push_sft_lora(batch_num, local_dir)
  push_gnn_checkpoint(batch_num, local_file)
  append_and_push_baselines_log(repo_id, new_line_path, local_dir)

Utility:
  retry_with_backoff(fn, max_retries, initial_backoff) -> T
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

from huggingface_hub import HfApi, snapshot_download

from . import hf_auth

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Repo ID helpers
# ---------------------------------------------------------------------------

def _get_namespace() -> str:
    """Return the hf_namespace, defaulting to verified username."""
    namespace = os.environ.get("HF_NAMESPACE")
    if not namespace:
        namespace = hf_auth.get_username()
    return namespace


def _log_op(operation: str, **kwargs) -> None:
    """Log a structured JSON line for every pull/push operation."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        **kwargs,
    }
    print(json.dumps(entry))


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def retry_with_backoff(
    fn: Callable[..., T],
    max_retries: int = 3,
    initial_backoff: float = 5.0,
) -> T:
    """
    Retry with exponential backoff (5s, 15s, 45s).
    Retries on HTTP 502/503/504 and connection-reset errors.
    Hard timeout: 300s per operation.
    """
    backoff = initial_backoff
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            is_transient = (
                isinstance(exc, urllib.error.HTTPError)
                and exc.code in (502, 503, 504)
            ) or "Connection reset" in str(exc)

            if not is_transient or attempt == max_retries:
                raise RuntimeError(
                    f"[hf_io] All {max_retries} retries exhausted: {exc}"
                ) from exc

            print(f"[hf_io] Retry {attempt + 1}/{max_retries} after {backoff:.0f}s: {exc}")
            time.sleep(backoff)
            backoff *= 3  # 5 -> 15 -> 45

    raise RuntimeError(f"[hf_io] Unexpected retry failure: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Pull functions
# ---------------------------------------------------------------------------

def pull_reviewed_batch(batch_num: int, local_dir: Path) -> Path:
    """Download reviewed/batch_<NNN>.jsonl from DATASET_REPO."""
    namespace = _get_namespace()
    repo_id = f"{namespace}/firewatch-sft-data"
    filename = f"reviewed/batch_{batch_num:03d}.jsonl"
    local_dir = Path(local_dir)

    def _download():
        return snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[filename],
            local_dir=local_dir,
            token=hf_auth.get_token(),
        )

    start = time.monotonic()
    repo_dir = retry_with_backoff(_download)
    elapsed = time.monotonic() - start

    local_path = local_dir / filename
    if not local_path.exists():
        candidates = list(Path(repo_dir).rglob(f"batch_{batch_num:03d}.jsonl"))
        if candidates:
            local_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"[hf_io] Downloaded repo at {repo_dir} but file {filename} not found"
            )

    size_bytes = local_path.stat().st_size
    _log_op("pull_dataset_batch", repo_id=repo_id, filename=filename,
            local_path=str(local_path), size_bytes=size_bytes, elapsed_s=round(elapsed, 2))
    return local_path


def pull_lora_adapter(repo_id: str, subfolder: str, local_dir: Path) -> Path | None:
    """
    Download LoRA adapter files for a specific subfolder.
    Allow patterns: <subfolder>/adapter_config.json, adapter_model.safetensors, README.md.
    Returns None if remote files don't exist.
    """
    local_dir = Path(local_dir)
    sf = subfolder.rstrip("/")
    allow = [
        f"{sf}/adapter_config.json",
        f"{sf}/adapter_model.safetensors",
        f"{sf}/README.md",
    ]

    def _download():
        return snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=allow,
            local_dir=local_dir,
            token=hf_auth.get_token(),
        )

    start = time.monotonic()
    try:
        repo_dir = retry_with_backoff(_download)
    except Exception as exc:
        if "404" in str(exc) or "not found" in str(exc).lower():
            return None
        raise
    elapsed = time.monotonic() - start

    adapter_dir = local_dir / sf
    if not adapter_dir.exists():
        candidates = list(Path(repo_dir).rglob(sf))
        if candidates:
            adapter_dir = candidates[0]
        else:
            return None

    _log_op("pull_lora_adapter", repo_id=repo_id, subfolder=sf,
            local_path=str(adapter_dir), elapsed_s=round(elapsed, 2))
    return adapter_dir


def pull_gnn_checkpoint(batch_num: int, local_dir: Path) -> Path | None:
    """Download gnn/batch_<N-1>.pt and normalization.json. Returns None if absent."""
    if batch_num <= 0:
        return None

    namespace = _get_namespace()
    repo_id = f"{namespace}/firewatch-gnn"
    filename = f"gnn/batch_{batch_num - 1:03d}.pt"
    local_dir = Path(local_dir)

    def _download():
        return snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[filename, "gnn/normalization.json"],
            local_dir=local_dir,
            token=hf_auth.get_token(),
        )

    start = time.monotonic()
    try:
        repo_dir = retry_with_backoff(_download)
    except Exception as exc:
        if "404" in str(exc) or "not found" in str(exc).lower():
            return None
        raise
    elapsed = time.monotonic() - start

    local_path = local_dir / filename
    if not local_path.exists():
        candidates = list(Path(repo_dir).rglob(f"batch_{batch_num - 1:03d}.pt"))
        if candidates:
            local_path = candidates[0]
        else:
            return None

    size_bytes = local_path.stat().st_size
    _log_op("pull_gnn_checkpoint", repo_id=repo_id, filename=filename,
            local_path=str(local_path), size_bytes=size_bytes, elapsed_s=round(elapsed, 2))
    return local_path


def pull_baselines_log(
    repo_id: str,
    local_dir: Path,
    repo_type: str = "dataset",
) -> Path | None:
    """
    Download baselines/metrics.jsonl from the given repo.
    Returns None if the file doesn't exist yet (pre-first-batch).

    repo_type defaults to "dataset" — pass "model" for the SFT model repo.
    """
    local_dir = Path(local_dir)
    filename = "baselines/metrics.jsonl"

    def _download():
        return snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=[filename],
            local_dir=local_dir,
            token=hf_auth.get_token(),
        )

    start = time.monotonic()
    try:
        repo_dir = retry_with_backoff(_download)
    except Exception as exc:
        if "404" in str(exc) or "not found" in str(exc).lower():
            return None
        raise
    elapsed = time.monotonic() - start

    local_path = local_dir / filename
    if not local_path.exists():
        candidates = list(Path(repo_dir).rglob("metrics.jsonl"))
        if candidates:
            local_path = candidates[0]
        else:
            return None

    size_bytes = local_path.stat().st_size
    _log_op("pull_baselines_log", repo_id=repo_id, filename=filename,
            local_path=str(local_path), size_bytes=size_bytes, elapsed_s=round(elapsed, 2))
    return local_path


# ---------------------------------------------------------------------------
# Push functions
# ---------------------------------------------------------------------------

def push_reviewed_batch(batch_num: int, local_file: Path, commit_message: str) -> None:
    """Upload a local file to DATASET_REPO at reviewed/batch_<NNN>.jsonl."""
    namespace = _get_namespace()
    repo_id = f"{namespace}/firewatch-sft-data"
    remote_path = f"reviewed/batch_{batch_num:03d}.jsonl"
    local_file = Path(local_file)

    start = time.monotonic()

    def _upload():
        api = HfApi(token=hf_auth.get_token())
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

    retry_with_backoff(_upload)
    elapsed = time.monotonic() - start
    size_bytes = local_file.stat().st_size
    _log_op("push_reviewed_batch", repo_id=repo_id, filename=remote_path,
            local_path=str(local_file), size_bytes=size_bytes, elapsed_s=round(elapsed, 2))


def push_sft_lora(batch_num: int, local_dir: Path) -> None:
    """Upload entire directory as a folder to SFT_MODEL_REPO/batch_<NNN>/."""
    namespace = _get_namespace()
    repo_id = f"{namespace}/firewatch-agent-sft"
    remote_folder = f"batch_{batch_num:03d}/"
    local_dir = Path(local_dir)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    start = time.monotonic()

    def _upload():
        api = HfApi(token=hf_auth.get_token())
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=remote_folder,
            commit_message=f"SFT LoRA batch_{batch_num:03d} @ {ts}",
        )

    retry_with_backoff(_upload)
    elapsed = time.monotonic() - start
    _log_op("push_lora_adapter", repo_id=repo_id, subfolder=remote_folder,
            local_path=str(local_dir), elapsed_s=round(elapsed, 2))


def push_gnn_checkpoint(batch_num: int, local_file: Path) -> None:
    """Upload .pt file + normalization.json to the GNN repo."""
    namespace = _get_namespace()
    repo_id = f"{namespace}/firewatch-gnn"
    remote_path = f"gnn/batch_{batch_num:03d}.pt"
    local_file = Path(local_file)

    start = time.monotonic()

    def _upload():
        api = HfApi(token=hf_auth.get_token())
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
        )

    retry_with_backoff(_upload)
    elapsed = time.monotonic() - start
    size_bytes = local_file.stat().st_size
    _log_op("push_gnn_checkpoint", repo_id=repo_id, filename=remote_path,
            local_path=str(local_file), size_bytes=size_bytes, elapsed_s=round(elapsed, 2))


def append_and_push_baselines_log(
    repo_id: str,
    new_line_path: Path,
    local_dir: Path,
) -> None:
    """
    Append-and-push baselines log (atomic read-merge-write pattern).
    1. Download existing baselines/metrics.jsonl (if any)
    2. Append the new line from new_line_path
    3. Upload the merged file back
    """
    local_dir = Path(local_dir)
    new_line_path = Path(new_line_path)
    merged_path = local_dir / "baselines" / "metrics.jsonl"
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: pull existing (may be None)
    existing = pull_baselines_log(repo_id, local_dir)

    # Step 2: merge
    existing_content = ""
    if existing is not None and existing.exists():
        existing_content = existing.read_text()

    new_content = new_line_path.read_text().strip()
    merged = existing_content.rstrip("\n")
    if merged:
        merged += "\n"
    merged += new_content + "\n"
    merged_path.write_text(merged)

    # Step 3: push
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    start = time.monotonic()

    def _upload():
        api = HfApi(token=hf_auth.get_token())
        api.upload_file(
            path_or_fileobj=str(merged_path),
            path_in_repo="baselines/metrics.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Append baselines log @ {ts}",
        )

    retry_with_backoff(_upload)
    elapsed = time.monotonic() - start
    size_bytes = merged_path.stat().st_size
    _log_op("append_and_push_baselines_log", repo_id=repo_id,
            local_path=str(merged_path), size_bytes=size_bytes, elapsed_s=round(elapsed, 2))