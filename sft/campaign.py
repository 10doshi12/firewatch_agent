"""
SFT campaign modes: paired_15 (30 data files, 15 training steps) vs legacy_30.

Training ids are **incremental**: each run loads GNN + LoRA **saved on Hub from the
previous run** (``batch_{k-1}``) and after training writes ``batch_k``. Detection
picks the smallest ``k`` with reviewed data but no ``batch_k/`` in the SFT model repo.
"""

from __future__ import annotations

import os
import sys
from typing import Final

TRAINING_RUNS_PAIRED: Final[int] = 15
DATA_BATCHES_TOTAL: Final[int] = 30
FINAL_TRAINING_RUN_PAIRED: Final[int] = 14


def data_batches_for_run(run_idx: int) -> tuple[int, int]:
    """Data batch indices (0..29) for training run `run_idx` (0..14)."""
    if not (0 <= run_idx < TRAINING_RUNS_PAIRED):
        raise ValueError(f"run_idx must be 0..{TRAINING_RUNS_PAIRED - 1}, got {run_idx}")
    a = 2 * run_idx
    b = 2 * run_idx + 1
    if b >= DATA_BATCHES_TOTAL:
        raise ValueError("invalid run_idx for 30 data batches")
    return a, b


def _parse_reviewed_batch_nums(files: list[str]) -> set[int]:
    out: set[int] = set()
    for f in files:
        if f.startswith("reviewed/batch_") and f.endswith(".jsonl"):
            name = f.split("/")[-1].replace("batch_", "").replace(".jsonl", "")
            try:
                out.add(int(name))
            except ValueError:
                continue
    return out


def _parse_trained_lora_runs(files: list[str]) -> set[int]:
    out: set[int] = set()
    for f in files:
        parts = f.split("/")
        if len(parts) >= 2 and parts[0].startswith("batch_"):
            name = parts[0].replace("batch_", "")
            try:
                out.add(int(name))
            except ValueError:
                continue
    return out


def detect_next_training_run_paired(namespace: str) -> int | None:
    from huggingface_hub import HfApi

    from shared import hf_auth  # noqa: WPS433

    api = HfApi(token=hf_auth.get_token())
    dataset_repo = f"{namespace}/firewatch-sft-data"
    model_repo = f"{namespace}/firewatch-agent-sft"

    try:
        dfiles = api.list_repo_files(dataset_repo, repo_type="dataset")
    except Exception as exc:
        if "404" in str(exc):
            print("[train] No dataset repo on HF", file=sys.stderr)
            sys.exit(1)
        raise

    try:
        mfiles = api.list_repo_files(model_repo, repo_type="model")
    except Exception:
        mfiles = []

    reviewed = _parse_reviewed_batch_nums(dfiles)
    trained = _parse_trained_lora_runs(mfiles)

    for k in range(TRAINING_RUNS_PAIRED):
        if k in trained:
            continue
        a, b = data_batches_for_run(k)
        if a in reviewed and b in reviewed:
            return k

    if trained.issuperset(set(range(TRAINING_RUNS_PAIRED))):
        print(
            f"[train] SFT campaign complete — all {TRAINING_RUNS_PAIRED} training runs have LoRAs"
        )
    else:
        print(
            "[train] No trainable run: need both reviewed/batch_2k and batch_2k+1 on the dataset "
            "repo, and no LoRA at batch_k/ on the SFT model repo."
        )
    return None


def detect_next_data_batch_legacy(namespace: str) -> int | None:
    from huggingface_hub import HfApi

    from shared import hf_auth  # noqa: WPS433

    api = HfApi(token=hf_auth.get_token())
    dataset_repo = f"{namespace}/firewatch-sft-data"
    model_repo = f"{namespace}/firewatch-agent-sft"

    try:
        files = api.list_repo_files(dataset_repo, repo_type="dataset")
    except Exception as exc:
        if "404" in str(exc):
            print("[train] No dataset repo on HF", file=sys.stderr)
            sys.exit(1)
        raise

    reviewed = sorted(
        f for f in files
        if f.startswith("reviewed/batch_") and f.endswith(".jsonl")
    )
    if not reviewed:
        print("[train] No reviewed batches on HF", file=sys.stderr)
        sys.exit(1)

    reviewed_nums: set[int] = set()
    for f in reviewed:
        name = f.split("/")[-1].replace("batch_", "").replace(".jsonl", "")
        try:
            reviewed_nums.add(int(name))
        except ValueError:
            continue

    trained_nums: set[int] = set()
    try:
        mfiles = api.list_repo_files(model_repo, repo_type="model")
        for f in mfiles:
            parts = f.split("/")
            if len(parts) >= 2 and parts[0].startswith("batch_"):
                name = parts[0].replace("batch_", "")
                try:
                    trained_nums.add(int(name))
                except ValueError:
                    continue
    except Exception:
        pass

    untrained = sorted(reviewed_nums - trained_nums)
    if not untrained:
        print("[train] SFT campaign complete — all reviewed batches have trained LoRAs")
        return None
    return untrained[0]


def detect_next_sft_step(namespace: str, campaign: str | None) -> int | None:
    mode = (campaign or os.environ.get("SFT_CAMPAIGN") or "paired_15").strip().lower()
    if mode in ("legacy_30", "legacy", "30"):
        return detect_next_data_batch_legacy(namespace)
    if mode in ("paired_15", "paired", "15"):
        return detect_next_training_run_paired(namespace)
    raise ValueError(f"Unknown sft.campaign: {campaign!r} (use paired_15 or legacy_30)")
