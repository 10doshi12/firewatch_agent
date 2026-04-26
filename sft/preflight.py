"""
preflight.py — cheap readiness checks before real SFT GPU training.

This module verifies auth, Hub repo layout, reviewed data compliance,
**Unsloth** (required — no dense fallback), CUDA, and disk before `sft.train`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from huggingface_hub import HfApi

from data_gen.check_batch import check_jsonl_file
from shared import hf_auth, hf_io
from shared import model_runtime
from shared.platform import WORKING_DIR, verify_disk_space
from sft.campaign import data_batches_for_run
from sft.train import detect_current_batch, load_config


@dataclass
class PreflightResult:
    """Structured preflight result for tests, notebooks, and CLI output."""

    ok: bool
    namespace: str
    batch_num: int | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, str | int | float | bool | None] = field(default_factory=dict)
    campaign_done: bool = False


# Exit codes match firewatch_agent/sft/train.py — kept in sync with start.sh.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CAMPAIGN_COMPLETE = 2


REQUIRED_REPOS: tuple[tuple[str, str, bool], ...] = (
    ("firewatch-sft-data", "dataset", True),
    ("firewatch-agent-sft", "model", True),
    ("firewatch-gnn", "model", True),
    ("firewatch-agent-grpo", "model", False),
)


def _resolve_namespace(config: dict, username: str) -> str:
    return hf_auth.resolve_namespace(config, username)


def _check_repos(api: HfApi, namespace: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for suffix, repo_type, required in REQUIRED_REPOS:
        repo_id = f"{namespace}/{suffix}"
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
        except Exception as exc:
            message = f"{repo_id} ({repo_type}) not reachable: {exc}"
            if required:
                errors.append(message)
            else:
                warnings.append(message)
    return errors, warnings


def _detect_batch(namespace: str, config: dict) -> tuple[int | None, list[str]]:
    try:
        campaign = (config.get("sft") or {}).get("campaign", "paired_15")
        return detect_current_batch(namespace, campaign), []
    except SystemExit as exc:
        return None, [f"could not detect current batch: exited with {exc.code}"]
    except Exception as exc:
        return None, [f"could not detect current batch: {exc}"]


def _check_unsloth(_config: dict) -> tuple[list[str], list[str]]:
    # Resolve via `model_runtime` so tests can patch `model_runtime.try_import_unsloth`.
    FastLanguageModel, error = model_runtime.try_import_unsloth()
    if FastLanguageModel is not None:
        return [], []

    err = error or "unknown import error"
    return [
        "Unsloth is required for SFT training but failed to import. "
        f"Install the `unsloth` package in a GPU environment (e.g. `pip install unsloth` "
        f"or the Colab wheel). No dense/PyTorch fallback is supported. "
        f"Details: {err}"
    ], []


def _check_cuda(require_cuda: bool) -> tuple[list[str], dict[str, str | bool | None]]:
    details: dict[str, str | bool | None] = {"cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return (["CUDA is not available for real SFT"] if require_cuda else []), details
    try:
        details["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        details["cuda_device"] = None
    return [], details


def run_preflight(
    config_path: Path | None = None,
    *,
    require_cuda: bool = True,
    disk_threshold_gb: float = 20.0,
) -> PreflightResult:
    """Run cheap checks needed before invoking real `sft.train`."""
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, str | int | float | bool | None] = {}
    namespace = ""
    batch_num: int | None = None

    config = load_config(config_path)

    try:
        username = hf_auth.get_username()
        token = hf_auth.get_token()
        namespace = _resolve_namespace(config, username)
        os.environ["HF_NAMESPACE"] = namespace
        details["hf_username"] = username
    except Exception as exc:
        errors.append(f"HF auth failed: {exc}")
        return PreflightResult(False, namespace, batch_num, errors, warnings, details)

    api = HfApi(token=token)
    repo_errors, repo_warnings = _check_repos(api, namespace)
    errors.extend(repo_errors)
    warnings.extend(repo_warnings)

    campaign = (config.get("sft") or {}).get("campaign", "paired_15")
    details["sft_campaign"] = campaign
    detected_batch, batch_errors = _detect_batch(namespace, config)
    batch_num = detected_batch
    errors.extend(batch_errors)

    # Campaign complete: auth + repos OK and detection cleanly returned None.
    # Skip Unsloth/CUDA/disk checks — there's no training to gate.
    campaign_done = (
        not errors
        and batch_num is None
        and not batch_errors
    )
    if campaign_done:
        details["namespace"] = namespace
        details["batch_num"] = None
        details["campaign_done"] = True
        return PreflightResult(
            ok=True,
            namespace=namespace,
            batch_num=None,
            errors=errors,
            warnings=warnings,
            details=details,
            campaign_done=True,
        )

    if batch_num is not None:
        local_dir = WORKING_DIR / "sft_preflight"
        local_dir.mkdir(parents=True, exist_ok=True)
        paired = str(campaign).lower() in ("paired_15", "paired", "15")
        try:
            if paired:
                a, b = data_batches_for_run(batch_num)
                path_a = hf_io.pull_reviewed_batch(a, local_dir)
                path_b = hf_io.pull_reviewed_batch(b, local_dir)
                total = 0
                for p in (path_a, path_b):
                    br = check_jsonl_file(p, expected_count=50)
                    total += br.example_count
                    if not br.ok:
                        errors.extend([f"{p.name}: {e}" for e in br.errors])
                details["reviewed_data_batches"] = f"{a:03d}+{b:03d}"
                details["reviewed_batch_paths"] = [str(path_a), str(path_b)]
                details["reviewed_example_count"] = total
            else:
                batch_path = hf_io.pull_reviewed_batch(batch_num, local_dir)
                batch_result = check_jsonl_file(batch_path, expected_count=50)
                details["reviewed_batch_path"] = str(batch_path)
                details["reviewed_example_count"] = batch_result.example_count
                if not batch_result.ok:
                    errors.extend(batch_result.errors)
        except Exception as exc:
            errors.append(f"could not pull reviewed data for next step {batch_num}: {exc}")

    unsloth_errors, unsloth_warnings = _check_unsloth(config)
    errors.extend(unsloth_errors)
    warnings.extend(unsloth_warnings)

    cuda_errors, cuda_details = _check_cuda(require_cuda=require_cuda)
    errors.extend(cuda_errors)
    details.update(cuda_details)

    enough_space, free_gb = verify_disk_space(threshold_gb=disk_threshold_gb)
    details["free_disk_gb"] = round(free_gb, 2)
    if not enough_space:
        errors.append(
            f"insufficient disk space: {free_gb:.1f}GB free, "
            f"requires {disk_threshold_gb:.1f}GB"
        )

    details["namespace"] = namespace
    details["batch_num"] = batch_num
    return PreflightResult(not errors, namespace, batch_num, errors, warnings, details)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for Firewatch SFT training")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--allow-cpu", action="store_true", help="Do not fail when CUDA is unavailable")
    parser.add_argument("--disk-threshold-gb", type=float, default=20.0)
    args = parser.parse_args()

    result = run_preflight(
        config_path=args.config,
        require_cuda=not args.allow_cpu,
        disk_threshold_gb=args.disk_threshold_gb,
    )
    print(json.dumps(result.details, indent=2, sort_keys=True))
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    if not result.ok:
        for error in result.errors:
            print(f"ERROR: {error}")
        sys.exit(EXIT_ERROR)
    if result.campaign_done:
        print("OK: SFT campaign complete — no untrained runs remain")
        sys.exit(EXIT_CAMPAIGN_COMPLETE)
    print("OK: SFT preflight passed")


if __name__ == "__main__":
    main()
