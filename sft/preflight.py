"""
preflight.py — cheap readiness checks before real SFT GPU training.

This module verifies auth, Hub repo layout, reviewed data compliance,
Unsloth availability, CUDA, and disk before `sft.train` loads the base LLM.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from huggingface_hub import HfApi

from data_gen.check_batch import check_jsonl_file
from shared import hf_auth, hf_io
from shared.platform import WORKING_DIR, verify_disk_space
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


REQUIRED_REPOS: tuple[tuple[str, str, bool], ...] = (
    ("firewatch-sft-data", "dataset", True),
    ("firewatch-agent-sft", "model", True),
    ("firewatch-gnn", "model", True),
    ("firewatch-agent-grpo", "model", False),
)


def _resolve_namespace(config: dict, username: str) -> str:
    namespace = config.get("hf_namespace") or os.environ.get("HF_NAMESPACE") or username
    return str(namespace)


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


def _detect_batch(namespace: str) -> tuple[int | None, list[str]]:
    try:
        return detect_current_batch(namespace), []
    except SystemExit as exc:
        return None, [f"could not detect current batch: exited with {exc.code}"]
    except Exception as exc:
        return None, [f"could not detect current batch: {exc}"]


def _check_unsloth() -> list[str]:
    if importlib.util.find_spec("unsloth") is None:
        return ["Unsloth is not importable in this runtime"]
    return []


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
        details["hf_username"] = username
    except Exception as exc:
        errors.append(f"HF auth failed: {exc}")
        return PreflightResult(False, namespace, batch_num, errors, warnings, details)

    api = HfApi(token=token)
    repo_errors, repo_warnings = _check_repos(api, namespace)
    errors.extend(repo_errors)
    warnings.extend(repo_warnings)

    detected_batch, batch_errors = _detect_batch(namespace)
    batch_num = detected_batch
    errors.extend(batch_errors)

    if batch_num is not None:
        local_dir = WORKING_DIR / "sft_preflight"
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            batch_path = hf_io.pull_reviewed_batch(batch_num, local_dir)
        except Exception as exc:
            errors.append(f"could not pull reviewed batch {batch_num:03d}: {exc}")
        else:
            batch_result = check_jsonl_file(batch_path, expected_count=50)
            details["reviewed_batch_path"] = str(batch_path)
            details["reviewed_example_count"] = batch_result.example_count
            if not batch_result.ok:
                errors.extend(batch_result.errors)

    errors.extend(_check_unsloth())

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
        sys.exit(1)
    print("OK: SFT preflight passed")


if __name__ == "__main__":
    main()
