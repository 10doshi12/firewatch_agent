"""
Tests for sft.train / sft.preflight exit-code contracts.

These codes are consumed by hf_space_sft_worker/start.sh and MUST match.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_agent_root = Path(__file__).resolve().parent.parent
if str(_agent_root) not in sys.path:
    sys.path.insert(0, str(_agent_root))


def _import_train():
    from sft import train
    return train


def _import_preflight():
    from sft import preflight
    return preflight


class TestSFTTrainExitCodes:
    def test_constants_match_worker_contract(self):
        train = _import_train()
        assert train.EXIT_OK == 0
        assert train.EXIT_ERROR == 1
        assert train.EXIT_CAMPAIGN_COMPLETE == 2
        assert train.EXIT_OOM == 3

    def test_campaign_complete_exits_with_code_2(self, monkeypatch):
        train = _import_train()
        monkeypatch.setattr(
            train, "run_sft_batch", lambda config_path=None: train.EXIT_CAMPAIGN_COMPLETE
        )
        monkeypatch.setattr(sys, "argv", ["train.py"])
        with pytest.raises(SystemExit) as exc_info:
            train.main()
        assert exc_info.value.code == 2

    def test_success_exits_with_code_0(self, monkeypatch):
        train = _import_train()
        monkeypatch.setattr(
            train, "run_sft_batch", lambda config_path=None: train.EXIT_OK
        )
        monkeypatch.setattr(sys, "argv", ["train.py"])
        with pytest.raises(SystemExit) as exc_info:
            train.main()
        assert exc_info.value.code == 0

    def test_cuda_oom_class_exits_with_code_3(self, monkeypatch):
        import torch

        train = _import_train()
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is None:
            pytest.skip("torch.cuda.OutOfMemoryError not available in this torch")

        def boom(config_path=None):
            raise oom_cls("CUDA out of memory. Tried to allocate 24.00 GiB")

        monkeypatch.setattr(train, "run_sft_batch", boom)
        monkeypatch.setattr(sys, "argv", ["train.py"])
        with pytest.raises(SystemExit) as exc_info:
            train.main()
        assert exc_info.value.code == 3

    def test_runtime_error_oom_string_exits_with_code_3(self, monkeypatch):
        train = _import_train()

        def boom(config_path=None):
            raise RuntimeError(
                "CUDA error: out of memory while allocating tensor"
            )

        monkeypatch.setattr(train, "run_sft_batch", boom)
        monkeypatch.setattr(sys, "argv", ["train.py"])
        with pytest.raises(SystemExit) as exc_info:
            train.main()
        assert exc_info.value.code == 3

    def test_generic_error_exits_with_code_1(self, monkeypatch):
        train = _import_train()

        def boom(config_path=None):
            raise ValueError("config malformed")

        monkeypatch.setattr(train, "run_sft_batch", boom)
        monkeypatch.setattr(sys, "argv", ["train.py"])
        with pytest.raises(SystemExit) as exc_info:
            train.main()
        assert exc_info.value.code == 1

    def test_runtime_error_non_oom_exits_with_code_1(self, monkeypatch):
        train = _import_train()

        def boom(config_path=None):
            raise RuntimeError("Hub upload rejected: 401 Unauthorized")

        monkeypatch.setattr(train, "run_sft_batch", boom)
        monkeypatch.setattr(sys, "argv", ["train.py"])
        with pytest.raises(SystemExit) as exc_info:
            train.main()
        assert exc_info.value.code == 1


class TestPreflightExitCodes:
    def test_campaign_done_exits_with_code_2(self, monkeypatch):
        preflight = _import_preflight()
        result = preflight.PreflightResult(
            ok=True,
            namespace="test/ns",
            batch_num=None,
            errors=[],
            warnings=[],
            details={"campaign_done": True, "namespace": "test/ns", "batch_num": None},
            campaign_done=True,
        )
        monkeypatch.setattr(preflight, "run_preflight", lambda **kw: result)
        monkeypatch.setattr(sys, "argv", ["preflight.py"])
        with pytest.raises(SystemExit) as exc_info:
            preflight.main()
        assert exc_info.value.code == 2

    def test_errors_exit_with_code_1(self, monkeypatch):
        preflight = _import_preflight()
        result = preflight.PreflightResult(
            ok=False,
            namespace="test/ns",
            batch_num=None,
            errors=["repo missing"],
            warnings=[],
            details={"namespace": "test/ns"},
            campaign_done=False,
        )
        monkeypatch.setattr(preflight, "run_preflight", lambda **kw: result)
        monkeypatch.setattr(sys, "argv", ["preflight.py"])
        with pytest.raises(SystemExit) as exc_info:
            preflight.main()
        assert exc_info.value.code == 1

    def test_pass_does_not_exit(self, monkeypatch):
        preflight = _import_preflight()
        result = preflight.PreflightResult(
            ok=True,
            namespace="test/ns",
            batch_num=0,
            errors=[],
            warnings=[],
            details={"namespace": "test/ns", "batch_num": 0},
            campaign_done=False,
        )
        monkeypatch.setattr(preflight, "run_preflight", lambda **kw: result)
        monkeypatch.setattr(sys, "argv", ["preflight.py"])
        # Successful preflight reaches "OK: SFT preflight passed" without sys.exit.
        preflight.main()
