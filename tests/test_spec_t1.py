"""
Tests for SPEC-T1 — Module 1: Portability & State Abstraction

Tests platform detection, hf_auth, hf_io, and data_gen modules.
All tests are offline — no network calls.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =====================================================================
# platform.py tests
# =====================================================================

class TestPlatform:
    def test_platform_is_local(self):
        """On dev machine, platform should be 'local'."""
        from shared.platform import PLATFORM
        assert PLATFORM == "local"

    def test_working_dir_exists(self):
        from shared.platform import WORKING_DIR
        assert WORKING_DIR.exists() or WORKING_DIR.parent.exists()

    def test_checkpoints_dir_created(self):
        from shared.platform import CHECKPOINTS_DIR
        assert CHECKPOINTS_DIR.exists()

    def test_hf_cache_dir_is_set(self):
        from shared.platform import HF_CACHE_DIR
        assert HF_CACHE_DIR is not None

    def test_verify_disk_space_returns_tuple(self):
        from shared.platform import verify_disk_space
        ok, free_gb = verify_disk_space(threshold_gb=1.0)
        assert isinstance(ok, bool)
        assert isinstance(free_gb, float)
        assert ok is True  # dev machine should have > 1GB

    def test_hf_telemetry_disabled(self):
        assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"

    def test_env_vars_set(self):
        """SPEC-T1 §7: HF_HUB_DISABLE_TELEMETRY must be '1'."""
        assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"


# =====================================================================
# hf_auth.py tests
# =====================================================================

class TestHfAuth:
    def test_get_token_from_env(self):
        """When HF_TOKEN is set, get_token returns it."""
        from shared.hf_auth import get_token
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token_123"}):
            token = get_token()
            assert token == "hf_test_token_123"

    def test_load_token_raises_when_missing(self):
        """On local with no HF_TOKEN, load_token raises."""
        from shared import hf_auth
        # Reset cached state
        hf_auth._token_loaded = False
        with patch.dict(os.environ, {}, clear=True):
            # Remove HF_TOKEN if present
            os.environ.pop("HF_TOKEN", None)
            with pytest.raises(RuntimeError, match="HF_TOKEN not set"):
                hf_auth.load_token()
        # Restore
        hf_auth._token_loaded = False

    def test_verify_token_calls_whoami(self):
        """verify_token should call HfApi.whoami."""
        from shared.hf_auth import verify_token
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "test_user"}
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            username = verify_token("fake_token")
            assert username == "test_user"
            mock_api.whoami.assert_called_once()

    def test_idempotent_load(self):
        """SPEC-T1 §4.4: multiple calls should be idempotent."""
        from shared import hf_auth
        hf_auth._token_loaded = False
        hf_auth._verified_username = None
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}):
            hf_auth._token_loaded = True
            token1 = hf_auth.load_token()
            token2 = hf_auth.load_token()
            assert token1 == token2


# =====================================================================
# hf_io.py tests
# =====================================================================

class TestHfIo:
    def test_retry_with_backoff_success(self):
        """Successful call on first attempt."""
        from shared.hf_io import retry_with_backoff
        result = retry_with_backoff(lambda: 42)
        assert result == 42

    def test_retry_with_backoff_eventual_success(self):
        """Retries on transient error, then succeeds."""
        from shared.hf_io import retry_with_backoff
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection reset by peer")
            return "ok"

        result = retry_with_backoff(flaky, max_retries=3, initial_backoff=0.01)
        assert result == "ok"
        assert call_count == 2

    def test_retry_with_backoff_exhausted(self):
        """All retries fail -> raises RuntimeError."""
        from shared.hf_io import retry_with_backoff

        def always_fail():
            raise ConnectionError("Connection reset by peer")

        with pytest.raises(RuntimeError, match="retries exhausted"):
            retry_with_backoff(always_fail, max_retries=2, initial_backoff=0.01)

    def test_pull_functions_exist(self):
        """All 4 pull functions are importable."""
        from shared.hf_io import (
            pull_reviewed_batch,
            pull_lora_adapter,
            pull_gnn_checkpoint,
            pull_baselines_log,
        )
        assert callable(pull_reviewed_batch)
        assert callable(pull_lora_adapter)
        assert callable(pull_gnn_checkpoint)
        assert callable(pull_baselines_log)

    def test_push_functions_exist(self):
        """All 4 push functions are importable."""
        from shared.hf_io import (
            push_reviewed_batch,
            push_sft_lora,
            push_gnn_checkpoint,
            append_and_push_baselines_log,
        )
        assert callable(push_reviewed_batch)
        assert callable(push_sft_lora)
        assert callable(push_gnn_checkpoint)
        assert callable(append_and_push_baselines_log)

    def test_pull_gnn_checkpoint_returns_none_for_batch_0(self):
        """Batch 0 has no prior checkpoint."""
        from shared.hf_io import pull_gnn_checkpoint
        result = pull_gnn_checkpoint(0, Path("/tmp"))
        assert result is None

    def test_pull_reviewed_batch_uses_dataset_repo_type(self, tmp_path: Path):
        """Dataset pulls must pass repo_type='dataset' to snapshot_download."""
        from shared import hf_io

        seen: dict[str, object] = {}
        local_file = tmp_path / "reviewed" / "batch_000.jsonl"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text("{}\n")

        def fake_snapshot_download(**kwargs):
            seen.update(kwargs)
            return str(tmp_path)

        with patch("shared.hf_io.snapshot_download", side_effect=fake_snapshot_download):
            with patch("shared.hf_auth.get_token", return_value="hf_test"):
                with patch("shared.hf_auth.get_username", return_value="testns"):
                    path = hf_io.pull_reviewed_batch(0, tmp_path)

        assert path == local_file
        assert seen.get("repo_type") == "dataset"


# =====================================================================
# validate.py tests
# =====================================================================

class TestValidate:
    def _make_valid_example(self) -> dict:
        return {
            "task_seed_id": "task_easy",
            "tier": "easy",
            "fault_type": "oom",
            "variation_strategy": "baseline",
            "observation": {
                "tick": 1,
                "budget": 30.0,
                "alerts": ["OOMKilled on auth-service"],
                "service_metrics": {"auth-service": {"status": "down"}},
                "logs": {"auth-service": ["OOMKilled"]},
            },
            "gold_action_sequence": [
                {"action": "fetch_logs", "params": {"service": "auth-service"}},
            ],
            "gold_alternatives": [],
            "expected_score_range": {"min": 0.7, "max": 1.0},
            "suboptimal_paths": [],
        }

    def test_valid_example_passes(self):
        from data_gen.validate import validate_example
        errs = validate_example(self._make_valid_example())
        assert errs == []

    def test_missing_field_detected(self):
        from data_gen.validate import validate_example
        ex = self._make_valid_example()
        del ex["tier"]
        errs = validate_example(ex)
        assert any("tier" in e for e in errs)

    def test_invalid_tier_detected(self):
        from data_gen.validate import validate_example
        ex = self._make_valid_example()
        ex["tier"] = "nightmare"
        errs = validate_example(ex)
        assert any("tier" in e for e in errs)

    def test_missing_observation_field(self):
        from data_gen.validate import validate_example
        ex = self._make_valid_example()
        del ex["observation"]["alerts"]
        errs = validate_example(ex)
        assert any("alerts" in e for e in errs)

    def test_validate_batch(self):
        from data_gen.validate import validate_batch
        batch = [self._make_valid_example() for _ in range(3)]
        errs = validate_batch(batch)
        assert errs == []

    def test_validate_batch_with_error(self):
        from data_gen.validate import validate_batch
        batch = [self._make_valid_example()]
        batch[0]["tier"] = "invalid"
        errs = validate_batch(batch)
        assert len(errs) > 0


# =====================================================================
# data_gen orchestration tests
# =====================================================================

class TestRunGeneratorCli:
    def test_resolve_batch_zero_maps_to_script_one(self):
        from data_gen.run_generator import resolve_script_and_batch

        script_num, batch_num = resolve_script_and_batch(script="01", batch=None)
        assert script_num == 1
        assert batch_num == 0

        script_num, batch_num = resolve_script_and_batch(script=None, batch=0)
        assert script_num == 1
        assert batch_num == 0

    def test_resolve_conflicting_script_and_batch_raises(self):
        from data_gen.run_generator import resolve_script_and_batch

        with pytest.raises(ValueError, match="conflicting"):
            resolve_script_and_batch(script="02", batch=0)

    def test_normalize_example_contract_converts_legacy_generator_shapes(self):
        from data_gen.run_generator import normalize_example_contract

        example = TestValidate()._make_valid_example()
        example["task_seed_id"] = "task_easy_oom_baseline"
        example["expected_score_range"] = [0.7, 1.0]
        example["gold_action_sequence"] = [
            "fetch_logs(auth-service)",
            "declare_resolved",
        ]

        normalize_example_contract(example, source_task={"fault_service": "auth-service"})

        assert example["fault_service"] == "auth-service"
        assert example["expected_score_range"] == {"min": 0.7, "max": 1.0}
        assert example["gold_action_sequence"] == [
            {"action": "fetch_logs", "params": {"service": "auth-service"}},
            {"action": "declare_resolved", "params": {}},
        ]


class TestBatchCompliance:
    def _make_strict_example(self) -> dict:
        example = TestValidate()._make_valid_example()
        example["example_id"] = "example-001"
        example["source_script"] = "gen_01_mixed_metric_a.py"
        example["fault_service"] = "auth-service"
        return example

    def test_load_jsonl_reports_malformed_json(self, tmp_path: Path):
        from data_gen.check_batch import check_jsonl_file

        batch_path = tmp_path / "batch_000.jsonl"
        batch_path.write_text('{"example_id": "ok"}\n{not-json}\n')

        result = check_jsonl_file(batch_path)

        assert result.ok is False
        assert any("invalid JSON" in err for err in result.errors)

    def test_strict_check_requires_example_id_source_and_root_cause(self):
        from data_gen.check_batch import check_examples

        example = self._make_strict_example()
        del example["example_id"]
        del example["source_script"]
        del example["fault_service"]
        example["gold_action_sequence"] = [{"action": "fetch_logs", "params": {}}]

        result = check_examples([example], expected_count=1)

        assert result.ok is False
        assert any("example_id" in err for err in result.errors)
        assert any("source_script" in err for err in result.errors)
        assert any("root-cause" in err for err in result.errors)

    def test_strict_check_rejects_invalid_score_range_and_action_shape(self):
        from data_gen.check_batch import check_examples

        example = self._make_strict_example()
        example["expected_score_range"] = {"min": 1.2, "max": 0.3}
        example["gold_action_sequence"] = [{"params": {"service": "auth-service"}}]

        result = check_examples([example], expected_count=1)

        assert result.ok is False
        assert any("expected_score_range" in err for err in result.errors)
        assert any("action name" in err for err in result.errors)


class TestDataUpload:
    def test_upload_batch_uses_shared_import_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from data_gen import upload
        from shared import hf_io

        reviewed_dir = tmp_path / "reviewed"
        reviewed_dir.mkdir()
        reviewed_file = reviewed_dir / "batch_000.jsonl"
        reviewed_file.write_text(
            json.dumps(
                {
                    "example_id": "example-001",
                    "source_script": "gen_01_mixed_metric_a.py",
                }
            )
            + "\n"
        )

        calls: list[tuple[int, Path, str]] = []

        def fake_push_reviewed_batch(batch_num: int, local_file: Path, commit_message: str) -> None:
            calls.append((batch_num, local_file, commit_message))

        monkeypatch.setattr(upload, "REVIEWED_DIR", reviewed_dir)
        monkeypatch.setattr(hf_io, "push_reviewed_batch", fake_push_reviewed_batch)

        upload.upload_batch(0)

        assert calls
        assert calls[0][0] == 0
        assert calls[0][1] == reviewed_file
        assert "gen_01_mixed_metric_a.py" in calls[0][2]


# =====================================================================
# Shared __init__.py exports test
# =====================================================================

class TestSharedExports:
    def test_all_exports_importable(self):
        """Every name in __all__ should be importable."""
        from shared import __all__ as exports
        import shared
        for name in exports:
            assert hasattr(shared, name), f"Missing export: {name}"
