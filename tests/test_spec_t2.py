"""
Tests for SPEC-T2 — Module 2: SFT + GNN Sequential Training Engine

All tests are offline — no network calls, no GPU required.
Tests GNN adjacency, model, serializer, dataset, prompt, and orchestrator logic.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Ensure imports resolve
_agent_root = Path(__file__).resolve().parent.parent
if str(_agent_root) not in sys.path:
    sys.path.insert(0, str(_agent_root))

_project_root = _agent_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# =====================================================================
# GNN Adjacency tests
# =====================================================================


class TestGNNAdjacency:
    def test_service_names_are_sorted(self):
        from gnn.adjacency import SERVICE_NAMES
        assert SERVICE_NAMES == tuple(sorted(SERVICE_NAMES))

    def test_num_services_matches(self):
        from gnn.adjacency import NUM_SERVICES, SERVICE_NAMES
        assert NUM_SERVICES == len(SERVICE_NAMES)

    def test_num_services_is_correct(self):
        """Should match FirewatchEnv ALL_SERVICES count."""
        import importlib
        import sys
        # Import config directly to avoid openenv dependency in __init__
        env_config_path = str(_project_root / "firewatch_env")
        if env_config_path not in sys.path:
            sys.path.insert(0, env_config_path)
        from config import ALL_SERVICES  # type: ignore[import]
        from gnn.adjacency import NUM_SERVICES
        assert NUM_SERVICES == len(ALL_SERVICES)

    def test_service_to_idx_complete(self):
        from gnn.adjacency import SERVICE_NAMES, SERVICE_TO_IDX
        for i, name in enumerate(SERVICE_NAMES):
            assert SERVICE_TO_IDX[name] == i

    def test_edge_index_shape(self):
        from gnn.adjacency import EDGE_INDEX
        assert EDGE_INDEX.dim() == 2
        assert EDGE_INDEX.shape[0] == 2
        assert EDGE_INDEX.shape[1] > 0  # at least self-loops

    def test_edge_index_has_self_loops(self):
        from gnn.adjacency import EDGE_INDEX, NUM_SERVICES
        src, dst = EDGE_INDEX
        self_loop_count = (src == dst).sum().item()
        assert self_loop_count >= NUM_SERVICES

    def test_edge_index_bidirectional(self):
        """Forward edges should have corresponding reverse edges."""
        from gnn.adjacency import ADJACENCY_DICT, EDGE_INDEX, SERVICE_TO_IDX
        src, dst = EDGE_INDEX
        edges = set(zip(src.tolist(), dst.tolist()))
        # Check at least one forward/reverse pair from the graph
        for svc, deps in ADJACENCY_DICT.items():
            if svc not in SERVICE_TO_IDX:
                continue
            for dep in deps:
                if dep not in SERVICE_TO_IDX:
                    continue
                s, d = SERVICE_TO_IDX[svc], SERVICE_TO_IDX[dep]
                assert (s, d) in edges, f"Missing forward edge {svc} -> {dep}"
                assert (d, s) in edges, f"Missing reverse edge {dep} -> {svc}"

    def test_adjacency_dict_matches_config(self):
        import sys
        env_config_path = str(_project_root / "firewatch_env")
        if env_config_path not in sys.path:
            sys.path.insert(0, env_config_path)
        from config import FULL_DEPENDENCY_GRAPH  # type: ignore[import]
        from gnn.adjacency import ADJACENCY_DICT
        for svc, deps in FULL_DEPENDENCY_GRAPH.items():
            assert ADJACENCY_DICT[svc] == deps


# =====================================================================
# GNN Model tests
# =====================================================================


class TestGNNModel:
    def test_model_instantiation(self):
        from gnn.model import GraphSAGEModel
        model = GraphSAGEModel(in_channels=24, hidden=64, num_classes=43)
        assert model.num_classes == 43
        assert model.hidden_dim == 64

    def test_forward_pass(self):
        from gnn.adjacency import EDGE_INDEX, NUM_SERVICES
        from gnn.model import GraphSAGEModel

        model = GraphSAGEModel(
            in_channels=24, hidden=64, num_classes=NUM_SERVICES
        )
        x = torch.randn(NUM_SERVICES, 24)
        logits, embeddings = model(x, EDGE_INDEX)

        assert logits.shape == (NUM_SERVICES, NUM_SERVICES)
        assert embeddings.shape == (NUM_SERVICES, 64)

    def test_default_num_classes(self):
        from gnn.adjacency import NUM_SERVICES
        from gnn.model import GraphSAGEModel
        model = GraphSAGEModel()
        assert model.num_classes == NUM_SERVICES


# =====================================================================
# GNN Serializer tests
# =====================================================================


class TestGNNSerializer:
    def test_blurb_format(self):
        from gnn.adjacency import NUM_SERVICES, SERVICE_NAMES
        from gnn.serializer import serialize_blurb

        logits = torch.randn(NUM_SERVICES)
        blurb = serialize_blurb(logits, SERVICE_NAMES)

        assert "[Graph analysis]" in blurb
        assert "Root-cause probabilities:" in blurb
        assert "Top-3 suspect services" in blurb
        assert "Downstream blast-radius" in blurb

    def test_blurb_default_service_names(self):
        from gnn.adjacency import NUM_SERVICES
        from gnn.serializer import serialize_blurb

        logits = torch.randn(NUM_SERVICES)
        blurb = serialize_blurb(logits)
        assert "[Graph analysis]" in blurb

    def test_blurb_probabilities_sum_to_one(self):
        from gnn.adjacency import NUM_SERVICES
        from gnn.serializer import serialize_blurb

        logits = torch.randn(NUM_SERVICES)
        blurb = serialize_blurb(logits)

        # Extract probabilities from blurb
        prob_line = [l for l in blurb.split("\n") if "Root-cause" in l][0]
        probs = []
        for part in prob_line.split(": ", 1)[1].split(", "):
            val = float(part.split("=")[1])
            probs.append(val)
        total = sum(probs)
        assert abs(total - 1.0) < 0.05  # softmax sums to ~1.0


# =====================================================================
# GNN Training tests
# =====================================================================


class TestGNNTraining:
    def _make_example(self, fault_service: str = "auth-service") -> dict:
        return {
            "example_id": "test-001",
            "task_seed_id": "task_easy",
            "tier": "easy",
            "fault_type": "oom",
            "fault_service": fault_service,
            "variation_strategy": "baseline",
            "observation": {
                "tick": 1,
                "budget": 30.0,
                "alerts": ["OOMKilled on auth-service"],
                "service_metrics": {
                    "auth-service": {
                        "status": "down",
                        "http_server_error_rate": 0.95,
                        "http_server_request_duration_p99": 5.0,
                        "process_memory_utilization": 0.99,
                    },
                    "api-gateway": {
                        "status": "degraded",
                        "http_server_error_rate": 0.15,
                    },
                    "db-proxy": {
                        "status": "healthy",
                        "http_server_error_rate": 0.01,
                    },
                },
                "logs": {"auth-service": ["OOMKilled"]},
            },
            "gold_action_sequence": [
                {"action": "fetch_logs", "params": {"service": "auth-service"}},
                {"action": "scale_replicas", "params": {"service": "auth-service"}},
            ],
            "gold_alternatives": [],
            "expected_score_range": {"min": 0.7, "max": 1.0},
            "suboptimal_paths": [],
        }

    def test_extract_node_features(self):
        from gnn.train_gnn import extract_node_features, NUM_FEATURES
        from gnn.adjacency import NUM_SERVICES

        example = self._make_example()
        features = extract_node_features(example)
        assert features.shape == (NUM_SERVICES, NUM_FEATURES)

    def test_get_root_cause_label(self):
        from gnn.train_gnn import get_root_cause_label
        from gnn.adjacency import SERVICE_TO_IDX

        example = self._make_example("auth-service")
        label = get_root_cause_label(example)
        assert label == SERVICE_TO_IDX["auth-service"]

    def test_welford_normalizer(self):
        from gnn.train_gnn import WelfordNormalizer

        norm = WelfordNormalizer(3)
        norm.update([1.0, 2.0, 3.0])
        norm.update([3.0, 4.0, 5.0])

        assert norm.n == 2
        assert abs(norm.mean[0] - 2.0) < 1e-6
        assert abs(norm.mean[1] - 3.0) < 1e-6

    def test_welford_roundtrip(self):
        from gnn.train_gnn import WelfordNormalizer

        norm = WelfordNormalizer(3)
        norm.update([1.0, 2.0, 3.0])
        norm.update([3.0, 4.0, 5.0])

        d = norm.to_dict()
        norm2 = WelfordNormalizer.from_dict(d)
        assert norm2.n == 2
        assert norm2.mean == norm.mean

    def test_train_gnn_smoke(self):
        """Smoke test: GNN training on small batch (CPU)."""
        from gnn.train_gnn import train_gnn

        examples = [self._make_example(f"auth-service") for _ in range(10)]
        for i, ex in enumerate(examples):
            ex["example_id"] = f"test-{i:03d}"

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path, norm_path = train_gnn(
                batch_examples=examples,
                batch_num=0,
                checkpoint_dir=Path(tmpdir),
                config={"max_epochs": 2, "patience": 2},
            )
            assert ckpt_path.exists()
            assert norm_path.exists()
            assert ckpt_path.suffix == ".pt"

            # Verify normalization stats are valid JSON
            norm_data = json.loads(norm_path.read_text())
            assert "mean" in norm_data
            assert "n" in norm_data


# =====================================================================
# SFT Dataset tests
# =====================================================================


class TestSFTDataset:
    def _make_valid_example(self) -> dict:
        return {
            "example_id": "test-001",
            "task_seed_id": "task_easy",
            "tier": "easy",
            "fault_type": "oom",
            "variation_strategy": "baseline",
            "observation": {
                "tick": 1,
                "budget": 30.0,
                "alerts": ["OOMKilled"],
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

    def test_load_batch(self):
        from sft.dataset import load_batch

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(50):
                ex = self._make_valid_example()
                ex["example_id"] = f"test-{i:03d}"
                f.write(json.dumps(ex) + "\n")
            tmpfile = f.name

        try:
            examples = load_batch(Path(tmpfile))
            assert len(examples) == 50
        finally:
            os.unlink(tmpfile)

    def test_load_batch_file_not_found(self):
        from sft.dataset import load_batch
        with pytest.raises(FileNotFoundError):
            load_batch(Path("/nonexistent/batch.jsonl"))

    def test_split_batch_deterministic(self):
        from sft.dataset import split_batch

        examples = [self._make_valid_example() for _ in range(50)]
        for i, ex in enumerate(examples):
            ex["example_id"] = f"test-{i:03d}"

        train1, val1 = split_batch(examples, batch_num=0)
        train2, val2 = split_batch(examples, batch_num=0)

        assert len(train1) == len(train2)
        assert len(val1) == len(val2)
        # Same seed = same split
        assert [e["example_id"] for e in train1] == [e["example_id"] for e in train2]

    def test_split_batch_ratio(self):
        from sft.dataset import split_batch

        examples = [self._make_valid_example() for _ in range(50)]
        train, val = split_batch(examples, batch_num=0)
        assert len(train) == 40
        assert len(val) == 10


# =====================================================================
# SFT Prompt tests
# =====================================================================


class TestSFTPrompt:
    def test_format_sft_prompt(self):
        from sft.prompt import format_sft_prompt

        example = {
            "observation": {
                "tick": 1,
                "budget": 30.0,
                "alerts": [{"severity": "critical", "alertname": "HighErrorRate",
                           "description": "error_rate is 0.95"}],
                "service_metrics": {
                    "auth-service": {
                        "status": "down",
                        "http_server_error_rate": 0.95,
                        "http_server_request_duration_p99": 5.0,
                        "process_memory_utilization": 0.99,
                        "process_cpu_utilization": 0.8,
                    },
                },
                "logs": {"auth-service": ["OOMKilled"]},
            },
            "gold_action_sequence": [
                {"action": "fetch_logs", "params": {"service": "auth-service"}},
            ],
        }

        prompt = format_sft_prompt(example)
        assert "system" in prompt
        assert "user" in prompt
        assert "assistant" in prompt
        assert "Site Reliability Engineer" in prompt["system"]
        assert "auth-service" in prompt["user"]

    def test_format_with_gnn_blurb(self):
        from sft.prompt import format_sft_prompt

        example = {
            "observation": {
                "tick": 1,
                "budget": 30.0,
                "alerts": [],
                "service_metrics": {},
                "logs": {},
            },
            "gold_action_sequence": [{"action": "declare_resolved", "params": {}}],
        }

        blurb = "[Graph analysis]\nTop suspect: auth-service (0.81)"
        prompt = format_sft_prompt(example, gnn_blurb=blurb)
        assert "Graph analysis" in prompt["user"]

    def test_format_chat_messages(self):
        from sft.prompt import format_chat_messages

        prompt = {
            "system": "You are an SRE.",
            "user": "Fix this.",
            "assistant": "OK.",
        }
        messages = format_chat_messages(prompt)
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"


# =====================================================================
# VRAM handoff test
# =====================================================================


class TestVRAMHandoff:
    def test_handoff_runs_without_error(self):
        """VRAM handoff should not raise even without CUDA."""
        from sft.train import vram_handoff
        from gnn.model import GraphSAGEModel

        model = GraphSAGEModel(in_channels=24, hidden=64, num_classes=43)
        vram_handoff(model)  # should not raise

    def test_handoff_none_model(self):
        from sft.train import vram_handoff
        vram_handoff(None)  # should not raise


# =====================================================================
# Config tests
# =====================================================================


class TestConfig:
    def test_config_yaml_loads(self):
        import yaml
        config_path = _agent_root / "config.yaml"
        assert config_path.exists(), "config.yaml not found"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "sft" in config
        assert "gnn" in config
        assert config["sft"]["lora_rank"] == 16
        assert config["gnn"]["hidden_dim"] == 64

    def test_load_config_function(self):
        from sft.train import load_config
        config = load_config()
        assert "sft" in config
        assert "gnn" in config


# =====================================================================
# SFT preflight tests
# =====================================================================


class TestSFTPreflight:
    def _write_reviewed_batch(self, tmp_path: Path, valid: bool = True) -> Path:
        example = TestSFTDataset()._make_valid_example()
        example["example_id"] = "example-001"
        example["source_script"] = "gen_01_mixed_metric_a.py"
        example["fault_service"] = "auth-service"
        if not valid:
            del example["example_id"]
        batch_path = tmp_path / "batch_000.jsonl"
        batch_path.write_text("\n".join(json.dumps(example) for _ in range(50)) + "\n")
        return batch_path

    def _patch_success_dependencies(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from sft import preflight

        batch_path = self._write_reviewed_batch(tmp_path)

        class FakeApi:
            def repo_info(self, repo_id: str, repo_type: str):
                return {"id": repo_id, "type": repo_type}

        monkeypatch.setattr(preflight, "load_config", lambda config_path=None: {"hf_namespace": "testns"})
        monkeypatch.setattr(preflight.hf_auth, "get_username", lambda: "hf-user")
        monkeypatch.setattr(preflight.hf_auth, "get_token", lambda: "hf-token")
        monkeypatch.setattr(preflight, "HfApi", lambda token=None: FakeApi())
        monkeypatch.setattr(preflight, "detect_current_batch", lambda namespace: 0)
        monkeypatch.setattr(preflight.hf_io, "pull_reviewed_batch", lambda batch_num, local_dir: batch_path)
        monkeypatch.setattr(preflight, "try_import_unsloth", lambda: (object(), None))
        monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(preflight.torch.cuda, "get_device_name", lambda index=0: "T4")
        monkeypatch.setattr(preflight, "verify_disk_space", lambda threshold_gb=20.0: (True, 42.0))

    def test_preflight_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        from sft import preflight

        self._patch_success_dependencies(monkeypatch, tmp_path)

        result = preflight.run_preflight()

        assert result.ok is True
        assert result.batch_num == 0
        assert result.namespace == "testns"
        assert result.errors == []

    def test_preflight_missing_required_repo_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        from sft import preflight

        self._patch_success_dependencies(monkeypatch, tmp_path)

        class FakeApi:
            def repo_info(self, repo_id: str, repo_type: str):
                if repo_id.endswith("firewatch-agent-sft"):
                    raise RuntimeError("404")
                return {"id": repo_id, "type": repo_type}

        monkeypatch.setattr(preflight, "HfApi", lambda token=None: FakeApi())

        result = preflight.run_preflight()

        assert result.ok is False
        assert any("firewatch-agent-sft" in err for err in result.errors)

    def test_preflight_missing_grpo_repo_warns_only(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        from sft import preflight

        self._patch_success_dependencies(monkeypatch, tmp_path)

        class FakeApi:
            def repo_info(self, repo_id: str, repo_type: str):
                if repo_id.endswith("firewatch-agent-grpo"):
                    raise RuntimeError("404")
                return {"id": repo_id, "type": repo_type}

        monkeypatch.setattr(preflight, "HfApi", lambda token=None: FakeApi())

        result = preflight.run_preflight()

        assert result.ok is True
        assert any("firewatch-agent-grpo" in warning for warning in result.warnings)

    def test_preflight_missing_cuda_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        from sft import preflight

        self._patch_success_dependencies(monkeypatch, tmp_path)
        monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: False)

        result = preflight.run_preflight()

        assert result.ok is False
        assert any("CUDA" in err for err in result.errors)

    def test_preflight_bad_reviewed_batch_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        from sft import preflight

        self._patch_success_dependencies(monkeypatch, tmp_path)
        bad_batch_path = self._write_reviewed_batch(tmp_path, valid=False)
        monkeypatch.setattr(preflight.hf_io, "pull_reviewed_batch", lambda batch_num, local_dir: bad_batch_path)

        result = preflight.run_preflight()

        assert result.ok is False
        assert any("example_id" in err for err in result.errors)

    def test_preflight_unsloth_failure_warns_when_fallback_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        from sft import preflight

        self._patch_success_dependencies(monkeypatch, tmp_path)
        monkeypatch.setattr(
            preflight,
            "load_config",
            lambda config_path=None: {
                "hf_namespace": "testns",
                "sft": {"fallback_base_model": "Qwen/Qwen2.5-3B-Instruct"},
            },
        )
        monkeypatch.setattr(
            preflight,
            "try_import_unsloth",
            lambda: (None, "ModuleNotFoundError: torchvision"),
        )

        result = preflight.run_preflight()

        assert result.ok is True
        assert any("fallback" in warning.lower() for warning in result.warnings)


# =====================================================================
# Model runtime selection tests
# =====================================================================


class TestModelRuntimeSelection:
    def test_training_falls_back_to_dense_model_and_optimizer(self):
        from shared.model_runtime import (
            resolve_base_model_for_training,
            resolve_optimizer_for_runtime,
        )

        sft_config = {
            "base_model": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
            "fallback_base_model": "Qwen/Qwen2.5-3B-Instruct",
            "optimizer": "adamw_8bit",
            "fallback_optimizer": "adamw_torch",
        }

        assert resolve_base_model_for_training(
            sft_config,
            use_low_bit_runtime=False,
            prev_lora_path=None,
        ) == "Qwen/Qwen2.5-3B-Instruct"
        assert resolve_optimizer_for_runtime(
            sft_config,
            use_low_bit_runtime=False,
        ) == "adamw_torch"

    def test_training_refuses_low_bit_previous_adapter_without_unsloth(self, tmp_path: Path):
        from shared.model_runtime import resolve_base_model_for_training

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        }))

        with pytest.raises(RuntimeError, match="Previous LoRA adapter requires the low-bit base model"):
            resolve_base_model_for_training(
                {"fallback_base_model": "Qwen/Qwen2.5-3B-Instruct"},
                use_low_bit_runtime=False,
                prev_lora_path=adapter_dir,
            )

    def test_inference_uses_adapter_recorded_dense_base_model(self, tmp_path: Path):
        from shared.model_runtime import resolve_base_model_for_inference

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        }))

        assert resolve_base_model_for_inference(
            {"base_model": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"},
            use_low_bit_runtime=False,
            lora_path=adapter_dir,
        ) == "Qwen/Qwen2.5-3B-Instruct"


# =====================================================================
# TRL / SFTConfig API compatibility
# =====================================================================


class TestTRLSequenceKwargs:
    def test_trl_sft_sequence_kwargs_matches_installed_trl(self):
        from sft.train import trl_sft_sequence_kwargs

        kw = trl_sft_sequence_kwargs(1536)
        assert len(kw) == 1
        key = next(iter(kw))
        assert key in ("max_length", "max_seq_length")
        assert kw[key] == 1536


# =====================================================================
# Cloud notebook launcher tests
# =====================================================================


class TestTrainingNotebooks:
    def test_colab_and_kaggle_notebooks_call_preflight_then_train(self):
        notebook_dir = _agent_root / "notebooks"
        for filename in ("firewatch_sft_colab.ipynb", "firewatch_sft_kaggle.ipynb"):
            notebook_path = notebook_dir / filename
            assert notebook_path.exists(), f"missing notebook: {filename}"
            content = notebook_path.read_text()
            assert ".venv/bin/python -m sft.preflight --config config.yaml" in content
            assert ".venv/bin/python -m sft.train --config config.yaml" in content
            assert content.index("sft.preflight") < content.index("sft.train")
            assert "unsloth" in content.lower()
            assert "pip install -q virtualenv" in content
            assert "python -m virtualenv .venv" in content
            assert ".venv/bin/python -m pip install" in content
            assert "--no-deps -e ." in content
            assert "torchvision==0.25.0" in content
