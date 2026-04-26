from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest.mock import MagicMock


_agent_root = Path(__file__).resolve().parent.parent
if str(_agent_root) not in sys.path:
    sys.path.insert(0, str(_agent_root))

_project_root = _agent_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def test_pull_sft_lora_uses_explicit_batch(monkeypatch, tmp_path):
    from grpo import train

    calls: list[tuple[str, str, Path]] = []

    def fake_pull_lora_adapter(repo_id: str, subfolder: str, local_dir: Path) -> Path:
        calls.append((repo_id, subfolder, local_dir))
        return local_dir / subfolder

    monkeypatch.setenv("GRPO_SFT_BATCH", "3")
    monkeypatch.setattr(train.hf_io, "pull_lora_adapter", fake_pull_lora_adapter)

    result = train.pull_sft_lora("test-ns", tmp_path)

    assert result == tmp_path / "batch_003"
    assert calls == [("test-ns/firewatch-agent-sft", "batch_003", tmp_path)]


def test_pull_gnn_latest_uses_explicit_batch(monkeypatch, tmp_path):
    from grpo import train

    calls: list[tuple[int, Path]] = []
    norm_path = tmp_path / "gnn" / "normalization.json"
    norm_path.parent.mkdir()
    norm_path.write_text("{}")

    def fake_pull_gnn_checkpoint(batch_num: int, local_dir: Path) -> Path:
        calls.append((batch_num, local_dir))
        return local_dir / "gnn" / f"batch_{batch_num - 1:03d}.pt"

    monkeypatch.setenv("GRPO_GNN_BATCH", "3")
    monkeypatch.setattr(train.hf_io, "pull_gnn_checkpoint", fake_pull_gnn_checkpoint)

    gnn_path, returned_norm = train.pull_gnn_latest("test-ns", tmp_path)

    assert gnn_path == tmp_path / "gnn" / "batch_003.pt"
    assert returned_norm == norm_path
    assert calls == [(4, tmp_path)]


def test_build_prompt_dataset_test_mode_uses_one_prompt(monkeypatch):
    from grpo import train

    monkeypatch.setenv("GRPO_TEST_RUN", "1")
    monkeypatch.setenv("GRPO_TEST_SEED", "1007")
    monkeypatch.setenv("GRPO_TEST_DIFFICULTY", "medium")

    dataset = train.build_prompt_dataset({"grpo": {"base_seed": 1000, "prompts_per_difficulty": 50}})

    assert dataset == [{"seed": 1007, "difficulty_idx": 1, "prompt_idx": 0}]


def test_apply_grpo_test_overrides_isolates_training_shape(monkeypatch):
    from grpo import train

    config = {
        "grpo": {
            "num_generations": 8,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_prompt_length": 2048,
            "max_completion_length": 256,
            "save_steps": 50,
        }
    }
    original = copy.deepcopy(config)

    train.apply_grpo_test_overrides(config, test_run=True)

    assert config["grpo"]["num_generations"] == 2
    assert config["grpo"]["num_train_epochs"] == 1
    assert config["grpo"]["per_device_train_batch_size"] == 2
    assert config["grpo"]["gradient_accumulation_steps"] == 1
    assert config["grpo"]["max_prompt_length"] == 1024
    assert config["grpo"]["max_completion_length"] == 128
    assert config["grpo"]["max_steps"] == 1
    assert original["grpo"]["num_generations"] == 8


def test_grpo_test_overrides_create_valid_trl_batch_shape(monkeypatch):
    from grpo import train
    from trl import GRPOConfig

    config = {"grpo": {}}
    monkeypatch.setenv("GRPO_TEST_RUN", "1")

    train.apply_grpo_test_overrides(config, test_run=True)
    grpo_config = config["grpo"]

    args = GRPOConfig(
        output_dir="/tmp/firewatch-grpo-test",
        num_generations=grpo_config["num_generations"],
        per_device_train_batch_size=grpo_config["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_config["gradient_accumulation_steps"],
        max_steps=grpo_config["max_steps"],
        report_to="none",
        bf16=False,
        fp16=False,
    )

    assert args.generation_batch_size % args.num_generations == 0


def test_push_grpo_checkpoint_uses_test_prefix(monkeypatch, tmp_path):
    from grpo import train

    uploads: list[str] = []
    fake_api = MagicMock()
    fake_api.upload_folder.side_effect = lambda **kwargs: uploads.append(kwargs["path_in_repo"])

    monkeypatch.setenv("GRPO_TEST_RUN", "1")
    monkeypatch.setattr(train.hf_auth, "get_token", lambda: "hf_test")
    monkeypatch.setattr(train, "HfApi", lambda token: fake_api)
    monkeypatch.setattr(train.hf_io, "retry_with_backoff", lambda fn: fn())

    train._push_grpo_checkpoint("test-ns", step=7, local_dir=tmp_path)

    assert uploads == ["test/checkpoint-7/", "test/latest/"]
    fake_api.create_repo.assert_called_once_with(
        "test-ns/firewatch-agent-grpo",
        repo_type="model",
        exist_ok=True,
        private=False,
    )


def test_grpo_metrics_writer_appends_and_syncs_on_interval(monkeypatch, tmp_path):
    from grpo import train

    sync_calls: list[tuple[str, str, Path]] = []

    def fake_sync(repo_id: str, remote_path: str, local_file: Path, local_dir: Path, commit_message: str) -> None:
        sync_calls.append((repo_id, remote_path, local_file))

    monkeypatch.setattr(train.hf_io, "append_and_push_dataset_jsonl", fake_sync)

    metrics_path = tmp_path / "grpo" / "metrics.jsonl"
    writer = train.GrpoMetricsWriter(
        metrics_path=metrics_path,
        namespace="test-ns",
        sync_every=2,
        sync_enabled=True,
    )

    writer.append({"event": "reward_eval", "reward": 0.5})
    assert sync_calls == []

    writer.append({"event": "reward_eval", "reward": -0.5})
    records = metrics_path.read_text().strip().splitlines()

    assert len(records) == 2
    assert sync_calls == [("test-ns/firewatch-sft-data", "grpo/metrics.jsonl", metrics_path)]


def test_grpo_metrics_writer_sync_final_ignores_upload_failure(monkeypatch, tmp_path):
    from grpo import train

    def fake_sync(*args, **kwargs) -> None:
        raise RuntimeError("temporary hub outage")

    monkeypatch.setattr(train.hf_io, "append_and_push_dataset_jsonl", fake_sync)

    writer = train.GrpoMetricsWriter(
        metrics_path=tmp_path / "metrics.jsonl",
        namespace="test-ns",
        sync_every=1,
        sync_enabled=True,
    )
    writer.append({"event": "grpo_complete", "wall_time_seconds": 1.0})

    writer.sync_final()

    assert (tmp_path / "metrics.jsonl").exists()


def test_run_post_grpo_baseline_uses_in_memory_policy(monkeypatch):
    from grpo import train

    class DummyModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def eval(self) -> None:
            self.calls.append("eval")

        def train(self) -> None:
            self.calls.append("train")

    class DummyEnvClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def disconnect(self) -> None:
            self.calls.append("disconnect")

        def connect(self) -> None:
            self.calls.append("connect")

    model = DummyModel()
    env_client = DummyEnvClient()
    tokenizer = object()
    gnn_model = object()
    normalizer = object()
    baseline_calls: list[dict] = []

    def fake_run_baseline(**kwargs) -> None:
        baseline_calls.append(kwargs)

    monkeypatch.setattr(train, "run_baseline", fake_run_baseline)

    train.run_post_grpo_baseline(
        namespace="test-ns",
        step=10,
        env_client=env_client,
        model=model,
        tokenizer=tokenizer,
        gnn_model=gnn_model,
        normalizer=normalizer,
        config_path=None,
    )

    assert baseline_calls == [{
        "model_variant": "grpo-step-10",
        "trigger": "post_grpo_step_10",
        "auto_triggered": True,
        "model_in_memory": (model, tokenizer, gnn_model, normalizer),
        "config_path": None,
    }]
    assert env_client.calls == ["disconnect", "connect"]
    assert model.calls == ["eval", "train"]


def test_grpo_baseline_interval_uses_env_before_config(monkeypatch):
    from grpo import train

    monkeypatch.setenv("GRPO_BASELINE_EVERY_STEPS", "10")

    assert train._grpo_baseline_every_steps({"grpo": {"baseline_every_steps": 50}}) == 10
