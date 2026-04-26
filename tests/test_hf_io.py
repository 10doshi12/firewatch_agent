from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock


_agent_root = Path(__file__).resolve().parent.parent
if str(_agent_root) not in sys.path:
    sys.path.insert(0, str(_agent_root))


def test_append_and_push_dataset_jsonl_merges_existing_remote_file(monkeypatch, tmp_path):
    from shared import hf_io

    repo_root = tmp_path / "downloaded"
    existing = repo_root / "grpo" / "metrics.jsonl"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"event":"old"}\n')

    local_file = tmp_path / "local" / "metrics.jsonl"
    local_file.parent.mkdir()
    local_file.write_text('{"event":"new"}\n')

    fake_api = MagicMock()
    upload_paths: list[str] = []
    fake_api.upload_file.side_effect = lambda **kwargs: upload_paths.append(kwargs["path_in_repo"])

    monkeypatch.setattr(hf_io, "snapshot_download", lambda **kwargs: str(repo_root))
    monkeypatch.setattr(hf_io, "HfApi", lambda token: fake_api)
    monkeypatch.setattr(hf_io.hf_auth, "get_token", lambda: "hf_test")
    monkeypatch.setattr(hf_io, "retry_with_backoff", lambda fn, *args, **kwargs: fn())

    hf_io.append_and_push_dataset_jsonl(
        repo_id="test-ns/firewatch-sft-data",
        remote_path="grpo/metrics.jsonl",
        local_file=local_file,
        local_dir=tmp_path / "merge",
        commit_message="sync grpo metrics",
    )

    merged = tmp_path / "merge" / "grpo" / "metrics.jsonl"
    assert merged.read_text().splitlines() == ['{"event":"old"}', '{"event":"new"}']
    assert upload_paths == ["grpo/metrics.jsonl"]
