from __future__ import annotations

import sys
from pathlib import Path


_agent_root = Path(__file__).resolve().parent.parent
if str(_agent_root) not in sys.path:
    sys.path.insert(0, str(_agent_root))

_project_root = _agent_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def test_parse_grpo_step_variant_for_in_memory_baseline():
    from eval import baseline

    parsed = baseline._parse_variant("grpo-step-10")

    assert parsed["type"] == "grpo-step"
    assert parsed["checkpoint_num"] == 10
    assert parsed["use_gnn"] is True
    assert parsed["lora_repo_suffix"] is None
    assert parsed["lora_subfolder"] is None


def test_unpack_in_memory_components_accepts_grpo_policy_tuple():
    from eval import baseline

    model = object()
    tokenizer = object()
    gnn_model = object()
    normalizer = object()

    assert baseline._unpack_in_memory_components(
        (model, tokenizer, gnn_model, normalizer)
    ) == (model, tokenizer, gnn_model, normalizer)
