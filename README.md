# firewatch_agent 🔥

> **The agent-side training and inference pipeline for FirewatchEnv — LLM + GNN + SFT + GRPO, zero env physics.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Unsloth](https://img.shields.io/badge/Unsloth-required%20for%20GPU-purple)](https://github.com/unslothai/unsloth)

*Sibling repo to [`firewatch_env`](https://github.com/10doshi12/firewatch_env). Meta PyTorch OpenEnv Hackathon India 2026.*

---

## What Is This?

`firewatch_agent` owns everything on the **agent side** of the FirewatchEnv RL loop. It knows nothing about the environment's internal physics — it only talks to the env over HTTP. `firewatch_env` knows nothing about the agent.

This repo provides:
- A **production-grade local baseline** runner (LLM + GNN + honest prompt), with no reward leakage
- A **GNN trainer** (GraphSAGE) for root-cause ranking from the service dependency graph
- A **Supervised Fine-Tuning (SFT)** pipeline for the LLM policy, using Unsloth 4-bit LoRA
- A **GRPO RL fine-tuning** pipeline against the live environment
- A **locked-checkpoint baseline evaluator** for regression detection after every training run
- An **offline analysis** module for generating reports and plots from run trajectories

---

## The Core Separation: Two Completely Different Programs

> Inference must **never** see rewards. Training must see rewards — they are the loss signal.

```
┌─────────────────────────────────────┐     ┌────────────────────────────────────────┐
│     INFERENCE  (no rewards)         │     │     TRAINING  (rewards required)       │
│                                     │     │                                        │
│  runners/inference.py               │     │  data_gen/ → sft/train.py              │
│  runners/honest_prompt.py           │     │           → grpo/train.py              │
│  runners/policy.py  (LLM+GNN)      │     │           → eval/baseline.py           │
│  runners/trajectory.py (JSONL log)  │     │                                        │
│                                     │     │  GPU required (Unsloth, CUDA)          │
│  CPU-only  ·  no Unsloth            │     │  CPU for data gen + review             │
└─────────────────────────────────────┘     └────────────────────────────────────────┘
         │  HTTP  │                                          │  HTTP  │
         └────────┴──────────────► firewatch_env server ◄───┴────────┘
```

Both programs share only the GNN checkpoint and (optionally) the SFT LoRA — nothing else crosses the boundary.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Setup](#2-setup)
3. [Inference — Baseline Runner](#3-inference--baseline-runner)
4. [Honesty Contract](#4-honesty-contract)
5. [Output Format](#5-output-format)
6. [GNN Module](#6-gnn-module)
7. [Training Pipeline Overview](#7-training-pipeline-overview)
8. [Data Generation](#8-data-generation)
9. [SFT Training](#9-sft-training)
10. [GRPO RL Fine-Tuning](#10-grpo-rl-fine-tuning)
11. [Baseline Evaluation](#11-baseline-evaluation)
12. [Offline Analysis](#12-offline-analysis)
13. [HuggingFace Artifact Layout](#13-huggingface-artifact-layout)
14. [Configuration Reference](#14-configuration-reference)
15. [Dependencies](#15-dependencies)
16. [Tests](#16-tests)
17. [Where to Run What](#17-where-to-run-what)

---

## 1. Repository Structure

```
firewatch_agent/
│
├── runners/                     # Production-side baseline (no rewards, no Unsloth)
│   ├── inference.py             # CLI entry point — run all 34 tasks or a smoke test
│   ├── honest_prompt.py         # Leakage-proof system + user prompt builder
│   ├── policy.py                # FirewatchPolicy: LLM + GNN + deterministic fallback
│   ├── llm_client.py            # Backend dispatch: OpenAI / OpenRouter / Ollama / echo
│   ├── http_sim_client.py       # Local HTTP client to firewatch_env (reset + step)
│   ├── gnn_baseline.py          # GNN mode: heuristic | untrained | from_checkpoint
│   └── trajectory.py            # Per-step JSONL logger for offline analysis + SFT harvest
│
├── data_gen/                    # SFT data generation — Module 0 / SPEC-T0
│   ├── run_generator.py         # Dispatches generator scripts by --script or --batch
│   ├── check_batch.py           # Compliance checks (raw and reviewed stages)
│   ├── review.py                # Human review interface
│   └── upload.py                # Uploads reviewed batch to HF dataset repo
│
├── data_gen_scripts/            # 30 generator scripts (gen_01_*.py … gen_30_*.py)
│                                # Each produces 50 examples → batch_NNN.jsonl
│
├── gnn/                         # GraphSAGE model + trainer — Module 3 / SPEC-T3
│   ├── model.py                 # GraphSAGE (2 layers, 64 hidden dim)
│   ├── trainer.py               # Training loop: max 250 epochs, patience 10
│   └── featurizer.py            # ServiceMetrics → 32-dim node feature vector
│
├── sft/                         # SFT trainer — Module 2 / SPEC-T2
│   ├── preflight.py             # Pre-flight check: HF auth, Unsloth, CUDA, disk, batches
│   └── train.py                 # Incremental LoRA fine-tuning (GNN-then-LLM)
│
├── grpo/                        # GRPO RL trainer — SPEC-T3
│   ├── train.py                 # GRPO loop: live env reward against locked SFT LoRA
│   └── sim_client.py            # GRPO-side env client (reset + step + reward read)
│
├── eval/                        # Locked-checkpoint evaluation — SPEC-T4
│   ├── baseline.py              # Runs locked checkpoints; compares to Hub metrics
│   └── regression_guard.py      # Raises if mean_reward or success_rate regresses
│
├── shared/                      # Shared utilities — SPEC-T1
│   ├── hf_auth.py               # HF token / namespace resolution
│   ├── io.py                    # File I/O helpers (JSONL, JSON, YAML)
│   └── model_runtime.py         # Unsloth / dense model loading helpers
│
├── analysis/                    # Offline analysis — static PNGs + Markdown report
│   └── analyze.py               # Reads runs/, SFT data, GRPO metrics; emits report.md
│
├── notebooks/                   # Colab/Kaggle launchers for real GPU SFT runs
├── tests/                       # Pytest suite (leakage guards, parser, schema, backends)
├── runs/                        # Per-run trajectory artefacts (gitignored)
│
├── config.yaml                  # All hyperparameters: SFT + GNN + GRPO
├── pyproject.toml               # Project metadata and dependencies
└── __init__.py
```

---

## 2. Setup

### Prerequisites

- Python 3.11+
- `uv` package manager
- A running `firewatch_env` server (see [firewatch_env](https://github.com/10doshi12/firewatch_env))
- For GPU training: CUDA environment + Unsloth (see §9)

### Install

```bash
cd firewatch_agent
uv sync                         # installs all dependencies incl. dev extras
uv pip install -e ".[fast]"     # optional: hf-transfer for faster Hub uploads
```

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | Yes (training) | HuggingFace API token |
| `HF_NAMESPACE` | No | Override Hub namespace (default: authenticated username) |
| `SPACE_URL` | No | Force a specific firewatch_env HF Space URL |
| `MAX_SFT_STEPS` | No | Steps per SFT process invocation (default: 1) |
| `SKIP_AUTO_BASELINE=1` | No | Skip post-SFT eval for speed |
| `SFT_APPLY_REGRESSION_OVERRIDE=1` | No | Apply `sft_regression_override.yaml` LR hint after regression |
| `GRPO_METRICS_PATH` | No | Override local GRPO metrics path |
| `GRPO_DATASET_SYNC_EVERY` | No | Override Hub sync interval for GRPO metrics |
| `FIREWATCH_SIM_URL` | No | Override sim_env_url for GRPO |

---

## 3. Inference — Baseline Runner

The inference runner talks to the env over HTTP and produces structured trajectory output. It runs without a GPU and without Unsloth.

### Start the Environment Server

```bash
# In one terminal, from the firewatch_env directory:
uv run server --host 0.0.0.0 --port 8000
```

### Run Inference

```bash
# Smoke test — one easy + one medium + one hard task
uv run python -m runners.inference --test-run

# Full evaluation — all 34 registered tasks
uv run python -m runners.inference

# Specific backend and model
uv run python -m runners.inference --backend ollama --model qwen2.5:14b-instruct

# GNN: untrained GraphSAGE baseline (ablation)
uv run python -m runners.inference --gnn untrained

# GNN: from a trained checkpoint
uv run python -m runners.inference --gnn from_checkpoint \
    --gnn-ckpt ../firewatch_agent_checkpoints/gnn/batch_010.pt \
    --gnn-norm ../firewatch_agent_checkpoints/gnn/normalization.json

# Ablation: put rewards in the prompt history (measures leakage effect)
uv run python -m runners.inference --inform-agent
```

### LLM Backends

`runners/llm_client.py` dispatches to four backends:

| Backend | Flag | Notes |
|---------|------|-------|
| OpenAI / OpenRouter | `--backend openai` (default) | Requires `HF_TOKEN` or `OPENAI_API_KEY` |
| Ollama | `--backend ollama` | Local model; no API key required |
| HuggingFace router | `--backend hf` | Uses HF inference router |
| Echo (dry run) | `--backend echo` | Returns fixed action JSON; for testing |

---

## 4. Honesty Contract

Enforced by `runners/honest_prompt.py` and **pinned by `tests/test_runner_honest_prompt.py`** — the tests will fail if any of these constraints are violated by a future code change.

Four leakage vectors that were removed from the legacy `firewatch_env/inference.py`:

| # | Removed Leakage | What It Did | Why It Inflated Scores |
|---|----------------|-------------|----------------------|
| 1 | Fault → remediation cheat sheet | Mapped fault type directly to correct action | Bypassed RCA entirely |
| 2 | Oracle `_recovery_hint` | Emitted "you MUST call declare_resolved NOW" | Solved agent's terminal decision |
| 3 | Fault-typed action menu | Phase-2 metric presence leaked fault category | LLM inferred fault from menu shape |
| 4 | Low `SUCCESS_SCORE_THRESHOLD` | Was 0.1 instead of 0.5 | Counted near-zero episodes as wins |

**What the honest prompt contains:**
- Active service telemetry (top 4 services by error rate)
- Dependency graph (compact, active services only)
- Ranked root-cause candidates from the GNN/heuristic (labelled as hints, not ground truth)
- Generic remediation vocabulary (same set regardless of Phase-2 metrics present)
- Fetched logs (if `fetch_logs` was called)
- Active alerts (top 4)
- Last 5 action history entries
- Neutral telemetry summary (current max error rate, degraded service count)

**What the prompt never contains:**
- Rewards or episode scores
- "Correct path" or task description text
- "You must call declare_resolved" or any imperative oracle
- Fault-type-specific action hints derived from observation fields

---

## 5. Output Format

### Standard Output (evaluator-compatible)

Same `[START] / [STEP] / [END]` format as the legacy runner — parseable by the OpenEnv evaluator:

```
[START] task=task_easy_oom_baseline env=firewatch-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=fetch_logs:auth-service done=false error=null
[STEP] step=2 action=scale_replicas:auth-service done=false error=null
[STEP] step=3 action=declare_resolved done=true error=null
[END] success=true steps=3
```

Set `INFERENCE_REPORT_REWARDS=1` to include `reward=` and `score=` in the STEP and END lines (off by default — leakage guard).

### Trajectory Files (`runs/<run-id>/`)

Every run writes structured artefacts to `runs/<run-id>/`:

| File | Content |
|------|---------|
| `metadata.json` | Run ID, model, backend, GNN mode, start time, task list |
| `steps.jsonl` | One JSON object per step: `{prompt, raw_response, action, reward, source, task_id, step}` |
| `episodes.jsonl` | One JSON object per episode: `{task_id, score, steps, success, wrong_actions, mttm}` |

The `steps.jsonl` files are a valid SFT data source. They contain `(prompt, raw_response, action, reward, source)` tuples that filter trivially into `(prompt, gold_action)` pairs for behavioural cloning, bootstrapping SFT data without hand-writing examples.

---

## 6. GNN Module

The GNN (`gnn/`) provides a root-cause ranking signal from the service dependency graph. It runs at inference time before the LLM call, providing candidates labelled by confidence score.

### Architecture

- **Model:** GraphSAGE (2 layers)
- **Hidden dimension:** 64
- **Input features:** 32-dimensional node vectors per service
  - 21 `ServiceMetrics` fields
  - 3 status one-hot encodings (`healthy`, `degraded/critical`, `down`)
  - 8 Phase 2/3 task-scoped metric fields
- **Dropout:** 0.1
- **Output:** Root-cause probability per node

### Training

```python
gnn:
  hidden_dim: 64
  num_layers: 2
  dropout: 0.1
  learning_rate: 1.0e-3
  max_epochs: 250
  patience: 10          # early stopping
  in_channels: 32
```

### GNN Modes at Inference

| Mode | Flag | Description |
|------|------|-------------|
| `heuristic` | (default) | Dependency-aware scoring: error rate + downstream blast radius |
| `untrained` | `--gnn untrained` | Random-initialized GraphSAGE — ablation baseline |
| `from_checkpoint` | `--gnn from_checkpoint --gnn-ckpt <path>` | Trained GNN from Hub checkpoint |

The GNN is a **hint**, not a controller — the LLM may disagree and act on different evidence. The prompt explicitly labels candidates as hints.

---

## 7. Training Pipeline Overview

```
data_gen_scripts/                    30 generator scripts
      ↓  (uv run python -m data_gen.run_generator)
data/raw/batch_NNN.jsonl             Raw generated examples
      ↓  (check_batch --stage raw)
      ↓  (review)
data/reviewed/batch_NNN.jsonl        Human-reviewed examples
      ↓  (check_batch --stage reviewed + upload)
HF: firewatch-sft-data/reviewed/     Durable batch store on Hub

      ↓  (sft/preflight → sft/train)          ← GPU (Unsloth required)
HF: firewatch-agent-sft/batch_k/     Incremental LoRA checkpoint
HF: firewatch-gnn/gnn/batch_k.pt     GNN checkpoint (co-trained)

      ↓  (grpo/train)                          ← GPU (live env + Unsloth)
HF: firewatch-agent-grpo/latest/     GRPO LoRA

      ↓  (eval/baseline)
HF: firewatch-sft-data/baselines/    metrics.jsonl comparison
```

### Run Order (One Training Cycle)

```bash
# ─── (A) Generate and review SFT data ─────────────────────────────────────────
uv run python -m data_gen.run_generator --batch 0    # generates batch_000.jsonl
uv run python -m data_gen.check_batch --batch 0 --stage raw
uv run python -m data_gen.review --batch 0           # human edits
uv run python -m data_gen.check_batch --batch 0 --stage reviewed
uv run python -m data_gen.upload --batch 0

# ─── (B) SFT — GNN-then-LLM, per SPEC-T2 §4–§11 ─────────────────────────────
uv run python -m sft.preflight --config config.yaml  # hard checks before GPU spend
uv run python -m sft.train --config config.yaml

# ─── (C) GRPO — locked SFT LoRA + live env, per SPEC-T3 §7 ──────────────────
cd ../firewatch_env && uv run server --host 0.0.0.0 --port 8000 &
cd ../firewatch_agent
uv run python -m grpo.train --config config.yaml

# ─── (D) Locked-checkpoint baseline eval ──────────────────────────────────────
uv run python -m eval.baseline --config config.yaml
```

---

## 8. Data Generation

30 generator scripts, each producing **50 examples**, map to 30 batch files:

```
gen_01_*.py  →  batch_000.jsonl
gen_02_*.py  →  batch_001.jsonl
   ...
gen_30_*.py  →  batch_029.jsonl
```

`data_gen.run_generator` accepts either `--script 01` or `--batch 0` and prints the resolved mapping before writing. **Existing raw files are never overwritten.**

### Compliance Checks

`check_batch` validates both stages:

```bash
uv run python -m data_gen.check_batch --batch 0 --stage raw       # before review
uv run python -m data_gen.check_batch --batch 0 --stage reviewed  # after review
```

Checks include JSON schema validation, action type validity, prompt length bounds, and duplicate detection.

### Human Review

```bash
uv run python -m data_gen.review --batch 0
```

Opens each example in a terminal review loop. Reviewers can edit the gold action, discard examples, or approve as-is.

### Upload to Hub

```bash
uv run python -m data_gen.upload --batch 0
```

Uploads `data/reviewed/batch_000.jsonl` to `<namespace>/firewatch-sft-data/reviewed/batch_000.jsonl`.

---

## 9. SFT Training

### Hard Requirements

- **Unsloth is mandatory.** `sft/preflight.py` raises `ImportError` if Unsloth cannot be imported. There is no dense-model fallback.
- CUDA GPU required.
- For CUDA 12.1 + torch 2.5 Ampere / HF Space images:
  ```bash
  pip install "unsloth[cu121-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
  ```

### Pre-Flight Check

Always run before spending GPU time:

```bash
uv run python -m sft.preflight --config config.yaml
```

Checks: HF authentication, required Hub repos exist, reviewed batch discovery, batch compliance, **Unsloth import** (hard stop), CUDA availability, disk space.

### Base Model and LoRA

```yaml
sft:
  base_model: "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"    # 4-bit quantized
  max_seq_length: 1536
  lora_rank: 16
  lora_alpha: 16
  lora_dropout: 0.0
  lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

### Campaign Modes

Two modes select how data files and training steps are paired:

**`paired_15` (default in config.yaml)**
- 15 Hub training runs, indices 0–14
- Each run uses two reviewed data files: run `k` uses `batch_{2k}` and `batch_{2k+1}`
- Final artifacts stored at `firewatch-agent-sft/batch_014/` and `firewatch-agent-sft/latest/`
- Requires all 30 reviewed batches (batch_000–batch_029)

**`legacy_30`**
- 30 training steps, one data file per step (indices 0–29)
- Older scheme; still supported

### Incremental Training

Every run `k > 0` loads the GNN checkpoint and LoRA from run `k-1` on Hub before training. Training is always incremental — no run starts from scratch after the first.

```yaml
sft:
  campaign: paired_15
  max_sft_steps_per_invocation: 1    # set via MAX_SFT_STEPS env for multi-step runs
  llm_epochs_per_batch: 5
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8     # effective batch size = 8
  learning_rate: 2.0e-5
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  max_prompt_length: 1024
  max_completion_length: 256
  optimizer: "adamw_8bit"
```

### Regression Guard

After each SFT batch, `eval.regression_guard` compares new baseline metrics to the previous Hub save:

```yaml
sft:
  regression_guard: true
  regression_min_delta_success_rate: 0.0   # must not decrease
  regression_min_delta_mean_reward: 0.0    # must not decrease
```

Set `SFT_APPLY_REGRESSION_OVERRIDE=1` to apply a manual LR hint from `sft_regression_override.yaml` and continue despite regression.

---

## 10. GRPO RL Fine-Tuning

GRPO (Group Relative Policy Optimization) fine-tunes the SFT LoRA against **live environment rewards**.

### Prerequisites

- The SFT LoRA must already be trained and stored on Hub
- A running `firewatch_env` server (pointed to by `sim_env_url` in config.yaml, or `FIREWATCH_SIM_URL` env var)
- Unsloth + CUDA

### Run

```bash
# Start environment server (same host recommended)
cd ../firewatch_env && uv run server --host 0.0.0.0 --port 8000 &

# Run GRPO training
cd ../firewatch_agent
uv run python -m grpo.train --config config.yaml
```

### GRPO Hyperparameters

```yaml
grpo:
  num_generations: 8            # rollouts per prompt for group reward
  base_seed: 1000
  learning_rate: 1.0e-5         # lower than SFT (RL stability)
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  max_prompt_length: 2048       # longer than SFT (full observation)
  max_completion_length: 256
  max_grad_norm: 0.1            # clip to prevent reward spikes
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  optimizer: "adamw_8bit"
  save_steps: 50
```

### GRPO Metrics

GRPO metrics are appended locally after each reward evaluation at `grpo/metrics.jsonl` and synced periodically to `<namespace>/firewatch-sft-data/grpo/metrics.jsonl` on Hub. Override local path with `GRPO_METRICS_PATH`, adjust upload interval with `GRPO_DATASET_SYNC_EVERY`.

### Artifacts

After training: `<namespace>/firewatch-agent-grpo/latest/` (and checkpoint folders per `save_steps`).

---

## 11. Baseline Evaluation

`eval/baseline.py` runs **60 episodes** against locked checkpoints and compares results to the previous Hub metrics snapshot.

```bash
uv run python -m eval.baseline --config config.yaml
```

```yaml
sft:
  baseline_sim_episodes: 60
  baseline_sim_url: "http://127.0.0.1:8000"   # local sim URL inside HF Space
```

Results are written to `<namespace>/firewatch-sft-data/baselines/metrics.jsonl`. The regression guard runs automatically unless `SKIP_AUTO_BASELINE=1` is set.

---

## 12. Offline Analysis

After any inference, SFT, or GRPO run, generate static PNG plots and a Markdown investigation report:

```bash
uv run python -m analysis.analyze \
  --sft-dir ../sft_data/reviewed \
  --runs-dir runs \
  --grpo-log auto \
  --output-dir analysis_runs/latest
```

Outputs written under `analysis_runs/latest/`:

| File | Content |
|------|---------|
| `report.md` | Narrative summary: score trends, per-task breakdown, GRPO curve |
| `summary.json` | Machine-readable metrics: mean score, success rate, MTTM distribution |
| `plots/*.png` | Static PNG charts: score by task, learning curve, BCM over time |

---

## 13. HuggingFace Artifact Layout

All durable training state lives in HF Hub repos. The Hub namespace resolves as:
`config.hf_namespace` → `HF_NAMESPACE` env var → authenticated username.

| Repo | Type | Contents |
|------|------|---------|
| `<ns>/firewatch-sft-data` | Dataset | `reviewed/batch_NNN.jsonl` (0–029), `baselines/metrics.jsonl`, `grpo/metrics.jsonl` |
| `<ns>/firewatch-agent-sft` | Model | `batch_NNN/` LoRA adapters (000–014 for `paired_15`, or 000–029 for `legacy_30`), `latest/` |
| `<ns>/firewatch-gnn` | Model | `gnn/batch_NNN.pt` checkpoints, `gnn/normalization.json` |
| `<ns>/firewatch-agent-grpo` | Model | `latest/` and per-`save_steps` checkpoint folders |

> **Important:** The HF Space (the simulator/app endpoint) is **not** canonical model storage. If the Space needs trained artifacts, it downloads from the model repos, not the other way around.

---

## 14. Configuration Reference

All training hyperparameters are in `config.yaml`. The full file is 75 lines.

### Top-Level

```yaml
hf_namespace: null         # falls back to HF_NAMESPACE env or authenticated username
sim_env_url: "https://10doshi12-firewatch-env.hf.space"
```

### SFT Section

```yaml
sft:
  campaign: paired_15                              # paired_15 | legacy_30
  max_sft_steps_per_invocation: 1
  base_model: "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
  fallback_base_model: "Qwen/Qwen2.5-3B-Instruct" # edge cases only; SFT never uses dense
  max_seq_length: 1536
  lora_rank: 16
  lora_alpha: 16
  lora_dropout: 0.0
  lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  llm_epochs_per_batch: 5
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  max_prompt_length: 1024
  max_completion_length: 256
  optimizer: "adamw_8bit"
  fallback_optimizer: "adamw_torch"
  baseline_sim_episodes: 60
  baseline_sim_url: "http://127.0.0.1:8000"
  regression_guard: true
  regression_min_delta_success_rate: 0.0
  regression_min_delta_mean_reward: 0.0
```

### GNN Section

```yaml
gnn:
  hidden_dim: 64
  num_layers: 2
  dropout: 0.1
  learning_rate: 1.0e-3
  max_epochs: 250
  patience: 10
  in_channels: 32    # 21 ServiceMetrics + 3 status one-hot + 8 Phase 2/3 fields
```

### GRPO Section

```yaml
grpo:
  num_generations: 8
  base_seed: 1000
  learning_rate: 1.0e-5
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  max_prompt_length: 2048
  max_completion_length: 256
  max_grad_norm: 0.1
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  optimizer: "adamw_8bit"
  fallback_optimizer: "adamw_torch"
  save_steps: 50
```

---

## 15. Dependencies

From `pyproject.toml` (Python 3.11+):

| Package | Version | Purpose |
|---------|---------|---------|
| `bitsandbytes` | ≥ 0.44 | 4-bit quantization for SFT/GRPO |
| `datasets` | ≥ 2.14 | HF dataset handling |
| `huggingface-hub` | ≥ 0.24 | Hub upload/download |
| `jsonschema` | ≥ 4.0 | Data compliance checks |
| `peft` | ≥ 0.13 | LoRA adapter management |
| `pyyaml` | ≥ 6.0 | config.yaml loading |
| `torch` | ≥ 2.0 | PyTorch |
| `torch-geometric` | ≥ 2.5 | GraphSAGE (GNN) |
| `transformers` | ≥ 4.56, < 5.0 | LLM loading and tokenization |
| `trl` | ≥ 0.21, < 0.29 | GRPO trainer |
| `websockets` | ≥ 13.0 | WebSocket env communication |
| `matplotlib` | ≥ 3.5 | Analysis plots |
| `python-dotenv` | ≥ 1.0 | .env file support |
| `hf-transfer` | ≥ 0.1.6 (optional) | Fast Hub transfers (`[fast]` extra) |

**Dev extras:**
- `pytest >= 9.0.3`

**Not listed but required for SFT/GRPO:**
- `unsloth` — must be installed separately per your CUDA version (see §9). There is no fallback.

---

## 16. Tests

The test suite includes critical leakage guards — failing these indicates the honesty contract has been broken.

```bash
# Leakage guards (most important — run before any prompt change)
uv run pytest tests/test_runner_honest_prompt.py

# Policy: action parser + LLM fallback behavior
uv run pytest tests/test_runner_policy.py

# Trajectory JSONL schema validation
uv run pytest tests/test_runner_trajectory.py

# Backend dispatch (openai / ollama / hf / echo)
uv run pytest tests/test_runner_llm_client.py

# All tests
uv run pytest tests/ -v
```

### What the Leakage Tests Assert

`tests/test_runner_honest_prompt.py` pins the honesty contract against `runners/honest_prompt.py`. If any of the following words or patterns appear in the generated prompt, the test fails:

- `"reward"` or `"episode_score"` (score leakage)
- `"correct path"` or `"correct action"` (answer leakage)
- `"MUST"` + `"declare_resolved"` together (oracle directive)
- Any fault-type-specific remediation hint in the action menu (fault-type leakage via menu shape)

---

## 17. Where to Run What

| Task | Machine | GPU? | Unsloth? |
|------|---------|------|---------|
| Data generation (`data_gen.run_generator`) | Local | No | No |
| Compliance checks (`data_gen.check_batch`) | Local | No | No |
| Human review (`data_gen.review`) | Local | No | No |
| Upload (`data_gen.upload`) | Local | No | No |
| SFT preflight (`sft.preflight`) | Local | No | Yes (import check) |
| **SFT training** (`sft.train`) | **Colab / Kaggle / GPU Space** | **Yes** | **Yes** |
| **GRPO training** (`grpo.train`) | **Colab / Kaggle / GPU Space** | **Yes** | **Yes** |
| Inference / baseline runner | Local | No | No |
| Locked-checkpoint eval (`eval.baseline`) | GPU Space / Local | Yes (model load) | Yes |
| Offline analysis (`analysis.analyze`) | Local | No | No |
| Unit tests (`pytest tests/`) | Local | No | No |

### Recommended GPU Setup (Colab / Kaggle / HF Space)

The monorepo should include both `firewatch_env/` (sim) and `firewatch_agent/`:

```bash
# 1. Install Unsloth first (version matching your CUDA)
pip install "unsloth[cu121-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"

# 2. Install agent dependencies
cd firewatch_agent && uv sync

# 3. Set HF token
export HF_TOKEN=<your-token>

# 4. Start the sim server on the same host (if doing GRPO)
cd ../firewatch_env && uv run server --host 0.0.0.0 --port 8000 &

# 5. Preflight, then train
cd ../firewatch_agent
uv run python -m sft.preflight --config config.yaml
uv run python -m sft.train --config config.yaml
```

Optional Docker GPU image (if a `docker/hf-train-sft.Dockerfile` exists in the monorepo root):

```bash
docker build -f docker/hf-train-sft.Dockerfile -t firewatch-sft .
```

Override `CMD` to start both the environment server and `sft.train`.

---

## Relation to `firewatch_env`

```
firewatch_env/          ←  environment physics, simulation, HTTP server
firewatch_agent/        ←  agent inference, GNN, SFT, GRPO, eval
         ↕
    HTTP / JSON
```

`firewatch_agent` is a client. It never imports from `firewatch_env` — there is no shared Python package boundary. The only contract between them is the HTTP API (`/reset`, `/step`, `/state`, `/health`) and the JSON schema of `SystemObservation` and `FirewatchAction`.

This decoupling means you can run inference against any compliant OpenEnv server, not just the canonical `firewatch_env`. It also means you can train the GNN and LLM against a faster/slower/modified simulation without touching the agent code.

---

*Meta PyTorch OpenEnv Hackathon India 2026*
