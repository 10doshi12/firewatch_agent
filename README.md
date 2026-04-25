# firewatch_agent

Sibling to `firewatch_env/`. Owns everything on the **agent side** of the
FirewatchEnv RL loop:

  * production-style local baseline (LLM + GNN + honest prompt)
  * GNN training (Module 3 / SPEC-T3)
  * SFT training of the LLM policy (Module 2 / SPEC-T2)
  * GRPO RL fine-tuning against the live env (SPEC-T3)
  * baseline evaluation against locked checkpoints (SPEC-T4)

`firewatch_env/` knows nothing about the agent. `firewatch_agent/` knows
nothing about the env's internal physics — it only talks to the env over
HTTP / WebSocket.

---

## Two completely separate flows

> **Production / inference** and **training** are two different programs.
> Production never sees rewards. Training does. They share only the GNN
> checkpoint and (optionally) the SFT LoRA.

### 1. Production / baseline inference

```bash
cd firewatch_env
uv run server --host 0.0.0.0 --port 8000   # in one terminal

cd ../firewatch_agent
uv run python -m runners.inference --test-run        # one easy + one medium + one hard
uv run python -m runners.inference                    # all 34 tasks
uv run python -m runners.inference --backend ollama --model qwen2.5:14b-instruct
uv run python -m runners.inference --gnn untrained    # untrained GraphSAGE rather than heuristic
uv run python -m runners.inference --gnn from_checkpoint \
    --gnn-ckpt ../firewatch_agent_checkpoints/gnn/batch_010.pt \
    --gnn-norm ../firewatch_agent_checkpoints/gnn/normalization.json
uv run python -m runners.inference --inform-agent     # ablation: rewards in prompt history
```

Honesty contract enforced by `runners/honest_prompt.py` and pinned by
`tests/test_runner_honest_prompt.py`:

  * No fault → remediation cheat sheet.
  * No oracle "you MUST call declare_resolved NOW" block.
  * Action menu is the **generic** vocabulary regardless of which
    Phase-2 fault-typed metric is in the observation.
  * Reward, episode_score, "correct path", and task descriptions are
    never in the prompt.
  * `SUCCESS_SCORE_THRESHOLD = 0.5`.

Outputs:

  * **stdout** — same `[START] / [STEP] / [END]` lines the legacy
    runner emitted, so the evaluator parses both.
  * **`runs/<run-id>/`** — `metadata.json`, `steps.jsonl`,
    `episodes.jsonl`. Rewards and scores live here for offline plotting
    and SFT data extraction.

### 2. Training pipeline (SFT → GRPO)

```
data_gen / data_gen_scripts        →   SFT data (data/reviewed/batch_NNN.jsonl)
sft/train.py                       →   GNN ckpt + SFT LoRA  (per batch)
grpo/train.py                      →   GRPO LoRA (RL fine-tune)
eval/baseline.py                   →   locked-checkpoint score
```

Training is intentionally **not** a flag on the inference runner. They
are different programs because they have different invariants:

  * inference must not see rewards (leakage).
  * training must see rewards (it is the loss signal).

Run order, per training cycle:

```bash
# (A) Generate / review SFT data ---- once per batch -----------------
uv run python -m data_gen.run_generator --batch 0   # same as --script 01
uv run python -m data_gen.check_batch --batch 0 --stage raw
uv run python -m data_gen.review --batch 0          # human edits
uv run python -m data_gen.check_batch --batch 0 --stage reviewed
uv run python -m data_gen.upload --batch 0

# (B) SFT batch -- GNN-then-LLM, per SPEC-T2 §4-§11 -----------------
uv run python -m sft.preflight --config config.yaml
uv run python -m sft.train --config config.yaml

# (C) GRPO -- locked SFT LoRA + live env reward, per SPEC-T3 §7 -----
cd ../firewatch_env && uv run server --host 0.0.0.0 --port 8000 &
cd ../firewatch_agent
uv run python -m grpo.train --config config.yaml

# (D) Locked-checkpoint baseline eval -------------------------------
uv run python -m eval.baseline --config config.yaml
```

The `runs/<run-id>/steps.jsonl` files written by `runners/inference.py`
are also a valid SFT data source — they are
`(prompt, raw_response, action, reward, source)` tuples that filter
trivially into `(prompt, gold_action)` for behavioural cloning. This
is how you bootstrap SFT data from the production agent without
hand-writing examples.

#### Batch numbering

The training data convention is zero-indexed:

  * `gen_01_*.py` -> `batch_000.jsonl`
  * `gen_02_*.py` -> `batch_001.jsonl`
  * ...
  * `gen_30_*.py` -> `batch_029.jsonl`

`data_gen.run_generator` accepts either `--script 01` or `--batch 0`
and prints the resolved mapping before writing the raw batch. Existing
raw files are never overwritten.

#### Where to run training

Use your local machine for data generation, compliance checks, human
review, upload, and offline tests. These steps do not require a GPU.

Use Kaggle when the free T4 runtime is enough and `HF_TOKEN` is stored in
Kaggle Secrets. Use Colab when A100/T4 availability or interactive
debugging is better. The notebooks in `notebooks/` are thin launchers:
they install `uv` + Unsloth, load `HF_TOKEN`, run `sft.preflight`, then
run `sft.train`. They do not contain a separate training implementation.

Before any real 7B SFT run, `sft.preflight` checks HF auth, required Hub
repos, reviewed batch discovery, batch compliance, Unsloth import, CUDA,
and disk space. It should pass before loading the base model.

#### Hugging Face artifact layout

Canonical durable state lives in Hugging Face Hub repos:

  * Dataset repo: `<namespace>/firewatch-sft-data`
    * `reviewed/batch_NNN.jsonl`
    * optional `baselines/metrics.jsonl`
  * SFT model repo: `<namespace>/firewatch-agent-sft`
    * `batch_NNN/` LoRA adapter files
    * `latest/` only after final SFT batch
  * GNN model repo: `<namespace>/firewatch-gnn`
    * `gnn/batch_NNN.pt`
    * `gnn/normalization.json`
  * GRPO model repo: `<namespace>/firewatch-agent-grpo`
    * `latest/` and checkpoint folders after GRPO

The HF Space is the simulator/app endpoint referenced by `sim_env_url`.
Do not use the Space app repo as canonical model-weight storage. If the
Space needs trained artifacts later, it should download or mount them
from the model repos.

---

## Layout

```
firewatch_agent/
├── runners/                     # Production-side baseline runner (NEW)
│   ├── inference.py             # CLI entry point
│   ├── honest_prompt.py         # leakage-proof system + user prompt
│   ├── llm_client.py            # OpenAI / OpenRouter / Ollama / echo
│   ├── http_sim_client.py       # local HTTP client to firewatch_env
│   ├── gnn_baseline.py          # heuristic | untrained | from_checkpoint
│   ├── policy.py                # FirewatchPolicy (LLM + GNN + fallback)
│   └── trajectory.py            # JSONL logger for offline analysis
│
├── data_gen/, data_gen_scripts/ # SFT data generation (Module 0 / SPEC-T0)
├── notebooks/                    # Colab/Kaggle launchers for real SFT
├── shared/                      # HF auth, IO, platform utils (SPEC-T1)
├── sft/                         # SFT trainer (SPEC-T2)
├── gnn/                         # GraphSAGE model + trainer (SPEC-T3)
├── grpo/                        # GRPO trainer + sim_client (SPEC-T3)
├── eval/                        # Locked-checkpoint baseline (SPEC-T4)
├── tests/                       # Includes new runner tests
├── runs/                        # Per-run trajectory artefacts (gitignored)
├── config.yaml                  # SFT + GNN + GRPO hyperparameters
└── pyproject.toml
```

## Setup

```bash
cd firewatch_agent
uv sync                         # dev-only: pytest, etc.
uv pip install -e ".[fast]"     # extras for HF Hub fast transfer
```

Then either start the inference loop above, or kick off the SFT
training pipeline. The two flows do not require each other: you can run
inference without ever training, and you can train (CPU GNN + SFT)
without ever running inference.

## Tests

```bash
uv run pytest tests/test_runner_honest_prompt.py    # leakage guards
uv run pytest tests/test_runner_policy.py           # parser + fallback
uv run pytest tests/test_runner_trajectory.py       # JSONL schema
uv run pytest tests/test_runner_llm_client.py       # backend dispatch
uv run pytest tests/                                # everything
```
