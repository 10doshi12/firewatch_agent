"""
firewatch_agent — FirewatchEnv Three-Node Training Pipeline

Top-level package. Sub-packages:
  shared/    — Platform detection, HF auth, HF I/O (Module 1 / SPEC-T1)
  data_gen/  — Dataset generation, review, upload (Module 0 / SPEC-T0)
  gnn/       — GraphSAGE model, adjacency, training (Module 2 / SPEC-T2)
  sft/       — SFT dataset, prompt, training orchestrator (Module 2 / SPEC-T2)
  grpo/      — GRPO RL loop against live sim (Module 3 / SPEC-T3)
  eval/      — Baseline evaluation against sim (Module 4 / SPEC-T4)
"""
