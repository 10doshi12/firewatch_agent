"""
grpo/ — GRPO Reinforcement Learning Loop (SPEC-T3)

Trains the SFT LoRA adapter further via Group Relative Policy Optimization
against the live FirewatchEnv simulator on HuggingFace Spaces.

Modules:
  rollout          — Synchronous rollout: reset → observe → act → step loop
  reward_extractor — Sums per-step rewards into episode-level scalar
  train            — CLI orchestrator: pull state → load models → GRPO train → push
"""
