# Firewatch Analysis Report

## Summary
- SFT examples analyzed: 1500
- Inference episodes analyzed: 21
- Inference steps analyzed: 573
- GRPO reward evaluations analyzed: 174
- GRPO mean immediate reward: -0.141
- GRPO positive reward rate: 0.36
- Baseline records analyzed: 0
- GRPO rollout batches: 23
- GRPO mean within-batch reward σ: 1.125
- GRPO batches with σ>0: 87%
- GRPO mean valid-action rate / batch: 81%

## GRPO Action Mix
- `declare_resolved`: 53
- `fetch_logs`: 30
- `trace_dependencies`: 20
- `get_metrics_detail`: 18
- `diagnose`: 4
- `restart_service`: 3
- `check`: 2
- `collect_logs`: 2
- `CAT`: 1
- `DEBUG_TRACE`: 1

## Caveats
- GRPO rewards are single-step evaluations today, not full episode returns.
- Baseline and inference success metrics should not be compared directly to GRPO reward-eval rewards.

## Plots
- [plots/sft_action_distribution.png](plots/sft_action_distribution.png)
- [plots/sft_task_fault_coverage.png](plots/sft_task_fault_coverage.png)
- [plots/inference_success_by_difficulty.png](plots/inference_success_by_difficulty.png)
- [plots/decision_source_mix.png](plots/decision_source_mix.png)
- [plots/grpo_reward_eval.png](plots/grpo_reward_eval.png)
- [plots/grpo_action_distribution.png](plots/grpo_action_distribution.png)
- [plots/grpo_reward_by_action.png](plots/grpo_reward_by_action.png)
- [plots/grpo_batch_learnability.png](plots/grpo_batch_learnability.png)
- [plots/baseline_progression.png](plots/baseline_progression.png)
- [plots/grpo_pre_post_delta.png](plots/grpo_pre_post_delta.png)
