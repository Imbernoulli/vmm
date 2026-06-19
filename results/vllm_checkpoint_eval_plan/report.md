# vLLM Checkpoint Eval Plan

这个计划把 same-shape checkpoint 候选转成逐个 `vllm serve` 和 `run_vllm_downstream_eval.py` 命令。它不声称已经完成真实性能评测；只有 `serve_status = ready_to_host` 的 checkpoint 才能进入 GPU/vLLM 下游评测。

- Plan status: `waiting_for_checkpoint_materialization`
- Candidate rows: `3`
- Ready to host: `0`
- Missing checkpoints: `2`
- Tasks: `gsm8k,mmlu,safety,humaneval_compile`

## Plan

| order | method | status | checkpoint | port | output |
| ---: | --- | --- | --- | ---: | --- |
| 0 | `moe_route_aware_candidate` | `checkpoint_missing_until_materialized` | `results/checkpoints/moe_route_aware_candidate` | 8100 | `results/vllm_checkpoint_eval/moe_route_aware_candidate` |
| 1 | `moe_bias_calibrated_candidate` | `checkpoint_missing_until_materialized` | `results/checkpoints/moe_bias_calibrated_candidate` | 8101 | `results/vllm_checkpoint_eval/moe_bias_calibrated_candidate` |
| 2 | `toy_moe_expert_weight_candidate` | `not_vllm_loadable_toy_candidate` | `results/checkpoints/toy_moe_expert_weight_candidate` | 8102 | `results/vllm_checkpoint_eval/toy_moe_expert_weight_candidate` |

## Commands

- `results/vllm_checkpoint_eval_plan/serve_and_eval_commands.sh`
- `results/vllm_checkpoint_eval_plan/checkpoint_eval_plan.csv`
- `results/vllm_checkpoint_eval_plan/summary.json`
