# vLLM Checkpoint Eval Plan

这个计划把 same-shape checkpoint 候选转成逐个 `vllm serve` 和 `run_vllm_downstream_eval.py` 命令。它不声称已经完成真实性能评测；只有 `serve_status = ready_to_host` 的 checkpoint 才能进入 GPU/vLLM 下游评测。

- Plan status: `ready_to_host`
- Candidate rows: `4`
- Ready to host: `1`
- Missing checkpoints: `2`
- Tasks: `gsm8k,mmlu,safety,humaneval_compile`

## Plan

| order | method | status | checkpoint | port | output |
| ---: | --- | --- | --- | ---: | --- |
| 0 | `qwen_0_5b_instruct_coder_uniform_average` | `ready_to_host` | `results/checkpoints/qwen_0_5b_instruct_coder_uniform_average` | 8100 | `results/vllm_checkpoint_eval/qwen_0_5b_instruct_coder_uniform_average` |
| 1 | `moe_route_aware_candidate` | `checkpoint_missing_until_materialized` | `results/checkpoints/moe_route_aware_candidate` | 8101 | `results/vllm_checkpoint_eval/moe_route_aware_candidate` |
| 2 | `moe_bias_calibrated_candidate` | `checkpoint_missing_until_materialized` | `results/checkpoints/moe_bias_calibrated_candidate` | 8102 | `results/vllm_checkpoint_eval/moe_bias_calibrated_candidate` |
| 3 | `toy_moe_expert_weight_candidate` | `not_vllm_loadable_toy_candidate` | `results/checkpoints/toy_moe_expert_weight_candidate` | 8103 | `results/vllm_checkpoint_eval/toy_moe_expert_weight_candidate` |

## Commands

- `results/vllm_checkpoint_eval_plan/serve_and_eval_commands.sh`
- `results/vllm_checkpoint_eval_plan/checkpoint_eval_plan.csv`
- `results/vllm_checkpoint_eval_plan/summary.json`
