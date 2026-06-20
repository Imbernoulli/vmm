# vLLM Checkpoint Eval Plan

这个计划把 source baseline 和 same-shape checkpoint 候选转成逐个 `vllm serve` 和 `run_vllm_downstream_eval.py` 命令。它不声称已经完成真实性能评测；只有 `serve_status = ready_to_host` 且目录内存在 safetensors/index 的模型才能进入 GPU/vLLM 下游评测。

- Plan status: `hosted_eval_complete`
- Candidate rows: `14`
- Ready to host: `11`
- Completed evals: `1`
- Missing checkpoints: `2`
- Tasks: `gsm8k,mmlu,safety,humaneval_compile`

## Plan

| order | method | serve status | eval status | avg primary | worst primary | checkpoint | port | output |
| ---: | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| 0 | `source_qwen3_30b_instruct` | `ready_to_host` | `not_run` |  |  | `/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` | 8100 | `results/vllm_checkpoint_eval/source_qwen3_30b_instruct` |
| 1 | `source_qwen3_30b_coder` | `ready_to_host` | `not_run` |  |  | `/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120` | 8101 | `results/vllm_checkpoint_eval/source_qwen3_30b_coder` |
| 2 | `qwen3_moe_unified_route_guarded_candidate` | `ready_to_host` | `not_run` |  |  | `results/checkpoints/qwen3_moe_unified_route_guarded_candidate` | 8102 | `results/vllm_checkpoint_eval/qwen3_moe_unified_route_guarded_candidate` |
| 3 | `qwen3_moe_audit_gated_candidate` | `ready_to_host` | `not_run` |  |  | `results/checkpoints/qwen3_moe_audit_gated_candidate` | 8103 | `results/vllm_checkpoint_eval/qwen3_moe_audit_gated_candidate` |
| 4 | `qwen3_moe_trust_region_candidate` | `ready_to_host` | `not_run` |  |  | `results/checkpoints/qwen3_moe_trust_region_candidate` | 8104 | `results/vllm_checkpoint_eval/qwen3_moe_trust_region_candidate` |
| 5 | `qwen3_moe_expert_only_trust_region_candidate` | `ready_to_host` | `not_run` |  |  | `results/checkpoints/qwen3_moe_expert_only_trust_region_candidate` | 8105 | `results/vllm_checkpoint_eval/qwen3_moe_expert_only_trust_region_candidate` |
| 6 | `qwen3_moe_tail_trimmed_expert_only_candidate` | `ready_to_host` | `not_run` |  |  | `results/checkpoints/qwen3_moe_tail_trimmed_expert_only_candidate` | 8106 | `results/vllm_checkpoint_eval/qwen3_moe_tail_trimmed_expert_only_candidate` |
| 7 | `qwen3_moe_searched_no_gt065_max_retention_candidate` | `ready_to_host` | `not_run` |  |  | `results/checkpoints/qwen3_moe_searched_no_gt065_max_retention_candidate` | 8107 | `results/vllm_checkpoint_eval/qwen3_moe_searched_no_gt065_max_retention_candidate` |
| 8 | `qwen3_moe_layer_chunk_candidate` | `ready_to_host` | `not_run` |  |  | `results/checkpoints/qwen3_moe_layer_chunk_candidate` | 8108 | `results/vllm_checkpoint_eval/qwen3_moe_layer_chunk_candidate` |
| 9 | `qwen3_moe_unified_mechanism_candidate` | `ready_to_host` | `not_run` |  |  | `results/checkpoints/qwen3_moe_unified_mechanism_candidate` | 8109 | `results/vllm_checkpoint_eval/qwen3_moe_unified_mechanism_candidate` |
| 10 | `qwen_0_5b_instruct_coder_uniform_average` | `ready_to_host` | `complete` | 0.180 | 0.000 | `results/checkpoints/qwen_0_5b_instruct_coder_uniform_average` | 8110 | `results/vllm_checkpoint_eval/qwen_0_5b_instruct_coder_uniform_average` |
| 11 | `moe_route_aware_candidate` | `checkpoint_missing_until_materialized` | `not_run` |  |  | `results/checkpoints/moe_route_aware_candidate` | 8111 | `results/vllm_checkpoint_eval/moe_route_aware_candidate` |
| 12 | `moe_bias_calibrated_candidate` | `checkpoint_missing_until_materialized` | `not_run` |  |  | `results/checkpoints/moe_bias_calibrated_candidate` | 8112 | `results/vllm_checkpoint_eval/moe_bias_calibrated_candidate` |
| 13 | `toy_moe_expert_weight_candidate` | `not_vllm_loadable_toy_candidate` | `not_run` |  |  | `results/checkpoints/toy_moe_expert_weight_candidate` | 8113 | `results/vllm_checkpoint_eval/toy_moe_expert_weight_candidate` |

## Commands

- `results/vllm_checkpoint_eval_plan/serve_and_eval_commands.sh`
- `results/vllm_checkpoint_eval_plan/checkpoint_eval_plan.csv`
- `results/vllm_checkpoint_eval_plan/summary.json`
