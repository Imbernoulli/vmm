# Checkpoint Materialization Readiness

这个 audit 把 same-shape writer 命令、router-bias plan、dry-run manifest 和 vLLM eval plan 串起来。目标是区分三件事：方法是否有 recipe、checkpoint 是否已经写出、是否可以进入真实 vLLM 下游评测。

- Status: `hosted_eval_complete`
- Candidates: `17`
- Materialized checkpoints: `10`
- Blocked by placeholders: `4`
- Ready for vLLM eval: `9`
- Completed vLLM evals: `1`

| candidate | writer status | vLLM status | eval status | avg primary | worst primary | end-to-end status | next action |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| `qwen3_moe_unified_route_guarded_candidate` | `materialized_checkpoint_exists` | `ready_to_host` | `not_run` |  |  | `ready_for_vllm_eval` | host the vLLM plan checkpoint and run downstream eval |
| `qwen3_moe_audit_gated_candidate` | `materialized_checkpoint_exists` | `ready_to_host` | `not_run` |  |  | `ready_for_vllm_eval` | host the vLLM plan checkpoint and run downstream eval |
| `qwen3_moe_trust_region_candidate` | `materialized_checkpoint_exists` | `ready_to_host` | `not_run` |  |  | `ready_for_vllm_eval` | host the vLLM plan checkpoint and run downstream eval |
| `qwen3_moe_expert_only_trust_region_candidate` | `materialized_checkpoint_exists` | `ready_to_host` | `not_run` |  |  | `ready_for_vllm_eval` | host the vLLM plan checkpoint and run downstream eval |
| `qwen3_moe_tail_trimmed_expert_only_candidate` | `materialized_checkpoint_exists` | `ready_to_host` | `not_run` |  |  | `ready_for_vllm_eval` | host the vLLM plan checkpoint and run downstream eval |
| `qwen3_moe_searched_no_gt065_max_retention_candidate` | `materialized_checkpoint_exists` | `ready_to_host` | `not_run` |  |  | `ready_for_vllm_eval` | host the vLLM plan checkpoint and run downstream eval |
| `qwen3_moe_layer_chunk_candidate` | `materialized_checkpoint_exists` | `ready_to_host` | `not_run` |  |  | `ready_for_vllm_eval` | host the vLLM plan checkpoint and run downstream eval |
| `qwen3_moe_unified_mechanism_candidate` | `materialized_checkpoint_exists` | `ready_to_host` | `not_run` |  |  | `ready_for_vllm_eval` | host the vLLM plan checkpoint and run downstream eval |
| `qwen3_moe_mechanistic_unified_candidate` | `materialization_command_ready` | `checkpoint_missing_until_materialized` | `not_run` |  |  | `needs_checkpoint_materialization` | run writer command and verify safetensors output |
| `qwen3_moe_subspace_scaled_candidate` | `materialized_checkpoint_exists` | `ready_to_host` | `not_run` |  |  | `ready_for_vllm_eval` | host the vLLM plan checkpoint and run downstream eval |
| `moe_route_aware_candidate` | `blocked_by_placeholder_inputs` | `checkpoint_missing_until_materialized` | `not_run` |  |  | `blocked_before_materialization` | replace placeholder model paths/route weights, run writer dry-run, then materialize |
| `toy_moe_expert_weight_candidate` | `blocked_by_placeholder_inputs` | `not_vllm_loadable_toy_candidate` | `not_run` |  |  | `toy_writer_validation_only` | replace placeholder model paths/route weights, run writer dry-run, then materialize |
| `toy_moe_expert_matched_candidate` | `blocked_by_placeholder_inputs` | `not_in_vllm_plan` | `not_run` |  |  | `toy_writer_validation_only` | replace placeholder model paths/route weights, run writer dry-run, then materialize |
| `toy_moe_confidence_blended_combined_candidate` | `blocked_by_placeholder_inputs` | `not_in_vllm_plan` | `not_run` |  |  | `toy_writer_validation_only` | replace placeholder model paths/route weights, run writer dry-run, then materialize |
| `moe_bias_calibrated_candidate` | `needs_real_moe_source_paths_for_tensor_add_writer` | `checkpoint_missing_until_materialized` | `not_run` |  |  | `needs_checkpoint_materialization` | run write_same_shape_average_checkpoint.py with real MoE sources and --tensor-add-csv results/moe_router_bias_plan/router_bias_deltas.csv |
| `qwen_0_5b_writer_compatibility` | `dry_run_compatible_no_checkpoint_written` | `not_in_vllm_plan` | `not_run` |  |  | `needs_materialization_after_dry_run` | choose a non-rejected dense average coefficient, run writer without --dry-run, then run vLLM eval plan |
| `qwen_0_5b_instruct_coder_uniform_average` | `materialized_checkpoint_exists` | `ready_to_host` | `complete` | 0.180 | 0.000 | `hosted_eval_complete` | compare this negative baseline against source endpoints and optimized candidates |

## Files

- `results/checkpoint_materialization_readiness/candidate_readiness.csv`
- `results/checkpoint_materialization_readiness/summary.json`
