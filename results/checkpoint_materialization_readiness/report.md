# Checkpoint Materialization Readiness

这个 audit 把 same-shape writer 命令、router-bias plan、dry-run manifest 和 vLLM eval plan 串起来。目标是区分三件事：方法是否有 recipe、checkpoint 是否已经写出、是否可以进入真实 vLLM 下游评测。

- Status: `waiting_for_checkpoint_materialization`
- Candidates: `5`
- Materialized checkpoints: `0`
- Blocked by placeholders: `3`
- Ready for vLLM eval: `0`

| candidate | writer status | vLLM status | end-to-end status | next action |
| --- | --- | --- | --- | --- |
| `moe_route_aware_candidate` | `blocked_by_placeholder_inputs` | `checkpoint_missing_until_materialized` | `blocked_before_materialization` | replace placeholder model paths/route weights, run writer dry-run, then materialize |
| `toy_moe_expert_weight_candidate` | `blocked_by_placeholder_inputs` | `not_vllm_loadable_toy_candidate` | `toy_writer_validation_only` | replace placeholder model paths/route weights, run writer dry-run, then materialize |
| `toy_moe_expert_matched_candidate` | `blocked_by_placeholder_inputs` | `not_in_vllm_plan` | `toy_writer_validation_only` | replace placeholder model paths/route weights, run writer dry-run, then materialize |
| `moe_bias_calibrated_candidate` | `needs_real_moe_source_paths_for_tensor_add_writer` | `checkpoint_missing_until_materialized` | `needs_checkpoint_materialization` | run write_same_shape_average_checkpoint.py with real MoE sources and --tensor-add-csv results/moe_router_bias_plan/router_bias_deltas.csv |
| `qwen_0_5b_writer_compatibility` | `dry_run_compatible_no_checkpoint_written` | `not_in_vllm_plan` | `needs_materialization_after_dry_run` | choose a non-rejected dense average coefficient, run writer without --dry-run, then run vLLM eval plan |

## Files

- `results/checkpoint_materialization_readiness/candidate_readiness.csv`
- `results/checkpoint_materialization_readiness/summary.json`
