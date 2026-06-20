# Qwen3 MoE Eval Budget Queue Smoke

This smoke verifies that the budgeted vLLM runner defaults to the final-selection queue, while ablation-only candidates stay out of the default run.

- Status: `passed`
- Assertions: `11/11`
- Final queue methods: `4`
- Mechanism queue methods: `8`

| assertion | expected | actual | passed |
| --- | --- | --- | --- |
| `default_runner_request_is_final` | `final` | `final` | `True` |
| `runner_shell_defaults_to_final` | `requested="${1:-final}"` | `present` | `True` |
| `runner_preflight_defaults_to_final_scope` | `preflight_scope="${2:-final}"` | `present` | `True` |
| `final_queue_methods_match_sources_plus_trust_gate` | `qwen3_moe_mechanistic_unified_candidate,qwen3_moe_subspace_scaled_candidate,source_qwen3_30b_coder,source_qwen3_30b_instruct` | `qwen3_moe_mechanistic_unified_candidate,qwen3_moe_subspace_scaled_candidate,source_qwen3_30b_coder,source_qwen3_30b_instruct` | `True` |
| `mechanism_queue_methods_match_ablation_only_candidates` | `qwen3_moe_audit_gated_candidate,qwen3_moe_expert_only_trust_region_candidate,qwen3_moe_layer_chunk_candidate,qwen3_moe_searched_no_gt065_max_retention_candidate,qwen3_moe_tail_trimmed_expert_only_candidate,qwen3_moe_trust_region_candidate,qwen3_moe_unified_mechanism_candidate,qwen3_moe_unified_route_guarded_candidate` | `qwen3_moe_audit_gated_candidate,qwen3_moe_expert_only_trust_region_candidate,qwen3_moe_layer_chunk_candidate,qwen3_moe_searched_no_gt065_max_retention_candidate,qwen3_moe_tail_trimmed_expert_only_candidate,qwen3_moe_trust_region_candidate,qwen3_moe_unified_mechanism_candidate,qwen3_moe_unified_route_guarded_candidate` | `True` |
| `summary_final_core_method_count_matches_table` | `4` | `4` | `True` |
| `summary_final_prompt_budget_matches_table` | `6144` | `6144` | `True` |
| `summary_mechanism_count_matches_table` | `8` | `8` | `True` |
| `summary_mechanism_prompt_budget_matches_table` | `12288` | `12288` | `True` |
| `router_queue_not_ready_by_default` | `no ready router rows` | `0` | `True` |
| `task_manifest_alignment_covers_all_budget_rows` | `14` | `14` | `True` |

## Outputs

- `results/qwen3_moe_eval_budget_queue_smoke/eval_budget_queue_smoke_matrix.csv`
- `results/qwen3_moe_eval_budget_queue_smoke/summary.json`
- `results/qwen3_moe_eval_budget_queue_smoke/report.md`
