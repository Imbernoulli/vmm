# Qwen3 MoE Adaptive Eval Schedule Smoke

- Status: `passed`
- Assertions: `15/15`

| case | assertion | expected | actual | passed |
| --- | --- | --- | --- | --- |
| `awaiting_sources` | `top_method` | `source_qwen3_30b_instruct` | `source_qwen3_30b_instruct` | `True` |
| `awaiting_sources` | `unified_candidate_status` | `awaiting_source_controls` | `awaiting_source_controls` | `True` |
| `probe_selected` | `top_method` | `qwen3_moe_unified_mechanism_candidate` | `qwen3_moe_unified_mechanism_candidate` | `True` |
| `probe_selected` | `unified_candidate_status` | `selected_for_initial_probe` | `selected_for_initial_probe` | `True` |
| `promising_escalates` | `top_method` | `qwen3_moe_unified_mechanism_candidate` | `qwen3_moe_unified_mechanism_candidate` | `True` |
| `promising_escalates` | `unified_candidate_status` | `promising_needs_powered_eval` | `promising_needs_powered_eval` | `True` |
| `full_ready` | `top_method` | `qwen3_moe_searched_no_gt065_max_retention_candidate` | `qwen3_moe_searched_no_gt065_max_retention_candidate` | `True` |
| `full_ready` | `unified_candidate_status` | `ready_for_final_selector` | `ready_for_final_selector` | `True` |
| `dominated_pruned` | `top_method` | `qwen3_moe_searched_no_gt065_max_retention_candidate` | `qwen3_moe_searched_no_gt065_max_retention_candidate` | `True` |
| `dominated_pruned` | `unified_candidate_status` | `pruned_by_source_frontier_probe` | `pruned_by_source_frontier_probe` | `True` |
| `paired_regression_pruned` | `top_method` | `qwen3_moe_searched_no_gt065_max_retention_candidate` | `qwen3_moe_searched_no_gt065_max_retention_candidate` | `True` |
| `paired_regression_pruned` | `unified_candidate_status` | `pruned_by_paired_source_probe` | `pruned_by_paired_source_probe` | `True` |
| `coverage_selects_complement` | `subspace_selected_for_coverage` | `True` | `True` | `True` |
| `coverage_selects_complement` | `covered_mechanism_test_count` | `2` | `2` | `True` |
| `coverage_selects_complement` | `subspace_probe_tasks` | `gsm8k,humaneval_compile` | `gsm8k,humaneval_compile` | `True` |
