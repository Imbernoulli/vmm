# Qwen3 Source-Set Complementarity Gate

This gate decides whether a source set is worth averaging before spending candidate/eval budget. It separates the source frontier from the merge method: if one source already dominates the measured tasks, averaging is treated as repair or ablation rather than a final-candidate prior.

## Summary

- Status: `source_set_complementarity_gate_ready`
- Current source set: `instruct+coder`
- Current gate: `source_dominated_not_averageable_as_final`
- Current dominant source: `instruct`
- Current frontier avg gain vs best single: `0.0000`
- Best observed merge: `merge_instruct+coder_routercal`
- Best observed avg gap to source frontier: `-0.0694`
- Strong complementary measured sets: `2`
- Weak complementary measured sets: `0`
- Recommended action: `do_not_expect_average_to_beat_current_source_frontier; use average only as repair_ablation_until_new_complementary_sources_pass_endpoint_eval`

## Measured Source Sets

| source set | gate | dominant | strict winners | frontier avg gain | frontier worst gain | action |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `coder+thinking` | `complementary_source_set_candidate` | `None` | 2 | 0.0083 | 0.0250 | spend_average_budget_after_topology_and_connectivity_gates |
| `base+coder+thinking` | `complementary_source_set_candidate` | `None` | 2 | 0.0083 | 0.0250 | spend_average_budget_after_topology_and_connectivity_gates |
| `base+instruct` | `source_dominated_not_averageable_as_final` | `instruct` | 1 | 0.0000 | 0.0000 | use_dominant_source_as_frontier; keep averages as repair_or_negative_control |
| `base+coder` | `source_dominated_not_averageable_as_final` | `coder` | 1 | 0.0000 | 0.0000 | use_dominant_source_as_frontier; keep averages as repair_or_negative_control |
| `base+thinking` | `source_dominated_not_averageable_as_final` | `thinking` | 1 | 0.0000 | 0.0000 | use_dominant_source_as_frontier; keep averages as repair_or_negative_control |
| `instruct+coder` | `source_dominated_not_averageable_as_final` | `instruct` | 1 | 0.0000 | 0.0000 | use_dominant_source_as_frontier; keep averages as repair_or_negative_control |
| `instruct+thinking` | `source_dominated_not_averageable_as_final` | `instruct` | 1 | 0.0000 | 0.0000 | use_dominant_source_as_frontier; keep averages as repair_or_negative_control |
| `base+instruct+coder` | `source_dominated_not_averageable_as_final` | `instruct` | 1 | 0.0000 | 0.0000 | use_dominant_source_as_frontier; keep averages as repair_or_negative_control |
| `base+instruct+thinking` | `source_dominated_not_averageable_as_final` | `instruct` | 1 | 0.0000 | 0.0000 | use_dominant_source_as_frontier; keep averages as repair_or_negative_control |
| `instruct+coder+thinking` | `source_dominated_not_averageable_as_final` | `instruct` | 1 | 0.0000 | 0.0000 | use_dominant_source_as_frontier; keep averages as repair_or_negative_control |
| `base+instruct+coder+thinking` | `source_dominated_not_averageable_as_final` | `instruct` | 1 | 0.0000 | 0.0000 | use_dominant_source_as_frontier; keep averages as repair_or_negative_control |

## Observed Merge Gaps

| merge | source set | variant | avg | gap to frontier | task regressions |
| --- | --- | --- | ---: | ---: | ---: |
| `merge_instruct+coder_routercal` | `instruct+coder` | `router_calibrated` | 0.8278 | -0.0694 | 3 |
| `merge_instruct+coder` | `instruct+coder` | `naive_average` | 0.7944 | -0.1028 | 3 |
| `merge_instruct+coder+thinking` | `instruct+coder+thinking` | `naive_average` | 0.8333 | -0.0639 | 2 |

## Registry Scenarios

| scenario | priority | source-set gate action | first average candidate |
| --- | --- | --- | --- |
| `dense_7b_general_code_math_reasoning` | `p0_first_wave` | `run_endpoint_eval_then_source_set_gate` | `coefficient_search_after_connectivity_gate` |
| `dense_7b_domain_extension` | `p1_after_7b_core` | `queue_after_p0_source_set_gate` | `anchor_plus_domain_delta_with_small_lambda_sweep` |
| `dense_32b_reasoning_long_reasoning` | `p1_scale_validation` | `queue_after_p0_source_set_gate` | `greedy_soup_or_task_arithmetic_only_inside_low_barrier_component` |
| `moe_30b_general_code_route_aware` | `p0_moe_wave` | `run_endpoint_eval_then_source_set_gate` | `router_frozen_shared_merge_plus_expert_matched_average` |
| `moe_30b_downstream_adapter_average` | `p1_after_real_routing_probe` | `queue_after_p0_source_set_gate` | `same-shape_adapter_average_or_route_weighted_full_delta` |
| `negative_controls` | `always` | `keep_negative_controls` | `none_control_only` |

## Outputs

- `source_set_gate`: `results/qwen3_source_set_complementarity_gate/source_set_gate.csv`
- `observed_merge_gaps`: `results/qwen3_source_set_complementarity_gate/observed_merge_gaps.csv`
- `recommended_scenarios`: `results/qwen3_source_set_complementarity_gate/recommended_scenarios.csv`
- `summary`: `results/qwen3_source_set_complementarity_gate/summary.json`
- `report`: `results/qwen3_source_set_complementarity_gate/report.md`
