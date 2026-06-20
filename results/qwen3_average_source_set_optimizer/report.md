# Qwen3 Average Source-Set Optimizer

This optimizer sits after the source-set complementarity gate. It asks whether measured endpoint complementarity is large enough to pay for observed merging interference before a same-shape average receives final-candidate budget.

## Summary

- Status: `source_set_surplus_optimizer_ready`
- Interference budget: `0.0694` from `max_floor_observed_negative_gap_q0.50`
- Source sets scored: `11`
- Final-budget source sets: `0`
- Probe-only source sets: `2`
- Top source set: `coder+thinking`
- Top optimizer gate: `probe_only_below_interference_budget`
- Top frontier avg gain: `0.0083`
- Top surplus vs interference: `-0.0611`
- Top task surplus-positive: `0/3`
- Top no-gain tasks: `2/3`
- Top best task gain: `mmlu` / `0.0250`
- Top blocking tasks: `mmlu,gsm8k,humaneval`
- Top source weights: `{"coder": 0.6666666666666666, "thinking": 0.3333333333333333}`
- Recommended action: `run_larger_endpoint_eval_and_small_weighted_probe_no_final_acceptance`

## Candidate Source Sets

| source set | gate | priority | avg gain | interference | surplus | weights | action |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `coder+thinking` | `probe_only_below_interference_budget` | 0.633 | 0.0083 | 0.0694 | -0.0611 | `{"coder": 0.6666666666666666, "thinking": 0.3333333333333333}` | run_larger_endpoint_eval_and_small_weighted_probe_no_final_acceptance |
| `base+coder+thinking` | `probe_only_below_interference_budget` | 0.633 | 0.0083 | 0.0694 | -0.0611 | `{"base": 0.0, "coder": 0.6666666666666666, "thinking": 0.3333333333333333}` | run_larger_endpoint_eval_and_small_weighted_probe_no_final_acceptance; drop_zero_vote_sources_or_keep_as_anchor_only |
| `base+instruct` | `reject_source_dominated` | 0.100 | 0.0000 | 0.0694 | -0.0694 | `{"base": 0.0, "instruct": 1.0}` | do_not_materialize_except_negative_control; drop_zero_vote_sources_or_keep_as_anchor_only |
| `base+coder` | `reject_source_dominated` | 0.100 | 0.0000 | 0.0694 | -0.0694 | `{"base": 0.16666666666666666, "coder": 0.8333333333333334}` | do_not_materialize_except_negative_control |
| `base+thinking` | `reject_source_dominated` | 0.100 | 0.0000 | 0.0694 | -0.0694 | `{"base": 0.0, "thinking": 1.0}` | do_not_materialize_except_negative_control; drop_zero_vote_sources_or_keep_as_anchor_only |
| `instruct+thinking` | `reject_source_dominated` | 0.100 | 0.0000 | 0.0694 | -0.0694 | `{"instruct": 1.0, "thinking": 0.0}` | do_not_materialize_except_negative_control; drop_zero_vote_sources_or_keep_as_anchor_only |
| `base+instruct+coder` | `reject_source_dominated` | 0.100 | 0.0000 | 0.0694 | -0.0694 | `{"base": 0.0, "coder": 0.3333333333333333, "instruct": 0.6666666666666666}` | do_not_materialize_except_negative_control; drop_zero_vote_sources_or_keep_as_anchor_only |
| `base+instruct+thinking` | `reject_source_dominated` | 0.100 | 0.0000 | 0.0694 | -0.0694 | `{"base": 0.0, "instruct": 1.0, "thinking": 0.0}` | do_not_materialize_except_negative_control; drop_zero_vote_sources_or_keep_as_anchor_only |
| `base+instruct+coder+thinking` | `reject_source_dominated` | 0.100 | 0.0000 | 0.0694 | -0.0694 | `{"base": 0.0, "coder": 0.3333333333333333, "instruct": 0.6666666666666666, "thinking": 0.0}` | do_not_materialize_except_negative_control; drop_zero_vote_sources_or_keep_as_anchor_only |
| `instruct+coder+thinking` | `reject_source_dominated` | 0.036 | 0.0000 | 0.0694 | -0.0694 | `{"coder": 0.3333333333333333, "instruct": 0.6666666666666666, "thinking": 0.0}` | do_not_materialize_except_negative_control; drop_zero_vote_sources_or_keep_as_anchor_only |
| `instruct+coder` | `reject_source_dominated` | 0.031 | 0.0000 | 0.0694 | -0.0694 | `{"coder": 0.3333333333333333, "instruct": 0.6666666666666666}` | do_not_materialize_except_negative_control |

## Task-Level Surplus

| source set | task | frontier source | gain | surplus | observed gap | status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `coder+thinking` | `mmlu` | `thinking` | 0.0250 | -0.0444 | n/a | `gain_below_interference_budget` |
| `coder+thinking` | `gsm8k` | `coder` | 0.0000 | -0.0694 | n/a | `no_task_frontier_gain` |
| `coder+thinking` | `humaneval` | `coder` | 0.0000 | -0.0694 | n/a | `no_task_frontier_gain` |

## Eval Plan

| candidate | status | tasks | examples/task | command |
| --- | --- | --- | ---: | --- |
| `qwen3_moe_coder_thinking_frontier_weighted_probe` | `endpoint_expansion_or_probe_only` | `mmlu,gsm8k,humaneval` | 32 | `python scripts/build_fp_downstream_matrix.py --help` |
| `qwen3_moe_base_coder_thinking_frontier_weighted_probe` | `endpoint_expansion_or_probe_only` | `mmlu,gsm8k,humaneval` | 32 | `python scripts/build_fp_downstream_matrix.py --help` |
| `qwen3_moe_base_instruct_frontier_weighted_probe` | `hold_source_dominated` | `` | 0 | `` |
| `qwen3_moe_base_coder_frontier_weighted_probe` | `hold_source_dominated` | `` | 0 | `` |
| `qwen3_moe_base_thinking_frontier_weighted_probe` | `hold_source_dominated` | `` | 0 | `` |
| `qwen3_moe_instruct_thinking_frontier_weighted_probe` | `hold_source_dominated` | `` | 0 | `` |
| `qwen3_moe_base_coder_instruct_frontier_weighted_probe` | `hold_source_dominated` | `` | 0 | `` |
| `qwen3_moe_base_instruct_thinking_frontier_weighted_probe` | `hold_source_dominated` | `` | 0 | `` |
| `qwen3_moe_base_coder_instruct_thinking_frontier_weighted_probe` | `hold_source_dominated` | `` | 0 | `` |
| `qwen3_moe_coder_instruct_thinking_frontier_weighted_probe` | `hold_source_dominated` | `` | 0 | `` |
| `qwen3_moe_coder_instruct_frontier_weighted_probe` | `hold_source_dominated` | `` | 0 | `` |

## Source Discovery Queue

| scenario | priority | action | carry-forward probe set |
| --- | --- | --- | --- |
| `moe_30b_general_code_route_aware` | `p0_moe_wave` | search_for_stronger_endpoint_complementarity_then_apply_surplus_gate | `coder+thinking` |
| `measured_qwen3_moe_coder_thinking_probe` | `p0_probe_only` | expand_endpoint_eval_or_materialize_small_probe_only_because_surplus_is_below_interference_budget | `coder+thinking` |
| `dense_7b_general_code_math_reasoning` | `p0_first_wave` | run_endpoint_frontier_before_average | `coder+thinking` |
| `moe_30b_downstream_adapter_average` | `p1_after_real_routing_probe` | search_for_stronger_endpoint_complementarity_then_apply_surplus_gate | `coder+thinking` |
| `dense_32b_reasoning_long_reasoning` | `p1_scale_validation` | run_endpoint_frontier_before_average | `coder+thinking` |
| `dense_7b_domain_extension` | `p1_after_7b_core` | run_endpoint_frontier_before_average | `coder+thinking` |
| `negative_controls` | `always` | keep_controls_only | `coder+thinking` |

## Literature Priors

| key | source | mechanism used here |
| --- | --- | --- |
| `mode_connectivity` | https://arxiv.org/abs/1802.10026 | A source-set average is plausible only when the probed path stays in a low-loss component. |
| `model_soups` | https://arxiv.org/abs/2203.05482 | Weight averaging helps when fine-tunes are in one basin; endpoint fallback remains part of the recipe. |
| `ties` | https://arxiv.org/abs/2306.01708 | Coordinate interference can erase useful deltas, so source/task gains must exceed merge interference. |
| `expert_merging` | https://arxiv.org/abs/2509.25712 | Layer/chunk coefficients should be driven by calibration behavior, not a fixed global coefficient. |
| `sub_moe` | https://arxiv.org/abs/2506.23266 | MoE expert similarity and subspace conflict are better signals than tensor names alone. |
| `harc` | https://arxiv.org/abs/2606.03391 | MoE router perturbations can break top-k dispatch, so source-set averaging must keep router repair as a separate gate. |

## Outputs

- `candidate_source_sets`: `results/qwen3_average_source_set_optimizer/candidate_source_sets.csv`
- `task_surplus`: `results/qwen3_average_source_set_optimizer/task_surplus.csv`
- `source_weight_recipes`: `results/qwen3_average_source_set_optimizer/source_weight_recipes.csv`
- `eval_plan`: `results/qwen3_average_source_set_optimizer/eval_plan.csv`
- `source_discovery_queue`: `results/qwen3_average_source_set_optimizer/source_discovery_queue.csv`
- `summary`: `results/qwen3_average_source_set_optimizer/summary.json`
- `report`: `results/qwen3_average_source_set_optimizer/report.md`
