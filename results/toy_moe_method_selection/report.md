# MoE Merge Method Selection

这个报告把 MoE 方法分数和 routing readiness 合在一起，给出是否 materialize 的决策。它的边界是保守的：endpoint 只能作 baseline，低 route-overlap / 低 top-1 agreement 的 average 会被拒绝，能过性能和 routing gate 的方法才进入 checkpoint writer 或下一轮 held-out eval。

- Recommended method: `matched_router_calibrated_average`
- Selection status: `has_candidate`
- Base worst accuracy: `0.775`

## Decision Table

| method | kind | worst acc | avg acc | calibrate flags | min top-k Jaccard | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| matched_router_sweep_selected_average | merge_candidate | 0.838 | 0.839 | 0 | 0.8275 | `candidate_with_router_guard` |
| matched_router_calibrated_average | merge_candidate | 0.838 | 0.839 | 0 | 0.8275 | `candidate_with_router_guard` |
| expert_weight_search_router_calibrated_average | merge_candidate | 0.828 | 0.831 | 0 | 0.8308 | `candidate_with_router_guard` |
| matched_router_weight_search_average | merge_candidate | 0.818 | 0.819 | 0 | 0.97 | `candidate_with_router_guard` |
| expert_weight_search_average | merge_candidate | 0.802 | 0.806 | 0 | 1 | `candidate_with_router_guard` |
| expert_matched_average | merge_candidate | 0.800 | 0.809 | 0 | 0.98 | `candidate_with_router_guard` |
| expert_matched_regmean_average | merge_candidate | 0.792 | 0.807 | 0 | 1 | `candidate_with_router_guard` |
| route_aware_expert_average | merge_candidate | 0.790 | 0.799 | 0 | 1 | `candidate_with_router_guard` |
| matched_router_frozen_average | merge_candidate | 0.787 | 0.801 | 0 | 1 | `candidate_with_router_guard` |
| expert_matched_dare_average | merge_candidate | 0.780 | 0.796 | 0 | 1 | `candidate_with_router_guard` |
| expert_matched_ties_dare_average | merge_candidate | 0.777 | 0.796 | 0 | 1 | `candidate_with_router_guard` |
| expert_matched_ties_average | merge_candidate | 0.775 | 0.792 | 0 | 1 | `candidate_with_router_guard` |
| router_frozen_average | merge_candidate | 0.615 | 0.652 | 0 | 1 | `reject_underperforms_base` |
| general_endpoint | endpoint_baseline | 0.780 | 0.796 | 0 | 0.965 | `baseline_only` |
| code_endpoint_permuted | endpoint_baseline | 0.802 | 0.810 | 2 | 0.2042 | `baseline_only` |
| base | base_baseline | 0.775 | 0.786 | 0 | n/a | `baseline_only` |
| all_weight_average | merge_candidate | 0.620 | 0.654 | 1 | 0.4883 | `reject_routing_breakdown` |

## Recommendation

推荐先 materialize/复评 `matched_router_calibrated_average`。

- worst accuracy: `0.838`
- avg accuracy: `0.839`
- decision: `candidate_with_router_guard`
- reason: candidate passes routing-overlap gate after router calibration; keep load-balance and held-out route checks.

## 规则

- `base` 和 endpoint 只能说明 anchor/source 能力，不算平均结果。
- `calibrate_router_before_average` 计数大于阈值时，拒绝直接 materialize。
- 候选方法必须至少不低于 base worst accuracy 阈值。
- 通过 route-overlap 但有 load concentration 的方法标为 `candidate_with_router_guard`，表示 materialize 后仍需做 load-balance 和 held-out route-overlap 检查。

## Files

- `results/toy_moe_method_selection/method_selection.csv`
- `results/toy_moe_method_selection/summary.json`
