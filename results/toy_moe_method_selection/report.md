# MoE Merge Method Selection

这个报告把 MoE 方法分数和 routing readiness 合在一起，给出是否 materialize 的决策。它的边界是保守的：endpoint 只能作 baseline，低 route-overlap / 低 top-1 agreement 的 average 会被拒绝，能过性能和 routing gate 的方法才进入 checkpoint writer 或下一轮 held-out eval。

- Recommended soft-router method: `expert_output_projection_router_calibrated_average`
- Recommended sparse `hard_top2` method: `unified_confidence_blended_route_kd_seed_average`
- Capacity-aware sparse `hard_top2` method: `unified_moe_bias_capacity_average`
- Sparse accuracy/overflow Pareto frontier: `unified_confidence_blended_route_kd_seed_average, unified_confidence_blended_moe_average, unified_output_projection_moe_average, unified_moe_bias_capacity_average, unified_output_projection_bias_capacity_average, matched_router_kd_average`
- Selection status: `has_candidate`
- Base worst accuracy: `0.7325`

## Decision Table

| method | kind | soft worst acc | hard top-2 worst acc | top-k overflow | avg acc | calibrate flags | min top-k Jaccard | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| expert_output_projection_router_calibrated_average | merge_candidate | 0.807 | 0.647 | 0.0550 | 0.835 | 0 | 0.84 | `candidate_with_router_guard` |
| unified_output_projection_calibrated_seed_average | merge_candidate | 0.807 | 0.647 | 0.0550 | 0.835 | 0 | 0.84 | `candidate_with_router_guard` |
| confidence_blended_router_calibrated_average | merge_candidate | 0.805 | 0.642 | 0.0563 | 0.833 | 0 | 0.84 | `candidate_with_router_guard` |
| unified_confidence_blended_calibrated_seed_average | merge_candidate | 0.805 | 0.642 | 0.0563 | 0.833 | 0 | 0.84 | `candidate_with_router_guard` |
| expert_weight_search_router_calibrated_average | merge_candidate | 0.802 | 0.642 | 0.0563 | 0.830 | 0 | 0.84 | `candidate_with_router_guard` |
| unified_calibrated_seed_average | merge_candidate | 0.802 | 0.642 | 0.0563 | 0.830 | 0 | 0.84 | `candidate_with_router_guard` |
| matched_router_calibrated_average | merge_candidate | 0.797 | 0.665 | 0.0537 | 0.829 | 0 | 0.8367 | `candidate_with_router_guard` |
| matched_router_sweep_selected_average | merge_candidate | 0.797 | 0.665 | 0.0537 | 0.829 | 0 | 0.8367 | `candidate_with_router_guard` |
| unified_output_projection_moe_average | merge_candidate | 0.795 | 0.685 | 0.0700 | 0.829 | 0 | 0.895 | `candidate_with_router_guard` |
| unified_output_projection_route_kd_seed_average | merge_candidate | 0.795 | 0.680 | 0.0788 | 0.821 | 0 | 0.865 | `candidate_with_router_guard` |
| unified_confidence_blended_moe_average | merge_candidate | 0.790 | 0.690 | 0.0762 | 0.824 | 0 | 0.8767 | `candidate_with_router_guard` |
| unified_confidence_blended_route_kd_seed_average | merge_candidate | 0.785 | 0.693 | 0.0788 | 0.820 | 0 | 0.865 | `candidate_with_router_guard` |
| unified_moe_average | merge_candidate | 0.785 | 0.690 | 0.0775 | 0.823 | 0 | 0.8733 | `candidate_with_router_guard` |
| unified_output_projection_bias_capacity_average | merge_candidate | 0.780 | 0.675 | 0.0450 | 0.816 | 0 | 0.915 | `candidate_with_router_guard` |
| unified_route_kd_seed_average | merge_candidate | 0.777 | 0.688 | 0.0788 | 0.816 | 0 | 0.865 | `candidate_with_router_guard` |
| unified_moe_bias_capacity_average | merge_candidate | 0.770 | 0.682 | 0.0475 | 0.809 | 0 | 0.9217 | `candidate_with_router_guard` |
| unified_confidence_blended_bias_capacity_average | merge_candidate | 0.770 | 0.680 | 0.0475 | 0.810 | 0 | 0.9217 | `candidate_with_router_guard` |
| matched_router_route_kd_average | merge_candidate | 0.762 | 0.685 | 0.0788 | 0.805 | 0 | 0.865 | `candidate_with_router_guard` |
| unified_confidence_blended_router_kd_seed_average | merge_candidate | 0.762 | 0.600 | 0.0338 | 0.801 | 0 | 0.8983 | `candidate_with_router_guard` |
| confidence_blended_expert_average | merge_candidate | 0.760 | 0.627 | 0.0675 | 0.797 | 0 | 1 | `candidate_with_router_guard` |
| unified_output_projection_router_kd_seed_average | merge_candidate | 0.760 | 0.608 | 0.0338 | 0.796 | 0 | 0.8983 | `candidate_with_router_guard` |
| unified_router_kd_seed_average | merge_candidate | 0.760 | 0.598 | 0.0338 | 0.800 | 0 | 0.8983 | `candidate_with_router_guard` |
| expert_output_projection_average | merge_candidate | 0.757 | 0.635 | 0.0675 | 0.795 | 0 | 1 | `candidate_with_router_guard` |
| matched_router_topk_calibrated_average | merge_candidate | 0.755 | 0.657 | 0.0788 | 0.792 | 0 | 0.895 | `candidate_with_router_guard` |
| expert_weight_search_average | merge_candidate | 0.755 | 0.632 | 0.0675 | 0.795 | 0 | 1 | `candidate_with_router_guard` |
| route_aware_expert_average | merge_candidate | 0.750 | 0.650 | 0.0675 | 0.790 | 0 | 1 | `candidate_with_router_guard` |
| matched_router_weight_search_average | merge_candidate | 0.750 | 0.657 | 0.0688 | 0.792 | 0 | 0.965 | `candidate_with_router_guard` |
| expert_matched_regmean_average | merge_candidate | 0.750 | 0.640 | 0.0675 | 0.785 | 0 | 1 | `candidate_with_router_guard` |
| expert_matched_average | merge_candidate | 0.750 | 0.652 | 0.0675 | 0.791 | 0 | 0.965 | `candidate_with_router_guard` |
| matched_router_hessian_average | merge_candidate | 0.750 | 0.650 | 0.0663 | 0.789 | 0 | 0.9633 | `candidate_with_router_guard` |
| matched_router_kd_average | merge_candidate | 0.745 | 0.660 | 0.0338 | 0.786 | 0 | 0.8983 | `candidate_with_router_guard` |
| matched_router_frozen_average | merge_candidate | 0.743 | 0.660 | 0.0675 | 0.785 | 0 | 1 | `candidate_with_router_guard` |
| expert_matched_dare_average | merge_candidate | 0.733 | 0.660 | 0.0675 | 0.782 | 0 | 1 | `candidate_with_router_guard` |
| expert_matched_ties_dare_average | merge_candidate | 0.713 | 0.662 | 0.0675 | 0.774 | 0 | 1 | `reject_underperforms_base` |
| expert_matched_ties_average | merge_candidate | 0.710 | 0.655 | 0.0675 | 0.769 | 0 | 1 | `reject_underperforms_base` |
| router_frozen_average | merge_candidate | 0.555 | 0.507 | 0.0675 | 0.631 | 0 | 1 | `reject_underperforms_base` |
| general_endpoint | endpoint_baseline | 0.723 | 0.647 | 0.0650 | 0.772 | 0 | 0.9517 | `baseline_only` |
| code_endpoint_permuted | endpoint_baseline | 0.757 | 0.637 | 0.0688 | 0.804 | 2 | 0.2075 | `baseline_only` |
| base | base_baseline | 0.733 | 0.615 | 0.0675 | 0.778 | 0 | n/a | `baseline_only` |
| all_weight_average | merge_candidate | 0.545 | 0.562 | 0.1062 | 0.624 | 1 | 0.475 | `reject_routing_breakdown` |

## Recommendation

推荐先 materialize/复评 `expert_output_projection_router_calibrated_average`。

- worst accuracy: `0.807`
- avg accuracy: `0.835`
- decision: `candidate_with_router_guard`
- reason: candidate passes routing-overlap gate after router calibration; keep load-balance and held-out route checks.

如果部署路径使用 `hard_top2` sparse dispatch，优先复评 `unified_confidence_blended_route_kd_seed_average`。

- hard_top2 worst accuracy: `0.693`
- soft worst accuracy: `0.785`
- decision: `candidate_with_router_guard`

如果同时惩罚 `hard_top2` accuracy loss 和 capacity overflow，当前优先复评 `unified_moe_bias_capacity_average`。

- capacity-aware score: `0.6300`
- hard_top2 worst accuracy: `0.682`
- max top-k overflow fraction: `0.0475`
- worst overflow category: `general`

## Sparse Pareto Frontier

这些点在 hard top-2 accuracy 和 top-k capacity overflow 上互不支配；部署前应围绕这几个点做 vLLM 下游评测，而不是只看 soft-router 分数。

| method | hard top-2 worst acc | top-k overflow | worst category | soft worst acc |
| --- | ---: | ---: | --- | ---: |
| unified_confidence_blended_route_kd_seed_average | 0.693 | 0.0788 | general | 0.785 |
| unified_confidence_blended_moe_average | 0.690 | 0.0762 | general | 0.790 |
| unified_output_projection_moe_average | 0.685 | 0.0700 | general | 0.795 |
| unified_moe_bias_capacity_average | 0.682 | 0.0475 | general | 0.770 |
| unified_output_projection_bias_capacity_average | 0.675 | 0.0450 | general | 0.780 |
| matched_router_kd_average | 0.660 | 0.0338 | code | 0.745 |

## 规则

- `base` 和 endpoint 只能说明 anchor/source 能力，不算平均结果。
- `calibrate_router_before_average` 计数大于阈值时，拒绝直接 materialize。
- 候选方法必须至少不低于 base worst accuracy 阈值。
- 通过 route-overlap 但有 load concentration 的方法标为 `candidate_with_router_guard`，表示 materialize 后仍需做 load-balance 和 held-out route-overlap 检查。

## Files

- `results/toy_moe_method_selection/method_selection.csv`
- `results/toy_moe_method_selection/sparse_pareto_frontier.csv`
- `results/toy_moe_method_selection/summary.json`
