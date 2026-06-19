# Toy MoE Route-Aware Merge

这个实验用一个很小的 soft-router MoE 做可控验证：base 先在 general/code 两类合成任务上训练，然后从同一 base fine-tune 两个同构 source。为了模拟 MoE 中常见的 expert-index 语义漂移，code source 在保持函数等价的前提下被 permute experts 和 router rows。

它验证的点很具体：直接 all-weight average 会把不同语义的 expert index 相加；expert matching 和 route-frequency expert weights 可以缓解这个问题；router 是否能开放平均要看 route overlap、load concentration 和 top-k margin。

## 关键结果

- Best method by worst accuracy: `expert_output_projection_router_calibrated_average` = `0.807`.
- All-weight average worst accuracy: `0.545`.
- Expert-matched average worst accuracy: `0.750`.
- Matched + router-frozen average worst accuracy: `0.743`.
- Expert-matched RegMean average worst accuracy: `0.750`.
- Expert-matched TIES average worst accuracy: `0.710`.
- Expert-matched DARE average worst accuracy: `0.733`.
- Expert-matched TIES+DARE average worst accuracy: `0.713`.
- Matched + router-weight-search average worst accuracy: `0.750`.
- Matched + Hessian-router average worst accuracy: `0.750`.
- Matched + Router-KD average worst accuracy: `0.745`.
- Matched + route-KD average worst accuracy: `0.762`.
- Matched + router-calibrated average worst accuracy: `0.797`.
- Matched + router-topk-calibrated average worst accuracy: `0.755`.
- Matched + router-sweep-selected average worst accuracy: `0.797`.
- Expert-weight search average worst accuracy: `0.755`.
- Expert-weight search + router-calibrated worst accuracy: `0.802`.
- Expert output-projection average worst accuracy: `0.757`.
- Expert output-projection + router-calibrated worst accuracy: `0.807`.
- Unified expert/router objective worst accuracy: `0.785`.
- Unified + router-bias capacity calibration worst accuracy: `0.770`.
- Unified router/capacity sweep selected router seed `router_kd_seed` and capacity loss `0.000` with held-out selection capacity-aware score `0.665`.
- Route-aware expert average worst accuracy: `0.750`.
- Lowest MoE connectivity barrier: `direct_matched_general_to_code` = `0.0000` worst-loss barrier.
- Direct unmatched source barrier: `0.0341`.
- Direct matched source barrier: `0.0000`.
- Matched + router-calibrated hard top-1 worst accuracy: `0.608`.
- Matched + router-calibrated hard top-2 worst accuracy: `0.665`.
- Matched + Hessian-router hard top-2 worst accuracy: `0.650`.
- Matched + Router-KD hard top-2 worst accuracy: `0.660`.
- Matched + route-KD hard top-2 worst accuracy: `0.685`.
- Route-KD hard top-2 delta vs router-calibrated: `0.020`.
- Matched + router-topk-calibrated hard top-2 worst accuracy: `0.657`.
- Capacity factor `1.25` max top-k overflow fraction: `0.106`.
- Top-k router calibration delta vs soft router calibration under hard top-2: `-0.008`.
- Recovered expert matching mean cosine: `0.977`.
- Code source permutation: `[2, 0, 3, 1]`.

## Method Table

| method | general acc | code acc | worst acc | avg loss |
| --- | ---: | ---: | ---: | ---: |
| expert_output_projection_router_calibrated_average | 0.863 | 0.807 | 0.807 | 0.603 |
| expert_weight_search_router_calibrated_average | 0.858 | 0.802 | 0.802 | 0.602 |
| unified_calibrated_seed_average | 0.858 | 0.802 | 0.802 | 0.602 |
| matched_router_calibrated_average | 0.860 | 0.797 | 0.797 | 0.603 |
| matched_router_sweep_selected_average | 0.860 | 0.797 | 0.797 | 0.603 |
| unified_moe_average | 0.860 | 0.785 | 0.785 | 0.607 |
| unified_route_kd_seed_average | 0.855 | 0.777 | 0.777 | 0.609 |
| unified_moe_bias_capacity_average | 0.848 | 0.770 | 0.770 | 0.609 |
| matched_router_route_kd_average | 0.848 | 0.762 | 0.762 | 0.610 |
| unified_router_kd_seed_average | 0.840 | 0.760 | 0.760 | 0.621 |
| code_endpoint_permuted | 0.850 | 0.757 | 0.757 | 0.620 |
| expert_output_projection_average | 0.833 | 0.757 | 0.757 | 0.623 |
| expert_weight_search_average | 0.835 | 0.755 | 0.755 | 0.622 |
| matched_router_topk_calibrated_average | 0.830 | 0.755 | 0.755 | 0.609 |
| matched_router_weight_search_average | 0.835 | 0.750 | 0.750 | 0.621 |
| expert_matched_regmean_average | 0.820 | 0.750 | 0.750 | 0.620 |
| expert_matched_average | 0.833 | 0.750 | 0.750 | 0.621 |
| matched_router_hessian_average | 0.828 | 0.750 | 0.750 | 0.621 |
| route_aware_expert_average | 0.830 | 0.750 | 0.750 | 0.623 |
| matched_router_kd_average | 0.828 | 0.745 | 0.745 | 0.623 |
| matched_router_frozen_average | 0.828 | 0.743 | 0.743 | 0.624 |
| expert_matched_dare_average | 0.833 | 0.733 | 0.733 | 0.624 |
| base | 0.823 | 0.733 | 0.733 | 0.631 |
| general_endpoint | 0.823 | 0.723 | 0.723 | 0.623 |
| expert_matched_ties_dare_average | 0.835 | 0.713 | 0.713 | 0.622 |
| expert_matched_ties_average | 0.828 | 0.710 | 0.710 | 0.624 |
| router_frozen_average | 0.708 | 0.555 | 0.555 | 0.677 |
| all_weight_average | 0.703 | 0.545 | 0.545 | 0.674 |

## Interpretation

- `all_weight_average` 是朴素 baseline：router 和 expert tensors 都按同名 index 平均，因此在 expert permutation 后会暴露 MoE index-alignment 风险。
- `expert_matched_average` 先用 unlabeled calibration input 的 expert-output cosine 做 Hungarian matching，再平均；这对应 Sub-MoE / Expert Merging 里强调的 function-aware expert alignment。
- `matched_router_frozen_average` 直接验证 MoE 特有假设：先对齐 expert 功能，再固定 token-to-expert dispatch，只平均非 router 权重。
- `expert_matched_regmean_average` 在 expert matching 后只对 expert Linear 层做 activation-covariance RegMean，router 仍固定为 base；这把 Dense RegMean 转成了 MoE expert-local 版本。
- `expert_matched_ties_average` / `expert_matched_dare_average` / `expert_matched_ties_dare_average` 把 Dense sparse task-vector merging 迁移到 MoE expert 子网；router 不参与稀疏合并。
- `matched_router_weight_search_average` 不做梯度训练，只对 router tensor 的 general/code task-vector 系数做 guarded search；这是 checkpoint-only 的 MoE router probe。
- `matched_router_hessian_average` 只解 router：用 source router softmax Hessian 和输入协方差做二阶加权最小二乘，检验 routing breakdown 是否来自线性 router averaging 的非线性 mismatch。
- `matched_router_kd_average` 不用标签，只让 router 蒸馏 general/code source logits；这对应 Router KD 的轻量 router-expert mismatch 修复假设。
- `matched_router_route_kd_average` 不用标签，直接蒸馏 source router 的 full route distribution 和 top-1 route；它检验 route-level signal 是否比 output-level KD 更适合 MoE merging。
- `matched_router_calibrated_average` 冻结 matched experts，只用小校准集更新 router，并用 base-router KL 约束防止 dispatch 漂移。
- `matched_router_topk_calibrated_average` 在 router-only calibration 里显式加入 hard top-2 dispatch loss，用来检验 soft-router 优化是否能迁移到真实 sparse dispatch。
- `matched_router_sweep_selected_average` 对 router calibration 的 KL 系数做 sweep，先过 route-overlap guard，再按 calibration worst-loss 选择候选；它把 router overlap/load 和任务精度放到同一个 probe 里。
- `expert_weight_search_average` 在同一个 expert 数和 tensor shape 内，对每个 expert 的 general/code delta 系数做校准集 min-max 坐标搜索；router 仍固定为 base。
- `expert_weight_search_router_calibrated_average` 在 per-expert 系数搜索后，只开放 router 做 guarded calibration。
- `expert_output_projection_average` 不用标签分数搜索，而是用 route-conditioned expert output residual 解每个 expert 的 source-delta 权重；它检验 output-space projection 是否能解释 expert merging。
- `expert_output_projection_router_calibrated_average` 在 output-space expert 权重后只校准 router，用来区分 expert 输出拟合和 router dispatch 校准的贡献。
- `unified_moe_average` 先用 per-expert source weight search 处理 expert 语义和重要性，再只更新 router；目标同时包含 soft/hard task loss、source route KD、source output KD、base-router KL、load-balance 和 differentiable capacity-overflow surrogate。router seed 和 capacity loss 系数都不是手工固定，而是在独立 selection split 上按 hard top-2 worst accuracy 减 max overflow 自动选择。
- `unified_moe_bias_capacity_average` 从 unified 结果出发只训练 router bias，检验全局 expert 负载偏置能否降低 capacity overflow，同时避免重学完整 router 几何。
- `route_aware_expert_average` 冻结 base router，并按 base router 在 general/code prompt 上的 route mass 给每个 expert 设置 source delta 权重；这对应 route-weight recipes 的 toy 版本。
- 这个实验不是 Qwen3 结果，但它把 MoE merging 的特质从报告落成了可跑的 probe：expert index、router overlap、expert load 和 category route mass 都会影响 average 是否安全。

## Files

- `method_metrics.csv`
- `dispatch_mode_metrics.csv`
- `router_summary.csv`
- `expert_load.csv`
- `router_capacity_metrics.csv`
- `route_overlap.csv`
- `expert_match.csv`
- `route_weights_by_expert.csv`
- `connectivity_path_metrics.csv`
- `connectivity_summary.csv`
- `connectivity_paths.png`
- `expert_regmean_covariances.csv`
- `expert_regmean_layers.csv`
- `expert_sparse_task_vectors.csv`
- `expert_search_weights_by_expert.csv`
- `expert_weight_search_trace.csv`
- `expert_output_projection_weights_by_expert.csv`
- `router_weight_search.csv`
- `router_hessian_average.csv`
- `router_kd_trace.csv`
- `router_route_kd_trace.csv`
- `unified_moe_trace.csv`
- `unified_moe_capacity_sweep.csv`
- `router_bias_capacity_trace.csv`
- `router_bias_capacity_sweep.csv`
- `router_calibration_sweep.csv`
- `toy_moe_merge.png`
- `summary.json`
