# Qwen3 MoE Mechanistic Sensitivity

这个分析重放 mechanistic unified candidate 的同一套 `B/H/I` scale law，并逐个移除内部特征族，观察 expert scale、hard-cap 绑定、retention 和 objective proxy 如何变化。

- Status: `mechanistic_sensitivity_ready`
- Baseline candidate: `s0.08_b1.65_h0.75_i0.75`
- Effective hard cap: `0.6490`
- Baseline fixed objective: `0.1822`
- Baseline reselected objective: `0.1822`
- Strongest fixed objective regression: `no_category_prior` (`delta=0.0034`)
- Strongest scale sensitivity: `no_subspace_conflict` (`route-mass weighted abs shift=0.0086`)

## Feature-Family Ablations

| ablation | fixed obj delta | fixed retention delta | fixed risk-delta delta | changed >0.01 | cap-bound groups | reselected candidate | reselected obj delta |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `baseline` | 0.000000 | 0.000000 | 0.000000 | 0 | 30 | `s0.08_b1.65_h0.75_i0.75` | 0.000000 |
| `no_category_prior` | 0.003357 | -0.001544 | -0.000405 | 0 | 18 | `s0.04_b1.15_h1.25_i0.75` | 0.000780 |
| `no_subspace_conflict` | 0.002812 | 0.007942 | 0.002223 | 2282 | 52 | `s0.16_b1.65_h1.25_i0.75` | -0.000215 |
| `no_source_conflict` | 0.002718 | 0.007590 | 0.002014 | 1336 | 32 | `s0.16_b1.65_h1.25_i0.75` | -0.000379 |
| `no_router_boundary` | 0.002605 | 0.007332 | 0.002046 | 2431 | 45 | `s0.16_b1.65_h1.00_i0.75` | -0.000063 |
| `no_load_pressure` | 0.000668 | 0.002333 | 0.000624 | 3 | 34 | `s0.08_b1.65_h1.25_i0.75` | 0.000023 |
| `no_expert_geometry` | 0.000396 | 0.001096 | 0.000290 | 0 | 44 | `s0.08_b1.65_h1.00_i0.75` | 0.000133 |
| `no_delta_pressure` | 0.000234 | 0.000675 | 0.000168 | 7 | 51 | `s0.08_b1.65_h0.75_i0.75` | 0.000234 |
| `no_feedback_prior` | 0.000000 | 0.000000 | 0.000000 | 0 | 30 | `s0.08_b1.65_h0.75_i0.75` | 0.000000 |

## Top Shrink Correlations

| feature | family | corr with shrink | corr with restore | corr with predicted delta |
| --- | --- | ---: | ---: | ---: |
| `feature_layer_geometry` | `layer_geometry` | 0.6981 | -0.0407 | 0.0907 |
| `feature_expert_internal_geometry` | `expert_geometry` | 0.6585 | -0.0881 | -0.0680 |
| `feature_subspace_conflict` | `subspace_conflict` | 0.6518 | -0.0893 | -0.0732 |
| `feature_router_instability` | `router_boundary` | 0.6242 | -0.0311 | 0.0471 |
| `feature_subspace_route_conflict` | `subspace_conflict` | 0.4582 | -0.1016 | -0.0146 |
| `route_mass_pressure` | `benefit` | -0.3194 | -0.0802 | 0.1189 |
| `feature_low_route_mass` | `low_route_mass` | 0.3194 | 0.0802 | -0.1189 |
| `feature_expert_route_geometry` | `expert_geometry` | 0.3192 | -0.1118 | 0.0136 |
| `original_weight_coder` | `benefit` | -0.2153 | 0.3666 | 0.9696 |
| `category_coder_prior` | `benefit` | -0.2107 | 0.1583 | 0.3049 |
| `coder_route_share` | `benefit` | -0.2105 | 0.3662 | 0.9599 |
| `feature_load_pressure` | `load_pressure` | -0.1635 | -0.0003 | -0.0388 |

## Interpretation

这里的 objective delta 是 counterfactual full-score：先移除某类特征求 scale，再放回完整 B/H/I objective 里打分。正值表示该特征族保护了当前 proxy；负值表示移除后完整 proxy 反而更好，应作为下一轮 ablation hypothesis，不能直接当成下游结论。
`no_category_prior` 是当前最大的完整目标退化来源，含义是这类信号不是装饰性规则；它改变了同构 expert average 在 benefit、curvature、interference 三项之间的可行折中。
`no_subspace_conflict` 对 scale 分布最敏感，说明 unified 算法不能只保留全局 cap，还要让局部 expert/subspace 结构进入 trust-region。
这个结果不替代最终 vLLM selector；它只说明 B/H/I 内部哪些信号真正改变了 same-shape expert scales。

## Outputs

- `feature_family_ablation`: `results/qwen3_moe_mechanistic_sensitivity/feature_family_ablation.csv`
- `feature_correlations`: `results/qwen3_moe_mechanistic_sensitivity/feature_correlations.csv`
- `top_affected_groups`: `results/qwen3_moe_mechanistic_sensitivity/top_affected_groups.csv`
- `summary`: `results/qwen3_moe_mechanistic_sensitivity/summary.json`
- `report`: `results/qwen3_moe_mechanistic_sensitivity/report.md`
