# MoE Same-Shape Average Plan

Generated at: `2026-06-19T03:52:20.350136+00:00`

这个报告把 Dense/Qwen Average 决策、MoE router probe 和 expert load probe 转成同构 MoE checkpoint 的合并计划。这里的同构约束是硬约束：不增加 experts，不改 router shape，不做运行时 ensemble；所有策略最后都要写回原模型结构。

## 全局判断

- Dense/Qwen decision verdict: `avoid_uniform_average`。
- 默认平均权重来源：`validation_grid_or_layerwise_coefficients`。
- `avoid_uniform_average` 数量：`3`；`coefficient_search` 数量：`2`。

## 参数组计划

| parameter group | same-shape action | weight source | reason |
| --- | --- | --- | --- |
| embedding / lm_head | `anchor_or_fisher_weighted_average` | general_retention_and_token_nll | 这些张量控制全局 token 分布；只有 general retention 不下降时才平均。 |
| shared attention / norms | `coefficient_search_or_layerwise_average` | validation_grid_or_layerwise_coefficients | 现有 Dense/Qwen probe 显示均匀平均可能穿过高 loss 区域。 |
| shared dense MLP | `conflict_aware_coordinate_average` | weighted_sign_conflict + Fisher/activation statistics | MLP delta 往往包含专长冲突；冲突高时用 TIES/DELLA/Fisher 风格的 coordinate weighting。 |
| MoE router | `router_frozen_until_probe` | router_entropy + load_balance + route_overlap | 没有传入 MoE routing probe。 |
| MoE experts | `expert_matched_route_frequency_average` | expert output similarity + route frequency + NLL sensitivity | 保持 expert 数和 tensor shape 不变；如果 expert index 语义不确定，先匹配再平均。 |
| LoRA / adapters | `adapter_delta_average_or_distill_back` | rank overlap + adapter output similarity | mixture 可作上界，但最终 artifact 应压回一个同构 adapter/checkpoint。 |

## Router Probe Summary

当前没有传入 MoE routing probe 输出；router 策略默认是 `router_frozen_until_probe`。

## Router Plan

没有 router-level CSV。运行 `scripts/probe_moe_routing.py` 后用 `--router-dir` 传入输出目录。

## Expert Plan

没有 expert-load CSV。运行 `scripts/probe_moe_routing.py` 后会生成 per-expert route-frequency plan。

## 研究依据到实现规则

- HARC / routing-breakdown 方向说明 MoE router 对 softmax/top-k 扰动非常敏感，所以 router 不应默认同权平均。
- Sub-MoE 方向说明 expert specialization 会造成参数冲突，因此要先按输出相似度/路由频率聚类或匹配，再做 expert-wise average。
- Expert Merging / layer-wise coefficient 方向说明不同层的重要性不同，因此 shared attention、MLP、router、expert FFN 应分组设权重。
- MergeME/WEMoE 类方法可以作为上界或启发，但本项目最终仍要压回同构 checkpoint。

## Materialization

选好全局、layer/module、router 和 expert 权重后，用 `scripts/write_same_shape_average_checkpoint.py` 写出 safetensors checkpoint。writer 会先验证 tensor name/shape，再按 `theta_base + sum_i w_i * (theta_i - theta_base)` materialize；对于 MoE，默认建议先 `--freeze-router`，再根据 routing probe 决定是否开放小 lambda router average。

## Files

- `results/moe_average_plan/parameter_group_plan.csv`
- `results/moe_average_plan/router_plan.csv`
- `results/moe_average_plan/expert_plan.csv`
- `results/moe_average_plan/summary.json`
