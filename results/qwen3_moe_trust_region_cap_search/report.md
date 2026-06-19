# Qwen3 MoE Trust-Region Cap-Law 搜索

这个实验不是静态比较算法名，而是对 Qwen3 MoE 合并规则本身做一次内部参数优化：在真实 route mass、risk flag 和 safetensors delta probe 上搜索可解释的 expert cap law。它不替代 vLLM 下游评测；它的作用是把“为什么要这么合并”转成可检查、可物化的规则。

## 结果

- 状态：`cap_law_search_ready`
- 真实 expert groups：`5243`
- 搜索 cap laws：`432`
- Pareto frontier laws：`88`
- 选中的 no `>0.75` law：`grid_00000_b0.75_m0.60_h0.00_s0.00_r0.00_l0.00_c0.00`
- 选中的 no `>0.65` law：`grid_00288_b0.65_m0.60_h0.00_s0.00_r0.00_l0.00_c0.00`
- 当前 trust-region retention：`0.9818`，仍有 `129` 个 group 高于 `0.65`
- 简单 uniform `0.65` cap retention：`0.9823`，高于 `0.65` 的 group 为 `0`

## 为什么要做这个 Probe

对 routed expert group `g`，当前 same-shape 合并规则可以近似写成：

```text
theta_out[g] = theta_base[g] + s_g * w_g * (theta_coder[g] - theta_base[g])
s_g = min(1, cap_g / relative_delta_g)
cap_g = base_cap - penalties(route_load, mixed_source, router_fragility, low_evidence, category_mismatch)
```

`w_g` 是 route/source 规则给 Coder delta 的原始权重，`s_g` 是 delta trust region 给它加的缩放。搜索目标不是下游分数，而是一个安全/效用代理：尽量压低 routed experts 里的高 relative-delta tail，同时尽量保留 route-mass-weighted 的 Coder contribution。

主要发现是：当前手写 risk penalties 自身不是 delta-threshold efficient。简单 uniform `0.65` cap 可以去掉剩余高 tail，而且 route-weighted nonbase mass retention 还略高于当前 trust-region 规则。这并不证明它下游一定更好；它证明下一轮 vLLM gate 应该把“简单 tail cap”和“更复杂的风险标记 law”放到同一组 source/candidate eval 里判定。

## 选中的规则

| role | law | >0.75 groups | >0.65 groups | retention | norm ratio | max rel-delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `reference` | `route_guarded_uncapped_expert_rules` | 302 | 405 | 1.0000 | 1.0000 | 1.3272 |
| `reference` | `current_trust_region_cap_law` | 0 | 129 | 0.9818 | 0.8683 | 0.7500 |
| `reference` | `uniform_065_tail_cap` | 0 | 0 | 0.9823 | 0.8738 | 0.6500 |
| `searched_no_gt075_max_retention` | `grid_00000_b0.75_m0.60_h0.00_s0.00_r0.00_l0.00_c0.00` | 0 | 405 | 0.9913 | 0.9224 | 0.7500 |
| `searched_no_gt065_max_retention` | `grid_00288_b0.65_m0.60_h0.00_s0.00_r0.00_l0.00_c0.00` | 0 | 0 | 0.9823 | 0.8738 | 0.6500 |
| `searched_min_internal_risk_score` | `grid_00288_b0.65_m0.60_h0.00_s0.00_r0.00_l0.00_c0.00` | 0 | 0 | 0.9823 | 0.8738 | 0.6500 |

## Risk-Flag Ablation

这里检查每个风险标记单独加 `0.05` penalty 时，是否真的减少高 delta tail。结果显示它们主要只是降低 norm/retention，不能单独减少 `>0.75` 或 `>0.65` group，因此不能只凭这些 flag 就说复杂 law 更优。

| flag | groups | extra scaled | >0.75 reduction | >0.65 reduction | retention loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shared_mixed` | 2218 | 2 | 0 | 0 | 0.0000 |
| `category_mismatch` | 152 | 52 | 0 | 0 | 0.0005 |
| `low_route_evidence` | 984 | 180 | 0 | 0 | 0.0008 |
| `high_load` | 1650 | 53 | 0 | 0 | 0.0013 |
| `fragile_router` | 2632 | 214 | 0 | 0 | 0.0025 |

## Pareto Frontier 样例

| law | >0.75 | >0.65 | retention | norm ratio | internal risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| `grid_00288_b0.65_m0.60_h0.00_s0.00_r0.00_l0.00_c0.00` | 0 | 0 | 0.9823 | 0.8738 | 62.2281 |
| `grid_00360_b0.65_m0.65_h0.00_s0.00_r0.00_l0.00_c0.00` | 0 | 0 | 0.9823 | 0.8738 | 62.2281 |
| `grid_00361_b0.65_m0.65_h0.00_s0.00_r0.00_l0.00_c0.05` | 0 | 0 | 0.9823 | 0.8738 | 62.2281 |
| `grid_00362_b0.65_m0.65_h0.00_s0.00_r0.00_l0.05_c0.00` | 0 | 0 | 0.9823 | 0.8738 | 62.2281 |
| `grid_00323_b0.65_m0.60_h0.05_s0.00_r0.05_l0.05_c0.05` | 0 | 0 | 0.9763 | 0.8492 | 62.3226 |
| `grid_00347_b0.65_m0.60_h0.10_s0.00_r0.05_l0.05_c0.05` | 0 | 0 | 0.9763 | 0.8492 | 62.3226 |
| `grid_00335_b0.65_m0.60_h0.05_s0.03_r0.05_l0.05_c0.05` | 0 | 0 | 0.9763 | 0.8491 | 62.3235 |
| `grid_00359_b0.65_m0.60_h0.10_s0.03_r0.05_l0.05_c0.05` | 0 | 0 | 0.9763 | 0.8491 | 62.3235 |
| `grid_00251_b0.70_m0.65_h0.05_s0.00_r0.05_l0.05_c0.05` | 0 | 47 | 0.9829 | 0.8767 | 302.2192 |
| `grid_00275_b0.70_m0.65_h0.10_s0.00_r0.05_l0.05_c0.05` | 0 | 47 | 0.9829 | 0.8767 | 302.2192 |
| `grid_00263_b0.70_m0.65_h0.05_s0.03_r0.05_l0.05_c0.05` | 0 | 47 | 0.9828 | 0.8767 | 302.2197 |
| `grid_00287_b0.70_m0.65_h0.10_s0.03_r0.05_l0.05_c0.05` | 0 | 47 | 0.9828 | 0.8767 | 302.2197 |

## 文献连接

- HARC (2026) 指出 MoE routing breakdown 会来自 softmax/top-k router 扰动；这里先冻结 router，把 expert delta 单独优化： https://arxiv.org/abs/2606.03391
- Expert Merging (2025) 用 unlabeled calibration behavior 学 layer/chunk-wise coefficients；这里对应到 layer/expert 粒度的参数审计和 cap-law 搜索： https://arxiv.org/abs/2509.25712
- 近期 LLM model-merging 系统研究报告很多通用 merge 算法在 LLM 上会失败，所以这里保留 endpoint fallback 和 vLLM gate： https://arxiv.org/abs/2511.21437
- Sub-MoE 用 expert-output similarity/subspace 做 expert 合并/压缩，这支持把 expert identity/subspace probe 和 router probe 分开处理： https://arxiv.org/abs/2506.23266

## 输出

- `report`: `results/qwen3_moe_trust_region_cap_search/report.md`
- `summary`: `results/qwen3_moe_trust_region_cap_search/summary.json`
- `cap_law_search`: `results/qwen3_moe_trust_region_cap_search/cap_law_search.csv`
- `reference_laws`: `results/qwen3_moe_trust_region_cap_search/reference_laws.csv`
- `pareto_frontier`: `results/qwen3_moe_trust_region_cap_search/pareto_frontier.csv`
- `selected_cap_laws`: `results/qwen3_moe_trust_region_cap_search/selected_cap_laws.csv`
- `risk_flag_ablation`: `results/qwen3_moe_trust_region_cap_search/risk_flag_ablation.csv`
- `selected_rule_artifacts`: `results/qwen3_moe_trust_region_cap_search/selected_rule_artifacts.json`
