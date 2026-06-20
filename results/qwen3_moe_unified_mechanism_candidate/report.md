# Qwen3 MoE Unified Mechanism Candidate

这个实验把“Average”写成同结构约束优化，而不是命名算法选择：先冻结高风险 router，保持 expert identity，再用真实 route mass、router fragility、load、source conflict 和 safetensors delta probe 生成 per-expert 缩放。

## Result

- Status: `unified_mechanism_candidate_ready`
- Expert groups: `5243`
- Searched candidates: `19`
- Selected candidate: `router_evidence_risk_s0.75`
- Selection family: `router_and_evidence_weighted_risk`
- Nonbase route-mass retention: `0.9758`
- Max predicted routed relative delta: `0.6234`
- Groups over hard cap `0.65`: `0`
- Risk-weighted predicted relative delta: `0.2273`
- Geometry-weighted predicted relative delta: `0.2184`
- Expert geometry used: `True`

## Why This Is The Current Unified Rule

理论上，uniform average、task-vector merge、TIES、Fisher/RegMean 都可以看成在同一个参数空间里求一个同结构解；真正的区别是约束和局部几何假设。对当前 Qwen3 MoE，最强的内部证据不是“某个算法名更好”，而是 router/top-k dispatch 对扰动敏感、expert identity 必须先固定、routed expert 的 high-delta tail 和 expert 内部几何风险必须被同时限制。

因此本脚本求的是：在不改结构、不改 router、不增加 expert 的条件下，保留足够的 route-mass-weighted Coder contribution，同时最小化 route/risk/geometry weighted predicted delta。旧的 uniform `0.65` cap 仍作为 baseline 参与搜索；如果 layer/geometry-aware prior 在 retention 约束内降低高风险 expert 的移动，它会被选中。

## Candidate Search

| candidate | family | pass cap | retention | norm ratio | max rel-delta | risk rel-delta | geom rel-delta | objective |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `uniform_0.65` | `threshold_efficient_cap` | `True` | 0.9823 | 0.8738 | 0.6500 | 0.2290 | 0.2198 | 1.4661 |
| `geometry_cap_s0.25` | `geometry_weighted_cap` | `True` | 0.9807 | 0.8671 | 0.6450 | 0.2286 | 0.2194 | 1.4712 |
| `router_evidence_risk_s0.25` | `router_and_evidence_weighted_risk` | `True` | 0.9803 | 0.8647 | 0.6410 | 0.2285 | 0.2193 | 1.4722 |
| `smooth_risk_s0.25` | `continuous_mechanism_risk` | `True` | 0.9804 | 0.8657 | 0.6417 | 0.2285 | 0.2194 | 1.4722 |
| `load_aware_risk_s0.25` | `load_weighted_risk` | `True` | 0.9807 | 0.8680 | 0.6450 | 0.2286 | 0.2194 | 1.4726 |
| `geometry_cap_s0.50` | `geometry_weighted_cap` | `True` | 0.9791 | 0.8603 | 0.6400 | 0.2281 | 0.2191 | 1.4768 |
| `router_evidence_risk_s0.50` | `router_and_evidence_weighted_risk` | `True` | 0.9781 | 0.8553 | 0.6321 | 0.2279 | 0.2189 | 1.4789 |
| `smooth_risk_s0.50` | `continuous_mechanism_risk` | `True` | 0.9784 | 0.8573 | 0.6335 | 0.2280 | 0.2190 | 1.4790 |
| `load_aware_risk_s0.50` | `load_weighted_risk` | `True` | 0.9789 | 0.8621 | 0.6401 | 0.2281 | 0.2191 | 1.4799 |
| `geometry_cap_s0.75` | `geometry_weighted_cap` | `True` | 0.9773 | 0.8533 | 0.6351 | 0.2276 | 0.2187 | 1.4832 |
| `router_evidence_risk_s0.75` | `router_and_evidence_weighted_risk` | `True` | 0.9758 | 0.8457 | 0.6234 | 0.2273 | 0.2184 | 1.4869 |
| `smooth_risk_s0.75` | `continuous_mechanism_risk` | `True` | 0.9761 | 0.8487 | 0.6262 | 0.2274 | 0.2185 | 1.4872 |

## Mechanism Constraints

- Expert identity: `same-name expert average only after identity gate`; current Qwen3 gate allows identity expert rules.
- Router: `freeze_router`; router calibration remains a separately gated ablation.
- Shared attention: frozen in this candidate because delta frontier says attention utility needs downstream eval, not norm-only evidence.
- Endpoint fallback: downstream selector must still reject this candidate if source endpoints dominate it.

## Literature Priors

- `mode_connectivity`: https://arxiv.org/abs/1802.10026
- `model_soups`: https://arxiv.org/abs/2203.05482
- `fisher_merging`: https://arxiv.org/abs/2111.09832
- `ties`: https://arxiv.org/abs/2306.01708
- `git_rebasin`: https://arxiv.org/abs/2209.04836
- `moe_routing_breakdown`: https://arxiv.org/abs/2606.03391

## Outputs

- `report`: `results/qwen3_moe_unified_mechanism_candidate/report.md`
- `summary`: `results/qwen3_moe_unified_mechanism_candidate/summary.json`
- `selected_candidate`: `results/qwen3_moe_unified_mechanism_candidate/selected_candidate.json`
- `candidate_search`: `results/qwen3_moe_unified_mechanism_candidate/candidate_search.csv`
- `unified_group_rules`: `results/qwen3_moe_unified_mechanism_candidate/unified_group_rules.csv`
- `tensor_rules`: `results/qwen3_moe_unified_mechanism_candidate/tensor_rules.txt`
- `writer_command`: `results/qwen3_moe_unified_mechanism_candidate/writer_command.txt`
- `dry_run_command`: `results/qwen3_moe_unified_mechanism_candidate/dry_run_command.txt`
- `checkpoint_output_dir`: `results/checkpoints/qwen3_moe_unified_mechanism_candidate`
