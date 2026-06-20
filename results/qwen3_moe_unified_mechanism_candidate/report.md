# Qwen3 MoE Unified Mechanism Candidate

这个实验把“Average”写成同结构约束优化，而不是命名算法选择：先冻结高风险 router，保持 expert identity，再用真实 route mass、router fragility、load、source conflict、safetensors delta 和 expert subspace conflict probe 生成 per-expert 缩放。

## Result

- Status: `unified_mechanism_candidate_ready`
- Expert groups: `5243`
- Searched candidates: `28`
- Selected candidate: `subspace_cap_s1.00`
- Selection family: `subspace_weighted_cap`
- Nonbase route-mass retention: `0.9763`
- Max predicted routed relative delta: `0.6438`
- Groups over hard cap `0.65`: `0`
- Risk-weighted predicted relative delta: `0.2247`
- Geometry-weighted predicted relative delta: `0.2184`
- Subspace-weighted predicted relative delta: `0.2148`
- Expert geometry used: `True`
- Subspace conflict probe used: `True`
- High-subspace-conflict mean scale: `0.9614`
- Materialized checkpoint rule status: `stale_or_different_rules`

## Why This Is The Current Unified Rule

理论上，uniform average、task-vector merge、TIES、Fisher/RegMean 都可以看成在同一个参数空间里求一个同结构解；真正的区别是约束和局部几何假设。对当前 Qwen3 MoE，最强的内部证据不是“某个算法名更好”，而是 router/top-k dispatch 对扰动敏感、expert identity 必须先固定、routed expert 的 high-delta tail、expert 内部几何风险和局部子空间冲突必须被同时限制。

因此本脚本求的是：在不改结构、不改 router、不增加 expert 的条件下，保留足够的 route-mass-weighted Coder contribution，同时最小化 route/risk/geometry/subspace weighted predicted delta。旧的 uniform `0.65` cap 仍作为 baseline 参与搜索；如果 layer/geometry/subspace-aware prior 在 retention 约束内降低高风险 expert 的移动，它会被选中。

## Candidate Search

| candidate | family | pass cap | retention | norm ratio | max rel-delta | risk rel-delta | geom rel-delta | subspace rel-delta | objective |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uniform_0.65` | `threshold_efficient_cap` | `True` | 0.9823 | 0.8738 | 0.6500 | 0.2263 | 0.2198 | 0.2163 | 1.6576 |
| `subspace_probe_prior` | `subspace_conflict_probe_prior` | `True` | 0.9821 | 0.8736 | 0.6500 | 0.2263 | 0.2197 | 0.2163 | 1.6588 |
| `subspace_probe_cap` | `subspace_conflict_probe_cap` | `True` | 0.9806 | 0.8658 | 0.6500 | 0.2258 | 0.2193 | 0.2157 | 1.6612 |
| `subspace_cap_s0.25` | `subspace_weighted_cap` | `True` | 0.9809 | 0.8676 | 0.6485 | 0.2260 | 0.2195 | 0.2160 | 1.6614 |
| `geometry_cap_s0.25` | `geometry_weighted_cap` | `True` | 0.9807 | 0.8671 | 0.6450 | 0.2259 | 0.2194 | 0.2160 | 1.6624 |
| `smooth_risk_s0.25` | `continuous_mechanism_risk` | `True` | 0.9805 | 0.8660 | 0.6431 | 0.2259 | 0.2194 | 0.2159 | 1.6630 |
| `router_evidence_risk_s0.25` | `router_and_evidence_weighted_risk` | `True` | 0.9803 | 0.8649 | 0.6419 | 0.2258 | 0.2194 | 0.2159 | 1.6631 |
| `load_aware_risk_s0.25` | `load_weighted_risk` | `True` | 0.9807 | 0.8683 | 0.6458 | 0.2259 | 0.2194 | 0.2160 | 1.6635 |
| `subspace_cap_s0.50` | `subspace_weighted_cap` | `True` | 0.9795 | 0.8613 | 0.6469 | 0.2256 | 0.2191 | 0.2156 | 1.6655 |
| `geometry_cap_s0.50` | `geometry_weighted_cap` | `True` | 0.9791 | 0.8603 | 0.6400 | 0.2255 | 0.2191 | 0.2156 | 1.6677 |
| `smooth_risk_s0.50` | `continuous_mechanism_risk` | `True` | 0.9786 | 0.8582 | 0.6361 | 0.2254 | 0.2190 | 0.2155 | 1.6688 |
| `router_evidence_risk_s0.50` | `router_and_evidence_weighted_risk` | `True` | 0.9783 | 0.8558 | 0.6337 | 0.2253 | 0.2189 | 0.2154 | 1.6691 |

Selected row outside the top objective slice:

| candidate | family | pass cap | retention | norm ratio | max rel-delta | risk rel-delta | geom rel-delta | subspace rel-delta | objective |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `subspace_cap_s1.00` | `subspace_weighted_cap` | `True` | 0.9763 | 0.8483 | 0.6438 | 0.2247 | 0.2184 | 0.2148 | 1.6753 |

## Mechanism Constraints

- Expert identity: `same-name expert average only after identity gate`; current Qwen3 gate allows identity expert rules.
- Router: `freeze_router`; router calibration remains a separately gated ablation.
- Shared attention: frozen in this candidate because delta frontier says attention utility needs downstream eval, not norm-only evidence.
- Endpoint fallback: downstream selector must still reject this candidate if source endpoints dominate it.
- Materialization freshness: `fresh` means the existing checkpoint manifest was written from the current tensor rules; `stale_or_different_rules` means rerun the writer and delta audit before treating checkpoint metrics as final.

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
