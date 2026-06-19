# Qwen3 MoE Unified Mechanism Candidate

这个实验把“Average”写成同结构约束优化，而不是命名算法选择：先冻结高风险 router，保持 expert identity，再用真实 route mass、router fragility、load、source conflict 和 safetensors delta probe 生成 per-expert 缩放。

## Result

- Status: `unified_mechanism_candidate_ready`
- Expert groups: `5243`
- Searched candidates: `10`
- Selected candidate: `uniform_0.65`
- Selection family: `threshold_efficient_cap`
- Nonbase route-mass retention: `0.9823`
- Max predicted routed relative delta: `0.6500`
- Groups over hard cap `0.65`: `0`
- Matches validated no-gt-0.65 rules: `True`

## Why This Is The Current Unified Rule

理论上，uniform average、task-vector merge、TIES、Fisher/RegMean 都可以看成在同一个参数空间里求一个同结构解；真正的区别是约束和局部几何假设。对当前 Qwen3 MoE，最强的内部证据不是“某个算法名更好”，而是 router/top-k dispatch 对扰动敏感、expert identity 必须先固定、routed expert 的 high-delta tail 必须被限制。

因此本脚本求的是：在不改结构、不改 router、不增加 expert 的条件下，最大化 route-mass-weighted Coder contribution，同时让 routed expert 的预测 relative-delta tail 不超过 hard cap。更复杂的 risk-dependent cap 也参与搜索；如果它只降低 retention 而不进一步降低 hard-tail violation，就被自动拒绝。

## Candidate Search

| candidate | family | pass cap | retention | norm ratio | max rel-delta | risk-weighted rel-delta | objective |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `uniform_0.65` | `threshold_efficient_cap` | `True` | 0.9823 | 0.8738 | 0.6500 | 0.2492 | 1.1487 |
| `load_aware_risk_s0.25` | `load_weighted_risk` | `True` | 0.9807 | 0.8681 | 0.6446 | 0.2486 | 1.1621 |
| `smooth_risk_s0.25` | `continuous_mechanism_risk` | `True` | 0.9804 | 0.8658 | 0.6410 | 0.2486 | 1.1629 |
| `router_evidence_risk_s0.25` | `router_and_evidence_weighted_risk` | `True` | 0.9803 | 0.8648 | 0.6409 | 0.2485 | 1.1635 |
| `load_aware_risk_s0.50` | `load_weighted_risk` | `True` | 0.9789 | 0.8624 | 0.6392 | 0.2480 | 1.1770 |
| `smooth_risk_s0.50` | `continuous_mechanism_risk` | `True` | 0.9785 | 0.8577 | 0.6320 | 0.2479 | 1.1782 |
| `router_evidence_risk_s0.50` | `router_and_evidence_weighted_risk` | `True` | 0.9782 | 0.8556 | 0.6318 | 0.2478 | 1.1794 |
| `smooth_risk_s0.75` | `continuous_mechanism_risk` | `True` | 0.9763 | 0.8493 | 0.6250 | 0.2471 | 1.1959 |
| `router_evidence_risk_s0.75` | `router_and_evidence_weighted_risk` | `True` | 0.9758 | 0.8460 | 0.6234 | 0.2470 | 1.1976 |
| `smooth_risk_s1.00` | `continuous_mechanism_risk` | `True` | 0.9740 | 0.8407 | 0.6177 | 0.2464 | 1.2148 |

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
