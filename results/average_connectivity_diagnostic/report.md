# Average Connectivity Diagnostic

这个诊断把 Dense 和 MoE 的直线路径证据放到同一个判据里：一个 average 是否可以接受，不看方法名，而看它有没有越过 endpoint frontier、midpoint 是否安全、是否真的观测到任务互补，以及局部二阶近似是否足够可信。

## Current Result

- Cases: `6`
- Rejected path/family cases: `5`
- Rejected fixed-midpoint cases: `5`
- Endpoint-frontier wins: `1`
- Complementary cases observed: `0`
- Dense midpoint gap: `2.9168`
- Qwen3 MoE Instruct/Coder interior gap: `0.1189`

## Diagnostics

| case | domain | metric | endpoint best | best interior | endpoint gap | midpoint gap | barrier | complementarity | decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `dense_instruct_coder_source_path` | `dense` | `worst` | 2.9940 | 3.4092 | 0.4152 | 2.9168 | 2.0295 | `False` | `reject_interior_average` |
| `dense_base_anchored_lambda_family` | `dense` | `worst` | 3.1737 | 3.0727 | -0.1010 | 2.8661 | 2.8661 | `False` | `family_beats_endpoint_but_not_as_fixed_average` |
| `dense_fisher_local_quadratic_check` | `dense` | `actual_over_predicted_degradation` | n/a | n/a | n/a | 37.8604 | n/a | `False` | `reject_local_quadratic_as_sufficient_gate` |
| `qwen3_moe_instruct_coder_source_path` | `moe` | `worst` | 2.4757 | 2.5947 | 0.1189 | 0.4683 | 0.1097 | `False` | `reject_interior_average` |
| `qwen3_moe_base_coder_source_path` | `moe` | `worst` | 2.3603 | 2.4661 | 0.1058 | 0.5398 | 0.0728 | `False` | `reject_interior_average` |
| `qwen3_moe_thinking_coder_complementary_path` | `moe` | `avg` | 0.5659 | 0.5877 | 0.0218 | 0.0898 | 0.0000 | `False` | `reject_interior_average` |

## Rules

| rule | accept if | reject if | meaning |
| --- | --- | --- | --- |
| `endpoint_frontier_rule` | best interior score <= best endpoint score - 0.01 | best interior score is worse than the best endpoint frontier | A straight-line or coefficient average must beat the source frontier, not only look smooth. |
| `midpoint_safety_rule` | midpoint score <= best endpoint score + 0.01 | fixed 0.5/0.5 midpoint has a positive endpoint-frontier gap | Uniform averaging is a special case and should be rejected separately from coefficient search. |
| `complementarity_rule` | different endpoints win different tasks and an interior point beats the best source average | one endpoint dominates all measured tasks or the best interior only ties the best source | Specialist labels are not enough; complementarity must appear in measured task losses. |
| `local_quadratic_rule` | curvature prediction matches actual midpoint degradation within a small ratio | actual/predicted degradation ratio is large | Fisher/RegMean-style local arguments cannot justify a nonlocal merge by themselves. |

## Outputs

- `results/average_connectivity_diagnostic/path_diagnostics.csv`
- `results/average_connectivity_diagnostic/acceptance_rules.json`
- `results/average_connectivity_diagnostic/connectivity_gaps.png`
- `results/average_connectivity_diagnostic/summary.json`
- `results/average_connectivity_diagnostic/report.md`
