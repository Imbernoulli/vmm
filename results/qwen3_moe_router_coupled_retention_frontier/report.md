# Qwen3 MoE Router-Coupled Retention Frontier

This probe answers whether the router-boundary term should become part of the default unified average, not just whether an aggressive ablation can reduce a proxy. It reruns the router-coupled search with a much finer near-zero shrink grid and explicitly enforces the mechanistic retention gate.

- Status: `router_coupled_retention_frontier_ready`
- Gate: `direct_router_boundary_term_not_default`
- Recommended action: `keep_router_fragility_inside_BHI_and_keep_direct_extra_shrink_as_ablation`
- Base retention: `0.965035`
- Min retention gate: `0.965`
- Passing default-gate candidates: `146/770`
- Best constrained candidate: `router_q0.85_s0.00020_cap0.00010`
- Constrained retention / delta: `0.965003` / `-3.16543e-05`
- Constrained router-coupled reduction: `1.16455e-05`
- Stress candidate: `router_q0.75_s0.01000_cap0.01000`
- Stress retention / delta: `0.961943` / `-0.0030917`
- Stress router-coupled reduction: `0.00113497`
- Constrained/stress effect fraction: `0.0102607`

## Interpretation

The router-boundary signal is real, but the default mechanistic solution is already sitting almost exactly on the retention boundary. The strongest default-gate-safe direct router shrink has only a tiny proxy effect, while the first materially visible direct shrink violates the retention gate. So the default algorithm should keep router fragility inside B/H/I as an interference feature and leave direct extra shrink as an ablation unless downstream vLLM evidence proves the retention trade-off is worth it.

## Frontier

| candidate | pass default | retention | retention delta | changed | coupled reduction | risk reduction | strength | cap | q |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `router_q0.85_s0.00020_cap0.00010` | `True` | 0.965003 | -3.16543e-05 | 638 | 1.16455e-05 | 9.68567e-06 | 0.0002 | 0.0001 | 0.85 |
| `router_q0.85_s0.00030_cap0.00010` | `True` | 0.965003 | -3.16543e-05 | 638 | 1.16455e-05 | 9.68567e-06 | 0.0003 | 0.0001 | 0.85 |
| `router_q0.85_s0.00040_cap0.00010` | `True` | 0.965003 | -3.16543e-05 | 638 | 1.16455e-05 | 9.68567e-06 | 0.0004 | 0.0001 | 0.85 |
| `router_q0.85_s0.00050_cap0.00010` | `True` | 0.965003 | -3.16543e-05 | 638 | 1.16455e-05 | 9.68567e-06 | 0.0005 | 0.0001 | 0.85 |
| `router_q0.85_s0.00075_cap0.00010` | `True` | 0.965003 | -3.16543e-05 | 638 | 1.16455e-05 | 9.68567e-06 | 0.00075 | 0.0001 | 0.85 |
| `router_q0.85_s0.00100_cap0.00010` | `True` | 0.965003 | -3.16543e-05 | 638 | 1.16455e-05 | 9.68567e-06 | 0.001 | 0.0001 | 0.85 |
| `router_q0.85_s0.00200_cap0.00010` | `True` | 0.965003 | -3.16543e-05 | 638 | 1.16455e-05 | 9.68567e-06 | 0.002 | 0.0001 | 0.85 |
| `router_q0.85_s0.00300_cap0.00010` | `True` | 0.965003 | -3.16543e-05 | 638 | 1.16455e-05 | 9.68567e-06 | 0.003 | 0.0001 | 0.85 |
| `router_q0.85_s0.00400_cap0.00010` | `True` | 0.965003 | -3.16543e-05 | 638 | 1.16455e-05 | 9.68567e-06 | 0.004 | 0.0001 | 0.85 |
| `router_q0.85_s0.00500_cap0.00010` | `True` | 0.965003 | -3.16543e-05 | 638 | 1.16455e-05 | 9.68567e-06 | 0.005 | 0.0001 | 0.85 |
| `router_q0.75_s0.01000_cap0.01000` | `False` | 0.961943 | -0.0030917 | 972 | 0.00113497 | 0.000943053 | 0.01 | 0.01 | 0.75 |
| `router_q0.80_s0.01000_cap0.01000` | `False` | 0.962059 | -0.00297511 | 810 | 0.00109596 | 0.000912042 | 0.01 | 0.01 | 0.8 |
| `router_q0.85_s0.01000_cap0.01000` | `False` | 0.962242 | -0.00279284 | 638 | 0.00103417 | 0.000861287 | 0.01 | 0.01 | 0.85 |
| `router_q0.75_s0.01000_cap0.00750` | `False` | 0.96245 | -0.00258459 | 972 | 0.000943627 | 0.000783208 | 0.01 | 0.0075 | 0.75 |
| `router_q0.80_s0.01000_cap0.00750` | `False` | 0.962567 | -0.002468 | 810 | 0.000904617 | 0.000752197 | 0.01 | 0.0075 | 0.8 |
| `router_q0.90_s0.01000_cap0.01000` | `False` | 0.962633 | -0.00240195 | 436 | 0.000899167 | 0.000751526 | 0.01 | 0.01 | 0.9 |
| `router_q0.75_s0.00750_cap0.00750` | `False` | 0.962716 | -0.00231877 | 972 | 0.000851227 | 0.00070729 | 0.0075 | 0.0075 | 0.75 |
| `router_q0.75_s0.00750_cap0.01000` | `False` | 0.962716 | -0.00231877 | 972 | 0.000851227 | 0.00070729 | 0.0075 | 0.01 | 0.75 |
| `router_q0.85_s0.01000_cap0.00750` | `False` | 0.962749 | -0.00228573 | 638 | 0.000842823 | 0.000701442 | 0.01 | 0.0075 | 0.85 |
| `router_q0.80_s0.00750_cap0.00750` | `False` | 0.962803 | -0.00223133 | 810 | 0.00082197 | 0.000684032 | 0.0075 | 0.0075 | 0.8 |

## Constrained Changed Experts

| layer | expert | category | route mass | base scale | constrained scale | extra shrink | pressure |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20 | 38 | `general` | 1.1555 | 0.980817 | 0.980717 | 0.0001 | 0.659877 |
| 25 | 104 | `general` | 1.30002 | 0.979811 | 0.979711 | 0.0001 | 0.586042 |
| 18 | 8 | `code` | 0.965749 | 0.98854 | 0.98844 | 0.0001 | 0.510152 |
| 24 | 65 | `safety` | 0.847552 | 0.977682 | 0.977582 | 0.0001 | 0.440476 |
| 23 | 21 | `code` | 0.939255 | 0.984875 | 0.984775 | 0.0001 | 0.440292 |
| 21 | 10 | `code` | 1.03665 | 0.989278 | 0.989178 | 0.0001 | 0.437532 |
| 17 | 83 | `safety` | 0.794408 | 0.976242 | 0.976142 | 0.0001 | 0.417969 |
| 22 | 98 | `safety` | 0.747714 | 0.979868 | 0.979768 | 0.0001 | 0.407043 |
| 11 | 21 | `code` | 0.951697 | 0.982709 | 0.982609 | 0.0001 | 0.403694 |
| 15 | 10 | `code` | 0.714189 | 0.982615 | 0.982515 | 0.0001 | 0.387706 |
| 20 | 116 | `safety` | 0.673112 | 0.978627 | 0.978527 | 0.0001 | 0.381324 |
| 22 | 53 | `safety` | 0.665883 | 0.977985 | 0.977885 | 0.0001 | 0.365182 |

## Outputs

- `retention_frontier`: `results/qwen3_moe_router_coupled_retention_frontier/retention_frontier.csv`
- `retention_constrained_group_rules`: `results/qwen3_moe_router_coupled_retention_frontier/retention_constrained_group_rules.csv`
- `selected_retention_constrained_candidate`: `results/qwen3_moe_router_coupled_retention_frontier/selected_retention_constrained_candidate.json`
- `selected_stress_candidate`: `results/qwen3_moe_router_coupled_retention_frontier/selected_stress_candidate.json`
- `tensor_rules`: `results/qwen3_moe_router_coupled_retention_frontier/tensor_rules.txt`
- `writer_command`: `results/qwen3_moe_router_coupled_retention_frontier/writer_command.txt`
- `dry_run_command`: `results/qwen3_moe_router_coupled_retention_frontier/dry_run_command.txt`
- `checkpoint_output_dir`: `results/checkpoints/qwen3_moe_router_coupled_retention_constrained_candidate`
- `summary`: `results/qwen3_moe_router_coupled_retention_frontier/summary.json`
- `report`: `results/qwen3_moe_router_coupled_retention_frontier/report.md`
