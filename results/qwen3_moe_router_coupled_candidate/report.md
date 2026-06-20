# Qwen3 MoE Router-Coupled Candidate

这个候选不是替换当前 mechanistic unified；它是一个显式 ablation target。做法是从 `s0.08_b1.65_h0.75_i0.75` 的 B/H/I expert scale 出发，只对 router-coupled risk 最高的 experts 施加额外小幅 shrink，用来验证 MoE 的离散 dispatch 边界是否需要比当前默认规则更保守。

- Status: `router_coupled_candidate_ready`
- Selection gate: `ablation_only_waiting_vllm`
- Selected candidate: `router_q0.75_s0.0100_cap0.0100`
- Changed groups: `972`
- Nonbase retention: `0.9619` (`delta=-0.0031`)
- Max predicted relative delta: `0.6490`
- Router-coupled delta reduction: `0.0011`
- Risk delta reduction: `0.0009`
- Mean extra shrink on changed groups: `0.0069`
- Writer candidate kind: `same_shape_router_frozen_expert_ablation`

## Candidate Search

| candidate | retention | retention delta | changed | coupled delta reduction | risk reduction | pass retention |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `router_q0.75_s0.0100_cap0.0100` | 0.9619 | -0.0031 | 972 | 0.0011 | 0.0009 | `False` |
| `router_q0.80_s0.0100_cap0.0100` | 0.9621 | -0.0030 | 810 | 0.0011 | 0.0009 | `False` |
| `router_q0.85_s0.0100_cap0.0100` | 0.9622 | -0.0028 | 638 | 0.0010 | 0.0009 | `False` |
| `router_q0.75_s0.0100_cap0.0075` | 0.9624 | -0.0026 | 972 | 0.0009 | 0.0008 | `False` |
| `router_q0.80_s0.0100_cap0.0075` | 0.9626 | -0.0025 | 810 | 0.0009 | 0.0008 | `False` |
| `router_q0.90_s0.0100_cap0.0100` | 0.9626 | -0.0024 | 436 | 0.0009 | 0.0008 | `False` |
| `router_q0.75_s0.0075_cap0.0075` | 0.9627 | -0.0023 | 972 | 0.0009 | 0.0007 | `False` |
| `router_q0.75_s0.0075_cap0.0100` | 0.9627 | -0.0023 | 972 | 0.0009 | 0.0007 | `False` |
| `router_q0.85_s0.0100_cap0.0075` | 0.9627 | -0.0023 | 638 | 0.0008 | 0.0007 | `False` |
| `router_q0.80_s0.0075_cap0.0075` | 0.9628 | -0.0022 | 810 | 0.0008 | 0.0007 | `False` |
| `router_q0.80_s0.0075_cap0.0100` | 0.9628 | -0.0022 | 810 | 0.0008 | 0.0007 | `False` |
| `router_q0.85_s0.0075_cap0.0075` | 0.9629 | -0.0021 | 638 | 0.0008 | 0.0006 | `False` |
| `router_q0.85_s0.0075_cap0.0100` | 0.9629 | -0.0021 | 638 | 0.0008 | 0.0006 | `False` |
| `router_q0.90_s0.0100_cap0.0075` | 0.9631 | -0.0019 | 436 | 0.0007 | 0.0006 | `False` |

## Top Changed Experts

| layer | expert | category | route mass | base scale | coupled scale | extra shrink | pressure | reason |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | 38 | `general` | 1.1555 | 0.9808 | 0.9708 | 0.0100 | 0.6599 | `route_important_fragile_boundary` |
| 25 | 104 | `general` | 1.3000 | 0.9798 | 0.9698 | 0.0100 | 0.5860 | `route_important_fragile_boundary` |
| 18 | 8 | `code` | 0.9657 | 0.9885 | 0.9785 | 0.0100 | 0.5102 | `route_important_fragile_boundary` |
| 24 | 65 | `safety` | 0.8476 | 0.9777 | 0.9677 | 0.0100 | 0.4405 | `route_important_fragile_boundary` |
| 23 | 21 | `code` | 0.9393 | 0.9849 | 0.9749 | 0.0100 | 0.4403 | `route_important_fragile_boundary` |
| 21 | 10 | `code` | 1.0366 | 0.9893 | 0.9793 | 0.0100 | 0.4375 | `route_important_fragile_boundary` |
| 17 | 83 | `safety` | 0.7944 | 0.9762 | 0.9662 | 0.0100 | 0.4180 | `route_important_fragile_boundary` |
| 22 | 98 | `safety` | 0.7477 | 0.9799 | 0.9699 | 0.0100 | 0.4070 | `route_important_fragile_boundary` |
| 11 | 21 | `code` | 0.9517 | 0.9827 | 0.9727 | 0.0100 | 0.4037 | `route_important_fragile_boundary` |
| 15 | 10 | `code` | 0.7142 | 0.9826 | 0.9726 | 0.0100 | 0.3877 | `route_important_fragile_boundary` |
| 20 | 116 | `safety` | 0.6731 | 0.9786 | 0.9686 | 0.0100 | 0.3813 | `route_important_fragile_boundary` |
| 22 | 53 | `safety` | 0.6659 | 0.9780 | 0.9680 | 0.0100 | 0.3652 | `route_important_fragile_boundary` |
| 14 | 107 | `code` | 0.6818 | 0.9857 | 0.9757 | 0.0100 | 0.3539 | `route_important_fragile_boundary` |
| 20 | 46 | `code` | 0.5983 | 0.9855 | 0.9755 | 0.0100 | 0.3407 | `route_important_fragile_boundary` |
| 19 | 109 | `math` | 0.6630 | 0.9869 | 0.9769 | 0.0100 | 0.3376 | `route_important_fragile_boundary` |
| 26 | 107 | `code` | 0.8161 | 0.9892 | 0.9792 | 0.0100 | 0.3232 | `route_important_fragile_boundary` |

## Interpretation

这个 ablation 降低了 router-coupled risk proxy，但 retention 低于当前 mechanistic solver 的 `0.965` gate，因此不应成为默认 unified candidate。它的用途是进入 mechanism/ablation eval queue，检验额外 router-boundary 保守性是否能换来真实下游稳健性。
如果 vLLM 下游评测没有收益，应保持当前 B/H/I；如果收益显著，再把 router-coupled shrink 作为可条件开启的 MoE-specific term。

## Outputs

- `candidate_search`: `results/qwen3_moe_router_coupled_candidate/candidate_search.csv`
- `router_coupled_group_rules`: `results/qwen3_moe_router_coupled_candidate/router_coupled_group_rules.csv`
- `selected_candidate`: `results/qwen3_moe_router_coupled_candidate/selected_candidate.json`
- `tensor_rules`: `results/qwen3_moe_router_coupled_candidate/tensor_rules.txt`
- `writer_command`: `results/qwen3_moe_router_coupled_candidate/writer_command.txt`
- `dry_run_command`: `results/qwen3_moe_router_coupled_candidate/dry_run_command.txt`
- `checkpoint_output_dir`: `results/checkpoints/qwen3_moe_router_coupled_candidate`
- `summary`: `results/qwen3_moe_router_coupled_candidate/summary.json`
- `report`: `results/qwen3_moe_router_coupled_candidate/report.md`
