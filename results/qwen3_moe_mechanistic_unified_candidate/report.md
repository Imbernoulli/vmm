# Qwen3 MoE Mechanistic Unified Candidate

这个候选不再把 Average 写成“选某个算法名”。它把同结构 MoE 平均写成每个 routed expert group 的近似二次目标：

`J_g(s) = 1/2 H_g ||delta_g||^2 s^2 + I_g s^2 - B_g s`

其中 `B_g` 来自 route mass、Coder route share、source specialization 和当前非 base 权重；`H_g` 来自 delta pressure、expert geometry、layer geometry、router fragility 和 subspace conflict；`I_g` 来自 source conflict、fragile top-k boundary、high-load/shared experts 和 category mismatch。求出的 `s` 是同结构规则里的非 base scale，并继续受 hard-cap 与 vLLM feedback gate 约束。

## Result

- Status: `mechanistic_unified_candidate_ready`
- Expert groups: `5243`
- Selected candidate: `s0.04_b1.65_h0.75_i0.75`
- Search points: `144`
- Nominal hard cap: `0.6500`
- Materialization safety margin: `0.0010`
- Effective solver hard cap: `0.6490`
- Nonbase route-mass retention: `0.9621`
- Max predicted routed relative delta: `0.6490`
- Hard-cap violations: `0`
- Risk-weighted predicted delta: `0.2214`
- Benefit-weighted scale: `0.9705`
- Mean mechanistic loss proxy: `0.0122`
- Mean scale: `0.9524`
- High-benefit low-risk mean scale: `0.8615`
- High-interference low-benefit mean scale: `1.0000`
- High-subspace-conflict mean scale: `0.9194`
- Writer manifest validated: `True`
- Writer manifest dry-run: `False`
- Writer tensor rules / hits: `5243` / `15729`
- Writer freeze-router hits: `48`

## Mechanism Interpretation

平均是否有效取决于同一个结构里的两个条件：第一，两个 checkpoint 的对应功能单元是否在同一个低损失 basin 或已对齐；第二，该功能单元的边际收益是否大于沿这条 weight-space 方向移动造成的曲率和干扰成本。Dense 模型里这通常表现为 mode connectivity、符号冲突和局部 curvature；MoE 里还多了 top-k 路由边界、expert identity、expert load、channel/chunk 子空间冲突。这个脚本把这些内部参数统一成 `B/H/I`，因此输出的是“为什么这组 expert 该动多少”，不是“某场景套某算法”。

`scale` 不是收益本身；高收益 group 如果原始 `delta` 已经很大，也会被 hard cap 限制。报告里的 reason 因此区分了 `preserve_high_benefit_low_curvature` 和 `preserve_benefit_but_delta_cap_limited`。

## Candidate Search

| candidate | pass cap | retention | max delta | risk delta | benefit scale | loss proxy | mean scale | objective |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `s0.04_b1.65_h0.75_i0.75` | `True` | 0.9621 | 0.6490 | 0.2214 | 0.9705 | 0.0122 | 0.9524 | 0.1614 |
| `s0.04_b1.65_h1.00_i0.75` | `True` | 0.9616 | 0.6490 | 0.2213 | 0.9702 | 0.0122 | 0.9521 | 0.1624 |
| `s0.04_b1.65_h1.25_i0.75` | `True` | 0.9612 | 0.6490 | 0.2212 | 0.9699 | 0.0122 | 0.9519 | 0.1633 |
| `s0.04_b1.40_h0.75_i0.75` | `True` | 0.9605 | 0.6490 | 0.2210 | 0.9690 | 0.0121 | 0.9512 | 0.1647 |
| `s0.08_b1.65_h0.75_i0.75` | `True` | 0.9601 | 0.6490 | 0.2208 | 0.9681 | 0.0119 | 0.9475 | 0.1653 |
| `s0.04_b1.40_h1.00_i0.75` | `True` | 0.9601 | 0.6490 | 0.2209 | 0.9687 | 0.0121 | 0.9510 | 0.1656 |
| `s0.04_b1.40_h1.25_i0.75` | `True` | 0.9597 | 0.6490 | 0.2208 | 0.9684 | 0.0121 | 0.9507 | 0.1665 |
| `s0.04_b1.65_h0.75_i1.00` | `True` | 0.9595 | 0.6490 | 0.2207 | 0.9678 | 0.0120 | 0.9499 | 0.1671 |
| `s0.08_b1.65_h1.00_i0.75` | `True` | 0.9593 | 0.6490 | 0.2206 | 0.9675 | 0.0118 | 0.9469 | 0.1672 |
| `s0.04_b1.65_h1.00_i1.00` | `True` | 0.9591 | 0.6490 | 0.2206 | 0.9676 | 0.0120 | 0.9497 | 0.1679 |
| `s0.04_b1.15_h0.75_i0.75` | `True` | 0.9589 | 0.6490 | 0.2206 | 0.9674 | 0.0120 | 0.9500 | 0.1683 |
| `s0.04_b1.65_h1.25_i1.00` | `True` | 0.9588 | 0.6490 | 0.2206 | 0.9673 | 0.0120 | 0.9495 | 0.1686 |
| `s0.08_b1.65_h1.25_i0.75` | `True` | 0.9585 | 0.6490 | 0.2204 | 0.9669 | 0.0118 | 0.9464 | 0.1690 |
| `s0.04_b1.15_h1.00_i0.75` | `True` | 0.9585 | 0.6490 | 0.2205 | 0.9671 | 0.0120 | 0.9497 | 0.1692 |

## Top Shrink Reasons

| reason | groups | route mass | mean benefit | mean curvature | mean interference | mean scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `balanced` | 4468 | 414.4203 | 0.2202 | 0.3682 | 0.2974 | 0.9588 |
| `route_important_fragile_boundary` | 402 | 135.4564 | 0.4176 | 0.5058 | 0.4647 | 0.9709 |
| `preserve_high_benefit_low_curvature` | 139 | 18.0120 | 0.7033 | 0.3880 | 0.2440 | 0.9966 |
| `preserve_benefit_but_delta_cap_limited` | 234 | 8.1113 | 0.7126 | 0.4359 | 0.2461 | 0.7727 |

## Outputs

- `report`: `results/qwen3_moe_mechanistic_unified_candidate/report.md`
- `summary`: `results/qwen3_moe_mechanistic_unified_candidate/summary.json`
- `candidate_search`: `results/qwen3_moe_mechanistic_unified_candidate/candidate_search.csv`
- `mechanistic_group_rules`: `results/qwen3_moe_mechanistic_unified_candidate/mechanistic_group_rules.csv`
- `selected_candidate`: `results/qwen3_moe_mechanistic_unified_candidate/selected_candidate.json`
- `tensor_rules`: `results/qwen3_moe_mechanistic_unified_candidate/tensor_rules.txt`
- `writer_command`: `results/qwen3_moe_mechanistic_unified_candidate/writer_command.txt`
- `dry_run_command`: `results/qwen3_moe_mechanistic_unified_candidate/dry_run_command.txt`
- `literature_sources`: `results/qwen3_moe_mechanistic_unified_candidate/literature_sources.json`
- `checkpoint_output_dir`: `results/checkpoints/qwen3_moe_mechanistic_unified_candidate`
