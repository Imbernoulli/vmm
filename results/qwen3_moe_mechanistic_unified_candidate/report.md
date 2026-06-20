# Qwen3 MoE Mechanistic Unified Candidate

这个候选不再把 Average 写成“选某个算法名”。它把同结构 MoE 平均写成每个 routed expert group 的近似二次目标：

`J_g(s) = 1/2 H_g ||delta_g||^2 s^2 + I_g s^2 - B_g s`

其中 `B_g` 来自 route mass、Coder route share、source specialization 和当前非 base 权重；`H_g` 来自 delta pressure、expert geometry、layer geometry、router fragility 和 subspace conflict；`I_g` 来自 source conflict、fragile top-k boundary、high-load/shared experts 和 category mismatch。求出的 `s` 是同结构规则里的非 base scale，并继续受 hard-cap 与 vLLM feedback gate 约束。

## Result

- Status: `mechanistic_unified_candidate_ready`
- Expert groups: `5243`
- Selected candidate: `s0.16_b1.65_h1.25_i1.00`
- Search points: `144`
- Nonbase route-mass retention: `0.9653`
- Max predicted routed relative delta: `0.6500`
- Hard-cap violations: `0`
- Risk-weighted predicted delta: `0.2303`
- Benefit-weighted scale: `0.9772`
- Mean mechanistic loss proxy: `0.0290`
- Mean scale: `0.9589`
- High-benefit low-risk mean scale: `0.8653`
- High-interference low-benefit mean scale: `0.9533`
- High-subspace-conflict mean scale: `0.9282`

## Mechanism Interpretation

平均是否有效取决于同一个结构里的两个条件：第一，两个 checkpoint 的对应功能单元是否在同一个低损失 basin 或已对齐；第二，该功能单元的边际收益是否大于沿这条 weight-space 方向移动造成的曲率和干扰成本。Dense 模型里这通常表现为 mode connectivity、符号冲突和局部 curvature；MoE 里还多了 top-k 路由边界、expert identity、expert load、channel/chunk 子空间冲突。这个脚本把这些内部参数统一成 `B/H/I`，因此输出的是“为什么这组 expert 该动多少”，不是“某场景套某算法”。

`scale` 不是收益本身；高收益 group 如果原始 `delta` 已经很大，也会被 hard cap 限制。报告里的 reason 因此区分了 `preserve_high_benefit_low_curvature` 和 `preserve_benefit_but_delta_cap_limited`。

## Candidate Search

| candidate | pass cap | retention | max delta | risk delta | benefit scale | loss proxy | mean scale | objective |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `s0.16_b1.65_h1.25_i1.00` | `True` | 0.9653 | 0.6500 | 0.2303 | 0.9772 | 0.0290 | 0.9589 | 0.1815 |
| `s0.12_b1.40_h1.00_i1.00` | `True` | 0.9650 | 0.6500 | 0.2303 | 0.9763 | 0.0290 | 0.9607 | 0.1818 |
| `s0.16_b1.15_h0.75_i0.75` | `True` | 0.9659 | 0.6500 | 0.2305 | 0.9773 | 0.0290 | 0.9602 | 0.1818 |
| `s0.12_b1.15_h1.25_i0.75` | `True` | 0.9649 | 0.6500 | 0.2303 | 0.9771 | 0.0292 | 0.9624 | 0.1819 |
| `s0.12_b1.65_h0.75_i1.25` | `True` | 0.9652 | 0.6500 | 0.2304 | 0.9759 | 0.0289 | 0.9596 | 0.1819 |
| `s0.16_b1.65_h1.00_i1.00` | `True` | 0.9673 | 0.6500 | 0.2309 | 0.9786 | 0.0291 | 0.9602 | 0.1823 |
| `s0.12_b1.15_h1.00_i0.75` | `True` | 0.9666 | 0.6500 | 0.2308 | 0.9783 | 0.0293 | 0.9635 | 0.1823 |
| `s0.08_b1.40_h0.75_i1.25` | `True` | 0.9655 | 0.6500 | 0.2305 | 0.9762 | 0.0291 | 0.9630 | 0.1824 |
| `s0.12_b1.40_h0.75_i1.00` | `True` | 0.9667 | 0.6500 | 0.2308 | 0.9776 | 0.0291 | 0.9618 | 0.1824 |
| `s0.08_b0.90_h1.00_i0.75` | `True` | 0.9656 | 0.6500 | 0.2306 | 0.9770 | 0.0293 | 0.9650 | 0.1824 |
| `s0.16_b1.40_h1.25_i0.75` | `True` | 0.9677 | 0.6500 | 0.2310 | 0.9799 | 0.0295 | 0.9629 | 0.1824 |
| `s0.08_b1.15_h1.00_i1.00` | `True` | 0.9649 | 0.6500 | 0.2304 | 0.9761 | 0.0292 | 0.9635 | 0.1824 |
| `s0.08_b1.15_h0.75_i1.00` | `True` | 0.9660 | 0.6500 | 0.2307 | 0.9769 | 0.0292 | 0.9642 | 0.1825 |
| `s0.08_b1.65_h1.25_i1.25` | `True` | 0.9670 | 0.6500 | 0.2309 | 0.9781 | 0.0294 | 0.9644 | 0.1827 |

## Top Shrink Reasons

| reason | groups | route mass | mean benefit | mean curvature | mean interference | mean scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `balanced` | 4372 | 414.5307 | 0.2374 | 0.4319 | 0.3759 | 0.9678 |
| `route_important_fragile_boundary` | 402 | 135.4564 | 0.4212 | 0.5842 | 0.5936 | 0.9835 |
| `shrink_high_interference_low_benefit` | 200 | 9.8881 | 0.2537 | 0.5443 | 0.6710 | 0.9435 |
| `preserve_high_benefit_low_curvature` | 58 | 8.5385 | 0.7204 | 0.4114 | 0.3274 | 0.9986 |
| `shrink_curved_subspace_conflict` | 89 | 4.0721 | 0.6250 | 0.7280 | 0.5407 | 0.6460 |
| `preserve_benefit_but_delta_cap_limited` | 122 | 3.5141 | 0.7090 | 0.4524 | 0.2073 | 0.7953 |

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
