# Qwen3 MoE Mechanistic Evidence Audit

这个审计把 mechanistic unified candidate 的每个 routed expert group 拆成 `B/H/I`、梯度、hard-cap 绑定和 scale 响应。它不是再报一个算法名，而是检查当前统一 scale law 是否真的由内部参数驱动。

## Result

- Status: `mechanistic_evidence_audit_ready`
- Selected candidate: `s0.04_b1.65_h0.75_i0.75`
- Expert groups: `5243`
- Nominal/effective hard cap: `0.6500` / `0.6490`
- Materialization safety margin: `0.0010`
- Hard-cap bound groups: `636`
- Gradient/sign agreement: `1.0000`
- Objective proxy improved groups: `0.8989`
- Route-mass weighted objective gain vs prior: `0.0017`
- Route-mass weighted selected scale: `0.9734`
- Dominant binding: `cost_gradient_shrink`

## Local Scale Law

当前同结构 average 被写成每个 expert group 的局部二次近似：

`J_g(s) = 1/2 H_g ||delta_g||^2 s^2 + I_g s^2 - B_g s`

`B_g` 是保留非 base source 的收益代理量；`H_g ||delta_g||^2` 是沿当前权重方向移动的局部曲率成本；`I_g` 是 source conflict、router/top-k 边界、load 与 subspace conflict 的干扰成本。若没有 damping、retention 和 hard cap，驻点是 `s*=B/(H||delta||^2+2I)`；当前候选使用更保守的一步更新，并把 `s * ||delta|| <= hard_cap` 当成硬约束。

## Binding Summary

| binding | groups | route mass | mean scale | scale delta | B | Hdelta2 | I | grad@prior | cap frac | obj gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cost_gradient_shrink` | 3999 | 379.7696 | 0.9563 | -0.0069 | 0.5631 | 0.0471 | 0.4252 | -0.3075 | 0.0000 | 0.0024 |
| `hard_cap_bound` | 636 | 126.8242 | 0.9844 | 0.0004 | 0.8818 | 0.0589 | 0.2496 | 0.3256 | 1.0000 | 0.0001 |
| `benefit_gradient_restore` | 608 | 69.4062 | 0.8938 | 0.0026 | 0.8156 | 0.1470 | 0.3087 | 0.1076 | 0.0000 | 0.0006 |

## Feature Response

| feature | corr(scale) | corr(delta scale) | route-weighted feature |
| --- | ---: | ---: | ---: |
| `interference_score` | -0.6554 | -0.6319 | 0.3309 |
| `feature_expert_internal_geometry` | -0.6176 | -0.5990 | 0.4958 |
| `feature_subspace_conflict` | -0.6116 | -0.5959 | 0.5095 |
| `feature_router_instability` | -0.6018 | -0.5502 | 0.6168 |
| `curvature_score` | -0.5450 | -0.2436 | 0.4129 |
| `feature_subspace_route_conflict` | -0.3692 | -0.4183 | 0.4099 |
| `feature_expert_route_geometry` | -0.2488 | -0.3370 | 0.5898 |
| `delta_pressure` | -0.0089 | 0.5465 | 0.2800 |
| `coder_route_share` | 0.0262 | 0.5846 | 0.2500 |
| `benefit_score` | 0.1574 | 0.6485 | 0.3413 |
| `load_pressure` | 0.3243 | 0.0715 | 0.5223 |
| `route_mass_pressure` | 0.4658 | 0.2836 | 0.5894 |

## Hard Cases

- Rows exported: `82`
- File: `results/qwen3_moe_mechanistic_evidence_audit/hard_cases.csv`

## Outputs

- `report`: `results/qwen3_moe_mechanistic_evidence_audit/report.md`
- `summary`: `results/qwen3_moe_mechanistic_evidence_audit/summary.json`
- `group_evidence`: `results/qwen3_moe_mechanistic_evidence_audit/group_mechanistic_evidence.csv`
- `binding_summary`: `results/qwen3_moe_mechanistic_evidence_audit/binding_summary.csv`
- `feature_deciles`: `results/qwen3_moe_mechanistic_evidence_audit/feature_decile_response.csv`
- `feature_correlations`: `results/qwen3_moe_mechanistic_evidence_audit/feature_correlations.csv`
- `hard_cases`: `results/qwen3_moe_mechanistic_evidence_audit/hard_cases.csv`
