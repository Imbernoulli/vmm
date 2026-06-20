# Qwen3 MoE Mechanistic Evidence Audit

这个审计把 mechanistic unified candidate 的每个 routed expert group 拆成 `B/H/I`、梯度、hard-cap 绑定和 scale 响应。它不是再报一个算法名，而是检查当前统一 scale law 是否真的由内部参数驱动。

## Result

- Status: `mechanistic_evidence_audit_ready`
- Selected candidate: `s0.08_b1.65_h0.75_i0.75`
- Expert groups: `5243`
- Hard-cap bound groups: `319`
- Gradient/sign agreement: `1.0000`
- Objective proxy improved groups: `0.9453`
- Route-mass weighted objective gain vs prior: `0.0075`
- Route-mass weighted selected scale: `0.9802`
- Dominant binding: `cost_gradient_shrink`

## Local Scale Law

当前同结构 average 被写成每个 expert group 的局部二次近似：

`J_g(s) = 1/2 H_g ||delta_g||^2 s^2 + I_g s^2 - B_g s`

`B_g` 是保留非 base source 的收益代理量；`H_g ||delta_g||^2` 是沿当前权重方向移动的局部曲率成本；`I_g` 是 source conflict、router/top-k 边界、load 与 subspace conflict 的干扰成本。若没有 damping、retention 和 hard cap，驻点是 `s*=B/(H||delta||^2+2I)`；当前候选使用更保守的一步更新，并把 `s * ||delta|| <= hard_cap` 当成硬约束。

## Binding Summary

| binding | groups | route mass | mean scale | scale delta | B | Hdelta2 | I | grad@prior | cap frac | obj gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cost_gradient_shrink` | 4736 | 501.4174 | 0.9690 | -0.0179 | 0.6302 | 0.0614 | 0.5057 | -0.4357 | 0.0000 | 0.0085 |
| `hard_cap_bound` | 319 | 65.5487 | 0.9901 | 0.0018 | 0.9399 | 0.0595 | 0.3660 | 0.1492 | 1.0000 | 0.0001 |
| `benefit_gradient_restore` | 188 | 9.0339 | 0.7404 | 0.0087 | 1.2949 | 0.5935 | 0.3884 | 0.2353 | 0.0000 | 0.0032 |

## Feature Response

| feature | corr(scale) | corr(delta scale) | route-weighted feature |
| --- | ---: | ---: | ---: |
| `curvature_score` | -0.5935 | -0.3784 | 0.4837 |
| `feature_router_instability` | -0.5827 | -0.5709 | 0.6168 |
| `feature_expert_internal_geometry` | -0.5661 | -0.5779 | 0.4958 |
| `interference_score` | -0.5600 | -0.6524 | 0.4449 |
| `feature_subspace_conflict` | -0.5593 | -0.5729 | 0.5095 |
| `feature_subspace_route_conflict` | -0.3445 | -0.4294 | 0.4099 |
| `feature_expert_route_geometry` | -0.2307 | -0.3439 | 0.5898 |
| `delta_pressure` | -0.1821 | 0.2407 | 0.2800 |
| `coder_route_share` | -0.1647 | 0.2451 | 0.2500 |
| `benefit_score` | 0.0139 | 0.4241 | 0.3499 |
| `load_pressure` | 0.3391 | 0.1206 | 0.5223 |
| `route_mass_pressure` | 0.4253 | 0.2387 | 0.5894 |

## Hard Cases

- Rows exported: `120`
- File: `results/qwen3_moe_mechanistic_evidence_audit/hard_cases.csv`

## Outputs

- `report`: `results/qwen3_moe_mechanistic_evidence_audit/report.md`
- `summary`: `results/qwen3_moe_mechanistic_evidence_audit/summary.json`
- `group_evidence`: `results/qwen3_moe_mechanistic_evidence_audit/group_mechanistic_evidence.csv`
- `binding_summary`: `results/qwen3_moe_mechanistic_evidence_audit/binding_summary.csv`
- `feature_deciles`: `results/qwen3_moe_mechanistic_evidence_audit/feature_decile_response.csv`
- `feature_correlations`: `results/qwen3_moe_mechanistic_evidence_audit/feature_correlations.csv`
- `hard_cases`: `results/qwen3_moe_mechanistic_evidence_audit/hard_cases.csv`
