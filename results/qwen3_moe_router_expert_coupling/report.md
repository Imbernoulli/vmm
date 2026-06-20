# Qwen3 MoE Router-Expert Coupling

这个 probe 把 router top-k 边界脆弱性和 routed expert scale law 按 layer/expert join 起来。目标不是再说 router 要不要动，而是检查：脆弱 router 层下面的 experts 是否已经被 B/H/I trust-region 自动收紧。

- Status: `router_expert_coupling_ready`
- Gate: `router_expert_coupling_active`
- Fragility -> router-instability feature corr: `0.6947`
- Fragility -> scale-shrink corr: `0.5831`
- Safe-lambda -> scale-shrink corr: `-0.1472`
- High-fragility weighted scale/shrink: `0.9701` / `0.0199`
- Low-fragility weighted scale/shrink: `0.9909` / `0.0061`
- High-vs-low shrink lift: `0.0138`
- Top coupled layer: `L20` risk `6.5776`

## Layer Coupling

| layer | fragility | safe lambda | weighted scale | weighted shrink | router feature | risk sum | top expert |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.7157 | 0.0439 | 0.9726 | 0.0205 | 0.9582 | 6.5776 | 38 |
| 17 | 0.7523 | 0.0261 | 0.9631 | 0.0219 | 0.8117 | 6.3802 | 83 |
| 15 | 0.7040 | 0.0438 | 0.9650 | 0.0221 | 0.8983 | 6.3323 | 10 |
| 18 | 0.7267 | 0.0391 | 0.9634 | 0.0202 | 0.8974 | 6.3295 | 8 |
| 22 | 0.6760 | 0.0500 | 0.9674 | 0.0207 | 0.9790 | 6.3057 | 98 |
| 19 | 0.6911 | 0.0436 | 0.9664 | 0.0188 | 0.9365 | 6.1677 | 109 |
| 14 | 0.6738 | 0.0444 | 0.9696 | 0.0225 | 0.9220 | 6.1190 | 107 |
| 24 | 0.6790 | 0.0392 | 0.9732 | 0.0208 | 0.9099 | 6.0144 | 65 |
| 12 | 0.7180 | 0.0338 | 0.9696 | 0.0205 | 0.7905 | 5.8829 | 56 |
| 13 | 0.7122 | 0.0384 | 0.9732 | 0.0209 | 0.7729 | 5.7471 | 104 |
| 29 | 0.6910 | 0.0396 | 0.9696 | 0.0214 | 0.8048 | 5.6688 | 83 |
| 23 | 0.6413 | 0.0500 | 0.9687 | 0.0211 | 0.8649 | 5.5387 | 21 |
| 25 | 0.6884 | 0.0459 | 0.9701 | 0.0199 | 0.7765 | 5.4762 | 104 |
| 11 | 0.7151 | 0.0352 | 0.9733 | 0.0212 | 0.6853 | 5.3432 | 21 |
| 28 | 0.6094 | 0.0500 | 0.9733 | 0.0198 | 0.8948 | 5.2778 | 73 |
| 21 | 0.6385 | 0.0425 | 0.9708 | 0.0201 | 0.8183 | 5.2728 | 10 |

## Top Router-Coupled Experts

| layer | expert | category | route mass | fragility | router feature | scale | shrink | coupled risk | reason |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | 38 | `general` | 1.1555 | 0.7157 | 0.9582 | 0.9808 | 0.0192 | 0.6714 | `route_important_fragile_boundary` |
| 25 | 104 | `general` | 1.3000 | 0.6884 | 0.7765 | 0.9798 | 0.0202 | 0.6547 | `route_important_fragile_boundary` |
| 18 | 8 | `code` | 0.9657 | 0.7267 | 0.8974 | 0.9885 | 0.0115 | 0.5325 | `route_important_fragile_boundary` |
| 11 | 21 | `code` | 0.9517 | 0.7151 | 0.6853 | 0.9827 | 0.0173 | 0.4820 | `route_important_fragile_boundary` |
| 21 | 10 | `code` | 1.0366 | 0.6385 | 0.8183 | 0.9893 | 0.0107 | 0.4746 | `route_important_fragile_boundary` |
| 23 | 21 | `code` | 0.9393 | 0.6413 | 0.8649 | 0.9849 | 0.0151 | 0.4684 | `route_important_fragile_boundary` |
| 17 | 83 | `safety` | 0.7944 | 0.7523 | 0.8117 | 0.9762 | 0.0238 | 0.4586 | `route_important_fragile_boundary` |
| 24 | 65 | `safety` | 0.8476 | 0.6790 | 0.9099 | 0.9777 | 0.0223 | 0.4582 | `route_important_fragile_boundary` |
| 22 | 98 | `safety` | 0.7477 | 0.6760 | 0.9790 | 0.9799 | 0.0201 | 0.4105 | `route_important_fragile_boundary` |
| 15 | 10 | `code` | 0.7142 | 0.7040 | 0.8983 | 0.9826 | 0.0174 | 0.4060 | `route_important_fragile_boundary` |
| 20 | 116 | `safety` | 0.6731 | 0.7157 | 0.9582 | 0.9786 | 0.0214 | 0.3879 | `route_important_fragile_boundary` |
| 9 | 10 | `code` | 1.0058 | 0.6389 | 0.6618 | 0.9988 | 0.0012 | 0.3792 | `balanced` |
| 22 | 53 | `safety` | 0.6659 | 0.6760 | 0.9790 | 0.9780 | 0.0220 | 0.3683 | `route_important_fragile_boundary` |
| 14 | 107 | `code` | 0.6818 | 0.6738 | 0.9220 | 0.9857 | 0.0143 | 0.3659 | `route_important_fragile_boundary` |
| 26 | 107 | `code` | 0.8161 | 0.6268 | 0.7835 | 0.9892 | 0.0108 | 0.3572 | `route_important_fragile_boundary` |
| 35 | 21 | `code` | 1.0404 | 0.5526 | 0.7050 | 0.9979 | 0.0021 | 0.3545 | `balanced` |

## Correlation Checks

| driver | target | weighted corr |
| --- | --- | ---: |
| `boundary_fragility_score` | `feature_router_instability` | 0.6947 |
| `boundary_fragility_score` | `scale_shrink` | 0.5831 |
| `boundary_fragility_score` | `mechanistic_selected_scale` | -0.1947 |
| `safe_lambda_proxy` | `feature_router_instability` | -0.0674 |
| `safe_lambda_proxy` | `scale_shrink` | -0.1472 |
| `safe_lambda_proxy` | `mechanistic_selected_scale` | 0.0510 |

## Interpretation

Current B/H/I already turns fragile router layers into stronger routed-expert shrink, so removing router-boundary terms would be a mechanistic regression.
Dense 模型的插值风险主要是连续 loss barrier；MoE 还多了离散 dispatch 边界。这里的结果说明 router fragility 不能只作为 router freeze 的理由，还应该作为 expert trust-region 的一部分，因为高脆弱层的 expert scale 确实被系统性收紧。
这个 probe 仍不替代最终 vLLM selector；它只证明 router-boundary 项在当前 B/H/I scale law 里有可观测机制作用。

## Outputs

- `layer_router_expert_coupling`: `results/qwen3_moe_router_expert_coupling/layer_router_expert_coupling.csv`
- `top_router_coupled_experts`: `results/qwen3_moe_router_expert_coupling/top_router_coupled_experts.csv`
- `coupling_correlations`: `results/qwen3_moe_router_expert_coupling/coupling_correlations.csv`
- `literature_sources`: `results/qwen3_moe_router_expert_coupling/literature_sources.json`
- `summary`: `results/qwen3_moe_router_expert_coupling/summary.json`
- `report`: `results/qwen3_moe_router_expert_coupling/report.md`
