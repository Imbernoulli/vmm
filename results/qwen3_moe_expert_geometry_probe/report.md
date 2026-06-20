# Qwen3 MoE Expert Geometry Probe

这个 probe 直接读取 Qwen3-30B-A3B Instruct 和 Coder 的 routed expert 权重，按 `gate_proj/up_proj/down_proj`、expert、layer 和 channel chunk 计算参数几何。它回答的不是哪个算法名最好，而是为什么某些 expert average 需要更保守：identity 可以成立，但专家内部方向、delta tail 和 route mass 可能仍然冲突。

## Result

- Status: `passed`
- Projection tensors: `18432`
- Experts: `6144` across `48` layers
- Mean / p05 projection cosine: `0.3855` / `0.1183`
- Mean / p95 projection relative delta: `1.0620` / `1.2754`
- High internal-geometry-risk experts: `931`
- High route+geometry-risk experts: `204`
- Top route-geometry layer: `17`
- Top route-geometry expert: layer `13`, expert `104`

## Why This Matters

MoE average 有两个不同层面的风险。第一，expert identity/gauge 要先对齐；否则同名 expert 不代表同一个函数。第二，即使 identity 通过，某些 expert 的 `gate/up/down` 内部参数方向和 channel chunk delta 仍然比其他 expert 更尖锐；如果这些 expert 又有高 route mass 或高 router fragility，就不能用同一个全局平均系数处理。

因此新的 unified rule 应把 `route_geometry_risk_score` 当作 cap-law 或 layer/chunk 系数的输入：router 仍冻结，expert identity 仍保持 same-shape，同一结构内只调整每层/每专家的非 base 权重或 delta cap。

## Projection Summary

| projection | tensors | mean rel-delta | p95 rel-delta | max rel-delta | mean cosine | min cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gate_proj` | 6144 | 1.0617 | 1.2727 | 1.5615 | 0.3846 | -0.0002 |
| `up_proj` | 6144 | 1.0864 | 1.2884 | 1.5359 | 0.3586 | -0.0004 |
| `down_proj` | 6144 | 1.0380 | 1.2625 | 1.4928 | 0.4134 | -0.0013 |

## Highest-Risk Layers

| layer | route-weighted risk | max risk | p95 rel-delta | min cosine | high-risk experts |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.7143 | 0.9220 | 1.3299 | 0.0007 | 15 |
| 15 | 0.7027 | 0.8763 | 1.3173 | 0.0004 | 13 |
| 16 | 0.6881 | 0.8795 | 1.3308 | 0.0012 | 10 |
| 14 | 0.6870 | 0.8696 | 1.3158 | -0.0000 | 12 |
| 13 | 0.6760 | 0.9297 | 1.3022 | 0.0007 | 9 |
| 12 | 0.6752 | 0.9134 | 1.3115 | 0.0004 | 11 |
| 23 | 0.6749 | 0.8693 | 1.3010 | 0.0011 | 10 |
| 22 | 0.6718 | 0.8950 | 1.2959 | 0.0380 | 9 |
| 20 | 0.6714 | 0.8530 | 1.2907 | 0.0333 | 10 |
| 29 | 0.6655 | 0.8815 | 1.2826 | 0.0167 | 10 |
| 21 | 0.6622 | 0.9154 | 1.3138 | 0.0129 | 8 |
| 24 | 0.6618 | 0.8509 | 1.2940 | 0.0910 | 10 |

## Highest-Risk Experts

| layer | expert | action | route+geometry risk | internal risk | rel-delta | cosine | route mass |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 13 | 104 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.9297 | 0.8811 | 1.2908 | 0.0858 | 0.6229 |
| 17 | 83 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.9220 | 0.8621 | 1.2728 | 0.1271 | 0.7944 |
| 21 | 69 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.9154 | 0.9096 | 1.3196 | 0.0292 | 0.4065 |
| 17 | 78 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.9144 | 0.8500 | 1.2975 | 0.1119 | 0.3401 |
| 12 | 119 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.9134 | 0.8469 | 1.2679 | 0.1112 | 0.4105 |
| 11 | 21 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.9057 | 0.8417 | 1.2601 | 0.1404 | 0.9517 |
| 22 | 69 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.8950 | 0.7921 | 1.2009 | 0.2214 | 0.5826 |
| 17 | 58 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.8917 | 0.8016 | 1.2422 | 0.2055 | 0.3672 |
| 29 | 83 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.8815 | 0.7816 | 1.2369 | 0.2035 | 0.6500 |
| 17 | 18 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.8801 | 0.8050 | 1.2391 | 0.1699 | 0.5055 |
| 16 | 115 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.8795 | 0.7843 | 1.2083 | 0.2007 | 0.7047 |
| 16 | 73 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.8781 | 0.8407 | 1.2664 | 0.1319 | 0.4380 |
| 16 | 87 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.8780 | 0.8131 | 1.2487 | 0.1589 | 0.3234 |
| 15 | 70 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.8763 | 0.8514 | 1.2627 | 0.1247 | 0.3642 |
| 25 | 104 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.8716 | 0.7647 | 1.1701 | 0.2618 | 1.3000 |
| 8 | 38 | `route_important_geometry_risk_use_layer_chunk_or_lower_cap` | 0.8701 | 0.7969 | 1.2286 | 0.1955 | 0.9400 |

## Top Channel Chunks

| layer | expert | projection | chunk | rel-delta | cosine | delta share |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 15 | 38 | `gate_proj` | 1 | 1.5888 | 0.0024 | 0.1208 |
| 15 | 38 | `gate_proj` | 6 | 1.5735 | -0.0007 | 0.1282 |
| 15 | 38 | `gate_proj` | 0 | 1.5664 | -0.0027 | 0.1277 |
| 15 | 38 | `gate_proj` | 2 | 1.5662 | 0.0026 | 0.1236 |
| 15 | 38 | `gate_proj` | 4 | 1.5609 | 0.0006 | 0.1267 |
| 15 | 38 | `gate_proj` | 7 | 1.5555 | -0.0004 | 0.1255 |
| 15 | 38 | `gate_proj` | 3 | 1.5533 | 0.0010 | 0.1247 |
| 15 | 38 | `up_proj` | 1 | 1.5469 | -0.0007 | 0.1235 |
| 15 | 38 | `up_proj` | 4 | 1.5455 | -0.0017 | 0.1258 |
| 15 | 38 | `up_proj` | 0 | 1.5415 | 0.0034 | 0.1262 |
| 15 | 38 | `up_proj` | 6 | 1.5403 | 0.0015 | 0.1286 |
| 15 | 38 | `up_proj` | 2 | 1.5394 | 0.0002 | 0.1237 |
| 15 | 38 | `up_proj` | 3 | 1.5315 | -0.0014 | 0.1245 |
| 15 | 38 | `gate_proj` | 5 | 1.5291 | 0.0023 | 0.1228 |
| 15 | 38 | `up_proj` | 7 | 1.5289 | 0.0014 | 0.1248 |
| 15 | 38 | `up_proj` | 5 | 1.5131 | -0.0009 | 0.1228 |

## Literature Priors

- `git_re_basin`: https://arxiv.org/abs/2209.04836
- `model_soups`: https://arxiv.org/abs/2203.05482
- `mergeme`: https://arxiv.org/abs/2502.00997
- `harc_routing_breakdown`: https://arxiv.org/abs/2606.03391
- `router_kd_calibration`: https://arxiv.org/abs/2603.02217

## Outputs

- `report`: `results/qwen3_moe_expert_geometry_probe/report.md`
- `summary`: `results/qwen3_moe_expert_geometry_probe/summary.json`
- `projection_geometry`: `results/qwen3_moe_expert_geometry_probe/projection_geometry.csv`
- `expert_geometry`: `results/qwen3_moe_expert_geometry_probe/expert_geometry.csv`
- `layer_geometry`: `results/qwen3_moe_expert_geometry_probe/layer_geometry.csv`
- `projection_summary`: `results/qwen3_moe_expert_geometry_probe/projection_summary.csv`
- `top_chunk_geometry`: `results/qwen3_moe_expert_geometry_probe/top_chunk_geometry.csv`
