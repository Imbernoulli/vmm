# MoE Routing Readiness

这个报告把 MoE routing probe 的原始 CSV 转成合并前的风险诊断。它回答的不是“该不该做 MoE”，而是更具体的四个问题：router 是否会 collapse，两个 source 的路由是否漂移，top-k 边界是否脆弱，以及哪些 experts 需要 route/category-aware 权重。

- Readiness status: `high_risk_calibrate_router_before_merge`
- Router dirs: `results/moe_routing_probe/qwen3_30b_instruct_vs_coder`
- Router rows: `576`；expert rows: `73728`；specialization rows: `5102`

## Router Readiness

Router action counts: `{"calibrate_router_before_average": 493, "small_lambda_router_with_overlap_guard": 46, "router_probe_passed_for_small_lambda": 31, "freeze_router_and_check_load_balance": 6}`

| method | router | category | max top1 | effective fraction | top-k Jaccard | risk flags | action |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.4.mlp.gate | general | 0.08 | 0.148 | 0.4882 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.15.mlp.gate | general | 0.16 | 0.1122 | 0.4093 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.14.mlp.gate | general | 0.28 | 0.08584 | 0.3766 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.10.mlp.gate | long_context | 0.09677 | 0.1408 | 0.4527 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.11.mlp.gate | long_context | 0.1613 | 0.1249 | 0.4332 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.12.mlp.gate | long_context | 0.2258 | 0.1037 | 0.4142 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.13.mlp.gate | long_context | 0.1613 | 0.09742 | 0.4239 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.14.mlp.gate | long_context | 0.3226 | 0.08811 | 0.365 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.15.mlp.gate | long_context | 0.129 | 0.1287 | 0.3959 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.13.mlp.gate | general | 0.12 | 0.1227 | 0.3959 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.4.mlp.gate | long_context | 0.1613 | 0.1208 | 0.4758 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.12.mlp.gate | general | 0.12 | 0.1297 | 0.4725 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.6.mlp.gate | long_context | 0.3226 | 0.06405 | 0.4386 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.35.mlp.gate | long_context | 0.129 | 0.1332 | 0.4797 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | long_context | 0.2258 | 0.08205 | 0.4087 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.26.mlp.gate | long_context | 0.129 | 0.1332 | 0.3632 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.27.mlp.gate | long_context | 0.1935 | 0.1088 | 0.3901 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.47.mlp.gate | long_context | 0.1935 | 0.09496 | 0.4742 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.11.mlp.gate | finance | 0.09375 | 0.1392 | 0.4256 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.12.mlp.gate | finance | 0.25 | 0.106 | 0.3963 | low_topk_route_overlap|low_top1_route_agreement|fragile_topk_boundary | `calibrate_router_before_average` |

## Expert Load Risks

Expert action counts: `{"anchor_heavy_until_rare_task_probe": 38163, "low_lambda_or_route_frequency_average": 31106, "protect_or_source_weight_high_load_expert": 4459}`

| method | router | category | expert | top-k over uniform | flags | action |
| --- | --- | --- | ---: | ---: | --- | --- |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.37.mlp.gate | general | 104 | 16 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.37.mlp.gate | agentic_code | 104 | 16 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.37.mlp.gate | general | 104 | 16 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.37.mlp.gate | legal | 104 | 16 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.37.mlp.gate | safety | 104 | 16 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.37.mlp.gate | code | 104 | 15.47 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | general | 104 | 15.36 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.37.mlp.gate | math | 104 | 15.27 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.31.mlp.gate | math | 18 | 15.27 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.31.mlp.gate | math | 18 | 15.03 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.37.mlp.gate | finance | 104 | 15 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.37.mlp.gate | long_context | 104 | 14.97 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.17.mlp.gate | long_context | 83 | 14.97 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.29.mlp.gate | long_context | 83 | 14.97 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | code | 104 | 14.93 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.6.mlp.gate | code | 80 | 14.93 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.39.mlp.gate | math | 59 | 14.91 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.19.mlp.gate | math | 18 | 14.91 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.37.mlp.gate | code | 104 | 14.9 | overused_expert | `protect_or_source_weight_high_load_expert` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | safety | 104 | 14.86 | overused_expert | `protect_or_source_weight_high_load_expert` |

## Category Specialization

Specialization action counts: `{"shared_or_mixed_expert": 3693, "category_specialized_route_weight": 1409}`

| method | router | expert | dominant category | share | action |
| --- | --- | ---: | --- | ---: | --- |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.0.mlp.gate | 8 | math | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.0.mlp.gate | 19 | math | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.9.mlp.gate | 120 | math | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.9.mlp.gate | 115 | safety | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.9.mlp.gate | 70 | math | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.0.mlp.gate | 24 | general | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | 69 | math | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | 71 | safety | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | 73 | safety | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | 60 | math | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | 64 | code | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | 98 | code | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | 86 | math | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | 87 | math | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | 88 | math | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | 114 | math | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | 117 | math | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.25.mlp.gate | 105 | safety | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.26.mlp.gate | 2 | math | 1 | `category_specialized_route_weight` |
| /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe | model.layers.24.mlp.gate | 121 | math | 1 | `category_specialized_route_weight` |

## 规则依据

- [HARC / routing-breakdown](https://arxiv.org/abs/2606.03391) 分析说明 MoE router 的 softmax/top-k 对参数扰动敏感，因此 `low_topk_route_overlap`、`low_top1_route_agreement` 和 load concentration 都应阻止直接 router average。
- [Sub-MoE](https://arxiv.org/abs/2506.23266) / [MergeMoE](https://arxiv.org/abs/2510.14436) 强调 expert specialization 和 expert output alignment，因此高负载或强 category-specialized experts 应先做 route/source-aware 权重，而不是同权平均。
- [Expert Merging](https://arxiv.org/abs/2509.25712) 强调 layer/chunk-wise coefficients；本报告输出的风险表应和 `build_moe_route_weight_recipes.py` 的 tensor rules 以及后续 layer-wise coefficient search 联动。

## 下一步命令

```bash
python scripts/probe_moe_routing.py --model Qwen/Qwen3-30B-A3B --compare-model Qwen/Qwen3-Coder-30B-A3B-Instruct --prompts prompts/qwen_moe_route_probe_prompts.jsonl --device-map auto --dtype bfloat16 --use-chat-template --output-dir results/moe_routing_probe/qwen3_30b_general_vs_code
PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code
PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code --source general --source code --category-source general=general --category-source code=code --category-source math=general --category-source safety=general
```

## Files

- `results/moe_routing_readiness/qwen3_30b_instruct_vs_coder/router_readiness.csv`
- `results/moe_routing_readiness/qwen3_30b_instruct_vs_coder/expert_load_risks.csv`
- `results/moe_routing_readiness/qwen3_30b_instruct_vs_coder/category_specialization.csv`
- `results/moe_routing_readiness/qwen3_30b_instruct_vs_coder/summary.json`
