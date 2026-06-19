# MoE Routing Readiness

这个报告把 MoE routing probe 的原始 CSV 转成合并前的风险诊断。它回答的不是“该不该做 MoE”，而是更具体的四个问题：router 是否会 collapse，两个 source 的路由是否漂移，top-k 边界是否脆弱，以及哪些 experts 需要 route/category-aware 权重。

- Readiness status: `high_risk_calibrate_router_before_merge`
- Router dirs: `results/toy_moe_merge`
- Router rows: `54`；expert rows: `216`；specialization rows: `108`

## Router Readiness

Router action counts: `{"freeze_router_and_check_load_balance": 26, "router_probe_passed_for_small_lambda": 25, "calibrate_router_before_average": 3}`

| method | router | category | max top1 | effective fraction | top-k Jaccard | risk flags | action |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| code_endpoint_permuted | toy_router | general | 0.575 | 0.6769 | 0.2075 | top1_load_concentration|low_topk_route_overlap|low_top1_route_agreement | `calibrate_router_before_average` |
| code_endpoint_permuted | toy_router | code | 0.465 | 0.6429 | 0.2942 | low_topk_route_overlap|low_top1_route_agreement | `calibrate_router_before_average` |
| all_weight_average | toy_router | general | 0.43 | 0.8276 | 0.475 | low_topk_route_overlap|low_top1_route_agreement | `calibrate_router_before_average` |
| general_endpoint | toy_router | general | 0.61 | 0.6379 | 0.9833 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| all_weight_average | toy_router | code | 0.5125 | 0.7408 | 0.63 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| base | toy_router | general | 0.5825 | 0.6717 | n/a | top1_load_concentration | `freeze_router_and_check_load_balance` |
| expert_matched_dare_average | toy_router | general | 0.5825 | 0.6717 | 1 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| expert_matched_ties_dare_average | toy_router | general | 0.5825 | 0.6717 | 1 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| router_frozen_average | toy_router | general | 0.5825 | 0.6717 | 1 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| expert_matched_average | toy_router | general | 0.605 | 0.6445 | 0.9867 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| matched_router_frozen_average | toy_router | general | 0.5825 | 0.6717 | 1 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| expert_matched_regmean_average | toy_router | general | 0.5825 | 0.6717 | 1 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| expert_matched_ties_average | toy_router | general | 0.5825 | 0.6717 | 1 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| unified_route_kd_seed_average | toy_router | general | 0.6225 | 0.6187 | 0.92 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| matched_router_calibrated_average | toy_router | general | 0.645 | 0.6097 | 0.8883 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| matched_router_topk_calibrated_average | toy_router | general | 0.705 | 0.561 | 0.895 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| expert_weight_search_average | toy_router | general | 0.5825 | 0.6717 | 1 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| matched_router_sweep_selected_average | toy_router | general | 0.645 | 0.6097 | 0.8883 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| matched_router_kd_average | toy_router | general | 0.5375 | 0.7812 | 0.8983 | top1_load_concentration | `freeze_router_and_check_load_balance` |
| matched_router_route_kd_average | toy_router | general | 0.6225 | 0.6187 | 0.92 | top1_load_concentration | `freeze_router_and_check_load_balance` |

## Expert Load Risks

Expert action counts: `{"low_lambda_or_route_frequency_average": 216}`

| method | router | category | expert | top-k over uniform | flags | action |
| --- | --- | --- | ---: | ---: | --- | --- |
| all_weight_average | toy_router | code | 3 | 1.615 | none | `low_lambda_or_route_frequency_average` |
| unified_route_kd_seed_average | toy_router | general | 0 | 1.565 | none | `low_lambda_or_route_frequency_average` |
| matched_router_topk_calibrated_average | toy_router | general | 0 | 1.565 | none | `low_lambda_or_route_frequency_average` |
| matched_router_route_kd_average | toy_router | general | 0 | 1.565 | none | `low_lambda_or_route_frequency_average` |
| unified_moe_average | toy_router | general | 0 | 1.56 | none | `low_lambda_or_route_frequency_average` |
| matched_router_hessian_average | toy_router | general | 0 | 1.51 | none | `low_lambda_or_route_frequency_average` |
| general_endpoint | toy_router | general | 0 | 1.5 | none | `low_lambda_or_route_frequency_average` |
| expert_matched_average | toy_router | general | 0 | 1.495 | none | `low_lambda_or_route_frequency_average` |
| matched_router_weight_search_average | toy_router | general | 0 | 1.49 | none | `low_lambda_or_route_frequency_average` |
| code_endpoint_permuted | toy_router | general | 1 | 1.49 | none | `low_lambda_or_route_frequency_average` |
| expert_matched_regmean_average | toy_router | general | 0 | 1.48 | none | `low_lambda_or_route_frequency_average` |
| expert_matched_ties_average | toy_router | general | 0 | 1.48 | none | `low_lambda_or_route_frequency_average` |
| router_frozen_average | toy_router | general | 0 | 1.48 | none | `low_lambda_or_route_frequency_average` |
| matched_router_frozen_average | toy_router | general | 0 | 1.48 | none | `low_lambda_or_route_frequency_average` |
| base | toy_router | general | 0 | 1.48 | none | `low_lambda_or_route_frequency_average` |
| route_aware_expert_average | toy_router | general | 0 | 1.48 | none | `low_lambda_or_route_frequency_average` |
| expert_output_projection_average | toy_router | general | 0 | 1.48 | none | `low_lambda_or_route_frequency_average` |
| expert_weight_search_average | toy_router | general | 0 | 1.48 | none | `low_lambda_or_route_frequency_average` |
| expert_matched_ties_dare_average | toy_router | general | 0 | 1.48 | none | `low_lambda_or_route_frequency_average` |
| expert_matched_dare_average | toy_router | general | 0 | 1.48 | none | `low_lambda_or_route_frequency_average` |

## Category Specialization

Specialization action counts: `{"shared_or_mixed_expert": 80, "category_specialized_route_weight": 28}`

| method | router | expert | dominant category | share | action |
| --- | --- | ---: | --- | ---: | --- |
| base | toy_router | 0 | general | 0.8087 | `category_specialized_route_weight` |
| expert_matched_ties_average | toy_router | 0 | general | 0.8087 | `category_specialized_route_weight` |
| expert_matched_regmean_average | toy_router | 0 | general | 0.8087 | `category_specialized_route_weight` |
| expert_matched_dare_average | toy_router | 0 | general | 0.8087 | `category_specialized_route_weight` |
| expert_matched_ties_dare_average | toy_router | 0 | general | 0.8087 | `category_specialized_route_weight` |
| expert_output_projection_average | toy_router | 0 | general | 0.8087 | `category_specialized_route_weight` |
| expert_weight_search_average | toy_router | 0 | general | 0.8087 | `category_specialized_route_weight` |
| matched_router_frozen_average | toy_router | 0 | general | 0.8087 | `category_specialized_route_weight` |
| router_frozen_average | toy_router | 0 | general | 0.8087 | `category_specialized_route_weight` |
| route_aware_expert_average | toy_router | 0 | general | 0.8087 | `category_specialized_route_weight` |
| matched_router_weight_search_average | toy_router | 0 | general | 0.7926 | `category_specialized_route_weight` |
| code_endpoint_permuted | toy_router | 1 | general | 0.7926 | `category_specialized_route_weight` |
| matched_router_hessian_average | toy_router | 0 | general | 0.7885 | `category_specialized_route_weight` |
| matched_router_topk_calibrated_average | toy_router | 0 | general | 0.7884 | `category_specialized_route_weight` |
| unified_route_kd_seed_average | toy_router | 0 | general | 0.7864 | `category_specialized_route_weight` |
| matched_router_route_kd_average | toy_router | 0 | general | 0.7864 | `category_specialized_route_weight` |
| general_endpoint | toy_router | 0 | general | 0.7853 | `category_specialized_route_weight` |
| expert_matched_average | toy_router | 0 | general | 0.7848 | `category_specialized_route_weight` |
| all_weight_average | toy_router | 0 | general | 0.7742 | `category_specialized_route_weight` |
| unified_moe_average | toy_router | 0 | general | 0.7573 | `category_specialized_route_weight` |

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

- `results/toy_moe_routing_readiness/router_readiness.csv`
- `results/toy_moe_routing_readiness/expert_load_risks.csv`
- `results/toy_moe_routing_readiness/category_specialization.csv`
- `results/toy_moe_routing_readiness/summary.json`
