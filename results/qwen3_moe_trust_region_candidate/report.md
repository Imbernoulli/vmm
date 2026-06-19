# Qwen3 MoE Trust-Region Candidate

这个候选把 route-frequency expert weights、routing/load readiness 和 materialized delta audit 合在一起。
核心约束是：router 仍冻结，目标结构不变；只按 MoE 内部风险缩小非 base source delta。

- Status: `trust_region_rules_ready`
- Expert rules: `5243`
- Scaled expert rules: `405`
- Scaled beyond simple delta cap: `103`
- Mean effective nonbase weight: `0.201` -> `0.186`
- Max routed expert relative delta estimate: `1.327` -> `0.750`
- Estimated total relative delta norm: `0.249`
- Estimated routed tensors >1.0 / >0.75 / >0.65: `0` / `0` / `354`
- Writer dry-run: `True`; expert/attention/router hits `15729` / `288` / `48`

## Risk Flags

| flag | count |
| --- | ---: |
| `category_source_mismatch` | 152 |
| `delta_above_base_cap` | 302 |
| `fragile_router_layer` | 2632 |
| `high_load_expert` | 1650 |
| `high_load_weight_limit` | 62 |
| `low_route_evidence` | 984 |
| `shared_mixed_expert` | 2218 |
| `shared_weight_limit` | 1 |

## Action Counts

| action | count |
| --- | ---: |
| `keep_route_weight_rule` | 420 |
| `keep_with_moe_risk_monitor` | 4418 |
| `scale_nonbase_delta_to_moe_trust_region` | 405 |

## Strongest Trust-Region Gates

| layer | expert | original nonbase | new nonbase | target cap | route max rel | expected max rel | scale | flags |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 12 | 71 | 0.850 | 0.412 | 0.600 | 1.237 | 0.600 | 0.485 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 20 | 118 | 0.850 | 0.442 | 0.600 | 1.153 | 0.600 | 0.520 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 11 | 31 | 0.850 | 0.444 | 0.600 | 1.148 | 0.600 | 0.523 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 10 | 36 | 0.850 | 0.446 | 0.600 | 1.145 | 0.600 | 0.524 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 14 | 24 | 0.850 | 0.447 | 0.600 | 1.142 | 0.600 | 0.526 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 15 | 38 | 0.850 | 0.448 | 0.700 | 1.327 | 0.700 | 0.527 | `delta_above_base_cap|fragile_router_layer` |
| 21 | 67 | 0.850 | 0.453 | 0.600 | 1.126 | 0.600 | 0.533 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 16 | 116 | 0.850 | 0.454 | 0.600 | 1.123 | 0.600 | 0.534 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 28 | 116 | 0.850 | 0.454 | 0.600 | 1.122 | 0.600 | 0.535 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 17 | 25 | 0.850 | 0.456 | 0.600 | 1.120 | 0.600 | 0.536 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 18 | 108 | 0.850 | 0.457 | 0.600 | 1.115 | 0.600 | 0.538 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 12 | 11 | 0.850 | 0.458 | 0.600 | 1.114 | 0.600 | 0.538 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 23 | 117 | 0.850 | 0.458 | 0.600 | 1.114 | 0.600 | 0.538 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 13 | 87 | 0.850 | 0.459 | 0.600 | 1.111 | 0.600 | 0.540 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 24 | 93 | 0.850 | 0.460 | 0.600 | 1.109 | 0.600 | 0.541 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 24 | 77 | 0.850 | 0.462 | 0.600 | 1.105 | 0.600 | 0.543 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 27 | 12 | 0.850 | 0.465 | 0.600 | 1.097 | 0.600 | 0.547 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 12 | 93 | 0.850 | 0.466 | 0.600 | 1.095 | 0.600 | 0.548 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 23 | 77 | 0.850 | 0.466 | 0.600 | 1.095 | 0.600 | 0.548 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 32 | 123 | 0.850 | 0.473 | 0.600 | 1.079 | 0.600 | 0.556 | `delta_above_base_cap|low_route_evidence` |

## Highest-Risk Kept Or Scaled Experts

| layer | expert | action | total route | max load/uniform | mean top-k Jaccard | categories | category share | route max rel | scale | flags |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 12 | 71 | `scale_nonbase_delta_to_moe_trust_region` | 0.009 | 0.000 | 0.417 | 0 | 0.000 | 1.237 | 0.485 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 20 | 118 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.401 | 0 | 0.000 | 1.153 | 0.520 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 11 | 31 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.426 | 0 | 0.000 | 1.148 | 0.523 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 10 | 36 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.449 | 0 | 0.000 | 1.145 | 0.524 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 14 | 24 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.533 | 0.402 | 8 | 1.000 | 1.142 | 0.526 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 15 | 38 | `scale_nonbase_delta_to_moe_trust_region` | 0.022 | 0.000 | 0.406 | 0 | 0.000 | 1.327 | 0.527 | `delta_above_base_cap|fragile_router_layer` |
| 21 | 67 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.372 | 0 | 0.000 | 1.126 | 0.533 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 16 | 116 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.410 | 0 | 0.000 | 1.123 | 0.534 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 28 | 116 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.401 | 0 | 0.000 | 1.122 | 0.535 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 17 | 25 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 2.207 | 0.395 | 8 | 1.000 | 1.120 | 0.536 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 18 | 108 | `scale_nonbase_delta_to_moe_trust_region` | 0.008 | 2.759 | 0.384 | 8 | 0.843 | 1.115 | 0.538 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 12 | 11 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.417 | 0 | 0.000 | 1.114 | 0.538 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 23 | 117 | `scale_nonbase_delta_to_moe_trust_region` | 0.018 | 0.000 | 0.307 | 0 | 0.000 | 1.114 | 0.538 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 13 | 87 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.424 | 0 | 0.000 | 1.111 | 0.540 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 24 | 93 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.356 | 0 | 0.000 | 1.109 | 0.541 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 24 | 77 | `scale_nonbase_delta_to_moe_trust_region` | 0.009 | 0.552 | 0.356 | 8 | 1.000 | 1.105 | 0.543 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 27 | 12 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.401 | 0 | 0.000 | 1.097 | 0.547 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 12 | 93 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.417 | 0 | 0.000 | 1.095 | 0.548 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 23 | 77 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.307 | 0 | 0.000 | 1.095 | 0.548 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 32 | 123 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.462 | 0 | 0.000 | 1.079 | 0.556 | `delta_above_base_cap|low_route_evidence` |
| 33 | 67 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.479 | 0 | 0.000 | 1.073 | 0.559 | `delta_above_base_cap|low_route_evidence` |
| 34 | 54 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.427 | 0 | 0.000 | 1.070 | 0.561 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 9 | 13 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.461 | 0 | 0.000 | 1.066 | 0.563 | `delta_above_base_cap|low_route_evidence` |
| 26 | 14 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.399 | 0 | 0.000 | 1.066 | 0.563 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 26 | 9 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.399 | 0 | 0.000 | 1.059 | 0.566 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 26 | 83 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.399 | 0 | 0.000 | 1.059 | 0.567 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 9 | 6 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.461 | 0 | 0.000 | 1.058 | 0.567 | `delta_above_base_cap|low_route_evidence` |
| 25 | 126 | `scale_nonbase_delta_to_moe_trust_region` | 0.018 | 0.000 | 0.427 | 0 | 0.000 | 1.054 | 0.569 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |
| 5 | 114 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.502 | 0 | 0.000 | 1.052 | 0.570 | `delta_above_base_cap|low_route_evidence` |
| 34 | 7 | `scale_nonbase_delta_to_moe_trust_region` | 0.013 | 0.000 | 0.427 | 0 | 0.000 | 1.050 | 0.571 | `delta_above_base_cap|fragile_router_layer|low_route_evidence` |

## Files

- `results/qwen3_moe_trust_region_candidate/trust_region_source_weights_by_expert.csv`
- `results/qwen3_moe_trust_region_candidate/estimated_tensor_delta.csv`
- `results/qwen3_moe_trust_region_candidate/tensor_rules.txt`
- `results/qwen3_moe_trust_region_candidate/writer_command.txt`
- `results/qwen3_moe_trust_region_candidate/dry_run_command.txt`
- `results/qwen3_moe_trust_region_candidate/dry_run/merge_manifest.json`
- `results/qwen3_moe_trust_region_candidate/summary.json`
