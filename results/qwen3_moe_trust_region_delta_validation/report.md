# Qwen3 MoE Trust-Region Delta Validation

这个验证把 trust-region 规则的逐 tensor delta 预测和真实物化 safetensors 的 delta audit 对齐。
目标是确认 expert-level nonbase delta 缩放确实按预期进入参数文件，而不是只看 group-level 汇总。

- Status: `passed`
- Tensor count: `18867`
- Max abs relative-delta prediction error: `0.000093`
- P99 abs relative-delta prediction error: `0.000007`
- Tensors above tolerance: `0`
- Routed actual/predicted tensors >0.75: `14` / `0`
- Routed >0.75 within rounding slop: `14`
- Routed max actual/predicted relative delta: `0.750022` / `0.750000`

## Group Prediction Error

| group | tensors | max abs rel error | p99 abs rel error | actual >0.75 | predicted >0.75 | rounding slop >0.75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attention` | 288 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| `embedding_or_head` | 2 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| `norm` | 97 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| `routed_expert_ffn` | 18432 | 0.000093 | 0.000007 | 14 | 0 | 14 |
| `router` | 48 | 0.000000 | 0.000000 | 0 | 0 | 0 |

## Action Prediction Error

| action | tensors | max abs rel error | actual >0.75 | rounding slop >0.75 |
| --- | ---: | ---: | ---: | ---: |
| `keep_route_weight_rule` | 1260 | 0.000000 | 0 | 0 |
| `keep_with_moe_risk_monitor` | 13254 | 0.000000 | 0 | 0 |
| `non_expert_or_unruled` | 2703 | 0.000000 | 0 | 0 |
| `scale_nonbase_delta_to_moe_trust_region` | 1215 | 0.000093 | 14 | 14 |

## Risk Flag Prediction Error

| risk flag | tensors | max abs rel error | actual >0.75 | rounding slop >0.75 |
| --- | ---: | ---: | ---: | ---: |
| `delta_above_base_cap` | 906 | 0.000093 | 14 | 14 |
| `category_source_mismatch` | 456 | 0.000067 | 0 | 0 |
| `fragile_router_layer` | 7896 | 0.000093 | 0 | 0 |
| `high_load_expert` | 4950 | 0.000018 | 0 | 0 |
| `high_load_weight_limit` | 186 | 0.000018 | 0 | 0 |
| `low_route_evidence` | 2952 | 0.000093 | 0 | 0 |
| `no_risk_flag` | 3963 | 0.000000 | 0 | 0 |
| `shared_mixed_expert` | 6654 | 0.000006 | 0 | 0 |
| `shared_weight_limit` | 3 | 0.000006 | 0 | 0 |

## Routed Threshold Residuals

| tensor | action | actual rel | predicted rel | abs error | flags |
| --- | --- | ---: | ---: | ---: | --- |
| `model.layers.0.mlp.experts.56.up_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750022 | 0.750000 | 0.000022 | `delta_above_base_cap` |
| `model.layers.6.mlp.experts.112.down_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750012 | 0.750000 | 0.000012 | `delta_above_base_cap` |
| `model.layers.42.mlp.experts.87.gate_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750011 | 0.750000 | 0.000011 | `delta_above_base_cap` |
| `model.layers.8.mlp.experts.15.up_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750007 | 0.750000 | 0.000007 | `delta_above_base_cap` |
| `model.layers.32.mlp.experts.15.up_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750005 | 0.750000 | 0.000005 | `delta_above_base_cap` |
| `model.layers.32.mlp.experts.0.up_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750004 | 0.750000 | 0.000004 | `delta_above_base_cap` |
| `model.layers.45.mlp.experts.40.gate_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750003 | 0.750000 | 0.000003 | `delta_above_base_cap` |
| `model.layers.33.mlp.experts.109.up_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750002 | 0.750000 | 0.000002 | `delta_above_base_cap` |
| `model.layers.47.mlp.experts.65.gate_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750002 | 0.750000 | 0.000002 | `delta_above_base_cap` |
| `model.layers.9.mlp.experts.99.up_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750002 | 0.750000 | 0.000002 | `delta_above_base_cap` |
| `model.layers.6.mlp.experts.120.up_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750001 | 0.750000 | 0.000001 | `delta_above_base_cap` |
| `model.layers.42.mlp.experts.88.gate_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750001 | 0.750000 | 0.000001 | `delta_above_base_cap` |
| `model.layers.2.mlp.experts.101.up_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750000 | 0.750000 | 0.000000 | `delta_above_base_cap` |
| `model.layers.37.mlp.experts.51.up_proj.weight` | `scale_nonbase_delta_to_moe_trust_region` | 0.750000 | 0.750000 | 0.000000 | `delta_above_base_cap` |

## Files

- `results/qwen3_moe_trust_region_delta_validation/tensor_prediction_error.csv`
- `results/qwen3_moe_trust_region_delta_validation/group_prediction_error_summary.csv`
- `results/qwen3_moe_trust_region_delta_validation/action_prediction_error_summary.csv`
- `results/qwen3_moe_trust_region_delta_validation/risk_flag_prediction_error_summary.csv`
- `results/qwen3_moe_trust_region_delta_validation/threshold_residuals.csv`
- `results/qwen3_moe_trust_region_delta_validation/summary.json`
