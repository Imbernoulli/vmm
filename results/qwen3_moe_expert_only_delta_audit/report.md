# Materialized Checkpoint Delta Audit

这个审计直接读取已物化的 safetensors checkpoint，而不是只相信 writer manifest。它检查输出 tensor 是否与 base 同构，并量化 candidate 相对 base 的实际参数改动。

- Status: `passed`
- Tensor count: `18867`
- Changed tensors: `10353`
- Changed numel fraction: `0.533`
- Total relative delta norm: `0.246`
- Max abs delta: `0.598`
- Router changed tensors: `0/48`

## Group Summary

| group | tensors | changed | changed numel frac | delta norm | relative delta norm | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `routed_expert_ffn` | 18432 | 10353 | 0.562 | 1017.191 | 0.255 | 0.598 |
| `attention` | 288 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `embedding_or_head` | 2 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `norm` | 97 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `router` | 48 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

## Layer Summary

| layer | tensors | changed | changed numel frac | delta norm | relative delta norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
|  | 3 | 0 | 0.000 | 0.000 | 0.000 |
| 0 | 393 | 279 | 0.704 | 127.603 | 0.263 |
| 1 | 393 | 288 | 0.727 | 130.047 | 0.232 |
| 2 | 393 | 252 | 0.636 | 138.342 | 0.242 |
| 3 | 393 | 246 | 0.621 | 137.669 | 0.238 |
| 4 | 393 | 240 | 0.606 | 147.730 | 0.257 |
| 5 | 393 | 201 | 0.507 | 123.106 | 0.214 |
| 6 | 393 | 204 | 0.515 | 144.391 | 0.251 |
| 7 | 393 | 201 | 0.507 | 132.206 | 0.228 |
| 8 | 393 | 222 | 0.560 | 157.432 | 0.273 |
| 9 | 393 | 204 | 0.515 | 143.502 | 0.251 |
| 10 | 393 | 222 | 0.560 | 150.970 | 0.265 |
| 11 | 393 | 204 | 0.515 | 148.994 | 0.262 |
| 12 | 393 | 222 | 0.560 | 161.033 | 0.284 |
| 13 | 393 | 216 | 0.545 | 145.875 | 0.258 |
| 14 | 393 | 198 | 0.500 | 141.309 | 0.250 |
| 15 | 393 | 225 | 0.568 | 151.609 | 0.267 |
| 16 | 393 | 195 | 0.492 | 145.857 | 0.256 |
| 17 | 393 | 201 | 0.507 | 162.967 | 0.288 |
| 18 | 393 | 210 | 0.530 | 145.146 | 0.253 |
| 19 | 393 | 183 | 0.462 | 151.456 | 0.262 |
| 20 | 393 | 213 | 0.538 | 161.895 | 0.278 |
| 21 | 393 | 201 | 0.507 | 151.140 | 0.259 |
| 22 | 393 | 222 | 0.560 | 179.274 | 0.308 |
| 23 | 393 | 219 | 0.553 | 161.764 | 0.281 |
| 24 | 393 | 213 | 0.538 | 158.881 | 0.275 |
| 25 | 393 | 228 | 0.576 | 157.036 | 0.270 |
| 26 | 393 | 207 | 0.523 | 163.116 | 0.277 |
| 27 | 393 | 198 | 0.500 | 145.236 | 0.246 |
| 28 | 393 | 207 | 0.523 | 142.724 | 0.243 |
| 29 | 393 | 207 | 0.523 | 147.517 | 0.250 |
| 30 | 393 | 204 | 0.515 | 155.526 | 0.263 |
| 31 | 393 | 180 | 0.454 | 148.277 | 0.248 |
| 32 | 393 | 219 | 0.553 | 153.863 | 0.256 |
| 33 | 393 | 213 | 0.538 | 141.757 | 0.236 |
| 34 | 393 | 210 | 0.530 | 149.026 | 0.249 |
| 35 | 393 | 231 | 0.583 | 141.109 | 0.235 |
| 36 | 393 | 225 | 0.568 | 157.463 | 0.261 |
| 37 | 393 | 213 | 0.538 | 149.514 | 0.246 |
| 38 | 393 | 195 | 0.492 | 120.835 | 0.198 |
| 39 | 393 | 207 | 0.523 | 135.840 | 0.221 |
| 40 | 393 | 204 | 0.515 | 129.711 | 0.209 |
| 41 | 393 | 219 | 0.553 | 134.332 | 0.214 |
| 42 | 393 | 240 | 0.606 | 131.618 | 0.207 |
| 43 | 393 | 228 | 0.576 | 149.166 | 0.234 |
| 44 | 393 | 204 | 0.515 | 141.719 | 0.221 |
| 45 | 393 | 183 | 0.462 | 137.462 | 0.213 |
| 46 | 393 | 216 | 0.545 | 157.070 | 0.242 |
| 47 | 393 | 234 | 0.591 | 135.115 | 0.208 |

## Top 20 Changed Tensors

| tensor | group | layer | numel | delta norm | relative delta norm | max abs delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `model.layers.46.mlp.experts.101.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 25.533 | 0.746 | 0.131 |
| `model.layers.46.mlp.experts.101.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 25.338 | 0.750 | 0.195 |
| `model.layers.46.mlp.experts.101.gate_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 24.295 | 0.736 | 0.123 |
| `model.layers.47.mlp.experts.65.up_proj.weight` | `routed_expert_ffn` | 47 | 1572864 | 24.076 | 0.720 | 0.097 |
| `model.layers.45.mlp.experts.40.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 23.831 | 0.743 | 0.093 |
| `model.layers.47.mlp.experts.65.gate_proj.weight` | `routed_expert_ffn` | 47 | 1572864 | 23.804 | 0.750 | 0.101 |
| `model.layers.44.mlp.experts.4.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 23.756 | 0.710 | 0.165 |
| `model.layers.42.mlp.experts.87.up_proj.weight` | `routed_expert_ffn` | 42 | 1572864 | 23.628 | 0.747 | 0.102 |
| `model.layers.42.mlp.experts.87.gate_proj.weight` | `routed_expert_ffn` | 42 | 1572864 | 23.216 | 0.750 | 0.107 |
| `model.layers.43.mlp.experts.13.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 23.209 | 0.744 | 0.098 |
| `model.layers.42.mlp.experts.88.up_proj.weight` | `routed_expert_ffn` | 42 | 1572864 | 22.943 | 0.735 | 0.108 |
| `model.layers.45.mlp.experts.40.gate_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 22.893 | 0.750 | 0.098 |
| `model.layers.38.mlp.experts.35.up_proj.weight` | `routed_expert_ffn` | 38 | 1572864 | 22.813 | 0.749 | 0.104 |
| `model.layers.45.mlp.experts.40.down_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 22.733 | 0.703 | 0.115 |
| `model.layers.43.mlp.experts.23.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 22.712 | 0.729 | 0.094 |
| `model.layers.33.mlp.experts.20.up_proj.weight` | `routed_expert_ffn` | 33 | 1572864 | 22.702 | 0.750 | 0.156 |
| `model.layers.8.mlp.experts.119.gate_proj.weight` | `routed_expert_ffn` | 8 | 1572864 | 22.649 | 0.730 | 0.090 |
| `model.layers.47.mlp.experts.65.down_proj.weight` | `routed_expert_ffn` | 47 | 1572864 | 22.580 | 0.694 | 0.227 |
| `model.layers.42.mlp.experts.88.gate_proj.weight` | `routed_expert_ffn` | 42 | 1572864 | 22.545 | 0.750 | 0.086 |
| `model.layers.37.mlp.experts.51.up_proj.weight` | `routed_expert_ffn` | 37 | 1572864 | 22.450 | 0.750 | 0.135 |

## Files

- `results/qwen3_moe_expert_only_delta_audit/tensor_delta_audit.csv`
- `results/qwen3_moe_expert_only_delta_audit/group_delta_summary.csv`
- `results/qwen3_moe_expert_only_delta_audit/layer_delta_summary.csv`
- `results/qwen3_moe_expert_only_delta_audit/summary.json`
