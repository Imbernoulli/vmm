# Materialized Checkpoint Delta Audit

这个审计直接读取已物化的 safetensors checkpoint，而不是只相信 writer manifest。它检查输出 tensor 是否与 base 同构，并量化 candidate 相对 base 的实际参数改动。

- Status: `passed`
- Tensor count: `18867`
- Changed tensors: `10641`
- Changed numel fraction: `0.563`
- Total relative delta norm: `0.249`
- Max abs delta: `1.688`
- Router changed tensors: `0/48`

## Group Summary

| group | tensors | changed | changed numel frac | delta norm | relative delta norm | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `routed_expert_ffn` | 18432 | 10353 | 0.562 | 1017.191 | 0.255 | 0.598 |
| `attention` | 288 | 288 | 1.000 | 149.056 | 0.189 | 1.688 |
| `embedding_or_head` | 2 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `norm` | 97 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `router` | 48 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

## Layer Summary

| layer | tensors | changed | changed numel frac | delta norm | relative delta norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
|  | 3 | 0 | 0.000 | 0.000 | 0.000 |
| 0 | 393 | 285 | 0.735 | 129.039 | 0.266 |
| 1 | 393 | 294 | 0.757 | 131.950 | 0.235 |
| 2 | 393 | 258 | 0.666 | 140.352 | 0.246 |
| 3 | 393 | 252 | 0.651 | 139.552 | 0.241 |
| 4 | 393 | 246 | 0.636 | 149.687 | 0.260 |
| 5 | 393 | 207 | 0.538 | 125.574 | 0.218 |
| 6 | 393 | 210 | 0.545 | 146.153 | 0.254 |
| 7 | 393 | 207 | 0.538 | 134.155 | 0.232 |
| 8 | 393 | 228 | 0.591 | 159.316 | 0.277 |
| 9 | 393 | 210 | 0.545 | 145.662 | 0.255 |
| 10 | 393 | 228 | 0.591 | 152.955 | 0.269 |
| 11 | 393 | 210 | 0.545 | 151.249 | 0.266 |
| 12 | 393 | 228 | 0.591 | 163.196 | 0.288 |
| 13 | 393 | 222 | 0.576 | 148.437 | 0.263 |
| 14 | 393 | 204 | 0.530 | 143.476 | 0.254 |
| 15 | 393 | 231 | 0.598 | 153.660 | 0.270 |
| 16 | 393 | 201 | 0.523 | 148.047 | 0.260 |
| 17 | 393 | 207 | 0.538 | 165.015 | 0.291 |
| 18 | 393 | 216 | 0.560 | 147.230 | 0.257 |
| 19 | 393 | 189 | 0.492 | 153.299 | 0.265 |
| 20 | 393 | 219 | 0.568 | 163.494 | 0.280 |
| 21 | 393 | 207 | 0.538 | 152.943 | 0.262 |
| 22 | 393 | 228 | 0.591 | 180.644 | 0.310 |
| 23 | 393 | 225 | 0.583 | 163.318 | 0.284 |
| 24 | 393 | 219 | 0.568 | 160.480 | 0.277 |
| 25 | 393 | 234 | 0.606 | 158.673 | 0.273 |
| 26 | 393 | 213 | 0.553 | 164.423 | 0.279 |
| 27 | 393 | 204 | 0.530 | 146.710 | 0.248 |
| 28 | 393 | 213 | 0.553 | 144.302 | 0.245 |
| 29 | 393 | 213 | 0.553 | 149.215 | 0.253 |
| 30 | 393 | 210 | 0.545 | 156.845 | 0.265 |
| 31 | 393 | 186 | 0.485 | 149.572 | 0.250 |
| 32 | 393 | 225 | 0.583 | 155.113 | 0.258 |
| 33 | 393 | 219 | 0.568 | 142.928 | 0.238 |
| 34 | 393 | 216 | 0.560 | 150.125 | 0.251 |
| 35 | 393 | 237 | 0.613 | 142.188 | 0.237 |
| 36 | 393 | 231 | 0.598 | 158.467 | 0.263 |
| 37 | 393 | 219 | 0.568 | 150.404 | 0.248 |
| 38 | 393 | 201 | 0.523 | 121.786 | 0.200 |
| 39 | 393 | 213 | 0.553 | 136.674 | 0.222 |
| 40 | 393 | 210 | 0.545 | 130.750 | 0.210 |
| 41 | 393 | 225 | 0.583 | 135.281 | 0.216 |
| 42 | 393 | 246 | 0.636 | 132.573 | 0.209 |
| 43 | 393 | 234 | 0.606 | 150.070 | 0.235 |
| 44 | 393 | 210 | 0.545 | 142.903 | 0.223 |
| 45 | 393 | 189 | 0.492 | 138.304 | 0.214 |
| 46 | 393 | 222 | 0.576 | 158.064 | 0.244 |
| 47 | 393 | 240 | 0.621 | 136.296 | 0.210 |

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

- `results/qwen3_moe_trust_region_delta_audit/tensor_delta_audit.csv`
- `results/qwen3_moe_trust_region_delta_audit/group_delta_summary.csv`
- `results/qwen3_moe_trust_region_delta_audit/layer_delta_summary.csv`
- `results/qwen3_moe_trust_region_delta_audit/summary.json`
