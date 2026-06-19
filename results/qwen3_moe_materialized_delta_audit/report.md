# Materialized Checkpoint Delta Audit

这个审计直接读取已物化的 safetensors checkpoint，而不是只相信 writer manifest。它检查输出 tensor 是否与 base 同构，并量化 candidate 相对 base 的实际参数改动。

- Status: `passed`
- Tensor count: `18867`
- Changed tensors: `10641`
- Changed numel fraction: `0.563`
- Total relative delta norm: `0.286`
- Max abs delta: `1.688`
- Router changed tensors: `0/48`

## Group Summary

| group | tensors | changed | changed numel frac | delta norm | relative delta norm | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `routed_expert_ffn` | 18432 | 10353 | 0.562 | 1171.486 | 0.293 | 0.744 |
| `attention` | 288 | 288 | 1.000 | 149.056 | 0.189 | 1.688 |
| `embedding_or_head` | 2 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `norm` | 97 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `router` | 48 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

## Layer Summary

| layer | tensors | changed | changed numel frac | delta norm | relative delta norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
|  | 3 | 0 | 0.000 | 0.000 | 0.000 |
| 0 | 393 | 285 | 0.735 | 140.285 | 0.290 |
| 1 | 393 | 294 | 0.757 | 137.030 | 0.244 |
| 2 | 393 | 258 | 0.666 | 157.238 | 0.275 |
| 3 | 393 | 252 | 0.651 | 159.998 | 0.276 |
| 4 | 393 | 246 | 0.636 | 178.429 | 0.310 |
| 5 | 393 | 207 | 0.538 | 133.647 | 0.232 |
| 6 | 393 | 210 | 0.545 | 171.008 | 0.297 |
| 7 | 393 | 207 | 0.538 | 147.321 | 0.254 |
| 8 | 393 | 228 | 0.591 | 180.124 | 0.313 |
| 9 | 393 | 210 | 0.545 | 176.146 | 0.308 |
| 10 | 393 | 228 | 0.591 | 171.882 | 0.302 |
| 11 | 393 | 210 | 0.545 | 177.913 | 0.313 |
| 12 | 393 | 228 | 0.591 | 192.690 | 0.340 |
| 13 | 393 | 222 | 0.576 | 180.228 | 0.319 |
| 14 | 393 | 204 | 0.530 | 159.652 | 0.283 |
| 15 | 393 | 231 | 0.598 | 176.030 | 0.310 |
| 16 | 393 | 201 | 0.523 | 171.678 | 0.302 |
| 17 | 393 | 207 | 0.538 | 196.574 | 0.347 |
| 18 | 393 | 216 | 0.560 | 180.043 | 0.314 |
| 19 | 393 | 189 | 0.492 | 181.122 | 0.313 |
| 20 | 393 | 219 | 0.568 | 198.096 | 0.340 |
| 21 | 393 | 207 | 0.538 | 188.193 | 0.322 |
| 22 | 393 | 228 | 0.591 | 205.833 | 0.354 |
| 23 | 393 | 225 | 0.583 | 193.229 | 0.336 |
| 24 | 393 | 219 | 0.568 | 188.821 | 0.326 |
| 25 | 393 | 234 | 0.606 | 183.755 | 0.316 |
| 26 | 393 | 213 | 0.553 | 201.528 | 0.343 |
| 27 | 393 | 204 | 0.530 | 168.061 | 0.284 |
| 28 | 393 | 213 | 0.553 | 163.928 | 0.279 |
| 29 | 393 | 213 | 0.553 | 178.901 | 0.304 |
| 30 | 393 | 210 | 0.545 | 184.168 | 0.311 |
| 31 | 393 | 186 | 0.485 | 172.259 | 0.288 |
| 32 | 393 | 225 | 0.583 | 176.498 | 0.293 |
| 33 | 393 | 219 | 0.568 | 166.605 | 0.278 |
| 34 | 393 | 216 | 0.560 | 183.943 | 0.307 |
| 35 | 393 | 237 | 0.613 | 158.400 | 0.264 |
| 36 | 393 | 231 | 0.598 | 177.833 | 0.295 |
| 37 | 393 | 219 | 0.568 | 170.635 | 0.281 |
| 38 | 393 | 201 | 0.523 | 128.698 | 0.211 |
| 39 | 393 | 213 | 0.553 | 146.757 | 0.239 |
| 40 | 393 | 210 | 0.545 | 144.613 | 0.233 |
| 41 | 393 | 225 | 0.583 | 149.109 | 0.238 |
| 42 | 393 | 246 | 0.636 | 142.733 | 0.225 |
| 43 | 393 | 234 | 0.606 | 160.907 | 0.252 |
| 44 | 393 | 210 | 0.545 | 151.779 | 0.237 |
| 45 | 393 | 189 | 0.492 | 159.703 | 0.247 |
| 46 | 393 | 222 | 0.576 | 172.164 | 0.265 |
| 47 | 393 | 240 | 0.621 | 147.314 | 0.227 |

## Top 20 Changed Tensors

| tensor | group | layer | numel | delta norm | relative delta norm | max abs delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `model.layers.27.mlp.experts.21.gate_proj.weight` | `routed_expert_ffn` | 27 | 1572864 | 33.359 | 1.083 | 0.134 |
| `model.layers.26.mlp.experts.116.gate_proj.weight` | `routed_expert_ffn` | 26 | 1572864 | 32.904 | 1.087 | 0.137 |
| `model.layers.27.mlp.experts.21.up_proj.weight` | `routed_expert_ffn` | 27 | 1572864 | 32.841 | 1.098 | 0.133 |
| `model.layers.15.mlp.experts.21.gate_proj.weight` | `routed_expert_ffn` | 15 | 1572864 | 32.774 | 1.117 | 0.170 |
| `model.layers.26.mlp.experts.116.up_proj.weight` | `routed_expert_ffn` | 26 | 1572864 | 32.704 | 1.101 | 0.142 |
| `model.layers.20.mlp.experts.118.down_proj.weight` | `routed_expert_ffn` | 20 | 1572864 | 32.633 | 1.126 | 0.132 |
| `model.layers.21.mlp.experts.67.up_proj.weight` | `routed_expert_ffn` | 21 | 1572864 | 32.553 | 1.126 | 0.130 |
| `model.layers.34.mlp.experts.54.up_proj.weight` | `routed_expert_ffn` | 34 | 1572864 | 32.467 | 1.070 | 0.187 |
| `model.layers.21.mlp.experts.67.gate_proj.weight` | `routed_expert_ffn` | 21 | 1572864 | 32.458 | 1.111 | 0.139 |
| `model.layers.24.mlp.experts.77.gate_proj.weight` | `routed_expert_ffn` | 24 | 1572864 | 32.430 | 1.096 | 0.208 |
| `model.layers.28.mlp.experts.116.down_proj.weight` | `routed_expert_ffn` | 28 | 1572864 | 32.429 | 1.099 | 0.156 |
| `model.layers.28.mlp.experts.116.up_proj.weight` | `routed_expert_ffn` | 28 | 1572864 | 32.427 | 1.122 | 0.188 |
| `model.layers.15.mlp.experts.21.up_proj.weight` | `routed_expert_ffn` | 15 | 1572864 | 32.415 | 1.129 | 0.138 |
| `model.layers.20.mlp.experts.118.up_proj.weight` | `routed_expert_ffn` | 20 | 1572864 | 32.397 | 1.153 | 0.143 |
| `model.layers.22.mlp.experts.114.gate_proj.weight` | `routed_expert_ffn` | 22 | 1572864 | 32.329 | 1.087 | 0.134 |
| `model.layers.21.mlp.experts.115.gate_proj.weight` | `routed_expert_ffn` | 21 | 1572864 | 32.313 | 1.092 | 0.128 |
| `model.layers.27.mlp.experts.12.gate_proj.weight` | `routed_expert_ffn` | 27 | 1572864 | 32.301 | 1.084 | 0.128 |
| `model.layers.22.mlp.experts.114.up_proj.weight` | `routed_expert_ffn` | 22 | 1572864 | 32.264 | 1.104 | 0.126 |
| `model.layers.20.mlp.experts.118.gate_proj.weight` | `routed_expert_ffn` | 20 | 1572864 | 32.253 | 1.132 | 0.123 |
| `model.layers.27.mlp.experts.12.up_proj.weight` | `routed_expert_ffn` | 27 | 1572864 | 32.226 | 1.097 | 0.129 |

## Files

- `results/qwen3_moe_materialized_delta_audit/tensor_delta_audit.csv`
- `results/qwen3_moe_materialized_delta_audit/group_delta_summary.csv`
- `results/qwen3_moe_materialized_delta_audit/layer_delta_summary.csv`
- `results/qwen3_moe_materialized_delta_audit/summary.json`
