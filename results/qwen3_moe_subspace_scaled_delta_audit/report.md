# Materialized Checkpoint Delta Audit

这个审计直接读取已物化的 safetensors checkpoint，而不是只相信 writer manifest。它检查输出 tensor 是否与 base 同构，并量化 candidate 相对 base 的实际参数改动。

- Status: `passed`
- Tensor count: `18867`
- Changed tensors: `10353`
- Changed numel fraction: `0.533`
- Total relative delta norm: `0.240`
- Max abs delta: `0.574`
- Router changed tensors: `0/48`

## Group Summary

| group | tensors | changed | changed numel frac | delta norm | relative delta norm | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `routed_expert_ffn` | 18432 | 10353 | 0.562 | 990.303 | 0.248 | 0.574 |
| `attention` | 288 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `embedding_or_head` | 2 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `norm` | 97 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `router` | 48 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

## Layer Summary

| layer | tensors | changed | changed numel frac | delta norm | relative delta norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
|  | 3 | 0 | 0.000 | 0.000 | 0.000 |
| 0 | 393 | 279 | 0.704 | 123.839 | 0.256 |
| 1 | 393 | 288 | 0.727 | 129.859 | 0.231 |
| 2 | 393 | 252 | 0.636 | 133.652 | 0.234 |
| 3 | 393 | 246 | 0.621 | 136.691 | 0.236 |
| 4 | 393 | 240 | 0.606 | 144.719 | 0.251 |
| 5 | 393 | 201 | 0.507 | 121.009 | 0.210 |
| 6 | 393 | 204 | 0.515 | 137.910 | 0.239 |
| 7 | 393 | 201 | 0.507 | 130.535 | 0.225 |
| 8 | 393 | 222 | 0.560 | 150.522 | 0.261 |
| 9 | 393 | 204 | 0.515 | 140.721 | 0.246 |
| 10 | 393 | 222 | 0.560 | 146.795 | 0.258 |
| 11 | 393 | 204 | 0.515 | 143.784 | 0.253 |
| 12 | 393 | 222 | 0.560 | 155.917 | 0.275 |
| 13 | 393 | 216 | 0.545 | 143.093 | 0.254 |
| 14 | 393 | 198 | 0.500 | 137.360 | 0.243 |
| 15 | 393 | 225 | 0.568 | 144.212 | 0.254 |
| 16 | 393 | 195 | 0.492 | 139.491 | 0.245 |
| 17 | 393 | 201 | 0.507 | 154.373 | 0.272 |
| 18 | 393 | 210 | 0.530 | 138.096 | 0.241 |
| 19 | 393 | 183 | 0.462 | 145.534 | 0.252 |
| 20 | 393 | 213 | 0.538 | 157.215 | 0.270 |
| 21 | 393 | 201 | 0.507 | 147.374 | 0.252 |
| 22 | 393 | 222 | 0.560 | 170.889 | 0.294 |
| 23 | 393 | 219 | 0.553 | 156.368 | 0.272 |
| 24 | 393 | 213 | 0.538 | 153.963 | 0.266 |
| 25 | 393 | 228 | 0.576 | 150.889 | 0.259 |
| 26 | 393 | 207 | 0.523 | 159.453 | 0.271 |
| 27 | 393 | 198 | 0.500 | 140.224 | 0.237 |
| 28 | 393 | 207 | 0.523 | 139.498 | 0.237 |
| 29 | 393 | 207 | 0.523 | 144.097 | 0.245 |
| 30 | 393 | 204 | 0.515 | 149.266 | 0.252 |
| 31 | 393 | 180 | 0.454 | 144.287 | 0.241 |
| 32 | 393 | 219 | 0.553 | 149.895 | 0.249 |
| 33 | 393 | 213 | 0.538 | 138.074 | 0.230 |
| 34 | 393 | 210 | 0.530 | 147.210 | 0.246 |
| 35 | 393 | 231 | 0.583 | 140.194 | 0.234 |
| 36 | 393 | 225 | 0.568 | 155.712 | 0.258 |
| 37 | 393 | 213 | 0.538 | 146.036 | 0.241 |
| 38 | 393 | 195 | 0.492 | 117.347 | 0.192 |
| 39 | 393 | 207 | 0.523 | 133.399 | 0.217 |
| 40 | 393 | 204 | 0.515 | 132.383 | 0.213 |
| 41 | 393 | 219 | 0.553 | 135.281 | 0.216 |
| 42 | 393 | 240 | 0.606 | 128.718 | 0.203 |
| 43 | 393 | 228 | 0.576 | 146.166 | 0.229 |
| 44 | 393 | 204 | 0.515 | 139.278 | 0.217 |
| 45 | 393 | 183 | 0.462 | 133.968 | 0.207 |
| 46 | 393 | 216 | 0.545 | 154.198 | 0.238 |
| 47 | 393 | 234 | 0.591 | 133.355 | 0.206 |

## Top 20 Changed Tensors

| tensor | group | layer | numel | delta norm | relative delta norm | max abs delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `model.layers.46.mlp.experts.56.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 20.555 | 0.604 | 0.155 |
| `model.layers.46.mlp.experts.101.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 20.452 | 0.598 | 0.105 |
| `model.layers.44.mlp.experts.4.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 20.384 | 0.610 | 0.142 |
| `model.layers.44.mlp.experts.106.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 20.367 | 0.598 | 0.133 |
| `model.layers.43.mlp.experts.25.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 20.347 | 0.605 | 0.113 |
| `model.layers.46.mlp.experts.101.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 20.295 | 0.601 | 0.156 |
| `model.layers.44.mlp.experts.18.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 20.054 | 0.603 | 0.102 |
| `model.layers.41.mlp.experts.23.up_proj.weight` | `routed_expert_ffn` | 41 | 1572864 | 20.043 | 0.614 | 0.084 |
| `model.layers.42.mlp.experts.54.up_proj.weight` | `routed_expert_ffn` | 42 | 1572864 | 20.034 | 0.616 | 0.089 |
| `model.layers.43.mlp.experts.105.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 20.008 | 0.602 | 0.081 |
| `model.layers.46.mlp.experts.56.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 19.981 | 0.572 | 0.288 |
| `model.layers.45.mlp.experts.6.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 19.939 | 0.603 | 0.109 |
| `model.layers.41.mlp.experts.109.up_proj.weight` | `routed_expert_ffn` | 41 | 1572864 | 19.901 | 0.617 | 0.083 |
| `model.layers.46.mlp.experts.47.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 19.721 | 0.587 | 0.081 |
| `model.layers.46.mlp.experts.72.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 19.686 | 0.589 | 0.139 |
| `model.layers.42.mlp.experts.105.up_proj.weight` | `routed_expert_ffn` | 42 | 1572864 | 19.666 | 0.613 | 0.083 |
| `model.layers.44.mlp.experts.106.down_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 19.637 | 0.570 | 0.416 |
| `model.layers.43.mlp.experts.103.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 19.637 | 0.606 | 0.170 |
| `model.layers.43.mlp.experts.9.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 19.608 | 0.607 | 0.115 |
| `model.layers.37.mlp.experts.115.up_proj.weight` | `routed_expert_ffn` | 37 | 1572864 | 19.592 | 0.621 | 0.099 |

## Files

- `results/qwen3_moe_subspace_scaled_delta_audit/tensor_delta_audit.csv`
- `results/qwen3_moe_subspace_scaled_delta_audit/group_delta_summary.csv`
- `results/qwen3_moe_subspace_scaled_delta_audit/layer_delta_summary.csv`
- `results/qwen3_moe_subspace_scaled_delta_audit/summary.json`
