# Materialized Checkpoint Delta Audit

这个审计直接读取已物化的 safetensors checkpoint，而不是只相信 writer manifest。它检查输出 tensor 是否与 base 同构，并量化 candidate 相对 base 的实际参数改动。

- Status: `passed`
- Tensor count: `18867`
- Changed tensors: `10353`
- Changed numel fraction: `0.533`
- Total relative delta norm: `0.240`
- Max abs delta: `0.594`
- Router changed tensors: `0/48`

## Group Summary

| group | tensors | changed | changed numel frac | delta norm | relative delta norm | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `routed_expert_ffn` | 18432 | 10353 | 0.562 | 993.813 | 0.249 | 0.594 |
| `attention` | 288 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `embedding_or_head` | 2 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `norm` | 97 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `router` | 48 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

## Layer Summary

| layer | tensors | changed | changed numel frac | delta norm | relative delta norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
|  | 3 | 0 | 0.000 | 0.000 | 0.000 |
| 0 | 393 | 279 | 0.704 | 123.365 | 0.255 |
| 1 | 393 | 288 | 0.727 | 129.870 | 0.231 |
| 2 | 393 | 252 | 0.636 | 133.676 | 0.234 |
| 3 | 393 | 246 | 0.621 | 137.054 | 0.237 |
| 4 | 393 | 240 | 0.606 | 145.450 | 0.253 |
| 5 | 393 | 201 | 0.507 | 121.219 | 0.210 |
| 6 | 393 | 204 | 0.515 | 137.924 | 0.239 |
| 7 | 393 | 201 | 0.507 | 130.788 | 0.226 |
| 8 | 393 | 222 | 0.560 | 150.521 | 0.261 |
| 9 | 393 | 204 | 0.515 | 140.602 | 0.246 |
| 10 | 393 | 222 | 0.560 | 145.396 | 0.255 |
| 11 | 393 | 204 | 0.515 | 143.481 | 0.252 |
| 12 | 393 | 222 | 0.560 | 154.847 | 0.273 |
| 13 | 393 | 216 | 0.545 | 142.393 | 0.252 |
| 14 | 393 | 198 | 0.500 | 137.408 | 0.244 |
| 15 | 393 | 225 | 0.568 | 144.057 | 0.254 |
| 16 | 393 | 195 | 0.492 | 139.236 | 0.245 |
| 17 | 393 | 201 | 0.507 | 153.606 | 0.271 |
| 18 | 393 | 210 | 0.530 | 138.082 | 0.241 |
| 19 | 393 | 183 | 0.462 | 147.309 | 0.255 |
| 20 | 393 | 213 | 0.538 | 158.612 | 0.272 |
| 21 | 393 | 201 | 0.507 | 147.610 | 0.252 |
| 22 | 393 | 222 | 0.560 | 170.428 | 0.293 |
| 23 | 393 | 219 | 0.553 | 156.199 | 0.271 |
| 24 | 393 | 213 | 0.538 | 154.195 | 0.266 |
| 25 | 393 | 228 | 0.576 | 151.370 | 0.260 |
| 26 | 393 | 207 | 0.523 | 159.028 | 0.270 |
| 27 | 393 | 198 | 0.500 | 140.800 | 0.238 |
| 28 | 393 | 207 | 0.523 | 140.041 | 0.238 |
| 29 | 393 | 207 | 0.523 | 144.383 | 0.245 |
| 30 | 393 | 204 | 0.515 | 150.874 | 0.255 |
| 31 | 393 | 180 | 0.454 | 145.690 | 0.244 |
| 32 | 393 | 219 | 0.553 | 150.723 | 0.250 |
| 33 | 393 | 213 | 0.538 | 138.500 | 0.231 |
| 34 | 393 | 210 | 0.530 | 147.841 | 0.247 |
| 35 | 393 | 231 | 0.583 | 141.247 | 0.235 |
| 36 | 393 | 225 | 0.568 | 156.453 | 0.259 |
| 37 | 393 | 213 | 0.538 | 146.142 | 0.241 |
| 38 | 393 | 195 | 0.492 | 118.060 | 0.193 |
| 39 | 393 | 207 | 0.523 | 134.037 | 0.218 |
| 40 | 393 | 204 | 0.515 | 133.727 | 0.215 |
| 41 | 393 | 219 | 0.553 | 136.267 | 0.217 |
| 42 | 393 | 240 | 0.606 | 129.347 | 0.204 |
| 43 | 393 | 228 | 0.576 | 149.392 | 0.234 |
| 44 | 393 | 204 | 0.515 | 141.813 | 0.221 |
| 45 | 393 | 183 | 0.462 | 136.834 | 0.212 |
| 46 | 393 | 216 | 0.545 | 157.040 | 0.242 |
| 47 | 393 | 234 | 0.591 | 134.450 | 0.207 |

## Top 20 Changed Tensors

| tensor | group | layer | numel | delta norm | relative delta norm | max abs delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `model.layers.46.mlp.experts.56.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 21.644 | 0.636 | 0.164 |
| `model.layers.44.mlp.experts.106.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 21.505 | 0.631 | 0.141 |
| `model.layers.43.mlp.experts.25.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 21.452 | 0.637 | 0.119 |
| `model.layers.44.mlp.experts.4.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 21.324 | 0.638 | 0.148 |
| `model.layers.43.mlp.experts.105.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 21.321 | 0.642 | 0.087 |
| `model.layers.45.mlp.experts.6.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 21.117 | 0.638 | 0.116 |
| `model.layers.46.mlp.experts.56.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 21.039 | 0.602 | 0.304 |
| `model.layers.44.mlp.experts.18.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 21.002 | 0.632 | 0.107 |
| `model.layers.46.mlp.experts.47.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 20.812 | 0.619 | 0.085 |
| `model.layers.46.mlp.experts.101.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 20.760 | 0.607 | 0.107 |
| `model.layers.41.mlp.experts.23.up_proj.weight` | `routed_expert_ffn` | 41 | 1572864 | 20.736 | 0.635 | 0.087 |
| `model.layers.44.mlp.experts.106.down_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 20.734 | 0.602 | 0.439 |
| `model.layers.45.mlp.experts.105.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 20.682 | 0.631 | 0.089 |
| `model.layers.46.mlp.experts.120.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 20.625 | 0.622 | 0.143 |
| `model.layers.46.mlp.experts.101.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 20.601 | 0.610 | 0.158 |
| `model.layers.42.mlp.experts.54.up_proj.weight` | `routed_expert_ffn` | 42 | 1572864 | 20.566 | 0.632 | 0.091 |
| `model.layers.47.mlp.experts.51.up_proj.weight` | `routed_expert_ffn` | 47 | 1572864 | 20.545 | 0.608 | 0.198 |
| `model.layers.47.mlp.experts.119.gate_proj.weight` | `routed_expert_ffn` | 47 | 1572864 | 20.535 | 0.632 | 0.207 |
| `model.layers.46.mlp.experts.25.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 20.473 | 0.635 | 0.244 |
| `model.layers.45.mlp.experts.61.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 20.451 | 0.628 | 0.102 |

## Files

- `results/qwen3_moe_unified_mechanism_delta_audit/tensor_delta_audit.csv`
- `results/qwen3_moe_unified_mechanism_delta_audit/group_delta_summary.csv`
- `results/qwen3_moe_unified_mechanism_delta_audit/layer_delta_summary.csv`
- `results/qwen3_moe_unified_mechanism_delta_audit/summary.json`
