# Materialized Checkpoint Delta Audit

这个审计直接读取已物化的 safetensors checkpoint，而不是只相信 writer manifest。它检查输出 tensor 是否与 base 同构，并量化 candidate 相对 base 的实际参数改动。

- Status: `passed`
- Tensor count: `18867`
- Changed tensors: `10353`
- Changed numel fraction: `0.533`
- Total relative delta norm: `0.248`
- Max abs delta: `0.627`
- Router changed tensors: `0/48`

## Group Summary

| group | tensors | changed | changed numel frac | delta norm | relative delta norm | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `routed_expert_ffn` | 18432 | 10353 | 0.562 | 1023.648 | 0.256 | 0.627 |
| `attention` | 288 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `embedding_or_head` | 2 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `norm` | 97 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `router` | 48 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

## Layer Summary

| layer | tensors | changed | changed numel frac | delta norm | relative delta norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
|  | 3 | 0 | 0.000 | 0.000 | 0.000 |
| 0 | 393 | 279 | 0.704 | 126.369 | 0.261 |
| 1 | 393 | 288 | 0.727 | 130.904 | 0.233 |
| 2 | 393 | 252 | 0.636 | 136.814 | 0.240 |
| 3 | 393 | 246 | 0.621 | 139.989 | 0.242 |
| 4 | 393 | 240 | 0.606 | 149.815 | 0.260 |
| 5 | 393 | 201 | 0.507 | 122.367 | 0.212 |
| 6 | 393 | 204 | 0.515 | 142.012 | 0.246 |
| 7 | 393 | 201 | 0.507 | 133.893 | 0.231 |
| 8 | 393 | 222 | 0.560 | 155.348 | 0.270 |
| 9 | 393 | 204 | 0.515 | 145.231 | 0.254 |
| 10 | 393 | 222 | 0.560 | 151.739 | 0.267 |
| 11 | 393 | 204 | 0.515 | 147.823 | 0.260 |
| 12 | 393 | 222 | 0.560 | 162.388 | 0.287 |
| 13 | 393 | 216 | 0.545 | 148.106 | 0.262 |
| 14 | 393 | 198 | 0.500 | 141.906 | 0.252 |
| 15 | 393 | 225 | 0.568 | 149.900 | 0.264 |
| 16 | 393 | 195 | 0.492 | 145.497 | 0.256 |
| 17 | 393 | 201 | 0.507 | 161.638 | 0.285 |
| 18 | 393 | 210 | 0.530 | 144.194 | 0.252 |
| 19 | 393 | 183 | 0.462 | 153.385 | 0.265 |
| 20 | 393 | 213 | 0.538 | 164.075 | 0.281 |
| 21 | 393 | 201 | 0.507 | 153.132 | 0.262 |
| 22 | 393 | 222 | 0.560 | 178.897 | 0.307 |
| 23 | 393 | 219 | 0.553 | 162.371 | 0.282 |
| 24 | 393 | 213 | 0.538 | 160.875 | 0.278 |
| 25 | 393 | 228 | 0.576 | 158.151 | 0.272 |
| 26 | 393 | 207 | 0.523 | 165.280 | 0.281 |
| 27 | 393 | 198 | 0.500 | 144.770 | 0.245 |
| 28 | 393 | 207 | 0.523 | 143.993 | 0.245 |
| 29 | 393 | 207 | 0.523 | 149.845 | 0.254 |
| 30 | 393 | 204 | 0.515 | 155.768 | 0.263 |
| 31 | 393 | 180 | 0.454 | 150.338 | 0.251 |
| 32 | 393 | 219 | 0.553 | 154.997 | 0.258 |
| 33 | 393 | 213 | 0.538 | 142.479 | 0.238 |
| 34 | 393 | 210 | 0.530 | 152.689 | 0.255 |
| 35 | 393 | 231 | 0.583 | 143.980 | 0.240 |
| 36 | 393 | 225 | 0.568 | 159.578 | 0.264 |
| 37 | 393 | 213 | 0.538 | 149.732 | 0.247 |
| 38 | 393 | 195 | 0.492 | 119.256 | 0.195 |
| 39 | 393 | 207 | 0.523 | 136.275 | 0.222 |
| 40 | 393 | 204 | 0.515 | 135.182 | 0.218 |
| 41 | 393 | 219 | 0.553 | 138.421 | 0.221 |
| 42 | 393 | 240 | 0.606 | 131.157 | 0.207 |
| 43 | 393 | 228 | 0.576 | 150.787 | 0.236 |
| 44 | 393 | 204 | 0.515 | 143.217 | 0.223 |
| 45 | 393 | 183 | 0.462 | 139.384 | 0.216 |
| 46 | 393 | 216 | 0.545 | 159.734 | 0.246 |
| 47 | 393 | 234 | 0.591 | 135.867 | 0.210 |

## Top 20 Changed Tensors

| tensor | group | layer | numel | delta norm | relative delta norm | max abs delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `model.layers.46.mlp.experts.101.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 22.128 | 0.647 | 0.114 |
| `model.layers.46.mlp.experts.56.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 22.117 | 0.650 | 0.167 |
| `model.layers.46.mlp.experts.101.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 21.959 | 0.650 | 0.169 |
| `model.layers.44.mlp.experts.106.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 21.896 | 0.643 | 0.143 |
| `model.layers.43.mlp.experts.25.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 21.876 | 0.650 | 0.121 |
| `model.layers.44.mlp.experts.4.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 21.736 | 0.650 | 0.151 |
| `model.layers.43.mlp.experts.105.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 21.602 | 0.650 | 0.088 |
| `model.layers.45.mlp.experts.6.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 21.507 | 0.650 | 0.118 |
| `model.layers.46.mlp.experts.56.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 21.499 | 0.615 | 0.312 |
| `model.layers.46.mlp.experts.47.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 21.443 | 0.638 | 0.088 |
| `model.layers.41.mlp.experts.23.up_proj.weight` | `routed_expert_ffn` | 41 | 1572864 | 21.212 | 0.650 | 0.089 |
| `model.layers.42.mlp.experts.54.up_proj.weight` | `routed_expert_ffn` | 42 | 1572864 | 21.144 | 0.650 | 0.094 |
| `model.layers.47.mlp.experts.119.gate_proj.weight` | `routed_expert_ffn` | 47 | 1572864 | 21.133 | 0.650 | 0.213 |
| `model.layers.44.mlp.experts.106.down_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 21.111 | 0.613 | 0.447 |
| `model.layers.46.mlp.experts.6.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 21.062 | 0.650 | 0.274 |
| `model.layers.46.mlp.experts.101.gate_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 21.055 | 0.638 | 0.107 |
| `model.layers.45.mlp.experts.105.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 21.027 | 0.641 | 0.091 |
| `model.layers.44.mlp.experts.18.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 21.002 | 0.632 | 0.107 |
| `model.layers.41.mlp.experts.109.up_proj.weight` | `routed_expert_ffn` | 41 | 1572864 | 20.977 | 0.650 | 0.088 |
| `model.layers.47.mlp.experts.51.up_proj.weight` | `routed_expert_ffn` | 47 | 1572864 | 20.960 | 0.620 | 0.203 |

## Files

- `results/qwen3_moe_searched_no_gt065_delta_audit/tensor_delta_audit.csv`
- `results/qwen3_moe_searched_no_gt065_delta_audit/group_delta_summary.csv`
- `results/qwen3_moe_searched_no_gt065_delta_audit/layer_delta_summary.csv`
- `results/qwen3_moe_searched_no_gt065_delta_audit/summary.json`
