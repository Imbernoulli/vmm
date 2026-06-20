# Materialized Checkpoint Delta Audit

这个审计直接读取已物化的 safetensors checkpoint，而不是只相信 writer manifest。它检查输出 tensor 是否与 base 同构，并量化 candidate 相对 base 的实际参数改动。

- Status: `passed`
- Tensor count: `18867`
- Changed tensors: `10353`
- Changed numel fraction: `0.533`
- Total relative delta norm: `0.238`
- Max abs delta: `0.598`
- Routed expert FFN relative delta norm: `0.246`
- Max routed expert FFN tensor relative delta: `0.649`
- Routed expert FFN tensors >0.65 / >0.6505: `0` / `0`
- Attention changed tensors: `0/288`
- Router changed tensors: `0/48`

## Group Summary

| group | tensors | changed | changed numel frac | delta norm | relative delta norm | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `routed_expert_ffn` | 18432 | 10353 | 0.562 | 984.912 | 0.246 | 0.598 |
| `attention` | 288 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `embedding_or_head` | 2 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `norm` | 97 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `router` | 48 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

## Layer Summary

| layer | tensors | changed | changed numel frac | delta norm | relative delta norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
|  | 3 | 0 | 0.000 | 0.000 | 0.000 |
| 0 | 393 | 279 | 0.704 | 123.623 | 0.255 |
| 1 | 393 | 288 | 0.727 | 128.349 | 0.229 |
| 2 | 393 | 252 | 0.636 | 133.751 | 0.234 |
| 3 | 393 | 246 | 0.621 | 136.353 | 0.235 |
| 4 | 393 | 240 | 0.606 | 144.500 | 0.251 |
| 5 | 393 | 201 | 0.507 | 119.980 | 0.208 |
| 6 | 393 | 204 | 0.515 | 137.364 | 0.238 |
| 7 | 393 | 201 | 0.507 | 129.694 | 0.224 |
| 8 | 393 | 222 | 0.560 | 149.702 | 0.260 |
| 9 | 393 | 204 | 0.515 | 139.530 | 0.244 |
| 10 | 393 | 222 | 0.560 | 142.996 | 0.251 |
| 11 | 393 | 204 | 0.515 | 140.924 | 0.248 |
| 12 | 393 | 222 | 0.560 | 151.897 | 0.268 |
| 13 | 393 | 216 | 0.545 | 139.831 | 0.248 |
| 14 | 393 | 198 | 0.500 | 134.601 | 0.239 |
| 15 | 393 | 225 | 0.568 | 141.023 | 0.248 |
| 16 | 393 | 195 | 0.492 | 137.138 | 0.241 |
| 17 | 393 | 201 | 0.507 | 151.179 | 0.267 |
| 18 | 393 | 210 | 0.530 | 136.120 | 0.238 |
| 19 | 393 | 183 | 0.462 | 145.410 | 0.251 |
| 20 | 393 | 213 | 0.538 | 156.415 | 0.268 |
| 21 | 393 | 201 | 0.507 | 145.243 | 0.248 |
| 22 | 393 | 222 | 0.560 | 167.426 | 0.288 |
| 23 | 393 | 219 | 0.553 | 153.634 | 0.267 |
| 24 | 393 | 213 | 0.538 | 151.549 | 0.262 |
| 25 | 393 | 228 | 0.576 | 149.217 | 0.256 |
| 26 | 393 | 207 | 0.523 | 156.500 | 0.266 |
| 27 | 393 | 198 | 0.500 | 138.754 | 0.235 |
| 28 | 393 | 207 | 0.523 | 137.941 | 0.235 |
| 29 | 393 | 207 | 0.523 | 142.148 | 0.241 |
| 30 | 393 | 204 | 0.515 | 149.359 | 0.252 |
| 31 | 393 | 180 | 0.454 | 144.049 | 0.241 |
| 32 | 393 | 219 | 0.553 | 150.147 | 0.250 |
| 33 | 393 | 213 | 0.538 | 138.182 | 0.230 |
| 34 | 393 | 210 | 0.530 | 146.406 | 0.245 |
| 35 | 393 | 231 | 0.583 | 140.183 | 0.234 |
| 36 | 393 | 225 | 0.568 | 154.519 | 0.256 |
| 37 | 393 | 213 | 0.538 | 147.287 | 0.243 |
| 38 | 393 | 195 | 0.492 | 118.251 | 0.194 |
| 39 | 393 | 207 | 0.523 | 133.990 | 0.218 |
| 40 | 393 | 204 | 0.515 | 134.304 | 0.216 |
| 41 | 393 | 219 | 0.553 | 137.006 | 0.218 |
| 42 | 393 | 240 | 0.606 | 129.881 | 0.205 |
| 43 | 393 | 228 | 0.576 | 150.151 | 0.235 |
| 44 | 393 | 204 | 0.515 | 142.291 | 0.222 |
| 45 | 393 | 183 | 0.462 | 137.576 | 0.213 |
| 46 | 393 | 216 | 0.545 | 157.167 | 0.242 |
| 47 | 393 | 234 | 0.591 | 133.801 | 0.206 |

## Top 20 Changed Tensors

| tensor | group | layer | numel | delta norm | relative delta norm | max abs delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `model.layers.46.mlp.experts.56.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 22.082 | 0.649 | 0.167 |
| `model.layers.44.mlp.experts.106.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 21.862 | 0.642 | 0.143 |
| `model.layers.43.mlp.experts.25.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 21.829 | 0.649 | 0.121 |
| `model.layers.44.mlp.experts.4.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 21.702 | 0.649 | 0.151 |
| `model.layers.43.mlp.experts.105.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 21.569 | 0.649 | 0.088 |
| `model.layers.45.mlp.experts.6.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 21.474 | 0.649 | 0.118 |
| `model.layers.46.mlp.experts.56.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 21.465 | 0.615 | 0.310 |
| `model.layers.46.mlp.experts.47.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 21.180 | 0.630 | 0.087 |
| `model.layers.41.mlp.experts.23.up_proj.weight` | `routed_expert_ffn` | 41 | 1572864 | 21.179 | 0.649 | 0.089 |
| `model.layers.42.mlp.experts.54.up_proj.weight` | `routed_expert_ffn` | 42 | 1572864 | 21.113 | 0.649 | 0.094 |
| `model.layers.44.mlp.experts.106.down_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 21.078 | 0.612 | 0.446 |
| `model.layers.44.mlp.experts.18.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 21.002 | 0.632 | 0.107 |
| `model.layers.45.mlp.experts.105.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 20.995 | 0.640 | 0.091 |
| `model.layers.46.mlp.experts.101.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 20.911 | 0.611 | 0.108 |
| `model.layers.46.mlp.experts.6.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 20.813 | 0.642 | 0.271 |
| `model.layers.47.mlp.experts.119.gate_proj.weight` | `routed_expert_ffn` | 47 | 1572864 | 20.796 | 0.640 | 0.210 |
| `model.layers.46.mlp.experts.120.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 20.790 | 0.627 | 0.144 |
| `model.layers.45.mlp.experts.110.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 20.790 | 0.642 | 0.092 |
| `model.layers.42.mlp.experts.105.up_proj.weight` | `routed_expert_ffn` | 42 | 1572864 | 20.775 | 0.648 | 0.088 |
| `model.layers.46.mlp.experts.101.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 20.751 | 0.614 | 0.159 |

## Files

- `results/qwen3_moe_mechanistic_unified_delta_audit/tensor_delta_audit.csv`
- `results/qwen3_moe_mechanistic_unified_delta_audit/group_delta_summary.csv`
- `results/qwen3_moe_mechanistic_unified_delta_audit/layer_delta_summary.csv`
- `results/qwen3_moe_mechanistic_unified_delta_audit/summary.json`
