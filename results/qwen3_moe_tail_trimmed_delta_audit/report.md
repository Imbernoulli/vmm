# Materialized Checkpoint Delta Audit

这个审计直接读取已物化的 safetensors checkpoint，而不是只相信 writer manifest。它检查输出 tensor 是否与 base 同构，并量化 candidate 相对 base 的实际参数改动。

- Status: `passed`
- Tensor count: `18867`
- Changed tensors: `10353`
- Changed numel fraction: `0.533`
- Total relative delta norm: `0.243`
- Max abs delta: `0.598`
- Router changed tensors: `0/48`

## Group Summary

| group | tensors | changed | changed numel frac | delta norm | relative delta norm | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `routed_expert_ffn` | 18432 | 10353 | 0.562 | 1005.252 | 0.252 | 0.598 |
| `attention` | 288 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `embedding_or_head` | 2 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `norm` | 97 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `router` | 48 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

## Layer Summary

| layer | tensors | changed | changed numel frac | delta norm | relative delta norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
|  | 3 | 0 | 0.000 | 0.000 | 0.000 |
| 0 | 393 | 279 | 0.704 | 124.812 | 0.258 |
| 1 | 393 | 288 | 0.727 | 130.047 | 0.232 |
| 2 | 393 | 252 | 0.636 | 134.909 | 0.236 |
| 3 | 393 | 246 | 0.621 | 137.093 | 0.237 |
| 4 | 393 | 240 | 0.606 | 146.338 | 0.254 |
| 5 | 393 | 201 | 0.507 | 121.786 | 0.211 |
| 6 | 393 | 204 | 0.515 | 140.466 | 0.244 |
| 7 | 393 | 201 | 0.507 | 132.206 | 0.228 |
| 8 | 393 | 222 | 0.560 | 153.219 | 0.266 |
| 9 | 393 | 204 | 0.515 | 142.466 | 0.249 |
| 10 | 393 | 222 | 0.560 | 149.789 | 0.263 |
| 11 | 393 | 204 | 0.515 | 146.783 | 0.258 |
| 12 | 393 | 222 | 0.560 | 159.219 | 0.281 |
| 13 | 393 | 216 | 0.545 | 144.830 | 0.257 |
| 14 | 393 | 198 | 0.500 | 140.366 | 0.249 |
| 15 | 393 | 225 | 0.568 | 148.916 | 0.262 |
| 16 | 393 | 195 | 0.492 | 144.164 | 0.253 |
| 17 | 393 | 201 | 0.507 | 159.764 | 0.282 |
| 18 | 393 | 210 | 0.530 | 142.688 | 0.249 |
| 19 | 393 | 183 | 0.462 | 150.477 | 0.260 |
| 20 | 393 | 213 | 0.538 | 160.398 | 0.275 |
| 21 | 393 | 201 | 0.507 | 150.086 | 0.257 |
| 22 | 393 | 222 | 0.560 | 176.591 | 0.303 |
| 23 | 393 | 219 | 0.553 | 159.733 | 0.278 |
| 24 | 393 | 213 | 0.538 | 157.669 | 0.272 |
| 25 | 393 | 228 | 0.576 | 154.899 | 0.266 |
| 26 | 393 | 207 | 0.523 | 161.997 | 0.275 |
| 27 | 393 | 198 | 0.500 | 143.497 | 0.243 |
| 28 | 393 | 207 | 0.523 | 142.085 | 0.242 |
| 29 | 393 | 207 | 0.523 | 146.894 | 0.249 |
| 30 | 393 | 204 | 0.515 | 153.310 | 0.259 |
| 31 | 393 | 180 | 0.454 | 147.410 | 0.247 |
| 32 | 393 | 219 | 0.553 | 151.613 | 0.252 |
| 33 | 393 | 213 | 0.538 | 139.269 | 0.232 |
| 34 | 393 | 210 | 0.530 | 148.460 | 0.248 |
| 35 | 393 | 231 | 0.583 | 141.109 | 0.235 |
| 36 | 393 | 225 | 0.568 | 156.906 | 0.260 |
| 37 | 393 | 213 | 0.538 | 146.308 | 0.241 |
| 38 | 393 | 195 | 0.492 | 118.070 | 0.193 |
| 39 | 393 | 207 | 0.523 | 134.043 | 0.218 |
| 40 | 393 | 204 | 0.515 | 129.711 | 0.209 |
| 41 | 393 | 219 | 0.553 | 133.699 | 0.213 |
| 42 | 393 | 240 | 0.606 | 128.683 | 0.203 |
| 43 | 393 | 228 | 0.576 | 145.962 | 0.229 |
| 44 | 393 | 204 | 0.515 | 139.513 | 0.218 |
| 45 | 393 | 183 | 0.462 | 135.427 | 0.210 |
| 46 | 393 | 216 | 0.545 | 155.560 | 0.240 |
| 47 | 393 | 234 | 0.591 | 133.581 | 0.206 |

## Top 20 Changed Tensors

| tensor | group | layer | numel | delta norm | relative delta norm | max abs delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `model.layers.46.mlp.experts.101.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 22.129 | 0.647 | 0.114 |
| `model.layers.46.mlp.experts.56.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 22.117 | 0.650 | 0.167 |
| `model.layers.46.mlp.experts.101.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 21.960 | 0.650 | 0.169 |
| `model.layers.44.mlp.experts.4.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 21.736 | 0.650 | 0.151 |
| `model.layers.46.mlp.experts.56.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 21.499 | 0.615 | 0.312 |
| `model.layers.46.mlp.experts.101.gate_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 21.057 | 0.638 | 0.107 |
| `model.layers.47.mlp.experts.65.up_proj.weight` | `routed_expert_ffn` | 47 | 1572864 | 20.866 | 0.624 | 0.084 |
| `model.layers.45.mlp.experts.61.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 20.661 | 0.634 | 0.104 |
| `model.layers.45.mlp.experts.40.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 20.653 | 0.644 | 0.080 |
| `model.layers.47.mlp.experts.65.gate_proj.weight` | `routed_expert_ffn` | 47 | 1572864 | 20.630 | 0.650 | 0.088 |
| `model.layers.44.mlp.experts.87.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 20.534 | 0.650 | 0.162 |
| `model.layers.46.mlp.experts.56.gate_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 20.523 | 0.632 | 0.158 |
| `model.layers.37.mlp.experts.115.up_proj.weight` | `routed_expert_ffn` | 37 | 1572864 | 20.518 | 0.650 | 0.104 |
| `model.layers.42.mlp.experts.87.up_proj.weight` | `routed_expert_ffn` | 42 | 1572864 | 20.477 | 0.647 | 0.088 |
| `model.layers.43.mlp.experts.95.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 20.391 | 0.632 | 0.120 |
| `model.layers.44.mlp.experts.87.down_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 20.278 | 0.644 | 0.268 |
| `model.layers.44.mlp.experts.4.down_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 20.273 | 0.603 | 0.188 |
| `model.layers.43.mlp.experts.22.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 20.222 | 0.644 | 0.145 |
| `model.layers.41.mlp.experts.94.up_proj.weight` | `routed_expert_ffn` | 41 | 1572864 | 20.215 | 0.648 | 0.083 |
| `model.layers.44.mlp.experts.106.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 20.210 | 0.593 | 0.132 |

## Files

- `results/qwen3_moe_tail_trimmed_delta_audit/tensor_delta_audit.csv`
- `results/qwen3_moe_tail_trimmed_delta_audit/group_delta_summary.csv`
- `results/qwen3_moe_tail_trimmed_delta_audit/layer_delta_summary.csv`
- `results/qwen3_moe_tail_trimmed_delta_audit/summary.json`
