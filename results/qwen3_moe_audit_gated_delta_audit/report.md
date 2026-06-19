# Materialized Checkpoint Delta Audit

这个审计直接读取已物化的 safetensors checkpoint，而不是只相信 writer manifest。它检查输出 tensor 是否与 base 同构，并量化 candidate 相对 base 的实际参数改动。

- Status: `passed`
- Tensor count: `18867`
- Changed tensors: `10641`
- Changed numel fraction: `0.563`
- Total relative delta norm: `0.264`
- Max abs delta: `1.688`
- Router changed tensors: `0/48`

## Group Summary

| group | tensors | changed | changed numel frac | delta norm | relative delta norm | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `routed_expert_ffn` | 18432 | 10353 | 0.562 | 1080.594 | 0.270 | 0.723 |
| `attention` | 288 | 288 | 1.000 | 149.056 | 0.189 | 1.688 |
| `embedding_or_head` | 2 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `norm` | 97 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `router` | 48 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

## Layer Summary

| layer | tensors | changed | changed numel frac | delta norm | relative delta norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
|  | 3 | 0 | 0.000 | 0.000 | 0.000 |
| 0 | 393 | 285 | 0.735 | 134.129 | 0.277 |
| 1 | 393 | 294 | 0.757 | 135.095 | 0.241 |
| 2 | 393 | 258 | 0.666 | 147.517 | 0.258 |
| 3 | 393 | 252 | 0.651 | 149.006 | 0.257 |
| 4 | 393 | 246 | 0.636 | 161.796 | 0.281 |
| 5 | 393 | 207 | 0.538 | 127.393 | 0.221 |
| 6 | 393 | 210 | 0.545 | 152.051 | 0.264 |
| 7 | 393 | 207 | 0.538 | 140.761 | 0.243 |
| 8 | 393 | 228 | 0.591 | 166.572 | 0.289 |
| 9 | 393 | 210 | 0.545 | 155.990 | 0.273 |
| 10 | 393 | 228 | 0.591 | 161.064 | 0.283 |
| 11 | 393 | 210 | 0.545 | 156.532 | 0.275 |
| 12 | 393 | 228 | 0.591 | 173.326 | 0.306 |
| 13 | 393 | 222 | 0.576 | 159.496 | 0.283 |
| 14 | 393 | 204 | 0.530 | 150.342 | 0.266 |
| 15 | 393 | 231 | 0.598 | 159.975 | 0.282 |
| 16 | 393 | 201 | 0.523 | 156.409 | 0.275 |
| 17 | 393 | 207 | 0.538 | 173.918 | 0.307 |
| 18 | 393 | 216 | 0.560 | 156.607 | 0.273 |
| 19 | 393 | 189 | 0.492 | 167.091 | 0.289 |
| 20 | 393 | 219 | 0.568 | 176.272 | 0.302 |
| 21 | 393 | 207 | 0.538 | 165.422 | 0.283 |
| 22 | 393 | 228 | 0.591 | 190.623 | 0.327 |
| 23 | 393 | 225 | 0.583 | 173.918 | 0.302 |
| 24 | 393 | 219 | 0.568 | 172.075 | 0.297 |
| 25 | 393 | 234 | 0.606 | 171.461 | 0.295 |
| 26 | 393 | 213 | 0.553 | 175.186 | 0.298 |
| 27 | 393 | 204 | 0.530 | 153.069 | 0.259 |
| 28 | 393 | 213 | 0.553 | 152.232 | 0.259 |
| 29 | 393 | 213 | 0.553 | 160.955 | 0.273 |
| 30 | 393 | 210 | 0.545 | 168.126 | 0.284 |
| 31 | 393 | 186 | 0.485 | 160.577 | 0.269 |
| 32 | 393 | 225 | 0.583 | 164.383 | 0.273 |
| 33 | 393 | 219 | 0.568 | 152.254 | 0.254 |
| 34 | 393 | 216 | 0.560 | 164.569 | 0.275 |
| 35 | 393 | 237 | 0.613 | 151.576 | 0.253 |
| 36 | 393 | 231 | 0.598 | 167.809 | 0.278 |
| 37 | 393 | 219 | 0.568 | 160.065 | 0.264 |
| 38 | 393 | 201 | 0.523 | 125.475 | 0.206 |
| 39 | 393 | 213 | 0.553 | 143.037 | 0.232 |
| 40 | 393 | 210 | 0.545 | 142.227 | 0.229 |
| 41 | 393 | 225 | 0.583 | 146.366 | 0.233 |
| 42 | 393 | 246 | 0.636 | 137.956 | 0.217 |
| 43 | 393 | 234 | 0.606 | 158.805 | 0.249 |
| 44 | 393 | 210 | 0.545 | 149.424 | 0.233 |
| 45 | 393 | 189 | 0.492 | 151.042 | 0.234 |
| 46 | 393 | 222 | 0.576 | 168.878 | 0.260 |
| 47 | 393 | 240 | 0.621 | 141.108 | 0.218 |

## Top 20 Changed Tensors

| tensor | group | layer | numel | delta norm | relative delta norm | max abs delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `model.layers.46.mlp.experts.101.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 25.533 | 0.746 | 0.131 |
| `model.layers.46.mlp.experts.101.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 25.338 | 0.750 | 0.195 |
| `model.layers.45.mlp.experts.6.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 24.816 | 0.750 | 0.137 |
| `model.layers.46.mlp.experts.47.up_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 24.742 | 0.736 | 0.102 |
| `model.layers.41.mlp.experts.23.up_proj.weight` | `routed_expert_ffn` | 41 | 1572864 | 24.475 | 0.750 | 0.103 |
| `model.layers.47.mlp.experts.119.gate_proj.weight` | `routed_expert_ffn` | 47 | 1572864 | 24.384 | 0.750 | 0.246 |
| `model.layers.46.mlp.experts.101.gate_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 24.295 | 0.736 | 0.123 |
| `model.layers.45.mlp.experts.105.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 24.262 | 0.740 | 0.105 |
| `model.layers.41.mlp.experts.109.up_proj.weight` | `routed_expert_ffn` | 41 | 1572864 | 24.204 | 0.750 | 0.101 |
| `model.layers.44.mlp.experts.63.up_proj.weight` | `routed_expert_ffn` | 44 | 1572864 | 24.165 | 0.745 | 0.102 |
| `model.layers.47.mlp.experts.65.up_proj.weight` | `routed_expert_ffn` | 47 | 1572864 | 24.076 | 0.720 | 0.097 |
| `model.layers.42.mlp.experts.105.up_proj.weight` | `routed_expert_ffn` | 42 | 1572864 | 24.051 | 0.750 | 0.102 |
| `model.layers.45.mlp.experts.110.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 24.039 | 0.742 | 0.106 |
| `model.layers.45.mlp.experts.90.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 24.025 | 0.741 | 0.100 |
| `model.layers.33.mlp.experts.30.up_proj.weight` | `routed_expert_ffn` | 33 | 1572864 | 23.993 | 0.750 | 0.263 |
| `model.layers.43.mlp.experts.25.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 23.981 | 0.713 | 0.133 |
| `model.layers.46.mlp.experts.6.down_proj.weight` | `routed_expert_ffn` | 46 | 1572864 | 23.960 | 0.739 | 0.312 |
| `model.layers.45.mlp.experts.40.up_proj.weight` | `routed_expert_ffn` | 45 | 1572864 | 23.831 | 0.743 | 0.093 |
| `model.layers.43.mlp.experts.24.up_proj.weight` | `routed_expert_ffn` | 43 | 1572864 | 23.809 | 0.750 | 0.099 |
| `model.layers.47.mlp.experts.65.gate_proj.weight` | `routed_expert_ffn` | 47 | 1572864 | 23.804 | 0.750 | 0.101 |

## Files

- `results/qwen3_moe_audit_gated_delta_audit/tensor_delta_audit.csv`
- `results/qwen3_moe_audit_gated_delta_audit/group_delta_summary.csv`
- `results/qwen3_moe_audit_gated_delta_audit/layer_delta_summary.csv`
- `results/qwen3_moe_audit_gated_delta_audit/summary.json`
