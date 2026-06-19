# Qwen3 MoE Expert-Only Trust-Region Candidate

这个候选是一个 attention ablation：保留 trust-region routed expert rules，但冻结 shared attention。
目的不是声称一定更好，而是在后续 vLLM eval 中隔离 Coder attention delta 是否真的有收益。

- Status: `attention_ablation_rules_ready`
- Expert rules: `5243`
- Attention rule: `freeze_shared_attention`
- Trust-region relative delta norm: `0.249`
- Estimated expert-only relative delta norm: `0.246`
- Estimated reduction: `0.003`
- Attention relative delta norm: `0.189`
- Attention delta energy fraction: `0.021`
- Writer dry-run: `True`
- Checkpoint materialized: `True`
- Materialized shards: `16`

## Attention Projection Summary

| kind | tensors | relative delta norm | mean tensor rel | p95 tensor rel | max tensor rel | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `k_norm` | 48 | 0.026 | 0.024 | 0.042 | 0.047 | 1.375 |
| `k_proj` | 48 | 0.196 | 0.192 | 0.234 | 0.294 | 0.141 |
| `o_proj` | 48 | 0.174 | 0.172 | 0.228 | 0.257 | 0.305 |
| `q_norm` | 48 | 0.028 | 0.024 | 0.040 | 0.075 | 1.688 |
| `q_proj` | 48 | 0.222 | 0.221 | 0.267 | 0.290 | 0.363 |
| `v_proj` | 48 | 0.138 | 0.137 | 0.186 | 0.295 | 0.072 |

## Highest Attention Layers

| layer | tensors | relative delta norm | max tensor rel |
| ---: | ---: | ---: | ---: |
| 0 | 6 | 0.251 | 0.295 |
| 13 | 6 | 0.246 | 0.281 |
| 12 | 6 | 0.232 | 0.268 |
| 17 | 6 | 0.231 | 0.260 |
| 11 | 6 | 0.228 | 0.254 |
| 16 | 6 | 0.222 | 0.264 |
| 5 | 6 | 0.218 | 0.249 |
| 9 | 6 | 0.218 | 0.262 |
| 18 | 6 | 0.217 | 0.263 |
| 15 | 6 | 0.217 | 0.255 |
| 10 | 6 | 0.215 | 0.240 |
| 14 | 6 | 0.214 | 0.258 |
| 4 | 6 | 0.212 | 0.246 |
| 2 | 6 | 0.211 | 0.243 |
| 8 | 6 | 0.211 | 0.247 |
| 19 | 6 | 0.207 | 0.249 |
| 1 | 6 | 0.206 | 0.238 |
| 21 | 6 | 0.202 | 0.245 |
| 7 | 6 | 0.201 | 0.240 |
| 25 | 6 | 0.200 | 0.234 |

## Files

- `results/qwen3_moe_expert_only_trust_region_candidate/attention_kind_summary.csv`
- `results/qwen3_moe_expert_only_trust_region_candidate/attention_layer_summary.csv`
- `results/qwen3_moe_expert_only_trust_region_candidate/tensor_rules.txt`
- `results/qwen3_moe_expert_only_trust_region_candidate/writer_command.txt`
- `results/qwen3_moe_expert_only_trust_region_candidate/dry_run_command.txt`
- `results/qwen3_moe_expert_only_trust_region_candidate/summary.json`
