# Qwen3 MoE Expert Subspace Conflict Probe

这个 probe 不再只问 expert index 是否对齐，而是读取真实 routed expert projection 的 channel/chunk 几何，检查同名 expert 内部是否存在局部子空间冲突。它的作用是给 MoE average 增加一个更细的 gate：identity mapping 通过以后，仍要判断哪些高路由质量 expert 需要更低非 base 权重、layer/chunk 系数或额外 output-space probe。

- Status: `ready_for_subspace_ablation`
- Projection tensors: `18432`
- Experts: `6144`
- Layers: `48`
- High subspace-conflict experts: `1323`
- Route-important high subspace-conflict experts: `242`
- Experts needing extra subspace scale beyond current unified rule: `17`
- Mean coder weight reduction if this ablation is materialized: `0.000041`
- Total coder weight reduction if this ablation is materialized: `0.253078`
- Top layer by route-weighted subspace conflict: `L17`
- Next action: `materialize_subspace_scaled_ablation_after_source_eval_budget`

## Layer Risk

| layer | high experts | route-high experts | extra-scaled experts | weighted score | mean scale | coder weight reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 80 | 22 | 0 | 0.7472 | 1.0000 | 0.000000 |
| 16 | 63 | 15 | 1 | 0.6984 | 0.9994 | 0.039260 |
| 13 | 79 | 15 | 0 | 0.6945 | 1.0000 | 0.000000 |
| 15 | 67 | 10 | 1 | 0.6896 | 0.9994 | 0.040275 |
| 14 | 76 | 16 | 0 | 0.6857 | 1.0000 | 0.000000 |
| 12 | 78 | 10 | 0 | 0.6832 | 1.0000 | 0.000000 |
| 23 | 56 | 16 | 0 | 0.6745 | 1.0000 | 0.000000 |
| 11 | 68 | 12 | 1 | 0.6661 | 0.9990 | 0.059435 |
| 29 | 49 | 10 | 0 | 0.6446 | 1.0000 | 0.000000 |
| 22 | 58 | 9 | 0 | 0.6433 | 1.0000 | 0.000000 |
| 10 | 64 | 9 | 2 | 0.6278 | 1.0000 | 0.001967 |
| 25 | 47 | 5 | 1 | 0.6214 | 0.9995 | 0.027483 |

## Actions

| action | experts | route mass | mean conflict | mean scale | coder weight reduction |
|---|---:|---:|---:|---:|---:|
| `route_important_subspace_conflict_lower_cap_or_chunk` | 7 | 1.2569 | 0.8360 | 0.9288 | 0.234963 |
| `lower_nonbase_weight_for_subspace_conflict` | 1081 | 42.5809 | 0.8163 | 1.0000 | 0.018115 |
| `identity_average_subspace_ok` | 3242 | 324.6910 | 0.3529 | 1.0000 | 0.000000 |
| `monitor_subspace_conflict_in_vllm_gate` | 1579 | 147.9740 | 0.6289 | 1.0000 | 0.000000 |
| `current_unified_cap_covers_route_subspace_conflict` | 235 | 59.4972 | 0.8042 | 1.0000 | 0.000000 |

## Top Expert Conflicts

| layer | expert | projection | driver | conflict | route mass | expected delta | cap | scale | action |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 21 | 69 | `up_proj` | `chunk_delta_spike` | 0.9367 | 0.4065 | 0.1246 | 0.5000 | 1.0000 | `current_unified_cap_covers_route_subspace_conflict` |
| 13 | 104 | `gate_proj` | `chunk_delta_spike` | 0.9053 | 0.6229 | 0.1173 | 0.5000 | 1.0000 | `current_unified_cap_covers_route_subspace_conflict` |
| 17 | 83 | `up_proj` | `chunk_delta_spike` | 0.8859 | 0.7944 | 0.2387 | 0.5500 | 1.0000 | `current_unified_cap_covers_route_subspace_conflict` |
| 10 | 98 | `down_proj` | `low_channel_cosine` | 0.8782 | 0.3571 | 0.2295 | 0.5500 | 1.0000 | `current_unified_cap_covers_route_subspace_conflict` |
| 17 | 78 | `up_proj` | `chunk_delta_spike` | 0.8745 | 0.3401 | 0.1564 | 0.5500 | 1.0000 | `current_unified_cap_covers_route_subspace_conflict` |
| 15 | 70 | `up_proj` | `chunk_delta_spike` | 0.8745 | 0.3642 | 0.2224 | 0.5500 | 1.0000 | `current_unified_cap_covers_route_subspace_conflict` |
| 12 | 119 | `up_proj` | `chunk_delta_spike` | 0.8729 | 0.4105 | 0.0000 | 0.5500 | 1.0000 | `current_unified_cap_covers_route_subspace_conflict` |
| 11 | 21 | `up_proj` | `chunk_delta_spike` | 0.8705 | 0.9517 | 0.3216 | 0.5500 | 1.0000 | `current_unified_cap_covers_route_subspace_conflict` |
| 17 | 21 | `gate_proj` | `chunk_delta_spike` | 0.8899 | 0.2950 | 0.2809 | 0.5500 | 1.0000 | `current_unified_cap_covers_route_subspace_conflict` |
| 16 | 73 | `up_proj` | `chunk_delta_spike` | 0.8662 | 0.4380 | 0.3693 | 0.5500 | 1.0000 | `current_unified_cap_covers_route_subspace_conflict` |
| 23 | 125 | `gate_proj` | `low_channel_cosine` | 0.9071 | 0.2771 | 0.1697 | 0.5000 | 1.0000 | `current_unified_cap_covers_route_subspace_conflict` |
| 17 | 55 | `down_proj` | `chunk_delta_spike` | 0.8877 | 0.2888 | 0.1693 | 0.5500 | 1.0000 | `current_unified_cap_covers_route_subspace_conflict` |

## Extra Scale Targets

| layer | expert | projection | conflict | route mass | expected delta | cap | scale | coder before-after | action |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 11 | 23 | `down_proj` | 0.9004 | 0.1410 | 0.5768 | 0.5000 | 0.8668 | 0.4463-0.3868 | `route_important_subspace_conflict_lower_cap_or_chunk` |
| 27 | 31 | `gate_proj` | 0.7343 | 0.1633 | 0.6003 | 0.5500 | 0.9162 | 0.4998-0.4579 | `route_important_subspace_conflict_lower_cap_or_chunk` |
| 15 | 31 | `up_proj` | 0.7571 | 0.2562 | 0.5992 | 0.5500 | 0.9179 | 0.4908-0.4505 | `route_important_subspace_conflict_lower_cap_or_chunk` |
| 16 | 64 | `up_proj` | 0.7867 | 0.2048 | 0.5987 | 0.5500 | 0.9186 | 0.4824-0.4432 | `route_important_subspace_conflict_lower_cap_or_chunk` |
| 25 | 31 | `up_proj` | 0.8122 | 0.1376 | 0.5850 | 0.5500 | 0.9402 | 0.4593-0.4318 | `route_important_subspace_conflict_lower_cap_or_chunk` |
| 5 | 9 | `gate_proj` | 0.8997 | 0.1635 | 0.5827 | 0.5500 | 0.9438 | 0.4592-0.4334 | `route_important_subspace_conflict_lower_cap_or_chunk` |
| 8 | 118 | `up_proj` | 0.8081 | 0.0266 | 0.6060 | 0.6000 | 0.9900 | 0.4730-0.4683 | `lower_nonbase_weight_for_subspace_conflict` |
| 8 | 101 | `up_proj` | 0.8319 | 0.0218 | 0.6030 | 0.6000 | 0.9951 | 0.4796-0.4772 | `lower_nonbase_weight_for_subspace_conflict` |
| 6 | 112 | `down_proj` | 0.9198 | 0.0399 | 0.6028 | 0.6000 | 0.9953 | 0.4822-0.4799 | `lower_nonbase_weight_for_subspace_conflict` |
| 33 | 67 | `up_proj` | 0.7797 | 0.0133 | 0.6023 | 0.6000 | 0.9962 | 0.4772-0.4754 | `lower_nonbase_weight_for_subspace_conflict` |
| 5 | 114 | `up_proj` | 0.7320 | 0.0133 | 0.6023 | 0.6000 | 0.9963 | 0.4865-0.4847 | `lower_nonbase_weight_for_subspace_conflict` |
| 33 | 11 | `up_proj` | 0.7786 | 0.0290 | 0.6021 | 0.6000 | 0.9965 | 0.4793-0.4776 | `lower_nonbase_weight_for_subspace_conflict` |
| 32 | 9 | `up_proj` | 0.9616 | 0.1904 | 0.5011 | 0.5000 | 0.9978 | 0.3744-0.3736 | `route_important_subspace_conflict_lower_cap_or_chunk` |
| 6 | 111 | `up_proj` | 0.7724 | 0.0259 | 0.6012 | 0.6000 | 0.9981 | 0.4778-0.4768 | `lower_nonbase_weight_for_subspace_conflict` |
| 10 | 121 | `up_proj` | 0.8281 | 0.0753 | 0.6011 | 0.6000 | 0.9981 | 0.4695-0.4686 | `lower_nonbase_weight_for_subspace_conflict` |
| 10 | 6 | `down_proj` | 0.8550 | 0.0218 | 0.6011 | 0.6000 | 0.9981 | 0.5327-0.5317 | `lower_nonbase_weight_for_subspace_conflict` |
| 32 | 123 | `up_proj` | 0.7871 | 0.0133 | 0.6007 | 0.6000 | 0.9988 | 0.4732-0.4727 | `lower_nonbase_weight_for_subspace_conflict` |

## Candidate Recipe

```bash
python scripts/write_same_shape_average_checkpoint.py --base /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --source instruct=/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --source coder=/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120 --source-weight instruct=0.0 --source-weight coder=0.0 --freeze-router --tensor-rule-file results/qwen3_moe_expert_subspace_conflict_probe/subspace_adjusted_tensor_rules.txt --output-dir results/checkpoints/qwen3_moe_subspace_scaled_candidate
```

## Files

- `results/qwen3_moe_expert_subspace_conflict_probe/projection_subspace_scores.csv`
- `results/qwen3_moe_expert_subspace_conflict_probe/expert_subspace_conflicts.csv`
- `results/qwen3_moe_expert_subspace_conflict_probe/layer_subspace_conflicts.csv`
- `results/qwen3_moe_expert_subspace_conflict_probe/subspace_action_summary.csv`
- `results/qwen3_moe_expert_subspace_conflict_probe/subspace_adjusted_group_rules.csv`
- `results/qwen3_moe_expert_subspace_conflict_probe/subspace_adjusted_tensor_rules.txt`
- `results/qwen3_moe_expert_subspace_conflict_probe/summary.json`
