# MoE Route-Weight Recipes

这个报告把 MoE routing/expert-load probe 或显式 expert 权重转成同构 checkpoint writer 可以读取的 tensor-rule 权重。目标不是增加 experts 或做 ensemble，而是在原 expert 数、原 router shape 下，给每个 expert 设置更合理的 source delta 系数。

- Recipe status: `route_weight_rules_ready`
- Recipe kind: `route_frequency`
- Sources: `instruct, coder`
- Router dirs: `results/moe_routing_probe/qwen3_30b_instruct_vs_coder`
- Expert weight CSVs: `none`
- Expert weight category filter: `none`
- Expert tensor rules: `5243`
- Packed expert slice rules: `0`
- Tensor rule file: `results/qwen3_moe_unified_route_guarded_candidate/tensor_rules.txt`
- Packed expert rule file: `results/qwen3_moe_unified_route_guarded_candidate/packed_expert_rules.csv`

## 权重规则

如果传入 `--expert-weight-csv`，脚本直接使用 `weight_<source>` 列；否则先按 prompt category 把 route mass 分给对应 source，然后做归一化：

```text
route_mass[source, layer, expert] = sum topk_fraction(category -> source)
writer_weight[source, layer, expert] = (1 - anchor_floor) * normalize(route_mass)
```

剩下的 `anchor_floor` 留给 base/anchor checkpoint，低使用率 expert 默认保持 anchor-heavy/frozen。这样做是为了避免 MoE average 直接把低证据、低路由频率的 experts 拉偏。

如果传入 `--model-source`，脚本只把某个 source 自己 checkpoint 的 route mass 分给该 source；例如 code 类只使用 coder checkpoint 的 code 路由，general/safety/legal 等只使用 instruct checkpoint 的对应路由。这样避免把两个模型路由冲突直接相加。

- Model-source filters: `[{"model_substring": "Qwen3-30B-A3B-Instruct-2507", "source": "instruct"}, {"model_substring": "Qwen3-Coder-30B-A3B-Instruct", "source": "coder"}]`

## 当前专家权重摘要

Action counts: `{"dominant_source_expert_delta": 2904, "mixed_source_expert_delta": 2088, "anchor_heavy_or_freeze": 251}`

| layer | expert | total top-k fraction | dominant source | dominant weight | action |
| --- | ---: | ---: | --- | ---: | --- |
| 0 | 0 | 0.1161 | instruct | 0.5887 | `mixed_source_expert_delta` |
| 0 | 1 | 0.1535 | instruct | 0.6293 | `mixed_source_expert_delta` |
| 0 | 2 | 0.1147 | instruct | 0.6531 | `mixed_source_expert_delta` |
| 0 | 3 | 0.2525 | instruct | 0.7001 | `dominant_source_expert_delta` |
| 0 | 4 | 0.02211 | instruct | 0.6652 | `mixed_source_expert_delta` |
| 0 | 6 | 0.01049 | instruct | 0.4604 | `mixed_source_expert_delta` |
| 0 | 7 | 0.05765 | instruct | 0.6636 | `mixed_source_expert_delta` |
| 0 | 8 | 0.005682 | instruct | 0.85 | `dominant_source_expert_delta` |
| 0 | 10 | 0.01328 | coder | 0.85 | `dominant_source_expert_delta` |
| 0 | 11 | 0.04228 | instruct | 0.7633 | `dominant_source_expert_delta` |
| 0 | 12 | 0.314 | instruct | 0.6792 | `mixed_source_expert_delta` |
| 0 | 13 | 0.03297 | instruct | 0.85 | `dominant_source_expert_delta` |
| 0 | 14 | 0.04166 | instruct | 0.85 | `dominant_source_expert_delta` |
| 0 | 15 | 0.00431 | instruct | 0 | `anchor_heavy_or_freeze` |
| 0 | 16 | 0.1909 | instruct | 0.6725 | `mixed_source_expert_delta` |
| 0 | 17 | 0.06204 | instruct | 0.6089 | `mixed_source_expert_delta` |
| 0 | 18 | 0.04854 | instruct | 0.5557 | `mixed_source_expert_delta` |
| 0 | 19 | 0.003788 | instruct | 0 | `anchor_heavy_or_freeze` |
| 0 | 20 | 0.03985 | coder | 0.85 | `dominant_source_expert_delta` |
| 0 | 21 | 0.1907 | instruct | 0.6566 | `mixed_source_expert_delta` |

## Prompt Category Source Map

| category | prompts | mapped source | mapping |
| --- | ---: | --- | --- |
| agentic_code | 1 | coder | explicit |
| code | 2 | coder | explicit |
| finance | 1 | instruct | explicit |
| general | 2 | instruct | explicit |
| legal | 1 | instruct | explicit |
| long_context | 1 | instruct | explicit |
| math | 2 | instruct | explicit |
| safety | 2 | instruct | explicit |

## Routing Probe Plan

| probe | model | compare model | prompts | output |
| --- | --- | --- | ---: | --- |
| qwen3_30b_general_vs_code | `Qwen/Qwen3-30B-A3B` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 12 | `results/moe_routing_probe/qwen3_30b_general_vs_code` |

## Writer Dry-Run Command

```bash
python scripts/write_same_shape_average_checkpoint.py --base /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --source instruct=/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --source coder=/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120 --source-weight instruct=0.0 --source-weight coder=0.0 --freeze-router --tensor-rule-file results/qwen3_moe_unified_route_guarded_candidate/tensor_rules.txt --output-dir results/checkpoints/qwen3_moe_unified_route_guarded_candidate --dry-run
```

## 需要先跑的 Routing Probe

```bash
python scripts/probe_moe_routing.py --model Qwen/Qwen3-30B-A3B --compare-model Qwen/Qwen3-Coder-30B-A3B-Instruct --prompts prompts/qwen_moe_route_probe_prompts.jsonl --device-map auto --dtype bfloat16 --max-length 768 --use-chat-template --output-dir results/moe_routing_probe/qwen3_30b_general_vs_code
```

然后重新生成 route weights：

```bash
PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code --source instruct --source coder --category-source agentic_code=coder --category-source code=coder --category-source finance=instruct --category-source general=instruct --category-source legal=instruct --category-source long_context=instruct --category-source math=instruct --category-source safety=instruct
```

## Files

- `results/qwen3_moe_unified_route_guarded_candidate/source_weights_by_expert.csv`
- `results/qwen3_moe_unified_route_guarded_candidate/tensor_rules.txt`
- `results/qwen3_moe_unified_route_guarded_candidate/packed_expert_rules.csv`
- `results/qwen3_moe_unified_route_guarded_candidate/writer_command.txt`
- `results/qwen3_moe_unified_route_guarded_candidate/routing_probe_plan.csv`
- `results/qwen3_moe_unified_route_guarded_candidate/category_source_plan.csv`
- `results/qwen3_moe_unified_route_guarded_candidate/summary.json`
