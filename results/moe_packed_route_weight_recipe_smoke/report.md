# MoE Route-Weight Recipes

这个报告把 MoE routing/expert-load probe 或显式 expert 权重转成同构 checkpoint writer 可以读取的 tensor-rule 权重。目标不是增加 experts 或做 ensemble，而是在原 expert 数、原 router shape 下，给每个 expert 设置更合理的 source delta 系数。

- Recipe status: `explicit_expert_weight_rules_ready`
- Recipe kind: `explicit_expert_weights`
- Sources: `general, code`
- Router dirs: `none`
- Expert weight CSVs: `results/moe_packed_route_weight_recipe_smoke/expert_weights.csv`
- Expert weight category filter: `none`
- Expert tensor rules: `3`
- Packed expert slice rules: `12`
- Tensor rule file: `results/moe_packed_route_weight_recipe_smoke/tensor_rules.txt`
- Packed expert rule file: `results/moe_packed_route_weight_recipe_smoke/packed_expert_rules.csv`

## 权重规则

如果传入 `--expert-weight-csv`，脚本直接使用 `weight_<source>` 列；否则先按 prompt category 把 route mass 分给对应 source，然后做归一化：

```text
route_mass[source, layer, expert] = sum topk_fraction(category -> source)
writer_weight[source, layer, expert] = (1 - anchor_floor) * normalize(route_mass)
```

剩下的 `anchor_floor` 留给 base/anchor checkpoint，低使用率 expert 默认保持 anchor-heavy/frozen。这样做是为了避免 MoE average 直接把低证据、低路由频率的 experts 拉偏。

## 拓扑线索

- MoE model: `qwen3_6_35b_a3b` / `qwen3_5_moe`
- Experts: `256`；active per token: `8`；active fraction: `0.03125`
- Local weights available: `True`

## 当前专家权重摘要

Action counts: `{"synthetic_packed_expert_remap_weight": 3}`

| layer | expert | total top-k fraction | dominant source | dominant weight | action |
| --- | ---: | ---: | --- | ---: | --- |
| 0 | 0 | 0 | code | 0.75 | `synthetic_packed_expert_remap_weight` |
| 0 | 1 | 0 | general | 0.9 | `synthetic_packed_expert_remap_weight` |
| 1 | 0 | 0 | general | 0.5 | `synthetic_packed_expert_remap_weight` |

## Prompt Category Source Map

| category | prompts | mapped source | mapping |
| --- | ---: | --- | --- |
| agentic_code | 1 | code | category_heuristic |
| code | 2 | code | source_name_match |
| finance | 1 | general | fallback |
| general | 2 | general | source_name_match |
| legal | 1 | general | fallback |
| long_context | 1 | general | fallback |
| math | 2 | general | fallback |
| safety | 2 | general | fallback |

## Routing Probe Plan

| probe | model | compare model | prompts | output |
| --- | --- | --- | ---: | --- |
| qwen3_30b_general_vs_code | `Qwen/Qwen3-30B-A3B` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 12 | `results/moe_routing_probe/qwen3_30b_general_vs_code` |

## Writer Dry-Run Command

```bash
python scripts/write_same_shape_average_checkpoint.py --base MOE_BASE_OR_ANCHOR_PATH --source general=GENERAL_MODEL_PATH --source code=CODE_MODEL_PATH --source-weight general=0.0 --source-weight code=0.0 --freeze-router --tensor-rule-file results/moe_packed_route_weight_recipe_smoke/tensor_rules.txt --packed-expert-rule-csv results/moe_packed_route_weight_recipe_smoke/packed_expert_rules.csv --output-dir results/checkpoints/moe_packed_route_weight_candidate --dry-run
```

## 需要先跑的 Routing Probe

```bash
python scripts/probe_moe_routing.py --model Qwen/Qwen3-30B-A3B --compare-model Qwen/Qwen3-Coder-30B-A3B-Instruct --prompts prompts/qwen_moe_route_probe_prompts.jsonl --device-map auto --dtype bfloat16 --max-length 768 --use-chat-template --output-dir results/moe_routing_probe/qwen3_30b_general_vs_code
```

然后重新生成 route weights：

```bash
PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code --source general --source code --category-source agentic_code=code --category-source code=code --category-source finance=general --category-source general=general --category-source legal=general --category-source long_context=general --category-source math=general --category-source safety=general
```

## Files

- `results/moe_packed_route_weight_recipe_smoke/source_weights_by_expert.csv`
- `results/moe_packed_route_weight_recipe_smoke/tensor_rules.txt`
- `results/moe_packed_route_weight_recipe_smoke/packed_expert_rules.csv`
- `results/moe_packed_route_weight_recipe_smoke/writer_command.txt`
- `results/moe_packed_route_weight_recipe_smoke/routing_probe_plan.csv`
- `results/moe_packed_route_weight_recipe_smoke/category_source_plan.csv`
- `results/moe_packed_route_weight_recipe_smoke/summary.json`
