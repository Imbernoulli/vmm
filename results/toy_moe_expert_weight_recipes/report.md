# MoE Route-Weight Recipes

这个报告把 MoE routing/expert-load probe 或显式 expert 权重转成同构 checkpoint writer 可以读取的 tensor-rule 权重。目标不是增加 experts 或做 ensemble，而是在原 expert 数、原 router shape 下，给每个 expert 设置更合理的 source delta 系数。

- Recipe status: `explicit_expert_weight_rules_ready`
- Recipe kind: `explicit_expert_weights`
- Sources: `general, code`
- Router dirs: `none`
- Expert weight CSVs: `results/toy_moe_merge/expert_search_weights_by_expert.csv`
- Expert tensor rules: `4`
- Tensor rule file: `results/toy_moe_expert_weight_recipes/tensor_rules.txt`

## 权重规则

如果传入 `--expert-weight-csv`，脚本直接使用 `weight_<source>` 列；否则先按 prompt category 把 route mass 分给对应 source，然后做归一化：

```text
route_mass[source, layer, expert] = sum topk_fraction(category -> source)
writer_weight[source, layer, expert] = (1 - anchor_floor) * normalize(route_mass)
```

剩下的 `anchor_floor` 留给 base/anchor checkpoint，低使用率 expert 默认保持 anchor-heavy/frozen。这样做是为了避免 MoE average 直接把低证据、低路由频率的 experts 拉偏。

## 当前专家权重摘要

Action counts: `{"calibration_searched_expert_delta": 4}`

| layer | expert | total top-k fraction | dominant source | dominant weight | action |
| --- | ---: | ---: | --- | ---: | --- |
|  | 0 | 0 | general | 0.75 | `calibration_searched_expert_delta` |
|  | 1 | 0 | code | 0.75 | `calibration_searched_expert_delta` |
|  | 2 | 0 | code | 0.5564 | `calibration_searched_expert_delta` |
|  | 3 | 0 | general | 0.5 | `calibration_searched_expert_delta` |

## Writer Dry-Run Command

```bash
python scripts/write_same_shape_average_checkpoint.py --base MOE_BASE_OR_ANCHOR_PATH --source general=GENERAL_MODEL_PATH --source code=CODE_MODEL_PATH --source-weight general=0.0 --source-weight code=0.0 --freeze-router --tensor-rule-file results/toy_moe_expert_weight_recipes/tensor_rules.txt --output-dir results/checkpoints/toy_moe_expert_weight_candidate --dry-run
```

## 需要先跑的 Routing Probe

```bash
python scripts/probe_moe_routing.py --model Qwen/Qwen3-30B-A3B --compare-model Qwen/Qwen3-Coder-30B-A3B-Instruct --prompts prompts/qwen_moe_route_probe_prompts.jsonl --device-map auto --dtype bfloat16 --use-chat-template --output-dir results/moe_routing_probe/qwen3_30b_general_vs_code
```

然后重新生成 route weights：

```bash
PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code --source general --source code --category-source general=general --category-source code=code --category-source math=general --category-source safety=general
```

## Files

- `results/toy_moe_expert_weight_recipes/source_weights_by_expert.csv`
- `results/toy_moe_expert_weight_recipes/tensor_rules.txt`
- `results/toy_moe_expert_weight_recipes/writer_command.txt`
- `results/toy_moe_expert_weight_recipes/summary.json`
