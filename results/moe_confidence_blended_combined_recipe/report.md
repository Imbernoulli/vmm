# MoE Combined Materialization Recipe

这个 recipe 把 confidence-blended expert tensor rules、expert-output alias remap 和 router-bias capacity delta 合成同一个 same-shape writer command。它不扩展 experts，不改 router shape，也不改输出 tensor names；所有动作都发生在已有 tensor 上。

- Status: `combined_writer_command_ready`
- Tensor rules: `5`
- Alias rules: `4`
- Router-bias delta rows: `4`
- Freeze router: `True`
- Dry run command: `True`

## 组合逻辑

- 默认 source delta weight 设为 0，避免未覆盖 tensor 被无意平均。
- `tensor_rules.txt` 决定共享 attention 和每个 expert FFN 的 source 权重。
- `source_tensor_aliases.txt` 只改变从某个 source 读取哪个 expert tensor，不改变输出 checkpoint 的 expert index。
- `router_bias_deltas.csv` 在 frozen/base router bias 上叠加 capacity correction。

## Writer Command

```bash
python scripts/write_same_shape_average_checkpoint.py --base MOE_BASE_OR_ANCHOR_PATH --source general=GENERAL_MODEL_PATH --source-weight general=0.0 --source code=CODE_MODEL_PATH --source-weight code=0.0 --freeze-router --tensor-rule-file results/toy_moe_confidence_blended_recipes/tensor_rules.txt --source-tensor-alias-file results/toy_moe_expert_remap_plan/source_tensor_aliases.txt --tensor-add-csv results/moe_confidence_blended_router_bias_plan/router_bias_deltas.csv --output-dir results/checkpoints/toy_moe_confidence_blended_combined_candidate --dry-run
```

## Files

- `results/toy_moe_confidence_blended_recipes/tensor_rules.txt`
- `results/toy_moe_expert_remap_plan/source_tensor_aliases.txt`
- `results/moe_confidence_blended_router_bias_plan/router_bias_deltas.csv`
- `results/moe_confidence_blended_combined_recipe/writer_command.txt`
- `results/moe_confidence_blended_combined_recipe/summary.json`
