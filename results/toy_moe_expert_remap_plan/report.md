# MoE Expert Remap Plan

这个报告把 expert-output matching 结果转成 same-shape checkpoint writer 可以读取的 source tensor alias 规则。它解决的是 MoE average 中最实际的一步：输出 checkpoint 的 expert index 不变，但从 source checkpoint 读取已经匹配过的 expert tensor。

- Source with remap: `code`
- Recommended upstream method: `expert_matched_average`
- Remap status: `ready`
- Alias rules: `4`
- Min output cosine: `0.9426`

## Expert Mapping

| output expert | matched source expert | output cosine | status |
| ---: | ---: | ---: | --- |
| 0 | 1 | 0.9957 | `use_alias` |
| 1 | 3 | 0.9426 | `use_alias` |
| 2 | 0 | 0.9724 | `use_alias` |
| 3 | 2 | 0.9985 | `use_alias` |

## Writer Dry-Run Command

```bash
python scripts/write_same_shape_average_checkpoint.py --base MOE_BASE_OR_ANCHOR_PATH --source general=GENERAL_MODEL_PATH --source code=CODE_MODEL_PATH --source-weight general=0.5 --source-weight code=0.5 --freeze-router --source-tensor-alias-file results/toy_moe_expert_remap_plan/source_tensor_aliases.txt --output-dir results/checkpoints/moe_expert_matched_candidate --dry-run
```

## Interpretation

- `source_tensor_aliases.txt` 不改变输出 tensor names/shapes，只改变某个 source 在读取 tensor 时用哪个 expert index。
- 这和 `expert_matched_average` 的语义一致：先对齐 expert 功能，再做同构平均。
- 如果某个 match 的 output cosine 低于阈值，报告会标成 `needs_manual_review`，不自动写 alias rule。

## Files

- `results/toy_moe_expert_remap_plan/expert_remap.csv`
- `results/toy_moe_expert_remap_plan/source_tensor_aliases.txt`
- `results/toy_moe_expert_remap_plan/writer_command.txt`
- `results/toy_moe_expert_remap_plan/summary.json`
