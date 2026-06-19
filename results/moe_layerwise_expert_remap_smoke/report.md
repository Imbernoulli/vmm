# MoE Layer-Wise Expert Remap Smoke

这个 smoke 验证 `build_moe_expert_remap_plan.py` 在 `expert_match.csv` 带 `layer_id` 时会生成 layer-scoped source tensor alias，而不是把某个 expert id 映射错误地应用到所有层。

- Status: `passed`
- Input rows: `4`
- Alias rules: `3`
- Layer-aware rules: `3`
- Manual review rows: `1`

## Files

- `results/moe_layerwise_expert_remap_smoke/checks.csv`
- `results/moe_layerwise_expert_remap_smoke/expert_remap.csv`
- `results/moe_layerwise_expert_remap_smoke/source_tensor_aliases.txt`
- `results/moe_layerwise_expert_remap_smoke/summary.json`
