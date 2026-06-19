# MoE Combined Writer Smoke

这个 smoke 构造 tiny MoE-like checkpoint，并在一次真实 same-shape writer 调用里同时启用 expert tensor rules、source expert alias remap、freeze-router 和 router-bias additive delta。
数值检查用 swapped code experts 验证 alias 确实先于 expert rule 生效；如果 alias 没有生效，expert 0/1 的 expected mean 会错。

- Status: `passed`
- Checked tensors: `7`
- Failed tensors: `0`
- Tensor rules: `3`
- Alias rules: `2`
- Code aliased tensors: `2`
- Additive deltas: `2` values across `1` tensors

## Files

- `results/moe_combined_writer_smoke/tensor_checks.csv`
- `results/moe_combined_writer_smoke/merge_manifest.json`
- `results/moe_combined_writer_smoke/summary.json`
