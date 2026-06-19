# MoE Packed-Expert Writer Smoke

这个 smoke 验证 same-shape writer 能处理真实 Qwen MoE 这种 packed expert tensor：输出 tensor 名字和 shape 不变，但第 0 维的 expert slice 可以按 CSV 指定的 source expert slice 与权重写入。
测试里 `down_proj` 和 `gate_up_proj` 都是 `(num_experts, ...)` 形式；code/general 的 source expert index 被故意交叉，用逐张量数值检查验证 remap 和权重确实生效。

- Status: `passed`
- Checked tensors: `6`
- Failed tensors: `0`
- Packed rule tensors: `2`
- Packed rule slices: `3`
- Packed rule values: `5`

## Files

- `results/moe_packed_expert_writer_smoke/tensor_checks.csv`
- `results/moe_packed_expert_writer_smoke/merge_manifest.json`
- `results/moe_packed_expert_writer_smoke/summary.json`
