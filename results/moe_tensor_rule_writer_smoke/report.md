# MoE Tensor-Rule Writer Smoke

这个 smoke 构造 tiny MoE-like safetensors checkpoint，调用真实 same-shape writer 写出权重，再逐张量检查 tensor-rule、freeze-router 和非浮点复制是否生效。

- Status: `passed`
- Checked tensors: `6`
- Failed tensors: `0`
- Rule counts: `{"default": 1, "freeze_router": 1, "tensor_rule:.*experts\\.0\\..*": 1, "tensor_rule:.*experts\\.1\\..*": 1, "tensor_rule:.*self_attn.*": 1}`

## Files

- `results/moe_tensor_rule_writer_smoke/tensor_checks.csv`
- `results/moe_tensor_rule_writer_smoke/merge_manifest.json`
- `results/moe_tensor_rule_writer_smoke/summary.json`
