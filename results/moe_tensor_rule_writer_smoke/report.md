# MoE Tensor-Rule Writer Smoke

这个 smoke 构造 tiny MoE-like safetensors checkpoint，调用真实 same-shape writer 写出权重，再逐张量检查 tensor-rule、freeze-router 和非浮点复制是否生效。
其中 router weight 先被 freeze，再通过 safetensors full-tensor delta 写入校准增量；router bias 通过 `tensor,index,delta` CSV 做 additive correction，用来验证两类 router calibration 都可以写进同构 checkpoint。

- Status: `passed`
- Checked tensors: `7`
- Failed tensors: `0`
- Rule counts: `{"default": 1, "freeze_router": 2, "tensor_rule:.*experts\\.0\\..*": 1, "tensor_rule:.*experts\\.1\\..*": 1, "tensor_rule:.*self_attn.*": 1}`
- Additive deltas: `2` values across `1` tensors
- Safetensors tensor deltas: `4` values across `1` tensors

## Files

- `results/moe_tensor_rule_writer_smoke/tensor_checks.csv`
- `results/moe_tensor_rule_writer_smoke/merge_manifest.json`
- `results/moe_tensor_rule_writer_smoke/summary.json`
