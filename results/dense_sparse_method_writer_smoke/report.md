# Dense Sparse-Method Writer Smoke

这个 smoke 验证 same-shape writer 的 coordinate-wise sparse merge method。`mlp.up_proj` 使用 TIES-style trim/sign-elect/merge；`self_attn.q_proj` 保持普通线性平均。

- Status: `passed`
- Checked tensors: `3`
- Failed tensors: `0`
- Method counts: `{"linear": 1, "tensor_method:.*mlp\\.up_proj.*:ties": 1}`

## Files

- `results/dense_sparse_method_writer_smoke/tensor_checks.csv`
- `results/dense_sparse_method_writer_smoke/merge_manifest.json`
- `results/dense_sparse_method_writer_smoke/summary.json`
