# Same-Shape Checkpoint Writer Smoke

这个 smoke 验证 `scripts/write_same_shape_average_checkpoint.py` 可以在不写出大模型权重的情况下，对 Qwen-compatible checkpoints 做同构检查。

## Run

```text
python scripts/write_same_shape_average_checkpoint.py \
  --base /srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-0.5B \
  --source instruct=/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --source coder=/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25 \
  --source-weight instruct=1.0 \
  --source-weight coder=0.0 \
  --freeze-router \
  --dry-run \
  --output-dir results/same_shape_writer_smoke
```

## 结果

- 只运行 dry-run：没有写出 weight shard。
- 检查 base tensors：`290`。
- `instruct` missing tensors：`0`；extra tensors：`0`；shape mismatches：`0`。
- `coder` missing tensors：`0`；extra tensors：`0`；shape mismatches：`0`。
- Rule counts：`default=290`。

这证明候选 Qwen2.5-0.5B base/instruct/coder checkpoints 对 writer 来说是同构兼容的。它不说明当前权重最优；权重应该来自 Average decision report 和下游 held-out validation。
