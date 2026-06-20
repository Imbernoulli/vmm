# MoE Router Delta Calibration

这个脚本把离线 route probe 得到的 hidden states 和 teacher router logits 转成同构 checkpoint writer 可写入的 safetensors delta。它不是直接平均 router weight；目标函数是 route-KD/top-1 route imitation，并用 capacity、load-balance 和 delta trust term 约束移动幅度。

- Status: `passed`
- Routers: `2`
- Delta tensors: `2`
- Mean initial KL: `0.239217`
- Mean final KL: `0.215267`
- Mean initial top-1 agreement: `0.4481`
- Mean final top-1 agreement: `0.4870`
- Max final hard top-1 capacity overflow: `0.012175`
- Max final hard top-k capacity overflow: `0.000000`
- Router cap mode: `per_router_table`
- Router cap range: `0.02` / `0.08`
- Max router hard top-1 overflow increase: `0.000000`
- Max router hard top-k overflow increase: `0.000000`
- Selection policy: `capacity_aware`
- Selection split: `validation`
- Mean selected epoch: `3.50`
- Mean train/selection samples: `307.0` / `77.0`
- Mean train/validation groups: `0.0` / `0.0`
- Mean train/final validation KL: `0.201274` / `0.215267`
- Max validation KL gap: `0.016903`
- Max validation top-1 drop: `0.047252`

## Writer

```bash
python scripts/write_same_shape_average_checkpoint.py --base SMOKE_BASE_CHECKPOINT --source SOURCE_NAME=SOURCE_CHECKPOINT --source-weight SOURCE_NAME=0.0 --freeze-router --tensor-delta-safetensors results/moe_router_delta_calibration_smoke/router_delta.safetensors --output-dir results/moe_router_delta_calibration_smoke/checkpoint_with_calibrated_router
```

## Router Metrics

| tensor | selected epoch | train KL | validation KL | KL gap | train top1 | validation top1 | top1 drop | final rel delta | top1 overflow initial-final | top-k overflow initial-final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `model.layers.0.router.weight` | 4 | 0.165653 | 0.182556 | 0.016903 | 0.5537 | 0.5065 | 0.0473 | 0.0200 | 0.0122-0.0000 | 0.0000-0.0000 |
| `model.layers.1.router.weight` | 3 | 0.236894 | 0.247977 | 0.011083 | 0.4951 | 0.4675 | 0.0276 | 0.0800 | 0.0122-0.0122 | 0.0000-0.0000 |

## Files

- `results/moe_router_delta_calibration_smoke/router_delta.safetensors`
- `results/moe_router_delta_calibration_smoke/router_delta_summary.csv`
- `results/moe_router_delta_calibration_smoke/training_trace.csv`
- `results/moe_router_delta_calibration_smoke/router_cap_table.csv`
- `results/moe_router_delta_calibration_smoke/summary.json`
