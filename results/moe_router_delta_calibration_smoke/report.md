# MoE Router Delta Calibration

这个脚本把离线 route probe 得到的 hidden states 和 teacher router logits 转成同构 checkpoint writer 可写入的 safetensors delta。它不是直接平均 router weight；目标函数是 route-KD/top-1 route imitation，并用 capacity、load-balance 和 delta trust term 约束移动幅度。

- Status: `passed`
- Routers: `1`
- Delta tensors: `1`
- Mean initial KL: `0.189350`
- Mean final KL: `0.055732`
- Mean initial top-1 agreement: `0.4545`
- Mean final top-1 agreement: `0.7792`
- Max final hard top-1 capacity overflow: `0.063312`
- Max final hard top-k capacity overflow: `0.000000`
- Max router hard top-1 overflow increase: `0.051136`
- Max router hard top-k overflow increase: `0.000000`
- Selection policy: `capacity_aware`
- Selection split: `validation`
- Mean selected epoch: `2.00`
- Mean train/selection samples: `307.0` / `77.0`
- Mean train/final validation KL: `0.051894` / `0.055732`
- Max validation KL gap: `0.003838`
- Max validation top-1 drop: `-0.007234`

## Writer

```bash
python scripts/write_same_shape_average_checkpoint.py --base SMOKE_BASE_CHECKPOINT --source SOURCE_NAME=SOURCE_CHECKPOINT --source-weight SOURCE_NAME=0.0 --freeze-router --tensor-delta-safetensors results/moe_router_delta_calibration_smoke/router_delta.safetensors --output-dir results/moe_router_delta_calibration_smoke/checkpoint_with_calibrated_router
```

## Router Metrics

| tensor | selected epoch | train KL | validation KL | KL gap | train top1 | validation top1 | top1 drop | final rel delta | top1 overflow initial-final | top-k overflow initial-final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `model.layers.0.router.weight` | 2 | 0.051894 | 0.055732 | 0.003838 | 0.7720 | 0.7792 | -0.0072 | 0.5000 | 0.0122-0.0633 | 0.0000-0.0000 |

## Files

- `results/moe_router_delta_calibration_smoke/router_delta.safetensors`
- `results/moe_router_delta_calibration_smoke/router_delta_summary.csv`
- `results/moe_router_delta_calibration_smoke/training_trace.csv`
- `results/moe_router_delta_calibration_smoke/summary.json`
