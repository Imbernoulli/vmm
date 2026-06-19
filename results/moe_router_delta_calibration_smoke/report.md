# MoE Router Delta Calibration

这个脚本把离线 route probe 得到的 hidden states 和 teacher router logits 转成同构 checkpoint writer 可写入的 safetensors delta。它不是直接平均 router weight；目标函数是 route-KD/top-1 route imitation，并用 capacity、load-balance 和 delta trust term 约束移动幅度。

- Status: `passed`
- Routers: `1`
- Delta tensors: `1`
- Mean initial KL: `0.175224`
- Mean final KL: `0.051595`
- Mean initial top-1 agreement: `0.5260`
- Mean final top-1 agreement: `0.7708`
- Max final hard top-1 capacity overflow: `0.088542`
- Max final hard top-k capacity overflow: `0.000000`

## Writer

```bash
python scripts/write_same_shape_average_checkpoint.py --base SMOKE_BASE_CHECKPOINT --source SOURCE_NAME=SOURCE_CHECKPOINT --source-weight SOURCE_NAME=0.0 --freeze-router --tensor-delta-safetensors results/moe_router_delta_calibration_smoke/router_delta.safetensors --output-dir results/moe_router_delta_calibration_smoke/checkpoint_with_calibrated_router
```

## Router Metrics

| tensor | initial KL | final KL | initial top1 | final top1 | final rel delta | final top1 load | final top-k load | final top-k overflow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `model.layers.0.router.weight` | 0.175224 | 0.051595 | 0.5260 | 0.7708 | 0.5000 | 0.4010 | 0.2734 | 0.0000 |

## Files

- `results/moe_router_delta_calibration_smoke/router_delta.safetensors`
- `results/moe_router_delta_calibration_smoke/router_delta_summary.csv`
- `results/moe_router_delta_calibration_smoke/training_trace.csv`
- `results/moe_router_delta_calibration_smoke/summary.json`
