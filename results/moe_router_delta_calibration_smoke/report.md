# MoE Router Delta Calibration

这个脚本把离线 route probe 得到的 hidden states 和 teacher router logits 转成同构 checkpoint writer 可写入的 safetensors delta。它不是直接平均 router weight；目标函数是 route-KD/top-1 route imitation，并用 capacity、load-balance 和 delta trust term 约束移动幅度。

- Status: `passed`
- Routers: `1`
- Delta tensors: `1`
- Mean initial KL: `0.175224`
- Mean final KL: `0.050264`
- Mean initial top-1 agreement: `0.5260`
- Mean final top-1 agreement: `0.7760`
- Max final hard top-1 capacity overflow: `0.088542`
- Max final hard top-k capacity overflow: `0.000000`
- Max router hard top-1 overflow increase: `0.033854`
- Max router hard top-k overflow increase: `0.000000`
- Selection policy: `capacity_aware`
- Mean selected epoch: `7.00`

## Writer

```bash
python scripts/write_same_shape_average_checkpoint.py --base SMOKE_BASE_CHECKPOINT --source SOURCE_NAME=SOURCE_CHECKPOINT --source-weight SOURCE_NAME=0.0 --freeze-router --tensor-delta-safetensors results/moe_router_delta_calibration_smoke/router_delta.safetensors --output-dir results/moe_router_delta_calibration_smoke/checkpoint_with_calibrated_router
```

## Router Metrics

| tensor | selected epoch | initial KL | final KL | initial top1 | final top1 | final rel delta | top1 overflow initial-final | top-k overflow initial-final | top1 load initial-final | top-k load initial-final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `model.layers.0.router.weight` | 7 | 0.175224 | 0.050264 | 0.5260 | 0.7760 | 0.5000 | 0.0547-0.0885 | 0.0000-0.0000 | 0.3672-0.4010 | 0.2747-0.2695 |

## Files

- `results/moe_router_delta_calibration_smoke/router_delta.safetensors`
- `results/moe_router_delta_calibration_smoke/router_delta_summary.csv`
- `results/moe_router_delta_calibration_smoke/training_trace.csv`
- `results/moe_router_delta_calibration_smoke/summary.json`
