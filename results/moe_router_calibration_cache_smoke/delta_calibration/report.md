# MoE Router Delta Calibration

这个脚本把离线 route probe 得到的 hidden states 和 teacher router logits 转成同构 checkpoint writer 可写入的 safetensors delta。它不是直接平均 router weight；目标函数是 route-KD/top-1 route imitation，并用 capacity、load-balance 和 delta trust term 约束移动幅度。

- Status: `passed`
- Routers: `2`
- Delta tensors: `2`
- Mean initial KL: `0.062390`
- Mean final KL: `0.020388`
- Mean initial top-1 agreement: `0.7031`
- Mean final top-1 agreement: `0.8802`

## Writer

```bash
python scripts/write_same_shape_average_checkpoint.py --base SMOKE_BASE_CHECKPOINT --source SOURCE_NAME=SOURCE_CHECKPOINT --source-weight SOURCE_NAME=0.0 --freeze-router --tensor-delta-safetensors results/moe_router_calibration_cache_smoke/delta_calibration/router_delta.safetensors --output-dir results/moe_router_calibration_cache_smoke/delta_calibration/checkpoint_with_calibrated_router
```

## Router Metrics

| tensor | initial KL | final KL | initial top1 | final top1 | final rel delta |
|---|---:|---:|---:|---:|---:|
| `blocks.0.router.weight` | 0.083914 | 0.029580 | 0.7188 | 0.8646 | 0.5000 |
| `blocks.1.router.weight` | 0.040866 | 0.011197 | 0.6875 | 0.8958 | 0.5000 |

## Files

- `results/moe_router_calibration_cache_smoke/delta_calibration/router_delta.safetensors`
- `results/moe_router_calibration_cache_smoke/delta_calibration/router_delta_summary.csv`
- `results/moe_router_calibration_cache_smoke/delta_calibration/training_trace.csv`
- `results/moe_router_calibration_cache_smoke/delta_calibration/summary.json`
