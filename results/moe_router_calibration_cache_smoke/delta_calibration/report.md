# MoE Router Delta Calibration

这个脚本把离线 route probe 得到的 hidden states 和 teacher router logits 转成同构 checkpoint writer 可写入的 safetensors delta。它不是直接平均 router weight；目标函数是 route-KD/top-1 route imitation，并用 capacity、load-balance 和 delta trust term 约束移动幅度。

- Status: `passed`
- Routers: `2`
- Delta tensors: `2`
- Mean initial KL: `0.047360`
- Mean final KL: `0.033399`
- Mean initial top-1 agreement: `0.7292`
- Mean final top-1 agreement: `0.8125`
- Max final hard top-1 capacity overflow: `0.125000`
- Max final hard top-k capacity overflow: `0.125000`
- Max router hard top-1 overflow increase: `0.000000`
- Max router hard top-k overflow increase: `0.000000`
- Selection policy: `capacity_aware`
- Selection split: `group_validation`
- Mean selected epoch: `3.50`
- Mean train/selection samples: `72.0` / `24.0`
- Mean train/validation groups: `3.0` / `1.0`
- Mean train/final validation KL: `0.052711` / `0.033399`
- Max validation KL gap: `-0.004130`
- Max validation top-1 drop: `0.013889`

## Writer

```bash
python scripts/write_same_shape_average_checkpoint.py --base SMOKE_BASE_CHECKPOINT --source SOURCE_NAME=SOURCE_CHECKPOINT --source-weight SOURCE_NAME=0.0 --freeze-router --tensor-delta-safetensors results/moe_router_calibration_cache_smoke/delta_calibration/router_delta.safetensors --output-dir results/moe_router_calibration_cache_smoke/delta_calibration/checkpoint_with_calibrated_router
```

## Router Metrics

| tensor | selected epoch | train KL | validation KL | KL gap | train top1 | validation top1 | top1 drop | final rel delta | top1 overflow initial-final | top-k overflow initial-final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `blocks.0.router.weight` | 0 | 0.092538 | 0.058045 | -0.034493 | 0.7222 | 0.7083 | 0.0139 | 0.0000 | 0.0208-0.0208 | 0.1250-0.1250 |
| `blocks.1.router.weight` | 7 | 0.012884 | 0.008754 | -0.004130 | 0.8889 | 0.9167 | -0.0278 | 0.5000 | 0.1667-0.1250 | 0.0000-0.0000 |

## Files

- `results/moe_router_calibration_cache_smoke/delta_calibration/router_delta.safetensors`
- `results/moe_router_calibration_cache_smoke/delta_calibration/router_delta_summary.csv`
- `results/moe_router_calibration_cache_smoke/delta_calibration/training_trace.csv`
- `results/moe_router_calibration_cache_smoke/delta_calibration/summary.json`
