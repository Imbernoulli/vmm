# MoE Router Calibration Cache

这个脚本把 student 模型的 router 输入 hidden states 和 teacher 模型的 router logits 对齐，写成 `train_moe_router_delta_calibration.py` 可直接读取的 cache。它补上了从真实 forward probe 到 router-delta 训练之间的接口。

- Status: `passed`
- Student: `SMOKE_STUDENT`
- Teacher: `SMOKE_TEACHER`
- Cache-ready routers: `2` / `2`
- Total cache rows: `192`
- Mean student->teacher KL: `0.062390`
- Delta calibration: `passed`
- Checkpoint materialization: `passed`

## Next Commands

```bash
python scripts/train_moe_router_delta_calibration.py --base STUDENT_BASE_CHECKPOINT --cache results/moe_router_calibration_cache_smoke/router_calibration_cache.pt --output-dir results/moe_router_calibration_cache_smoke/delta_calibration
python scripts/write_same_shape_average_checkpoint.py --base STUDENT_BASE_CHECKPOINT --source SOURCE_NAME=SOURCE_CHECKPOINT --source-weight SOURCE_NAME=0.0 --freeze-router --tensor-delta-safetensors results/moe_router_calibration_cache_smoke/delta_calibration/router_delta.safetensors --output-dir CHECKPOINT_WITH_CALIBRATED_ROUTER
```

## Router Rows

| tensor | rows | groups | hidden | experts | KL | top1 | top-k |
|---|---:|---:|---:|---:|---:|---:|---:|
| `blocks.0.router.weight` | 96 | 4 | 6 | 4 | 0.083914 | 0.7188 | 0.6875 |
| `blocks.1.router.weight` | 96 | 4 | 6 | 4 | 0.040866 | 0.6875 | 0.8194 |

## Files

- `results/moe_router_calibration_cache_smoke/router_calibration_cache.pt`
- `results/moe_router_calibration_cache_smoke/cache_summary.csv`
- `results/moe_router_calibration_cache_smoke/summary.json`
- `results/moe_router_calibration_cache_smoke/report.md`
- `results/moe_router_calibration_cache_smoke/materialization_checks.csv`
