# Qwen3 MoE HARC Router Candidate

这个 gate 把 HARC router solver 产生的同形状 `router_delta.safetensors` 写入一个可直接进入 vLLM 评测队列的同结构 checkpoint。它验证的不是“router 不动”，而是“router 只按 HARC delta 改动，非 router 权重保持不变”。

- Status: `harc_router_candidate_waiting_for_solver_delta`
- Solver status: `harc_solver_missing_cache`
- Base checkpoint: `results/checkpoints/qwen3_moe_unified_mechanism_candidate` (exists `True`)
- Router delta: `results/qwen3_moe_harc_router_solver/router_delta.safetensors` (exists `False`)
- Candidate checkpoint: `results/checkpoints/qwen3_moe_harc_router_candidate` (exists `False`)
- Delta tensors: `0`
- Materialization checks: `0/0` passed, max error `n/a`
- Recommended action: `collect_real_router_cache_then_rerun_harc_solver`

## Requirements

| requirement | passed | evidence |
| --- | ---: | --- |
| `solver_ready` | `False` | `harc_solver_missing_cache` |
| `router_delta_exists` | `False` | `results/qwen3_moe_harc_router_solver/router_delta.safetensors` |
| `base_checkpoint_exists` | `True` | `results/checkpoints/qwen3_moe_unified_mechanism_candidate` |
| `checkpoint_materialized` | `False` | `results/checkpoints/qwen3_moe_harc_router_candidate` |
| `router_delta_applied` | `False` | `0/0` |
| `non_router_unchanged` | `False` | `0/0` |

## Writer Command

```bash
python scripts/write_same_shape_average_checkpoint.py --base results/checkpoints/qwen3_moe_unified_mechanism_candidate --source same=results/checkpoints/qwen3_moe_unified_mechanism_candidate --source-weight same=0.0 --freeze-router --tensor-delta-safetensors results/qwen3_moe_harc_router_solver/router_delta.safetensors --output-dir results/checkpoints/qwen3_moe_harc_router_candidate
```
