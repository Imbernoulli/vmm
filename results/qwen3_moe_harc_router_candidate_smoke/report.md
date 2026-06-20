# Qwen3 MoE HARC Router Candidate

这个 gate 把 HARC router solver 产生的同形状 `router_delta.safetensors` 写入一个可直接进入 vLLM 评测队列的同结构 checkpoint。它验证的不是“router 不动”，而是“router 只按 HARC delta 改动，非 router 权重保持不变”。

- Status: `harc_router_candidate_materialized`
- Solver status: `smoke_ready`
- Base checkpoint: `results/qwen3_moe_harc_router_candidate_smoke/mock_inputs/base` (exists `True`)
- Router delta: `results/qwen3_moe_harc_router_candidate_smoke/mock_inputs/router_delta.safetensors` (exists `True`)
- Candidate checkpoint: `results/qwen3_moe_harc_router_candidate_smoke/checkpoint_with_harc_router` (exists `True`)
- Delta tensors: `2`
- Materialization checks: `3/3` passed, max error `0.000000`
- Recommended action: `smoke_passed_materialization_contract_ready`

## Requirements

| requirement | passed | evidence |
| --- | ---: | --- |
| `solver_ready` | `True` | `smoke_ready` |
| `router_delta_exists` | `True` | `results/qwen3_moe_harc_router_candidate_smoke/mock_inputs/router_delta.safetensors` |
| `base_checkpoint_exists` | `True` | `results/qwen3_moe_harc_router_candidate_smoke/mock_inputs/base` |
| `checkpoint_materialized` | `True` | `results/qwen3_moe_harc_router_candidate_smoke/checkpoint_with_harc_router` |
| `router_delta_applied` | `True` | `2/2` |
| `non_router_unchanged` | `True` | `1/1` |

## Writer Command

```bash
python scripts/write_same_shape_average_checkpoint.py --base results/qwen3_moe_harc_router_candidate_smoke/mock_inputs/base --source same=results/qwen3_moe_harc_router_candidate_smoke/mock_inputs/base --source-weight same=0.0 --freeze-router --tensor-delta-safetensors results/qwen3_moe_harc_router_candidate_smoke/mock_inputs/router_delta.safetensors --output-dir results/qwen3_moe_harc_router_candidate_smoke/checkpoint_with_harc_router
```
