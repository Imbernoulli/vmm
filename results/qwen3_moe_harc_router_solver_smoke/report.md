# Qwen3 MoE HARC Router Delta Solver

This script solves a training-free second-order router calibration problem from a router hidden/logit cache. It uses the softmax Hessian as the local KL curvature and a matrix-free conjugate-gradient solve, then writes same-shape router delta tensors.

## Result

- Status: `harc_solver_ready`
- Routers: `2`
- Delta tensors: `2`
- Mean route KL: `0.031806` -> `0.000006`
- Mean top-1 agreement: `0.662109` -> `0.998047`
- Max CG relative residual: `0.000031`
- Max relative delta norm: `0.463745`
- Max cap utilization: `0.927491`
- Recommended action: `materialize_harc_router_delta_then_run_matched_vllm_gate`

## Requirements

| requirement | passed | evidence | next action |
| --- | --- | --- | --- |
| `router_cache_exists` | `True` | cache=SMOKE_ROUTER_CACHE; exists=True | collect router calibration cache before solving HARC deltas |
| `base_checkpoint_exists` | `True` | base=SMOKE_BASE_CHECKPOINT; exists=True | point --base to the same-shape checkpoint whose routers will receive the delta |
| `router_tensors_loaded` | `True` | routers=2; load_error=None | verify cache tensor names match the base checkpoint safetensors index |
| `cg_converged` | `True` | max relative residual=0.000031 | increase --cg-max-iters or --ridge if residual is high |
| `kl_improved` | `True` | KL 0.031806 -> 0.000006 | check cache/source mismatch if HARC does not improve route KL |
| `relative_norm_cap_respected` | `True` | max cap utilization=0.927491 | lower --max-relative-norm or use a router cap table if needed |

## Router Metrics

| tensor | KL initial-final | top1 initial-final | rel delta | CG residual | cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model.layers.0.mlp.gate.weight` | 0.025798-0.000005 | 0.7500-1.0000 | 0.3633 | 0.000031 | 0.5000 |
| `model.layers.1.mlp.gate.weight` | 0.037815-0.000007 | 0.5742-0.9961 | 0.4637 | 0.000029 | 0.5000 |

## Writer

```bash
python scripts/write_same_shape_average_checkpoint.py --base SMOKE_BASE_CHECKPOINT --source SOURCE_NAME=SOURCE_CHECKPOINT --source-weight SOURCE_NAME=0.0 --freeze-router --tensor-delta-safetensors results/qwen3_moe_harc_router_solver_smoke/router_delta.safetensors --output-dir results/qwen3_moe_harc_router_solver_smoke/checkpoint_with_harc_router
```

## Files

- `results/qwen3_moe_harc_router_solver_smoke/router_delta.safetensors`
- `results/qwen3_moe_harc_router_solver_smoke/router_delta_summary.csv`
- `results/qwen3_moe_harc_router_solver_smoke/solver_trace.csv`
- `results/qwen3_moe_harc_router_solver_smoke/harc_solver_requirements.csv`
- `results/qwen3_moe_harc_router_solver_smoke/summary.json`
- `results/qwen3_moe_harc_router_solver_smoke/report.md`
