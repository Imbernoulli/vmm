# Qwen3 MoE HARC Router Delta Solver

This script solves a training-free second-order router calibration problem from a router hidden/logit cache. It uses the softmax Hessian as the local KL curvature and a matrix-free conjugate-gradient solve, then writes same-shape router delta tensors.

## Result

- Status: `harc_solver_missing_cache`
- Routers: `0`
- Delta tensors: `0`
- Mean route KL: `n/a` -> `n/a`
- Mean top-1 agreement: `n/a` -> `n/a`
- Max CG relative residual: `n/a`
- Max relative delta norm: `n/a`
- Max cap utilization: `n/a`
- Recommended action: `collect_cache_or_fix_base_before_harc_solver`

## Requirements

| requirement | passed | evidence | next action |
| --- | --- | --- | --- |
| `router_cache_exists` | `False` | cache=results/qwen3_moe_router_calibration_cache/router_calibration_cache.pt; exists=False | collect router calibration cache before solving HARC deltas |
| `base_checkpoint_exists` | `True` | base=results/checkpoints/qwen3_moe_unified_mechanism_candidate; exists=True | point --base to the same-shape checkpoint whose routers will receive the delta |
| `router_tensors_loaded` | `False` | routers=0; load_error=None | verify cache tensor names match the base checkpoint safetensors index |
| `cg_converged` | `False` | max relative residual=n/a | increase --cg-max-iters or --ridge if residual is high |
| `kl_improved` | `False` | KL n/a -> n/a | check cache/source mismatch if HARC does not improve route KL |
| `relative_norm_cap_respected` | `False` | max cap utilization=n/a | lower --max-relative-norm or use a router cap table if needed |

## Router Metrics

| tensor | KL initial-final | top1 initial-final | rel delta | CG residual | cap |
| --- | ---: | ---: | ---: | ---: | ---: |

## Writer

```bash
HARC solver has not written router deltas yet.
```

## Files

- `results/qwen3_moe_harc_router_solver/router_delta.safetensors`
- `results/qwen3_moe_harc_router_solver/router_delta_summary.csv`
- `results/qwen3_moe_harc_router_solver/solver_trace.csv`
- `results/qwen3_moe_harc_router_solver/harc_solver_requirements.csv`
- `results/qwen3_moe_harc_router_solver/summary.json`
- `results/qwen3_moe_harc_router_solver/report.md`
