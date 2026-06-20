# Qwen3 MoE HARC Router Stats

This collector converts router hidden states and teacher logits into the statistics needed by the HARC-style router objective: softmax Hessian summaries, hidden covariance summaries, top-k boundary margin, and a per-router precision proxy.

## Result

- Status: `harc_router_stats_missing_cache`
- Cache: `results/qwen3_moe_router_calibration_cache/router_calibration_cache.pt`
- Routers: `0/0` valid
- First-stage coverage: `0/15` (`missing`)
- Mean Hessian trace: `n/a`
- Mean hidden covariance trace: `n/a`
- Top router by precision proxy: `None`

## Requirements

| requirement | passed | evidence | next action |
| --- | --- | --- | --- |
| `router_calibration_cache_exists` | `False` | cache=results/qwen3_moe_router_calibration_cache/router_calibration_cache.pt; exists=False | run router calibration cache collection on the GPU host |
| `router_records_present` | `False` | routers=0 | verify router hooks matched model.layers.*.mlp.gate/router modules |
| `hidden_logits_shape_valid` | `False` | valid routers=0; total rows=0 | collect 2D hidden states and matching teacher logits for each router |
| `softmax_hessian_positive` | `False` | mean Hessian trace=n/a | check teacher logits are not all degenerate one-hot or empty |
| `hidden_covariance_positive` | `False` | mean covariance trace=n/a | increase prompt/token coverage if hidden covariance is zero |
| `first_stage_priority_covered` | `False` | coverage=missing; 0/15 layers | collect all high-priority HARC layers before solving the matrix-free system |

## Top Router Stats

| layer | tensor | role | rows | experts | Hessian trace | cov trace | boundary proxy | precision proxy |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

## Solver Input

- Objective: `min_Wm sum_i E_x KL(softmax(W_i x) || softmax(W_m x))`
- Matrix-free matvec: `sum_i E_x [x x^T dW^T H_i] without materializing kron(H_i, xx^T)`
- Ready router tensors: `0`

## Outputs

- `results/qwen3_moe_harc_router_stats/router_harc_stats.csv`
- `results/qwen3_moe_harc_router_stats/harc_stats_requirements.csv`
- `results/qwen3_moe_harc_router_stats/harc_solver_inputs.json`
- `results/qwen3_moe_harc_router_stats/summary.json`
- `results/qwen3_moe_harc_router_stats/report.md`
