# Qwen3 MoE HARC Readiness Gate

This gate turns the MoE router-breakdown mechanism into an executable decision: direct router averaging stays rejected, and HARC-style router distribution matching is allowed only as a calibrated repair path with matched downstream acceptance.

## Result

- Status: `harc_ready_for_matrix_free_solver`
- Preconditions: `6/6`
- HARC cache: `available`
- HARC stats: `harc_router_stats_ready` (`2/2` routers, first-stage `2/2`)
- Top priority layer: `L1` score `0.8560`
- Recommended action: `run_matrix_free_harc_solver_then_matched_vllm_gate`

## Requirements

| requirement | role | passed | evidence | next action |
| --- | --- | --- | --- | --- |
| `direct_router_average_rejected` | `harc_precondition` | `True` | router action=freeze_router; allowed layers=0/4; top-k jaccard mean/min=0.4200/0.2000 | freeze direct router average and use calibrated router objective |
| `topk_boundary_fragility_detected` | `harc_precondition` | `True` | high fragility layers=2/4; min safe lambda=0.0180; top layer=L1 score=0.8200 | prioritize fragile router layers for HARC statistics |
| `router_only_repair_signal_positive` | `harc_precondition` | `True` | worst NLL reduction=0.1200; avg NLL reduction=0.0800; acceptance=mechanism_supported_but_do_not_accept_without_matched_vllm_eval | keep router calibration as repair, not acceptance, until matched vLLM passes |
| `router_expert_coupling_active` | `harc_precondition` | `True` | gate=router_expert_coupling_active; fragility->feature corr=0.7000; shrink corr=0.5000 | calibrate router together with expert-cap law, not as an isolated tensor average |
| `safe_default_router_calibration_frontier_exists` | `harc_precondition` | `True` | default candidates=2/4; recommended=['cap001', 'margin_profile']; blocker=baseline_eval,source_eval,candidate_eval,audit | compare HARC-style calibration against route-KD cap001 and margin_profile |
| `calibration_job_preflight_ready` | `harc_precondition` | `True` | job status=job_ready_awaiting_gpu; prompts=True; source controls=True; student=True; teacher=True | run preflight on GPU host, then collect router logits/hidden states |
| `hessian_covariance_cache_available` | `harc_solver_requirement` | `True` | stats=harc_router_stats_ready; dir=results/qwen3_moe_harc_readiness_gate_smoke/mock_inputs/harc_solver_ready/harc_stats; routers=2/2; first-stage=2/2 (complete) | collect H_i=diag(r)-rr^T and hidden covariance per router layer with collect_qwen3_moe_harc_router_stats.py |

## Layer Priority

| layer | score | role | fragility | safe lambda | coupled risk | route mass |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 0.8560 | `critical_topk_boundary_layer` | 0.8200 | 0.0180 | 6.0000 | 12.00 |
| 2 | 0.2900 | `collect_hessian_covariance_first` | 0.5500 | 0.0500 | 2.0000 | 6.00 |

## HARC Objective

- `min_Wm sum_i E_x KL(softmax(W_i x) || softmax(W_m x))`
- `(sum_i E[H_i kron xx^T]) vec(W_m^T) = sum_i E[H_i kron xx^T] vec(W_i^T)`
- Same-shape constraint: `only update existing router tensors; output checkpoint keeps the same architecture and tensor shapes`

## Outputs

- `results/qwen3_moe_harc_readiness_gate_smoke/mock_inputs/case_outputs/harc_solver_ready/harc_readiness_requirements.csv`
- `results/qwen3_moe_harc_readiness_gate_smoke/mock_inputs/case_outputs/harc_solver_ready/layer_harc_priority.csv`
- `results/qwen3_moe_harc_readiness_gate_smoke/mock_inputs/case_outputs/harc_solver_ready/harc_solver_plan.json`
- `results/qwen3_moe_harc_readiness_gate_smoke/mock_inputs/case_outputs/harc_solver_ready/summary.json`
