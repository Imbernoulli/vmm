# Qwen3 MoE HARC Readiness Gate

This gate turns the MoE router-breakdown mechanism into an executable decision: direct router averaging stays rejected, and HARC-style router distribution matching is allowed only as a calibrated repair path with matched downstream acceptance.

## Result

- Status: `harc_ready_for_curvature_collection_waiting_cache`
- Preconditions: `6/6`
- HARC cache: `missing`
- Top priority layer: `L17` score `0.7379`
- Recommended action: `collect_hessian_covariance_router_stats_then_run_harc_solver`

## Requirements

| requirement | role | passed | evidence | next action |
| --- | --- | --- | --- | --- |
| `direct_router_average_rejected` | `harc_precondition` | `True` | router action=freeze_router; allowed layers=0/48; top-k jaccard mean/min=0.4539/0.2422 | freeze direct router average and use calibrated router objective |
| `topk_boundary_fragility_detected` | `harc_precondition` | `True` | high fragility layers=24/48; min safe lambda=0.0197; top layer=L17 score=0.7523 | prioritize fragile router layers for HARC statistics |
| `router_only_repair_signal_positive` | `harc_precondition` | `True` | worst NLL reduction=0.2214; avg NLL reduction=0.1610; acceptance=mechanism_supported_but_do_not_accept_without_matched_vllm_eval | keep router calibration as repair, not acceptance, until matched vLLM passes |
| `router_expert_coupling_active` | `harc_precondition` | `True` | gate=router_expert_coupling_active; fragility->feature corr=0.6947; shrink corr=0.5831 | calibrate router together with expert-cap law, not as an isolated tensor average |
| `safe_default_router_calibration_frontier_exists` | `harc_precondition` | `True` | default candidates=2/4; recommended=['cap001', 'margin_profile']; blocker=baseline_eval,source_eval,candidate_eval,audit,group_validation,capacity_metrics | compare HARC-style calibration against route-KD cap001 and margin_profile |
| `calibration_job_preflight_ready` | `harc_precondition` | `True` | job status=job_ready_awaiting_gpu; prompts=True; source controls=True; student=True; teacher=True | run preflight on GPU host, then collect router logits/hidden states |
| `hessian_covariance_cache_available` | `harc_solver_requirement` | `False` | cache dir=results/qwen3_moe_harc_router_stats exists=False | collect H_i=diag(r)-rr^T and hidden covariance per router layer |

## Layer Priority

| layer | score | role | fragility | safe lambda | coupled risk | route mass |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 17 | 0.7379 | `collect_hessian_covariance_first` | 0.7523 | 0.0261 | 6.3802 | 12.00 |
| 18 | 0.6737 | `collect_hessian_covariance_first` | 0.7267 | 0.0391 | 6.3295 | 12.00 |
| 12 | 0.6725 | `collect_hessian_covariance_first` | 0.7180 | 0.0338 | 5.8829 | 12.00 |
| 20 | 0.6607 | `collect_hessian_covariance_first` | 0.7157 | 0.0439 | 6.5776 | 12.00 |
| 15 | 0.6461 | `collect_hessian_covariance_first` | 0.7040 | 0.0438 | 6.3323 | 12.00 |
| 13 | 0.6459 | `collect_hessian_covariance_first` | 0.7122 | 0.0384 | 5.7471 | 12.00 |
| 11 | 0.6428 | `collect_hessian_covariance_first` | 0.7151 | 0.0352 | 5.3432 | 12.00 |
| 24 | 0.6407 | `collect_hessian_covariance_first` | 0.6790 | 0.0392 | 6.0144 | 12.00 |
| 19 | 0.6347 | `collect_hessian_covariance_first` | 0.6911 | 0.0436 | 6.1677 | 12.00 |
| 29 | 0.6292 | `collect_hessian_covariance_first` | 0.6910 | 0.0396 | 5.6688 | 12.00 |
| 14 | 0.6223 | `collect_hessian_covariance_first` | 0.6738 | 0.0444 | 6.1190 | 12.00 |
| 4 | 0.6190 | `collect_hessian_covariance_first` | 0.7026 | 0.0308 | 4.6917 | 12.00 |
| 22 | 0.6088 | `collect_hessian_covariance_first` | 0.6760 | 0.0500 | 6.3057 | 12.00 |
| 25 | 0.5947 | `collect_hessian_covariance_first` | 0.6884 | 0.0459 | 5.4762 | 12.00 |
| 21 | 0.5797 | `track_in_margin_profile` | 0.6385 | 0.0425 | 5.2728 | 12.00 |
| 47 | 0.5728 | `track_in_margin_profile` | 0.6664 | 0.0324 | 3.9006 | 12.00 |

## HARC Objective

- `min_Wm sum_i E_x KL(softmax(W_i x) || softmax(W_m x))`
- `(sum_i E[H_i kron xx^T]) vec(W_m^T) = sum_i E[H_i kron xx^T] vec(W_i^T)`
- Same-shape constraint: `only update existing router tensors; output checkpoint keeps the same architecture and tensor shapes`

## Outputs

- `results/qwen3_moe_harc_readiness_gate/harc_readiness_requirements.csv`
- `results/qwen3_moe_harc_readiness_gate/layer_harc_priority.csv`
- `results/qwen3_moe_harc_readiness_gate/harc_solver_plan.json`
- `results/qwen3_moe_harc_readiness_gate/summary.json`
