# Qwen3 MoE Router Calibration Frontier

这个 artifact 只回答 router calibration 这一个机制问题：direct router average 已经被 top-k margin probe 拒绝；如果要动 router，只能作为 route-KD/HARC-style 的小 delta，并且必须先过 margin、audit、capacity、source endpoint 和 matched vLLM gates。

## Result

- Status: `router_calibration_frontier_ready`
- Safe-lambda proxy: `0.0197`
- High-fragility router layers: `24/48`
- Default-run router-cal candidates: `2/4`
- Stress-only candidates: `2`
- NLL repair signal: worst reduction `0.2214`
- Generation repair signal: avg gain `0.0333`, gap to best parent `-0.0694`
- Selection status: `awaiting_baseline_eval`
- Recommended default candidates: `cap001, margin_profile`
- Acceptance blocker: `baseline_eval,source_eval,candidate_eval,audit,group_validation,capacity_metrics`

## Frontier

| cap | mode | cap/safe | margin | role | repair score | next action |
| --- | --- | ---: | --- | --- | ---: | --- |
| `cap001` | `global` | 0.5070 | `True` | `default_probe_waiting_downstream_gate` | 0.2214 | run_route_kd_materialization_audit_and_matched_vllm_eval |
| `margin_profile` | `per_router_margin_profile` | 2.5348 | `True` | `default_probe_waiting_downstream_gate` | 0.2214 | run_route_kd_materialization_audit_and_matched_vllm_eval |
| `cap0025` | `global` | 1.2674 | `False` | `stress_ablation_not_default` | 0.1747 | keep_for_stress_only_until_safe_cap_wins_or_vllm_evidence_overrides |
| `cap005` | `global` | 2.5348 | `False` | `stress_ablation_not_default` | 0.0874 | keep_for_stress_only_until_safe_cap_wins_or_vllm_evidence_overrides |

## Contract

| requirement | mechanism | status | passed | observed | action |
| --- | --- | --- | --- | --- | --- |
| `direct_router_average_rejected` | `topk_boundary_stability` | `router_margin_fragility_rejects_direct_router_average` | `True` | allowed_layers=0/48; min_safe_lambda=0.0197 | freeze source-averaged router; only test calibrated router deltas |
| `safe_router_delta_frontier_exists` | `margin_capped_route_kd` | `job_ready_awaiting_gpu` | `True` | default_run_candidates=2/4 | run only cap001 and margin-profile by default; keep larger caps as stress ablations |
| `local_router_repair_signal` | `router_expert_mismatch` | `signal_present` | `True` | nll_worst_reduction=0.2214; generation_avg_gain=0.0333 | treat router calibration as active repair lever, not final acceptance |
| `matched_vllm_source_dominance_gate` | `selector_level_no_regression` | `awaiting_baseline_eval` | `False` | baseline_eval=False; source_eval=False; eligible=0/4 | reject or wait until baseline, source endpoints, candidate eval, audit, and manifest gates pass |
| `same_shape_router_only_delta` | `architecture_invariant` | `False` | `False` | audit_completed=False; candidate_eval=False | accept only audited router tensors; no expert, attention, embedding, config, or topology changes |

## Algorithm Consequence

Router calibration is now an explicit frontier, not a default average step. `cap001` and `margin_profile` are the only default probes because they respect the observed router top-k margin; `cap0025` and `cap005` remain stress ablations. Even if local NLL and small generation evidence are positive, the unified algorithm still cannot append a router delta until matched vLLM source-dominance gates pass.

## Literature Hooks

- [When Model Merging Breaks Routing: Training-Free Calibration for MoE](https://arxiv.org/abs/2606.03391): Direct router averaging can cross top-k dispatch boundaries; router movement should be a calibrated, margin-gated intervention.
- [Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression](https://arxiv.org/abs/2603.02217): After experts are edited or merged, router-expert mismatch is a separate failure mode; router-only KD is a small repair lever.
- [Model Merging by Output-Space Projection](https://arxiv.org/abs/2605.29101): Output-space calibration is useful only when the local repair signal transfers to downstream held-out behavior.
- [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482): Averaging is accepted only after validation, with source or baseline fallback kept in the candidate set.

## Outputs

- `frontier`: `results/qwen3_moe_router_calibration_frontier/router_calibration_frontier.csv`
- `contract`: `results/qwen3_moe_router_calibration_frontier/router_calibration_contract.csv`
- `literature`: `results/qwen3_moe_router_calibration_frontier/literature_sources.json`
- `summary`: `results/qwen3_moe_router_calibration_frontier/summary.json`
- `report`: `results/qwen3_moe_router_calibration_frontier/report.md`
