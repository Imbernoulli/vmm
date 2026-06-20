# Average Invariant Audit

This audit turns model-averaging literature and current Dense/MoE probes into executable invariants. It is not a method leaderboard: it explains which assumptions must hold before an average can be accepted, and which assumptions currently fail.

## Result

- Invariants: `10`
- Hard gates not yet accepting average: `4`
- Default-accepted method families: `0`
- Current algorithm: `same_shape_guarded_average`
- Final selection status: `awaiting_source_eval`

## Invariants

| invariant | domain | hard | status | current value | algorithm action |
| --- | --- | --- | --- | --- | --- |
| `same_shape_contract` | `dense+moe` | `True` | `pass` | same_shape=True | Reject topology-expanding methods as final outputs; use them only as upper-bound probes. |
| `endpoint_frontier` | `dense+moe` | `True` | `reject_default_average` | path_rejected=5/6; frontier_wins=1 | Always include source endpoints and allow endpoint/anchor fallback. |
| `fixed_midpoint_safety` | `dense+moe` | `True` | `reject_default_average` | midpoint_rejected=5/6; dense_midpoint_gap=2.9168; qwen3_midpoint_gap=0.4683 | Keep uniform average as a negative baseline unless midpoint gap is within tolerance. |
| `local_quadratic_validity` | `dense` | `False` | `reject_as_sufficient_gate` | Fisher actual/predicted degradation ratios are far above a small local-error regime. | Use local curvature as a feature, not as a proof of merge safety. |
| `expert_identity_alignment` | `moe` | `True` | `pass` | identity_layers=48/48; identity_fraction=1.0000 | Use identity slices for Qwen3; require remap/alignment for other MoEs. |
| `router_stability` | `moe` | `True` | `reject_router_movement` | allowed_router_layers=0/48; min_top1=0.0690; min_topk_jaccard=0.2422 | Freeze router by default; test router-KD/HARC-style calibration as a separate intervention. |
| `expert_internal_geometry` | `moe` | `False` | `conditional_geometry_shrink` | high_internal=931; high_route_geometry=204; mean_cos=0.3862 | Use route/geometry-weighted caps and layer/chunk coefficients. |
| `routed_delta_trust_region` | `moe` | `True` | `pass` | selected_max_predicted_delta=0.6438; groups_gt_hard_cap=0; delta_audit_status=passed | Only materialize candidates that pass cap and same-shape delta audit. |
| `router_calibration_not_acceptance` | `moe` | `False` | `mechanism_supported_not_sufficient` | worst_nll_reduction=0.2214; worst_gap_to_best_source=0.1265 | Queue router-calibrated candidates for matched vLLM evaluation and source-dominance selection. |
| `matched_downstream_dominance` | `dense+moe` | `True` | `awaiting_eval` | usable_candidates=0/9; usable_eval_bundles=0/11 | Do not claim a unified average wins until source and candidate eval bundles pass audit. |

## Method Matrix

| method family | current gate | required invariants | why | use if |
| --- | --- | --- | --- | --- |
| `Uniform / linear average` | `reject_as_default` | endpoint_frontier;fixed_midpoint_safety | Current Dense and Qwen3 MoE midpoint/path evidence is dominated by source endpoints. | Only if the midpoint itself passes endpoint-frontier and downstream gates. |
| `Model soups / greedy soup` | `conditional` | endpoint_frontier;matched_downstream_dominance | Soup logic is valid inside one low-error basin, but current source paths are not proven to be in that basin. | Greedy validation includes endpoints and rejects any candidate that hurts the frontier. |
| `Task arithmetic / coefficient search` | `conditional_endpoint_fallback` | endpoint_frontier;fixed_midpoint_safety | Coefficient search may find an anchor/base fallback while the raw average remains unsafe. | Search over layer/module/expert coefficients and permit endpoint or anchor selection. |
| `TIES / DARE / DELLA / STAR` | `conditional_no_router_blindness` | router_stability;expert_internal_geometry;routed_delta_trust_region | Coordinate conflict methods do not solve MoE dispatch, expert identity, or route-load risk by themselves. | Apply after expert identity/remap, with router frozen/calibrated and routed delta caps. |
| `Fisher / RegMean / RegMean++` | `conditional_not_sufficient` | local_quadratic_validity;matched_downstream_dominance | Local or layerwise regression arguments can underestimate nonlocal merge barriers. | Use as a feature or tensor rule, then validate on held-out NLL and downstream eval. |
| `WEMoE / dynamic MoE upscaling` | `disallowed_as_final_output` | same_shape_contract | Dynamic expert modules can reduce interference but change the model structure requested here. | Use as an upper bound or teacher, then distill/compress back into same-shape rules. |
| `Expert Merging++ / layer chunking` | `active_structural_candidate` | expert_internal_geometry;routed_delta_trust_region;matched_downstream_dominance | Current Qwen3 geometry shows layer/expert heterogeneity; layer/chunk coefficients directly target it. | Keep same-shape tensor rules and require final vLLM source-frontier dominance. |
| `Sub-MoE / expert-output clustering` | `probe_only_under_same_shape_contract` | same_shape_contract;expert_identity_alignment | Expert compression can change expert count, but output similarity is useful as a probe. | Use output/subspace similarity to decide remaps or caps without changing tensor topology. |
| `HARC / router-only calibration` | `active_separate_intervention` | router_stability;matched_downstream_dominance | Router-only NLL probe improves linear MoE but still does not dominate the best source. | Train/calibrate router deltas under caps, audit them, then evaluate with source endpoints. |
| `Unified same-shape mechanism optimizer` | `structurally_ready_awaiting_eval` | matched_downstream_dominance | It satisfies same-shape, identity, frozen-router, geometry, and delta-cap gates, but final acceptance is still downstream. | Accept only after eval-bundle audit and final selector beat source frontier. |

## Selector-Level Statement

The optimizer is not a proof that averaging always wins. Its defensible guarantee is selector-level: because endpoints are always in the candidate set and final acceptance requires audited matched eval with source-dominance, task-regression, uncertainty, and paired-prediction gates, the deployed choice falls back to the best source whenever the same-shape average lacks measured evidence.

## Outputs

- `results/average_invariant_audit/invariant_table.csv`
- `results/average_invariant_audit/method_invariant_matrix.csv`
- `results/average_invariant_audit/algorithm_spec.json`
- `results/average_invariant_audit/literature_sources.json`
- `results/average_invariant_audit/invariant_status.png`
- `results/average_invariant_audit/summary.json`
