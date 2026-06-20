# Average Trust-Region Bounds

这个产物把 Dense 和 MoE averaging 的 probe 转成可执行 trust-region 约束：不是问某个算法名是否流行，而是问当前证据允许哪些参数方向移动多远。

- Status: `trust_region_bounds_ready_waiting_vllm`
- Constraints: `11`
- Passed / rejected / waiting: `2` / `7` / `2`
- Dense local task-vector lambda bound: `0.3416`
- Dense safe uniform lambda from held-out path: `0.0000`
- MoE router safe lambda proxy: `0.0197`
- Mechanistic expert delta cap: `0.6490`

## Constraint Bounds

| domain | mechanism | status | measured | allowed | candidate | over bound | action |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `dense` | `local_quadratic_trust_radius` | `reject_linear_task_vector_average` | 42.8604 | 0.3416 | 1.0000 | 2.9278 | Search a held-out coefficient family and allow endpoint/anchor fallback. |
| `dense` | `heldout_lambda_frontier` | `reject_uniform_linear_lambda` | 6.0398 | 0.0000 | 1.0000 | n/a | Use the observed best lambda config instead of fixed 0.5/0.5 averaging. |
| `dense` | `sparse_sign_trust_radius` | `keep_sparse_methods_conditional` | n/a | n/a | 1.0000 | n/a | Treat sign/sparsity conflict as an ablation until it beats endpoint and anchor gates. |
| `moe` | `qwen3_instruct_coder_source_line` | `reject_source_to_source_linear_average` | 0.1189 | 0.0000 | 0.1189 | n/a | Use route/evidence/geometry/subspace constrained same-shape expert rules, not source midpoint. |
| `moe` | `qwen3_base_coder_source_line` | `reject_base_to_specialist_source_line` | 0.1058 | 0.0000 | 0.1058 | n/a | Do not assume base-anchored source deltas are safe without the same path gate. |
| `moe` | `qwen3_specialist_complementarity` | `do_not_assume_complementarity_is_averageable` | 0.0000 | 0.0000 | 0.0000 | n/a | Keep specialist complementarity as an eval hypothesis, not as an average acceptance rule. |
| `moe` | `router_topk_margin` | `reject_direct_router_average` | 0.0197 | 0.0197 | 0.5000 | 25.3483 | Freeze router for expert candidates; only allow separately audited route-KD deltas. |
| `moe` | `unified_routed_expert_cap` | `expert_delta_cap_passed` | 0.6438 | 0.6500 | 0.6438 | 0.9905 | Keep the unified mechanism candidate provisional until downstream source-dominance gates pass. |
| `moe` | `mechanistic_scale_law_cap` | `mechanistic_cap_and_retention_passed` | 0.6490 | 0.6490 | 0.6490 | 1.0000 | Use as a structural-frontier candidate, but do not override a statistically separated downstream leader. |
| `moe` | `router_calibration_acceptance` | `router_calibration_promising_but_unaccepted` | 0.2214 | 4.0000 | 0.0000 | 0.0000 | Run matched frozen-router baseline/source/candidate vLLM eval before attaching router deltas. |
| `moe` | `final_average_acceptance` | `awaiting_matched_vllm_eval` | 0.0000 | 10.0000 | 0.0000 | 0.0000 | Accept no average until source endpoints and same-shape candidates pass locked-manifest eval bundle audit. |

## Decisions

| scope | decision | evidence | next gate |
| --- | --- | --- | --- |
| `dense` | `reject fixed midpoint; use coefficient search with endpoint or anchor fallback` | linear worst NLL = 6.0398; best lambda-family worst NLL = 3.0727; best endpoint worst NLL = 3.1737; max accepted uniform lambda = 0.0000. | `held-out generation or vLLM eval for any new interior point` |
| `moe_router` | `freeze direct router average; route-KD/router calibration remains a separate ablation` | min safe-lambda proxy = 0.0197; direct midpoint router lambda = 0.5000; high-fragility layers = 24/48. | `matched frozen-router baseline/source/candidate vLLM eval` |
| `moe_experts` | `allow capped same-shape expert movement only inside routed delta and retention bounds` | candidate s0.08_b1.65_h0.75_i0.75 max predicted relative delta = 0.6490; effective cap = 0.6490; route-mass retention = 0.9650; min retention = 0.9650. | `locked-manifest downstream source-dominance selector` |
| `final_selection` | `do not accept any average before audited downstream eval bundles exist` | final selector status = awaiting_source_eval; eligible candidates = 0/10; reason = Both Qwen3 source endpoints must complete audited vLLM eval before final candidate selection.. | `qwen3_moe_eval_bundle_audit plus final candidate selector` |

## Outputs

- `constraints`: `results/average_trust_region_bounds/trust_region_constraints.csv`
- `decisions`: `results/average_trust_region_bounds/trust_region_decisions.csv`
- `algorithm`: `results/average_trust_region_bounds/algorithm.json`
- `summary`: `results/average_trust_region_bounds/summary.json`
- `report`: `results/average_trust_region_bounds/report.md`
