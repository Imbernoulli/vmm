# Unified Average Optimizer

这个脚本把 Dense 和 MoE 的 probe 结果统一成同一个操作选择器：先测几何和对称性，再决定能不能平均、平均多少、哪些结构必须冻结或校准。它不是按算法名投票，而是让每个平均动作都绑定一个可测机制。

## Current Decision

- Dense: `avoid_linear_midpoint_use_probe_selected_anchor_or_low_lambda`；linear worst NLL `8.9477`，unified worst NLL `5.1830`。
- Dense lambda connectivity: midpoint worst NLL `6.0398`，best lambda-family worst NLL `3.0727`。
- MoE: `align_experts_freeze_router_then_gate_candidate_by_vllm`；真实 OLMoE same-name average degradation `5.4910`，Qwen3 router action `freeze_router`。
- Qwen3 MoE straight-line connectivity: best interior worst NLL `2.5947` vs best endpoint `2.4757`；interior gap `0.1189`，general barrier `0.1097`。
- Qwen3 complementary path: best merge avg NLL `0.5659` vs best source avg `0.5659`；merge beats sources `False`。
- Qwen3 Base->Coder path: best interior worst NLL `2.4661` vs best endpoint `2.3603`；interior gap `0.1058`，general barrier `0.0728`。
- Qwen3 unified mechanism: `subspace_cap_s1.00`；retention `0.9763`，subspace-weighted rel-delta `0.2148`，high-subspace mean scale `0.9614`，materialized rules `fresh`，audit relative norm `0.2404`，routed >0.65 `0`。
- Qwen3 subspace-scaled ablation: audit relative norm `0.2395`，mechanistic->subspace norm delta `0.001304`，routed >0.65 `0`。
- Qwen3 router margin fragility: high layers `24/48`，top `L17` score `0.7523`，min safe-lambda proxy `0.0197`。
- Qwen3 router-coupled retention frontier: gate `direct_router_boundary_term_not_default`，default-gate candidates `146/770`，effect fraction `0.0103`，constrained `router_q0.85_s0.00020_cap0.00010`，stress `router_q0.75_s0.01000_cap0.01000`。
- Qwen3 router NLL probe: worst-NLL reduction `0.2214`，code gap to best source `-0.0139`。
- Qwen3 generation matrix: Instruct+Coder avg `0.7944` -> router-cal avg `0.8278`；avg gain `0.0333`，gap to best parent `-0.0694`。
- Qwen3 generation attribution: router-cal recovers `0.3243` of avg naive drop and beats pair frontier on `0/5` scores。
- Qwen3 generation confidence: positive tasks vs naive `2/3`，confident positives `0/3`，confident source-frontier wins `0/3`；avg gain interval `[-0.1703, 0.2312]`。
- Qwen3 source-set complementarity: `instruct+coder` gate `source_dominated_not_averageable_as_final`，dominant `instruct`，frontier avg gain `0.0000`，best observed merge gap `-0.0694`。
- Qwen3 source-set surplus: top `coder+thinking` gate `probe_only_below_interference_budget`，frontier avg gain `0.0083` vs interference budget `0.0694`，surplus `-0.0611`，weights `{"coder": 0.6666666666666666, "thinking": 0.3333333333333333}`。
- Qwen3 router calibration: `awaiting_baseline_eval`。
- Qwen3 final selection: `awaiting_source_eval`，eligible `0/11`。
- Qwen3 final selector rank gate: confidence band `True`，rank mode `None`，band size `0`，point leader `None`。

## Mechanism Features

| domain | probe | signal | value | threshold | evidence |
| --- | --- | --- | ---: | ---: | --- |
| `dense` | `curvature_law_general` | `high_nonlocal_barrier` | 42.8604 | 5.0000 | actual/predicted general degradation = 42.8604; uniform worst NLL = 5.9109; fisher worst NLL = 5.2492 |
| `dense` | `curvature_law_code` | `high_nonlocal_barrier` | 26.6575 | 5.0000 | actual/predicted code degradation = 26.6575 |
| `dense` | `heldout_unified_selector` | `allow_endpoint_or_anchor_fallback` | 5.1830 | 8.9477 | unified test worst NLL = 5.1830; linear = 8.9477; TIES = 9.1097; best endpoint = 5.1510 |
| `dense` | `dense_lambda_connectivity` | `straight_line_midpoint_rejected` | 6.0398 | 3.0727 | linear midpoint worst NLL = 6.0398; best lambda-family worst NLL = 3.0727; best config = {"code": 0.5414439073133305, "general": 3.072743579141455, "lambda": 0.0, "mode": "uniform", "worst": 3.072743579141455}; best source endpoint worst NLL = 3.1737 |
| `dense` | `generation_smoke` | `linear_generation_regression` | 0.0000 | 0.5000 | linear avg accuracy = 0.0000; unified avg accuracy = 0.5000; best smoke method = coder |
| `moe` | `controlled_expert_gauge` | `expert_permutation_is_function_preserving` | 0.0000 | 0.0000 | gauge-equivalent B MSE = 0.00000000; same-name worst = 0.5105; aligned worst = 0.1252 |
| `moe` | `real_olmoe_gauge_selfmerge` | `reject_same_name_average_without_alignment` | 5.4910 | 1.0000 | baseline NLL = 4.1678; same-name average NLL = 9.6588; aligned average NLL = 4.1678; layers recovered = 16/16 |
| `moe` | `qwen3_expert_identity` | `identity_alignment_is_allowed_for_this_pair` | 1.0000 | 1.0000 | identity-optimal layer fraction = 1.0000; argmax identity fraction = 1.0000 |
| `moe` | `qwen3_router_move_gate` | `freeze_router_or_train_route_kd_delta` | 0.0000 | 48.0000 | allowed router layers = 0/48; top-k Jaccard mean/min = 0.4539/0.2422; top1 agreement mean/min = 0.4125/0.0690 |
| `moe` | `qwen3_router_margin_fragility` | `topk_boundary_lambda_cap_rejects_direct_router_average` | 0.7523 | 0.6200 | high-fragility layers = 24/48; top layer = L17 score 0.7523; top category = long_context score 0.7329; min safe-lambda proxy = 0.0197 |
| `moe` | `qwen3_router_coupled_retention_frontier` | `direct_router_boundary_shrink_not_default_under_retention` | 0.0103 | 0.0500 | gate = direct_router_boundary_term_not_default; default-gate candidates = 146/770; constrained = router_q0.85_s0.00020_cap0.00010 retention_delta -0.000032 coupled_reduction 0.00001165; stress = router_q0.75_s0.01000_cap0.01000 coupled_reduction 0.001135; effect fraction = 0.0103 |
| `moe` | `qwen3_straight_line_connectivity` | `reject_source_to_source_linear_interpolation` | 0.1189 | 0.0000 | best interior worst NLL = 2.5947; best endpoint worst NLL = 2.4757; interior gap = 0.1189; general barrier = 0.1097; task-vector cosine vs base = 0.1799 |
| `moe` | `qwen3_complementary_pair_connectivity` | `do_not_assume_specialist_complementarity_is_averageable` | 0.0000 | 0.0000 | best merge avg NLL = 0.5659; best source avg NLL = 0.5659; merge-source gap = 0.0000; best merge t = 0.0000; merge beats both sources = False |
| `moe` | `qwen3_base_to_coder_connectivity` | `source_delta_from_base_is_not_safe_without_gate` | 0.1058 | 0.0000 | best interior worst NLL = 2.4661; best endpoint worst NLL = 2.3603; interior gap = 0.1058; general barrier = 0.0728; task-vector norm = 4379.0027 |
| `moe` | `qwen3_unified_mechanism_optimizer` | `use_router_evidence_geometry_subspace_risk_caps` | 0.2247 | 0.6500 | selected = subspace_cap_s1.00; family = subspace_weighted_cap; retention = 0.9763; risk-weighted predicted rel delta = 0.2247; geometry-weighted predicted rel delta = 0.2184; subspace-weighted predicted rel delta = 0.2148; high-subspace mean scale = 0.9614 |
| `moe` | `qwen3_unified_materialized_audit` | `materialized_same_shape_tail_reduction` | 0.2404 | 0.2435 | rule status = fresh; audit status = passed; total relative norm = 0.2404; router changed = 0/48; layer/chunk->unified norm reduction = 0.0031; routed >0.65 reduction = 89 |
| `moe` | `qwen3_subspace_scaled_materialized_audit` | `test_uncovered_subspace_conflict_shrink` | 0.2395 | 0.2404 | subspace total relative norm = 0.2395; mechanistic->subspace norm delta = 0.001304; subspace routed >0.65 = 0; router changed = 0 |
| `moe` | `qwen3_router_calibration_nll_probe` | `router_dispatch_is_real_optimization_lever` | 0.2214 | 0.0000 | status = router_calibration_improves_linear_merge_but_needs_downstream_gate; linear worst NLL = 2.6355; router-cal worst NLL = 2.4140; worst reduction = 0.2214; code gap to best source = -0.0139 |
| `moe` | `qwen3_generation_downstream_routercal_matrix` | `router_calibration_recovers_generation_interference_but_not_endpoint_dominance` | 0.0333 | 0.0000 | status = generation_downstream_matrix_ready; Instruct+Coder avg = 0.7944; +router-cal avg = 0.8278; avg gain = 0.0333; HumanEval gain = 0.0750; gap to best parent avg = -0.0694 |
| `moe` | `qwen3_generation_routercal_effect_attribution` | `router_calibration_is_repair_not_acceptance_rule` | 0.3243 | 1.0000 | status = generation_mechanism_attribution_ready; avg naive drop = 0.1028; avg recovery fraction = 0.3243; HumanEval recovery = 0.5000; beats pair frontier = 0/5 |
| `moe` | `qwen3_generation_confidence_audit` | `generation_gain_directional_not_confident_or_source_dominant` | 0.0000 | 3.0000 | status = generation_confidence_audit_ready; positive tasks vs naive = 2/3; confident positive tasks = 0/3; confident source-frontier wins = 0/3; avg gain interval = [-0.1703, 0.2312] |
| `moe` | `qwen3_source_set_complementarity_gate` | `current_source_set_is_source_dominated_not_complementary` | 0.0000 | 0.0200 | current source set = instruct+coder; gate = source_dominated_not_averageable_as_final; dominant source = instruct; frontier avg gain = 0.0000; best observed merge = merge_instruct+coder_routercal gap -0.0694; complementary measured sets = 2 |
| `moe` | `qwen3_source_set_surplus_optimizer` | `measured_complementarity_is_below_observed_interference_budget` | -0.0611 | 0.0000 | status = source_set_surplus_optimizer_ready; top source set = coder+thinking; top gate = probe_only_below_interference_budget; frontier avg gain = 0.0083; interference budget = 0.0694; surplus = -0.0611; final-budget sets = 0; probe-only sets = 2 |
| `moe` | `qwen3_router_calibration_gate` | `do_not_accept_router_delta_without_baseline_eval` | 0.0000 | 4.0000 | status = awaiting_baseline_eval; eligible router-cal candidates = 0/4; reason = Run the frozen-router searched_no_gt065 baseline eval before deciding whether router calibration helps. |
| `moe` | `qwen3_final_candidate_selection` | `await_matched_vllm_before_accepting_average` | 0.0000 | 11.0000 | status = awaiting_source_eval; eligible candidates = 0/11; reason = Both Qwen3 source endpoints must complete audited vLLM eval before final candidate selection. |
| `moe` | `qwen3_final_selector_confidence_band` | `use_structural_tiebreak_only_when_downstream_scores_are_statistically_tied` | 0.0000 | 0.0000 | confidence tie band = True; rank mode = None; point leader = None; band size = 0; structural-frontier eligible = 0; rank band methods = [] |

## Operations

| stage | operation | selected action | why |
| --- | --- | --- | --- |
| `dense_connectivity_gate` | `do_not_use_linear_midpoint_by_default` | search base-anchored coefficient family; current config = {"density": 1.0, "importance": "uniform", "lam": 0.0, "router": "average", "sign_resolve": true} | It prevents a fixed 0.5 midpoint from crossing a measured high-loss barrier. |
| `dense_sparse_coordinate_gate` | `make TIES/DARE-style sparsity conditional` | only materialize sparse conflict rules when held-out and vLLM gates pass | It keeps sign-conflict probes as diagnostics without letting them delete useful dense capacity. |
| `moe_expert_identity_gate` | `canonicalize expert gauge before averaging` | run layer-wise expert alignment; for Qwen3 Instruct/Coder the mapping is currently identity | It removes a discrete symmetry error before any continuous weight interpolation is attempted. |
| `moe_router_gate` | `freeze direct router movement` | freeze_router | It avoids averaging a discrete top-k dispatch boundary that has high measured source disagreement. |
| `moe_router_margin_cap_gate` | `bound router movement by observed top-k margins` | freeze_router; direct router average rejected by router_margin_fragility_rejects_direct_router_average | It prevents a small weight-space router step from crossing a discrete dispatch boundary and sending tokens to untrained expert combinations. |
| `moe_router_coupled_shrink_gate` | `keep router-boundary fragility inside B/H/I instead of direct extra shrink` | keep_router_fragility_inside_BHI_and_keep_direct_extra_shrink_as_ablation; constrained=router_q0.85_s0.00020_cap0.00010 coupled_reduction=0.00001165; stress=router_q0.75_s0.01000_cap0.01000 coupled_reduction=0.001135 | It avoids spending retention budget on a direct router-boundary term whose safe version is too weak; the same signal stays useful as an interference feature inside the expert scale objective. |
| `moe_straight_line_connectivity_gate` | `reject_unconditional_source_to_source_linear_interpolation` | use route/evidence/geometry-constrained same-shape candidates instead of a source-to-source midpoint | It treats model connectivity as measured evidence: a smooth-looking line is not accepted unless an interior point beats the source frontier. |
| `moe_source_set_complementarity_gate` | `require source-set complementarity before expecting average to beat sources` | do_not_expect_average_to_beat_current_source_frontier; use average only as repair_ablation_until_new_complementary_sources_pass_endpoint_eval | It avoids spending algorithmic effort trying to make an average beat a source set whose measured frontier is already a single dominant endpoint. |
| `moe_source_set_surplus_gate` | `require complementarity surplus to cover observed merge interference` | probe_only_below_interference_budget: run_larger_endpoint_eval_and_small_weighted_probe_no_final_acceptance; source_weights={"coder": 0.6666666666666666, "thinking": 0.3333333333333333} | It prevents a weakly complementary source set from being promoted to final-average budget when measured average interference is larger than the available task-frontier gain. |
| `moe_expert_delta_optimizer` | `apply retention-constrained router/evidence/geometry/subspace caps` | subspace_cap_s1.00 with hard cap 0.6500; subspace-weighted rel-delta = 0.2148; materialized rule status = fresh; layer/chunk->unified routed >0.65 reduction = 89; mechanistic->subspace norm delta = 0.001304 | It keeps useful Coder-route mass while shrinking high-risk routed expert deltas and local subspace conflicts instead of using one global coefficient. |
| `moe_router_calibration_gate` | `treat router calibration as a separately audited ablation` | nll_probe_worst_reduction=0.2214; generation_avg_gain=0.0333; generation_recovery_fraction=0.3243; confidence_positive_tasks=0/3; confidence_source_frontier_wins=0/3; awaiting_baseline_eval: Run the frozen-router searched_no_gt065 baseline eval before deciding whether router calibration helps. | It keeps router calibration as an active MoE-specific lever while still requiring source-dominance and task-regression gates before acceptance. |
| `moe_statistical_selector_gate` | `rank structural mechanisms only inside a downstream confidence tie band` | confidence_tie_band=True; rank_mode=None; point_leader=None; band_size=0; structural_frontier_eligible=0 | It avoids overfitting tiny finite-sample score gaps while still refusing to let structural cleanliness override a statistically separated downstream winner. |
| `moe_candidate_gate` | `select only after audited downstream eval` | keep all registered Qwen3 candidates provisional until eval-bundle audit passes | It prevents structural cleanliness from being mistaken for actual downstream dominance. |

## Falsification Tests

| hypothesis | status | current evidence | falsification test | next command |
| --- | --- | --- | --- | --- |
| `dense_same_basin_required` | `rejected_for_current_midpoint` | curvature ratios general/code = 42.8604/26.6575; midpoint worst NLL = 6.0398; best lambda-family worst NLL = 3.0727 | A lambda/path sweep plus held-out generation eval must show an interior same-shape checkpoint beating the endpoint frontier on worst-task score. | `python scripts/fp_dense_lambda.py --out results/fp_dense_lambda` |
| `dense_coordinate_conflict_is_diagnostic_not_default` | `conditional_only` | linear worst NLL = 8.9477; unified worst NLL = 5.1830; best endpoint worst NLL = 5.1510 | A TIES/DARE/DELLA-style materialization must improve worst-task score over the current anchor without endpoint domination. | `python scripts/fp_merge_compare.py --help` |
| `moe_expert_gauge_alignment_precedes_average` | `alignment_required` | real same-name degradation = 5.4910; aligned degradation = 0.0000; Qwen3 identity fraction = 1.0000 | Expert output cosine/route coactivation must show same-index experts are functionally matched, or a remap must recover the self-merge baseline. | `python scripts/fp_moe_barrier.py --help` |
| `moe_direct_router_average_crosses_topk_boundaries` | `direct_router_average_rejected` | router action = freeze_router; high-fragility layers = 24/48; min safe-lambda proxy = 0.0197 | A router-moving candidate must preserve top-k overlap/load and beat the frozen-router candidate under the same downstream manifest. | `results/qwen3_moe_router_calibration_job/run_router_calibration_job.sh preflight` |
| `moe_direct_router_boundary_shrink_is_not_default` | `direct_shrink_ablation_only` | frontier gate = direct_router_boundary_term_not_default; default-gate candidates = 146/770; effect fraction = 0.0103; constrained reduction = 0.00001165; stress reduction = 0.001135 | A direct router-boundary shrink must pass retention/hard-cap gates and achieve a material fraction of the aggressive ablation effect, then beat the B/H/I-only default under the locked downstream manifest. | `python scripts/analyze_qwen3_moe_router_coupled_retention_frontier.py --output-dir results/qwen3_moe_router_coupled_retention_frontier` |
| `moe_source_to_source_line_not_averageable` | `straight_line_rejected` | Instruct/Coder interior gap = 0.1189; Base/Coder interior gap = 0.1058; complementary merge beats sources = False | A straight-line or 2D plane sweep must find an interior point that beats both source endpoints on the paired downstream frontier. | `python scripts/fp_moe_barrier.py --out results/fp_moe_barrier` |
| `source_set_complementarity_precedes_average` | `current_qwen3_pair_source_dominated` | source set = instruct+coder; gate = source_dominated_not_averageable_as_final; dominant source = instruct; frontier avg gain = 0.0000; best observed merge gap = -0.0694; complementary measured sets = 2 | Endpoint eval for a source set must show a material task-frontier gain over its best single source, then the same-shape average must approach or beat that frontier under locked downstream eval. | `python scripts/build_qwen3_source_set_complementarity_gate.py --output-dir results/qwen3_source_set_complementarity_gate` |
| `source_set_surplus_must_exceed_interference` | `top_measured_source_set_probe_only_below_interference_budget` | top source set = coder+thinking; top gate = probe_only_below_interference_budget; frontier avg gain = 0.0083; interference budget = 0.0694; surplus = -0.0611; source weights = {"coder": 0.6666666666666666, "thinking": 0.3333333333333333} | A measured source set must show frontier_avg_gain minus observed merge-interference budget >= 0, then its same-shape average must pass locked downstream eval. | `python scripts/build_qwen3_average_source_set_optimizer.py --output-dir results/qwen3_average_source_set_optimizer` |
| `moe_risk_weighted_expert_caps_preserve_useful_route_mass` | `structurally_passed_waiting_downstream_eval` | candidate = subspace_cap_s1.00; retention = 0.9763; subspace-weighted rel-delta = 0.2148; routed >0.65 = 0 | Budgeted vLLM eval must show the unified candidate is non-dominated by both sources and does not regress any task beyond tolerance. | `results/qwen3_moe_eval_budget_plan/run_eval_budget.sh final` |
| `router_calibration_repairs_dispatch_but_is_not_acceptance` | `promising_but_unaccepted` | NLL worst reduction = 0.2214; generation avg gain = 0.0333; confident positive tasks = 0/3; source-frontier wins = 0/3 | Router-calibrated candidates must beat the frozen-router baseline under paired source controls and maintain router-only audit caps. | `results/qwen3_moe_router_calibration_job/run_router_calibration_job.sh all` |
| `downstream_source_dominance_is_final_gate` | `awaiting_source_eval` | final selection status = awaiting_source_eval; eligible candidates = 0/11; router calibration status = awaiting_baseline_eval | All candidates and sources must be scored on the locked manifest; dominated averages are rejected even if their structural audit looks clean. | `python scripts/select_qwen3_moe_unified_result.py` |
| `moe_structural_tiebreak_requires_statistical_equivalence` | `policy_ready_waiting_eval` | confidence tie band = True; rank mode = None; point leader = None; rank band size = 0; structural-frontier eligible = 0 | Selector smoke and locked-manifest eval must show that structural frontier/safety affects rank only inside a confidence-overlap band; if intervals separate, the point leader wins. | `python scripts/select_qwen3_moe_final_candidate.py --smoke-matrix --output-dir results/qwen3_moe_final_candidate_selection_smoke` |

## Evidence Ledger

| hypothesis | verdict | evidence tier | current action | gate still needed |
| --- | --- | --- | --- | --- |
| `dense_same_basin_required` | `supports_current_action` | `path_nll_plus_curvature_proxy` | reject_dense_linear_midpoint_use_low_lambda_or_endpoint_anchor | held-out generation or vLLM eval for any new interior dense point |
| `dense_coordinate_conflict_is_diagnostic_not_default` | `supports_conditional_action` | `heldout_nll_selector` | keep_sparse_conflict_rules_as_ablation_not_default | materialized sparse candidate must beat anchor and source frontier |
| `moe_expert_gauge_alignment_precedes_average` | `supports_current_action` | `controlled_and_real_gauge_probe` | canonicalize_or_verify_expert_identity_before_expert_average | per-layer expert matching or verified identity for each target pair |
| `moe_direct_router_average_crosses_topk_boundaries` | `supports_current_action` | `router_margin_and_topk_proxy` | freeze_router_or_only_allow_capped_route_kd_delta | router-moving candidate must pass route overlap, load, audit, and downstream gates |
| `moe_direct_router_boundary_shrink_is_not_default` | `supports_current_action` | `retention_constrained_router_coupled_frontier` | use_router_fragility_as_BHI_interference_feature_keep_direct_shrink_ablation_only | a direct-shrink ablation must beat the B/H/I-only default under matched vLLM eval before promotion |
| `moe_source_to_source_line_not_averageable` | `supports_current_action` | `source_to_source_nll_path_probe` | reject_unconditional_moe_source_to_source_midpoint | a future plane/path sweep must find a non-endpoint downstream win |
| `source_set_complementarity_precedes_average` | `supports_current_action` | `generation_source_frontier_gate` | treat_current_instruct_coder_average_as_repair_or_ablation_until_new_complementary_sources_pass_endpoint_eval | new source sets must show material endpoint complementarity and then pass locked downstream eval |
| `source_set_surplus_must_exceed_interference` | `supports_current_action` | `source_frontier_gain_vs_observed_merge_gap` | keep_top_measured_complementary_source_set_probe_only_until_surplus_beats_interference | find a source set whose frontier gain exceeds observed interference, then pass locked downstream eval |
| `moe_risk_weighted_expert_caps_preserve_useful_route_mass` | `awaiting_downstream_eval` | `structural_audit_without_downstream_acceptance` | keep_unified_mechanism_candidate_provisional | budgeted vLLM eval versus both sources and registered candidates |
| `router_calibration_repairs_dispatch_but_is_not_acceptance` | `promising_but_unaccepted` | `nll_probe_and_generation_smoke` | train_and_eval_router_calibration_as_ablation_not_default | paired vLLM eval with frozen-router baseline and source controls |
| `downstream_source_dominance_is_final_gate` | `awaiting_downstream_eval` | `missing_required_downstream_eval` | do_not_accept_any_average_until_locked_manifest_eval_completes | complete budgeted vLLM eval bundle audit |
| `moe_structural_tiebreak_requires_statistical_equivalence` | `supports_current_action` | `selector_policy_and_smoke_matrix` | apply_structural_frontier_safety_only_inside_confidence_tie_band | complete locked-manifest vLLM eval so the confidence band contains real candidate scores |

## Algorithm Contract

- Contract status: `blocked_on_downstream_eval`
- Requirements passed: `11/13`
- Blocking requirements: `2`

| requirement | mechanism | status | passed | observed | next command |
| --- | --- | --- | --- | --- | --- |
| `same_shape_output_contract` | `architecture_invariant` | `passed` | `True` | algorithm contract writes same-shape operations only | `python scripts/build_unified_average_optimizer.py --output-dir results/unified_average_optimizer` |
| `dense_midpoint_path_gate` | `mode_connectivity_and_curvature` | `passed` | `True` | dense decision=avoid_linear_midpoint_use_probe_selected_anchor_or_low_lambda; midpoint worst=6.0398; best lambda-family worst=3.0727; verdict=supports_current_action | `python scripts/fp_dense_lambda.py --out results/fp_dense_lambda` |
| `moe_expert_identity_gate` | `expert_gauge_alignment` | `passed` | `True` | identity_fraction=1.0000; real same-name degradation=5.4910; verdict=supports_current_action | `python scripts/fp_moe_real_probe.py --help` |
| `moe_router_boundary_gate` | `topk_margin_fragility` | `passed` | `True` | router_action=freeze_router; high_fragility_layers=24/48; min_safe_lambda=0.0197; verdict=supports_current_action | `python scripts/build_qwen3_moe_router_margin_fragility.py --output-dir results/qwen3_moe_router_margin_fragility` |
| `moe_router_coupled_retention_frontier_gate` | `router_boundary_signal_budgeting` | `passed` | `True` | frontier_gate=direct_router_boundary_term_not_default; default_gate_candidates=146/770; effect_fraction=0.0103; constrained=router_q0.85_s0.00020_cap0.00010; stress=router_q0.75_s0.01000_cap0.01000; verdict=supports_current_action | `python scripts/analyze_qwen3_moe_router_coupled_retention_frontier.py --output-dir results/qwen3_moe_router_coupled_retention_frontier` |
| `moe_connectivity_gate` | `source_to_source_path_probe` | `passed` | `True` | instruct/coder interior gap=0.1189; base/coder interior gap=0.1058; complementary beats sources=False; verdict=supports_current_action | `python scripts/fp_moe_barrier.py --out results/fp_moe_barrier` |
| `source_set_complementarity_gate` | `endpoint_frontier_before_average` | `passed` | `True` | source_set=instruct+coder; gate=source_dominated_not_averageable_as_final; dominant=instruct; frontier_avg_gain=0.0000; best_observed_gap=-0.0694; verdict=supports_current_action | `python scripts/build_qwen3_source_set_complementarity_gate.py --output-dir results/qwen3_source_set_complementarity_gate` |
| `source_set_surplus_budget_gate` | `frontier_gain_minus_merge_interference` | `passed` | `True` | top_source_set=coder+thinking; top_gate=probe_only_below_interference_budget; frontier_avg_gain=0.0083; interference_budget=0.0694; surplus=-0.0611; final_budget_sets=0; verdict=supports_current_action | `python scripts/build_qwen3_average_source_set_optimizer.py --output-dir results/qwen3_average_source_set_optimizer` |
| `mechanistic_candidate_structural_gate` | `route_geometry_subspace_cap_audit` | `passed` | `True` | rule_status=fresh; audit_status=passed; routed_gt_0.65=0; router_changed=0/48; manifest_match=True | `python scripts/audit_materialized_checkpoint_delta.py --base BASE --candidate results/checkpoints/qwen3_moe_unified_mechanism_candidate --output-dir results/qwen3_moe_unified_mechanism_delta_audit` |
| `router_calibration_separate_acceptance_gate` | `router_repair_not_default_average` | `passed` | `True` | router_calibration_status=awaiting_baseline_eval; eligible=0/4; verdict=promising_but_unaccepted | `results/qwen3_moe_router_calibration_job/run_router_calibration_job.sh all` |
| `statistical_structural_tiebreak_gate` | `confidence_band_selector` | `passed` | `True` | tie_band=True; rank_mode=None; band_size=0; point_leader=None; verdict=supports_current_action | `python scripts/select_qwen3_moe_final_candidate.py --smoke-matrix --output-dir results/qwen3_moe_final_candidate_selection_smoke` |
| `downstream_source_dominance_gate` | `locked_manifest_vllm_eval` | `blocked_on_downstream_eval` | `False` | final_status=awaiting_source_eval; eligible=0/11; downstream_verdict=awaiting_downstream_eval; unified_verdict=awaiting_downstream_eval | `results/qwen3_moe_eval_budget_plan/run_eval_budget.sh final` |
| `final_unified_average_acceptance` | `all_gates_joint` | `blocked_on_downstream_eval` | `False` | structural_gate=True; downstream_terminal=False; unified_accepted=False; source_fallback=False | `results/qwen3_moe_eval_budget_plan/run_eval_budget.sh final` |

## Next Experiments

| rank | experiment | status | priority | driving verdict | command | expected update |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | `budgeted_qwen3_moe_downstream_eval` | `blocked_on_gpu_vllm` | 1.00 | `downstream_source_dominance_is_final_gate=awaiting_downstream_eval` | `results/qwen3_moe_eval_budget_plan/run_eval_budget.sh final` | select_qwen3_moe_unified_result can move from awaiting_source_eval to accept/reject/source fallback |
| 2 | `router_calibration_active_candidates` | `blocked_on_gpu_vllm` | 0.95 | `router_calibration_repairs_dispatch_but_is_not_acceptance=promising_but_unaccepted` | `results/qwen3_moe_router_calibration_job/run_router_calibration_job.sh all` | router calibration selector can decide cap001/margin_profile vs freeze-router baseline |
| 3 | `source_set_surplus_discovery` | `ready_for_endpoint_expansion_no_final_candidate` | 0.91 | `source_set_surplus_must_exceed_interference=supports_current_action` | `python scripts/build_qwen3_average_source_set_optimizer.py --output-dir results/qwen3_average_source_set_optimizer` | expand endpoint/source-set eval or add more complementary downstream sources before final average materialization |
| 4 | `mechanism_effect_attribution_refresh` | `awaiting_eval_bundle` | 0.88 | `moe_risk_weighted_expert_caps_preserve_useful_route_mass=awaiting_downstream_eval` | `python scripts/attribute_qwen3_moe_mechanism_effects.py` | operation_decisions can stop relying on structural-only risk reductions |
| 5 | `unified_optimizer_refresh` | `ready` | 0.82 | `downstream_source_dominance_is_final_gate=awaiting_downstream_eval` | `python scripts/build_unified_average_optimizer.py --output-dir results/unified_average_optimizer` | mechanism_hypotheses and next_experiment_queue refresh from current artifacts |
| 6 | `dense_low_loss_path_recheck` | `lower_priority_until_qwen_moe_eval_unblocked` | 0.55 | `dense_same_basin_required=supports_current_action` | `python scripts/fp_dense_lambda.py --out results/fp_dense_lambda` | dense_same_basin_required hypothesis can move from rejected to supported for a specific pair |

## Literature Priors

| key | source | mechanism used here |
| --- | --- | --- |
| `loss_landscape` | https://arxiv.org/abs/1712.09913 | Loss landscapes should be inspected on meaningful weight-space directions; visual smoothness is not evidence that a source-to-source midpoint is safe. |
| `mode_connectivity` | https://arxiv.org/abs/1802.10026 | A weight average is trusted only when the probed path stays in a low-loss basin. |
| `model_soups` | https://arxiv.org/abs/2203.05482 | Same-basin finetunes can average well, but endpoint fallback is part of the recipe. |
| `git_rebasin` | https://arxiv.org/abs/2209.04836 | Permutation symmetry must be canonicalized before weight-space merging. |
| `ties` | https://arxiv.org/abs/2306.01708 | Coordinate sign conflict is a real dense failure signal, but it still needs held-out gating. |
| `dare` | https://arxiv.org/abs/2311.03099 | Delta pruning/rescaling is useful only when the retained delta is not too large or noisy. |
| `mergeme` | https://arxiv.org/abs/2502.00997 | MoE merging must handle parameter interference and routing, not just average experts. |
| `expert_merging` | https://arxiv.org/abs/2509.25712 | Layer and chunk coefficients should be guided by unlabeled hidden/logit alignment rather than a fixed global merge weight. |
| `sub_moe` | https://arxiv.org/abs/2506.23266 | Expert output similarity/subspace structure is a better merge signal than tensor names alone. |
| `regmean++` | https://arxiv.org/abs/2508.03121 | Activation regression should account for intra- and cross-layer dependencies; layer-wise closed forms are useful diagnostics but can miss propagation effects. |
| `mergemoe` | https://arxiv.org/abs/2510.14436 | MoE expert merging can be formulated through output-space matching and optimization. |
| `namex` | https://arxiv.org/abs/2510.16138 | Expert weights should reflect cooperation/competition rather than a fixed uniform prior. |
| `harc` | https://arxiv.org/abs/2606.03391 | MoE router movement must be gated by top-k boundary stability, not treated like an ordinary dense tensor. |
| `router_kd_calibration` | https://arxiv.org/abs/2603.02217 | Expert edits create router-expert mismatch; router-only KD is a lightweight repair lever but still needs downstream acceptance gates. |
| `output_space_projection` | https://arxiv.org/abs/2605.29101 | Output residual projection provides a convex calibration objective and a captured-residual diagnostic for when output-space coefficients should improve a merge. |

## Outputs

- `results/unified_average_optimizer/mechanism_features.csv`
- `results/unified_average_optimizer/operation_decisions.csv`
- `results/unified_average_optimizer/mechanism_hypotheses.csv`
- `results/unified_average_optimizer/hypothesis_evidence_ledger.csv`
- `results/unified_average_optimizer/algorithm_contract.csv`
- `results/unified_average_optimizer/next_experiment_queue.csv`
- `results/unified_average_optimizer/algorithm.json`
- `results/unified_average_optimizer/summary.json`
- `results/unified_average_optimizer/report.md`
