# Result Summary

Generated at: `2026-06-20T13:27:47.878472+00:00`

## Coverage

Complete: `94`; partial: `1`; missing: `0`.

| item | status | evidence |
| --- | --- | --- |
| 2D task-vector merge landscape | complete | Digits and CIFAR grid metrics plus merge landscape figures. |
| Per-task basin overlay | complete | results/digits_merge/figures/per_task_basin_overlay.png. |
| Task-arithmetic lambda sweep | complete | Digits, CIFAR, and Qwen path/lambda sweeps. |
| Merge-method overlay | complete | Digits method table and overlay cover average, task arithmetic, SLERP, TIES, DARE, TIES+DARE, Fisher, RegMean, layer-wise task arithmetic, and validation grid search. |
| Layer-wise interference atlas | complete | Digits, CIFAR, and pairwise single-digit conflict tables/figures. |
| One-class expert surrogate | complete | Ten single-digit experts and all 45 pairwise merges. |
| Randomness and alignment analysis | complete | Independent-initialization MLP path before/after Hungarian hidden-unit alignment. |
| Natural-image small-model case study | complete | CIFAR-10 vehicle/animal GroupNorm CNN merge landscape. |
| CLIP or ViT task-vector phase | complete | CIFAR100 ViT-style from-scratch transformer and ImageNet-pretrained ViT-B/16 frozen-backbone transfer merge studies are present. |
| Qwen-compatible LLM probe | complete | Safetensors probe and same-file smoke test. |
| Real Qwen LLM path sweep | complete | Qwen2.5-1.5B base-to-instruct path is evaluated with fixed NLL prompts plus GSM8K, MMLU, and HumanEval benchmark slices. |
| Multi-expert LLM merge | complete | Qwen2.5-0.5B base, Qwen2.5-0.5B-Instruct, and Qwen2.5-Coder-0.5B-Instruct are evaluated in a two-expert merge plane. |
| Dense curvature-displacement mechanism probe | complete | results/fp_curvature_law/report.md compares diagonal-Fisher second-order midpoint predictions against real Qwen instruct/coder interpolation loss. |
| Unified merge-family selector | complete | results/fp_merge_compare_dense/report.md evaluates a finite family containing linear average, task arithmetic, sign-elect, and magnitude-weighted variants, then selects by held-out worst-task NLL. |
| Dense exact-answer generation smoke | complete | results/fp_gen_eval_dense/report.md evaluates base, endpoints, linear average, and unified lambda=0 on built-in math/code-output generation tasks without executing model-generated code. |
| Qwen3 MoE generation-level downstream matrix | complete | results/fp_downstream_matrix/report.md compares official Qwen3 MoE parents, naive averages, and router-calibrated averages on MMLU/GSM8K/HumanEval generation tasks; it is auxiliary evidence, not the final vLLM selector. |
| Qwen3 MoE generation-level mechanism attribution | complete | results/fp_downstream_attribution/report.md attributes the generation matrix into naive-average regression, router-calibration recovery, and remaining source-frontier gap by task. |
| Qwen3 MoE generation confidence audit | complete | results/fp_downstream_confidence_audit/report.md adds Wilson aggregate uncertainty bounds to the generation matrix and shows router calibration is directional but not yet a confident source-frontier win. |
| Formal LLM benchmark slices | complete | Representative Qwen2.5-1.5B benchmark slices cover MMLU, GSM8K, HumanEval canonical-solution NLL, and BeaverTails safety/refusal NLL. |
| vLLM hosted downstream evaluation | partial | scripts/run_vllm_downstream_eval.py can build a served-model eval plan from the Qwen target registry; the generic registry run remains endpoint_unavailable, while checkpoint-specific hosted eval is tracked separately. |
| Materialized checkpoint vLLM hosted eval | complete | results/vllm_checkpoint_eval/qwen_0_5b_instruct_coder_uniform_average/report.md contains a real vLLM-hosted GSM8K/MMLU/safety/HumanEval compile eval for the materialized Qwen2.5-0.5B uniform-average checkpoint. |
| Qwen source-vs-merge vLLM comparison | complete | results/vllm_source_merge_comparison/report.md compares Qwen2.5-0.5B base/instruct/coder source endpoints against the materialized uniform-average checkpoint under the same vLLM downstream tasks. |
| Probe-guided dense average candidate vLLM eval | complete | results/probe_guided_dense_average_candidate/report.md selects a non-uniform Qwen instruct/coder bridge from the NLL grid, materializes the same-shape checkpoint locally, and records its real vLLM downstream eval. |
| Qwen dense module-wise guard ablation vLLM eval | complete | results/qwen_dense_module_guarded_candidate/report.md, results/qwen_dense_norm_guarded_candidate/report.md, and results/qwen_dense_selective_norm_guarded_candidate/report.md compare module-level, norm-only, and selective-norm tensor-rule variants against the global bridge under the same vLLM downstream tasks. |
| Qwen dense sparse-method candidate | complete | results/qwen_dense_sparse_method_candidate/report.md and results/qwen_dense_attention_sparse_method_candidate/report.md compare broad attention+MLP sparse rules against an attention-only sparse rule under real vLLM eval. |
| vLLM downstream eval contract smoke | complete | results/vllm_downstream_eval_smoke/smoke_report.md validates the OpenAI-compatible HTTP request, answer parsing, scoring, model ranking, and artifact writing path using a local mock endpoint. |
| vLLM checkpoint eval plan | complete | results/vllm_checkpoint_eval_plan/report.md turns same-shape checkpoint candidates into one-checkpoint-at-a-time vLLM serve/eval commands while keeping missing checkpoints separate from completed metrics. |
| Checkpoint materialization readiness audit | complete | results/checkpoint_materialization_readiness/report.md audits writer commands, placeholders, dry-run outputs, checkpoint existence, and vLLM eval readiness in one table. |
| MoE materialization pipeline plan | complete | results/moe_materialization_pipeline_plan/report.md connects Qwen MoE target selection, topology, routing probe, readiness, route weights, expert remap, router-bias deltas, checkpoint writer, and vLLM eval gates. |
| Probe-gated unified average plan | complete | results/probe_gated_unified_average_plan/report.md turns Dense vLLM ablations and toy MoE mechanism contrasts into a same-shape intervention gate rather than a static method ranking. |
| Probe-guided Average decision report | complete | results/average_decision_report/report.md converts merge grids, conflict probes, and optional MoE routing probes into same-shape average decisions. |
| Dense/MoE averaging literature matrix | complete | results/model_averaging_literature_review/report.md maps recent model averaging and MoE merging papers to probes, failure signals, and same-shape writer actions. |
| Average method gate matrix | complete | results/average_method_gate_matrix/report.md turns common Dense/MoE averaging method families into current-evidence accept/reject/conditional gates. |
| Average trust-region bounds | complete | results/average_trust_region_bounds/report.md converts Dense curvature failure, held-out lambda paths, MoE source-line barriers, router top-k margins, and routed expert caps into executable average-movement bounds. |
| Average connectivity diagnostic | complete | results/average_connectivity_diagnostic/report.md unifies Dense/MoE endpoint-frontier, midpoint, barrier, complementarity, and local-quadratic gates. |
| Average invariant audit | complete | results/average_invariant_audit/report.md converts model-averaging literature and current Dense/MoE probes into executable acceptance invariants and method gates. |
| Qwen target model registry | complete | results/qwen_target_model_registry/report.md maps representative official, third-party, downstream, and adapter-pool Qwen candidates to scenarios, eval slices, probes, and same-shape topology gates. |
| MoE same-shape averaging plan | complete | results/moe_average_plan/report.md maps router/expert probes into same-shape router, shared-module, expert, and adapter averaging actions. |
| Same-shape checkpoint writer | complete | scripts/write_same_shape_average_checkpoint.py writes same-shape safetensors checkpoints; results/same_shape_writer_smoke/report.md validates Qwen2.5-0.5B base/instruct/coder dry-run compatibility. |
| Dense sparse-method writer smoke | complete | results/dense_sparse_method_writer_smoke/report.md verifies coordinate-wise TIES-style trim/sign-elect/merge inside the same-shape checkpoint writer. |
| MoE tensor-rule writer materialization | complete | results/moe_tensor_rule_writer_smoke/report.md writes a tiny MoE-like safetensors checkpoint and verifies tensor-rule, freeze-router, router-bias additive deltas, full-tensor router deltas, and non-floating tensor behavior numerically. |
| MoE router delta calibration smoke | complete | results/moe_router_delta_calibration_smoke/report.md trains a same-shape router safetensors delta from hidden/router-logit cache, improving route KL and top-1 agreement under global/per-router cap-table relative-norm caps. |
| MoE router calibration cache smoke | complete | results/moe_router_calibration_cache_smoke/report.md captures student router hidden states and teacher router logits from forward hooks, then verifies the cache by training a same-shape router delta. |
| MoE combined writer smoke | complete | results/moe_combined_writer_smoke/report.md verifies expert tensor rules, source expert alias remap, freeze-router, and router-bias additive deltas in one same-shape writer call. |
| MoE packed-expert writer smoke | complete | results/moe_packed_expert_writer_smoke/report.md verifies first-dimension packed expert slice weights and source-expert remaps for Qwen-style packed MoE tensors. |
| MoE layer-wise expert remap smoke | complete | results/moe_layerwise_expert_remap_smoke/report.md verifies layer-scoped source tensor alias rules for real multi-layer MoE expert matching. |
| Checkpoint topology inspection | complete | results/checkpoint_topology_inspect/report.md inspects Qwen MoE/Dense configs and safetensors headers without loading weights. |
| Average candidate recipes | complete | results/average_candidate_recipes/report.md converts probe decisions into conservative same-shape materialization recipes and skips endpoint-only pseudo-averages. |
| MoE route-weight recipes | complete | results/moe_route_weight_recipes/report.md converts MoE routing/expert-load probes into tensor-rule files for same-shape checkpoint materialization; current recipe is waiting for real routing probe data. |
| MoE packed route-weight recipe smoke | complete | results/moe_packed_route_weight_recipe_smoke/report.md verifies route/expert weights can emit Qwen-style packed_expert_rules.csv with source-expert remap columns. |
| MoE router-bias additive capacity plan | complete | results/moe_router_bias_plan/report.md converts expert_load.csv into writer-ready router-bias additive deltas for same-shape capacity correction. |
| MoE confidence-blended router-bias capacity plan | complete | results/moe_confidence_blended_router_bias_plan/report.md applies the same writer-ready capacity correction to the confidence-blended unified MoE candidate. |
| MoE searched expert-weight recipes | complete | results/toy_moe_expert_weight_recipes/report.md converts calibration-searched per-expert source weights into same-shape checkpoint writer tensor rules. |
| MoE output-projection expert-weight recipes | complete | results/toy_moe_output_projection_recipes/report.md converts route-conditioned output-space expert weights into same-shape checkpoint writer tensor rules. |
| MoE confidence-blended expert-weight recipes | complete | results/toy_moe_confidence_blended_recipes/report.md converts projection-confidence-gated expert weights into same-shape checkpoint writer tensor rules. |
| MoE confidence-blended combined materialization recipe | complete | results/moe_confidence_blended_combined_recipe/report.md composes expert weights, expert alias remap, and router-bias capacity deltas into one same-shape writer command. |
| MoE routing readiness diagnostics | complete | results/moe_routing_readiness/report.md turns router_summary, route_overlap, and expert_load CSVs into router collapse, drift, boundary-fragility, and expert-load risk actions. |
| MoE routing probe CLI | complete | scripts/probe_moe_routing.py captures MoE router hooks and writes router_summary.csv, expert_load.csv, optional route_overlap.csv, summary.json, and report.md; results/moe_routing_probe_smoke/report.md validates the contract on a tiny local MoE. |
| MoE routing probe smoke | complete | results/moe_routing_probe_smoke/report.md proves the routing probe captures two tiny MoE gates and produces router, expert-load, token-route, comparison, and route-overlap CSVs. |
| Toy MoE route-aware merge | complete | results/toy_moe_merge/report.md runs a small same-shape MoE averaging experiment showing expert-index mismatch and expert-matched/router-calibrated fixes. |
| First-principles MoE mechanism probe | complete | results/fp_moe_mechanism/report.md isolates function-preserving expert permutation, expert alignment, router calibration, and Fisher ablations with real forward/backward passes. |
| Real MoE expert-gauge self-merge probe | complete | results/fp_moe_real_probe/report.md runs a function-preserving expert/router permutation on a real packed OLMoE checkpoint and shows same-name averaging fails unless expert identity is recovered. |
| MoE probe-gated selector | complete | results/moe_probe_gated_selector/report.md combines real OLMoE gauge evidence, Qwen3 expert correspondence, and toy route/capacity selection into a same-shape MoE average gate. |
| Qwen3 MoE unified average preflight | complete | results/moe_unified_preflight_qwen3_30b/report.md verifies Qwen3-30B Instruct/Coder same-shape config, router tensor contract, routed expert layout, expert identity gate, and the emitted real routing probe command. |
| Qwen3 MoE real routing readiness | complete | results/moe_routing_readiness/qwen3_30b_instruct_vs_coder/report.md analyzes the real Qwen3-30B Instruct/Coder route overlap and expert load probe, showing direct router averaging is high risk and needs calibration or freeze. |
| Qwen3 MoE route-guarded unified candidate | complete | results/qwen3_moe_unified_route_guarded_candidate/report.md converts the real Qwen3 route/load probe into source-route-conditioned same-shape tensor rules and a validated writer dry-run command. |
| Qwen3 MoE mechanism-gated vLLM eval gate | complete | results/qwen3_moe_mechanism_eval_gate/report.md turns two source endpoints and all registered same-shape Qwen3 MoE candidates into mechanism tests, a one-model-at-a-time vLLM run script, and endpoint-fallback selection rules. |
| Qwen3 MoE statistically powered vLLM eval budget | complete | results/qwen3_moe_eval_budget_plan/report.md raises the Qwen3 source/candidate vLLM run from a 64-example smoke floor to a Wilson/paired-test budgeted eval script; results/qwen3_moe_eval_budget_queue_smoke/report.md verifies the default final queue excludes ablation-only candidates. |
| Qwen3 MoE adaptive vLLM eval scheduler | complete | results/qwen3_moe_adaptive_eval_schedule/report.md turns the fixed Qwen3 MoE budget into a sequential source-control, mechanism-targeted probe-task, and full-budget escalation schedule; results/qwen3_moe_adaptive_eval_schedule_smoke/report.md covers source-missing, probe-selected, promising-escalation, full-ready, dominated-prune, coverage-selection, and task-selection branches. |
| Qwen3 MoE eval task manifest preflight | complete | results/qwen3_moe_eval_manifest_preflight/report.md checks that all budgeted source/candidate evals share one canonical task manifest and that the manifest contains the required task/example keys before vLLM runs. |
| Qwen3 MoE mechanism leverage map | complete | results/qwen3_moe_mechanism_levers/report.md ranks MoE-specific failure mechanisms, next experiments, and importance-guided layer/chunk calibration slots from real Qwen3 probes, including expert geometry and subspace conflict probes. |
| Qwen3 MoE expert geometry probe | complete | results/qwen3_moe_expert_geometry_probe/report.md reads 18,432 routed expert tensors from real Qwen3 Instruct/Coder safetensors and joins internal geometry risk with route/load context. |
| Qwen3 MoE expert subspace conflict probe | complete | results/qwen3_moe_expert_subspace_conflict_probe/report.md converts real expert channel/chunk geometry into subspace conflict gates and a candidate scale plan for uncovered high-risk experts. |
| Qwen3 MoE layer/chunk coefficient candidate | complete | results/qwen3_moe_layer_chunk_candidate/report.md converts the mechanism leverage layer scores into writer-ready same-shape tensor rules; results/qwen3_moe_layer_chunk_delta_audit/report.md verifies the materialized same-shape checkpoint. |
| Qwen3 MoE unified downstream result selector | complete | results/qwen3_moe_unified_result_selection/report.md gates the unified same-shape average against both Qwen3 source endpoints after matched vLLM eval; results/qwen3_moe_unified_result_selection_smoke/report.md covers candidate-win, source-dominance, task-regression, and no-gain branches. |
| Qwen3 MoE final candidate selector | complete | results/qwen3_moe_final_candidate_selection/report.md ranks all registered same-shape Qwen3 MoE candidates against both source endpoints after eval-bundle audit, with source-dominance, task-regression, score-confidence, paired-prediction, checkpoint-audit, and provisional-selection gates. |
| Qwen3 MoE candidate trust-region gate | complete | results/qwen3_moe_candidate_trust_region_gate/report.md marks old high-risk candidates as ablation-only and exposes only strict routed-expert trust-region candidates to final default selection. |
| Unified Dense/MoE average optimizer | complete | results/unified_average_optimizer/report.md converts Dense barrier probes, Dense/Qwen3 MoE straight-line connectivity, MoE gauge probes, Qwen3 expert identity, router movement, router margin fragility, router-only NLL calibration evidence, unified mechanism caps, router-calibration gating, and final candidate-selection gates into one same-shape operation policy. |
| Qwen3 MoE vLLM eval bundle audit | complete | results/qwen3_moe_eval_bundle_audit/report.md checks every Qwen3 source/candidate eval output for model-id, task-manifest sha, task, example-count, prediction, primary-score, and paired prediction-key consistency before selector use; results/qwen3_moe_eval_bundle_audit_smoke/report.md covers valid, stale-model, missing-task, low-example, key-mismatch, and manifest-mismatch bundles. |
| Qwen3 MoE mechanism effect attribution | complete | results/qwen3_moe_mechanism_effect_attribution/report.md decomposes the Qwen3 MoE source-frontier -> route-guarded -> audit-gated -> trust-region -> expert-only -> tail-trimmed -> searched-cap -> layer/chunk -> unified-mechanism chain into structural and downstream score deltas, gated by the eval-bundle audit. |
| Qwen3 MoE downstream feedback optimizer | complete | results/qwen3_moe_feedback_optimizer/report.md converts source-frontier task regressions from vLLM eval into bounded routed-expert rule updates; results/qwen3_moe_feedback_optimizer_smoke/report.md verifies code-regression restoration, non-code source-regression shrinkage, hard-cap enforcement, no-update awaiting-eval behavior, and eval-bundle-to-feedback integration. |
| Qwen3 MoE mechanistic unified candidate | complete | results/qwen3_moe_mechanistic_unified_candidate/report.md solves per-expert nonbase scale from benefit, curvature, and interference proxies, using real route mass, expert geometry, subspace conflict, delta pressure, and feedback priors; results/qwen3_moe_mechanistic_evidence_audit/report.md checks the B/H/I gradient, hard-cap binding, and internal-feature scale response; results/qwen3_moe_mechanistic_sensitivity/report.md reruns feature-family counterfactual full-score ablations to identify which internal signals protect the complete B/H/I objective; results/qwen3_moe_router_expert_coupling/report.md joins router top-k boundary fragility with expert scales to verify router-boundary risk becomes expert trust-region shrink; results/qwen3_moe_router_coupled_candidate/report.md materializes that coupling into a writer-ready ablation-only same-shape candidate; results/qwen3_moe_mechanistic_unified_candidate_smoke/report.md verifies monotonic mechanism behavior, hard-cap enforcement, and feedback shrink gating. |
| Qwen3 MoE post-vLLM eval refresh pipeline | complete | results/qwen3_moe_post_eval_refresh/report.md runs eval-bundle audit, unified/final selection, mechanism attribution, downstream feedback optimization, mechanistic unified candidate generation, mechanistic evidence audit, mechanistic sensitivity attribution, router-expert coupling attribution, router-coupled ablation candidate generation, unified average optimizer refresh, smoke checks, and collect_results in a fixed post-eval order after remote vLLM outputs land. |
| Qwen3 MoE searched cap-law materialized candidate | complete | results/qwen3_moe_searched_no_gt065_delta_audit/report.md verifies the materialized searched 0.65 cap-law checkpoint and adds it to the Qwen3 MoE eval gate. |
| Qwen3 MoE router move gate | complete | results/qwen3_moe_router_move_gate/report.md combines router tensor deltas with real routing readiness and rejects direct router-weight movement for all 48 layers. |
| Qwen3 MoE router margin fragility probe | complete | results/qwen3_moe_router_margin_fragility/report.md ranks router layers and prompt categories by top-k boundary fragility from real Qwen3 route margins, overlap, and router movement. |
| Qwen3 MoE router calibration NLL probe | complete | results/qwen3_moe_router_calibration_nll_probe/report.md formalizes the real Qwen3 router-only training probe, showing the averaged MoE improves when only router dispatch is recalibrated while keeping experts frozen. |
| Qwen3 MoE router calibration job | complete | results/qwen3_moe_router_calibration_job/report.md turns the rejected direct-router-move result into a margin-capped route-KD router-calibration sweep job and locks source/baseline/candidate vLLM evals to one task manifest. |
| Qwen3 MoE router calibration result selector | complete | results/qwen3_moe_router_calibration_selection/report.md accepts a router-calibrated cap only when matched vLLM eval, router-only tensor audit, top-k margin cap compliance, and source/baseline dominance gates pass. |
| Qwen3 MoE trust-region cap-law search | complete | results/qwen3_moe_trust_region_cap_search/report.md searches interpretable expert cap laws over real Qwen3 route-mass, risk-flag, and safetensors-delta probes and emits writer-ready next-candidate rules. |
| Qwen3 MoE unified mechanism candidate | complete | results/qwen3_moe_unified_mechanism_candidate/report.md turns route mass, router fragility, load, source-conflict, delta, expert geometry, and subspace-conflict probes into one same-shape constrained optimizer and writer-ready candidate. |
| Toy MoE multi-method routing readiness | complete | results/toy_moe_routing_readiness/report.md applies the generic readiness gate to toy MoE methods and flags all-weight routing drift separately from expert-matched/route-aware variants. |
| Toy MoE merge method selection | complete | results/toy_moe_method_selection/report.md combines method metrics, routing readiness, and sparse capacity overflow into materialization gates plus a hard-top2/overflow Pareto frontier. |
| Toy MoE expert remap plan | complete | results/toy_moe_expert_remap_plan/report.md turns expert-output matching into source tensor alias rules for same-shape checkpoint materialization. |
| Interactive explainer UI | complete | Dashboard includes a draggable precomputed merge-plane explorer with task-pair, method, objective, raw/normalized plane, alpha/beta, and lambda controls. |

## Key Metrics

| experiment | metric | value |
| --- | --- | ---: |
| digits merge | linear-average worst accuracy | 0.922 |
| digits merge | layer-wise task arithmetic worst accuracy | 0.928 |
| digits merge | RegMean linear-layer worst accuracy | 0.939 |
| digits merge | max grid worst accuracy | 0.961 |
| digits merge | global task-vector cosine | 0.138 |
| single-digit pairs | mean linear worst accuracy | 0.986 |
| single-digit pairs | weighted conflict vs drop Spearman | 0.165 |
| alignment | midpoint accuracy before to after | 0.944 to 0.971 |
| alignment | loss barrier before to after | 0.064 to 0.006 |
| CIFAR | linear-average worst accuracy | 0.249 |
| CIFAR | validation-grid best worst accuracy | 0.426 |
| CIFAR100 ViT-style | linear-average worst accuracy | 0.076 |
| CIFAR100 ViT-style | best method worst accuracy | 0.197 |
| pretrained ViT transfer | linear-average worst accuracy | 0.763 |
| pretrained ViT transfer | best method worst accuracy | 0.783 |
| Qwen path | best average-NLL lambda | 0.75 |
| Qwen path | instruction NLL at base to best | 3.612 to 1.811 |
| Qwen GSM8K slice | best strict exact match | 0.083 at lambda 0.75 |
| Qwen GSM8K slice | best loose exact match | 0.250 at lambda 0.75 |
| Qwen MMLU slice | best accuracy | 0.750 at lambda 0.75 |
| Qwen MMLU slice | best correct / total | 18/24 |
| Qwen HumanEval NLL slice | best solution NLL | 0.964 at lambda 1.00 |
| Qwen safety/refusal slice | best avg safety NLL | 2.546 at lambda 0.75 |
| Qwen multi-expert | best average-NLL method | instruct_expert (3.009) |
| Qwen multi-expert | linear-average avg / worst NLL | 5.591 / 9.553 |
| Qwen multi-expert | instruct/coder weighted conflict | 0.386 |
| dense curvature law | general actual / predicted degradation | 42.860 |
| dense curvature law | code actual / predicted degradation | 26.658 |
| dense curvature law | uniform / Fisher worst NLL | 5.911 / 5.249 |
| dense curvature law | top interference tensor | model.embed_tokens.weight |
| dense unified selector | selected lambda / test worst NLL | 0.00 / 5.183 |
| dense unified selector | linear / TIES worst delta vs unified | 3.765 / 3.927 |
| dense unified selector | unified minus best endpoint worst NLL | 0.032 |
| dense generation smoke | best method / linear avg accuracy | coder / 0.000 |
| dense generation smoke | unified avg delta vs linear | 0.500 |
| dense generation smoke | coder worst delta vs unified | 0.500 |
| Qwen3 MoE downstream generation matrix | best avg model / avg | instruct / 0.897 |
| Qwen3 MoE downstream generation matrix | Instruct+Coder avg -> +router-cal avg | 0.794 -> 0.828 |
| Qwen3 MoE downstream generation matrix | router-cal avg gain / HumanEval gain / gap to best parent | 0.033 / 0.075 / -0.069 |
| Qwen3 MoE downstream attribution | avg drop / router-cal recovery fraction / gap | 0.103 / 0.324 / -0.069 |
| Qwen3 MoE downstream attribution | HumanEval recovery / scores beating pair frontier | 0.500 / 0/5 |
| Qwen3 MoE downstream confidence | positive / confident-positive tasks vs naive | 2/3 / 0/3 |
| Qwen3 MoE downstream confidence | confident source-frontier wins / avg gain interval | 0/3 / [-0.170, 0.231] |
| first-principles MoE mechanism | gauge-equivalent B MSE | 7.66e-16 |
| first-principles MoE mechanism | router agreement raw to aligned | 0.035 to 0.795 |
| first-principles MoE mechanism | same-name to aligned worst loss | 0.511 to 0.125 |
| first-principles MoE mechanism | aligned + router-calibrated worst loss | 0.110 |
| first-principles MoE mechanism | Fisher worst-loss reduction after alignment | -0.027 |
| MoE probe-gated selector | global gauge rule | reject_same_name_average_without_alignment |
| MoE probe-gated selector | Qwen3 expert identity decision | identity_expert_average_allowed_with_routing_gate |
| MoE probe-gated selector | next blocking probe | materialized_route_guarded_candidate_vllm_eval |
| Qwen3 MoE unified preflight | status | same_shape_and_identity_ready_route_runtime_blocked_here |
| Qwen3 MoE unified preflight | same-shape / expert identity / CUDA | True / pass / False |
| Qwen3 MoE unified preflight | router / routed expert tensors | 48 / 18432 |
| Qwen3 MoE routing probe | prompts / routers / overlap rows | 12 / 48 / 576 |
| Qwen3 MoE routing probe | mean/min top-k Jaccard | 0.454 / 0.242 |
| Qwen3 MoE routing probe | mean/min top1 agreement | 0.413 / 0.069 |
| Qwen3 MoE routing readiness | status | high_risk_calibrate_router_before_merge |
| Qwen3 MoE routing readiness | calibrate / small-lambda / passed / freeze rows | 493 / 46 / 31 / 6 |
| Qwen3 MoE route-guarded candidate | status / frozen router | route_weight_rules_ready / True |
| Qwen3 MoE route-guarded candidate | expert rules / route rows used / skipped | 5243 / 35432 / 73728 |
| Qwen3 MoE route-guarded candidate | manifest expert / attention / router hits | 15729 / 288 / 48 |
| Qwen3 MoE materialized delta audit | status / changed tensors / router changed | passed / 10641 / 0/48 |
| Qwen3 MoE materialized delta audit | changed numel frac / relative delta norm / max abs delta | 0.563 / 0.286 / 1.688 |
| Qwen3 MoE audit-gated candidate | status / scaled rules / dry-run router hits | audit_gated_rules_ready / 302 / 48 |
| Qwen3 MoE audit-gated candidate | mean nonbase weight / max audited rel-delta / min scale | 0.201->0.193 / 1.327 / 0.565 |
| Qwen3 MoE audit-gated delta audit | status / total relative norm / router changed | passed / 0.264 / 0/48 |
| Qwen3 MoE audit-gated delta audit | max routed rel-delta / routed tensors >1.0 / >0.75 | 0.750 / 0 / 164 |
| Qwen3 MoE trust-region candidate | status / scaled rules / beyond delta cap | trust_region_rules_ready / 405 / 103 |
| Qwen3 MoE trust-region candidate | estimated total rel-norm / max routed rel-delta / >0.75 | 0.249 / 0.750 / 0 |
| Qwen3 MoE trust-region candidate | dry-run expert / attention / router hits | 15729 / 288 / 48 |
| Qwen3 MoE trust-region delta audit | status / total relative norm / router changed | passed / 0.249 / 0/48 |
| Qwen3 MoE trust-region delta audit | max routed rel-delta / routed tensors >1.0 / >0.75 | 0.750 / 0 / 14 |
| Qwen3 MoE trust-region delta validation | status / max abs pred error / p99 pred error | passed / 0.000 / 0.000 |
| Qwen3 MoE trust-region delta validation | tensors above tolerance / actual >0.75 / rounding slop | 0 / 14 / 14 |
| Qwen3 MoE expert-only ablation | status / attention rule / dry-run router hits | attention_ablation_rules_ready / freeze_shared_attention / 48 |
| Qwen3 MoE expert-only ablation | materialized / shards / dry-run router hits | True / 16 / 48 |
| Qwen3 MoE expert-only ablation | estimated rel-norm / reduction / attention energy frac | 0.246 / 0.003 / 0.021 |
| Qwen3 MoE expert-only ablation | frozen tensors / expert / attention hits | 3891 / 15729 / 288 |
| Qwen3 MoE expert-only delta audit | status / total relative norm / router changed | passed / 0.246 / 0/48 |
| Qwen3 MoE expert-only delta audit | max routed rel-delta / routed tensors >1.0 / >0.75 | 0.750 / 0 / 14 |
| Qwen3 MoE tail-trimmed expert-only candidate | status / target cap / scaled groups | tail_trimmed_rules_ready / 0.650 / 140 |
| Qwen3 MoE tail-trimmed expert-only candidate | estimated rel-norm / routed max / >0.65 | 0.243 / 0.650 / 0 |
| Qwen3 MoE tail-trimmed delta audit | status / total relative norm / router changed | passed / 0.243 / 0/48 |
| Qwen3 MoE tail-trimmed delta audit | max routed rel-delta / routed tensors >1.0 / >0.75 | 0.650 / 0 / 0 |
| Qwen3 MoE searched cap-law delta audit | status / total relative norm / router changed | passed / 0.248 / 0/48 |
| Qwen3 MoE searched cap-law delta audit | max routed rel-delta / >0.75 / >0.65 / >0.6505 | 0.650 / 0 / 245 / 0 |
| Qwen3 MoE layer/chunk delta audit | status / total relative norm / router changed | passed / 0.243 / 0/48 |
| Qwen3 MoE layer/chunk delta audit | max routed rel-delta / >0.75 / >0.65 / >0.6505 | 0.650 / 0 / 89 / 0 |
| Qwen3 MoE delta frontier | best safety candidate / next required gate | mechanistic_unified / vllm_downstream_eval_trust_region_vs_expert_only_tail_trimmed_vs_searched_cap_law_vs_layer_chunk_vs_unified_vs_mechanistic_vs_subspace_scaled |
| Qwen3 MoE delta frontier | structural dominated / mechanistic nearest / distance | 8 / unified_mechanism / 0.021 |
| Qwen3 MoE delta frontier | audit->trust routed >0.75 reduction / trust->expert-only routed >0.75 reduction | 150 / 0 |
| Qwen3 MoE delta frontier | trust vs expert-only total rel-norm / attention norm reduction | 0.249->0.246 / 0.189 |
| Qwen3 MoE delta frontier | expert-only->tail-trimmed rel-norm reduction / routed >0.65 reduction | 0.003 / 286 |
| Qwen3 MoE delta frontier | tail-trimmed vs searched rel-norm delta / >0.6505 counts | 0.004 / 0->0 |
| Qwen3 MoE delta frontier | searched->layer/chunk rel-norm reduction / >0.65 reduction / >0.6505 | 0.004 / 156 / 0 |
| Qwen3 MoE delta frontier | unified matches searched / unified rel-norm / router changed | False / 0.240 / 0 |
| Qwen3 MoE delta frontier | layer/chunk->unified rel-norm reduction / >0.65 reduction / unified >0.6505 | 0.003 / 89 / 0 |
| Qwen3 MoE delta frontier | unified->mechanistic rel-norm reduction / >0.65 delta / mechanistic >0.6505 | 0.002 / 0 / 0 |
| Qwen3 MoE delta frontier | mechanistic->subspace rel-norm delta / >0.65 reduction / subspace >0.6505 | 0.001 / 0 / 0 |
| Qwen3 MoE mechanism eval gate | status / selection / selected | awaiting_remote_vllm_eval / awaiting_source_eval / None |
| Qwen3 MoE mechanism eval gate | ready / completed / awaiting tests | 12 / 0 / 12 |
| Qwen3 MoE mechanism eval gate | local GPU / best delta-safety candidate | nvidia_smi_failed / mechanistic_unified |
| Qwen3 MoE mechanism eval gate | unified serve / audit / optimizer test | ready_to_host / True / awaiting_eval |
| Qwen3 MoE eval budget plan | status / current -> recommended examples | ready_for_budgeted_remote_vllm_eval / 64 -> 384 |
| Qwen3 MoE eval budget plan | planned / ready / pending methods | 15 / 12 / 2 |
| Qwen3 MoE eval budget plan | current / recommended / extra prompt budget | 3840 / 23040 / 19200 |
| Qwen3 MoE eval budget plan | ready current / recommended / extra prompt budget | 3072 / 18432 / 15360 |
| Qwen3 MoE eval budget plan | default queue / final methods / final prompts | final / 4 / 6144 |
| Qwen3 MoE eval budget plan | mechanism ablation methods / prompts | 9 / 13824 |
| Qwen3 MoE eval budget plan | Wilson n / paired n / capped tasks | 381 / 248 / humaneval_compile |
| Qwen3 MoE eval budget plan | task manifest aligned / canonical manifest | 15/15 / results/qwen3_moe_mechanism_eval_gate/task_manifest.json |
| Qwen3 MoE eval budget queue smoke | status / assertions | passed / 11/11 |
| Qwen3 MoE eval budget queue smoke | final / mechanism / router methods | 4 / 9 / 2 |
| Qwen3 MoE adaptive eval schedule | status / top action / top method | adaptive_schedule_ready / run_or_extend_source_control_probe / source_qwen3_30b_instruct |
| Qwen3 MoE adaptive eval schedule | source controls / round1 probes / probe->full examples | False / 6 / 64 -> 384 |
| Qwen3 MoE adaptive eval schedule | runnable methods / prompt budget / round1 probe prompts | 8 / 1664 / 1152 |
| Qwen3 MoE adaptive eval schedule | round1 policy / covered mechanism tests | greedy_mechanism_coverage_then_priority / 6 |
| Qwen3 MoE adaptive eval schedule | structural frontier / best structural method / score | True / qwen3_moe_mechanistic_unified_candidate / 0.993 |
| Qwen3 MoE adaptive eval schedule | structural dominance / frontier members / dominated methods | True / 2 / 8 |
| Qwen3 MoE adaptive eval schedule | paired gate status counts / alpha | {'awaiting_source_controls': 10, 'source_control': 2, 'checkpoint_missing': 2} / 0.050 |
| Qwen3 MoE adaptive eval schedule smoke | status / assertions | passed / 17/17 |
| Qwen3 MoE eval manifest preflight | status / tasks sufficient / methods aligned | task_manifest_ready / 4/4 / 15/15 |
| Qwen3 MoE eval budget plan | router active / ready / pending / plan-pruned caps | 2 / 0 / 2 / 2 |
| Qwen3 MoE mechanism levers | top lever / priority / next test | source_and_candidate_downstream_eval / 0.98 / results/qwen3_moe_eval_budget_plan/run_eval_budget.sh final |
| Qwen3 MoE mechanism levers | fine calibration layers / top layer score | 12,13,17,20,21,22,23,26 / 17:0.905 |
| Qwen3 MoE mechanism levers | expert geometry used / top geometry layer | True / 17:0.714 |
| Qwen3 MoE mechanism levers | expert subspace used / high / extra-scaled / top layer | True / 1323 / 17 / 17 |
| Qwen3 MoE expert geometry probe | projection tensors / experts / layers | 18432 / 6144 / 48 |
| Qwen3 MoE expert geometry probe | mean-p05 cosine / mean-p95 rel-delta | 0.386-0.118 / 1.062-1.275 |
| Qwen3 MoE expert geometry probe | high internal / route+geometry risk experts | 931 / 204 |
| Qwen3 MoE expert geometry probe | top layer / top expert risk | 17 / 13:104 (0.930) |
| Qwen3 MoE expert subspace conflict probe | projections / high / route-high / extra-scaled | 18432 / 1323 / 242 / 17 |
| Qwen3 MoE expert subspace conflict probe | top layer / max conflict / coder reduction / next action | 17 / 1.000 / 0.253078 / materialize_subspace_scaled_ablation_after_source_eval_budget |
| Qwen3 MoE expert subspace conflict probe | dry-run / floating / tensor-rule hits / frozen-router hits | True / 18867 / 15729 / 48 |
| Qwen3 MoE layer/chunk candidate | schedule / feasible schedules / changed groups | policy_095_098_100 / 3/15 / 2623 |
| Qwen3 MoE layer/chunk candidate | retention / risk delta reduction / max rel-delta | 0.985 / 0.021 / 0.650 |
| Qwen3 MoE layer/chunk candidate | dry-run / floating / frozen / tensor-rule hits | True / 18867 / 3891 / 15729 |
| Qwen3 MoE unified result selector | status / selected / reason | awaiting_source_eval / None / Both Qwen3 source endpoints must complete matched vLLM downstream eval before accepting an average. |
| Qwen3 MoE unified result selector | source complete / unified complete / eligible | False / False / 0/3 |
| Qwen3 MoE unified result selector smoke | status / passed cases | passed / 4/4 |
| Qwen3 MoE candidate trust-region gate | status / final-selectable / ablation-only | candidate_trust_region_gate_ready / 2/11 / 9 |
| Qwen3 MoE candidate trust-region gate | strict cap / selected methods | 0.650 / ['qwen3_moe_mechanistic_unified_candidate', 'qwen3_moe_subspace_scaled_candidate'] |
| Qwen3 MoE final candidate selector | status / selected / eligible | awaiting_source_eval / None / 0/11 |
| Qwen3 MoE final candidate selector | usable / complete / best source | 0/11 / False / None |
| Qwen3 MoE final candidate selector | uncertainty / paired gates / paired alpha | True / True / 0.050 |
| Qwen3 MoE final candidate selector | structural frontier / dominated / safety / tie tolerance | None / None / n/a / 0.000 |
| Qwen3 MoE final candidate selector | rank mode / confidence band / band size / point leader | None / True / 0 / None |
| Qwen3 MoE final candidate selector smoke | status / passed cases | passed / 11/11 |
| unified average optimizer | status / dense / MoE | built_waiting_for_qwen3_vllm_eval / avoid_linear_midpoint_use_probe_selected_anchor_or_low_lambda / align_experts_freeze_router_then_gate_candidate_by_vllm |
| unified average optimizer | hypotheses / queue / top experiment | 10 / 5 / budgeted_qwen3_moe_downstream_eval |
| unified average optimizer | evidence ledger / verdicts | 10 / {'awaiting_downstream_eval': 2, 'promising_but_unaccepted': 1, 'supports_conditional_action': 1, 'supports_current_action': 6} |
| unified average optimizer | contract status / passed / blocked | blocked_on_downstream_eval / 9/11 / ['downstream_source_dominance_gate', 'final_unified_average_acceptance'] |
| unified average optimizer ledger smoke | status / passed cases / assertions | passed / 5/5 / 29/29 |
| unified average optimizer | dense linear / unified / endpoint worst NLL | 8.948 / 5.183 / 5.151 |
| unified average optimizer | dense lambda midpoint / best-family worst NLL | 6.040 / 3.073 |
| unified average optimizer | real MoE gauge / router / Qwen3 final | 5.491 -> 0.000 / freeze_router / awaiting_source_eval (0/11) |
| unified average optimizer | Qwen3 unified candidate / subspace-delta / rule status | subspace_cap_s1.00 / 0.215 / fresh |
| unified average optimizer | Qwen3 unified audit norm / >0.65 / manifest max diff | 0.240 / 0 / 0.000 |
| unified average optimizer | final selector confidence band / rank mode / band size | True / None / 0 |
| unified average optimizer | router margin high layers / top / min safe-lambda | 24/48 / L17 0.752 / 0.020 |
| unified average optimizer | router-coupled frontier gate / pass / effect | direct_router_boundary_term_not_default / 146/770 / 0.0103 |
| unified average optimizer | Qwen3 MoE straight-line interior gap / general barrier | 0.119 / 0.110 |
| unified average optimizer | Qwen3 Base->Coder interior gap / complementary win | 0.106 / False |
| unified average optimizer | layer/chunk->unified norm / >0.65 reduction | 0.003 / 89 |
| unified average optimizer | router calibration status / eligible | awaiting_baseline_eval / 0/4 |
| unified average optimizer | router NLL probe worst reduction / code gap | 0.221 / -0.014 |
| Qwen3 MoE eval bundle audit | status / usable / invalid complete | awaiting_eval / 0/13 / 0 |
| Qwen3 MoE eval bundle audit | source usable / candidate usable / unified usable | 0/2 / 0/11 / False |
| Qwen3 MoE eval bundle audit | pairable sources / failed methods | 0 / 0 |
| Qwen3 MoE eval bundle audit smoke | status / passed cases | passed / 6/6 |
| Qwen3 MoE mechanism attribution | status / scored / regressions | awaiting_eval / 0/10 / 0 |
| Qwen3 MoE mechanism attribution | best avg / best worst transition | None / None |
| Qwen3 MoE mechanism attribution smoke | status / passed cases | passed / 3/3 |
| Qwen3 MoE feedback optimizer | status / scored tasks / regressions / changed groups | awaiting_eval / 0/4 / 0 / 0 |
| Qwen3 MoE feedback optimizer | candidate / base selection / frontier-dominated | qwen3_moe_unified_mechanism_candidate / auto_selected / False-True |
| Qwen3 MoE feedback optimizer | feedback base candidates considered | 1 |
| Qwen3 MoE feedback optimizer | materialization gate | do_not_materialize_feedback_candidate_yet |
| Qwen3 MoE feedback optimizer | nonbase ratio / max expected delta / hard-cap violations | 1.000 / 0.644 / 0 |
| Qwen3 MoE feedback optimizer smoke | status / passed cases | passed / 15/15 |
| Qwen3 MoE mechanistic unified candidate | selected / candidates / feedback | s0.08_b1.65_h0.75_i0.75 / 144 / awaiting_eval |
| Qwen3 MoE mechanistic unified candidate | nominal cap / effective cap / write margin | 0.650 / 0.649 / 0.001 |
| Qwen3 MoE mechanistic unified candidate | retention / max rel-delta / hard-cap violations | 0.965 / 0.649 / 0 |
| Qwen3 MoE mechanistic unified candidate | risk-delta / benefit-scale / loss proxy | 0.230 / 0.976 / 0.029 |
| Qwen3 MoE mechanistic unified candidate | writer manifest / dry-run / tensor-rule hits / freeze-router hits | True / False / 15729 / 48 |
| Qwen3 MoE mechanistic evidence audit | gradient agree / objective improved / hard-cap bound | 1.000 / 0.945 / 319 |
| Qwen3 MoE mechanistic evidence audit | dominant binding / suppressing features | cost_gradient_shrink / curvature_score, feature_router_instability, feature_expert_internal_geometry |
| Qwen3 MoE mechanistic sensitivity | strongest objective / delta / reselected | no_category_prior / 0.003 / s0.04_b1.15_h1.25_i0.75 |
| Qwen3 MoE mechanistic sensitivity | strongest scale / shift / top shrink feature | no_subspace_conflict / 0.0086 / feature_layer_geometry (0.698) |
| Qwen3 MoE router-expert coupling | gate / fragility->feature / fragility->shrink | router_expert_coupling_active / 0.695 / 0.583 |
| Qwen3 MoE router-expert coupling | high-low shrink lift / top layer / high-low scale | 0.0138 / L20 / 0.970-0.991 |
| Qwen3 MoE router-coupled candidate | gate / selected / changed groups | ablation_only_waiting_vllm / router_q0.75_s0.0100_cap0.0100 / 972 |
| Qwen3 MoE router-coupled candidate | retention delta / coupled delta reduction / risk reduction | -0.0031 / 0.0011 / 0.0009 |
| Qwen3 MoE router-coupled retention frontier | gate / constrained / stress | direct_router_boundary_term_not_default / router_q0.85_s0.00020_cap0.00010 / router_q0.75_s0.01000_cap0.01000 |
| Qwen3 MoE router-coupled retention frontier | pass default / effect fraction / action | 146/770 / 0.0103 / keep_router_fragility_inside_BHI_and_keep_direct_extra_shrink_as_ablation |
| Qwen3 MoE mechanistic unified smoke | status / passed cases | passed / 4/4 |
| Qwen3 MoE post-eval refresh | status / passed steps / audit usable | passed / 28/28 / 0/13 |
| Qwen3 MoE post-eval refresh | selection / final selection / attribution scored / plan steps | awaiting_source_eval / awaiting_source_eval / 0/10 / 28/28 |
| Qwen3 MoE post-eval refresh | feedback status / scored tasks / changed groups | awaiting_eval / 0/4 / 0 |
| Qwen3 MoE post-eval refresh | mechanistic status / retention / hard-cap violations | mechanistic_unified_candidate_ready / 0.965 / 0 |
| Qwen3 MoE post-eval refresh | sensitivity objective / scale | no_category_prior 0.003 / no_subspace_conflict 0.0086 |
| Qwen3 MoE post-eval refresh | router-expert coupling | router_expert_coupling_active / 0.695 / 0.583 / L20 |
| Qwen3 MoE post-eval refresh | router-coupled candidate | ablation_only_waiting_vllm / router_q0.75_s0.0100_cap0.0100 / -0.0031 / 0.0011 |
| Qwen3 MoE router move gate | status / action / allowed layers | router_move_rejected_freeze_router / freeze_router / 0/48 |
| Qwen3 MoE router move gate | unsafe / calibrate / freeze rows | 499 / 493 / 6 |
| Qwen3 MoE router move gate | router rel-norm / mean-min top-k Jaccard / min top1 | 0.739 / 0.454-0.242 / 0.069 |
| Qwen3 MoE router margin fragility | status / high-fragility layers / top layer | router_margin_fragility_rejects_direct_router_average / 24/48 / L17 |
| Qwen3 MoE router margin fragility | top score / min safe-lambda proxy / top category | 0.752 / 0.020 / long_context |
| Qwen3 MoE router calibration NLL probe | status / worst / avg reduction | router_calibration_improves_linear_merge_but_needs_downstream_gate / 0.221 / 0.161 |
| Qwen3 MoE router calibration NLL probe | code gap / worst gap to best source | -0.014 / 0.127 |
| Qwen3 MoE router calibration job | status / local GPU / candidates / stages | job_ready_awaiting_gpu / unavailable / 4 / 22 |
| Qwen3 MoE router calibration job | source controls / ready | 2 / True |
| Qwen3 MoE router calibration job | task manifest / create-if-missing | results/qwen3_moe_mechanism_eval_gate/task_manifest.json / True |
| Qwen3 MoE router calibration job | margin safe-lambda / planned-pass caps | 0.020 / 2/4 |
| Qwen3 MoE router calibration job | default-run caps | 2/4 |
| Qwen3 MoE router calibration job | margin-profile enabled / cap rows / min-mean-max | True / 48 / 0.020-0.045-0.050 |
| Qwen3 MoE router calibration job | inputs student / teacher / prompts | True / True / True |
| Qwen3 MoE router calibration selector | status / selected / eligible | awaiting_baseline_eval / None / 0/4 |
| Qwen3 MoE router calibration selector | source required-complete / baseline eval / candidate eval / audit | True-False / False / False / False |
| Qwen3 MoE router calibration selector | training / hard route-load / group validation | False / False / False |
| Qwen3 MoE router calibration selector | margin gate / safe-lambda / high layers | True / 0.020 / 24/48 |
| Qwen3 MoE router calibration selector | active / plan-pruned candidates | 2 / 2 |
| Qwen3 MoE router row-validation negative smoke | status / eligible / group validation | awaiting_router_calibration_eval / 0/3 / False |
| Qwen3 MoE router row-validation negative smoke | first decision reason | router_validation_not_group_heldout |
| Qwen3 MoE router source-dominance negative smoke | status / selected / eligible | keep_frozen_router_baseline / qwen3_moe_searched_no_gt065_max_retention_candidate / 0/3 |
| Qwen3 MoE router source-dominance negative smoke | first decision reason | source_endpoint_dominates |
| Qwen3 MoE router no-gain negative smoke | status / selected / eligible | keep_frozen_router_baseline / qwen3_moe_searched_no_gt065_max_retention_candidate / 0/3 |
| Qwen3 MoE router no-gain negative smoke | first decision reason | no_downstream_gain |
| Qwen3 MoE router task-regression negative smoke | status / selected / eligible | keep_frozen_router_baseline / qwen3_moe_searched_no_gt065_max_retention_candidate / 0/3 |
| Qwen3 MoE router task-regression negative smoke | first decision reason | task_score_regression |
| Qwen3 MoE router selector matrix smoke | status / passed cases | passed / 6/6 |
| Qwen3 MoE cap-law search | searched / frontier / expert groups | 432 / 88 / 5243 |
| Qwen3 MoE cap-law search | current trust vs uniform 0.65 retention | 0.982 / 0.982 |
| Qwen3 MoE cap-law search | current trust vs uniform 0.65 >0.65 groups | 129 / 0 |
| Qwen3 MoE cap-law search | extra risk penalties threshold-efficient | False |
| Qwen3 MoE cap-law search | validated dry-run rules / expert hits / router hits | 1 / 15729 / 48 |
| Qwen3 MoE unified mechanism candidate | selected / family / candidates | subspace_cap_s1.00 / subspace_weighted_cap / 28 |
| Qwen3 MoE unified mechanism candidate | retention / max rel-delta / hard-cap violations | 0.976 / 0.644 / 0 |
| Qwen3 MoE unified mechanism candidate | risk-delta / geometry-delta / subspace-delta | 0.225 / 0.218 / 0.215 |
| Qwen3 MoE unified mechanism candidate | geometry used / subspace used / high-subspace scale | True / True / 0.961 |
| Qwen3 MoE unified mechanism candidate | router / attention policy | freeze_router / freeze_shared_attention_pending_downstream_eval |
| Qwen3 MoE unified mechanism candidate | materialized rules / manifest match / max diff | fresh / True / 0.000 |
| Qwen3 MoE unified mechanism candidate | matches validated no-gt-0.65 rules / max diff | False / 0.062 |
| real MoE gauge self-merge | baseline / same-name / aligned NLL | 4.168 / 9.659 / 4.168 |
| real MoE gauge self-merge | same-name degradation vs baseline | 5.491 |
| real MoE gauge self-merge | recovered expert permutations | 16 / 16 |
| real Qwen3 MoE correspondence | identity-optimal layers / mean diag cosine | 1.000 / 0.183 |
| toy MoE route-aware merge | all-weight average worst accuracy | 0.545 |
| toy MoE route-aware merge | expert-matched average worst accuracy | 0.750 |
| toy MoE connectivity | best path / barrier | direct_matched_general_to_code / 0.000 |
| toy MoE connectivity | direct unmatched barrier | 0.034 |
| toy MoE connectivity | direct matched barrier | 0.000 |
| toy MoE route-aware merge | matched + router-frozen worst accuracy | 0.743 |
| toy MoE route-aware merge | expert-matched RegMean worst accuracy | 0.750 |
| toy MoE route-aware merge | expert-matched RegMean delta vs frozen | 0.007 |
| toy MoE route-aware merge | expert-matched TIES worst accuracy | 0.710 |
| toy MoE route-aware merge | expert-matched DARE worst accuracy | 0.733 |
| toy MoE route-aware merge | expert-matched TIES+DARE worst accuracy | 0.713 |
| toy MoE route-aware merge | best sparse expert delta vs matched average | -0.017 |
| toy MoE route-aware merge | guarded router-weight selected general/code | 0.00 / 1.00 |
| toy MoE route-aware merge | guarded router-weight eligible / total | 15 / 15 |
| toy MoE route-aware merge | matched + router-weight-search worst accuracy | 0.750 |
| toy MoE route-aware merge | matched + Hessian-router average worst accuracy | 0.750 |
| toy MoE route-aware merge | matched + Router-KD average worst accuracy | 0.745 |
| toy MoE route-aware merge | matched + route-KD average worst accuracy | 0.762 |
| toy MoE route-aware merge | matched + router-calibrated worst accuracy | 0.797 |
| toy MoE route-aware merge | matched + router-topk-calibrated worst accuracy | 0.755 |
| toy MoE hard dispatch | matched + router-calibrated hard top-1 worst accuracy | 0.608 |
| toy MoE hard dispatch | matched + router-calibrated hard top-2 worst accuracy | 0.665 |
| toy MoE hard dispatch | matched + router-topk-calibrated hard top-2 worst accuracy | 0.657 |
| toy MoE hard dispatch | matched + Hessian-router hard top-2 worst accuracy | 0.650 |
| toy MoE hard dispatch | matched + Router-KD hard top-2 worst accuracy | 0.660 |
| toy MoE hard dispatch | matched + route-KD hard top-2 worst accuracy | 0.685 |
| toy MoE hard dispatch | route-KD hard top-2 delta vs router-calibrated | 0.020 |
| toy MoE hard dispatch | route-KD hard top-2 delta vs output-KD | 0.025 |
| toy MoE unified objective | hard top-2 worst accuracy | 0.690 |
| toy MoE unified objective | hard top-2 delta vs route-KD | 0.005 |
| toy MoE unified objective | selected capacity loss coef | 0.000 |
| toy MoE unified objective | selected router seed | router_kd_seed |
| toy MoE unified objective | capacity-sweep candidates | 24 |
| toy MoE unified objective | capacity-sweep select score | 0.665 |
| toy MoE unified objective | capacity-sweep test score | 0.612 |
| toy MoE unified output-projection objective | worst accuracy | 0.795 |
| toy MoE unified output-projection objective | hard top-2 worst accuracy | 0.685 |
| toy MoE unified output-projection objective | delta vs unified hard top-2 | -0.005 |
| toy MoE unified output-projection objective | capacity-sweep select score | 0.665 |
| toy MoE unified output-projection objective | selected router seed | router_kd_seed |
| toy MoE bias capacity | selected capacity loss coef | 1.000 |
| toy MoE bias capacity | hard top-2 worst accuracy | 0.682 |
| toy MoE bias capacity | max top-k overflow fraction | 0.048 |
| toy MoE bias capacity | capacity-sweep test score | 0.635 |
| toy MoE capacity | max top-k overflow fraction | 0.106 |
| toy MoE capacity | worst overflow method/category | all_weight_average / code |
| toy MoE capacity | route-KD max top-k overflow fraction | 0.079 |
| toy MoE capacity | route-KD minus calibrated overflow | 0.025 |
| toy MoE unified objective | max top-k overflow fraction | 0.077 |
| toy MoE hard dispatch | soft to hard top-1 delta | -0.190 |
| toy MoE hard dispatch | top-k vs soft-calibrated hard top-2 delta | -0.008 |
| toy MoE route-aware merge | guarded router-sweep selected KL | 0.25 |
| toy MoE route-aware merge | guarded router-sweep eligible / total | 3 / 5 |
| toy MoE route-aware merge | router-sweep selected min top-k Jaccard | 0.837 |
| toy MoE route-aware merge | matched + router-sweep-selected worst accuracy | 0.797 |
| toy MoE route-aware merge | expert-weight search worst accuracy | 0.755 |
| toy MoE route-aware merge | expert-weight search + router-calibrated worst accuracy | 0.802 |
| toy MoE output projection | expert output-projection worst accuracy | 0.757 |
| toy MoE output projection | output-projection + router-calibrated worst accuracy | 0.807 |
| toy MoE output projection | mean captured output residual fraction | 0.616 |
| toy MoE output projection | delta vs matched-calibrated | 0.010 |
| toy MoE confidence-blended expert | router-calibrated worst accuracy | 0.805 |
| toy MoE confidence-blended expert | mean projection confidence | 0.616 |
| toy MoE unified objective | worst accuracy | 0.785 |
| toy MoE unified objective | delta vs expert-search router-calibrated | -0.017 |
| toy MoE unified objective | delta vs route-KD | 0.023 |
| toy MoE confidence-blended unified | worst accuracy | 0.790 |
| toy MoE confidence-blended unified | hard top-2 worst accuracy | 0.690 |
| toy MoE confidence-blended unified | max top-k overflow fraction | 0.076 |
| toy MoE confidence-blended unified | delta vs old unified | 0.005 |
| toy MoE unified output-projection bias-capacity | worst accuracy | 0.780 |
| toy MoE unified output-projection bias-capacity | delta vs output-projection unified | -0.015 |
| toy MoE unified output-projection bias-capacity | selected capacity-aware score | 0.667 |
| toy MoE confidence-blended bias-capacity | worst accuracy | 0.770 |
| toy MoE confidence-blended bias-capacity | selected capacity-aware score | 0.677 |
| toy MoE route-aware merge | route-aware average worst accuracy | 0.750 |
| toy MoE route-aware merge | matched + router-frozen minus all-weight worst accuracy | 0.198 |
| toy MoE route-aware merge | matched router calibration gain over frozen | 0.055 |
| toy MoE route-aware merge | Hessian-router delta vs expert-matched | 0.000 |
| toy MoE route-aware merge | Hessian-router delta vs router-calibrated | -0.047 |
| toy MoE route-aware merge | Router-KD delta vs expert-matched | -0.005 |
| toy MoE route-aware merge | Router-KD delta vs router-calibrated | -0.052 |
| toy MoE route-aware merge | route-KD delta vs Router-KD | 0.017 |
| toy MoE route-aware merge | route-KD delta vs router-calibrated | -0.035 |
| toy MoE route-aware merge | top-k router calibration delta vs soft calibration | -0.042 |
| toy MoE route-aware merge | expert search router-calibrated delta vs matched-calibrated | 0.005 |
| toy MoE route-aware merge | route-aware minus all-weight worst accuracy | 0.205 |
| toy MoE routing readiness | readiness status | high_risk_calibrate_router_before_merge |
| toy MoE routing readiness | all-weight calibrate-router flags | 1 |
| toy MoE method selection | recommended method | expert_output_projection_router_calibrated_average |
| toy MoE method selection | recommended hard top-2 method | unified_confidence_blended_route_kd_seed_average |
| toy MoE method selection | recommended hard top-2 worst accuracy | 0.693 |
| toy MoE method selection | capacity-aware hard top-2 method | unified_moe_bias_capacity_average |
| toy MoE method selection | capacity-aware top-k overflow | 0.048 |
| toy MoE method selection | hard top-2 / overflow Pareto methods | unified_confidence_blended_route_kd_seed_average, unified_confidence_blended_moe_average, unified_output_projection_moe_average, unified_moe_bias_capacity_average, unified_output_projection_bias_capacity_average, matched_router_kd_average |
| toy MoE method selection | all-weight decision | reject_routing_breakdown |
| toy MoE expert remap plan | remap status | ready |
| toy MoE expert remap plan | source tensor alias rules | 4 |
| toy MoE expert remap plan | layer-aware alias rules | 0 |
| toy MoE expert remap plan | min expert-output cosine | 0.941 |
| MoE layer-wise expert remap smoke | status | passed |
| MoE layer-wise expert remap smoke | alias / layer-aware / manual-review rows | 3 / 3 / 1 |
| vLLM hosted downstream eval | status | endpoint_unavailable |
| vLLM hosted downstream eval | queued served models | 7 |
| vLLM hosted downstream eval | candidate table | results/qwen_target_model_registry/model_registry.csv |
| vLLM downstream eval smoke | status | passed |
| vLLM downstream eval smoke | good / bad avg primary | 1.000 / 0.000 |
| vLLM checkpoint eval plan | status | hosted_eval_complete |
| vLLM checkpoint eval plan | ready / missing / not-loadable | 13 / 3 / 1 |
| vLLM checkpoint eval plan | unified serve / eval output | ready_to_host / results/vllm_checkpoint_eval/qwen3_moe_unified_mechanism_candidate |
| vLLM hosted eval results | completed eval dirs | 10 |
| vLLM hosted eval results | best eval avg / worst primary | source_qwen_0_5b_base / 0.375 / 0.094 |
| vLLM source-vs-merge comparison | status | merge_underperforms_all_sources |
| vLLM source-vs-merge comparison | best source / merge avg / delta | Qwen2.5-0.5B Base / 0.180 / -0.195 |
| vLLM source-vs-merge comparison | merge rank / source endpoints better | 4 / 3 |
| probe-guided dense average | selected alpha / beta | 0.250 / 1.000 |
| probe-guided dense average | vLLM avg / delta vs uniform / delta vs best source | 0.203 / 0.023 / -0.172 |
| Qwen dense guard probe | norm mean tensor cosine / sign conflict | -0.164 / 0.441 |
| Qwen dense guard ablation | module-guarded vLLM avg / delta vs global bridge | 0.160 / -0.043 |
| Qwen dense guard ablation | norm-only vLLM avg / delta vs global bridge | 0.203 / 0.000 |
| Qwen dense guard ablation | selective-norm vLLM avg / delta vs global bridge | 0.191 / -0.012 |
| Qwen dense broad sparse-method candidate | selected tensors / applied sparse rules / vLLM avg / delta vs global | 99 / 99 / 0.156 / -0.047 |
| Qwen dense attention sparse-method candidate | selected tensors / applied sparse rules / vLLM avg / delta vs global | 49 / 49 / 0.203 / 0.000 |
| checkpoint materialization readiness | status | hosted_eval_complete |
| checkpoint materialization readiness | materialized / blocked / ready / completed | 11 / 4 / 10 / 1 |
| checkpoint materialization readiness | unified writer / vLLM / end-to-end | materialized_checkpoint_exists / ready_to_host / ready_for_vllm_eval |
| MoE materialization pipeline | status | waiting_for_real_moe_probe_or_paths |
| MoE materialization pipeline | current blocking stage | exact_moe_topology |
| MoE materialization pipeline | ready / waiting gates | 3 / 6 |
| probe-gated unified average | dense default action | probe_guided_global_bridge_only |
| probe-gated unified average | dense bridge delta / module-guard delta | 0.023 / -0.043 |
| probe-gated unified average | MoE default action | expert_identity_plus_confidence_blended_expert_weights_plus_guarded_router_plus_capacity_gate |
| probe-gated unified average | MoE expert gain / overflow delta / real blocker | 0.205 / -0.029 / exact_moe_topology |
| Average decision report | avoid uniform average decisions | 3 |
| Average decision report | coefficient-search decisions | 2 |
| model averaging literature review | sources reviewed | 22 |
| model averaging literature review | method / probe / MoE-stage counts | 7 / 7 / 7 |
| average method gate matrix | accepted / rejected-default / conditional / active / required | 0 / 1 / 3 / 1 / 2 |
| average method gate matrix | dense midpoint / best-family / Qwen3 interior gap | 6.040 / 3.073 / 0.119 |
| average trust-region bounds | status / constraints / passed-rejected-waiting | trust_region_bounds_ready_waiting_vllm / 11 / 2-7-2 |
| average trust-region bounds | Dense lambda bound / safe uniform lambda / router safe lambda | 0.342 / 0.000 / 0.020 |
| average trust-region bounds | router midpoint over bound / mechanistic cap / selected max delta | 25.348 / 0.649 / 0.649 |
| average trust-region bounds smoke | status / assertions | passed / 11/11 |
| average connectivity diagnostic | path rejected / midpoint rejected / frontier wins | 5/6 / 5/6 / 1 |
| average connectivity diagnostic | Dense midpoint gap / Dense anchor gap / Qwen3 MoE gap | 2.917 / -0.101 / 0.119 |
| average invariant audit | invariants / hard blockers / default accepted methods | 10 / 4 / 0 |
| average invariant audit | same-shape / router allowed layers / final selector | True / 0/48 / awaiting_source_eval |
| average invariant audit | selected candidate / retention / predicted max delta | subspace_cap_s1.00 / 0.976 / 0.644 |
| Qwen target model registry | candidate dense / MoE models | 12 / 5 |
| Qwen target model registry | downstream or third-party candidates | 9 |
| Qwen target model registry | recommended first scenario | dense_7b_general_code_math_reasoning |
| Qwen target model registry | manual resolution or selection required | 6 |
| MoE routing probe smoke | routers / prompts | 2 / 3 |
| MoE routing probe smoke | router / expert / overlap rows | 6 / 24 / 6 |
| MoE average plan | router plan rows | 0 |
| MoE average plan | expert plan rows | 0 |
| same-shape writer smoke | Qwen-compatible tensors checked | 290 |
| dense sparse-method writer smoke | status | passed |
| dense sparse-method writer smoke | checked / failed tensors / method rules | 3 / 0 / 1 |
| MoE tensor-rule writer smoke | status | passed |
| MoE tensor-rule writer smoke | checked / failed tensors | 7 / 0 |
| MoE tensor-rule writer smoke | additive bias delta tensors / values | 1 / 2 |
| MoE tensor-rule writer smoke | safetensors tensor delta tensors / values | 1 / 4 |
| MoE router delta calibration smoke | status / routers / delta tensors | passed / 2 / 2 |
| MoE router delta calibration smoke | route KL initial-final / top1 initial-final / max rel delta | 0.2392-0.2153 / 0.4481-0.4870 / 0.0800 |
| MoE router delta calibration smoke | cap mode / min-mean-max cap / max utilization | per_router_table / 0.0200-0.0500-0.0800 / 1.0000 |
| MoE router delta calibration smoke | selection policy-split / selected epoch / score | capacity_aware-validation / 3.50 / 0.2168 |
| MoE router delta calibration smoke | train/selection samples / validation fraction | 307.0/77.0 / 0.201 |
| MoE router delta calibration smoke | train/validation groups | 0.0/0.0 |
| MoE router delta calibration smoke | train-validation KL / top1 gap | 0.2013-0.2153 / 0.5244-0.4870 / 0.0169/0.0473 |
| MoE router delta calibration smoke | hard top1/top-k overflow initial-final / increase | 0.0122-0.0122 / 0.0000-0.0000 / 0.0000/0.0000 |
| MoE router delta calibration smoke | hard top1/top-k max load initial-final | 0.3247-0.3247 / 0.2727-0.2727 |
| MoE router calibration cache smoke | status / ready routers / cache rows | passed / 2/2 / 192 |
| MoE router calibration cache smoke | materialization status / checked / failed | passed / 2 / 0 |
| MoE router calibration cache smoke | cache KL / trained KL initial-final / trained top1 initial-final | 0.0624 / 0.0474-0.0334 / 0.7292-0.8125 |
| MoE router calibration cache smoke | selection split / samples / groups | capacity_aware-group_validation / 72.0/24.0 / 3.0/1.0 |
| MoE combined writer smoke | status | passed |
| MoE combined writer smoke | checked / failed tensors | 7 / 0 |
| MoE combined writer smoke | alias rules / aliased tensors / additive values | 2 / 2 / 2 |
| MoE packed-expert writer smoke | status | passed |
| MoE packed-expert writer smoke | checked / failed tensors | 6 / 0 |
| MoE packed-expert writer smoke | packed rule tensors / slices / values | 2 / 3 / 5 |
| checkpoint topology | inspected MoE configs | 1 |
| checkpoint topology | primary real MoE source | qwen3_6_35b_a3b / weights=True |
| checkpoint topology | experts config / packed weights / routed expert bytes | 256 / 256 / 66035122176 |
| average candidate recipes | endpoint-only skips | 1 |
| average candidate recipes | MoE templates awaiting routing probe | 1 |
| MoE route-weight recipes | recipe status | waiting_for_routing_probe |
| MoE route-weight recipes | expert tensor rules | 0 |
| MoE packed route-weight recipe smoke | packed rules / tensors / slices | 12 / 4 / 6 |
| MoE packed route-weight recipe smoke | writer command uses packed CSV | True |
| MoE router-bias plan | status | router_bias_delta_ready |
| MoE router-bias plan | nonzero delta rows | 4 |
| MoE confidence-blended router-bias plan | status | router_bias_delta_ready |
| MoE confidence-blended router-bias plan | nonzero delta rows | 4 |
| MoE searched expert-weight recipes | recipe status | explicit_expert_weight_rules_ready |
| MoE searched expert-weight recipes | expert tensor rules | 4 |
| MoE output-projection expert-weight recipes | recipe status | explicit_expert_weight_rules_ready |
| MoE output-projection expert-weight recipes | expert tensor rules | 4 |
| MoE confidence-blended expert-weight recipes | recipe status | explicit_expert_weight_rules_ready |
| MoE confidence-blended expert-weight recipes | expert tensor rules | 4 |
| MoE confidence-blended combined recipe | status | combined_writer_command_ready |
| MoE confidence-blended combined recipe | tensor / alias / bias-delta rules | 5 / 4 / 4 |
| MoE routing readiness | readiness status | missing_router_summary |
| MoE routing readiness | router / expert risk rows | 0 / 0 |
