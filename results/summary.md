# Result Summary

Generated at: `2026-06-19T20:22:23.639609+00:00`

## Coverage

Complete: `60`; partial: `1`; missing: `0`.

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
| Qwen target model registry | complete | results/qwen_target_model_registry/report.md maps representative official, third-party, downstream, and adapter-pool Qwen candidates to scenarios, eval slices, probes, and same-shape topology gates. |
| MoE same-shape averaging plan | complete | results/moe_average_plan/report.md maps router/expert probes into same-shape router, shared-module, expert, and adapter averaging actions. |
| Same-shape checkpoint writer | complete | scripts/write_same_shape_average_checkpoint.py writes same-shape safetensors checkpoints; results/same_shape_writer_smoke/report.md validates Qwen2.5-0.5B base/instruct/coder dry-run compatibility. |
| Dense sparse-method writer smoke | complete | results/dense_sparse_method_writer_smoke/report.md verifies coordinate-wise TIES-style trim/sign-elect/merge inside the same-shape checkpoint writer. |
| MoE tensor-rule writer materialization | complete | results/moe_tensor_rule_writer_smoke/report.md writes a tiny MoE-like safetensors checkpoint and verifies tensor-rule, freeze-router, router-bias additive deltas, and non-floating tensor behavior numerically. |
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
| Qwen3 MoE delta frontier | best safety candidate / next required gate | expert_only / vllm_downstream_eval_trust_region_vs_expert_only_attention_ablation |
| Qwen3 MoE delta frontier | audit->trust routed >0.75 reduction / trust->expert-only routed >0.75 reduction | 150 / 0 |
| Qwen3 MoE delta frontier | trust vs expert-only total rel-norm / attention norm reduction | 0.249->0.246 / 0.189 |
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
| vLLM checkpoint eval plan | ready / missing / not-loadable | 7 / 2 / 1 |
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
| checkpoint materialization readiness | materialized / blocked / ready / completed | 5 / 4 / 4 / 1 |
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
