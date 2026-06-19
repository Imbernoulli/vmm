# Result Summary

Generated at: `2026-06-19T12:55:03.894128+00:00`

## Coverage

Complete: `32`; partial: `1`; missing: `0`.

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
| Formal LLM benchmark slices | complete | Representative Qwen2.5-1.5B benchmark slices cover MMLU, GSM8K, HumanEval canonical-solution NLL, and BeaverTails safety/refusal NLL. |
| vLLM hosted downstream evaluation | partial | scripts/run_vllm_downstream_eval.py can build a served-model eval plan from the Qwen target registry and evaluate those ids through an OpenAI-compatible vLLM endpoint on GSM8K, MMLU, safety, and HumanEval compile slices; current result is endpoint_unavailable until a vLLM server is reachable. |
| Probe-guided Average decision report | complete | results/average_decision_report/report.md converts merge grids, conflict probes, and optional MoE routing probes into same-shape average decisions. |
| Dense/MoE averaging literature matrix | complete | results/model_averaging_literature_review/report.md maps recent model averaging and MoE merging papers to probes, failure signals, and same-shape writer actions. |
| Qwen target model registry | complete | results/qwen_target_model_registry/report.md maps representative official, third-party, downstream, and adapter-pool Qwen candidates to scenarios, eval slices, probes, and same-shape topology gates. |
| MoE same-shape averaging plan | complete | results/moe_average_plan/report.md maps router/expert probes into same-shape router, shared-module, expert, and adapter averaging actions. |
| Same-shape checkpoint writer | complete | scripts/write_same_shape_average_checkpoint.py writes same-shape safetensors checkpoints; results/same_shape_writer_smoke/report.md validates Qwen2.5-0.5B base/instruct/coder dry-run compatibility. |
| MoE tensor-rule writer materialization | complete | results/moe_tensor_rule_writer_smoke/report.md writes a tiny MoE-like safetensors checkpoint and verifies tensor-rule, freeze-router, router-bias additive deltas, and non-floating tensor behavior numerically. |
| Checkpoint topology inspection | complete | results/checkpoint_topology_inspect/report.md inspects Qwen MoE/Dense configs and safetensors headers without loading weights. |
| Average candidate recipes | complete | results/average_candidate_recipes/report.md converts probe decisions into conservative same-shape materialization recipes and skips endpoint-only pseudo-averages. |
| MoE route-weight recipes | complete | results/moe_route_weight_recipes/report.md converts MoE routing/expert-load probes into tensor-rule files for same-shape checkpoint materialization; current recipe is waiting for real routing probe data. |
| MoE router-bias additive capacity plan | complete | results/moe_router_bias_plan/report.md converts expert_load.csv into writer-ready router-bias additive deltas for same-shape capacity correction. |
| MoE searched expert-weight recipes | complete | results/toy_moe_expert_weight_recipes/report.md converts calibration-searched per-expert source weights into same-shape checkpoint writer tensor rules. |
| MoE routing readiness diagnostics | complete | results/moe_routing_readiness/report.md turns router_summary, route_overlap, and expert_load CSVs into router collapse, drift, boundary-fragility, and expert-load risk actions. |
| MoE routing probe CLI | complete | scripts/probe_moe_routing.py captures MoE router hooks and writes router_summary.csv, expert_load.csv, optional route_overlap.csv, summary.json, and report.md; results/moe_routing_probe_smoke/report.md validates the contract on a tiny local MoE. |
| MoE routing probe smoke | complete | results/moe_routing_probe_smoke/report.md proves the routing probe captures two tiny MoE gates and produces router, expert-load, token-route, comparison, and route-overlap CSVs. |
| Toy MoE route-aware merge | complete | results/toy_moe_merge/report.md runs a small same-shape MoE averaging experiment showing expert-index mismatch and expert-matched/router-calibrated fixes. |
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
| toy MoE unified objective | worst accuracy | 0.785 |
| toy MoE unified objective | delta vs expert-search router-calibrated | -0.017 |
| toy MoE unified objective | delta vs route-KD | 0.023 |
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
| toy MoE method selection | recommended hard top-2 method | unified_moe_average |
| toy MoE method selection | recommended hard top-2 worst accuracy | 0.690 |
| toy MoE method selection | capacity-aware hard top-2 method | unified_moe_bias_capacity_average |
| toy MoE method selection | capacity-aware top-k overflow | 0.048 |
| toy MoE method selection | hard top-2 / overflow Pareto methods | unified_moe_average, unified_moe_bias_capacity_average, matched_router_kd_average |
| toy MoE method selection | all-weight decision | reject_routing_breakdown |
| toy MoE expert remap plan | remap status | ready |
| toy MoE expert remap plan | source tensor alias rules | 4 |
| toy MoE expert remap plan | min expert-output cosine | 0.943 |
| vLLM hosted downstream eval | status | endpoint_unavailable |
| vLLM hosted downstream eval | queued served models | 7 |
| vLLM hosted downstream eval | candidate table | results/qwen_target_model_registry/model_registry.csv |
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
| MoE tensor-rule writer smoke | status | passed |
| MoE tensor-rule writer smoke | checked / failed tensors | 7 / 0 |
| MoE tensor-rule writer smoke | additive bias delta tensors / values | 1 / 2 |
| checkpoint topology | inspected MoE configs | 1 |
| average candidate recipes | endpoint-only skips | 1 |
| average candidate recipes | MoE templates awaiting routing probe | 1 |
| MoE route-weight recipes | recipe status | waiting_for_routing_probe |
| MoE route-weight recipes | expert tensor rules | 0 |
| MoE router-bias plan | status | router_bias_delta_ready |
| MoE router-bias plan | nonzero delta rows | 4 |
| MoE searched expert-weight recipes | recipe status | explicit_expert_weight_rules_ready |
| MoE searched expert-weight recipes | expert tensor rules | 4 |
| MoE routing readiness | readiness status | waiting_for_routing_probe |
| MoE routing readiness | router / expert risk rows | 0 / 0 |
