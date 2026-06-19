# Result Summary

Generated at: `2026-06-19T09:42:56.179458+00:00`

## Coverage

Complete: `29`; partial: `1`; missing: `0`.

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
| vLLM hosted downstream evaluation | partial | scripts/run_vllm_downstream_eval.py calls an OpenAI-compatible vLLM endpoint for GSM8K, MMLU, safety, and HumanEval compile slices; current result is endpoint_unavailable until a vLLM server is reachable. |
| Probe-guided Average decision report | complete | results/average_decision_report/report.md converts merge grids, conflict probes, and optional MoE routing probes into same-shape average decisions. |
| Dense/MoE averaging literature matrix | complete | results/model_averaging_literature_review/report.md maps recent model averaging and MoE merging papers to probes, failure signals, and same-shape writer actions. |
| Qwen target model registry | complete | results/qwen_target_model_registry/report.md maps representative official, third-party, downstream, and adapter-pool Qwen candidates to scenarios, eval slices, probes, and same-shape topology gates. |
| MoE same-shape averaging plan | complete | results/moe_average_plan/report.md maps router/expert probes into same-shape router, shared-module, expert, and adapter averaging actions. |
| Same-shape checkpoint writer | complete | scripts/write_same_shape_average_checkpoint.py writes same-shape safetensors checkpoints; results/same_shape_writer_smoke/report.md validates Qwen2.5-0.5B base/instruct/coder dry-run compatibility. |
| Checkpoint topology inspection | complete | results/checkpoint_topology_inspect/report.md inspects Qwen MoE/Dense configs and safetensors headers without loading weights. |
| Average candidate recipes | complete | results/average_candidate_recipes/report.md converts probe decisions into conservative same-shape materialization recipes and skips endpoint-only pseudo-averages. |
| MoE route-weight recipes | complete | results/moe_route_weight_recipes/report.md converts MoE routing/expert-load probes into tensor-rule files for same-shape checkpoint materialization; current recipe is waiting for real routing probe data. |
| MoE routing readiness diagnostics | complete | results/moe_routing_readiness/report.md turns router_summary, route_overlap, and expert_load CSVs into router collapse, drift, boundary-fragility, and expert-load risk actions. |
| MoE routing probe CLI | complete | scripts/probe_moe_routing.py captures MoE router hooks and writes router_summary.csv, expert_load.csv, optional route_overlap.csv, summary.json, and report.md; results/moe_routing_probe_smoke/report.md validates the contract on a tiny local MoE. |
| MoE routing probe smoke | complete | results/moe_routing_probe_smoke/report.md proves the routing probe captures two tiny MoE gates and produces router, expert-load, token-route, comparison, and route-overlap CSVs. |
| Toy MoE route-aware merge | complete | results/toy_moe_merge/report.md runs a small same-shape MoE averaging experiment showing expert-index mismatch and expert-matched/router-calibrated fixes. |
| Toy MoE multi-method routing readiness | complete | results/toy_moe_routing_readiness/report.md applies the generic readiness gate to toy MoE methods and flags all-weight routing drift separately from expert-matched/route-aware variants. |
| Toy MoE merge method selection | complete | results/toy_moe_method_selection/report.md combines method metrics and routing readiness to reject all-weight average and recommend matched router-calibrated averaging with router guard. |
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
| toy MoE route-aware merge | all-weight average worst accuracy | 0.620 |
| toy MoE route-aware merge | expert-matched average worst accuracy | 0.800 |
| toy MoE route-aware merge | matched + router-frozen worst accuracy | 0.787 |
| toy MoE route-aware merge | matched + router-calibrated worst accuracy | 0.838 |
| toy MoE route-aware merge | route-aware average worst accuracy | 0.790 |
| toy MoE route-aware merge | matched + router-frozen minus all-weight worst accuracy | 0.167 |
| toy MoE route-aware merge | matched router calibration gain over frozen | 0.050 |
| toy MoE route-aware merge | route-aware minus all-weight worst accuracy | 0.170 |
| toy MoE routing readiness | readiness status | high_risk_calibrate_router_before_merge |
| toy MoE routing readiness | all-weight calibrate-router flags | 1 |
| toy MoE method selection | recommended method | matched_router_calibrated_average |
| toy MoE method selection | all-weight decision | reject_routing_breakdown |
| toy MoE expert remap plan | remap status | ready |
| toy MoE expert remap plan | source tensor alias rules | 4 |
| toy MoE expert remap plan | min expert-output cosine | 0.943 |
| vLLM hosted downstream eval | status | endpoint_unavailable |
| Average decision report | avoid uniform average decisions | 3 |
| Average decision report | coefficient-search decisions | 2 |
| model averaging literature review | sources reviewed | 21 |
| model averaging literature review | method / probe / MoE-stage counts | 6 / 6 / 6 |
| Qwen target model registry | candidate dense / MoE models | 12 / 5 |
| Qwen target model registry | downstream or third-party candidates | 9 |
| Qwen target model registry | recommended first scenario | dense_7b_general_code_math_reasoning |
| Qwen target model registry | manual resolution or selection required | 6 |
| MoE routing probe smoke | routers / prompts | 2 / 3 |
| MoE routing probe smoke | router / expert / overlap rows | 6 / 24 / 6 |
| MoE average plan | router plan rows | 0 |
| MoE average plan | expert plan rows | 0 |
| same-shape writer smoke | Qwen-compatible tensors checked | 290 |
| checkpoint topology | inspected MoE configs | 1 |
| average candidate recipes | endpoint-only skips | 1 |
| average candidate recipes | MoE templates awaiting routing probe | 1 |
| MoE route-weight recipes | recipe status | waiting_for_routing_probe |
| MoE route-weight recipes | expert tensor rules | 0 |
| MoE routing readiness | readiness status | waiting_for_routing_probe |
| MoE routing readiness | router / expert risk rows | 0 / 0 |
