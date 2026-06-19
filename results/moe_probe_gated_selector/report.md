# MoE Probe-Gated Selector

这份结果把真实 MoE gauge 反事实、Qwen3 expert correspondence 和 toy MoE route/capacity selector 合成一套同构 MoE average gate。它不是静态说某个算法最好，而是决定什么时候必须对齐 experts、什么时候可以暂用 identity、什么时候必须等 routing/load probe。

## Current Decision

- Global MoE gauge rule: `reject_same_name_average_without_alignment`
- Qwen3 Instruct/Coder expert identity: `identity_expert_average_allowed_with_routing_gate`
- Qwen3 preflight: `same_shape_identity_ready_wait_for_route_runtime`
- Qwen3 routing: `reject_direct_router_average_calibrate_or_freeze`
- Next blocking probe: `materialized_route_guarded_candidate_vllm_eval`
- Same-shape invariant: `No selector stage changes model class, tokenizer, hidden size, router shape, expert count, or tensor shape.`

## Evidence Cases

| case | decision | expert gate | router gate | capacity gate | evidence |
| --- | --- | --- | --- | --- | --- |
| `real_olmoe_gauge_selfmerge` | `reject_same_name_average_without_alignment` | `required` | `permute_router_rows_with_experts` | `not_tested_in_this_probe` | same-name NLL degradation=5.491; aligned degradation=0.000000; permutation recovery=16/16 |
| `qwen3_instruct_coder_cross_correspondence` | `identity_expert_average_allowed_with_routing_gate` | `identity_mapping_allowed` | `routing_probe_required_before_router_average` | `expert_load_probe_required` | identity layers=1.000; argmax identity=1.000; diag/offdiag cosine ratio=1339.5 |
| `qwen3_unified_preflight_contract` | `same_shape_identity_ready_wait_for_route_runtime` | `pass` | `real_route_probe_required` | `real_expert_load_probe_required` | status=same_shape_and_identity_ready_route_runtime_blocked_here; same_shape=True; routers=48; routed_expert_tensors=18432; identity=pass; cuda=False |
| `qwen3_real_route_load_readiness` | `reject_direct_router_average_calibrate_or_freeze` | `identity_mapping_allowed_but_route_weighted_expert_weights_required` | `calibrate_or_freeze_router_before_average` | `protect_high_load_experts_and_check_capacity` | status=high_risk_calibrate_router_before_merge; calibrate=493; small_lambda=46; passed=31; freeze=6; overused_expert=4459 |
| `toy_moe_method_selector` | `use_dispatch_aware_selector_not_static_average` | `expert_match_or_confidence_blend` | `calibrate_or_route_kd_with_overlap_guard` | `pass` | all_weight decision=reject_routing_breakdown; soft=expert_output_projection_router_calibrated_average; hard_top2=unified_confidence_blended_route_kd_seed_average; capacity=unified_moe_bias_capacity_average; capacity overflow=0.0475 |

## Selector Stages

| stage | gate | action if pass | action if fail |
| --- | --- | --- | --- |
| `topology` | same_shape_config_and_routed_expert_layout | continue | do_not_materialize |
| `expert_identity` | expert correspondence or gauge self-merge evidence | use identity mapping for Qwen3 pair | apply layerwise expert alias/remap before any expert averaging |
| `expert_weighting` | route-conditioned output or held-out task loss | confidence-blend searched and output-projection expert weights | freeze expert source or stay at endpoint/base |
| `router` | route overlap, top1 agreement, and router loss | small-step router calibration, route-KD, or freeze-router policy | freeze router and reject router averaging |
| `capacity` | top-k expert load overflow under serving dispatch | pass | router-bias capacity correction or reject sparse-serving candidate |
| `behavior_eval` | NLL plus generation/vLLM held-out tasks | materialize same-shape checkpoint | do_not_publish_candidate_as_average |

## Interpretation

真实 OLMoE 反事实说明 expert index 是 gauge，不是稳定语义；Qwen3 cross-correspondence 和 preflight 说明这对官方同族 checkpoint 目前 identity mapping 与同构合同可信，但真实 route/load probe 显示 router overlap 与 load 仍是独立失败模式。因此真实 Qwen3 MoE materialization 的下一步不是直接 average，而是基于 route/load 结果生成 route/category-aware expert weights，并选择 freeze-router、small-step calibration、route-KD 或 router-bias capacity correction，再进入 vLLM 行为评测。

## Files

- `results/moe_probe_gated_selector/selector_cases.csv`
- `results/moe_probe_gated_selector/selector_stages.csv`
- `results/moe_probe_gated_selector/summary.json`
