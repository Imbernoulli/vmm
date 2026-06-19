# Probe-Gated Unified Average Plan

这份结果不是按场景枚举“哪个算法最好”，而是把 probe 观察到的机制转成同构 checkpoint 的 gate。Dense 和 MoE 共用一个原则：只有当 probe 和 held-out eval 都支持某个 intervention 时，才把它变成默认写出规则。

## Current Decision

- Dense default: `probe_guided_global_bridge_only`
- MoE default: `expert_identity_plus_confidence_blended_expert_weights_plus_guarded_router_plus_capacity_gate`
- Real Qwen MoE blocker: `exact_moe_topology`
- Same-shape invariant: `No stage changes tokenizer, model class, tensor names, tensor shapes, router shape, or expert count.`

## Mechanism Contrasts

| domain | contrast | deltas | mechanism | action |
| --- | --- | ---: | --- | --- |
| Dense | `uniform_average_vs_best_source` | avg=-0.195, worst=-0.094, probe=n/a | midpoint ridge and endpoint skill mismatch | `reject_uniform_average` |
| Dense | `probe_guided_bridge_vs_uniform` | avg=0.023, worst=0.000, probe=1.921 | validation NLL selects a coder-anchored bridge away from the midpoint ridge | `keep_global_probe_guided_bridge_as_dense_baseline` |
| Dense | `aggressive_module_guard_vs_global_bridge` | avg=-0.043, worst=n/a, probe=n/a | module-level conflict is too coarse; freezing anchors and damping MLP removes useful adaptation | `reject_aggressive_module_guard` |
| Dense | `norm_only_guard_vs_global_bridge` | avg=0.000, worst=n/a, probe=n/a | normalization deltas shift task distribution rather than uniformly improving all tasks | `hold_norm_guard_for_targeted_ablation_only` |
| Dense | `selective_norm_guard_vs_global_bridge` | avg=-0.012, worst=n/a, probe=n/a | highest-conflict norm tensors are not sufficient causal levers | `reject_static_high_conflict_tensor_freeze` |
| MoE | `expert_identity_alignment` | soft=0.205, hard_top2=0.090, overflow=-0.039 | same-name expert tensors are semantically permuted | `apply_layerwise_expert_alias_or_matching_before_expert_average` |
| MoE | `router_calibration_after_matching` | soft=0.055, hard_top2=0.005, overflow=-0.014 | expert weights and router dispatch must be co-calibrated | `calibrate_or_distill_router_under_route_overlap_guard` |
| MoE | `route_conditioned_output_projection` | soft=0.005, hard_top2=0.005, overflow=-0.001 | expert source weights should explain routed output residuals, not only parameter deltas | `use_projection_when_captured_fraction_is_high` |
| MoE | `projection_confidence_blend` | soft=-0.005, hard_top2=0.005, overflow=0.006 | projection is reliable for some experts and over-moves others | `blend_projection_with_search_using_expert_captured_fraction` |
| MoE | `capacity_bias_correction` | soft=-0.020, hard_top2=-0.010, overflow=-0.029 | top-k capacity overflow is a separate failure mode from task loss | `apply_router_bias_delta_when_overflow_is_above_capacity_budget` |

## Unified Gate

| stage | decision | same-shape action | evidence |
| --- | --- | --- | --- |
| `dense_global_coefficients` | `enabled_for_dense_baseline` | write base + 0.25 * instruct_delta + 1.0 * coder_delta | bridge avg primary delta vs uniform = 0.023 |
| `dense_module_freeze_guard` | `rejected_as_default` | do not freeze full embedding/norm groups or damp all MLP tensors by default | module-guard delta vs bridge = -0.043 |
| `dense_static_tensor_freeze` | `rejected_as_default` | only use static tensor freezes inside a scored ablation sweep | selective-norm delta vs bridge = -0.012 |
| `moe_expert_identity` | `required_for_moe` | generate layer-scoped source_tensor_aliases before expert averaging | expert matching soft worst-acc gain = 0.205 |
| `moe_expert_weights` | `enabled_for_moe` | blend search weights and output-projection weights per expert using captured_fraction | captured_fraction min/mean/max = 0.238/0.616/0.957 |
| `moe_router` | `guarded_calibration_required` | freeze, KD-calibrate, or small-lambda calibrate router only after route overlap checks | matched router calibration soft worst-acc gain = 0.055 |
| `moe_capacity` | `capacity_gate_not_unconditional` | write router-bias deltas only when overflow reduction is worth the accuracy cost | confidence-blended bias-capacity overflow delta = -0.029 |
| `real_qwen_moe_materialization` | `blocked_until_real_probe` | do not emit final Qwen MoE writer command until exact headers and real routing probe exist | current blocking stage = exact_moe_topology |

## Files

- `results/probe_gated_unified_average_plan/dense_mechanism_contrasts.csv`
- `results/probe_gated_unified_average_plan/moe_mechanism_contrasts.csv`
- `results/probe_gated_unified_average_plan/intervention_plan.csv`
- `results/probe_gated_unified_average_plan/summary.json`
