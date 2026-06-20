# Qwen3 MoE Mechanism-Gated vLLM Eval Gate

这份 gate 的目的不是静态宣布哪个算法最好，而是把每个内部机制变成可证伪的下游评测问题：router 是否应该冻结、shared attention delta 是否有用、route/load/category 信号是否真的能预测 expert 风险、0.65 tail trim 是否会伤害能力。

- Gate status: `awaiting_remote_vllm_eval`
- Local GPU available: `False` (`nvidia_smi_failed`)
- Source endpoints: `2`
- Same-shape candidates: `12`
- Ready-to-host rows: `12`
- Completed Qwen3 eval rows: `0`
- Current selection status: `awaiting_source_eval`
- Selected method: `None`

## Unified Rule

当前 unified average 不是一个固定的 `0.5/0.5` 公式，而是一个机制门控的同构输出规则：

```text
1. 先检查 same-shape 和 expert identity；identity 不成立就先 remap expert。
2. router overlap/load 风险高时先 freeze router；router calibration 只作为单独 ablation 进入。
3. routed experts 用 source-route-conditioned delta，并按 route/load/category/router-fragility 设 trust region。
4. shared attention 是否移动只看 trust-region vs expert-only 的同任务 vLLM 结果。
5. 0.65 tail trim 是否默认启用只看 tail-trimmed vs expert-only 的同任务 vLLM 结果。
6. hand-built risk penalties 是否保留，只看 searched no-gt-0.65 vs tail-trimmed 的同任务 vLLM 结果。
7. 如果所有候选被 source endpoint 支配，输出同构 endpoint/no-average。
```

一个简化的 expert 规则可以写成：

```text
theta_out[g] = theta_base[g] + s_g * w_g(source, route_mass, category) * (theta_source[g] - theta_base[g])
s_g = min(1, cap_g * ||theta_base[g]|| / ||w_g * (theta_source[g] - theta_base[g])||)
cap_g = f(route_load, category_specialization, router_fragility, delta_audit_tail)
```

这解释了为什么不能只靠某个静态算法名：Fisher/RegMean/TIES 这些方法给的是候选变换或局部解释，真正进入 unified 规则前必须通过同构、路由、delta audit 和下游任务门控。

## Mechanism Tests

| test | comparison | status | avg delta | worst delta | delta norm reduction | routed >0.75 reduction | question |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `source_control_floor` | `source_qwen3_30b_instruct` -> `source_qwen3_30b_coder` | `awaiting_eval` |  |  |  |  | How different are the source endpoints on the same downstream tasks? |
| `tail_delta_cap` | `qwen3_moe_unified_route_guarded_candidate` -> `qwen3_moe_audit_gated_candidate` | `awaiting_eval` |  |  | 0.022 | 675.000 | Does clipping extreme routed-expert deltas help? |
| `route_load_trust_region` | `qwen3_moe_audit_gated_candidate` -> `qwen3_moe_trust_region_candidate` | `awaiting_eval` |  |  | 0.015 | 150.000 | Do route/load/category/fragility probes identify the expert groups that need a tighter cap? |
| `shared_attention_ablation` | `qwen3_moe_trust_region_candidate` -> `qwen3_moe_expert_only_trust_region_candidate` | `awaiting_eval` |  |  | 0.003 | 0.000 | Should the unified MoE rule move shared attention, or keep it fixed? |
| `second_stage_tail_trim` | `qwen3_moe_expert_only_trust_region_candidate` -> `qwen3_moe_tail_trimmed_expert_only_candidate` | `awaiting_eval` |  |  | 0.003 | 14.000 | Does the stricter 0.65 routed-expert tail cap remove risk without removing ability? |
| `risk_penalty_simplification` | `qwen3_moe_tail_trimmed_expert_only_candidate` -> `qwen3_moe_searched_no_gt065_max_retention_candidate` | `awaiting_eval` |  |  | -0.004 | 0.000 | Are hand-built risk penalties necessary after a uniform 0.65 expert cap is enforced? |
| `layer_chunk_sensitivity` | `qwen3_moe_searched_no_gt065_max_retention_candidate` -> `qwen3_moe_layer_chunk_candidate` | `awaiting_eval` |  |  | 0.004 | 0.000 | Do importance-guided layer/chunk coefficients improve the unified MoE rule beyond a uniform expert cap? |
| `candidate_vs_sources` | `source_qwen3_30b_instruct` -> `qwen3_moe_unified_mechanism_candidate` | `awaiting_eval` |  |  |  |  | Does any same-shape candidate avoid Pareto domination by the two source endpoints? |
| `unified_mechanism_optimizer` | `qwen3_moe_layer_chunk_candidate` -> `qwen3_moe_unified_mechanism_candidate` | `awaiting_eval` |  |  | 0.003 | 0.000 | Does the router/evidence/geometry-risk optimizer improve downstream behavior beyond the layer/chunk candidate? |
| `expert_subspace_conflict_ablation` | `qwen3_moe_mechanistic_unified_candidate` -> `qwen3_moe_subspace_scaled_candidate` | `awaiting_eval` |  |  | -0.001 | 0.000 | Do uncovered high subspace-conflict experts need additional non-base shrink after the unified mechanism cap? |
| `mechanistic_unified_optimizer` | `qwen3_moe_unified_mechanism_candidate` -> `qwen3_moe_mechanistic_unified_candidate` | `awaiting_eval` |  |  | 0.002 | 0.000 | Does the benefit/curvature/interference objective explain a better scale law than the current risk-weighted cap search? |
| `router_coupled_boundary_ablation` | `qwen3_moe_mechanistic_unified_candidate` -> `qwen3_moe_router_coupled_candidate` | `awaiting_eval` |  |  |  |  | Does the layer-level router-boundary fragility signal justify extra expert shrink after the B/H/I scale law? |
| `harc_router_calibration_ablation` | `qwen3_moe_unified_mechanism_candidate` -> `qwen3_moe_harc_router_candidate` | `awaiting_eval` |  |  |  |  | Does HARC-style second-order router calibration recover downstream score beyond the frozen-router unified mechanism candidate? |

## Eval Gate Plan

| order | method | role | serve | eval | avg | worst | routed >0.75 | attention changed | mechanism |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `source_qwen3_30b_instruct` | `source` | `ready_to_host` | `not_run` |  |  |  |  | general/instruction endpoint |
| 1 | `source_qwen3_30b_coder` | `source` | `ready_to_host` | `not_run` |  |  |  |  | code endpoint |
| 2 | `qwen3_moe_unified_route_guarded_candidate` | `candidate` | `ready_to_host` | `not_run` |  |  | 839.000 | 288.000 | freeze router + route-conditioned expert weights + small attention step |
| 3 | `qwen3_moe_audit_gated_candidate` | `candidate` | `ready_to_host` | `not_run` |  |  | 164.000 | 288.000 | route-conditioned experts + file-level relative-delta cap |
| 4 | `qwen3_moe_trust_region_candidate` | `candidate` | `ready_to_host` | `not_run` |  |  | 14.000 | 288.000 | route/load/category/router-fragility trust-region caps |
| 5 | `qwen3_moe_expert_only_trust_region_candidate` | `candidate` | `ready_to_host` | `not_run` |  |  | 14.000 | 0.000 | trust-region experts + frozen shared attention + frozen router |
| 6 | `qwen3_moe_tail_trimmed_expert_only_candidate` | `candidate` | `ready_to_host` | `not_run` |  |  | 0.000 | 0.000 | expert-only + second-stage routed-expert tail cap at 0.65 |
| 7 | `qwen3_moe_searched_no_gt065_max_retention_candidate` | `candidate` | `ready_to_host` | `not_run` |  |  | 0.000 | 0.000 | freeze router/attention + source-route expert weights + searched uniform 0.65 cap |
| 8 | `qwen3_moe_layer_chunk_candidate` | `candidate` | `ready_to_host` | `not_run` |  |  | 0.000 | 0.000 | freeze router/attention + source-route expert weights + importance-guided layer/chunk coefficients |
| 9 | `qwen3_moe_unified_mechanism_candidate` | `candidate` | `ready_to_host` | `not_run` |  |  | 0.000 | 0.000 | mechanism-optimized same-shape MoE average with frozen router/attention and router/evidence/geometry-risk expert caps |
| 10 | `qwen3_moe_mechanistic_unified_candidate` | `candidate` | `ready_to_host` | `not_run` |  |  | 0.000 | 0.000 | damped per-expert benefit/curvature/interference optimizer with frozen router/attention |
| 11 | `qwen3_moe_subspace_scaled_candidate` | `candidate` | `ready_to_host` | `not_run` |  |  | 0.000 | 0.000 | unified mechanism candidate plus extra shrink for uncovered expert channel/chunk subspace conflicts |
| 12 | `qwen3_moe_router_coupled_candidate` | `candidate` | `checkpoint_missing_until_materialized` | `not_run` |  |  |  |  | mechanistic unified candidate plus extra shrink for high router-boundary-fragility expert groups |
| 13 | `qwen3_moe_harc_router_candidate` | `candidate` | `checkpoint_missing_until_harc_solver_delta` | `not_run` |  |  |  |  | unified mechanism candidate plus matrix-free HARC router calibration delta |

## Selection State

| method | eligible | dominated by source | avg | worst | gsm8k | mmlu | safety | humaneval | delta norm |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `source_qwen3_30b_instruct` | `False` | `` |  |  |  |  |  |  |  |
| `source_qwen3_30b_coder` | `False` | `` |  |  |  |  |  |  |  |
| `qwen3_moe_unified_route_guarded_candidate` | `False` | `` |  |  |  |  |  |  | 0.286 |
| `qwen3_moe_audit_gated_candidate` | `False` | `` |  |  |  |  |  |  | 0.264 |
| `qwen3_moe_trust_region_candidate` | `False` | `` |  |  |  |  |  |  | 0.249 |
| `qwen3_moe_expert_only_trust_region_candidate` | `False` | `` |  |  |  |  |  |  | 0.246 |
| `qwen3_moe_tail_trimmed_expert_only_candidate` | `False` | `` |  |  |  |  |  |  | 0.243 |
| `qwen3_moe_searched_no_gt065_max_retention_candidate` | `False` | `` |  |  |  |  |  |  | 0.248 |
| `qwen3_moe_layer_chunk_candidate` | `False` | `` |  |  |  |  |  |  | 0.243 |
| `qwen3_moe_unified_mechanism_candidate` | `False` | `` |  |  |  |  |  |  | 0.240 |
| `qwen3_moe_mechanistic_unified_candidate` | `False` | `` |  |  |  |  |  |  | 0.238 |
| `qwen3_moe_subspace_scaled_candidate` | `False` | `` |  |  |  |  |  |  | 0.240 |
| `qwen3_moe_router_coupled_candidate` | `False` | `` |  |  |  |  |  |  |  |
| `qwen3_moe_harc_router_candidate` | `False` | `` |  |  |  |  |  |  |  |

## How To Run On GPU

在 GPU host 上从仓库根目录运行：

```bash
results/qwen3_moe_mechanism_eval_gate/run_eval_gate.sh all
python scripts/build_qwen3_moe_mechanism_eval_gate.py
python scripts/collect_results.py
```

也可以只跑一个方法：

```bash
results/qwen3_moe_mechanism_eval_gate/run_eval_gate.sh qwen3_moe_tail_trimmed_expert_only_candidate
```

## Literature Hooks

- [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913): Treat loss surfaces as 2D slices through weight space; for this project the axes are task vectors or source deltas rather than random directions.
- [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026): If two checkpoints are connected by a low-loss path, averaging may work; if the straight line crosses a barrier, the selector should shrink or reject the merge.
- [Essentially No Barriers in Neural Network Energy Landscape](https://arxiv.org/abs/1803.00885): Nonlinear connectivity can exist even when straight-line averaging is poor, motivating path probes rather than only midpoint scores.
- [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482): Weight averaging is plausible when fine-tuned models sit in one low-error basin; the gate must test that assumption instead of assuming it.
- [Merging Models with Fisher-Weighted Averaging](https://arxiv.org/abs/2111.09832): Fisher/Laplace weighting gives a local quadratic explanation, but our Qwen dense probe shows local curvature can underpredict nonlocal barriers.
- [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708): Sign and magnitude conflicts are useful probes, but sparse conflict rules still need held-out/vLLM gates before touching broad LLM modules.
- [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836): Permutation symmetries explain why expert identity alignment must precede same-name averaging.
- [MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs](https://arxiv.org/abs/2502.00997): MoE merging needs explicit handling of parameter interference and routing, not only uniform expert averaging.
- [When Model Merging Breaks Routing: Training-Free Calibration for MoE](https://arxiv.org/abs/2606.03391): Router perturbations can cause routing breakdown; the current Qwen3 rule freezes router first and leaves router calibration as a separate ablation.

## Outputs

- `results/qwen3_moe_mechanism_eval_gate/eval_gate_plan.csv`
- `results/qwen3_moe_mechanism_eval_gate/mechanism_tests.csv`
- `results/qwen3_moe_mechanism_eval_gate/method_selection.csv`
- `results/qwen3_moe_mechanism_eval_gate/selection_rules.json`
- `results/qwen3_moe_mechanism_eval_gate/run_eval_gate.sh`
- `results/qwen3_moe_mechanism_eval_gate/literature_sources.json`
