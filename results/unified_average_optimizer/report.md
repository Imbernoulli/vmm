# Unified Average Optimizer

这个脚本把 Dense 和 MoE 的 probe 结果统一成同一个操作选择器：先测几何和对称性，再决定能不能平均、平均多少、哪些结构必须冻结或校准。它不是按算法名投票，而是让每个平均动作都绑定一个可测机制。

## Current Decision

- Dense: `avoid_linear_midpoint_use_probe_selected_anchor_or_low_lambda`；linear worst NLL `8.9477`，unified worst NLL `5.1830`。
- MoE: `align_experts_freeze_router_then_gate_candidate_by_vllm`；真实 OLMoE same-name average degradation `5.4910`，Qwen3 router action `freeze_router`。
- Qwen3 final selection: `awaiting_source_eval`，eligible `0/7`。

## Mechanism Features

| domain | probe | signal | value | threshold | evidence |
| --- | --- | --- | ---: | ---: | --- |
| `dense` | `curvature_law_general` | `high_nonlocal_barrier` | 42.8604 | 5.0000 | actual/predicted general degradation = 42.8604; uniform worst NLL = 5.9109; fisher worst NLL = 5.2492 |
| `dense` | `curvature_law_code` | `high_nonlocal_barrier` | 26.6575 | 5.0000 | actual/predicted code degradation = 26.6575 |
| `dense` | `heldout_unified_selector` | `allow_endpoint_or_anchor_fallback` | 5.1830 | 8.9477 | unified test worst NLL = 5.1830; linear = 8.9477; TIES = 9.1097; best endpoint = 5.1510 |
| `dense` | `generation_smoke` | `linear_generation_regression` | 0.0000 | 0.5000 | linear avg accuracy = 0.0000; unified avg accuracy = 0.5000; best smoke method = coder |
| `moe` | `controlled_expert_gauge` | `expert_permutation_is_function_preserving` | 0.0000 | 0.0000 | gauge-equivalent B MSE = 0.00000000; same-name worst = 0.5105; aligned worst = 0.1252 |
| `moe` | `real_olmoe_gauge_selfmerge` | `reject_same_name_average_without_alignment` | 5.4910 | 1.0000 | baseline NLL = 4.1678; same-name average NLL = 9.6588; aligned average NLL = 4.1678; layers recovered = 16/16 |
| `moe` | `qwen3_expert_identity` | `identity_alignment_is_allowed_for_this_pair` | 1.0000 | 1.0000 | identity-optimal layer fraction = 1.0000; argmax identity fraction = 1.0000 |
| `moe` | `qwen3_router_move_gate` | `freeze_router_or_train_route_kd_delta` | 0.0000 | 48.0000 | allowed router layers = 0/48; top-k Jaccard mean/min = 0.4539/0.2422; top1 agreement mean/min = 0.4125/0.0690 |
| `moe` | `qwen3_final_candidate_selection` | `await_matched_vllm_before_accepting_average` | 0.0000 | 7.0000 | status = awaiting_source_eval; eligible candidates = 0/7; reason = Both Qwen3 source endpoints must complete audited vLLM eval before final candidate selection. |

## Operations

| stage | operation | selected action | why |
| --- | --- | --- | --- |
| `dense_connectivity_gate` | `do_not_use_linear_midpoint_by_default` | search base-anchored coefficient family; current config = {"density": 1.0, "importance": "uniform", "lam": 0.0, "router": "average", "sign_resolve": true} | It prevents a fixed 0.5 midpoint from crossing a measured high-loss barrier. |
| `dense_sparse_coordinate_gate` | `make TIES/DARE-style sparsity conditional` | only materialize sparse conflict rules when held-out and vLLM gates pass | It keeps sign-conflict probes as diagnostics without letting them delete useful dense capacity. |
| `moe_expert_identity_gate` | `canonicalize expert gauge before averaging` | run layer-wise expert alignment; for Qwen3 Instruct/Coder the mapping is currently identity | It removes a discrete symmetry error before any continuous weight interpolation is attempted. |
| `moe_router_gate` | `freeze direct router movement` | freeze_router | It avoids averaging a discrete top-k dispatch boundary that has high measured source disagreement. |
| `moe_candidate_gate` | `select only after audited downstream eval` | keep all seven Qwen3 candidates provisional until eval-bundle audit passes | It prevents structural cleanliness from being mistaken for actual downstream dominance. |

## Literature Priors

| key | source | mechanism used here |
| --- | --- | --- |
| `mode_connectivity` | https://arxiv.org/abs/1802.10026 | A weight average is trusted only when the probed path stays in a low-loss basin. |
| `model_soups` | https://arxiv.org/abs/2203.05482 | Same-basin finetunes can average well, but endpoint fallback is part of the recipe. |
| `git_rebasin` | https://arxiv.org/abs/2209.04836 | Permutation symmetry must be canonicalized before weight-space merging. |
| `ties` | https://arxiv.org/abs/2306.01708 | Coordinate sign conflict is a real dense failure signal, but it still needs held-out gating. |
| `dare` | https://arxiv.org/abs/2311.03099 | Delta pruning/rescaling is useful only when the retained delta is not too large or noisy. |
| `mergeme` | https://arxiv.org/abs/2502.00997 | MoE merging must handle parameter interference and routing, not just average experts. |
| `sub_moe` | https://arxiv.org/abs/2506.23266 | Expert output similarity/subspace structure is a better merge signal than tensor names alone. |
| `mergemoe` | https://arxiv.org/abs/2510.14436 | MoE expert merging can be formulated through output-space matching and optimization. |
| `namex` | https://arxiv.org/abs/2510.16138 | Expert weights should reflect cooperation/competition rather than a fixed uniform prior. |

## Outputs

- `results/unified_average_optimizer/mechanism_features.csv`
- `results/unified_average_optimizer/operation_decisions.csv`
- `results/unified_average_optimizer/algorithm.json`
- `results/unified_average_optimizer/summary.json`
- `results/unified_average_optimizer/report.md`
