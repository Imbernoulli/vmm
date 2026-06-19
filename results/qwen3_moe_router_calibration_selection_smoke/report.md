# Qwen3 MoE Router Calibration Selection

这一步只解决一个问题：在 frozen-router 的 `searched_no_gt065` 基线之上，是否应该加入一个小的 route-KD router delta。结论不会按算法名决定，而是按机制证据决定：下游任务、router-only 审计、delta cap、source endpoint 支配关系必须同时通过。

- Selection status: `selected_router_calibrated_candidate`
- Selected method: `smoke_router_calibrated_cap0025`
- Reason: Selected the router-calibrated cap that improved downstream scores, stayed within cap, changed only router tensors, and was not source dominated.
- Baseline eval completed: `True`
- Source eval required: `True`
- Source eval completed: `True`
- Candidate eval completed: `True`
- Audit completed: `True`
- Training completed: `True`
- Capacity metrics completed: `True`
- Eligible candidates: `1/3`

## Baseline

| eval dir | status | avg | worst | gsm8k | mmlu | safety | humaneval |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `results/qwen3_moe_router_calibration_selection_smoke/input_job/eval_baseline` | `complete` | 0.5000 | 0.3000 | 0.4200 | 0.5300 | 0.6200 | 0.3000 |

## Candidate Gate

| cap | method | split | selected epoch | KL gap | top1 drop | decision | avg delta | worst delta | worst task delta | router max rel | top1/top-k overflow | top1/top-k increase | load pass | gen pass | router-only | cap pass | score | reason |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- |
| 0.0100 | `smoke_router_calibrated_cap001` | `validation` | 10.0 | 0.0200 | 0.0200 | `reject_or_wait` | 0.0010 | 0.0000 | -0.0010 | 0.0080 | 0.0000/0.0000 | 0.0000/0.0000 | `True` | `True` | `True` | `True` | -0.0018 | `no_downstream_gain` |
| 0.0250 | `smoke_router_calibrated_cap0025` | `validation` | 10.0 | 0.0200 | 0.0200 | `candidate_eligible` | 0.0150 | 0.0100 | 0.0000 | 0.0220 | 0.0150/0.0200 | 0.0050/0.0150 | `True` | `True` | `True` | `True` | 0.0055 | `passes_all_gates` |
| 0.0500 | `smoke_router_calibrated_cap005` | `validation` | 10.0 | 0.0200 | 0.0200 | `reject_or_wait` | 0.0250 | 0.0150 | 0.0000 | 0.0710 | 0.0900/0.0400 | 0.0600/0.0250 | `False` | `True` | `False` | `False` | -0.0055 | `top1_capacity_overflow_increase,audit_not_router_only,router_delta_cap_violation` |

## Source Controls

| eval dir | status | avg | worst |
| --- | --- | ---: | ---: |
| `results/qwen3_moe_router_calibration_selection_smoke/input_job/eval_source_instruct` | `complete` | 0.4900 | 0.2800 |
| `results/qwen3_moe_router_calibration_selection_smoke/input_job/eval_source_coder` | `complete` | 0.4700 | 0.2600 |

## Unified Rule Update

如果 selection status 是 `selected_router_calibrated_candidate`，统一方法可以在 `searched_no_gt065` expert/attention 冻结策略后追加该 cap 的 router delta；否则 unified 默认继续保持 frozen router。这样算法不会在 router 机制证据不足时强行动 router。

## Decision Rules

- Baseline searched_no_gt065 eval must be complete on the same vLLM task set.
- Both source endpoint evals must be complete unless --allow-missing-source-eval is explicitly set.
- Every cap candidate must have a materialized delta audit and vLLM eval before final selection.
- Every cap candidate must have router training metrics with hard top-1/top-k route-load statistics.
- The audit must show only router tensors changed, with no shape/dtype mismatch.
- The maximum per-router relative delta norm must stay inside the planned cap.
- Hard top-1 route capacity overflow may not exceed 0.1.
- Hard top-k route capacity overflow may not exceed 0.05.
- Hard top-1 route capacity overflow may not increase over the frozen-router start by more than 0.05.
- Hard top-k route capacity overflow may not increase over the frozen-router start by more than 0.02.
- Validation route-KL gap over train route-KL may not exceed 0.2.
- Validation top-1 agreement drop from train top-1 agreement may not exceed 0.2.
- Average primary score may not drop more than 0.005.
- Worst primary score may not drop more than 0.01.
- No available task primary score may drop more than 0.02.
- At least one downstream primary/task score must improve by 0.002 or more.
- A candidate is rejected when a source endpoint dominates it on all available scores.

## Literature Hooks

- [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482): Weight averaging is only justified when checkpoints behave as if they are in one low-error basin; the selector therefore rejects router deltas that do not improve downstream scores over the frozen-router baseline.
- [Merging Models with Fisher-Weighted Averaging](https://arxiv.org/abs/2111.09832): A local quadratic view motivates small trust-region updates, but the actual acceptance criterion is still held-out downstream behavior.
- [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708): Interference is treated as a measurable signal: the router calibration delta must be sparse in module scope and must not introduce non-router changes.
- [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836): Expert identity and permutation alignment remain upstream gates; this selector only decides whether a small router delta should be added after the frozen-router expert candidate.
- [What Matters for Model Merging at Scale?](https://arxiv.org/abs/2410.03617): Large-model merging can work, but endpoint controls are required; router-calibrated candidates are rejected if dominated by source endpoints.
- [Model Merging by Output-Space Projection](https://arxiv.org/abs/2605.29101): The route-KD cache is an output-space calibration signal for routers; this script uses downstream scores to decide whether that local calibration transfers to the full model.

## Outputs

- `results/qwen3_moe_router_calibration_selection_smoke/selection_table.csv`
- `results/qwen3_moe_router_calibration_selection_smoke/decision_rules.json`
- `results/qwen3_moe_router_calibration_selection_smoke/summary.json`
