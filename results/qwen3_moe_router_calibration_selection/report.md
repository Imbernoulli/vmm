# Qwen3 MoE Router Calibration Selection

这一步只解决一个问题：在 frozen-router 的 `searched_no_gt065` 基线之上，是否应该加入一个小的 route-KD router delta。结论不会按算法名决定，而是按机制证据决定：下游任务、router-only 审计、delta cap、source endpoint 支配关系必须同时通过。

- Selection status: `awaiting_baseline_eval`
- Selected method: `None`
- Reason: Run the frozen-router searched_no_gt065 baseline eval before deciding whether router calibration helps.
- Baseline eval completed: `False`
- Source eval required: `True`
- Source eval completed: `False`
- Candidate eval completed: `False`
- Audit completed: `False`
- Training completed: `False`
- Capacity metrics completed: `False`
- Group validation completed: `False`
- Router margin gate completed: `True`
- Router margin safe-lambda proxy: `0.0197`
- Router margin high-fragility layers: `24/48`
- Task manifest gate completed: `False`
- Eligible candidates: `0/4`
- Active candidates: `2`
- Plan-pruned candidates: `2`
- Baseline task manifest sha: `None`

## Baseline

| eval dir | status | avg | worst | gsm8k | mmlu | safety | humaneval |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `results/vllm_checkpoint_eval/qwen3_moe_searched_no_gt065_max_retention_candidate` | `not_run` |  |  |  |  |  |  |

## Candidate Gate

| cap | mode | method | manifest | split | groups | selected epoch | KL gap | top1 drop | decision | avg delta | worst delta | worst task delta | router max rel | table violations | margin pass | plan-pruned | top1/top-k overflow | top1/top-k increase | load pass | gen pass | group pass | router-only | cap pass | score | reason |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | --- |
| 0.0100 | `global` | `qwen3_moe_router_calibrated_searched_no_gt065_cap001_candidate` | `awaiting_eval` | `None` | / |  |  |  | `reject_or_wait` |  |  |  |  | 0 | `False` | `False` | / | / | `False` | `False` | `False` | `False` | `False` | -0.0001 | `awaiting_baseline_eval,awaiting_source_eval,awaiting_router_training,awaiting_candidate_eval,awaiting_audit` |
| 0.0250 | `global` | `qwen3_moe_router_calibrated_searched_no_gt065_cap0025_candidate` | `awaiting_eval` | `None` | / |  |  |  | `reject_or_wait` |  |  |  |  | 0 | `False` | `True` | / | / | `False` | `False` | `False` | `False` | `False` | -0.0003 | `awaiting_baseline_eval,awaiting_source_eval,awaiting_router_training,awaiting_candidate_eval,awaiting_audit,router_margin_planned_cap_violation` |
| 0.0500 | `global` | `qwen3_moe_router_calibrated_searched_no_gt065_cap005_candidate` | `awaiting_eval` | `None` | / |  |  |  | `reject_or_wait` |  |  |  |  | 0 | `False` | `True` | / | / | `False` | `False` | `False` | `False` | `False` | -0.0005 | `awaiting_baseline_eval,awaiting_source_eval,awaiting_router_training,awaiting_candidate_eval,awaiting_audit,router_margin_planned_cap_violation` |
| 0.0500 | `per_router_margin_profile` | `qwen3_moe_router_calibrated_searched_no_gt065_margin_profile_candidate` | `awaiting_eval` | `None` | / |  |  |  | `reject_or_wait` |  |  |  |  | 0 | `False` | `False` | / | / | `False` | `False` | `False` | `False` | `False` | -0.0005 | `awaiting_baseline_eval,awaiting_source_eval,awaiting_router_training,awaiting_candidate_eval,awaiting_audit` |

## Source Controls

| eval dir | status | avg | worst |
| --- | --- | ---: | ---: |
| `results/vllm_checkpoint_eval/source_qwen3_30b_instruct` | `not_run` |  |  |
| `results/vllm_checkpoint_eval/source_qwen3_30b_coder` | `not_run` |  |  |

## Unified Rule Update

如果 selection status 是 `selected_router_calibrated_candidate`，统一方法可以在 `searched_no_gt065` expert/attention 冻结策略后追加该 cap 的 router delta；否则 unified 默认继续保持 frozen router。这样算法不会在 router 机制证据不足时强行动 router。

## Decision Rules

- Baseline searched_no_gt065 eval must be complete on the same vLLM task set.
- Both source endpoint evals must be complete unless --allow-missing-source-eval is explicitly set.
- Every cap candidate must have a materialized delta audit and vLLM eval before final selection.
- Every cap candidate must have router training metrics with hard top-1/top-k route-load statistics.
- Router route-KD validation must use group-heldout prompt/batch splits with at least 1 train groups and 1 validation groups, unless --allow-row-validation is explicitly set.
- Baseline, source, and candidate downstream eval summaries must carry the same task_manifest_sha256.
- The audit must show only router tensors changed, with no shape/dtype mismatch.
- The router relative delta norm must stay inside the planned global cap or, for margin-profile candidates, every router tensor must stay inside its cap-table entry.
- The planned and audited router delta must stay inside the observed top-k margin safe-lambda proxy unless --disable-router-margin-gate is explicitly set; current tolerance is 0.02.
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
- [When Model Merging Breaks Routing: Training-Free Calibration for MoE](https://arxiv.org/abs/2606.03391): Router movement is gated by observed top-k boundary margins; a route-KD delta may not exceed the measured safe-lambda proxy without fresh downstream evidence.

## Outputs

- `results/qwen3_moe_router_calibration_selection/selection_table.csv`
- `results/qwen3_moe_router_calibration_selection/decision_rules.json`
- `results/qwen3_moe_router_calibration_selection/summary.json`
