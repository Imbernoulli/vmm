# Toy MoE Route-Aware Merge

这个实验用一个很小的 soft-router MoE 做可控验证：base 先在 general/code 两类合成任务上训练，然后从同一 base fine-tune 两个同构 source。为了模拟 MoE 中常见的 expert-index 语义漂移，code source 在保持函数等价的前提下被 permute experts 和 router rows。

它验证的点很具体：直接 all-weight average 会把不同语义的 expert index 相加；expert matching 和 route-frequency expert weights 可以缓解这个问题；router 是否能开放平均要看 route overlap、load concentration 和 top-k margin。

## 关键结果

- Best method by worst accuracy: `code_endpoint_permuted` = `0.802`.
- All-weight average worst accuracy: `0.620`.
- Expert-matched average worst accuracy: `0.800`.
- Matched + router-frozen average worst accuracy: `0.787`.
- Route-aware expert average worst accuracy: `0.790`.
- Recovered expert matching mean cosine: `0.977`.
- Code source permutation: `[2, 0, 3, 1]`.

## Method Table

| method | general acc | code acc | worst acc | avg loss |
| --- | ---: | ---: | ---: | ---: |
| code_endpoint_permuted | 0.818 | 0.802 | 0.802 | 0.618 |
| expert_matched_average | 0.818 | 0.800 | 0.800 | 0.616 |
| route_aware_expert_average | 0.807 | 0.790 | 0.790 | 0.620 |
| matched_router_frozen_average | 0.815 | 0.787 | 0.787 | 0.619 |
| general_endpoint | 0.812 | 0.780 | 0.780 | 0.615 |
| base | 0.797 | 0.775 | 0.775 | 0.628 |
| all_weight_average | 0.688 | 0.620 | 0.620 | 0.663 |
| router_frozen_average | 0.690 | 0.615 | 0.615 | 0.666 |

## Interpretation

- `all_weight_average` 是朴素 baseline：router 和 expert tensors 都按同名 index 平均，因此在 expert permutation 后会暴露 MoE index-alignment 风险。
- `expert_matched_average` 先用 unlabeled calibration input 的 expert-output cosine 做 Hungarian matching，再平均；这对应 Sub-MoE / Expert Merging 里强调的 function-aware expert alignment。
- `matched_router_frozen_average` 直接验证 MoE 特有假设：先对齐 expert 功能，再固定 token-to-expert dispatch，只平均非 router 权重。
- `route_aware_expert_average` 冻结 base router，并按 base router 在 general/code prompt 上的 route mass 给每个 expert 设置 source delta 权重；这对应 route-weight recipes 的 toy 版本。
- 这个实验不是 Qwen3 结果，但它把 MoE merging 的特质从报告落成了可跑的 probe：expert index、router overlap、expert load 和 category route mass 都会影响 average 是否安全。

## Files

- `method_metrics.csv`
- `router_summary.csv`
- `expert_load.csv`
- `route_overlap.csv`
- `expert_match.csv`
- `route_weights_by_expert.csv`
- `toy_moe_merge.png`
- `summary.json`
