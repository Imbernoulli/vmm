# MoE Average Mechanism Probe

这个实验只回答一个问题：为什么同构 MoE checkpoint 不能直接按同名 tensor average。我们把 B 模型做了一个函数等价的 expert/router row 置换；B 的输出几乎不变，但同名 expert index 的语义被打乱，因此它能隔离 MoE 特有的平均失败机制。

## 结果摘要

- 函数等价置换后的 B 与原 B probe MSE: `7.66403e-16`。
- same-name uniform worst loss: `0.5105`。
- expert-aligned uniform worst loss: `0.1252`。
- aligned + router calibration worst loss: `0.1095`。
- aligned + Fisher worst loss: `0.1520`。
- 本次 best merge: `uniform_aligned_routercal`，worst loss `0.1095`。
- 本次 overall lowest: `base`，worst loss `0.1067`。

## 机制结论

| mechanism | baseline | intervention | worst-loss reduction | implication |
| --- | --- | --- | ---: | --- |
| `expert_identity_alignment` | `uniform_same_name` | `uniform_aligned` | 0.3853 | same tensor name is not a stable expert identity after a legal MoE gauge permutation |
| `router_calibration_after_alignment` | `uniform_aligned` | `uniform_aligned_routercal` | 0.0157 | after experts move, the top-k router boundary must be re-fit to the merged experts |
| `route_conditioned_fisher` | `uniform_aligned` | `fisher_aligned` | -0.0267 | curvature weighting is not automatically safe; it must pass a held-out gate |
| `router_calibration_after_fisher` | `fisher_aligned` | `fisher_aligned_routercal` | 0.0068 | router calibration can repair some dispatch drift even when expert weighting is imperfect |
| `router_cannot_fix_misaligned_experts` | `uniform_same_name` | `uniform_same_name_routercal` | 0.1812 | router-only fitting is limited if the averaged experts themselves mix unrelated functions |

## Unified Rule

不是固定说 TIES/Fisher/RegMean 哪个永远最好，而是用同一个 gate：先检查模型是否同构；MoE 先做 expert function alignment；expert 权重只在 route-conditioned probe 上降低 held-out loss 时启用；router 只允许在 expert 已对齐之后做小步校准或蒸馏；capacity correction 只在 sparse top-k overflow 超预算时启用。所有步骤保持 tokenizer、模型类、tensor name/shape、expert 数不变。

## Files

- `summary.json`
- `method_metrics.csv`
- `mechanism_deltas.csv`
- `moe_mechanism.png`
