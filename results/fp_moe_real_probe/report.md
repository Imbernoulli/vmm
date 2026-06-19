# Real MoE Expert-Gauge Self-Merge Probe

这个实验在真实 MoE LLM checkpoint 上做一个函数等价反事实：把每一层的 expert slice 和 router row 同步置换，模型函数应保持不变；然后比较同名 average 和恢复 expert 对齐后的 average。

## Result

- Model: `allenai/OLMoE-1B-7B-0924-Instruct`
- MoE layers: `16`; experts/layer: `64`; format: `packed`
- Baseline NLL: `4.1678`
- Gauge-permuted NLL: `4.1656`; delta `-0.002217`
- Naive same-name average NLL: `9.6588`; delta `5.4910`
- Aligned average NLL: `4.1678`; delta `0.000000`
- Exact permutation recovery: `16/16` layers

## Interpretation

如果 gauge-permuted NLL 与 baseline 一致，而 same-name average 退化，就说明 MoE 的 expert index 不是稳定语义。即使两个 checkpoint 表示同一个函数，只要 expert gauge 不一致，按同名 tensor average 也会破坏模型。恢复 expert 对齐后再 average 是同构 MoE merge 的必要前置步骤。

## Qwen3 Cross-Correspondence

同目录下还有一个真实 Qwen3 MoE cross-correspondence probe：`Qwen/Qwen3-30B-A3B-Instruct-2507` vs `Qwen/Qwen3-Coder-30B-A3B-Instruct`，base-subtracted 后逐层匹配 routed expert deltas。结果是 `48/48` 层 identity-optimal，mean argmax-is-identity fraction `1.000`，mean diagonal cosine `0.183`，mean off-diagonal cosine `0.00014`。

这说明这对官方同源 Qwen3 MoE 目前没有明显 expert permutation；真实 merge 可以先保留 identity mapping，但仍要保留 expert-alignment gate，因为 OLMoE self-merge 反事实已经证明，一旦 gauge 不一致，同名 average 会直接失败。

## Files

- `summary.json`
- `gauge_selfmerge.json`
- `qwen3_instruct_coder/cross_correspondence.json`
