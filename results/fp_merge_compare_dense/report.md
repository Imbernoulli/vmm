# Unified Merge Family Probe

这个实验不是预先判断哪个命名算法最好，而是把多个方法写成同一个候选族，再用 held-out worst-task NLL 选择候选，最后在 disjoint test split 上复验。

## Result

- candidate count: `24`
- grid profile: `linear`
- selected config: `{"density": 1.0, "importance": "uniform", "lam": 0.0, "router": "average", "sign_resolve": true}`
- selected held-out worst NLL: `5.4525`
- unified test worst NLL: `5.1830`
- best endpoint worst NLL: `5.1510`

## Mechanism

`linear average`、`task arithmetic`、sign-elect、magnitude-weighted merge 都是 `base + lambda * combine(delta_A, delta_B)` 的特例。选择器测的是：midpoint 是否跨 barrier、任务向量是否需要缩放、同坐标 sign 冲突是否应该被过滤、以及重要性是否集中在大幅度坐标上。

在有限候选族内，held-out 选择等价于选经验风险最低的候选；它不能保证击败所有未知算法，但能避免把某个固定方法当成先验真理。真正上线前仍需要 vLLM hosted downstream eval。

## Files

- `summary.json`
- `method_metrics.csv`
- `selection_trace.csv`
