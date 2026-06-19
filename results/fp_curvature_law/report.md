# Dense Curvature-Displacement Probe

这个实验用真实 Qwen2.5-0.5B Instruct / Coder 权重和真实 Wikitext、HumanEval NLL 来问一个更底层的问题：Dense checkpoint averaging 的退化能不能被局部二阶 Fisher 曲率解释。

## 结果摘要

- task-vector cosine: `0.140`，说明 instruct 与 coder task vector 方向不是完全相反，但也远不是同一个方向。
- uniform midpoint 的 general NLL 从 Instruct endpoint 的 `3.099` 升到 `5.911`，code NLL 从 Coder endpoint 的 `0.515` 升到 `1.747`。
- diagonal-Fisher 二阶预测 general degradation 只有 `0.0656`，真实是 `2.8120`，actual/predicted = `42.86`。
- diagonal-Fisher 二阶预测 code degradation 只有 `0.0462`，真实是 `1.2316`，actual/predicted = `26.66`。
- Fisher merge 的 worst NLL 是 `5.249`，比 uniform midpoint 的 `5.911` 好，但仍明显处在高 loss barrier 上。

## 机制解释

这说明 Qwen instruct/coder 的 midpoint 退化不是一个局部小凸二次项能解释的现象。局部 Fisher 只能描述 endpoint 附近的小位移；而 uniform average 跨过的是两个 task-specialized checkpoints 之间的非局部路径，真实 loss barrier 比二阶预测大一个数量级以上。

因此 unified dense average 不能只靠 “0.5/0.5 平均” 或单次 Fisher 权重。更合理的 gate 是：先用真实 eval 或 NLL path 判断 A->B 是否有 barrier；如果 midpoint barrier 很高，就做 coefficient/layer/tensor gate；如果某个 curvature-aware 或 sparse-coordinate intervention 在 held-out eval 上没有降低 worst loss，就不能把它设成默认。

## Files

- `summary.json`
- `curvature_law.png`
- `run.log`
