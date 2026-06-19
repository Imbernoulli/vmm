# Qwen 0.5B Source-vs-Merge vLLM Comparison

## 结论

同一套 vLLM 下游评测里，`Instruct/Coder 0.5/0.5 Uniform Average` 不是有效折中点：avg primary `0.180`，比最佳源模型 `Qwen2.5-0.5B Base` 的 `0.375` 低 `0.195`；worst primary `0.000`，仍然是零。

这说明这里的失败不是静态 probe 的误判，而是在真实 endpoint 上也成立：uniform average 没有同时保留 instruct/coder/base 的能力，反而落在一个多任务都不强的区域。

## Model Scores

| model | role | avg primary | worst primary | delta vs best source avg |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-0.5B Base | source | 0.375 | 0.094 | 0.000 |
| Qwen2.5-0.5B Instruct | source | 0.227 | 0.000 | -0.148 |
| Qwen2.5-Coder-0.5B-Instruct | source | 0.199 | 0.000 | -0.176 |
| Instruct/Coder 0.5/0.5 Uniform Average | merge | 0.180 | 0.000 | -0.195 |

## Task-Level Primary Metrics

| task | merge score | best source | best source score | merge delta |
| --- | ---: | --- | ---: | ---: |
| gsm8k | 0.000 | Qwen2.5-0.5B Base | 0.094 | -0.094 |
| humaneval_compile | 0.000 | Qwen2.5-0.5B Base | 0.609 | -0.609 |
| mmlu | 0.219 | Qwen2.5-0.5B Instruct | 0.344 | -0.125 |
| safety | 0.500 | Qwen2.5-0.5B Instruct | 0.547 | -0.047 |

## 机理解释

- Base 的 aggregate 分数最高，主要来自 `humaneval_compile=0.609`；这个指标只检查生成片段能否编译，不等价于 instruction/code benchmark 的真实 pass rate，也不能说明 base 已经是目标平均模型。
- Instruct 在 MMLU 和 safety 上强于 coder；coder 的 GSM8K strict 只高于 instruct，但这个 64-sample slice 里 base 仍是 GSM8K 和 compile 的最高点。这些 endpoint skill 没有被 `0.5/0.5` 权重同时继承。
- Uniform merge 的 safety policy accuracy 是 `0.500`，但 safe_non_refusal 是 `1.000`、unsafe_refusal 是 `0.000`。这不是安全行为变好，而是几乎不拒绝 unsafe prompts。
- 这个结果和前面的 Qwen multi-expert NLL plane 一致：`alpha=0.5,beta=0.5` 位于高 worst-NLL ridge，因此真实生成评测也退化。
- 下一步的统一算法不能只问“哪个算法在哪个场景最好”，而要把 source endpoint ability、任务向量连通性、模块级 conflict、输出空间投影残差、MoE router/expert load 一起作为 gate，再写回同构 checkpoint。

## Artifacts

- Model score CSV: `results/vllm_source_merge_comparison/model_scores.csv`
- Task metric CSV: `results/vllm_source_merge_comparison/task_metrics.csv`
- Figure: `results/vllm_source_merge_comparison/source_vs_merge_primary_scores.png`
- Summary JSON: `results/vllm_source_merge_comparison/summary.json`
