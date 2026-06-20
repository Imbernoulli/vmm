# Qwen3 MoE Unified Result Selector

这个 selector 只回答一个问题：source endpoints 和 unified mechanism candidate 完成同一套 vLLM 下游任务后，是否接受这个 Average，还是回退到同结构 source endpoint。

- Status: `awaiting_source_eval`
- Selected: `None`
- Reason: `Both Qwen3 source endpoints must complete matched vLLM downstream eval before accepting an average.`
- Source eval complete: `False`
- Unified eval complete: `False`
- Unified audit passed: `True`
- Alias status: `None`

## Selection Table

| method | eval | audit | avg | worst | gsm8k | mmlu | safety | humaneval | dominated | regression | eligible |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `source_qwen3_30b_instruct` | `False` | `True` | None | None | None | None | None | None | `` | `` | `False` |
| `source_qwen3_30b_coder` | `False` | `True` | None | None | None | None | None | None | `` | `` | `False` |
| `qwen3_moe_unified_mechanism_candidate` | `False` | `True` | None | None | None | None | None | None | `` | `` | `False` |

## Outputs

- `results/qwen3_moe_unified_result_selection/selection_table.csv`
- `results/qwen3_moe_unified_result_selection/summary.json`
- `results/qwen3_moe_unified_result_selection/decision_rules.json`
