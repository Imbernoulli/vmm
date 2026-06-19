# Checkpoint Topology Inspect

这个报告只读 config 和 safetensors header，不加载权重内容。它用于 Average 前的第一层同构检查：模型是否是 MoE、专家数/激活专家数是多少、router/expert/shared 参数是否可被分组处理。

## Models

| model | model_type | layers | hidden | experts | active/top-k | active fraction | weights | total bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| qwen3_6_35b_a3b | qwen3_5_moe | 40 | 2048 | 256 | 8 | 0.0312 | yes | 71903645408 |
| qwen2_5_0_5b_dense | qwen2 | 24 | 896 | None | None | n/a | yes | 988065536 |

## Average-Relevant Notes

### qwen3_6_35b_a3b
- MoE config: `256` experts, `8` active per token; active fraction `0.03125`.
- Average implication: router、shared modules、routed experts 必须分组处理；不能把 router 当普通 dense 层同权平均。
- `router`: tensors `41`, bytes `42991616`.
- `routed_expert`: tensors `82`, bytes `66035122176`.
- `shared_expert`: tensors `164`, bytes `258117632`.
- `attention`: tensors `392`, bytes `2909788928`.
- `dense_mlp`: tensors `108`, bytes `535781088`.
- `embedding`: tensors `1`, bytes `1017118720`.
- `lm_head`: tensors `1`, bytes `1017118720`.

### qwen2_5_0_5b_dense
- Dense config: 没有 MoE expert/router 字段；可走 dense average / task-vector / layer-wise coefficient 路线。
- `attention`: tensors `168`, bytes `88135680`.
- `dense_mlp`: tensors `72`, bytes `627572736`.
- `embedding`: tensors `1`, bytes `272269312`.

## Pairwise Config Compatibility

| left | right | same config | mismatched fields |
| --- | --- | --- | --- |
| qwen3_6_35b_a3b | qwen2_5_0_5b_dense | False | model_type,hidden_size,num_hidden_layers,num_attention_heads,moe_intermediate_size,shared_expert_intermediate_size,num_experts,num_experts_per_tok,vocab_size |

## Files

- `summary.json`
- `*_groups.csv` when safetensors headers are available
- `*_experts.csv` when expert tensors are visible in headers
