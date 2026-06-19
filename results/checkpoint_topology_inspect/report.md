# Checkpoint Topology Inspect

这个报告只读 config 和 safetensors header，不加载权重内容。它用于 Average 前的第一层同构检查：模型是否是 MoE、专家数/激活专家数是多少、router/expert/shared 参数是否可被分组处理。

## Models

| model | model_type | layers | hidden | experts | active/top-k | active fraction | weights | total bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| qwen3_5_35b_a3b | qwen3_5_moe | 40 | 2048 | 256 | 8 | 0.0312 | no | n/a |
| qwen3_0_6b_dense | qwen3 | 28 | 1024 | None | None | n/a | yes | 1503264768 |

## Average-Relevant Notes

### qwen3_5_35b_a3b
- MoE config: `256` experts, `8` active per token; active fraction `0.03125`.
- Average implication: router、shared modules、routed experts 必须分组处理；不能把 router 当普通 dense 层同权平均。
- 本地没有 safetensors 权重 shard；当前只完成 config-level topology probe。

### qwen3_0_6b_dense
- Dense config: 没有 MoE expert/router 字段；可走 dense average / task-vector / layer-wise coefficient 路线。
- `attention`: tensors `196`, bytes `352393216`.
- `dense_mlp`: tensors `84`, bytes `528482304`.
- `embedding`: tensors `1`, bytes `311164928`.
- `lm_head`: tensors `1`, bytes `311164928`.

## Pairwise Config Compatibility

| left | right | same config | mismatched fields |
| --- | --- | --- | --- |
| qwen3_5_35b_a3b | qwen3_0_6b_dense | False | model_type,hidden_size,num_hidden_layers,num_key_value_heads,moe_intermediate_size,shared_expert_intermediate_size,num_experts,num_experts_per_tok,vocab_size |

## Files

- `summary.json`
- `*_groups.csv` when safetensors headers are available
- `*_experts.csv` when expert tensors are visible in headers
