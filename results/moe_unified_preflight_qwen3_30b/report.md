# Qwen3 MoE Unified Average Preflight

这一步不是继续做静态方法排名，而是把 unified average 的必要条件拆成可验证合同：同构 topology、router tensor、packed expert layout、expert identity、真实 route/load，以及最终 vLLM 行为评测。当前脚本只读 config 和 safetensors header，不加载 30B 权重内容。

## Current Result

- Status: `same_shape_and_identity_ready_route_runtime_blocked_here`
- Pair: `qwen3_30b_instruct` vs `qwen3_30b_coder`
- Same-shape contract: `True`
- Expert identity gate: `pass`
- CUDA available in this process: `False`
- Router tensors compared: `48`
- Routed expert tensors compared: `18432`

## Why This Is The Unified Algorithm Gate

MoE 的输出可以粗略写成 `shared(x) + sum_e route_e(x; R) * expert_e(x; W_e)`。因此 average 失败不是一个单一原因：同名 expert 可能只是 gauge index，expert 函数可能不在同一个语义坐标，router 的 top-k 边界可能漂移，serving 时还会出现 capacity/load 问题。unified 方法应该先验证这些机制，再决定 expert remap、expert weights、router freeze/small-step/route-KD 和 router-bias capacity correction；最终 checkpoint 仍保持同结构。

## Model Contract

- Model type: `qwen3_moe`
- Layers: `48`
- Experts per layer: `128`
- Active experts per token: `8`

| field | left | right | match |
| --- | --- | --- | --- |
| `model_type` | `qwen3_moe` | `qwen3_moe` | `True` |
| `hidden_size` | `2048` | `2048` | `True` |
| `num_hidden_layers` | `48` | `48` | `True` |
| `num_attention_heads` | `32` | `32` | `True` |
| `num_key_value_heads` | `4` | `4` | `True` |
| `moe_intermediate_size` | `768` | `768` | `True` |
| `shared_expert_intermediate_size` | `None` | `None` | `True` |
| `num_experts` | `128` | `128` | `True` |
| `num_experts_per_tok` | `8` | `8` | `True` |
| `vocab_size` | `151936` | `151936` | `True` |

## Gate Table

| stage | status | action | evidence |
| --- | --- | --- | --- |
| `same_shape_config` | `pass` | `continue` | All average-critical config fields match. |
| `router_tensor_contract` | `pass` | `allow_router_probe` | 48 router tensors compared; all shapes match=True. |
| `routed_expert_layout` | `pass` | `allow_identity_or_remap_gate` | 18432 routed expert tensors compared; per-layer layout match=True. |
| `expert_identity` | `pass` | `use_identity_expert_slices_first` | Expert identity is stable enough to use identity slices first. |
| `runtime_route_probe` | `blocked_in_this_process` | `run the emitted command on a GPU/vLLM host; keep materialization blocked here` | torch.cuda.is_available=False. |
| `behavior_eval` | `waiting` | `host candidate with vLLM after route/load gates pass` | No same-shape Qwen3 MoE candidate should be published before route overlap, load, and downstream eval. |

## Next Executable Probe

```bash
python scripts/probe_moe_routing.py --model /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --compare-model /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120 --tokenizer /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --prompts prompts/qwen_moe_route_probe_prompts.jsonl --device-map auto --dtype bfloat16 --use-chat-template --local-files-only --top-k 8 --output-dir results/moe_routing_probe/qwen3_30b_instruct_vs_coder
PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir results/moe_routing_probe/qwen3_30b_instruct_vs_coder --output-dir results/moe_routing_readiness/qwen3_30b_instruct_vs_coder
```

## Files

- `results/moe_unified_preflight_qwen3_30b/config_contract.csv`
- `results/moe_unified_preflight_qwen3_30b/router_contract.csv`
- `results/moe_unified_preflight_qwen3_30b/expert_contract.csv`
- `results/moe_unified_preflight_qwen3_30b/layout_contract.csv`
- `results/moe_unified_preflight_qwen3_30b/unified_gate_table.csv`
- `results/moe_unified_preflight_qwen3_30b/routing_probe_command.txt`
