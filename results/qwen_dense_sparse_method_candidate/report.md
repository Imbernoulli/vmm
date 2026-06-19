# Qwen Dense Sparse-Method Candidate

## 结论

`qwen_0_5b_sparse_method_bridge` 把已有 global bridge 的 source weights `instruct=0.25`、`coder=1.0` 保留不变，但对 high-conflict attention,mlp tensor 加上 `ties` coordinate rule。

这不是再做模块级 freeze。它对应 TIES/DARE/DELLA/Breadcrumbs 这类 sparse task-vector 机制：先用 probe 找到符号冲突集中的坐标，再在同构 writer 中对这些坐标做 trim/sign-elect/merge。

## Selection

| metric | value |
| --- | ---: |
| selected tensors | 99 |
| selected parameters | 240730112 |
| selected parameter fraction | 0.4873 |
| min sign conflict | 0.440 |
| max cosine | 0.160 |
| density | 0.500 |
| dry-run status | passed |
| dry-run sparse method tensors | 99 |

Projection counts: `{"attn_k": 21, "attn_o": 2, "attn_q": 23, "attn_v": 3, "mlp_down": 15, "mlp_gate": 22, "mlp_up": 13}`

## vLLM Eval Result

| metric | value |
| --- | ---: |
| status | complete |
| avg primary | 0.156 |
| worst primary | 0.000 |
| delta vs global bridge avg | -0.047 |
| delta vs uniform avg | -0.023 |
| delta vs best source avg | -0.219 |

| task | primary metric | score | delta vs global bridge | delta vs uniform |
| --- | --- | ---: | ---: | ---: |
| gsm8k | strict_exact | 0.016 | -0.047 | 0.016 |
| mmlu | accuracy | 0.219 | -0.031 | 0.000 |
| safety | policy_accuracy | 0.391 | -0.109 | -0.109 |
| humaneval_compile | compile_rate | 0.000 | 0.000 | 0.000 |

Full vLLM report: `results/vllm_checkpoint_eval/qwen_0_5b_sparse_method_bridge/report.md`

## Selected Tensor Preview

| tensor | projection | numel | cosine | sign conflict |
| --- | --- | ---: | ---: | ---: |
| `model.layers.23.self_attn.k_proj.weight` | attn_k | 114688 | 0.027 | 0.485 |
| `model.layers.22.self_attn.k_proj.weight` | attn_k | 114688 | 0.015 | 0.479 |
| `model.layers.17.self_attn.k_proj.weight` | attn_k | 114688 | 0.037 | 0.472 |
| `model.layers.16.self_attn.k_proj.weight` | attn_k | 114688 | 0.025 | 0.471 |
| `model.layers.20.self_attn.k_proj.weight` | attn_k | 114688 | 0.044 | 0.471 |
| `model.layers.21.self_attn.k_proj.weight` | attn_k | 114688 | 0.051 | 0.467 |
| `model.layers.18.self_attn.k_proj.weight` | attn_k | 114688 | 0.060 | 0.466 |
| `model.layers.11.self_attn.k_proj.weight` | attn_k | 114688 | 0.066 | 0.464 |
| `model.layers.13.self_attn.k_proj.weight` | attn_k | 114688 | 0.059 | 0.464 |
| `model.layers.16.self_attn.q_proj.weight` | attn_q | 802816 | 0.094 | 0.462 |
| `model.layers.9.self_attn.k_proj.weight` | attn_k | 114688 | 0.077 | 0.461 |
| `model.layers.22.self_attn.q_proj.weight` | attn_q | 802816 | 0.124 | 0.460 |
| `model.layers.14.self_attn.k_proj.weight` | attn_k | 114688 | 0.080 | 0.459 |
| `model.layers.23.self_attn.q_proj.weight` | attn_q | 802816 | 0.136 | 0.458 |
| `model.layers.5.self_attn.k_proj.weight` | attn_k | 114688 | 0.079 | 0.458 |
| `model.layers.15.self_attn.k_proj.weight` | attn_k | 114688 | 0.077 | 0.457 |
| `model.layers.19.self_attn.k_proj.weight` | attn_k | 114688 | 0.083 | 0.456 |
| `model.layers.4.self_attn.k_proj.weight` | attn_k | 114688 | 0.081 | 0.455 |
| `model.layers.17.self_attn.q_proj.weight` | attn_q | 802816 | 0.112 | 0.455 |
| `model.layers.0.self_attn.v_proj.weight` | attn_v | 114688 | 0.117 | 0.454 |

## Writer Commands

Dry-run validation:

```bash
python scripts/write_same_shape_average_checkpoint.py --base /srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-0.5B --source instruct=/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source coder=/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25 --source-weight instruct=0.25 --source-weight coder=1.0 --tensor-method-rule-file results/qwen_dense_sparse_method_candidate/tensor_method_rules.txt --output-dir /srv/home/bohanlyu/visualizing-model-merging/results/qwen_dense_sparse_method_candidate/dry_run --dry-run
```

Materialize candidate:

```bash
python scripts/write_same_shape_average_checkpoint.py --base /srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-0.5B --source instruct=/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source coder=/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25 --source-weight instruct=0.25 --source-weight coder=1.0 --tensor-method-rule-file results/qwen_dense_sparse_method_candidate/tensor_method_rules.txt --output-dir results/checkpoints/qwen_0_5b_sparse_method_bridge
```

## vLLM Eval

```bash
CUDA_VISIBLE_DEVICES=1 /srv/home/bohanlyu/miniconda3/envs/cogdoc/bin/vllm serve results/checkpoints/qwen_0_5b_sparse_method_bridge --served-model-name candidate_qwen_0_5b_sparse_method_bridge --host 127.0.0.1 --port 8100 --dtype bfloat16 --tensor-parallel-size 1
```

```bash
python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8100/v1 --models candidate_qwen_0_5b_sparse_method_bridge --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/qwen_0_5b_sparse_method_bridge
```

## Files

- `results/qwen_dense_sparse_method_candidate/selected_tensors.csv`
- `results/qwen_dense_sparse_method_candidate/tensor_method_rules.txt`
- `results/qwen_dense_sparse_method_candidate/dry_run/merge_manifest.json`
- `results/qwen_dense_sparse_method_candidate/summary.json`
