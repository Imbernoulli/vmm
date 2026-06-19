# Qwen Dense Norm-Guarded Bridge Candidate

## 结论

`qwen_0_5b_norm_guarded_bridge` 是 global bridge 后的模块级 ablation：保留 `alpha=0.25,beta=1.0` 的基本方向，只冻结 norm/layernorm，其它模块保持 global bridge。

## Module Probe

| group | tensors | params | cosine | sign conflict | instruct/coder L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| mlp | 72 | 313786368 | 0.142 | 0.442 | 0.070 |
| embedding_anchor | 1 | 136134656 | 0.142 | 0.448 | 0.092 |
| attention | 168 | 44067840 | 0.138 | 0.442 | 0.064 |
| norm_anchor | 49 | 43904 | 0.090 | 0.441 | 0.084 |

## Tensor Rules

| group | instruct | coder | reason |
| --- | ---: | ---: | --- |
| norm_anchor | 0.00 | 0.00 | freeze only the highest-conflict normalization anchors |
| embedding_anchor | 0.25 | 1.00 | keep the global bridge on token embeddings for lexical/task adaptation |
| attention | 0.25 | 1.00 | keep the NLL-selected bridge on attention tensors |
| mlp | 0.25 | 1.00 | keep the global bridge on MLP tensors to isolate norm freezing |

## Materialization

```bash
python scripts/write_same_shape_average_checkpoint.py --base /srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-0.5B --source instruct=/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source coder=/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25 --source-weight instruct=0.0 --source-weight coder=0.0 --tensor-rule-file results/qwen_dense_norm_guarded_candidate/tensor_rules.txt --output-dir results/checkpoints/qwen_0_5b_norm_guarded_bridge
```

## vLLM Eval

```bash
CUDA_VISIBLE_DEVICES=1 /srv/home/bohanlyu/miniconda3/envs/cogdoc/bin/vllm serve results/checkpoints/qwen_0_5b_norm_guarded_bridge --served-model-name candidate_qwen_0_5b_norm_guarded_bridge --host 127.0.0.1 --port 8100 --dtype bfloat16 --tensor-parallel-size 1
```

```bash
python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8100/v1 --models candidate_qwen_0_5b_norm_guarded_bridge --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/qwen_0_5b_norm_guarded_bridge
```

## vLLM Eval Result

真实 endpoint eval 已完成：avg primary `0.203`，相对 global bridge `0.000`，相对 best source `-0.172`。

| task | score | global bridge | delta |
| --- | ---: | ---: | ---: |
| gsm8k | 0.047 | 0.062 | -0.016 |
| mmlu | 0.219 | 0.250 | -0.031 |
| safety | 0.547 | 0.500 | 0.047 |
| humaneval_compile | 0.000 | 0.000 | 0.000 |

## Artifacts

- Tensor stats: `results/qwen_dense_norm_guarded_candidate/tensor_conflict.csv`
- Module summary: `results/qwen_dense_norm_guarded_candidate/module_conflict.csv`
- Tensor rules: `results/qwen_dense_norm_guarded_candidate/tensor_rules.txt`
- Summary: `results/qwen_dense_norm_guarded_candidate/summary.json`
