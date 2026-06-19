# Probe-Guided Dense Average Candidate

## 结论

从 Qwen instruct/coder NLL grid 里选出的下一轮同构 checkpoint candidate 是 `qwen_0_5b_probe_guided_bridge_a025_b100`：`alpha=0.25`、`beta=1.00`。

它不是端点复制，也不是 `0.5/0.5` uniform average。相对 uniform midpoint，worst NLL 降低 `1.921`，avg NLL 降低 `2.551`；但它仍然必须用真实 vLLM endpoint 评测验证。

## Selection Metrics

| metric | value |
| --- | ---: |
| alpha | 0.25 |
| beta | 1.00 |
| avg NLL | 3.040 |
| worst NLL | 7.632 |
| general NLL | 7.632 |
| instruction NLL | 0.902 |
| code NLL | 0.586 |
| uniform avg NLL | 5.591 |
| uniform worst NLL | 9.553 |

## 机制解释

- `0.5/0.5` midpoint 落在高 worst-NLL ridge 上，说明两个 task deltas 在这个切片里不是简单同 basin 线性连通。
- 选出的 bridge candidate 沿着 coder endpoint 保留代码 delta，同时只加入一小段 instruct delta；它的目标是测试“低剂量跨任务注入”是否比全量对半平均更稳定。
- 如果 vLLM 下游评测仍然差，结论不是换一个固定系数，而是进入 layer/module-wise weighting：对冲突层降权，对稳定共享层保留平均。

## Materialization

Checkpoint output: `results/checkpoints/qwen_0_5b_probe_guided_bridge_a025_b100`

```bash
python scripts/write_same_shape_average_checkpoint.py --base /srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-0.5B --source instruct=/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source coder=/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25 --source-weight instruct=0.25 --source-weight coder=1.0 --output-dir results/checkpoints/qwen_0_5b_probe_guided_bridge_a025_b100
```

## vLLM Eval

Eval output: `results/vllm_checkpoint_eval/qwen_0_5b_probe_guided_bridge_a025_b100`

```bash
CUDA_VISIBLE_DEVICES=1 /srv/home/bohanlyu/miniconda3/envs/cogdoc/bin/vllm serve results/checkpoints/qwen_0_5b_probe_guided_bridge_a025_b100 --served-model-name candidate_qwen_0_5b_probe_guided_bridge_a025_b100 --host 127.0.0.1 --port 8100 --dtype bfloat16 --tensor-parallel-size 1
```

```bash
python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8100/v1 --models candidate_qwen_0_5b_probe_guided_bridge_a025_b100 --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/qwen_0_5b_probe_guided_bridge_a025_b100
```

## vLLM Eval Result

真实 endpoint eval 已完成：avg primary `0.203`，worst primary `0.000`。相对 uniform average，avg primary 提升 `0.023`；相对最佳源模型仍低 `0.172`。

| task | score | uniform score | delta vs uniform |
| --- | ---: | ---: | ---: |
| gsm8k | 0.062 | 0.000 | 0.062 |
| mmlu | 0.250 | 0.219 | 0.031 |
| safety | 0.500 | 0.500 | 0.000 |
| humaneval_compile | 0.000 | 0.000 | 0.000 |

这说明 NLL-grid probe 选出的 bridge 确实避开了最差 midpoint：GSM8K 和 MMLU 都比 uniform 回升，safety 的 unsafe refusal 从 `0.000` 回到 `0.500`。但 compile 仍是 `0.000`，整体仍低于 instruct/base；所以 global scalar coefficient 只能算第一层 gate，下一步需要 layer/module-wise 权重。

## Artifacts

- Summary: `results/probe_guided_dense_average_candidate/summary.json`
- Candidate table: `results/probe_guided_dense_average_candidate/candidate_scores.csv`
- Writer command: `results/probe_guided_dense_average_candidate/writer_command.txt`
