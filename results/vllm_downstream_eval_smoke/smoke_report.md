# vLLM Downstream Eval Contract Smoke

这个 smoke 启动一个本地 OpenAI-compatible mock endpoint，并通过真实 HTTP 调用 `run_vllm_downstream_eval.py`，验证下游评测 harness 的请求、解析、打分、排序和产物写出路径。

- Status: `passed`
- Good model avg primary: `1.000`
- Bad model avg primary: `0.000`
- Metric rows: `8`

## Files

- `results/vllm_downstream_eval_smoke/metrics.csv`
- `results/vllm_downstream_eval_smoke/model_summary.csv`
- `results/vllm_downstream_eval_smoke/smoke_summary.json`
