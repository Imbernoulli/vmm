# vLLM Downstream Eval

Status: `endpoint_unavailable`.

- Base URL: `http://127.0.0.1:8000/v1`
- Models: `Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-Coder-7B-Instruct, Qwen/Qwen2.5-Math-7B-Instruct, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, Qwen/Qwen3-30B-A3B-Base, Qwen/Qwen3-30B-A3B, Qwen/Qwen3-Coder-30B-A3B-Instruct`
- Error: `URLError: <urlopen error [Errno 1] Operation not permitted>`

Start a vLLM OpenAI-compatible server and rerun this script to produce real downstream task metrics.

## Eval Plan

| order | method | served model id | source |
| ---: | --- | --- | --- |
| 0 | Qwen/Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | results/qwen_target_model_registry/model_registry.csv |
| 1 | Qwen/Qwen2.5-Coder-7B-Instruct | `Qwen/Qwen2.5-Coder-7B-Instruct` | results/qwen_target_model_registry/model_registry.csv |
| 2 | Qwen/Qwen2.5-Math-7B-Instruct | `Qwen/Qwen2.5-Math-7B-Instruct` | results/qwen_target_model_registry/model_registry.csv |
| 3 | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | results/qwen_target_model_registry/model_registry.csv |
| 4 | Qwen/Qwen3-30B-A3B-Base | `Qwen/Qwen3-30B-A3B-Base` | results/qwen_target_model_registry/model_registry.csv |
| 5 | Qwen/Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B` | results/qwen_target_model_registry/model_registry.csv |
| 6 | Qwen/Qwen3-Coder-30B-A3B-Instruct | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | results/qwen_target_model_registry/model_registry.csv |
