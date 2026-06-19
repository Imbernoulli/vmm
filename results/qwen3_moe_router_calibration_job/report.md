# Qwen3 MoE Router Calibration Job

这个作业把已完成的 router calibration smoke 扩展到真实 Qwen3 candidate：先在 `searched_no_gt065` checkpoint 上收集 student router hidden states，用 Coder source 作为 teacher router logits，再训练 capped route-KD router delta，最后写出 router-calibrated ablation checkpoint 并进入 vLLM 下游评测。

- Status: `job_ready_awaiting_gpu`
- Student checkpoint exists: `True`
- Teacher checkpoint exists: `True`
- Prompt pack exists: `True`
- Local GPU status: `unavailable`
- Router caps: `0.01, 0.025, 0.05`
- Baseline eval dir: `results/vllm_checkpoint_eval/qwen3_moe_searched_no_gt065_max_retention_candidate`
- Source control count: `2`
- Candidate count: `3`
- Selection output: `results/qwen3_moe_router_calibration_selection/summary.json`

## Why This Ablation

Direct Instruct/Coder router weight movement was rejected by the router move gate. This job tests a narrower mechanism: keep the best frozen-router expert candidate fixed, then add a small route-KD router delta learned from real hidden states and teacher logits. If downstream scores improve without routing collapse, router calibration becomes a valid next component; otherwise the unified rule keeps router frozen.

## Source Controls

| method | checkpoint | port | eval output |
|---|---|---:|---|
| `source_qwen3_30b_instruct` | `/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` | 8100 | `results/vllm_checkpoint_eval/source_qwen3_30b_instruct` |
| `source_qwen3_30b_coder` | `/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120` | 8101 | `results/vllm_checkpoint_eval/source_qwen3_30b_coder` |

## Candidates

| cap | method | checkpoint | port |
|---:|---|---|---:|
| 0.010 | `qwen3_moe_router_calibrated_searched_no_gt065_cap001_candidate` | `results/checkpoints/qwen3_moe_router_calibrated_searched_no_gt065_cap001_candidate` | 8108 |
| 0.025 | `qwen3_moe_router_calibrated_searched_no_gt065_cap0025_candidate` | `results/checkpoints/qwen3_moe_router_calibrated_searched_no_gt065_cap0025_candidate` | 8109 |
| 0.050 | `qwen3_moe_router_calibrated_searched_no_gt065_cap005_candidate` | `results/checkpoints/qwen3_moe_router_calibrated_searched_no_gt065_cap005_candidate` | 8110 |

## Stages

| stage | cap | expected output |
|---|---:|---|
| `collect_router_cache` | `shared` | `results/qwen3_moe_router_calibration_job/cache/router_calibration_cache.pt` |
| `vllm_eval_source_control` | `source` | `results/vllm_checkpoint_eval/source_qwen3_30b_instruct` |
| `vllm_eval_source_control` | `source` | `results/vllm_checkpoint_eval/source_qwen3_30b_coder` |
| `vllm_eval_baseline` | `baseline` | `results/vllm_checkpoint_eval/qwen3_moe_searched_no_gt065_max_retention_candidate` |
| `train_router_delta` | `cap001` | `results/qwen3_moe_router_calibration_job/delta_cap001/router_delta.safetensors` |
| `materialize_checkpoint` | `cap001` | `results/checkpoints/qwen3_moe_router_calibrated_searched_no_gt065_cap001_candidate` |
| `audit_delta` | `cap001` | `results/qwen3_moe_router_calibration_job/audit_cap001` |
| `vllm_eval` | `cap001` | `results/vllm_checkpoint_eval/qwen3_moe_router_calibrated_searched_no_gt065_cap001_candidate` |
| `train_router_delta` | `cap0025` | `results/qwen3_moe_router_calibration_job/delta_cap0025/router_delta.safetensors` |
| `materialize_checkpoint` | `cap0025` | `results/checkpoints/qwen3_moe_router_calibrated_searched_no_gt065_cap0025_candidate` |
| `audit_delta` | `cap0025` | `results/qwen3_moe_router_calibration_job/audit_cap0025` |
| `vllm_eval` | `cap0025` | `results/vllm_checkpoint_eval/qwen3_moe_router_calibrated_searched_no_gt065_cap0025_candidate` |
| `train_router_delta` | `cap005` | `results/qwen3_moe_router_calibration_job/delta_cap005/router_delta.safetensors` |
| `materialize_checkpoint` | `cap005` | `results/checkpoints/qwen3_moe_router_calibrated_searched_no_gt065_cap005_candidate` |
| `audit_delta` | `cap005` | `results/qwen3_moe_router_calibration_job/audit_cap005` |
| `vllm_eval` | `cap005` | `results/vllm_checkpoint_eval/qwen3_moe_router_calibrated_searched_no_gt065_cap005_candidate` |
| `select_router_calibration_result` | `all` | `results/qwen3_moe_router_calibration_selection/summary.json` |

## Run

```bash
results/qwen3_moe_router_calibration_job/run_router_calibration_job.sh all
```

## Outputs

- `source_control_plan`: `results/qwen3_moe_router_calibration_job/source_control_plan.csv`
- `candidate_plan`: `results/qwen3_moe_router_calibration_job/candidate_plan.csv`
- `stage_plan`: `results/qwen3_moe_router_calibration_job/stage_plan.csv`
- `run_script`: `results/qwen3_moe_router_calibration_job/run_router_calibration_job.sh`
- `selection`: `results/qwen3_moe_router_calibration_selection/summary.json`
- `summary`: `results/qwen3_moe_router_calibration_job/summary.json`
- `report`: `results/qwen3_moe_router_calibration_job/report.md`
