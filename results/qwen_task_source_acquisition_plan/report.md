# Qwen Task Source Acquisition Plan

这个 artifact 把 task-level source gap 翻译成具体候选源模型和 vLLM jobs。它的目的不是直接平均更多模型，而是先证明每个 same-shape group 的 endpoint frontier 有足够 task surplus。

- Status: `task_source_acquisition_plan_ready`
- Task gaps: `3`
- Candidate rows: `35`
- Eval jobs: `8`
- Top job: `gsm8k_qwen3_moe_30b_a3b_source_acquisition`
- Top task: `gsm8k`

## Eval Jobs

| job | task | group | models | tasks | needed | command |
| --- | --- | --- | ---: | --- | ---: | --- |
| `gsm8k_qwen3_moe_30b_a3b_source_acquisition` | `gsm8k` | `qwen3_moe_30b_a3b` | 3 | `gsm8k,mmlu` | 0.0694 | `python scripts/run_vllm_downstream_eval.py --base-url http://HOST:PORT/v1 --models Qwen/Qwen3-30B-A3B-Thinking-2507,Qwen/Qwen3-30B-A3B,Qwen/Qwen3-30B-A3B-Base --tasks gsm8k,mmlu --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_task_source_acquisition_plan/task_manifests/gsm8k_qwen3_moe_30b_a3b_source_acquisition_task_manifest.json --create-task-manifest-if-missing --output-dir results/qwen_task_source_acquisition_plan/vllm_eval/gsm8k_qwen3_moe_30b_a3b_source_acquisition` |
| `humaneval_qwen3_moe_30b_a3b_source_acquisition` | `humaneval` | `qwen3_moe_30b_a3b` | 4 | `humaneval_compile,mmlu` | 0.0694 | `python scripts/run_vllm_downstream_eval.py --base-url http://HOST:PORT/v1 --models Qwen/Qwen3-Coder-30B-A3B-Instruct,Qwen/Qwen3-30B-A3B,Qwen/Qwen3-30B-A3B-Thinking-2507,Qwen/Qwen3-30B-A3B-Base --tasks humaneval_compile,mmlu --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_task_source_acquisition_plan/task_manifests/humaneval_qwen3_moe_30b_a3b_source_acquisition_task_manifest.json --create-task-manifest-if-missing --output-dir results/qwen_task_source_acquisition_plan/vllm_eval/humaneval_qwen3_moe_30b_a3b_source_acquisition` |
| `gsm8k_qwen2_dense_32b_source_acquisition` | `gsm8k` | `qwen2_dense_32b` | 3 | `gsm8k,mmlu` | 0.0694 | `python scripts/run_vllm_downstream_eval.py --base-url http://HOST:PORT/v1 --models a-m-team/AM-Thinking-v1,deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,Qwen/Qwen2.5-32B-Instruct --tasks gsm8k,mmlu --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_task_source_acquisition_plan/task_manifests/gsm8k_qwen2_dense_32b_source_acquisition_task_manifest.json --create-task-manifest-if-missing --output-dir results/qwen_task_source_acquisition_plan/vllm_eval/gsm8k_qwen2_dense_32b_source_acquisition` |
| `gsm8k_qwen2_dense_7b_source_acquisition` | `gsm8k` | `qwen2_dense_7b` | 3 | `gsm8k,mmlu` | 0.0694 | `python scripts/run_vllm_downstream_eval.py --base-url http://HOST:PORT/v1 --models Qwen/Qwen2.5-Math-7B-Instruct,deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,Qwen/Qwen2.5-Coder-7B-Instruct --tasks gsm8k,mmlu --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_task_source_acquisition_plan/task_manifests/gsm8k_qwen2_dense_7b_source_acquisition_task_manifest.json --create-task-manifest-if-missing --output-dir results/qwen_task_source_acquisition_plan/vllm_eval/gsm8k_qwen2_dense_7b_source_acquisition` |
| `humaneval_qwen2_dense_7b_source_acquisition` | `humaneval` | `qwen2_dense_7b` | 2 | `humaneval_compile,mmlu` | 0.0694 | `python scripts/run_vllm_downstream_eval.py --base-url http://HOST:PORT/v1 --models Qwen/Qwen2.5-Coder-7B-Instruct,Qwen/Qwen2.5-Math-7B-Instruct --tasks humaneval_compile,mmlu --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_task_source_acquisition_plan/task_manifests/humaneval_qwen2_dense_7b_source_acquisition_task_manifest.json --create-task-manifest-if-missing --output-dir results/qwen_task_source_acquisition_plan/vllm_eval/humaneval_qwen2_dense_7b_source_acquisition` |
| `mmlu_qwen3_moe_30b_a3b_source_acquisition` | `mmlu` | `qwen3_moe_30b_a3b` | 3 | `mmlu,gsm8k,humaneval_compile` | 0.0444 | `python scripts/run_vllm_downstream_eval.py --base-url http://HOST:PORT/v1 --models Qwen/Qwen3-30B-A3B,Qwen/Qwen3-30B-A3B-Thinking-2507,Qwen/Qwen3-30B-A3B-Base --tasks mmlu,gsm8k,humaneval_compile --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_task_source_acquisition_plan/task_manifests/mmlu_qwen3_moe_30b_a3b_source_acquisition_task_manifest.json --create-task-manifest-if-missing --output-dir results/qwen_task_source_acquisition_plan/vllm_eval/mmlu_qwen3_moe_30b_a3b_source_acquisition` |
| `mmlu_qwen2_dense_32b_source_acquisition` | `mmlu` | `qwen2_dense_32b` | 3 | `mmlu,gsm8k,humaneval_compile` | 0.0444 | `python scripts/run_vllm_downstream_eval.py --base-url http://HOST:PORT/v1 --models Qwen/Qwen2.5-32B,Qwen/Qwen2.5-32B-Instruct,deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tasks mmlu,gsm8k,humaneval_compile --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_task_source_acquisition_plan/task_manifests/mmlu_qwen2_dense_32b_source_acquisition_task_manifest.json --create-task-manifest-if-missing --output-dir results/qwen_task_source_acquisition_plan/vllm_eval/mmlu_qwen2_dense_32b_source_acquisition` |
| `mmlu_qwen2_dense_7b_source_acquisition` | `mmlu` | `qwen2_dense_7b` | 3 | `mmlu,gsm8k,humaneval_compile` | 0.0444 | `python scripts/run_vllm_downstream_eval.py --base-url http://HOST:PORT/v1 --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-Coder-7B-Instruct,deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --tasks mmlu,gsm8k,humaneval_compile --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_task_source_acquisition_plan/task_manifests/mmlu_qwen2_dense_7b_source_acquisition_task_manifest.json --create-task-manifest-if-missing --output-dir results/qwen_task_source_acquisition_plan/vllm_eval/mmlu_qwen2_dense_7b_source_acquisition` |

## Top Candidate Sources

| task | group | model | role | score | status |
| --- | --- | --- | --- | ---: | --- |
| `gsm8k` | `qwen2_dense_32b` | `a-m-team/AM-Thinking-v1` | `reasoning_rl_expert` | 1.3600 | `ready_for_topology_inspect` |
| `gsm8k` | `qwen2_dense_32b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | `reasoning_distill_expert` | 1.3600 | `ready_for_topology_inspect` |
| `gsm8k` | `qwen2_dense_32b` | `Long1K-32B` | `long_reasoning_expert` | 1.0000 | `manual_weight_id_resolution_required` |
| `gsm8k` | `qwen2_dense_32b` | `Qwen/Qwen2.5-32B-Instruct` | `general_instruction_anchor` | 0.8700 | `ready_for_topology_inspect` |
| `gsm8k` | `qwen2_dense_32b` | `DianJin-R1-32B` | `domain_finance_expert` | 0.6100 | `manual_weight_id_resolution_required` |
| `gsm8k` | `qwen2_dense_7b` | `Qwen/Qwen2.5-Math-7B-Instruct` | `math_expert` | 1.3800 | `ready_for_topology_inspect` |
| `gsm8k` | `qwen2_dense_7b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `reasoning_distill_expert` | 1.3600 | `ready_for_topology_inspect` |
| `gsm8k` | `qwen2_dense_7b` | `Qwen/Qwen2.5-Coder-7B-Instruct` | `code_expert` | 0.9900 | `ready_for_topology_inspect` |
| `gsm8k` | `qwen3_moe_30b_a3b` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | `moe_thinking_reasoning_anchor` | 1.3100 | `ready_for_endpoint_eval` |
| `gsm8k` | `qwen3_moe_30b_a3b` | `Qwen/Qwen3-30B-A3B` | `moe_general_anchor` | 1.1400 | `ready_for_topology_inspect` |
| `gsm8k` | `qwen3_moe_30b_a3b` | `Qwen/Qwen3-30B-A3B-Base` | `moe_base` | 0.9700 | `ready_for_topology_inspect` |
| `gsm8k` | `qwen3_moe_30b_a3b` | `hf_tree:Qwen/Qwen3-30B-A3B` | `moe_downstream_adapter_pool` | 0.4700 | `manual_candidate_selection_required` |
| `humaneval` | `qwen2_dense_32b` | `a-m-team/AM-Thinking-v1` | `reasoning_rl_expert` | 1.1400 | `ready_for_topology_inspect` |
| `humaneval` | `qwen2_dense_32b` | `Long1K-32B` | `long_reasoning_expert` | 0.6100 | `manual_weight_id_resolution_required` |
| `humaneval` | `qwen2_dense_7b` | `Qwen/Qwen2.5-Coder-7B-Instruct` | `code_expert` | 1.3800 | `ready_for_topology_inspect` |
| `humaneval` | `qwen2_dense_7b` | `Qwen/Qwen2.5-Math-7B-Instruct` | `math_expert` | 1.1600 | `ready_for_topology_inspect` |
| `humaneval` | `qwen3_moe_30b_a3b` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `moe_code_agent_expert` | 1.4800 | `ready_for_topology_inspect` |
| `humaneval` | `qwen3_moe_30b_a3b` | `Qwen/Qwen3-30B-A3B` | `moe_general_anchor` | 1.3100 | `ready_for_topology_inspect` |
| `humaneval` | `qwen3_moe_30b_a3b` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | `moe_thinking_reasoning_anchor` | 1.1400 | `ready_for_endpoint_eval` |
| `humaneval` | `qwen3_moe_30b_a3b` | `Qwen/Qwen3-30B-A3B-Base` | `moe_base` | 0.9700 | `ready_for_topology_inspect` |
| `humaneval` | `qwen3_moe_30b_a3b` | `hf_tree:Qwen/Qwen3-Coder-30B-A3B-Instruct` | `moe_coder_downstream_adapter_pool` | 0.8100 | `manual_candidate_selection_required` |
| `humaneval` | `qwen3_moe_30b_a3b` | `hf_tree:Qwen/Qwen3-30B-A3B` | `moe_downstream_adapter_pool` | 0.4700 | `manual_candidate_selection_required` |
| `mmlu` | `qwen2_dense_32b` | `Qwen/Qwen2.5-32B` | `base_anchor` | 1.2900 | `ready_for_topology_inspect` |
| `mmlu` | `qwen2_dense_32b` | `Qwen/Qwen2.5-32B-Instruct` | `general_instruction_anchor` | 1.1200 | `ready_for_topology_inspect` |

## Acceptance Gates

| task | group | top candidate | gate | pass action | fail action |
| --- | --- | --- | --- | --- | --- |
| `gsm8k` | `qwen3_moe_30b_a3b` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | `awaiting_vllm_source_frontier_eval` | add_group_to_source_set_surplus_optimizer_then_run_same_shape_average_probe | do_not_spend_average_budget_on_this_group_for_this_task |
| `humaneval` | `qwen3_moe_30b_a3b` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `awaiting_vllm_source_frontier_eval` | add_group_to_source_set_surplus_optimizer_then_run_same_shape_average_probe | do_not_spend_average_budget_on_this_group_for_this_task |
| `gsm8k` | `qwen2_dense_32b` | `a-m-team/AM-Thinking-v1` | `awaiting_vllm_source_frontier_eval` | add_group_to_source_set_surplus_optimizer_then_run_same_shape_average_probe | do_not_spend_average_budget_on_this_group_for_this_task |
| `gsm8k` | `qwen2_dense_7b` | `Qwen/Qwen2.5-Math-7B-Instruct` | `awaiting_vllm_source_frontier_eval` | add_group_to_source_set_surplus_optimizer_then_run_same_shape_average_probe | do_not_spend_average_budget_on_this_group_for_this_task |
| `humaneval` | `qwen2_dense_7b` | `Qwen/Qwen2.5-Coder-7B-Instruct` | `awaiting_vllm_source_frontier_eval` | add_group_to_source_set_surplus_optimizer_then_run_same_shape_average_probe | do_not_spend_average_budget_on_this_group_for_this_task |
| `mmlu` | `qwen3_moe_30b_a3b` | `Qwen/Qwen3-30B-A3B` | `awaiting_vllm_source_frontier_eval` | add_group_to_source_set_surplus_optimizer_then_run_same_shape_average_probe | do_not_spend_average_budget_on_this_group_for_this_task |
| `mmlu` | `qwen2_dense_32b` | `Qwen/Qwen2.5-32B` | `awaiting_vllm_source_frontier_eval` | add_group_to_source_set_surplus_optimizer_then_run_same_shape_average_probe | do_not_spend_average_budget_on_this_group_for_this_task |
| `mmlu` | `qwen2_dense_7b` | `Qwen/Qwen2.5-7B-Instruct` | `awaiting_vllm_source_frontier_eval` | add_group_to_source_set_surplus_optimizer_then_run_same_shape_average_probe | do_not_spend_average_budget_on_this_group_for_this_task |

## Literature Hooks

- [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482): Weight averaging is most plausible when fine-tuned models occupy a shared low-loss basin and source endpoints have complementary validation signal.
- [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708): Sign and redundancy interference mean weak source complementarity should block final average promotion before delta surgery.
- [Expert Merging: Model Merging with Unsupervised Expert Alignment and Importance-Guided Layer Chunking](https://arxiv.org/abs/2509.25712): For MoE, source and task evidence should decide which same-shape expert families deserve layer/chunk coefficient calibration.
- [When Model Merging Breaks Routing: Training-Free Calibration for MoE](https://arxiv.org/abs/2606.03391): Router calibration is useful only after source/candidate eval shows the average is worth repairing.

## Outputs

- `report`: `results/qwen_task_source_acquisition_plan/report.md`
- `summary`: `results/qwen_task_source_acquisition_plan/summary.json`
- `candidate_task_sources`: `results/qwen_task_source_acquisition_plan/candidate_task_sources.csv`
- `vllm_source_acquisition_jobs`: `results/qwen_task_source_acquisition_plan/vllm_source_acquisition_jobs.csv`
- `source_acquisition_gates`: `results/qwen_task_source_acquisition_plan/source_acquisition_gates.csv`
- `literature_sources`: `results/qwen_task_source_acquisition_plan/literature_sources.json`
