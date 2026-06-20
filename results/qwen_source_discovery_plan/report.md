# Qwen Source Discovery Plan

## Result

当前最好的 measured source set 是 `coder+thinking`，source weights `{"coder": 0.6666666666666666, "thinking": 0.3333333333333333}`；它的 endpoint frontier avg gain 是 `0.0083`，但已观测 merge interference budget 是 `0.0694`，还差 `0.0611`。任务级 blocker 是 `gsm8k`，需要额外 `0.0694`。所以它只能进入 probe-only / endpoint-expansion，不能作为 final average budget。

下一批 source discovery 的最高优先场景是 `dense_7b_general_code_math_reasoning`：`run_endpoint_eval_plus_connectivity_probe_then_surplus_gate`。这一步的目标不是机械比较算法，而是找到 `source_frontier_gain - observed_merge_interference_budget >= 0` 的源集合，然后再做同构 average。

## Planned Source Sets

| candidate | scenario | gate | mechanism | extra gain needed | next action |
| --- | --- | --- | --- | ---: | --- |
| `qwen3_moe_coder_thinking_frontier_weighted_probe` | `measured_qwen3_moe_source_set` | `ready_for_endpoint_expansion_probe_only` | measured_complementarity_below_interference_budget | 0.0611 | expand_endpoint_eval_and_small_weighted_probe_before_final_average |
| `dense_7b_general_code_math_reasoning_planned_source_set` | `dense_7b_general_code_math_reasoning` | `ready_for_endpoint_eval_and_surplus_gate` | connectivity_barrier_then_source_surplus_gate | 0.0694 | run_endpoint_eval_plus_connectivity_probe_then_surplus_gate |
| `moe_30b_general_code_route_aware_planned_source_set` | `moe_30b_general_code_route_aware` | `ready_for_endpoint_eval_and_surplus_gate` | router_expert_probe_then_source_surplus_gate | 0.0694 | run_endpoint_eval_plus_router_expert_probe_then_surplus_gate |
| `dense_32b_reasoning_long_reasoning_planned_source_set` | `dense_32b_reasoning_long_reasoning` | `ready_after_p0_wave` | connectivity_barrier_then_source_surplus_gate | 0.0694 | run_endpoint_eval_plus_connectivity_probe_then_surplus_gate |
| `moe_30b_downstream_adapter_average_planned_source_set` | `moe_30b_downstream_adapter_average` | `manual_model_resolution_first` | router_expert_probe_then_source_surplus_gate | 0.0694 | resolve_public_same_shape_weight_ids_before_endpoint_eval |
| `dense_7b_domain_extension_planned_source_set` | `dense_7b_domain_extension` | `manual_model_resolution_first` | connectivity_barrier_then_source_surplus_gate | 0.0694 | resolve_public_same_shape_weight_ids_before_endpoint_eval |

## Task Gap Targets

| task | capability | frontier source | gain | additional needed | status | next action |
| --- | --- | --- | ---: | ---: | --- | --- |
| `gsm8k` | `math_reasoning` | `coder` | 0.0000 | 0.0694 | `no_task_frontier_gain` | find_or_eval_a_source_that_beats_the_current_best_single_on_this_task |
| `humaneval` | `code_generation_agentic` | `coder` | 0.0000 | 0.0694 | `no_task_frontier_gain` | find_or_eval_a_source_that_beats_the_current_best_single_on_this_task |
| `mmlu` | `general_knowledge_instruction` | `thinking` | 0.0250 | 0.0444 | `gain_below_interference_budget` | expand_endpoint_eval_and_search_a_stronger_specialist_for_this_task |

## Scenario Priority

| rank | scenario | priority | ready/manual | mechanism focus | next action |
| ---: | --- | ---: | --- | --- | --- |
| 1 | `dense_7b_general_code_math_reasoning` | 0.95 | 4/0 | connectivity_barrier_then_source_surplus_gate | run_endpoint_eval_plus_connectivity_probe_then_surplus_gate |
| 2 | `moe_30b_general_code_route_aware` | 0.95 | 3/0 | router_expert_probe_then_source_surplus_gate | run_endpoint_eval_plus_router_expert_probe_then_surplus_gate |
| 3 | `dense_32b_reasoning_long_reasoning` | 0.72 | 4/1 | connectivity_barrier_then_source_surplus_gate | run_endpoint_eval_plus_connectivity_probe_then_surplus_gate |
| 4 | `moe_30b_downstream_adapter_average` | 0.45 | 0/2 | router_expert_probe_then_source_surplus_gate | resolve_public_same_shape_weight_ids_before_endpoint_eval |
| 5 | `dense_7b_domain_extension` | 0.44 | 0/2 | connectivity_barrier_then_source_surplus_gate | resolve_public_same_shape_weight_ids_before_endpoint_eval |
| 6 | `negative_controls` | 0.20 | 0/0 | negative_control_endpoint_and_uniform_average | keep_endpoint_uniform_and_writer_controls_labeled |

## Endpoint Eval Expansion

| scenario | capability | benchmarks | metric | min examples | runner |
| --- | --- | --- | --- | ---: | --- |
| `measured_qwen3_moe_source_set` | `math_reasoning` | GSM8K, MATH-500, AIME slice | exact_match, gold-answer NLL, CoT/TIR format_success | 256 | `endpoint_expansion_or_probe_only` |
| `measured_qwen3_moe_source_set` | `code_generation_agentic` | HumanEval, MBPP, LiveCodeBench, repo-level/tool-call slice | pass@1, canonical solution NLL, syntax/test failure type, tool-call format_success | 128 | `endpoint_expansion_or_probe_only` |
| `measured_qwen3_moe_source_set` | `general_knowledge_instruction` | MMLU, C-Eval, CMMLU, IFEval, small chat-format pack | accuracy, response-only NLL, format_success, general_retention | 256 | `endpoint_expansion_or_probe_only` |
| `measured_qwen3_moe_source_set` | `general_knowledge_instruction` | MMLU, C-Eval, CMMLU, IFEval, small chat-format pack | accuracy, response-only NLL, format_success, general_retention | 256 | `endpoint_expansion_or_probe_only` |
| `measured_qwen3_moe_source_set` | `math_reasoning` | GSM8K, MATH-500, AIME slice | exact_match, gold-answer NLL, CoT/TIR format_success | 256 | `endpoint_expansion_or_probe_only` |
| `measured_qwen3_moe_source_set` | `code_generation_agentic` | HumanEval, MBPP, LiveCodeBench, repo-level/tool-call slice | pass@1, canonical solution NLL, syntax/test failure type, tool-call format_success | 128 | `endpoint_expansion_or_probe_only` |
| `measured_qwen3_moe_source_set` | `moe_routing` | general/math/code/domain route-probe prompt pack | router entropy, top-k expert frequency, route overlap, expert load balance | 64 | `endpoint_expansion_or_probe_only` |
| `measured_qwen3_moe_source_set` | `connectivity_geometry` | anchor-to-expert lambda sweeps; pairwise alpha/beta planes | barrier, worst_NLL, plane residual, delta cosine, sign/weighted conflict | 64 | `endpoint_expansion_or_probe_only` |
| `dense_7b_general_code_math_reasoning` | `general_knowledge_instruction` | MMLU, C-Eval, CMMLU, IFEval, small chat-format pack | accuracy, response-only NLL, format_success, general_retention | 256 | `vllm_endpoint_eval` |
| `dense_7b_general_code_math_reasoning` | `math_reasoning` | GSM8K, MATH-500, AIME slice | exact_match, gold-answer NLL, CoT/TIR format_success | 256 | `vllm_endpoint_eval` |
| `dense_7b_general_code_math_reasoning` | `code_generation_agentic` | HumanEval, MBPP, LiveCodeBench, repo-level/tool-call slice | pass@1, canonical solution NLL, syntax/test failure type, tool-call format_success | 128 | `vllm_endpoint_eval` |
| `dense_7b_general_code_math_reasoning` | `safety_refusal` | BeaverTails, AdvBench, safe/unsafe refusal prompt pack | safe_response NLL, unsafe_refusal NLL, over-trigger/under-trigger rate | 128 | `vllm_endpoint_eval` |
| `dense_7b_general_code_math_reasoning` | `connectivity_geometry` | anchor-to-expert lambda sweeps; pairwise alpha/beta planes | barrier, worst_NLL, plane residual, delta cosine, sign/weighted conflict | 64 | `probe_then_eval` |
| `moe_30b_general_code_route_aware` | `general_knowledge_instruction` | MMLU, C-Eval, CMMLU, IFEval, small chat-format pack | accuracy, response-only NLL, format_success, general_retention | 256 | `vllm_endpoint_eval` |
| `moe_30b_general_code_route_aware` | `math_reasoning` | GSM8K, MATH-500, AIME slice | exact_match, gold-answer NLL, CoT/TIR format_success | 256 | `vllm_endpoint_eval` |
| `moe_30b_general_code_route_aware` | `code_generation_agentic` | HumanEval, MBPP, LiveCodeBench, repo-level/tool-call slice | pass@1, canonical solution NLL, syntax/test failure type, tool-call format_success | 128 | `vllm_endpoint_eval` |
| `moe_30b_general_code_route_aware` | `safety_refusal` | BeaverTails, AdvBench, safe/unsafe refusal prompt pack | safe_response NLL, unsafe_refusal NLL, over-trigger/under-trigger rate | 128 | `vllm_endpoint_eval` |
| `moe_30b_general_code_route_aware` | `connectivity_geometry` | anchor-to-expert lambda sweeps; pairwise alpha/beta planes | barrier, worst_NLL, plane residual, delta cosine, sign/weighted conflict | 64 | `probe_then_eval` |
| `moe_30b_general_code_route_aware` | `moe_routing` | general/math/code/domain route-probe prompt pack | router entropy, top-k expert frequency, route overlap, expert load balance | 64 | `probe_then_eval` |
| `dense_32b_reasoning_long_reasoning` | `general_knowledge_instruction` | MMLU, C-Eval, CMMLU, IFEval, small chat-format pack | accuracy, response-only NLL, format_success, general_retention | 256 | `vllm_endpoint_eval` |
| `dense_32b_reasoning_long_reasoning` | `math_reasoning` | GSM8K, MATH-500, AIME slice | exact_match, gold-answer NLL, CoT/TIR format_success | 256 | `vllm_endpoint_eval` |
| `dense_32b_reasoning_long_reasoning` | `code_generation_agentic` | HumanEval, MBPP, LiveCodeBench, repo-level/tool-call slice | pass@1, canonical solution NLL, syntax/test failure type, tool-call format_success | 128 | `vllm_endpoint_eval` |
| `dense_32b_reasoning_long_reasoning` | `reasoning_style` | GPQA-Diamond slice, AIME, long CoT prompt pack | answer accuracy, chain length, think-tag compliance, language-mix rate | 128 | `vllm_endpoint_eval` |
| `dense_32b_reasoning_long_reasoning` | `safety_refusal` | BeaverTails, AdvBench, safe/unsafe refusal prompt pack | safe_response NLL, unsafe_refusal NLL, over-trigger/under-trigger rate | 128 | `vllm_endpoint_eval` |
| `dense_32b_reasoning_long_reasoning` | `connectivity_geometry` | anchor-to-expert lambda sweeps; pairwise alpha/beta planes | barrier, worst_NLL, plane residual, delta cosine, sign/weighted conflict | 64 | `probe_then_eval` |
| `moe_30b_downstream_adapter_average` | `domain_finance_or_other_lab_finetune` | CFLUE, FinQA, CCC, or candidate-specific domain suite | domain exact/graded score, compliance format, held-in_retention | 128 | `vllm_endpoint_eval` |
| `moe_30b_downstream_adapter_average` | `general_knowledge_instruction` | MMLU, C-Eval, CMMLU, IFEval, small chat-format pack | accuracy, response-only NLL, format_success, general_retention | 256 | `vllm_endpoint_eval` |
| `moe_30b_downstream_adapter_average` | `safety_refusal` | BeaverTails, AdvBench, safe/unsafe refusal prompt pack | safe_response NLL, unsafe_refusal NLL, over-trigger/under-trigger rate | 128 | `vllm_endpoint_eval` |
| `moe_30b_downstream_adapter_average` | `connectivity_geometry` | anchor-to-expert lambda sweeps; pairwise alpha/beta planes | barrier, worst_NLL, plane residual, delta cosine, sign/weighted conflict | 64 | `probe_then_eval` |
| `moe_30b_downstream_adapter_average` | `moe_routing` | general/math/code/domain route-probe prompt pack | router entropy, top-k expert frequency, route overlap, expert load balance | 64 | `probe_then_eval` |
| `dense_7b_domain_extension` | `domain_finance_or_other_lab_finetune` | CFLUE, FinQA, CCC, or candidate-specific domain suite | domain exact/graded score, compliance format, held-in_retention | 128 | `vllm_endpoint_eval` |
| `dense_7b_domain_extension` | `general_knowledge_instruction` | MMLU, C-Eval, CMMLU, IFEval, small chat-format pack | accuracy, response-only NLL, format_success, general_retention | 256 | `vllm_endpoint_eval` |
| `dense_7b_domain_extension` | `safety_refusal` | BeaverTails, AdvBench, safe/unsafe refusal prompt pack | safe_response NLL, unsafe_refusal NLL, over-trigger/under-trigger rate | 128 | `vllm_endpoint_eval` |
| `dense_7b_domain_extension` | `connectivity_geometry` | anchor-to-expert lambda sweeps; pairwise alpha/beta planes | barrier, worst_NLL, plane residual, delta cosine, sign/weighted conflict | 64 | `probe_then_eval` |

## Source Discovery Queue

| rank | item | status | priority | command | expected update |
| ---: | --- | --- | ---: | --- | --- |
| 1 | `measured_coder_thinking_endpoint_expansion` | `ready_endpoint_expansion_probe_only` | 0.98 | `python scripts/run_vllm_downstream_eval.py --models SERVED_CODER,SERVED_THINKING --tasks gsm8k,humaneval_compile,mmlu --max-examples 256 --output-dir results/qwen_source_discovery_plan/measured_coder_thinking_vllm` | move Coder+Thinking from probe-only to final-budget only if surplus becomes non-negative |
| 2 | `dense_7b_general_code_math_reasoning_endpoint_frontier` | `ready_for_endpoint_eval_and_surplus_gate` | 0.95 | `python scripts/run_vllm_downstream_eval.py --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-Coder-7B-Instruct,Qwen/Qwen2.5-Math-7B-Instruct,deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --tasks mmlu,gsm8k,humaneval_compile,safety --max-examples 256 --output-dir results/qwen_source_discovery_plan/vllm_endpoint_eval` | endpoint frontier plus connectivity plane decide if Dense source set can enter average budget |
| 3 | `moe_30b_general_code_route_aware_endpoint_frontier` | `ready_for_endpoint_eval_and_surplus_gate` | 0.95 | `python scripts/run_vllm_downstream_eval.py --models Qwen/Qwen3-30B-A3B-Base,Qwen/Qwen3-30B-A3B,Qwen/Qwen3-Coder-30B-A3B-Instruct --tasks mmlu,gsm8k,humaneval_compile --max-examples 256 --output-dir results/qwen_source_discovery_plan/vllm_endpoint_eval` | router/expert probes plus endpoint frontier decide if MoE source set can enter surplus optimizer |
| 4 | `dense_32b_reasoning_long_reasoning_endpoint_frontier` | `ready_after_p0_wave` | 0.72 | `python scripts/run_vllm_downstream_eval.py --models Qwen/Qwen2.5-32B,Qwen/Qwen2.5-32B-Instruct,a-m-team/AM-Thinking-v1,deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tasks mmlu,gsm8k,humaneval_compile,safety --max-examples 256 --output-dir results/qwen_source_discovery_plan/vllm_endpoint_eval` | endpoint frontier plus connectivity plane decide if Dense source set can enter average budget |
| 5 | `moe_30b_downstream_adapter_average_resolve_weights` | `manual_model_resolution_first` | 0.40 | `python scripts/build_qwen_target_model_registry.py --output-dir results/qwen_target_model_registry` | replace paper/pool placeholders with concrete same-shape endpoints or adapters |
| 6 | `dense_7b_domain_extension_resolve_weights` | `manual_model_resolution_first` | 0.39 | `python scripts/build_qwen_target_model_registry.py --output-dir results/qwen_target_model_registry` | replace paper/pool placeholders with concrete same-shape endpoints or adapters |

## Literature Priors

| key | source | mechanism used here |
| --- | --- | --- |
| `mode_connectivity` | https://arxiv.org/abs/1802.10026 | Only average inside a measured low-loss component; source identity alone is not enough. |
| `model_soups` | https://arxiv.org/abs/2203.05482 | Average same-basin finetunes, but keep endpoint fallback and held-out selection. |
| `ties` | https://arxiv.org/abs/2306.01708 | Task-vector interference means endpoint complementarity must be larger than merge damage. |
| `expert_merging` | https://arxiv.org/abs/2509.25712 | MoE coefficients should follow calibration behavior instead of one global average. |
| `sub_moe` | https://arxiv.org/abs/2506.23266 | Expert similarity, expert output behavior, and subspace conflict are primary MoE probes. |
| `harc` | https://arxiv.org/abs/2606.03391 | Router top-k boundary movement must be audited before any MoE router change is accepted. |

## Outputs

- `results/qwen_source_discovery_plan/candidate_source_sets.csv`
- `results/qwen_source_discovery_plan/task_gap_targets.csv`
- `results/qwen_source_discovery_plan/endpoint_eval_expansion.csv`
- `results/qwen_source_discovery_plan/scenario_priority.csv`
- `results/qwen_source_discovery_plan/source_discovery_queue.csv`
- `results/qwen_source_discovery_plan/summary.json`
- `results/qwen_source_discovery_plan/report.md`
