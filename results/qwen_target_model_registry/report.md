# Qwen 目标模型、场景与评测 Probe Registry

生成时间：`2026-06-19T09:09:40.294017+00:00`

## 先给结论

第一轮真实实验建议从 `dense_7b_general_code_math_reasoning` 开始：`Qwen/Qwen2.5-7B-Instruct` 作为通用 anchor，加入官方 Coder、官方 Math、`DeepSeek-R1-Distill-Qwen-7B` 这三个同源专长/下游分支。

这比只比较 Base/Instruct 更接近真实下游用户场景：候选里显式包含第三方蒸馏、SFT/RL reasoning 模型、论文中的金融/长推理微调模型，以及 Hugging Face 模型树里的同构 adapters/finetunes 候选池。

Average 的硬约束保持不变：最终输出必须和输入模型同构。Dense 不能变成 ensemble；MoE 不能改变 layer 数、router shape、expert 数或每个 expert 的张量 shape。DianJin/Long1K 这类论文候选先标成需要人工确认权重 ID，避免把还未验证的模型当作可直接 materialize 的 checkpoint。

## Registry 规模

- 候选条目：`17`，其中 dense `12`，MoE `5`。
- 官方模型：`8`；第三方/下游/候选池：`9`。
- 可直接进入 topology inspect：`11`；需要先人工确认权重或具体候选：`6`。
- 场景：`6`；评测/probe 维度：`9`。

## P0 候选模型

| phase | role | model_id | source_type | merge_role | scenario | materialization_status |
| --- | --- | --- | --- | --- | --- | --- |
| dense_7b | general_instruction_anchor | Qwen/Qwen2.5-7B-Instruct | official_posttrained | anchor | dense_7b_general_code_math_reasoning | ready_for_topology_inspect |
| dense_7b | code_expert | Qwen/Qwen2.5-Coder-7B-Instruct | official_specialist | expert | dense_7b_general_code_math_reasoning | ready_for_topology_inspect |
| dense_7b | math_expert | Qwen/Qwen2.5-Math-7B-Instruct | official_specialist | expert | dense_7b_general_code_math_reasoning | ready_for_topology_inspect |
| dense_7b | reasoning_distill_expert | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | third_party_distill | downstream_expert | dense_7b_general_code_math_reasoning | ready_for_topology_inspect |
| dense_32b | base_anchor | Qwen/Qwen2.5-32B | official_base | base_or_delta_reference | dense_32b_reasoning_long_reasoning | ready_for_topology_inspect |
| dense_32b | general_instruction_anchor | Qwen/Qwen2.5-32B-Instruct | official_posttrained | anchor | dense_32b_reasoning_long_reasoning | ready_for_topology_inspect |
| dense_32b | reasoning_rl_expert | a-m-team/AM-Thinking-v1 | third_party_sft_rl | downstream_expert | dense_32b_reasoning_long_reasoning | ready_for_topology_inspect |
| dense_32b | reasoning_distill_expert | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | third_party_distill | downstream_expert | dense_32b_reasoning_long_reasoning | ready_for_topology_inspect |
| moe_30b_a3b | moe_base | Qwen/Qwen3-30B-A3B-Base | official_base_moe | base_or_delta_reference | moe_30b_general_code_route_aware | ready_for_topology_inspect |
| moe_30b_a3b | moe_general_anchor | Qwen/Qwen3-30B-A3B | official_posttrained_moe | anchor | moe_30b_general_code_route_aware | ready_for_topology_inspect |
| moe_30b_a3b | moe_code_agent_expert | Qwen/Qwen3-Coder-30B-A3B-Instruct | official_specialist_moe | expert | moe_30b_general_code_route_aware | ready_for_topology_inspect |

## 需要先确认的下游候选

| phase | role | model_id | source_type | topology_action | materialization_status |
| --- | --- | --- | --- | --- | --- |
| dense_7b | domain_finance_expert | DianJin-R1-7B | paper_downstream_finetune | resolve_public_weight_id_then_inspect_topology | manual_weight_id_resolution_required |
| dense_7b | user_lab_adapter_pool | hf_tree:Qwen/Qwen2.5-7B-Instruct | downstream_adapter_finetune_pool | select_same-shape_full_or_adapter_candidates_only | manual_candidate_selection_required |
| dense_32b | long_reasoning_expert | Long1K-32B | paper_downstream_finetune | resolve_public_weight_id_then_inspect_topology | manual_weight_id_resolution_required |
| dense_32b | domain_finance_expert | DianJin-R1-32B | paper_downstream_finetune | resolve_public_weight_id_then_inspect_topology | manual_weight_id_resolution_required |
| moe_30b_a3b | moe_downstream_adapter_pool | hf_tree:Qwen/Qwen3-30B-A3B | downstream_adapter_finetune_pool | select_same-shape_moe_or_adapter_candidates_only | manual_candidate_selection_required |
| moe_30b_a3b | moe_coder_downstream_adapter_pool | hf_tree:Qwen/Qwen3-Coder-30B-A3B-Instruct | downstream_adapter_finetune_pool | select_same-shape_moe_or_adapter_candidates_only | manual_candidate_selection_required |

## 场景矩阵

| scenario_id | priority | objective | first_average_candidate | success_gate |
| --- | --- | --- | --- | --- |
| dense_7b_general_code_math_reasoning | p0_first_wave | 合成一个同构 Qwen2.5 7B assistant，保留通用指令、代码、数学和 R1 蒸馏推理能力。 | coefficient_search_after_connectivity_gate | worst_score > uniform_average and each held-in retention >= 0.90 with general retention >= 0.95. |
| dense_7b_domain_extension | p1_after_7b_core | 加入金融/医疗/法律等实验室或行业微调分支，检验领域能力能否压回同构 7B checkpoint。 | anchor_plus_domain_delta_with_small_lambda_sweep | domain held-in improves over anchor while general/safety retention does not cross the drop threshold. |
| dense_32b_reasoning_long_reasoning | p1_scale_validation | 在 Qwen2.5 32B 上验证 reasoning RL、R1 distill 和长 CoT 分支是否更容易或更难平均。 | greedy_soup_or_task_arithmetic_only_inside_low_barrier_component | reasoning score improves over general anchor and no held-in branch drops below 0.90 retention. |
| moe_30b_general_code_route_aware | p0_moe_wave | 合成同 expert 数、同 router shape 的 Qwen3-30B-A3B route-aware MoE，保留通用和代码/agentic 能力。 | router_frozen_shared_merge_plus_expert_matched_average | route overlap and expert load stay within readiness gate while worst_score beats all-weight average. |
| moe_30b_downstream_adapter_average | p1_after_real_routing_probe | 选择真实下游 Qwen3 MoE adapters/finetunes，判断 adapter-level 或 expert-wise average 能否压回同构 MoE。 | same-shape_adapter_average_or_route_weighted_full_delta | final output remains one same-shape adapter or checkpoint and beats endpoint-only/negative baselines. |
| negative_controls | always | 保留朴素平均、端点、endpoint-only best grid 和 all-weight MoE average 作为负/对照项，避免误报。 | none_control_only | control remains explicitly labeled and is not selected unless it passes same-shape average criteria. |

## 评测与 Probe 矩阵

| capability | benchmark_slice | metric | held_in_source | dense_or_moe |
| --- | --- | --- | --- | --- |
| general_knowledge_instruction | MMLU, C-Eval, CMMLU, IFEval, small chat-format pack | accuracy, response-only NLL, format_success, general_retention | Qwen instruct/general anchors | dense_and_moe |
| math_reasoning | GSM8K, MATH-500, AIME slice | exact_match, gold-answer NLL, CoT/TIR format_success | Qwen2.5-Math and DeepSeek/AM/Long1K reasoning branches | dense_and_moe |
| code_generation_agentic | HumanEval, MBPP, LiveCodeBench, repo-level/tool-call slice | pass@1, canonical solution NLL, syntax/test failure type, tool-call format_success | Qwen2.5-Coder and Qwen3-Coder branches | dense_and_moe |
| reasoning_style | GPQA-Diamond slice, AIME, long CoT prompt pack | answer accuracy, chain length, think-tag compliance, language-mix rate | DeepSeek-R1-Distill, AM-Thinking, Long1K | dense |
| domain_finance_or_other_lab_finetune | CFLUE, FinQA, CCC, or candidate-specific domain suite | domain exact/graded score, compliance format, held-in_retention | DianJin-R1 or selected same-shape domain finetunes/adapters | dense_and_moe |
| safety_refusal | BeaverTails, AdvBench, safe/unsafe refusal prompt pack | safe_response NLL, unsafe_refusal NLL, over-trigger/under-trigger rate | general instruct anchor and safety-aligned endpoints | dense_and_moe |
| connectivity_geometry | anchor-to-expert lambda sweeps; pairwise alpha/beta planes | barrier, worst_NLL, plane residual, delta cosine, sign/weighted conflict | all selected endpoints | dense_and_moe |
| moe_routing | general/math/code/domain route-probe prompt pack | router entropy, top-k expert frequency, route overlap, expert load balance | Qwen3-30B-A3B and Qwen3-Coder-30B-A3B | moe |
| materialization | same-shape checkpoint writer dry run plus held-out eval slice | tensor shape match, missing tensor count, loaded checkpoint eval, endpoint/control gap | final selected average recipe | dense_and_moe |

## 执行顺序

1. 先对 P0 模型做只读 `config/tokenizer/safetensors header` 检查，确认同构性和 chat template 差异。
2. 所有端点先跑同一套小切片评测，端点不过关的模型不进入 average。
3. Dense 先跑 lambda sweep、pairwise barrier 和 alpha/beta plane；MoE 先跑 router top-k、expert load、route overlap。
4. 只有低 barrier/低冲突/route readiness 通过后，才进入 coefficient search、TIES/DARE/RegMean/layer-wise 或 route-aware expert merge。
5. 最终写出 same-shape checkpoint 或 same-shape adapter，再在 held-out slice 上报告一次。

## 文件

- 机器可读模型表：`results/qwen_target_model_registry/model_registry.csv`
- 场景矩阵：`results/qwen_target_model_registry/scenario_matrix.csv`
- 评测/probe 矩阵：`results/qwen_target_model_registry/eval_probe_matrix.csv`
- 汇总 JSON：`results/qwen_target_model_registry/summary.json`

## 证据链接

- Qwen/Qwen2.5-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen2.5-Coder-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- Qwen/Qwen2.5-Math-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- DianJin-R1-7B: https://arxiv.org/abs/2504.15716
- Qwen/Qwen2.5-32B: https://huggingface.co/Qwen/Qwen2.5-32B
- Qwen/Qwen2.5-32B-Instruct: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
- a-m-team/AM-Thinking-v1: https://huggingface.co/a-m-team/AM-Thinking-v1
- deepseek-ai/DeepSeek-R1-Distill-Qwen-32B: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- Long1K-32B: https://arxiv.org/abs/2503.18069
- Qwen/Qwen3-30B-A3B-Base: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base
- Qwen/Qwen3-30B-A3B: https://huggingface.co/Qwen/Qwen3-30B-A3B
- Qwen/Qwen3-Coder-30B-A3B-Instruct: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
