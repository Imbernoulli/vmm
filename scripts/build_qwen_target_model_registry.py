#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


MODEL_FIELDS = [
    "phase",
    "priority",
    "role",
    "model_id",
    "model_family",
    "architecture",
    "scale",
    "source_type",
    "merge_role",
    "scenario",
    "topology_action",
    "eval_focus",
    "probe_focus",
    "materialization_status",
    "source_url",
    "evidence_note",
]

SCENARIO_FIELDS = [
    "scenario_id",
    "priority",
    "objective",
    "input_roles",
    "required_models",
    "primary_risk",
    "primary_metrics",
    "required_probes",
    "first_average_candidate",
    "success_gate",
]

EVAL_FIELDS = [
    "capability",
    "benchmark_slice",
    "metric",
    "held_in_source",
    "probe_reason",
    "dense_or_moe",
]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rel(path: str | Path) -> str:
    return str(repo_path(path).relative_to(REPO_ROOT))


def write_csv(path: str | Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def model_rows() -> list[dict[str, str]]:
    return [
        {
            "phase": "dense_7b",
            "priority": "p0",
            "role": "general_instruction_anchor",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_family": "Qwen2.5",
            "architecture": "dense_qwen2",
            "scale": "7B",
            "source_type": "official_posttrained",
            "merge_role": "anchor",
            "scenario": "dense_7b_general_code_math_reasoning",
            "topology_action": "inspect_config_tokenizer_safetensors_header",
            "eval_focus": "MMLU/C-Eval/CMMLU, IFEval, safety/refusal, general chat format",
            "probe_focus": "anchor-to-expert lambda paths; general_retention baseline",
            "materialization_status": "ready_for_topology_inspect",
            "source_url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            "evidence_note": "Official instruction-tuned Qwen2.5 7B; model card reports Qwen2.5 architecture, Apache-2.0 license, 28 layers, and large downstream adapter/finetune tree.",
        },
        {
            "phase": "dense_7b",
            "priority": "p0",
            "role": "code_expert",
            "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "model_family": "Qwen2.5-Coder",
            "architecture": "dense_qwen2",
            "scale": "7B",
            "source_type": "official_specialist",
            "merge_role": "expert",
            "scenario": "dense_7b_general_code_math_reasoning",
            "topology_action": "inspect_config_tokenizer_safetensors_header",
            "eval_focus": "HumanEval, MBPP, LiveCodeBench slice, code NLL, code format/test failure type",
            "probe_focus": "code-task delta norm, sign conflict, code/general barrier",
            "materialization_status": "ready_for_topology_inspect",
            "source_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct",
            "evidence_note": "Official code-specific Qwen2.5 branch; model card emphasizes code generation, code reasoning, and code fixing.",
        },
        {
            "phase": "dense_7b",
            "priority": "p0",
            "role": "math_expert",
            "model_id": "Qwen/Qwen2.5-Math-7B-Instruct",
            "model_family": "Qwen2.5-Math",
            "architecture": "dense_qwen2",
            "scale": "7B",
            "source_type": "official_specialist",
            "merge_role": "expert",
            "scenario": "dense_7b_general_code_math_reasoning",
            "topology_action": "inspect_config_tokenizer_safetensors_header",
            "eval_focus": "GSM8K, MATH-500, AIME slice, answer NLL, CoT/TIR format",
            "probe_focus": "math-task delta norm, math/code barrier, CoT length/format drift",
            "materialization_status": "ready_for_topology_inspect",
            "source_url": "https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct",
            "evidence_note": "Official math-specific Qwen2.5 branch; model card says the series targets Chinese/English math with CoT and tool-integrated reasoning.",
        },
        {
            "phase": "dense_7b",
            "priority": "p0",
            "role": "reasoning_distill_expert",
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "model_family": "DeepSeek-R1-Distill-Qwen",
            "architecture": "dense_qwen2",
            "scale": "7B",
            "source_type": "third_party_distill",
            "merge_role": "downstream_expert",
            "scenario": "dense_7b_general_code_math_reasoning",
            "topology_action": "inspect_config_tokenizer_safetensors_header",
            "eval_focus": "MATH-500, GPQA slice, long CoT behavior, GSM8K, instruction/safety retention",
            "probe_focus": "reasoning-vs-math delta cosine, CoT style drift, endpoint KL",
            "materialization_status": "ready_for_topology_inspect",
            "source_url": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "evidence_note": "Third-party distilled Qwen branch; DeepSeek model card maps the 7B distill model to Qwen2.5-Math-7B.",
        },
        {
            "phase": "dense_7b",
            "priority": "p1",
            "role": "domain_finance_expert",
            "model_id": "DianJin-R1-7B",
            "model_family": "DianJin-R1",
            "architecture": "dense_qwen2",
            "scale": "7B",
            "source_type": "paper_downstream_finetune",
            "merge_role": "optional_downstream_expert",
            "scenario": "dense_7b_domain_extension",
            "topology_action": "resolve_public_weight_id_then_inspect_topology",
            "eval_focus": "CFLUE, FinQA, CCC, finance compliance format, general retention",
            "probe_focus": "domain/general barrier, domain held-in retention, safety/refusal drift",
            "materialization_status": "manual_weight_id_resolution_required",
            "source_url": "https://arxiv.org/abs/2504.15716",
            "evidence_note": "Paper reports DianJin-R1-7B is fine-tuned from Qwen2.5-7B-Instruct; exact public checkpoint ID should be verified before use.",
        },
        {
            "phase": "dense_7b",
            "priority": "p1",
            "role": "user_lab_adapter_pool",
            "model_id": "hf_tree:Qwen/Qwen2.5-7B-Instruct",
            "model_family": "Qwen2.5 downstream ecosystem",
            "architecture": "dense_qwen2_or_adapter",
            "scale": "7B",
            "source_type": "downstream_adapter_finetune_pool",
            "merge_role": "candidate_pool",
            "scenario": "dense_7b_domain_extension",
            "topology_action": "select_same-shape_full_or_adapter_candidates_only",
            "eval_focus": "task-specific held-in benchmark chosen by candidate domain",
            "probe_focus": "adapter/full-delta norm, tokenizer/chat-template compatibility, endpoint quality gate",
            "materialization_status": "manual_candidate_selection_required",
            "source_url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            "evidence_note": "Model tree exposes many adapters, finetunes, and merges; choose concrete same-shape candidates rather than treating the pool itself as a model.",
        },
        {
            "phase": "dense_32b",
            "priority": "p0",
            "role": "base_anchor",
            "model_id": "Qwen/Qwen2.5-32B",
            "model_family": "Qwen2.5",
            "architecture": "dense_qwen2",
            "scale": "32B",
            "source_type": "official_base",
            "merge_role": "base_or_delta_reference",
            "scenario": "dense_32b_reasoning_long_reasoning",
            "topology_action": "inspect_config_tokenizer_safetensors_header",
            "eval_focus": "base NLL, MMLU/C-Eval/CMMLU, downstream delta reference",
            "probe_focus": "task vectors relative to base; topology compatibility for 32B branches",
            "materialization_status": "ready_for_topology_inspect",
            "source_url": "https://huggingface.co/Qwen/Qwen2.5-32B",
            "evidence_note": "Official Qwen2.5 32B base; model card reports 64 layers and recommends post-training for conversation use.",
        },
        {
            "phase": "dense_32b",
            "priority": "p0",
            "role": "general_instruction_anchor",
            "model_id": "Qwen/Qwen2.5-32B-Instruct",
            "model_family": "Qwen2.5",
            "architecture": "dense_qwen2",
            "scale": "32B",
            "source_type": "official_posttrained",
            "merge_role": "anchor",
            "scenario": "dense_32b_reasoning_long_reasoning",
            "topology_action": "inspect_config_tokenizer_safetensors_header",
            "eval_focus": "general instruction, safety/refusal, format retention",
            "probe_focus": "general_retention baseline; 32B endpoint compatibility",
            "materialization_status": "ready_for_topology_inspect",
            "source_url": "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct",
            "evidence_note": "Official 32B instruction anchor for 32B downstream reasoning branches.",
        },
        {
            "phase": "dense_32b",
            "priority": "p0",
            "role": "reasoning_rl_expert",
            "model_id": "a-m-team/AM-Thinking-v1",
            "model_family": "AM-Thinking",
            "architecture": "dense_qwen2",
            "scale": "32B",
            "source_type": "third_party_sft_rl",
            "merge_role": "downstream_expert",
            "scenario": "dense_32b_reasoning_long_reasoning",
            "topology_action": "inspect_config_tokenizer_safetensors_header",
            "eval_focus": "AIME, MATH-500, GPQA, LiveCodeBench, long reasoning format",
            "probe_focus": "reasoning delta direction, RL-style format drift, code/math co-retention",
            "materialization_status": "ready_for_topology_inspect",
            "source_url": "https://huggingface.co/a-m-team/AM-Thinking-v1",
            "evidence_note": "Model card says AM-Thinking-v1 is built on Qwen2.5-32B-Base with SFT and two-stage RL.",
        },
        {
            "phase": "dense_32b",
            "priority": "p0",
            "role": "reasoning_distill_expert",
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "model_family": "DeepSeek-R1-Distill-Qwen",
            "architecture": "dense_qwen2",
            "scale": "32B",
            "source_type": "third_party_distill",
            "merge_role": "downstream_expert",
            "scenario": "dense_32b_reasoning_long_reasoning",
            "topology_action": "inspect_config_tokenizer_safetensors_header",
            "eval_focus": "AIME, MATH-500, GPQA, CoT length, general/safety retention",
            "probe_focus": "distill-vs-RL reasoning delta conflict, endpoint KL, barrier",
            "materialization_status": "ready_for_topology_inspect",
            "source_url": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "evidence_note": "DeepSeek model card maps the 32B distill model to Qwen2.5-32B.",
        },
        {
            "phase": "dense_32b",
            "priority": "p1",
            "role": "long_reasoning_expert",
            "model_id": "Long1K-32B",
            "model_family": "Long1K",
            "architecture": "dense_qwen2",
            "scale": "32B",
            "source_type": "paper_downstream_finetune",
            "merge_role": "optional_downstream_expert",
            "scenario": "dense_32b_reasoning_long_reasoning",
            "topology_action": "resolve_public_weight_id_then_inspect_topology",
            "eval_focus": "long CoT, MATH, GPQA, answer-length and format stability",
            "probe_focus": "chain length drift, long-vs-short reasoning barrier, held-in retention",
            "materialization_status": "manual_weight_id_resolution_required",
            "source_url": "https://arxiv.org/abs/2503.18069",
            "evidence_note": "Paper reports Long1K-32B is fine-tuned from Qwen2.5-32B-Instruct and says model/code/data are open-sourced; verify exact weight ID before use.",
        },
        {
            "phase": "dense_32b",
            "priority": "p1",
            "role": "domain_finance_expert",
            "model_id": "DianJin-R1-32B",
            "model_family": "DianJin-R1",
            "architecture": "dense_qwen2",
            "scale": "32B",
            "source_type": "paper_downstream_finetune",
            "merge_role": "optional_downstream_expert",
            "scenario": "dense_32b_domain_extension",
            "topology_action": "resolve_public_weight_id_then_inspect_topology",
            "eval_focus": "CFLUE, FinQA, CCC, finance reasoning format, general retention",
            "probe_focus": "domain/reasoning conflict, safety/compliance retention, barrier",
            "materialization_status": "manual_weight_id_resolution_required",
            "source_url": "https://arxiv.org/abs/2504.15716",
            "evidence_note": "Paper reports DianJin-R1-32B is fine-tuned from Qwen2.5-32B-Instruct; exact public checkpoint ID should be verified before use.",
        },
        {
            "phase": "moe_30b_a3b",
            "priority": "p0",
            "role": "moe_base",
            "model_id": "Qwen/Qwen3-30B-A3B-Base",
            "model_family": "Qwen3",
            "architecture": "qwen3_moe",
            "scale": "30B-A3B",
            "source_type": "official_base_moe",
            "merge_role": "base_or_delta_reference",
            "scenario": "moe_30b_general_code_route_aware",
            "topology_action": "inspect_moe_config_router_expert_tensors",
            "eval_focus": "base NLL and router baseline on general/math/code prompts",
            "probe_focus": "router top-k distribution, expert load, route entropy, expert tensor groups",
            "materialization_status": "ready_for_topology_inspect",
            "source_url": "https://huggingface.co/Qwen/Qwen3-30B-A3B-Base",
            "evidence_note": "Official Qwen3 MoE base; model card reports 30.5B total, 3.3B activated, 48 layers, 128 experts, 8 activated experts.",
        },
        {
            "phase": "moe_30b_a3b",
            "priority": "p0",
            "role": "moe_general_anchor",
            "model_id": "Qwen/Qwen3-30B-A3B",
            "model_family": "Qwen3",
            "architecture": "qwen3_moe",
            "scale": "30B-A3B",
            "source_type": "official_posttrained_moe",
            "merge_role": "anchor",
            "scenario": "moe_30b_general_code_route_aware",
            "topology_action": "inspect_moe_config_router_expert_tensors",
            "eval_focus": "thinking/non-thinking general instruction, math, code, agent/tool prompts",
            "probe_focus": "router entropy, route overlap to base/code, thinking-mode format drift",
            "materialization_status": "ready_for_topology_inspect",
            "source_url": "https://huggingface.co/Qwen/Qwen3-30B-A3B",
            "evidence_note": "Official Qwen3 post-trained MoE; model card reports thinking/non-thinking support and 128 experts with 8 activated.",
        },
        {
            "phase": "moe_30b_a3b",
            "priority": "p0",
            "role": "moe_code_agent_expert",
            "model_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "model_family": "Qwen3-Coder",
            "architecture": "qwen3_moe",
            "scale": "30B-A3B",
            "source_type": "official_specialist_moe",
            "merge_role": "expert",
            "scenario": "moe_30b_general_code_route_aware",
            "topology_action": "inspect_moe_config_router_expert_tensors",
            "eval_focus": "HumanEval, MBPP, LiveCodeBench, repo-level coding, tool/function calling",
            "probe_focus": "route overlap, expert output similarity, code-specialized expert load",
            "materialization_status": "ready_for_topology_inspect",
            "source_url": "https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "evidence_note": "Official Qwen3-Coder MoE; model card reports agentic coding, 30.5B total/3.3B activated, 128 experts, 8 activated experts.",
        },
        {
            "phase": "moe_30b_a3b",
            "priority": "p1",
            "role": "moe_downstream_adapter_pool",
            "model_id": "hf_tree:Qwen/Qwen3-30B-A3B",
            "model_family": "Qwen3 downstream MoE ecosystem",
            "architecture": "qwen3_moe_or_adapter",
            "scale": "30B-A3B",
            "source_type": "downstream_adapter_finetune_pool",
            "merge_role": "candidate_pool",
            "scenario": "moe_30b_downstream_adapter_average",
            "topology_action": "select_same-shape_moe_or_adapter_candidates_only",
            "eval_focus": "candidate-specific held-in task plus general/code/math retention",
            "probe_focus": "router drift, adapter delta placement, expert-load specialization",
            "materialization_status": "manual_candidate_selection_required",
            "source_url": "https://huggingface.co/Qwen/Qwen3-30B-A3B",
            "evidence_note": "Use concrete Qwen3-30B-A3B downstream MoE full-finetunes or adapters only after topology and license checks.",
        },
        {
            "phase": "moe_30b_a3b",
            "priority": "p1",
            "role": "moe_coder_downstream_adapter_pool",
            "model_id": "hf_tree:Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "model_family": "Qwen3-Coder downstream MoE ecosystem",
            "architecture": "qwen3_moe_or_adapter",
            "scale": "30B-A3B",
            "source_type": "downstream_adapter_finetune_pool",
            "merge_role": "candidate_pool",
            "scenario": "moe_30b_downstream_adapter_average",
            "topology_action": "select_same-shape_moe_or_adapter_candidates_only",
            "eval_focus": "repo coding, tool calling, candidate-specific held-in task",
            "probe_focus": "route-aware adapter average, expert output matching, route overlap",
            "materialization_status": "manual_candidate_selection_required",
            "source_url": "https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "evidence_note": "Use concrete downstream Qwen3-Coder 30B-A3B adapters/finetunes after verifying same router/expert topology.",
        },
    ]


def scenario_rows() -> list[dict[str, str]]:
    return [
        {
            "scenario_id": "dense_7b_general_code_math_reasoning",
            "priority": "p0_first_wave",
            "objective": "合成一个同构 Qwen2.5 7B assistant，保留通用指令、代码、数学和 R1 蒸馏推理能力。",
            "input_roles": "general_instruction_anchor; code_expert; math_expert; reasoning_distill_expert",
            "required_models": "Qwen/Qwen2.5-7B-Instruct; Qwen/Qwen2.5-Coder-7B-Instruct; Qwen/Qwen2.5-Math-7B-Instruct; deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "primary_risk": "专长分支改变输出协议和 CoT 风格；朴素 0.5/0.5 平均可能落在高 NLL ridge。",
            "primary_metrics": "avg_score; worst_score; held-in_retention; general_retention; format_success; safety_retention",
            "required_probes": "topology inspect; endpoint eval; anchor-to-expert lambda sweep; pairwise barrier; alpha/beta grid; cosine/sign conflict; answer-format probes",
            "first_average_candidate": "coefficient_search_after_connectivity_gate",
            "success_gate": "worst_score > uniform_average and each held-in retention >= 0.90 with general retention >= 0.95.",
        },
        {
            "scenario_id": "dense_7b_domain_extension",
            "priority": "p1_after_7b_core",
            "objective": "加入金融/医疗/法律等实验室或行业微调分支，检验领域能力能否压回同构 7B checkpoint。",
            "input_roles": "general_instruction_anchor; selected_domain_expert; optional code/math/reasoning expert",
            "required_models": "Qwen/Qwen2.5-7B-Instruct plus a verified same-shape downstream full-finetune or adapter",
            "primary_risk": "领域数据可能牺牲安全、拒答和通用指令能力；公开权重 ID 与 license 需要单独确认。",
            "primary_metrics": "domain held-in score; general_retention; safety_retention; format_success; worst_score",
            "required_probes": "topology/license gate; domain/general barrier; refusal over/under-trigger; domain answer-format probe",
            "first_average_candidate": "anchor_plus_domain_delta_with_small_lambda_sweep",
            "success_gate": "domain held-in improves over anchor while general/safety retention does not cross the drop threshold.",
        },
        {
            "scenario_id": "dense_32b_reasoning_long_reasoning",
            "priority": "p1_scale_validation",
            "objective": "在 Qwen2.5 32B 上验证 reasoning RL、R1 distill 和长 CoT 分支是否更容易或更难平均。",
            "input_roles": "base_anchor; general_instruction_anchor; reasoning_rl_expert; reasoning_distill_expert; optional long_reasoning_expert",
            "required_models": "Qwen/Qwen2.5-32B; Qwen/Qwen2.5-32B-Instruct; a-m-team/AM-Thinking-v1; deepseek-ai/DeepSeek-R1-Distill-Qwen-32B; optional Long1K-32B",
            "primary_risk": "多个推理分支可能同向增强，也可能在回答长度、think tags、代码能力和安全策略上互扰。",
            "primary_metrics": "AIME/MATH/GPQA; LiveCodeBench; chain length; held-in/general/safety retention; cost",
            "required_probes": "base-relative task vectors; reasoning branch barrier graph; answer length/style probes; layer-wise conflict",
            "first_average_candidate": "greedy_soup_or_task_arithmetic_only_inside_low_barrier_component",
            "success_gate": "reasoning score improves over general anchor and no held-in branch drops below 0.90 retention.",
        },
        {
            "scenario_id": "moe_30b_general_code_route_aware",
            "priority": "p0_moe_wave",
            "objective": "合成同 expert 数、同 router shape 的 Qwen3-30B-A3B route-aware MoE，保留通用和代码/agentic 能力。",
            "input_roles": "moe_base; moe_general_anchor; moe_code_agent_expert",
            "required_models": "Qwen/Qwen3-30B-A3B-Base; Qwen/Qwen3-30B-A3B; Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "primary_risk": "router/expert 不能当普通 dense 层平均；expert index mismatch 或 router drift 会导致 collapse。",
            "primary_metrics": "general/code/math score; route overlap; expert entropy/load; activated params; latency/tokens per second",
            "required_probes": "router top-k distribution; route overlap; expert load; expert output similarity; expert delta norm/cosine",
            "first_average_candidate": "router_frozen_shared_merge_plus_expert_matched_average",
            "success_gate": "route overlap and expert load stay within readiness gate while worst_score beats all-weight average.",
        },
        {
            "scenario_id": "moe_30b_downstream_adapter_average",
            "priority": "p1_after_real_routing_probe",
            "objective": "选择真实下游 Qwen3 MoE adapters/finetunes，判断 adapter-level 或 expert-wise average 能否压回同构 MoE。",
            "input_roles": "moe_general_anchor; moe_downstream_adapter_or_finetune; optional moe_code_agent_expert",
            "required_models": "verified Qwen3-30B-A3B or Qwen3-Coder-30B-A3B same-shape downstream candidates",
            "primary_risk": "LoRA/adapters 的能力可能本来就需要输入自适应路由；直接压成单 delta 可能损失上界能力。",
            "primary_metrics": "candidate held-in score; general/code retention; adapter-vs-full checkpoint delta; route readiness",
            "required_probes": "adapter tensor placement; router drift; route-frequency tensor rules; MoLE-style upper-bound comparison",
            "first_average_candidate": "same-shape_adapter_average_or_route_weighted_full_delta",
            "success_gate": "final output remains one same-shape adapter or checkpoint and beats endpoint-only/negative baselines.",
        },
        {
            "scenario_id": "negative_controls",
            "priority": "always",
            "objective": "保留朴素平均、端点、endpoint-only best grid 和 all-weight MoE average 作为负/对照项，避免误报。",
            "input_roles": "all selected scenarios",
            "required_models": "same endpoints as each scenario",
            "primary_risk": "把端点、投影点或失败的 uniform average 误解成真正的 average 成功。",
            "primary_metrics": "uniform_average_delta; endpoint gap; best grid endpoint flag; readiness action",
            "required_probes": "average decision report; candidate recipes; same-shape writer dry run; held-out test report",
            "first_average_candidate": "none_control_only",
            "success_gate": "control remains explicitly labeled and is not selected unless it passes same-shape average criteria.",
        },
    ]


def eval_probe_rows() -> list[dict[str, str]]:
    return [
        {
            "capability": "general_knowledge_instruction",
            "benchmark_slice": "MMLU, C-Eval, CMMLU, IFEval, small chat-format pack",
            "metric": "accuracy, response-only NLL, format_success, general_retention",
            "held_in_source": "Qwen instruct/general anchors",
            "probe_reason": "判断合并后是否只是专长变强而通用能力塌掉。",
            "dense_or_moe": "dense_and_moe",
        },
        {
            "capability": "math_reasoning",
            "benchmark_slice": "GSM8K, MATH-500, AIME slice",
            "metric": "exact_match, gold-answer NLL, CoT/TIR format_success",
            "held_in_source": "Qwen2.5-Math and DeepSeek/AM/Long1K reasoning branches",
            "probe_reason": "数学分支是第一批同源专长能力，能直接暴露 average 是否保留 held-in 能力。",
            "dense_or_moe": "dense_and_moe",
        },
        {
            "capability": "code_generation_agentic",
            "benchmark_slice": "HumanEval, MBPP, LiveCodeBench, repo-level/tool-call slice",
            "metric": "pass@1, canonical solution NLL, syntax/test failure type, tool-call format_success",
            "held_in_source": "Qwen2.5-Coder and Qwen3-Coder branches",
            "probe_reason": "代码分支最容易和通用/安全/长 CoT 风格互扰，NLL 和单元测试要同时看。",
            "dense_or_moe": "dense_and_moe",
        },
        {
            "capability": "reasoning_style",
            "benchmark_slice": "GPQA-Diamond slice, AIME, long CoT prompt pack",
            "metric": "answer accuracy, chain length, think-tag compliance, language-mix rate",
            "held_in_source": "DeepSeek-R1-Distill, AM-Thinking, Long1K",
            "probe_reason": "推理模型可能保留知识但破坏输出协议，必须把风格和长度单独量化。",
            "dense_or_moe": "dense",
        },
        {
            "capability": "domain_finance_or_other_lab_finetune",
            "benchmark_slice": "CFLUE, FinQA, CCC, or candidate-specific domain suite",
            "metric": "domain exact/graded score, compliance format, held-in_retention",
            "held_in_source": "DianJin-R1 or selected same-shape domain finetunes/adapters",
            "probe_reason": "代表真实下游实验室/行业微调，不应只停留在官方模型分支。",
            "dense_or_moe": "dense_and_moe",
        },
        {
            "capability": "safety_refusal",
            "benchmark_slice": "BeaverTails, AdvBench, safe/unsafe refusal prompt pack",
            "metric": "safe_response NLL, unsafe_refusal NLL, over-trigger/under-trigger rate",
            "held_in_source": "general instruct anchor and safety-aligned endpoints",
            "probe_reason": "合并专长能力不能用安全和拒答稳定性换来。",
            "dense_or_moe": "dense_and_moe",
        },
        {
            "capability": "connectivity_geometry",
            "benchmark_slice": "anchor-to-expert lambda sweeps; pairwise alpha/beta planes",
            "metric": "barrier, worst_NLL, plane residual, delta cosine, sign/weighted conflict",
            "held_in_source": "all selected endpoints",
            "probe_reason": "probe 的核心是判断哪些模型、层、模块处在可平均连通区域里。",
            "dense_or_moe": "dense_and_moe",
        },
        {
            "capability": "moe_routing",
            "benchmark_slice": "general/math/code/domain route-probe prompt pack",
            "metric": "router entropy, top-k expert frequency, route overlap, expert load balance",
            "held_in_source": "Qwen3-30B-A3B and Qwen3-Coder-30B-A3B",
            "probe_reason": "MoE average 成败取决于 token 到 expert 的分配是否仍然稳定。",
            "dense_or_moe": "moe",
        },
        {
            "capability": "materialization",
            "benchmark_slice": "same-shape checkpoint writer dry run plus held-out eval slice",
            "metric": "tensor shape match, missing tensor count, loaded checkpoint eval, endpoint/control gap",
            "held_in_source": "final selected average recipe",
            "probe_reason": "Average 最终必须写成同构 checkpoint 或 adapter，不能只停在图、CSV 或 ensemble。",
            "dense_or_moe": "dense_and_moe",
        },
    ]


def status_counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row[field]
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_summary(models: list[dict[str, str]], scenarios: list[dict[str, str]], eval_probes: list[dict[str, str]]) -> dict[str, Any]:
    dense = [row for row in models if row["architecture"].startswith("dense")]
    moe = [row for row in models if "moe" in row["architecture"]]
    downstream = [
        row
        for row in models
        if row["source_type"].startswith("third_party")
        or row["source_type"].startswith("paper_downstream")
        or row["source_type"].startswith("downstream")
    ]
    ready = [row for row in models if row["materialization_status"] == "ready_for_topology_inspect"]
    manual = [row for row in models if row["materialization_status"] != "ready_for_topology_inspect"]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_count": len(models),
        "dense_model_count": len(dense),
        "moe_model_count": len(moe),
        "official_count": sum(1 for row in models if row["source_type"].startswith("official")),
        "downstream_or_third_party_count": len(downstream),
        "ready_for_topology_inspect_count": len(ready),
        "manual_resolution_or_selection_count": len(manual),
        "scenario_count": len(scenarios),
        "eval_probe_count": len(eval_probes),
        "phase_counts": status_counts(models, "phase"),
        "source_type_counts": status_counts(models, "source_type"),
        "materialization_status_counts": status_counts(models, "materialization_status"),
        "recommended_first_scenario": "dense_7b_general_code_math_reasoning",
        "recommended_first_models": [
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen2.5-Math-7B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        ],
        "recommended_first_moe_scenario": "moe_30b_general_code_route_aware",
        "recommended_first_moe_models": [
            "Qwen/Qwen3-30B-A3B-Base",
            "Qwen/Qwen3-30B-A3B",
            "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        ],
        "required_next_artifacts": [
            "topology inspect for every p0 candidate",
            "endpoint eval table before any averaging",
            "connectivity/barrier graph before selecting average coefficients",
            "MoE routing probe on Qwen3-30B-A3B and Qwen3-Coder-30B-A3B before router/expert averaging",
            "same-shape checkpoint writer dry run followed by held-out eval",
        ],
    }


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "").replace("|", "/") for column in columns) + " |")
    return lines


def build_report(
    models: list[dict[str, str]],
    scenarios: list[dict[str, str]],
    eval_probes: list[dict[str, str]],
    summary: dict[str, Any],
) -> str:
    p0_models = [row for row in models if row["priority"] == "p0"]
    manual_models = [row for row in models if row["materialization_status"] != "ready_for_topology_inspect"]
    source_links = []
    seen: set[str] = set()
    for row in models:
        url = row["source_url"]
        if url in seen:
            continue
        seen.add(url)
        source_links.append(f"- {row['model_id']}: {url}")

    lines = [
        "# Qwen 目标模型、场景与评测 Probe Registry",
        "",
        f"生成时间：`{summary['generated_at']}`",
        "",
        "## 先给结论",
        "",
        (
            "第一轮真实实验建议从 `dense_7b_general_code_math_reasoning` 开始："
            "`Qwen/Qwen2.5-7B-Instruct` 作为通用 anchor，加入官方 Coder、官方 Math、"
            "`DeepSeek-R1-Distill-Qwen-7B` 这三个同源专长/下游分支。"
        ),
        "",
        (
            "这比只比较 Base/Instruct 更接近真实下游用户场景：候选里显式包含第三方蒸馏、"
            "SFT/RL reasoning 模型、论文中的金融/长推理微调模型，以及 Hugging Face 模型树里的"
            "同构 adapters/finetunes 候选池。"
        ),
        "",
        (
            "Average 的硬约束保持不变：最终输出必须和输入模型同构。Dense 不能变成 ensemble；"
            "MoE 不能改变 layer 数、router shape、expert 数或每个 expert 的张量 shape。"
            "DianJin/Long1K 这类论文候选先标成需要人工确认权重 ID，避免把还未验证的模型当作可直接 materialize 的 checkpoint。"
        ),
        "",
        "## Registry 规模",
        "",
        f"- 候选条目：`{summary['model_count']}`，其中 dense `{summary['dense_model_count']}`，MoE `{summary['moe_model_count']}`。",
        f"- 官方模型：`{summary['official_count']}`；第三方/下游/候选池：`{summary['downstream_or_third_party_count']}`。",
        f"- 可直接进入 topology inspect：`{summary['ready_for_topology_inspect_count']}`；需要先人工确认权重或具体候选：`{summary['manual_resolution_or_selection_count']}`。",
        f"- 场景：`{summary['scenario_count']}`；评测/probe 维度：`{summary['eval_probe_count']}`。",
        "",
        "## P0 候选模型",
        "",
    ]
    lines.extend(
        markdown_table(
            p0_models,
            ["phase", "role", "model_id", "source_type", "merge_role", "scenario", "materialization_status"],
        )
    )
    lines.extend(
        [
            "",
            "## 需要先确认的下游候选",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            manual_models,
            ["phase", "role", "model_id", "source_type", "topology_action", "materialization_status"],
        )
    )
    lines.extend(
        [
            "",
            "## 场景矩阵",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            scenarios,
            ["scenario_id", "priority", "objective", "first_average_candidate", "success_gate"],
        )
    )
    lines.extend(
        [
            "",
            "## 评测与 Probe 矩阵",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            eval_probes,
            ["capability", "benchmark_slice", "metric", "held_in_source", "dense_or_moe"],
        )
    )
    lines.extend(
        [
            "",
            "## 执行顺序",
            "",
            "1. 先对 P0 模型做只读 `config/tokenizer/safetensors header` 检查，确认同构性和 chat template 差异。",
            "2. 所有端点先跑同一套小切片评测，端点不过关的模型不进入 average。",
            "3. Dense 先跑 lambda sweep、pairwise barrier 和 alpha/beta plane；MoE 先跑 router top-k、expert load、route overlap。",
            "4. 只有低 barrier/低冲突/route readiness 通过后，才进入 coefficient search、TIES/DARE/RegMean/layer-wise 或 route-aware expert merge。",
            "5. 最终写出 same-shape checkpoint 或 same-shape adapter，再在 held-out slice 上报告一次。",
            "",
            "## 文件",
            "",
            "- 机器可读模型表：`results/qwen_target_model_registry/model_registry.csv`",
            "- 场景矩阵：`results/qwen_target_model_registry/scenario_matrix.csv`",
            "- 评测/probe 矩阵：`results/qwen_target_model_registry/eval_probe_matrix.csv`",
            "- 汇总 JSON：`results/qwen_target_model_registry/summary.json`",
            "",
            "## 证据链接",
            "",
        ]
    )
    lines.extend(source_links)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/qwen_target_model_registry")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    models = model_rows()
    scenarios = scenario_rows()
    eval_probes = eval_probe_rows()
    summary = build_summary(models, scenarios, eval_probes)

    write_csv(output_dir / "model_registry.csv", models, MODEL_FIELDS)
    write_csv(output_dir / "scenario_matrix.csv", scenarios, SCENARIO_FIELDS)
    write_csv(output_dir / "eval_probe_matrix.csv", eval_probes, EVAL_FIELDS)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(build_report(models, scenarios, eval_probes, summary), encoding="utf-8")

    print(f"Wrote {rel(output_dir / 'model_registry.csv')}")
    print(f"Wrote {rel(output_dir / 'scenario_matrix.csv')}")
    print(f"Wrote {rel(output_dir / 'eval_probe_matrix.csv')}")
    print(f"Wrote {rel(output_dir / 'summary.json')}")
    print(f"Wrote {rel(output_dir / 'report.md')}")


if __name__ == "__main__":
    main()
