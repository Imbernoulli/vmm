#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

LITERATURE_PRIORS = [
    {
        "key": "mode_connectivity",
        "source": "https://arxiv.org/abs/1802.10026",
        "mechanism": "Only average inside a measured low-loss component; source identity alone is not enough.",
    },
    {
        "key": "model_soups",
        "source": "https://arxiv.org/abs/2203.05482",
        "mechanism": "Average same-basin finetunes, but keep endpoint fallback and held-out selection.",
    },
    {
        "key": "ties",
        "source": "https://arxiv.org/abs/2306.01708",
        "mechanism": "Task-vector interference means endpoint complementarity must be larger than merge damage.",
    },
    {
        "key": "expert_merging",
        "source": "https://arxiv.org/abs/2509.25712",
        "mechanism": "MoE coefficients should follow calibration behavior instead of one global average.",
    },
    {
        "key": "sub_moe",
        "source": "https://arxiv.org/abs/2506.23266",
        "mechanism": "Expert similarity, expert output behavior, and subspace conflict are primary MoE probes.",
    },
    {
        "key": "harc",
        "source": "https://arxiv.org/abs/2606.03391",
        "mechanism": "Router top-k boundary movement must be audited before any MoE router change is accepted.",
    },
]


CAPABILITY_MIN_EXAMPLES = {
    "general_knowledge_instruction": 256,
    "math_reasoning": 256,
    "code_generation_agentic": 128,
    "reasoning_style": 128,
    "domain_finance_or_other_lab_finetune": 128,
    "safety_refusal": 128,
    "connectivity_geometry": 64,
    "moe_routing": 64,
    "materialization": 16,
}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    value = clean_value(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def fnum(value: Any) -> float | None:
    value = clean_value(value)
    return None if value is None else float(value)


def maybe_int(value: Any) -> int | None:
    value = clean_value(value)
    return None if value is None else int(value)


def fmt(value: Any, digits: int = 4) -> str:
    value = fnum(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def split_semicolon(text: Any) -> list[str]:
    text = "" if clean_value(text) is None else str(text)
    return [part.strip() for part in text.split(";") if part.strip()]


def served_model_list(text: Any) -> str:
    models = [
        item
        for item in split_semicolon(text)
        if not item.lower().startswith("optional ")
        and "verified " not in item.lower()
        and " plus " not in item.lower()
    ]
    return ",".join(models)


def priority_score(priority: Any, *, ready_count: int, manual_count: int, architecture_kind: str) -> float:
    label = str(priority)
    if label == "always":
        base = 0.20
    elif label.startswith("p0_first"):
        base = 0.95
    elif label.startswith("p0_moe"):
        base = 0.94
    elif label.startswith("p0"):
        base = 0.93
    elif label.startswith("p1_scale"):
        base = 0.76
    elif label.startswith("p1"):
        base = 0.70
    else:
        base = 0.55
    if ready_count < 2 and label != "always":
        base -= 0.18
    if manual_count:
        base -= min(0.16, manual_count * 0.04)
    if architecture_kind == "moe":
        base += 0.01
    return max(0.0, float(base))


def infer_architecture_kind(rows: pd.DataFrame, scenario_id: str) -> str:
    archs = " ".join(str(item) for item in rows.get("architecture", pd.Series(dtype=str)).dropna().unique())
    scenario = str(scenario_id)
    if "moe" in archs.lower() or scenario.startswith("moe_"):
        return "moe"
    if "dense" in archs.lower() or scenario.startswith("dense_"):
        return "dense"
    return "control"


def readiness_gate(ready_count: int, manual_count: int, priority: str, scenario_id: str) -> str:
    if scenario_id == "negative_controls":
        return "control_only"
    if manual_count and ready_count < 2:
        return "manual_model_resolution_first"
    if str(priority).startswith("p0") and ready_count >= 2:
        return "ready_for_endpoint_eval_and_surplus_gate"
    if ready_count >= 2:
        return "ready_after_p0_wave"
    return "needs_source_resolution"


def mechanism_focus(architecture_kind: str) -> str:
    if architecture_kind == "moe":
        return "router_expert_probe_then_source_surplus_gate"
    if architecture_kind == "dense":
        return "connectivity_barrier_then_source_surplus_gate"
    return "negative_control_endpoint_and_uniform_average"


def scenario_capabilities(scenario_id: str, architecture_kind: str) -> list[str]:
    if scenario_id == "negative_controls":
        return ["materialization"]
    if "domain" in scenario_id or "adapter" in scenario_id:
        caps = [
            "domain_finance_or_other_lab_finetune",
            "general_knowledge_instruction",
            "safety_refusal",
            "connectivity_geometry",
        ]
    elif "32b" in scenario_id:
        caps = [
            "general_knowledge_instruction",
            "math_reasoning",
            "code_generation_agentic",
            "reasoning_style",
            "safety_refusal",
            "connectivity_geometry",
        ]
    else:
        caps = [
            "general_knowledge_instruction",
            "math_reasoning",
            "code_generation_agentic",
            "safety_refusal",
            "connectivity_geometry",
        ]
    if architecture_kind == "moe":
        caps.append("moe_routing")
    return caps


def build_scenario_priority(registry: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, scenario in scenarios.iterrows():
        scenario_id = str(scenario["scenario_id"])
        model_rows = registry[registry["scenario"].astype(str) == scenario_id]
        architecture_kind = infer_architecture_kind(model_rows, scenario_id)
        ready_rows = model_rows[
            model_rows.get("materialization_status", pd.Series(dtype=str))
            .astype(str)
            .str.contains("ready_for_topology_inspect", na=False)
        ]
        manual_rows = model_rows[
            model_rows.get("materialization_status", pd.Series(dtype=str))
            .astype(str)
            .str.contains("manual|resolve", case=False, na=False)
        ]
        ready_count = int(len(ready_rows))
        manual_count = int(len(manual_rows))
        priority = str(scenario.get("priority"))
        gate = readiness_gate(ready_count, manual_count, priority, scenario_id)
        score = priority_score(
            priority,
            ready_count=ready_count,
            manual_count=manual_count,
            architecture_kind=architecture_kind,
        )
        if gate == "control_only":
            next_action = "keep_endpoint_uniform_and_writer_controls_labeled"
        elif gate == "manual_model_resolution_first":
            next_action = "resolve_public_same_shape_weight_ids_before_endpoint_eval"
        elif architecture_kind == "moe":
            next_action = "run_endpoint_eval_plus_router_expert_probe_then_surplus_gate"
        else:
            next_action = "run_endpoint_eval_plus_connectivity_probe_then_surplus_gate"

        rows.append(
            {
                "scenario_id": scenario_id,
                "priority": priority,
                "priority_score": score,
                "architecture_kind": architecture_kind,
                "source_count": int(len(model_rows)),
                "ready_model_count": ready_count,
                "manual_resolution_count": manual_count,
                "source_roles": "; ".join(str(item) for item in model_rows.get("role", pd.Series(dtype=str)).dropna()),
                "ready_models": "; ".join(str(item) for item in ready_rows.get("model_id", pd.Series(dtype=str)).dropna()),
                "manual_models": "; ".join(str(item) for item in manual_rows.get("model_id", pd.Series(dtype=str)).dropna()),
                "required_models": scenario.get("required_models"),
                "first_average_candidate": scenario.get("first_average_candidate"),
                "success_gate": scenario.get("success_gate"),
                "readiness_gate": gate,
                "mechanism_focus": mechanism_focus(architecture_kind),
                "next_action": next_action,
            }
        )
    out = pd.DataFrame(rows).sort_values(["priority_score", "scenario_id"], ascending=[False, True])
    out = out.reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    columns = [
        "rank",
        "scenario_id",
        "priority",
        "priority_score",
        "architecture_kind",
        "source_count",
        "ready_model_count",
        "manual_resolution_count",
        "source_roles",
        "ready_models",
        "manual_models",
        "required_models",
        "first_average_candidate",
        "success_gate",
        "readiness_gate",
        "mechanism_focus",
        "next_action",
    ]
    return out[columns]


def measured_top_row(
    optimizer: dict[str, Any],
    top_candidates: pd.DataFrame,
) -> dict[str, Any]:
    top = optimizer.get("top_source_set") or {}
    interference = fnum(optimizer.get("interference_budget")) or 0.0
    gain = fnum(top.get("frontier_avg_gain_vs_best_single")) or 0.0
    additional = max(0.0, interference - gain)
    matched = pd.DataFrame()
    if not top_candidates.empty and top.get("source_set"):
        matched = top_candidates[top_candidates["source_set"].astype(str) == str(top.get("source_set"))]
    first = matched.iloc[0].to_dict() if not matched.empty else {}
    return {
        "candidate_id": top.get("candidate_id") or "qwen3_moe_measured_top_source_set_probe",
        "scenario_id": "measured_qwen3_moe_source_set",
        "source_set": top.get("source_set"),
        "source_set_type": "measured_generation_frontier",
        "architecture_kind": "moe",
        "source_roles": "measured endpoints from downstream matrix",
        "required_models": top.get("source_set"),
        "ready_model_count": maybe_int(first.get("active_source_count")),
        "manual_resolution_count": 0,
        "frontier_avg_gain_vs_best_single": gain,
        "interference_budget": interference,
        "frontier_avg_surplus_vs_interference": fnum(top.get("frontier_avg_surplus_vs_interference")),
        "additional_frontier_avg_gain_needed": additional,
        "optimizer_gate": top.get("optimizer_gate"),
        "source_weights": top.get("source_weights"),
        "readiness_gate": "ready_for_endpoint_expansion_probe_only",
        "mechanism_focus": "measured_complementarity_below_interference_budget",
        "next_action": "expand_endpoint_eval_and_small_weighted_probe_before_final_average",
        "same_shape_policy": "preserve Qwen3 MoE config/tokenizer/tensor names/tensor shapes/expert count/router shape",
    }


def build_candidate_source_sets(
    scenario_priority: pd.DataFrame,
    optimizer: dict[str, Any],
    top_candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows = [measured_top_row(optimizer, top_candidates)]
    interference = fnum(optimizer.get("interference_budget")) or 0.0
    for _, scenario in scenario_priority.iterrows():
        scenario_id = str(scenario["scenario_id"])
        if scenario_id == "negative_controls":
            continue
        if scenario["readiness_gate"] not in {
            "ready_for_endpoint_eval_and_surplus_gate",
            "ready_after_p0_wave",
            "manual_model_resolution_first",
        }:
            continue
        rows.append(
            {
                "candidate_id": f"{scenario_id}_planned_source_set",
                "scenario_id": scenario_id,
                "source_set": scenario["required_models"],
                "source_set_type": "registry_planned_source_set",
                "architecture_kind": scenario["architecture_kind"],
                "source_roles": scenario["source_roles"],
                "required_models": scenario["required_models"],
                "ready_model_count": scenario["ready_model_count"],
                "manual_resolution_count": scenario["manual_resolution_count"],
                "frontier_avg_gain_vs_best_single": None,
                "interference_budget": interference,
                "frontier_avg_surplus_vs_interference": None,
                "additional_frontier_avg_gain_needed": interference,
                "optimizer_gate": "endpoint_eval_required_before_average",
                "source_weights": None,
                "readiness_gate": scenario["readiness_gate"],
                "mechanism_focus": scenario["mechanism_focus"],
                "next_action": scenario["next_action"],
                "same_shape_policy": (
                    "preserve dense Qwen config/tokenizer/tensor names/tensor shapes"
                    if scenario["architecture_kind"] == "dense"
                    else "preserve Qwen MoE config/tokenizer/tensor names/tensor shapes/expert count/router shape"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_endpoint_eval_expansion(
    scenario_priority: pd.DataFrame,
    eval_matrix: pd.DataFrame,
    measured: dict[str, Any],
) -> pd.DataFrame:
    by_capability = {str(row["capability"]): row.to_dict() for _, row in eval_matrix.iterrows()}
    rows: list[dict[str, Any]] = []

    def add_row(
        *,
        scenario_id: str,
        plan_id: str,
        capability: str,
        purpose: str,
        architecture_kind: str,
        runner_scope: str,
    ) -> None:
        row = by_capability.get(capability, {})
        route_required = capability == "moe_routing" or architecture_kind == "moe"
        rows.append(
            {
                "plan_id": plan_id,
                "scenario_id": scenario_id,
                "capability": capability,
                "benchmark_slice": row.get("benchmark_slice"),
                "metric": row.get("metric"),
                "held_in_source": row.get("held_in_source"),
                "purpose": purpose,
                "dense_or_moe": row.get("dense_or_moe"),
                "min_examples": CAPABILITY_MIN_EXAMPLES.get(capability, 64),
                "route_probe_required": route_required,
                "acceptance_signal": "frontier_gain_over_best_single_then_no_task_regression",
                "runner_scope": runner_scope,
            }
        )

    measured_caps = [
        "general_knowledge_instruction",
        "math_reasoning",
        "code_generation_agentic",
        "moe_routing",
        "connectivity_geometry",
    ]
    for capability in measured_caps:
        add_row(
            scenario_id="measured_qwen3_moe_source_set",
            plan_id=f"measured_{capability}",
            capability=capability,
            purpose=(
                "shrink uncertainty for the current top measured source set; "
                f"additional avg frontier gain needed {fmt(measured.get('additional_frontier_avg_gain_needed'))}"
            ),
            architecture_kind="moe",
            runner_scope="endpoint_expansion_or_probe_only",
        )

    for _, scenario in scenario_priority.iterrows():
        scenario_id = str(scenario["scenario_id"])
        if scenario_id == "negative_controls":
            continue
        if str(scenario["readiness_gate"]) not in {
            "ready_for_endpoint_eval_and_surplus_gate",
            "ready_after_p0_wave",
            "manual_model_resolution_first",
        }:
            continue
        for capability in scenario_capabilities(scenario_id, str(scenario["architecture_kind"])):
            add_row(
                scenario_id=scenario_id,
                plan_id=f"{scenario_id}_{capability}",
                capability=capability,
                purpose=(
                    "measure endpoint frontier before any same-shape average; "
                    f"scenario gate {scenario['readiness_gate']}"
                ),
                architecture_kind=str(scenario["architecture_kind"]),
                runner_scope="vllm_endpoint_eval" if "routing" not in capability and "connectivity" not in capability else "probe_then_eval",
            )
    return pd.DataFrame(rows)


def build_source_discovery_queue(
    scenario_priority: pd.DataFrame,
    candidate_source_sets: pd.DataFrame,
    measured: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "base_rank": 1,
            "queue_item": "measured_coder_thinking_endpoint_expansion",
            "scenario_id": "measured_qwen3_moe_source_set",
            "mechanism_target": "verify whether the measured Coder+Thinking complementarity is real and large enough",
            "status": "ready_endpoint_expansion_probe_only",
            "why_now": (
                f"Current frontier avg gain {fmt(measured.get('frontier_avg_gain_vs_best_single'))} "
                f"is below the observed interference budget {fmt(measured.get('interference_budget'))}; "
                f"additional gain needed {fmt(measured.get('additional_frontier_avg_gain_needed'))}."
            ),
            "command": (
                "python scripts/run_vllm_downstream_eval.py --models SERVED_CODER,SERVED_THINKING "
                "--tasks mmlu,gsm8k,humaneval_compile --max-examples 256 "
                "--output-dir results/qwen_source_discovery_plan/measured_coder_thinking_vllm"
            ),
            "preflight_command": "results/qwen3_moe_eval_budget_plan/run_eval_budget.sh preflight final",
            "priority_score": 0.98,
            "expected_decision_update": "move Coder+Thinking from probe-only to final-budget only if surplus becomes non-negative",
        }
    ]

    ready = scenario_priority[
        scenario_priority["readiness_gate"].isin(
            ["ready_for_endpoint_eval_and_surplus_gate", "ready_after_p0_wave"]
        )
    ]
    for offset, (_, scenario) in enumerate(ready.iterrows(), start=2):
        scenario_id = str(scenario["scenario_id"])
        if scenario_id == "negative_controls":
            continue
        models_arg = served_model_list(scenario["required_models"])
        if scenario["architecture_kind"] == "moe":
            command = (
                f"python scripts/run_vllm_downstream_eval.py --models {models_arg} "
                "--tasks mmlu,gsm8k,humaneval_compile "
                "--max-examples 256 --output-dir results/qwen_source_discovery_plan/vllm_endpoint_eval"
            )
            expected = "router/expert probes plus endpoint frontier decide if MoE source set can enter surplus optimizer"
        else:
            command = (
                f"python scripts/run_vllm_downstream_eval.py --models {models_arg} "
                "--tasks mmlu,gsm8k,humaneval_compile,safety "
                "--max-examples 256 --output-dir results/qwen_source_discovery_plan/vllm_endpoint_eval"
            )
            expected = "endpoint frontier plus connectivity plane decide if Dense source set can enter average budget"
        rows.append(
            {
                "base_rank": offset,
                "queue_item": f"{scenario_id}_endpoint_frontier",
                "scenario_id": scenario_id,
                "mechanism_target": scenario["mechanism_focus"],
                "status": scenario["readiness_gate"],
                "why_now": scenario["next_action"],
                "command": command,
                "preflight_command": "python scripts/build_qwen_target_model_registry.py --output-dir results/qwen_target_model_registry",
                "priority_score": float(scenario["priority_score"]),
                "expected_decision_update": expected,
            }
        )

    manual = scenario_priority[scenario_priority["readiness_gate"] == "manual_model_resolution_first"]
    for offset, (_, scenario) in enumerate(manual.iterrows(), start=len(rows) + 1):
        rows.append(
            {
                "base_rank": offset,
                "queue_item": f"{scenario['scenario_id']}_resolve_weights",
                "scenario_id": scenario["scenario_id"],
                "mechanism_target": "downstream_lab_or_domain_finetune_source_resolution",
                "status": "manual_model_resolution_first",
                "why_now": "Downstream user/lab finetunes matter, but only concrete same-shape weights can be evaluated.",
                "command": "python scripts/build_qwen_target_model_registry.py --output-dir results/qwen_target_model_registry",
                "preflight_command": "python -m py_compile scripts/build_qwen_target_model_registry.py",
                "priority_score": max(0.30, float(scenario["priority_score"]) - 0.05),
                "expected_decision_update": "replace paper/pool placeholders with concrete same-shape endpoints or adapters",
            }
        )

    out = pd.DataFrame(rows).sort_values(["priority_score", "base_rank"], ascending=[False, True])
    out = out.reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    columns = [
        "rank",
        "queue_item",
        "scenario_id",
        "mechanism_target",
        "status",
        "why_now",
        "command",
        "preflight_command",
        "priority_score",
        "expected_decision_update",
    ]
    return out[columns]


def build_report(
    summary: dict[str, Any],
    scenario_priority: pd.DataFrame,
    candidate_source_sets: pd.DataFrame,
    endpoint_eval_expansion: pd.DataFrame,
    source_discovery_queue: pd.DataFrame,
) -> str:
    top = summary.get("top_measured_source_set") or {}
    top_scenario = summary.get("top_scenario") or {}
    lines = [
        "# Qwen Source Discovery Plan",
        "",
        "## Result",
        "",
        (
            f"当前最好的 measured source set 是 `{top.get('source_set')}`，source weights "
            f"`{top.get('source_weights')}`；它的 endpoint frontier avg gain 是 "
            f"`{fmt(top.get('frontier_avg_gain_vs_best_single'))}`，但已观测 merge interference budget 是 "
            f"`{fmt(summary.get('interference_budget'))}`，还差 "
            f"`{fmt(summary.get('measured_additional_frontier_avg_gain_needed'))}`。所以它只能进入 probe-only / endpoint-expansion，不能作为 final average budget。"
        ),
        "",
        (
            f"下一批 source discovery 的最高优先场景是 `{top_scenario.get('scenario_id')}`："
            f"`{top_scenario.get('next_action')}`。这一步的目标不是机械比较算法，而是找到 "
            "`source_frontier_gain - observed_merge_interference_budget >= 0` 的源集合，然后再做同构 average。"
        ),
        "",
        "## Planned Source Sets",
        "",
        "| candidate | scenario | gate | mechanism | extra gain needed | next action |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for _, row in candidate_source_sets.iterrows():
        lines.append(
            f"| `{row['candidate_id']}` | `{row['scenario_id']}` | `{row['readiness_gate']}` | "
            f"{row['mechanism_focus']} | {fmt(row['additional_frontier_avg_gain_needed'])} | "
            f"{row['next_action']} |"
        )

    lines.extend(
        [
            "",
            "## Scenario Priority",
            "",
            "| rank | scenario | priority | ready/manual | mechanism focus | next action |",
            "| ---: | --- | ---: | --- | --- | --- |",
        ]
    )
    for _, row in scenario_priority.iterrows():
        lines.append(
            f"| {int(row['rank'])} | `{row['scenario_id']}` | {fmt(row['priority_score'], 2)} | "
            f"{row['ready_model_count']}/{row['manual_resolution_count']} | "
            f"{row['mechanism_focus']} | {row['next_action']} |"
        )

    lines.extend(
        [
            "",
            "## Endpoint Eval Expansion",
            "",
            "| scenario | capability | benchmarks | metric | min examples | runner |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for _, row in endpoint_eval_expansion.iterrows():
        lines.append(
            f"| `{row['scenario_id']}` | `{row['capability']}` | {row['benchmark_slice']} | "
            f"{row['metric']} | {int(row['min_examples'])} | `{row['runner_scope']}` |"
        )

    lines.extend(
        [
            "",
            "## Source Discovery Queue",
            "",
            "| rank | item | status | priority | command | expected update |",
            "| ---: | --- | --- | ---: | --- | --- |",
        ]
    )
    for _, row in source_discovery_queue.iterrows():
        lines.append(
            f"| {int(row['rank'])} | `{row['queue_item']}` | `{row['status']}` | "
            f"{fmt(row['priority_score'], 2)} | `{row['command']}` | {row['expected_decision_update']} |"
        )

    lines.extend(
        [
            "",
            "## Literature Priors",
            "",
            "| key | source | mechanism used here |",
            "| --- | --- | --- |",
        ]
    )
    for prior in LITERATURE_PRIORS:
        lines.append(f"| `{prior['key']}` | {prior['source']} | {prior['mechanism']} |")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['candidate_source_sets']}`",
            f"- `{summary['outputs']['endpoint_eval_expansion']}`",
            f"- `{summary['outputs']['scenario_priority']}`",
            f"- `{summary['outputs']['source_discovery_queue']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    registry = read_csv(args.model_registry)
    scenarios = read_csv(args.scenario_matrix)
    eval_matrix = read_csv(args.eval_probe_matrix)
    optimizer = read_json(args.average_source_set_optimizer)
    optimizer_candidates = read_csv(args.average_source_set_candidates)
    source_set_gate = read_csv(args.source_set_gate)

    scenario_priority = build_scenario_priority(registry, scenarios)
    measured = measured_top_row(optimizer, optimizer_candidates)
    candidate_source_sets = build_candidate_source_sets(scenario_priority, optimizer, optimizer_candidates)
    endpoint_eval_expansion = build_endpoint_eval_expansion(scenario_priority, eval_matrix, measured)
    source_discovery_queue = build_source_discovery_queue(scenario_priority, candidate_source_sets, measured)

    non_control = scenario_priority[scenario_priority["scenario_id"].astype(str) != "negative_controls"]
    top_scenario = non_control.iloc[0].to_dict() if not non_control.empty else {}
    top_queue = source_discovery_queue.iloc[0].to_dict() if not source_discovery_queue.empty else {}
    top_measured = optimizer.get("top_source_set") or {}
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = output_dir / "candidate_source_sets.csv"
    endpoint_path = output_dir / "endpoint_eval_expansion.csv"
    scenario_path = output_dir / "scenario_priority.csv"
    queue_path = output_dir / "source_discovery_queue.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    summary = {
        "schema_version": 1,
        "status": "source_discovery_plan_ready",
        "registry_model_count": int(len(registry)),
        "scenario_count": int(len(scenario_priority)),
        "candidate_source_set_count": int(len(candidate_source_sets)),
        "endpoint_eval_expansion_count": int(len(endpoint_eval_expansion)),
        "source_discovery_queue_count": int(len(source_discovery_queue)),
        "ready_p0_scenario_count": int(
            len(
                scenario_priority[
                    scenario_priority["priority"].astype(str).str.startswith("p0")
                    & scenario_priority["readiness_gate"].isin(["ready_for_endpoint_eval_and_surplus_gate"])
                ]
            )
        ),
        "manual_resolution_scenario_count": int(
            len(scenario_priority[scenario_priority["readiness_gate"] == "manual_model_resolution_first"])
        ),
        "interference_budget": fnum(optimizer.get("interference_budget")),
        "interference_budget_source": optimizer.get("interference_budget_source"),
        "measured_additional_frontier_avg_gain_needed": measured.get(
            "additional_frontier_avg_gain_needed"
        ),
        "top_measured_source_set": top_measured,
        "top_scenario": {
            "scenario_id": top_scenario.get("scenario_id"),
            "priority_score": fnum(top_scenario.get("priority_score")),
            "readiness_gate": top_scenario.get("readiness_gate"),
            "next_action": top_scenario.get("next_action"),
            "mechanism_focus": top_scenario.get("mechanism_focus"),
        },
        "top_queue_item": {
            "queue_item": top_queue.get("queue_item"),
            "scenario_id": top_queue.get("scenario_id"),
            "status": top_queue.get("status"),
            "priority_score": fnum(top_queue.get("priority_score")),
            "command": top_queue.get("command"),
        },
        "source_set_gate_rows": int(len(source_set_gate)),
        "literature_priors": LITERATURE_PRIORS,
        "mechanism_equation": "promote average only if source_frontier_avg_gain - observed_merge_interference_budget >= 0",
        "outputs": {
            "candidate_source_sets": rel(candidate_path),
            "endpoint_eval_expansion": rel(endpoint_path),
            "scenario_priority": rel(scenario_path),
            "source_discovery_queue": rel(queue_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }

    candidate_source_sets.to_csv(candidate_path, index=False)
    endpoint_eval_expansion.to_csv(endpoint_path, index=False)
    scenario_priority.to_csv(scenario_path, index=False)
    source_discovery_queue.to_csv(queue_path, index=False)
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(
        build_report(summary, scenario_priority, candidate_source_sets, endpoint_eval_expansion, source_discovery_queue),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Qwen source discovery and endpoint-eval expansion plan.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen_source_discovery_plan"))
    parser.add_argument("--model-registry", type=Path, default=Path("results/qwen_target_model_registry/model_registry.csv"))
    parser.add_argument("--scenario-matrix", type=Path, default=Path("results/qwen_target_model_registry/scenario_matrix.csv"))
    parser.add_argument("--eval-probe-matrix", type=Path, default=Path("results/qwen_target_model_registry/eval_probe_matrix.csv"))
    parser.add_argument(
        "--average-source-set-optimizer",
        type=Path,
        default=Path("results/qwen3_average_source_set_optimizer/summary.json"),
    )
    parser.add_argument(
        "--average-source-set-candidates",
        type=Path,
        default=Path("results/qwen3_average_source_set_optimizer/candidate_source_sets.csv"),
    )
    parser.add_argument(
        "--source-set-gate",
        type=Path,
        default=Path("results/qwen3_source_set_complementarity_gate/source_set_gate.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args)
    print(f"Wrote Qwen source discovery plan to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; top_scenario={summary['top_scenario']['scenario_id']}; "
        f"additional_gain_needed={fmt(summary['measured_additional_frontier_avg_gain_needed'])}"
    )


if __name__ == "__main__":
    main()
