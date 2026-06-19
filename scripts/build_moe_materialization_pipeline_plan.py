#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def read_json_if_exists(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def shell_quote(command: str) -> str:
    return command.replace("\n", " ").strip()


def gate(
    *,
    stage: str,
    status: str,
    evidence: str,
    next_action: str,
    command: str = "",
    artifact: str = "",
) -> dict[str, str]:
    return {
        "stage": stage,
        "status": status,
        "evidence": evidence,
        "next_action": next_action,
        "command": shell_quote(command),
        "artifact": artifact,
    }


def registry_gate(registry: dict[str, Any]) -> dict[str, str]:
    models = registry.get("recommended_first_moe_models", [])
    status = "complete" if models else "missing"
    return gate(
        stage="target_model_selection",
        status=status,
        evidence=f"{len(models)} recommended p0 MoE models: {', '.join(models) if models else 'none'}",
        next_action="inspect exact topology/header for every selected source model",
        command="python scripts/build_qwen_target_model_registry.py",
        artifact="results/qwen_target_model_registry/report.md",
    )


def topology_gate(topology: dict[str, Any], registry: dict[str, Any]) -> dict[str, str]:
    models = topology.get("models", [])
    moe_models = [model for model in models if model.get("config", {}).get("is_moe_config")]
    weights_available = [model for model in moe_models if model.get("headers", {}).get("weights_available")]
    target_models = registry.get("recommended_first_moe_models", [])
    if len(weights_available) >= len(target_models) and target_models:
        status = "complete"
        next_action = "run routing probes on all exact p0 MoE sources"
    elif moe_models:
        status = "partial_config_only"
        next_action = "inspect exact Qwen3-30B-A3B source checkpoints with safetensors headers"
    else:
        status = "missing"
        next_action = "run topology inspection on p0 MoE source checkpoints"
    return gate(
        stage="exact_moe_topology",
        status=status,
        evidence=(
            f"inspected_moe_configs={len(moe_models)}, weights_available={len(weights_available)}, "
            f"target_p0_moe={len(target_models)}"
        ),
        next_action=next_action,
        command="python scripts/inspect_checkpoint_topology.py --model NAME=MODEL_PATH --output-dir results/checkpoint_topology_inspect",
        artifact="results/checkpoint_topology_inspect/report.md",
    )


def routing_probe_gate(route_weight_summary: dict[str, Any]) -> dict[str, str]:
    plans = route_weight_summary.get("routing_probe_plan", [])
    command = plans[0].get("command", "") if plans else ""
    output_dir = plans[0].get("output_dir", "") if plans else ""
    expected_summary = repo_path(output_dir) / "summary.json" if output_dir else None
    if expected_summary and expected_summary.exists():
        status = "complete"
        evidence = f"routing probe exists at {rel(expected_summary)}"
        next_action = "run routing readiness and route-weight recipe builders"
    elif plans:
        status = "ready_to_run"
        evidence = f"{len(plans)} routing probe command planned; expected output {output_dir or 'unknown'}"
        next_action = "run the planned Qwen3 MoE routing probe before materialization"
    else:
        status = "missing_plan"
        evidence = "no routing probe command available"
        next_action = "build route-weight recipe plan with model/prompt/source mapping"
    return gate(
        stage="real_moe_routing_probe",
        status=status,
        evidence=evidence,
        next_action=next_action,
        command=command,
        artifact="results/moe_route_weight_recipes/routing_probe_plan.csv",
    )


def readiness_gate(readiness: dict[str, Any], route_weight_summary: dict[str, Any]) -> dict[str, str]:
    status = str(readiness.get("readiness_status", "missing"))
    plans = route_weight_summary.get("routing_probe_plan", [])
    output_dir = plans[0].get("output_dir", "results/moe_routing_probe/qwen3_30b_general_vs_code") if plans else "results/moe_routing_probe/qwen3_30b_general_vs_code"
    if status == "waiting_for_routing_probe":
        next_action = "run routing probe, then rerun readiness analysis on its output"
    else:
        next_action = "review router/expert actions before enabling any router or expert averaging"
    return gate(
        stage="routing_readiness_gate",
        status=status,
        evidence=(
            f"router_rows={readiness.get('router_rows', 0)}, "
            f"expert_rows={readiness.get('expert_rows', 0)}, "
            f"specialization_rows={readiness.get('specialization_rows', 0)}"
        ),
        next_action=next_action,
        command=f"PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir {output_dir}",
        artifact="results/moe_routing_readiness/report.md",
    )


def route_weight_gate(route_weight_summary: dict[str, Any]) -> dict[str, str]:
    status = str(route_weight_summary.get("recipe_status", "missing"))
    plans = route_weight_summary.get("routing_probe_plan", [])
    command = plans[0].get("next_recipe_command", "") if plans else ""
    if status == "waiting_for_routing_probe":
        next_action = "run routing probe first; route weights require real expert_load.csv"
    elif status.endswith("ready") or route_weight_summary.get("tensor_rule_count", 0):
        next_action = "feed tensor_rules.txt into the combined writer recipe"
    else:
        next_action = "inspect recipe report and missing inputs"
    return gate(
        stage="route_weight_tensor_rules",
        status=status,
        evidence=(
            f"expert_rules={route_weight_summary.get('expert_rule_count', 0)}, "
            f"tensor_rules={route_weight_summary.get('tensor_rule_count', 0)}, "
            f"router_dirs={len(route_weight_summary.get('router_dirs', []))}"
        ),
        next_action=next_action,
        command=command,
        artifact="results/moe_route_weight_recipes/report.md",
    )


def expert_remap_gate(layer_smoke: dict[str, Any]) -> dict[str, str]:
    status = "template_validated" if layer_smoke.get("status") == "passed" else "needs_writer_fix"
    return gate(
        stage="layerwise_expert_remap",
        status=status,
        evidence=(
            f"layer_aware_rules={layer_smoke.get('layer_aware_rule_count', 0)}, "
            f"manual_review_rows={layer_smoke.get('manual_review_count', 0)} in smoke"
        ),
        next_action="run layer-wise expert-output matching on real Qwen3 MoE sources, then build source tensor aliases",
        command=(
            "PYTHONPATH=src python scripts/build_moe_expert_remap_plan.py "
            "--expert-match results/moe_expert_match/qwen3_30b_general_vs_code/expert_match.csv "
            "--output-dir results/moe_expert_remap/qwen3_30b_general_vs_code --source-name code"
        ),
        artifact="results/moe_layerwise_expert_remap_smoke/report.md",
    )


def router_bias_gate(route_weight_summary: dict[str, Any]) -> dict[str, str]:
    plans = route_weight_summary.get("routing_probe_plan", [])
    output_dir = plans[0].get("output_dir", "results/moe_routing_probe/qwen3_30b_general_vs_code") if plans else "results/moe_routing_probe/qwen3_30b_general_vs_code"
    expert_load = repo_path(output_dir) / "expert_load.csv"
    status = "ready_to_build" if expert_load.exists() else "waiting_for_routing_probe"
    return gate(
        stage="router_bias_capacity_delta",
        status=status,
        evidence=f"expert_load_exists={expert_load.exists()} at {rel(expert_load)}",
        next_action="convert real top-k load imbalance into same-shape router-bias deltas",
        command=(
            "PYTHONPATH=src python scripts/build_moe_router_bias_plan.py "
            f"--router-dir {output_dir} --output-dir results/moe_router_bias_plan/qwen3_30b_general_vs_code "
            "--router-bias-template '{router}.bias'"
        ),
        artifact="results/moe_confidence_blended_router_bias_plan/report.md",
    )


def combined_recipe_gate(combined: dict[str, Any]) -> dict[str, str]:
    status = str(combined.get("recipe_status", "missing"))
    missing = combined.get("missing_inputs", [])
    return gate(
        stage="combined_same_shape_writer_recipe",
        status="template_validated" if status == "combined_writer_command_ready" and not missing else status,
        evidence=(
            f"tensor_rules={combined.get('tensor_rule_count', 0)}, "
            f"alias_rules={combined.get('alias_rule_count', 0)}, "
            f"bias_delta_rows={combined.get('router_bias_delta_rows', 0)}"
        ),
        next_action="replace toy/template inputs with real route-weight rules, layer-wise aliases, and router-bias deltas",
        command=(
            "PYTHONPATH=src python scripts/build_moe_combined_materialization_recipe.py "
            "--tensor-rule-file results/moe_route_weight_recipes/tensor_rules.txt "
            "--source-tensor-alias-file results/moe_expert_remap/qwen3_30b_general_vs_code/source_tensor_aliases.txt "
            "--tensor-add-csv results/moe_router_bias_plan/qwen3_30b_general_vs_code/router_bias_deltas.csv"
        ),
        artifact="results/moe_confidence_blended_combined_recipe/report.md",
    )


def materialization_gate(readiness: dict[str, Any]) -> dict[str, str]:
    blocked = int(readiness.get("blocked_by_placeholder_count", 0))
    materialized = int(readiness.get("materialized_count", 0))
    return gate(
        stage="checkpoint_materialization",
        status="blocked_by_real_source_paths" if blocked else "ready_or_materialized",
        evidence=f"materialized={materialized}, blocked_by_placeholders={blocked}",
        next_action="after real source paths and recipes are ready, run writer dry-run, then materialize safetensors",
        command="PYTHONPATH=src python scripts/build_checkpoint_materialization_readiness.py --output-dir results/checkpoint_materialization_readiness",
        artifact="results/checkpoint_materialization_readiness/report.md",
    )


def vllm_gate(vllm: dict[str, Any]) -> dict[str, str]:
    return gate(
        stage="hosted_downstream_eval",
        status="waiting_for_materialized_moe_checkpoint",
        evidence=(
            f"checkpoint_eval_plan_status={vllm.get('status')}, "
            f"completed={vllm.get('completed_eval_count', 0)}, "
            f"missing_checkpoint={vllm.get('missing_checkpoint_count', 0)}"
        ),
        next_action="host the materialized MoE checkpoint with vLLM and run the same downstream eval harness",
        command="PYTHONPATH=src python scripts/build_vllm_checkpoint_eval_plan.py --output-dir results/vllm_checkpoint_eval_plan",
        artifact="results/vllm_checkpoint_eval_plan/report.md",
    )


def build_report(summary: dict[str, Any], gates: list[dict[str, str]]) -> str:
    lines = [
        "# MoE Materialization Pipeline Plan",
        "",
        "这个 plan 把真实 Qwen MoE average 从目标模型选择到 vLLM eval 的每个 gate 串起来。它区分 toy/template 已验证能力和真实 Qwen3 MoE 仍缺的 probe 输入，避免把 toy readiness 误当成真实 checkpoint readiness。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Current blocking stage: `{summary['current_blocking_stage']}`",
        f"- Gates: `{summary['gate_count']}`",
        f"- Ready/complete gates: `{summary['ready_or_complete_count']}`",
        f"- Waiting/blocked gates: `{summary['waiting_or_blocked_count']}`",
        "",
        "| stage | status | evidence | next action |",
        "| --- | --- | --- | --- |",
    ]
    for row in gates:
        lines.append(
            f"| `{row['stage']}` | `{row['status']}` | {row['evidence']} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Next Commands",
            "",
            "```bash",
        ]
    )
    for row in gates:
        if row["command"]:
            lines.append(f"# {row['stage']}")
            lines.append(row["command"])
    lines.extend(
        [
            "```",
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['stage_gates']}`",
            f"- `{summary['outputs']['next_commands']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an end-to-end gate plan for real MoE materialization.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_materialization_pipeline_plan"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = read_json("results/qwen_target_model_registry/summary.json")
    topology = read_json("results/checkpoint_topology_inspect/summary.json")
    route_weight = read_json("results/moe_route_weight_recipes/summary.json")
    routing_readiness = read_json("results/moe_routing_readiness/summary.json")
    layer_smoke = read_json("results/moe_layerwise_expert_remap_smoke/summary.json")
    combined = read_json("results/moe_confidence_blended_combined_recipe/summary.json")
    materialization = read_json("results/checkpoint_materialization_readiness/summary.json")
    vllm = read_json_if_exists("results/vllm_checkpoint_eval_plan/summary.json")

    gates = [
        registry_gate(registry),
        topology_gate(topology, registry),
        routing_probe_gate(route_weight),
        readiness_gate(routing_readiness, route_weight),
        route_weight_gate(route_weight),
        expert_remap_gate(layer_smoke),
        router_bias_gate(route_weight),
        combined_recipe_gate(combined),
        materialization_gate(materialization),
        vllm_gate(vllm),
    ]
    waiting_statuses = {"missing", "missing_plan", "waiting_for_routing_probe", "blocked_by_real_source_paths", "waiting_for_materialized_moe_checkpoint", "partial_config_only", "ready_to_run"}
    current_blocker = next((row["stage"] for row in gates if row["status"] in waiting_statuses), "")
    ready_statuses = {"complete", "template_validated", "ready_or_materialized"}
    summary = {
        "schema_version": 1,
        "status": "waiting_for_real_moe_probe_or_paths" if current_blocker else "ready_for_materialization",
        "current_blocking_stage": current_blocker,
        "gate_count": len(gates),
        "ready_or_complete_count": sum(1 for row in gates if row["status"] in ready_statuses),
        "waiting_or_blocked_count": sum(1 for row in gates if row["status"] in waiting_statuses),
        "recommended_first_moe_models": registry.get("recommended_first_moe_models", []),
        "same_shape_constraint": "The pipeline never changes router shape, expert count, hidden size, tokenizer, or model class.",
        "outputs": {
            "stage_gates": rel(output_dir / "stage_gates.csv"),
            "next_commands": rel(output_dir / "next_commands.sh"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    pd.DataFrame(gates).to_csv(output_dir / "stage_gates.csv", index=False)
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for row in gates:
        if row["command"]:
            commands.append(f"# {row['stage']} [{row['status']}]")
            commands.append(row["command"])
            commands.append("")
    command_path = output_dir / "next_commands.sh"
    command_path.write_text("\n".join(commands), encoding="utf-8")
    command_path.chmod(command_path.stat().st_mode | stat.S_IXUSR)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary, gates), encoding="utf-8")
    print(f"Wrote MoE materialization pipeline plan to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
