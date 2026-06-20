#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def command_text(command: list[str]) -> str:
    return " ".join(command)


def run_step(step: dict[str, Any], *, plan_only: bool) -> dict[str, Any]:
    started = time.time()
    result = {
        "step": step["step"],
        "kind": step["kind"],
        "command": command_text(step["command"]),
        "status": "planned" if plan_only else "running",
        "returncode": None,
        "duration_sec": 0.0,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if plan_only:
        return result
    completed = subprocess.run(
        step["command"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    result.update(
        {
            "status": "passed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "duration_sec": time.time() - started,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    )
    return result


def build_steps(args: argparse.Namespace) -> list[dict[str, Any]]:
    py = "python"
    steps = [
        {
            "step": "build_candidate_trust_region_gate",
            "kind": "gate",
            "command": [
                py,
                "scripts/build_qwen3_moe_candidate_trust_region_gate.py",
                "--gate-plan",
                str(args.gate_dir / "eval_gate_plan.csv"),
                "--output-dir",
                str(args.candidate_trust_region_gate_dir),
            ],
        },
        {
            "step": "plan_eval_budget",
            "kind": "planner",
            "command": [
                py,
                "scripts/plan_qwen3_moe_eval_budget.py",
                "--gate-dir",
                str(args.gate_dir),
                "--candidate-trust-gate",
                str(args.candidate_trust_region_gate_dir / "candidate_trust_region_gate.csv"),
                "--output-dir",
                str(args.eval_budget_dir),
            ],
        },
        {
            "step": "audit_eval_bundles",
            "kind": "gate",
            "command": [
                py,
                "scripts/audit_qwen3_moe_eval_bundle.py",
                "--gate-dir",
                str(args.gate_dir),
                "--output-dir",
                str(args.audit_dir),
            ],
        },
        {
            "step": "select_unified_result",
            "kind": "selector",
            "command": [
                py,
                "scripts/select_qwen3_moe_unified_result.py",
                "--gate-dir",
                str(args.gate_dir),
                "--output-dir",
                str(args.selection_dir),
            ],
        },
        {
            "step": "select_final_candidate",
            "kind": "selector",
            "command": [
                py,
                "scripts/select_qwen3_moe_final_candidate.py",
                "--gate-dir",
                str(args.gate_dir),
                "--audit-dir",
                str(args.audit_dir),
                "--output-dir",
                str(args.final_selection_dir),
                "--candidate-trust-gate",
                str(args.candidate_trust_region_gate_dir / "candidate_trust_region_gate.csv"),
            ],
        },
        {
            "step": "attribute_mechanism_effects",
            "kind": "attribution",
            "command": [
                py,
                "scripts/attribute_qwen3_moe_mechanism_effects.py",
                "--gate-dir",
                str(args.gate_dir),
                "--audit-dir",
                str(args.audit_dir),
                "--output-dir",
                str(args.attribution_dir),
            ],
        },
        {
            "step": "build_feedback_optimizer",
            "kind": "optimizer",
            "command": [
                py,
                "scripts/build_qwen3_moe_feedback_optimizer.py",
                "--gate-dir",
                str(args.gate_dir),
                "--audit-dir",
                str(args.audit_dir),
                "--output-dir",
                str(args.feedback_dir),
            ],
        },
        {
            "step": "build_mechanistic_unified_candidate",
            "kind": "optimizer",
            "command": [
                py,
                "scripts/build_qwen3_moe_mechanistic_unified_candidate.py",
                "--output-dir",
                str(args.mechanistic_dir),
            ],
        },
        {
            "step": "audit_mechanistic_evidence",
            "kind": "attribution",
            "command": [
                py,
                "scripts/audit_qwen3_moe_mechanistic_evidence.py",
                "--output-dir",
                str(args.mechanistic_evidence_dir),
            ],
        },
        {
            "step": "analyze_mechanistic_sensitivity",
            "kind": "attribution",
            "command": [
                py,
                "scripts/analyze_qwen3_moe_mechanistic_sensitivity.py",
                "--output-dir",
                str(args.mechanistic_sensitivity_dir),
            ],
        },
        {
            "step": "analyze_router_expert_coupling",
            "kind": "attribution",
            "command": [
                py,
                "scripts/analyze_qwen3_moe_router_expert_coupling.py",
                "--output-dir",
                str(args.router_expert_coupling_dir),
            ],
        },
        {
            "step": "build_router_coupled_candidate",
            "kind": "optimizer",
            "command": [
                py,
                "scripts/build_qwen3_moe_router_coupled_candidate.py",
                "--output-dir",
                str(args.router_coupled_candidate_dir),
            ],
        },
        {
            "step": "analyze_router_coupled_retention_frontier",
            "kind": "attribution",
            "command": [
                py,
                "scripts/analyze_qwen3_moe_router_coupled_retention_frontier.py",
                "--output-dir",
                str(args.router_coupled_retention_frontier_dir),
            ],
        },
        {
            "step": "build_source_set_complementarity_gate",
            "kind": "gate",
            "command": [
                py,
                "scripts/build_qwen3_source_set_complementarity_gate.py",
                "--output-dir",
                str(args.source_set_complementarity_dir),
            ],
        },
        {
            "step": "build_average_source_set_optimizer",
            "kind": "optimizer",
            "command": [
                py,
                "scripts/build_qwen3_average_source_set_optimizer.py",
                "--output-dir",
                str(args.average_source_set_optimizer_dir),
            ],
        },
        {
            "step": "build_qwen_source_discovery_plan",
            "kind": "planner",
            "command": [
                py,
                "scripts/build_qwen_source_discovery_plan.py",
                "--output-dir",
                str(args.qwen_source_discovery_plan_dir),
            ],
        },
        {
            "step": "build_qwen_source_discovery_eval_plan",
            "kind": "planner",
            "command": [
                py,
                "scripts/build_qwen_source_discovery_eval_plan.py",
                "--source-discovery-dir",
                str(args.qwen_source_discovery_plan_dir),
                "--output-dir",
                str(args.qwen_source_discovery_eval_plan_dir),
            ],
        },
        {
            "step": "audit_qwen_source_discovery_served_model_preflight",
            "kind": "gate",
            "command": [
                py,
                "scripts/audit_vllm_served_model_preflight.py",
                "--eval-jobs",
                str(args.qwen_source_discovery_eval_plan_dir / "vllm_eval_jobs.csv"),
                "--output-dir",
                str(args.qwen_source_discovery_served_model_preflight_dir),
            ],
        },
        {
            "step": "build_qwen_source_frontier_eval_feedback",
            "kind": "gate",
            "command": [
                py,
                "scripts/build_qwen_source_frontier_eval_feedback.py",
                "--eval-jobs",
                str(args.qwen_source_discovery_eval_plan_dir / "vllm_eval_jobs.csv"),
                "--average-source-set-optimizer",
                str(args.average_source_set_optimizer_dir / "summary.json"),
                "--output-dir",
                str(args.qwen_source_frontier_eval_feedback_dir),
            ],
        },
        {
            "step": "build_router_calibration_frontier",
            "kind": "gate",
            "command": [
                py,
                "scripts/build_qwen3_moe_router_calibration_frontier.py",
                "--output-dir",
                str(args.router_calibration_frontier_dir),
            ],
        },
        {
            "step": "collect_qwen3_moe_harc_router_stats",
            "kind": "probe",
            "command": [
                py,
                "scripts/collect_qwen3_moe_harc_router_stats.py",
                "--output-dir",
                str(args.harc_router_stats_dir),
            ],
        },
        {
            "step": "build_qwen3_moe_harc_readiness_gate",
            "kind": "gate",
            "command": [
                py,
                "scripts/build_qwen3_moe_harc_readiness_gate.py",
                "--output-dir",
                str(args.harc_readiness_gate_dir),
                "--harc-stats-dir",
                str(args.harc_router_stats_dir),
                "--harc-stats-summary",
                str(args.harc_router_stats_dir / "summary.json"),
            ],
        },
        {
            "step": "build_unified_average_optimizer",
            "kind": "optimizer",
            "command": [
                py,
                "scripts/build_unified_average_optimizer.py",
                "--output-dir",
                str(args.unified_optimizer_dir),
                "--qwen-source-discovery-plan",
                str(args.qwen_source_discovery_plan_dir / "summary.json"),
                "--qwen-source-discovery-eval-plan",
                str(args.qwen_source_discovery_eval_plan_dir / "summary.json"),
                "--qwen-source-frontier-eval-feedback",
                str(args.qwen_source_frontier_eval_feedback_dir / "summary.json"),
                "--qwen3-router-calibration-frontier",
                str(args.router_calibration_frontier_dir / "summary.json"),
            ],
        },
        {
            "step": "build_average_method_gate_matrix",
            "kind": "optimizer",
            "command": [
                py,
                "scripts/build_average_method_gate_matrix.py",
                "--output-dir",
                str(args.average_method_gate_dir),
                "--optimizer-summary",
                str(args.unified_optimizer_dir / "summary.json"),
                "--optimizer-features",
                str(args.unified_optimizer_dir / "mechanism_features.csv"),
            ],
        },
        {
            "step": "build_average_trust_region_bounds",
            "kind": "optimizer",
            "command": [
                py,
                "scripts/build_average_trust_region_bounds.py",
                "--output-dir",
                str(args.average_trust_region_bounds_dir),
            ],
        },
        {
            "step": "analyze_mechanism_levers",
            "kind": "attribution",
            "command": [
                py,
                "scripts/analyze_qwen3_moe_mechanism_levers.py",
                "--eval-budget-dir",
                str(args.eval_budget_dir),
                "--output-dir",
                str(args.mechanism_levers_dir),
            ],
        },
    ]
    if args.include_smoke:
        steps.extend(
            [
                {
                    "step": "audit_eval_bundles_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/audit_qwen3_moe_eval_bundle.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.audit_smoke_dir),
                    ],
                },
                {
                    "step": "select_unified_result_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/select_qwen3_moe_unified_result.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.selection_smoke_dir),
                    ],
                },
                {
                    "step": "select_final_candidate_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/select_qwen3_moe_final_candidate.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.final_selection_smoke_dir),
                    ],
                },
                {
                    "step": "eval_budget_queue_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/smoke_qwen3_moe_eval_budget_queue.py",
                        "--eval-budget-dir",
                        str(args.eval_budget_dir),
                        "--candidate-trust-gate",
                        str(args.candidate_trust_region_gate_dir / "candidate_trust_region_gate.csv"),
                        "--output-dir",
                        str(args.eval_budget_smoke_dir),
                    ],
                },
                {
                    "step": "build_qwen_source_frontier_eval_feedback_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/build_qwen_source_frontier_eval_feedback.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.qwen_source_frontier_eval_feedback_smoke_dir),
                    ],
                },
                {
                    "step": "collect_qwen3_moe_harc_router_stats_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/collect_qwen3_moe_harc_router_stats.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.harc_router_stats_smoke_dir),
                    ],
                },
                {
                    "step": "build_qwen3_moe_harc_readiness_gate_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/build_qwen3_moe_harc_readiness_gate.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.harc_readiness_gate_smoke_dir),
                    ],
                },
                {
                    "step": "attribute_mechanism_effects_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/attribute_qwen3_moe_mechanism_effects.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.attribution_smoke_dir),
                    ],
                },
                {
                    "step": "build_feedback_optimizer_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/build_qwen3_moe_feedback_optimizer.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.feedback_smoke_dir),
                    ],
                },
                {
                    "step": "build_mechanistic_unified_candidate_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/build_qwen3_moe_mechanistic_unified_candidate.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.mechanistic_smoke_dir),
                    ],
                },
                {
                    "step": "unified_average_optimizer_ledger_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/smoke_unified_average_optimizer_ledger.py",
                        "--summary",
                        str(args.unified_optimizer_dir / "summary.json"),
                        "--output-dir",
                        str(args.unified_optimizer_smoke_dir),
                    ],
                },
                {
                    "step": "average_method_gate_matrix_consistency_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/smoke_average_method_gate_matrix.py",
                        "--optimizer-summary",
                        str(args.unified_optimizer_dir / "summary.json"),
                        "--method-gate-dir",
                        str(args.average_method_gate_dir),
                        "--output-dir",
                        str(args.average_method_gate_smoke_dir),
                    ],
                },
                {
                    "step": "average_trust_region_bounds_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/smoke_average_trust_region_bounds.py",
                        "--bounds-dir",
                        str(args.average_trust_region_bounds_dir),
                        "--output-dir",
                        str(args.average_trust_region_bounds_smoke_dir),
                    ],
                },
            ]
        )
    if not args.skip_collect:
        steps.append(
            {
                "step": "collect_results",
                "kind": "summary",
                "command": [py, "scripts/collect_results.py"],
            }
        )
    return steps


def downstream_status(args: argparse.Namespace) -> dict[str, Any]:
    audit = read_json(repo_path(args.audit_dir) / "summary.json")
    selection = read_json(repo_path(args.selection_dir) / "summary.json")
    final_selection = read_json(repo_path(args.final_selection_dir) / "summary.json")
    candidate_trust_region_gate = read_json(repo_path(args.candidate_trust_region_gate_dir) / "summary.json")
    eval_budget = read_json(repo_path(args.eval_budget_dir) / "summary.json")
    attribution = read_json(repo_path(args.attribution_dir) / "summary.json")
    feedback = read_json(repo_path(args.feedback_dir) / "summary.json")
    mechanistic = read_json(repo_path(args.mechanistic_dir) / "summary.json")
    mechanistic_evidence = read_json(repo_path(args.mechanistic_evidence_dir) / "summary.json")
    mechanistic_sensitivity = read_json(repo_path(args.mechanistic_sensitivity_dir) / "summary.json")
    router_expert_coupling = read_json(repo_path(args.router_expert_coupling_dir) / "summary.json")
    router_coupled_candidate = read_json(repo_path(args.router_coupled_candidate_dir) / "summary.json")
    router_coupled_frontier = read_json(
        repo_path(args.router_coupled_retention_frontier_dir) / "summary.json"
    )
    source_set_complementarity = read_json(
        repo_path(args.source_set_complementarity_dir) / "summary.json"
    )
    average_source_set_optimizer = read_json(
        repo_path(args.average_source_set_optimizer_dir) / "summary.json"
    )
    qwen_source_discovery_plan = read_json(
        repo_path(args.qwen_source_discovery_plan_dir) / "summary.json"
    )
    qwen_source_discovery_eval_plan = read_json(
        repo_path(args.qwen_source_discovery_eval_plan_dir) / "summary.json"
    )
    qwen_source_discovery_served_model_preflight = read_json(
        repo_path(args.qwen_source_discovery_served_model_preflight_dir) / "summary.json"
    )
    qwen_source_frontier_eval_feedback = read_json(
        repo_path(args.qwen_source_frontier_eval_feedback_dir) / "summary.json"
    )
    qwen_source_frontier_eval_feedback_smoke = read_json(
        repo_path(args.qwen_source_frontier_eval_feedback_smoke_dir) / "summary.json"
    )
    router_calibration_frontier = read_json(
        repo_path(args.router_calibration_frontier_dir) / "summary.json"
    )
    harc_router_stats = read_json(repo_path(args.harc_router_stats_dir) / "summary.json")
    harc_router_stats_smoke = read_json(repo_path(args.harc_router_stats_smoke_dir) / "summary.json")
    harc_readiness_gate = read_json(repo_path(args.harc_readiness_gate_dir) / "summary.json")
    harc_readiness_gate_smoke = read_json(
        repo_path(args.harc_readiness_gate_smoke_dir) / "summary.json"
    )
    unified_optimizer = read_json(repo_path(args.unified_optimizer_dir) / "summary.json")
    unified_optimizer_smoke = read_json(repo_path(args.unified_optimizer_smoke_dir) / "summary.json")
    average_method_gate = read_json(repo_path(args.average_method_gate_dir) / "summary.json")
    average_method_gate_smoke = read_json(repo_path(args.average_method_gate_smoke_dir) / "summary.json")
    eval_budget_smoke = read_json(repo_path(args.eval_budget_smoke_dir) / "summary.json")
    average_trust_region_bounds = read_json(
        repo_path(args.average_trust_region_bounds_dir) / "summary.json"
    )
    average_trust_region_bounds_smoke = read_json(
        repo_path(args.average_trust_region_bounds_smoke_dir) / "summary.json"
    )
    mechanism_levers = read_json(repo_path(args.mechanism_levers_dir) / "summary.json")
    final_current = final_selection.get("current_selection") or {}
    optimizer_moe = unified_optimizer.get("moe") or {}
    optimizer_top = unified_optimizer.get("top_next_experiment") or {}
    return {
        "audit_status": audit.get("status"),
        "audit_usable_for_selection": audit.get("usable_for_selection_count"),
        "audit_method_count": audit.get("method_count"),
        "selection_status": selection.get("status"),
        "selected_method": (selection.get("current_selection") or {}).get("selected_method"),
        "selection_reason": (selection.get("current_selection") or {}).get("reason"),
        "final_selection_status": final_selection.get("status"),
        "final_selected_method": final_current.get("selected_method"),
        "final_selection_reason": final_current.get("reason"),
        "final_eligible_candidate_count": final_current.get("eligible_candidate_count"),
        "final_candidate_count": final_current.get("candidate_count"),
        "candidate_trust_region_gate_status": candidate_trust_region_gate.get("status"),
        "candidate_trust_region_final_selectable": candidate_trust_region_gate.get(
            "final_selectable_candidate_count"
        ),
        "candidate_trust_region_candidates": candidate_trust_region_gate.get("candidate_count"),
        "candidate_trust_region_ablation_only": candidate_trust_region_gate.get(
            "ablation_only_candidate_count"
        ),
        "eval_budget_status": eval_budget.get("status"),
        "eval_budget_default_runner_request": eval_budget.get("default_runner_request"),
        "eval_budget_final_core_method_count": eval_budget.get("final_core_method_count"),
        "eval_budget_final_core_prompt_budget": eval_budget.get("final_core_recommended_prompt_budget"),
        "eval_budget_mechanism_ablation_method_count": eval_budget.get("mechanism_ablation_method_count"),
        "eval_budget_recommended_max_examples": eval_budget.get("recommended_max_examples"),
        "eval_budget_smoke_status": eval_budget_smoke.get("status"),
        "eval_budget_smoke_passed": eval_budget_smoke.get("passed_assertion_count"),
        "eval_budget_smoke_assertions": eval_budget_smoke.get("assertion_count"),
        "attribution_status": attribution.get("status"),
        "attribution_scored_transition_count": attribution.get("scored_transition_count"),
        "attribution_transition_count": attribution.get("transition_count"),
        "feedback_status": feedback.get("status"),
        "feedback_scored_task_count": feedback.get("scored_task_count"),
        "feedback_task_count": feedback.get("task_count"),
        "feedback_regression_task_count": feedback.get("regression_task_count"),
        "feedback_changed_group_count": feedback.get("changed_group_count"),
        "mechanistic_status": mechanistic.get("status"),
        "mechanistic_selected_candidate": mechanistic.get("selected_candidate_id"),
        "mechanistic_retention": mechanistic.get("selected_nonbase_mass_retention"),
        "mechanistic_hard_cap_violations": mechanistic.get("selected_hard_cap_violation_count"),
        "mechanistic_evidence_status": mechanistic_evidence.get("status"),
        "mechanistic_evidence_gradient_agreement": mechanistic_evidence.get("gradient_sign_agreement_rate"),
        "mechanistic_evidence_objective_improved_fraction": mechanistic_evidence.get(
            "objective_proxy_improved_group_fraction"
        ),
        "mechanistic_evidence_hard_cap_bound_group_count": mechanistic_evidence.get(
            "hard_cap_bound_group_count"
        ),
        "mechanistic_sensitivity_status": mechanistic_sensitivity.get("status"),
        "mechanistic_sensitivity_strongest_objective_ablation": (
            mechanistic_sensitivity.get("strongest_fixed_objective_regression") or {}
        ).get("ablation"),
        "mechanistic_sensitivity_strongest_objective_delta": (
            mechanistic_sensitivity.get("strongest_fixed_objective_regression") or {}
        ).get("fixed_objective_delta"),
        "mechanistic_sensitivity_strongest_scale_ablation": (
            mechanistic_sensitivity.get("strongest_scale_sensitivity") or {}
        ).get("ablation"),
        "mechanistic_sensitivity_scale_shift": (
            mechanistic_sensitivity.get("strongest_scale_sensitivity") or {}
        ).get("route_mass_weighted_abs_scale_shift"),
        "router_expert_coupling_status": router_expert_coupling.get("status"),
        "router_expert_coupling_gate": router_expert_coupling.get("gate"),
        "router_expert_coupling_fragility_router_feature_corr": router_expert_coupling.get(
            "fragility_router_feature_corr"
        ),
        "router_expert_coupling_fragility_scale_shrink_corr": router_expert_coupling.get(
            "fragility_scale_shrink_corr"
        ),
        "router_expert_coupling_shrink_lift": router_expert_coupling.get(
            "high_vs_low_weighted_shrink_lift"
        ),
        "router_expert_coupling_top_layer": router_expert_coupling.get("top_coupled_layer_id"),
        "router_coupled_candidate_status": router_coupled_candidate.get("status"),
        "router_coupled_candidate_selection_gate": router_coupled_candidate.get("selection_gate"),
        "router_coupled_candidate_selected": router_coupled_candidate.get("selected_candidate_id"),
        "router_coupled_candidate_retention": router_coupled_candidate.get("selected_nonbase_mass_retention"),
        "router_coupled_candidate_retention_delta": router_coupled_candidate.get(
            "selected_retention_delta_vs_mechanistic"
        ),
        "router_coupled_candidate_delta_reduction": router_coupled_candidate.get(
            "selected_router_coupled_delta_reduction_vs_mechanistic"
        ),
        "router_coupled_frontier_status": router_coupled_frontier.get("status"),
        "router_coupled_frontier_gate": router_coupled_frontier.get("gate"),
        "router_coupled_frontier_action": router_coupled_frontier.get(
            "recommended_unified_action"
        ),
        "router_coupled_frontier_effect_fraction": router_coupled_frontier.get(
            "constrained_effect_fraction_vs_stress"
        ),
        "router_coupled_frontier_default_gate_candidates": router_coupled_frontier.get(
            "default_gate_candidate_count"
        ),
        "router_coupled_frontier_candidate_count": router_coupled_frontier.get(
            "candidate_count"
        ),
        "source_set_complementarity_status": source_set_complementarity.get("status"),
        "source_set_complementarity_current_gate": source_set_complementarity.get("current_gate"),
        "source_set_complementarity_current_dominant_source": source_set_complementarity.get(
            "current_dominant_source"
        ),
        "source_set_complementarity_frontier_avg_gain": source_set_complementarity.get(
            "current_frontier_avg_gain_vs_best_single"
        ),
        "source_set_complementarity_best_observed_gap": source_set_complementarity.get(
            "current_best_observed_avg_gap_to_frontier"
        ),
        "source_set_complementarity_complementary_count": source_set_complementarity.get(
            "complementary_source_set_count"
        ),
        "average_source_set_optimizer_status": average_source_set_optimizer.get("status"),
        "average_source_set_optimizer_top_source_set": (
            average_source_set_optimizer.get("top_source_set") or {}
        ).get("source_set"),
        "average_source_set_optimizer_top_gate": (
            average_source_set_optimizer.get("top_source_set") or {}
        ).get("optimizer_gate"),
        "average_source_set_optimizer_top_gain": (
            average_source_set_optimizer.get("top_source_set") or {}
        ).get("frontier_avg_gain_vs_best_single"),
        "average_source_set_optimizer_interference_budget": average_source_set_optimizer.get(
            "interference_budget"
        ),
        "average_source_set_optimizer_top_surplus": (
            average_source_set_optimizer.get("top_source_set") or {}
        ).get("frontier_avg_surplus_vs_interference"),
        "average_source_set_optimizer_final_budget_candidates": average_source_set_optimizer.get(
            "final_average_budget_candidate_count"
        ),
        "average_source_set_optimizer_probe_only": average_source_set_optimizer.get(
            "probe_only_source_set_count"
        ),
        "qwen_source_discovery_plan_status": qwen_source_discovery_plan.get("status"),
        "qwen_source_discovery_top_scenario": (
            qwen_source_discovery_plan.get("top_scenario") or {}
        ).get("scenario_id"),
        "qwen_source_discovery_top_action": (
            qwen_source_discovery_plan.get("top_scenario") or {}
        ).get("next_action"),
        "qwen_source_discovery_top_queue_item": (
            qwen_source_discovery_plan.get("top_queue_item") or {}
        ).get("queue_item"),
        "qwen_source_discovery_measured_additional_gain_needed": qwen_source_discovery_plan.get(
            "measured_additional_frontier_avg_gain_needed"
        ),
        "qwen_source_discovery_eval_plan_status": qwen_source_discovery_eval_plan.get("status"),
        "qwen_source_discovery_eval_plan_job_count": qwen_source_discovery_eval_plan.get(
            "eval_job_count"
        ),
        "qwen_source_discovery_eval_plan_top_job": (
            qwen_source_discovery_eval_plan.get("top_eval_job") or {}
        ).get("job_id"),
        "qwen_source_discovery_eval_plan_task_status": qwen_source_discovery_eval_plan.get(
            "task_name_compatibility_status"
        ),
        "qwen_source_discovery_eval_plan_tasks": ",".join(
            qwen_source_discovery_eval_plan.get("task_names") or []
        ),
        "qwen_source_discovery_served_preflight_status": qwen_source_discovery_served_model_preflight.get(
            "status"
        ),
        "qwen_source_discovery_served_preflight_endpoint": qwen_source_discovery_served_model_preflight.get(
            "endpoint_probe_status"
        ),
        "qwen_source_discovery_served_preflight_required": qwen_source_discovery_served_model_preflight.get(
            "unique_required_model_count"
        ),
        "qwen_source_discovery_served_preflight_missing": qwen_source_discovery_served_model_preflight.get(
            "missing_required_model_count"
        ),
        "qwen_source_discovery_served_preflight_ready_manifests": qwen_source_discovery_served_model_preflight.get(
            "ready_manifest_count"
        ),
        "qwen_source_discovery_served_preflight_manifests": qwen_source_discovery_served_model_preflight.get(
            "manifest_check_count"
        ),
        "qwen_source_discovery_served_preflight_blocker": qwen_source_discovery_served_model_preflight.get(
            "blocking_reason"
        ),
        "qwen_source_frontier_eval_feedback_status": qwen_source_frontier_eval_feedback.get(
            "status"
        ),
        "qwen_source_frontier_eval_feedback_scored": qwen_source_frontier_eval_feedback.get(
            "scored_job_count"
        ),
        "qwen_source_frontier_eval_feedback_jobs": qwen_source_frontier_eval_feedback.get(
            "job_count"
        ),
        "qwen_source_frontier_eval_feedback_final_candidates": qwen_source_frontier_eval_feedback.get(
            "final_average_budget_candidate_count"
        ),
        "qwen_source_frontier_eval_feedback_probe_only": qwen_source_frontier_eval_feedback.get(
            "probe_only_candidate_count"
        ),
        "qwen_source_frontier_eval_feedback_top_job": (
            qwen_source_frontier_eval_feedback.get("top_scored_job") or {}
        ).get("job_id"),
        "qwen_source_frontier_eval_feedback_top_gate": (
            qwen_source_frontier_eval_feedback.get("top_scored_job") or {}
        ).get("decision_gate"),
        "qwen_source_frontier_eval_feedback_top_surplus": (
            qwen_source_frontier_eval_feedback.get("top_scored_job") or {}
        ).get("surplus_vs_interference"),
        "qwen_source_frontier_eval_feedback_blocker": qwen_source_frontier_eval_feedback.get(
            "blocking_reason"
        ),
        "qwen_source_frontier_eval_feedback_smoke_status": qwen_source_frontier_eval_feedback_smoke.get(
            "status"
        ),
        "qwen_source_frontier_eval_feedback_smoke_passed": (
            qwen_source_frontier_eval_feedback_smoke.get("smoke_checks") or {}
        ).get("passed"),
        "qwen_source_frontier_eval_feedback_smoke_scored": qwen_source_frontier_eval_feedback_smoke.get(
            "scored_job_count"
        ),
        "qwen_source_frontier_eval_feedback_smoke_jobs": qwen_source_frontier_eval_feedback_smoke.get(
            "job_count"
        ),
        "qwen_source_frontier_eval_feedback_smoke_final_candidates": (
            qwen_source_frontier_eval_feedback_smoke.get("final_average_budget_candidate_count")
        ),
        "router_calibration_frontier_status": router_calibration_frontier.get("status"),
        "router_calibration_frontier_default_candidates": router_calibration_frontier.get(
            "default_candidate_count"
        ),
        "router_calibration_frontier_candidate_count": router_calibration_frontier.get(
            "candidate_count"
        ),
        "router_calibration_frontier_recommended": ",".join(
            router_calibration_frontier.get("recommended_default_candidates") or []
        ),
        "router_calibration_frontier_blocker": router_calibration_frontier.get(
            "acceptance_blocker"
        ),
        "router_calibration_frontier_nll_signal": router_calibration_frontier.get(
            "nll_worst_reduction_signal"
        ),
        "router_calibration_frontier_generation_signal": router_calibration_frontier.get(
            "generation_avg_gain_signal"
        ),
        "harc_router_stats_status": harc_router_stats.get("status"),
        "harc_router_stats_valid_routers": harc_router_stats.get("valid_router_count"),
        "harc_router_stats_routers": harc_router_stats.get("router_count"),
        "harc_router_stats_first_stage_covered": harc_router_stats.get(
            "first_stage_covered_layer_count"
        ),
        "harc_router_stats_first_stage_required": harc_router_stats.get(
            "first_stage_required_layer_count"
        ),
        "harc_router_stats_first_stage_status": harc_router_stats.get(
            "first_stage_coverage_status"
        ),
        "harc_router_stats_mean_hessian_trace": harc_router_stats.get("mean_hessian_trace"),
        "harc_router_stats_mean_cov_trace": harc_router_stats.get("mean_hidden_cov_trace"),
        "harc_router_stats_smoke_status": harc_router_stats_smoke.get("status"),
        "harc_router_stats_smoke_checks_passed": (
            harc_router_stats_smoke.get("smoke_checks") or {}
        ).get("passed"),
        "harc_router_stats_smoke_checks_total": (
            harc_router_stats_smoke.get("smoke_checks") or {}
        ).get("total"),
        "harc_readiness_gate_status": harc_readiness_gate.get("status"),
        "harc_readiness_gate_preconditions": harc_readiness_gate.get("precondition_count"),
        "harc_readiness_gate_passed_preconditions": harc_readiness_gate.get(
            "precondition_passed_count"
        ),
        "harc_readiness_gate_cache": harc_readiness_gate.get("hessian_covariance_cache_status"),
        "harc_readiness_gate_top_layer": harc_readiness_gate.get("top_harc_layer"),
        "harc_readiness_gate_top_score": harc_readiness_gate.get("top_harc_priority_score"),
        "harc_readiness_gate_first_stage_layers": harc_readiness_gate.get(
            "first_stage_layer_count"
        ),
        "harc_readiness_gate_action": harc_readiness_gate.get("recommended_action"),
        "harc_readiness_gate_smoke_status": harc_readiness_gate_smoke.get("status"),
        "harc_readiness_gate_smoke_passed": harc_readiness_gate_smoke.get("passed_case_count"),
        "harc_readiness_gate_smoke_cases": harc_readiness_gate_smoke.get("case_count"),
        "unified_optimizer_status": unified_optimizer.get("status"),
        "unified_optimizer_contract_status": unified_optimizer.get("contract_status"),
        "unified_optimizer_contract_passed": unified_optimizer.get("contract_passed_requirement_count"),
        "unified_optimizer_contract_requirements": unified_optimizer.get("contract_requirement_count"),
        "unified_optimizer_contract_blocking": unified_optimizer.get("contract_blocking_requirements", []),
        "unified_optimizer_top_experiment": optimizer_top.get("experiment"),
        "unified_optimizer_top_experiment_status": optimizer_top.get("status"),
        "unified_optimizer_final_confidence_tie_band": optimizer_moe.get("qwen3_final_confidence_tie_band"),
        "unified_optimizer_final_rank_mode": optimizer_moe.get("qwen3_final_selection_rank_mode"),
        "unified_optimizer_final_rank_band_size": optimizer_moe.get("qwen3_final_selection_rank_band_size"),
        "unified_optimizer_smoke_status": unified_optimizer_smoke.get("status"),
        "unified_optimizer_smoke_passed": unified_optimizer_smoke.get("passed_case_count"),
        "unified_optimizer_smoke_cases": unified_optimizer_smoke.get("case_count"),
        "average_method_gate_status": average_method_gate.get("status"),
        "average_method_gate_default_accepted": average_method_gate.get("accepted_by_default_count"),
        "average_method_gate_default_rejected": average_method_gate.get("default_rejected_count"),
        "average_method_gate_conditional": average_method_gate.get("conditional_count"),
        "average_method_gate_smoke_status": average_method_gate_smoke.get("status"),
        "average_method_gate_smoke_passed": average_method_gate_smoke.get("passed_assertion_count"),
        "average_method_gate_smoke_assertions": average_method_gate_smoke.get("assertion_count"),
        "average_trust_region_bounds_status": average_trust_region_bounds.get("status"),
        "average_trust_region_bounds_constraints": average_trust_region_bounds.get("constraint_count"),
        "average_trust_region_bounds_passed": average_trust_region_bounds.get("passed_count"),
        "average_trust_region_bounds_rejected": average_trust_region_bounds.get("rejected_count"),
        "average_trust_region_bounds_waiting": average_trust_region_bounds.get("waiting_count"),
        "dense_local_task_vector_lambda_bound": average_trust_region_bounds.get(
            "dense_local_task_vector_lambda_bound"
        ),
        "moe_router_safe_lambda_proxy": average_trust_region_bounds.get("moe_router_safe_lambda_proxy"),
        "moe_direct_router_average_over_safe_bound": average_trust_region_bounds.get(
            "moe_direct_router_average_over_safe_bound"
        ),
        "average_trust_region_bounds_smoke_status": average_trust_region_bounds_smoke.get("status"),
        "average_trust_region_bounds_smoke_passed": average_trust_region_bounds_smoke.get(
            "passed_assertion_count"
        ),
        "average_trust_region_bounds_smoke_assertions": average_trust_region_bounds_smoke.get(
            "assertion_count"
        ),
        "mechanism_levers_status": mechanism_levers.get("status"),
        "mechanism_levers_top_lever": mechanism_levers.get("top_lever"),
        "mechanism_levers_top_next_test": mechanism_levers.get("top_lever_next_test"),
    }


def build_report(summary: dict[str, Any]) -> str:
    downstream = summary.get("downstream") or {}
    lines = [
        "# Qwen3 MoE Post-Eval Refresh",
        "",
        "这个脚本在远端 vLLM eval 落盘后按固定顺序刷新 eval bundle audit、unified/final selector、mechanism attribution、feedback/mechanistic optimizer、mechanistic sensitivity、router-expert coupling、router-coupled candidate、router-coupled retention frontier、unified average optimizer、average method gate matrix、average trust-region bounds 和总汇总，避免手工漏跑或用到旧结果。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Plan only: `{summary['plan_only']}`",
        f"- Steps passed: `{summary['passed_step_count']}/{summary['step_count']}`",
        f"- Audit: `{downstream.get('audit_status', 'n/a')}` (`{downstream.get('audit_usable_for_selection', 'n/a')}/{downstream.get('audit_method_count', 'n/a')}` usable)",
        f"- Selection: `{downstream.get('selection_status', 'n/a')}` -> `{downstream.get('selected_method', 'n/a')}`",
        f"- Final selection: `{downstream.get('final_selection_status', 'n/a')}` -> `{downstream.get('final_selected_method', 'n/a')}` (`{downstream.get('final_eligible_candidate_count', 'n/a')}/{downstream.get('final_candidate_count', 'n/a')}` eligible)",
        f"- Candidate trust-region gate: `{downstream.get('candidate_trust_region_gate_status', 'n/a')}` (`{downstream.get('candidate_trust_region_final_selectable', 'n/a')}/{downstream.get('candidate_trust_region_candidates', 'n/a')}` final-selectable, `{downstream.get('candidate_trust_region_ablation_only', 'n/a')}` ablation-only)",
        f"- Eval budget queue: `{downstream.get('eval_budget_status', 'n/a')}` (default `{downstream.get('eval_budget_default_runner_request', 'n/a')}`, final `{downstream.get('eval_budget_final_core_method_count', 'n/a')}` methods / `{downstream.get('eval_budget_final_core_prompt_budget', 'n/a')}` prompts, max examples `{downstream.get('eval_budget_recommended_max_examples', 'n/a')}`)",
        f"- Eval budget queue smoke: `{downstream.get('eval_budget_smoke_status', 'n/a')}` (`{downstream.get('eval_budget_smoke_passed', 'n/a')}/{downstream.get('eval_budget_smoke_assertions', 'n/a')}` assertions)",
        f"- Attribution: `{downstream.get('attribution_status', 'n/a')}` (`{downstream.get('attribution_scored_transition_count', 'n/a')}/{downstream.get('attribution_transition_count', 'n/a')}` scored)",
        f"- Feedback optimizer: `{downstream.get('feedback_status', 'n/a')}` (`{downstream.get('feedback_scored_task_count', 'n/a')}/{downstream.get('feedback_task_count', 'n/a')}` scored, `{downstream.get('feedback_changed_group_count', 'n/a')}` changed groups)",
        f"- Mechanistic unified: `{downstream.get('mechanistic_status', 'n/a')}` -> `{downstream.get('mechanistic_selected_candidate', 'n/a')}` (`retention={downstream.get('mechanistic_retention', 'n/a')}`, `violations={downstream.get('mechanistic_hard_cap_violations', 'n/a')}`)",
        f"- Mechanistic evidence: `{downstream.get('mechanistic_evidence_status', 'n/a')}` (`gradient_agreement={downstream.get('mechanistic_evidence_gradient_agreement', 'n/a')}`, `objective_improved={downstream.get('mechanistic_evidence_objective_improved_fraction', 'n/a')}`)",
        f"- Mechanistic sensitivity: `{downstream.get('mechanistic_sensitivity_status', 'n/a')}` (objective `{downstream.get('mechanistic_sensitivity_strongest_objective_ablation', 'n/a')}` delta `{downstream.get('mechanistic_sensitivity_strongest_objective_delta', 'n/a')}`, scale `{downstream.get('mechanistic_sensitivity_strongest_scale_ablation', 'n/a')}` shift `{downstream.get('mechanistic_sensitivity_scale_shift', 'n/a')}`)",
        f"- Router-expert coupling: `{downstream.get('router_expert_coupling_gate', 'n/a')}` (fragility->feature `{downstream.get('router_expert_coupling_fragility_router_feature_corr', 'n/a')}`, fragility->shrink `{downstream.get('router_expert_coupling_fragility_scale_shrink_corr', 'n/a')}`, shrink lift `{downstream.get('router_expert_coupling_shrink_lift', 'n/a')}`, top layer `L{downstream.get('router_expert_coupling_top_layer', 'n/a')}`)",
        f"- Router-coupled candidate: `{downstream.get('router_coupled_candidate_selection_gate', 'n/a')}` -> `{downstream.get('router_coupled_candidate_selected', 'n/a')}` (`retention={downstream.get('router_coupled_candidate_retention', 'n/a')}`, `retention_delta={downstream.get('router_coupled_candidate_retention_delta', 'n/a')}`, `coupled_delta_reduction={downstream.get('router_coupled_candidate_delta_reduction', 'n/a')}`)",
        f"- Router-coupled retention frontier: `{downstream.get('router_coupled_frontier_gate', 'n/a')}` (`effect_fraction={downstream.get('router_coupled_frontier_effect_fraction', 'n/a')}`, candidates `{downstream.get('router_coupled_frontier_default_gate_candidates', 'n/a')}/{downstream.get('router_coupled_frontier_candidate_count', 'n/a')}` pass default gate)",
        f"- Source-set complementarity: `{downstream.get('source_set_complementarity_current_gate', 'n/a')}` (dominant `{downstream.get('source_set_complementarity_current_dominant_source', 'n/a')}`, frontier avg gain `{downstream.get('source_set_complementarity_frontier_avg_gain', 'n/a')}`, best observed gap `{downstream.get('source_set_complementarity_best_observed_gap', 'n/a')}`, complementary sets `{downstream.get('source_set_complementarity_complementary_count', 'n/a')}`)",
        f"- Average source-set optimizer: `{downstream.get('average_source_set_optimizer_top_gate', 'n/a')}` for `{downstream.get('average_source_set_optimizer_top_source_set', 'n/a')}` (gain `{downstream.get('average_source_set_optimizer_top_gain', 'n/a')}` vs interference budget `{downstream.get('average_source_set_optimizer_interference_budget', 'n/a')}`, surplus `{downstream.get('average_source_set_optimizer_top_surplus', 'n/a')}`, final-budget `{downstream.get('average_source_set_optimizer_final_budget_candidates', 'n/a')}`, probe-only `{downstream.get('average_source_set_optimizer_probe_only', 'n/a')}`)",
        f"- Qwen source discovery plan: `{downstream.get('qwen_source_discovery_plan_status', 'n/a')}` (top scenario `{downstream.get('qwen_source_discovery_top_scenario', 'n/a')}`, queue `{downstream.get('qwen_source_discovery_top_queue_item', 'n/a')}`, additional gain needed `{downstream.get('qwen_source_discovery_measured_additional_gain_needed', 'n/a')}`)",
        f"- Qwen source discovery eval plan: `{downstream.get('qwen_source_discovery_eval_plan_status', 'n/a')}` (`{downstream.get('qwen_source_discovery_eval_plan_job_count', 'n/a')}` jobs, top `{downstream.get('qwen_source_discovery_eval_plan_top_job', 'n/a')}`, tasks `{downstream.get('qwen_source_discovery_eval_plan_tasks', 'n/a')}`, task names `{downstream.get('qwen_source_discovery_eval_plan_task_status', 'n/a')}`)",
        f"- Qwen source discovery served-model preflight: `{downstream.get('qwen_source_discovery_served_preflight_status', 'n/a')}` (endpoint `{downstream.get('qwen_source_discovery_served_preflight_endpoint', 'n/a')}`, required `{downstream.get('qwen_source_discovery_served_preflight_required', 'n/a')}`, missing `{downstream.get('qwen_source_discovery_served_preflight_missing', 'n/a')}`, manifests `{downstream.get('qwen_source_discovery_served_preflight_ready_manifests', 'n/a')}/{downstream.get('qwen_source_discovery_served_preflight_manifests', 'n/a')}`, blocker `{downstream.get('qwen_source_discovery_served_preflight_blocker', 'n/a')}`)",
        f"- Qwen source frontier eval feedback: `{downstream.get('qwen_source_frontier_eval_feedback_status', 'n/a')}` (scored `{downstream.get('qwen_source_frontier_eval_feedback_scored', 'n/a')}/{downstream.get('qwen_source_frontier_eval_feedback_jobs', 'n/a')}`, final candidates `{downstream.get('qwen_source_frontier_eval_feedback_final_candidates', 'n/a')}`, probe-only `{downstream.get('qwen_source_frontier_eval_feedback_probe_only', 'n/a')}`, top `{downstream.get('qwen_source_frontier_eval_feedback_top_job', 'n/a')}` / `{downstream.get('qwen_source_frontier_eval_feedback_top_gate', 'n/a')}`, surplus `{downstream.get('qwen_source_frontier_eval_feedback_top_surplus', 'n/a')}`, blocker `{downstream.get('qwen_source_frontier_eval_feedback_blocker', 'n/a')}`)",
        f"- Qwen source frontier eval feedback smoke: `{downstream.get('qwen_source_frontier_eval_feedback_smoke_status', 'n/a')}` (passed `{downstream.get('qwen_source_frontier_eval_feedback_smoke_passed', 'n/a')}`, scored `{downstream.get('qwen_source_frontier_eval_feedback_smoke_scored', 'n/a')}/{downstream.get('qwen_source_frontier_eval_feedback_smoke_jobs', 'n/a')}`, final candidates `{downstream.get('qwen_source_frontier_eval_feedback_smoke_final_candidates', 'n/a')}`)",
        f"- Router calibration frontier: `{downstream.get('router_calibration_frontier_status', 'n/a')}` (`{downstream.get('router_calibration_frontier_default_candidates', 'n/a')}/{downstream.get('router_calibration_frontier_candidate_count', 'n/a')}` default, recommended `{downstream.get('router_calibration_frontier_recommended', 'n/a')}`, blocker `{downstream.get('router_calibration_frontier_blocker', 'n/a')}`, nll `{downstream.get('router_calibration_frontier_nll_signal', 'n/a')}`, generation `{downstream.get('router_calibration_frontier_generation_signal', 'n/a')}`)",
        f"- HARC router stats: `{downstream.get('harc_router_stats_status', 'n/a')}` (`{downstream.get('harc_router_stats_valid_routers', 'n/a')}/{downstream.get('harc_router_stats_routers', 'n/a')}` routers, first-stage `{downstream.get('harc_router_stats_first_stage_covered', 'n/a')}/{downstream.get('harc_router_stats_first_stage_required', 'n/a')}` `{downstream.get('harc_router_stats_first_stage_status', 'n/a')}`, Hessian `{downstream.get('harc_router_stats_mean_hessian_trace', 'n/a')}`, cov `{downstream.get('harc_router_stats_mean_cov_trace', 'n/a')}`)",
        f"- HARC router stats smoke: `{downstream.get('harc_router_stats_smoke_status', 'n/a')}` (`{downstream.get('harc_router_stats_smoke_checks_passed', 'n/a')}/{downstream.get('harc_router_stats_smoke_checks_total', 'n/a')}` checks)",
        f"- HARC readiness gate: `{downstream.get('harc_readiness_gate_status', 'n/a')}` (`{downstream.get('harc_readiness_gate_passed_preconditions', 'n/a')}/{downstream.get('harc_readiness_gate_preconditions', 'n/a')}` preconditions, cache `{downstream.get('harc_readiness_gate_cache', 'n/a')}`, top layer `L{downstream.get('harc_readiness_gate_top_layer', 'n/a')}` score `{downstream.get('harc_readiness_gate_top_score', 'n/a')}`, first-stage layers `{downstream.get('harc_readiness_gate_first_stage_layers', 'n/a')}`, action `{downstream.get('harc_readiness_gate_action', 'n/a')}`)",
        f"- HARC readiness smoke: `{downstream.get('harc_readiness_gate_smoke_status', 'n/a')}` (`{downstream.get('harc_readiness_gate_smoke_passed', 'n/a')}/{downstream.get('harc_readiness_gate_smoke_cases', 'n/a')}` cases)",
        f"- Unified average optimizer: `{downstream.get('unified_optimizer_status', 'n/a')}` (top next experiment `{downstream.get('unified_optimizer_top_experiment', 'n/a')}` / `{downstream.get('unified_optimizer_top_experiment_status', 'n/a')}`)",
        f"- Unified algorithm contract: `{downstream.get('unified_optimizer_contract_status', 'n/a')}` (`{downstream.get('unified_optimizer_contract_passed', 'n/a')}/{downstream.get('unified_optimizer_contract_requirements', 'n/a')}` passed, blocking `{downstream.get('unified_optimizer_contract_blocking', [])}`)",
        f"- Unified selector rank gate in optimizer: confidence band `{downstream.get('unified_optimizer_final_confidence_tie_band', 'n/a')}`, rank mode `{downstream.get('unified_optimizer_final_rank_mode', 'n/a')}`, band size `{downstream.get('unified_optimizer_final_rank_band_size', 'n/a')}`",
        f"- Unified optimizer ledger smoke: `{downstream.get('unified_optimizer_smoke_status', 'n/a')}` (`{downstream.get('unified_optimizer_smoke_passed', 'n/a')}/{downstream.get('unified_optimizer_smoke_cases', 'n/a')}` cases)",
        f"- Average method gate matrix: `{downstream.get('average_method_gate_status', 'n/a')}` (`accepted_by_default={downstream.get('average_method_gate_default_accepted', 'n/a')}`, `rejected={downstream.get('average_method_gate_default_rejected', 'n/a')}`, `conditional={downstream.get('average_method_gate_conditional', 'n/a')}`)",
        f"- Average method gate smoke: `{downstream.get('average_method_gate_smoke_status', 'n/a')}` (`{downstream.get('average_method_gate_smoke_passed', 'n/a')}/{downstream.get('average_method_gate_smoke_assertions', 'n/a')}` assertions)",
        f"- Average trust-region bounds: `{downstream.get('average_trust_region_bounds_status', 'n/a')}` (`passed={downstream.get('average_trust_region_bounds_passed', 'n/a')}`, `rejected={downstream.get('average_trust_region_bounds_rejected', 'n/a')}`, `waiting={downstream.get('average_trust_region_bounds_waiting', 'n/a')}`); dense lambda bound `{downstream.get('dense_local_task_vector_lambda_bound', 'n/a')}`, router midpoint over safe bound `{downstream.get('moe_direct_router_average_over_safe_bound', 'n/a')}`",
        f"- Average trust-region smoke: `{downstream.get('average_trust_region_bounds_smoke_status', 'n/a')}` (`{downstream.get('average_trust_region_bounds_smoke_passed', 'n/a')}/{downstream.get('average_trust_region_bounds_smoke_assertions', 'n/a')}` assertions)",
        f"- Mechanism levers: `{downstream.get('mechanism_levers_status', 'n/a')}` (top `{downstream.get('mechanism_levers_top_lever', 'n/a')}` -> `{downstream.get('mechanism_levers_top_next_test', 'n/a')}`)",
        "",
        "| step | kind | status | returncode | seconds |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in summary["steps"]:
        lines.append(
            f"| `{row['step']}` | `{row['kind']}` | `{row['status']}` | "
            f"{row['returncode']} | {float(row['duration_sec']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
        ]
    )
    for row in summary["steps"]:
        lines.append(f"- `{row['command']}`")
    return "\n".join(lines) + "\n"


def write_outputs(output_dir: Path, step_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = [row for row in step_rows if row["status"] == "failed"]
    passed = [row for row in step_rows if row["status"] == "passed"]
    planned = [row for row in step_rows if row["status"] == "planned"]
    status = "planned" if args.plan_only else ("failed" if failed else "passed")
    summary = {
        "schema_version": 1,
        "status": status,
        "plan_only": bool(args.plan_only),
        "step_count": int(len(step_rows)),
        "passed_step_count": int(len(passed)),
        "planned_step_count": int(len(planned)),
        "failed_step_count": int(len(failed)),
        "downstream": downstream_status(args) if not args.plan_only else {},
        "steps": step_rows,
        "outputs": {
            "steps": rel(output_dir / "steps.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    pd.DataFrame(step_rows).to_csv(output_dir / "steps.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Qwen3 MoE post-vLLM eval gates in the correct order.")
    parser.add_argument("--gate-dir", type=Path, default=Path("results/qwen3_moe_mechanism_eval_gate"))
    parser.add_argument("--audit-dir", type=Path, default=Path("results/qwen3_moe_eval_bundle_audit"))
    parser.add_argument("--selection-dir", type=Path, default=Path("results/qwen3_moe_unified_result_selection"))
    parser.add_argument("--final-selection-dir", type=Path, default=Path("results/qwen3_moe_final_candidate_selection"))
    parser.add_argument(
        "--candidate-trust-region-gate-dir",
        type=Path,
        default=Path("results/qwen3_moe_candidate_trust_region_gate"),
    )
    parser.add_argument("--eval-budget-dir", type=Path, default=Path("results/qwen3_moe_eval_budget_plan"))
    parser.add_argument("--attribution-dir", type=Path, default=Path("results/qwen3_moe_mechanism_effect_attribution"))
    parser.add_argument("--feedback-dir", type=Path, default=Path("results/qwen3_moe_feedback_optimizer"))
    parser.add_argument(
        "--mechanistic-dir",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_unified_candidate"),
    )
    parser.add_argument(
        "--mechanistic-evidence-dir",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_evidence_audit"),
    )
    parser.add_argument(
        "--mechanistic-sensitivity-dir",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_sensitivity"),
    )
    parser.add_argument(
        "--router-expert-coupling-dir",
        type=Path,
        default=Path("results/qwen3_moe_router_expert_coupling"),
    )
    parser.add_argument(
        "--router-coupled-candidate-dir",
        type=Path,
        default=Path("results/qwen3_moe_router_coupled_candidate"),
    )
    parser.add_argument(
        "--router-coupled-retention-frontier-dir",
        type=Path,
        default=Path("results/qwen3_moe_router_coupled_retention_frontier"),
    )
    parser.add_argument(
        "--source-set-complementarity-dir",
        type=Path,
        default=Path("results/qwen3_source_set_complementarity_gate"),
    )
    parser.add_argument(
        "--average-source-set-optimizer-dir",
        type=Path,
        default=Path("results/qwen3_average_source_set_optimizer"),
    )
    parser.add_argument(
        "--qwen-source-discovery-plan-dir",
        type=Path,
        default=Path("results/qwen_source_discovery_plan"),
    )
    parser.add_argument(
        "--qwen-source-discovery-eval-plan-dir",
        type=Path,
        default=Path("results/qwen_source_discovery_eval_plan"),
    )
    parser.add_argument(
        "--qwen-source-discovery-served-model-preflight-dir",
        type=Path,
        default=Path("results/qwen_source_discovery_served_model_preflight"),
    )
    parser.add_argument(
        "--qwen-source-frontier-eval-feedback-dir",
        type=Path,
        default=Path("results/qwen_source_frontier_eval_feedback"),
    )
    parser.add_argument(
        "--qwen-source-frontier-eval-feedback-smoke-dir",
        type=Path,
        default=Path("results/qwen_source_frontier_eval_feedback_smoke"),
    )
    parser.add_argument(
        "--router-calibration-frontier-dir",
        type=Path,
        default=Path("results/qwen3_moe_router_calibration_frontier"),
    )
    parser.add_argument(
        "--harc-router-stats-dir",
        type=Path,
        default=Path("results/qwen3_moe_harc_router_stats"),
    )
    parser.add_argument(
        "--harc-readiness-gate-dir",
        type=Path,
        default=Path("results/qwen3_moe_harc_readiness_gate"),
    )
    parser.add_argument(
        "--unified-optimizer-dir",
        type=Path,
        default=Path("results/unified_average_optimizer"),
    )
    parser.add_argument(
        "--average-method-gate-dir",
        type=Path,
        default=Path("results/average_method_gate_matrix"),
    )
    parser.add_argument(
        "--average-trust-region-bounds-dir",
        type=Path,
        default=Path("results/average_trust_region_bounds"),
    )
    parser.add_argument("--mechanism-levers-dir", type=Path, default=Path("results/qwen3_moe_mechanism_levers"))
    parser.add_argument("--audit-smoke-dir", type=Path, default=Path("results/qwen3_moe_eval_bundle_audit_smoke"))
    parser.add_argument("--selection-smoke-dir", type=Path, default=Path("results/qwen3_moe_unified_result_selection_smoke"))
    parser.add_argument(
        "--final-selection-smoke-dir",
        type=Path,
        default=Path("results/qwen3_moe_final_candidate_selection_smoke"),
    )
    parser.add_argument(
        "--attribution-smoke-dir",
        type=Path,
        default=Path("results/qwen3_moe_mechanism_effect_attribution_smoke"),
    )
    parser.add_argument(
        "--feedback-smoke-dir",
        type=Path,
        default=Path("results/qwen3_moe_feedback_optimizer_smoke"),
    )
    parser.add_argument(
        "--mechanistic-smoke-dir",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_unified_candidate_smoke"),
    )
    parser.add_argument(
        "--unified-optimizer-smoke-dir",
        type=Path,
        default=Path("results/unified_average_optimizer_ledger_smoke"),
    )
    parser.add_argument(
        "--average-method-gate-smoke-dir",
        type=Path,
        default=Path("results/average_method_gate_matrix_consistency_smoke"),
    )
    parser.add_argument(
        "--eval-budget-smoke-dir",
        type=Path,
        default=Path("results/qwen3_moe_eval_budget_queue_smoke"),
    )
    parser.add_argument(
        "--average-trust-region-bounds-smoke-dir",
        type=Path,
        default=Path("results/average_trust_region_bounds_smoke"),
    )
    parser.add_argument(
        "--harc-readiness-gate-smoke-dir",
        type=Path,
        default=Path("results/qwen3_moe_harc_readiness_gate_smoke"),
    )
    parser.add_argument(
        "--harc-router-stats-smoke-dir",
        type=Path,
        default=Path("results/qwen3_moe_harc_router_stats_smoke"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_post_eval_refresh"))
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = build_steps(args)
    rows = []
    for step in steps:
        result = run_step(step, plan_only=args.plan_only)
        rows.append(result)
        if result["status"] == "failed":
            break
    summary = write_outputs(args.output_dir, rows, args)
    print(f"Wrote Qwen3 MoE post-eval refresh to {repo_path(args.output_dir).resolve()}")
    print(f"Status: {summary['status']}; steps={summary['passed_step_count']}/{summary['step_count']}")
    if summary["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
