#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def fmt(value: Any, digits: int = 4) -> str:
    value = fnum(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def clean_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def clean_row(row: pd.Series) -> dict[str, Any]:
    return {str(key): clean_value(value) for key, value in row.items()}


def feature_by_name(features: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if features.empty:
        return {}
    return {str(row["probe"]): clean_row(row) for _, row in features.iterrows()}


def source_links(source_matrix: pd.DataFrame, names: str) -> list[dict[str, str]]:
    if source_matrix.empty or not names:
        return []
    requested = {name.strip() for name in str(names).split(";") if name.strip()}
    rows = source_matrix[source_matrix["short_name"].isin(requested)]
    return [
        {
            "short_name": str(row["short_name"]),
            "year": str(row["year"]),
            "url": str(row["url"]),
        }
        for _, row in rows.iterrows()
    ]


def build_gate_rows(
    method_matrix: pd.DataFrame,
    source_matrix: pd.DataFrame,
    optimizer_summary: dict[str, Any],
    optimizer_features: pd.DataFrame,
) -> list[dict[str, Any]]:
    dense = optimizer_summary.get("dense", {})
    moe = optimizer_summary.get("moe", {})
    features = feature_by_name(optimizer_features)
    dense_lambda = features.get("dense_lambda_connectivity", {})
    moe_line = features.get("qwen3_straight_line_connectivity", {})
    moe_base_coder = features.get("qwen3_base_to_coder_connectivity", {})
    moe_complementary = features.get("qwen3_complementary_pair_connectivity", {})
    router_gate = features.get("qwen3_router_move_gate", {})
    router_coupled = features.get("qwen3_router_coupled_retention_frontier", {})
    router_calibration = features.get("qwen3_router_calibration_nll_probe", {})
    final_gate = features.get("qwen3_final_candidate_selection", {})
    olmoe_gauge = features.get("real_olmoe_gauge_selfmerge", {})
    qwen_identity = features.get("qwen3_expert_identity", {})

    statuses: dict[str, dict[str, str]] = {
        "Uniform / linear average": {
            "current_gate": "rejected_as_default",
            "dense_status": "reject_midpoint_for_current_qwen_dense_pair",
            "moe_status": "reject_source_to_source_midpoint_for_current_qwen3_moe_pair",
            "why": (
                f"Dense midpoint worst NLL {fmt(dense.get('lambda_linear_worst_nll'))} is far above the "
                f"best lambda-family value {fmt(dense.get('lambda_best_worst_nll'))}; Qwen3 MoE best interior "
                f"worst NLL is above the best endpoint by {fmt(moe.get('qwen3_interpolation_interior_gap_nll'))}."
            ),
            "allowed_action": "Use only as a negative baseline; do not materialize as the selected average.",
            "blocking_probe": "connectivity_path_must_beat_endpoint_frontier",
        },
        "Task arithmetic / coefficient search": {
            "current_gate": "conditional_allowed",
            "dense_status": "allowed_only_with_heldout_lambda_selection",
            "moe_status": "allowed_only_as_layer_expert_coefficient_rules",
            "why": (
                f"Current Dense coefficient family selects lambda config {dense.get('lambda_best_config')}; "
                "Qwen3 MoE uses route/evidence/geometry coefficients rather than a global source delta."
            ),
            "allowed_action": "Search coefficients under endpoint fallback and same-shape writer constraints.",
            "blocking_probe": "heldout_loss_or_vllm_gate_must_accept_coefficients",
        },
        "Sign / sparsity conflict methods": {
            "current_gate": "conditional_diagnostic",
            "dense_status": "diagnostic_until_it_beats_anchor",
            "moe_status": "expert_ffn_only_after_alignment_never_raw_router",
            "why": (
                f"Dense unified worst NLL {fmt(dense.get('unified_worst_nll'))} is used as the held-out gate; "
                "sparse rules are not accepted just because sign conflict exists."
            ),
            "allowed_action": "Emit tensor rules only for modules that pass held-out and vLLM gates.",
            "blocking_probe": "sparse_rule_must_improve_worst_task_and_preserve_critical_tensors",
        },
        "Importance / activation-aware average": {
            "current_gate": "conditional_needs_calibration_match",
            "dense_status": "fisher_is_not_sufficient_on_current_qwen_dense_probe",
            "moe_status": "needs_route_conditioned_sensitivity_before_acceptance",
            "why": (
                f"Dense curvature ratios general/code are {fmt(dense.get('curvature_ratio_general'))}/"
                f"{fmt(dense.get('curvature_ratio_code'))}, so local quadratic/Fisher evidence is too weak alone."
            ),
            "allowed_action": "Use Fisher/RegMean-style weights only after activation or NLL sensitivity predicts held-out behavior.",
            "blocking_probe": "activation_covariance_or_fisher_prediction_must_match_actual_path_loss",
        },
        "Output-space calibrated average": {
            "current_gate": "active_moe_lever_pending_downstream_gate",
            "dense_status": "available_when_output_residual_is_measured",
            "moe_status": "router_route_kd_active_but_not_accepted_yet",
            "why": (
                f"Router-only calibration reduces linear MoE worst NLL by {fmt(moe.get('qwen3_router_calibration_nll_worst_reduction'))}, "
                f"but final router-calibration selector status is {moe.get('qwen3_router_calibration_status')}."
            ),
            "allowed_action": "Train or fit output/router residual corrections, then require source/baseline vLLM dominance gates.",
            "blocking_probe": "matched_baseline_source_candidate_vllm_eval_required",
        },
        "Alignment before averaging": {
            "current_gate": "required_precondition",
            "dense_status": "required_when_barrier_or_permutation_symmetry_is_detected",
            "moe_status": "required_for_expert_gauge_qwen3_identity_currently_passes",
            "why": (
                f"Real OLMoE same-name degradation is {fmt(moe.get('real_gauge_naive_degradation'))} while aligned degradation is "
                f"{fmt(moe.get('real_gauge_aligned_degradation'))}; Qwen3 expert identity fraction is "
                f"{fmt(moe.get('qwen3_identity_fraction'))}."
            ),
            "allowed_action": "Canonicalize or remap features/experts before any average; Qwen3 can use identity expert mapping for this pair.",
            "blocking_probe": "expert_or_feature_matching_must_be_known_before_tensor_average",
        },
        "Router-aware MoE average": {
            "current_gate": "required_for_moe",
            "dense_status": "not_applicable",
            "moe_status": "freeze_router_or_train_audited_route_kd_delta",
            "why": (
                f"Direct router movement allows {fmt(router_gate.get('value'), 0)}/{fmt(router_gate.get('threshold'), 0)} layers; "
                f"direct router-boundary shrink has retention-safe effect fraction {fmt(router_coupled.get('value'), 4)} "
                f"against threshold {fmt(router_coupled.get('threshold'), 4)}; "
                f"router calibration NLL reduction is {fmt(router_calibration.get('value'))}, but candidate acceptance is still "
                f"{moe.get('qwen3_router_calibration_status')}."
            ),
            "allowed_action": "Freeze routers for expert candidate generation; treat router delta as separately audited ablation.",
            "blocking_probe": "hard_route_load_capacity_and_matched_vllm_eval_must_pass",
        },
    }

    rows = []
    for _, method in method_matrix.iterrows():
        family = str(method["method_family"])
        status = statuses.get(
            family,
            {
                "current_gate": "not_evaluated",
                "dense_status": "not_evaluated",
                "moe_status": "not_evaluated",
                "why": "No current rule is registered for this method family.",
                "allowed_action": str(method.get("recommended_action", "")),
                "blocking_probe": str(method.get("primary_probe", "")),
            },
        )
        rows.append(
            {
                "method_family": family,
                "current_gate": status["current_gate"],
                "dense_status": status["dense_status"],
                "moe_status": status["moe_status"],
                "allowed_action": status["allowed_action"],
                "blocking_probe": status["blocking_probe"],
                "why": status["why"],
                "literature_sources": str(method.get("sources", "")),
                "source_links": json.dumps(source_links(source_matrix, str(method.get("sources", ""))), sort_keys=True),
                "primary_probe": str(method.get("primary_probe", "")),
                "failure_signal": str(method.get("failure_signal", "")),
                "same_shape_constraint": "output checkpoint keeps target config/tokenizer/model class/tensor names/tensor shapes",
            }
        )
    return rows


def build_probe_rows(optimizer_summary: dict[str, Any], optimizer_features: pd.DataFrame) -> list[dict[str, Any]]:
    dense = optimizer_summary.get("dense", {})
    moe = optimizer_summary.get("moe", {})
    features = feature_by_name(optimizer_features)
    return [
        {
            "gate": "dense_midpoint_rejection",
            "probe": "dense_lambda_connectivity",
            "value": dense.get("lambda_linear_worst_nll"),
            "threshold": dense.get("lambda_best_worst_nll"),
            "required_before_accepting": "uniform average or fixed lambda midpoint",
            "status": "failed_for_current_dense_pair",
            "evidence": (features.get("dense_lambda_connectivity") or {}).get("evidence"),
        },
        {
            "gate": "moe_source_to_source_interpolation",
            "probe": "qwen3_straight_line_connectivity",
            "value": moe.get("qwen3_interpolation_interior_gap_nll"),
            "threshold": 0.0,
            "required_before_accepting": "Qwen3 MoE source-to-source midpoint",
            "status": "failed_for_current_qwen3_pair",
            "evidence": (features.get("qwen3_straight_line_connectivity") or {}).get("evidence"),
        },
        {
            "gate": "moe_complementarity_claim",
            "probe": "qwen3_complementary_pair_connectivity",
            "value": moe.get("qwen3_complementary_best_merge_avg_nll"),
            "threshold": moe.get("qwen3_complementary_best_source_avg_nll"),
            "required_before_accepting": "claim that specialist endpoints combine under averaging",
            "status": "not_supported_by_current_probe",
            "evidence": (features.get("qwen3_complementary_pair_connectivity") or {}).get("evidence"),
        },
        {
            "gate": "expert_gauge_alignment",
            "probe": "real_olmoe_gauge_selfmerge",
            "value": moe.get("real_gauge_naive_degradation"),
            "threshold": 0.0,
            "required_before_accepting": "same-name expert average",
            "status": "alignment_required",
            "evidence": (features.get("real_olmoe_gauge_selfmerge") or {}).get("evidence"),
        },
        {
            "gate": "router_movement",
            "probe": "qwen3_router_move_gate",
            "value": 0,
            "threshold": 48,
            "required_before_accepting": "direct router weight average",
            "status": "failed_freeze_or_calibrate",
            "evidence": (features.get("qwen3_router_move_gate") or {}).get("evidence"),
        },
        {
            "gate": "router_coupled_direct_shrink",
            "probe": "qwen3_router_coupled_retention_frontier",
            "value": moe.get("qwen3_router_coupled_frontier_effect_fraction"),
            "threshold": moe.get("qwen3_router_coupled_frontier_minimum_effect_fraction"),
            "required_before_accepting": "direct router-boundary extra shrink as a default expert-scale term",
            "status": moe.get("qwen3_router_coupled_frontier_gate"),
            "evidence": (features.get("qwen3_router_coupled_retention_frontier") or {}).get("evidence"),
        },
        {
            "gate": "router_calibration",
            "probe": "qwen3_router_calibration_nll_probe",
            "value": moe.get("qwen3_router_calibration_nll_worst_reduction"),
            "threshold": 0.0,
            "required_before_accepting": "router-only delta after expert candidate",
            "status": moe.get("qwen3_router_calibration_status"),
            "evidence": (features.get("qwen3_router_calibration_nll_probe") or {}).get("evidence"),
        },
        {
            "gate": "final_downstream_acceptance",
            "probe": "qwen3_final_candidate_selection",
            "value": moe.get("qwen3_eligible_candidates"),
            "threshold": moe.get("qwen3_candidate_count"),
            "required_before_accepting": "any Qwen3 same-shape average as final answer",
            "status": moe.get("qwen3_final_selection_status"),
            "evidence": (features.get("qwen3_final_candidate_selection") or {}).get("evidence"),
        },
    ]


def build_report(summary: dict[str, Any], method_table: pd.DataFrame, probe_table: pd.DataFrame) -> str:
    lines = [
        "# Average Method Gate Matrix",
        "",
        "这个结果把常见 Dense/MoE averaging 方法从“方法名列表”改成当前证据下的执行门槛。每个方法族都必须说明：在 Dense 里是否允许、在 MoE 里是否允许、阻塞 probe 是什么，以及同构 checkpoint 约束下能做什么。",
        "",
        "## Current Takeaway",
        "",
        f"- Accepted-by-default method families: `{summary['accepted_by_default_count']}`",
        f"- Conditional method families: `{summary['conditional_count']}`",
        f"- Active but still gated method families: `{summary['active_lever_count']}`",
        f"- Required precondition method families: `{summary['required_precondition_count']}`",
        f"- Rejected-as-default method families: `{summary['default_rejected_count']}`",
        f"- Qwen3 final selector status: `{summary['qwen3_final_selection_status']}`",
        "",
        "## Method Gates",
        "",
        "| method family | gate | dense status | MoE status | allowed action | blocking probe |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in method_table.iterrows():
        lines.append(
            f"| `{row['method_family']}` | `{row['current_gate']}` | `{row['dense_status']}` | "
            f"`{row['moe_status']}` | {row['allowed_action']} | `{row['blocking_probe']}` |"
        )
    lines.extend(
        [
            "",
            "## Probe Gates",
            "",
            "| gate | probe | value | threshold | status | evidence |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for _, row in probe_table.iterrows():
        lines.append(
            f"| `{row['gate']}` | `{row['probe']}` | {fmt(row['value'])} | {fmt(row['threshold'])} | "
            f"`{row['status']}` | {row['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['method_gate_matrix']}`",
            f"- `{summary['outputs']['probe_gate_matrix']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    method_matrix = read_csv(args.method_matrix)
    source_matrix = read_csv(args.source_matrix)
    optimizer_summary = read_json(args.optimizer_summary)
    optimizer_features = read_csv(args.optimizer_features)
    method_rows = build_gate_rows(method_matrix, source_matrix, optimizer_summary, optimizer_features)
    probe_rows = build_probe_rows(optimizer_summary, optimizer_features)
    method_table = pd.DataFrame(method_rows)
    probe_table = pd.DataFrame(probe_rows)

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method_path = output_dir / "method_gate_matrix.csv"
    probe_path = output_dir / "probe_gate_matrix.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    current_gates = method_table["current_gate"].astype(str)
    summary = {
        "schema_version": 1,
        "status": "built_from_current_probe_evidence",
        "method_family_count": int(len(method_table)),
        "accepted_by_default_count": int((current_gates == "accepted_by_default").sum()),
        "conditional_count": int(current_gates.str.startswith("conditional").sum()),
        "active_lever_count": int(current_gates.str.startswith("active").sum()),
        "required_precondition_count": int(current_gates.str.startswith("required").sum()),
        "default_rejected_count": int((current_gates == "rejected_as_default").sum()),
        "qwen3_final_selection_status": (optimizer_summary.get("moe") or {}).get("qwen3_final_selection_status"),
        "qwen3_router_calibration_status": (optimizer_summary.get("moe") or {}).get(
            "qwen3_router_calibration_status"
        ),
        "qwen3_router_coupled_frontier_gate": (optimizer_summary.get("moe") or {}).get(
            "qwen3_router_coupled_frontier_gate"
        ),
        "qwen3_router_coupled_frontier_effect_fraction": (optimizer_summary.get("moe") or {}).get(
            "qwen3_router_coupled_frontier_effect_fraction"
        ),
        "dense_lambda_linear_worst_nll": (optimizer_summary.get("dense") or {}).get("lambda_linear_worst_nll"),
        "dense_lambda_best_worst_nll": (optimizer_summary.get("dense") or {}).get("lambda_best_worst_nll"),
        "qwen3_interpolation_interior_gap_nll": (optimizer_summary.get("moe") or {}).get(
            "qwen3_interpolation_interior_gap_nll"
        ),
        "outputs": {
            "method_gate_matrix": rel(method_path),
            "probe_gate_matrix": rel(probe_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }

    method_table.to_csv(method_path, index=False)
    probe_table.to_csv(probe_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, method_table, probe_table), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current-evidence gates for common averaging methods.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/average_method_gate_matrix"))
    parser.add_argument(
        "--method-matrix",
        type=Path,
        default=Path("results/model_averaging_literature_review/method_matrix.csv"),
    )
    parser.add_argument(
        "--source-matrix",
        type=Path,
        default=Path("results/model_averaging_literature_review/source_matrix.csv"),
    )
    parser.add_argument(
        "--optimizer-summary",
        type=Path,
        default=Path("results/unified_average_optimizer/summary.json"),
    )
    parser.add_argument(
        "--optimizer-features",
        type=Path,
        default=Path("results/unified_average_optimizer/mechanism_features.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args)
    print(f"Wrote average method gate matrix to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; methods={summary['method_family_count']}; "
        f"conditional={summary['conditional_count']}"
    )


if __name__ == "__main__":
    main()
