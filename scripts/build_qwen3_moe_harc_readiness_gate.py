#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


LITERATURE_SOURCES = [
    {
        "key": "harc",
        "title": "When Model Merging Breaks Routing: Training-Free Calibration for MoE",
        "url": "https://arxiv.org/abs/2606.03391",
        "mechanism": "MoE router merging should match source routing distributions; a second-order KL approximation yields a Hessian/input-covariance weighted linear system.",
    },
    {
        "key": "model_soups",
        "title": "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time",
        "url": "https://arxiv.org/abs/2203.05482",
        "mechanism": "Weight averaging is accepted only after validation against source or held-out controls.",
    },
    {
        "key": "ties",
        "title": "TIES-Merging: Resolving Interference When Merging Models",
        "url": "https://arxiv.org/abs/2306.01708",
        "mechanism": "Dense-model merging needs explicit interference handling; MoE routers need an additional top-k boundary gate.",
    },
    {
        "key": "regmean++",
        "title": "RegMean++: Enhancing Effectiveness and Generalization of Regression Mean for Model Merging",
        "url": "https://arxiv.org/abs/2508.03121",
        "mechanism": "Activation/statistics-aware closed-form merging is useful, but layer-local statistics must be checked against propagation and validation gates.",
    },
]


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


def read_csv(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def fnum(value: Any, default: float | None = None) -> float | None:
    value = clean_value(value)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 4) -> str:
    value = fnum(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.min()) if len(values) else 0.0
    hi = float(values.max()) if len(values) else 0.0
    if hi <= lo + EPS:
        return pd.Series(0.0, index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def build_layer_priority(layer_fragility: pd.DataFrame, layer_coupling: pd.DataFrame) -> pd.DataFrame:
    if layer_fragility.empty:
        return pd.DataFrame()
    keep = [
        "layer",
        "router",
        "boundary_fragility_score",
        "safe_lambda_proxy",
        "unsafe_fraction",
        "mean_topk_jaccard",
        "mean_top1_agreement",
        "mean_top1_margin",
        "router_relative_delta_norm",
        "fragility_rank",
        "router_margin_action",
    ]
    frame = layer_fragility[[col for col in keep if col in layer_fragility.columns]].copy()
    if not layer_coupling.empty:
        coupling_cols = [
            "layer_id",
            "route_mass",
            "weighted_router_instability_feature",
            "weighted_scale_shrink",
            "router_coupled_risk_sum",
            "route_mass_weighted_router_residual_pressure",
            "top_expert_id",
        ]
        frame = frame.merge(
            layer_coupling[[col for col in coupling_cols if col in layer_coupling.columns]],
            left_on="layer",
            right_on="layer_id",
            how="left",
        )
    for column in [
        "route_mass",
        "weighted_router_instability_feature",
        "weighted_scale_shrink",
        "router_coupled_risk_sum",
        "route_mass_weighted_router_residual_pressure",
    ]:
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    safe_lambda = pd.to_numeric(frame.get("safe_lambda_proxy", 0.0), errors="coerce").fillna(0.0)
    frame["low_margin_capacity_score"] = (1.0 - (safe_lambda / 0.05).clip(0.0, 1.0)).astype(float)
    frame["coupled_risk_score"] = normalize(frame["router_coupled_risk_sum"])
    frame["route_mass_score"] = normalize(frame["route_mass"])
    frame["harc_priority_score"] = (
        0.40 * pd.to_numeric(frame["boundary_fragility_score"], errors="coerce").fillna(0.0)
        + 0.25 * frame["coupled_risk_score"]
        + 0.20 * frame["low_margin_capacity_score"]
        + 0.10 * pd.to_numeric(frame.get("unsafe_fraction", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        + 0.05 * frame["route_mass_score"]
    )
    frame["harc_layer_role"] = "track_in_margin_profile"
    frame.loc[
        (frame["harc_priority_score"] >= 0.60)
        | (pd.to_numeric(frame.get("fragility_rank", 999), errors="coerce").fillna(999) <= 12),
        "harc_layer_role",
    ] = "collect_hessian_covariance_first"
    frame.loc[
        pd.to_numeric(frame.get("safe_lambda_proxy", 0.0), errors="coerce").fillna(0.0) < 0.025,
        "harc_layer_role",
    ] = "critical_topk_boundary_layer"
    frame["required_harc_statistics"] = "softmax_hessian_diag_offdiag,input_hidden_covariance,topk_assignment_overlap"
    return frame.sort_values(["harc_priority_score", "boundary_fragility_score"], ascending=[False, False])


def build_requirements(
    router_move: dict[str, Any],
    margin: dict[str, Any],
    nll: dict[str, Any],
    coupling: dict[str, Any],
    frontier: dict[str, Any],
    job: dict[str, Any],
    stats_dir: Path,
    *,
    min_nll_reduction: float,
    min_coupling_corr: float,
) -> pd.DataFrame:
    direct_rejected = (
        router_move.get("recommended_unified_router_action") == "freeze_router"
        and int(router_move.get("allowed_router_layer_count") or 0) == 0
    )
    margin_fragile = (
        margin.get("recommended_unified_router_action") == "freeze_router"
        and int(margin.get("high_fragility_layer_count") or 0) > 0
    )
    local_repair = (fnum(nll.get("worst_nll_reduction_vs_linear"), 0.0) or 0.0) >= min_nll_reduction
    expert_coupled = (
        coupling.get("gate") == "router_expert_coupling_active"
        and (fnum(coupling.get("fragility_router_feature_corr"), 0.0) or 0.0) >= min_coupling_corr
    )
    default_frontier = int(frontier.get("default_candidate_count") or 0) > 0
    job_ready = (
        str(job.get("status", "")).startswith("job_ready")
        and bool(job.get("prompts_exists"))
        and bool(job.get("source_controls_ready"))
        and bool(job.get("student_exists"))
        and bool(job.get("teacher_exists"))
    )
    cache_ready = repo_path(stats_dir).exists()
    rows = [
        {
            "requirement": "direct_router_average_rejected",
            "role": "harc_precondition",
            "passed": direct_rejected,
            "evidence": (
                f"router action={router_move.get('recommended_unified_router_action')}; "
                f"allowed layers={router_move.get('allowed_router_layer_count')}/"
                f"{router_move.get('router_layer_count')}; top-k jaccard mean/min="
                f"{fmt(router_move.get('mean_topk_jaccard'))}/{fmt(router_move.get('min_topk_jaccard'))}"
            ),
            "next_action": "freeze direct router average and use calibrated router objective",
        },
        {
            "requirement": "topk_boundary_fragility_detected",
            "role": "harc_precondition",
            "passed": margin_fragile,
            "evidence": (
                f"high fragility layers={margin.get('high_fragility_layer_count')}/"
                f"{margin.get('router_layer_count')}; min safe lambda={fmt(margin.get('min_safe_lambda_proxy'))}; "
                f"top layer=L{margin.get('top_fragile_layer')} score={fmt(margin.get('top_fragility_score'))}"
            ),
            "next_action": "prioritize fragile router layers for HARC statistics",
        },
        {
            "requirement": "router_only_repair_signal_positive",
            "role": "harc_precondition",
            "passed": local_repair,
            "evidence": (
                f"worst NLL reduction={fmt(nll.get('worst_nll_reduction_vs_linear'))}; "
                f"avg NLL reduction={fmt(nll.get('avg_nll_reduction_vs_linear'))}; "
                f"acceptance={nll.get('acceptance_decision')}"
            ),
            "next_action": "keep router calibration as repair, not acceptance, until matched vLLM passes",
        },
        {
            "requirement": "router_expert_coupling_active",
            "role": "harc_precondition",
            "passed": expert_coupled,
            "evidence": (
                f"gate={coupling.get('gate')}; fragility->feature corr="
                f"{fmt(coupling.get('fragility_router_feature_corr'))}; shrink corr="
                f"{fmt(coupling.get('fragility_scale_shrink_corr'))}"
            ),
            "next_action": "calibrate router together with expert-cap law, not as an isolated tensor average",
        },
        {
            "requirement": "safe_default_router_calibration_frontier_exists",
            "role": "harc_precondition",
            "passed": default_frontier,
            "evidence": (
                f"default candidates={frontier.get('default_candidate_count')}/"
                f"{frontier.get('candidate_count')}; recommended={frontier.get('recommended_default_candidates')}; "
                f"blocker={frontier.get('acceptance_blocker')}"
            ),
            "next_action": "compare HARC-style calibration against route-KD cap001 and margin_profile",
        },
        {
            "requirement": "calibration_job_preflight_ready",
            "role": "harc_precondition",
            "passed": job_ready,
            "evidence": (
                f"job status={job.get('status')}; prompts={job.get('prompts_exists')}; "
                f"source controls={job.get('source_controls_ready')}; student={job.get('student_exists')}; "
                f"teacher={job.get('teacher_exists')}"
            ),
            "next_action": "run preflight on GPU host, then collect router logits/hidden states",
        },
        {
            "requirement": "hessian_covariance_cache_available",
            "role": "harc_solver_requirement",
            "passed": cache_ready,
            "evidence": f"cache dir={rel(stats_dir)} exists={cache_ready}",
            "next_action": "collect H_i=diag(r)-rr^T and hidden covariance per router layer",
        },
    ]
    return pd.DataFrame(rows)


def decide_status(requirements: pd.DataFrame) -> str:
    lookup = {str(row["requirement"]): bool(row["passed"]) for _, row in requirements.iterrows()}
    if not lookup.get("direct_router_average_rejected", False):
        return "harc_not_recommended_router_average_not_rejected"
    if not lookup.get("router_only_repair_signal_positive", False):
        return "harc_not_recommended_no_local_repair_signal"
    preconditions = requirements[requirements["role"] == "harc_precondition"]
    if not bool(preconditions["passed"].all()):
        return "harc_waiting_for_precondition_evidence"
    if lookup.get("hessian_covariance_cache_available", False):
        return "harc_ready_for_matrix_free_solver"
    return "harc_ready_for_curvature_collection_waiting_cache"


def solver_plan(args: argparse.Namespace, status: str) -> dict[str, Any]:
    return {
        "objective": "min_Wm sum_i E_x KL(softmax(W_i x) || softmax(W_m x))",
        "quadratic_proxy": "0.5 * (W_m x - W_i x)^T H_i (W_m x - W_i x)",
        "hessian": "H_i = diag(r_i) - r_i r_i^T",
        "linear_system": "(sum_i E[H_i kron xx^T]) vec(W_m^T) = sum_i E[H_i kron xx^T] vec(W_i^T)",
        "solver": "matrix_free_conjugate_gradient",
        "same_shape_constraint": "only update existing router tensors; output checkpoint keeps the same architecture and tensor shapes",
        "current_status": status,
        "next_command": (
            "collect router logits and hidden states for source/candidate prompts, then solve the "
            "matrix-free HARC linear system before matched vLLM acceptance"
        ),
        "harc_stats_dir": rel(args.harc_stats_dir),
    }


def build_report(summary: dict[str, Any], requirements: pd.DataFrame, layers: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE HARC Readiness Gate",
        "",
        "This gate turns the MoE router-breakdown mechanism into an executable decision: direct router averaging stays rejected, and HARC-style router distribution matching is allowed only as a calibrated repair path with matched downstream acceptance.",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Preconditions: `{summary['precondition_passed_count']}/{summary['precondition_count']}`",
        f"- HARC cache: `{summary['hessian_covariance_cache_status']}`",
        f"- Top priority layer: `L{summary.get('top_harc_layer')}` score `{fmt(summary.get('top_harc_priority_score'))}`",
        f"- Recommended action: `{summary['recommended_action']}`",
        "",
        "## Requirements",
        "",
        "| requirement | role | passed | evidence | next action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in requirements.iterrows():
        lines.append(
            f"| `{row['requirement']}` | `{row['role']}` | `{bool(row['passed'])}` | "
            f"{row['evidence']} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Layer Priority",
            "",
            "| layer | score | role | fragility | safe lambda | coupled risk | route mass |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in layers.head(16).iterrows():
        lines.append(
            f"| {int(row['layer'])} | {fmt(row['harc_priority_score'])} | "
            f"`{row['harc_layer_role']}` | {fmt(row['boundary_fragility_score'])} | "
            f"{fmt(row['safe_lambda_proxy'])} | {fmt(row['router_coupled_risk_sum'])} | "
            f"{fmt(row['route_mass'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## HARC Objective",
            "",
            f"- `{summary['solver_plan']['objective']}`",
            f"- `{summary['solver_plan']['linear_system']}`",
            f"- Same-shape constraint: `{summary['solver_plan']['same_shape_constraint']}`",
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['requirements']}`",
            f"- `{summary['outputs']['layer_priority']}`",
            f"- `{summary['outputs']['solver_plan']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    router_move = read_json(args.router_move_summary)
    margin = read_json(args.router_margin_summary)
    nll = read_json(args.router_calibration_nll_summary)
    coupling = read_json(args.router_expert_coupling_summary)
    frontier = read_json(args.router_calibration_frontier_summary)
    job = read_json(args.router_calibration_job_summary)
    layers = build_layer_priority(read_csv(args.layer_fragility), read_csv(args.layer_coupling))
    requirements = build_requirements(
        router_move,
        margin,
        nll,
        coupling,
        frontier,
        job,
        args.harc_stats_dir,
        min_nll_reduction=args.min_nll_reduction,
        min_coupling_corr=args.min_coupling_corr,
    )
    status = decide_status(requirements)
    preconditions = requirements[requirements["role"] == "harc_precondition"]
    cache_ready = bool(
        requirements.loc[
            requirements["requirement"] == "hessian_covariance_cache_available",
            "passed",
        ].iloc[0]
    )
    top = layers.iloc[0].to_dict() if not layers.empty else {}
    if status == "harc_ready_for_matrix_free_solver":
        action = "run_matrix_free_harc_solver_then_matched_vllm_gate"
    elif status == "harc_ready_for_curvature_collection_waiting_cache":
        action = "collect_hessian_covariance_router_stats_then_run_harc_solver"
    elif status == "harc_not_recommended_router_average_not_rejected":
        action = "do_not_spend_harc_budget_until_direct_router_average_is_shown_unsafe"
    elif status == "harc_not_recommended_no_local_repair_signal":
        action = "do_not_spend_harc_budget_until_router_only_repair_has_positive_signal"
    else:
        action = "complete_missing_precondition_probe_before_harc"
    summary = {
        "schema_version": 1,
        "status": status,
        "precondition_count": int(len(preconditions)),
        "precondition_passed_count": int(preconditions["passed"].sum()),
        "hessian_covariance_cache_status": "available" if cache_ready else "missing",
        "recommended_action": action,
        "top_harc_layer": int(top["layer"]) if top else None,
        "top_harc_priority_score": fnum(top.get("harc_priority_score")) if top else None,
        "critical_layer_count": int((layers["harc_layer_role"] == "critical_topk_boundary_layer").sum())
        if not layers.empty
        else 0,
        "first_stage_layer_count": int(
            layers["harc_layer_role"].isin(
                ["critical_topk_boundary_layer", "collect_hessian_covariance_first"]
            ).sum()
        )
        if not layers.empty
        else 0,
        "solver_plan": solver_plan(args, status),
        "literature_sources": LITERATURE_SOURCES,
        "outputs": {
            "requirements": rel(output_dir / "harc_readiness_requirements.csv"),
            "layer_priority": rel(output_dir / "layer_harc_priority.csv"),
            "solver_plan": rel(output_dir / "harc_solver_plan.json"),
            "literature": rel(output_dir / "literature_sources.json"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    requirements.to_csv(output_dir / "harc_readiness_requirements.csv", index=False)
    layers.to_csv(output_dir / "layer_harc_priority.csv", index=False)
    (output_dir / "harc_solver_plan.json").write_text(
        json.dumps(json_safe(summary["solver_plan"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "literature_sources.json").write_text(
        json.dumps(json_safe(LITERATURE_SOURCES), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(build_report(summary, requirements, layers), encoding="utf-8")
    return summary


def write_mock_case(root: Path, case: str, *, direct_rejected: bool, local_repair: bool) -> argparse.Namespace:
    case_dir = root / case
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "router_move.json").write_text(
        json.dumps(
            {
                "recommended_unified_router_action": "freeze_router" if direct_rejected else "allow_small_router_average",
                "allowed_router_layer_count": 0 if direct_rejected else 4,
                "router_layer_count": 4,
                "mean_topk_jaccard": 0.42,
                "min_topk_jaccard": 0.20,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "margin.json").write_text(
        json.dumps(
            {
                "recommended_unified_router_action": "freeze_router",
                "high_fragility_layer_count": 2,
                "router_layer_count": 4,
                "min_safe_lambda_proxy": 0.018,
                "top_fragile_layer": 1,
                "top_fragility_score": 0.82,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "nll.json").write_text(
        json.dumps(
            {
                "worst_nll_reduction_vs_linear": 0.12 if local_repair else -0.01,
                "avg_nll_reduction_vs_linear": 0.08 if local_repair else -0.01,
                "acceptance_decision": "mechanism_supported_but_do_not_accept_without_matched_vllm_eval",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "coupling.json").write_text(
        json.dumps(
            {
                "gate": "router_expert_coupling_active",
                "fragility_router_feature_corr": 0.7,
                "fragility_scale_shrink_corr": 0.5,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "frontier.json").write_text(
        json.dumps(
            {
                "default_candidate_count": 2,
                "candidate_count": 4,
                "recommended_default_candidates": ["cap001", "margin_profile"],
                "acceptance_blocker": "baseline_eval,source_eval,candidate_eval,audit",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "job.json").write_text(
        json.dumps(
            {
                "status": "job_ready_awaiting_gpu",
                "prompts_exists": True,
                "source_controls_ready": True,
                "student_exists": True,
                "teacher_exists": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "layer": 1,
                "router": "mock.layers.1.mlp.gate",
                "boundary_fragility_score": 0.82,
                "safe_lambda_proxy": 0.018,
                "unsafe_fraction": 1.0,
                "mean_topk_jaccard": 0.35,
                "mean_top1_agreement": 0.30,
                "mean_top1_margin": 0.03,
                "router_relative_delta_norm": 0.8,
                "fragility_rank": 1,
                "router_margin_action": "freeze_router_prioritize_calibration",
            },
            {
                "layer": 2,
                "router": "mock.layers.2.mlp.gate",
                "boundary_fragility_score": 0.55,
                "safe_lambda_proxy": 0.05,
                "unsafe_fraction": 0.7,
                "mean_topk_jaccard": 0.55,
                "mean_top1_agreement": 0.45,
                "mean_top1_margin": 0.06,
                "router_relative_delta_norm": 0.4,
                "fragility_rank": 2,
                "router_margin_action": "freeze_router",
            },
        ]
    ).to_csv(case_dir / "layer_fragility.csv", index=False)
    pd.DataFrame(
        [
            {
                "layer_id": 1,
                "route_mass": 12.0,
                "weighted_router_instability_feature": 0.9,
                "weighted_scale_shrink": 0.02,
                "router_coupled_risk_sum": 6.0,
                "route_mass_weighted_router_residual_pressure": 0.1,
                "top_expert_id": 7,
            },
            {
                "layer_id": 2,
                "route_mass": 6.0,
                "weighted_router_instability_feature": 0.4,
                "weighted_scale_shrink": 0.01,
                "router_coupled_risk_sum": 2.0,
                "route_mass_weighted_router_residual_pressure": 0.0,
                "top_expert_id": 3,
            },
        ]
    ).to_csv(case_dir / "layer_coupling.csv", index=False)
    return argparse.Namespace(
        output_dir=root / "case_outputs" / case,
        router_move_summary=case_dir / "router_move.json",
        router_margin_summary=case_dir / "margin.json",
        router_calibration_nll_summary=case_dir / "nll.json",
        router_expert_coupling_summary=case_dir / "coupling.json",
        router_calibration_frontier_summary=case_dir / "frontier.json",
        router_calibration_job_summary=case_dir / "job.json",
        layer_fragility=case_dir / "layer_fragility.csv",
        layer_coupling=case_dir / "layer_coupling.csv",
        harc_stats_dir=case_dir / "missing_harc_stats",
        min_nll_reduction=0.01,
        min_coupling_corr=0.3,
    )


def build_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mock_root = output_dir / "mock_inputs"
    cases = [
        ("harc_ready", True, True, "harc_ready_for_curvature_collection_waiting_cache"),
        ("router_safe", False, True, "harc_not_recommended_router_average_not_rejected"),
        ("no_repair", True, False, "harc_not_recommended_no_local_repair_signal"),
    ]
    rows = []
    for case, direct_rejected, local_repair, expected in cases:
        case_args = write_mock_case(
            mock_root,
            case,
            direct_rejected=direct_rejected,
            local_repair=local_repair,
        )
        summary = build(case_args)
        rows.append(
            {
                "case": case,
                "expected_status": expected,
                "actual_status": summary["status"],
                "passed": summary["status"] == expected,
                "top_harc_layer": summary.get("top_harc_layer"),
                "preconditions": f"{summary['precondition_passed_count']}/{summary['precondition_count']}",
            }
        )
    smoke = pd.DataFrame(rows)
    passed = bool(smoke["passed"].all())
    smoke_summary = {
        "schema_version": 1,
        "status": "smoke_passed" if passed else "smoke_failed",
        "case_count": int(len(smoke)),
        "passed_case_count": int(smoke["passed"].sum()),
        "smoke_input_dir": rel(mock_root),
        "outputs": {
            "cases": rel(output_dir / "smoke_cases.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    smoke.to_csv(output_dir / "smoke_cases.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(smoke_summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Qwen3 MoE HARC Readiness Smoke",
        "",
        f"- Status: `{smoke_summary['status']}`",
        f"- Cases: `{smoke_summary['passed_case_count']}/{smoke_summary['case_count']}`",
        "",
        "| case | expected | actual | passed |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in smoke.iterrows():
        lines.append(
            f"| `{row['case']}` | `{row['expected_status']}` | `{row['actual_status']}` | `{row['passed']}` |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)
    return smoke_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HARC-style readiness gate for Qwen3 MoE router calibration.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_harc_readiness_gate"))
    parser.add_argument(
        "--router-move-summary",
        type=Path,
        default=Path("results/qwen3_moe_router_move_gate/summary.json"),
    )
    parser.add_argument(
        "--router-margin-summary",
        type=Path,
        default=Path("results/qwen3_moe_router_margin_fragility/summary.json"),
    )
    parser.add_argument(
        "--router-calibration-nll-summary",
        type=Path,
        default=Path("results/qwen3_moe_router_calibration_nll_probe/summary.json"),
    )
    parser.add_argument(
        "--router-expert-coupling-summary",
        type=Path,
        default=Path("results/qwen3_moe_router_expert_coupling/summary.json"),
    )
    parser.add_argument(
        "--router-calibration-frontier-summary",
        type=Path,
        default=Path("results/qwen3_moe_router_calibration_frontier/summary.json"),
    )
    parser.add_argument(
        "--router-calibration-job-summary",
        type=Path,
        default=Path("results/qwen3_moe_router_calibration_job/summary.json"),
    )
    parser.add_argument(
        "--layer-fragility",
        type=Path,
        default=Path("results/qwen3_moe_router_margin_fragility/layer_margin_fragility.csv"),
    )
    parser.add_argument(
        "--layer-coupling",
        type=Path,
        default=Path("results/qwen3_moe_router_expert_coupling/layer_router_expert_coupling.csv"),
    )
    parser.add_argument(
        "--harc-stats-dir",
        type=Path,
        default=Path("results/qwen3_moe_harc_router_stats"),
    )
    parser.add_argument("--min-nll-reduction", type=float, default=0.01)
    parser.add_argument("--min-coupling-corr", type=float, default=0.30)
    parser.add_argument("--smoke-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_smoke(args) if args.smoke_matrix else build(args)
    print(f"Wrote Qwen3 MoE HARC readiness gate to {repo_path(args.output_dir).resolve()}")
    if args.smoke_matrix:
        print(f"Status: {summary['status']}; cases={summary['passed_case_count']}/{summary['case_count']}")
    else:
        print(
            "Status: "
            f"{summary['status']}; preconditions={summary['precondition_passed_count']}/"
            f"{summary['precondition_count']}; top_layer=L{summary.get('top_harc_layer')}"
        )


if __name__ == "__main__":
    main()
