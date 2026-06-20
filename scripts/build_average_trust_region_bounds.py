#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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


def fint(value: Any) -> int | None:
    value = clean_value(value)
    return None if value is None else int(value)


def fmt(value: Any, digits: int = 4) -> str:
    value = fnum(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def max_safe_lambda(rows: list[dict[str, Any]], endpoint_worst: float | None, mode: str) -> float | None:
    if endpoint_worst is None:
        return None
    accepted = [
        fnum(row.get("lambda"))
        for row in rows
        if str(row.get("mode")) == mode
        and fnum(row.get("lambda")) is not None
        and fnum(row.get("worst")) is not None
        and float(row["worst"]) <= endpoint_worst
    ]
    return max(accepted) if accepted else None


def accepted_interior_count(path: list[dict[str, Any]], endpoint_worst: float | None) -> int:
    if endpoint_worst is None:
        return 0
    count = 0
    for row in path:
        t = fnum(row.get("t"))
        worst = fnum(row.get("worst"))
        if t is None or worst is None or t <= 0.0 or t >= 1.0:
            continue
        if worst <= endpoint_worst:
            count += 1
    return count


def constraint_row(
    *,
    domain: str,
    mechanism: str,
    bound_type: str,
    measured_value: Any,
    allowed_value: Any,
    candidate_value: Any,
    status: str,
    action: str,
    evidence: str,
) -> dict[str, Any]:
    measured = fnum(measured_value)
    allowed = fnum(allowed_value)
    candidate = fnum(candidate_value)
    excess_ratio = None
    margin = None
    if candidate is not None and allowed is not None:
        margin = allowed - candidate
        if allowed != 0.0:
            excess_ratio = candidate / allowed
    return {
        "domain": domain,
        "mechanism": mechanism,
        "bound_type": bound_type,
        "measured_value": measured,
        "allowed_value": allowed,
        "candidate_value": candidate,
        "margin_to_bound": margin,
        "candidate_over_bound": excess_ratio,
        "status": status,
        "action": action,
        "evidence": evidence,
    }


def build_constraints(inputs: dict[str, dict[str, Any]], curvature_ratio_limit: float) -> pd.DataFrame:
    dense_curvature = inputs["dense_curvature"]
    dense_lambda = inputs["dense_lambda"]
    moe_line = inputs["moe_line"]
    moe_complementary = inputs["moe_complementary"]
    moe_base_coder = inputs["moe_base_coder"]
    router_margin = inputs["router_margin"]
    unified = inputs["unified_candidate"]
    mechanistic = inputs["mechanistic_candidate"]
    final_selector = inputs["final_selector"]
    router_cal_nll = inputs["router_calibration_nll"]
    router_selector = inputs["router_selector"]
    downstream_confidence = inputs["downstream_confidence"]

    curve = dense_curvature.get("curvature_law") or {}
    ratio_general = fnum(curve.get("ratio_general"))
    ratio_code = fnum(curve.get("ratio_code"))
    max_ratio = max([v for v in [ratio_general, ratio_code] if v is not None], default=None)
    local_lambda_bound = None
    if max_ratio is not None and max_ratio > 0:
        local_lambda_bound = math.sqrt(curvature_ratio_limit / max_ratio)

    dense_endpoint = fnum(dense_lambda.get("best_endpoint_worst"))
    dense_safe_uniform_lambda = max_safe_lambda(dense_lambda.get("rows") or [], dense_endpoint, "uniform")
    dense_safe_sign_lambda = max_safe_lambda(dense_lambda.get("rows") or [], dense_endpoint, "sign_resolved")
    dense_linear_worst = fnum(dense_lambda.get("linear_worst"))
    dense_best_worst = fnum(dense_lambda.get("unified_best_worst"))

    moe_endpoint = fnum(moe_line.get("endpoint_best_worst"))
    moe_best_interior = fnum(moe_line.get("best_interior_worst"))
    moe_gap = None if moe_endpoint is None or moe_best_interior is None else moe_best_interior - moe_endpoint
    moe_accepted_interiors = accepted_interior_count(moe_line.get("interpolation") or [], moe_endpoint)

    base_coder_endpoint = fnum(moe_base_coder.get("endpoint_best_worst"))
    base_coder_best_interior = fnum(moe_base_coder.get("best_interior_worst"))
    base_coder_gap = (
        None
        if base_coder_endpoint is None or base_coder_best_interior is None
        else base_coder_best_interior - base_coder_endpoint
    )
    base_coder_accepted_interiors = accepted_interior_count(
        moe_base_coder.get("interpolation") or [], base_coder_endpoint
    )

    complementary_best_merge = fnum(moe_complementary.get("best_merge_avg_nll"))
    complementary_best_source = fnum(moe_complementary.get("best_source_avg_nll"))
    complementary_gap = (
        None
        if complementary_best_merge is None or complementary_best_source is None
        else complementary_best_merge - complementary_best_source
    )

    router_safe_lambda = fnum(router_margin.get("min_safe_lambda_proxy"))
    router_average_lambda = 0.5
    router_eligible = fint((router_selector.get("current_selection") or {}).get("eligible_candidate_count"))
    router_candidates = fint((router_selector.get("current_selection") or {}).get("candidate_count"))
    final_current = final_selector.get("current_selection") or {}
    final_eligible = fint(final_current.get("eligible_candidate_count"))
    final_candidates = fint(final_current.get("candidate_count"))

    unified_max_delta = fnum(unified.get("selected_max_predicted_relative_delta"))
    unified_hard_cap = fnum(unified.get("hard_cap"))
    mechanistic_max_delta = fnum(mechanistic.get("selected_max_predicted_relative_delta"))
    mechanistic_cap = fnum(mechanistic.get("effective_hard_cap") or mechanistic.get("hard_cap"))
    mechanistic_retention = fnum(mechanistic.get("selected_nonbase_mass_retention"))
    mechanistic_min_retention = fnum(mechanistic.get("min_retention"))

    routercal_worst_reduction = fnum(router_cal_nll.get("worst_nll_reduction_vs_linear"))
    confidence_positive = fint(downstream_confidence.get("routercal_confident_positive_task_count_vs_naive"))
    task_count = fint(downstream_confidence.get("task_count"))

    rows = [
        constraint_row(
            domain="dense",
            mechanism="local_quadratic_trust_radius",
            bound_type="lambda_bound_from_actual_over_predicted_curvature",
            measured_value=max_ratio,
            allowed_value=local_lambda_bound,
            candidate_value=1.0,
            status="reject_linear_task_vector_average"
            if local_lambda_bound is not None and local_lambda_bound < 1.0
            else "local_quadratic_bound_allows_linear",
            action="Search a held-out coefficient family and allow endpoint/anchor fallback.",
            evidence=(
                f"actual/predicted curvature ratios general/code = {fmt(ratio_general)}/{fmt(ratio_code)}; "
                f"using ratio limit {fmt(curvature_ratio_limit)}, the derived full-task-vector lambda bound is "
                f"{fmt(local_lambda_bound)}."
            ),
        ),
        constraint_row(
            domain="dense",
            mechanism="heldout_lambda_frontier",
            bound_type="max_uniform_lambda_not_worse_than_endpoint",
            measured_value=dense_linear_worst,
            allowed_value=dense_safe_uniform_lambda,
            candidate_value=1.0,
            status="reject_uniform_linear_lambda"
            if dense_safe_uniform_lambda is None or dense_safe_uniform_lambda < 1.0
            else "uniform_linear_lambda_allowed",
            action="Use the observed best lambda config instead of fixed 0.5/0.5 averaging.",
            evidence=(
                f"linear worst NLL = {fmt(dense_linear_worst)}; best lambda-family worst NLL = "
                f"{fmt(dense_best_worst)}; best endpoint worst NLL = {fmt(dense_endpoint)}; "
                f"max accepted uniform lambda = {fmt(dense_safe_uniform_lambda)}."
            ),
        ),
        constraint_row(
            domain="dense",
            mechanism="sparse_sign_trust_radius",
            bound_type="max_sign_resolved_lambda_not_worse_than_endpoint",
            measured_value=None,
            allowed_value=dense_safe_sign_lambda,
            candidate_value=1.0,
            status="keep_sparse_methods_conditional"
            if dense_safe_sign_lambda is None or dense_safe_sign_lambda < 1.0
            else "sign_resolved_linear_lambda_allowed",
            action="Treat sign/sparsity conflict as an ablation until it beats endpoint and anchor gates.",
            evidence=f"max accepted sign-resolved lambda = {fmt(dense_safe_sign_lambda)} under endpoint worst NLL {fmt(dense_endpoint)}.",
        ),
        constraint_row(
            domain="moe",
            mechanism="qwen3_instruct_coder_source_line",
            bound_type="interior_path_must_beat_endpoint_frontier",
            measured_value=moe_gap,
            allowed_value=0.0,
            candidate_value=moe_gap,
            status="reject_source_to_source_linear_average"
            if moe_gap is None or moe_gap > 0.0 or moe_accepted_interiors == 0
            else "source_line_has_accepted_interior",
            action="Use route/evidence/geometry/subspace constrained same-shape expert rules, not source midpoint.",
            evidence=(
                f"best interior worst NLL = {fmt(moe_best_interior)}; best endpoint worst NLL = {fmt(moe_endpoint)}; "
                f"interior gap = {fmt(moe_gap)}; accepted interior count = {moe_accepted_interiors}."
            ),
        ),
        constraint_row(
            domain="moe",
            mechanism="qwen3_base_coder_source_line",
            bound_type="base_to_specialist_delta_must_beat_endpoint_frontier",
            measured_value=base_coder_gap,
            allowed_value=0.0,
            candidate_value=base_coder_gap,
            status="reject_base_to_specialist_source_line"
            if base_coder_gap is None or base_coder_gap > 0.0 or base_coder_accepted_interiors == 0
            else "base_to_specialist_line_has_accepted_interior",
            action="Do not assume base-anchored source deltas are safe without the same path gate.",
            evidence=(
                f"best interior worst NLL = {fmt(base_coder_best_interior)}; best endpoint worst NLL = "
                f"{fmt(base_coder_endpoint)}; interior gap = {fmt(base_coder_gap)}; "
                f"accepted interior count = {base_coder_accepted_interiors}."
            ),
        ),
        constraint_row(
            domain="moe",
            mechanism="qwen3_specialist_complementarity",
            bound_type="complementary_pair_merge_must_beat_best_source",
            measured_value=complementary_gap,
            allowed_value=0.0,
            candidate_value=complementary_gap,
            status="do_not_assume_complementarity_is_averageable"
            if not bool(moe_complementary.get("merge_beats_both_sources_on_avg"))
            else "complementary_average_candidate_allowed",
            action="Keep specialist complementarity as an eval hypothesis, not as an average acceptance rule.",
            evidence=(
                f"best merge avg NLL = {fmt(complementary_best_merge)}; best source avg NLL = "
                f"{fmt(complementary_best_source)}; best merge t = {fmt(moe_complementary.get('best_merge_t'))}; "
                f"merge beats both sources = {moe_complementary.get('merge_beats_both_sources_on_avg')}."
            ),
        ),
        constraint_row(
            domain="moe",
            mechanism="router_topk_margin",
            bound_type="router_lambda_bound_from_min_safe_margin",
            measured_value=router_safe_lambda,
            allowed_value=router_safe_lambda,
            candidate_value=router_average_lambda,
            status="reject_direct_router_average"
            if router_safe_lambda is None or router_average_lambda > router_safe_lambda
            else "direct_router_average_allowed",
            action="Freeze router for expert candidates; only allow separately audited route-KD deltas.",
            evidence=(
                f"min safe-lambda proxy = {fmt(router_safe_lambda)}; direct midpoint router lambda = "
                f"{fmt(router_average_lambda)}; high-fragility layers = "
                f"{router_margin.get('high_fragility_layer_count')}/{router_margin.get('router_layer_count')}."
            ),
        ),
        constraint_row(
            domain="moe",
            mechanism="unified_routed_expert_cap",
            bound_type="predicted_relative_delta_hard_cap",
            measured_value=unified_max_delta,
            allowed_value=unified_hard_cap,
            candidate_value=unified_max_delta,
            status="expert_delta_cap_passed"
            if unified_max_delta is not None and unified_hard_cap is not None and unified_max_delta <= unified_hard_cap
            else "expert_delta_cap_failed",
            action="Keep the unified mechanism candidate provisional until downstream source-dominance gates pass.",
            evidence=(
                f"candidate {unified.get('selected_candidate_id')} max predicted relative delta = "
                f"{fmt(unified_max_delta)}; hard cap = {fmt(unified_hard_cap)}; retention = "
                f"{fmt(unified.get('selected_nonbase_mass_retention'))}; routed >0.65 groups = "
                f"{unified.get('selected_routed_gt_065_groups')}."
            ),
        ),
        constraint_row(
            domain="moe",
            mechanism="mechanistic_scale_law_cap",
            bound_type="effective_hard_cap_after_materialization_margin",
            measured_value=mechanistic_max_delta,
            allowed_value=mechanistic_cap,
            candidate_value=mechanistic_max_delta,
            status="mechanistic_cap_and_retention_passed"
            if (
                mechanistic_max_delta is not None
                and mechanistic_cap is not None
                and mechanistic_retention is not None
                and mechanistic_min_retention is not None
                and mechanistic_max_delta <= mechanistic_cap
                and mechanistic_retention >= mechanistic_min_retention
            )
            else "mechanistic_cap_or_retention_failed",
            action="Use as a structural-frontier candidate, but do not override a statistically separated downstream leader.",
            evidence=(
                f"candidate {mechanistic.get('selected_candidate_id')} max predicted relative delta = "
                f"{fmt(mechanistic_max_delta)}; effective cap = {fmt(mechanistic_cap)}; route-mass retention = "
                f"{fmt(mechanistic_retention)}; min retention = {fmt(mechanistic_min_retention)}."
            ),
        ),
        constraint_row(
            domain="moe",
            mechanism="router_calibration_acceptance",
            bound_type="router_calibration_needs_matched_baseline_eval",
            measured_value=routercal_worst_reduction,
            allowed_value=router_candidates,
            candidate_value=router_eligible,
            status="router_calibration_promising_but_unaccepted"
            if routercal_worst_reduction is not None and routercal_worst_reduction > 0.0 and not router_eligible
            else "router_calibration_gate_state",
            action="Run matched frozen-router baseline/source/candidate vLLM eval before attaching router deltas.",
            evidence=(
                f"router-cal worst-NLL reduction vs linear = {fmt(routercal_worst_reduction)}; "
                f"eligible router-cal candidates = {router_eligible}/{router_candidates}; "
                f"confidence-positive generation tasks = {confidence_positive}/{task_count}."
            ),
        ),
        constraint_row(
            domain="moe",
            mechanism="final_average_acceptance",
            bound_type="audited_downstream_source_dominance_gate",
            measured_value=final_eligible,
            allowed_value=final_candidates,
            candidate_value=final_eligible,
            status="awaiting_matched_vllm_eval"
            if not final_eligible
            else "downstream_candidates_available_for_selection",
            action="Accept no average until source endpoints and same-shape candidates pass locked-manifest eval bundle audit.",
            evidence=(
                f"final selector status = {final_current.get('status')}; eligible candidates = "
                f"{final_eligible}/{final_candidates}; reason = {final_current.get('reason')}."
            ),
        ),
    ]
    return pd.DataFrame(rows)


def build_algorithm(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "same_shape_average_trust_region_policy",
        "contract": "The output model must keep the input model class, config, tokenizer, tensor names, tensor shapes, router count, and expert count.",
        "dense_rule": [
            "Estimate whether the local quadratic/Fisher approximation is valid along the candidate task-vector direction.",
            "If actual/predicted degradation is above the trust threshold, reject fixed midpoint averaging.",
            "Search coefficients on held-out tasks and keep endpoint/anchor fallback in the candidate set.",
        ],
        "moe_rule": [
            "Canonicalize or verify expert identity before any expert tensor average.",
            "Reject direct router movement unless router top-k margin/load gates allow a bounded delta.",
            "Move routed experts under route/evidence/geometry/subspace caps and keep structural candidates provisional.",
            "Accept an average only after audited source/candidate vLLM eval proves source dominance and task-regression gates.",
        ],
        "current_bounds": {
            "dense_local_task_vector_lambda_bound": summary.get("dense_local_task_vector_lambda_bound"),
            "dense_safe_uniform_lambda": summary.get("dense_safe_uniform_lambda"),
            "moe_router_safe_lambda_proxy": summary.get("moe_router_safe_lambda_proxy"),
            "mechanistic_effective_expert_delta_cap": summary.get("mechanistic_effective_expert_delta_cap"),
        },
    }


def build_report(summary: dict[str, Any], constraints: pd.DataFrame, decisions: pd.DataFrame) -> str:
    lines = [
        "# Average Trust-Region Bounds",
        "",
        "这个产物把 Dense 和 MoE averaging 的 probe 转成可执行 trust-region 约束：不是问某个算法名是否流行，而是问当前证据允许哪些参数方向移动多远。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Constraints: `{summary['constraint_count']}`",
        f"- Passed / rejected / waiting: `{summary['passed_count']}` / `{summary['rejected_count']}` / `{summary['waiting_count']}`",
        f"- Dense local task-vector lambda bound: `{fmt(summary.get('dense_local_task_vector_lambda_bound'))}`",
        f"- Dense safe uniform lambda from held-out path: `{fmt(summary.get('dense_safe_uniform_lambda'))}`",
        f"- MoE router safe lambda proxy: `{fmt(summary.get('moe_router_safe_lambda_proxy'))}`",
        f"- Mechanistic expert delta cap: `{fmt(summary.get('mechanistic_effective_expert_delta_cap'))}`",
        "",
        "## Constraint Bounds",
        "",
        "| domain | mechanism | status | measured | allowed | candidate | over bound | action |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in constraints.iterrows():
        lines.append(
            "| "
            f"`{row['domain']}` | `{row['mechanism']}` | `{row['status']}` | "
            f"{fmt(row['measured_value'])} | {fmt(row['allowed_value'])} | {fmt(row['candidate_value'])} | "
            f"{fmt(row['candidate_over_bound'])} | {row['action']} |"
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "| scope | decision | evidence | next gate |",
            "| --- | --- | --- | --- |",
        ]
    )
    for _, row in decisions.iterrows():
        lines.append(
            f"| `{row['scope']}` | `{row['decision']}` | {row['evidence']} | `{row['next_gate']}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `constraints`: `{summary['outputs']['constraints']}`",
            f"- `decisions`: `{summary['outputs']['decisions']}`",
            f"- `algorithm`: `{summary['outputs']['algorithm']}`",
            f"- `summary`: `{summary['outputs']['summary']}`",
            f"- `report`: `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified Dense/MoE average trust-region bounds.")
    parser.add_argument("--dense-curvature", type=Path, default=Path("results/fp_curvature_law/summary.json"))
    parser.add_argument("--dense-lambda", type=Path, default=Path("results/fp_dense_lambda/summary.json"))
    parser.add_argument("--moe-line", type=Path, default=Path("results/fp_moe_barrier/summary.json"))
    parser.add_argument("--moe-complementary", type=Path, default=Path("results/fp_moe_complementary/summary.json"))
    parser.add_argument("--moe-base-coder", type=Path, default=Path("results/fp_moe_forgetting_base_coder/summary.json"))
    parser.add_argument(
        "--router-margin", type=Path, default=Path("results/qwen3_moe_router_margin_fragility/summary.json")
    )
    parser.add_argument(
        "--unified-candidate", type=Path, default=Path("results/qwen3_moe_unified_mechanism_candidate/summary.json")
    )
    parser.add_argument(
        "--mechanistic-candidate",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_unified_candidate/summary.json"),
    )
    parser.add_argument(
        "--final-selector", type=Path, default=Path("results/qwen3_moe_final_candidate_selection/summary.json")
    )
    parser.add_argument(
        "--router-calibration-nll",
        type=Path,
        default=Path("results/qwen3_moe_router_calibration_nll_probe/summary.json"),
    )
    parser.add_argument(
        "--router-selector", type=Path, default=Path("results/qwen3_moe_router_calibration_selection/summary.json")
    )
    parser.add_argument(
        "--downstream-confidence",
        type=Path,
        default=Path("results/fp_downstream_confidence_audit/summary.json"),
    )
    parser.add_argument("--curvature-ratio-limit", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/average_trust_region_bounds"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "dense_curvature": read_json(args.dense_curvature),
        "dense_lambda": read_json(args.dense_lambda),
        "moe_line": read_json(args.moe_line),
        "moe_complementary": read_json(args.moe_complementary),
        "moe_base_coder": read_json(args.moe_base_coder),
        "router_margin": read_json(args.router_margin),
        "unified_candidate": read_json(args.unified_candidate),
        "mechanistic_candidate": read_json(args.mechanistic_candidate),
        "final_selector": read_json(args.final_selector),
        "router_calibration_nll": read_json(args.router_calibration_nll),
        "router_selector": read_json(args.router_selector),
        "downstream_confidence": read_json(args.downstream_confidence),
    }
    constraints = build_constraints(inputs, args.curvature_ratio_limit)
    rejected_mask = constraints["status"].astype(str).str.contains("reject|conditional|do_not")
    passed_mask = constraints["status"].astype(str).str.contains("passed|allowed")
    waiting_mask = constraints["status"].astype(str).str.contains("awaiting|unaccepted")

    by_mechanism = {str(row["mechanism"]): row.to_dict() for _, row in constraints.iterrows()}
    dense_local = by_mechanism.get("local_quadratic_trust_radius", {})
    dense_lambda = by_mechanism.get("heldout_lambda_frontier", {})
    router_bound = by_mechanism.get("router_topk_margin", {})
    mechanistic_bound = by_mechanism.get("mechanistic_scale_law_cap", {})
    final_bound = by_mechanism.get("final_average_acceptance", {})
    status = (
        "trust_region_bounds_ready_waiting_vllm"
        if str(final_bound.get("status")) == "awaiting_matched_vllm_eval"
        else "trust_region_bounds_ready"
    )
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    constraints_path = output_dir / "trust_region_constraints.csv"
    decisions_path = output_dir / "trust_region_decisions.csv"
    algorithm_path = output_dir / "algorithm.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    decisions = pd.DataFrame(
        [
            {
                "scope": "dense",
                "decision": "reject fixed midpoint; use coefficient search with endpoint or anchor fallback",
                "evidence": dense_lambda.get("evidence"),
                "next_gate": "held-out generation or vLLM eval for any new interior point",
            },
            {
                "scope": "moe_router",
                "decision": "freeze direct router average; route-KD/router calibration remains a separate ablation",
                "evidence": router_bound.get("evidence"),
                "next_gate": "matched frozen-router baseline/source/candidate vLLM eval",
            },
            {
                "scope": "moe_experts",
                "decision": "allow capped same-shape expert movement only inside routed delta and retention bounds",
                "evidence": mechanistic_bound.get("evidence"),
                "next_gate": "locked-manifest downstream source-dominance selector",
            },
            {
                "scope": "final_selection",
                "decision": "do not accept any average before audited downstream eval bundles exist",
                "evidence": final_bound.get("evidence"),
                "next_gate": "qwen3_moe_eval_bundle_audit plus final candidate selector",
            },
        ]
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "constraint_count": int(len(constraints)),
        "passed_count": int(passed_mask.sum()),
        "rejected_count": int(rejected_mask.sum()),
        "waiting_count": int(waiting_mask.sum()),
        "curvature_ratio_limit": args.curvature_ratio_limit,
        "dense_local_task_vector_lambda_bound": clean_value(dense_local.get("allowed_value")),
        "dense_safe_uniform_lambda": clean_value(dense_lambda.get("allowed_value")),
        "dense_linear_candidate_over_safe_uniform_bound": clean_value(dense_lambda.get("candidate_over_bound")),
        "moe_router_safe_lambda_proxy": clean_value(router_bound.get("allowed_value")),
        "moe_direct_router_average_over_safe_bound": clean_value(router_bound.get("candidate_over_bound")),
        "mechanistic_effective_expert_delta_cap": clean_value(mechanistic_bound.get("allowed_value")),
        "mechanistic_selected_max_predicted_relative_delta": clean_value(mechanistic_bound.get("candidate_value")),
        "final_selection_status": clean_value(final_bound.get("status")),
        "outputs": {
            "constraints": rel(constraints_path),
            "decisions": rel(decisions_path),
            "algorithm": rel(algorithm_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    constraints.to_csv(constraints_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    algorithm_path.write_text(
        json.dumps(json_safe(build_algorithm(summary)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, constraints, decisions), encoding="utf-8")
    print(f"Wrote average trust-region bounds to {output_dir.resolve()}")
    print(
        f"Status: {summary['status']}; constraints {summary['constraint_count']}; "
        f"passed/rejected/waiting {summary['passed_count']}/{summary['rejected_count']}/{summary['waiting_count']}"
    )


if __name__ == "__main__":
    main()
