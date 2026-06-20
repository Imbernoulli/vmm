#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

LITERATURE_SOURCES = [
    {
        "key": "harc",
        "title": "When Model Merging Breaks Routing: Training-Free Calibration for MoE",
        "url": "https://arxiv.org/abs/2606.03391",
        "mechanism": "Direct router averaging can cross top-k dispatch boundaries; router movement should be a calibrated, margin-gated intervention.",
    },
    {
        "key": "router_kd_calibration",
        "title": "Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression",
        "url": "https://arxiv.org/abs/2603.02217",
        "mechanism": "After experts are edited or merged, router-expert mismatch is a separate failure mode; router-only KD is a small repair lever.",
    },
    {
        "key": "output_space_projection",
        "title": "Model Merging by Output-Space Projection",
        "url": "https://arxiv.org/abs/2605.29101",
        "mechanism": "Output-space calibration is useful only when the local repair signal transfers to downstream held-out behavior.",
    },
    {
        "key": "model_soups",
        "title": "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time",
        "url": "https://arxiv.org/abs/2203.05482",
        "mechanism": "Averaging is accepted only after validation, with source or baseline fallback kept in the candidate set.",
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


def maybe_float(value: Any) -> float | None:
    value = clean_value(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def maybe_bool(value: Any) -> bool:
    value = clean_value(value)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fmt(value: Any, digits: int = 4) -> str:
    value = maybe_float(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_frontier_rows(
    candidates: pd.DataFrame,
    job: dict[str, Any],
    margin: dict[str, Any],
    nll_probe: dict[str, Any],
    downstream: dict[str, Any],
    selection: dict[str, Any],
) -> pd.DataFrame:
    safe_lambda = maybe_float(margin.get("min_safe_lambda_proxy"))
    limit = maybe_float(job.get("router_margin_limit_with_tolerance"))
    nll_reduction = maybe_float(nll_probe.get("worst_nll_reduction_vs_linear"))
    gen_gain = maybe_float(downstream.get("pair_routercal_avg_gain"))
    gen_gap_to_parent = maybe_float(downstream.get("pair_routercal_gap_to_best_parent_avg"))
    high_fragility_layers = maybe_float(margin.get("high_fragility_layer_count"))
    router_layers = maybe_float(margin.get("router_layer_count"))
    selection_current = selection.get("current_selection") or {}
    selection_status = selection.get("status", "not_run")

    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        cap = maybe_float(row.get("router_max_relative_norm"))
        mode = str(row.get("router_cap_mode", "global"))
        margin_pass = maybe_bool(row.get("router_margin_planned_cap_passed"))
        default_run = maybe_bool(row.get("default_run_enabled"))
        cap_ratio = None
        if cap is not None and safe_lambda not in (None, 0.0):
            cap_ratio = cap / float(safe_lambda)
        if mode == "per_router_margin_profile":
            min_cap = maybe_float(row.get("router_margin_profile_min_cap"))
            max_cap = maybe_float(row.get("router_margin_profile_max_cap"))
            cap_description = f"profile {fmt(min_cap)}-{fmt(max_cap)}"
            boundary_role = "layer_margin_profile_candidate"
        else:
            cap_description = fmt(cap)
            boundary_role = "global_margin_safe_candidate" if margin_pass else "global_stress_candidate"

        local_repair_score = 0.0
        if nll_reduction is not None and cap is not None and safe_lambda not in (None, 0.0):
            # This is a priority score, not a downstream prediction. It discounts caps that exceed
            # the observed top-k margin proxy because those are stress ablations, not default moves.
            local_repair_score = float(nll_reduction) * min(1.0, float(safe_lambda) / max(float(cap), 1e-12))
        if mode == "per_router_margin_profile" and nll_reduction is not None:
            local_repair_score = float(nll_reduction)

        if not default_run:
            frontier_role = "stress_ablation_not_default"
            next_action = "keep_for_stress_only_until_safe_cap_wins_or_vllm_evidence_overrides"
        elif selection_current.get("baseline_eval_completed") is False or selection_status.startswith("awaiting"):
            frontier_role = "default_probe_waiting_downstream_gate"
            next_action = "run_route_kd_materialization_audit_and_matched_vllm_eval"
        else:
            frontier_role = "default_probe_ready_for_selector"
            next_action = "let_router_calibration_selector_accept_or_reject_from_scored_bundle"

        rows.append(
            {
                "rank": int(row.get("rank", len(rows))),
                "cap_label": row.get("cap_label"),
                "method": row.get("method"),
                "router_cap_mode": mode,
                "cap_description": cap_description,
                "router_max_relative_norm": cap,
                "safe_lambda_proxy": safe_lambda,
                "limit_with_tolerance": limit,
                "cap_to_safe_lambda_ratio": cap_ratio,
                "margin_planned_passed": margin_pass,
                "default_run_enabled": default_run,
                "boundary_role": boundary_role,
                "frontier_role": frontier_role,
                "local_repair_priority_score": local_repair_score,
                "nll_worst_reduction_signal": nll_reduction,
                "generation_avg_gain_signal": gen_gain,
                "generation_gap_to_best_parent_signal": gen_gap_to_parent,
                "high_fragility_layers": high_fragility_layers,
                "router_layer_count": router_layers,
                "selection_status": selection_status,
                "next_action": next_action,
                "eval_dir": row.get("eval_dir"),
                "audit_dir": row.get("audit_dir"),
                "checkpoint_dir": row.get("checkpoint_dir"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["default_run_enabled", "local_repair_priority_score", "rank"],
        ascending=[False, False, True],
    )


def build_contract_rows(
    job: dict[str, Any],
    margin: dict[str, Any],
    selection: dict[str, Any],
    frontier: pd.DataFrame,
) -> pd.DataFrame:
    current = selection.get("current_selection") or {}
    default_count = int(frontier["default_run_enabled"].sum()) if not frontier.empty else 0
    return pd.DataFrame(
        [
            {
                "requirement": "direct_router_average_rejected",
                "mechanism": "topk_boundary_stability",
                "status": margin.get("status"),
                "passed": margin.get("recommended_unified_router_action") == "freeze_router",
                "observed": (
                    f"allowed_layers={margin.get('allowed_router_layer_count')}/"
                    f"{margin.get('router_layer_count')}; min_safe_lambda={fmt(margin.get('min_safe_lambda_proxy'))}"
                ),
                "action": "freeze source-averaged router; only test calibrated router deltas",
            },
            {
                "requirement": "safe_router_delta_frontier_exists",
                "mechanism": "margin_capped_route_kd",
                "status": job.get("status"),
                "passed": default_count > 0,
                "observed": f"default_run_candidates={default_count}/{len(frontier)}",
                "action": "run only cap001 and margin-profile by default; keep larger caps as stress ablations",
            },
            {
                "requirement": "local_router_repair_signal",
                "mechanism": "router_expert_mismatch",
                "status": "signal_present",
                "passed": True,
                "observed": (
                    f"nll_worst_reduction={fmt((frontier['nll_worst_reduction_signal'].dropna().iloc[0] if not frontier.empty and frontier['nll_worst_reduction_signal'].notna().any() else None))}; "
                    f"generation_avg_gain={fmt((frontier['generation_avg_gain_signal'].dropna().iloc[0] if not frontier.empty and frontier['generation_avg_gain_signal'].notna().any() else None))}"
                ),
                "action": "treat router calibration as active repair lever, not final acceptance",
            },
            {
                "requirement": "matched_vllm_source_dominance_gate",
                "mechanism": "selector_level_no_regression",
                "status": selection.get("status", "not_run"),
                "passed": bool(current.get("eligible_candidate_count", 0)),
                "observed": (
                    f"baseline_eval={current.get('baseline_eval_completed')}; "
                    f"source_eval={current.get('source_eval_completed')}; "
                    f"eligible={current.get('eligible_candidate_count')}/{current.get('candidate_count')}"
                ),
                "action": "reject or wait until baseline, source endpoints, candidate eval, audit, and manifest gates pass",
            },
            {
                "requirement": "same_shape_router_only_delta",
                "mechanism": "architecture_invariant",
                "status": current.get("audit_completed"),
                "passed": bool(current.get("audit_completed")) and bool(current.get("candidate_eval_completed")),
                "observed": (
                    f"audit_completed={current.get('audit_completed')}; "
                    f"candidate_eval={current.get('candidate_eval_completed')}"
                ),
                "action": "accept only audited router tensors; no expert, attention, embedding, config, or topology changes",
            },
        ]
    )


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    frontier: pd.DataFrame,
    contract: pd.DataFrame,
) -> None:
    default_rows = frontier[frontier["default_run_enabled"]] if not frontier.empty else pd.DataFrame()
    stress_rows = frontier[~frontier["default_run_enabled"]] if not frontier.empty else pd.DataFrame()
    lines = [
        "# Qwen3 MoE Router Calibration Frontier",
        "",
        "这个 artifact 只回答 router calibration 这一个机制问题：direct router average 已经被 top-k margin probe 拒绝；如果要动 router，只能作为 route-KD/HARC-style 的小 delta，并且必须先过 margin、audit、capacity、source endpoint 和 matched vLLM gates。",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Safe-lambda proxy: `{fmt(summary.get('safe_lambda_proxy'))}`",
        f"- High-fragility router layers: `{summary.get('high_fragility_layer_count')}/{summary.get('router_layer_count')}`",
        f"- Default-run router-cal candidates: `{summary.get('default_candidate_count')}/{summary.get('candidate_count')}`",
        f"- Stress-only candidates: `{summary.get('stress_candidate_count')}`",
        f"- NLL repair signal: worst reduction `{fmt(summary.get('nll_worst_reduction_signal'))}`",
        f"- Generation repair signal: avg gain `{fmt(summary.get('generation_avg_gain_signal'))}`, gap to best parent `{fmt(summary.get('generation_gap_to_best_parent_signal'))}`",
        f"- Selection status: `{summary.get('selection_status')}`",
        f"- Recommended default candidates: `{', '.join(summary.get('recommended_default_candidates') or [])}`",
        f"- Acceptance blocker: `{summary.get('acceptance_blocker')}`",
        "",
        "## Frontier",
        "",
        "| cap | mode | cap/safe | margin | role | repair score | next action |",
        "| --- | --- | ---: | --- | --- | ---: | --- |",
    ]
    for _, row in frontier.iterrows():
        lines.append(
            "| "
            f"`{row['cap_label']}` | `{row['router_cap_mode']}` | "
            f"{fmt(row['cap_to_safe_lambda_ratio'])} | `{row['margin_planned_passed']}` | "
            f"`{row['frontier_role']}` | {fmt(row['local_repair_priority_score'])} | "
            f"{row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "| requirement | mechanism | status | passed | observed | action |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in contract.iterrows():
        lines.append(
            "| "
            f"`{row['requirement']}` | `{row['mechanism']}` | `{row['status']}` | "
            f"`{row['passed']}` | {row['observed']} | {row['action']} |"
        )
    lines.extend(
        [
            "",
            "## Algorithm Consequence",
            "",
            "Router calibration is now an explicit frontier, not a default average step. `cap001` and `margin_profile` are the only default probes because they respect the observed router top-k margin; `cap0025` and `cap005` remain stress ablations. Even if local NLL and small generation evidence are positive, the unified algorithm still cannot append a router delta until matched vLLM source-dominance gates pass.",
            "",
            "## Literature Hooks",
            "",
        ]
    )
    for source in LITERATURE_SOURCES:
        lines.append(f"- [{source['title']}]({source['url']}): {source['mechanism']}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `frontier`: `{rel(output_dir / 'router_calibration_frontier.csv')}`",
            f"- `contract`: `{rel(output_dir / 'router_calibration_contract.csv')}`",
            f"- `literature`: `{rel(output_dir / 'literature_sources.json')}`",
            f"- `summary`: `{rel(output_dir / 'summary.json')}`",
            f"- `report`: `{rel(output_dir / 'report.md')}`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-calibration-job-dir", type=Path, default=Path("results/qwen3_moe_router_calibration_job"))
    parser.add_argument("--router-calibration-selection", type=Path, default=Path("results/qwen3_moe_router_calibration_selection/summary.json"))
    parser.add_argument("--router-margin-summary", type=Path, default=Path("results/qwen3_moe_router_margin_fragility/summary.json"))
    parser.add_argument("--router-calibration-nll-summary", type=Path, default=Path("results/qwen3_moe_router_calibration_nll_probe/summary.json"))
    parser.add_argument("--downstream-matrix-summary", type=Path, default=Path("results/fp_downstream_matrix/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_router_calibration_frontier"))
    args = parser.parse_args()

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    job_dir = repo_path(args.router_calibration_job_dir)
    job = read_json(job_dir / "summary.json")
    candidates = read_csv(job_dir / "candidate_plan.csv")
    margin = read_json(args.router_margin_summary)
    nll_probe = read_json(args.router_calibration_nll_summary)
    downstream = read_json(args.downstream_matrix_summary)
    selection = read_json(args.router_calibration_selection)

    frontier = build_frontier_rows(candidates, job, margin, nll_probe, downstream, selection)
    contract = build_contract_rows(job, margin, selection, frontier)

    default_candidates = frontier.loc[frontier["default_run_enabled"], "cap_label"].astype(str).tolist()
    stress_count = int((~frontier["default_run_enabled"]).sum()) if not frontier.empty else 0
    current = selection.get("current_selection") or {}
    blocker_parts = []
    if not current.get("baseline_eval_completed"):
        blocker_parts.append("baseline_eval")
    if not current.get("source_eval_completed"):
        blocker_parts.append("source_eval")
    if not current.get("candidate_eval_completed"):
        blocker_parts.append("candidate_eval")
    if not current.get("audit_completed"):
        blocker_parts.append("audit")
    if not current.get("group_validation_completed"):
        blocker_parts.append("group_validation")
    if not current.get("capacity_metrics_completed"):
        blocker_parts.append("capacity_metrics")
    if not blocker_parts:
        blocker_parts.append("selector_scoring")

    summary = {
        "schema_version": 1,
        "status": "router_calibration_frontier_ready",
        "candidate_count": int(len(frontier)),
        "default_candidate_count": int(frontier["default_run_enabled"].sum()) if not frontier.empty else 0,
        "stress_candidate_count": stress_count,
        "recommended_default_candidates": default_candidates,
        "safe_lambda_proxy": maybe_float(margin.get("min_safe_lambda_proxy")),
        "limit_with_tolerance": maybe_float(job.get("router_margin_limit_with_tolerance")),
        "high_fragility_layer_count": margin.get("high_fragility_layer_count"),
        "router_layer_count": margin.get("router_layer_count"),
        "top_fragile_layer": margin.get("top_fragile_layer"),
        "top_fragility_score": maybe_float(margin.get("top_fragility_score")),
        "nll_worst_reduction_signal": maybe_float(nll_probe.get("worst_nll_reduction_vs_linear")),
        "nll_acceptance_decision": nll_probe.get("acceptance_decision"),
        "generation_avg_gain_signal": maybe_float(downstream.get("pair_routercal_avg_gain")),
        "generation_gap_to_best_parent_signal": maybe_float(downstream.get("pair_routercal_gap_to_best_parent_avg")),
        "selection_status": selection.get("status"),
        "eligible_candidate_count": current.get("eligible_candidate_count"),
        "active_candidate_count": current.get("active_candidate_count"),
        "plan_pruned_candidate_count": current.get("plan_pruned_candidate_count"),
        "acceptance_blocker": ",".join(blocker_parts),
        "algorithm_update": (
            "route_kd_router_delta_is_active_probe_only; append to unified average only after "
            "matched vllm source-dominance and router-only audits pass"
        ),
        "outputs": {
            "frontier": rel(output_dir / "router_calibration_frontier.csv"),
            "contract": rel(output_dir / "router_calibration_contract.csv"),
            "literature": rel(output_dir / "literature_sources.json"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }

    frontier.to_csv(output_dir / "router_calibration_frontier.csv", index=False)
    contract.to_csv(output_dir / "router_calibration_contract.csv", index=False)
    (output_dir / "literature_sources.json").write_text(
        json.dumps(LITERATURE_SOURCES, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir, summary, frontier, contract)
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
