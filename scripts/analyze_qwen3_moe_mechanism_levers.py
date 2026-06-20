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
        "key": "harc_routing_breakdown",
        "title": "When Model Merging Breaks Routing: Training-Free Calibration for MoE",
        "url": "https://arxiv.org/abs/2606.03391",
        "mechanism": "MoE averages can fail through router/top-k dispatch breakdown; router movement needs calibration evidence.",
    },
    {
        "key": "router_kd_calibration",
        "title": "Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression",
        "url": "https://arxiv.org/abs/2603.02217",
        "mechanism": "After expert edits/merges, router-expert mismatch is a distinct failure mode; lightweight router KD can recover routing.",
    },
    {
        "key": "expert_merging_chunking",
        "title": "Expert Merging: Model Merging with Unsupervised Expert Alignment and Importance-Guided Layer Chunking",
        "url": "https://arxiv.org/abs/2509.25712",
        "mechanism": "Layer/chunk coefficient learning on unlabeled calibration data is useful because merge sensitivity is heterogeneous across layers.",
    },
    {
        "key": "sub_moe_subspace",
        "title": "Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging",
        "url": "https://arxiv.org/abs/2506.23266",
        "mechanism": "Expert-output similarity and shared subspaces help separate coherent expert groups from conflicting specializations.",
    },
    {
        "key": "mergeme",
        "title": "MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs",
        "url": "https://arxiv.org/abs/2502.00997",
        "mechanism": "MoE merging needs explicit interference mitigation and routing heuristics beyond unweighted expert averaging.",
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


def maybe_float(value: Any, default: float = 0.0) -> float:
    value = clean_value(value)
    return default if value is None else float(value)


def maybe_int(value: Any, default: int = 0) -> int:
    value = clean_value(value)
    return default if value is None else int(value)


def norm01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo + 1e-12:
        return pd.Series(0.0, index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def find_pair(pairwise: pd.DataFrame, from_candidate: str, to_candidate: str) -> dict[str, Any]:
    if pairwise.empty:
        return {}
    rows = pairwise[
        (pairwise["from_candidate"] == from_candidate)
        & (pairwise["to_candidate"] == to_candidate)
    ]
    return {} if rows.empty else rows.iloc[0].to_dict()


def find_candidate(frontier: pd.DataFrame, candidate: str) -> dict[str, Any]:
    if frontier.empty:
        return {}
    rows = frontier[frontier["candidate"] == candidate]
    return {} if rows.empty else rows.iloc[0].to_dict()


def build_task_gap_policy(task_gap_targets: pd.DataFrame) -> pd.DataFrame:
    if task_gap_targets.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in task_gap_targets.iterrows():
        status = str(row.get("status"))
        task = str(row.get("task"))
        capability = str(row.get("capability"))
        additional_needed = maybe_float(row.get("additional_task_gain_needed"))
        if status == "no_task_frontier_gain":
            hypothesis = "the current source set has no measurable endpoint frontier signal on this task"
            average_policy = "block_final_average_budget_and_search_or_eval_a_stronger_task_source"
            probe = "host specialist endpoints, score the same task manifest, then recompute source surplus before changing MoE weights"
            moe_lever = "source_frontier_first"
            priority_bonus = 0.08
        elif status == "gain_below_interference_budget":
            hypothesis = "the task has a positive source frontier signal, but it is smaller than observed merge interference"
            average_policy = "expand_endpoint_eval_and_only_then_try_output_or_layer_chunk_alignment"
            probe = "increase task examples and compare source frontier against the observed interference budget"
            moe_lever = "output_or_layer_chunk_alignment_after_positive_source_signal"
            priority_bonus = 0.04
        elif status == "observed_merge_loses_frontier":
            hypothesis = "the average is actively regressing a task where sources already contain usable signal"
            average_policy = "repair_task_regression_with_router_expert_or_subspace_caps_before_acceptance"
            probe = "paired vLLM regression audit plus router/expert attribution for the regressed task"
            moe_lever = "regression_repair"
            priority_bonus = 0.06
        else:
            hypothesis = "the task is not currently a blocker, but it remains part of the locked source frontier gate"
            average_policy = "keep_as_guardrail_for_future_average_acceptance"
            probe = "keep task in paired source/candidate eval"
            moe_lever = "guardrail"
            priority_bonus = 0.0
        rows.append(
            {
                "task": task,
                "runner_task": row.get("runner_task"),
                "capability": capability,
                "frontier_source": row.get("frontier_source"),
                "frontier_gain_vs_best_single": maybe_float(row.get("frontier_gain_vs_best_single")),
                "interference_budget": maybe_float(row.get("interference_budget")),
                "additional_task_gain_needed": additional_needed,
                "status": status,
                "mechanism_hypothesis": hypothesis,
                "average_policy": average_policy,
                "recommended_probe": probe,
                "preferred_moe_lever": moe_lever,
                "priority_score": additional_needed + priority_bonus,
            }
        )
    return pd.DataFrame(rows).sort_values(["priority_score", "task"], ascending=[False, True]).reset_index(drop=True)


def build_layer_chunking_plan(
    layer_frontier: pd.DataFrame,
    router_layers: pd.DataFrame,
    expert_geometry_layers: pd.DataFrame,
) -> pd.DataFrame:
    if layer_frontier.empty:
        return pd.DataFrame()
    layers = layer_frontier.copy()
    if not router_layers.empty:
        layers = layers.merge(router_layers, left_on="layer", right_on="layer", how="left")
    if not expert_geometry_layers.empty:
        geometry = expert_geometry_layers.rename(columns={"layer_id": "layer"}).copy()
        keep = [
            "layer",
            "route_mass_weighted_route_geometry_risk_score",
            "route_mass_weighted_internal_geometry_risk_score",
            "high_route_geometry_risk_experts",
            "p95_combined_relative_delta",
            "min_combined_cosine",
        ]
        layers = layers.merge(geometry[[column for column in keep if column in geometry]], on="layer", how="left")
    for column in ["calibrate_rows", "probe_rows", "router_relative_delta_norm", "min_topk_jaccard"]:
        if column not in layers:
            layers[column] = 0.0
    geometry_columns = [
        "route_mass_weighted_route_geometry_risk_score",
        "route_mass_weighted_internal_geometry_risk_score",
        "high_route_geometry_risk_experts",
        "p95_combined_relative_delta",
        "min_combined_cosine",
    ]
    for column in geometry_columns:
        if column not in layers:
            layers[column] = 0.0
        layers[column] = pd.to_numeric(layers[column], errors="coerce").fillna(0.0)
    geometry_available = bool(
        "route_mass_weighted_route_geometry_risk_score" in layers
        and float(layers["route_mass_weighted_route_geometry_risk_score"].max()) > 0.0
    )
    layers["calibrate_fraction"] = (
        pd.to_numeric(layers["calibrate_rows"], errors="coerce").fillna(0.0)
        / pd.to_numeric(layers["probe_rows"], errors="coerce").replace(0, 1).fillna(1.0)
    )
    if geometry_available:
        geometry_pressure = (
            0.50 * norm01(layers["route_mass_weighted_route_geometry_risk_score"])
            + 0.25 * norm01(layers["high_route_geometry_risk_experts"])
            + 0.15 * norm01(layers["p95_combined_relative_delta"])
            + 0.10 * norm01(1.0 - layers["min_combined_cosine"])
        )
        layers["layer_importance_score"] = (
            0.24 * norm01(layers["route_to_trust_reduction"])
            + 0.20 * norm01(layers["trust_region_relative_delta_norm"])
            + 0.18 * norm01(layers["calibrate_fraction"])
            + 0.12 * norm01(layers["router_relative_delta_norm"])
            + 0.10 * norm01(1.0 - pd.to_numeric(layers["min_topk_jaccard"], errors="coerce").fillna(1.0))
            + 0.16 * geometry_pressure
        )
    else:
        layers["layer_importance_score"] = (
            0.30 * norm01(layers["route_to_trust_reduction"])
            + 0.25 * norm01(layers["trust_region_relative_delta_norm"])
            + 0.20 * norm01(layers["calibrate_fraction"])
            + 0.15 * norm01(layers["router_relative_delta_norm"])
            + 0.10 * norm01(1.0 - pd.to_numeric(layers["min_topk_jaccard"], errors="coerce").fillna(1.0))
        )
    ranked = layers.sort_values("layer_importance_score", ascending=False).reset_index(drop=True)
    policies = []
    for idx, row in ranked.iterrows():
        if idx < 8:
            policy = "per_layer_coefficients"
            coefficient_slots = 4
            reason = "highest expert-delta/router/internal-geometry sensitivity; allocate fine calibration coefficients"
        elif idx < 24:
            policy = "two_layer_chunk_coefficients"
            coefficient_slots = 2
            reason = "medium sensitivity; share coefficients across small adjacent chunks"
        else:
            policy = "coarse_shared_coefficients"
            coefficient_slots = 1
            reason = "lower sensitivity; keep coefficient count small"
        policies.append((policy, coefficient_slots, reason))
    ranked["chunk_policy"] = [item[0] for item in policies]
    ranked["recommended_coefficient_slots"] = [item[1] for item in policies]
    ranked["chunk_reason"] = [item[2] for item in policies]
    columns = [
        "layer",
        "layer_importance_score",
        "chunk_policy",
        "recommended_coefficient_slots",
        "route_guarded_relative_delta_norm",
        "trust_region_relative_delta_norm",
        "route_to_trust_reduction",
        "trust_to_expert_only_reduction",
        "calibrate_fraction",
        "router_relative_delta_norm",
        "min_topk_jaccard",
        "min_top1_agreement",
        "route_mass_weighted_route_geometry_risk_score",
        "high_route_geometry_risk_experts",
        "p95_combined_relative_delta",
        "min_combined_cosine",
        "chunk_reason",
    ]
    return ranked[columns]


def build_levers(
    *,
    delta_summary: dict[str, Any],
    pairwise: pd.DataFrame,
    candidate_frontier: pd.DataFrame,
    cap_summary: dict[str, Any],
    risk_ablation: pd.DataFrame,
    router_summary: dict[str, Any],
    eval_budget_summary: dict[str, Any],
    final_selection_summary: dict[str, Any],
    chunking_plan: pd.DataFrame,
    subspace_summary: dict[str, Any],
    qwen_source_discovery_summary: dict[str, Any],
    average_source_set_optimizer: dict[str, Any],
    task_gap_policy: pd.DataFrame,
) -> pd.DataFrame:
    route_to_audit = find_pair(pairwise, "route_guarded", "audit_gated")
    audit_to_trust = find_pair(pairwise, "audit_gated", "trust_region")
    trust_to_expert = find_pair(pairwise, "trust_region", "expert_only")
    expert_to_tail = find_pair(pairwise, "expert_only", "tail_trimmed")
    tail_to_searched = find_pair(pairwise, "tail_trimmed", "searched_no_gt065")
    tail = find_candidate(candidate_frontier, "tail_trimmed")
    searched = find_candidate(candidate_frontier, "searched_no_gt065")
    risk_flag_reductions = 0
    risk_flag_retention_loss = 0.0
    if not risk_ablation.empty:
        risk_flag_reductions = int(
            pd.to_numeric(risk_ablation["routed_gt_075_reduction"], errors="coerce").fillna(0).sum()
            + pd.to_numeric(risk_ablation["routed_gt_065_reduction"], errors="coerce").fillna(0).sum()
        )
        risk_flag_retention_loss = float(
            pd.to_numeric(risk_ablation["nonbase_mass_retention_loss"], errors="coerce").fillna(0).sum()
        )
    top_chunks = ",".join(
        str(int(row["layer"])) for _, row in chunking_plan.head(8).sort_values("layer").iterrows()
    ) if not chunking_plan.empty else ""
    top_geometry_layers = ",".join(
        str(int(row["layer"]))
        for _, row in chunking_plan.sort_values(
            "route_mass_weighted_route_geometry_risk_score",
            ascending=False,
        )
        .head(8)
        .sort_values("layer")
        .iterrows()
    ) if not chunking_plan.empty and "route_mass_weighted_route_geometry_risk_score" in chunking_plan else ""
    subspace_available = bool(subspace_summary)
    subspace_quantitative_evidence = (
        f"projection tensors {subspace_summary.get('projection_tensor_count')}; "
        f"high-conflict experts {subspace_summary.get('high_subspace_conflict_expert_count')}; "
        f"route-important high-conflict experts "
        f"{subspace_summary.get('route_important_high_subspace_conflict_expert_count')}; "
        f"extra-scaled experts {subspace_summary.get('subspace_extra_scaled_expert_count')}; "
        f"top layer L{subspace_summary.get('top_subspace_conflict_layer')}"
    ) if subspace_available else "Qwen3 identity gate passes, but no expert-output subspace clustering artifact is tracked for candidate generation."
    subspace_current_action = (
        "use subspace-conflict score as a pre-materialization gate; keep current unified caps for covered experts and reserve a subspace-scaled ablation for the 17 uncovered experts"
        if subspace_available
        else "keep identity as required preflight; add expert-output/subspace probe before averaging unrelated downstream MoE fine-tunes"
    )
    subspace_next_test = (
        "materialize/evaluate the subspace-scaled candidate only after the source/candidate budgeted vLLM gate, or use the gate before third-party MoE merges"
        if subspace_available
        else "collect per-expert output embeddings and cluster/subspace similarity before materializing third-party MoE merges"
    )
    top_source_set = average_source_set_optimizer.get("top_source_set") or {}
    top_task_gap = task_gap_policy.iloc[0].to_dict() if not task_gap_policy.empty else {}
    task_gap_text = "; ".join(
        f"{row['task']}:{row['status']} needs {maybe_float(row['additional_task_gain_needed']):.4f}"
        for _, row in task_gap_policy.iterrows()
    )
    measured_queue = qwen_source_discovery_summary.get("top_queue_item") or {}
    levers = [
        {
            "mechanism": "source_task_gap_frontier_acquisition",
            "evidence_type": "task_surplus_gate",
            "quantitative_evidence": (
                f"top source set {top_source_set.get('source_set')}; surplus "
                f"{top_source_set.get('frontier_avg_surplus_vs_interference')}; "
                f"task blockers {task_gap_text or 'none'}"
            ),
            "inferred_failure_mode": (
                "the current sources do not yet contain enough task-frontier complementarity to justify "
                "spending final average budget on MoE weight optimization"
            ),
            "current_action": (
                "prioritize task-source acquisition/eval; do not promote any average until "
                "task-level source surplus covers observed merge interference"
            ),
            "next_test": measured_queue.get(
                "command",
                "python scripts/build_qwen_source_discovery_plan.py --output-dir results/qwen_source_discovery_plan",
            ),
            "priority_score": 0.995 if not task_gap_policy.empty else 0.58,
            "confidence": "high" if not task_gap_policy.empty else "medium",
            "source_artifacts": (
                "results/qwen_source_discovery_plan/task_gap_targets.csv; "
                "results/qwen3_average_source_set_optimizer/task_surplus.csv"
            ),
        },
        {
            "mechanism": "source_and_candidate_downstream_eval",
            "evidence_type": "statistical_eval_budget",
            "quantitative_evidence": (
                f"examples {eval_budget_summary.get('current_gate_examples')} -> "
                f"{eval_budget_summary.get('recommended_max_examples')}; extra prompts "
                f"{eval_budget_summary.get('total_additional_prompt_budget')}"
            ),
            "inferred_failure_mode": "64-example smoke cannot prove source dominance, task regression, or paired prediction loss.",
            "current_action": "run budgeted one-model-at-a-time vLLM eval before accepting any average",
            "next_test": "results/qwen3_moe_eval_budget_plan/run_eval_budget.sh final",
            "priority_score": 0.98,
            "confidence": "high",
            "source_artifacts": "results/qwen3_moe_eval_budget_plan/report.md",
        },
        {
            "mechanism": "router_direct_movement",
            "evidence_type": "router_gate",
            "quantitative_evidence": (
                f"allowed layers {router_summary.get('allowed_router_layer_count')}/"
                f"{router_summary.get('router_layer_count')}; min top1 "
                f"{router_summary.get('min_top1_agreement')}; router rel-norm "
                f"{router_summary.get('total_router_relative_delta_norm')}"
            ),
            "inferred_failure_mode": "shared router tensors have category-specific unsafe slices, so direct router averaging can misroute tokens.",
            "current_action": "freeze router for same-shape candidate; only consider calibrated router deltas",
            "next_test": "route-KD/HARC-style router calibration after frozen-router baseline and sources finish vLLM eval",
            "priority_score": 0.94,
            "confidence": "high",
            "source_artifacts": "results/qwen3_moe_router_move_gate/report.md",
        },
        {
            "mechanism": "routed_expert_tail_cap_0_75",
            "evidence_type": "delta_frontier",
            "quantitative_evidence": (
                f"route->audit removes >0.75 by {route_to_audit.get('routed_gt_075_reduction')} and >1.0 by "
                f"{route_to_audit.get('routed_gt_1_reduction')}"
            ),
            "inferred_failure_mode": "a few routed experts take oversized Coder deltas and dominate the unsafe tail.",
            "current_action": "keep file-level/audit-level relative-delta cap as a mandatory safety gate",
            "next_test": "compare route_guarded vs audit_gated under budgeted vLLM eval",
            "priority_score": 0.86,
            "confidence": "high",
            "source_artifacts": "results/qwen3_moe_delta_frontier/report.md",
        },
        {
            "mechanism": "route_load_trust_region",
            "evidence_type": "delta_frontier",
            "quantitative_evidence": (
                f"audit->trust removes >0.75 by {audit_to_trust.get('routed_gt_075_reduction')} and >0.65 by "
                f"{audit_to_trust.get('routed_gt_065_reduction')}"
            ),
            "inferred_failure_mode": "route/load/category signals mainly lower the remaining high-delta expert tail.",
            "current_action": "keep trust-region as an ablation, but do not assume its extra risk flags improve utility",
            "next_test": "compare audit_gated vs trust_region and inspect paired task regressions",
            "priority_score": 0.78,
            "confidence": "medium_high",
            "source_artifacts": "results/qwen3_moe_delta_frontier/report.md",
        },
        {
            "mechanism": "shared_attention_delta",
            "evidence_type": "attention_ablation",
            "quantitative_evidence": (
                f"trust->expert-only removes attention relative norm {trust_to_expert.get('attention_relative_delta_norm_reduction')} "
                f"with routed-tail reduction {trust_to_expert.get('routed_gt_075_reduction')}"
            ),
            "inferred_failure_mode": "attention movement is not routed; norm-only probes cannot decide if it carries useful ability.",
            "current_action": "freeze attention in current unified candidate, but keep trust_region vs expert_only eval as the utility test",
            "next_test": "budgeted paired vLLM comparison of trust_region and expert_only",
            "priority_score": 0.74,
            "confidence": "medium",
            "source_artifacts": "results/qwen3_moe_delta_frontier/report.md",
        },
        {
            "mechanism": "tail_cap_0_65",
            "evidence_type": "delta_frontier",
            "quantitative_evidence": (
                f"expert_only->tail_trimmed removes >0.75 by {expert_to_tail.get('routed_gt_075_reduction')} and >0.65 by "
                f"{expert_to_tail.get('routed_gt_065_reduction')}; tail max rel {tail.get('routed_max_tensor_relative_delta')}"
            ),
            "inferred_failure_mode": "after attention freeze, a small residual routed expert tail remains above the safer cap.",
            "current_action": "evaluate tail-trimmed as the conservative expert-only candidate",
            "next_test": "budgeted paired vLLM comparison of expert_only vs tail_trimmed",
            "priority_score": 0.80,
            "confidence": "medium_high",
            "source_artifacts": "results/qwen3_moe_delta_frontier/report.md",
        },
        {
            "mechanism": "risk_penalty_complexity",
            "evidence_type": "cap_law_search",
            "quantitative_evidence": (
                f"risk flag ablation tail reductions {risk_flag_reductions}; summed retention loss {risk_flag_retention_loss:.6g}; "
                f"searched no-gt-0.65 rel norm {searched.get('total_relative_delta_norm')}"
            ),
            "inferred_failure_mode": "hand-built risk penalties can reduce retention without reducing the relevant tail threshold.",
            "current_action": "prefer the simpler uniform 0.65 cap unless downstream eval proves risk penalties preserve task behavior",
            "next_test": "budgeted paired vLLM comparison of tail_trimmed vs searched_no_gt065 vs layer_chunk vs unified mechanism",
            "priority_score": 0.83,
            "confidence": "medium_high",
            "source_artifacts": "results/qwen3_moe_trust_region_cap_search/report.md",
        },
        {
            "mechanism": "importance_guided_layer_chunking",
            "evidence_type": "layer_delta_router_geometry_join",
            "quantitative_evidence": (
                f"top fine-calibration layers: {top_chunks}; top expert-geometry layers: {top_geometry_layers}"
            ),
            "inferred_failure_mode": "a single global coefficient ignores layer heterogeneity in expert delta, router sensitivity, and internal expert geometry.",
            "current_action": "use high-sensitivity layers for future unlabeled coefficient calibration; keep low-sensitivity layers coarse",
            "next_test": "learn per-layer or chunk coefficients on hidden/logit calibration cache under same-shape writer rules",
            "priority_score": 0.72,
            "confidence": "medium",
            "source_artifacts": (
                "results/qwen3_moe_mechanism_levers/layer_chunking_plan.csv; "
                "results/qwen3_moe_expert_geometry_probe/report.md"
            ),
        },
        {
            "mechanism": "expert_identity_and_subspace_probe",
            "evidence_type": "subspace_conflict_probe" if subspace_available else "literature_gap",
            "quantitative_evidence": subspace_quantitative_evidence,
            "inferred_failure_mode": "same-index experts may be shape-compatible while still hiding localized channel/chunk subspace conflicts.",
            "current_action": subspace_current_action,
            "next_test": subspace_next_test,
            "priority_score": 0.73 if subspace_available else 0.66,
            "confidence": "medium",
            "source_artifacts": (
                "results/qwen3_moe_expert_subspace_conflict_probe/report.md; "
                "results/qwen3_moe_expert_geometry_probe/report.md"
            ) if subspace_available else "results/fp_moe_real_probe/report.md; results/moe_probe_gated_selector/report.md",
        },
    ]
    final_status = (final_selection_summary.get("current_selection") or {}).get("status")
    if final_status not in {None, "awaiting_source_eval", "awaiting_candidate_eval"}:
        eval_index = 1 if len(levers) > 1 else 0
        levers[eval_index]["priority_score"] = 0.60
        levers[eval_index]["quantitative_evidence"] += f"; final selector status {final_status}"
    if not task_gap_policy.empty and top_task_gap:
        levers[0]["quantitative_evidence"] += (
            f"; top task gap {top_task_gap.get('task')} "
            f"{top_task_gap.get('status')} needs "
            f"{maybe_float(top_task_gap.get('additional_task_gain_needed')):.4f}"
        )
    return pd.DataFrame(levers).sort_values("priority_score", ascending=False).reset_index(drop=True)


def build_experiment_queue(levers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rank, (_, row) in enumerate(levers.iterrows(), start=1):
        if rank > 6:
            break
        rows.append(
            {
                "rank": rank,
                "mechanism": row["mechanism"],
                "action": row["current_action"],
                "test_or_command": row["next_test"],
                "priority_score": row["priority_score"],
                "confidence": row["confidence"],
            }
        )
    return pd.DataFrame(rows)


def fmt(value: Any) -> str:
    value = clean_value(value)
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_report(
    summary: dict[str, Any],
    levers: pd.DataFrame,
    queue: pd.DataFrame,
    chunking: pd.DataFrame,
    task_gap_policy: pd.DataFrame,
) -> str:
    lines = [
        "# Qwen3 MoE Mechanism Leverage Map",
        "",
        "这个 artifact 把现有 Qwen3 MoE probe 结果压成一个优化优先级表：不是问哪个算法名更好，而是问哪个机制最可能解释平均失败，以及下一步应该用什么实验验证。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Lever count: `{summary['lever_count']}`",
        f"- Top lever: `{summary['top_lever']}`",
        f"- Top next test: `{summary['top_lever_next_test']}`",
        f"- Task-gap blockers: `{summary['task_gap_blocker_count']}`",
        f"- Top task gap: `{summary['top_task_gap_task']}` / `{summary['top_task_gap_status']}` needs `{fmt(summary['top_task_gap_additional_gain_needed'])}`",
        f"- Fine calibration layers: `{summary['fine_calibration_layers']}`",
        "",
        "## Levers",
        "",
        "| mechanism | priority | confidence | evidence | action |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for _, row in levers.iterrows():
        lines.append(
            f"| `{row['mechanism']}` | {fmt(row['priority_score'])} | `{row['confidence']}` | "
            f"{row['quantitative_evidence']} | {row['current_action']} |"
        )
    lines.extend(
        [
            "",
            "## Next Experiments",
            "",
            "| rank | mechanism | test or command |",
            "| ---: | --- | --- |",
        ]
    )
    for _, row in queue.iterrows():
        lines.append(f"| {int(row['rank'])} | `{row['mechanism']}` | `{row['test_or_command']}` |")
    if not task_gap_policy.empty:
        lines.extend(
            [
                "",
                "## Task-Gap Mechanism Policy",
                "",
                "这个表把 source surplus 的 task-level blocker 翻译成 average policy：source frontier 没有通过时，不把失败归因给某个 averaging 算法超参，而是先证明任务源互补性。",
                "",
                "| task | capability | status | gain | needed | average policy | probe |",
                "| --- | --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        for _, row in task_gap_policy.iterrows():
            lines.append(
                f"| `{row['task']}` | `{row['capability']}` | `{row['status']}` | "
                f"{fmt(row['frontier_gain_vs_best_single'])} | "
                f"{fmt(row['additional_task_gain_needed'])} | "
                f"{row['average_policy']} | {row['recommended_probe']} |"
            )
    if not chunking.empty:
        lines.extend(
            [
                "",
                "## Layer/Chunk Calibration Plan",
                "",
                "这个表把 Expert Merging/importance-guided chunking 的思想落到当前 Qwen3 数据上：高敏感层给更多校准系数，低敏感层共享粗粒度系数。",
                "",
                "| layer | score | policy | slots | route->trust | calibrate frac | router rel | geometry risk | high-geom experts |",
                "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in chunking.head(16).iterrows():
            lines.append(
                f"| {int(row['layer'])} | {fmt(row['layer_importance_score'])} | "
                f"`{row['chunk_policy']}` | {int(row['recommended_coefficient_slots'])} | "
                f"{fmt(row['route_to_trust_reduction'])} | {fmt(row['calibrate_fraction'])} | "
                f"{fmt(row['router_relative_delta_norm'])} | "
                f"{fmt(row.get('route_mass_weighted_route_geometry_risk_score'))} | "
                f"{fmt(row.get('high_route_geometry_risk_experts'))} |"
            )
    lines.extend(
        [
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
        ]
    )
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    levers: pd.DataFrame,
    queue: pd.DataFrame,
    chunking: pd.DataFrame,
    subspace_summary: dict[str, Any],
    task_gap_policy: pd.DataFrame,
) -> dict[str, Any]:
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    levers_path = output_dir / "mechanism_levers.csv"
    queue_path = output_dir / "next_experiment_queue.csv"
    chunking_path = output_dir / "layer_chunking_plan.csv"
    task_gap_path = output_dir / "task_gap_policy.csv"
    literature_path = output_dir / "literature_sources.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    levers.to_csv(levers_path, index=False)
    queue.to_csv(queue_path, index=False)
    chunking.to_csv(chunking_path, index=False)
    task_gap_policy.to_csv(task_gap_path, index=False)
    literature_path.write_text(
        json.dumps(json_safe(LITERATURE_SOURCES), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fine_layers = (
        ",".join(str(int(row["layer"])) for _, row in chunking[chunking["chunk_policy"] == "per_layer_coefficients"].sort_values("layer").iterrows())
        if not chunking.empty
        else ""
    )
    top = levers.iloc[0].to_dict() if not levers.empty else {}
    top_task_gap = task_gap_policy.iloc[0].to_dict() if not task_gap_policy.empty else {}
    geometry_available = (
        not chunking.empty
        and "route_mass_weighted_route_geometry_risk_score" in chunking
        and float(chunking["route_mass_weighted_route_geometry_risk_score"].max()) > 0.0
    )
    top_geometry = (
        chunking.sort_values("route_mass_weighted_route_geometry_risk_score", ascending=False).iloc[0].to_dict()
        if geometry_available
        else {}
    )
    summary = {
        "schema_version": 1,
        "status": "mechanism_leverage_map_ready",
        "lever_count": int(len(levers)),
        "top_lever": top.get("mechanism"),
        "top_lever_priority": maybe_float(top.get("priority_score")),
        "top_lever_next_test": top.get("next_test"),
        "task_gap_blocker_count": int(len(task_gap_policy)),
        "task_gap_tasks": ",".join(str(row["task"]) for _, row in task_gap_policy.iterrows()),
        "top_task_gap_task": top_task_gap.get("task"),
        "top_task_gap_status": top_task_gap.get("status"),
        "top_task_gap_capability": top_task_gap.get("capability"),
        "top_task_gap_additional_gain_needed": maybe_float(
            top_task_gap.get("additional_task_gain_needed"),
            default=0.0,
        )
        if top_task_gap
        else None,
        "top_task_gap_average_policy": top_task_gap.get("average_policy"),
        "fine_calibration_layers": fine_layers,
        "expert_geometry_probe_used": geometry_available,
        "expert_subspace_probe_used": bool(subspace_summary),
        "high_subspace_conflict_expert_count": maybe_int(
            subspace_summary.get("high_subspace_conflict_expert_count")
        )
        if subspace_summary
        else None,
        "subspace_extra_scaled_expert_count": maybe_int(
            subspace_summary.get("subspace_extra_scaled_expert_count")
        )
        if subspace_summary
        else None,
        "top_subspace_conflict_layer": maybe_int(subspace_summary.get("top_subspace_conflict_layer"))
        if subspace_summary
        else None,
        "top_expert_geometry_layer": maybe_int(top_geometry.get("layer")) if top_geometry else None,
        "top_expert_geometry_layer_risk": maybe_float(
            top_geometry.get("route_mass_weighted_route_geometry_risk_score")
        )
        if top_geometry
        else None,
        "queue_count": int(len(queue)),
        "literature_count": int(len(LITERATURE_SOURCES)),
        "outputs": {
            "report": rel(report_path),
            "summary": rel(summary_path),
            "mechanism_levers": rel(levers_path),
            "next_experiment_queue": rel(queue_path),
            "layer_chunking_plan": rel(chunking_path),
            "task_gap_policy": rel(task_gap_path),
            "literature_sources": rel(literature_path),
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, levers, queue, chunking, task_gap_policy), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Qwen3 MoE averaging mechanisms and next experiments.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_mechanism_levers"))
    parser.add_argument("--delta-frontier-dir", type=Path, default=Path("results/qwen3_moe_delta_frontier"))
    parser.add_argument("--cap-search-dir", type=Path, default=Path("results/qwen3_moe_trust_region_cap_search"))
    parser.add_argument("--router-gate-dir", type=Path, default=Path("results/qwen3_moe_router_move_gate"))
    parser.add_argument("--eval-budget-dir", type=Path, default=Path("results/qwen3_moe_eval_budget_plan"))
    parser.add_argument("--final-selection-dir", type=Path, default=Path("results/qwen3_moe_final_candidate_selection"))
    parser.add_argument(
        "--expert-geometry-layer",
        type=Path,
        default=Path("results/qwen3_moe_expert_geometry_probe/layer_geometry.csv"),
    )
    parser.add_argument(
        "--expert-subspace-dir",
        type=Path,
        default=Path("results/qwen3_moe_expert_subspace_conflict_probe"),
    )
    parser.add_argument(
        "--qwen-source-discovery-plan",
        type=Path,
        default=Path("results/qwen_source_discovery_plan/summary.json"),
    )
    parser.add_argument(
        "--qwen-source-task-gap-targets",
        type=Path,
        default=Path("results/qwen_source_discovery_plan/task_gap_targets.csv"),
    )
    parser.add_argument(
        "--average-source-set-optimizer",
        type=Path,
        default=Path("results/qwen3_average_source_set_optimizer/summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    delta_dir = repo_path(args.delta_frontier_dir)
    cap_dir = repo_path(args.cap_search_dir)
    router_dir = repo_path(args.router_gate_dir)
    eval_budget_dir = repo_path(args.eval_budget_dir)
    final_dir = repo_path(args.final_selection_dir)
    subspace_dir = repo_path(args.expert_subspace_dir)
    delta_summary = read_json(delta_dir / "summary.json")
    pairwise = read_csv(delta_dir / "pairwise_delta_reductions.csv")
    candidate_frontier = read_csv(delta_dir / "candidate_delta_frontier.csv")
    layer_frontier = read_csv(delta_dir / "layer_delta_frontier.csv")
    cap_summary = read_json(cap_dir / "summary.json")
    risk_ablation = read_csv(cap_dir / "risk_flag_ablation.csv")
    router_summary = read_json(router_dir / "summary.json")
    router_layers = read_csv(router_dir / "router_layer_move_gate.csv")
    expert_geometry_layers = read_csv(args.expert_geometry_layer)
    eval_budget_summary = read_json(eval_budget_dir / "summary.json")
    final_selection_summary = read_json(final_dir / "summary.json")
    subspace_summary = read_json(subspace_dir / "summary.json")
    qwen_source_discovery_summary = read_json(args.qwen_source_discovery_plan)
    average_source_set_optimizer = read_json(args.average_source_set_optimizer)
    task_gap_targets = read_csv(args.qwen_source_task_gap_targets)
    task_gap_policy = build_task_gap_policy(task_gap_targets)
    chunking = build_layer_chunking_plan(layer_frontier, router_layers, expert_geometry_layers)
    levers = build_levers(
        delta_summary=delta_summary,
        pairwise=pairwise,
        candidate_frontier=candidate_frontier,
        cap_summary=cap_summary,
        risk_ablation=risk_ablation,
        router_summary=router_summary,
        eval_budget_summary=eval_budget_summary,
        final_selection_summary=final_selection_summary,
        chunking_plan=chunking,
        subspace_summary=subspace_summary,
        qwen_source_discovery_summary=qwen_source_discovery_summary,
        average_source_set_optimizer=average_source_set_optimizer,
        task_gap_policy=task_gap_policy,
    )
    queue = build_experiment_queue(levers)
    summary = write_outputs(args.output_dir, levers, queue, chunking, subspace_summary, task_gap_policy)
    print(f"Wrote Qwen3 MoE mechanism leverage map to {repo_path(args.output_dir).resolve()}")
    print(f"Top lever: {summary['top_lever']} ({summary['top_lever_priority']:.3f})")


if __name__ == "__main__":
    main()
