#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_JOB_DIR = Path("results/qwen3_moe_router_calibration_job")
DEFAULT_OUTPUT_DIR = Path("results/qwen3_moe_router_calibration_selection")
DEFAULT_BASELINE_EVAL_DIR = Path("results/vllm_checkpoint_eval/qwen3_moe_searched_no_gt065_max_retention_candidate")
DEFAULT_BASELINE_AUDIT_DIR = Path("results/qwen3_moe_searched_no_gt065_delta_audit")
DEFAULT_SOURCE_EVAL_DIRS = [
    Path("results/vllm_checkpoint_eval/source_qwen3_30b_instruct"),
    Path("results/vllm_checkpoint_eval/source_qwen3_30b_coder"),
]

TASK_SCORE_COLUMNS = {
    "gsm8k": "task_gsm8k_score",
    "mmlu": "task_mmlu_score",
    "safety": "task_safety_score",
    "humaneval_compile": "task_humaneval_compile_score",
}
SCORE_COLUMNS = ["avg_primary_score", "worst_primary_score", *TASK_SCORE_COLUMNS.values()]

LITERATURE_HOOKS = [
    {
        "key": "model_soups",
        "title": "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time",
        "url": "https://arxiv.org/abs/2203.05482",
        "mechanism": "Weight averaging is only justified when checkpoints behave as if they are in one low-error basin; the selector therefore rejects router deltas that do not improve downstream scores over the frozen-router baseline.",
    },
    {
        "key": "fisher_merging",
        "title": "Merging Models with Fisher-Weighted Averaging",
        "url": "https://arxiv.org/abs/2111.09832",
        "mechanism": "A local quadratic view motivates small trust-region updates, but the actual acceptance criterion is still held-out downstream behavior.",
    },
    {
        "key": "ties",
        "title": "TIES-Merging: Resolving Interference When Merging Models",
        "url": "https://arxiv.org/abs/2306.01708",
        "mechanism": "Interference is treated as a measurable signal: the router calibration delta must be sparse in module scope and must not introduce non-router changes.",
    },
    {
        "key": "git_rebasin",
        "title": "Git Re-Basin: Merging Models modulo Permutation Symmetries",
        "url": "https://arxiv.org/abs/2209.04836",
        "mechanism": "Expert identity and permutation alignment remain upstream gates; this selector only decides whether a small router delta should be added after the frozen-router expert candidate.",
    },
    {
        "key": "large_scale_merging",
        "title": "What Matters for Model Merging at Scale?",
        "url": "https://arxiv.org/abs/2410.03617",
        "mechanism": "Large-model merging can work, but endpoint controls are required; router-calibrated candidates are rejected if dominated by source endpoints.",
    },
    {
        "key": "output_space_projection",
        "title": "Model Merging by Output-Space Projection",
        "url": "https://arxiv.org/abs/2605.29101",
        "mechanism": "The route-KD cache is an output-space calibration signal for routers; this script uses downstream scores to decide whether that local calibration transfers to the full model.",
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


def maybe_float(value: Any) -> float | None:
    value = clean_value(value)
    return None if value is None else float(value)


def maybe_int(value: Any) -> int | None:
    value = clean_value(value)
    return None if value is None else int(value)


def read_json_if_exists(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def primary_metric(row: pd.Series | dict[str, Any]) -> tuple[str, float] | None:
    for key in ("strict_exact", "accuracy", "policy_accuracy", "compile_rate"):
        value = maybe_float(row.get(key))
        if value is not None:
            return key, value
    return None


def read_eval_state(eval_dir: str | Path) -> dict[str, Any]:
    root = repo_path(eval_dir)
    summary = read_json_if_exists(root / "summary.json")
    model_summary = read_csv_if_exists(root / "model_summary.csv")
    metrics = read_csv_if_exists(root / "metrics.csv")

    status = str(summary.get("status", "not_run"))
    if not model_summary.empty:
        model_row = model_summary.iloc[0].to_dict()
    else:
        rows = summary.get("model_summary") or []
        model_row = rows[0] if rows else {}

    task_scores: dict[str, float] = {}
    if not metrics.empty:
        for _, metric_row in metrics.iterrows():
            task = str(metric_row.get("task", "unknown"))
            primary = primary_metric(metric_row)
            if primary is not None:
                _, score = primary
                task_scores[task] = score

    state: dict[str, Any] = {
        "eval_dir": rel(root),
        "eval_exists": bool(root.exists()),
        "eval_status": status,
        "eval_completed": status == "complete",
        "model": clean_value(model_row.get("model")),
        "avg_primary_score": maybe_float(model_row.get("avg_primary_score")),
        "worst_primary_score": maybe_float(model_row.get("worst_primary_score")),
        "task_scores": task_scores,
        "summary_path": rel(root / "summary.json") if (root / "summary.json").exists() else None,
        "model_summary_path": rel(root / "model_summary.csv") if (root / "model_summary.csv").exists() else None,
        "metrics_path": rel(root / "metrics.csv") if (root / "metrics.csv").exists() else None,
    }
    for task, column in TASK_SCORE_COLUMNS.items():
        state[column] = task_scores.get(task)
    return state


def read_training_state(delta_dir: str | Path) -> dict[str, Any]:
    root = repo_path(delta_dir)
    summary = read_json_if_exists(root / "summary.json")
    trace = read_csv_if_exists(root / "training_trace.csv")
    final_loss = None
    if not trace.empty and "loss" in trace:
        final_loss = maybe_float(trace.iloc[-1].get("loss"))
    return {
        "delta_dir": rel(root),
        "training_status": summary.get("status", "not_run") if summary else "not_run",
        "training_summary_exists": bool(summary),
        "mean_initial_route_kl": maybe_float(summary.get("mean_initial_route_kl")),
        "mean_final_route_kl": maybe_float(summary.get("mean_final_route_kl")),
        "mean_initial_top1_agreement": maybe_float(summary.get("mean_initial_top1_agreement")),
        "mean_final_top1_agreement": maybe_float(summary.get("mean_final_top1_agreement")),
        "max_final_relative_delta_norm": maybe_float(summary.get("max_final_relative_delta_norm")),
        "final_training_loss": final_loss,
        "router_delta_safetensors": summary.get("outputs", {}).get("router_delta_safetensors")
        if summary
        else rel(root / "router_delta.safetensors"),
    }


def read_audit_state(audit_dir: str | Path, planned_cap: float, cap_tolerance: float) -> dict[str, Any]:
    root = repo_path(audit_dir)
    summary = read_json_if_exists(root / "summary.json")
    tensors = read_csv_if_exists(root / "tensor_delta_audit.csv")

    changed_tensors = maybe_int(summary.get("changed_tensors"))
    router_changed_tensors = maybe_int(summary.get("router_changed_tensors"))
    router_tensors = maybe_int(summary.get("router_tensors"))
    non_router_changed_tensors = None
    router_max_relative_delta_norm = None
    router_delta_norm = None
    if not tensors.empty and {"group", "changed"}.issubset(tensors.columns):
        changed_mask = bool_series(tensors["changed"])
        router_mask = tensors["group"].astype(str) == "router"
        non_router_changed_tensors = int((changed_mask & ~router_mask).sum())
        if "relative_delta_norm" in tensors:
            router_rel = pd.to_numeric(tensors.loc[router_mask, "relative_delta_norm"], errors="coerce").dropna()
            if not router_rel.empty:
                router_max_relative_delta_norm = float(router_rel.max())
        if "delta_norm2" in tensors:
            router_delta_norm = math.sqrt(
                float(pd.to_numeric(tensors.loc[router_mask, "delta_norm2"], errors="coerce").fillna(0.0).sum())
            )
        elif "delta_norm" in tensors:
            router_delta_norm = math.sqrt(
                float((pd.to_numeric(tensors.loc[router_mask, "delta_norm"], errors="coerce").fillna(0.0) ** 2).sum())
            )
    elif changed_tensors is not None and router_changed_tensors is not None:
        non_router_changed_tensors = changed_tensors - router_changed_tensors

    if router_max_relative_delta_norm is None:
        group_summary = summary.get("group_summary") or []
        for row in group_summary:
            if row.get("group") == "router":
                router_max_relative_delta_norm = maybe_float(row.get("relative_delta_norm"))
                router_delta_norm = maybe_float(row.get("delta_norm"))
                break

    has_shape_or_dtype_problem = bool(
        maybe_int(summary.get("shape_mismatch_count")) or maybe_int(summary.get("dtype_mismatch_count"))
    )
    audit_exists = bool(summary)
    router_only_changed = (
        audit_exists
        and non_router_changed_tensors == 0
        and (router_changed_tensors or 0) > 0
        and not has_shape_or_dtype_problem
    )
    cap_passed = (
        router_max_relative_delta_norm is not None
        and router_max_relative_delta_norm <= planned_cap * (1.0 + cap_tolerance)
    )
    return {
        "audit_dir": rel(root),
        "audit_exists": audit_exists,
        "audit_status": summary.get("status", "not_run") if summary else "not_run",
        "audit_shape_mismatch_count": maybe_int(summary.get("shape_mismatch_count")),
        "audit_dtype_mismatch_count": maybe_int(summary.get("dtype_mismatch_count")),
        "changed_tensors": changed_tensors,
        "router_tensors": router_tensors,
        "router_changed_tensors": router_changed_tensors,
        "non_router_changed_tensors": non_router_changed_tensors,
        "router_only_changed": router_only_changed,
        "router_max_relative_delta_norm": router_max_relative_delta_norm,
        "router_delta_norm": router_delta_norm,
        "planned_router_cap": float(planned_cap),
        "cap_passed": cap_passed,
        "audit_passed_for_router_calibration": bool(router_only_changed and cap_passed),
        "audit_summary_path": rel(root / "summary.json") if (root / "summary.json").exists() else None,
        "tensor_delta_audit": rel(root / "tensor_delta_audit.csv") if (root / "tensor_delta_audit.csv").exists() else None,
    }


def score_dict(row: dict[str, Any]) -> dict[str, float]:
    out = {}
    for column in SCORE_COLUMNS:
        value = maybe_float(row.get(column))
        if value is not None:
            out[column] = value
    return out


def score_delta(candidate: dict[str, Any], baseline: dict[str, Any], column: str) -> float | None:
    left = maybe_float(candidate.get(column))
    right = maybe_float(baseline.get(column))
    if left is None or right is None:
        return None
    return left - right


def common_score_columns(*rows: dict[str, Any]) -> list[str]:
    columns = []
    for column in SCORE_COLUMNS:
        if all(maybe_float(row.get(column)) is not None for row in rows):
            columns.append(column)
    return columns


def dominates(left: dict[str, Any], right: dict[str, Any], columns: list[str], eps: float = 1e-12) -> bool:
    pairs = []
    for column in columns:
        left_value = maybe_float(left.get(column))
        right_value = maybe_float(right.get(column))
        if left_value is None or right_value is None:
            continue
        pairs.append((left_value, right_value))
    if not pairs:
        return False
    return all(left_value >= right_value - eps for left_value, right_value in pairs) and any(
        left_value > right_value + eps for left_value, right_value in pairs
    )


def build_candidate_rows(
    candidate_plan: pd.DataFrame,
    baseline_eval: dict[str, Any],
    source_evals: list[dict[str, Any]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_complete = bool(baseline_eval.get("eval_completed"))
    source_complete = all(bool(row.get("eval_completed")) for row in source_evals) if source_evals else False

    for _, plan_row in candidate_plan.iterrows():
        method = str(plan_row["method"])
        cap = float(plan_row["router_max_relative_norm"])
        eval_state = read_eval_state(plan_row["eval_dir"])
        audit_state = read_audit_state(plan_row["audit_dir"], cap, args.cap_tolerance)
        training_state = read_training_state(plan_row["delta_dir"])

        row: dict[str, Any] = {
            "cap_label": str(plan_row["cap_label"]),
            "method": method,
            "router_max_relative_norm": cap,
            "checkpoint_dir": plan_row.get("checkpoint_dir"),
            "eval_dir": eval_state["eval_dir"],
            "audit_dir": audit_state["audit_dir"],
            "delta_dir": training_state["delta_dir"],
            **eval_state,
            **audit_state,
            **training_state,
        }

        deltas = {
            column: score_delta(row, baseline_eval, column)
            for column in SCORE_COLUMNS
            if maybe_float(row.get(column)) is not None and maybe_float(baseline_eval.get(column)) is not None
        }
        for column, value in deltas.items():
            row[f"delta_vs_baseline_{column}"] = value
        task_deltas = [
            value for column, value in deltas.items() if column.startswith("task_") and maybe_float(value) is not None
        ]
        row["worst_task_delta_vs_baseline"] = min(task_deltas) if task_deltas else None
        row["best_task_delta_vs_baseline"] = max(task_deltas) if task_deltas else None

        baseline_columns = common_score_columns(row, baseline_eval)
        row["baseline_dominates"] = bool(baseline_complete and dominates(baseline_eval, row, baseline_columns))
        dominated_by_sources = []
        if source_complete:
            for source in source_evals:
                columns = common_score_columns(row, source)
                if dominates(source, row, columns):
                    dominated_by_sources.append(str(source.get("model") or source.get("eval_dir")))
        row["dominated_by_source"] = ",".join(dominated_by_sources)

        avg_drop = maybe_float(row.get("delta_vs_baseline_avg_primary_score"))
        worst_drop = maybe_float(row.get("delta_vs_baseline_worst_primary_score"))
        worst_task_delta = maybe_float(row.get("worst_task_delta_vs_baseline"))
        best_delta = max((value for value in deltas.values() if value is not None), default=None)

        preserves_average = avg_drop is not None and avg_drop >= -args.max_avg_drop
        preserves_worst = worst_drop is not None and worst_drop >= -args.max_worst_drop
        preserves_tasks = worst_task_delta is None or worst_task_delta >= -args.max_task_drop
        has_downstream_gain = best_delta is not None and best_delta >= args.min_gain

        rejection_reasons = []
        if not baseline_complete:
            rejection_reasons.append("awaiting_baseline_eval")
        if not eval_state["eval_completed"]:
            rejection_reasons.append("awaiting_candidate_eval")
        if not audit_state["audit_exists"]:
            rejection_reasons.append("awaiting_audit")
        if audit_state["audit_exists"] and not audit_state["router_only_changed"]:
            rejection_reasons.append("audit_not_router_only")
        if audit_state["audit_exists"] and not audit_state["cap_passed"]:
            rejection_reasons.append("router_delta_cap_violation")
        if baseline_complete and eval_state["eval_completed"] and not preserves_average:
            rejection_reasons.append("avg_score_regression")
        if baseline_complete and eval_state["eval_completed"] and not preserves_worst:
            rejection_reasons.append("worst_score_regression")
        if baseline_complete and eval_state["eval_completed"] and not preserves_tasks:
            rejection_reasons.append("task_score_regression")
        if baseline_complete and eval_state["eval_completed"] and not has_downstream_gain:
            rejection_reasons.append("no_downstream_gain")
        if dominated_by_sources:
            rejection_reasons.append("source_endpoint_dominates")

        eligible = (
            baseline_complete
            and eval_state["eval_completed"]
            and audit_state["audit_passed_for_router_calibration"]
            and preserves_average
            and preserves_worst
            and preserves_tasks
            and has_downstream_gain
            and not dominated_by_sources
        )
        row["preserves_average"] = preserves_average
        row["preserves_worst"] = preserves_worst
        row["preserves_tasks"] = preserves_tasks
        row["has_downstream_gain"] = has_downstream_gain
        row["selection_eligible"] = bool(eligible)
        row["decision"] = "candidate_eligible" if eligible else "reject_or_wait"
        row["decision_reason"] = ",".join(rejection_reasons) if rejection_reasons else "passes_all_gates"
        row["selection_score"] = selection_score(row)
        rows.append(row)

    return pd.DataFrame(rows)


def selection_score(row: dict[str, Any]) -> float:
    avg = maybe_float(row.get("delta_vs_baseline_avg_primary_score")) or 0.0
    worst = maybe_float(row.get("delta_vs_baseline_worst_primary_score")) or 0.0
    task = maybe_float(row.get("worst_task_delta_vs_baseline")) or 0.0
    cap = maybe_float(row.get("router_max_relative_norm")) or 0.0
    router_norm = maybe_float(row.get("router_max_relative_delta_norm")) or cap
    route_kl_gain = 0.0
    initial_kl = maybe_float(row.get("mean_initial_route_kl"))
    final_kl = maybe_float(row.get("mean_final_route_kl"))
    if initial_kl is not None and final_kl is not None:
        route_kl_gain = initial_kl - final_kl
    return avg + 0.5 * worst + 0.25 * task + 0.05 * route_kl_gain - 0.01 * router_norm


def build_selection(
    table: pd.DataFrame,
    baseline_eval: dict[str, Any],
    source_evals: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_complete = bool(baseline_eval.get("eval_completed"))
    candidate_eval_complete = bool((table["eval_completed"].astype(bool)).all()) if not table.empty else False
    audit_complete = bool((table["audit_exists"].astype(bool)).all()) if not table.empty else False
    source_complete = all(bool(row.get("eval_completed")) for row in source_evals) if source_evals else False
    eligible = table[table["selection_eligible"].astype(bool)].copy() if not table.empty else pd.DataFrame()

    if not baseline_complete:
        status = "awaiting_baseline_eval"
        selected_method = None
        reason = "Run the frozen-router searched_no_gt065 baseline eval before deciding whether router calibration helps."
    elif not candidate_eval_complete or not audit_complete:
        status = "awaiting_router_calibration_eval"
        selected_method = None
        reason = "The cap sweep is not complete; all candidates need audit and matched vLLM downstream eval."
    elif eligible.empty:
        status = "keep_frozen_router_baseline"
        selected_method = "qwen3_moe_searched_no_gt065_max_retention_candidate"
        reason = "No router-calibrated cap improved downstream scores while preserving baseline/source-control gates."
    else:
        eligible = eligible.sort_values(
            ["selection_score", "avg_primary_score", "worst_primary_score", "router_max_relative_norm"],
            ascending=[False, False, False, True],
        )
        selected = eligible.iloc[0]
        status = "selected_router_calibrated_candidate"
        selected_method = str(selected["method"])
        reason = (
            "Selected the router-calibrated cap that improved downstream scores, stayed within cap, "
            "changed only router tensors, and was not source dominated."
        )

    return {
        "schema_version": 1,
        "status": status,
        "selected_method": selected_method,
        "reason": reason,
        "baseline_eval_completed": baseline_complete,
        "candidate_eval_completed": candidate_eval_complete,
        "audit_completed": audit_complete,
        "source_eval_completed": source_complete,
        "eligible_candidate_count": int(len(eligible)),
        "candidate_count": int(len(table)),
    }


def build_decision_rules(args: argparse.Namespace, selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "router_calibrated_moe_average_acceptance_gate",
        "current_selection": selection,
        "problem_statement": (
            "The target model must stay same-shape with the input Qwen3 MoE checkpoint. "
            "Router calibration is not accepted because an algorithm name sounds better; it is accepted only if the mechanism survives matched downstream eval and tensor audit."
        ),
        "theory_summary": [
            {
                "claim": "Averaging is a local operation in weight space.",
                "implication": "If the path between checkpoints has a loss barrier or a discrete router flip, a midpoint/delta can be worse even when local curvature looks small.",
            },
            {
                "claim": "MoE routers create discontinuous functional changes.",
                "implication": "A small router weight delta can change expert assignment for many tokens; therefore the default unified method freezes routers unless route-KD calibration gives downstream evidence.",
            },
            {
                "claim": "A better unified algorithm needs an abstention rule.",
                "implication": "When no calibrated cap is non-dominated by the frozen-router baseline/source endpoints, the correct same-shape output is the baseline/no-router-move candidate.",
            },
        ],
        "acceptance_gates": [
            "Baseline searched_no_gt065 eval must be complete on the same vLLM task set.",
            "Every cap candidate must have a materialized delta audit and vLLM eval before final selection.",
            "The audit must show only router tensors changed, with no shape/dtype mismatch.",
            "The maximum per-router relative delta norm must stay inside the planned cap.",
            f"Average primary score may not drop more than {args.max_avg_drop}.",
            f"Worst primary score may not drop more than {args.max_worst_drop}.",
            f"No available task primary score may drop more than {args.max_task_drop}.",
            f"At least one downstream primary/task score must improve by {args.min_gain} or more.",
            "If source endpoint evals are available, a candidate is rejected when a source dominates it on all available scores.",
        ],
        "ranking": [
            "Among accepted candidates, sort by selection_score.",
            "selection_score = avg_gain + 0.5 * worst_gain + 0.25 * worst_task_gain + 0.05 * route_kl_gain - 0.01 * router_delta_norm.",
            "Use avg score, worst score, and smaller cap as tie breakers.",
        ],
        "literature_hooks": LITERATURE_HOOKS,
    }


def fmt(value: Any, digits: int = 4) -> str:
    value = clean_value(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(
    summary: dict[str, Any],
    table: pd.DataFrame,
    baseline_eval: dict[str, Any],
    source_evals: list[dict[str, Any]],
    rules: dict[str, Any],
) -> str:
    selection = summary["current_selection"]
    lines = [
        "# Qwen3 MoE Router Calibration Selection",
        "",
        "这一步只解决一个问题：在 frozen-router 的 `searched_no_gt065` 基线之上，是否应该加入一个小的 route-KD router delta。结论不会按算法名决定，而是按机制证据决定：下游任务、router-only 审计、delta cap、source endpoint 支配关系必须同时通过。",
        "",
        f"- Selection status: `{selection['status']}`",
        f"- Selected method: `{selection.get('selected_method')}`",
        f"- Reason: {selection['reason']}",
        f"- Baseline eval completed: `{selection['baseline_eval_completed']}`",
        f"- Candidate eval completed: `{selection['candidate_eval_completed']}`",
        f"- Audit completed: `{selection['audit_completed']}`",
        f"- Eligible candidates: `{selection['eligible_candidate_count']}/{selection['candidate_count']}`",
        "",
        "## Baseline",
        "",
        "| eval dir | status | avg | worst | gsm8k | mmlu | safety | humaneval |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| `{baseline_eval['eval_dir']}` | `{baseline_eval['eval_status']}` | "
            f"{fmt(baseline_eval.get('avg_primary_score'))} | {fmt(baseline_eval.get('worst_primary_score'))} | "
            f"{fmt(baseline_eval.get('task_gsm8k_score'))} | {fmt(baseline_eval.get('task_mmlu_score'))} | "
            f"{fmt(baseline_eval.get('task_safety_score'))} | {fmt(baseline_eval.get('task_humaneval_compile_score'))} |"
        ),
        "",
        "## Candidate Gate",
        "",
        "| cap | method | decision | avg delta | worst delta | worst task delta | router max rel | router-only | cap pass | score | reason |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {fmt(row['router_max_relative_norm'])} | `{row['method']}` | `{row['decision']}` | "
            f"{fmt(row.get('delta_vs_baseline_avg_primary_score'))} | "
            f"{fmt(row.get('delta_vs_baseline_worst_primary_score'))} | "
            f"{fmt(row.get('worst_task_delta_vs_baseline'))} | "
            f"{fmt(row.get('router_max_relative_delta_norm'))} | `{row.get('router_only_changed')}` | "
            f"`{row.get('cap_passed')}` | {fmt(row.get('selection_score'))} | `{row.get('decision_reason')}` |"
        )
    lines.extend(
        [
            "",
            "## Source Controls",
            "",
            "| eval dir | status | avg | worst |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for source in source_evals:
        lines.append(
            f"| `{source['eval_dir']}` | `{source['eval_status']}` | "
            f"{fmt(source.get('avg_primary_score'))} | {fmt(source.get('worst_primary_score'))} |"
        )
    lines.extend(
        [
            "",
            "## Unified Rule Update",
            "",
            "如果 selection status 是 `selected_router_calibrated_candidate`，统一方法可以在 `searched_no_gt065` expert/attention 冻结策略后追加该 cap 的 router delta；否则 unified 默认继续保持 frozen router。这样算法不会在 router 机制证据不足时强行动 router。",
            "",
            "## Decision Rules",
            "",
        ]
    )
    for gate in rules["acceptance_gates"]:
        lines.append(f"- {gate}")
    lines.extend(
        [
            "",
            "## Literature Hooks",
            "",
        ]
    )
    for source in LITERATURE_HOOKS:
        lines.append(f"- [{source['title']}]({source['url']}): {source['mechanism']}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['selection_table']}`",
            f"- `{summary['outputs']['decision_rules']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_eval_dir(root: Path, model: str, scores: dict[str, float]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    avg = float(scores["avg_primary_score"])
    worst = float(scores["worst_primary_score"])
    pd.DataFrame(
        [
            {
                "model": model,
                "task_count": 4,
                "avg_primary_score": avg,
                "worst_primary_score": worst,
                "rank": 1,
            }
        ]
    ).to_csv(root / "model_summary.csv", index=False)
    pd.DataFrame(
        [
            {"task": "gsm8k", "examples": 8, "strict_exact": scores["task_gsm8k_score"], "model": model},
            {"task": "mmlu", "examples": 8, "accuracy": scores["task_mmlu_score"], "model": model},
            {"task": "safety", "examples": 8, "policy_accuracy": scores["task_safety_score"], "model": model},
            {
                "task": "humaneval_compile",
                "examples": 8,
                "compile_rate": scores["task_humaneval_compile_score"],
                "model": model,
            },
        ]
    ).to_csv(root / "metrics.csv", index=False)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "best_avg_primary_model": model,
                "model_summary": [
                    {
                        "model": model,
                        "task_count": 4,
                        "avg_primary_score": avg,
                        "worst_primary_score": worst,
                        "rank": 1,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_audit_dir(root: Path, cap: float, router_max: float, *, non_router_changed: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(4):
        rows.append(
            {
                "tensor": f"model.layers.{idx}.mlp.gate.weight",
                "group": "router",
                "changed": True,
                "delta_norm": router_max * 10.0,
                "delta_norm2": (router_max * 10.0) ** 2,
                "base_norm": 10.0,
                "relative_delta_norm": router_max if idx == 0 else min(router_max, cap * 0.8),
                "max_abs_delta": 0.01,
            }
        )
    if non_router_changed:
        rows.append(
            {
                "tensor": "model.layers.0.self_attn.q_proj.weight",
                "group": "attention",
                "changed": True,
                "delta_norm": 1.0,
                "delta_norm2": 1.0,
                "base_norm": 10.0,
                "relative_delta_norm": 0.1,
                "max_abs_delta": 0.02,
            }
        )
    tensor_df = pd.DataFrame(rows)
    tensor_df.to_csv(root / "tensor_delta_audit.csv", index=False)
    group_df = (
        tensor_df.groupby("group")
        .agg(
            tensor_count=("tensor", "count"),
            changed_tensors=("changed", "sum"),
            delta_norm=("delta_norm", "max"),
            relative_delta_norm=("relative_delta_norm", "max"),
            max_abs_delta=("max_abs_delta", "max"),
        )
        .reset_index()
    )
    group_df.to_csv(root / "group_delta_summary.csv", index=False)
    pd.DataFrame([{"layer": 0, "tensor_count": len(tensor_df), "changed_tensors": len(tensor_df)}]).to_csv(
        root / "layer_delta_summary.csv", index=False
    )
    router_changed = int((tensor_df["group"] == "router").sum())
    (root / "summary.json").write_text(
        json.dumps(
            {
                "status": "needs_review",
                "shape_mismatch_count": 0,
                "dtype_mismatch_count": 0,
                "changed_tensors": int(len(tensor_df)),
                "router_tensors": 4,
                "router_changed_tensors": router_changed,
                "relative_delta_norm": router_max,
                "group_summary": group_df.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_training_dir(root: Path, cap: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "passed",
        "mean_initial_route_kl": 0.12,
        "mean_final_route_kl": max(0.01, 0.12 - cap),
        "mean_initial_top1_agreement": 0.70,
        "mean_final_top1_agreement": min(0.95, 0.70 + cap * 2),
        "max_final_relative_delta_norm": cap,
        "outputs": {"router_delta_safetensors": rel(root / "router_delta.safetensors")},
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame([{"epoch": 0, "loss": 0.5}, {"epoch": 1, "loss": 0.2}]).to_csv(
        root / "training_trace.csv", index=False
    )


def write_smoke_inputs(output_dir: Path) -> Path:
    job_dir = output_dir / "input_job"
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    baseline_eval = job_dir / "eval_baseline"
    write_eval_dir(
        baseline_eval,
        "qwen3_moe_searched_no_gt065_max_retention_candidate",
        {
            "avg_primary_score": 0.500,
            "worst_primary_score": 0.300,
            "task_gsm8k_score": 0.420,
            "task_mmlu_score": 0.530,
            "task_safety_score": 0.620,
            "task_humaneval_compile_score": 0.300,
        },
    )
    source_eval_dirs = [job_dir / "eval_source_instruct", job_dir / "eval_source_coder"]
    write_eval_dir(
        source_eval_dirs[0],
        "source_qwen3_30b_instruct",
        {
            "avg_primary_score": 0.490,
            "worst_primary_score": 0.280,
            "task_gsm8k_score": 0.390,
            "task_mmlu_score": 0.540,
            "task_safety_score": 0.620,
            "task_humaneval_compile_score": 0.280,
        },
    )
    write_eval_dir(
        source_eval_dirs[1],
        "source_qwen3_30b_coder",
        {
            "avg_primary_score": 0.470,
            "worst_primary_score": 0.260,
            "task_gsm8k_score": 0.360,
            "task_mmlu_score": 0.490,
            "task_safety_score": 0.550,
            "task_humaneval_compile_score": 0.410,
        },
    )

    specs = [
        ("cap001", 0.010, 0.008, False, 0.501, 0.300, 0.421, 0.529, 0.620, 0.300),
        ("cap0025", 0.025, 0.022, False, 0.515, 0.310, 0.430, 0.535, 0.620, 0.320),
        ("cap005", 0.050, 0.071, True, 0.525, 0.315, 0.435, 0.540, 0.620, 0.340),
    ]
    candidate_rows = []
    for idx, (label, cap, router_max, non_router_changed, avg, worst, gsm8k, mmlu, safety, humaneval) in enumerate(specs):
        method = f"smoke_router_calibrated_{label}"
        delta_dir = job_dir / f"delta_{label}"
        audit_dir = job_dir / f"audit_{label}"
        eval_dir = job_dir / f"eval_{label}"
        write_training_dir(delta_dir, cap)
        write_audit_dir(audit_dir, cap, router_max, non_router_changed=non_router_changed)
        write_eval_dir(
            eval_dir,
            method,
            {
                "avg_primary_score": avg,
                "worst_primary_score": worst,
                "task_gsm8k_score": gsm8k,
                "task_mmlu_score": mmlu,
                "task_safety_score": safety,
                "task_humaneval_compile_score": humaneval,
            },
        )
        candidate_rows.append(
            {
                "rank": idx,
                "cap_label": label,
                "router_max_relative_norm": cap,
                "method": method,
                "delta_dir": rel(delta_dir),
                "audit_dir": rel(audit_dir),
                "eval_dir": rel(eval_dir),
                "checkpoint_dir": rel(job_dir / f"checkpoint_{label}"),
            }
        )
    pd.DataFrame(candidate_rows).to_csv(job_dir / "candidate_plan.csv", index=False)
    (job_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "smoke_ready",
                "candidate_count": len(candidate_rows),
                "baseline_eval_dir": rel(baseline_eval),
                "source_eval_dirs": [rel(path) for path in source_eval_dirs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return job_dir


def run_selection(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.job_dir = write_smoke_inputs(output_dir)
        args.baseline_eval_dir = args.job_dir / "eval_baseline"
        args.source_eval_dir = [args.job_dir / "eval_source_instruct", args.job_dir / "eval_source_coder"]

    candidate_plan = read_csv_if_exists(repo_path(args.job_dir) / "candidate_plan.csv")
    if candidate_plan.empty:
        raise FileNotFoundError(f"Missing candidate plan: {repo_path(args.job_dir) / 'candidate_plan.csv'}")

    baseline_eval = read_eval_state(args.baseline_eval_dir)
    baseline_audit = read_json_if_exists(args.baseline_audit_dir / "summary.json")
    source_evals = [read_eval_state(path) for path in args.source_eval_dir]
    table = build_candidate_rows(candidate_plan, baseline_eval, source_evals, args)
    selection = build_selection(table, baseline_eval, source_evals)
    rules = build_decision_rules(args, selection)

    selection_table_path = output_dir / "selection_table.csv"
    rules_path = output_dir / "decision_rules.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    sources_path = output_dir / "literature_sources.json"

    table.to_csv(selection_table_path, index=False)
    rules_path.write_text(json.dumps(json_safe(rules), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    sources_path.write_text(json.dumps(LITERATURE_HOOKS, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "schema_version": 1,
        "status": selection["status"],
        "smoke": bool(args.smoke),
        "job_dir": rel(args.job_dir),
        "baseline_eval": baseline_eval,
        "baseline_audit_status": baseline_audit.get("status") if baseline_audit else "not_available",
        "source_evals": source_evals,
        "current_selection": selection,
        "outputs": {
            "selection_table": rel(selection_table_path),
            "decision_rules": rel(rules_path),
            "literature_sources": rel(sources_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, table, baseline_eval, source_evals, rules), encoding="utf-8")
    print(f"Wrote Qwen3 MoE router calibration selection to {output_dir.resolve()}")
    print(f"Status: {selection['status']}")
    print(f"Selected: {selection.get('selected_method')}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a Qwen3 MoE router-calibrated cap from audit + vLLM eval results.")
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-eval-dir", type=Path, default=DEFAULT_BASELINE_EVAL_DIR)
    parser.add_argument("--baseline-audit-dir", type=Path, default=DEFAULT_BASELINE_AUDIT_DIR)
    parser.add_argument("--source-eval-dir", type=Path, action="append", default=None)
    parser.add_argument("--max-avg-drop", type=float, default=0.005)
    parser.add_argument("--max-worst-drop", type=float, default=0.01)
    parser.add_argument("--max-task-drop", type=float, default=0.02)
    parser.add_argument("--min-gain", type=float, default=0.002)
    parser.add_argument("--cap-tolerance", type=float, default=0.02)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.source_eval_dir is None:
        args.source_eval_dir = DEFAULT_SOURCE_EVAL_DIRS
    return args


def main() -> None:
    run_selection(parse_args())


if __name__ == "__main__":
    main()
