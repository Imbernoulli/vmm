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
        "selection_policy": summary.get("selection_policy"),
        "selection_split": summary.get("selection_split"),
        "mean_train_samples": maybe_float(summary.get("mean_train_samples")),
        "mean_selection_samples": maybe_float(summary.get("mean_selection_samples")),
        "mean_validation_fraction": maybe_float(summary.get("mean_validation_fraction")),
        "mean_train_group_count": maybe_float(summary.get("mean_train_group_count")),
        "mean_validation_group_count": maybe_float(summary.get("mean_validation_group_count")),
        "mean_selected_epoch": maybe_float(summary.get("mean_selected_epoch")),
        "min_selected_epoch": maybe_int(summary.get("min_selected_epoch")),
        "max_selected_epoch": maybe_int(summary.get("max_selected_epoch")),
        "mean_selection_score": maybe_float(summary.get("mean_selection_score")),
        "mean_train_final_route_kl": maybe_float(summary.get("mean_train_final_route_kl")),
        "mean_train_final_top1_agreement": maybe_float(summary.get("mean_train_final_top1_agreement")),
        "mean_route_kl_generalization_gap": maybe_float(summary.get("mean_route_kl_generalization_gap")),
        "max_route_kl_generalization_gap": maybe_float(summary.get("max_route_kl_generalization_gap")),
        "mean_top1_generalization_drop": maybe_float(summary.get("mean_top1_generalization_drop")),
        "max_top1_generalization_drop": maybe_float(summary.get("max_top1_generalization_drop")),
        "mean_initial_capacity_overflow_fraction": maybe_float(summary.get("mean_initial_capacity_overflow_fraction")),
        "max_initial_capacity_overflow_fraction": maybe_float(summary.get("max_initial_capacity_overflow_fraction")),
        "mean_final_capacity_overflow_fraction": maybe_float(summary.get("mean_final_capacity_overflow_fraction")),
        "max_final_capacity_overflow_fraction": maybe_float(summary.get("max_final_capacity_overflow_fraction")),
        "mean_initial_top1_capacity_overflow_fraction": maybe_float(
            summary.get("mean_initial_top1_capacity_overflow_fraction")
        ),
        "max_initial_top1_capacity_overflow_fraction": maybe_float(
            summary.get("max_initial_top1_capacity_overflow_fraction")
        ),
        "mean_final_top1_capacity_overflow_fraction": maybe_float(
            summary.get("mean_final_top1_capacity_overflow_fraction")
        ),
        "max_final_top1_capacity_overflow_fraction": maybe_float(
            summary.get("max_final_top1_capacity_overflow_fraction")
        ),
        "max_router_top1_capacity_overflow_increase": maybe_float(
            summary.get("max_router_top1_capacity_overflow_increase")
        ),
        "mean_initial_topk_capacity_overflow_fraction": maybe_float(
            summary.get("mean_initial_topk_capacity_overflow_fraction")
        ),
        "max_initial_topk_capacity_overflow_fraction": maybe_float(
            summary.get("max_initial_topk_capacity_overflow_fraction")
        ),
        "mean_final_topk_capacity_overflow_fraction": maybe_float(
            summary.get("mean_final_topk_capacity_overflow_fraction")
        ),
        "max_final_topk_capacity_overflow_fraction": maybe_float(
            summary.get("max_final_topk_capacity_overflow_fraction")
        ),
        "max_router_topk_capacity_overflow_increase": maybe_float(
            summary.get("max_router_topk_capacity_overflow_increase")
        ),
        "max_initial_top1_load_fraction": maybe_float(summary.get("max_initial_top1_load_fraction")),
        "max_final_top1_load_fraction": maybe_float(summary.get("max_final_top1_load_fraction")),
        "max_initial_topk_load_fraction": maybe_float(summary.get("max_initial_topk_load_fraction")),
        "max_final_topk_load_fraction": maybe_float(summary.get("max_final_topk_load_fraction")),
        "mean_initial_top1_load_entropy": maybe_float(summary.get("mean_initial_top1_load_entropy")),
        "mean_final_top1_load_entropy": maybe_float(summary.get("mean_final_top1_load_entropy")),
        "mean_initial_topk_load_entropy": maybe_float(summary.get("mean_initial_topk_load_entropy")),
        "mean_final_topk_load_entropy": maybe_float(summary.get("mean_final_topk_load_entropy")),
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
    source_required = not bool(args.allow_missing_source_eval)

    for _, plan_row in candidate_plan.iterrows():
        method = str(plan_row["method"])
        cap = float(plan_row["router_max_relative_norm"])
        eval_state = read_eval_state(plan_row["eval_dir"])
        audit_state = read_audit_state(plan_row["audit_dir"], cap, args.cap_tolerance)
        training_state = read_training_state(plan_row["delta_dir"])
        training_passed = training_state["training_status"] == "passed"
        top1_overflow = training_state.get("max_final_top1_capacity_overflow_fraction")
        topk_overflow = training_state.get("max_final_topk_capacity_overflow_fraction")
        top1_overflow_increase = training_state.get("max_router_top1_capacity_overflow_increase")
        topk_overflow_increase = training_state.get("max_router_topk_capacity_overflow_increase")
        route_kl_gap = training_state.get("max_route_kl_generalization_gap")
        top1_drop = training_state.get("max_top1_generalization_drop")
        capacity_metrics_ready = (
            bool(training_state["training_summary_exists"])
            and top1_overflow is not None
            and topk_overflow is not None
            and top1_overflow_increase is not None
            and topk_overflow_increase is not None
        )
        generalization_metrics_ready = (
            bool(training_state["training_summary_exists"]) and route_kl_gap is not None and top1_drop is not None
        )
        train_group_count = training_state.get("mean_train_group_count")
        validation_group_count = training_state.get("mean_validation_group_count")
        group_validation_required = not bool(args.allow_row_validation)
        group_validation_metrics_ready = (
            bool(training_state["training_summary_exists"])
            and training_state.get("selection_split") == "group_validation"
            and train_group_count is not None
            and validation_group_count is not None
        )
        group_validation_passed = bool(
            not group_validation_required
            or (
                group_validation_metrics_ready
                and train_group_count >= float(args.min_train_groups)
                and validation_group_count >= float(args.min_validation_groups)
            )
        )
        top1_capacity_passed = (
            top1_overflow is not None and top1_overflow <= float(args.max_top1_capacity_overflow)
        )
        topk_capacity_passed = (
            topk_overflow is not None and topk_overflow <= float(args.max_topk_capacity_overflow)
        )
        top1_capacity_not_worse = (
            top1_overflow_increase is not None
            and top1_overflow_increase <= float(args.max_top1_capacity_overflow_increase)
        )
        topk_capacity_not_worse = (
            topk_overflow_increase is not None
            and topk_overflow_increase <= float(args.max_topk_capacity_overflow_increase)
        )
        route_kl_generalizes = route_kl_gap is not None and route_kl_gap <= float(args.max_route_kl_validation_gap)
        top1_generalizes = top1_drop is not None and top1_drop <= float(args.max_top1_validation_drop)
        router_load_capacity_passed = bool(
            capacity_metrics_ready
            and top1_capacity_passed
            and topk_capacity_passed
            and top1_capacity_not_worse
            and topk_capacity_not_worse
        )
        router_generalization_passed = bool(
            generalization_metrics_ready
            and route_kl_generalizes
            and top1_generalizes
        )

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
            "training_passed": training_passed,
            "capacity_metrics_ready": capacity_metrics_ready,
            "top1_capacity_passed": top1_capacity_passed,
            "topk_capacity_passed": topk_capacity_passed,
            "top1_capacity_not_worse": top1_capacity_not_worse,
            "topk_capacity_not_worse": topk_capacity_not_worse,
            "router_load_capacity_passed": router_load_capacity_passed,
            "generalization_metrics_ready": generalization_metrics_ready,
            "route_kl_generalizes": route_kl_generalizes,
            "top1_generalizes": top1_generalizes,
            "router_generalization_passed": router_generalization_passed,
            "group_validation_required": group_validation_required,
            "group_validation_metrics_ready": group_validation_metrics_ready,
            "group_validation_passed": group_validation_passed,
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
        if source_required and not source_complete:
            rejection_reasons.append("awaiting_source_eval")
        if not training_state["training_summary_exists"]:
            rejection_reasons.append("awaiting_router_training")
        elif not training_passed:
            rejection_reasons.append("router_training_not_passed")
        elif not capacity_metrics_ready:
            rejection_reasons.append("awaiting_router_load_metrics")
        elif not top1_capacity_passed:
            rejection_reasons.append("top1_capacity_overflow")
        elif not topk_capacity_passed:
            rejection_reasons.append("topk_capacity_overflow")
        elif not top1_capacity_not_worse:
            rejection_reasons.append("top1_capacity_overflow_increase")
        elif not topk_capacity_not_worse:
            rejection_reasons.append("topk_capacity_overflow_increase")
        if training_state["training_summary_exists"] and not generalization_metrics_ready:
            rejection_reasons.append("awaiting_router_generalization_metrics")
        elif generalization_metrics_ready and not route_kl_generalizes:
            rejection_reasons.append("router_route_kl_validation_gap")
        elif generalization_metrics_ready and not top1_generalizes:
            rejection_reasons.append("router_top1_validation_drop")
        if training_state["training_summary_exists"] and group_validation_required and not group_validation_metrics_ready:
            rejection_reasons.append("router_validation_not_group_heldout")
        elif training_state["training_summary_exists"] and not group_validation_passed:
            rejection_reasons.append("insufficient_router_validation_groups")
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
            and (source_complete or not source_required)
            and training_passed
            and router_load_capacity_passed
            and router_generalization_passed
            and group_validation_passed
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
    top1_overflow = maybe_float(row.get("max_final_top1_capacity_overflow_fraction")) or 0.0
    topk_overflow = maybe_float(row.get("max_final_topk_capacity_overflow_fraction")) or 0.0
    top1_overflow_increase = max(0.0, maybe_float(row.get("max_router_top1_capacity_overflow_increase")) or 0.0)
    topk_overflow_increase = max(0.0, maybe_float(row.get("max_router_topk_capacity_overflow_increase")) or 0.0)
    route_kl_gap = max(0.0, maybe_float(row.get("max_route_kl_generalization_gap")) or 0.0)
    top1_drop = max(0.0, maybe_float(row.get("max_top1_generalization_drop")) or 0.0)
    return (
        avg
        + 0.5 * worst
        + 0.25 * task
        + 0.05 * route_kl_gain
        - 0.01 * router_norm
        - 0.10 * top1_overflow
        - 0.25 * topk_overflow
        - 0.15 * top1_overflow_increase
        - 0.35 * topk_overflow_increase
        - 0.10 * route_kl_gap
        - 0.05 * top1_drop
    )


def build_selection(
    table: pd.DataFrame,
    baseline_eval: dict[str, Any],
    source_evals: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline_complete = bool(baseline_eval.get("eval_completed"))
    candidate_eval_complete = bool((table["eval_completed"].astype(bool)).all()) if not table.empty else False
    audit_complete = bool((table["audit_exists"].astype(bool)).all()) if not table.empty else False
    training_complete = bool((table["training_summary_exists"].astype(bool)).all()) if not table.empty else False
    capacity_metrics_complete = bool((table["capacity_metrics_ready"].astype(bool)).all()) if not table.empty else False
    group_validation_complete = bool((table["group_validation_passed"].astype(bool)).all()) if not table.empty else False
    source_complete = all(bool(row.get("eval_completed")) for row in source_evals) if source_evals else False
    source_required = not bool(args.allow_missing_source_eval)
    eligible = table[table["selection_eligible"].astype(bool)].copy() if not table.empty else pd.DataFrame()

    if not baseline_complete:
        status = "awaiting_baseline_eval"
        selected_method = None
        reason = "Run the frozen-router searched_no_gt065 baseline eval before deciding whether router calibration helps."
    elif source_required and not source_complete:
        status = "awaiting_source_eval"
        selected_method = None
        reason = "Run both source endpoint evals before allowing router calibration selection or frozen-router fallback."
    elif (
        not candidate_eval_complete
        or not audit_complete
        or not training_complete
        or not capacity_metrics_complete
        or not group_validation_complete
    ):
        status = "awaiting_router_calibration_eval"
        selected_method = None
        reason = (
            "The cap sweep is not complete; all candidates need router training summaries, hard route-load metrics, "
            "group-heldout validation, audit, and matched vLLM downstream eval."
        )
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
        "training_completed": training_complete,
        "capacity_metrics_completed": capacity_metrics_complete,
        "group_validation_completed": group_validation_complete,
        "source_eval_completed": source_complete,
        "source_eval_required": source_required,
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
                "claim": "Route KL alone can hide expert-load collapse.",
                "implication": "A candidate must also pass hard top-1/top-k capacity-overflow gates computed from actual token assignments, not only softmax mean load.",
            },
            {
                "claim": "Router calibration should be a local correction, not a new routing regime.",
                "implication": "The selector rejects candidates whose hard capacity overflow increases too much relative to the frozen-router starting point, even if their absolute overflow is below a fixed cap.",
            },
            {
                "claim": "Router probes need held-out evidence.",
                "implication": "A calibrated router delta can only pass selection when its route-KD and top-1 agreement transfer from the training cache to the held-out selection cache.",
            },
            {
                "claim": "Token-row random validation can leak prompt-specific routing patterns.",
                "implication": "By default the selector requires group-heldout validation, so every prompt/batch group used for selection is absent from the router-delta training rows.",
            },
            {
                "claim": "A better unified algorithm needs an abstention rule.",
                "implication": "When no calibrated cap is non-dominated by the frozen-router baseline/source endpoints, the correct same-shape output is the baseline/no-router-move candidate.",
            },
        ],
        "acceptance_gates": [
            "Baseline searched_no_gt065 eval must be complete on the same vLLM task set.",
            "Both source endpoint evals must be complete unless --allow-missing-source-eval is explicitly set.",
            "Every cap candidate must have a materialized delta audit and vLLM eval before final selection.",
            "Every cap candidate must have router training metrics with hard top-1/top-k route-load statistics.",
            (
                "Router route-KD validation must use group-heldout prompt/batch splits "
                f"with at least {args.min_train_groups} train groups and {args.min_validation_groups} validation groups, "
                "unless --allow-row-validation is explicitly set."
            ),
            "The audit must show only router tensors changed, with no shape/dtype mismatch.",
            "The maximum per-router relative delta norm must stay inside the planned cap.",
            f"Hard top-1 route capacity overflow may not exceed {args.max_top1_capacity_overflow}.",
            f"Hard top-k route capacity overflow may not exceed {args.max_topk_capacity_overflow}.",
            f"Hard top-1 route capacity overflow may not increase over the frozen-router start by more than {args.max_top1_capacity_overflow_increase}.",
            f"Hard top-k route capacity overflow may not increase over the frozen-router start by more than {args.max_topk_capacity_overflow_increase}.",
            f"Validation route-KL gap over train route-KL may not exceed {args.max_route_kl_validation_gap}.",
            f"Validation top-1 agreement drop from train top-1 agreement may not exceed {args.max_top1_validation_drop}.",
            f"Average primary score may not drop more than {args.max_avg_drop}.",
            f"Worst primary score may not drop more than {args.max_worst_drop}.",
            f"No available task primary score may drop more than {args.max_task_drop}.",
            f"At least one downstream primary/task score must improve by {args.min_gain} or more.",
            "A candidate is rejected when a source endpoint dominates it on all available scores.",
        ],
        "ranking": [
            "Among accepted candidates, sort by selection_score.",
            "selection_score = avg_gain + 0.5 * worst_gain + 0.25 * worst_task_gain + 0.05 * route_kl_gain - 0.01 * router_delta_norm - 0.10 * top1_overflow - 0.25 * topk_overflow - 0.15 * top1_overflow_increase - 0.35 * topk_overflow_increase - 0.10 * route_kl_validation_gap - 0.05 * top1_validation_drop.",
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
        f"- Source eval required: `{selection['source_eval_required']}`",
        f"- Source eval completed: `{selection['source_eval_completed']}`",
        f"- Candidate eval completed: `{selection['candidate_eval_completed']}`",
        f"- Audit completed: `{selection['audit_completed']}`",
        f"- Training completed: `{selection['training_completed']}`",
        f"- Capacity metrics completed: `{selection['capacity_metrics_completed']}`",
        f"- Group validation completed: `{selection['group_validation_completed']}`",
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
        "| cap | method | split | groups | selected epoch | KL gap | top1 drop | decision | avg delta | worst delta | worst task delta | router max rel | top1/top-k overflow | top1/top-k increase | load pass | gen pass | group pass | router-only | cap pass | score | reason |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for _, row in table.iterrows():
        overflow_text = (
            f"{fmt(row.get('max_final_top1_capacity_overflow_fraction'))}/"
            f"{fmt(row.get('max_final_topk_capacity_overflow_fraction'))}"
        )
        increase_text = (
            f"{fmt(row.get('max_router_top1_capacity_overflow_increase'))}/"
            f"{fmt(row.get('max_router_topk_capacity_overflow_increase'))}"
        )
        lines.append(
            f"| {fmt(row['router_max_relative_norm'])} | `{row['method']}` | "
            f"`{row.get('selection_split')}` | "
            f"{fmt(row.get('mean_train_group_count'), 1)}/{fmt(row.get('mean_validation_group_count'), 1)} | "
            f"{fmt(row.get('mean_selected_epoch'), 1)} | "
            f"{fmt(row.get('max_route_kl_generalization_gap'))} | "
            f"{fmt(row.get('max_top1_generalization_drop'))} | `{row['decision']}` | "
            f"{fmt(row.get('delta_vs_baseline_avg_primary_score'))} | "
            f"{fmt(row.get('delta_vs_baseline_worst_primary_score'))} | "
            f"{fmt(row.get('worst_task_delta_vs_baseline'))} | "
            f"{fmt(row.get('router_max_relative_delta_norm'))} | {overflow_text} | "
            f"{increase_text} | "
            f"`{row.get('router_load_capacity_passed')}` | `{row.get('router_generalization_passed')}` | "
            f"`{row.get('group_validation_passed')}` | "
            f"`{row.get('router_only_changed')}` | "
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


def write_training_dir(
    root: Path,
    cap: float,
    initial_top1_overflow: float,
    final_top1_overflow: float,
    initial_topk_overflow: float,
    final_topk_overflow: float,
    *,
    selection_split: str = "group_validation",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    top1_increase = final_top1_overflow - initial_top1_overflow
    topk_increase = final_topk_overflow - initial_topk_overflow
    summary = {
        "status": "passed",
        "schema_version": 2,
        "mean_initial_route_kl": 0.12,
        "mean_final_route_kl": max(0.01, 0.12 - cap),
        "mean_initial_top1_agreement": 0.70,
        "mean_final_top1_agreement": min(0.95, 0.70 + cap * 2),
        "max_final_relative_delta_norm": cap,
        "selection_policy": "capacity_aware",
        "selection_split": selection_split,
        "mean_train_samples": 128.0,
        "mean_selection_samples": 32.0,
        "mean_validation_fraction": 0.2,
        "mean_train_group_count": 8.0,
        "mean_validation_group_count": 2.0,
        "mean_selected_epoch": 10.0,
        "min_selected_epoch": 10,
        "max_selected_epoch": 10,
        "mean_selection_score": max(0.0, 0.12 - cap),
        "mean_train_final_route_kl": max(0.01, 0.10 - cap),
        "mean_train_final_top1_agreement": min(0.98, 0.72 + cap * 2),
        "mean_route_kl_generalization_gap": 0.01,
        "max_route_kl_generalization_gap": 0.02,
        "mean_top1_generalization_drop": 0.01,
        "max_top1_generalization_drop": 0.02,
        "mean_initial_capacity_overflow_fraction": initial_topk_overflow / 2.0,
        "max_initial_capacity_overflow_fraction": initial_topk_overflow,
        "mean_final_capacity_overflow_fraction": final_topk_overflow / 2.0,
        "max_final_capacity_overflow_fraction": final_topk_overflow,
        "mean_initial_top1_capacity_overflow_fraction": initial_top1_overflow / 2.0,
        "max_initial_top1_capacity_overflow_fraction": initial_top1_overflow,
        "mean_final_top1_capacity_overflow_fraction": final_top1_overflow / 2.0,
        "max_final_top1_capacity_overflow_fraction": final_top1_overflow,
        "max_router_top1_capacity_overflow_increase": top1_increase,
        "mean_initial_topk_capacity_overflow_fraction": initial_topk_overflow / 2.0,
        "max_initial_topk_capacity_overflow_fraction": initial_topk_overflow,
        "mean_final_topk_capacity_overflow_fraction": final_topk_overflow / 2.0,
        "max_final_topk_capacity_overflow_fraction": final_topk_overflow,
        "max_router_topk_capacity_overflow_increase": topk_increase,
        "max_initial_top1_load_fraction": 0.20 + initial_top1_overflow,
        "max_final_top1_load_fraction": 0.20 + final_top1_overflow,
        "max_initial_topk_load_fraction": 0.18 + initial_topk_overflow,
        "max_final_topk_load_fraction": 0.18 + final_topk_overflow,
        "mean_initial_top1_load_entropy": max(0.0, 0.96 - initial_top1_overflow),
        "mean_final_top1_load_entropy": max(0.0, 0.96 - final_top1_overflow),
        "mean_initial_topk_load_entropy": max(0.0, 0.98 - initial_topk_overflow),
        "mean_final_topk_load_entropy": max(0.0, 0.98 - final_topk_overflow),
        "outputs": {"router_delta_safetensors": rel(root / "router_delta.safetensors")},
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame([{"epoch": 0, "loss": 0.5}, {"epoch": 1, "loss": 0.2}]).to_csv(
        root / "training_trace.csv", index=False
    )


def write_smoke_inputs(
    output_dir: Path,
    *,
    row_validation_negative: bool = False,
    source_dominance_negative: bool = False,
) -> Path:
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
    instruct_source_scores = (
        {
            "avg_primary_score": 0.550,
            "worst_primary_score": 0.360,
            "task_gsm8k_score": 0.460,
            "task_mmlu_score": 0.570,
            "task_safety_score": 0.650,
            "task_humaneval_compile_score": 0.360,
        }
        if source_dominance_negative
        else {
            "avg_primary_score": 0.490,
            "worst_primary_score": 0.280,
            "task_gsm8k_score": 0.390,
            "task_mmlu_score": 0.540,
            "task_safety_score": 0.620,
            "task_humaneval_compile_score": 0.280,
        }
    )
    write_eval_dir(
        source_eval_dirs[0],
        "source_qwen3_30b_instruct",
        instruct_source_scores,
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

    smoke_split = "validation" if row_validation_negative else "group_validation"
    specs = [
        (
            "cap001",
            0.010,
            0.008,
            False,
            smoke_split,
            0.000,
            0.000,
            0.000,
            0.000,
            0.501,
            0.300,
            0.421,
            0.529,
            0.620,
            0.300,
        ),
        (
            "cap0025",
            0.025,
            0.022,
            False,
            smoke_split,
            0.010,
            0.015,
            0.005,
            0.020,
            0.515,
            0.310,
            0.430,
            0.535,
            0.620,
            0.320,
        ),
        (
            "cap005",
            0.050,
            0.071,
            True,
            smoke_split,
            0.030,
            0.090,
            0.015,
            0.040,
            0.525,
            0.315,
            0.435,
            0.540,
            0.620,
            0.340,
        ),
    ]
    candidate_rows = []
    for idx, (
        label,
        cap,
        router_max,
        non_router_changed,
        selection_split,
        initial_top1_overflow,
        final_top1_overflow,
        initial_topk_overflow,
        final_topk_overflow,
        avg,
        worst,
        gsm8k,
        mmlu,
        safety,
        humaneval,
    ) in enumerate(specs):
        method = f"smoke_router_calibrated_{label}"
        delta_dir = job_dir / f"delta_{label}"
        audit_dir = job_dir / f"audit_{label}"
        eval_dir = job_dir / f"eval_{label}"
        write_training_dir(
            delta_dir,
            cap,
            initial_top1_overflow,
            final_top1_overflow,
            initial_topk_overflow,
            final_topk_overflow,
            selection_split=selection_split,
        )
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
    smoke_modes = [
        bool(args.smoke),
        bool(args.row_validation_negative_smoke),
        bool(args.source_dominance_negative_smoke),
    ]
    if sum(smoke_modes) > 1:
        raise ValueError("Use only one smoke mode at a time.")
    if any(smoke_modes):
        args.job_dir = write_smoke_inputs(
            output_dir,
            row_validation_negative=args.row_validation_negative_smoke,
            source_dominance_negative=args.source_dominance_negative_smoke,
        )
        args.baseline_eval_dir = args.job_dir / "eval_baseline"
        args.source_eval_dir = [args.job_dir / "eval_source_instruct", args.job_dir / "eval_source_coder"]

    candidate_plan = read_csv_if_exists(repo_path(args.job_dir) / "candidate_plan.csv")
    if candidate_plan.empty:
        raise FileNotFoundError(f"Missing candidate plan: {repo_path(args.job_dir) / 'candidate_plan.csv'}")

    baseline_eval = read_eval_state(args.baseline_eval_dir)
    baseline_audit = read_json_if_exists(args.baseline_audit_dir / "summary.json")
    source_evals = [read_eval_state(path) for path in args.source_eval_dir]
    table = build_candidate_rows(candidate_plan, baseline_eval, source_evals, args)
    selection = build_selection(table, baseline_eval, source_evals, args)
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
        "smoke": any(smoke_modes),
        "row_validation_negative_smoke": bool(args.row_validation_negative_smoke),
        "source_dominance_negative_smoke": bool(args.source_dominance_negative_smoke),
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
    parser.add_argument(
        "--max-top1-capacity-overflow",
        type=float,
        default=0.10,
        help="Reject router-calibrated candidates whose final hard top-1 route-load overflow exceeds this fraction.",
    )
    parser.add_argument(
        "--max-topk-capacity-overflow",
        type=float,
        default=0.05,
        help="Reject router-calibrated candidates whose final hard top-k route-load overflow exceeds this fraction.",
    )
    parser.add_argument(
        "--max-top1-capacity-overflow-increase",
        type=float,
        default=0.05,
        help="Reject router-calibrated candidates whose hard top-1 overflow increases too much over the frozen-router start.",
    )
    parser.add_argument(
        "--max-topk-capacity-overflow-increase",
        type=float,
        default=0.02,
        help="Reject router-calibrated candidates whose hard top-k overflow increases too much over the frozen-router start.",
    )
    parser.add_argument(
        "--max-route-kl-validation-gap",
        type=float,
        default=0.20,
        help="Reject router-calibrated candidates whose held-out route KL is too much worse than train route KL.",
    )
    parser.add_argument(
        "--max-top1-validation-drop",
        type=float,
        default=0.20,
        help="Reject router-calibrated candidates whose held-out top-1 agreement drops too much below train top-1.",
    )
    parser.add_argument(
        "--min-train-groups",
        type=int,
        default=1,
        help="Minimum group-heldout training groups required for router calibration acceptance.",
    )
    parser.add_argument(
        "--min-validation-groups",
        type=int,
        default=1,
        help="Minimum group-heldout validation groups required for router calibration acceptance.",
    )
    parser.add_argument(
        "--allow-row-validation",
        action="store_true",
        help="Debug escape hatch: allow row-level validation splits instead of requiring group-heldout prompt/batch splits.",
    )
    parser.add_argument(
        "--row-validation-negative-smoke",
        action="store_true",
        help="Build a complete smoke job with row-level validation splits; default selection should reject it.",
    )
    parser.add_argument(
        "--source-dominance-negative-smoke",
        action="store_true",
        help="Build a complete smoke job where a source endpoint dominates otherwise valid router-calibrated candidates.",
    )
    parser.add_argument(
        "--allow-missing-source-eval",
        action="store_true",
        help="Debug escape hatch: allow selection without completed source endpoint evals.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.source_eval_dir is None:
        args.source_eval_dir = DEFAULT_SOURCE_EVAL_DIRS
    return args


def main() -> None:
    run_selection(parse_args())


if __name__ == "__main__":
    main()
