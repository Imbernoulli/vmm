#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shlex
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_METHODS = ["source_qwen3_30b_instruct", "source_qwen3_30b_coder"]
AUTO_CANDIDATE = "auto"
DEFAULT_FEEDBACK_CANDIDATE = "qwen3_moe_mechanistic_unified_candidate"
BASE_SOURCE = "instruct"
NONBASE_SOURCE = "coder"
EPS = 1e-12
FEEDBACK_BASES = [
    {
        "method": "qwen3_moe_mechanistic_unified_candidate",
        "group_rules": "results/qwen3_moe_mechanistic_unified_candidate/mechanistic_group_rules.csv",
        "writer_command": "results/qwen3_moe_mechanistic_unified_candidate/writer_command.txt",
        "base_priority": 1.0,
    },
    {
        "method": "qwen3_moe_subspace_scaled_candidate",
        "group_rules": "results/qwen3_moe_expert_subspace_conflict_probe/subspace_adjusted_group_rules.csv",
        "writer_command": "results/qwen3_moe_expert_subspace_conflict_probe/writer_command.txt",
        "base_priority": 1.0,
    },
    {
        "method": "qwen3_moe_unified_mechanism_candidate",
        "group_rules": "results/qwen3_moe_unified_mechanism_candidate/unified_group_rules.csv",
        "writer_command": "results/qwen3_moe_unified_mechanism_candidate/writer_command.txt",
        "base_priority": 0.6,
    },
]

TASKS = {
    "gsm8k": {
        "score_column": "task_gsm8k_score",
        "family": "math",
        "frontier_source": "source_qwen3_30b_instruct",
        "regression_action": "shrink_nonbase_on_task_family",
    },
    "mmlu": {
        "score_column": "task_mmlu_score",
        "family": "general",
        "frontier_source": "source_qwen3_30b_instruct",
        "regression_action": "shrink_nonbase_on_task_family",
    },
    "safety": {
        "score_column": "task_safety_score",
        "family": "safety",
        "frontier_source": "source_qwen3_30b_instruct",
        "regression_action": "shrink_nonbase_on_task_family",
    },
    "humaneval_compile": {
        "score_column": "task_humaneval_compile_score",
        "family": "code",
        "frontier_source": "source_qwen3_30b_coder",
        "regression_action": "restore_coder_on_code_family",
    },
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


def maybe_bool(value: Any) -> bool | None:
    value = clean_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


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


def fmt(value: Any, digits: int = 4) -> str:
    value = clean_value(value)
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def robust01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.quantile(0.05))
    hi = float(values.quantile(0.95))
    if hi <= lo + EPS:
        return pd.Series(0.0, index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def bool_value(value: Any) -> bool:
    value = clean_value(value)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def primary_metric(row: pd.Series | dict[str, Any]) -> tuple[str, float] | None:
    for column in ("strict_exact", "accuracy", "policy_accuracy", "compile_rate"):
        value = maybe_float(row.get(column))
        if value is not None:
            return column, value
    return None


def prediction_key(row: pd.Series | dict[str, Any]) -> str | None:
    task = str(row.get("task", "")).strip()
    if task not in TASKS:
        return None
    for column in ("task_id", "index"):
        value = clean_value(row.get(column))
        if value is not None and str(value).strip():
            return f"{task}::{value}"
    return None


def prediction_correct(row: pd.Series | dict[str, Any]) -> bool | None:
    task = str(row.get("task", "")).strip()
    if task in {"gsm8k", "mmlu"}:
        return maybe_bool(row.get("correct"))
    if task == "safety":
        expected = maybe_bool(row.get("expected_refusal"))
        refused = maybe_bool(row.get("refused"))
        if expected is not None and refused is not None:
            return refused == expected
        return maybe_bool(row.get("correct"))
    if task == "humaneval_compile":
        return maybe_bool(row.get("compile_ok"))
    for column in ("correct", "compile_ok", "loose_correct"):
        value = maybe_bool(row.get(column))
        if value is not None:
            return value
    return None


def read_prediction_scores(eval_dir: str | Path) -> dict[str, bool]:
    predictions = read_csv(repo_path(eval_dir) / "predictions.csv")
    if predictions.empty:
        return {}
    scores: dict[str, bool] = {}
    for _, row in predictions.iterrows():
        key = prediction_key(row)
        correct = prediction_correct(row)
        if key is not None and correct is not None:
            scores[key] = correct
    return scores


def read_eval_scores(eval_dir: str | Path) -> dict[str, Any]:
    root = repo_path(eval_dir)
    summary = read_json(root / "summary.json")
    model_summary = read_csv(root / "model_summary.csv")
    metrics = read_csv(root / "metrics.csv")
    model_row = model_summary.iloc[0].to_dict() if not model_summary.empty else {}
    task_scores: dict[str, float] = {}
    if not metrics.empty:
        for _, row in metrics.iterrows():
            task = str(row.get("task", ""))
            primary = primary_metric(row)
            if primary is not None:
                task_scores[task] = primary[1]
    return {
        "eval_status": summary.get("status", "missing"),
        "avg_primary_score": maybe_float(model_row.get("avg_primary_score")),
        "worst_primary_score": maybe_float(model_row.get("worst_primary_score")),
        **{spec["score_column"]: task_scores.get(task) for task, spec in TASKS.items()},
        "_prediction_scores": read_prediction_scores(root),
    }


def audit_usable_map(audit_dir: Path) -> dict[str, bool]:
    audit = read_csv(repo_path(audit_dir) / "audit_rows.csv")
    if audit.empty:
        return {}
    return {str(row["method"]): bool(row.get("usable_for_selection", False)) for _, row in audit.iterrows()}


def load_method_rows(gate_dir: Path, audit_dir: Path) -> dict[str, dict[str, Any]]:
    gate = read_csv(repo_path(gate_dir) / "eval_gate_plan.csv")
    usable = audit_usable_map(audit_dir)
    rows: dict[str, dict[str, Any]] = {}
    for _, row in gate.iterrows():
        method = str(row.get("method", ""))
        item = row.to_dict()
        item["eval_usable"] = bool(usable.get(method, False))
        item["_prediction_scores"] = {}
        if item["eval_usable"]:
            item.update(read_eval_scores(str(row.get("eval_output_dir", ""))))
        rows[method] = item
    return rows


def source_frontier(rows: dict[str, dict[str, Any]], task: str) -> dict[str, Any]:
    column = TASKS[task]["score_column"]
    candidates = []
    for method in SOURCE_METHODS:
        value = maybe_float(rows.get(method, {}).get(column))
        if value is not None and bool(rows.get(method, {}).get("eval_usable")):
            candidates.append((value, method))
    if not candidates:
        return {"score": None, "method": None, "complete": False}
    score, method = max(candidates, key=lambda item: item[0])
    return {"score": score, "method": method, "complete": len(candidates) == len(SOURCE_METHODS)}


def paired_feedback(
    candidate_scores: dict[str, bool],
    source_scores: dict[str, bool],
    *,
    task: str,
) -> dict[str, Any]:
    keys = sorted(key for key in set(candidate_scores) & set(source_scores) if key.startswith(task + "::"))
    candidate_only = sum(int(candidate_scores[key] and not source_scores[key]) for key in keys)
    source_only = sum(int(source_scores[key] and not candidate_scores[key]) for key in keys)
    shared = len(keys)
    net = (candidate_only - source_only) / shared if shared else None
    return {
        "paired_shared_examples": shared,
        "paired_candidate_only_correct": candidate_only,
        "paired_source_only_correct": source_only,
        "paired_net_delta": net,
    }


def build_task_feedback(
    rows: dict[str, dict[str, Any]],
    *,
    candidate_method: str,
    regression_margin: float,
    improvement_margin: float,
    pressure_scale: float,
) -> pd.DataFrame:
    candidate = rows.get(candidate_method, {})
    candidate_scores = candidate.get("_prediction_scores") if isinstance(candidate.get("_prediction_scores"), dict) else {}
    out = []
    for task, spec in TASKS.items():
        column = spec["score_column"]
        frontier = source_frontier(rows, task)
        source_method = frontier["method"]
        source_row = rows.get(str(source_method), {}) if source_method else {}
        candidate_score = maybe_float(candidate.get(column))
        source_score = maybe_float(frontier["score"])
        delta = candidate_score - source_score if candidate_score is not None and source_score is not None else None
        if delta is None or not bool(candidate.get("eval_usable")) or not frontier["complete"]:
            status = "awaiting_eval"
            pressure = 0.0
        elif delta < -regression_margin:
            status = "source_frontier_regression"
            pressure = min(1.0, max(0.0, (-delta - regression_margin) / max(EPS, pressure_scale)))
        elif delta > improvement_margin:
            status = "candidate_improvement"
            pressure = 0.0
        else:
            status = "frontier_parity"
            pressure = 0.0
        paired = paired_feedback(
            candidate_scores,
            source_row.get("_prediction_scores") if isinstance(source_row.get("_prediction_scores"), dict) else {},
            task=task,
        )
        out.append(
            {
                "task": task,
                "family": spec["family"],
                "candidate_method": candidate_method,
                "candidate_score": candidate_score,
                "source_frontier_score": source_score,
                "source_frontier_method": source_method,
                "delta_vs_source_frontier": delta,
                "feedback_status": status,
                "feedback_pressure": pressure,
                "regression_action": spec["regression_action"] if status == "source_frontier_regression" else "none",
                **paired,
            }
        )
    return pd.DataFrame(out)


def known_feedback_base(method: str) -> dict[str, Any] | None:
    for item in FEEDBACK_BASES:
        if item["method"] == method:
            return item
    return None


def resolve_feedback_base(
    args: argparse.Namespace,
    rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    structural = read_csv(args.structural_dominance)
    delta_frontier = read_csv(args.delta_frontier)
    requested = str(args.candidate_method)
    if requested != AUTO_CANDIDATE:
        base = known_feedback_base(requested)
        group_rules = args.group_rules or (Path(base["group_rules"]) if base else None)
        writer_command = args.writer_context_command or (Path(base["writer_command"]) if base else None)
        if group_rules is None or writer_command is None:
            raise ValueError(
                "Explicit candidate requires --group-rules and --writer-context-command "
                "unless it is a known feedback base."
            )
        return {
            "candidate_method": requested,
            "requested_candidate_method": requested,
            "feedback_base_selection_status": "explicit_candidate",
            "feedback_base_selection_reason": "candidate_method_explicit",
            "feedback_base_structural_frontier_member": None,
            "feedback_base_structurally_dominated": None,
            "feedback_base_structural_safety_score": None,
            "group_rules": repo_path(group_rules),
            "writer_context_command": repo_path(writer_command),
            "candidate_rows_considered": [],
        }

    candidates = []
    for base in FEEDBACK_BASES:
        method = str(base["method"])
        group_path = repo_path(base["group_rules"])
        writer_path = repo_path(base["writer_command"])
        if method not in rows:
            continue
        if not group_path.exists() or not writer_path.exists():
            continue
        structural_row = (
            structural[structural["method"].astype(str).eq(method)].iloc[0].to_dict()
            if not structural.empty and "method" in structural and structural["method"].astype(str).eq(method).any()
            else {}
        )
        delta_row = (
            delta_frontier[delta_frontier["method"].astype(str).eq(method)].iloc[0].to_dict()
            if not delta_frontier.empty
            and "method" in delta_frontier
            and delta_frontier["method"].astype(str).eq(method).any()
            else {}
        )
        dominated = bool_value(structural_row.get("structurally_dominated"))
        frontier_member = not dominated if structural_row else None
        safety_score = maybe_float(structural_row.get("structural_safety_score"))
        total_delta = maybe_float(delta_row.get("total_relative_delta_norm"))
        score = float(base.get("base_priority", 0.0))
        if safety_score is not None:
            score += 0.30 * safety_score
        if total_delta is not None:
            score += 0.05 * max(0.0, min(1.0, (0.30 - total_delta) / 0.10))
        if frontier_member is True:
            score += 0.30
        if dominated:
            score -= 0.30
        candidates.append(
            {
                "method": method,
                "selection_score": score,
                "group_rules": group_path,
                "writer_context_command": writer_path,
                "structural_frontier_member": frontier_member,
                "structurally_dominated": dominated if structural_row else None,
                "structural_safety_score": safety_score,
                "total_relative_delta_norm": total_delta,
                "base_priority": float(base.get("base_priority", 0.0)),
            }
        )
    if not candidates:
        fallback = known_feedback_base(DEFAULT_FEEDBACK_CANDIDATE)
        if fallback is None:
            raise ValueError("No known feedback base is configured.")
        return {
            "candidate_method": DEFAULT_FEEDBACK_CANDIDATE,
            "requested_candidate_method": requested,
            "feedback_base_selection_status": "fallback_no_candidate_rows",
            "feedback_base_selection_reason": "no known feedback base was present in the eval gate rows",
            "feedback_base_structural_frontier_member": None,
            "feedback_base_structurally_dominated": None,
            "feedback_base_structural_safety_score": None,
            "group_rules": repo_path(fallback["group_rules"]),
            "writer_context_command": repo_path(fallback["writer_command"]),
            "candidate_rows_considered": [],
        }
    selected = max(candidates, key=lambda item: (item["selection_score"], item["base_priority"], item["method"]))
    reason = (
        "selected highest-scoring known group-rule candidate using structural dominance; "
        f"frontier={selected['structural_frontier_member']}, dominated={selected['structurally_dominated']}"
    )
    return {
        "candidate_method": selected["method"],
        "requested_candidate_method": requested,
        "feedback_base_selection_status": "auto_selected",
        "feedback_base_selection_reason": reason,
        "feedback_base_structural_frontier_member": selected["structural_frontier_member"],
        "feedback_base_structurally_dominated": selected["structurally_dominated"],
        "feedback_base_structural_safety_score": selected["structural_safety_score"],
        "group_rules": selected["group_rules"],
        "writer_context_command": selected["writer_context_command"],
        "candidate_rows_considered": [
            {
                "method": item["method"],
                "selection_score": item["selection_score"],
                "structural_frontier_member": item["structural_frontier_member"],
                "structurally_dominated": item["structurally_dominated"],
                "structural_safety_score": item["structural_safety_score"],
                "total_relative_delta_norm": item["total_relative_delta_norm"],
            }
            for item in sorted(candidates, key=lambda row: row["selection_score"], reverse=True)
        ],
    }


def normalize_group_rule_columns(groups: pd.DataFrame) -> pd.DataFrame:
    groups = groups.copy()
    aliases = {
        "selected_scale": ["mechanistic_selected_scale", "subspace_extra_scale", "prior_scale"],
        "selected_weight_coder": [
            "mechanistic_weight_coder",
            "subspace_adjusted_weight_coder",
            "prior_weight_coder",
            "unified_weight_coder",
        ],
        "selected_weight_instruct": [
            "mechanistic_weight_instruct",
            "prior_weight_instruct",
            "unified_weight_instruct",
        ],
        "original_weight_coder": ["unified_weight_coder", "prior_weight_coder"],
        "original_weight_instruct": ["unified_weight_instruct", "prior_weight_instruct"],
        "selected_expected_max_relative_delta_norm": [
            "mechanistic_expected_max_relative_delta_norm",
            "unified_expected_max_relative_delta_norm",
        ],
    }
    for target, sources in aliases.items():
        if target in groups.columns:
            continue
        for source in sources:
            if source in groups.columns:
                groups[target] = groups[source]
                break
    if "audit_max_relative_delta_norm" not in groups.columns and {
        "selected_expected_max_relative_delta_norm",
        "selected_scale",
    }.issubset(groups.columns):
        expected = pd.to_numeric(
            groups["selected_expected_max_relative_delta_norm"], errors="coerce"
        ).fillna(0.0)
        scale = pd.to_numeric(groups["selected_scale"], errors="coerce").fillna(1.0).clip(lower=EPS)
        groups["audit_max_relative_delta_norm"] = expected / scale
    if "mechanism_risk_score" not in groups.columns:
        risk_columns = [
            "interference_score",
            "curvature_score",
            "feature_subspace_conflict",
            "feature_expert_route_geometry",
            "feature_expert_internal_geometry",
            "feature_router_instability",
            "route_mass_pressure",
            "load_pressure",
            "delta_pressure",
        ]
        present = [column for column in risk_columns if column in groups.columns]
        if present:
            risk = pd.Series(0.0, index=groups.index)
            weights = {
                "interference_score": 0.22,
                "curvature_score": 0.18,
                "feature_subspace_conflict": 0.14,
                "feature_expert_route_geometry": 0.12,
                "feature_expert_internal_geometry": 0.12,
                "feature_router_instability": 0.10,
                "route_mass_pressure": 0.05,
                "load_pressure": 0.04,
                "delta_pressure": 0.03,
            }
            used_weight = 0.0
            for column in present:
                weight = weights[column]
                risk += weight * pd.to_numeric(groups[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
                used_weight += weight
            groups["mechanism_risk_score"] = (risk / max(EPS, used_weight)).clip(0.0, 1.0)
    return groups


def load_group_rules(group_rules: Path, expert_context: Path) -> pd.DataFrame:
    groups = normalize_group_rule_columns(pd.read_csv(repo_path(group_rules)))
    if "tensor_pattern" in groups.columns:
        groups["tensor_pattern"] = groups["tensor_pattern"].fillna("").astype(str)
        groups = groups[groups["tensor_pattern"].str.len() > 0].copy()
    context = pd.read_csv(repo_path(expert_context))
    keep = [
        "layer_id",
        "expert_id",
        "dominant_category",
        "route_mass_instruct",
        "route_mass_coder",
        "max_dominant_category_share",
        "min_categories_observed",
        "max_topk_over_uniform",
        "mean_topk_over_uniform",
    ]
    present = [column for column in keep if column in context.columns]
    if present:
        groups = groups.merge(context[present], on=["layer_id", "expert_id"], how="left")
    if "dominant_category" not in groups.columns:
        groups["dominant_category"] = ""
    groups["dominant_category"] = groups["dominant_category"].fillna("")
    fallback_category = groups["dominant_source"].map(lambda value: "code" if str(value) == "coder" else "general")
    groups["dominant_category"] = groups["dominant_category"].where(groups["dominant_category"].astype(str).str.len() > 0, fallback_category)
    defaults = {
        "selected_scale": 1.0,
        "selected_weight_coder": 0.0,
        "selected_weight_instruct": 0.0,
        "original_weight_coder": 0.0,
        "original_weight_instruct": 0.0,
        "audit_max_relative_delta_norm": 0.0,
        "selected_expected_max_relative_delta_norm": 0.0,
        "mechanism_risk_score": 0.0,
        "feature_subspace_conflict": 0.0,
        "feature_expert_route_geometry": 0.0,
        "feature_router_instability": 0.0,
        "feature_load_pressure": 0.0,
        "total_topk_fraction": 0.0,
        "route_mass_coder": 0.0,
        "route_mass_instruct": 0.0,
    }
    for column, default in defaults.items():
        if column not in groups.columns:
            groups[column] = default
        groups[column] = pd.to_numeric(groups[column], errors="coerce").fillna(default)
    if "tensor_pattern" not in groups.columns:
        raise ValueError("group rules must include tensor_pattern")
    return groups


def task_exposure(groups: pd.DataFrame, family: str) -> pd.Series:
    category = groups["dominant_category"].astype(str)
    exact = (category == family).astype(float)
    if family == "general":
        related = category.isin(["math", "safety"]).astype(float) * 0.35
    elif family in {"math", "safety"}:
        related = (category == "general").astype(float) * 0.25
    else:
        related = (groups["dominant_source"].astype(str) == "coder").astype(float) * 0.25
    category_weight = pd.concat([exact, related], axis=1).max(axis=1).clip(0.05, 1.0)
    route_weight = (0.35 + 0.65 * robust01(groups["total_topk_fraction"])).clip(0.35, 1.0)
    return (category_weight * route_weight).clip(0.0, 1.0)


def apply_feedback_to_groups(
    groups: pd.DataFrame,
    task_feedback: pd.DataFrame,
    *,
    hard_cap: float,
    max_restore_step: float,
    max_shrink_step: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = groups.copy()
    base_scale = out["selected_scale"].clip(0.0, 1.0)
    scale = base_scale.copy()
    action_parts = pd.Series("", index=out.index, dtype=object)
    risk = out["mechanism_risk_score"].clip(0.0, 1.0)
    coder_present = (out["original_weight_coder"].abs() > EPS).astype(float)

    update_rows = []
    for _, row in task_feedback.iterrows():
        if row["feedback_status"] != "source_frontier_regression":
            update_rows.append(
                {
                    "task": row["task"],
                    "family": row["family"],
                    "action": "no_parameter_update",
                    "pressure": float(row["feedback_pressure"]),
                    "affected_group_count": 0,
                    "mean_scale_delta": 0.0,
                }
            )
            continue
        family = str(row["family"])
        exposure = task_exposure(out, family)
        before = scale.copy()
        pressure = float(row["feedback_pressure"])
        if row["regression_action"] == "restore_coder_on_code_family":
            coder_source = (out["dominant_source"].astype(str) == "coder").astype(float)
            restore = max_restore_step * pressure * exposure * (0.45 + 0.55 * coder_source) * (1.0 - 0.45 * risk)
            restore = restore.clip(lower=0.0) * coder_present
            scale = (scale + restore).clip(0.0, 1.0)
            changed = restore > 1e-9
            action_parts.loc[changed] = action_parts.loc[changed].map(
                lambda text: (text + "|restore_coder_for_code_regression").strip("|")
            )
            action = "restore_coder_for_code_regression"
        else:
            shrink = max_shrink_step * pressure * exposure * (0.35 + 0.65 * risk)
            shrink = shrink.clip(lower=0.0, upper=0.95) * coder_present
            scale = (scale * (1.0 - shrink)).clip(0.0, 1.0)
            changed = shrink > 1e-9
            action_parts.loc[changed] = action_parts.loc[changed].map(
                lambda text: (text + "|shrink_coder_for_source_regression").strip("|")
            )
            action = "shrink_coder_for_source_regression"
        delta = scale - before
        update_rows.append(
            {
                "task": row["task"],
                "family": family,
                "action": action,
                "pressure": pressure,
                "affected_group_count": int((delta.abs() > 1e-9).sum()),
                "mean_scale_delta": float(delta.mean()),
                "route_mass_weighted_scale_delta": float(
                    (out["total_topk_fraction"].clip(lower=0.0) * delta).sum()
                    / max(EPS, float(out["total_topk_fraction"].clip(lower=0.0).sum()))
                ),
            }
        )

    cap_scale = pd.Series(1.0, index=out.index)
    audit_max = out["audit_max_relative_delta_norm"].clip(lower=0.0)
    over_cap = audit_max * scale > hard_cap + 1e-9
    cap_scale.loc[over_cap] = hard_cap / audit_max.loc[over_cap].clip(lower=EPS)
    capped_before = scale.copy()
    scale = pd.concat([scale, cap_scale], axis=1).min(axis=1).clip(0.0, 1.0)
    capped = scale < capped_before - 1e-9
    action_parts.loc[capped] = action_parts.loc[capped].map(lambda text: (text + "|hard_cap_enforced").strip("|"))

    out["feedback_selected_scale"] = scale
    out["feedback_scale_delta"] = scale - base_scale
    out["feedback_selected_weight_coder"] = out["original_weight_coder"] * scale
    out["feedback_selected_weight_instruct"] = out["selected_weight_instruct"]
    out["feedback_expected_max_relative_delta_norm"] = out["audit_max_relative_delta_norm"] * scale
    out["feedback_action"] = action_parts.where(action_parts.astype(str).str.len() > 0, "unchanged")
    out["feedback_tensor_rule"] = out.apply(
        lambda item: (
            f"{item['tensor_pattern']}::"
            f"{NONBASE_SOURCE}={float(item['feedback_selected_weight_coder']):.6g},"
            f"{BASE_SOURCE}={float(item['feedback_selected_weight_instruct']):.6g}"
        ),
        axis=1,
    )
    return out, pd.DataFrame(update_rows)


def feedback_metrics(groups: pd.DataFrame, *, hard_cap: float) -> dict[str, Any]:
    route_mass = groups["total_topk_fraction"].clip(lower=0.0)
    original_nonbase = groups["selected_weight_coder"].abs().clip(lower=0.0)
    feedback_nonbase = groups["feedback_selected_weight_coder"].abs().clip(lower=0.0)
    original_mass = float((route_mass * original_nonbase).sum())
    feedback_mass = float((route_mass * feedback_nonbase).sum())
    changed = groups["feedback_scale_delta"].abs() > 1e-9
    return {
        "changed_group_count": int(changed.sum()),
        "mean_feedback_scale_delta": float(groups["feedback_scale_delta"].mean()),
        "min_feedback_scale_delta": float(groups["feedback_scale_delta"].min()),
        "max_feedback_scale_delta": float(groups["feedback_scale_delta"].max()),
        "route_mass_weighted_nonbase_before": original_mass,
        "route_mass_weighted_nonbase_after": feedback_mass,
        "route_mass_weighted_nonbase_ratio": feedback_mass / max(EPS, original_mass),
        "max_feedback_expected_relative_delta": float(groups["feedback_expected_max_relative_delta_norm"].max()),
        "groups_over_hard_cap_after_feedback": int(
            (groups["feedback_expected_max_relative_delta_norm"] > hard_cap + 1e-9).sum()
        ),
    }


def extract_writer_context(command_path: Path) -> tuple[str, dict[str, str]]:
    command = repo_path(command_path).read_text(encoding="utf-8").strip()
    parts = shlex.split(command)
    base = ""
    sources: dict[str, str] = {}
    idx = 0
    while idx < len(parts):
        if parts[idx] == "--base" and idx + 1 < len(parts):
            base = parts[idx + 1]
            idx += 2
            continue
        if parts[idx] == "--source" and idx + 1 < len(parts):
            raw = parts[idx + 1]
            if "=" in raw:
                name, path = raw.split("=", 1)
                sources[name] = path
            idx += 2
            continue
        idx += 1
    if not base or BASE_SOURCE not in sources or NONBASE_SOURCE not in sources:
        raise ValueError(f"Could not recover base/source paths from {command_path}")
    return base, sources


def write_tensor_rules_and_commands(groups: pd.DataFrame, output_dir: Path, writer_context_command: Path) -> dict[str, str]:
    output_dir = repo_path(output_dir)
    tensor_rules = output_dir / "tensor_rules.txt"
    with tensor_rules.open("w", encoding="utf-8") as handle:
        handle.write("# Feedback-adjusted Qwen3 MoE expert rules.\n")
        handle.write("# Router remains frozen; feedback only changes routed-expert coder deltas.\n")
        for _, row in groups.sort_values(["layer_id", "expert_id"]).iterrows():
            handle.write(str(row["feedback_tensor_rule"]) + "\n")
    base_path, sources = extract_writer_context(writer_context_command)
    checkpoint_output_dir = "results/checkpoints/qwen3_moe_feedback_mechanism_candidate"
    command = (
        "python scripts/write_same_shape_average_checkpoint.py "
        f"--base {shlex.quote(base_path)} "
        f"--source {BASE_SOURCE}={shlex.quote(sources[BASE_SOURCE])} "
        f"--source {NONBASE_SOURCE}={shlex.quote(sources[NONBASE_SOURCE])} "
        f"--source-weight {BASE_SOURCE}=0.0 --source-weight {NONBASE_SOURCE}=0.0 "
        f"--freeze-router --tensor-rule-file {rel(tensor_rules)} "
        f"--output-dir {checkpoint_output_dir}"
    )
    writer_command = output_dir / "writer_command.txt"
    dry_run_command = output_dir / "dry_run_command.txt"
    writer_command.write_text(command + "\n", encoding="utf-8")
    dry_run_command.write_text(command + " --dry-run\n", encoding="utf-8")
    return {
        "tensor_rules": rel(tensor_rules),
        "writer_command": rel(writer_command),
        "dry_run_command": rel(dry_run_command),
        "checkpoint_output_dir": checkpoint_output_dir,
    }


def build_report(summary: dict[str, Any], task_feedback: pd.DataFrame, updates: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE Feedback Optimizer",
        "",
        "This stage is the downstream-feedback half of the MoE averaging rule. It does not pick an algorithm name; it converts scored vLLM source-frontier regressions into bounded expert-rule updates.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Candidate method: `{summary['candidate_method']}`",
        f"- Feedback base selection: `{summary['feedback_base_selection_status']}`",
        f"- Feedback base frontier/dominated: `{summary['feedback_base_structural_frontier_member']}` / `{summary['feedback_base_structurally_dominated']}`",
        f"- Feedback base candidates considered: `{len(summary.get('feedback_base_candidate_rows_considered', []))}`",
        f"- Scored tasks: `{summary['scored_task_count']}/{summary['task_count']}`",
        f"- Regression tasks: `{summary['regression_task_count']}`",
        f"- Changed expert groups: `{summary['changed_group_count']}`",
        f"- Materialization gate: `{summary['materialization_gate']}`",
        f"- Route-mass nonbase ratio after feedback: `{fmt(summary['route_mass_weighted_nonbase_ratio'])}`",
        f"- Max expected relative delta after feedback: `{fmt(summary['max_feedback_expected_relative_delta'])}`",
        f"- Groups over hard cap after feedback: `{summary['groups_over_hard_cap_after_feedback']}`",
        "",
        "## Task Feedback",
        "",
        "| task | family | status | candidate | source frontier | delta | pressure | action | paired net |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for _, row in task_feedback.iterrows():
        lines.append(
            f"| `{row['task']}` | `{row['family']}` | `{row['feedback_status']}` | "
            f"{fmt(row['candidate_score'])} | {fmt(row['source_frontier_score'])} | "
            f"{fmt(row['delta_vs_source_frontier'])} | {fmt(row['feedback_pressure'])} | "
            f"`{row['regression_action']}` | {fmt(row['paired_net_delta'])} |"
        )
    base_candidates = summary.get("feedback_base_candidate_rows_considered") or []
    if base_candidates:
        lines.extend(
            [
                "",
                "## Feedback Base Candidates",
                "",
                "| method | selection score | frontier | dominated | structural safety | total relative delta |",
                "| --- | ---: | --- | --- | ---: | ---: |",
            ]
        )
        for row in base_candidates:
            lines.append(
                f"| `{row['method']}` | {fmt(row['selection_score'])} | "
                f"`{row['structural_frontier_member']}` | `{row['structurally_dominated']}` | "
                f"{fmt(row['structural_safety_score'])} | {fmt(row['total_relative_delta_norm'])} |"
            )
    lines.extend(
        [
            "",
            "## Feature Updates",
            "",
            "| task | action | affected groups | mean scale delta |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for _, row in updates.iterrows():
        lines.append(
            f"| `{row['task']}` | `{row['action']}` | {int(row['affected_group_count'])} | "
            f"{fmt(float(row['mean_scale_delta']))} |"
        )
    lines.extend(["", "## Outputs", ""])
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def run_real(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_method_rows(args.gate_dir, args.audit_dir)
    feedback_base = resolve_feedback_base(args, rows)
    task_feedback = build_task_feedback(
        rows,
        candidate_method=feedback_base["candidate_method"],
        regression_margin=args.regression_margin,
        improvement_margin=args.improvement_margin,
        pressure_scale=args.pressure_scale,
    )
    groups = load_group_rules(feedback_base["group_rules"], args.expert_context)
    feedback_groups, updates = apply_feedback_to_groups(
        groups,
        task_feedback,
        hard_cap=args.hard_cap,
        max_restore_step=args.max_restore_step,
        max_shrink_step=args.max_shrink_step,
    )
    metrics = feedback_metrics(feedback_groups, hard_cap=args.hard_cap)
    scored = task_feedback[task_feedback["feedback_status"] != "awaiting_eval"]
    regressions = task_feedback[task_feedback["feedback_status"] == "source_frontier_regression"]
    if scored.empty:
        status = "awaiting_eval"
    elif not regressions.empty and metrics["changed_group_count"] > 0:
        status = "feedback_rules_ready"
    else:
        status = "no_feedback_change_needed"
    task_path = output_dir / "task_feedback.csv"
    update_path = output_dir / "feature_update_summary.csv"
    group_path = output_dir / "feedback_group_rules.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    task_feedback.to_csv(task_path, index=False)
    updates.to_csv(update_path, index=False)
    feedback_groups.to_csv(group_path, index=False)
    artifacts = write_tensor_rules_and_commands(
        feedback_groups,
        output_dir,
        feedback_base["writer_context_command"],
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "candidate_method": feedback_base["candidate_method"],
        "requested_candidate_method": feedback_base["requested_candidate_method"],
        "feedback_base_selection_status": feedback_base["feedback_base_selection_status"],
        "feedback_base_selection_reason": feedback_base["feedback_base_selection_reason"],
        "feedback_base_structural_frontier_member": feedback_base[
            "feedback_base_structural_frontier_member"
        ],
        "feedback_base_structurally_dominated": feedback_base["feedback_base_structurally_dominated"],
        "feedback_base_structural_safety_score": feedback_base[
            "feedback_base_structural_safety_score"
        ],
        "feedback_base_candidate_rows_considered": feedback_base["candidate_rows_considered"],
        "group_rules_source": rel(feedback_base["group_rules"]),
        "writer_context_command_source": rel(feedback_base["writer_context_command"]),
        "materialization_gate": "materialize_feedback_candidate"
        if status == "feedback_rules_ready"
        else "do_not_materialize_feedback_candidate_yet",
        "task_count": int(len(task_feedback)),
        "scored_task_count": int(len(scored)),
        "regression_task_count": int(len(regressions)),
        "hard_cap": float(args.hard_cap),
        **metrics,
        "outputs": {
            "task_feedback": rel(task_path),
            "feature_update_summary": rel(update_path),
            "feedback_group_rules": rel(group_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
            **artifacts,
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, task_feedback, updates), encoding="utf-8")
    print(f"Wrote Qwen3 MoE feedback optimizer to {output_dir.resolve()}")
    print(f"Status: {summary['status']}; regressions={summary['regression_task_count']}; changed_groups={summary['changed_group_count']}")


def write_smoke_eval_output(root: Path, *, scores: dict[str, float], predictions: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    model_summary = {
        "avg_primary_score": sum(scores.values()) / max(1, len(scores)),
        "worst_primary_score": min(scores.values()) if scores else None,
        **{TASKS[task]["score_column"]: score for task, score in scores.items()},
    }
    metrics = []
    for task, score in scores.items():
        row: dict[str, Any] = {
            "task": task,
            "examples": 2,
            "strict_exact": None,
            "accuracy": None,
            "policy_accuracy": None,
            "compile_rate": None,
        }
        if task == "gsm8k":
            row["strict_exact"] = score
        elif task == "mmlu":
            row["accuracy"] = score
        elif task == "safety":
            row["policy_accuracy"] = score
        elif task == "humaneval_compile":
            row["compile_rate"] = score
        metrics.append(row)
    pd.DataFrame([model_summary]).to_csv(root / "model_summary.csv", index=False)
    pd.DataFrame(metrics).to_csv(root / "metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(root / "predictions.csv", index=False)
    (root / "summary.json").write_text(json.dumps({"status": "complete"}, indent=2) + "\n", encoding="utf-8")


def integration_smoke_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="qwen3_moe_feedback_smoke_") as temp:
        root = Path(temp)
        gate_dir = root / "gate"
        audit_dir = root / "audit"
        eval_root = root / "eval"
        gate_dir.mkdir()
        audit_dir.mkdir()
        methods = {
            "source_qwen3_30b_instruct": {
                "scores": {
                    "gsm8k": 0.80,
                    "mmlu": 0.80,
                    "safety": 0.90,
                    "humaneval_compile": 0.20,
                },
                "predictions": [
                    {"task": "safety", "task_id": "s0", "expected_refusal": True, "refused": True},
                    {"task": "safety", "task_id": "s1", "expected_refusal": True, "refused": True},
                ],
            },
            "source_qwen3_30b_coder": {
                "scores": {
                    "gsm8k": 0.30,
                    "mmlu": 0.40,
                    "safety": 0.20,
                    "humaneval_compile": 0.80,
                },
                "predictions": [
                    {"task": "humaneval_compile", "task_id": "h0", "compile_ok": True},
                    {"task": "humaneval_compile", "task_id": "h1", "compile_ok": True},
                ],
            },
            DEFAULT_FEEDBACK_CANDIDATE: {
                "scores": {
                    "gsm8k": 0.82,
                    "mmlu": 0.79,
                    "safety": 0.65,
                    "humaneval_compile": 0.65,
                },
                "predictions": [
                    {"task": "safety", "task_id": "s0", "expected_refusal": True, "refused": False},
                    {"task": "safety", "task_id": "s1", "expected_refusal": True, "refused": True},
                    {"task": "humaneval_compile", "task_id": "h0", "compile_ok": False},
                    {"task": "humaneval_compile", "task_id": "h1", "compile_ok": True},
                ],
            },
        }
        gate_rows = []
        audit_rows = []
        for method, payload in methods.items():
            output_dir = eval_root / method
            write_smoke_eval_output(output_dir, scores=payload["scores"], predictions=payload["predictions"])
            gate_rows.append(
                {
                    "method": method,
                    "role": "source" if method in SOURCE_METHODS else "candidate",
                    "eval_output_dir": str(output_dir),
                }
            )
            audit_rows.append({"method": method, "usable_for_selection": True})
        pd.DataFrame(gate_rows).to_csv(gate_dir / "eval_gate_plan.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(audit_dir / "audit_rows.csv", index=False)

        loaded = load_method_rows(gate_dir, audit_dir)
        task_feedback = build_task_feedback(
            loaded,
            candidate_method=DEFAULT_FEEDBACK_CANDIDATE,
            regression_margin=args.regression_margin,
            improvement_margin=args.improvement_margin,
            pressure_scale=args.pressure_scale,
        )
        groups = pd.DataFrame(
            [
                {
                    "layer_id": 0,
                    "expert_id": 0,
                    "dominant_source": "coder",
                    "dominant_category": "code",
                    "original_weight_instruct": 0.10,
                    "original_weight_coder": 0.80,
                    "selected_weight_instruct": 0.10,
                    "selected_weight_coder": 0.40,
                    "selected_scale": 0.50,
                    "audit_max_relative_delta_norm": 0.80,
                    "selected_expected_max_relative_delta_norm": 0.40,
                    "mechanism_risk_score": 0.20,
                    "total_topk_fraction": 0.40,
                    "tensor_pattern": ".*layers\\.0\\..*experts\\.0\\..*",
                },
                {
                    "layer_id": 0,
                    "expert_id": 1,
                    "dominant_source": "instruct",
                    "dominant_category": "safety",
                    "original_weight_instruct": 0.75,
                    "original_weight_coder": 0.15,
                    "selected_weight_instruct": 0.75,
                    "selected_weight_coder": 0.15,
                    "selected_scale": 1.00,
                    "audit_max_relative_delta_norm": 0.30,
                    "selected_expected_max_relative_delta_norm": 0.30,
                    "mechanism_risk_score": 0.80,
                    "total_topk_fraction": 0.35,
                    "tensor_pattern": ".*layers\\.0\\..*experts\\.1\\..*",
                },
            ]
        )
        adjusted, _ = apply_feedback_to_groups(
            groups,
            task_feedback,
            hard_cap=args.hard_cap,
            max_restore_step=args.max_restore_step,
            max_shrink_step=args.max_shrink_step,
        )
        metrics = feedback_metrics(adjusted, hard_cap=args.hard_cap)
        regressions = task_feedback[task_feedback["feedback_status"] == "source_frontier_regression"]
        materialization_gate = (
            "materialize_feedback_candidate"
            if not regressions.empty and metrics["changed_group_count"] > 0
            else "do_not_materialize_feedback_candidate_yet"
        )
        code_scale = float(adjusted.loc[adjusted["expert_id"] == 0, "feedback_selected_scale"].iloc[0])
        safety_scale = float(adjusted.loc[adjusted["expert_id"] == 1, "feedback_selected_scale"].iloc[0])
        human = task_feedback[task_feedback["task"] == "humaneval_compile"].iloc[0]
        safety = task_feedback[task_feedback["task"] == "safety"].iloc[0]
        return [
            {
                "case": "integration_eval_bundle",
                "assertion": "humaneval_regression_detected",
                "expected": "source_frontier_regression",
                "actual": str(human["feedback_status"]),
                "passed": str(human["feedback_status"]) == "source_frontier_regression",
            },
            {
                "case": "integration_eval_bundle",
                "assertion": "safety_regression_detected",
                "expected": "source_frontier_regression",
                "actual": str(safety["feedback_status"]),
                "passed": str(safety["feedback_status"]) == "source_frontier_regression",
            },
            {
                "case": "integration_eval_bundle",
                "assertion": "paired_predictions_loaded",
                "expected": "negative humaneval paired delta",
                "actual": fmt(human["paired_net_delta"]),
                "passed": maybe_float(human["paired_net_delta"]) is not None
                and float(human["paired_net_delta"]) < 0.0,
            },
            {
                "case": "integration_eval_bundle",
                "assertion": "code_scale_restored",
                "expected": ">0.50",
                "actual": f"{code_scale:.6f}",
                "passed": code_scale > 0.50,
            },
            {
                "case": "integration_eval_bundle",
                "assertion": "safety_scale_shrunk",
                "expected": "<1.00",
                "actual": f"{safety_scale:.6f}",
                "passed": safety_scale < 1.00,
            },
            {
                "case": "integration_eval_bundle",
                "assertion": "materialization_gate_opens",
                "expected": "materialize_feedback_candidate",
                "actual": materialization_gate,
                "passed": materialization_gate == "materialize_feedback_candidate",
            },
        ]


def run_smoke(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = pd.DataFrame(
        [
            {
                "layer_id": 0,
                "expert_id": 0,
                "dominant_source": "coder",
                "dominant_category": "code",
                "original_weight_instruct": 0.10,
                "original_weight_coder": 0.80,
                "selected_weight_instruct": 0.10,
                "selected_weight_coder": 0.40,
                "selected_scale": 0.50,
                "audit_max_relative_delta_norm": 0.80,
                "selected_expected_max_relative_delta_norm": 0.40,
                "mechanism_risk_score": 0.20,
                "total_topk_fraction": 0.40,
                "tensor_pattern": ".*layers\\.0\\..*experts\\.0\\..*",
            },
            {
                "layer_id": 0,
                "expert_id": 1,
                "dominant_source": "coder",
                "dominant_category": "math",
                "original_weight_instruct": 0.40,
                "original_weight_coder": 0.40,
                "selected_weight_instruct": 0.40,
                "selected_weight_coder": 0.32,
                "selected_scale": 0.80,
                "audit_max_relative_delta_norm": 0.60,
                "selected_expected_max_relative_delta_norm": 0.48,
                "mechanism_risk_score": 0.90,
                "total_topk_fraction": 0.30,
                "tensor_pattern": ".*layers\\.0\\..*experts\\.1\\..*",
            },
            {
                "layer_id": 0,
                "expert_id": 2,
                "dominant_source": "instruct",
                "dominant_category": "safety",
                "original_weight_instruct": 0.80,
                "original_weight_coder": 0.05,
                "selected_weight_instruct": 0.80,
                "selected_weight_coder": 0.05,
                "selected_scale": 1.00,
                "audit_max_relative_delta_norm": 0.10,
                "selected_expected_max_relative_delta_norm": 0.10,
                "mechanism_risk_score": 0.10,
                "total_topk_fraction": 0.20,
                "tensor_pattern": ".*layers\\.0\\..*experts\\.2\\..*",
            },
        ]
    )
    feedback = pd.DataFrame(
        [
            {
                "task": "humaneval_compile",
                "family": "code",
                "feedback_status": "source_frontier_regression",
                "feedback_pressure": 1.0,
                "regression_action": "restore_coder_on_code_family",
            },
            {
                "task": "gsm8k",
                "family": "math",
                "feedback_status": "source_frontier_regression",
                "feedback_pressure": 1.0,
                "regression_action": "shrink_nonbase_on_task_family",
            },
        ]
    )
    adjusted, updates = apply_feedback_to_groups(
        groups,
        feedback,
        hard_cap=0.65,
        max_restore_step=args.max_restore_step,
        max_shrink_step=args.max_shrink_step,
    )
    code_scale = float(adjusted.loc[adjusted["expert_id"] == 0, "feedback_selected_scale"].iloc[0])
    math_scale = float(adjusted.loc[adjusted["expert_id"] == 1, "feedback_selected_scale"].iloc[0])
    max_delta = float(adjusted["feedback_expected_max_relative_delta_norm"].max())
    empty_feedback = pd.DataFrame(
        [
            {
                "task": "gsm8k",
                "family": "math",
                "feedback_status": "awaiting_eval",
                "feedback_pressure": 0.0,
                "regression_action": "none",
            }
        ]
    )
    empty_adjusted, _ = apply_feedback_to_groups(
        groups,
        empty_feedback,
        hard_cap=0.65,
        max_restore_step=args.max_restore_step,
        max_shrink_step=args.max_shrink_step,
    )
    mechanistic_rules = normalize_group_rule_columns(
        pd.DataFrame(
            [
                {
                    "layer_id": 1,
                    "expert_id": 7,
                    "dominant_source": "coder",
                    "dominant_category": "code",
                    "original_weight_instruct": 0.20,
                    "original_weight_coder": 0.70,
                    "mechanistic_selected_scale": 0.64,
                    "mechanistic_weight_instruct": 0.20,
                    "mechanistic_weight_coder": 0.448,
                    "mechanistic_expected_max_relative_delta_norm": 0.61,
                    "audit_max_relative_delta_norm": 0.95,
                    "interference_score": 0.70,
                    "curvature_score": 0.60,
                    "feature_subspace_conflict": 0.80,
                    "feature_expert_internal_geometry": 0.40,
                    "tensor_pattern": ".*layers\\.1\\..*experts\\.7\\..*",
                }
            ]
        )
    )
    auto_base = resolve_feedback_base(
        args,
        {
            "qwen3_moe_unified_mechanism_candidate": {},
            "qwen3_moe_mechanistic_unified_candidate": {},
            "qwen3_moe_subspace_scaled_candidate": {},
        },
    )
    auto_methods = {str(row["method"]) for row in auto_base["candidate_rows_considered"]}
    matrix = pd.DataFrame(
        [
            {
                "case": "code_regression_restore",
                "assertion": "code_scale_increases",
                "expected": ">0.50",
                "actual": f"{code_scale:.6f}",
                "passed": code_scale > 0.50,
            },
            {
                "case": "math_regression_shrink",
                "assertion": "math_scale_decreases",
                "expected": "<0.80",
                "actual": f"{math_scale:.6f}",
                "passed": math_scale < 0.80,
            },
            {
                "case": "hard_cap",
                "assertion": "max_delta_capped",
                "expected": "<=0.65",
                "actual": f"{max_delta:.6f}",
                "passed": max_delta <= 0.650000001,
            },
            {
                "case": "awaiting_eval",
                "assertion": "no_feedback_without_scores",
                "expected": "0 changed groups",
                "actual": str(int((empty_adjusted["feedback_scale_delta"].abs() > 1e-9).sum())),
                "passed": int((empty_adjusted["feedback_scale_delta"].abs() > 1e-9).sum()) == 0,
            },
            {
                "case": "mechanistic_group_rules",
                "assertion": "mechanistic_scale_normalized",
                "expected": "0.640000",
                "actual": f"{float(mechanistic_rules['selected_scale'].iloc[0]):.6f}",
                "passed": abs(float(mechanistic_rules["selected_scale"].iloc[0]) - 0.64) <= 1e-12,
            },
            {
                "case": "mechanistic_group_rules",
                "assertion": "mechanistic_weight_normalized",
                "expected": "0.448000",
                "actual": f"{float(mechanistic_rules['selected_weight_coder'].iloc[0]):.6f}",
                "passed": abs(float(mechanistic_rules["selected_weight_coder"].iloc[0]) - 0.448) <= 1e-12,
            },
            {
                "case": "auto_feedback_base",
                "assertion": "mechanistic_selected",
                "expected": "qwen3_moe_mechanistic_unified_candidate",
                "actual": str(auto_base["candidate_method"]),
                "passed": str(auto_base["candidate_method"]) == "qwen3_moe_mechanistic_unified_candidate",
            },
            {
                "case": "auto_feedback_base",
                "assertion": "subspace_candidate_considered",
                "expected": "True",
                "actual": str("qwen3_moe_subspace_scaled_candidate" in auto_methods),
                "passed": "qwen3_moe_subspace_scaled_candidate" in auto_methods,
            },
        ]
    )
    matrix = pd.concat([matrix, pd.DataFrame(integration_smoke_rows(args))], ignore_index=True)
    matrix_path = output_dir / "feedback_optimizer_smoke_matrix.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    matrix.to_csv(matrix_path, index=False)
    summary = {
        "schema_version": 1,
        "status": "passed" if bool(matrix["passed"].all()) else "failed",
        "case_count": int(len(matrix)),
        "passed_case_count": int(matrix["passed"].sum()),
        "failed_case_count": int((~matrix["passed"]).sum()),
        "outputs": {
            "matrix": rel(matrix_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    lines = [
        "# Qwen3 MoE Feedback Optimizer Smoke",
        "",
        f"- Status: `{summary['status']}`",
        f"- Cases passed: `{summary['passed_case_count']}/{summary['case_count']}`",
        "",
        "| case | assertion | expected | actual | passed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in matrix.iterrows():
        lines.append(f"| `{row['case']}` | `{row['assertion']}` | `{row['expected']}` | `{row['actual']}` | `{row['passed']}` |")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Qwen3 MoE feedback optimizer smoke to {output_dir.resolve()}")
    print(f"Status: {summary['status']}; cases={summary['passed_case_count']}/{summary['case_count']}")
    if summary["status"] != "passed":
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a downstream-feedback optimizer for Qwen3 MoE averaging.")
    parser.add_argument("--gate-dir", type=Path, default=Path("results/qwen3_moe_mechanism_eval_gate"))
    parser.add_argument("--audit-dir", type=Path, default=Path("results/qwen3_moe_eval_bundle_audit"))
    parser.add_argument(
        "--group-rules",
        "--unified-group-rules",
        dest="group_rules",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--expert-context",
        type=Path,
        default=Path("results/qwen3_moe_trust_region_candidate/trust_region_source_weights_by_expert.csv"),
    )
    parser.add_argument(
        "--writer-context-command",
        type=Path,
        default=None,
    )
    parser.add_argument("--candidate-method", default=AUTO_CANDIDATE)
    parser.add_argument(
        "--delta-frontier",
        type=Path,
        default=Path("results/qwen3_moe_delta_frontier/candidate_delta_frontier.csv"),
    )
    parser.add_argument(
        "--structural-dominance",
        type=Path,
        default=Path("results/qwen3_moe_delta_frontier/structural_dominance.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_feedback_optimizer"))
    parser.add_argument("--hard-cap", type=float, default=0.65)
    parser.add_argument("--regression-margin", type=float, default=0.02)
    parser.add_argument("--improvement-margin", type=float, default=0.02)
    parser.add_argument("--pressure-scale", type=float, default=0.12)
    parser.add_argument("--max-restore-step", type=float, default=0.10)
    parser.add_argument("--max-shrink-step", type=float, default=0.12)
    parser.add_argument("--smoke-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_matrix:
        run_smoke(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
