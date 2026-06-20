#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_METHODS = ["source_qwen3_30b_instruct", "source_qwen3_30b_coder"]
TASK_SCORE_COLUMNS = [
    "task_gsm8k_score",
    "task_mmlu_score",
    "task_safety_score",
    "task_humaneval_compile_score",
]
TASK_TO_SCORE_COLUMN = {
    "gsm8k": "task_gsm8k_score",
    "mmlu": "task_mmlu_score",
    "safety": "task_safety_score",
    "humaneval_compile": "task_humaneval_compile_score",
}
PAIRED_TASKS = set(TASK_TO_SCORE_COLUMN)
SCORE_COLUMNS = ["avg_primary_score", "worst_primary_score", *TASK_SCORE_COLUMNS]


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


def primary_metric(row: pd.Series | dict[str, Any]) -> tuple[str, float] | None:
    for column in ("strict_exact", "accuracy", "policy_accuracy", "compile_rate"):
        value = maybe_float(row.get(column))
        if value is not None:
            return column, value
    return None


def prediction_key(row: pd.Series | dict[str, Any]) -> str | None:
    task = str(row.get("task", "")).strip()
    if not task:
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
        task = str(row.get("task", "")).strip()
        if key is not None and correct is not None and task in PAIRED_TASKS:
            scores[key] = correct
    return scores


def wilson_interval(score: float | None, n: int | None, z: float) -> tuple[float | None, float | None]:
    if score is None or n is None or n <= 0:
        return None, None
    p = min(1.0, max(0.0, float(score)))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def aggregate_interval(
    scores: dict[str, float],
    bounds: dict[str, tuple[float | None, float | None]],
) -> dict[str, float | None]:
    if not scores:
        return {
            "avg_primary_lower": None,
            "avg_primary_upper": None,
            "worst_primary_lower": None,
            "worst_primary_upper": None,
        }
    lowers = [bounds[task][0] for task in scores if bounds.get(task, (None, None))[0] is not None]
    uppers = [bounds[task][1] for task in scores if bounds.get(task, (None, None))[1] is not None]
    if len(lowers) != len(scores) or len(uppers) != len(scores):
        return {
            "avg_primary_lower": None,
            "avg_primary_upper": None,
            "worst_primary_lower": None,
            "worst_primary_upper": None,
        }
    return {
        "avg_primary_lower": sum(lowers) / len(lowers),
        "avg_primary_upper": sum(uppers) / len(uppers),
        "worst_primary_lower": min(lowers),
        "worst_primary_upper": min(uppers),
    }


def read_eval_scores(eval_dir: str | Path, *, confidence_z: float) -> dict[str, Any]:
    root = repo_path(eval_dir)
    summary = read_json(root / "summary.json")
    model_summary = read_csv(root / "model_summary.csv")
    metrics = read_csv(root / "metrics.csv")
    prediction_scores = read_prediction_scores(root)
    status = str(summary.get("status", "missing"))
    model_row = model_summary.iloc[0].to_dict() if not model_summary.empty else {}
    task_scores: dict[str, float] = {}
    task_bounds: dict[str, tuple[float | None, float | None]] = {}
    task_examples: dict[str, int] = {}
    if not metrics.empty:
        for _, metric_row in metrics.iterrows():
            task = str(metric_row.get("task", ""))
            primary = primary_metric(metric_row)
            if primary is not None:
                task_scores[task] = primary[1]
                examples = clean_value(metric_row.get("examples"))
                n = int(examples) if examples is not None else None
                task_examples[task] = n or 0
                task_bounds[task] = wilson_interval(primary[1], n, confidence_z)
    intervals = aggregate_interval(task_scores, task_bounds)
    task_interval_values: dict[str, float | int | None] = {}
    for task, column in TASK_TO_SCORE_COLUMN.items():
        lower, upper = task_bounds.get(task, (None, None))
        task_interval_values[f"{column}_n"] = task_examples.get(task)
        task_interval_values[f"{column}_lower"] = lower
        task_interval_values[f"{column}_upper"] = upper
    return {
        "eval_status": status,
        "eval_completed": status == "complete",
        "avg_primary_score": maybe_float(model_row.get("avg_primary_score")),
        "worst_primary_score": maybe_float(model_row.get("worst_primary_score")),
        **intervals,
        "task_gsm8k_score": task_scores.get("gsm8k"),
        "task_mmlu_score": task_scores.get("mmlu"),
        "task_safety_score": task_scores.get("safety"),
        "task_humaneval_compile_score": task_scores.get("humaneval_compile"),
        **task_interval_values,
        "_prediction_scores": prediction_scores,
    }


def bundle_usable_map(audit_dir: Path) -> dict[str, bool]:
    rows = read_csv(repo_path(audit_dir) / "audit_rows.csv")
    if rows.empty:
        return {}
    return {str(row["method"]): bool(row.get("usable_for_selection", False)) for _, row in rows.iterrows()}


def score_pairs(left: pd.Series | dict[str, Any], right: pd.Series | dict[str, Any]) -> list[tuple[float, float]]:
    pairs = []
    for column in SCORE_COLUMNS:
        left_value = maybe_float(left.get(column))
        right_value = maybe_float(right.get(column))
        if left_value is not None and right_value is not None:
            pairs.append((left_value, right_value))
    return pairs


def dominates(left: pd.Series | dict[str, Any], right: pd.Series | dict[str, Any], eps: float = 1e-9) -> bool:
    pairs = score_pairs(left, right)
    if not pairs:
        return False
    return all(left_value >= right_value - eps for left_value, right_value in pairs) and any(
        left_value > right_value + eps for left_value, right_value in pairs
    )


def task_regressions(sources: pd.DataFrame, candidate: pd.Series, tolerance: float) -> list[str]:
    regressions = []
    for column in TASK_SCORE_COLUMNS:
        candidate_value = maybe_float(candidate.get(column))
        source_values = [maybe_float(value) for value in sources[column].tolist() if maybe_float(value) is not None]
        if candidate_value is None or len(source_values) < len(SOURCE_METHODS):
            continue
        if candidate_value < min(source_values) - tolerance:
            regressions.append(column)
    return regressions


def confidence_task_regressions(sources: pd.DataFrame, candidate: pd.Series, tolerance: float) -> list[str]:
    regressions = []
    for column in TASK_SCORE_COLUMNS:
        candidate_lower = maybe_float(candidate.get(f"{column}_lower"))
        if f"{column}_lower" not in sources:
            continue
        source_lowers = [
            maybe_float(value)
            for value in sources[f"{column}_lower"].tolist()
            if maybe_float(value) is not None
        ]
        if candidate_lower is None or len(source_lowers) < len(SOURCE_METHODS):
            continue
        if candidate_lower < min(source_lowers) - tolerance:
            regressions.append(column)
    return regressions


def interval_dominates(left: pd.Series | dict[str, Any], right: pd.Series | dict[str, Any], eps: float = 1e-9) -> bool:
    pairs = []
    for column in SCORE_COLUMNS:
        left_lower = maybe_float(left.get(f"{column}_lower"))
        right_upper = maybe_float(right.get(f"{column}_upper"))
        if left_lower is not None and right_upper is not None:
            pairs.append((left_lower, right_upper))
    if not pairs:
        return False
    return all(left_lower >= right_upper - eps for left_lower, right_upper in pairs) and any(
        left_lower > right_upper + eps for left_lower, right_upper in pairs
    )


def exact_paired_source_advantage_pvalue(candidate_only: int, source_only: int) -> float:
    if source_only <= candidate_only:
        return 1.0
    discordant = candidate_only + source_only
    if discordant <= 0:
        return 1.0
    if discordant > 1024:
        # Normal approximation to the one-sided sign test when exact summation would underflow.
        mean = discordant / 2.0
        variance = discordant / 4.0
        z = (candidate_only + 0.5 - mean) / math.sqrt(variance)
        return 0.5 * math.erfc(-z / math.sqrt(2.0))
    return sum(math.comb(discordant, i) for i in range(candidate_only + 1)) / (2.0**discordant)


def paired_task_regressions(
    sources: pd.DataFrame,
    candidate: pd.Series,
    *,
    tolerance_rate: float,
    alpha: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    candidate_scores = candidate.get("_prediction_scores")
    if not isinstance(candidate_scores, dict) or not candidate_scores:
        return [], []
    regressions: list[str] = []
    rows: list[dict[str, Any]] = []
    for _, source in sources.iterrows():
        source_scores = source.get("_prediction_scores")
        if not isinstance(source_scores, dict) or not source_scores:
            continue
        by_task: dict[str, list[str]] = {}
        shared_keys = sorted(set(candidate_scores) & set(source_scores))
        for key in shared_keys:
            task = key.split("::", 1)[0]
            if task in PAIRED_TASKS:
                by_task.setdefault(task, []).append(key)
        for task, keys in sorted(by_task.items()):
            candidate_only = sum(int(candidate_scores[key] and not source_scores[key]) for key in keys)
            source_only = sum(int(source_scores[key] and not candidate_scores[key]) for key in keys)
            shared = len(keys)
            net_delta = (candidate_only - source_only) / max(shared, 1)
            pvalue = exact_paired_source_advantage_pvalue(candidate_only, source_only)
            regression = source_only > candidate_only and net_delta < -tolerance_rate and pvalue <= alpha
            if regression:
                regressions.append(TASK_TO_SCORE_COLUMN[task])
            rows.append(
                {
                    "source": source.get("method"),
                    "task": task,
                    "shared": shared,
                    "candidate_only_correct": candidate_only,
                    "source_only_correct": source_only,
                    "net_delta": net_delta,
                    "pvalue": pvalue,
                    "regression": regression,
                }
            )
    return sorted(set(regressions)), rows


def load_real_rows(gate_dir: Path, audit_dir: Path, *, confidence_z: float) -> pd.DataFrame:
    gate = read_csv(repo_path(gate_dir) / "eval_gate_plan.csv")
    usable = bundle_usable_map(audit_dir)
    rows = []
    for _, row in gate.iterrows():
        item = row.to_dict()
        method = str(item["method"])
        item["eval_usable"] = bool(usable.get(method, False))
        if item["eval_usable"]:
            item.update(read_eval_scores(str(item.get("eval_output_dir", "")), confidence_z=confidence_z))
        else:
            item["eval_status"] = item.get("eval_status") or "not_usable"
            item["eval_completed"] = False
            for column in SCORE_COLUMNS:
                item[column] = None
            for column in ["avg_primary", "worst_primary", *TASK_SCORE_COLUMNS]:
                item[f"{column}_lower"] = None
                item[f"{column}_upper"] = None
            for column in TASK_SCORE_COLUMNS:
                item[f"{column}_n"] = None
        rows.append(item)
    return pd.DataFrame(rows)


def build_selection_table(
    rows: pd.DataFrame,
    *,
    task_regression_tolerance: float,
    use_uncertainty_gate: bool,
    use_paired_gate: bool,
    paired_loss_tolerance_rate: float,
    paired_alpha: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = rows.copy()
    sources = rows[rows["method"].isin(SOURCE_METHODS)].copy()
    candidates = rows[rows["role"] == "candidate"].copy()
    sources_complete = bool(len(sources) == len(SOURCE_METHODS) and sources["eval_usable"].astype(bool).all())
    candidates_complete = bool(not candidates.empty and candidates["eval_usable"].astype(bool).all())
    usable_candidates = candidates[candidates["eval_usable"].astype(bool)].copy()
    best_source_by_avg = None
    best_source_by_worst = None
    if not sources.empty and sources["eval_usable"].astype(bool).any():
        usable_sources = sources[sources["eval_usable"].astype(bool)].copy()
        if not usable_sources.empty:
            best_source_by_avg = usable_sources.sort_values(
                ["avg_primary_score", "worst_primary_score"], ascending=False
            ).iloc[0]["method"]
            best_source_by_worst = usable_sources.sort_values(
                ["worst_primary_score", "avg_primary_score"], ascending=False
            ).iloc[0]["method"]

    table_rows = []
    for _, row in rows.iterrows():
        is_candidate = str(row.get("role")) == "candidate"
        dominated_by = []
        confidence_dominated_by = []
        regressions = []
        confidence_regressions = []
        paired_regressions = []
        paired_rows: list[dict[str, Any]] = []
        eligible = False
        rejection_reasons = []
        if is_candidate:
            if not bool(row.get("eval_usable", False)):
                rejection_reasons.append("awaiting_candidate_eval")
            if not bool(row.get("audit_passed", False)):
                rejection_reasons.append("checkpoint_audit_failed")
            if sources_complete and bool(row.get("eval_usable", False)):
                for _, source in sources.iterrows():
                    if dominates(source, row):
                        dominated_by.append(str(source["method"]))
                    if use_uncertainty_gate and interval_dominates(source, row):
                        confidence_dominated_by.append(str(source["method"]))
                regressions = task_regressions(sources, row, task_regression_tolerance)
                if use_uncertainty_gate:
                    confidence_regressions = confidence_task_regressions(sources, row, task_regression_tolerance)
                if use_paired_gate:
                    paired_regressions, paired_rows = paired_task_regressions(
                        sources,
                        row,
                        tolerance_rate=paired_loss_tolerance_rate,
                        alpha=paired_alpha,
                    )
                if dominated_by:
                    rejection_reasons.append("source_endpoint_dominates")
                if confidence_dominated_by:
                    rejection_reasons.append("source_confidence_interval_dominates")
                if regressions:
                    rejection_reasons.append("task_score_regression")
                if confidence_regressions:
                    rejection_reasons.append("task_confidence_regression")
                if paired_regressions:
                    rejection_reasons.append("paired_prediction_regression")
            elif is_candidate and bool(row.get("eval_usable", False)):
                rejection_reasons.append("awaiting_source_eval")
            eligible = bool(
                is_candidate
                and sources_complete
                and bool(row.get("eval_usable", False))
                and bool(row.get("audit_passed", False))
                and not dominated_by
                and not confidence_dominated_by
                and not regressions
                and not confidence_regressions
                and not paired_regressions
            )
        paired_shared = sum(int(item["shared"]) for item in paired_rows)
        paired_candidate_only = sum(int(item["candidate_only_correct"]) for item in paired_rows)
        paired_source_only = sum(int(item["source_only_correct"]) for item in paired_rows)
        paired_min_pvalue = min([float(item["pvalue"]) for item in paired_rows], default=None)
        table_rows.append(
            {
                "method": row.get("method"),
                "role": row.get("role"),
                "eval_usable": bool(row.get("eval_usable", False)),
                "audit_passed": bool(row.get("audit_passed", False)),
                "avg_primary_score": row.get("avg_primary_score"),
                "worst_primary_score": row.get("worst_primary_score"),
                "avg_primary_lower": row.get("avg_primary_lower"),
                "avg_primary_upper": row.get("avg_primary_upper"),
                "worst_primary_lower": row.get("worst_primary_lower"),
                "worst_primary_upper": row.get("worst_primary_upper"),
                "task_gsm8k_score": row.get("task_gsm8k_score"),
                "task_mmlu_score": row.get("task_mmlu_score"),
                "task_safety_score": row.get("task_safety_score"),
                "task_humaneval_compile_score": row.get("task_humaneval_compile_score"),
                "total_relative_delta_norm": row.get("total_relative_delta_norm"),
                "routed_tensors_gt_0_75": row.get("routed_tensors_gt_0_75"),
                "dominated_by_source": ",".join(dominated_by),
                "confidence_dominated_by_source": ",".join(confidence_dominated_by),
                "task_regression_columns": ",".join(regressions),
                "task_confidence_regression_columns": ",".join(confidence_regressions),
                "task_paired_regression_columns": ",".join(paired_regressions),
                "paired_shared_examples": paired_shared if paired_rows else None,
                "paired_candidate_only_correct": paired_candidate_only if paired_rows else None,
                "paired_source_only_correct": paired_source_only if paired_rows else None,
                "paired_min_pvalue": paired_min_pvalue,
                "rejection_reasons": ",".join(rejection_reasons),
                "selection_eligible": eligible,
            }
        )
    table = pd.DataFrame(table_rows)
    eligible = table[table["selection_eligible"].astype(bool)].copy() if not table.empty else pd.DataFrame()
    if not eligible.empty:
        ranked = eligible.sort_values(
            ["avg_primary_score", "worst_primary_score", "total_relative_delta_norm"],
            ascending=[False, False, True],
        )
        selected = ranked.iloc[0].to_dict()
        status = "select_candidate" if candidates_complete else "provisional_candidate"
        selection = {
            "status": status,
            "selected_method": selected.get("method"),
            "reason": "candidate_survives_source_dominance_and_task_regression_gates",
        }
    elif not sources_complete:
        selection = {
            "status": "awaiting_source_eval",
            "selected_method": None,
            "reason": "Both Qwen3 source endpoints must complete audited vLLM eval before final candidate selection.",
        }
    elif not candidates_complete:
        selection = {
            "status": "awaiting_candidate_eval",
            "selected_method": best_source_by_avg,
            "reason": "Sources are usable but not all candidates have audited vLLM eval; current fallback is best source.",
        }
    else:
        selection = {
            "status": "keep_source_endpoint",
            "selected_method": best_source_by_avg,
            "reason": "no_candidate_survives_source_dominance_and_task_regression_gates",
        }
    selection.update(
        {
            "sources_complete": sources_complete,
            "candidates_complete": candidates_complete,
            "source_count": int(len(sources)),
            "candidate_count": int(len(candidates)),
            "usable_candidate_count": int(len(usable_candidates)),
            "eligible_candidate_count": int(len(eligible)),
            "best_source_by_avg": best_source_by_avg,
            "best_source_by_worst": best_source_by_worst,
            "score_columns": SCORE_COLUMNS,
            "uncertainty_gate": use_uncertainty_gate,
            "paired_prediction_gate": use_paired_gate,
            "paired_loss_tolerance_rate": paired_loss_tolerance_rate,
            "paired_alpha": paired_alpha,
        }
    )
    return table, selection


def build_report(summary: dict[str, Any], table: pd.DataFrame) -> str:
    def fmt(value: Any) -> str:
        value = clean_value(value)
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    selection = summary["current_selection"]
    lines = [
        "# Qwen3 MoE Final Candidate Selection",
        "",
        "这个 selector 在远端 vLLM eval 结果通过 bundle audit 后，对两个 source endpoint 和八个 same-shape Qwen3 MoE candidates 做最终选择。它不会只看内部 delta，也不会只看单个 unified 候选；candidate 必须同时通过 source dominance、task regression 和 checkpoint audit gate。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Selected: `{selection.get('selected_method')}`",
        f"- Reason: `{selection.get('reason')}`",
        f"- Sources complete: `{selection.get('sources_complete')}`",
        f"- Candidates complete: `{selection.get('candidates_complete')}`",
        f"- Eligible candidates: `{selection.get('eligible_candidate_count')}/{selection.get('candidate_count')}`",
        f"- Uncertainty gate: `{selection.get('uncertainty_gate')}`",
        f"- Paired prediction gate: `{selection.get('paired_prediction_gate')}`",
        f"- Paired alpha: `{selection.get('paired_alpha')}`",
        "",
        "| method | role | usable | audit | avg | avg CI | worst | worst CI | rel norm | dominated | conf dom | regressions | conf regressions | paired net | paired p | paired regressions | eligible |",
        "| --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for _, row in table.iterrows():
        paired_net = None
        paired_shared = clean_value(row.get("paired_shared_examples"))
        if paired_shared:
            paired_net = (
                int(clean_value(row.get("paired_candidate_only_correct")) or 0)
                - int(clean_value(row.get("paired_source_only_correct")) or 0)
            )
        lines.append(
            f"| `{row['method']}` | `{row['role']}` | `{bool(row['eval_usable'])}` | "
            f"`{bool(row['audit_passed'])}` | {fmt(row.get('avg_primary_score'))} | "
            f"[{fmt(row.get('avg_primary_lower'))}, {fmt(row.get('avg_primary_upper'))}] | "
            f"{fmt(row.get('worst_primary_score'))} | "
            f"[{fmt(row.get('worst_primary_lower'))}, {fmt(row.get('worst_primary_upper'))}] | "
            f"{fmt(row.get('total_relative_delta_norm'))} | "
            f"`{row.get('dominated_by_source', '')}` | "
            f"`{row.get('confidence_dominated_by_source', '')}` | "
            f"`{row.get('task_regression_columns', '')}` | "
            f"`{row.get('task_confidence_regression_columns', '')}` | "
            f"{fmt(paired_net)} | "
            f"{fmt(row.get('paired_min_pvalue'))} | "
            f"`{row.get('task_paired_regression_columns', '')}` | "
            f"`{bool(row.get('selection_eligible'))}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['selection_table']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['decision_rules']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    table: pd.DataFrame,
    selection: dict[str, Any],
    *,
    smoke_case: str | None = None,
) -> dict[str, Any]:
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "selection_table.csv"
    summary_path = output_dir / "summary.json"
    rules_path = output_dir / "decision_rules.json"
    report_path = output_dir / "report.md"
    table.to_csv(table_path, index=False)
    rules = {
        "schema_version": 1,
        "target_contract": "same-shape Qwen3 MoE output; no ensemble, no added experts, no architecture change",
        "required_controls": SOURCE_METHODS,
        "candidate_policy": [
            "both source endpoints must pass eval-bundle audit",
            "candidate must pass eval-bundle audit",
            "candidate checkpoint audit must pass",
            "candidate must not be dominated by either source endpoint",
            "candidate must not be dominated by either source endpoint after score confidence intervals",
            "candidate must not regress a task below both source endpoints",
            "candidate task confidence lower bound must not fall below both source lower bounds",
            "candidate must not have statistically significant paired prediction regression on shared eval examples",
            "rank eligible candidates by avg primary score, then worst primary score, then lower total relative delta norm",
        ],
    }
    summary = {
        "schema_version": 1,
        "status": selection["status"],
        "smoke_case": smoke_case,
        "current_selection": selection,
        "outputs": {
            "selection_table": rel(table_path),
            "decision_rules": rel(rules_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    rules_path.write_text(json.dumps(json_safe(rules), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, table), encoding="utf-8")
    return summary


def add_smoke_intervals(rows: list[dict[str, Any]], *, default_n: int = 128, z: float = 1.96) -> list[dict[str, Any]]:
    for row in rows:
        task_scores = {}
        task_bounds = {}
        for _, column in TASK_TO_SCORE_COLUMN.items():
            score = maybe_float(row.get(column))
            n = int(row.get(f"{column}_n") or default_n)
            row[f"{column}_n"] = n
            lower, upper = wilson_interval(score, n, z)
            row[f"{column}_lower"] = lower
            row[f"{column}_upper"] = upper
            if score is not None:
                task_scores[column] = score
                task_bounds[column] = (lower, upper)
        intervals = aggregate_interval(task_scores, task_bounds)
        row.update(intervals)
    return rows


def smoke_rows(case: str) -> pd.DataFrame:
    tasks = {
        "task_gsm8k_score": 0.55,
        "task_mmlu_score": 0.60,
        "task_safety_score": 0.70,
        "task_humaneval_compile_score": 0.35,
    }
    rows = [
        {
            "method": "source_qwen3_30b_instruct",
            "role": "source",
            "eval_usable": True,
            "audit_passed": True,
            "avg_primary_score": 0.62,
            "worst_primary_score": 0.40,
            **tasks,
        },
        {
            "method": "source_qwen3_30b_coder",
            "role": "source",
            "eval_usable": True,
            "audit_passed": True,
            "avg_primary_score": 0.58,
            "worst_primary_score": 0.42,
            "task_gsm8k_score": 0.50,
            "task_mmlu_score": 0.55,
            "task_safety_score": 0.45,
            "task_humaneval_compile_score": 0.60,
        },
        {
            "method": "qwen3_moe_tail_trimmed_expert_only_candidate",
            "role": "candidate",
            "eval_usable": True,
            "audit_passed": True,
            "avg_primary_score": 0.66,
            "worst_primary_score": 0.48,
            "task_gsm8k_score": 0.58,
            "task_mmlu_score": 0.61,
            "task_safety_score": 0.68,
            "task_humaneval_compile_score": 0.55,
            "total_relative_delta_norm": 0.243,
            "routed_tensors_gt_0_75": 0,
        },
        {
            "method": "qwen3_moe_searched_no_gt065_max_retention_candidate",
            "role": "candidate",
            "eval_usable": True,
            "audit_passed": True,
            "avg_primary_score": 0.65,
            "worst_primary_score": 0.47,
            "task_gsm8k_score": 0.57,
            "task_mmlu_score": 0.60,
            "task_safety_score": 0.67,
            "task_humaneval_compile_score": 0.54,
            "total_relative_delta_norm": 0.248,
            "routed_tensors_gt_0_75": 0,
        },
    ]
    rows = add_smoke_intervals(rows)
    if case == "source_dominance":
        rows[2].update(
            avg_primary_score=0.55,
            worst_primary_score=0.35,
            task_gsm8k_score=0.45,
            task_mmlu_score=0.50,
            task_safety_score=0.40,
            task_humaneval_compile_score=0.30,
        )
        rows[3].update(
            avg_primary_score=0.54,
            worst_primary_score=0.34,
            task_gsm8k_score=0.44,
            task_mmlu_score=0.49,
            task_safety_score=0.39,
            task_humaneval_compile_score=0.29,
        )
        rows = add_smoke_intervals(rows)
    elif case == "task_regression":
        rows[2]["task_safety_score"] = 0.30
        rows[3]["task_safety_score"] = 0.31
        rows = add_smoke_intervals(rows)
    elif case == "uncertain_small_sample":
        for column in TASK_SCORE_COLUMNS:
            rows[2][f"{column}_n"] = 6
        rows[2].update(
            avg_primary_score=0.67,
            worst_primary_score=0.46,
            task_gsm8k_score=0.59,
            task_mmlu_score=0.62,
            task_safety_score=0.69,
            task_humaneval_compile_score=0.46,
        )
        rows[3]["eval_usable"] = False
        rows = add_smoke_intervals(rows)
    elif case == "paired_regression":
        rows[2]["_prediction_scores"] = {f"gsm8k::{idx}": idx < 3 for idx in range(12)}
        rows[0]["_prediction_scores"] = {f"gsm8k::{idx}": True for idx in range(12)}
        rows[1]["_prediction_scores"] = {f"gsm8k::{idx}": idx < 6 for idx in range(12)}
        rows[3]["eval_usable"] = False
    elif case == "paired_noisy_delta":
        rows[2]["_prediction_scores"] = {f"gsm8k::{idx}": idx in {0, 1, 2, 3, 4} for idx in range(12)}
        rows[0]["_prediction_scores"] = {f"gsm8k::{idx}": idx in {0, 1, 2, 3, 4, 5} for idx in range(12)}
        rows[1]["_prediction_scores"] = {f"gsm8k::{idx}": idx in {0, 1, 2, 3, 4} for idx in range(12)}
        rows[3]["eval_usable"] = False
    elif case == "partial":
        rows[3]["eval_usable"] = False
    elif case != "candidate_win":
        raise ValueError(f"Unknown smoke case: {case}")
    return pd.DataFrame(rows)


def run_smoke_matrix(args: argparse.Namespace) -> dict[str, Any]:
    expected = {
        "candidate_win": ("select_candidate", "qwen3_moe_tail_trimmed_expert_only_candidate"),
        "source_dominance": ("keep_source_endpoint", "source_qwen3_30b_instruct"),
        "task_regression": ("keep_source_endpoint", "source_qwen3_30b_instruct"),
        "uncertain_small_sample": ("awaiting_candidate_eval", "source_qwen3_30b_instruct"),
        "paired_regression": ("awaiting_candidate_eval", "source_qwen3_30b_instruct"),
        "paired_noisy_delta": ("provisional_candidate", "qwen3_moe_tail_trimmed_expert_only_candidate"),
        "partial": ("provisional_candidate", "qwen3_moe_tail_trimmed_expert_only_candidate"),
    }
    rows = []
    for case, (expected_status, expected_selected) in expected.items():
        table, selection = build_selection_table(
            smoke_rows(case),
            task_regression_tolerance=args.task_regression_tolerance,
            use_uncertainty_gate=not args.disable_uncertainty_gate,
            use_paired_gate=not args.disable_paired_gate,
            paired_loss_tolerance_rate=args.paired_loss_tolerance_rate,
            paired_alpha=args.paired_alpha,
        )
        rows.append(
            {
                "case": case,
                "status": selection["status"],
                "expected_status": expected_status,
                "selected_method": selection.get("selected_method"),
                "expected_selected_method": expected_selected,
                "passed": selection["status"] == expected_status
                and selection.get("selected_method") == expected_selected,
                "eligible_candidate_count": selection.get("eligible_candidate_count"),
            }
        )
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.DataFrame(rows)
    matrix_path = output_dir / "selector_matrix.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    matrix.to_csv(matrix_path, index=False)
    passed = int(matrix["passed"].astype(bool).sum())
    summary = {
        "schema_version": 1,
        "status": "passed" if passed == len(matrix) else "failed",
        "case_count": int(len(matrix)),
        "passed_case_count": passed,
        "failed_case_count": int(len(matrix) - passed),
        "outputs": {
            "matrix": rel(matrix_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
        "cases": matrix.to_dict(orient="records"),
    }
    lines = [
        "# Qwen3 MoE Final Candidate Selector Smoke",
        "",
        f"- Status: `{summary['status']}`",
        f"- Cases: `{passed}/{len(matrix)}`",
        "",
        "| case | status | selected | eligible | passed |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for _, row in matrix.iterrows():
        lines.append(
            f"| `{row['case']}` | `{row['status']}` | `{row['selected_method']}` | "
            f"{int(row['eligible_candidate_count'])} | `{bool(row['passed'])}` |"
        )
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if summary["status"] != "passed":
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the final Qwen3 MoE same-shape candidate after audited vLLM eval.")
    parser.add_argument("--gate-dir", type=Path, default=Path("results/qwen3_moe_mechanism_eval_gate"))
    parser.add_argument("--audit-dir", type=Path, default=Path("results/qwen3_moe_eval_bundle_audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_final_candidate_selection"))
    parser.add_argument("--smoke-matrix", action="store_true")
    parser.add_argument("--task-regression-tolerance", type=float, default=1e-9)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--disable-uncertainty-gate", action="store_true")
    parser.add_argument("--disable-paired-gate", action="store_true")
    parser.add_argument("--paired-loss-tolerance-rate", type=float, default=0.0)
    parser.add_argument("--paired-alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_matrix:
        summary = run_smoke_matrix(args)
        print(f"Wrote Qwen3 MoE final candidate selector smoke to {repo_path(args.output_dir).resolve()}")
        print(f"Status: {summary['status']}; cases {summary['passed_case_count']}/{summary['case_count']}")
        return
    rows = load_real_rows(args.gate_dir, args.audit_dir, confidence_z=args.confidence_z)
    table, selection = build_selection_table(
        rows,
        task_regression_tolerance=args.task_regression_tolerance,
        use_uncertainty_gate=not args.disable_uncertainty_gate,
        use_paired_gate=not args.disable_paired_gate,
        paired_loss_tolerance_rate=args.paired_loss_tolerance_rate,
        paired_alpha=args.paired_alpha,
    )
    summary = write_outputs(args.output_dir, table, selection)
    print(f"Wrote Qwen3 MoE final candidate selection to {repo_path(args.output_dir).resolve()}")
    print(f"Status: {summary['status']}; selected={summary['current_selection'].get('selected_method')}")


if __name__ == "__main__":
    main()
