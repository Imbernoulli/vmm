#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_METHODS = {"source_qwen3_30b_instruct", "source_qwen3_30b_coder"}
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
DEFAULT_TASK_ORDER = ["gsm8k", "mmlu", "safety", "humaneval_compile"]
TASKS_BY_MECHANISM_TEST = {
    "candidate_vs_sources": DEFAULT_TASK_ORDER,
    "unified_mechanism_optimizer": ["gsm8k", "mmlu", "humaneval_compile"],
    "expert_subspace_conflict_ablation": ["gsm8k", "humaneval_compile"],
    "layer_chunk_sensitivity": ["mmlu", "humaneval_compile"],
    "risk_penalty_simplification": ["gsm8k", "humaneval_compile"],
    "second_stage_tail_trim": ["gsm8k", "humaneval_compile"],
    "shared_attention_ablation": ["mmlu", "humaneval_compile"],
    "route_load_trust_region": ["gsm8k", "mmlu", "humaneval_compile"],
    "tail_delta_cap": DEFAULT_TASK_ORDER,
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


def maybe_int(value: Any) -> int | None:
    value = clean_value(value)
    return None if value is None else int(value)


def bool_value(value: Any) -> bool:
    value = clean_value(value)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def read_csv(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cli_arg_value(command: str, option: str) -> str | None:
    if not command:
        return None
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token == option and index + 1 < len(tokens):
            return tokens[index + 1]
        prefix = option + "="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def replace_cli_arg(command: str, option: str, value: int) -> str:
    if not command:
        return command
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token == option and index + 1 < len(tokens):
            tokens[index + 1] = str(value)
            return " ".join(shlex.quote(item) for item in tokens)
        prefix = option + "="
        if token.startswith(prefix):
            tokens[index] = f"{option}={value}"
            return " ".join(shlex.quote(item) for item in tokens)
    tokens.extend([option, str(value)])
    return " ".join(shlex.quote(item) for item in tokens)


def replace_cli_arg_text(command: str, option: str, value: str) -> str:
    if not command:
        return command
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token == option and index + 1 < len(tokens):
            tokens[index + 1] = value
            return " ".join(shlex.quote(item) for item in tokens)
        prefix = option + "="
        if token.startswith(prefix):
            tokens[index] = f"{option}={value}"
            return " ".join(shlex.quote(item) for item in tokens)
    tokens.extend([option, value])
    return " ".join(shlex.quote(item) for item in tokens)


def parse_tasks(value: Any) -> list[str]:
    return [part.strip() for part in str(clean_value(value) or "").split(",") if part.strip()]


def tasks_from_command(command: str) -> list[str]:
    return parse_tasks(cli_arg_value(command, "--tasks") or ",".join(DEFAULT_TASK_ORDER))


def serve_model_path(command: str) -> str:
    if not command:
        return ""
    tokens = shlex.split(command)
    for index, token in enumerate(tokens[:-1]):
        if token == "serve":
            return tokens[index + 1]
    return ""


def checkpoint_ready(row: pd.Series | dict[str, Any]) -> bool:
    explicit = clean_value(row.get("checkpoint_exists"))
    if explicit is not None:
        return bool_value(explicit)
    checkpoint_path = clean_value(row.get("checkpoint_path"))
    if checkpoint_path is not None and repo_path(str(checkpoint_path)).exists():
        return True
    model_path = serve_model_path(str(row.get("serve_command") or ""))
    if model_path and repo_path(model_path).exists():
        return True
    return str(clean_value(row.get("serve_status")) or "") == "ready_to_host"


def primary_metric(row: pd.Series | dict[str, Any]) -> tuple[str, float] | None:
    for column in ("strict_exact", "accuracy", "policy_accuracy", "compile_rate"):
        value = maybe_float(row.get(column))
        if value is not None:
            return column, value
    return None


def prediction_key(row: pd.Series | dict[str, Any]) -> str | None:
    task = str(row.get("task", "")).strip()
    if task not in PAIRED_TASKS:
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


def exact_paired_source_advantage_pvalue(candidate_only: int, source_only: int) -> float:
    if source_only <= candidate_only:
        return 1.0
    discordant = candidate_only + source_only
    if discordant <= 0:
        return 1.0
    if discordant > 1024:
        mean = discordant / 2.0
        variance = discordant / 4.0
        z = (candidate_only + 0.5 - mean) / (variance**0.5)
        return 0.5 * math.erfc(-z / (2.0**0.5))
    return sum(math.comb(discordant, item) for item in range(candidate_only + 1)) / (2.0**discordant)


def score_from_eval_dir(eval_output_dir: str | Path) -> dict[str, Any]:
    root = repo_path(eval_output_dir)
    summary = read_json(root / "summary.json")
    model_summary = read_csv(root / "model_summary.csv")
    metrics = read_csv(root / "metrics.csv")
    row = model_summary.iloc[0].to_dict() if not model_summary.empty else {}
    out: dict[str, Any] = {
        "eval_status_observed": summary.get("status") or row.get("eval_status") or "missing",
        "observed_examples_min": None,
    }
    for column in SCORE_COLUMNS:
        out[column] = maybe_float(row.get(column))
    observed_examples = []
    if not metrics.empty:
        for _, metric_row in metrics.iterrows():
            task = str(metric_row.get("task") or "")
            primary = primary_metric(metric_row)
            if primary is not None:
                out[f"task_{task}_score"] = primary[1]
            examples = maybe_int(metric_row.get("examples"))
            if examples is not None:
                observed_examples.append(examples)
    if observed_examples:
        out["observed_examples_min"] = int(min(observed_examples))
    out["_prediction_scores"] = read_prediction_scores(root)
    return out


def row_scores(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    has_inline_scores = False
    for column in SCORE_COLUMNS:
        value = maybe_float(row.get(column))
        out[column] = value
        has_inline_scores = has_inline_scores or value is not None
    if has_inline_scores:
        out["eval_status_observed"] = str(row.get("eval_status") or "complete")
        out["observed_examples_min"] = maybe_int(
            row.get("observed_examples_min") or row.get("recommended_max_examples") or row.get("current_max_examples")
        )
        out["_prediction_scores"] = clean_value(row.get("_prediction_scores")) or {}
        return out
    return score_from_eval_dir(str(row.get("eval_output_dir") or ""))


def source_frontier(rows: pd.DataFrame) -> tuple[bool, dict[str, float | None]]:
    frontier: dict[str, float | None] = {}
    source_rows = rows[rows["method"].isin(SOURCE_METHODS)]
    sources_complete = not source_rows.empty and len(source_rows) == len(SOURCE_METHODS)
    for column in SCORE_COLUMNS:
        values = [maybe_float(row.get(column)) for _, row in source_rows.iterrows()]
        values = [value for value in values if value is not None]
        frontier[column] = max(values) if values else None
        sources_complete = sources_complete and len(values) == len(SOURCE_METHODS)
    return sources_complete, frontier


def mechanism_reference_count(mechanism_budget: pd.DataFrame, method: str) -> int:
    if mechanism_budget.empty:
        return 0
    count = 0
    for _, row in mechanism_budget.iterrows():
        methods = str(row.get("required_methods") or "")
        if method in {part.strip() for part in methods.split(",") if part.strip()}:
            count += 1
    return count


def parse_required_methods(value: Any) -> list[str]:
    return [part.strip() for part in str(clean_value(value) or "").split(",") if part.strip()]


def mechanism_test_weight(test: str) -> float:
    weights = {
        "candidate_vs_sources": 2.00,
        "unified_mechanism_optimizer": 1.70,
        "expert_subspace_conflict_ablation": 1.45,
        "layer_chunk_sensitivity": 1.30,
        "risk_penalty_simplification": 1.20,
        "second_stage_tail_trim": 1.05,
        "shared_attention_ablation": 0.95,
        "route_load_trust_region": 0.90,
        "tail_delta_cap": 0.85,
        "source_control_floor": 0.00,
    }
    return weights.get(test, 1.0)


def mechanism_tests_for_method(mechanism_budget: pd.DataFrame, method: str) -> list[str]:
    if mechanism_budget.empty:
        return []
    tests = []
    for _, row in mechanism_budget.iterrows():
        test = str(row.get("test") or "")
        required = set(parse_required_methods(row.get("required_methods")))
        if method in required and method not in SOURCE_METHODS and test:
            tests.append(test)
    return tests


def covered_tests_for_selection(mechanism_budget: pd.DataFrame, selected_methods: set[str]) -> set[str]:
    covered = set()
    controls = set(SOURCE_METHODS) | set(selected_methods)
    for _, row in mechanism_budget.iterrows():
        test = str(row.get("test") or "")
        required = set(parse_required_methods(row.get("required_methods")))
        candidate_required = {method for method in required if method not in SOURCE_METHODS}
        if test and candidate_required and required.issubset(controls):
            covered.add(test)
    return covered


def probe_tasks_for_tests(tests: list[str], available_tasks: list[str]) -> tuple[list[str], str]:
    if not tests:
        return list(available_tasks), "default_all_tasks_no_mechanism_context"
    task_set: set[str] = set()
    for test in tests:
        task_set.update(TASKS_BY_MECHANISM_TEST.get(test, DEFAULT_TASK_ORDER))
    ordered = [task for task in DEFAULT_TASK_ORDER if task in task_set and task in available_tasks]
    if not ordered:
        ordered = list(available_tasks)
    return ordered, "mechanism_tests:" + ",".join(sorted(tests))


def row_recommended_tasks(
    row: pd.Series,
    decision: dict[str, Any],
    covered_tests: list[str],
    available_tasks: list[str],
) -> tuple[list[str], str]:
    if not decision["recommended_max_examples"]:
        return [], "no_eval_action"
    role = str(row.get("role") or "")
    action = str(decision["eval_action"])
    if role == "source":
        return list(available_tasks), "source_controls_need_all_tasks"
    if action == "extend_promising_candidate_to_full_budget":
        return list(available_tasks), "full_budget_escalation_needs_all_tasks"
    if action in {"run_initial_mechanism_probe", "queue_after_source_controls"}:
        return probe_tasks_for_tests(covered_tests, available_tasks)
    return list(available_tasks), "default_all_tasks_for_eval_action"


def choose_round1_probe_methods(
    candidate_rows: pd.DataFrame,
    mechanism_budget: pd.DataFrame,
    max_round1_candidates: int,
) -> tuple[set[str], dict[str, Any], dict[str, list[str]]]:
    selected: set[str] = set()
    ranked = candidate_rows.sort_values(["mechanism_priority", "gate_order"], ascending=[False, True])
    available = [str(row["method"]) for _, row in ranked.iterrows() if checkpoint_ready(row)]
    method_rows = {str(row["method"]): row for _, row in ranked.iterrows()}
    while len(selected) < max_round1_candidates and len(selected) < len(available):
        covered_now = covered_tests_for_selection(mechanism_budget, selected)
        best_method = None
        best_score = None
        for method in available:
            if method in selected:
                continue
            prospective = set(selected)
            prospective.add(method)
            newly_covered = covered_tests_for_selection(mechanism_budget, prospective) - covered_now
            method_tests = set(mechanism_tests_for_method(mechanism_budget, method))
            partial_weight = 0.15 * sum(
                mechanism_test_weight(test) for test in method_tests if test not in covered_now
            )
            coverage_weight = sum(mechanism_test_weight(test) for test in newly_covered)
            priority = float(method_rows[method].get("mechanism_priority") or 0.0)
            score = coverage_weight + partial_weight + 0.10 * priority
            key = (score, priority, -int(method_rows[method].get("gate_order") or 9999))
            if best_score is None or key > best_score:
                best_score = key
                best_method = method
        if best_method is None:
            break
        selected.add(best_method)
    final_covered = sorted(covered_tests_for_selection(mechanism_budget, selected))
    coverage_by_method = {
        method: sorted(mechanism_tests_for_method(mechanism_budget, method)) for method in available
    }
    coverage_summary = {
        "round1_selection_policy": "greedy_mechanism_coverage_then_priority",
        "round1_covered_mechanism_tests": final_covered,
        "round1_covered_mechanism_test_count": len(final_covered),
        "round1_candidate_method_count": len(selected),
    }
    return selected, coverage_summary, coverage_by_method


def mechanism_priority(row: pd.Series, mechanism_budget: pd.DataFrame) -> float:
    method = str(row["method"])
    if method in SOURCE_METHODS:
        return 1.20
    manual = {
        "qwen3_moe_unified_mechanism_candidate": 1.00,
        "qwen3_moe_searched_no_gt065_max_retention_candidate": 0.94,
        "qwen3_moe_layer_chunk_candidate": 0.92,
        "qwen3_moe_subspace_scaled_candidate": 0.90,
        "qwen3_moe_tail_trimmed_expert_only_candidate": 0.86,
        "qwen3_moe_expert_only_trust_region_candidate": 0.82,
        "qwen3_moe_trust_region_candidate": 0.78,
        "qwen3_moe_audit_gated_candidate": 0.70,
        "qwen3_moe_unified_route_guarded_candidate": 0.66,
    }.get(method, 0.55)
    reference_bonus = min(0.12, 0.03 * mechanism_reference_count(mechanism_budget, method))
    missing_penalty = 0.20 if not checkpoint_ready(row) else 0.0
    pruned_penalty = 0.25 if bool_value(row.get("plan_level_pruned", False)) else 0.0
    return round(max(0.0, manual + reference_bonus - missing_penalty - pruned_penalty), 4)


def max_task_regression(candidate: pd.Series, frontier: dict[str, float | None]) -> float | None:
    regressions = []
    for column in TASK_SCORE_COLUMNS:
        source_value = frontier.get(column)
        candidate_value = maybe_float(candidate.get(column))
        if source_value is not None and candidate_value is not None:
            regressions.append(float(source_value) - candidate_value)
    return max(regressions) if regressions else None


def paired_probe(
    candidate: pd.Series,
    sources: pd.DataFrame,
    *,
    tolerance_rate: float,
    alpha: float,
) -> dict[str, Any]:
    candidate_scores = candidate.get("_prediction_scores")
    if not isinstance(candidate_scores, dict) or not candidate_scores:
        return {
            "paired_shared_examples": 0,
            "paired_candidate_only_correct": 0,
            "paired_source_only_correct": 0,
            "paired_net_delta": None,
            "paired_min_pvalue": None,
            "task_paired_regression_columns": "",
            "paired_gate_status": "paired_predictions_unavailable",
        }
    rows = []
    regressions: list[str] = []
    for _, source in sources.iterrows():
        source_scores = source.get("_prediction_scores")
        if not isinstance(source_scores, dict) or not source_scores:
            continue
        by_task: dict[str, list[str]] = {}
        for key in sorted(set(candidate_scores) & set(source_scores)):
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
                    "shared": shared,
                    "candidate_only": candidate_only,
                    "source_only": source_only,
                    "net_delta": net_delta,
                    "pvalue": pvalue,
                }
            )
    shared_total = sum(int(row["shared"]) for row in rows)
    candidate_only_total = sum(int(row["candidate_only"]) for row in rows)
    source_only_total = sum(int(row["source_only"]) for row in rows)
    net_delta_total = (
        (candidate_only_total - source_only_total) / shared_total if shared_total else None
    )
    return {
        "paired_shared_examples": shared_total,
        "paired_candidate_only_correct": candidate_only_total,
        "paired_source_only_correct": source_only_total,
        "paired_net_delta": net_delta_total,
        "paired_min_pvalue": min([float(row["pvalue"]) for row in rows], default=None),
        "task_paired_regression_columns": ",".join(sorted(set(regressions))),
        "paired_gate_status": "paired_regression"
        if regressions
        else ("paired_pass" if rows else "paired_predictions_unavailable"),
    }


def no_paired_probe(status: str = "not_applicable") -> dict[str, Any]:
    return {
        "paired_shared_examples": 0,
        "paired_candidate_only_correct": 0,
        "paired_source_only_correct": 0,
        "paired_net_delta": None,
        "paired_min_pvalue": None,
        "task_paired_regression_columns": "",
        "paired_gate_status": status,
    }


def candidate_decision(
    row: pd.Series,
    *,
    sources: pd.DataFrame,
    sources_complete: bool,
    frontier: dict[str, float | None],
    selected_probe_methods: set[str],
    full_examples: int,
    probe_examples: int,
    close_margin: float,
    task_regression_margin: float,
    paired_loss_tolerance_rate: float,
    paired_alpha: float,
) -> dict[str, Any]:
    method = str(row["method"])
    role = str(row.get("role") or "")
    observed_examples = maybe_int(row.get("observed_examples_min")) or 0
    avg = maybe_float(row.get("avg_primary_score"))
    worst = maybe_float(row.get("worst_primary_score"))
    best_avg = frontier.get("avg_primary_score")
    best_worst = frontier.get("worst_primary_score")

    if role == "source":
        if avg is None or observed_examples < probe_examples:
            return {
                "stage": "round0_source_controls",
                "eval_action": "run_or_extend_source_control_probe",
                "recommended_max_examples": max(probe_examples, observed_examples),
                "decision_status": "source_control_required",
                "decision_reason": "Both source endpoints must be scored before any same-shape average can be accepted or pruned.",
                **no_paired_probe("source_control"),
            }
        if observed_examples < full_examples:
            return {
                "stage": "source_probe_complete",
                "eval_action": "hold_source_full_budget_until_candidate_survives_probe",
                "recommended_max_examples": 0,
                "decision_status": "source_control_probe_complete",
                "decision_reason": "Source probe exists; defer full-budget source expansion until at least one candidate survives the probe round.",
                **no_paired_probe("source_control"),
            }
        return {
            "stage": "complete_source_control",
            "eval_action": "no_eval_needed",
            "recommended_max_examples": observed_examples,
            "decision_status": "source_control_complete",
            "decision_reason": "Source endpoint already has full-budget scores.",
            **no_paired_probe("source_control"),
        }

    if not checkpoint_ready(row):
        return {
            "stage": "materialize_before_eval",
            "eval_action": "materialize_checkpoint_first",
            "recommended_max_examples": 0,
            "decision_status": "checkpoint_missing",
            "decision_reason": "The method is in the eval plan but its checkpoint path is not present yet.",
            **no_paired_probe("checkpoint_missing"),
        }
    if not sources_complete:
        if method in selected_probe_methods:
            action = "queue_after_source_controls"
        else:
            action = "hold_until_source_controls_and_probe_slots"
        return {
            "stage": "blocked_by_source_controls",
            "eval_action": action,
            "recommended_max_examples": probe_examples if method in selected_probe_methods else 0,
            "decision_status": "awaiting_source_controls",
            "decision_reason": "Candidate pruning/acceptance requires both source endpoints on the same task manifest first.",
            **no_paired_probe("awaiting_source_controls"),
        }
    if avg is None or worst is None or observed_examples < probe_examples:
        if method in selected_probe_methods:
            return {
                "stage": "round1_mechanism_probe",
                "eval_action": "run_initial_mechanism_probe",
                "recommended_max_examples": probe_examples,
                "decision_status": "selected_for_initial_probe",
                "decision_reason": "Selected by mechanism diversity/priority for the first candidate probe round.",
                **no_paired_probe("awaiting_candidate_predictions"),
            }
        return {
            "stage": "hold_probe_queue",
            "eval_action": "hold_until_probe_round_expands",
            "recommended_max_examples": 0,
            "decision_status": "not_in_first_probe_round",
            "decision_reason": "Lower mechanism priority; wait until higher-value probes finish or produce an ambiguity.",
            **no_paired_probe("not_in_probe_round"),
        }

    task_regression = max_task_regression(row, frontier)
    paired = paired_probe(
        row,
        sources,
        tolerance_rate=paired_loss_tolerance_rate,
        alpha=paired_alpha,
    )
    avg_gap = None if best_avg is None else avg - best_avg
    worst_gap = None if best_worst is None else worst - best_worst
    task_ok = task_regression is None or task_regression <= task_regression_margin
    paired_ok = not paired["task_paired_regression_columns"]
    close_or_better = (
        (avg_gap is not None and avg_gap >= -close_margin)
        or (worst_gap is not None and worst_gap >= -close_margin)
    )
    dominated = (
        avg_gap is not None
        and worst_gap is not None
        and avg_gap < -close_margin
        and worst_gap < -close_margin
        and not task_ok
    )
    if not paired_ok:
        return {
            "stage": "stopped_after_probe",
            "eval_action": "prune_paired_regressing_candidate",
            "recommended_max_examples": observed_examples,
            "decision_status": "pruned_by_paired_source_probe",
            "decision_reason": "Shared prediction keys show a statistically significant source-only advantage.",
            **paired,
        }
    if dominated:
        return {
            "stage": "stopped_after_probe",
            "eval_action": "prune_dominated_candidate",
            "recommended_max_examples": observed_examples,
            "decision_status": "pruned_by_source_frontier_probe",
            "decision_reason": "Probe scores are below the source frontier on avg/worst and exceed task-regression tolerance.",
            **paired,
        }
    if close_or_better and task_ok and paired_ok and observed_examples < full_examples:
        return {
            "stage": "round2_full_budget",
            "eval_action": "extend_promising_candidate_to_full_budget",
            "recommended_max_examples": full_examples,
            "decision_status": "promising_needs_powered_eval",
            "decision_reason": "Initial probe is close to or better than the source frontier without a large task regression.",
            **paired,
        }
    if close_or_better and task_ok and paired_ok:
        return {
            "stage": "ready_for_selector",
            "eval_action": "no_eval_needed",
            "recommended_max_examples": observed_examples,
            "decision_status": "ready_for_final_selector",
            "decision_reason": "Candidate has full-budget scores and is not source-dominated under configured margins.",
            **paired,
        }
    return {
        "stage": "hold_after_probe",
        "eval_action": "hold_for_mechanism_review",
        "recommended_max_examples": observed_examples,
        "decision_status": "ambiguous_or_task_regressing_probe",
        "decision_reason": "Probe is neither clearly dominated nor cleanly promising; inspect paired predictions/mechanism attribution before extending.",
        **paired,
    }


def enrich_budget(method_budget: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in method_budget.iterrows():
        item = row.to_dict()
        scores = row_scores(row)
        item.update(scores)
        rows.append(item)
    return pd.DataFrame(rows)


def build_schedule(
    method_budget: pd.DataFrame,
    mechanism_budget: pd.DataFrame,
    *,
    probe_examples: int,
    full_examples: int,
    max_round1_candidates: int,
    close_margin: float,
    task_regression_margin: float,
    paired_loss_tolerance_rate: float,
    paired_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    budget = enrich_budget(method_budget)
    if budget.empty:
        raise ValueError("method budget is empty")
    budget["mechanism_priority"] = budget.apply(lambda row: mechanism_priority(row, mechanism_budget), axis=1)
    candidate_rows = budget[~budget["method"].isin(SOURCE_METHODS)].copy()
    candidate_rows = candidate_rows.sort_values(["mechanism_priority", "gate_order"], ascending=[False, True])
    selected_probe_methods, coverage_summary, coverage_by_method = choose_round1_probe_methods(
        candidate_rows,
        mechanism_budget,
        max_round1_candidates,
    )
    sources_complete, frontier = source_frontier(budget)
    sources = budget[budget["method"].isin(SOURCE_METHODS)].copy()

    rows = []
    for _, row in budget.iterrows():
        method = str(row["method"])
        decision = candidate_decision(
            row,
            sources=sources,
            sources_complete=sources_complete,
            frontier=frontier,
            selected_probe_methods=selected_probe_methods,
            full_examples=full_examples,
            probe_examples=probe_examples,
            close_margin=close_margin,
            task_regression_margin=task_regression_margin,
            paired_loss_tolerance_rate=paired_loss_tolerance_rate,
            paired_alpha=paired_alpha,
        )
        covered_tests = coverage_by_method.get(method, [])
        eval_command = str(row.get("eval_command_recommended") or row.get("eval_command_current") or "")
        available_tasks = parse_tasks(row.get("tasks")) or tasks_from_command(eval_command)
        recommended_tasks, task_selection_reason = row_recommended_tasks(
            row,
            decision,
            covered_tests,
            available_tasks,
        )
        if decision["recommended_max_examples"]:
            eval_command = replace_cli_arg(eval_command, "--max-examples", int(decision["recommended_max_examples"]))
            eval_command = replace_cli_arg_text(eval_command, "--tasks", ",".join(recommended_tasks))
        else:
            eval_command = ""
        rows.append(
            {
                "schedule_rank": 0,
                "method": row["method"],
                "role": row.get("role"),
                "gate_order": maybe_int(row.get("gate_order")),
                "stage": decision["stage"],
                "eval_action": decision["eval_action"],
                "decision_status": decision["decision_status"],
                "mechanism_priority": row["mechanism_priority"],
                "selected_for_round1_probe": method in selected_probe_methods,
                "round1_selection_policy": coverage_summary["round1_selection_policy"]
                if method in selected_probe_methods
                else "",
                "covered_mechanism_tests": ",".join(covered_tests),
                "recommended_tasks": ",".join(recommended_tasks),
                "recommended_task_count": len(recommended_tasks),
                "task_selection_reason": task_selection_reason,
                "recommended_prompt_budget": int(decision["recommended_max_examples"]) * len(recommended_tasks),
                "serve_status": row.get("serve_status"),
                "checkpoint_exists": checkpoint_ready(row),
                "eval_status_observed": row.get("eval_status_observed"),
                "observed_examples_min": row.get("observed_examples_min"),
                "recommended_max_examples": decision["recommended_max_examples"],
                "avg_primary_score": row.get("avg_primary_score"),
                "worst_primary_score": row.get("worst_primary_score"),
                "max_task_regression_vs_source_frontier": max_task_regression(row, frontier)
                if str(row.get("role") or "") != "source"
                else None,
                "paired_gate_status": decision["paired_gate_status"],
                "paired_shared_examples": decision["paired_shared_examples"],
                "paired_candidate_only_correct": decision["paired_candidate_only_correct"],
                "paired_source_only_correct": decision["paired_source_only_correct"],
                "paired_net_delta": decision["paired_net_delta"],
                "paired_min_pvalue": decision["paired_min_pvalue"],
                "task_paired_regression_columns": decision["task_paired_regression_columns"],
                "decision_reason": decision["decision_reason"],
                "task_manifest": row.get("task_manifest") or cli_arg_value(eval_command, "--task-manifest"),
                "eval_output_dir": row.get("eval_output_dir"),
                "base_url": row.get("base_url"),
                "serve_command": row.get("serve_command"),
                "adaptive_eval_command": eval_command,
            }
        )
    schedule = pd.DataFrame(rows)
    action_order = {
        "run_or_extend_source_control_probe": 0,
        "extend_promising_candidate_to_full_budget": 1,
        "run_initial_mechanism_probe": 2,
        "queue_after_source_controls": 3,
        "hold_until_probe_round_expands": 5,
        "hold_until_source_controls_and_probe_slots": 6,
        "hold_for_mechanism_review": 7,
        "hold_source_full_budget_until_candidate_survives_probe": 8,
        "materialize_checkpoint_first": 8,
        "prune_paired_regressing_candidate": 9,
        "prune_dominated_candidate": 9,
        "no_eval_needed": 10,
    }
    schedule["_action_order"] = schedule["eval_action"].map(action_order).fillna(99)
    schedule = schedule.sort_values(
        ["_action_order", "mechanism_priority", "gate_order", "method"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    schedule["schedule_rank"] = range(1, len(schedule) + 1)
    schedule = schedule.drop(columns=["_action_order"])

    mechanism_rows = []
    for _, row in mechanism_budget.iterrows():
        required_methods = [part.strip() for part in str(row.get("required_methods") or "").split(",") if part.strip()]
        statuses = {
            method: str(
                schedule.loc[schedule["method"] == method, "decision_status"].iloc[0]
                if not schedule.loc[schedule["method"] == method].empty
                else "missing_from_schedule"
            )
            for method in required_methods
        }
        if any(status == "ready_for_final_selector" for status in statuses.values()):
            action = "ready_for_mechanism_selector"
        elif any(status == "promising_needs_powered_eval" for status in statuses.values()):
            action = "extend_promising_required_method"
        elif any(status == "selected_for_initial_probe" for status in statuses.values()):
            action = "run_required_probe"
        elif any(status == "awaiting_source_controls" for status in statuses.values()):
            action = "wait_for_source_controls"
        elif any(status == "pruned_by_source_frontier_probe" for status in statuses.values()):
            action = "drop_pruned_branch_or_replace_candidate"
        else:
            action = "hold"
        mechanism_rows.append(
            {
                "test": row.get("test"),
                "status": row.get("status"),
                "required_methods": ",".join(required_methods),
                "required_method_statuses": json.dumps(statuses, sort_keys=True),
                "adaptive_action": action,
                "mechanism_question": row.get("mechanism_question"),
            }
        )
    mechanism_schedule = pd.DataFrame(mechanism_rows)
    round1_rows = schedule[schedule["selected_for_round1_probe"].astype(bool)]
    runnable_rows = schedule[schedule["recommended_max_examples"].astype(int) > 0]
    summary = {
        "schema_version": 1,
        "status": "adaptive_schedule_ready",
        "method_count": int(len(schedule)),
        "source_controls_complete": bool(sources_complete),
        "round1_probe_candidate_count": int(schedule["selected_for_round1_probe"].sum()),
        "round1_selection_policy": coverage_summary["round1_selection_policy"],
        "round1_covered_mechanism_tests": coverage_summary["round1_covered_mechanism_tests"],
        "round1_covered_mechanism_test_count": coverage_summary["round1_covered_mechanism_test_count"],
        "round1_probe_task_budget": int(round1_rows["recommended_prompt_budget"].sum()),
        "runnable_prompt_budget": int(runnable_rows["recommended_prompt_budget"].sum()),
        "runnable_method_count": int(len(runnable_rows)),
        "eval_action_counts": {
            str(key): int(value) for key, value in schedule["eval_action"].value_counts().to_dict().items()
        },
        "decision_status_counts": {
            str(key): int(value) for key, value in schedule["decision_status"].value_counts().to_dict().items()
        },
        "top_eval_action": str(schedule.iloc[0]["eval_action"]) if not schedule.empty else None,
        "top_method": str(schedule.iloc[0]["method"]) if not schedule.empty else None,
        "probe_examples": int(probe_examples),
        "full_examples": int(full_examples),
        "close_margin": float(close_margin),
        "task_regression_margin": float(task_regression_margin),
        "paired_loss_tolerance_rate": float(paired_loss_tolerance_rate),
        "paired_alpha": float(paired_alpha),
    }
    return schedule, mechanism_schedule, summary


def build_runner(schedule: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Adaptive Qwen3 MoE eval runner. Start only one vLLM server at a time.",
        f"# Top action: {summary.get('top_eval_action')} / {summary.get('top_method')}",
        "",
    ]
    runnable = schedule[
        schedule["eval_action"].isin(
            {
                "run_or_extend_source_control_probe",
                "run_initial_mechanism_probe",
                "extend_promising_candidate_to_full_budget",
            }
        )
    ]
    if runnable.empty:
        lines.append('echo "No runnable eval action in the current adaptive schedule."')
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            'method="${1:-}"',
            'if [[ -z "$method" ]]; then',
            '  echo "Usage: $0 METHOD" >&2',
            '  echo "Runnable methods:" >&2',
        ]
    )
    for _, row in runnable.iterrows():
        lines.append(f'  echo "  {row["method"]} ({row["eval_action"]})" >&2')
    lines.extend(["  exit 2", "fi", ""])
    for _, row in runnable.iterrows():
        method = str(row["method"])
        serve_command = str(row.get("serve_command") or "")
        eval_command = str(row.get("adaptive_eval_command") or "")
        base_url = str(row.get("base_url") or "")
        log_path = f"results/qwen3_moe_adaptive_eval_schedule/logs/{method}.serve.log"
        lines.extend(
            [
                f'if [[ "$method" == {shlex.quote(method)} ]]; then',
                "  mkdir -p results/qwen3_moe_adaptive_eval_schedule/logs",
                f"  bash -lc {shlex.quote(serve_command)} >{shlex.quote(log_path)} 2>&1 &",
                "  server_pid=$!",
                "  cleanup() {",
                "    if kill -0 \"$server_pid\" >/dev/null 2>&1; then",
                "      kill \"$server_pid\" >/dev/null 2>&1 || true",
                "      wait \"$server_pid\" >/dev/null 2>&1 || true",
                "    fi",
                "  }",
                "  trap cleanup EXIT",
                "  for _ in $(seq 1 180); do",
                f"    if curl -fsS {shlex.quote(base_url.rstrip('/') + '/models')} >/dev/null 2>&1; then",
                "      break",
                "    fi",
                "    sleep 2",
                "  done",
                f"  curl -fsS {shlex.quote(base_url.rstrip('/') + '/models')} >/dev/null",
                f"  bash -lc {shlex.quote(eval_command)}",
                "  exit 0",
                "fi",
                "",
            ]
        )
    lines.extend(['echo "Unknown method: $method" >&2', "exit 2"])
    return "\n".join(lines) + "\n"


def build_report(summary: dict[str, Any], schedule: pd.DataFrame, mechanism_schedule: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE Adaptive Eval Schedule",
        "",
        "This scheduler turns mechanism evidence into a sequential vLLM budget: source controls first, high-value mechanism probes second, full-budget expansion only for candidates that are not source-dominated.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Source controls complete: `{summary['source_controls_complete']}`",
        f"- Round-1 probe candidates: `{summary['round1_probe_candidate_count']}`",
        f"- Round-1 coverage policy: `{summary['round1_selection_policy']}`",
        f"- Round-1 covered mechanism tests: `{summary['round1_covered_mechanism_test_count']}`",
        f"- Round-1 probe prompt budget: `{summary['round1_probe_task_budget']}`",
        f"- Runnable prompt budget now: `{summary['runnable_prompt_budget']}`",
        f"- Top action: `{summary['top_eval_action']}` for `{summary['top_method']}`",
        "",
        "## Method Schedule",
        "",
        "| rank | method | action | status | tasks | examples | prompts | priority | covered tests | paired gate | paired net | paired p | reason |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for _, row in schedule.iterrows():
        paired_net = row["paired_net_delta"]
        paired_p = row["paired_min_pvalue"]
        paired_net_text = "" if clean_value(paired_net) is None else f"{float(paired_net):.4f}"
        paired_p_text = "" if clean_value(paired_p) is None else f"{float(paired_p):.4f}"
        lines.append(
            f"| {int(row['schedule_rank'])} | `{row['method']}` | `{row['eval_action']}` | "
            f"`{row['decision_status']}` | `{row['recommended_tasks']}` | "
            f"{int(row['recommended_max_examples'])} | {int(row['recommended_prompt_budget'])} | "
            f"{float(row['mechanism_priority']):.3f} | `{row['covered_mechanism_tests']}` | "
            f"`{row['paired_gate_status']}` | "
            f"{paired_net_text} | {paired_p_text} | {row['decision_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism Gates",
            "",
            "| test | action | required method statuses |",
            "| --- | --- | --- |",
        ]
    )
    for _, row in mechanism_schedule.iterrows():
        lines.append(
            f"| `{row['test']}` | `{row['adaptive_action']}` | `{row['required_method_statuses']}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['schedule']}`",
            f"- `{summary['outputs']['mechanism_schedule']}`",
            f"- `{summary['outputs']['run_script']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    schedule: pd.DataFrame,
    mechanism_schedule: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Any]:
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = output_dir / "adaptive_schedule.csv"
    mechanism_path = output_dir / "mechanism_schedule.csv"
    run_script_path = output_dir / "run_adaptive_eval.sh"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary = dict(summary)
    summary["outputs"] = {
        "schedule": rel(schedule_path),
        "mechanism_schedule": rel(mechanism_path),
        "run_script": rel(run_script_path),
        "summary": rel(summary_path),
        "report": rel(report_path),
    }
    schedule.to_csv(schedule_path, index=False)
    mechanism_schedule.to_csv(mechanism_path, index=False)
    run_script_path.write_text(build_runner(schedule, summary), encoding="utf-8")
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, schedule, mechanism_schedule), encoding="utf-8")
    return summary


def smoke_budget(case: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    base_rows = [
        {
            "gate_order": 0,
            "method": "source_qwen3_30b_instruct",
            "role": "source",
            "checkpoint_exists": True,
            "serve_status": "ready_to_host",
            "recommended_max_examples": 384,
            "current_max_examples": 64,
            "eval_command_recommended": "python eval.py --max-examples 384",
        },
        {
            "gate_order": 1,
            "method": "source_qwen3_30b_coder",
            "role": "source",
            "checkpoint_exists": True,
            "serve_status": "ready_to_host",
            "recommended_max_examples": 384,
            "current_max_examples": 64,
            "eval_command_recommended": "python eval.py --max-examples 384",
        },
        {
            "gate_order": 2,
            "method": "qwen3_moe_unified_mechanism_candidate",
            "role": "candidate",
            "checkpoint_exists": True,
            "serve_status": "ready_to_host",
            "recommended_max_examples": 384,
            "current_max_examples": 64,
            "eval_command_recommended": "python eval.py --max-examples 384",
        },
        {
            "gate_order": 3,
            "method": "qwen3_moe_searched_no_gt065_max_retention_candidate",
            "role": "candidate",
            "checkpoint_exists": True,
            "serve_status": "ready_to_host",
            "recommended_max_examples": 384,
            "current_max_examples": 64,
            "eval_command_recommended": "python eval.py --max-examples 384",
        },
    ]
    mechanism = pd.DataFrame(
        [
            {
                "test": "candidate_vs_sources",
                "status": "awaiting_eval",
                "required_methods": "source_qwen3_30b_instruct,qwen3_moe_unified_mechanism_candidate",
                "mechanism_question": "smoke",
            }
        ]
    )
    expected = {
        "top_method": "source_qwen3_30b_instruct",
        "candidate_status": "awaiting_source_controls",
    }
    if case == "awaiting_sources":
        return pd.DataFrame(base_rows), mechanism, expected

    for row in base_rows[:2]:
        row.update(
            {
                "eval_status": "complete",
                "observed_examples_min": 64,
                "avg_primary_score": 0.70 if row["method"].endswith("instruct") else 0.66,
                "worst_primary_score": 0.54,
                "task_gsm8k_score": 0.66,
                "task_mmlu_score": 0.70,
                "task_safety_score": 0.72,
                "task_humaneval_compile_score": 0.48 if row["method"].endswith("instruct") else 0.72,
            }
        )
    instruct_scores = {f"gsm8k::{idx}": idx < 48 for idx in range(64)}
    coder_scores = {f"gsm8k::{idx}": idx < 45 for idx in range(64)}
    base_rows[0]["_prediction_scores"] = instruct_scores
    base_rows[1]["_prediction_scores"] = coder_scores
    expected["top_method"] = "qwen3_moe_unified_mechanism_candidate"
    if case == "probe_selected":
        expected["candidate_status"] = "selected_for_initial_probe"
        return pd.DataFrame(base_rows), mechanism, expected

    candidate = base_rows[2]
    candidate.update(
        {
            "eval_status": "complete",
            "observed_examples_min": 64,
            "avg_primary_score": 0.69,
            "worst_primary_score": 0.53,
            "task_gsm8k_score": 0.65,
            "task_mmlu_score": 0.69,
            "task_safety_score": 0.70,
            "task_humaneval_compile_score": 0.70,
            "_prediction_scores": {f"gsm8k::{idx}": idx < 47 for idx in range(64)},
        }
    )
    if case == "promising_escalates":
        expected["candidate_status"] = "promising_needs_powered_eval"
        return pd.DataFrame(base_rows), mechanism, expected
    if case == "full_ready":
        candidate["observed_examples_min"] = 384
        expected["candidate_status"] = "ready_for_final_selector"
        expected["top_method"] = "qwen3_moe_searched_no_gt065_max_retention_candidate"
        return pd.DataFrame(base_rows), mechanism, expected
    if case == "dominated_pruned":
        candidate.update(
            {
                "avg_primary_score": 0.58,
                "worst_primary_score": 0.42,
                "task_gsm8k_score": 0.50,
                "task_mmlu_score": 0.55,
                "task_safety_score": 0.60,
                "task_humaneval_compile_score": 0.50,
            }
        )
        expected["candidate_status"] = "pruned_by_source_frontier_probe"
        expected["top_method"] = "qwen3_moe_searched_no_gt065_max_retention_candidate"
        return pd.DataFrame(base_rows), mechanism, expected
    if case == "paired_regression_pruned":
        candidate.update(
            {
                "avg_primary_score": 0.69,
                "worst_primary_score": 0.53,
                "_prediction_scores": {f"gsm8k::{idx}": idx < 20 for idx in range(64)},
            }
        )
        expected["candidate_status"] = "pruned_by_paired_source_probe"
        expected["top_method"] = "qwen3_moe_searched_no_gt065_max_retention_candidate"
        return pd.DataFrame(base_rows), mechanism, expected
    raise ValueError(f"Unknown smoke case: {case}")


def run_smoke(args: argparse.Namespace) -> None:
    rows = []
    cases = [
        "awaiting_sources",
        "probe_selected",
        "promising_escalates",
        "full_ready",
        "dominated_pruned",
        "paired_regression_pruned",
    ]
    for case in cases:
        method_budget, mechanism_budget, expected = smoke_budget(case)
        schedule, _, _ = build_schedule(
            method_budget,
            mechanism_budget,
            probe_examples=args.probe_examples,
            full_examples=args.full_examples,
            max_round1_candidates=args.max_round1_candidates,
            close_margin=args.close_margin,
            task_regression_margin=args.task_regression_margin,
            paired_loss_tolerance_rate=args.paired_loss_tolerance_rate,
            paired_alpha=args.paired_alpha,
        )
        top_method = str(schedule.iloc[0]["method"])
        candidate = schedule[schedule["method"] == "qwen3_moe_unified_mechanism_candidate"].iloc[0]
        rows.append(
            {
                "case": case,
                "assertion": "top_method",
                "expected": expected["top_method"],
                "actual": top_method,
                "passed": top_method == expected["top_method"],
            }
        )
        rows.append(
            {
                "case": case,
                "assertion": "unified_candidate_status",
                "expected": expected["candidate_status"],
                "actual": candidate["decision_status"],
                "passed": str(candidate["decision_status"]) == expected["candidate_status"],
            }
        )
    coverage_method_budget = pd.DataFrame(
        [
            {
                "gate_order": 0,
                "method": "source_qwen3_30b_instruct",
                "role": "source",
                "checkpoint_exists": True,
                "serve_status": "ready_to_host",
                "avg_primary_score": 0.70,
                "worst_primary_score": 0.50,
                "observed_examples_min": 64,
            },
            {
                "gate_order": 1,
                "method": "source_qwen3_30b_coder",
                "role": "source",
                "checkpoint_exists": True,
                "serve_status": "ready_to_host",
                "avg_primary_score": 0.70,
                "worst_primary_score": 0.50,
                "observed_examples_min": 64,
            },
            {
                "gate_order": 2,
                "method": "qwen3_moe_unified_mechanism_candidate",
                "role": "candidate",
                "checkpoint_exists": True,
                "serve_status": "ready_to_host",
            },
            {
                "gate_order": 3,
                "method": "qwen3_moe_layer_chunk_candidate",
                "role": "candidate",
                "checkpoint_exists": True,
                "serve_status": "ready_to_host",
            },
            {
                "gate_order": 4,
                "method": "qwen3_moe_subspace_scaled_candidate",
                "role": "candidate",
                "checkpoint_exists": True,
                "serve_status": "ready_to_host",
            },
        ]
    )
    coverage_mechanism_budget = pd.DataFrame(
        [
            {
                "test": "candidate_vs_sources",
                "required_methods": "source_qwen3_30b_instruct,qwen3_moe_unified_mechanism_candidate",
                "mechanism_question": "smoke",
            },
            {
                "test": "expert_subspace_conflict_ablation",
                "required_methods": "qwen3_moe_unified_mechanism_candidate,qwen3_moe_subspace_scaled_candidate",
                "mechanism_question": "smoke",
            },
        ]
    )
    coverage_schedule, _, coverage_summary = build_schedule(
        coverage_method_budget,
        coverage_mechanism_budget,
        probe_examples=args.probe_examples,
        full_examples=args.full_examples,
        max_round1_candidates=2,
        close_margin=args.close_margin,
        task_regression_margin=args.task_regression_margin,
        paired_loss_tolerance_rate=args.paired_loss_tolerance_rate,
        paired_alpha=args.paired_alpha,
    )
    selected = set(
        coverage_schedule.loc[coverage_schedule["selected_for_round1_probe"].astype(bool), "method"].astype(str)
    )
    rows.append(
        {
            "case": "coverage_selects_complement",
            "assertion": "subspace_selected_for_coverage",
            "expected": "True",
            "actual": str("qwen3_moe_subspace_scaled_candidate" in selected),
            "passed": "qwen3_moe_subspace_scaled_candidate" in selected,
        }
    )
    rows.append(
        {
            "case": "coverage_selects_complement",
            "assertion": "covered_mechanism_test_count",
            "expected": "2",
            "actual": str(coverage_summary["round1_covered_mechanism_test_count"]),
            "passed": coverage_summary["round1_covered_mechanism_test_count"] == 2,
        }
    )
    subspace_row = coverage_schedule[
        coverage_schedule["method"] == "qwen3_moe_subspace_scaled_candidate"
    ].iloc[0]
    rows.append(
        {
            "case": "coverage_selects_complement",
            "assertion": "subspace_probe_tasks",
            "expected": "gsm8k,humaneval_compile",
            "actual": str(subspace_row["recommended_tasks"]),
            "passed": str(subspace_row["recommended_tasks"]) == "gsm8k,humaneval_compile",
        }
    )
    matrix = pd.DataFrame(rows)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "adaptive_eval_schedule_smoke_matrix.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary = {
        "schema_version": 1,
        "status": "passed" if bool(matrix["passed"].all()) else "failed",
        "case_count": int(matrix["case"].nunique()),
        "assertion_count": int(len(matrix)),
        "passed_assertion_count": int(matrix["passed"].sum()),
        "failed_assertion_count": int((~matrix["passed"]).sum()),
        "outputs": {
            "matrix": rel(matrix_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    lines = [
        "# Qwen3 MoE Adaptive Eval Schedule Smoke",
        "",
        f"- Status: `{summary['status']}`",
        f"- Assertions: `{summary['passed_assertion_count']}/{summary['assertion_count']}`",
        "",
        "| case | assertion | expected | actual | passed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in matrix.iterrows():
        lines.append(
            f"| `{row['case']}` | `{row['assertion']}` | `{row['expected']}` | "
            f"`{row['actual']}` | `{bool(row['passed'])}` |"
        )
    matrix.to_csv(matrix_path, index=False)
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote adaptive eval schedule smoke to {output_dir.resolve()}")
    print(f"Status: {summary['status']}; assertions {summary['passed_assertion_count']}/{summary['assertion_count']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an adaptive vLLM eval schedule for Qwen3 MoE averaging.")
    parser.add_argument("--method-budget", type=Path, default=Path("results/qwen3_moe_eval_budget_plan/method_budget.csv"))
    parser.add_argument(
        "--mechanism-budget",
        type=Path,
        default=Path("results/qwen3_moe_eval_budget_plan/mechanism_budget.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_adaptive_eval_schedule"))
    parser.add_argument("--probe-examples", type=int, default=64)
    parser.add_argument("--full-examples", type=int, default=384)
    parser.add_argument("--max-round1-candidates", type=int, default=5)
    parser.add_argument("--close-margin", type=float, default=0.02)
    parser.add_argument("--task-regression-margin", type=float, default=0.05)
    parser.add_argument("--paired-loss-tolerance-rate", type=float, default=0.0)
    parser.add_argument("--paired-alpha", type=float, default=0.05)
    parser.add_argument("--smoke-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_matrix:
        run_smoke(args)
        return
    method_budget = read_csv(args.method_budget)
    mechanism_budget = read_csv(args.mechanism_budget)
    schedule, mechanism_schedule, summary = build_schedule(
        method_budget,
        mechanism_budget,
        probe_examples=args.probe_examples,
        full_examples=args.full_examples,
        max_round1_candidates=args.max_round1_candidates,
        close_margin=args.close_margin,
        task_regression_margin=args.task_regression_margin,
        paired_loss_tolerance_rate=args.paired_loss_tolerance_rate,
        paired_alpha=args.paired_alpha,
    )
    summary = write_outputs(args.output_dir, schedule, mechanism_schedule, summary)
    print(f"Wrote Qwen3 MoE adaptive eval schedule to {repo_path(args.output_dir).resolve()}")
    print(f"Status: {summary['status']}; top={summary['top_eval_action']} {summary['top_method']}")


if __name__ == "__main__":
    main()
