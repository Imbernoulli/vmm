#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

LITERATURE_PRIORS = [
    {
        "key": "mode_connectivity",
        "source": "https://arxiv.org/abs/1802.10026",
        "mechanism": "A source-set average is plausible only when the probed path stays in a low-loss component.",
    },
    {
        "key": "model_soups",
        "source": "https://arxiv.org/abs/2203.05482",
        "mechanism": "Weight averaging helps when fine-tunes are in one basin; endpoint fallback remains part of the recipe.",
    },
    {
        "key": "ties",
        "source": "https://arxiv.org/abs/2306.01708",
        "mechanism": "Coordinate interference can erase useful deltas, so source/task gains must exceed merge interference.",
    },
    {
        "key": "expert_merging",
        "source": "https://arxiv.org/abs/2509.25712",
        "mechanism": "Layer/chunk coefficients should be driven by calibration behavior, not a fixed global coefficient.",
    },
    {
        "key": "sub_moe",
        "source": "https://arxiv.org/abs/2506.23266",
        "mechanism": "MoE expert similarity and subspace conflict are better signals than tensor names alone.",
    },
    {
        "key": "harc",
        "source": "https://arxiv.org/abs/2606.03391",
        "mechanism": "MoE router perturbations can break top-k dispatch, so source-set averaging must keep router repair as a separate gate.",
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


def fnum(value: Any) -> float | None:
    value = clean_value(value)
    return None if value is None else float(value)


def fmt(value: Any, digits: int = 4) -> str:
    value = fnum(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def source_key(sources: list[str] | tuple[str, ...]) -> str:
    return "+".join(sorted(str(source) for source in sources))


def parse_sources(text: str) -> list[str]:
    return [part.strip() for part in str(text).split("+") if part.strip()]


def task_columns(matrix: pd.DataFrame) -> list[str]:
    ignored = {"name", "avg", "worst"}
    return [column for column in matrix.columns if column not in ignored]


def normalized_entropy(weights: dict[str, float]) -> float:
    positive = [weight for weight in weights.values() if weight > 0.0]
    if len(positive) <= 1:
        return 0.0
    entropy = -sum(weight * math.log(weight) for weight in positive)
    return entropy / math.log(len(weights))


def observed_interference_budget(
    observed: pd.DataFrame,
    *,
    quantile: float,
    floor: float,
) -> dict[str, Any]:
    if observed.empty or "avg_gap_to_source_frontier" not in observed.columns:
        return {
            "budget": float(floor),
            "source": "floor_no_observed_merges",
            "negative_gap_count": 0,
            "median_abs_negative_gap": None,
            "best_repaired_abs_gap": None,
        }
    gaps = pd.to_numeric(observed["avg_gap_to_source_frontier"], errors="coerce").dropna()
    negative = gaps[gaps < 0.0].abs()
    if negative.empty:
        return {
            "budget": float(floor),
            "source": "floor_no_negative_observed_gap",
            "negative_gap_count": 0,
            "median_abs_negative_gap": None,
            "best_repaired_abs_gap": None,
        }
    quantile_budget = float(negative.quantile(quantile))
    return {
        "budget": max(float(floor), quantile_budget),
        "source": f"max_floor_observed_negative_gap_q{quantile:.2f}",
        "negative_gap_count": int(len(negative)),
        "median_abs_negative_gap": float(negative.quantile(0.5)),
        "best_repaired_abs_gap": float(negative.min()),
    }


def source_scores(matrix: pd.DataFrame) -> dict[str, dict[str, float]]:
    tasks = task_columns(matrix)
    source_df = matrix[~matrix["name"].astype(str).str.startswith("merge_")]
    return {
        str(row["name"]): {task: float(row[task]) for task in tasks}
        for _, row in source_df.iterrows()
    }


def task_frontier_weights(
    scores: dict[str, dict[str, float]],
    sources: list[str],
    tasks: list[str],
    *,
    task_win_margin: float,
) -> tuple[dict[str, float], dict[str, str], dict[str, list[str]]]:
    votes = {source: 0.0 for source in sources}
    winners: dict[str, str] = {}
    tied_winners: dict[str, list[str]] = {}
    for task in tasks:
        ordered = sorted(((source, scores[source][task]) for source in sources), key=lambda item: item[1], reverse=True)
        best_score = ordered[0][1]
        task_winners = [source for source, value in ordered if best_score - value <= task_win_margin]
        tied_winners[task] = task_winners
        if len(task_winners) == 1:
            winners[task] = task_winners[0]
        else:
            winners[task] = "tie:" + "+".join(task_winners)
        for source in task_winners:
            votes[source] += 1.0 / len(task_winners)
    total_votes = sum(votes.values()) or 1.0
    weights = {source: votes[source] / total_votes for source in sources}
    return weights, winners, tied_winners


def build_candidate_rows(
    source_sets: pd.DataFrame,
    observed: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    interference_budget: float,
    task_win_margin: float,
    probe_surplus_margin: float,
) -> pd.DataFrame:
    scores = source_scores(matrix)
    tasks = task_columns(matrix)
    observed_by_key: dict[str, dict[str, Any]] = {}
    if not observed.empty:
        for key, rows in observed.groupby("source_set_key"):
            best = rows.sort_values(
                ["avg_gap_to_source_frontier", "worst_gap_to_source_frontier"],
                ascending=[False, False],
            ).iloc[0]
            observed_by_key[str(key)] = best.to_dict()

    rows: list[dict[str, Any]] = []
    for _, source_set in source_sets.iterrows():
        sources = parse_sources(str(source_set["source_set"]))
        key = str(source_set["source_set_key"])
        weights, winners, tied_winners = task_frontier_weights(
            scores,
            sources,
            tasks,
            task_win_margin=task_win_margin,
        )
        active_sources = [source for source, weight in weights.items() if weight > 0.0]
        reduced_key = source_key(active_sources)
        frontier_avg_gain = float(source_set["frontier_avg_gain_vs_best_single"])
        frontier_worst_gain = float(source_set["frontier_worst_gain_vs_best_single"])
        surplus_vs_interference = frontier_avg_gain - interference_budget
        gate = str(source_set["gate"])
        best_observed = observed_by_key.get(key, {})
        best_observed_gap = fnum(best_observed.get("avg_gap_to_source_frontier"))

        if gate == "source_dominated_not_averageable_as_final":
            optimizer_gate = "reject_source_dominated"
            action = "do_not_materialize_except_negative_control"
            priority = 0.10
        elif surplus_vs_interference >= 0.0 and frontier_worst_gain >= 0.0:
            optimizer_gate = "final_average_budget_candidate"
            action = "materialize_frontier_weighted_same_shape_candidate_then_run_locked_vllm"
            priority = 0.95 + min(0.20, surplus_vs_interference)
        elif frontier_avg_gain > 0.0 or frontier_worst_gain >= probe_surplus_margin:
            optimizer_gate = "probe_only_below_interference_budget"
            action = "run_larger_endpoint_eval_and_small_weighted_probe_no_final_acceptance"
            priority = 0.62 + max(0.0, frontier_avg_gain) + max(0.0, frontier_worst_gain) * 0.2
        else:
            optimizer_gate = "deprioritize_until_new_tasks_or_sources"
            action = "search_more_complementary_sources_before_average"
            priority = 0.25

        if best_observed_gap is not None and best_observed_gap < 0.0:
            priority -= min(0.20, abs(best_observed_gap))
        if reduced_key != key:
            action += "; drop_zero_vote_sources_or_keep_as_anchor_only"

        rows.append(
            {
                "source_set": source_set["source_set"],
                "source_set_key": key,
                "gate_from_complementarity": gate,
                "optimizer_gate": optimizer_gate,
                "priority_score": max(0.0, float(priority)),
                "frontier_avg_gain_vs_best_single": frontier_avg_gain,
                "frontier_worst_gain_vs_best_single": frontier_worst_gain,
                "interference_budget": float(interference_budget),
                "frontier_avg_surplus_vs_interference": surplus_vs_interference,
                "dominant_source": source_set.get("dominant_source"),
                "strict_task_winner_count": int(source_set["strict_task_winner_count"]),
                "task_frontier_winners": json.dumps(winners, sort_keys=True),
                "task_frontier_tied_winners": json.dumps(tied_winners, sort_keys=True),
                "source_weights": json.dumps(weights, sort_keys=True),
                "active_source_count": len(active_sources),
                "active_source_set_key": reduced_key,
                "source_weight_entropy": normalized_entropy(weights),
                "best_single_avg_source": source_set["best_single_avg_source"],
                "best_single_worst_source": source_set["best_single_worst_source"],
                "best_observed_merge": best_observed.get("merge_model"),
                "best_observed_variant": best_observed.get("variant"),
                "best_observed_avg_gap_to_frontier": best_observed_gap,
                "candidate_id": f"qwen3_moe_{key.replace('+', '_')}_frontier_weighted_probe",
                "router_policy": "freeze_router_to_best_single_or_anchor_until_router_gate_passes",
                "expert_policy": "identity_or_aligned_experts_with_task_frontier_source_weights",
                "same_shape_policy": "preserve config tokenizer model class tensor names tensor shapes expert count and router shape",
                "recommended_action": action,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["priority_score", "frontier_avg_surplus_vs_interference", "frontier_worst_gain_vs_best_single"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_eval_plan(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in candidate_rows.iterrows():
        gate = str(row["optimizer_gate"])
        if gate == "reject_source_dominated":
            tasks = ""
            examples = 0
            command = ""
            status = "hold_source_dominated"
        elif gate == "final_average_budget_candidate":
            tasks = "mmlu,gsm8k,humaneval,safety"
            examples = 64
            command = "results/qwen3_moe_eval_budget_plan/run_eval_budget.sh final"
            status = "queue_after_materialization"
        elif gate == "probe_only_below_interference_budget":
            tasks = "mmlu,gsm8k,humaneval"
            examples = 32
            command = "python scripts/build_fp_downstream_matrix.py --help"
            status = "endpoint_expansion_or_probe_only"
        else:
            tasks = ""
            examples = 0
            command = ""
            status = "wait_for_new_sources"
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "source_set": row["source_set"],
                "optimizer_gate": gate,
                "status": status,
                "tasks": tasks,
                "examples_per_task": examples,
                "prompt_budget": examples * len([task for task in tasks.split(",") if task]),
                "command": command,
                "reason": row["recommended_action"],
            }
        )
    return pd.DataFrame(rows)


def decode_json_dict(value: Any) -> dict[str, Any]:
    value = clean_value(value)
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_task_surplus_rows(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    interference_budget: float,
    task_win_margin: float,
) -> pd.DataFrame:
    scores = source_scores(matrix)
    tasks = task_columns(matrix)
    observed_by_key: dict[str, dict[str, Any]] = {}
    if not observed.empty:
        for key, rows in observed.groupby("source_set_key"):
            best = rows.sort_values(
                ["avg_gap_to_source_frontier", "worst_gap_to_source_frontier"],
                ascending=[False, False],
            ).iloc[0]
            observed_by_key[str(key)] = best.to_dict()

    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        sources = parse_sources(str(candidate["source_set"]))
        key = str(candidate["source_set_key"])
        best_single = str(candidate["best_single_avg_source"])
        observed_row = observed_by_key.get(key, {})
        observed_task_gaps = decode_json_dict(observed_row.get("task_gaps_to_frontier"))
        weights = decode_json_dict(candidate.get("source_weights"))
        for task in tasks:
            ordered = sorted(
                ((source, scores[source][task]) for source in sources),
                key=lambda item: item[1],
                reverse=True,
            )
            frontier_source = ordered[0][0]
            frontier_score = float(ordered[0][1])
            tied_frontier_sources = [
                source for source, value in ordered if frontier_score - float(value) <= task_win_margin
            ]
            best_single_score = float(scores[best_single][task])
            frontier_gain = frontier_score - best_single_score
            observed_gap = fnum(observed_task_gaps.get(task))
            observed_score = None if observed_gap is None else frontier_score + observed_gap
            task_surplus = frontier_gain - float(interference_budget)
            if frontier_gain <= task_win_margin:
                status = "no_task_frontier_gain"
                action = "do_not_use_this_task_as_average_evidence"
            elif task_surplus < 0.0:
                status = "gain_below_interference_budget"
                action = "expand_endpoint_eval_or_find_stronger_source_for_this_task"
            elif observed_gap is not None and observed_gap < -task_win_margin:
                status = "observed_merge_loses_frontier"
                action = "do_not_promote_average_until_candidate_repairs_this_task"
            else:
                status = "task_surplus_candidate"
                action = "eligible_task_signal_after_locked_eval"
            rows.append(
                {
                    "source_set": candidate["source_set"],
                    "source_set_key": key,
                    "optimizer_gate": candidate["optimizer_gate"],
                    "task": task,
                    "frontier_source": frontier_source,
                    "tied_frontier_sources": json.dumps(tied_frontier_sources, sort_keys=True),
                    "frontier_score": frontier_score,
                    "best_single_avg_source": best_single,
                    "best_single_task_score": best_single_score,
                    "frontier_gain_vs_best_single": frontier_gain,
                    "interference_budget": float(interference_budget),
                    "task_surplus_vs_interference": task_surplus,
                    "observed_merge": observed_row.get("merge_model"),
                    "observed_variant": observed_row.get("variant"),
                    "observed_gap_to_frontier": observed_gap,
                    "observed_task_score": observed_score,
                    "source_weight_for_frontier_source": fnum(weights.get(frontier_source)),
                    "status": status,
                    "recommended_action": action,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["source_set_key", "task_surplus_vs_interference", "frontier_gain_vs_best_single"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_discovery_rows(candidates: pd.DataFrame, scenario_matrix: pd.DataFrame) -> pd.DataFrame:
    top_probe = candidates[candidates["optimizer_gate"] == "probe_only_below_interference_budget"]
    top_probe_source = None if top_probe.empty else str(top_probe.iloc[0]["source_set"])
    rows: list[dict[str, Any]] = []
    if not scenario_matrix.empty:
        for _, row in scenario_matrix.iterrows():
            scenario_id = str(row["scenario_id"])
            if scenario_id == "negative_controls":
                action = "keep_controls_only"
                priority = 0.05
            elif "moe" in scenario_id:
                action = "search_for_stronger_endpoint_complementarity_then_apply_surplus_gate"
                priority = 0.90 if str(row["priority"]).startswith("p0") else 0.70
            else:
                action = "run_endpoint_frontier_before_average"
                priority = 0.75 if str(row["priority"]).startswith("p0") else 0.50
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "priority": row.get("priority"),
                    "source_roles": row.get("input_roles"),
                    "required_models": row.get("required_models"),
                    "action": action,
                    "priority_score": priority,
                    "carry_forward_measured_probe_set": top_probe_source,
                }
            )
    rows.append(
        {
            "scenario_id": "measured_qwen3_moe_coder_thinking_probe",
            "priority": "p0_probe_only",
            "source_roles": "code_agent_expert; reasoning_thinking_expert",
            "required_models": "measured source names: coder; thinking",
            "action": "expand_endpoint_eval_or_materialize_small_probe_only_because_surplus_is_below_interference_budget",
            "priority_score": 0.88,
            "carry_forward_measured_probe_set": top_probe_source,
        }
    )
    out = pd.DataFrame(rows)
    return out.sort_values(["priority_score", "scenario_id"], ascending=[False, True]).reset_index(drop=True)


def build_report(
    summary: dict[str, Any],
    candidates: pd.DataFrame,
    task_surplus: pd.DataFrame,
    eval_plan: pd.DataFrame,
    discovery: pd.DataFrame,
) -> str:
    top = summary.get("top_source_set") or {}
    lines = [
        "# Qwen3 Average Source-Set Optimizer",
        "",
        "This optimizer sits after the source-set complementarity gate. It asks whether measured endpoint complementarity is large enough to pay for observed merging interference before a same-shape average receives final-candidate budget.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Interference budget: `{fmt(summary['interference_budget'])}` from `{summary['interference_budget_source']}`",
        f"- Source sets scored: `{summary['source_set_count']}`",
        f"- Final-budget source sets: `{summary['final_average_budget_candidate_count']}`",
        f"- Probe-only source sets: `{summary['probe_only_source_set_count']}`",
        f"- Top source set: `{top.get('source_set')}`",
        f"- Top optimizer gate: `{top.get('optimizer_gate')}`",
        f"- Top frontier avg gain: `{fmt(top.get('frontier_avg_gain_vs_best_single'))}`",
        f"- Top surplus vs interference: `{fmt(top.get('frontier_avg_surplus_vs_interference'))}`",
        f"- Top task surplus-positive: `{summary['top_task_surplus_positive_count']}/{summary['task_count']}`",
        f"- Top no-gain tasks: `{summary['top_no_gain_task_count']}/{summary['task_count']}`",
        f"- Top best task gain: `{summary['top_best_task_gain_task']}` / `{fmt(summary['top_best_task_gain'])}`",
        f"- Top blocking tasks: `{summary['top_blocking_tasks']}`",
        f"- Top source weights: `{top.get('source_weights')}`",
        f"- Recommended action: `{top.get('recommended_action')}`",
        "",
        "## Candidate Source Sets",
        "",
        "| source set | gate | priority | avg gain | interference | surplus | weights | action |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for _, row in candidates.iterrows():
        lines.append(
            f"| `{row['source_set']}` | `{row['optimizer_gate']}` | {fmt(row['priority_score'], 3)} | "
            f"{fmt(row['frontier_avg_gain_vs_best_single'])} | {fmt(row['interference_budget'])} | "
            f"{fmt(row['frontier_avg_surplus_vs_interference'])} | `{row['source_weights']}` | "
            f"{row['recommended_action']} |"
        )
    lines.extend(
        [
            "",
            "## Task-Level Surplus",
            "",
            "| source set | task | frontier source | gain | surplus | observed gap | status |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in task_surplus.iterrows():
        if row["source_set"] != top.get("source_set"):
            continue
        lines.append(
            f"| `{row['source_set']}` | `{row['task']}` | `{row['frontier_source']}` | "
            f"{fmt(row['frontier_gain_vs_best_single'])} | "
            f"{fmt(row['task_surplus_vs_interference'])} | "
            f"{fmt(row['observed_gap_to_frontier'])} | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Eval Plan",
            "",
            "| candidate | status | tasks | examples/task | command |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for _, row in eval_plan.iterrows():
        lines.append(
            f"| `{row['candidate_id']}` | `{row['status']}` | `{row['tasks']}` | "
            f"{int(row['examples_per_task'])} | `{row['command']}` |"
        )
    lines.extend(
        [
            "",
            "## Source Discovery Queue",
            "",
            "| scenario | priority | action | carry-forward probe set |",
            "| --- | --- | --- | --- |",
        ]
    )
    for _, row in discovery.iterrows():
        lines.append(
            f"| `{row['scenario_id']}` | `{row['priority']}` | {row['action']} | "
            f"`{row['carry_forward_measured_probe_set']}` |"
        )
    lines.extend(
        [
            "",
            "## Literature Priors",
            "",
            "| key | source | mechanism used here |",
            "| --- | --- | --- |",
        ]
    )
    for prior in LITERATURE_PRIORS:
        lines.append(f"| `{prior['key']}` | {prior['source']} | {prior['mechanism']} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `candidate_source_sets`: `{summary['outputs']['candidate_source_sets']}`",
            f"- `task_surplus`: `{summary['outputs']['task_surplus']}`",
            f"- `source_weight_recipes`: `{summary['outputs']['source_weight_recipes']}`",
            f"- `eval_plan`: `{summary['outputs']['eval_plan']}`",
            f"- `source_discovery_queue`: `{summary['outputs']['source_discovery_queue']}`",
            f"- `summary`: `{summary['outputs']['summary']}`",
            f"- `report`: `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_sets = read_csv(args.source_set_gate)
    observed = read_csv(args.observed_merge_gaps)
    matrix = read_csv(args.matrix)
    scenario_matrix = read_csv(args.scenario_matrix)
    if source_sets.empty:
        raise ValueError(f"Missing source-set gate table: {args.source_set_gate}")
    if matrix.empty:
        raise ValueError(f"Missing downstream matrix: {args.matrix}")

    budget_info = observed_interference_budget(
        observed,
        quantile=args.observed_interference_quantile,
        floor=args.min_interference_budget,
    )
    candidates = build_candidate_rows(
        source_sets,
        observed,
        matrix,
        interference_budget=float(budget_info["budget"]),
        task_win_margin=args.task_win_margin,
        probe_surplus_margin=args.probe_surplus_margin,
    )
    recipes = candidates[
        [
            "candidate_id",
            "source_set",
            "active_source_set_key",
            "source_weights",
            "router_policy",
            "expert_policy",
            "same_shape_policy",
            "optimizer_gate",
            "recommended_action",
        ]
    ].copy()
    task_surplus = build_task_surplus_rows(
        candidates,
        observed,
        matrix,
        interference_budget=float(budget_info["budget"]),
        task_win_margin=args.task_win_margin,
    )
    eval_plan = build_eval_plan(candidates)
    discovery = build_discovery_rows(candidates, scenario_matrix)

    top = candidates.iloc[0].to_dict() if not candidates.empty else {}
    top_task_surplus = task_surplus[task_surplus["source_set_key"] == top.get("source_set_key")]
    top_positive_tasks = top_task_surplus[
        pd.to_numeric(top_task_surplus["frontier_gain_vs_best_single"], errors="coerce") > args.task_win_margin
    ]
    top_surplus_positive_tasks = top_task_surplus[
        pd.to_numeric(top_task_surplus["task_surplus_vs_interference"], errors="coerce") >= 0.0
    ]
    top_no_gain_tasks = top_task_surplus[
        pd.to_numeric(top_task_surplus["frontier_gain_vs_best_single"], errors="coerce") <= args.task_win_margin
    ]
    top_blockers = top_task_surplus[
        top_task_surplus["status"].isin(["no_task_frontier_gain", "gain_below_interference_budget", "observed_merge_loses_frontier"])
    ]
    top_best_task = None
    if not top_task_surplus.empty:
        top_best_task = top_task_surplus.sort_values("frontier_gain_vs_best_single", ascending=False).iloc[0].to_dict()
    final_count = int((candidates["optimizer_gate"] == "final_average_budget_candidate").sum())
    probe_count = int((candidates["optimizer_gate"] == "probe_only_below_interference_budget").sum())
    reject_count = int((candidates["optimizer_gate"] == "reject_source_dominated").sum())

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_source_sets.csv"
    task_surplus_path = output_dir / "task_surplus.csv"
    recipe_path = output_dir / "source_weight_recipes.csv"
    eval_path = output_dir / "eval_plan.csv"
    discovery_path = output_dir / "source_discovery_queue.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    summary = {
        "schema_version": 1,
        "status": "source_set_surplus_optimizer_ready",
        "matrix": rel(args.matrix),
        "source_set_gate": rel(args.source_set_gate),
        "observed_merge_gaps": rel(args.observed_merge_gaps),
        "source_set_count": int(len(candidates)),
        "task_count": int(len(task_columns(matrix))),
        "final_average_budget_candidate_count": final_count,
        "probe_only_source_set_count": probe_count,
        "rejected_source_dominated_count": reject_count,
        "top_positive_task_count": int(len(top_positive_tasks)),
        "top_task_surplus_positive_count": int(len(top_surplus_positive_tasks)),
        "top_no_gain_task_count": int(len(top_no_gain_tasks)),
        "top_blocking_task_count": int(len(top_blockers)),
        "top_blocking_tasks": ",".join(str(task) for task in top_blockers["task"].tolist()),
        "top_best_task_gain_task": None if top_best_task is None else top_best_task.get("task"),
        "top_best_task_gain": None if top_best_task is None else fnum(top_best_task.get("frontier_gain_vs_best_single")),
        "top_best_task_frontier_source": None if top_best_task is None else top_best_task.get("frontier_source"),
        "interference_budget": float(budget_info["budget"]),
        "interference_budget_source": budget_info["source"],
        "negative_observed_merge_gap_count": budget_info["negative_gap_count"],
        "median_abs_negative_observed_gap": budget_info["median_abs_negative_gap"],
        "best_repaired_abs_observed_gap": budget_info["best_repaired_abs_gap"],
        "min_interference_budget": float(args.min_interference_budget),
        "probe_surplus_margin": float(args.probe_surplus_margin),
        "task_win_margin": float(args.task_win_margin),
        "top_source_set": {
            "source_set": top.get("source_set"),
            "source_set_key": top.get("source_set_key"),
            "optimizer_gate": top.get("optimizer_gate"),
            "priority_score": fnum(top.get("priority_score")),
            "frontier_avg_gain_vs_best_single": fnum(top.get("frontier_avg_gain_vs_best_single")),
            "frontier_worst_gain_vs_best_single": fnum(top.get("frontier_worst_gain_vs_best_single")),
            "frontier_avg_surplus_vs_interference": fnum(top.get("frontier_avg_surplus_vs_interference")),
            "source_weights": top.get("source_weights"),
            "active_source_set_key": top.get("active_source_set_key"),
            "candidate_id": top.get("candidate_id"),
            "recommended_action": top.get("recommended_action"),
        },
        "literature_priors": LITERATURE_PRIORS,
        "mechanism_equation": (
            "promote average only if source_frontier_gain - observed_merge_interference_budget >= 0; "
            "otherwise use the source set for endpoint expansion or probe-only materialization"
        ),
        "outputs": {
            "candidate_source_sets": rel(candidate_path),
            "task_surplus": rel(task_surplus_path),
            "source_weight_recipes": rel(recipe_path),
            "eval_plan": rel(eval_path),
            "source_discovery_queue": rel(discovery_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }

    candidates.to_csv(candidate_path, index=False)
    task_surplus.to_csv(task_surplus_path, index=False)
    recipes.to_csv(recipe_path, index=False)
    eval_plan.to_csv(eval_path, index=False)
    discovery.to_csv(discovery_path, index=False)
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, candidates, task_surplus, eval_plan, discovery), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Qwen3 source-set choice before same-shape averaging.")
    parser.add_argument("--matrix", type=Path, default=Path("results/fp_downstream_matrix/matrix.csv"))
    parser.add_argument(
        "--source-set-gate",
        type=Path,
        default=Path("results/qwen3_source_set_complementarity_gate/source_set_gate.csv"),
    )
    parser.add_argument(
        "--observed-merge-gaps",
        type=Path,
        default=Path("results/qwen3_source_set_complementarity_gate/observed_merge_gaps.csv"),
    )
    parser.add_argument(
        "--scenario-matrix",
        type=Path,
        default=Path("results/qwen_target_model_registry/scenario_matrix.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_average_source_set_optimizer"))
    parser.add_argument("--observed-interference-quantile", type=float, default=0.5)
    parser.add_argument("--min-interference-budget", type=float, default=0.03)
    parser.add_argument("--probe-surplus-margin", type=float, default=0.0)
    parser.add_argument("--task-win-margin", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args)
    top = summary.get("top_source_set") or {}
    print(f"Wrote Qwen3 average source-set optimizer to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; top={top.get('source_set')} gate={top.get('optimizer_gate')}"
    )


if __name__ == "__main__":
    main()
