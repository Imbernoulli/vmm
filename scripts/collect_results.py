#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
HASH_LIMIT_BYTES = 50 * 1024 * 1024


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rel(path: str | Path) -> str:
    return str(repo_path(path).relative_to(REPO_ROOT))


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(repo_path(path), **kwargs)


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def clean_row(row: pd.Series) -> dict[str, Any]:
    return {str(key): clean_value(value) for key, value in row.items()}


def best_row(df: pd.DataFrame, column: str, largest: bool = True) -> dict[str, Any]:
    idx = df[column].idxmax() if largest else df[column].idxmin()
    return clean_row(df.loc[idx])


def find_method(df: pd.DataFrame, method: str) -> dict[str, Any]:
    rows = df[df["method"] == method]
    if rows.empty:
        raise ValueError(f"Missing method row: {method}")
    return clean_row(rows.iloc[0])


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def sha256_file(path: Path) -> str | None:
    if path.stat().st_size > HASH_LIMIT_BYTES:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_kind(path: Path) -> str:
    if path.suffix in {".py"}:
        return "code"
    if path.suffix in {".md", ".html"}:
        return "document"
    if path.suffix in {".csv", ".json", ".jsonl", ".txt"}:
        return "data"
    if path.suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        return "figure"
    return "artifact"


def collect_artifacts() -> list[dict[str, Any]]:
    roots = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "RESEARCH_REPORT.md",
        REPO_ROOT / "PAPER.md",
        REPO_ROOT / "MODEL_AVERAGING_PROBES_AND_MOE_OPTIMIZATION.md",
        REPO_ROOT / "QWEN_DENSE_MOE_EXPERIMENT_PLAN.md",
        REPO_ROOT / "proposal.md",
        REPO_ROOT / "prompts",
        REPO_ROOT / "src",
        REPO_ROOT / "scripts",
        REPO_ROOT / "results",
    ]
    suffixes = {".py", ".md", ".csv", ".json", ".jsonl", ".txt", ".png", ".html"}
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix in suffixes:
                files.append(root)
            continue
        files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)

    artifacts = []
    for path in sorted(set(files)):
        relative = rel(path)
        if "/checkpoints/" in relative or relative.endswith(".pt"):
            continue
        if relative.endswith("/predictions.csv"):
            continue
        stat = path.stat()
        artifacts.append(
            {
                "path": relative,
                "kind": artifact_kind(path),
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return artifacts


def summarize_digits() -> dict[str, Any]:
    summary = read_json("results/digits_merge/summary.json")
    methods = read_csv("results/digits_merge/method_metrics.csv")
    grid = read_csv("results/digits_merge/grid_metrics.csv")
    interference = read_csv("results/digits_merge/interference.csv")
    merge_methods = methods[methods.get("kind", "") == "merge"]
    if merge_methods.empty:
        merge_methods = methods[~methods["method"].str.startswith("expert")]
    return {
        "summary": summary,
        "base": find_method(methods, "base"),
        "linear_average": find_method(methods, "linear_average"),
        "layerwise_task_arithmetic": find_method(methods, "layerwise_task_arithmetic"),
        "regmean_linear": find_method(methods, "regmean_linear"),
        "validation_grid_best": find_method(methods, "validation_grid_best"),
        "best_merge_method": best_row(merge_methods, "worst_acc", largest=True),
        "grid": {
            "points": int(len(grid)),
            "max_worst_acc": float(grid["worst_acc"].max()),
            "max_avg_acc": float(grid["avg_acc"].max()),
            "fraction_worst_acc_ge_0_90": float((grid["worst_acc"] >= 0.90).mean()),
            "fraction_worst_acc_ge_0_95": float((grid["worst_acc"] >= 0.95).mean()),
        },
        "top_weighted_conflict": best_row(interference, "weighted_conflict", largest=True),
        "figures": [
            rel("results/digits_merge/figures/merge_landscape.png"),
            rel("results/digits_merge/figures/per_task_basin_overlay.png"),
            rel("results/digits_merge/figures/lambda_sweep.png"),
            rel("results/digits_merge/figures/method_overlay.png"),
            rel("results/digits_merge/figures/interference_heatmap.png"),
        ],
    }


def summarize_pairwise() -> dict[str, Any]:
    summary = read_json("results/digit_pairwise_experts/summary.json")
    pairs = read_csv("results/digit_pairwise_experts/pairwise_metrics.csv")
    correlations = read_csv("results/digit_pairwise_experts/correlations.csv", index_col=0)
    corr_row = correlations.loc["linear_drop_from_base"]
    return {
        "summary": summary,
        "mean_linear_worst_acc": float(pairs["linear_worst_acc"].mean()),
        "worst_linear_pair": best_row(pairs, "linear_worst_acc", largest=False),
        "largest_drop_pair": best_row(pairs, "linear_drop_from_base", largest=True),
        "top_weighted_conflict_pair": best_row(pairs, "weighted_conflict", largest=True),
        "spearman_vs_linear_drop": {
            "cosine": maybe_float(corr_row["cosine"]),
            "sign_conflict": maybe_float(corr_row["sign_conflict"]),
            "weighted_conflict": maybe_float(corr_row["weighted_conflict"]),
            "max_layer_weighted_conflict": maybe_float(corr_row["max_layer_weighted_conflict"]),
        },
        "figures": [
            rel("results/digit_pairwise_experts/pairwise_heatmaps.png"),
            rel("results/digit_pairwise_experts/conflict_vs_drop.png"),
            rel("results/digit_pairwise_experts/layer_conflict_atlas.png"),
        ],
    }


def summarize_alignment() -> dict[str, Any]:
    summary = read_json("results/alignment_barrier/summary.json")
    path = read_csv("results/alignment_barrier/path_metrics.csv")
    return {
        "summary": summary,
        "min_before_loss": float(path["before_loss"].min()),
        "max_before_loss": float(path["before_loss"].max()),
        "min_after_loss": float(path["after_loss"].min()),
        "max_after_loss": float(path["after_loss"].max()),
        "figure": rel("results/alignment_barrier/interpolation_alignment.png"),
    }


def summarize_cifar() -> dict[str, Any]:
    summary = read_json("results/cifar_merge/summary.json")
    methods = read_csv("results/cifar_merge/method_metrics.csv")
    grid = read_csv("results/cifar_merge/grid_metrics.csv")
    interference = read_csv("results/cifar_merge/interference.csv")
    base = find_method(methods, "base")
    linear = find_method(methods, "linear_average")
    validation = find_method(methods, "validation_grid_best")
    return {
        "summary": summary,
        "base": base,
        "linear_average": linear,
        "validation_grid_best": validation,
        "best_method": best_row(methods, "worst_acc", largest=True),
        "linear_minus_base_worst_acc": float(linear["worst_acc"] - base["worst_acc"]),
        "validation_minus_base_worst_acc": float(validation["worst_acc"] - base["worst_acc"]),
        "validation_minus_linear_worst_acc": float(validation["worst_acc"] - linear["worst_acc"]),
        "grid": {
            "points": int(len(grid)),
            "max_worst_acc": float(grid["worst_acc"].max()),
            "max_avg_acc": float(grid["avg_acc"].max()),
            "fraction_worst_acc_ge_0_40": float((grid["worst_acc"] >= 0.40).mean()),
            "fraction_worst_acc_ge_0_45": float((grid["worst_acc"] >= 0.45).mean()),
        },
        "top_weighted_conflict": best_row(interference, "weighted_conflict", largest=True),
        "figures": [
            rel("results/cifar_merge/figures/merge_landscape.png"),
            rel("results/cifar_merge/figures/method_overlay.png"),
            rel("results/cifar_merge/figures/lambda_sweep.png"),
            rel("results/cifar_merge/figures/interference_heatmap.png"),
        ],
    }


def summarize_cifar100_vit() -> dict[str, Any]:
    summary = read_json("results/cifar100_vit_merge/summary.json")
    methods = read_csv("results/cifar100_vit_merge/method_metrics.csv")
    grid = read_csv("results/cifar100_vit_merge/grid_metrics.csv")
    interference = read_csv("results/cifar100_vit_merge/interference.csv")
    base = find_method(methods, "base")
    linear = find_method(methods, "linear_average")
    best = best_row(methods, "worst_acc", largest=True)
    return {
        "summary": summary,
        "base": base,
        "linear_average": linear,
        "best_method": best,
        "linear_minus_base_worst_acc": float(linear["worst_acc"] - base["worst_acc"]),
        "best_minus_base_worst_acc": float(best["worst_acc"] - base["worst_acc"]),
        "grid": {
            "points": int(len(grid)),
            "max_worst_acc": float(grid["worst_acc"].max()),
            "max_avg_acc": float(grid["avg_acc"].max()),
            "fraction_worst_acc_ge_0_15": float((grid["worst_acc"] >= 0.15).mean()),
            "fraction_worst_acc_ge_0_20": float((grid["worst_acc"] >= 0.20).mean()),
        },
        "top_weighted_conflict": best_row(interference, "weighted_conflict", largest=True),
        "figures": [
            rel("results/cifar100_vit_merge/figures/merge_landscape.png"),
            rel("results/cifar100_vit_merge/figures/method_overlay.png"),
            rel("results/cifar100_vit_merge/figures/lambda_sweep.png"),
            rel("results/cifar100_vit_merge/figures/interference_heatmap.png"),
            rel("results/cifar100_vit_merge/figures/pca_task_vectors.png"),
        ],
    }


def summarize_pretrained_vit_transfer() -> dict[str, Any]:
    summary = read_json("results/pretrained_vit_transfer_merge/summary.json")
    methods = read_csv("results/pretrained_vit_transfer_merge/method_metrics.csv")
    grid = read_csv("results/pretrained_vit_transfer_merge/grid_metrics.csv")
    interference = read_csv("results/pretrained_vit_transfer_merge/interference.csv")
    base = find_method(methods, "base")
    linear = find_method(methods, "linear_average")
    best = best_row(methods, "worst_acc", largest=True)
    return {
        "summary": summary,
        "base": base,
        "linear_average": linear,
        "best_method": best,
        "linear_minus_base_worst_acc": float(linear["worst_acc"] - base["worst_acc"]),
        "best_minus_base_worst_acc": float(best["worst_acc"] - base["worst_acc"]),
        "grid": {
            "points": int(len(grid)),
            "max_worst_acc": float(grid["worst_acc"].max()),
            "max_avg_acc": float(grid["avg_acc"].max()),
            "fraction_worst_acc_ge_0_75": float((grid["worst_acc"] >= 0.75).mean()),
            "fraction_worst_acc_ge_0_80": float((grid["worst_acc"] >= 0.80).mean()),
        },
        "top_weighted_conflict": best_row(interference, "weighted_conflict", largest=True),
        "figures": [
            rel("results/pretrained_vit_transfer_merge/figures/merge_landscape.png"),
            rel("results/pretrained_vit_transfer_merge/figures/method_overlay.png"),
            rel("results/pretrained_vit_transfer_merge/figures/lambda_sweep.png"),
            rel("results/pretrained_vit_transfer_merge/figures/interference_heatmap.png"),
        ],
    }


def summarize_qwen_path() -> dict[str, Any]:
    summary = read_json("results/qwen_path_sweep/summary.json")
    path = read_csv("results/qwen_path_sweep/path_metrics.csv")
    deltas = read_csv("results/qwen_path_sweep/delta_summary.csv")
    group_rows = []
    for group, rows in deltas.groupby("group"):
        group_rows.append(
            {
                "group": group,
                "numel": int(rows["numel"].sum()),
                "delta_norm": float(math.sqrt(float((rows["delta_norm"] ** 2).sum()))),
                "mean_abs_delta_weighted": float(
                    (rows["mean_abs_delta"] * rows["numel"]).sum() / rows["numel"].sum()
                ),
            }
        )
    group_df = pd.DataFrame(group_rows)
    return {
        "summary": summary,
        "lambda_0": clean_row(path[path["lambda"] == 0.0].iloc[0]),
        "lambda_1": clean_row(path[path["lambda"] == 1.0].iloc[0]),
        "best_avg": best_row(path, "avg_nll", largest=False),
        "best_instruction": best_row(path, "instruction_nll", largest=False),
        "best_general": best_row(path, "general_nll", largest=False),
        "best_worst": best_row(path, "worst_nll", largest=False),
        "top_tensor_delta_norm": best_row(deltas, "delta_norm", largest=True),
        "top_group_delta_norms": [
            clean_row(row) for _, row in group_df.sort_values("delta_norm", ascending=False).head(8).iterrows()
        ],
        "figures": [
            rel("results/qwen_path_sweep/qwen_path_sweep.png"),
            rel("results/qwen_path_sweep/delta_norms.png"),
        ],
    }


def summarize_qwen_probe_smoke() -> dict[str, Any]:
    manifest = read_json("results/qwen_probe_smoke/manifest.json")
    deltas = read_csv("results/qwen_probe_smoke/delta_summary.csv")
    return {
        "manifest": manifest,
        "rows": int(len(deltas)),
        "max_delta_norm": float(deltas["delta_norm"].max()) if not deltas.empty else None,
        "max_mean_abs_delta": float(deltas["mean_abs_delta"].max()) if not deltas.empty else None,
    }


def summarize_qwen_gsm8k() -> dict[str, Any]:
    summary = read_json("results/qwen_gsm8k_slice/summary.json")
    metrics = read_csv("results/qwen_gsm8k_slice/metrics.csv")
    return {
        "summary": summary,
        "best_strict": best_row(metrics, "exact_match", largest=True),
        "best_loose": best_row(metrics, "loose_exact_match", largest=True),
        "rows": [clean_row(row) for _, row in metrics.iterrows()],
        "figure": rel("results/qwen_gsm8k_slice/gsm8k_exact_match.png"),
    }


def summarize_qwen_mmlu() -> dict[str, Any]:
    summary = read_json("results/qwen_mmlu_slice/summary.json")
    metrics = read_csv("results/qwen_mmlu_slice/metrics.csv")
    return {
        "summary": summary,
        "best_accuracy": best_row(metrics, "accuracy", largest=True),
        "best_gold_nll": best_row(metrics, "avg_gold_nll", largest=False),
        "rows": [clean_row(row) for _, row in metrics.iterrows()],
        "figure": rel("results/qwen_mmlu_slice/mmlu_accuracy.png"),
    }


def summarize_qwen_humaneval() -> dict[str, Any]:
    summary = read_json("results/qwen_humaneval_nll_slice/summary.json")
    metrics = read_csv("results/qwen_humaneval_nll_slice/metrics.csv")
    return {
        "summary": summary,
        "best_solution_nll": best_row(metrics, "avg_solution_nll", largest=False),
        "rows": [clean_row(row) for _, row in metrics.iterrows()],
        "figure": rel("results/qwen_humaneval_nll_slice/humaneval_nll.png"),
    }


def summarize_qwen_safety() -> dict[str, Any]:
    summary = read_json("results/qwen_safety_refusal_slice/summary.json")
    metrics = read_csv("results/qwen_safety_refusal_slice/metrics.csv")
    return {
        "summary": summary,
        "best_avg_safety_nll": best_row(metrics, "avg_safety_nll", largest=False),
        "best_safe_response_nll": best_row(metrics, "safe_response_nll", largest=False),
        "best_unsafe_refusal_nll": best_row(metrics, "unsafe_refusal_nll", largest=False),
        "rows": [clean_row(row) for _, row in metrics.iterrows()],
        "figure": rel("results/qwen_safety_refusal_slice/safety_refusal_nll.png"),
    }


def summarize_qwen_multi_expert() -> dict[str, Any]:
    summary = read_json("results/qwen_multi_expert_merge/summary.json")
    methods = read_csv("results/qwen_multi_expert_merge/method_metrics.csv")
    grid = read_csv("results/qwen_multi_expert_merge/grid_metrics.csv")
    conflict = read_csv("results/qwen_multi_expert_merge/pairwise_conflict.csv")
    return {
        "summary": summary,
        "base": find_method(methods, "base"),
        "instruct_expert": find_method(methods, "instruct_expert"),
        "coder_expert": find_method(methods, "coder_expert"),
        "linear_average": find_method(methods, "linear_average"),
        "best_avg": best_row(methods, "avg_nll", largest=False),
        "best_worst": best_row(methods, "worst_nll", largest=False),
        "best_grid_avg": best_row(grid, "avg_nll", largest=False),
        "best_grid_worst": best_row(grid, "worst_nll", largest=False),
        "grid": {
            "points": int(len(grid)),
            "min_avg_nll": float(grid["avg_nll"].min()),
            "min_worst_nll": float(grid["worst_nll"].min()),
            "max_avg_nll": float(grid["avg_nll"].max()),
            "max_worst_nll": float(grid["worst_nll"].max()),
        },
        "instruct_coder_conflict": clean_row(conflict.iloc[0]) if not conflict.empty else None,
        "figures": [
            rel("results/qwen_multi_expert_merge/figures/merge_grid.png"),
            rel("results/qwen_multi_expert_merge/figures/diagonal_path.png"),
            rel("results/qwen_multi_expert_merge/figures/pairwise_conflict.png"),
        ],
    }


def summarize_average_decision_report() -> dict[str, Any]:
    summary = read_json("results/average_decision_report/summary.json")
    decisions = summary.get("decisions", [])
    verdict_counts: dict[str, int] = {}
    for row in decisions:
        verdict = str(row.get("verdict", "unknown"))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    return {
        "summary": summary,
        "verdict_counts": verdict_counts,
        "rows": decisions,
        "report": rel("results/average_decision_report/report.md"),
        "decision_table": rel("results/average_decision_report/decision_table.csv"),
        "parameter_group_actions": rel("results/average_decision_report/parameter_group_actions.csv"),
    }


def summarize_moe_average_plan() -> dict[str, Any]:
    summary = read_json("results/moe_average_plan/summary.json")
    parameter_plan = read_csv("results/moe_average_plan/parameter_group_plan.csv")
    router_plan = read_csv("results/moe_average_plan/router_plan.csv")
    expert_plan = read_csv("results/moe_average_plan/expert_plan.csv")
    return {
        "summary": summary,
        "parameter_groups": [clean_row(row) for _, row in parameter_plan.iterrows()],
        "router_plan_rows": int(len(router_plan)),
        "expert_plan_rows": int(len(expert_plan)),
        "report": rel("results/moe_average_plan/report.md"),
        "parameter_group_plan": rel("results/moe_average_plan/parameter_group_plan.csv"),
        "router_plan": rel("results/moe_average_plan/router_plan.csv"),
        "expert_plan": rel("results/moe_average_plan/expert_plan.csv"),
    }


def summarize_same_shape_writer_smoke() -> dict[str, Any]:
    manifest = read_json("results/same_shape_writer_smoke/merge_manifest.json")
    return {
        "manifest": manifest,
        "floating_tensors": int(manifest["floating_tensors"]),
        "dry_run": bool(manifest["dry_run"]),
        "source_summaries": manifest["source_summaries"],
        "report": rel("results/same_shape_writer_smoke/report.md"),
        "manifest_path": rel("results/same_shape_writer_smoke/merge_manifest.json"),
    }


def summarize_moe_tensor_rule_writer_smoke() -> dict[str, Any]:
    summary = read_json("results/moe_tensor_rule_writer_smoke/summary.json")
    checks = read_csv("results/moe_tensor_rule_writer_smoke/tensor_checks.csv")
    manifest = read_json("results/moe_tensor_rule_writer_smoke/merge_manifest.json")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "checked_tensors": int(summary.get("checked_tensors", len(checks))),
        "failed_tensors": int(summary.get("failed_tensors", (~checks["passed"]).sum())),
        "rule_counts": summary.get("rule_counts", {}),
        "floating_tensors": int(manifest.get("floating_tensors", 0)),
        "frozen_tensors": int(manifest.get("frozen_tensors", 0)),
        "additive_delta_tensors": int(manifest.get("additive_delta_tensors", 0)),
        "additive_delta_values": int(manifest.get("additive_delta_values", 0)),
        "tensor_add_delta_summary": manifest.get("tensor_add_delta_summary", {}),
        "report": rel("results/moe_tensor_rule_writer_smoke/report.md"),
        "tensor_checks": rel("results/moe_tensor_rule_writer_smoke/tensor_checks.csv"),
        "manifest_path": rel("results/moe_tensor_rule_writer_smoke/merge_manifest.json"),
    }


def summarize_moe_combined_writer_smoke() -> dict[str, Any]:
    summary = read_json("results/moe_combined_writer_smoke/summary.json")
    checks = read_csv("results/moe_combined_writer_smoke/tensor_checks.csv")
    manifest = read_json("results/moe_combined_writer_smoke/merge_manifest.json")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "checked_tensors": int(summary.get("checked_tensors", len(checks))),
        "failed_tensors": int(summary.get("failed_tensors", (~checks["passed"]).sum())),
        "tensor_rule_count": int(summary.get("tensor_rule_count", len(manifest.get("tensor_rules", [])))),
        "tensor_alias_rule_count": int(
            summary.get("tensor_alias_rule_count", len(manifest.get("tensor_alias_rules", [])))
        ),
        "code_aliased_tensors": int(summary.get("code_aliased_tensors", 0)),
        "additive_delta_tensors": int(summary.get("additive_delta_tensors", 0)),
        "additive_delta_values": int(summary.get("additive_delta_values", 0)),
        "rule_counts": summary.get("rule_counts", {}),
        "report": rel("results/moe_combined_writer_smoke/report.md"),
        "tensor_checks": rel("results/moe_combined_writer_smoke/tensor_checks.csv"),
        "manifest_path": rel("results/moe_combined_writer_smoke/merge_manifest.json"),
    }


def summarize_checkpoint_topology() -> dict[str, Any]:
    summary = read_json("results/checkpoint_topology_inspect/summary.json")
    models = summary.get("models", [])
    return {
        "summary": summary,
        "models": models,
        "comparisons": summary.get("comparisons", []),
        "report": rel("results/checkpoint_topology_inspect/report.md"),
        "compatibility": rel("results/checkpoint_topology_inspect/compatibility.csv"),
    }


def summarize_average_candidate_recipes() -> dict[str, Any]:
    summary = read_json("results/average_candidate_recipes/summary.json")
    recipes = read_csv("results/average_candidate_recipes/candidate_recipes.csv")
    return {
        "summary": summary,
        "status_counts": summary.get("status_counts", {}),
        "rows": [clean_row(row) for _, row in recipes.iterrows()],
        "report": rel("results/average_candidate_recipes/report.md"),
        "recipes": rel("results/average_candidate_recipes/candidate_recipes.csv"),
    }


def summarize_moe_route_weight_recipes() -> dict[str, Any]:
    summary = read_json("results/moe_route_weight_recipes/summary.json")
    source_weights = read_csv("results/moe_route_weight_recipes/source_weights_by_expert.csv")
    routing_probe_plan_path = repo_path("results/moe_route_weight_recipes/routing_probe_plan.csv")
    category_source_plan_path = repo_path("results/moe_route_weight_recipes/category_source_plan.csv")
    routing_probe_plan = read_csv(routing_probe_plan_path) if routing_probe_plan_path.exists() else pd.DataFrame()
    category_source_plan = read_csv(category_source_plan_path) if category_source_plan_path.exists() else pd.DataFrame()
    return {
        "summary": summary,
        "recipe_status": summary.get("recipe_status"),
        "expert_rule_count": int(summary.get("expert_rule_count", 0)),
        "tensor_rule_count": int(summary.get("tensor_rule_count", 0)),
        "source_weights_rows": int(len(source_weights)),
        "routing_probe_plan_rows": int(len(routing_probe_plan)),
        "category_source_plan_rows": int(len(category_source_plan)),
        "routing_probe_models": []
        if routing_probe_plan.empty
        else [str(model) for model in routing_probe_plan["model"].tolist()],
        "report": rel("results/moe_route_weight_recipes/report.md"),
        "source_weights": rel("results/moe_route_weight_recipes/source_weights_by_expert.csv"),
        "tensor_rules": rel("results/moe_route_weight_recipes/tensor_rules.txt"),
        "writer_command": rel("results/moe_route_weight_recipes/writer_command.txt"),
        "routing_probe_plan": rel(routing_probe_plan_path) if routing_probe_plan_path.exists() else None,
        "category_source_plan": rel(category_source_plan_path) if category_source_plan_path.exists() else None,
        "prompt_pack": rel("prompts/qwen_moe_route_probe_prompts.jsonl"),
    }


def summarize_moe_router_bias_plan_dir(path: str) -> dict[str, Any]:
    root = repo_path(path)
    summary = read_json(root / "summary.json")
    plan = read_csv(root / "router_bias_plan.csv")
    deltas = read_csv(root / "router_bias_deltas.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "methods": summary.get("methods", []),
        "router_count": int(summary.get("router_count", 0)),
        "delta_rows": int(summary.get("delta_rows", len(plan))),
        "nonzero_delta_rows": int(summary.get("nonzero_delta_rows", len(deltas))),
        "writer_csv_ready": bool(summary.get("writer_csv_ready", False)),
        "load_stat": summary.get("load_stat"),
        "report": rel(root / "report.md"),
        "router_bias_plan": rel(root / "router_bias_plan.csv"),
        "router_bias_deltas": rel(root / "router_bias_deltas.csv"),
    }


def summarize_moe_router_bias_plan() -> dict[str, Any]:
    return summarize_moe_router_bias_plan_dir("results/moe_router_bias_plan")


def summarize_moe_confidence_blended_router_bias_plan() -> dict[str, Any]:
    return summarize_moe_router_bias_plan_dir("results/moe_confidence_blended_router_bias_plan")


def summarize_toy_moe_recipe_dir(path: str) -> dict[str, Any]:
    root = repo_path(path)
    summary = read_json(root / "summary.json")
    source_weights = read_csv(root / "source_weights_by_expert.csv")
    return {
        "summary": summary,
        "recipe_status": summary.get("recipe_status"),
        "recipe_kind": summary.get("recipe_kind"),
        "expert_weight_category": summary.get("expert_weight_category"),
        "expert_rule_count": int(summary.get("expert_rule_count", 0)),
        "tensor_rule_count": int(summary.get("tensor_rule_count", 0)),
        "source_weights_rows": int(len(source_weights)),
        "report": rel(root / "report.md"),
        "source_weights": rel(root / "source_weights_by_expert.csv"),
        "tensor_rules": rel(root / "tensor_rules.txt"),
        "writer_command": rel(root / "writer_command.txt"),
    }


def summarize_toy_moe_expert_weight_recipes() -> dict[str, Any]:
    return summarize_toy_moe_recipe_dir("results/toy_moe_expert_weight_recipes")


def summarize_toy_moe_output_projection_recipes() -> dict[str, Any]:
    return summarize_toy_moe_recipe_dir("results/toy_moe_output_projection_recipes")


def summarize_toy_moe_confidence_blended_recipes() -> dict[str, Any]:
    return summarize_toy_moe_recipe_dir("results/toy_moe_confidence_blended_recipes")


def summarize_moe_confidence_blended_combined_recipe() -> dict[str, Any]:
    root = repo_path("results/moe_confidence_blended_combined_recipe")
    summary = read_json(root / "summary.json")
    return {
        "summary": summary,
        "recipe_status": summary.get("recipe_status"),
        "tensor_rule_count": int(summary.get("tensor_rule_count", 0)),
        "alias_rule_count": int(summary.get("alias_rule_count", 0)),
        "router_bias_delta_rows": int(summary.get("router_bias_delta_rows", 0)),
        "freeze_router": bool(summary.get("freeze_router", False)),
        "dry_run": bool(summary.get("dry_run", False)),
        "report": rel(root / "report.md"),
        "writer_command": rel(root / "writer_command.txt"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_moe_routing_readiness() -> dict[str, Any]:
    summary = read_json("results/moe_routing_readiness/summary.json")
    router_readiness = read_csv("results/moe_routing_readiness/router_readiness.csv")
    expert_risks = read_csv("results/moe_routing_readiness/expert_load_risks.csv")
    specialization = read_csv("results/moe_routing_readiness/category_specialization.csv")
    return {
        "summary": summary,
        "readiness_status": summary.get("readiness_status"),
        "router_rows": int(len(router_readiness)),
        "expert_rows": int(len(expert_risks)),
        "specialization_rows": int(len(specialization)),
        "report": rel("results/moe_routing_readiness/report.md"),
        "router_readiness": rel("results/moe_routing_readiness/router_readiness.csv"),
        "expert_load_risks": rel("results/moe_routing_readiness/expert_load_risks.csv"),
        "category_specialization": rel("results/moe_routing_readiness/category_specialization.csv"),
    }


def summarize_toy_moe_merge() -> dict[str, Any]:
    summary = read_json("results/toy_moe_merge/summary.json")
    methods = read_csv("results/toy_moe_merge/method_metrics.csv")
    dispatch_modes = read_csv("results/toy_moe_merge/dispatch_mode_metrics.csv")
    connectivity = read_csv("results/toy_moe_merge/connectivity_summary.csv")
    router_summary = read_csv("results/toy_moe_merge/router_summary.csv")
    router_capacity = read_csv("results/toy_moe_merge/router_capacity_metrics.csv")
    expert_match = read_csv("results/toy_moe_merge/expert_match.csv")
    expert_regmean_layers = read_csv("results/toy_moe_merge/expert_regmean_layers.csv")
    expert_regmean_covariances = read_csv("results/toy_moe_merge/expert_regmean_covariances.csv")
    expert_sparse_task_vectors = read_csv("results/toy_moe_merge/expert_sparse_task_vectors.csv")
    expert_output_projection_weights = read_csv("results/toy_moe_merge/expert_output_projection_weights_by_expert.csv")
    confidence_blended_expert_weights_path = repo_path(
        "results/toy_moe_merge/confidence_blended_expert_weights_by_expert.csv"
    )
    confidence_blended_expert_weights = (
        read_csv(confidence_blended_expert_weights_path)
        if confidence_blended_expert_weights_path.exists()
        else pd.DataFrame()
    )
    router_weight_search = read_csv("results/toy_moe_merge/router_weight_search.csv")
    router_hessian_average = read_csv("results/toy_moe_merge/router_hessian_average.csv")
    router_kd_trace = read_csv("results/toy_moe_merge/router_kd_trace.csv")
    router_route_kd_trace = read_csv("results/toy_moe_merge/router_route_kd_trace.csv")
    unified_trace_path = repo_path("results/toy_moe_merge/unified_moe_trace.csv")
    unified_moe_trace = read_csv(unified_trace_path) if unified_trace_path.exists() else pd.DataFrame()
    unified_capacity_sweep_path = repo_path("results/toy_moe_merge/unified_moe_capacity_sweep.csv")
    unified_moe_capacity_sweep = (
        read_csv(unified_capacity_sweep_path) if unified_capacity_sweep_path.exists() else pd.DataFrame()
    )
    unified_output_projection_trace_path = repo_path("results/toy_moe_merge/unified_output_projection_moe_trace.csv")
    unified_output_projection_moe_trace = (
        read_csv(unified_output_projection_trace_path)
        if unified_output_projection_trace_path.exists()
        else pd.DataFrame()
    )
    unified_output_projection_capacity_sweep_path = repo_path(
        "results/toy_moe_merge/unified_output_projection_moe_capacity_sweep.csv"
    )
    unified_output_projection_moe_capacity_sweep = (
        read_csv(unified_output_projection_capacity_sweep_path)
        if unified_output_projection_capacity_sweep_path.exists()
        else pd.DataFrame()
    )
    unified_confidence_blended_trace_path = repo_path(
        "results/toy_moe_merge/unified_confidence_blended_moe_trace.csv"
    )
    unified_confidence_blended_moe_trace = (
        read_csv(unified_confidence_blended_trace_path)
        if unified_confidence_blended_trace_path.exists()
        else pd.DataFrame()
    )
    unified_confidence_blended_capacity_sweep_path = repo_path(
        "results/toy_moe_merge/unified_confidence_blended_moe_capacity_sweep.csv"
    )
    unified_confidence_blended_moe_capacity_sweep = (
        read_csv(unified_confidence_blended_capacity_sweep_path)
        if unified_confidence_blended_capacity_sweep_path.exists()
        else pd.DataFrame()
    )
    router_bias_trace_path = repo_path("results/toy_moe_merge/router_bias_capacity_trace.csv")
    router_bias_capacity_trace = read_csv(router_bias_trace_path) if router_bias_trace_path.exists() else pd.DataFrame()
    router_bias_sweep_path = repo_path("results/toy_moe_merge/router_bias_capacity_sweep.csv")
    router_bias_capacity_sweep = read_csv(router_bias_sweep_path) if router_bias_sweep_path.exists() else pd.DataFrame()
    output_projection_bias_trace_path = repo_path(
        "results/toy_moe_merge/output_projection_bias_capacity_trace.csv"
    )
    output_projection_bias_capacity_trace = (
        read_csv(output_projection_bias_trace_path)
        if output_projection_bias_trace_path.exists()
        else pd.DataFrame()
    )
    output_projection_bias_sweep_path = repo_path(
        "results/toy_moe_merge/output_projection_bias_capacity_sweep.csv"
    )
    output_projection_bias_capacity_sweep = (
        read_csv(output_projection_bias_sweep_path)
        if output_projection_bias_sweep_path.exists()
        else pd.DataFrame()
    )
    confidence_blended_bias_trace_path = repo_path("results/toy_moe_merge/confidence_blended_bias_capacity_trace.csv")
    confidence_blended_bias_capacity_trace = (
        read_csv(confidence_blended_bias_trace_path)
        if confidence_blended_bias_trace_path.exists()
        else pd.DataFrame()
    )
    confidence_blended_bias_sweep_path = repo_path("results/toy_moe_merge/confidence_blended_bias_capacity_sweep.csv")
    confidence_blended_bias_capacity_sweep = (
        read_csv(confidence_blended_bias_sweep_path)
        if confidence_blended_bias_sweep_path.exists()
        else pd.DataFrame()
    )
    selected_unified_capacity = (
        unified_moe_capacity_sweep[
            unified_moe_capacity_sweep["selected_by_select_capacity_aware_score"].astype(bool)
        ]
        if not unified_moe_capacity_sweep.empty
        else pd.DataFrame()
    )
    selected_unified_output_projection_capacity = (
        unified_output_projection_moe_capacity_sweep[
            unified_output_projection_moe_capacity_sweep["selected_by_select_capacity_aware_score"].astype(bool)
        ]
        if not unified_output_projection_moe_capacity_sweep.empty
        else pd.DataFrame()
    )
    selected_unified_confidence_blended_capacity = (
        unified_confidence_blended_moe_capacity_sweep[
            unified_confidence_blended_moe_capacity_sweep["selected_by_select_capacity_aware_score"].astype(bool)
        ]
        if not unified_confidence_blended_moe_capacity_sweep.empty
        else pd.DataFrame()
    )
    selected_router_bias_capacity = (
        router_bias_capacity_sweep[
            router_bias_capacity_sweep["selected_by_select_capacity_aware_score"].astype(bool)
        ]
        if not router_bias_capacity_sweep.empty
        else pd.DataFrame()
    )
    selected_output_projection_bias_capacity = (
        output_projection_bias_capacity_sweep[
            output_projection_bias_capacity_sweep["selected_by_select_capacity_aware_score"].astype(bool)
        ]
        if not output_projection_bias_capacity_sweep.empty
        else pd.DataFrame()
    )
    selected_confidence_blended_bias_capacity = (
        confidence_blended_bias_capacity_sweep[
            confidence_blended_bias_capacity_sweep["selected_by_select_capacity_aware_score"].astype(bool)
        ]
        if not confidence_blended_bias_capacity_sweep.empty
        else pd.DataFrame()
    )
    selected_router_weight = router_weight_search[
        router_weight_search["selected_by_guarded_calib_worst_loss"].astype(bool)
    ]
    router_calibration_sweep = read_csv("results/toy_moe_merge/router_calibration_sweep.csv")
    selected_router_calibration = router_calibration_sweep[
        router_calibration_sweep["selected_by_guarded_calib_worst_loss"].astype(bool)
    ]
    return {
        "summary": summary,
        "best_method": find_method(methods, summary["best_method"]),
        "dispatch_mode_rows": int(len(dispatch_modes)),
        "dispatch_robustness": summary.get("dispatch_robustness", {}),
        "connectivity_summary_rows": int(len(connectivity)),
        "connectivity_best": clean_row(connectivity.sort_values("barrier_worst_loss").iloc[0])
        if not connectivity.empty
        else None,
        "connectivity": summary.get("connectivity", {}),
        "router_capacity": summary.get("router_capacity", {}),
        "router_capacity_rows": int(len(router_capacity)),
        "all_weight_average": find_method(methods, "all_weight_average"),
        "expert_matched_average": find_method(methods, "expert_matched_average"),
        "matched_router_frozen_average": find_method(methods, "matched_router_frozen_average"),
        "expert_matched_regmean_average": find_method(methods, "expert_matched_regmean_average"),
        "expert_matched_ties_average": find_method(methods, "expert_matched_ties_average"),
        "expert_matched_dare_average": find_method(methods, "expert_matched_dare_average"),
        "expert_matched_ties_dare_average": find_method(methods, "expert_matched_ties_dare_average"),
        "matched_router_weight_search_average": find_method(methods, "matched_router_weight_search_average"),
        "matched_router_hessian_average": find_method(methods, "matched_router_hessian_average"),
        "matched_router_kd_average": find_method(methods, "matched_router_kd_average"),
        "matched_router_route_kd_average": find_method(methods, "matched_router_route_kd_average"),
        "matched_router_calibrated_average": find_method(methods, "matched_router_calibrated_average"),
        "matched_router_topk_calibrated_average": find_method(
            methods, "matched_router_topk_calibrated_average"
        ),
        "matched_router_sweep_selected_average": find_method(methods, "matched_router_sweep_selected_average"),
        "expert_weight_search_average": find_method(methods, "expert_weight_search_average"),
        "expert_weight_search_router_calibrated_average": find_method(
            methods, "expert_weight_search_router_calibrated_average"
        ),
        "expert_output_projection_average": find_method(methods, "expert_output_projection_average"),
        "expert_output_projection_router_calibrated_average": find_method(
            methods, "expert_output_projection_router_calibrated_average"
        ),
        "confidence_blended_expert_average": find_method(methods, "confidence_blended_expert_average"),
        "confidence_blended_router_calibrated_average": find_method(
            methods, "confidence_blended_router_calibrated_average"
        ),
        "unified_moe_average": find_method(methods, "unified_moe_average"),
        "unified_output_projection_moe_average": find_method(methods, "unified_output_projection_moe_average"),
        "unified_confidence_blended_moe_average": find_method(methods, "unified_confidence_blended_moe_average"),
        "unified_moe_bias_capacity_average": find_method(methods, "unified_moe_bias_capacity_average"),
        "unified_output_projection_bias_capacity_average": find_method(
            methods, "unified_output_projection_bias_capacity_average"
        ),
        "unified_confidence_blended_bias_capacity_average": find_method(
            methods, "unified_confidence_blended_bias_capacity_average"
        ),
        "route_aware_expert_average": find_method(methods, "route_aware_expert_average"),
        "matched_router_frozen_minus_all_weight_worst_acc": float(
            summary.get("matched_router_frozen_minus_all_weight_worst_acc", 0.0)
        ),
        "matched_router_calibrated_minus_all_weight_worst_acc": float(
            summary.get("matched_router_calibrated_minus_all_weight_worst_acc", 0.0)
        ),
        "matched_router_calibrated_minus_frozen_worst_acc": float(
            summary.get("matched_router_calibrated_minus_frozen_worst_acc", 0.0)
        ),
        "matched_router_topk_calibrated_minus_matched_calibrated_worst_acc": float(
            summary.get("matched_router_topk_calibrated_minus_matched_calibrated_worst_acc", 0.0)
        ),
        "expert_matched_regmean_minus_matched_frozen_worst_acc": float(
            summary.get("expert_matched_regmean_minus_matched_frozen_worst_acc", 0.0)
        ),
        "expert_regmean_layers": int(len(expert_regmean_layers)),
        "expert_regmean_covariance_rows": int(len(expert_regmean_covariances)),
        "expert_sparse_task_vector_rows": int(len(expert_sparse_task_vectors)),
        "expert_matched_ties_minus_matched_average_worst_acc": float(
            summary.get("expert_matched_ties_minus_matched_average_worst_acc", 0.0)
        ),
        "expert_matched_dare_minus_matched_average_worst_acc": float(
            summary.get("expert_matched_dare_minus_matched_average_worst_acc", 0.0)
        ),
        "expert_matched_ties_dare_minus_matched_average_worst_acc": float(
            summary.get("expert_matched_ties_dare_minus_matched_average_worst_acc", 0.0)
        ),
        "router_weight_search_rows": int(len(router_weight_search)),
        "router_weight_search_eligible_count": int(router_weight_search["eligible_by_route_guard"].sum()),
        "router_weight_search_selected": clean_row(selected_router_weight.iloc[0])
        if not selected_router_weight.empty
        else None,
        "router_hessian_rows": int(len(router_hessian_average)),
        "matched_router_hessian_minus_expert_matched_worst_acc": float(
            summary.get("matched_router_hessian_minus_expert_matched_worst_acc", 0.0)
        ),
        "matched_router_hessian_minus_matched_calibrated_worst_acc": float(
            summary.get("matched_router_hessian_minus_matched_calibrated_worst_acc", 0.0)
        ),
        "router_kd_rows": int(len(router_kd_trace)),
        "matched_router_kd_minus_expert_matched_worst_acc": float(
            summary.get("matched_router_kd_minus_expert_matched_worst_acc", 0.0)
        ),
        "matched_router_kd_minus_matched_calibrated_worst_acc": float(
            summary.get("matched_router_kd_minus_matched_calibrated_worst_acc", 0.0)
        ),
        "router_route_kd_rows": int(len(router_route_kd_trace)),
        "matched_router_route_kd_minus_expert_matched_worst_acc": float(
            summary.get("matched_router_route_kd_minus_expert_matched_worst_acc", 0.0)
        ),
        "matched_router_route_kd_minus_matched_calibrated_worst_acc": float(
            summary.get("matched_router_route_kd_minus_matched_calibrated_worst_acc", 0.0)
        ),
        "matched_router_route_kd_minus_router_kd_worst_acc": float(
            summary.get("matched_router_route_kd_minus_router_kd_worst_acc", 0.0)
        ),
        "router_calibration_sweep_rows": int(len(router_calibration_sweep)),
        "router_calibration_sweep_eligible_count": int(router_calibration_sweep["eligible_by_route_guard"].sum()),
        "router_calibration_sweep_selected": clean_row(selected_router_calibration.iloc[0])
        if not selected_router_calibration.empty
        else None,
        "expert_weight_search_router_calibrated_minus_all_weight_worst_acc": float(
            summary.get("expert_weight_search_router_calibrated_minus_all_weight_worst_acc", 0.0)
        ),
        "expert_weight_search_router_calibrated_minus_matched_calibrated_worst_acc": float(
            summary.get("expert_weight_search_router_calibrated_minus_matched_calibrated_worst_acc", 0.0)
        ),
        "expert_output_projection_rows": int(len(expert_output_projection_weights)),
        "expert_output_projection": summary.get("expert_output_projection", {}),
        "confidence_blended_expert_rows": int(len(confidence_blended_expert_weights)),
        "confidence_blended_expert": summary.get("confidence_blended_expert", {}),
        "expert_output_projection_router_calibrated_minus_all_weight_worst_acc": float(
            summary.get("expert_output_projection_router_calibrated_minus_all_weight_worst_acc", 0.0)
        ),
        "expert_output_projection_router_calibrated_minus_matched_calibrated_worst_acc": float(
            summary.get("expert_output_projection_router_calibrated_minus_matched_calibrated_worst_acc", 0.0)
        ),
        "unified_moe_trace_rows": int(len(unified_moe_trace)),
        "unified_moe_capacity_sweep_rows": int(len(unified_moe_capacity_sweep)),
        "unified_moe_capacity_sweep_selected": clean_row(selected_unified_capacity.iloc[0])
        if not selected_unified_capacity.empty
        else None,
        "unified_output_projection_moe_trace_rows": int(len(unified_output_projection_moe_trace)),
        "unified_output_projection_moe_capacity_sweep_rows": int(
            len(unified_output_projection_moe_capacity_sweep)
        ),
        "unified_output_projection_moe_capacity_sweep_selected": clean_row(
            selected_unified_output_projection_capacity.iloc[0]
        )
        if not selected_unified_output_projection_capacity.empty
        else None,
        "unified_confidence_blended_moe_trace_rows": int(len(unified_confidence_blended_moe_trace)),
        "unified_confidence_blended_moe_capacity_sweep_rows": int(
            len(unified_confidence_blended_moe_capacity_sweep)
        ),
        "unified_confidence_blended_moe_capacity_sweep_selected": clean_row(
            selected_unified_confidence_blended_capacity.iloc[0]
        )
        if not selected_unified_confidence_blended_capacity.empty
        else None,
        "router_bias_capacity_trace_rows": int(len(router_bias_capacity_trace)),
        "router_bias_capacity_sweep_rows": int(len(router_bias_capacity_sweep)),
        "router_bias_capacity_sweep_selected": clean_row(selected_router_bias_capacity.iloc[0])
        if not selected_router_bias_capacity.empty
        else None,
        "output_projection_bias_capacity_trace_rows": int(len(output_projection_bias_capacity_trace)),
        "output_projection_bias_capacity_sweep_rows": int(len(output_projection_bias_capacity_sweep)),
        "output_projection_bias_capacity_sweep_selected": clean_row(
            selected_output_projection_bias_capacity.iloc[0]
        )
        if not selected_output_projection_bias_capacity.empty
        else None,
        "confidence_blended_bias_capacity_trace_rows": int(len(confidence_blended_bias_capacity_trace)),
        "confidence_blended_bias_capacity_sweep_rows": int(len(confidence_blended_bias_capacity_sweep)),
        "confidence_blended_bias_capacity_sweep_selected": clean_row(
            selected_confidence_blended_bias_capacity.iloc[0]
        )
        if not selected_confidence_blended_bias_capacity.empty
        else None,
        "unified_moe_bias_capacity_minus_unified_worst_acc": float(
            summary.get("unified_moe_bias_capacity_minus_unified_worst_acc", 0.0)
        ),
        "unified_moe_minus_expert_search_worst_acc": float(
            summary.get("unified_moe_minus_expert_search_worst_acc", 0.0)
        ),
        "unified_moe_minus_expert_search_router_calibrated_worst_acc": float(
            summary.get("unified_moe_minus_expert_search_router_calibrated_worst_acc", 0.0)
        ),
        "unified_moe_minus_route_kd_worst_acc": float(summary.get("unified_moe_minus_route_kd_worst_acc", 0.0)),
        "unified_output_projection_moe_minus_unified_worst_acc": float(
            summary.get("unified_output_projection_moe_minus_unified_worst_acc", 0.0)
        ),
        "unified_output_projection_moe_minus_output_projection_router_calibrated_worst_acc": float(
            summary.get(
                "unified_output_projection_moe_minus_output_projection_router_calibrated_worst_acc",
                0.0,
            )
        ),
        "unified_confidence_blended_moe_minus_unified_worst_acc": float(
            summary.get("unified_confidence_blended_moe_minus_unified_worst_acc", 0.0)
        ),
        "unified_confidence_blended_moe_minus_unified_output_projection_worst_acc": float(
            summary.get("unified_confidence_blended_moe_minus_unified_output_projection_worst_acc", 0.0)
        ),
        "unified_output_projection_bias_capacity_minus_unified_output_projection_worst_acc": float(
            summary.get("unified_output_projection_bias_capacity_minus_unified_output_projection_worst_acc", 0.0)
        ),
        "unified_output_projection_bias_capacity_minus_unified_bias_capacity_worst_acc": float(
            summary.get("unified_output_projection_bias_capacity_minus_unified_bias_capacity_worst_acc", 0.0)
        ),
        "unified_confidence_blended_bias_capacity_minus_unified_confidence_blended_worst_acc": float(
            summary.get("unified_confidence_blended_bias_capacity_minus_unified_confidence_blended_worst_acc", 0.0)
        ),
        "unified_confidence_blended_bias_capacity_minus_unified_bias_capacity_worst_acc": float(
            summary.get("unified_confidence_blended_bias_capacity_minus_unified_bias_capacity_worst_acc", 0.0)
        ),
        "route_aware_minus_all_weight_worst_acc": float(summary["route_aware_minus_all_weight_worst_acc"]),
        "expert_match_mean_cosine": float(expert_match["output_cosine"].mean()),
        "router_rows": int(len(router_summary)),
        "report": rel("results/toy_moe_merge/report.md"),
        "method_metrics": rel("results/toy_moe_merge/method_metrics.csv"),
        "dispatch_mode_metrics": rel("results/toy_moe_merge/dispatch_mode_metrics.csv"),
        "connectivity_summary": rel("results/toy_moe_merge/connectivity_summary.csv"),
        "connectivity_path_metrics": rel("results/toy_moe_merge/connectivity_path_metrics.csv"),
        "connectivity_figure": rel("results/toy_moe_merge/connectivity_paths.png"),
        "router_summary": rel("results/toy_moe_merge/router_summary.csv"),
        "router_capacity_metrics": rel("results/toy_moe_merge/router_capacity_metrics.csv"),
        "expert_load": rel("results/toy_moe_merge/expert_load.csv"),
        "route_overlap": rel("results/toy_moe_merge/route_overlap.csv"),
        "expert_match": rel("results/toy_moe_merge/expert_match.csv"),
        "route_weights": rel("results/toy_moe_merge/route_weights_by_expert.csv"),
        "expert_regmean_layers_file": rel("results/toy_moe_merge/expert_regmean_layers.csv"),
        "expert_regmean_covariances": rel("results/toy_moe_merge/expert_regmean_covariances.csv"),
        "expert_sparse_task_vectors": rel("results/toy_moe_merge/expert_sparse_task_vectors.csv"),
        "expert_search_weights": rel("results/toy_moe_merge/expert_search_weights_by_expert.csv"),
        "expert_weight_search_trace": rel("results/toy_moe_merge/expert_weight_search_trace.csv"),
        "expert_output_projection_weights": rel("results/toy_moe_merge/expert_output_projection_weights_by_expert.csv"),
        "confidence_blended_expert_weights": rel(confidence_blended_expert_weights_path)
        if confidence_blended_expert_weights_path.exists()
        else None,
        "router_weight_search": rel("results/toy_moe_merge/router_weight_search.csv"),
        "router_hessian_average": rel("results/toy_moe_merge/router_hessian_average.csv"),
        "router_kd_trace": rel("results/toy_moe_merge/router_kd_trace.csv"),
        "router_route_kd_trace": rel("results/toy_moe_merge/router_route_kd_trace.csv"),
        "unified_moe_trace": rel(unified_trace_path) if unified_trace_path.exists() else None,
        "unified_moe_capacity_sweep": rel(unified_capacity_sweep_path)
        if unified_capacity_sweep_path.exists()
        else None,
        "unified_output_projection_moe_trace": rel(unified_output_projection_trace_path)
        if unified_output_projection_trace_path.exists()
        else None,
        "unified_output_projection_moe_capacity_sweep": rel(unified_output_projection_capacity_sweep_path)
        if unified_output_projection_capacity_sweep_path.exists()
        else None,
        "unified_confidence_blended_moe_trace": rel(unified_confidence_blended_trace_path)
        if unified_confidence_blended_trace_path.exists()
        else None,
        "unified_confidence_blended_moe_capacity_sweep": rel(unified_confidence_blended_capacity_sweep_path)
        if unified_confidence_blended_capacity_sweep_path.exists()
        else None,
        "router_bias_capacity_trace": rel(router_bias_trace_path) if router_bias_trace_path.exists() else None,
        "router_bias_capacity_sweep": rel(router_bias_sweep_path) if router_bias_sweep_path.exists() else None,
        "output_projection_bias_capacity_trace": rel(output_projection_bias_trace_path)
        if output_projection_bias_trace_path.exists()
        else None,
        "output_projection_bias_capacity_sweep": rel(output_projection_bias_sweep_path)
        if output_projection_bias_sweep_path.exists()
        else None,
        "confidence_blended_bias_capacity_trace": rel(confidence_blended_bias_trace_path)
        if confidence_blended_bias_trace_path.exists()
        else None,
        "confidence_blended_bias_capacity_sweep": rel(confidence_blended_bias_sweep_path)
        if confidence_blended_bias_sweep_path.exists()
        else None,
        "router_calibration_sweep": rel("results/toy_moe_merge/router_calibration_sweep.csv"),
        "figure": rel("results/toy_moe_merge/toy_moe_merge.png"),
    }


def summarize_toy_moe_routing_readiness() -> dict[str, Any]:
    summary = read_json("results/toy_moe_routing_readiness/summary.json")
    router_readiness = read_csv("results/toy_moe_routing_readiness/router_readiness.csv")
    expert_risks = read_csv("results/toy_moe_routing_readiness/expert_load_risks.csv")
    specialization = read_csv("results/toy_moe_routing_readiness/category_specialization.csv")
    all_weight = router_readiness[router_readiness["method"] == "all_weight_average"]
    return {
        "summary": summary,
        "readiness_status": summary.get("readiness_status"),
        "router_rows": int(len(router_readiness)),
        "expert_rows": int(len(expert_risks)),
        "specialization_rows": int(len(specialization)),
        "all_weight_router_actions": all_weight["recommended_action"].value_counts().to_dict(),
        "report": rel("results/toy_moe_routing_readiness/report.md"),
        "router_readiness": rel("results/toy_moe_routing_readiness/router_readiness.csv"),
        "expert_load_risks": rel("results/toy_moe_routing_readiness/expert_load_risks.csv"),
        "category_specialization": rel("results/toy_moe_routing_readiness/category_specialization.csv"),
    }


def summarize_toy_moe_method_selection() -> dict[str, Any]:
    summary = read_json("results/toy_moe_method_selection/summary.json")
    selection = read_csv("results/toy_moe_method_selection/method_selection.csv")
    frontier_path = repo_path("results/toy_moe_method_selection/sparse_pareto_frontier.csv")
    sparse_frontier = read_csv(frontier_path) if frontier_path.exists() else pd.DataFrame()
    recommended_method = summary.get("recommended_method")
    recommended_sparse_method = summary.get("recommended_sparse_method")
    recommended_sparse_capacity_aware_method = summary.get("recommended_sparse_capacity_aware_method")
    recommended = selection[selection["method"] == recommended_method]
    sparse_recommended = selection[selection["method"] == recommended_sparse_method]
    sparse_capacity_recommended = selection[selection["method"] == recommended_sparse_capacity_aware_method]
    all_weight = selection[selection["method"] == "all_weight_average"]
    return {
        "summary": summary,
        "recommended_method": recommended_method,
        "recommended_decision": summary.get("recommended_decision"),
        "recommended_row": clean_row(recommended.iloc[0]) if not recommended.empty else None,
        "recommended_sparse_dispatch_mode": summary.get("recommended_sparse_dispatch_mode"),
        "recommended_sparse_method": recommended_sparse_method,
        "recommended_sparse_decision": summary.get("recommended_sparse_decision"),
        "recommended_sparse_worst_acc": maybe_float(summary.get("recommended_sparse_worst_acc")),
        "recommended_sparse_row": clean_row(sparse_recommended.iloc[0]) if not sparse_recommended.empty else None,
        "recommended_sparse_capacity_aware_method": recommended_sparse_capacity_aware_method,
        "recommended_sparse_capacity_aware_decision": summary.get("recommended_sparse_capacity_aware_decision"),
        "recommended_sparse_capacity_aware_score": maybe_float(
            summary.get("recommended_sparse_capacity_aware_score")
        ),
        "recommended_sparse_capacity_aware_worst_acc": maybe_float(
            summary.get("recommended_sparse_capacity_aware_worst_acc")
        ),
        "recommended_sparse_capacity_aware_topk_overflow_fraction": maybe_float(
            summary.get("recommended_sparse_capacity_aware_topk_overflow_fraction")
        ),
        "recommended_sparse_capacity_aware_row": clean_row(sparse_capacity_recommended.iloc[0])
        if not sparse_capacity_recommended.empty
        else None,
        "sparse_pareto_frontier_rows": int(summary.get("sparse_pareto_frontier_rows", len(sparse_frontier))),
        "sparse_pareto_frontier_methods": list(summary.get("sparse_pareto_frontier_methods", [])),
        "sparse_pareto_frontier": rel(frontier_path),
        "all_weight_decision": None if all_weight.empty else str(all_weight.iloc[0]["decision"]),
        "all_weight_calibrate_count": 0 if all_weight.empty else int(all_weight.iloc[0]["calibrate_router_count"]),
        "report": rel("results/toy_moe_method_selection/report.md"),
        "method_selection": rel("results/toy_moe_method_selection/method_selection.csv"),
    }


def summarize_toy_moe_expert_remap_plan() -> dict[str, Any]:
    summary = read_json("results/toy_moe_expert_remap_plan/summary.json")
    remap = read_csv("results/toy_moe_expert_remap_plan/expert_remap.csv")
    return {
        "summary": summary,
        "remap_status": summary.get("remap_status"),
        "alias_rule_count": int(summary.get("alias_rule_count", 0)),
        "layer_aware_rule_count": int(summary.get("layer_aware_rule_count", 0)),
        "manual_review_count": int(summary.get("manual_review_count", 0)),
        "min_output_cosine": maybe_float(summary.get("min_output_cosine")),
        "mean_output_cosine": maybe_float(summary.get("mean_output_cosine")),
        "remap_rows": [clean_row(row) for _, row in remap.iterrows()],
        "report": rel("results/toy_moe_expert_remap_plan/report.md"),
        "expert_remap": rel("results/toy_moe_expert_remap_plan/expert_remap.csv"),
        "source_tensor_aliases": rel("results/toy_moe_expert_remap_plan/source_tensor_aliases.txt"),
        "writer_command": rel("results/toy_moe_expert_remap_plan/writer_command.txt"),
    }


def summarize_moe_layerwise_expert_remap_smoke() -> dict[str, Any]:
    summary = read_json("results/moe_layerwise_expert_remap_smoke/summary.json")
    checks = read_csv("results/moe_layerwise_expert_remap_smoke/checks.csv")
    remap = read_csv("results/moe_layerwise_expert_remap_smoke/expert_remap.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "input_rows": int(summary.get("input_rows", len(remap))),
        "alias_rule_count": int(summary.get("alias_rule_count", 0)),
        "layer_aware_rule_count": int(summary.get("layer_aware_rule_count", 0)),
        "manual_review_count": int(summary.get("manual_review_count", 0)),
        "failed_checks": int((~checks["passed"]).sum()),
        "report": rel("results/moe_layerwise_expert_remap_smoke/report.md"),
        "checks": rel("results/moe_layerwise_expert_remap_smoke/checks.csv"),
        "expert_remap": rel("results/moe_layerwise_expert_remap_smoke/expert_remap.csv"),
        "source_tensor_aliases": rel("results/moe_layerwise_expert_remap_smoke/source_tensor_aliases.txt"),
    }


def summarize_vllm_downstream_eval() -> dict[str, Any]:
    summary_path = repo_path("results/vllm_downstream_eval/summary.json")
    if not summary_path.exists():
        return {
            "status": "missing",
            "model": None,
            "base_url": None,
            "tasks": None,
            "metrics": [],
            "error": "results/vllm_downstream_eval/summary.json is missing",
            "report": None,
        }

    summary = read_json(summary_path)
    status = str(summary.get("status", "unknown"))
    metrics_path = repo_path("results/vllm_downstream_eval/metrics.csv")
    metrics = []
    if metrics_path.exists():
        metrics = [clean_row(row) for _, row in read_csv(metrics_path).iterrows()]
    model_summary_path = repo_path("results/vllm_downstream_eval/model_summary.csv")
    model_summary = []
    if model_summary_path.exists():
        model_summary = [clean_row(row) for _, row in read_csv(model_summary_path).iterrows()]
    eval_plan_path = repo_path("results/vllm_downstream_eval/eval_plan.csv")
    eval_plan = []
    if eval_plan_path.exists():
        eval_plan = [clean_row(row) for _, row in read_csv(eval_plan_path).iterrows()]
    probe = summary.get("endpoint_probe", {})
    return {
        "summary": summary,
        "status": status,
        "model": summary.get("model"),
        "models": summary.get("models", [summary.get("model")] if summary.get("model") else []),
        "model_count": int(summary.get("model_count", 1 if summary.get("model") else 0)),
        "best_avg_primary_model": summary.get("best_avg_primary_model"),
        "eval_plan": eval_plan,
        "eval_plan_model_count": len(eval_plan),
        "eval_plan_path": rel(eval_plan_path) if eval_plan_path.exists() else None,
        "candidate_table": summary.get("candidate_table"),
        "candidate_query": summary.get("candidate_query"),
        "base_url": summary.get("base_url"),
        "tasks": summary.get("tasks"),
        "example_source": summary.get("example_source"),
        "metrics": metrics,
        "model_summary": model_summary,
        "error": None if status == "complete" else f"{probe.get('error_type')}: {probe.get('error')}",
        "report": rel("results/vllm_downstream_eval/report.md"),
        "summary_path": rel(summary_path),
        "metrics_path": rel(metrics_path) if metrics_path.exists() else None,
        "model_summary_path": rel(model_summary_path) if model_summary_path.exists() else None,
    }


def summarize_vllm_downstream_eval_smoke() -> dict[str, Any]:
    summary = read_json("results/vllm_downstream_eval_smoke/smoke_summary.json")
    metrics = read_csv("results/vllm_downstream_eval_smoke/metrics.csv")
    model_summary = read_csv("results/vllm_downstream_eval_smoke/model_summary.csv")
    checks = summary.get("checks", {})
    return {
        "summary": summary,
        "status": summary.get("status"),
        "task_rows": int(checks.get("task_rows", len(metrics))),
        "mock_good_avg_primary_score": maybe_float(checks.get("mock_good_avg_primary_score")),
        "mock_bad_avg_primary_score": maybe_float(checks.get("mock_bad_avg_primary_score")),
        "mock_good_rank_1": bool(checks.get("mock_good_rank_1", False)),
        "model_summary": [clean_row(row) for _, row in model_summary.iterrows()],
        "report": rel("results/vllm_downstream_eval_smoke/smoke_report.md"),
        "metrics": rel("results/vllm_downstream_eval_smoke/metrics.csv"),
        "model_summary_path": rel("results/vllm_downstream_eval_smoke/model_summary.csv"),
        "smoke_summary": rel("results/vllm_downstream_eval_smoke/smoke_summary.json"),
    }


def summarize_vllm_checkpoint_eval_plan() -> dict[str, Any]:
    summary = read_json("results/vllm_checkpoint_eval_plan/summary.json")
    plan = read_csv("results/vllm_checkpoint_eval_plan/checkpoint_eval_plan.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "candidate_count": int(summary.get("candidate_count", len(plan))),
        "ready_to_host_count": int(summary.get("ready_to_host_count", 0)),
        "completed_eval_count": int(summary.get("completed_eval_count", 0)),
        "missing_checkpoint_count": int(summary.get("missing_checkpoint_count", 0)),
        "not_vllm_loadable_count": int(summary.get("not_vllm_loadable_count", 0)),
        "tasks": summary.get("tasks"),
        "plan_rows": [clean_row(row) for _, row in plan.iterrows()],
        "report": rel("results/vllm_checkpoint_eval_plan/report.md"),
        "plan_csv": rel("results/vllm_checkpoint_eval_plan/checkpoint_eval_plan.csv"),
        "shell_script": rel("results/vllm_checkpoint_eval_plan/serve_and_eval_commands.sh"),
    }


def summarize_vllm_checkpoint_eval_results() -> dict[str, Any]:
    root = repo_path("results/vllm_checkpoint_eval")
    if not root.exists():
        return {"completed_count": 0, "results": [], "best_result": None}
    results = []
    for summary_path in sorted(root.glob("*/summary.json")):
        output_dir = summary_path.parent
        summary = read_json(summary_path)
        metrics_path = output_dir / "metrics.csv"
        model_summary_path = output_dir / "model_summary.csv"
        metrics = read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        model_summary = read_csv(model_summary_path) if model_summary_path.exists() else pd.DataFrame()
        first_model = clean_row(model_summary.iloc[0]) if not model_summary.empty else {}
        results.append(
            {
                "method": output_dir.name,
                "status": summary.get("status"),
                "model": summary.get("model"),
                "models": summary.get("models", []),
                "example_source": summary.get("example_source"),
                "max_examples_per_task": summary.get("max_examples_per_task"),
                "tasks": summary.get("tasks"),
                "avg_primary_score": maybe_float(first_model.get("avg_primary_score")),
                "worst_primary_score": maybe_float(first_model.get("worst_primary_score")),
                "task_count": int(first_model.get("task_count", len(metrics)) or 0),
                "metrics": [clean_row(row) for _, row in metrics.iterrows()],
                "report": rel(output_dir / "report.md"),
                "summary_path": rel(summary_path),
                "metrics_path": rel(metrics_path) if metrics_path.exists() else None,
                "model_summary_path": rel(model_summary_path) if model_summary_path.exists() else None,
            }
        )
    completed = [row for row in results if row["status"] == "complete"]
    best = None
    if completed:
        best = max(completed, key=lambda row: (row.get("avg_primary_score") or 0.0, row.get("worst_primary_score") or 0.0))
    return {
        "completed_count": len(completed),
        "result_count": len(results),
        "results": results,
        "best_result": best,
    }


def summarize_vllm_source_merge_comparison() -> dict[str, Any]:
    summary = read_json("results/vllm_source_merge_comparison/summary.json")
    model_scores = read_csv("results/vllm_source_merge_comparison/model_scores.csv")
    task_metrics = read_csv("results/vllm_source_merge_comparison/task_metrics.csv")
    merge_tasks = task_metrics[task_metrics["role"] == "merge"]
    return {
        "summary": summary,
        "status": summary.get("status"),
        "source_model_count": int(summary.get("source_model_count", 0)),
        "merge_model": summary.get("merge_model"),
        "merge_rank_by_avg_primary": int(summary.get("merge_rank_by_avg_primary", 0)),
        "source_models_better_than_merge_count": int(summary.get("source_models_better_than_merge_count", 0)),
        "best_source_model": summary.get("best_source_model"),
        "best_source_display_name": summary.get("best_source_display_name"),
        "best_source_avg_primary_score": maybe_float(summary.get("best_source_avg_primary_score")),
        "best_source_worst_primary_score": maybe_float(summary.get("best_source_worst_primary_score")),
        "merge_avg_primary_score": maybe_float(summary.get("merge_avg_primary_score")),
        "merge_worst_primary_score": maybe_float(summary.get("merge_worst_primary_score")),
        "merge_delta_vs_best_source_avg_primary": maybe_float(
            summary.get("merge_delta_vs_best_source_avg_primary")
        ),
        "merge_delta_vs_best_source_worst_primary": maybe_float(
            summary.get("merge_delta_vs_best_source_worst_primary")
        ),
        "best_source_by_task": summary.get("best_source_by_task", {}),
        "merge_by_task": summary.get("merge_by_task", {}),
        "model_scores": [clean_row(row) for _, row in model_scores.iterrows()],
        "merge_task_metrics": [clean_row(row) for _, row in merge_tasks.iterrows()],
        "report": rel("results/vllm_source_merge_comparison/report.md"),
        "model_scores_path": rel("results/vllm_source_merge_comparison/model_scores.csv"),
        "task_metrics_path": rel("results/vllm_source_merge_comparison/task_metrics.csv"),
        "figure": rel("results/vllm_source_merge_comparison/source_vs_merge_primary_scores.png"),
    }


def summarize_probe_guided_dense_average_candidate() -> dict[str, Any]:
    summary = read_json("results/probe_guided_dense_average_candidate/summary.json")
    candidate_scores = read_csv("results/probe_guided_dense_average_candidate/candidate_scores.csv")
    eval_summary = summary.get("vllm_eval") or {}
    return {
        "summary": summary,
        "status": summary.get("status"),
        "candidate_id": summary.get("candidate_id"),
        "alpha": maybe_float(summary.get("alpha")),
        "beta": maybe_float(summary.get("beta")),
        "avg_nll": maybe_float(summary.get("avg_nll")),
        "worst_nll": maybe_float(summary.get("worst_nll")),
        "uniform_avg_gain": maybe_float(summary.get("uniform_avg_gain")),
        "uniform_worst_gain": maybe_float(summary.get("uniform_worst_gain")),
        "vllm_eval_status": eval_summary.get("status"),
        "vllm_avg_primary_score": maybe_float(eval_summary.get("avg_primary_score")) if eval_summary else None,
        "vllm_worst_primary_score": maybe_float(eval_summary.get("worst_primary_score")) if eval_summary else None,
        "delta_vs_uniform_avg_primary": maybe_float(eval_summary.get("delta_vs_uniform_avg_primary")) if eval_summary else None,
        "delta_vs_best_source_avg_primary": maybe_float(eval_summary.get("delta_vs_best_source_avg_primary")) if eval_summary else None,
        "task_metrics": eval_summary.get("task_metrics", []),
        "top_candidates": [clean_row(row) for _, row in candidate_scores.head(8).iterrows()],
        "report": rel("results/probe_guided_dense_average_candidate/report.md"),
        "candidate_scores_path": rel("results/probe_guided_dense_average_candidate/candidate_scores.csv"),
        "writer_command": rel("results/probe_guided_dense_average_candidate/writer_command.txt"),
        "vllm_eval_report": eval_summary.get("report"),
    }


def summarize_qwen_dense_guarded_candidate(output_dir: str) -> dict[str, Any]:
    summary = read_json(f"{output_dir}/summary.json")
    module_summary = read_csv(f"{output_dir}/module_conflict.csv")
    eval_summary = summary.get("vllm_eval") or {}
    module_rows = [clean_row(row) for _, row in module_summary.iterrows()]
    modules_by_group = {str(row["group"]): row for row in module_rows}
    norm_anchor = modules_by_group.get("norm_anchor", {})
    return {
        "summary": summary,
        "status": summary.get("status"),
        "candidate_id": summary.get("candidate_id"),
        "variant": summary.get("variant"),
        "checkpoint_output_dir": summary.get("checkpoint_output_dir"),
        "vllm_eval_status": eval_summary.get("status"),
        "vllm_avg_primary_score": maybe_float(eval_summary.get("avg_primary_score")) if eval_summary else None,
        "vllm_worst_primary_score": maybe_float(eval_summary.get("worst_primary_score")) if eval_summary else None,
        "delta_vs_global_bridge_avg_primary": maybe_float(eval_summary.get("delta_vs_global_bridge_avg_primary"))
        if eval_summary
        else None,
        "delta_vs_best_source_avg_primary": maybe_float(eval_summary.get("delta_vs_best_source_avg_primary"))
        if eval_summary
        else None,
        "task_metrics": eval_summary.get("task_metrics", []),
        "module_rows": module_rows,
        "modules_by_group": modules_by_group,
        "norm_anchor_mean_tensor_cosine": maybe_float(norm_anchor.get("mean_tensor_cosine")),
        "norm_anchor_sign_conflict_rate": maybe_float(norm_anchor.get("sign_conflict_rate")),
        "report": rel(f"{output_dir}/report.md"),
        "tensor_conflict": rel(f"{output_dir}/tensor_conflict.csv"),
        "module_conflict": rel(f"{output_dir}/module_conflict.csv"),
        "tensor_rules": rel(f"{output_dir}/tensor_rules.txt"),
        "writer_command": rel(f"{output_dir}/writer_command.txt"),
    }


def summarize_checkpoint_materialization_readiness() -> dict[str, Any]:
    summary = read_json("results/checkpoint_materialization_readiness/summary.json")
    readiness = read_csv("results/checkpoint_materialization_readiness/candidate_readiness.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "candidate_count": int(summary.get("candidate_count", len(readiness))),
        "materialized_count": int(summary.get("materialized_count", 0)),
        "blocked_by_placeholder_count": int(summary.get("blocked_by_placeholder_count", 0)),
        "ready_for_vllm_eval_count": int(summary.get("ready_for_vllm_eval_count", 0)),
        "completed_vllm_eval_count": int(summary.get("completed_vllm_eval_count", 0)),
        "toy_validation_only_count": int(summary.get("toy_validation_only_count", 0)),
        "rows": [clean_row(row) for _, row in readiness.iterrows()],
        "report": rel("results/checkpoint_materialization_readiness/report.md"),
        "readiness_csv": rel("results/checkpoint_materialization_readiness/candidate_readiness.csv"),
    }


def summarize_moe_materialization_pipeline_plan() -> dict[str, Any]:
    summary = read_json("results/moe_materialization_pipeline_plan/summary.json")
    gates = read_csv("results/moe_materialization_pipeline_plan/stage_gates.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "current_blocking_stage": summary.get("current_blocking_stage"),
        "gate_count": int(summary.get("gate_count", len(gates))),
        "ready_or_complete_count": int(summary.get("ready_or_complete_count", 0)),
        "waiting_or_blocked_count": int(summary.get("waiting_or_blocked_count", 0)),
        "recommended_first_moe_models": summary.get("recommended_first_moe_models", []),
        "gates": [clean_row(row) for _, row in gates.iterrows()],
        "report": rel("results/moe_materialization_pipeline_plan/report.md"),
        "stage_gates": rel("results/moe_materialization_pipeline_plan/stage_gates.csv"),
        "next_commands": rel("results/moe_materialization_pipeline_plan/next_commands.sh"),
    }


def summarize_model_averaging_literature_review() -> dict[str, Any]:
    summary = read_json("results/model_averaging_literature_review/summary.json")
    methods = read_csv("results/model_averaging_literature_review/method_matrix.csv")
    probes = read_csv("results/model_averaging_literature_review/probe_matrix.csv")
    moe_stages = read_csv("results/model_averaging_literature_review/moe_optimization_matrix.csv")
    sources = read_csv("results/model_averaging_literature_review/source_matrix.csv")
    return {
        "summary": summary,
        "source_count": int(summary.get("source_count", len(sources))),
        "method_family_count": int(summary.get("method_family_count", len(methods))),
        "probe_count": int(summary.get("probe_count", len(probes))),
        "moe_stage_count": int(summary.get("moe_stage_count", len(moe_stages))),
        "dense_recommendation": summary.get("dense_recommendation"),
        "moe_recommendation": summary.get("moe_recommendation"),
        "report": rel("results/model_averaging_literature_review/report.md"),
        "method_matrix": rel("results/model_averaging_literature_review/method_matrix.csv"),
        "probe_matrix": rel("results/model_averaging_literature_review/probe_matrix.csv"),
        "moe_optimization_matrix": rel("results/model_averaging_literature_review/moe_optimization_matrix.csv"),
        "source_matrix": rel("results/model_averaging_literature_review/source_matrix.csv"),
    }


def summarize_qwen_target_model_registry() -> dict[str, Any]:
    summary = read_json("results/qwen_target_model_registry/summary.json")
    models = read_csv("results/qwen_target_model_registry/model_registry.csv")
    scenarios = read_csv("results/qwen_target_model_registry/scenario_matrix.csv")
    eval_probes = read_csv("results/qwen_target_model_registry/eval_probe_matrix.csv")
    first_scenario = summary.get("recommended_first_scenario")
    first_rows = models[models["scenario"] == first_scenario]
    moe_scenario = summary.get("recommended_first_moe_scenario")
    moe_rows = models[models["scenario"] == moe_scenario]
    return {
        "summary": summary,
        "model_count": int(summary.get("model_count", len(models))),
        "dense_model_count": int(summary.get("dense_model_count", 0)),
        "moe_model_count": int(summary.get("moe_model_count", 0)),
        "official_count": int(summary.get("official_count", 0)),
        "downstream_or_third_party_count": int(summary.get("downstream_or_third_party_count", 0)),
        "ready_for_topology_inspect_count": int(summary.get("ready_for_topology_inspect_count", 0)),
        "manual_resolution_or_selection_count": int(summary.get("manual_resolution_or_selection_count", 0)),
        "scenario_count": int(summary.get("scenario_count", len(scenarios))),
        "eval_probe_count": int(summary.get("eval_probe_count", len(eval_probes))),
        "recommended_first_scenario": first_scenario,
        "recommended_first_models": [str(row["model_id"]) for _, row in first_rows.iterrows() if row["priority"] == "p0"],
        "recommended_first_moe_scenario": moe_scenario,
        "recommended_first_moe_models": [str(row["model_id"]) for _, row in moe_rows.iterrows() if row["priority"] == "p0"],
        "report": rel("results/qwen_target_model_registry/report.md"),
        "model_registry": rel("results/qwen_target_model_registry/model_registry.csv"),
        "scenario_matrix": rel("results/qwen_target_model_registry/scenario_matrix.csv"),
        "eval_probe_matrix": rel("results/qwen_target_model_registry/eval_probe_matrix.csv"),
    }


def summarize_moe_routing_probe_smoke() -> dict[str, Any]:
    summary = read_json("results/moe_routing_probe_smoke/summary.json")
    return {
        "summary": summary,
        "router_count": int(summary.get("router_count", 0)),
        "prompt_count": int(summary.get("prompt_count", 0)),
        "route_overlap_rows": int(summary.get("row_counts", {}).get("route_overlap", 0)),
        "router_summary_rows": int(summary.get("row_counts", {}).get("router_summary", 0)),
        "expert_load_rows": int(summary.get("row_counts", {}).get("expert_load", 0)),
        "report": rel("results/moe_routing_probe_smoke/report.md"),
        "summary_path": rel("results/moe_routing_probe_smoke/summary.json"),
        "manifest": rel("results/moe_routing_probe_smoke/manifest.json"),
        "router_summary": rel("results/moe_routing_probe_smoke/router_summary.csv"),
        "expert_load": rel("results/moe_routing_probe_smoke/expert_load.csv"),
        "route_overlap": rel("results/moe_routing_probe_smoke/route_overlap.csv"),
    }


def coverage_checklist() -> list[dict[str, str]]:
    return [
        {
            "item": "2D task-vector merge landscape",
            "status": "complete",
            "evidence": "Digits and CIFAR grid metrics plus merge landscape figures.",
        },
        {
            "item": "Per-task basin overlay",
            "status": "complete",
            "evidence": "results/digits_merge/figures/per_task_basin_overlay.png.",
        },
        {
            "item": "Task-arithmetic lambda sweep",
            "status": "complete",
            "evidence": "Digits, CIFAR, and Qwen path/lambda sweeps.",
        },
        {
            "item": "Merge-method overlay",
            "status": "complete",
            "evidence": "Digits method table and overlay cover average, task arithmetic, SLERP, TIES, DARE, TIES+DARE, Fisher, RegMean, layer-wise task arithmetic, and validation grid search.",
        },
        {
            "item": "Layer-wise interference atlas",
            "status": "complete",
            "evidence": "Digits, CIFAR, and pairwise single-digit conflict tables/figures.",
        },
        {
            "item": "One-class expert surrogate",
            "status": "complete",
            "evidence": "Ten single-digit experts and all 45 pairwise merges.",
        },
        {
            "item": "Randomness and alignment analysis",
            "status": "complete",
            "evidence": "Independent-initialization MLP path before/after Hungarian hidden-unit alignment.",
        },
        {
            "item": "Natural-image small-model case study",
            "status": "complete",
            "evidence": "CIFAR-10 vehicle/animal GroupNorm CNN merge landscape.",
        },
        {
            "item": "CLIP or ViT task-vector phase",
            "status": "complete",
            "evidence": "CIFAR100 ViT-style from-scratch transformer and ImageNet-pretrained ViT-B/16 frozen-backbone transfer merge studies are present.",
        },
        {
            "item": "Qwen-compatible LLM probe",
            "status": "complete",
            "evidence": "Safetensors probe and same-file smoke test.",
        },
        {
            "item": "Real Qwen LLM path sweep",
            "status": "complete",
            "evidence": "Qwen2.5-1.5B base-to-instruct path is evaluated with fixed NLL prompts plus GSM8K, MMLU, and HumanEval benchmark slices.",
        },
        {
            "item": "Multi-expert LLM merge",
            "status": "complete",
            "evidence": "Qwen2.5-0.5B base, Qwen2.5-0.5B-Instruct, and Qwen2.5-Coder-0.5B-Instruct are evaluated in a two-expert merge plane.",
        },
        {
            "item": "Formal LLM benchmark slices",
            "status": "complete",
            "evidence": "Representative Qwen2.5-1.5B benchmark slices cover MMLU, GSM8K, HumanEval canonical-solution NLL, and BeaverTails safety/refusal NLL.",
        },
        {
            "item": "vLLM hosted downstream evaluation",
            "status": "partial",
            "evidence": "scripts/run_vllm_downstream_eval.py can build a served-model eval plan from the Qwen target registry; the generic registry run remains endpoint_unavailable, while checkpoint-specific hosted eval is tracked separately.",
        },
        {
            "item": "Materialized checkpoint vLLM hosted eval",
            "status": "complete",
            "evidence": "results/vllm_checkpoint_eval/qwen_0_5b_instruct_coder_uniform_average/report.md contains a real vLLM-hosted GSM8K/MMLU/safety/HumanEval compile eval for the materialized Qwen2.5-0.5B uniform-average checkpoint.",
        },
        {
            "item": "Qwen source-vs-merge vLLM comparison",
            "status": "complete",
            "evidence": "results/vllm_source_merge_comparison/report.md compares Qwen2.5-0.5B base/instruct/coder source endpoints against the materialized uniform-average checkpoint under the same vLLM downstream tasks.",
        },
        {
            "item": "Probe-guided dense average candidate vLLM eval",
            "status": "complete",
            "evidence": "results/probe_guided_dense_average_candidate/report.md selects a non-uniform Qwen instruct/coder bridge from the NLL grid, materializes the same-shape checkpoint locally, and records its real vLLM downstream eval.",
        },
        {
            "item": "Qwen dense module-wise guard ablation vLLM eval",
            "status": "complete",
            "evidence": "results/qwen_dense_module_guarded_candidate/report.md, results/qwen_dense_norm_guarded_candidate/report.md, and results/qwen_dense_selective_norm_guarded_candidate/report.md compare module-level, norm-only, and selective-norm tensor-rule variants against the global bridge under the same vLLM downstream tasks.",
        },
        {
            "item": "vLLM downstream eval contract smoke",
            "status": "complete",
            "evidence": "results/vllm_downstream_eval_smoke/smoke_report.md validates the OpenAI-compatible HTTP request, answer parsing, scoring, model ranking, and artifact writing path using a local mock endpoint.",
        },
        {
            "item": "vLLM checkpoint eval plan",
            "status": "complete",
            "evidence": "results/vllm_checkpoint_eval_plan/report.md turns same-shape checkpoint candidates into one-checkpoint-at-a-time vLLM serve/eval commands while keeping missing checkpoints separate from completed metrics.",
        },
        {
            "item": "Checkpoint materialization readiness audit",
            "status": "complete",
            "evidence": "results/checkpoint_materialization_readiness/report.md audits writer commands, placeholders, dry-run outputs, checkpoint existence, and vLLM eval readiness in one table.",
        },
        {
            "item": "MoE materialization pipeline plan",
            "status": "complete",
            "evidence": "results/moe_materialization_pipeline_plan/report.md connects Qwen MoE target selection, topology, routing probe, readiness, route weights, expert remap, router-bias deltas, checkpoint writer, and vLLM eval gates.",
        },
        {
            "item": "Probe-guided Average decision report",
            "status": "complete",
            "evidence": "results/average_decision_report/report.md converts merge grids, conflict probes, and optional MoE routing probes into same-shape average decisions.",
        },
        {
            "item": "Dense/MoE averaging literature matrix",
            "status": "complete",
            "evidence": "results/model_averaging_literature_review/report.md maps recent model averaging and MoE merging papers to probes, failure signals, and same-shape writer actions.",
        },
        {
            "item": "Qwen target model registry",
            "status": "complete",
            "evidence": "results/qwen_target_model_registry/report.md maps representative official, third-party, downstream, and adapter-pool Qwen candidates to scenarios, eval slices, probes, and same-shape topology gates.",
        },
        {
            "item": "MoE same-shape averaging plan",
            "status": "complete",
            "evidence": "results/moe_average_plan/report.md maps router/expert probes into same-shape router, shared-module, expert, and adapter averaging actions.",
        },
        {
            "item": "Same-shape checkpoint writer",
            "status": "complete",
            "evidence": "scripts/write_same_shape_average_checkpoint.py writes same-shape safetensors checkpoints; results/same_shape_writer_smoke/report.md validates Qwen2.5-0.5B base/instruct/coder dry-run compatibility.",
        },
        {
            "item": "MoE tensor-rule writer materialization",
            "status": "complete",
            "evidence": "results/moe_tensor_rule_writer_smoke/report.md writes a tiny MoE-like safetensors checkpoint and verifies tensor-rule, freeze-router, router-bias additive deltas, and non-floating tensor behavior numerically.",
        },
        {
            "item": "MoE combined writer smoke",
            "status": "complete",
            "evidence": "results/moe_combined_writer_smoke/report.md verifies expert tensor rules, source expert alias remap, freeze-router, and router-bias additive deltas in one same-shape writer call.",
        },
        {
            "item": "MoE layer-wise expert remap smoke",
            "status": "complete",
            "evidence": "results/moe_layerwise_expert_remap_smoke/report.md verifies layer-scoped source tensor alias rules for real multi-layer MoE expert matching.",
        },
        {
            "item": "Checkpoint topology inspection",
            "status": "complete",
            "evidence": "results/checkpoint_topology_inspect/report.md inspects Qwen MoE/Dense configs and safetensors headers without loading weights.",
        },
        {
            "item": "Average candidate recipes",
            "status": "complete",
            "evidence": "results/average_candidate_recipes/report.md converts probe decisions into conservative same-shape materialization recipes and skips endpoint-only pseudo-averages.",
        },
        {
            "item": "MoE route-weight recipes",
            "status": "complete",
            "evidence": "results/moe_route_weight_recipes/report.md converts MoE routing/expert-load probes into tensor-rule files for same-shape checkpoint materialization; current recipe is waiting for real routing probe data.",
        },
        {
            "item": "MoE router-bias additive capacity plan",
            "status": "complete",
            "evidence": "results/moe_router_bias_plan/report.md converts expert_load.csv into writer-ready router-bias additive deltas for same-shape capacity correction.",
        },
        {
            "item": "MoE confidence-blended router-bias capacity plan",
            "status": "complete",
            "evidence": "results/moe_confidence_blended_router_bias_plan/report.md applies the same writer-ready capacity correction to the confidence-blended unified MoE candidate.",
        },
        {
            "item": "MoE searched expert-weight recipes",
            "status": "complete",
            "evidence": "results/toy_moe_expert_weight_recipes/report.md converts calibration-searched per-expert source weights into same-shape checkpoint writer tensor rules.",
        },
        {
            "item": "MoE output-projection expert-weight recipes",
            "status": "complete",
            "evidence": "results/toy_moe_output_projection_recipes/report.md converts route-conditioned output-space expert weights into same-shape checkpoint writer tensor rules.",
        },
        {
            "item": "MoE confidence-blended expert-weight recipes",
            "status": "complete",
            "evidence": "results/toy_moe_confidence_blended_recipes/report.md converts projection-confidence-gated expert weights into same-shape checkpoint writer tensor rules.",
        },
        {
            "item": "MoE confidence-blended combined materialization recipe",
            "status": "complete",
            "evidence": "results/moe_confidence_blended_combined_recipe/report.md composes expert weights, expert alias remap, and router-bias capacity deltas into one same-shape writer command.",
        },
        {
            "item": "MoE routing readiness diagnostics",
            "status": "complete",
            "evidence": "results/moe_routing_readiness/report.md turns router_summary, route_overlap, and expert_load CSVs into router collapse, drift, boundary-fragility, and expert-load risk actions.",
        },
        {
            "item": "MoE routing probe CLI",
            "status": "complete",
            "evidence": "scripts/probe_moe_routing.py captures MoE router hooks and writes router_summary.csv, expert_load.csv, optional route_overlap.csv, summary.json, and report.md; results/moe_routing_probe_smoke/report.md validates the contract on a tiny local MoE.",
        },
        {
            "item": "MoE routing probe smoke",
            "status": "complete",
            "evidence": "results/moe_routing_probe_smoke/report.md proves the routing probe captures two tiny MoE gates and produces router, expert-load, token-route, comparison, and route-overlap CSVs.",
        },
        {
            "item": "Toy MoE route-aware merge",
            "status": "complete",
            "evidence": "results/toy_moe_merge/report.md runs a small same-shape MoE averaging experiment showing expert-index mismatch and expert-matched/router-calibrated fixes.",
        },
        {
            "item": "Toy MoE multi-method routing readiness",
            "status": "complete",
            "evidence": "results/toy_moe_routing_readiness/report.md applies the generic readiness gate to toy MoE methods and flags all-weight routing drift separately from expert-matched/route-aware variants.",
        },
        {
            "item": "Toy MoE merge method selection",
            "status": "complete",
            "evidence": "results/toy_moe_method_selection/report.md combines method metrics, routing readiness, and sparse capacity overflow into materialization gates plus a hard-top2/overflow Pareto frontier.",
        },
        {
            "item": "Toy MoE expert remap plan",
            "status": "complete",
            "evidence": "results/toy_moe_expert_remap_plan/report.md turns expert-output matching into source tensor alias rules for same-shape checkpoint materialization.",
        },
        {
            "item": "Interactive explainer UI",
            "status": "complete",
            "evidence": "Dashboard includes a draggable precomputed merge-plane explorer with task-pair, method, objective, raw/normalized plane, alpha/beta, and lambda controls.",
        },
    ]


def build_summary() -> dict[str, Any]:
    experiments = {
        "digits_merge": summarize_digits(),
        "digit_pairwise_experts": summarize_pairwise(),
        "alignment_barrier": summarize_alignment(),
        "cifar_merge": summarize_cifar(),
        "cifar100_vit_merge": summarize_cifar100_vit(),
        "pretrained_vit_transfer_merge": summarize_pretrained_vit_transfer(),
        "qwen_path_sweep": summarize_qwen_path(),
        "qwen_gsm8k_slice": summarize_qwen_gsm8k(),
        "qwen_mmlu_slice": summarize_qwen_mmlu(),
        "qwen_humaneval_nll_slice": summarize_qwen_humaneval(),
        "qwen_safety_refusal_slice": summarize_qwen_safety(),
        "qwen_multi_expert_merge": summarize_qwen_multi_expert(),
        "qwen_probe_smoke": summarize_qwen_probe_smoke(),
        "average_decision_report": summarize_average_decision_report(),
        "model_averaging_literature_review": summarize_model_averaging_literature_review(),
        "qwen_target_model_registry": summarize_qwen_target_model_registry(),
        "moe_routing_probe_smoke": summarize_moe_routing_probe_smoke(),
        "moe_average_plan": summarize_moe_average_plan(),
        "same_shape_writer_smoke": summarize_same_shape_writer_smoke(),
        "moe_tensor_rule_writer_smoke": summarize_moe_tensor_rule_writer_smoke(),
        "moe_combined_writer_smoke": summarize_moe_combined_writer_smoke(),
        "checkpoint_topology_inspect": summarize_checkpoint_topology(),
        "average_candidate_recipes": summarize_average_candidate_recipes(),
        "moe_route_weight_recipes": summarize_moe_route_weight_recipes(),
        "moe_router_bias_plan": summarize_moe_router_bias_plan(),
        "moe_confidence_blended_router_bias_plan": summarize_moe_confidence_blended_router_bias_plan(),
        "toy_moe_expert_weight_recipes": summarize_toy_moe_expert_weight_recipes(),
        "toy_moe_output_projection_recipes": summarize_toy_moe_output_projection_recipes(),
        "toy_moe_confidence_blended_recipes": summarize_toy_moe_confidence_blended_recipes(),
        "moe_confidence_blended_combined_recipe": summarize_moe_confidence_blended_combined_recipe(),
        "moe_routing_readiness": summarize_moe_routing_readiness(),
        "toy_moe_merge": summarize_toy_moe_merge(),
        "toy_moe_routing_readiness": summarize_toy_moe_routing_readiness(),
        "toy_moe_method_selection": summarize_toy_moe_method_selection(),
        "toy_moe_expert_remap_plan": summarize_toy_moe_expert_remap_plan(),
        "moe_layerwise_expert_remap_smoke": summarize_moe_layerwise_expert_remap_smoke(),
        "vllm_downstream_eval": summarize_vllm_downstream_eval(),
        "vllm_downstream_eval_smoke": summarize_vllm_downstream_eval_smoke(),
        "vllm_checkpoint_eval_plan": summarize_vllm_checkpoint_eval_plan(),
        "vllm_checkpoint_eval_results": summarize_vllm_checkpoint_eval_results(),
        "vllm_source_merge_comparison": summarize_vllm_source_merge_comparison(),
        "probe_guided_dense_average_candidate": summarize_probe_guided_dense_average_candidate(),
        "qwen_dense_module_guarded_candidate": summarize_qwen_dense_guarded_candidate(
            "results/qwen_dense_module_guarded_candidate"
        ),
        "qwen_dense_norm_guarded_candidate": summarize_qwen_dense_guarded_candidate(
            "results/qwen_dense_norm_guarded_candidate"
        ),
        "qwen_dense_selective_norm_guarded_candidate": summarize_qwen_dense_guarded_candidate(
            "results/qwen_dense_selective_norm_guarded_candidate"
        ),
        "checkpoint_materialization_readiness": summarize_checkpoint_materialization_readiness(),
        "moe_materialization_pipeline_plan": summarize_moe_materialization_pipeline_plan(),
    }
    coverage = coverage_checklist()
    counts = {
        status: sum(1 for item in coverage if item["status"] == status)
        for status in ("complete", "partial", "missing")
    }
    overall_status = "complete" if counts["partial"] == 0 and counts["missing"] == 0 else "partial_complete"
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "overall_status": overall_status,
        "coverage_counts": counts,
        "coverage": coverage,
        "experiments": experiments,
        "reproduction_commands": [
            "PYTHONPATH=src python scripts/run_digits_merge.py --output-dir results/digits_merge --device cpu",
            "PYTHONPATH=src python scripts/run_digit_pairwise_experts.py --output-dir results/digit_pairwise_experts --device cpu",
            "PYTHONPATH=src python scripts/run_alignment_barrier.py --output-dir results/alignment_barrier --device cpu",
            "PYTHONPATH=src python scripts/run_cifar_merge.py --output-dir results/cifar_merge",
            "PYTHONPATH=src python scripts/run_cifar100_vit_merge.py --output-dir results/cifar100_vit_merge",
            "PYTHONPATH=src python scripts/run_pretrained_vit_transfer_merge.py --output-dir results/pretrained_vit_transfer_merge",
            "PYTHONPATH=src python scripts/run_qwen_path_sweep.py --output-dir results/qwen_path_sweep --dtype bfloat16 --max-length 384",
            "PYTHONPATH=src python scripts/run_qwen_gsm8k_slice.py --output-dir results/qwen_gsm8k_slice",
            "PYTHONPATH=src python scripts/run_qwen_mmlu_slice.py --output-dir results/qwen_mmlu_slice",
            "PYTHONPATH=src python scripts/run_qwen_humaneval_nll_slice.py --output-dir results/qwen_humaneval_nll_slice",
            "PYTHONPATH=src python scripts/run_qwen_safety_refusal_slice.py --output-dir results/qwen_safety_refusal_slice",
            "PYTHONPATH=src python scripts/run_qwen_multi_expert_merge.py --output-dir results/qwen_multi_expert_merge",
            "PYTHONPATH=src python scripts/run_toy_moe_merge.py --output-dir results/toy_moe_merge --device cpu",
            "PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir results/toy_moe_merge --output-dir results/toy_moe_routing_readiness --topology-summary ''",
            "PYTHONPATH=src python scripts/select_moe_merge_method.py",
            "PYTHONPATH=src python scripts/build_moe_expert_remap_plan.py",
            "PYTHONPATH=scripts python scripts/smoke_moe_layerwise_expert_remap.py --output-dir results/moe_layerwise_expert_remap_smoke",
            (
                "python scripts/run_vllm_downstream_eval.py "
                "--candidate-table results/qwen_target_model_registry/model_registry.csv "
                "--candidate-method-column model_id --candidate-model-id-template '{model_id}' "
                "--candidate-query \"priority == 'p0' and phase in ['dense_7b', 'moe_30b_a3b']\" "
                "--base-url http://HOST:PORT/v1 --tasks gsm8k,mmlu,safety,humaneval_compile"
            ),
            "python scripts/build_model_averaging_literature_review.py",
            "python scripts/smoke_moe_routing_probe_contract.py",
            "python scripts/smoke_vllm_downstream_eval_contract.py --output-dir results/vllm_downstream_eval_smoke",
            "PYTHONPATH=src python scripts/build_vllm_checkpoint_eval_plan.py --output-dir results/vllm_checkpoint_eval_plan",
            "PYTHONPATH=src python scripts/build_vllm_source_merge_comparison.py --output-dir results/vllm_source_merge_comparison",
            "PYTHONPATH=src python scripts/build_probe_guided_dense_average_candidate.py --output-dir results/probe_guided_dense_average_candidate",
            "PYTHONPATH=src python scripts/build_qwen_dense_module_guarded_candidate.py --output-dir results/qwen_dense_module_guarded_candidate --variant module_guarded",
            "PYTHONPATH=src python scripts/build_qwen_dense_module_guarded_candidate.py --output-dir results/qwen_dense_norm_guarded_candidate --variant norm_only",
            "PYTHONPATH=src python scripts/build_qwen_dense_module_guarded_candidate.py --output-dir results/qwen_dense_selective_norm_guarded_candidate --variant selective_norm",
            "PYTHONPATH=src python scripts/build_average_decision_report.py",
            "python scripts/build_qwen_target_model_registry.py",
            "PYTHONPATH=src python scripts/build_moe_average_plan.py",
            "python scripts/write_same_shape_average_checkpoint.py --base BASE --source expert=EXPERT --dry-run --output-dir results/same_shape_writer_smoke",
            "python scripts/smoke_moe_tensor_rule_writer.py --output-dir results/moe_tensor_rule_writer_smoke",
            "PYTHONPATH=src python scripts/smoke_moe_combined_writer.py --output-dir results/moe_combined_writer_smoke",
            "python scripts/inspect_checkpoint_topology.py --model NAME=MODEL_PATH --output-dir results/checkpoint_topology_inspect",
            "PYTHONPATH=src python scripts/build_average_candidate_recipes.py",
            "PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code",
            "PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code --source general --source code",
            "PYTHONPATH=src python scripts/build_moe_router_bias_plan.py --router-dir results/toy_moe_merge --method unified_moe_average --router-bias-template '{router}.bias'",
            "PYTHONPATH=src python scripts/build_moe_router_bias_plan.py --router-dir results/toy_moe_merge --method unified_confidence_blended_moe_average --output-dir results/moe_confidence_blended_router_bias_plan --router-bias-template '{router}.bias'",
            "PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --output-dir results/toy_moe_expert_weight_recipes --expert-weight-csv results/toy_moe_merge/expert_search_weights_by_expert.csv --source general --source code --checkpoint-output-dir results/checkpoints/toy_moe_expert_weight_candidate --topology-summary ''",
            "PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --output-dir results/toy_moe_output_projection_recipes --expert-weight-csv results/toy_moe_merge/expert_output_projection_weights_by_expert.csv --expert-weight-category combined --source general --source code --checkpoint-output-dir results/checkpoints/toy_moe_output_projection_candidate --topology-summary ''",
            "PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --output-dir results/toy_moe_confidence_blended_recipes --expert-weight-csv results/toy_moe_merge/confidence_blended_expert_weights_by_expert.csv --source general --source code --checkpoint-output-dir results/checkpoints/toy_moe_confidence_blended_candidate --topology-summary ''",
            "PYTHONPATH=src python scripts/build_moe_combined_materialization_recipe.py",
            "PYTHONPATH=src python scripts/build_checkpoint_materialization_readiness.py --output-dir results/checkpoint_materialization_readiness",
            "PYTHONPATH=src python scripts/build_moe_materialization_pipeline_plan.py --output-dir results/moe_materialization_pipeline_plan",
            "PYTHONPATH=src python scripts/build_dashboard.py --output-dir results/dashboard",
            "PYTHONPATH=src python scripts/collect_results.py",
        ],
    }


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_markdown(summary: dict[str, Any]) -> str:
    exp = summary["experiments"]
    digits = exp["digits_merge"]
    pairwise = exp["digit_pairwise_experts"]
    alignment = exp["alignment_barrier"]["summary"]
    cifar = exp["cifar_merge"]
    vit = exp["cifar100_vit_merge"]
    pretrained_vit = exp["pretrained_vit_transfer_merge"]
    qwen = exp["qwen_path_sweep"]
    gsm8k = exp["qwen_gsm8k_slice"]
    mmlu = exp["qwen_mmlu_slice"]
    humaneval = exp["qwen_humaneval_nll_slice"]
    safety = exp["qwen_safety_refusal_slice"]
    qwen_multi = exp["qwen_multi_expert_merge"]
    qwen_multi_conflict = qwen_multi["instruct_coder_conflict"] or {}
    average_decision = exp["average_decision_report"]
    literature_review = exp["model_averaging_literature_review"]
    qwen_registry = exp["qwen_target_model_registry"]
    routing_probe_smoke = exp["moe_routing_probe_smoke"]
    moe_average_plan = exp["moe_average_plan"]
    writer_smoke = exp["same_shape_writer_smoke"]
    moe_tensor_rule_writer_smoke = exp["moe_tensor_rule_writer_smoke"]
    moe_combined_writer_smoke = exp["moe_combined_writer_smoke"]
    topology = exp["checkpoint_topology_inspect"]
    moe_models = [model for model in topology["models"] if model.get("config", {}).get("is_moe_config")]
    recipes = exp["average_candidate_recipes"]
    route_weight_recipes = exp["moe_route_weight_recipes"]
    router_bias_plan = exp["moe_router_bias_plan"]
    confidence_blended_router_bias_plan = exp["moe_confidence_blended_router_bias_plan"]
    toy_expert_weight_recipes = exp["toy_moe_expert_weight_recipes"]
    toy_output_projection_recipes = exp["toy_moe_output_projection_recipes"]
    toy_confidence_blended_recipes = exp["toy_moe_confidence_blended_recipes"]
    confidence_blended_combined_recipe = exp["moe_confidence_blended_combined_recipe"]
    routing_readiness = exp["moe_routing_readiness"]
    toy_moe = exp["toy_moe_merge"]
    selected_unified_capacity = toy_moe.get("unified_moe_capacity_sweep_selected") or {}
    selected_unified_output_projection_capacity = (
        toy_moe.get("unified_output_projection_moe_capacity_sweep_selected") or {}
    )
    selected_unified_confidence_blended_capacity = (
        toy_moe.get("unified_confidence_blended_moe_capacity_sweep_selected") or {}
    )
    selected_router_bias_capacity = toy_moe.get("router_bias_capacity_sweep_selected") or {}
    selected_output_projection_bias_capacity = toy_moe.get("output_projection_bias_capacity_sweep_selected") or {}
    selected_confidence_blended_bias_capacity = toy_moe.get("confidence_blended_bias_capacity_sweep_selected") or {}
    toy_moe_readiness = exp["toy_moe_routing_readiness"]
    toy_moe_selection = exp["toy_moe_method_selection"]
    toy_moe_remap = exp["toy_moe_expert_remap_plan"]
    layerwise_remap_smoke = exp["moe_layerwise_expert_remap_smoke"]
    vllm_eval = exp["vllm_downstream_eval"]
    vllm_eval_smoke = exp["vllm_downstream_eval_smoke"]
    vllm_checkpoint_eval_plan = exp["vllm_checkpoint_eval_plan"]
    vllm_checkpoint_eval_results = exp["vllm_checkpoint_eval_results"]
    vllm_checkpoint_best = vllm_checkpoint_eval_results.get("best_result") or {}
    vllm_source_merge = exp["vllm_source_merge_comparison"]
    probe_guided_dense = exp["probe_guided_dense_average_candidate"]
    qwen_dense_module_guarded = exp["qwen_dense_module_guarded_candidate"]
    qwen_dense_norm_guarded = exp["qwen_dense_norm_guarded_candidate"]
    qwen_dense_selective_norm_guarded = exp["qwen_dense_selective_norm_guarded_candidate"]
    materialization_readiness = exp["checkpoint_materialization_readiness"]
    moe_pipeline_plan = exp["moe_materialization_pipeline_plan"]
    coverage_counts = summary["coverage_counts"]
    lines = [
        "# Result Summary",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Coverage",
        "",
        (
            f"Complete: `{coverage_counts['complete']}`; "
            f"partial: `{coverage_counts['partial']}`; "
            f"missing: `{coverage_counts['missing']}`."
        ),
        "",
        "| item | status | evidence |",
        "| --- | --- | --- |",
    ]
    for item in summary["coverage"]:
        lines.append(f"| {item['item']} | {item['status']} | {item['evidence']} |")
    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            "| experiment | metric | value |",
            "| --- | --- | ---: |",
            (
                "| digits merge | linear-average worst accuracy | "
                f"{fmt(digits['linear_average']['worst_acc'])} |"
            ),
            (
                "| digits merge | layer-wise task arithmetic worst accuracy | "
                f"{fmt(digits['layerwise_task_arithmetic']['worst_acc'])} |"
            ),
            (
                "| digits merge | RegMean linear-layer worst accuracy | "
                f"{fmt(digits['regmean_linear']['worst_acc'])} |"
            ),
            (
                "| digits merge | max grid worst accuracy | "
                f"{fmt(digits['grid']['max_worst_acc'])} |"
            ),
            (
                "| digits merge | global task-vector cosine | "
                f"{fmt(digits['summary']['global_task_vector_cosine'])} |"
            ),
            (
                "| single-digit pairs | mean linear worst accuracy | "
                f"{fmt(pairwise['mean_linear_worst_acc'])} |"
            ),
            (
                "| single-digit pairs | weighted conflict vs drop Spearman | "
                f"{fmt(pairwise['spearman_vs_linear_drop']['weighted_conflict'])} |"
            ),
            (
                "| alignment | midpoint accuracy before to after | "
                f"{fmt(alignment['midpoint_before_acc'])} to {fmt(alignment['midpoint_after_acc'])} |"
            ),
            (
                "| alignment | loss barrier before to after | "
                f"{fmt(alignment['barrier_before'])} to {fmt(alignment['barrier_after'])} |"
            ),
            (
                "| CIFAR | linear-average worst accuracy | "
                f"{fmt(cifar['linear_average']['worst_acc'])} |"
            ),
            (
                "| CIFAR | validation-grid best worst accuracy | "
                f"{fmt(cifar['validation_grid_best']['worst_acc'])} |"
            ),
            (
                "| CIFAR100 ViT-style | linear-average worst accuracy | "
                f"{fmt(vit['linear_average']['worst_acc'])} |"
            ),
            (
                "| CIFAR100 ViT-style | best method worst accuracy | "
                f"{fmt(vit['best_method']['worst_acc'])} |"
            ),
            (
                "| pretrained ViT transfer | linear-average worst accuracy | "
                f"{fmt(pretrained_vit['linear_average']['worst_acc'])} |"
            ),
            (
                "| pretrained ViT transfer | best method worst accuracy | "
                f"{fmt(pretrained_vit['best_method']['worst_acc'])} |"
            ),
            (
                "| Qwen path | best average-NLL lambda | "
                f"{fmt(qwen['best_avg']['lambda'], 2)} |"
            ),
            (
                "| Qwen path | instruction NLL at base to best | "
                f"{fmt(qwen['lambda_0']['instruction_nll'])} to {fmt(qwen['best_instruction']['instruction_nll'])} |"
            ),
            (
                "| Qwen GSM8K slice | best strict exact match | "
                f"{fmt(gsm8k['best_strict']['exact_match'])} at lambda {fmt(gsm8k['best_strict']['lambda'], 2)} |"
            ),
            (
                "| Qwen GSM8K slice | best loose exact match | "
                f"{fmt(gsm8k['best_loose']['loose_exact_match'])} at lambda {fmt(gsm8k['best_loose']['lambda'], 2)} |"
            ),
            (
                "| Qwen MMLU slice | best accuracy | "
                f"{fmt(mmlu['best_accuracy']['accuracy'])} at lambda {fmt(mmlu['best_accuracy']['lambda'], 2)} |"
            ),
            (
                "| Qwen MMLU slice | best correct / total | "
                f"{int(mmlu['best_accuracy']['accuracy_count'])}/{int(mmlu['best_accuracy']['examples'])} |"
            ),
            (
                "| Qwen HumanEval NLL slice | best solution NLL | "
                f"{fmt(humaneval['best_solution_nll']['avg_solution_nll'])} at lambda {fmt(humaneval['best_solution_nll']['lambda'], 2)} |"
            ),
            (
                "| Qwen safety/refusal slice | best avg safety NLL | "
                f"{fmt(safety['best_avg_safety_nll']['avg_safety_nll'])} at lambda {fmt(safety['best_avg_safety_nll']['lambda'], 2)} |"
            ),
            (
                "| Qwen multi-expert | best average-NLL method | "
                f"{qwen_multi['best_avg']['method']} ({fmt(qwen_multi['best_avg']['avg_nll'])}) |"
            ),
            (
                "| Qwen multi-expert | linear-average avg / worst NLL | "
                f"{fmt(qwen_multi['linear_average']['avg_nll'])} / {fmt(qwen_multi['linear_average']['worst_nll'])} |"
            ),
            (
                "| Qwen multi-expert | instruct/coder weighted conflict | "
                f"{fmt(qwen_multi_conflict.get('weighted_conflict'))} |"
            ),
            (
                "| toy MoE route-aware merge | all-weight average worst accuracy | "
                f"{fmt(toy_moe['all_weight_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | expert-matched average worst accuracy | "
                f"{fmt(toy_moe['expert_matched_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE connectivity | best path / barrier | "
                f"{toy_moe['connectivity']['best_path']} / {fmt(toy_moe['connectivity']['best_barrier_worst_loss'])} |"
            ),
            (
                "| toy MoE connectivity | direct unmatched barrier | "
                f"{fmt(toy_moe['connectivity']['direct_unmatched_barrier_worst_loss'])} |"
            ),
            (
                "| toy MoE connectivity | direct matched barrier | "
                f"{fmt(toy_moe['connectivity']['direct_matched_barrier_worst_loss'])} |"
            ),
            (
                "| toy MoE route-aware merge | matched + router-frozen worst accuracy | "
                f"{fmt(toy_moe['matched_router_frozen_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | expert-matched RegMean worst accuracy | "
                f"{fmt(toy_moe['expert_matched_regmean_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | expert-matched RegMean delta vs frozen | "
                f"{fmt(toy_moe['expert_matched_regmean_minus_matched_frozen_worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | expert-matched TIES worst accuracy | "
                f"{fmt(toy_moe['expert_matched_ties_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | expert-matched DARE worst accuracy | "
                f"{fmt(toy_moe['expert_matched_dare_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | expert-matched TIES+DARE worst accuracy | "
                f"{fmt(toy_moe['expert_matched_ties_dare_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | best sparse expert delta vs matched average | "
                f"{fmt(max(toy_moe['expert_matched_ties_minus_matched_average_worst_acc'], toy_moe['expert_matched_dare_minus_matched_average_worst_acc'], toy_moe['expert_matched_ties_dare_minus_matched_average_worst_acc']))} |"
            ),
            (
                "| toy MoE route-aware merge | guarded router-weight selected general/code | "
                f"{fmt(toy_moe['router_weight_search_selected']['router_weight_general'], 2)} / "
                f"{fmt(toy_moe['router_weight_search_selected']['router_weight_code'], 2)} |"
            ),
            (
                "| toy MoE route-aware merge | guarded router-weight eligible / total | "
                f"{toy_moe['router_weight_search_eligible_count']} / {toy_moe['router_weight_search_rows']} |"
            ),
            (
                "| toy MoE route-aware merge | matched + router-weight-search worst accuracy | "
                f"{fmt(toy_moe['matched_router_weight_search_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | matched + Hessian-router average worst accuracy | "
                f"{fmt(toy_moe['matched_router_hessian_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | matched + Router-KD average worst accuracy | "
                f"{fmt(toy_moe['matched_router_kd_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | matched + route-KD average worst accuracy | "
                f"{fmt(toy_moe['matched_router_route_kd_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | matched + router-calibrated worst accuracy | "
                f"{fmt(toy_moe['matched_router_calibrated_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | matched + router-topk-calibrated worst accuracy | "
                f"{fmt(toy_moe['matched_router_topk_calibrated_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE hard dispatch | matched + router-calibrated hard top-1 worst accuracy | "
                f"{fmt(toy_moe['dispatch_robustness']['matched_router_calibrated_hard_top1_worst_acc'])} |"
            ),
            (
                "| toy MoE hard dispatch | matched + router-calibrated hard top-2 worst accuracy | "
                f"{fmt(toy_moe['dispatch_robustness']['matched_router_calibrated_hard_top2_worst_acc'])} |"
            ),
            (
                "| toy MoE hard dispatch | matched + router-topk-calibrated hard top-2 worst accuracy | "
                f"{fmt(toy_moe['dispatch_robustness']['matched_router_topk_calibrated_hard_top2_worst_acc'])} |"
            ),
            (
                "| toy MoE hard dispatch | matched + Hessian-router hard top-2 worst accuracy | "
                f"{fmt(toy_moe['dispatch_robustness']['matched_router_hessian_hard_top2_worst_acc'])} |"
            ),
            (
                "| toy MoE hard dispatch | matched + Router-KD hard top-2 worst accuracy | "
                f"{fmt(toy_moe['dispatch_robustness']['matched_router_kd_hard_top2_worst_acc'])} |"
            ),
            (
                "| toy MoE hard dispatch | matched + route-KD hard top-2 worst accuracy | "
                f"{fmt(toy_moe['dispatch_robustness']['matched_router_route_kd_hard_top2_worst_acc'])} |"
            ),
            (
                "| toy MoE hard dispatch | route-KD hard top-2 delta vs router-calibrated | "
                f"{fmt(toy_moe['dispatch_robustness']['route_kd_minus_calibrated_hard_top2_worst_acc'])} |"
            ),
            (
                "| toy MoE hard dispatch | route-KD hard top-2 delta vs output-KD | "
                f"{fmt(toy_moe['dispatch_robustness']['route_kd_minus_output_kd_hard_top2_worst_acc'])} |"
            ),
            (
                "| toy MoE unified objective | hard top-2 worst accuracy | "
                f"{fmt(toy_moe['dispatch_robustness']['unified_moe_hard_top2_worst_acc'])} |"
            ),
            (
                "| toy MoE unified objective | hard top-2 delta vs route-KD | "
                f"{fmt(toy_moe['dispatch_robustness']['unified_moe_minus_route_kd_hard_top2_worst_acc'])} |"
            ),
            (
                "| toy MoE unified objective | selected capacity loss coef | "
                f"{fmt(selected_unified_capacity.get('capacity_loss_coef'), 3)} |"
            ),
            (
                "| toy MoE unified objective | selected router seed | "
                f"{selected_unified_capacity.get('router_seed', 'n/a')} |"
            ),
            (
                "| toy MoE unified objective | capacity-sweep candidates | "
                f"{toy_moe['unified_moe_capacity_sweep_rows']} |"
            ),
            (
                "| toy MoE unified objective | capacity-sweep select score | "
                f"{fmt(selected_unified_capacity.get('select_capacity_aware_score'))} |"
            ),
            (
                "| toy MoE unified objective | capacity-sweep test score | "
                f"{fmt(selected_unified_capacity.get('test_capacity_aware_score'))} |"
            ),
            (
                "| toy MoE unified output-projection objective | worst accuracy | "
                f"{fmt(toy_moe['unified_output_projection_moe_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE unified output-projection objective | hard top-2 worst accuracy | "
                f"{fmt(toy_moe['dispatch_robustness'].get('unified_output_projection_moe_hard_top2_worst_acc'))} |"
            ),
            (
                "| toy MoE unified output-projection objective | delta vs unified hard top-2 | "
                f"{fmt(toy_moe['dispatch_robustness'].get('unified_output_projection_moe_minus_unified_hard_top2_worst_acc'))} |"
            ),
            (
                "| toy MoE unified output-projection objective | capacity-sweep select score | "
                f"{fmt(selected_unified_output_projection_capacity.get('select_capacity_aware_score'))} |"
            ),
            (
                "| toy MoE unified output-projection objective | selected router seed | "
                f"{selected_unified_output_projection_capacity.get('router_seed', 'n/a')} |"
            ),
            (
                "| toy MoE bias capacity | selected capacity loss coef | "
                f"{fmt(selected_router_bias_capacity.get('capacity_loss_coef'), 3)} |"
            ),
            (
                "| toy MoE bias capacity | hard top-2 worst accuracy | "
                f"{fmt(toy_moe['dispatch_robustness'].get('unified_moe_bias_capacity_hard_top2_worst_acc'))} |"
            ),
            (
                "| toy MoE bias capacity | max top-k overflow fraction | "
                f"{fmt(toy_moe['router_capacity'].get('unified_moe_bias_capacity_max_topk_overflow_fraction'))} |"
            ),
            (
                "| toy MoE bias capacity | capacity-sweep test score | "
                f"{fmt(selected_router_bias_capacity.get('test_capacity_aware_score'))} |"
            ),
            (
                "| toy MoE capacity | max top-k overflow fraction | "
                f"{fmt(toy_moe['router_capacity']['max_topk_overflow_fraction'])} |"
            ),
            (
                "| toy MoE capacity | worst overflow method/category | "
                f"{toy_moe['router_capacity']['worst_topk_overflow_method']} / "
                f"{toy_moe['router_capacity']['worst_topk_overflow_category']} |"
            ),
            (
                "| toy MoE capacity | route-KD max top-k overflow fraction | "
                f"{fmt(toy_moe['router_capacity']['matched_router_route_kd_max_topk_overflow_fraction'])} |"
            ),
            (
                "| toy MoE capacity | route-KD minus calibrated overflow | "
                f"{fmt(toy_moe['router_capacity']['route_kd_minus_calibrated_max_topk_overflow_fraction'])} |"
            ),
            (
                "| toy MoE unified objective | max top-k overflow fraction | "
                f"{fmt(toy_moe['router_capacity']['unified_moe_max_topk_overflow_fraction'])} |"
            ),
            (
                "| toy MoE hard dispatch | soft to hard top-1 delta | "
                f"{fmt(toy_moe['dispatch_robustness']['matched_router_calibrated_soft_to_hard_top1_worst_acc_delta'])} |"
            ),
            (
                "| toy MoE hard dispatch | top-k vs soft-calibrated hard top-2 delta | "
                f"{fmt(toy_moe['dispatch_robustness']['topk_calibrated_minus_soft_calibrated_hard_top2_worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | guarded router-sweep selected KL | "
                f"{fmt(toy_moe['router_calibration_sweep_selected']['kl_coef'], 2)} |"
            ),
            (
                "| toy MoE route-aware merge | guarded router-sweep eligible / total | "
                f"{toy_moe['router_calibration_sweep_eligible_count']} / {toy_moe['router_calibration_sweep_rows']} |"
            ),
            (
                "| toy MoE route-aware merge | router-sweep selected min top-k Jaccard | "
                f"{fmt(toy_moe['router_calibration_sweep_selected']['min_test_topk_jaccard'])} |"
            ),
            (
                "| toy MoE route-aware merge | matched + router-sweep-selected worst accuracy | "
                f"{fmt(toy_moe['matched_router_sweep_selected_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | expert-weight search worst accuracy | "
                f"{fmt(toy_moe['expert_weight_search_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | expert-weight search + router-calibrated worst accuracy | "
                f"{fmt(toy_moe['expert_weight_search_router_calibrated_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE output projection | expert output-projection worst accuracy | "
                f"{fmt(toy_moe['expert_output_projection_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE output projection | output-projection + router-calibrated worst accuracy | "
                f"{fmt(toy_moe['expert_output_projection_router_calibrated_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE output projection | mean captured output residual fraction | "
                f"{fmt(toy_moe['expert_output_projection'].get('mean_captured_fraction'))} |"
            ),
            (
                "| toy MoE output projection | delta vs matched-calibrated | "
                f"{fmt(toy_moe['expert_output_projection_router_calibrated_minus_matched_calibrated_worst_acc'])} |"
            ),
            (
                "| toy MoE confidence-blended expert | router-calibrated worst accuracy | "
                f"{fmt(toy_moe['confidence_blended_router_calibrated_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE confidence-blended expert | mean projection confidence | "
                f"{fmt(toy_moe['confidence_blended_expert'].get('mean_projection_captured_fraction'))} |"
            ),
            (
                "| toy MoE unified objective | worst accuracy | "
                f"{fmt(toy_moe['unified_moe_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE unified objective | delta vs expert-search router-calibrated | "
                f"{fmt(toy_moe['unified_moe_minus_expert_search_router_calibrated_worst_acc'])} |"
            ),
            (
                "| toy MoE unified objective | delta vs route-KD | "
                f"{fmt(toy_moe['unified_moe_minus_route_kd_worst_acc'])} |"
            ),
            (
                "| toy MoE confidence-blended unified | worst accuracy | "
                f"{fmt(toy_moe['unified_confidence_blended_moe_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE confidence-blended unified | hard top-2 worst accuracy | "
                f"{fmt(toy_moe['dispatch_robustness'].get('unified_confidence_blended_moe_hard_top2_worst_acc'))} |"
            ),
            (
                "| toy MoE confidence-blended unified | max top-k overflow fraction | "
                f"{fmt(toy_moe['router_capacity'].get('unified_confidence_blended_moe_max_topk_overflow_fraction'))} |"
            ),
            (
                "| toy MoE confidence-blended unified | delta vs old unified | "
                f"{fmt(toy_moe['unified_confidence_blended_moe_minus_unified_worst_acc'])} |"
            ),
            (
                "| toy MoE unified output-projection bias-capacity | worst accuracy | "
                f"{fmt(toy_moe['unified_output_projection_bias_capacity_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE unified output-projection bias-capacity | delta vs output-projection unified | "
                f"{fmt(toy_moe['unified_output_projection_bias_capacity_minus_unified_output_projection_worst_acc'])} |"
            ),
            (
                "| toy MoE unified output-projection bias-capacity | selected capacity-aware score | "
                f"{fmt(selected_output_projection_bias_capacity.get('select_capacity_aware_score'))} |"
            ),
            (
                "| toy MoE confidence-blended bias-capacity | worst accuracy | "
                f"{fmt(toy_moe['unified_confidence_blended_bias_capacity_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE confidence-blended bias-capacity | selected capacity-aware score | "
                f"{fmt(selected_confidence_blended_bias_capacity.get('select_capacity_aware_score'))} |"
            ),
            (
                "| toy MoE route-aware merge | route-aware average worst accuracy | "
                f"{fmt(toy_moe['route_aware_expert_average']['worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | matched + router-frozen minus all-weight worst accuracy | "
                f"{fmt(toy_moe['matched_router_frozen_minus_all_weight_worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | matched router calibration gain over frozen | "
                f"{fmt(toy_moe['matched_router_calibrated_minus_frozen_worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | Hessian-router delta vs expert-matched | "
                f"{fmt(toy_moe['matched_router_hessian_minus_expert_matched_worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | Hessian-router delta vs router-calibrated | "
                f"{fmt(toy_moe['matched_router_hessian_minus_matched_calibrated_worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | Router-KD delta vs expert-matched | "
                f"{fmt(toy_moe['matched_router_kd_minus_expert_matched_worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | Router-KD delta vs router-calibrated | "
                f"{fmt(toy_moe['matched_router_kd_minus_matched_calibrated_worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | route-KD delta vs Router-KD | "
                f"{fmt(toy_moe['matched_router_route_kd_minus_router_kd_worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | route-KD delta vs router-calibrated | "
                f"{fmt(toy_moe['matched_router_route_kd_minus_matched_calibrated_worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | top-k router calibration delta vs soft calibration | "
                f"{fmt(toy_moe['matched_router_topk_calibrated_minus_matched_calibrated_worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | expert search router-calibrated delta vs matched-calibrated | "
                f"{fmt(toy_moe['expert_weight_search_router_calibrated_minus_matched_calibrated_worst_acc'])} |"
            ),
            (
                "| toy MoE route-aware merge | route-aware minus all-weight worst accuracy | "
                f"{fmt(toy_moe['route_aware_minus_all_weight_worst_acc'])} |"
            ),
            (
                "| toy MoE routing readiness | readiness status | "
                f"{toy_moe_readiness['readiness_status']} |"
            ),
            (
                "| toy MoE routing readiness | all-weight calibrate-router flags | "
                f"{toy_moe_readiness['all_weight_router_actions'].get('calibrate_router_before_average', 0)} |"
            ),
            (
                "| toy MoE method selection | recommended method | "
                f"{toy_moe_selection['recommended_method']} |"
            ),
            (
                "| toy MoE method selection | recommended hard top-2 method | "
                f"{toy_moe_selection['recommended_sparse_method']} |"
            ),
            (
                "| toy MoE method selection | recommended hard top-2 worst accuracy | "
                f"{fmt(toy_moe_selection['recommended_sparse_worst_acc'])} |"
            ),
            (
                "| toy MoE method selection | capacity-aware hard top-2 method | "
                f"{toy_moe_selection['recommended_sparse_capacity_aware_method']} |"
            ),
            (
                "| toy MoE method selection | capacity-aware top-k overflow | "
                f"{fmt(toy_moe_selection['recommended_sparse_capacity_aware_topk_overflow_fraction'])} |"
            ),
            (
                "| toy MoE method selection | hard top-2 / overflow Pareto methods | "
                f"{', '.join(toy_moe_selection['sparse_pareto_frontier_methods'])} |"
            ),
            (
                "| toy MoE method selection | all-weight decision | "
                f"{toy_moe_selection['all_weight_decision']} |"
            ),
            (
                "| toy MoE expert remap plan | remap status | "
                f"{toy_moe_remap['remap_status']} |"
            ),
            (
                "| toy MoE expert remap plan | source tensor alias rules | "
                f"{toy_moe_remap['alias_rule_count']} |"
            ),
            (
                "| toy MoE expert remap plan | layer-aware alias rules | "
                f"{toy_moe_remap['layer_aware_rule_count']} |"
            ),
            (
                "| toy MoE expert remap plan | min expert-output cosine | "
                f"{fmt(toy_moe_remap['min_output_cosine'])} |"
            ),
            (
                "| MoE layer-wise expert remap smoke | status | "
                f"{layerwise_remap_smoke['status']} |"
            ),
            (
                "| MoE layer-wise expert remap smoke | alias / layer-aware / manual-review rows | "
                f"{layerwise_remap_smoke['alias_rule_count']} / "
                f"{layerwise_remap_smoke['layer_aware_rule_count']} / "
                f"{layerwise_remap_smoke['manual_review_count']} |"
            ),
            (
                "| vLLM hosted downstream eval | status | "
                f"{vllm_eval['status']} |"
            ),
            (
                "| vLLM hosted downstream eval | queued served models | "
                f"{vllm_eval['eval_plan_model_count']} |"
            ),
            (
                "| vLLM hosted downstream eval | candidate table | "
                f"{vllm_eval['candidate_table']} |"
            ),
            (
                "| vLLM downstream eval smoke | status | "
                f"{vllm_eval_smoke['status']} |"
            ),
            (
                "| vLLM downstream eval smoke | good / bad avg primary | "
                f"{fmt(vllm_eval_smoke['mock_good_avg_primary_score'])} / {fmt(vllm_eval_smoke['mock_bad_avg_primary_score'])} |"
            ),
            (
                "| vLLM checkpoint eval plan | status | "
                f"{vllm_checkpoint_eval_plan['status']} |"
            ),
            (
                "| vLLM checkpoint eval plan | ready / missing / not-loadable | "
                f"{vllm_checkpoint_eval_plan['ready_to_host_count']} / "
                f"{vllm_checkpoint_eval_plan['missing_checkpoint_count']} / "
                f"{vllm_checkpoint_eval_plan['not_vllm_loadable_count']} |"
            ),
            (
                "| vLLM hosted eval results | completed eval dirs | "
                f"{vllm_checkpoint_eval_results['completed_count']} |"
            ),
            (
                "| vLLM hosted eval results | best eval avg / worst primary | "
                f"{vllm_checkpoint_best.get('method')} / "
                f"{fmt(vllm_checkpoint_best.get('avg_primary_score'))} / "
                f"{fmt(vllm_checkpoint_best.get('worst_primary_score'))} |"
            ),
            (
                "| vLLM source-vs-merge comparison | status | "
                f"{vllm_source_merge['status']} |"
            ),
            (
                "| vLLM source-vs-merge comparison | best source / merge avg / delta | "
                f"{vllm_source_merge['best_source_display_name']} / "
                f"{fmt(vllm_source_merge['merge_avg_primary_score'])} / "
                f"{fmt(vllm_source_merge['merge_delta_vs_best_source_avg_primary'])} |"
            ),
            (
                "| vLLM source-vs-merge comparison | merge rank / source endpoints better | "
                f"{vllm_source_merge['merge_rank_by_avg_primary']} / "
                f"{vllm_source_merge['source_models_better_than_merge_count']} |"
            ),
            (
                "| probe-guided dense average | selected alpha / beta | "
                f"{fmt(probe_guided_dense['alpha'])} / {fmt(probe_guided_dense['beta'])} |"
            ),
            (
                "| probe-guided dense average | vLLM avg / delta vs uniform / delta vs best source | "
                f"{fmt(probe_guided_dense['vllm_avg_primary_score'])} / "
                f"{fmt(probe_guided_dense['delta_vs_uniform_avg_primary'])} / "
                f"{fmt(probe_guided_dense['delta_vs_best_source_avg_primary'])} |"
            ),
            (
                "| Qwen dense guard probe | norm mean tensor cosine / sign conflict | "
                f"{fmt(qwen_dense_module_guarded['norm_anchor_mean_tensor_cosine'])} / "
                f"{fmt(qwen_dense_module_guarded['norm_anchor_sign_conflict_rate'])} |"
            ),
            (
                "| Qwen dense guard ablation | module-guarded vLLM avg / delta vs global bridge | "
                f"{fmt(qwen_dense_module_guarded['vllm_avg_primary_score'])} / "
                f"{fmt(qwen_dense_module_guarded['delta_vs_global_bridge_avg_primary'])} |"
            ),
            (
                "| Qwen dense guard ablation | norm-only vLLM avg / delta vs global bridge | "
                f"{fmt(qwen_dense_norm_guarded['vllm_avg_primary_score'])} / "
                f"{fmt(qwen_dense_norm_guarded['delta_vs_global_bridge_avg_primary'])} |"
            ),
            (
                "| Qwen dense guard ablation | selective-norm vLLM avg / delta vs global bridge | "
                f"{fmt(qwen_dense_selective_norm_guarded['vllm_avg_primary_score'])} / "
                f"{fmt(qwen_dense_selective_norm_guarded['delta_vs_global_bridge_avg_primary'])} |"
            ),
            (
                "| checkpoint materialization readiness | status | "
                f"{materialization_readiness['status']} |"
            ),
            (
                "| checkpoint materialization readiness | materialized / blocked / ready / completed | "
                f"{materialization_readiness['materialized_count']} / "
                f"{materialization_readiness['blocked_by_placeholder_count']} / "
                f"{materialization_readiness['ready_for_vllm_eval_count']} / "
                f"{materialization_readiness['completed_vllm_eval_count']} |"
            ),
            (
                "| MoE materialization pipeline | status | "
                f"{moe_pipeline_plan['status']} |"
            ),
            (
                "| MoE materialization pipeline | current blocking stage | "
                f"{moe_pipeline_plan['current_blocking_stage']} |"
            ),
            (
                "| MoE materialization pipeline | ready / waiting gates | "
                f"{moe_pipeline_plan['ready_or_complete_count']} / {moe_pipeline_plan['waiting_or_blocked_count']} |"
            ),
            (
                "| Average decision report | avoid uniform average decisions | "
                f"{average_decision['verdict_counts'].get('avoid_uniform_average', 0)} |"
            ),
            (
                "| Average decision report | coefficient-search decisions | "
                f"{average_decision['verdict_counts'].get('coefficient_search', 0)} |"
            ),
            (
                "| model averaging literature review | sources reviewed | "
                f"{literature_review['source_count']} |"
            ),
            (
                "| model averaging literature review | method / probe / MoE-stage counts | "
                f"{literature_review['method_family_count']} / {literature_review['probe_count']} / {literature_review['moe_stage_count']} |"
            ),
            (
                "| Qwen target model registry | candidate dense / MoE models | "
                f"{qwen_registry['dense_model_count']} / {qwen_registry['moe_model_count']} |"
            ),
            (
                "| Qwen target model registry | downstream or third-party candidates | "
                f"{qwen_registry['downstream_or_third_party_count']} |"
            ),
            (
                "| Qwen target model registry | recommended first scenario | "
                f"{qwen_registry['recommended_first_scenario']} |"
            ),
            (
                "| Qwen target model registry | manual resolution or selection required | "
                f"{qwen_registry['manual_resolution_or_selection_count']} |"
            ),
            (
                "| MoE routing probe smoke | routers / prompts | "
                f"{routing_probe_smoke['router_count']} / {routing_probe_smoke['prompt_count']} |"
            ),
            (
                "| MoE routing probe smoke | router / expert / overlap rows | "
                f"{routing_probe_smoke['router_summary_rows']} / {routing_probe_smoke['expert_load_rows']} / {routing_probe_smoke['route_overlap_rows']} |"
            ),
            (
                "| MoE average plan | router plan rows | "
                f"{moe_average_plan['router_plan_rows']} |"
            ),
            (
                "| MoE average plan | expert plan rows | "
                f"{moe_average_plan['expert_plan_rows']} |"
            ),
            (
                "| same-shape writer smoke | Qwen-compatible tensors checked | "
                f"{writer_smoke['floating_tensors']} |"
            ),
            (
                "| MoE tensor-rule writer smoke | status | "
                f"{moe_tensor_rule_writer_smoke['status']} |"
            ),
            (
                "| MoE tensor-rule writer smoke | checked / failed tensors | "
                f"{moe_tensor_rule_writer_smoke['checked_tensors']} / {moe_tensor_rule_writer_smoke['failed_tensors']} |"
            ),
            (
                "| MoE tensor-rule writer smoke | additive bias delta tensors / values | "
                f"{moe_tensor_rule_writer_smoke['additive_delta_tensors']} / {moe_tensor_rule_writer_smoke['additive_delta_values']} |"
            ),
            (
                "| MoE combined writer smoke | status | "
                f"{moe_combined_writer_smoke['status']} |"
            ),
            (
                "| MoE combined writer smoke | checked / failed tensors | "
                f"{moe_combined_writer_smoke['checked_tensors']} / {moe_combined_writer_smoke['failed_tensors']} |"
            ),
            (
                "| MoE combined writer smoke | alias rules / aliased tensors / additive values | "
                f"{moe_combined_writer_smoke['tensor_alias_rule_count']} / "
                f"{moe_combined_writer_smoke['code_aliased_tensors']} / "
                f"{moe_combined_writer_smoke['additive_delta_values']} |"
            ),
            (
                "| checkpoint topology | inspected MoE configs | "
                f"{len(moe_models)} |"
            ),
            (
                "| average candidate recipes | endpoint-only skips | "
                f"{recipes['status_counts'].get('skip_endpoint_only', 0)} |"
            ),
            (
                "| average candidate recipes | MoE templates awaiting routing probe | "
                f"{recipes['status_counts'].get('template_waiting_for_routing_probe', 0)} |"
            ),
            (
                "| MoE route-weight recipes | recipe status | "
                f"{route_weight_recipes['recipe_status']} |"
            ),
            (
                "| MoE route-weight recipes | expert tensor rules | "
                f"{route_weight_recipes['expert_rule_count']} |"
            ),
            (
                "| MoE router-bias plan | status | "
                f"{router_bias_plan['status']} |"
            ),
            (
                "| MoE router-bias plan | nonzero delta rows | "
                f"{router_bias_plan['nonzero_delta_rows']} |"
            ),
            (
                "| MoE confidence-blended router-bias plan | status | "
                f"{confidence_blended_router_bias_plan['status']} |"
            ),
            (
                "| MoE confidence-blended router-bias plan | nonzero delta rows | "
                f"{confidence_blended_router_bias_plan['nonzero_delta_rows']} |"
            ),
            (
                "| MoE searched expert-weight recipes | recipe status | "
                f"{toy_expert_weight_recipes['recipe_status']} |"
            ),
            (
                "| MoE searched expert-weight recipes | expert tensor rules | "
                f"{toy_expert_weight_recipes['expert_rule_count']} |"
            ),
            (
                "| MoE output-projection expert-weight recipes | recipe status | "
                f"{toy_output_projection_recipes['recipe_status']} |"
            ),
            (
                "| MoE output-projection expert-weight recipes | expert tensor rules | "
                f"{toy_output_projection_recipes['expert_rule_count']} |"
            ),
            (
                "| MoE confidence-blended expert-weight recipes | recipe status | "
                f"{toy_confidence_blended_recipes['recipe_status']} |"
            ),
            (
                "| MoE confidence-blended expert-weight recipes | expert tensor rules | "
                f"{toy_confidence_blended_recipes['expert_rule_count']} |"
            ),
            (
                "| MoE confidence-blended combined recipe | status | "
                f"{confidence_blended_combined_recipe['recipe_status']} |"
            ),
            (
                "| MoE confidence-blended combined recipe | tensor / alias / bias-delta rules | "
                f"{confidence_blended_combined_recipe['tensor_rule_count']} / "
                f"{confidence_blended_combined_recipe['alias_rule_count']} / "
                f"{confidence_blended_combined_recipe['router_bias_delta_rows']} |"
            ),
            (
                "| MoE routing readiness | readiness status | "
                f"{routing_readiness['readiness_status']} |"
            ),
            (
                "| MoE routing readiness | router / expert risk rows | "
                f"{routing_readiness['router_rows']} / {routing_readiness['expert_rows']} |"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-out", default="results/summary.json")
    parser.add_argument("--markdown-out", default="results/summary.md")
    parser.add_argument("--manifest-out", default="ARTIFACT_MANIFEST.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary()
    write_json(args.summary_out, summary)
    markdown_target = repo_path(args.markdown_out)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.write_text(build_markdown(summary), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "summary": rel(args.summary_out),
        "artifacts": collect_artifacts(),
    }
    write_json(args.manifest_out, manifest)
    print(f"Wrote {rel(args.summary_out)}")
    print(f"Wrote {rel(args.markdown_out)}")
    print(f"Wrote {rel(args.manifest_out)}")
    print(f"Artifacts indexed: {len(manifest['artifacts'])}")


if __name__ == "__main__":
    main()
