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
    return {
        "summary": summary,
        "recipe_status": summary.get("recipe_status"),
        "expert_rule_count": int(summary.get("expert_rule_count", 0)),
        "tensor_rule_count": int(summary.get("tensor_rule_count", 0)),
        "source_weights_rows": int(len(source_weights)),
        "report": rel("results/moe_route_weight_recipes/report.md"),
        "source_weights": rel("results/moe_route_weight_recipes/source_weights_by_expert.csv"),
        "tensor_rules": rel("results/moe_route_weight_recipes/tensor_rules.txt"),
        "writer_command": rel("results/moe_route_weight_recipes/writer_command.txt"),
        "prompt_pack": rel("prompts/qwen_moe_route_probe_prompts.jsonl"),
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
    router_summary = read_csv("results/toy_moe_merge/router_summary.csv")
    expert_match = read_csv("results/toy_moe_merge/expert_match.csv")
    return {
        "summary": summary,
        "best_method": find_method(methods, summary["best_method"]),
        "all_weight_average": find_method(methods, "all_weight_average"),
        "expert_matched_average": find_method(methods, "expert_matched_average"),
        "route_aware_expert_average": find_method(methods, "route_aware_expert_average"),
        "route_aware_minus_all_weight_worst_acc": float(summary["route_aware_minus_all_weight_worst_acc"]),
        "expert_match_mean_cosine": float(expert_match["output_cosine"].mean()),
        "router_rows": int(len(router_summary)),
        "report": rel("results/toy_moe_merge/report.md"),
        "method_metrics": rel("results/toy_moe_merge/method_metrics.csv"),
        "router_summary": rel("results/toy_moe_merge/router_summary.csv"),
        "expert_load": rel("results/toy_moe_merge/expert_load.csv"),
        "route_overlap": rel("results/toy_moe_merge/route_overlap.csv"),
        "expert_match": rel("results/toy_moe_merge/expert_match.csv"),
        "route_weights": rel("results/toy_moe_merge/route_weights_by_expert.csv"),
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
    recommended_method = summary.get("recommended_method")
    recommended = selection[selection["method"] == recommended_method]
    all_weight = selection[selection["method"] == "all_weight_average"]
    return {
        "summary": summary,
        "recommended_method": recommended_method,
        "recommended_decision": summary.get("recommended_decision"),
        "recommended_row": clean_row(recommended.iloc[0]) if not recommended.empty else None,
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
        "manual_review_count": int(summary.get("manual_review_count", 0)),
        "min_output_cosine": maybe_float(summary.get("min_output_cosine")),
        "mean_output_cosine": maybe_float(summary.get("mean_output_cosine")),
        "remap_rows": [clean_row(row) for _, row in remap.iterrows()],
        "report": rel("results/toy_moe_expert_remap_plan/report.md"),
        "expert_remap": rel("results/toy_moe_expert_remap_plan/expert_remap.csv"),
        "source_tensor_aliases": rel("results/toy_moe_expert_remap_plan/source_tensor_aliases.txt"),
        "writer_command": rel("results/toy_moe_expert_remap_plan/writer_command.txt"),
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
            "item": "MoE routing readiness diagnostics",
            "status": "complete",
            "evidence": "results/moe_routing_readiness/report.md turns router_summary, route_overlap, and expert_load CSVs into router collapse, drift, boundary-fragility, and expert-load risk actions.",
        },
        {
            "item": "MoE routing probe CLI",
            "status": "complete",
            "evidence": "scripts/probe_moe_routing.py captures MoE router hooks and writes router_summary.csv, expert_load.csv, optional route_overlap.csv, summary.json, and report.md for downstream readiness and route-weight recipes.",
        },
        {
            "item": "Toy MoE route-aware merge",
            "status": "complete",
            "evidence": "results/toy_moe_merge/report.md runs a small same-shape MoE averaging experiment showing expert-index mismatch and route-aware/expert-matched fixes.",
        },
        {
            "item": "Toy MoE multi-method routing readiness",
            "status": "complete",
            "evidence": "results/toy_moe_routing_readiness/report.md applies the generic readiness gate to toy MoE methods and flags all-weight routing drift separately from expert-matched/route-aware variants.",
        },
        {
            "item": "Toy MoE merge method selection",
            "status": "complete",
            "evidence": "results/toy_moe_method_selection/report.md combines method metrics and routing readiness to reject all-weight average and recommend expert-matched averaging with router guard.",
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
        "moe_average_plan": summarize_moe_average_plan(),
        "same_shape_writer_smoke": summarize_same_shape_writer_smoke(),
        "checkpoint_topology_inspect": summarize_checkpoint_topology(),
        "average_candidate_recipes": summarize_average_candidate_recipes(),
        "moe_route_weight_recipes": summarize_moe_route_weight_recipes(),
        "moe_routing_readiness": summarize_moe_routing_readiness(),
        "toy_moe_merge": summarize_toy_moe_merge(),
        "toy_moe_routing_readiness": summarize_toy_moe_routing_readiness(),
        "toy_moe_method_selection": summarize_toy_moe_method_selection(),
        "toy_moe_expert_remap_plan": summarize_toy_moe_expert_remap_plan(),
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
            "python scripts/build_model_averaging_literature_review.py",
            "PYTHONPATH=src python scripts/build_average_decision_report.py",
            "PYTHONPATH=src python scripts/build_moe_average_plan.py",
            "python scripts/write_same_shape_average_checkpoint.py --base BASE --source expert=EXPERT --dry-run --output-dir results/same_shape_writer_smoke",
            "python scripts/inspect_checkpoint_topology.py --model NAME=MODEL_PATH --output-dir results/checkpoint_topology_inspect",
            "PYTHONPATH=src python scripts/build_average_candidate_recipes.py",
            "PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code",
            "PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code --source general --source code",
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
    moe_average_plan = exp["moe_average_plan"]
    writer_smoke = exp["same_shape_writer_smoke"]
    topology = exp["checkpoint_topology_inspect"]
    moe_models = [model for model in topology["models"] if model.get("config", {}).get("is_moe_config")]
    recipes = exp["average_candidate_recipes"]
    route_weight_recipes = exp["moe_route_weight_recipes"]
    routing_readiness = exp["moe_routing_readiness"]
    toy_moe = exp["toy_moe_merge"]
    toy_moe_readiness = exp["toy_moe_routing_readiness"]
    toy_moe_selection = exp["toy_moe_method_selection"]
    toy_moe_remap = exp["toy_moe_expert_remap_plan"]
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
                "| toy MoE route-aware merge | route-aware average worst accuracy | "
                f"{fmt(toy_moe['route_aware_expert_average']['worst_acc'])} |"
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
                "| toy MoE expert remap plan | min expert-output cosine | "
                f"{fmt(toy_moe_remap['min_output_cosine'])} |"
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
