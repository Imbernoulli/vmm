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


def maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


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
    excluded_files = {
        "scripts/fp_materialize_merge.py",  # scratch materialization helper outside the maintained writer path
        "scripts/fp_make_figures.py",  # stale draft figures with old first-principles numbers
        "scripts/fp_collect_matrix.py",  # scratch matrix collector awaiting vLLM-backed replacement
        "results/fp_curvature_law/run.log",
        "results/fp_dense_lambda/run.log",
        "results/fp_merge_compare_dense/run.log",
        "results/fp_moe_barrier/run.log",
        "results/fp_moe_complementary/run.log",
        "results/fp_moe_forgetting_base_coder/moe_barrier.png",  # title reflects the generic script, not the base->coder run
        "results/fp_moe_forgetting_base_coder/run.log",
        "results/fp_moe_mechanism/run.log",
        "results/fp_moe_mechanism_identity/run.log",
    }
    excluded_prefixes = {
        "results/fp_figures/",
        "results/fp_gen_eval_moe/",
        "results/fp_moe_real_probe/olmoe/",
        "results/qwen3_moe_router_calibration_selection_manifest_mismatch_negative_smoke/",
    }
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
        if relative in excluded_files or any(relative.startswith(prefix) for prefix in excluded_prefixes):
            continue
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


def summarize_fp_curvature_law() -> dict[str, Any]:
    root = repo_path("results/fp_curvature_law")
    summary = read_json(root / "summary.json")
    law = summary.get("curvature_law", {})
    merges = summary.get("merges", {})
    best_merge_name = min(merges, key=lambda k: merges[k]["worst"]) if merges else None
    return {
        "summary": summary,
        "geometry": summary.get("geometry", {}),
        "curvature_law": law,
        "best_merge_name": best_merge_name,
        "best_merge": merges.get(best_merge_name, {}) if best_merge_name else {},
        "uniform": merges.get("uniform", {}),
        "fisher": merges.get("fisher", {}),
        "top_interference_tensor": (summary.get("top_interference_tensors") or [{}])[0],
        "report": rel(root / "report.md"),
        "figure": rel(root / "curvature_law.png"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_fp_merge_compare() -> dict[str, Any]:
    root = repo_path("results/fp_merge_compare_dense")
    summary = read_json(root / "summary.json")
    methods = read_csv(root / "method_metrics.csv")
    trace = read_csv(root / "selection_trace.csv")
    results = summary.get("results", {})
    unified = results.get("unified", {})
    linear = results.get("linear", {})
    task_arith = results.get("task_arith_0.5", {})
    ties = results.get("ties_0.5", {})
    best_endpoint = maybe_float(summary.get("best_endpoint_worst"))
    unified_worst = maybe_float(unified.get("worst"))
    return {
        "summary": summary,
        "models": summary.get("models", {}),
        "grid_profile": summary.get("grid_profile"),
        "candidate_count": int(summary.get("candidate_count", len(trace))),
        "selected_config": unified.get("config", {}),
        "heldout_best_worst": maybe_float(unified.get("ho_worst")),
        "unified": unified,
        "linear": linear,
        "task_arith_0_5": task_arith,
        "ties_0_5": ties,
        "best_endpoint_worst": best_endpoint,
        "unified_worst_minus_best_endpoint": None
        if unified_worst is None or best_endpoint is None
        else unified_worst - best_endpoint,
        "linear_worst_delta_vs_unified": None
        if maybe_float(linear.get("worst")) is None or unified_worst is None
        else maybe_float(linear.get("worst")) - unified_worst,
        "ties_worst_delta_vs_unified": None
        if maybe_float(ties.get("worst")) is None or unified_worst is None
        else maybe_float(ties.get("worst")) - unified_worst,
        "method_rows": [clean_row(row) for _, row in methods.iterrows()],
        "top_selection_rows": [clean_row(row) for _, row in trace.head(8).iterrows()],
        "report": rel(root / "report.md"),
        "summary_path": rel(root / "summary.json"),
        "method_metrics": rel(root / "method_metrics.csv"),
        "selection_trace": rel(root / "selection_trace.csv"),
    }


def summarize_fp_gen_eval_dense() -> dict[str, Any]:
    root = repo_path("results/fp_gen_eval_dense")
    summary = read_json(root / "summary.json")
    methods = read_csv(root / "method_metrics.csv")
    predictions = read_csv(root / "predictions.csv")
    results = summary.get("results", {})
    linear = results.get("linear", {})
    unified = results.get("unified", {})
    coder = results.get("coder", {})
    return {
        "summary": summary,
        "models": summary.get("models", {}),
        "task_counts": summary.get("task_counts", {}),
        "best_method": summary.get("best_method"),
        "linear": linear,
        "unified": unified,
        "coder": coder,
        "unified_avg_delta_vs_linear": maybe_float(unified.get("avg_accuracy"))
        - maybe_float(linear.get("avg_accuracy"))
        if maybe_float(unified.get("avg_accuracy")) is not None and maybe_float(linear.get("avg_accuracy")) is not None
        else None,
        "unified_worst_delta_vs_linear": maybe_float(unified.get("worst_accuracy"))
        - maybe_float(linear.get("worst_accuracy"))
        if maybe_float(unified.get("worst_accuracy")) is not None and maybe_float(linear.get("worst_accuracy")) is not None
        else None,
        "coder_worst_delta_vs_unified": maybe_float(coder.get("worst_accuracy"))
        - maybe_float(unified.get("worst_accuracy"))
        if maybe_float(coder.get("worst_accuracy")) is not None and maybe_float(unified.get("worst_accuracy")) is not None
        else None,
        "method_rows": [clean_row(row) for _, row in methods.iterrows()],
        "prediction_rows": [clean_row(row) for _, row in predictions.iterrows()],
        "report": rel(root / "report.md"),
        "summary_path": rel(root / "summary.json"),
        "method_metrics": rel(root / "method_metrics.csv"),
        "predictions": rel(root / "predictions.csv"),
    }


def summarize_fp_downstream_matrix() -> dict[str, Any]:
    root = repo_path("results/fp_downstream_matrix")
    summary = read_json(root / "summary.json")
    matrix = read_csv(root / "matrix.csv")
    if not summary:
        return {
            "status": "missing",
            "model_count": 0,
            "matrix_rows": [],
            "report": rel(root / "report.md"),
            "summary_path": rel(root / "summary.json"),
            "matrix": rel(root / "matrix.csv"),
            "figure": rel(root / "downstream_matrix.png"),
        }
    return {
        "summary": summary,
        "status": summary.get("status"),
        "role": summary.get("role"),
        "model_count": maybe_int(summary.get("model_count")),
        "best_avg_model": summary.get("best_avg_model"),
        "best_avg": maybe_float(summary.get("best_avg")),
        "best_parent_model": summary.get("best_parent_model"),
        "best_parent_avg": maybe_float(summary.get("best_parent_avg")),
        "pair_merge_avg": maybe_float(summary.get("pair_merge_avg")),
        "pair_routercal_avg": maybe_float(summary.get("pair_routercal_avg")),
        "pair_routercal_avg_gain": maybe_float(summary.get("pair_routercal_avg_gain")),
        "pair_routercal_humaneval_gain": maybe_float(summary.get("pair_routercal_humaneval_gain")),
        "pair_routercal_gsm8k_gain": maybe_float(summary.get("pair_routercal_gsm8k_gain")),
        "pair_routercal_mmlu_gain": maybe_float(summary.get("pair_routercal_mmlu_gain")),
        "pair_routercal_gap_to_best_parent_avg": maybe_float(
            summary.get("pair_routercal_gap_to_best_parent_avg")
        ),
        "triple_merge_avg": maybe_float(summary.get("triple_merge_avg")),
        "triple_minus_pair_avg": maybe_float(summary.get("triple_minus_pair_avg")),
        "interpretation": summary.get("interpretation"),
        "matrix_rows": [clean_row(row) for _, row in matrix.iterrows()],
        "report": rel(root / "report.md"),
        "summary_path": rel(root / "summary.json"),
        "matrix": rel(root / "matrix.csv"),
        "figure": rel(root / "downstream_matrix.png"),
    }


def summarize_fp_downstream_attribution() -> dict[str, Any]:
    root = repo_path("results/fp_downstream_attribution")
    summary = read_json(root / "summary.json")
    transitions = read_csv(root / "transition_effects.csv")
    if not summary:
        return {
            "status": "missing",
            "score_count": 0,
            "transition_rows": [],
            "report": rel(root / "report.md"),
            "summary_path": rel(root / "summary.json"),
            "transition_effects": rel(root / "transition_effects.csv"),
        }
    return {
        "summary": summary,
        "status": summary.get("status"),
        "role": summary.get("role"),
        "score_count": maybe_int(summary.get("score_count")),
        "best_avg_model": summary.get("best_avg_model"),
        "best_avg": maybe_float(summary.get("best_avg")),
        "avg_naive_drop_vs_pair_frontier": maybe_float(
            summary.get("avg_naive_drop_vs_pair_frontier")
        ),
        "avg_routercal_gain_vs_naive": maybe_float(summary.get("avg_routercal_gain_vs_naive")),
        "avg_routercal_recovery_fraction": maybe_float(
            summary.get("avg_routercal_recovery_fraction")
        ),
        "avg_routercal_gap_vs_pair_frontier": maybe_float(
            summary.get("avg_routercal_gap_vs_pair_frontier")
        ),
        "humaneval_naive_drop_vs_pair_frontier": maybe_float(
            summary.get("humaneval_naive_drop_vs_pair_frontier")
        ),
        "humaneval_routercal_gain_vs_naive": maybe_float(
            summary.get("humaneval_routercal_gain_vs_naive")
        ),
        "humaneval_routercal_recovery_fraction": maybe_float(
            summary.get("humaneval_routercal_recovery_fraction")
        ),
        "mean_recovery_fraction_over_dropped_scores": maybe_float(
            summary.get("mean_recovery_fraction_over_dropped_scores")
        ),
        "routercal_beats_pair_frontier_count": maybe_int(
            summary.get("routercal_beats_pair_frontier_count")
        ),
        "routercal_beats_all_source_frontier_count": maybe_int(
            summary.get("routercal_beats_all_source_frontier_count")
        ),
        "interpretation": summary.get("interpretation"),
        "transition_rows": [clean_row(row) for _, row in transitions.iterrows()],
        "report": rel(root / "report.md"),
        "summary_path": rel(root / "summary.json"),
        "transition_effects": rel(root / "transition_effects.csv"),
    }


def summarize_fp_downstream_confidence_audit() -> dict[str, Any]:
    root = repo_path("results/fp_downstream_confidence_audit")
    summary = read_json(root / "summary.json")
    intervals = read_csv(root / "model_task_intervals.csv")
    comparisons = read_csv(root / "comparison_intervals.csv")
    if not summary:
        return {
            "status": "missing",
            "task_count": 0,
            "comparison_rows": [],
            "report": rel(root / "report.md"),
            "summary_path": rel(root / "summary.json"),
            "model_task_intervals": rel(root / "model_task_intervals.csv"),
            "comparison_intervals": rel(root / "comparison_intervals.csv"),
        }
    return {
        "summary": summary,
        "status": summary.get("status"),
        "role": summary.get("role"),
        "task_sample_sizes": summary.get("task_sample_sizes", {}),
        "task_count": maybe_int(summary.get("task_count")),
        "model_count": maybe_int(summary.get("model_count")),
        "routercal_positive_task_count_vs_naive": maybe_int(
            summary.get("routercal_positive_task_count_vs_naive")
        ),
        "routercal_confident_positive_task_count_vs_naive": maybe_int(
            summary.get("routercal_confident_positive_task_count_vs_naive")
        ),
        "routercal_confident_beats_pair_frontier_task_count": maybe_int(
            summary.get("routercal_confident_beats_pair_frontier_task_count")
        ),
        "routercal_confident_loses_pair_frontier_task_count": maybe_int(
            summary.get("routercal_confident_loses_pair_frontier_task_count")
        ),
        "routercal_avg_diff_vs_naive": maybe_float(summary.get("routercal_avg_diff_vs_naive")),
        "routercal_avg_diff_lower_vs_naive": maybe_float(
            summary.get("routercal_avg_diff_lower_vs_naive")
        ),
        "routercal_avg_diff_upper_vs_naive": maybe_float(
            summary.get("routercal_avg_diff_upper_vs_naive")
        ),
        "routercal_avg_gap_vs_pair_frontier": maybe_float(
            summary.get("routercal_avg_gap_vs_pair_frontier")
        ),
        "routercal_avg_gap_lower_vs_pair_frontier": maybe_float(
            summary.get("routercal_avg_gap_lower_vs_pair_frontier")
        ),
        "routercal_avg_gap_upper_vs_pair_frontier": maybe_float(
            summary.get("routercal_avg_gap_upper_vs_pair_frontier")
        ),
        "interpretation": summary.get("interpretation"),
        "interval_rows": [clean_row(row) for _, row in intervals.iterrows()],
        "comparison_rows": [clean_row(row) for _, row in comparisons.iterrows()],
        "report": rel(root / "report.md"),
        "summary_path": rel(root / "summary.json"),
        "model_task_intervals": rel(root / "model_task_intervals.csv"),
        "comparison_intervals": rel(root / "comparison_intervals.csv"),
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
        "tensor_delta_safetensors_tensors": int(manifest.get("tensor_delta_safetensors_tensors", 0)),
        "tensor_delta_safetensors_entries": int(manifest.get("tensor_delta_safetensors_entries", 0)),
        "tensor_delta_safetensors_values": int(manifest.get("tensor_delta_safetensors_values", 0)),
        "tensor_delta_safetensors_summary": manifest.get("tensor_delta_safetensors_summary", {}),
        "report": rel("results/moe_tensor_rule_writer_smoke/report.md"),
        "tensor_checks": rel("results/moe_tensor_rule_writer_smoke/tensor_checks.csv"),
        "manifest_path": rel("results/moe_tensor_rule_writer_smoke/merge_manifest.json"),
    }


def summarize_moe_router_delta_calibration_smoke() -> dict[str, Any]:
    summary = read_json("results/moe_router_delta_calibration_smoke/summary.json")
    metrics = read_csv("results/moe_router_delta_calibration_smoke/router_delta_summary.csv")
    final = metrics[metrics["stage"] == "final"]
    return {
        "summary": summary,
        "status": summary.get("status"),
        "router_count": int(summary.get("router_count", final["tensor"].nunique())),
        "delta_tensor_count": int(summary.get("delta_tensor_count", 0)),
        "mean_initial_route_kl": float(summary.get("mean_initial_route_kl", 0.0)),
        "mean_final_route_kl": float(summary.get("mean_final_route_kl", 0.0)),
        "mean_initial_top1_agreement": float(summary.get("mean_initial_top1_agreement", 0.0)),
        "mean_final_top1_agreement": float(summary.get("mean_final_top1_agreement", 0.0)),
        "max_final_relative_delta_norm": float(summary.get("max_final_relative_delta_norm", 0.0)),
        "selection_policy": summary.get("selection_policy"),
        "selection_split": summary.get("selection_split"),
        "mean_train_samples": float(summary.get("mean_train_samples", 0.0)),
        "mean_selection_samples": float(summary.get("mean_selection_samples", 0.0)),
        "mean_validation_fraction": float(summary.get("mean_validation_fraction", 0.0)),
        "mean_train_group_count": float(summary.get("mean_train_group_count", 0.0)),
        "mean_validation_group_count": float(summary.get("mean_validation_group_count", 0.0)),
        "mean_selected_epoch": float(summary.get("mean_selected_epoch", 0.0)),
        "mean_selection_score": float(summary.get("mean_selection_score", 0.0)),
        "mean_train_final_route_kl": float(summary.get("mean_train_final_route_kl", 0.0)),
        "mean_train_final_top1_agreement": float(summary.get("mean_train_final_top1_agreement", 0.0)),
        "max_route_kl_generalization_gap": float(summary.get("max_route_kl_generalization_gap", 0.0)),
        "max_top1_generalization_drop": float(summary.get("max_top1_generalization_drop", 0.0)),
        "max_initial_top1_capacity_overflow_fraction": float(
            summary.get("max_initial_top1_capacity_overflow_fraction", 0.0)
        ),
        "max_final_top1_capacity_overflow_fraction": float(
            summary.get("max_final_top1_capacity_overflow_fraction", 0.0)
        ),
        "max_router_top1_capacity_overflow_increase": float(
            summary.get("max_router_top1_capacity_overflow_increase", 0.0)
        ),
        "max_initial_topk_capacity_overflow_fraction": float(
            summary.get("max_initial_topk_capacity_overflow_fraction", 0.0)
        ),
        "max_final_topk_capacity_overflow_fraction": float(
            summary.get("max_final_topk_capacity_overflow_fraction", 0.0)
        ),
        "max_router_topk_capacity_overflow_increase": float(
            summary.get("max_router_topk_capacity_overflow_increase", 0.0)
        ),
        "max_initial_top1_load_fraction": float(summary.get("max_initial_top1_load_fraction", 0.0)),
        "max_final_top1_load_fraction": float(summary.get("max_final_top1_load_fraction", 0.0)),
        "max_initial_topk_load_fraction": float(summary.get("max_initial_topk_load_fraction", 0.0)),
        "max_final_topk_load_fraction": float(summary.get("max_final_topk_load_fraction", 0.0)),
        "max_relative_norm": summary.get("max_relative_norm"),
        "router_cap_mode": summary.get("router_cap_mode"),
        "router_cap_table": summary.get("router_cap_table"),
        "min_router_relative_norm_cap": maybe_float(summary.get("min_router_relative_norm_cap")),
        "mean_router_relative_norm_cap": maybe_float(summary.get("mean_router_relative_norm_cap")),
        "max_router_relative_norm_cap": maybe_float(summary.get("max_router_relative_norm_cap")),
        "max_cap_utilization": maybe_float(summary.get("max_cap_utilization")),
        "router_delta_safetensors": summary.get("outputs", {}).get(
            "router_delta_safetensors",
            rel("results/moe_router_delta_calibration_smoke/router_delta.safetensors"),
        ),
        "report": rel("results/moe_router_delta_calibration_smoke/report.md"),
        "router_delta_summary": rel("results/moe_router_delta_calibration_smoke/router_delta_summary.csv"),
        "router_cap_table_path": rel("results/moe_router_delta_calibration_smoke/router_cap_table.csv"),
        "training_trace": rel("results/moe_router_delta_calibration_smoke/training_trace.csv"),
    }


def summarize_moe_router_calibration_cache_smoke() -> dict[str, Any]:
    root = repo_path("results/moe_router_calibration_cache_smoke")
    summary = read_json(root / "summary.json")
    cache = read_csv(root / "cache_summary.csv")
    calibration = read_json(root / "delta_calibration/summary.json")
    return {
        "summary": summary,
        "calibration_summary": calibration,
        "status": summary.get("status"),
        "cache_ready_router_count": int(summary.get("cache_ready_router_count", 0)),
        "common_router_count": int(summary.get("common_router_count", len(cache))),
        "total_cache_rows": int(summary.get("total_cache_rows", 0)),
        "mean_student_teacher_route_kl": float(summary.get("mean_student_teacher_route_kl", 0.0)),
        "mean_student_teacher_top1_agreement": float(summary.get("mean_student_teacher_top1_agreement", 0.0)),
        "calibration_status": summary.get("calibration_status"),
        "calibration_mean_initial_route_kl": float(calibration.get("mean_initial_route_kl", 0.0)),
        "calibration_mean_final_route_kl": float(calibration.get("mean_final_route_kl", 0.0)),
        "calibration_mean_initial_top1_agreement": float(calibration.get("mean_initial_top1_agreement", 0.0)),
        "calibration_mean_final_top1_agreement": float(calibration.get("mean_final_top1_agreement", 0.0)),
        "calibration_selection_policy": calibration.get("selection_policy"),
        "calibration_selection_split": calibration.get("selection_split"),
        "calibration_mean_selected_epoch": float(calibration.get("mean_selected_epoch", 0.0)),
        "calibration_mean_selection_score": float(calibration.get("mean_selection_score", 0.0)),
        "calibration_mean_train_samples": float(calibration.get("mean_train_samples", 0.0)),
        "calibration_mean_selection_samples": float(calibration.get("mean_selection_samples", 0.0)),
        "calibration_mean_validation_fraction": float(calibration.get("mean_validation_fraction", 0.0)),
        "calibration_mean_train_group_count": float(calibration.get("mean_train_group_count", 0.0)),
        "calibration_mean_validation_group_count": float(calibration.get("mean_validation_group_count", 0.0)),
        "materialization_status": summary.get("materialization_status"),
        "materialization_checked_tensors": int(summary.get("materialization_checked_tensors") or 0),
        "materialization_failed_tensors": int(summary.get("materialization_failed_tensors") or 0),
        "router_delta_safetensors": calibration.get("outputs", {}).get(
            "router_delta_safetensors",
            rel(root / "delta_calibration/router_delta.safetensors"),
        ),
        "cache": rel(root / "router_calibration_cache.pt"),
        "cache_summary": rel(root / "cache_summary.csv"),
        "materialization_checks": rel(root / "materialization_checks.csv"),
        "materialized_checkpoint_manifest": rel(root / "checkpoint_with_calibrated_router/merge_manifest.json"),
        "report": rel(root / "report.md"),
        "delta_calibration_report": rel(root / "delta_calibration/report.md"),
    }


def summarize_dense_sparse_method_writer_smoke() -> dict[str, Any]:
    summary = read_json("results/dense_sparse_method_writer_smoke/summary.json")
    checks = read_csv("results/dense_sparse_method_writer_smoke/tensor_checks.csv")
    manifest = read_json("results/dense_sparse_method_writer_smoke/merge_manifest.json")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "checked_tensors": int(summary.get("checked_tensors", len(checks))),
        "failed_tensors": int(summary.get("failed_tensors", (~checks["passed"]).sum())),
        "method_counts": summary.get("method_counts", {}),
        "tensor_method_rule_count": len(manifest.get("tensor_method_rules", [])),
        "report": rel("results/dense_sparse_method_writer_smoke/report.md"),
        "tensor_checks": rel("results/dense_sparse_method_writer_smoke/tensor_checks.csv"),
        "manifest_path": rel("results/dense_sparse_method_writer_smoke/merge_manifest.json"),
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


def summarize_moe_packed_expert_writer_smoke() -> dict[str, Any]:
    summary = read_json("results/moe_packed_expert_writer_smoke/summary.json")
    checks = read_csv("results/moe_packed_expert_writer_smoke/tensor_checks.csv")
    manifest = read_json("results/moe_packed_expert_writer_smoke/merge_manifest.json")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "checked_tensors": int(summary.get("checked_tensors", len(checks))),
        "failed_tensors": int(summary.get("failed_tensors", (~checks["passed"]).sum())),
        "packed_expert_rule_tensors": int(summary.get("packed_expert_rule_tensors", 0)),
        "packed_expert_rule_slices": int(summary.get("packed_expert_rule_slices", 0)),
        "packed_expert_rule_values": int(summary.get("packed_expert_rule_values", 0)),
        "packed_expert_slice_rule_summary": manifest.get("packed_expert_slice_rule_summary", {}),
        "rule_counts": summary.get("rule_counts", {}),
        "report": rel("results/moe_packed_expert_writer_smoke/report.md"),
        "tensor_checks": rel("results/moe_packed_expert_writer_smoke/tensor_checks.csv"),
        "manifest_path": rel("results/moe_packed_expert_writer_smoke/merge_manifest.json"),
    }


def summarize_checkpoint_topology() -> dict[str, Any]:
    summary = read_json("results/checkpoint_topology_inspect/summary.json")
    models = summary.get("models", [])
    primary = models[0] if models else {}
    primary_config = primary.get("config", {})
    primary_headers = primary.get("headers", {})
    groups = primary_headers.get("groups", {})
    return {
        "summary": summary,
        "models": models,
        "comparisons": summary.get("comparisons", []),
        "primary_model": primary.get("name"),
        "primary_model_type": primary_config.get("model_type"),
        "primary_weights_available": bool(primary_headers.get("weights_available", False)),
        "primary_num_experts": maybe_int(primary_config.get("num_experts")),
        "primary_num_experts_per_tok": maybe_int(primary_config.get("num_experts_per_tok")),
        "primary_num_experts_with_weights": maybe_int(primary_headers.get("num_experts_with_weights")),
        "primary_packed_expert_tensor_count": maybe_int(primary_headers.get("packed_expert_tensor_count")),
        "primary_routed_expert_bytes": maybe_int((groups.get("routed_expert") or {}).get("bytes")),
        "primary_router_tensor_count": maybe_int((groups.get("router") or {}).get("tensors")),
        "primary_total_bytes": maybe_int(primary_headers.get("total_bytes")),
        "report": rel("results/checkpoint_topology_inspect/report.md"),
        "compatibility": rel("results/checkpoint_topology_inspect/compatibility.csv"),
    }


def summarize_moe_unified_preflight() -> dict[str, Any]:
    root = repo_path("results/moe_unified_preflight_qwen3_30b")
    summary = read_json(root / "summary.json")
    gates = read_csv(root / "unified_gate_table.csv")
    config = read_csv(root / "config_contract.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "same_shape_contract_pass": bool(summary.get("same_shape_contract_pass", False)),
        "expert_identity_status": (summary.get("expert_identity_gate") or {}).get("status"),
        "expert_identity_fraction": maybe_float(
            (summary.get("expert_identity_gate") or {}).get("frac_layers_identity_optimal")
        ),
        "cuda_available": bool(summary.get("cuda_available", False)),
        "router_tensor_rows": int(summary.get("router_tensor_rows", 0)),
        "routed_expert_tensor_rows": int(summary.get("routed_expert_tensor_rows", 0)),
        "gate_rows": [clean_row(row) for _, row in gates.iterrows()],
        "config_rows": [clean_row(row) for _, row in config.iterrows()],
        "report": rel(root / "report.md"),
        "summary_path": rel(root / "summary.json"),
        "gate_table": rel(root / "unified_gate_table.csv"),
        "router_contract": rel(root / "router_contract.csv"),
        "expert_contract": rel(root / "expert_contract.csv"),
        "routing_probe_command": rel(root / "routing_probe_command.txt"),
    }


def summarize_qwen3_moe_routing_probe() -> dict[str, Any]:
    root = repo_path("results/moe_routing_probe/qwen3_30b_instruct_vs_coder")
    summary = read_json(root / "summary.json")
    route_overlap = read_csv(root / "route_overlap.csv")
    router_summary = read_csv(root / "router_summary.csv")
    return {
        "summary": summary,
        "prompt_count": int(summary.get("prompt_count", 0)),
        "category_count": int(summary.get("category_count", 0)),
        "router_count": int(summary.get("router_count", 0)),
        "router_summary_rows": int(len(router_summary)),
        "route_overlap_rows": int(len(route_overlap)),
        "mean_top1_agreement": float(route_overlap["top1_agreement"].mean()),
        "min_top1_agreement": float(route_overlap["top1_agreement"].min()),
        "mean_topk_jaccard": float(route_overlap["topk_jaccard"].mean()),
        "min_topk_jaccard": float(route_overlap["topk_jaccard"].min()),
        "report": rel(root / "report.md"),
        "summary_path": rel(root / "summary.json"),
        "router_summary": rel(root / "router_summary.csv"),
        "compare_router_summary": rel(root / "compare_router_summary.csv"),
        "route_overlap": rel(root / "route_overlap.csv"),
    }


def summarize_qwen3_moe_routing_readiness() -> dict[str, Any]:
    root = repo_path("results/moe_routing_readiness/qwen3_30b_instruct_vs_coder")
    summary = read_json(root / "summary.json")
    router_readiness = read_csv(root / "router_readiness.csv")
    return {
        "summary": summary,
        "readiness_status": summary.get("readiness_status"),
        "router_rows": int(summary.get("router_rows", len(router_readiness))),
        "expert_rows": int(summary.get("expert_rows", 0)),
        "specialization_rows": int(summary.get("specialization_rows", 0)),
        "router_action_counts": summary.get("router_action_counts", {}),
        "expert_action_counts": summary.get("expert_action_counts", {}),
        "specialization_action_counts": summary.get("specialization_action_counts", {}),
        "mean_top1_agreement": float(router_readiness["top1_agreement"].mean()),
        "min_top1_agreement": float(router_readiness["top1_agreement"].min()),
        "mean_topk_jaccard": float(router_readiness["topk_jaccard"].mean()),
        "min_topk_jaccard": float(router_readiness["topk_jaccard"].min()),
        "report": rel(root / "report.md"),
        "summary_path": rel(root / "summary.json"),
        "router_readiness": rel(root / "router_readiness.csv"),
        "category_specialization": rel(root / "category_specialization.csv"),
    }


def summarize_qwen3_moe_route_guarded_candidate() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_unified_route_guarded_candidate")
    summary = read_json(root / "summary.json")
    source_weights = read_csv(root / "source_weights_by_expert.csv")
    writer_command = (root / "writer_command.txt").read_text(encoding="utf-8").strip()
    loaded_inputs = [
        row for row in summary.get("input_summaries", []) if row.get("expert_load_file") in {"expert_load.csv", "compare_expert_load.csv"}
    ]
    manifest_path = repo_path("results/checkpoints/qwen3_moe_unified_route_guarded_candidate/merge_manifest.json")
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    rule_counts = manifest.get("rule_counts", {})
    expert_tensor_rule_hits = sum(
        int(count) for key, count in rule_counts.items() if str(key).startswith("tensor_rule:.*layers\\.")
    )
    manifest_exists = bool(manifest)
    checkpoint_materialized = manifest_exists and not bool(manifest.get("dry_run", True))
    return {
        "summary": summary,
        "recipe_status": summary.get("recipe_status"),
        "recipe_kind": summary.get("recipe_kind"),
        "expert_rule_count": int(summary.get("expert_rule_count", len(source_weights))),
        "tensor_rule_count": int(summary.get("tensor_rule_count", 0)),
        "source_weight_rows": int(len(source_weights)),
        "source_route_conditioned_used_rows": int(sum(int(row.get("used_rows", 0)) for row in loaded_inputs)),
        "source_route_conditioned_skipped_rows": int(sum(int(row.get("skipped_model_rows", 0)) for row in loaded_inputs)),
        "freeze_router": bool(summary.get("freeze_router")),
        "shared_attention_coder_delta": maybe_float((summary.get("shared_attention_weights") or {}).get("coder")),
        "packed_expert_rules_enabled": bool(summary.get("packed_expert_rules_enabled")),
        "writer_manifest_exists": manifest_exists,
        "writer_checkpoint_materialized": checkpoint_materialized,
        "writer_manifest_floating_tensors": int(manifest.get("floating_tensors", 0)),
        "writer_manifest_frozen_tensors": int(manifest.get("frozen_tensors", 0)),
        "writer_dry_run_passed": manifest_exists,
        "writer_dry_run_floating_tensors": int(manifest.get("floating_tensors", 0)),
        "writer_dry_run_frozen_tensors": int(manifest.get("frozen_tensors", 0)),
        "writer_dry_run_freeze_router_hits": int(rule_counts.get("freeze_router", 0)),
        "writer_dry_run_shared_attention_hits": int(rule_counts.get("tensor_rule:.*self_attn.*", 0)),
        "writer_dry_run_expert_tensor_rule_hits": expert_tensor_rule_hits,
        "writer_command_has_real_paths": "MOE_BASE_OR_ANCHOR_PATH" not in writer_command and "MODEL_PATH" not in writer_command,
        "category_sources": summary.get("category_sources", {}),
        "model_sources": summary.get("model_sources", []),
        "report": rel(root / "report.md"),
        "source_weights": rel(root / "source_weights_by_expert.csv"),
        "tensor_rules": rel(root / "tensor_rules.txt"),
        "writer_command": rel(root / "writer_command.txt"),
        "manifest": rel(manifest_path) if manifest_path.exists() else None,
    }


def summarize_materialized_delta_audit_dir(path: str) -> dict[str, Any]:
    root = repo_path(path)
    summary = read_json(root / "summary.json")
    groups = read_csv(root / "group_delta_summary.csv")
    layers = read_csv(root / "layer_delta_summary.csv")
    tensors = read_csv(root / "tensor_delta_audit.csv")
    top_layers = layers.sort_values("delta_norm", ascending=False).head(8)
    routed = tensors[tensors["group"] == "routed_expert_ffn"]
    return {
        "summary": summary,
        "status": summary.get("status"),
        "tensor_count": int(summary.get("tensor_count", 0)),
        "changed_tensors": int(summary.get("changed_tensors", 0)),
        "changed_numel_fraction": float(summary.get("changed_numel", 0)) / max(1, int(summary.get("total_numel", 1))),
        "relative_delta_norm": maybe_float(summary.get("relative_delta_norm")),
        "max_abs_delta": maybe_float(summary.get("max_abs_delta")),
        "router_tensors": int(summary.get("router_tensors", 0)),
        "router_changed_tensors": int(summary.get("router_changed_tensors", 0)),
        "max_routed_tensor_relative_delta": None
        if routed.empty
        else float(routed["relative_delta_norm"].max()),
        "routed_tensors_relative_delta_gt_1": int((routed["relative_delta_norm"] > 1.0).sum()) if not routed.empty else 0,
        "routed_tensors_relative_delta_gt_075": int((routed["relative_delta_norm"] > 0.75).sum()) if not routed.empty else 0,
        "routed_tensors_relative_delta_gt_065": int((routed["relative_delta_norm"] > 0.65).sum()) if not routed.empty else 0,
        "routed_tensors_relative_delta_gt_06505": int((routed["relative_delta_norm"] > 0.6505).sum()) if not routed.empty else 0,
        "group_rows": [clean_row(row) for _, row in groups.iterrows()],
        "top_layer_rows": [clean_row(row) for _, row in top_layers.iterrows()],
        "top_changed_tensors": summary.get("top_changed_tensors", []),
        "report": rel(root / "report.md"),
        "tensor_delta_audit": rel(root / "tensor_delta_audit.csv"),
        "group_delta_summary": rel(root / "group_delta_summary.csv"),
        "layer_delta_summary": rel(root / "layer_delta_summary.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_materialized_delta_audit() -> dict[str, Any]:
    return summarize_materialized_delta_audit_dir("results/qwen3_moe_materialized_delta_audit")


def summarize_qwen3_moe_audit_gated_delta_audit() -> dict[str, Any]:
    return summarize_materialized_delta_audit_dir("results/qwen3_moe_audit_gated_delta_audit")


def summarize_qwen3_moe_trust_region_delta_audit() -> dict[str, Any]:
    return summarize_materialized_delta_audit_dir("results/qwen3_moe_trust_region_delta_audit")


def summarize_qwen3_moe_expert_only_delta_audit() -> dict[str, Any]:
    return summarize_materialized_delta_audit_dir("results/qwen3_moe_expert_only_delta_audit")


def summarize_qwen3_moe_tail_trimmed_delta_audit() -> dict[str, Any]:
    return summarize_materialized_delta_audit_dir("results/qwen3_moe_tail_trimmed_delta_audit")


def summarize_qwen3_moe_searched_no_gt065_delta_audit() -> dict[str, Any]:
    return summarize_materialized_delta_audit_dir("results/qwen3_moe_searched_no_gt065_delta_audit")


def summarize_qwen3_moe_layer_chunk_delta_audit() -> dict[str, Any]:
    return summarize_materialized_delta_audit_dir("results/qwen3_moe_layer_chunk_delta_audit")


def summarize_qwen3_moe_tail_trimmed_expert_only_candidate() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_tail_trimmed_expert_only_candidate")
    summary = read_json(root / "summary.json")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "target_cap": maybe_float(summary.get("target_cap")),
        "expert_group_count": int(summary.get("expert_group_count", 0)),
        "scaled_expert_group_count": int(summary.get("scaled_expert_group_count", 0)),
        "rule_count": int(summary.get("rule_count", 0)),
        "scaled_rule_count": int(summary.get("scaled_rule_count", 0)),
        "min_delta_scale": maybe_float(summary.get("min_delta_scale")),
        "mean_delta_scale_scaled_groups": maybe_float(summary.get("mean_delta_scale_scaled_groups")),
        "source_total_relative_delta_norm": maybe_float(summary.get("source_total_relative_delta_norm")),
        "estimated_total_relative_delta_norm": maybe_float(summary.get("estimated_total_relative_delta_norm")),
        "estimated_routed_max_relative_delta": maybe_float(summary.get("estimated_routed_max_relative_delta")),
        "estimated_routed_p99_relative_delta": maybe_float(summary.get("estimated_routed_p99_relative_delta")),
        "estimated_routed_tensors_gt_1": int(summary.get("estimated_routed_tensors_gt_1", 0)),
        "estimated_routed_tensors_gt_075": int(summary.get("estimated_routed_tensors_gt_075", 0)),
        "estimated_routed_tensors_gt_065": int(summary.get("estimated_routed_tensors_gt_065", 0)),
        "writer_dry_run_validated": bool(summary.get("writer_dry_run_validated", False)),
        "writer_dry_run_floating_tensors": int(summary.get("writer_dry_run_floating_tensors", 0)),
        "writer_dry_run_frozen_tensors": int(summary.get("writer_dry_run_frozen_tensors", 0)),
        "writer_dry_run_expert_tensor_rule_hits": int(
            summary.get("writer_dry_run_expert_tensor_rule_hits", 0)
        ),
        "writer_dry_run_shared_attention_hits": int(
            summary.get("writer_dry_run_shared_attention_hits", 0)
        ),
        "writer_dry_run_freeze_router_hits": int(summary.get("writer_dry_run_freeze_router_hits", 0)),
        "writer_materialized_validated": bool(summary.get("writer_materialized_validated", False)),
        "writer_materialized_shards": int(summary.get("writer_materialized_shards", 0)),
        "report": rel(root / "report.md"),
        "expert_tail_scales": rel(root / "expert_tail_scales.csv"),
        "tail_trimmed_rules": rel(root / "tail_trimmed_rules.csv"),
        "tensor_rules": rel(root / "tensor_rules.txt"),
        "writer_command": rel(root / "writer_command.txt"),
        "dry_run_command": rel(root / "dry_run_command.txt"),
        "dry_run_manifest": rel(root / "dry_run" / "merge_manifest.json"),
        "materialized_manifest": rel(
            repo_path("results/checkpoints/qwen3_moe_tail_trimmed_expert_only_candidate/merge_manifest.json")
        ),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_delta_frontier() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_delta_frontier")
    summary = read_json(root / "summary.json")
    candidates = read_csv(root / "candidate_delta_frontier.csv")
    pairwise = read_csv(root / "pairwise_delta_reductions.csv")
    structural_pairwise = read_csv(root / "structural_pairwise_distances.csv")
    structural_dominance = read_csv(root / "structural_dominance.csv")
    thresholds = read_csv(root / "tail_thresholds.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "candidate_count": int(summary.get("candidate_count", len(candidates))),
        "best_delta_safety_candidate": summary.get("best_delta_safety_candidate"),
        "structural_dominated_candidate_count": int(
            summary.get("structural_dominated_candidate_count", 0)
        ),
        "structural_dominated_candidates": summary.get("structural_dominated_candidates", []),
        "closest_structural_pair": summary.get("closest_structural_pair"),
        "mechanistic_nearest_structural_candidate": summary.get(
            "mechanistic_nearest_structural_candidate"
        ),
        "mechanistic_nearest_structural_distance": maybe_float(
            summary.get("mechanistic_nearest_structural_distance")
        ),
        "mechanistic_nearest_structural_safety_delta": maybe_float(
            summary.get("mechanistic_nearest_structural_safety_delta")
        ),
        "trust_region_total_relative_delta_norm": maybe_float(
            summary.get("trust_region_total_relative_delta_norm")
        ),
        "expert_only_total_relative_delta_norm": maybe_float(
            summary.get("expert_only_total_relative_delta_norm")
        ),
        "tail_trimmed_total_relative_delta_norm": maybe_float(
            summary.get("tail_trimmed_total_relative_delta_norm")
        ),
        "searched_no_gt065_total_relative_delta_norm": maybe_float(
            summary.get("searched_no_gt065_total_relative_delta_norm")
        ),
        "layer_chunk_total_relative_delta_norm": maybe_float(
            summary.get("layer_chunk_total_relative_delta_norm")
        ),
        "unified_mechanism_total_relative_delta_norm": maybe_float(
            summary.get("unified_mechanism_total_relative_delta_norm")
        ),
        "mechanistic_unified_total_relative_delta_norm": maybe_float(
            summary.get("mechanistic_unified_total_relative_delta_norm")
        ),
        "subspace_scaled_total_relative_delta_norm": maybe_float(
            summary.get("subspace_scaled_total_relative_delta_norm")
        ),
        "unified_mechanism_matches_searched_no_gt065_delta": bool(
            summary.get("unified_mechanism_matches_searched_no_gt065_delta", False)
        ),
        "unified_mechanism_router_changed_tensors": int(
            summary.get("unified_mechanism_router_changed_tensors", 0)
        ),
        "mechanistic_unified_router_changed_tensors": int(
            summary.get("mechanistic_unified_router_changed_tensors", 0)
        ),
        "subspace_scaled_router_changed_tensors": int(
            summary.get("subspace_scaled_router_changed_tensors", 0)
        ),
        "trust_to_expert_only_relative_norm_reduction": maybe_float(
            summary.get("trust_to_expert_only_relative_norm_reduction")
        ),
        "expert_only_to_tail_trimmed_relative_norm_reduction": maybe_float(
            summary.get("expert_only_to_tail_trimmed_relative_norm_reduction")
        ),
        "tail_trimmed_to_searched_no_gt065_relative_norm_delta": maybe_float(
            summary.get("tail_trimmed_to_searched_no_gt065_relative_norm_delta")
        ),
        "searched_no_gt065_to_layer_chunk_relative_norm_reduction": maybe_float(
            summary.get("searched_no_gt065_to_layer_chunk_relative_norm_reduction")
        ),
        "searched_no_gt065_to_layer_chunk_routed_gt_065_reduction": int(
            summary.get("searched_no_gt065_to_layer_chunk_routed_gt_065_reduction", 0)
        ),
        "trust_to_expert_only_attention_norm_reduction": maybe_float(
            summary.get("trust_to_expert_only_attention_norm_reduction")
        ),
        "audit_to_trust_routed_gt_075_reduction": int(
            summary.get("audit_to_trust_routed_gt_075_reduction", 0)
        ),
        "trust_to_expert_only_routed_gt_075_reduction": int(
            summary.get("trust_to_expert_only_routed_gt_075_reduction", 0)
        ),
        "expert_only_to_tail_trimmed_routed_gt_065_reduction": int(
            summary.get("expert_only_to_tail_trimmed_routed_gt_065_reduction", 0)
        ),
        "expert_only_to_tail_trimmed_routed_gt_075_reduction": int(
            summary.get("expert_only_to_tail_trimmed_routed_gt_075_reduction", 0)
        ),
        "tail_trimmed_to_searched_no_gt065_routed_gt_065_delta": int(
            summary.get("tail_trimmed_to_searched_no_gt065_routed_gt_065_delta", 0)
        ),
        "tail_trimmed_routed_gt_0_6505": int(summary.get("tail_trimmed_routed_gt_0_6505", 0)),
        "searched_no_gt065_routed_gt_0_6505": int(
            summary.get("searched_no_gt065_routed_gt_0_6505", 0)
        ),
        "layer_chunk_routed_gt_0_6505": int(summary.get("layer_chunk_routed_gt_0_6505", 0)),
        "layer_chunk_routed_gt_0_65": int(summary.get("layer_chunk_routed_gt_0_65", 0)),
        "unified_mechanism_routed_gt_0_6505": int(
            summary.get("unified_mechanism_routed_gt_0_6505", 0)
        ),
        "unified_mechanism_routed_gt_0_65": int(summary.get("unified_mechanism_routed_gt_0_65", 0)),
        "mechanistic_unified_routed_gt_0_6505": int(
            summary.get("mechanistic_unified_routed_gt_0_6505", 0)
        ),
        "mechanistic_unified_routed_gt_0_65": int(
            summary.get("mechanistic_unified_routed_gt_0_65", 0)
        ),
        "subspace_scaled_routed_gt_0_6505": int(
            summary.get("subspace_scaled_routed_gt_0_6505", 0)
        ),
        "subspace_scaled_routed_gt_0_65": int(summary.get("subspace_scaled_routed_gt_0_65", 0)),
        "layer_chunk_to_unified_relative_norm_reduction": maybe_float(
            summary.get("layer_chunk_to_unified_relative_norm_reduction")
        ),
        "layer_chunk_to_unified_routed_gt_065_reduction": int(
            summary.get("layer_chunk_to_unified_routed_gt_065_reduction", 0)
        ),
        "unified_to_mechanistic_relative_norm_reduction": maybe_float(
            summary.get("unified_to_mechanistic_relative_norm_reduction")
        ),
        "unified_to_mechanistic_routed_gt_065_reduction": int(
            summary.get("unified_to_mechanistic_routed_gt_065_reduction", 0)
        ),
        "mechanistic_to_subspace_relative_norm_delta": maybe_float(
            summary.get("mechanistic_to_subspace_relative_norm_delta")
        ),
        "mechanistic_to_subspace_routed_gt_065_reduction": int(
            summary.get("mechanistic_to_subspace_routed_gt_065_reduction", 0)
        ),
        "expert_only_attention_changed_tensors": int(
            summary.get("expert_only_attention_changed_tensors", 0)
        ),
        "tail_trimmed_attention_changed_tensors": int(
            summary.get("tail_trimmed_attention_changed_tensors", 0)
        ),
        "expert_only_router_changed_tensors": int(summary.get("expert_only_router_changed_tensors", 0)),
        "tail_trimmed_router_changed_tensors": int(summary.get("tail_trimmed_router_changed_tensors", 0)),
        "searched_no_gt065_attention_changed_tensors": int(
            summary.get("searched_no_gt065_attention_changed_tensors", 0)
        ),
        "layer_chunk_attention_changed_tensors": int(
            summary.get("layer_chunk_attention_changed_tensors", 0)
        ),
        "searched_no_gt065_router_changed_tensors": int(
            summary.get("searched_no_gt065_router_changed_tensors", 0)
        ),
        "layer_chunk_router_changed_tensors": int(
            summary.get("layer_chunk_router_changed_tensors", 0)
        ),
        "next_required_gate": summary.get("next_required_gate"),
        "candidate_rows": [clean_row(row) for _, row in candidates.iterrows()],
        "pairwise_rows": [clean_row(row) for _, row in pairwise.iterrows()],
        "structural_pairwise_rows": [clean_row(row) for _, row in structural_pairwise.iterrows()],
        "structural_dominance_rows": [clean_row(row) for _, row in structural_dominance.iterrows()],
        "threshold_rows": [clean_row(row) for _, row in thresholds.iterrows()],
        "report": rel(root / "report.md"),
        "candidate_frontier": rel(root / "candidate_delta_frontier.csv"),
        "group_frontier": rel(root / "group_delta_frontier.csv"),
        "pairwise_reductions": rel(root / "pairwise_delta_reductions.csv"),
        "structural_pairwise": rel(root / "structural_pairwise_distances.csv"),
        "structural_dominance": rel(root / "structural_dominance.csv"),
        "tail_thresholds": rel(root / "tail_thresholds.csv"),
        "layer_frontier": rel(root / "layer_delta_frontier.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_mechanism_eval_gate() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_mechanism_eval_gate")
    summary = read_json(root / "summary.json")
    gate = read_csv(root / "eval_gate_plan.csv")
    tests = read_csv(root / "mechanism_tests.csv")
    selection = read_csv(root / "method_selection.csv")
    current_selection = summary.get("current_selection", {})
    local_gpu = summary.get("local_gpu", {})
    unified_gate = gate[gate["method"] == "qwen3_moe_unified_mechanism_candidate"]
    unified_selection = selection[selection["method"] == "qwen3_moe_unified_mechanism_candidate"]
    unified_test = tests[tests["test"] == "unified_mechanism_optimizer"]
    unified_gate_row = {} if unified_gate.empty else unified_gate.iloc[0].to_dict()
    unified_selection_row = {} if unified_selection.empty else unified_selection.iloc[0].to_dict()
    unified_test_row = {} if unified_test.empty else unified_test.iloc[0].to_dict()
    return {
        "summary": summary,
        "status": summary.get("status"),
        "source_count": int(summary.get("source_count", 0)),
        "candidate_count": int(summary.get("candidate_count", 0)),
        "ready_to_host_count": int(summary.get("ready_to_host_count", 0)),
        "completed_eval_count": int(summary.get("completed_eval_count", 0)),
        "selection_status": current_selection.get("status"),
        "selected_method": current_selection.get("selected_method"),
        "best_delta_safety_candidate": summary.get("best_delta_safety_candidate"),
        "local_gpu_available": bool(local_gpu.get("available", False)),
        "local_gpu_status": local_gpu.get("status"),
        "awaiting_tests": int((tests["current_status"] == "awaiting_eval").sum()),
        "unified_candidate_serve_status": unified_gate_row.get("serve_status"),
        "unified_candidate_eval_status": unified_gate_row.get("eval_status"),
        "unified_candidate_end_to_end_status": unified_gate_row.get("end_to_end_status"),
        "unified_candidate_audit_passed": bool(unified_gate_row.get("audit_passed", False)),
        "unified_candidate_eval_output_dir": unified_gate_row.get("eval_output_dir"),
        "unified_candidate_selection_eligible": bool(
            unified_selection_row.get("selection_eligible", False)
        ),
        "unified_mechanism_optimizer_status": unified_test_row.get("current_status"),
        "mechanism_test_rows": [clean_row(row) for _, row in tests.iterrows()],
        "eval_gate_rows": [clean_row(row) for _, row in gate.iterrows()],
        "selection_rows": [clean_row(row) for _, row in selection.iterrows()],
        "report": rel(root / "report.md"),
        "eval_gate_plan": rel(root / "eval_gate_plan.csv"),
        "mechanism_tests": rel(root / "mechanism_tests.csv"),
        "method_selection": rel(root / "method_selection.csv"),
        "selection_rules": rel(root / "selection_rules.json"),
        "run_script": rel(root / "run_eval_gate.sh"),
        "literature_sources": rel(root / "literature_sources.json"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_eval_budget_plan() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_eval_budget_plan")
    summary = read_json(root / "summary.json")
    task_budget = read_csv(root / "task_budget.csv")
    method_budget = read_csv(root / "method_budget.csv")
    mechanism_budget = read_csv(root / "mechanism_budget.csv")
    task_manifest_alignment = read_csv(root / "task_manifest_alignment.csv")
    capped_tasks = task_budget[task_budget["budget_status"] == "target_not_met_dataset_cap"]
    return {
        "summary": summary,
        "status": summary.get("status"),
        "method_count": int(summary.get("method_count", len(method_budget))),
        "ready_to_host_method_count": int(
            summary.get("ready_to_host_method_count", (method_budget["serve_status"] == "ready_to_host").sum())
        ),
        "pending_materialization_method_count": int(summary.get("pending_materialization_method_count", 0)),
        "source_count": int(summary.get("source_count", 0)),
        "candidate_count": int(summary.get("candidate_count", 0)),
        "current_gate_examples": maybe_int(summary.get("current_gate_examples")),
        "recommended_max_examples": maybe_int(summary.get("recommended_max_examples")),
        "target_wilson_half_width": maybe_float(summary.get("target_wilson_half_width")),
        "wilson_required_examples": maybe_int(summary.get("wilson_required_examples")),
        "paired_required_examples": maybe_int(summary.get("paired_required_examples")),
        "paired_required_discordant": maybe_int(summary.get("paired_required_discordant")),
        "target_paired_net_loss_rate": maybe_float(summary.get("target_paired_net_loss_rate")),
        "paired_alpha": maybe_float(summary.get("paired_alpha")),
        "assumed_paired_discordance_rate": maybe_float(summary.get("assumed_paired_discordance_rate")),
        "total_current_prompt_budget": maybe_int(summary.get("total_current_prompt_budget")),
        "total_recommended_prompt_budget": maybe_int(summary.get("total_recommended_prompt_budget")),
        "total_additional_prompt_budget": maybe_int(summary.get("total_additional_prompt_budget")),
        "ready_to_host_current_prompt_budget": maybe_int(summary.get("ready_to_host_current_prompt_budget")),
        "ready_to_host_recommended_prompt_budget": maybe_int(summary.get("ready_to_host_recommended_prompt_budget")),
        "ready_to_host_additional_prompt_budget": maybe_int(summary.get("ready_to_host_additional_prompt_budget")),
        "default_runner_request": summary.get("default_runner_request"),
        "final_core_method_count": maybe_int(summary.get("final_core_method_count")),
        "final_core_ready_to_host_method_count": maybe_int(summary.get("final_core_ready_to_host_method_count")),
        "final_core_recommended_prompt_budget": maybe_int(summary.get("final_core_recommended_prompt_budget")),
        "mechanism_ablation_method_count": maybe_int(summary.get("mechanism_ablation_method_count")),
        "mechanism_ablation_ready_to_host_method_count": maybe_int(
            summary.get("mechanism_ablation_ready_to_host_method_count")
        ),
        "mechanism_ablation_recommended_prompt_budget": maybe_int(
            summary.get("mechanism_ablation_recommended_prompt_budget")
        ),
        "eval_queue_summary": summary.get("eval_queue_summary", []),
        "canonical_task_manifest": summary.get("canonical_task_manifest"),
        "task_manifest_path_count": maybe_int(summary.get("task_manifest_path_count")),
        "task_manifest_aligned_method_count": maybe_int(summary.get("task_manifest_aligned_method_count")),
        "task_manifest_unaligned_method_count": maybe_int(summary.get("task_manifest_unaligned_method_count")),
        "task_manifest_pairing_contract": summary.get("task_manifest_pairing_contract"),
        "router_calibration_active_candidate_count": int(
            (summary.get("router_calibration") or {}).get("active_candidate_count", 0)
        ),
        "router_calibration_active_ready_count": int(
            (summary.get("router_calibration") or {}).get("active_ready_count", 0)
        ),
        "router_calibration_active_pending_count": int(
            (summary.get("router_calibration") or {}).get("active_pending_count", 0)
        ),
        "router_calibration_plan_pruned_candidate_count": int(
            (summary.get("router_calibration") or {}).get("plan_pruned_candidate_count", 0)
        ),
        "dataset_capped_tasks": ",".join(str(row["task"]) for _, row in capped_tasks.iterrows()),
        "task_rows": [clean_row(row) for _, row in task_budget.iterrows()],
        "method_rows": [clean_row(row) for _, row in method_budget.iterrows()],
        "mechanism_rows": [clean_row(row) for _, row in mechanism_budget.iterrows()],
        "task_manifest_alignment_rows": [clean_row(row) for _, row in task_manifest_alignment.iterrows()],
        "report": rel(root / "report.md"),
        "task_budget": rel(root / "task_budget.csv"),
        "method_budget": rel(root / "method_budget.csv"),
        "mechanism_budget": rel(root / "mechanism_budget.csv"),
        "router_calibration_budget": rel(root / "router_calibration_budget.csv"),
        "task_manifest_alignment": rel(root / "task_manifest_alignment.csv"),
        "run_script": rel(root / "run_eval_budget.sh"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_eval_budget_queue_smoke() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_eval_budget_queue_smoke")
    summary = read_json(root / "summary.json")
    matrix = read_csv(root / "eval_budget_queue_smoke_matrix.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "assertion_count": int(summary.get("assertion_count", len(matrix))),
        "passed_assertion_count": int(summary.get("passed_assertion_count", 0)),
        "failed_assertion_count": int(summary.get("failed_assertion_count", 0)),
        "final_queue_method_count": maybe_int(summary.get("final_queue_method_count")),
        "mechanism_queue_method_count": maybe_int(summary.get("mechanism_queue_method_count")),
        "router_queue_method_count": maybe_int(summary.get("router_queue_method_count")),
        "final_queue_methods": summary.get("final_queue_methods", []),
        "mechanism_queue_methods": summary.get("mechanism_queue_methods", []),
        "matrix_rows": [clean_row(row) for _, row in matrix.iterrows()],
        "report": rel(root / "report.md"),
        "matrix": rel(root / "eval_budget_queue_smoke_matrix.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_adaptive_eval_schedule() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_adaptive_eval_schedule")
    summary = read_json(root / "summary.json")
    schedule = read_csv(root / "adaptive_schedule.csv")
    mechanisms = read_csv(root / "mechanism_schedule.csv")
    round1 = schedule[schedule["selected_for_round1_probe"].astype(bool)] if "selected_for_round1_probe" in schedule else pd.DataFrame()
    return {
        "summary": summary,
        "status": summary.get("status"),
        "method_count": int(summary.get("method_count", len(schedule))),
        "source_controls_complete": bool(summary.get("source_controls_complete", False)),
        "round1_probe_candidate_count": int(summary.get("round1_probe_candidate_count", len(round1))),
        "round1_selection_policy": summary.get("round1_selection_policy"),
        "round1_covered_mechanism_test_count": int(summary.get("round1_covered_mechanism_test_count", 0)),
        "round1_covered_mechanism_tests": summary.get("round1_covered_mechanism_tests", []),
        "structural_frontier_available": bool(summary.get("structural_frontier_available", False)),
        "best_structural_method": summary.get("best_structural_method"),
        "best_structural_safety_score": maybe_float(summary.get("best_structural_safety_score")),
        "structural_dominance_available": bool(summary.get("structural_dominance_available", False)),
        "structural_frontier_member_count": maybe_int(summary.get("structural_frontier_member_count")),
        "structurally_dominated_method_count": maybe_int(summary.get("structurally_dominated_method_count")),
        "round1_probe_task_budget": maybe_int(summary.get("round1_probe_task_budget")),
        "runnable_prompt_budget": maybe_int(summary.get("runnable_prompt_budget")),
        "runnable_method_count": maybe_int(summary.get("runnable_method_count")),
        "top_eval_action": summary.get("top_eval_action"),
        "top_method": summary.get("top_method"),
        "probe_examples": maybe_int(summary.get("probe_examples")),
        "full_examples": maybe_int(summary.get("full_examples")),
        "eval_action_counts": summary.get("eval_action_counts", {}),
        "decision_status_counts": summary.get("decision_status_counts", {}),
        "paired_gate_status_counts": (
            {str(key): int(value) for key, value in schedule["paired_gate_status"].value_counts().to_dict().items()}
            if "paired_gate_status" in schedule
            else {}
        ),
        "paired_loss_tolerance_rate": maybe_float(summary.get("paired_loss_tolerance_rate")),
        "paired_alpha": maybe_float(summary.get("paired_alpha")),
        "round1_probe_methods": ",".join(str(row["method"]) for _, row in round1.iterrows()),
        "schedule_rows": [clean_row(row) for _, row in schedule.iterrows()],
        "mechanism_rows": [clean_row(row) for _, row in mechanisms.iterrows()],
        "report": rel(root / "report.md"),
        "schedule": rel(root / "adaptive_schedule.csv"),
        "mechanism_schedule": rel(root / "mechanism_schedule.csv"),
        "run_script": rel(root / "run_adaptive_eval.sh"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_adaptive_eval_schedule_smoke() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_adaptive_eval_schedule_smoke")
    summary = read_json(root / "summary.json")
    matrix = read_csv(root / "adaptive_eval_schedule_smoke_matrix.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "case_count": int(summary.get("case_count", matrix["case"].nunique() if not matrix.empty else 0)),
        "assertion_count": int(summary.get("assertion_count", len(matrix))),
        "passed_assertion_count": int(summary.get("passed_assertion_count", 0)),
        "failed_assertion_count": int(summary.get("failed_assertion_count", 0)),
        "case_rows": [clean_row(row) for _, row in matrix.iterrows()],
        "report": rel(root / "report.md"),
        "matrix": rel(root / "adaptive_eval_schedule_smoke_matrix.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_eval_manifest_preflight() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_eval_manifest_preflight")
    summary = read_json(root / "summary.json")
    task_checks = read_csv(root / "task_manifest_checks.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "canonical_task_manifest": summary.get("canonical_task_manifest"),
        "manifest_exists": bool(summary.get("manifest_exists", False)),
        "manifest_sha256": summary.get("manifest_sha256"),
        "method_count": maybe_int(summary.get("method_count")),
        "task_manifest_aligned_method_count": maybe_int(summary.get("task_manifest_aligned_method_count")),
        "task_manifest_unaligned_method_count": maybe_int(
            summary.get("task_manifest_unaligned_method_count")
        ),
        "task_count": maybe_int(summary.get("task_count")),
        "task_sufficient_count": maybe_int(summary.get("task_sufficient_count")),
        "total_required_examples": maybe_int(summary.get("total_required_examples")),
        "total_manifest_examples": maybe_int(summary.get("total_manifest_examples")),
        "blocking_reason": summary.get("blocking_reason"),
        "task_rows": [clean_row(row) for _, row in task_checks.iterrows()],
        "report": rel(root / "report.md"),
        "task_manifest_checks": rel(root / "task_manifest_checks.csv"),
        "prepare_manifest_command": rel(root / "prepare_manifest_command.txt"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_mechanism_levers() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_mechanism_levers")
    summary = read_json(root / "summary.json")
    levers = read_csv(root / "mechanism_levers.csv")
    queue = read_csv(root / "next_experiment_queue.csv")
    chunking = read_csv(root / "layer_chunking_plan.csv")
    top_lever = clean_row(levers.iloc[0]) if not levers.empty else {}
    top_chunk = clean_row(chunking.iloc[0]) if not chunking.empty else {}
    return {
        "summary": summary,
        "status": summary.get("status"),
        "lever_count": int(summary.get("lever_count", len(levers))),
        "top_lever": summary.get("top_lever"),
        "top_lever_priority": maybe_float(summary.get("top_lever_priority")),
        "fine_calibration_layers": summary.get("fine_calibration_layers"),
        "expert_geometry_probe_used": bool(summary.get("expert_geometry_probe_used", False)),
        "expert_subspace_probe_used": bool(summary.get("expert_subspace_probe_used", False)),
        "high_subspace_conflict_expert_count": maybe_int(summary.get("high_subspace_conflict_expert_count")),
        "subspace_extra_scaled_expert_count": maybe_int(summary.get("subspace_extra_scaled_expert_count")),
        "top_subspace_conflict_layer": maybe_int(summary.get("top_subspace_conflict_layer")),
        "top_expert_geometry_layer": maybe_int(summary.get("top_expert_geometry_layer")),
        "top_expert_geometry_layer_risk": maybe_float(summary.get("top_expert_geometry_layer_risk")),
        "queue_count": int(summary.get("queue_count", len(queue))),
        "literature_count": int(summary.get("literature_count", 0)),
        "top_lever_action": top_lever.get("current_action"),
        "top_lever_next_test": top_lever.get("next_test"),
        "top_chunk_layer": maybe_int(top_chunk.get("layer")),
        "top_chunk_score": maybe_float(top_chunk.get("layer_importance_score")),
        "lever_rows": [clean_row(row) for _, row in levers.iterrows()],
        "queue_rows": [clean_row(row) for _, row in queue.iterrows()],
        "chunking_rows": [clean_row(row) for _, row in chunking.iterrows()],
        "report": rel(root / "report.md"),
        "mechanism_levers": rel(root / "mechanism_levers.csv"),
        "next_experiment_queue": rel(root / "next_experiment_queue.csv"),
        "layer_chunking_plan": rel(root / "layer_chunking_plan.csv"),
        "literature_sources": rel(root / "literature_sources.json"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_expert_subspace_conflict_probe() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_expert_subspace_conflict_probe")
    summary = read_json(root / "summary.json")
    experts = read_csv(root / "expert_subspace_conflicts.csv")
    layers = read_csv(root / "layer_subspace_conflicts.csv")
    actions = read_csv(root / "subspace_action_summary.csv")
    top_layer = clean_row(layers.iloc[0]) if not layers.empty else {}
    top_extra_scaled = (
        clean_row(experts[experts["subspace_extra_scale"] < 0.999].sort_values("subspace_extra_scale").iloc[0])
        if not experts.empty and "subspace_extra_scale" in experts and (experts["subspace_extra_scale"] < 0.999).any()
        else {}
    )
    return {
        "summary": summary,
        "status": summary.get("status"),
        "projection_tensor_count": int(summary.get("projection_tensor_count", 0)),
        "expert_count": int(summary.get("expert_count", 0)),
        "layer_count": int(summary.get("layer_count", 0)),
        "high_subspace_conflict_expert_count": int(summary.get("high_subspace_conflict_expert_count", 0)),
        "route_important_high_subspace_conflict_expert_count": int(
            summary.get("route_important_high_subspace_conflict_expert_count", 0)
        ),
        "subspace_extra_scaled_expert_count": int(summary.get("subspace_extra_scaled_expert_count", 0)),
        "subspace_adjusted_tensor_rule_count": int(summary.get("subspace_adjusted_tensor_rule_count", 0)),
        "mean_subspace_conflict_score": maybe_float(summary.get("mean_subspace_conflict_score")),
        "max_subspace_conflict_score": maybe_float(summary.get("max_subspace_conflict_score")),
        "mean_coder_weight_reduction": maybe_float(summary.get("mean_coder_weight_reduction")),
        "total_coder_weight_reduction": maybe_float(summary.get("total_coder_weight_reduction")),
        "dry_run_validated": bool(summary.get("dry_run_validated", False)),
        "dry_run_floating_tensors": int(summary.get("dry_run_floating_tensors", 0)),
        "dry_run_frozen_tensors": int(summary.get("dry_run_frozen_tensors", 0)),
        "dry_run_tensor_rule_count": int(summary.get("dry_run_tensor_rule_count", 0)),
        "dry_run_tensor_rule_hit_count": int(summary.get("dry_run_tensor_rule_hit_count", 0)),
        "dry_run_default_tensor_count": int(summary.get("dry_run_default_tensor_count", 0)),
        "dry_run_freeze_router_hits": int(summary.get("dry_run_freeze_router_hits", 0)),
        "top_subspace_conflict_layer": maybe_int(summary.get("top_subspace_conflict_layer")),
        "next_action": summary.get("next_action"),
        "top_layer": top_layer,
        "top_extra_scaled": top_extra_scaled,
        "action_rows": [clean_row(row) for _, row in actions.iterrows()],
        "top_layer_rows": [clean_row(row) for _, row in layers.head(12).iterrows()],
        "top_expert_rows": [clean_row(row) for _, row in experts.head(16).iterrows()],
        "report": rel(root / "report.md"),
        "projection_scores": rel(root / "projection_subspace_scores.csv"),
        "expert_conflicts": rel(root / "expert_subspace_conflicts.csv"),
        "layer_conflicts": rel(root / "layer_subspace_conflicts.csv"),
        "action_summary": rel(root / "subspace_action_summary.csv"),
        "subspace_adjusted_group_rules": rel(root / "subspace_adjusted_group_rules.csv"),
        "subspace_adjusted_tensor_rules": rel(root / "subspace_adjusted_tensor_rules.txt"),
        "dry_run_command": rel(root / "dry_run_command.txt"),
        "dry_run_manifest": rel(root / "dry_run/merge_manifest.json"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_expert_geometry_probe() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_expert_geometry_probe")
    summary = read_json(root / "summary.json")
    projection_summary = read_csv(root / "projection_summary.csv")
    layers = read_csv(root / "layer_geometry.csv")
    experts = read_csv(root / "expert_geometry.csv")
    top_layer = clean_row(layers.iloc[0]) if not layers.empty else {}
    top_expert = clean_row(experts.iloc[0]) if not experts.empty else {}
    return {
        "summary": summary,
        "status": summary.get("status"),
        "projection_tensor_count": int(summary.get("projection_tensor_count", 0)),
        "expert_count": int(summary.get("expert_count", 0)),
        "layer_count": int(summary.get("layer_count", 0)),
        "mean_projection_cosine": maybe_float(summary.get("mean_projection_cosine")),
        "p05_projection_cosine": maybe_float(summary.get("p05_projection_cosine")),
        "mean_projection_relative_delta": maybe_float(summary.get("mean_projection_relative_delta")),
        "p95_projection_relative_delta": maybe_float(summary.get("p95_projection_relative_delta")),
        "max_projection_relative_delta": maybe_float(summary.get("max_projection_relative_delta")),
        "mean_expert_combined_relative_delta": maybe_float(summary.get("mean_expert_combined_relative_delta")),
        "p95_expert_combined_relative_delta": maybe_float(summary.get("p95_expert_combined_relative_delta")),
        "max_expert_combined_relative_delta": maybe_float(summary.get("max_expert_combined_relative_delta")),
        "mean_expert_combined_cosine": maybe_float(summary.get("mean_expert_combined_cosine")),
        "min_expert_combined_cosine": maybe_float(summary.get("min_expert_combined_cosine")),
        "high_internal_geometry_risk_expert_count": int(
            summary.get("high_internal_geometry_risk_expert_count", 0)
        ),
        "high_route_geometry_risk_expert_count": int(
            summary.get("high_route_geometry_risk_expert_count", 0)
        ),
        "route_observed_expert_count": int(summary.get("route_observed_expert_count", 0)),
        "top_layer_by_route_geometry_risk": maybe_int(summary.get("top_layer_by_route_geometry_risk")),
        "top_layer_route_mass_weighted_route_geometry_risk": maybe_float(
            summary.get("top_layer_route_mass_weighted_route_geometry_risk")
        ),
        "top_expert_layer": maybe_int(summary.get("top_expert_layer")),
        "top_expert_id": maybe_int(summary.get("top_expert_id")),
        "top_expert_route_geometry_risk_score": maybe_float(
            summary.get("top_expert_route_geometry_risk_score")
        ),
        "top_expert_combined_relative_delta": maybe_float(
            summary.get("top_expert_combined_relative_delta")
        ),
        "recommended_unified_action": summary.get("recommended_unified_action"),
        "top_layer": top_layer,
        "top_expert": top_expert,
        "projection_rows": [clean_row(row) for _, row in projection_summary.iterrows()],
        "top_layer_rows": [clean_row(row) for _, row in layers.head(12).iterrows()],
        "top_expert_rows": [clean_row(row) for _, row in experts.head(16).iterrows()],
        "report": rel(root / "report.md"),
        "projection_summary": rel(root / "projection_summary.csv"),
        "projection_geometry": rel(root / "projection_geometry.csv"),
        "expert_geometry": rel(root / "expert_geometry.csv"),
        "layer_geometry": rel(root / "layer_geometry.csv"),
        "top_chunk_geometry": rel(root / "top_chunk_geometry.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_layer_chunk_candidate() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_layer_chunk_candidate")
    summary = read_json(root / "summary.json")
    search = read_csv(root / "schedule_search.csv")
    layers = read_csv(root / "layer_coefficients.csv")
    feasible = (
        search[search["feasible_for_selection"].astype(str).str.lower().eq("true")]
        if "feasible_for_selection" in search
        else pd.DataFrame()
    )
    return {
        "summary": summary,
        "status": summary.get("status"),
        "expert_group_count": int(summary.get("expert_group_count", 0)),
        "schedule_count": int(summary.get("schedule_count", len(search))),
        "feasible_schedule_count": int(len(feasible)),
        "selected_schedule_id": summary.get("selected_schedule_id"),
        "selected_route_mass_weighted_coder_retention": maybe_float(
            summary.get("selected_route_mass_weighted_coder_retention")
        ),
        "selected_fine_layer_coder_retention": maybe_float(
            summary.get("selected_fine_layer_coder_retention")
        ),
        "selected_risk_weighted_delta_reduction": maybe_float(
            summary.get("selected_risk_weighted_delta_reduction")
        ),
        "selected_delta_norm_proxy_ratio": maybe_float(summary.get("selected_delta_norm_proxy_ratio")),
        "selected_max_predicted_relative_delta": maybe_float(
            summary.get("selected_max_predicted_relative_delta")
        ),
        "selected_changed_group_count": int(summary.get("selected_changed_group_count", 0)),
        "selection_min_retention": maybe_float(summary.get("selection_min_retention")),
        "selection_hard_cap": maybe_float(summary.get("selection_hard_cap")),
        "dry_run_validated": bool(summary.get("dry_run_validated", False)),
        "dry_run_floating_tensors": int(summary.get("dry_run_floating_tensors", 0)),
        "dry_run_frozen_tensors": int(summary.get("dry_run_frozen_tensors", 0)),
        "dry_run_freeze_router": bool(summary.get("dry_run_freeze_router", False)),
        "dry_run_default_tensor_count": int(summary.get("dry_run_default_tensor_count", 0)),
        "dry_run_freeze_router_hits": int(summary.get("dry_run_freeze_router_hits", 0)),
        "dry_run_tensor_rule_count": int(summary.get("dry_run_tensor_rule_count", 0)),
        "dry_run_tensor_rule_hit_count": int(summary.get("dry_run_tensor_rule_hit_count", 0)),
        "schedule_rows": [clean_row(row) for _, row in search.iterrows()],
        "layer_rows": [clean_row(row) for _, row in layers.iterrows()],
        "report": rel(root / "report.md"),
        "selected_schedule": rel(root / "selected_schedule.json"),
        "schedule_search": rel(root / "schedule_search.csv"),
        "layer_coefficients": rel(root / "layer_coefficients.csv"),
        "selected_group_rules": rel(root / "selected_group_rules.csv"),
        "tensor_rules": rel(root / "tensor_rules.txt"),
        "writer_command": rel(root / "writer_command.txt"),
        "dry_run_command": rel(root / "dry_run_command.txt"),
        "dry_run_manifest": rel(repo_path("results/checkpoints/qwen3_moe_layer_chunk_candidate/merge_manifest.json")),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_unified_result_selection() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_unified_result_selection")
    summary = read_json(root / "summary.json")
    table = read_csv(root / "selection_table.csv")
    selection = summary.get("current_selection", {})
    eligible = table[table["selection_eligible"].astype(bool)] if "selection_eligible" in table else pd.DataFrame()
    unified_rows = table[table["method"] == "qwen3_moe_unified_mechanism_candidate"]
    unified_row = {} if unified_rows.empty else clean_row(unified_rows.iloc[0])
    return {
        "summary": summary,
        "status": summary.get("status"),
        "selected_method": selection.get("selected_method"),
        "selection_reason": selection.get("reason"),
        "sources_complete": bool(selection.get("sources_complete", False)),
        "unified_completed": bool(selection.get("unified_completed", False)),
        "unified_audit_passed": bool(selection.get("unified_audit_passed", False)),
        "alias_status": selection.get("alias_status"),
        "best_source_by_avg": selection.get("best_source_by_avg"),
        "avg_gain_vs_best_source": maybe_float(selection.get("avg_gain_vs_best_source")),
        "worst_gain_vs_best_source": maybe_float(selection.get("worst_gain_vs_best_source")),
        "eligible_candidate_count": int(len(eligible)),
        "candidate_count": int(summary.get("candidate_count", len(table))),
        "unified_selection_eligible": bool(unified_row.get("selection_eligible", False)),
        "unified_dominated_by_source": unified_row.get("dominated_by_source"),
        "unified_task_regression_columns": unified_row.get("task_regression_columns"),
        "candidate_rows": [clean_row(row) for _, row in table.iterrows()],
        "report": rel(root / "report.md"),
        "selection_table": rel(root / "selection_table.csv"),
        "decision_rules": rel(root / "decision_rules.json"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_unified_result_selection_smoke() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_unified_result_selection_smoke")
    summary = read_json(root / "summary.json")
    matrix = read_csv(root / "selector_matrix.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "case_count": int(summary.get("case_count", len(matrix))),
        "passed_case_count": int(summary.get("passed_case_count", 0)),
        "failed_case_count": int(summary.get("failed_case_count", 0)),
        "case_rows": [clean_row(row) for _, row in matrix.iterrows()],
        "report": rel(root / "report.md"),
        "matrix": rel(root / "selector_matrix.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_candidate_trust_region_gate() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_candidate_trust_region_gate")
    summary = read_json(root / "summary.json")
    table = read_csv(root / "candidate_trust_region_gate.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "candidate_count": int(summary.get("candidate_count", 0)),
        "final_selectable_candidate_count": int(summary.get("final_selectable_candidate_count", 0)),
        "ablation_only_candidate_count": int(summary.get("ablation_only_candidate_count", 0)),
        "structural_reject_candidate_count": int(summary.get("structural_reject_candidate_count", 0)),
        "strict_routed_max_relative_delta": maybe_float(summary.get("strict_routed_max_relative_delta")),
        "max_routed_tensors_gt_065": maybe_int(summary.get("max_routed_tensors_gt_065")),
        "final_selectable_methods": summary.get("final_selectable_methods", []),
        "gate_rows": [clean_row(row) for _, row in table.iterrows()],
        "report": rel(root / "report.md"),
        "gate": rel(root / "candidate_trust_region_gate.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_final_candidate_selection() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_final_candidate_selection")
    summary = read_json(root / "summary.json")
    table = read_csv(root / "selection_table.csv")
    selection = summary.get("current_selection", {})
    eligible = table[table["selection_eligible"].astype(bool)] if "selection_eligible" in table else pd.DataFrame()
    selected = table[table["method"] == selection.get("selected_method")] if "method" in table else pd.DataFrame()
    selected_row = clean_row(selected.iloc[0]) if not selected.empty else {}
    structural_frontier_eligible_count = 0
    if not eligible.empty and "structural_frontier_member" in eligible:
        structural_frontier_eligible_count = int(
            eligible["structural_frontier_member"].map(lambda value: maybe_bool(value) is True).sum()
        )
    return {
        "summary": summary,
        "status": summary.get("status"),
        "selected_method": selection.get("selected_method"),
        "selection_reason": selection.get("reason"),
        "sources_complete": bool(selection.get("sources_complete", False)),
        "candidates_complete": bool(selection.get("candidates_complete", False)),
        "uncertainty_gate": bool(selection.get("uncertainty_gate", False)),
        "paired_prediction_gate": bool(selection.get("paired_prediction_gate", False)),
        "paired_alpha": maybe_float(selection.get("paired_alpha")),
        "selection_score_tie_tolerance": maybe_float(selection.get("selection_score_tie_tolerance")),
        "confidence_tie_band": bool(selection.get("confidence_tie_band", False)),
        "selection_rank_mode": selection.get("selection_rank_mode"),
        "selection_point_leader_method": selection.get("selection_point_leader_method"),
        "selection_rank_band_size": maybe_int(selection.get("selection_rank_band_size")),
        "selection_rank_band_methods": selection.get("selection_rank_band_methods", []),
        "selection_rank_policy": selection.get("selection_rank_policy", []),
        "structural_dominance_available": bool(selection.get("structural_dominance_available", False)),
        "structural_frontier_eligible_count": int(
            selection.get("structural_frontier_eligible_count", structural_frontier_eligible_count) or 0
        ),
        "selected_structural_frontier_member": maybe_bool(
            selection.get("selected_structural_frontier_member", selected_row.get("structural_frontier_member"))
        ),
        "selected_structurally_dominated": maybe_bool(
            selection.get("selected_structurally_dominated", selected_row.get("structurally_dominated"))
        ),
        "selected_structural_safety_score": maybe_float(
            selection.get("selected_structural_safety_score", selected_row.get("structural_safety_score"))
        ),
        "usable_candidate_count": int(selection.get("usable_candidate_count", 0)),
        "eligible_candidate_count": int(selection.get("eligible_candidate_count", len(eligible))),
        "candidate_count": int(selection.get("candidate_count", len(table[table["role"] == "candidate"]))),
        "best_source_by_avg": selection.get("best_source_by_avg"),
        "best_source_by_worst": selection.get("best_source_by_worst"),
        "candidate_rows": [clean_row(row) for _, row in table.iterrows()],
        "report": rel(root / "report.md"),
        "selection_table": rel(root / "selection_table.csv"),
        "decision_rules": rel(root / "decision_rules.json"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_final_candidate_selection_smoke() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_final_candidate_selection_smoke")
    summary = read_json(root / "summary.json")
    matrix = read_csv(root / "selector_matrix.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "case_count": int(summary.get("case_count", len(matrix))),
        "passed_case_count": int(summary.get("passed_case_count", 0)),
        "failed_case_count": int(summary.get("failed_case_count", 0)),
        "case_rows": [clean_row(row) for _, row in matrix.iterrows()],
        "report": rel(root / "report.md"),
        "matrix": rel(root / "selector_matrix.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_unified_average_optimizer() -> dict[str, Any]:
    root = repo_path("results/unified_average_optimizer")
    summary = read_json(root / "summary.json")
    features = read_csv(root / "mechanism_features.csv")
    decisions = read_csv(root / "operation_decisions.csv")
    hypotheses = read_csv(root / "mechanism_hypotheses.csv")
    evidence_ledger = read_csv(root / "hypothesis_evidence_ledger.csv")
    algorithm_contract = read_csv(root / "algorithm_contract.csv")
    next_queue = read_csv(root / "next_experiment_queue.csv")
    dense = summary.get("dense", {})
    moe = summary.get("moe", {})
    return {
        "summary": summary,
        "status": summary.get("status"),
        "hypothesis_count": maybe_int(summary.get("hypothesis_count")),
        "hypothesis_status_counts": summary.get("hypothesis_status_counts", {}),
        "evidence_ledger_count": maybe_int(summary.get("evidence_ledger_count")),
        "evidence_verdict_counts": summary.get("evidence_verdict_counts", {}),
        "contract_status": summary.get("contract_status"),
        "contract_requirement_count": maybe_int(summary.get("contract_requirement_count")),
        "contract_passed_requirement_count": maybe_int(summary.get("contract_passed_requirement_count")),
        "contract_blocking_requirement_count": maybe_int(summary.get("contract_blocking_requirement_count")),
        "contract_failed_requirement_count": maybe_int(summary.get("contract_failed_requirement_count")),
        "contract_blocking_requirements": summary.get("contract_blocking_requirements", []),
        "next_experiment_count": maybe_int(summary.get("next_experiment_count")),
        "top_next_experiment": summary.get("top_next_experiment", {}),
        "dense_decision": dense.get("decision"),
        "dense_linear_worst_nll": maybe_float(dense.get("linear_worst_nll")),
        "dense_unified_worst_nll": maybe_float(dense.get("unified_worst_nll")),
        "dense_best_endpoint_worst_nll": maybe_float(dense.get("best_endpoint_worst_nll")),
        "dense_lambda_linear_worst_nll": maybe_float(dense.get("lambda_linear_worst_nll")),
        "dense_lambda_best_worst_nll": maybe_float(dense.get("lambda_best_worst_nll")),
        "dense_curvature_ratio_general": maybe_float(dense.get("curvature_ratio_general")),
        "dense_curvature_ratio_code": maybe_float(dense.get("curvature_ratio_code")),
        "moe_decision": moe.get("decision"),
        "real_gauge_naive_degradation": maybe_float(moe.get("real_gauge_naive_degradation")),
        "real_gauge_aligned_degradation": maybe_float(moe.get("real_gauge_aligned_degradation")),
        "qwen3_identity_fraction": maybe_float(moe.get("qwen3_identity_fraction")),
        "router_action": moe.get("router_action"),
        "qwen3_unified_candidate_id": moe.get("qwen3_unified_candidate_id"),
        "qwen3_unified_candidate_family": moe.get("qwen3_unified_candidate_family"),
        "qwen3_unified_candidate_count": maybe_int(moe.get("qwen3_unified_candidate_count")),
        "qwen3_unified_nonbase_mass_retention": maybe_float(
            moe.get("qwen3_unified_nonbase_mass_retention")
        ),
        "qwen3_unified_subspace_weighted_predicted_relative_delta": maybe_float(
            moe.get("qwen3_unified_subspace_weighted_predicted_relative_delta")
        ),
        "qwen3_unified_subspace_risk_weighted_coder_retention": maybe_float(
            moe.get("qwen3_unified_subspace_risk_weighted_coder_retention")
        ),
        "qwen3_unified_high_subspace_mean_scale": maybe_float(
            moe.get("qwen3_unified_high_subspace_mean_scale")
        ),
        "qwen3_unified_materialized_rule_status": moe.get("qwen3_unified_materialized_rule_status"),
        "qwen3_unified_matches_materialized_checkpoint_manifest": bool(
            moe.get("qwen3_unified_matches_materialized_checkpoint_manifest", False)
        ),
        "qwen3_unified_max_manifest_weight_abs_diff": maybe_float(
            moe.get("qwen3_unified_max_manifest_weight_abs_diff")
        ),
        "qwen3_unified_relative_delta_norm": maybe_float(
            moe.get("qwen3_unified_relative_delta_norm")
        ),
        "qwen3_unified_routed_gt_065": maybe_int(moe.get("qwen3_unified_routed_gt_065")),
        "qwen3_router_margin_status": moe.get("qwen3_router_margin_status"),
        "qwen3_router_margin_high_fragility_layers": maybe_int(
            moe.get("qwen3_router_margin_high_fragility_layers")
        ),
        "qwen3_router_margin_layer_count": maybe_int(moe.get("qwen3_router_margin_layer_count")),
        "qwen3_router_margin_top_layer": maybe_int(moe.get("qwen3_router_margin_top_layer")),
        "qwen3_router_margin_top_score": maybe_float(moe.get("qwen3_router_margin_top_score")),
        "qwen3_router_margin_top_category": moe.get("qwen3_router_margin_top_category"),
        "qwen3_router_margin_top_category_score": maybe_float(
            moe.get("qwen3_router_margin_top_category_score")
        ),
        "qwen3_router_margin_min_safe_lambda_proxy": maybe_float(
            moe.get("qwen3_router_margin_min_safe_lambda_proxy")
        ),
        "qwen3_layer_chunk_to_unified_relative_norm_reduction": maybe_float(
            moe.get("qwen3_layer_chunk_to_unified_relative_norm_reduction")
        ),
        "qwen3_layer_chunk_to_unified_routed_gt_065_reduction": maybe_int(
            moe.get("qwen3_layer_chunk_to_unified_routed_gt_065_reduction")
        ),
        "qwen3_router_calibration_nll_status": moe.get("qwen3_router_calibration_nll_status"),
        "qwen3_router_calibration_nll_worst_reduction": maybe_float(
            moe.get("qwen3_router_calibration_nll_worst_reduction")
        ),
        "qwen3_router_calibration_nll_avg_reduction": maybe_float(
            moe.get("qwen3_router_calibration_nll_avg_reduction")
        ),
        "qwen3_router_calibration_nll_code_gap_to_best_source": maybe_float(
            moe.get("qwen3_router_calibration_nll_code_gap_to_best_source")
        ),
        "qwen3_router_calibration_nll_worst_gap_to_best_source": maybe_float(
            moe.get("qwen3_router_calibration_nll_worst_gap_to_best_source")
        ),
        "qwen3_generation_matrix_status": moe.get("qwen3_generation_matrix_status"),
        "qwen3_generation_pair_routercal_avg_gain": maybe_float(
            moe.get("qwen3_generation_pair_routercal_avg_gain")
        ),
        "qwen3_generation_pair_routercal_gap_to_best_parent_avg": maybe_float(
            moe.get("qwen3_generation_pair_routercal_gap_to_best_parent_avg")
        ),
        "qwen3_generation_attribution_status": moe.get("qwen3_generation_attribution_status"),
        "qwen3_generation_avg_routercal_recovery_fraction": maybe_float(
            moe.get("qwen3_generation_avg_routercal_recovery_fraction")
        ),
        "qwen3_generation_routercal_beats_pair_frontier_count": maybe_int(
            moe.get("qwen3_generation_routercal_beats_pair_frontier_count")
        ),
        "qwen3_generation_confidence_status": moe.get("qwen3_generation_confidence_status"),
        "qwen3_generation_confidence_task_count": maybe_int(
            moe.get("qwen3_generation_confidence_task_count")
        ),
        "qwen3_generation_routercal_positive_tasks_vs_naive": maybe_int(
            moe.get("qwen3_generation_routercal_positive_tasks_vs_naive")
        ),
        "qwen3_generation_routercal_confident_positive_tasks": maybe_int(
            moe.get("qwen3_generation_routercal_confident_positive_tasks")
        ),
        "qwen3_generation_routercal_confident_source_frontier_wins": maybe_int(
            moe.get("qwen3_generation_routercal_confident_source_frontier_wins")
        ),
        "qwen3_generation_routercal_avg_gain_lower": maybe_float(
            moe.get("qwen3_generation_routercal_avg_gain_lower")
        ),
        "qwen3_generation_routercal_avg_gain_upper": maybe_float(
            moe.get("qwen3_generation_routercal_avg_gain_upper")
        ),
        "qwen3_router_calibration_status": moe.get("qwen3_router_calibration_status"),
        "qwen3_router_calibration_eligible_candidates": maybe_int(
            moe.get("qwen3_router_calibration_eligible_candidates")
        ),
        "qwen3_router_calibration_candidate_count": maybe_int(
            moe.get("qwen3_router_calibration_candidate_count")
        ),
        "qwen3_interpolation_interior_gap_nll": maybe_float(
            moe.get("qwen3_interpolation_interior_gap_nll")
        ),
        "qwen3_interpolation_general_barrier_nll": maybe_float(
            moe.get("qwen3_interpolation_general_barrier_nll")
        ),
        "qwen3_complementary_merge_beats_sources": bool(
            moe.get("qwen3_complementary_merge_beats_sources", False)
        ),
        "qwen3_base_coder_interior_gap_nll": maybe_float(moe.get("qwen3_base_coder_interior_gap_nll")),
        "qwen3_base_coder_general_barrier_nll": maybe_float(
            moe.get("qwen3_base_coder_general_barrier_nll")
        ),
        "qwen3_final_selection_status": moe.get("qwen3_final_selection_status"),
        "qwen3_eligible_candidates": maybe_int(moe.get("qwen3_eligible_candidates")),
        "qwen3_candidate_count": maybe_int(moe.get("qwen3_candidate_count")),
        "qwen3_final_confidence_tie_band": bool(moe.get("qwen3_final_confidence_tie_band", False)),
        "qwen3_final_selection_rank_mode": moe.get("qwen3_final_selection_rank_mode"),
        "qwen3_final_selection_point_leader_method": moe.get("qwen3_final_selection_point_leader_method"),
        "qwen3_final_selection_rank_band_size": maybe_int(moe.get("qwen3_final_selection_rank_band_size")),
        "qwen3_final_structural_frontier_eligible_count": maybe_int(
            moe.get("qwen3_final_structural_frontier_eligible_count")
        ),
        "feature_rows": [clean_row(row) for _, row in features.iterrows()],
        "decision_rows": [clean_row(row) for _, row in decisions.iterrows()],
        "hypothesis_rows": [clean_row(row) for _, row in hypotheses.iterrows()],
        "evidence_ledger_rows": [clean_row(row) for _, row in evidence_ledger.iterrows()],
        "algorithm_contract_rows": [clean_row(row) for _, row in algorithm_contract.iterrows()],
        "next_experiment_rows": [clean_row(row) for _, row in next_queue.iterrows()],
        "report": rel(root / "report.md"),
        "features": rel(root / "mechanism_features.csv"),
        "decisions": rel(root / "operation_decisions.csv"),
        "hypotheses": rel(root / "mechanism_hypotheses.csv"),
        "evidence_ledger": rel(root / "hypothesis_evidence_ledger.csv"),
        "algorithm_contract": rel(root / "algorithm_contract.csv"),
        "next_experiment_queue": rel(root / "next_experiment_queue.csv"),
        "algorithm": rel(root / "algorithm.json"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_unified_average_optimizer_ledger_smoke() -> dict[str, Any]:
    root = repo_path("results/unified_average_optimizer_ledger_smoke")
    summary = read_json(root / "summary.json")
    matrix = read_csv(root / "ledger_matrix.csv")
    queue_matrix_path = root / "queue_matrix.csv"
    queue_matrix = read_csv(queue_matrix_path) if queue_matrix_path.exists() else pd.DataFrame()
    contract_matrix_path = root / "contract_matrix.csv"
    contract_matrix = read_csv(contract_matrix_path) if contract_matrix_path.exists() else pd.DataFrame()
    return {
        "summary": summary,
        "status": summary.get("status"),
        "case_count": int(summary.get("case_count", matrix["case"].nunique() if not matrix.empty else 0)),
        "assertion_count": int(summary.get("assertion_count", len(matrix))),
        "passed_case_count": int(summary.get("passed_case_count", 0)),
        "failed_case_count": int(summary.get("failed_case_count", 0)),
        "passed_assertion_count": int(summary.get("passed_assertion_count", 0)),
        "failed_assertion_count": int(summary.get("failed_assertion_count", 0)),
        "case_rows": [clean_row(row) for _, row in matrix.iterrows()],
        "queue_case_rows": [clean_row(row) for _, row in queue_matrix.iterrows()],
        "contract_case_rows": [clean_row(row) for _, row in contract_matrix.iterrows()],
        "report": rel(root / "report.md"),
        "matrix": rel(root / "ledger_matrix.csv"),
        "queue_matrix": rel(queue_matrix_path) if queue_matrix_path.exists() else None,
        "contract_matrix": rel(contract_matrix_path) if contract_matrix_path.exists() else None,
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_router_calibration_nll_probe() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_router_calibration_nll_probe")
    summary = read_json(root / "summary.json")
    methods = read_csv(root / "method_metrics.csv")
    deltas = read_csv(root / "mechanism_deltas.csv")
    delta_by_name = {str(row["metric"]): clean_row(row) for _, row in deltas.iterrows()}
    return {
        "summary": summary,
        "status": summary.get("status"),
        "acceptance_decision": summary.get("acceptance_decision"),
        "steps": maybe_int(summary.get("steps")),
        "lr": maybe_float(summary.get("lr")),
        "linear_merge": find_method(methods, "linear_merge"),
        "router_calibrated": find_method(methods, "linear_merge_routercal"),
        "worst_nll_reduction_vs_linear": maybe_float(summary.get("worst_nll_reduction_vs_linear")),
        "avg_nll_reduction_vs_linear": maybe_float(summary.get("avg_nll_reduction_vs_linear")),
        "general_nll_reduction_vs_linear": maybe_float(summary.get("general_nll_reduction_vs_linear")),
        "code_nll_reduction_vs_linear": maybe_float(summary.get("code_nll_reduction_vs_linear")),
        "routercal_code_gap_to_best_source": maybe_float(summary.get("routercal_code_gap_to_best_source")),
        "routercal_worst_gap_to_best_source": maybe_float(summary.get("routercal_worst_gap_to_best_source")),
        "delta_rows": [clean_row(row) for _, row in deltas.iterrows()],
        "delta_by_name": delta_by_name,
        "report": rel(root / "report.md"),
        "method_metrics": rel(root / "method_metrics.csv"),
        "mechanism_deltas": rel(root / "mechanism_deltas.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_eval_bundle_audit() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_eval_bundle_audit")
    summary = read_json(root / "summary.json")
    rows = read_csv(root / "audit_rows.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "method_count": int(summary.get("method_count", len(rows))),
        "complete_eval_count": int(summary.get("complete_eval_count", 0)),
        "usable_for_selection_count": int(summary.get("usable_for_selection_count", 0)),
        "invalid_complete_count": int(summary.get("invalid_complete_count", 0)),
        "source_usable_count": int(summary.get("source_usable_count", 0)),
        "source_count": int(summary.get("source_count", 0)),
        "candidate_usable_count": int(summary.get("candidate_usable_count", 0)),
        "candidate_count": int(summary.get("candidate_count", 0)),
        "unified_usable": bool(summary.get("unified_usable", False)),
        "pairability_complete_source_count": int(summary.get("pairability_complete_source_count", 0)),
        "pairability_failed_method_count": int(summary.get("pairability_failed_method_count", 0)),
        "audit_rows": [clean_row(row) for _, row in rows.iterrows()],
        "report": rel(root / "report.md"),
        "audit_rows_path": rel(root / "audit_rows.csv"),
        "pairability": rel(root / "pairability.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_eval_bundle_audit_smoke() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_eval_bundle_audit_smoke")
    summary = read_json(root / "summary.json")
    matrix = read_csv(root / "audit_matrix.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "case_count": int(summary.get("case_count", len(matrix))),
        "passed_case_count": int(summary.get("passed_case_count", 0)),
        "failed_case_count": int(summary.get("failed_case_count", 0)),
        "case_rows": [clean_row(row) for _, row in matrix.iterrows()],
        "report": rel(root / "report.md"),
        "matrix": rel(root / "audit_matrix.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_mechanism_effect_attribution() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_mechanism_effect_attribution")
    summary = read_json(root / "summary.json")
    transitions = read_csv(root / "transition_effects.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "transition_count": int(summary.get("transition_count", len(transitions))),
        "scored_transition_count": int(summary.get("scored_transition_count", 0)),
        "improving_transition_count": int(summary.get("improving_transition_count", 0)),
        "regressing_transition_count": int(summary.get("regressing_transition_count", 0)),
        "best_avg_transition": summary.get("best_avg_transition"),
        "best_avg_delta": maybe_float(summary.get("best_avg_delta")),
        "best_worst_transition": summary.get("best_worst_transition"),
        "best_worst_delta": maybe_float(summary.get("best_worst_delta")),
        "transition_rows": [clean_row(row) for _, row in transitions.iterrows()],
        "report": rel(root / "report.md"),
        "transition_effects": rel(root / "transition_effects.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_mechanism_effect_attribution_smoke() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_mechanism_effect_attribution_smoke")
    summary = read_json(root / "summary.json")
    matrix = read_csv(root / "attribution_matrix.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "case_count": int(summary.get("case_count", len(matrix))),
        "passed_case_count": int(summary.get("passed_case_count", 0)),
        "failed_case_count": int(summary.get("failed_case_count", 0)),
        "case_rows": [clean_row(row) for _, row in matrix.iterrows()],
        "report": rel(root / "report.md"),
        "matrix": rel(root / "attribution_matrix.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_feedback_optimizer() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_feedback_optimizer")
    summary = read_json(root / "summary.json")
    task_feedback = read_csv(root / "task_feedback.csv")
    updates = read_csv(root / "feature_update_summary.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "candidate_method": summary.get("candidate_method"),
        "requested_candidate_method": summary.get("requested_candidate_method"),
        "feedback_base_selection_status": summary.get("feedback_base_selection_status"),
        "feedback_base_structural_frontier_member": summary.get(
            "feedback_base_structural_frontier_member"
        ),
        "feedback_base_structurally_dominated": summary.get("feedback_base_structurally_dominated"),
        "feedback_base_structural_safety_score": maybe_float(
            summary.get("feedback_base_structural_safety_score")
        ),
        "feedback_base_candidate_count": len(
            summary.get("feedback_base_candidate_rows_considered") or []
        ),
        "feedback_base_candidate_rows_considered": summary.get(
            "feedback_base_candidate_rows_considered", []
        ),
        "group_rules_source": summary.get("group_rules_source"),
        "writer_context_command_source": summary.get("writer_context_command_source"),
        "materialization_gate": summary.get("materialization_gate"),
        "task_count": int(summary.get("task_count", len(task_feedback))),
        "scored_task_count": int(summary.get("scored_task_count", 0)),
        "regression_task_count": int(summary.get("regression_task_count", 0)),
        "changed_group_count": int(summary.get("changed_group_count", 0)),
        "route_mass_weighted_nonbase_ratio": maybe_float(
            summary.get("route_mass_weighted_nonbase_ratio")
        ),
        "max_feedback_expected_relative_delta": maybe_float(
            summary.get("max_feedback_expected_relative_delta")
        ),
        "groups_over_hard_cap_after_feedback": int(
            summary.get("groups_over_hard_cap_after_feedback", 0)
        ),
        "task_rows": [clean_row(row) for _, row in task_feedback.iterrows()],
        "update_rows": [clean_row(row) for _, row in updates.iterrows()],
        "report": rel(root / "report.md"),
        "task_feedback": rel(root / "task_feedback.csv"),
        "feature_update_summary": rel(root / "feature_update_summary.csv"),
        "feedback_group_rules": rel(root / "feedback_group_rules.csv"),
        "tensor_rules": rel(root / "tensor_rules.txt"),
        "writer_command": rel(root / "writer_command.txt"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_feedback_optimizer_smoke() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_feedback_optimizer_smoke")
    summary = read_json(root / "summary.json")
    matrix = read_csv(root / "feedback_optimizer_smoke_matrix.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "case_count": int(summary.get("case_count", len(matrix))),
        "passed_case_count": int(summary.get("passed_case_count", 0)),
        "failed_case_count": int(summary.get("failed_case_count", 0)),
        "case_rows": [clean_row(row) for _, row in matrix.iterrows()],
        "report": rel(root / "report.md"),
        "matrix": rel(root / "feedback_optimizer_smoke_matrix.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_mechanistic_unified_candidate() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_mechanistic_unified_candidate")
    summary = read_json(root / "summary.json")
    search = read_csv(root / "candidate_search.csv")
    group_rules = read_csv(root / "mechanistic_group_rules.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "expert_group_count": int(summary.get("expert_group_count", len(group_rules))),
        "candidate_count": int(summary.get("candidate_count", len(search))),
        "selected_candidate_id": summary.get("selected_candidate_id"),
        "nominal_hard_cap": maybe_float(summary.get("nominal_hard_cap", summary.get("hard_cap"))),
        "materialization_safety_margin": maybe_float(summary.get("materialization_safety_margin")),
        "effective_hard_cap": maybe_float(summary.get("effective_hard_cap", summary.get("hard_cap"))),
        "selected_nonbase_mass_retention": maybe_float(summary.get("selected_nonbase_mass_retention")),
        "selected_max_predicted_relative_delta": maybe_float(summary.get("selected_max_predicted_relative_delta")),
        "selected_hard_cap_violation_count": int(summary.get("selected_hard_cap_violation_count", 0)),
        "selected_risk_weighted_predicted_delta": maybe_float(
            summary.get("selected_risk_weighted_predicted_delta")
        ),
        "selected_benefit_weighted_scale": maybe_float(summary.get("selected_benefit_weighted_scale")),
        "selected_mean_mechanistic_loss_proxy": maybe_float(
            summary.get("selected_mean_mechanistic_loss_proxy")
        ),
        "selected_mean_scale": maybe_float(summary.get("selected_mean_scale")),
        "selected_min_scale": maybe_float(summary.get("selected_min_scale")),
        "selected_high_benefit_low_risk_mean_scale": maybe_float(
            summary.get("selected_high_benefit_low_risk_mean_scale")
        ),
        "selected_high_interference_low_benefit_mean_scale": maybe_float(
            summary.get("selected_high_interference_low_benefit_mean_scale")
        ),
        "selected_high_subspace_mean_scale": maybe_float(summary.get("selected_high_subspace_mean_scale")),
        "feedback_status": summary.get("feedback_status"),
        "feedback_materialization_gate": summary.get("feedback_materialization_gate"),
        "writer_manifest_validated": bool(summary.get("writer_manifest_validated", False)),
        "writer_manifest_dry_run": summary.get("writer_manifest_dry_run"),
        "dry_run_validated": bool(summary.get("dry_run_validated", False)),
        "dry_run_tensor_rule_count": int(summary.get("dry_run_tensor_rule_count", 0)),
        "dry_run_tensor_rule_hit_count": int(summary.get("dry_run_tensor_rule_hit_count", 0)),
        "dry_run_freeze_router_hits": int(summary.get("dry_run_freeze_router_hits", 0)),
        "dry_run_floating_tensors": int(summary.get("dry_run_floating_tensors", 0)),
        "dry_run_frozen_tensors": int(summary.get("dry_run_frozen_tensors", 0)),
        "candidate_rows": [clean_row(row) for _, row in search.iterrows()],
        "report": rel(root / "report.md"),
        "candidate_search": rel(root / "candidate_search.csv"),
        "mechanistic_group_rules": rel(root / "mechanistic_group_rules.csv"),
        "tensor_rules": rel(root / "tensor_rules.txt"),
        "writer_command": rel(root / "writer_command.txt"),
        "dry_run_command": rel(root / "dry_run_command.txt"),
        "literature_sources": rel(root / "literature_sources.json"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_mechanistic_unified_candidate_smoke() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_mechanistic_unified_candidate_smoke")
    summary = read_json(root / "summary.json")
    matrix = read_csv(root / "mechanistic_unified_smoke_matrix.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "case_count": int(summary.get("case_count", len(matrix))),
        "passed_case_count": int(summary.get("passed_case_count", 0)),
        "failed_case_count": int(summary.get("failed_case_count", 0)),
        "case_rows": [clean_row(row) for _, row in matrix.iterrows()],
        "report": rel(root / "report.md"),
        "matrix": rel(root / "mechanistic_unified_smoke_matrix.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_mechanistic_evidence_audit() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_mechanistic_evidence_audit")
    summary = read_json(root / "summary.json")
    bindings = read_csv(root / "binding_summary.csv")
    correlations = read_csv(root / "feature_correlations.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "selected_candidate_id": summary.get("selected_candidate_id"),
        "group_count": int(summary.get("group_count", 0)),
        "nominal_hard_cap": maybe_float(summary.get("nominal_hard_cap", summary.get("hard_cap"))),
        "materialization_safety_margin": maybe_float(summary.get("materialization_safety_margin")),
        "effective_hard_cap": maybe_float(summary.get("effective_hard_cap", summary.get("hard_cap"))),
        "hard_cap_bound_group_count": int(summary.get("hard_cap_bound_group_count", 0)),
        "hard_cap_bound_route_mass": maybe_float(summary.get("hard_cap_bound_route_mass")),
        "gradient_sign_agreement_rate": maybe_float(summary.get("gradient_sign_agreement_rate")),
        "objective_proxy_improved_group_fraction": maybe_float(
            summary.get("objective_proxy_improved_group_fraction")
        ),
        "route_mass_weighted_objective_gain_vs_prior": maybe_float(
            summary.get("route_mass_weighted_objective_gain_vs_prior")
        ),
        "route_mass_weighted_selected_scale": maybe_float(
            summary.get("route_mass_weighted_selected_scale")
        ),
        "route_mass_weighted_scale_delta": maybe_float(summary.get("route_mass_weighted_scale_delta")),
        "dominant_binding": summary.get("dominant_binding"),
        "dominant_binding_group_count": int(summary.get("dominant_binding_group_count", 0)),
        "dominant_binding_route_mass": maybe_float(summary.get("dominant_binding_route_mass")),
        "most_scale_suppressing_features": summary.get("most_scale_suppressing_features") or [],
        "most_scale_preserving_features": summary.get("most_scale_preserving_features") or [],
        "binding_rows": [clean_row(row) for _, row in bindings.iterrows()],
        "correlation_rows": [clean_row(row) for _, row in correlations.iterrows()],
        "report": rel(root / "report.md"),
        "binding_summary": rel(root / "binding_summary.csv"),
        "feature_deciles": rel(root / "feature_decile_response.csv"),
        "feature_correlations": rel(root / "feature_correlations.csv"),
        "hard_cases": rel(root / "hard_cases.csv"),
        "group_evidence": rel(root / "group_mechanistic_evidence.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_mechanistic_sensitivity() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_mechanistic_sensitivity")
    summary = read_json(root / "summary.json")
    ablations = read_csv(root / "feature_family_ablation.csv")
    correlations = read_csv(root / "feature_correlations.csv")
    affected = read_csv(root / "top_affected_groups.csv")
    strongest_objective = summary.get("strongest_fixed_objective_regression") or {}
    strongest_scale = summary.get("strongest_scale_sensitivity") or {}
    top_shrink_feature = summary.get("top_shrink_feature") or {}
    return {
        "summary": summary,
        "status": summary.get("status"),
        "baseline_candidate_id": summary.get("baseline_candidate_id"),
        "baseline_reselected_candidate_id": summary.get("baseline_reselected_candidate_id"),
        "effective_hard_cap": maybe_float(summary.get("effective_hard_cap")),
        "ablation_count": int(summary.get("ablation_count", len(ablations))),
        "feature_correlation_count": int(summary.get("feature_correlation_count", len(correlations))),
        "affected_group_rows": int(summary.get("affected_group_rows", len(affected))),
        "baseline_fixed_objective": maybe_float(summary.get("baseline_fixed_objective")),
        "baseline_reselected_objective": maybe_float(summary.get("baseline_reselected_objective")),
        "strongest_objective_ablation": strongest_objective.get("ablation"),
        "strongest_objective_delta": maybe_float(strongest_objective.get("fixed_objective_delta")),
        "strongest_objective_retention_delta": maybe_float(
            strongest_objective.get("fixed_nonbase_mass_retention_delta")
        ),
        "strongest_objective_reselected_candidate_id": strongest_objective.get(
            "reselected_candidate_id"
        ),
        "strongest_scale_ablation": strongest_scale.get("ablation"),
        "strongest_scale_shift": maybe_float(strongest_scale.get("route_mass_weighted_abs_scale_shift")),
        "strongest_scale_groups_changed_gt_0_01": maybe_int(
            strongest_scale.get("groups_changed_gt_0_01")
        ),
        "top_shrink_feature": top_shrink_feature.get("feature"),
        "top_shrink_feature_family": top_shrink_feature.get("feature_family"),
        "top_shrink_feature_corr": maybe_float(top_shrink_feature.get("weighted_corr_with_shrink")),
        "ablation_rows": [clean_row(row) for _, row in ablations.iterrows()],
        "correlation_rows": [clean_row(row) for _, row in correlations.iterrows()],
        "affected_group_rows_data": [clean_row(row) for _, row in affected.iterrows()],
        "report": rel(root / "report.md"),
        "feature_family_ablation": rel(root / "feature_family_ablation.csv"),
        "feature_correlations": rel(root / "feature_correlations.csv"),
        "top_affected_groups": rel(root / "top_affected_groups.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_router_expert_coupling() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_router_expert_coupling")
    summary = read_json(root / "summary.json")
    layer_rows = read_csv(root / "layer_router_expert_coupling.csv")
    expert_rows = read_csv(root / "top_router_coupled_experts.csv")
    correlations = read_csv(root / "coupling_correlations.csv")
    high = summary.get("high_fragility_slice") or {}
    low = summary.get("low_fragility_slice") or {}
    return {
        "summary": summary,
        "status": summary.get("status"),
        "gate": summary.get("gate"),
        "gate_reason": summary.get("gate_reason"),
        "selected_mechanistic_candidate_id": summary.get("selected_mechanistic_candidate_id"),
        "group_count": int(summary.get("group_count", len(expert_rows))),
        "layer_count": int(summary.get("layer_count", len(layer_rows))),
        "high_fragility_layer_count": int(summary.get("high_fragility_layer_count", 0)),
        "top_coupled_layer_id": maybe_int(summary.get("top_coupled_layer_id")),
        "top_coupled_layer_risk": maybe_float(summary.get("top_coupled_layer_risk")),
        "top_coupled_layer_fragility": maybe_float(summary.get("top_coupled_layer_fragility")),
        "top_coupled_layer_weighted_shrink": maybe_float(
            summary.get("top_coupled_layer_weighted_shrink")
        ),
        "fragility_router_feature_corr": maybe_float(
            summary.get("fragility_router_feature_corr")
        ),
        "fragility_scale_shrink_corr": maybe_float(summary.get("fragility_scale_shrink_corr")),
        "safe_lambda_scale_shrink_corr": maybe_float(summary.get("safe_lambda_scale_shrink_corr")),
        "fragility_expected_delta_corr": maybe_float(summary.get("fragility_expected_delta_corr")),
        "high_vs_low_weighted_shrink_lift": maybe_float(
            summary.get("high_vs_low_weighted_shrink_lift")
        ),
        "high_fragility_weighted_scale": maybe_float(high.get("weighted_scale")),
        "high_fragility_weighted_shrink": maybe_float(high.get("weighted_scale_shrink")),
        "low_fragility_weighted_scale": maybe_float(low.get("weighted_scale")),
        "low_fragility_weighted_shrink": maybe_float(low.get("weighted_scale_shrink")),
        "layer_rows": [clean_row(row) for _, row in layer_rows.iterrows()],
        "expert_rows": [clean_row(row) for _, row in expert_rows.iterrows()],
        "correlation_rows": [clean_row(row) for _, row in correlations.iterrows()],
        "report": rel(root / "report.md"),
        "layer_router_expert_coupling": rel(root / "layer_router_expert_coupling.csv"),
        "top_router_coupled_experts": rel(root / "top_router_coupled_experts.csv"),
        "coupling_correlations": rel(root / "coupling_correlations.csv"),
        "literature_sources": rel(root / "literature_sources.json"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_router_coupled_candidate() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_router_coupled_candidate")
    summary = read_json(root / "summary.json")
    search = read_csv(root / "candidate_search.csv")
    group_rules = read_csv(root / "router_coupled_group_rules.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "selection_gate": summary.get("selection_gate"),
        "writer_candidate_kind": summary.get("writer_candidate_kind"),
        "base_mechanistic_candidate_id": summary.get("base_mechanistic_candidate_id"),
        "selected_candidate_id": summary.get("selected_candidate_id"),
        "candidate_count": int(summary.get("candidate_count", len(search))),
        "expert_group_count": int(summary.get("expert_group_count", len(group_rules))),
        "effective_hard_cap": maybe_float(summary.get("effective_hard_cap")),
        "min_retention": maybe_float(summary.get("min_retention")),
        "base_nonbase_mass_retention": maybe_float(summary.get("base_nonbase_mass_retention")),
        "selected_nonbase_mass_retention": maybe_float(summary.get("selected_nonbase_mass_retention")),
        "selected_retention_delta_vs_mechanistic": maybe_float(
            summary.get("selected_retention_delta_vs_mechanistic")
        ),
        "selected_changed_group_count": maybe_int(summary.get("selected_changed_group_count")),
        "selected_changed_group_mean_extra_shrink": maybe_float(
            summary.get("selected_changed_group_mean_extra_shrink")
        ),
        "selected_max_extra_shrink": maybe_float(summary.get("selected_max_extra_shrink")),
        "selected_max_predicted_relative_delta": maybe_float(
            summary.get("selected_max_predicted_relative_delta")
        ),
        "selected_hard_cap_violation_count": maybe_int(
            summary.get("selected_hard_cap_violation_count")
        ),
        "selected_router_coupled_delta_reduction_vs_mechanistic": maybe_float(
            summary.get("selected_router_coupled_delta_reduction_vs_mechanistic")
        ),
        "selected_risk_delta_reduction_vs_mechanistic": maybe_float(
            summary.get("selected_risk_delta_reduction_vs_mechanistic")
        ),
        "candidate_rows": [clean_row(row) for _, row in search.iterrows()],
        "report": rel(root / "report.md"),
        "candidate_search": rel(root / "candidate_search.csv"),
        "router_coupled_group_rules": rel(root / "router_coupled_group_rules.csv"),
        "selected_candidate": rel(root / "selected_candidate.json"),
        "tensor_rules": rel(root / "tensor_rules.txt"),
        "writer_command": rel(root / "writer_command.txt"),
        "dry_run_command": rel(root / "dry_run_command.txt"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_router_coupled_retention_frontier() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_router_coupled_retention_frontier")
    summary = read_json(root / "summary.json")
    frontier = read_csv(root / "retention_frontier.csv")
    constrained = summary.get("constrained") or {}
    stress = summary.get("stress") or {}
    return {
        "summary": summary,
        "status": summary.get("status"),
        "gate": summary.get("gate"),
        "recommended_unified_action": summary.get("recommended_unified_action"),
        "base_mechanistic_candidate_id": summary.get("base_mechanistic_candidate_id"),
        "effective_hard_cap": maybe_float(summary.get("effective_hard_cap")),
        "min_retention": maybe_float(summary.get("min_retention")),
        "base_nonbase_mass_retention": maybe_float(summary.get("base_nonbase_mass_retention")),
        "base_router_coupled_weighted_predicted_delta": maybe_float(
            summary.get("base_router_coupled_weighted_predicted_delta")
        ),
        "candidate_count": int(summary.get("candidate_count", len(frontier))),
        "default_gate_candidate_count": maybe_int(summary.get("default_gate_candidate_count")),
        "minimum_effective_fraction": maybe_float(summary.get("minimum_effective_fraction")),
        "constrained_effect_fraction_vs_stress": maybe_float(
            summary.get("constrained_effect_fraction_vs_stress")
        ),
        "constrained_candidate_id": constrained.get("candidate_id"),
        "constrained_retention": maybe_float(constrained.get("nonbase_mass_retention")),
        "constrained_retention_delta": maybe_float(constrained.get("retention_delta_vs_base")),
        "constrained_router_coupled_delta_reduction": maybe_float(
            constrained.get("router_coupled_delta_reduction_vs_base")
        ),
        "constrained_risk_delta_reduction": maybe_float(
            constrained.get("risk_delta_reduction_vs_base")
        ),
        "stress_candidate_id": stress.get("candidate_id"),
        "stress_retention": maybe_float(stress.get("nonbase_mass_retention")),
        "stress_retention_delta": maybe_float(stress.get("retention_delta_vs_base")),
        "stress_router_coupled_delta_reduction": maybe_float(
            stress.get("router_coupled_delta_reduction_vs_base")
        ),
        "stress_risk_delta_reduction": maybe_float(stress.get("risk_delta_reduction_vs_base")),
        "report": rel(root / "report.md"),
        "retention_frontier": rel(root / "retention_frontier.csv"),
        "retention_constrained_group_rules": rel(
            root / "retention_constrained_group_rules.csv"
        ),
        "selected_retention_constrained_candidate": rel(
            root / "selected_retention_constrained_candidate.json"
        ),
        "selected_stress_candidate": rel(root / "selected_stress_candidate.json"),
        "tensor_rules": rel(root / "tensor_rules.txt"),
        "writer_command": rel(root / "writer_command.txt"),
        "dry_run_command": rel(root / "dry_run_command.txt"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_post_eval_refresh() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_post_eval_refresh")
    summary = read_json(root / "summary.json")
    steps = read_csv(root / "steps.csv")
    downstream = summary.get("downstream") or {}
    return {
        "summary": summary,
        "status": summary.get("status"),
        "step_count": int(summary.get("step_count", len(steps))),
        "passed_step_count": int(summary.get("passed_step_count", 0)),
        "failed_step_count": int(summary.get("failed_step_count", 0)),
        "audit_status": downstream.get("audit_status"),
        "audit_usable_for_selection": maybe_int(downstream.get("audit_usable_for_selection")),
        "audit_method_count": maybe_int(downstream.get("audit_method_count")),
        "selection_status": downstream.get("selection_status"),
        "selected_method": downstream.get("selected_method"),
        "selection_reason": downstream.get("selection_reason"),
        "final_selection_status": downstream.get("final_selection_status"),
        "final_selected_method": downstream.get("final_selected_method"),
        "final_selection_reason": downstream.get("final_selection_reason"),
        "final_eligible_candidate_count": maybe_int(downstream.get("final_eligible_candidate_count")),
        "final_candidate_count": maybe_int(downstream.get("final_candidate_count")),
        "attribution_status": downstream.get("attribution_status"),
        "attribution_scored_transition_count": maybe_int(
            downstream.get("attribution_scored_transition_count")
        ),
        "attribution_transition_count": maybe_int(downstream.get("attribution_transition_count")),
        "feedback_status": downstream.get("feedback_status"),
        "feedback_scored_task_count": maybe_int(downstream.get("feedback_scored_task_count")),
        "feedback_task_count": maybe_int(downstream.get("feedback_task_count")),
        "feedback_regression_task_count": maybe_int(
            downstream.get("feedback_regression_task_count")
        ),
        "feedback_changed_group_count": maybe_int(downstream.get("feedback_changed_group_count")),
        "mechanistic_status": downstream.get("mechanistic_status"),
        "mechanistic_selected_candidate": downstream.get("mechanistic_selected_candidate"),
        "mechanistic_retention": maybe_float(downstream.get("mechanistic_retention")),
        "mechanistic_hard_cap_violations": maybe_int(downstream.get("mechanistic_hard_cap_violations")),
        "mechanistic_sensitivity_status": downstream.get("mechanistic_sensitivity_status"),
        "mechanistic_sensitivity_strongest_objective_ablation": downstream.get(
            "mechanistic_sensitivity_strongest_objective_ablation"
        ),
        "mechanistic_sensitivity_strongest_objective_delta": maybe_float(
            downstream.get("mechanistic_sensitivity_strongest_objective_delta")
        ),
        "mechanistic_sensitivity_strongest_scale_ablation": downstream.get(
            "mechanistic_sensitivity_strongest_scale_ablation"
        ),
        "mechanistic_sensitivity_scale_shift": maybe_float(
            downstream.get("mechanistic_sensitivity_scale_shift")
        ),
        "router_expert_coupling_status": downstream.get("router_expert_coupling_status"),
        "router_expert_coupling_gate": downstream.get("router_expert_coupling_gate"),
        "router_expert_coupling_fragility_router_feature_corr": maybe_float(
            downstream.get("router_expert_coupling_fragility_router_feature_corr")
        ),
        "router_expert_coupling_fragility_scale_shrink_corr": maybe_float(
            downstream.get("router_expert_coupling_fragility_scale_shrink_corr")
        ),
        "router_expert_coupling_shrink_lift": maybe_float(
            downstream.get("router_expert_coupling_shrink_lift")
        ),
        "router_expert_coupling_top_layer": maybe_int(
            downstream.get("router_expert_coupling_top_layer")
        ),
        "router_coupled_candidate_status": downstream.get("router_coupled_candidate_status"),
        "router_coupled_candidate_selection_gate": downstream.get(
            "router_coupled_candidate_selection_gate"
        ),
        "router_coupled_candidate_selected": downstream.get("router_coupled_candidate_selected"),
        "router_coupled_candidate_retention": maybe_float(
            downstream.get("router_coupled_candidate_retention")
        ),
        "router_coupled_candidate_retention_delta": maybe_float(
            downstream.get("router_coupled_candidate_retention_delta")
        ),
        "router_coupled_candidate_delta_reduction": maybe_float(
            downstream.get("router_coupled_candidate_delta_reduction")
        ),
        "router_coupled_frontier_status": downstream.get("router_coupled_frontier_status"),
        "router_coupled_frontier_gate": downstream.get("router_coupled_frontier_gate"),
        "router_coupled_frontier_action": downstream.get("router_coupled_frontier_action"),
        "router_coupled_frontier_effect_fraction": maybe_float(
            downstream.get("router_coupled_frontier_effect_fraction")
        ),
        "router_coupled_frontier_default_gate_candidates": maybe_int(
            downstream.get("router_coupled_frontier_default_gate_candidates")
        ),
        "router_coupled_frontier_candidate_count": maybe_int(
            downstream.get("router_coupled_frontier_candidate_count")
        ),
        "step_rows": [clean_row(row) for _, row in steps.iterrows()],
        "report": rel(root / "report.md"),
        "steps": rel(root / "steps.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_post_eval_refresh_plan() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_post_eval_refresh_plan")
    summary = read_json(root / "summary.json")
    steps = read_csv(root / "steps.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "step_count": int(summary.get("step_count", len(steps))),
        "planned_step_count": int(summary.get("planned_step_count", 0)),
        "step_rows": [clean_row(row) for _, row in steps.iterrows()],
        "report": rel(root / "report.md"),
        "steps": rel(root / "steps.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_router_move_gate() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_router_move_gate")
    summary = read_json(root / "summary.json")
    layer_gate = read_csv(root / "router_layer_move_gate.csv")
    router_delta = read_csv(root / "router_delta_summary.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "router_layer_count": int(summary.get("router_layer_count", 0)),
        "allowed_router_layer_count": int(summary.get("allowed_router_layer_count", 0)),
        "frozen_router_layer_count": int(summary.get("frozen_router_layer_count", 0)),
        "unsafe_readiness_rows": int(summary.get("unsafe_readiness_rows", 0)),
        "calibrate_readiness_rows": int(summary.get("calibrate_readiness_rows", 0)),
        "freeze_readiness_rows": int(summary.get("freeze_readiness_rows", 0)),
        "small_lambda_readiness_rows": int(summary.get("small_lambda_readiness_rows", 0)),
        "passed_readiness_rows": int(summary.get("passed_readiness_rows", 0)),
        "total_router_relative_delta_norm": maybe_float(summary.get("total_router_relative_delta_norm")),
        "max_router_relative_delta_norm": maybe_float(summary.get("max_router_relative_delta_norm")),
        "mean_topk_jaccard": maybe_float(summary.get("mean_topk_jaccard")),
        "min_topk_jaccard": maybe_float(summary.get("min_topk_jaccard")),
        "mean_top1_agreement": maybe_float(summary.get("mean_top1_agreement")),
        "min_top1_agreement": maybe_float(summary.get("min_top1_agreement")),
        "recommended_unified_router_action": summary.get("recommended_unified_router_action"),
        "layer_gate_rows": [clean_row(row) for _, row in layer_gate.iterrows()],
        "router_delta_rows": [clean_row(row) for _, row in router_delta.iterrows()],
        "report": rel(root / "report.md"),
        "router_layer_move_gate": rel(root / "router_layer_move_gate.csv"),
        "router_delta_summary": rel(root / "router_delta_summary.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_router_margin_fragility() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_router_margin_fragility")
    summary = read_json(root / "summary.json")
    layers = read_csv(root / "layer_margin_fragility.csv")
    categories = read_csv(root / "category_margin_fragility.csv")
    slices = read_csv(root / "slice_margin_fragility.csv")
    top_layer = clean_row(layers.iloc[0]) if not layers.empty else {}
    top_category = clean_row(categories.iloc[0]) if not categories.empty else {}
    return {
        "summary": summary,
        "status": summary.get("status"),
        "router_layer_count": int(summary.get("router_layer_count", len(layers))),
        "high_fragility_layer_count": int(summary.get("high_fragility_layer_count", 0)),
        "top_fragile_layer": maybe_int(summary.get("top_fragile_layer")),
        "top_fragility_score": maybe_float(summary.get("top_fragility_score")),
        "least_fragile_layer": maybe_int(summary.get("least_fragile_layer")),
        "least_fragility_score": maybe_float(summary.get("least_fragility_score")),
        "mean_fragility_score": maybe_float(summary.get("mean_fragility_score")),
        "min_safe_lambda_proxy": maybe_float(summary.get("min_safe_lambda_proxy")),
        "median_safe_lambda_proxy": maybe_float(summary.get("median_safe_lambda_proxy")),
        "top_fragile_category": summary.get("top_fragile_category"),
        "top_category_fragility_score": maybe_float(summary.get("top_category_fragility_score")),
        "unsafe_readiness_rows": int(summary.get("unsafe_readiness_rows", 0)),
        "calibrate_readiness_rows": int(summary.get("calibrate_readiness_rows", 0)),
        "freeze_readiness_rows": int(summary.get("freeze_readiness_rows", 0)),
        "allowed_router_layer_count": int(summary.get("allowed_router_layer_count", 0)),
        "recommended_unified_router_action": summary.get("recommended_unified_router_action"),
        "top_layer": top_layer,
        "top_category": top_category,
        "layer_rows": [clean_row(row) for _, row in layers.head(16).iterrows()],
        "category_rows": [clean_row(row) for _, row in categories.iterrows()],
        "slice_rows": [clean_row(row) for _, row in slices.head(16).iterrows()],
        "report": rel(root / "report.md"),
        "layer_fragility": rel(root / "layer_margin_fragility.csv"),
        "category_fragility": rel(root / "category_margin_fragility.csv"),
        "slice_fragility": rel(root / "slice_margin_fragility.csv"),
        "literature_sources": rel(root / "literature_sources.json"),
        "figure": rel(root / "router_margin_fragility.png"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_router_calibration_job() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_router_calibration_job")
    summary = read_json(root / "summary.json")
    source_plan = read_csv(root / "source_control_plan.csv")
    candidate_plan = read_csv(root / "candidate_plan.csv")
    stage_plan = read_csv(root / "stage_plan.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "student_exists": bool(summary.get("student_exists", False)),
        "teacher_exists": bool(summary.get("teacher_exists", False)),
        "tokenizer_exists": bool(summary.get("tokenizer_exists", False)),
        "prompts_exists": bool(summary.get("prompts_exists", False)),
        "local_gpu_status": summary.get("local_gpu_status"),
        "router_caps": summary.get("router_caps", []),
        "router_margin_safe_lambda_proxy": maybe_float(summary.get("router_margin_safe_lambda_proxy")),
        "router_margin_limit_with_tolerance": maybe_float(summary.get("router_margin_limit_with_tolerance")),
        "router_margin_profile_enabled": bool(summary.get("router_margin_profile_enabled", False)),
        "router_margin_profile_cap_rows": maybe_int(summary.get("router_margin_profile_cap_rows")),
        "router_margin_profile_min_cap": maybe_float(summary.get("router_margin_profile_min_cap")),
        "router_margin_profile_mean_cap": maybe_float(summary.get("router_margin_profile_mean_cap")),
        "router_margin_profile_max_cap": maybe_float(summary.get("router_margin_profile_max_cap")),
        "router_margin_profile_cap_table": summary.get("router_margin_profile_cap_table"),
        "router_margin_planned_pass_count": maybe_int(summary.get("router_margin_planned_pass_count")),
        "default_run_candidate_count": maybe_int(summary.get("default_run_candidate_count")),
        "task_manifest": summary.get("task_manifest"),
        "create_task_manifest_if_missing": bool(summary.get("create_task_manifest_if_missing", False)),
        "source_control_count": int(summary.get("source_control_count", len(source_plan))),
        "source_controls_ready": bool(summary.get("source_controls_ready", False)),
        "candidate_count": int(summary.get("candidate_count", len(candidate_plan))),
        "stage_count": int(summary.get("stage_count", len(stage_plan))),
        "mechanism": summary.get("mechanism"),
        "router_validation_gate": summary.get("router_validation_gate"),
        "source_rows": [clean_row(row) for _, row in source_plan.iterrows()],
        "candidate_rows": [clean_row(row) for _, row in candidate_plan.iterrows()],
        "stage_rows": [clean_row(row) for _, row in stage_plan.iterrows()],
        "report": rel(root / "report.md"),
        "source_control_plan": rel(root / "source_control_plan.csv"),
        "candidate_plan": rel(root / "candidate_plan.csv"),
        "stage_plan": rel(root / "stage_plan.csv"),
        "router_margin_profile_cap_table_path": rel(root / "router_margin_profile_caps.csv"),
        "run_script": rel(root / "run_router_calibration_job.sh"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_router_calibration_selection() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_router_calibration_selection")
    return summarize_qwen3_moe_router_calibration_selection_dir(root)


def summarize_qwen3_moe_router_calibration_selection_dir(root: str | Path) -> dict[str, Any]:
    root = repo_path(root)
    summary = read_json(root / "summary.json")
    table = read_csv(root / "selection_table.csv")
    selection = summary.get("current_selection", {})
    eligible = table[table["selection_eligible"].astype(bool)] if "selection_eligible" in table else pd.DataFrame()
    return {
        "summary": summary,
        "status": summary.get("status"),
        "selected_method": selection.get("selected_method"),
        "selection_reason": selection.get("reason"),
        "baseline_eval_completed": bool(selection.get("baseline_eval_completed", False)),
        "source_eval_required": bool(selection.get("source_eval_required", False)),
        "candidate_eval_completed": bool(selection.get("candidate_eval_completed", False)),
        "audit_completed": bool(selection.get("audit_completed", False)),
        "training_completed": bool(selection.get("training_completed", False)),
        "capacity_metrics_completed": bool(selection.get("capacity_metrics_completed", False)),
        "group_validation_completed": bool(selection.get("group_validation_completed", False)),
        "router_margin_gate_completed": bool(selection.get("router_margin_gate_completed", False)),
        "router_margin_gate_enabled": bool(selection.get("router_margin_gate_enabled", False)),
        "router_margin_min_safe_lambda_proxy": maybe_float(
            selection.get("router_margin_min_safe_lambda_proxy")
        ),
        "router_margin_limit_with_tolerance": maybe_float(
            selection.get("router_margin_limit_with_tolerance")
        ),
        "router_margin_high_fragility_layer_count": maybe_int(
            selection.get("router_margin_high_fragility_layer_count")
        ),
        "router_margin_layer_count": maybe_int(selection.get("router_margin_layer_count")),
        "router_margin_top_fragile_layer": maybe_int(selection.get("router_margin_top_fragile_layer")),
        "source_eval_completed": bool(selection.get("source_eval_completed", False)),
        "eligible_candidate_count": int(selection.get("eligible_candidate_count", len(eligible))),
        "candidate_count": int(selection.get("candidate_count", len(table))),
        "active_candidate_count": maybe_int(selection.get("active_candidate_count")),
        "plan_pruned_candidate_count": maybe_int(selection.get("plan_pruned_candidate_count")),
        "best_available_score": None if table.empty else maybe_float(table["selection_score"].max()),
        "candidate_rows": [clean_row(row) for _, row in table.iterrows()],
        "report": rel(root / "report.md"),
        "selection_table": rel(root / "selection_table.csv"),
        "decision_rules": rel(root / "decision_rules.json"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_router_calibration_row_validation_negative_smoke() -> dict[str, Any]:
    return summarize_qwen3_moe_router_calibration_selection_dir(
        "results/qwen3_moe_router_calibration_selection_row_validation_negative_smoke"
    )


def summarize_qwen3_moe_router_calibration_source_dominance_negative_smoke() -> dict[str, Any]:
    return summarize_qwen3_moe_router_calibration_selection_dir(
        "results/qwen3_moe_router_calibration_selection_source_dominance_negative_smoke"
    )


def summarize_qwen3_moe_router_calibration_no_gain_negative_smoke() -> dict[str, Any]:
    return summarize_qwen3_moe_router_calibration_selection_dir(
        "results/qwen3_moe_router_calibration_selection_no_gain_negative_smoke"
    )


def summarize_qwen3_moe_router_calibration_task_regression_negative_smoke() -> dict[str, Any]:
    return summarize_qwen3_moe_router_calibration_selection_dir(
        "results/qwen3_moe_router_calibration_selection_task_regression_negative_smoke"
    )


def summarize_qwen3_moe_router_calibration_selector_matrix_smoke() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_router_calibration_selector_matrix_smoke")
    summary = read_json(root / "summary.json")
    matrix = read_csv(root / "selector_matrix.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "case_count": int(summary.get("case_count", len(matrix))),
        "passed_case_count": int(summary.get("passed_case_count", 0)),
        "failed_case_count": int(summary.get("failed_case_count", 0)),
        "case_rows": [clean_row(row) for _, row in matrix.iterrows()],
        "report": rel(root / "report.md"),
        "matrix": rel(root / "selector_matrix.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_trust_region_cap_search() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_trust_region_cap_search")
    summary = read_json(root / "summary.json")
    selected = read_csv(root / "selected_cap_laws.csv")
    ablation = read_csv(root / "risk_flag_ablation.csv")
    artifacts = read_json(root / "selected_rule_artifacts.json")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "expert_group_count": int(summary.get("expert_group_count", 0)),
        "searched_law_count": int(summary.get("searched_law_count", 0)),
        "pareto_frontier_count": int(summary.get("pareto_frontier_count", 0)),
        "selected_no_gt075_law": summary.get("selected_no_gt075_law"),
        "selected_no_gt065_law": summary.get("selected_no_gt065_law"),
        "current_trust_retention": maybe_float(summary.get("current_trust_retention")),
        "uniform_065_retention": maybe_float(summary.get("uniform_065_retention")),
        "uniform_065_retention_delta_vs_current_trust": maybe_float(
            summary.get("uniform_065_retention_delta_vs_current_trust")
        ),
        "current_trust_routed_gt_065_groups": int(summary.get("current_trust_routed_gt_065_groups", 0)),
        "uniform_065_routed_gt_065_groups": int(summary.get("uniform_065_routed_gt_065_groups", 0)),
        "current_extra_risk_penalties_delta_threshold_efficient": bool(
            summary.get("current_extra_risk_penalties_delta_threshold_efficient", False)
        ),
        "dry_run_validated_rule_count": sum(1 for row in artifacts if row.get("dry_run_validated")),
        "max_dry_run_expert_rule_hits": max((int(row.get("dry_run_expert_rule_hits", 0)) for row in artifacts), default=0),
        "max_dry_run_freeze_router_hits": max((int(row.get("dry_run_freeze_router_hits", 0)) for row in artifacts), default=0),
        "selected_rows": [clean_row(row) for _, row in selected.iterrows()],
        "risk_flag_ablation_rows": [clean_row(row) for _, row in ablation.iterrows()],
        "report": rel(root / "report.md"),
        "cap_law_search": rel(root / "cap_law_search.csv"),
        "pareto_frontier": rel(root / "pareto_frontier.csv"),
        "selected_cap_laws": rel(root / "selected_cap_laws.csv"),
        "risk_flag_ablation": rel(root / "risk_flag_ablation.csv"),
        "selected_rule_artifacts": rel(root / "selected_rule_artifacts.json"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_unified_mechanism_candidate() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_unified_mechanism_candidate")
    summary = read_json(root / "summary.json")
    search = read_csv(root / "candidate_search.csv")
    group_rules = read_csv(root / "unified_group_rules.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "expert_group_count": int(summary.get("expert_group_count", len(group_rules))),
        "candidate_count": int(summary.get("candidate_count", len(search))),
        "selected_candidate_id": summary.get("selected_candidate_id"),
        "selected_candidate_family": summary.get("selected_candidate_family"),
        "selected_nonbase_mass_retention": maybe_float(summary.get("selected_nonbase_mass_retention")),
        "selected_delta_norm_proxy_ratio_vs_uncapped": maybe_float(
            summary.get("selected_delta_norm_proxy_ratio_vs_uncapped")
        ),
        "selected_max_predicted_relative_delta": maybe_float(summary.get("selected_max_predicted_relative_delta")),
        "selected_risk_weighted_predicted_relative_delta": maybe_float(
            summary.get("selected_risk_weighted_predicted_relative_delta")
        ),
        "selected_geometry_weighted_predicted_relative_delta": maybe_float(
            summary.get("selected_geometry_weighted_predicted_relative_delta")
        ),
        "selected_subspace_weighted_predicted_relative_delta": maybe_float(
            summary.get("selected_subspace_weighted_predicted_relative_delta")
        ),
        "selected_route_geometry_risk_weighted_coder_retention": maybe_float(
            summary.get("selected_route_geometry_risk_weighted_coder_retention")
        ),
        "selected_subspace_risk_weighted_coder_retention": maybe_float(
            summary.get("selected_subspace_risk_weighted_coder_retention")
        ),
        "selected_high_geometry_mean_scale": maybe_float(summary.get("selected_high_geometry_mean_scale")),
        "selected_high_subspace_mean_scale": maybe_float(summary.get("selected_high_subspace_mean_scale")),
        "selected_routed_gt_hard_cap_groups": int(summary.get("selected_routed_gt_hard_cap_groups", 0)),
        "selected_routed_gt_065_groups": int(summary.get("selected_routed_gt_065_groups", 0)),
        "selected_routed_gt_075_groups": int(summary.get("selected_routed_gt_075_groups", 0)),
        "selected_scaled_group_count": int(summary.get("selected_scaled_group_count", 0)),
        "expert_geometry_probe_used": bool(summary.get("expert_geometry_probe_used", False)),
        "subspace_conflict_probe_used": bool(summary.get("subspace_conflict_probe_used", False)),
        "layer_coefficients_used": bool(summary.get("layer_coefficients_used", False)),
        "matches_validated_reference_rules": bool(summary.get("matches_validated_reference_rules", False)),
        "materialized_checkpoint_rule_status": summary.get("materialized_checkpoint_rule_status"),
        "matches_materialized_checkpoint_manifest": bool(
            summary.get("matches_materialized_checkpoint_manifest", False)
        ),
        "max_materialized_checkpoint_weight_abs_diff": maybe_float(
            summary.get("max_materialized_checkpoint_weight_abs_diff")
        ),
        "candidate_rule_count": int(summary.get("candidate_rule_count", 0)),
        "reference_rule_count": int(summary.get("reference_rule_count", 0)),
        "max_reference_weight_abs_diff": maybe_float(summary.get("max_reference_weight_abs_diff")),
        "router_policy": summary.get("router_policy"),
        "shared_attention_policy": summary.get("shared_attention_policy"),
        "mechanism_features": summary.get("mechanism_features", {}),
        "candidate_rows": [clean_row(row) for _, row in search.iterrows()],
        "report": rel(root / "report.md"),
        "candidate_search": rel(root / "candidate_search.csv"),
        "unified_group_rules": rel(root / "unified_group_rules.csv"),
        "tensor_rules": rel(root / "tensor_rules.txt"),
        "writer_command": rel(root / "writer_command.txt"),
        "dry_run_command": rel(root / "dry_run_command.txt"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_trust_region_delta_validation() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_trust_region_delta_validation")
    summary = read_json(root / "summary.json")
    group_summary = read_csv(root / "group_prediction_error_summary.csv")
    action_summary = read_csv(root / "action_prediction_error_summary.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "tensor_count": int(summary.get("tensor_count", 0)),
        "max_abs_relative_error": maybe_float(summary.get("max_abs_relative_error")),
        "p99_abs_relative_error": maybe_float(summary.get("p99_abs_relative_error")),
        "mean_abs_relative_error": maybe_float(summary.get("mean_abs_relative_error")),
        "tensors_above_relative_tolerance": int(summary.get("tensors_above_relative_tolerance", 0)),
        "routed_actual_relative_delta_gt_075": int(summary.get("routed_actual_relative_delta_gt_075", 0)),
        "routed_predicted_relative_delta_gt_075": int(summary.get("routed_predicted_relative_delta_gt_075", 0)),
        "routed_above_075_rounding_slop": int(summary.get("routed_above_075_rounding_slop", 0)),
        "routed_max_actual_relative_delta_norm": maybe_float(summary.get("routed_max_actual_relative_delta_norm")),
        "routed_max_predicted_relative_delta_norm": maybe_float(summary.get("routed_max_predicted_relative_delta_norm")),
        "scaled_tensor_count": int(summary.get("scaled_tensor_count", 0)),
        "group_rows": [clean_row(row) for _, row in group_summary.iterrows()],
        "action_rows": [clean_row(row) for _, row in action_summary.iterrows()],
        "report": rel(root / "report.md"),
        "tensor_prediction_error": rel(root / "tensor_prediction_error.csv"),
        "threshold_residuals": rel(root / "threshold_residuals.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_audit_gated_candidate() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_audit_gated_candidate")
    summary = read_json(root / "summary.json")
    weights = read_csv(root / "audit_gated_source_weights_by_expert.csv")
    manifest_path = repo_path("results/checkpoints/qwen3_moe_audit_gated_candidate/merge_manifest.json")
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    rule_counts = manifest.get("rule_counts", {})
    expert_tensor_rule_hits = sum(
        int(count) for key, count in rule_counts.items() if str(key).startswith("tensor_rule:.*layers\\.")
    )
    return {
        "summary": summary,
        "status": summary.get("status"),
        "expert_rule_count": int(summary.get("expert_rule_count", len(weights))),
        "scaled_expert_rule_count": int(summary.get("scaled_expert_rule_count", 0)),
        "frozen_by_audit_cap_count": int(summary.get("frozen_by_audit_cap_count", 0)),
        "mean_original_effective_nonbase_weight": maybe_float(summary.get("mean_original_effective_nonbase_weight")),
        "mean_effective_nonbase_weight": maybe_float(summary.get("mean_effective_nonbase_weight")),
        "max_audit_relative_delta_before_cap": maybe_float(summary.get("max_audit_relative_delta_before_cap")),
        "min_audit_delta_scale": maybe_float(summary.get("min_audit_delta_scale")),
        "action_counts": summary.get("action_counts", {}),
        "writer_dry_run_passed": bool(manifest.get("dry_run", False)),
        "writer_dry_run_floating_tensors": int(manifest.get("floating_tensors", 0)),
        "writer_dry_run_frozen_tensors": int(manifest.get("frozen_tensors", 0)),
        "writer_dry_run_expert_tensor_rule_hits": expert_tensor_rule_hits,
        "writer_dry_run_shared_attention_hits": int(rule_counts.get("tensor_rule:.*self_attn.*", 0)),
        "writer_dry_run_freeze_router_hits": int(rule_counts.get("freeze_router", 0)),
        "report": rel(root / "report.md"),
        "audit_gated_source_weights": rel(root / "audit_gated_source_weights_by_expert.csv"),
        "tensor_rules": rel(root / "tensor_rules.txt"),
        "writer_command": rel(root / "writer_command.txt"),
        "manifest": rel(manifest_path) if manifest_path.exists() else None,
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_trust_region_candidate() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_trust_region_candidate")
    summary = read_json(root / "summary.json")
    weights = read_csv(root / "trust_region_source_weights_by_expert.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "expert_rule_count": int(summary.get("expert_rule_count", len(weights))),
        "scaled_expert_rule_count": int(summary.get("scaled_expert_rule_count", 0)),
        "scaled_beyond_delta_cap_count": int(summary.get("scaled_beyond_delta_cap_count", 0)),
        "mean_original_effective_nonbase_weight": maybe_float(summary.get("mean_original_effective_nonbase_weight")),
        "mean_effective_nonbase_weight": maybe_float(summary.get("mean_effective_nonbase_weight")),
        "max_route_audit_relative_delta_before_scale": maybe_float(
            summary.get("max_route_audit_relative_delta_before_scale")
        ),
        "max_expected_relative_delta_after_scale": maybe_float(summary.get("max_expected_relative_delta_after_scale")),
        "min_trust_delta_scale": maybe_float(summary.get("min_trust_delta_scale")),
        "min_trust_target_relative_delta": maybe_float(summary.get("min_trust_target_relative_delta")),
        "estimated_relative_delta_norm": maybe_float(summary.get("estimated_relative_delta_norm")),
        "estimated_routed_tensor_max_relative_delta": maybe_float(
            summary.get("estimated_routed_tensor_max_relative_delta")
        ),
        "estimated_routed_tensors_relative_delta_gt_1": int(
            summary.get("estimated_routed_tensors_relative_delta_gt_1", 0)
        ),
        "estimated_routed_tensors_relative_delta_gt_075": int(
            summary.get("estimated_routed_tensors_relative_delta_gt_075", 0)
        ),
        "estimated_routed_tensors_relative_delta_gt_065": int(
            summary.get("estimated_routed_tensors_relative_delta_gt_065", 0)
        ),
        "risk_flag_counts": summary.get("risk_flag_counts", {}),
        "action_counts": summary.get("action_counts", {}),
        "writer_dry_run_validated": bool(summary.get("writer_dry_run_validated", False)),
        "writer_dry_run_floating_tensors": int(summary.get("writer_dry_run_floating_tensors", 0)),
        "writer_dry_run_frozen_tensors": int(summary.get("writer_dry_run_frozen_tensors", 0)),
        "writer_dry_run_expert_tensor_rule_hits": int(summary.get("writer_dry_run_expert_tensor_rule_hits", 0)),
        "writer_dry_run_shared_attention_hits": int(summary.get("writer_dry_run_shared_attention_hits", 0)),
        "writer_dry_run_freeze_router_hits": int(summary.get("writer_dry_run_freeze_router_hits", 0)),
        "writer_checkpoint_materialized": bool(summary.get("writer_checkpoint_materialized", False)),
        "writer_materialized_floating_tensors": int(summary.get("writer_materialized_floating_tensors", 0)),
        "writer_materialized_frozen_tensors": int(summary.get("writer_materialized_frozen_tensors", 0)),
        "writer_materialized_shards": int(summary.get("writer_materialized_shards", 0)),
        "report": rel(root / "report.md"),
        "trust_region_source_weights": rel(root / "trust_region_source_weights_by_expert.csv"),
        "estimated_tensor_delta": rel(root / "estimated_tensor_delta.csv"),
        "tensor_rules": rel(root / "tensor_rules.txt"),
        "writer_command": rel(root / "writer_command.txt"),
        "dry_run_command": rel(root / "dry_run_command.txt"),
        "dry_run_manifest": rel(root / "dry_run" / "merge_manifest.json"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_qwen3_moe_expert_only_trust_region_candidate() -> dict[str, Any]:
    root = repo_path("results/qwen3_moe_expert_only_trust_region_candidate")
    summary = read_json(root / "summary.json")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "expert_rule_count": int(summary.get("expert_rule_count", 0)),
        "attention_rule": summary.get("attention_rule"),
        "current_trust_region_relative_delta_norm": maybe_float(
            summary.get("current_trust_region_relative_delta_norm")
        ),
        "estimated_expert_only_relative_delta_norm": maybe_float(
            summary.get("estimated_expert_only_relative_delta_norm")
        ),
        "estimated_relative_delta_norm_reduction": maybe_float(
            summary.get("estimated_relative_delta_norm_reduction")
        ),
        "attention_delta_energy_fraction": maybe_float(summary.get("attention_delta_energy_fraction")),
        "attention_relative_delta_norm": maybe_float(summary.get("attention_relative_delta_norm")),
        "attention_changed_tensors_removed": int(summary.get("attention_changed_tensors_removed", 0)),
        "writer_dry_run_validated": bool(summary.get("writer_dry_run_validated", False)),
        "writer_dry_run_floating_tensors": int(summary.get("writer_dry_run_floating_tensors", 0)),
        "writer_dry_run_frozen_tensors": int(summary.get("writer_dry_run_frozen_tensors", 0)),
        "writer_dry_run_expert_tensor_rule_hits": int(
            summary.get("writer_dry_run_expert_tensor_rule_hits", 0)
        ),
        "writer_dry_run_shared_attention_hits": int(
            summary.get("writer_dry_run_shared_attention_hits", 0)
        ),
        "writer_dry_run_freeze_router_hits": int(summary.get("writer_dry_run_freeze_router_hits", 0)),
        "writer_dry_run_shards": int(summary.get("writer_dry_run_shards", 0)),
        "writer_checkpoint_materialized": bool(summary.get("writer_checkpoint_materialized", False)),
        "writer_materialized_floating_tensors": int(summary.get("writer_materialized_floating_tensors", 0)),
        "writer_materialized_frozen_tensors": int(summary.get("writer_materialized_frozen_tensors", 0)),
        "writer_materialized_shards": int(summary.get("writer_materialized_shards", 0)),
        "report": rel(root / "report.md"),
        "attention_kind_summary": rel(root / "attention_kind_summary.csv"),
        "attention_layer_summary": rel(root / "attention_layer_summary.csv"),
        "tensor_rules": rel(root / "tensor_rules.txt"),
        "writer_command": rel(root / "writer_command.txt"),
        "dry_run_command": rel(root / "dry_run_command.txt"),
        "dry_run_manifest": rel(root / "dry_run" / "merge_manifest.json"),
        "materialized_manifest": rel(repo_path("results/checkpoints/qwen3_moe_expert_only_trust_region_candidate/merge_manifest.json")),
        "summary_path": rel(root / "summary.json"),
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


def summarize_moe_packed_route_weight_recipe_smoke() -> dict[str, Any]:
    root = repo_path("results/moe_packed_route_weight_recipe_smoke")
    summary = read_json(root / "summary.json")
    packed_rules = read_csv(root / "packed_expert_rules.csv")
    source_weights = read_csv(root / "source_weights_by_expert.csv")
    writer_command = (root / "writer_command.txt").read_text(encoding="utf-8")
    return {
        "summary": summary,
        "recipe_status": summary.get("recipe_status"),
        "packed_expert_rules_enabled": bool(summary.get("packed_expert_rules_enabled")),
        "packed_expert_rule_count": int(summary.get("packed_expert_rule_count", len(packed_rules))),
        "packed_expert_rule_tensor_count": int(summary.get("packed_expert_rule_tensor_count", 0)),
        "packed_expert_rule_slice_count": int(summary.get("packed_expert_rule_slice_count", 0)),
        "source_weight_rows": int(len(source_weights)),
        "topology_model": (summary.get("topology") or {}).get("name"),
        "writer_command_has_packed_rule": "--packed-expert-rule-csv" in writer_command,
        "report": rel(root / "report.md"),
        "source_weights": rel(root / "source_weights_by_expert.csv"),
        "packed_expert_rules": rel(root / "packed_expert_rules.csv"),
        "writer_command": rel(root / "writer_command.txt"),
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


def summarize_fp_moe_mechanism() -> dict[str, Any]:
    root = repo_path("results/fp_moe_mechanism")
    summary = read_json(root / "summary.json")
    methods = read_csv(root / "method_metrics.csv")
    mechanisms = read_csv(root / "mechanism_deltas.csv")

    mechanism_by_name = {
        str(row["mechanism"]): clean_row(row) for _, row in mechanisms.iterrows()
    }
    return {
        "summary": summary,
        "best_merge_method": summary.get("best_merge_method"),
        "best_overall_method": summary.get("best_overall_method"),
        "gauge_perm_applied_to_B": summary.get("gauge_perm_applied_to_B"),
        "gauge_equivalence_mse": maybe_float(summary.get("gauge_equivalence_mse")),
        "hungarian_perm": summary.get("hungarian_perm"),
        "router_agreement_raw": maybe_float(summary.get("router_agreement_raw")),
        "router_agreement_aligned": maybe_float(summary.get("router_agreement_aligned")),
        "uniform_same_name": find_method(methods, "uniform_same_name"),
        "uniform_same_name_routercal": find_method(methods, "uniform_same_name_routercal"),
        "uniform_aligned": find_method(methods, "uniform_aligned"),
        "uniform_aligned_routercal": find_method(methods, "uniform_aligned_routercal"),
        "fisher_aligned": find_method(methods, "fisher_aligned"),
        "fisher_aligned_routercal": find_method(methods, "fisher_aligned_routercal"),
        "base": find_method(methods, "base"),
        "expert_identity_alignment": mechanism_by_name.get("expert_identity_alignment", {}),
        "router_calibration_after_alignment": mechanism_by_name.get("router_calibration_after_alignment", {}),
        "route_conditioned_fisher": mechanism_by_name.get("route_conditioned_fisher", {}),
        "router_cannot_fix_misaligned_experts": mechanism_by_name.get("router_cannot_fix_misaligned_experts", {}),
        "mechanism_rows": [clean_row(row) for _, row in mechanisms.iterrows()],
        "report": rel(root / "report.md"),
        "method_metrics": rel(root / "method_metrics.csv"),
        "mechanism_deltas": rel(root / "mechanism_deltas.csv"),
        "figure": rel(root / "moe_mechanism.png"),
    }


def summarize_fp_moe_real_probe() -> dict[str, Any]:
    root = repo_path("results/fp_moe_real_probe")
    summary = read_json(root / "summary.json")
    qwen_cross_path = root / "qwen3_instruct_coder" / "cross_correspondence.json"
    qwen_cross = read_json(qwen_cross_path) if qwen_cross_path.exists() else None
    return {
        "summary": summary,
        "model": summary.get("model"),
        "n_moe_layers": int(summary.get("n_moe_layers", 0)),
        "moe_format": summary.get("moe_format"),
        "num_experts": maybe_int(summary.get("num_experts")),
        "baseline_nll": maybe_float(summary.get("baseline_nll")),
        "gauge_permuted_nll": maybe_float(summary.get("gauge_permuted_nll")),
        "naive_same_name_average_nll": maybe_float(summary.get("naive_sameNAME_average_nll")),
        "aligned_average_nll": maybe_float(summary.get("aligned_average_nll")),
        "naive_degradation_vs_baseline": maybe_float(summary.get("naive_degradation_vs_baseline")),
        "aligned_degradation_vs_baseline": maybe_float(summary.get("aligned_degradation_vs_baseline")),
        "layers_perm_recovered": maybe_int(summary.get("layers_perm_recovered")),
        "qwen3_cross": qwen_cross,
        "qwen3_frac_layers_identity_optimal": None
        if qwen_cross is None
        else maybe_float(qwen_cross.get("frac_layers_identity_optimal")),
        "qwen3_mean_argmax_is_identity_frac": None
        if qwen_cross is None
        else maybe_float(qwen_cross.get("mean_argmax_is_identity_frac")),
        "qwen3_mean_diag_cos": None if qwen_cross is None else maybe_float(qwen_cross.get("mean_diag_cos")),
        "qwen3_mean_matched_cos": None if qwen_cross is None else maybe_float(qwen_cross.get("mean_matched_cos")),
        "report": rel(root / "report.md"),
        "summary_path": rel(root / "summary.json"),
        "gauge_selfmerge": rel(root / "gauge_selfmerge.json"),
        "qwen3_cross_correspondence": rel(qwen_cross_path) if qwen_cross_path.exists() else None,
    }


def summarize_moe_probe_gated_selector() -> dict[str, Any]:
    root = repo_path("results/moe_probe_gated_selector")
    summary = read_json(root / "summary.json")
    cases = read_csv(root / "selector_cases.csv")
    stages = read_csv(root / "selector_stages.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "global_moe_gauge_decision": summary.get("global_moe_gauge_decision"),
        "qwen3_expert_identity_decision": summary.get("qwen3_expert_identity_decision"),
        "qwen3_preflight_decision": summary.get("qwen3_preflight_decision"),
        "qwen3_routing_decision": summary.get("qwen3_routing_decision"),
        "next_blocking_probe": summary.get("next_blocking_probe"),
        "real_gauge_naive_degradation": maybe_float(summary.get("real_gauge_naive_degradation")),
        "real_gauge_aligned_degradation": maybe_float(summary.get("real_gauge_aligned_degradation")),
        "qwen3_identity_fraction": maybe_float(summary.get("qwen3_identity_fraction")),
        "qwen3_argmax_identity_fraction": maybe_float(summary.get("qwen3_argmax_identity_fraction")),
        "toy_soft_recommendation": summary.get("toy_soft_recommendation"),
        "toy_sparse_recommendation": summary.get("toy_sparse_recommendation"),
        "toy_capacity_recommendation": summary.get("toy_capacity_recommendation"),
        "case_rows": [clean_row(row) for _, row in cases.iterrows()],
        "stage_rows": [clean_row(row) for _, row in stages.iterrows()],
        "report": rel(root / "report.md"),
        "summary_path": rel(root / "summary.json"),
        "case_table": rel(root / "selector_cases.csv"),
        "stage_table": rel(root / "selector_stages.csv"),
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
    unified = plan[plan["method"] == "qwen3_moe_unified_mechanism_candidate"]
    unified_row = {} if unified.empty else unified.iloc[0].to_dict()
    return {
        "summary": summary,
        "status": summary.get("status"),
        "candidate_count": int(summary.get("candidate_count", len(plan))),
        "ready_to_host_count": int(summary.get("ready_to_host_count", 0)),
        "completed_eval_count": int(summary.get("completed_eval_count", 0)),
        "missing_checkpoint_count": int(summary.get("missing_checkpoint_count", 0)),
        "not_vllm_loadable_count": int(summary.get("not_vllm_loadable_count", 0)),
        "tasks": summary.get("tasks"),
        "unified_candidate_serve_status": unified_row.get("serve_status"),
        "unified_candidate_eval_status": unified_row.get("eval_status"),
        "unified_candidate_eval_output_dir": unified_row.get("eval_output_dir"),
        "unified_candidate_checkpoint_path": unified_row.get("checkpoint_path"),
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


def summarize_qwen_dense_sparse_method_candidate(output_dir: str = "results/qwen_dense_sparse_method_candidate") -> dict[str, Any]:
    summary = read_json(f"{output_dir}/summary.json")
    selected = read_csv(f"{output_dir}/selected_tensors.csv")
    manifest = read_json(f"{output_dir}/dry_run/merge_manifest.json")
    method_counts = summary.get("dry_run_method_counts") or manifest.get("method_counts", {})
    sparse_applied = summary.get("dry_run_tensor_method_applied_count")
    if sparse_applied is None:
        sparse_applied = sum(int(count) for method, count in method_counts.items() if str(method).startswith("tensor_method:"))
    return {
        "summary": summary,
        "status": summary.get("status"),
        "candidate_id": summary.get("candidate_id"),
        "base_bridge_candidate": summary.get("base_bridge_candidate"),
        "method": summary.get("method"),
        "density": maybe_float(summary.get("density")),
        "source_weights": summary.get("source_weights", {}),
        "selected_tensor_count": maybe_int(summary.get("selected_tensor_count")),
        "selected_numel": maybe_int(summary.get("selected_numel")),
        "selected_numel_fraction": maybe_float(summary.get("selected_numel_fraction")),
        "projection_counts": summary.get("projection_counts", {}),
        "dry_run_status": summary.get("dry_run_status"),
        "dry_run_linear_count": maybe_int(method_counts.get("linear")),
        "dry_run_tensor_method_applied_count": maybe_int(sparse_applied),
        "dry_run_floating_tensors": maybe_int(manifest.get("floating_tensors")),
        "vllm_eval": summary.get("vllm_eval"),
        "vllm_avg_primary_score": maybe_float((summary.get("vllm_eval") or {}).get("avg_primary_score")),
        "vllm_delta_vs_base_bridge_avg_primary": maybe_float(
            (summary.get("vllm_eval") or {}).get("delta_vs_base_bridge_avg_primary")
        ),
        "vllm_delta_vs_uniform_avg_primary": maybe_float(
            (summary.get("vllm_eval") or {}).get("delta_vs_uniform_avg_primary")
        ),
        "top_selected_tensors": [clean_row(row) for _, row in selected.head(12).iterrows()],
        "report": rel(f"{output_dir}/report.md"),
        "selected_tensors_path": rel(f"{output_dir}/selected_tensors.csv"),
        "tensor_method_rules": rel(f"{output_dir}/tensor_method_rules.txt"),
        "dry_run_manifest": rel(f"{output_dir}/dry_run/merge_manifest.json"),
        "writer_command": rel(f"{output_dir}/writer_command.txt"),
        "vllm_commands": rel(f"{output_dir}/vllm_commands.json"),
    }


def summarize_checkpoint_materialization_readiness() -> dict[str, Any]:
    summary = read_json("results/checkpoint_materialization_readiness/summary.json")
    readiness = read_csv("results/checkpoint_materialization_readiness/candidate_readiness.csv")
    unified = readiness[readiness["candidate"] == "qwen3_moe_unified_mechanism_candidate"]
    unified_row = {} if unified.empty else unified.iloc[0].to_dict()
    return {
        "summary": summary,
        "status": summary.get("status"),
        "candidate_count": int(summary.get("candidate_count", len(readiness))),
        "materialized_count": int(summary.get("materialized_count", 0)),
        "blocked_by_placeholder_count": int(summary.get("blocked_by_placeholder_count", 0)),
        "ready_for_vllm_eval_count": int(summary.get("ready_for_vllm_eval_count", 0)),
        "completed_vllm_eval_count": int(summary.get("completed_vllm_eval_count", 0)),
        "toy_validation_only_count": int(summary.get("toy_validation_only_count", 0)),
        "unified_candidate_writer_status": unified_row.get("writer_status"),
        "unified_candidate_vllm_plan_status": unified_row.get("vllm_plan_status"),
        "unified_candidate_end_to_end_status": unified_row.get("end_to_end_status"),
        "unified_candidate_eval_output_dir": unified_row.get("vllm_eval_output_dir"),
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


def summarize_probe_gated_unified_average_plan() -> dict[str, Any]:
    summary = read_json("results/probe_gated_unified_average_plan/summary.json")
    dense = read_csv("results/probe_gated_unified_average_plan/dense_mechanism_contrasts.csv")
    moe = read_csv("results/probe_gated_unified_average_plan/moe_mechanism_contrasts.csv")
    plan = read_csv("results/probe_gated_unified_average_plan/intervention_plan.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "dense_default_action": summary.get("dense_default_action"),
        "dense_bridge_avg_primary_delta_vs_uniform": float(summary.get("dense_bridge_avg_primary_delta_vs_uniform", 0.0)),
        "dense_module_guard_delta_vs_bridge": float(summary.get("dense_module_guard_delta_vs_bridge", 0.0)),
        "moe_default_action": summary.get("moe_default_action"),
        "moe_expert_identity_soft_worst_acc_gain": float(summary.get("moe_expert_identity_soft_worst_acc_gain", 0.0)),
        "moe_capacity_topk_overflow_delta": float(summary.get("moe_capacity_topk_overflow_delta", 0.0)),
        "real_qwen_moe_blocker": summary.get("real_qwen_moe_blocker"),
        "dense_contrasts": [clean_row(row) for _, row in dense.iterrows()],
        "moe_contrasts": [clean_row(row) for _, row in moe.iterrows()],
        "intervention_plan": [clean_row(row) for _, row in plan.iterrows()],
        "report": rel("results/probe_gated_unified_average_plan/report.md"),
        "dense_mechanism_contrasts": rel("results/probe_gated_unified_average_plan/dense_mechanism_contrasts.csv"),
        "moe_mechanism_contrasts": rel("results/probe_gated_unified_average_plan/moe_mechanism_contrasts.csv"),
        "intervention_plan_csv": rel("results/probe_gated_unified_average_plan/intervention_plan.csv"),
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


def summarize_average_method_gate_matrix() -> dict[str, Any]:
    root = repo_path("results/average_method_gate_matrix")
    summary = read_json(root / "summary.json")
    method_gates = read_csv(root / "method_gate_matrix.csv")
    probe_gates = read_csv(root / "probe_gate_matrix.csv")
    rejected_rows = method_gates[method_gates["current_gate"].astype(str) == "rejected_as_default"]
    active_rows = method_gates[method_gates["current_gate"].astype(str).str.startswith("active")]
    required_rows = method_gates[method_gates["current_gate"].astype(str).str.startswith("required")]
    return {
        "summary": summary,
        "status": summary.get("status"),
        "method_family_count": int(summary.get("method_family_count", len(method_gates))),
        "accepted_by_default_count": int(summary.get("accepted_by_default_count", 0)),
        "conditional_count": int(summary.get("conditional_count", 0)),
        "active_lever_count": int(summary.get("active_lever_count", len(active_rows))),
        "required_precondition_count": int(summary.get("required_precondition_count", len(required_rows))),
        "default_rejected_count": int(summary.get("default_rejected_count", len(rejected_rows))),
        "qwen3_final_selection_status": summary.get("qwen3_final_selection_status"),
        "qwen3_router_calibration_status": summary.get("qwen3_router_calibration_status"),
        "dense_lambda_linear_worst_nll": maybe_float(summary.get("dense_lambda_linear_worst_nll")),
        "dense_lambda_best_worst_nll": maybe_float(summary.get("dense_lambda_best_worst_nll")),
        "qwen3_interpolation_interior_gap_nll": maybe_float(
            summary.get("qwen3_interpolation_interior_gap_nll")
        ),
        "rejected_method_families": [str(row["method_family"]) for _, row in rejected_rows.iterrows()],
        "active_method_families": [str(row["method_family"]) for _, row in active_rows.iterrows()],
        "required_method_families": [str(row["method_family"]) for _, row in required_rows.iterrows()],
        "method_rows": [clean_row(row) for _, row in method_gates.iterrows()],
        "probe_rows": [clean_row(row) for _, row in probe_gates.iterrows()],
        "report": rel(root / "report.md"),
        "method_gate_matrix": rel(root / "method_gate_matrix.csv"),
        "probe_gate_matrix": rel(root / "probe_gate_matrix.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_average_trust_region_bounds() -> dict[str, Any]:
    root = repo_path("results/average_trust_region_bounds")
    summary = read_json(root / "summary.json")
    constraints = read_csv(root / "trust_region_constraints.csv")
    decisions = read_csv(root / "trust_region_decisions.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "constraint_count": int(summary.get("constraint_count", len(constraints))),
        "passed_count": int(summary.get("passed_count", 0)),
        "rejected_count": int(summary.get("rejected_count", 0)),
        "waiting_count": int(summary.get("waiting_count", 0)),
        "dense_local_task_vector_lambda_bound": maybe_float(
            summary.get("dense_local_task_vector_lambda_bound")
        ),
        "dense_safe_uniform_lambda": maybe_float(summary.get("dense_safe_uniform_lambda")),
        "dense_linear_candidate_over_safe_uniform_bound": maybe_float(
            summary.get("dense_linear_candidate_over_safe_uniform_bound")
        ),
        "moe_router_safe_lambda_proxy": maybe_float(summary.get("moe_router_safe_lambda_proxy")),
        "moe_direct_router_average_over_safe_bound": maybe_float(
            summary.get("moe_direct_router_average_over_safe_bound")
        ),
        "mechanistic_effective_expert_delta_cap": maybe_float(
            summary.get("mechanistic_effective_expert_delta_cap")
        ),
        "mechanistic_selected_max_predicted_relative_delta": maybe_float(
            summary.get("mechanistic_selected_max_predicted_relative_delta")
        ),
        "final_selection_status": summary.get("final_selection_status"),
        "constraint_rows": [clean_row(row) for _, row in constraints.iterrows()],
        "decision_rows": [clean_row(row) for _, row in decisions.iterrows()],
        "report": rel(root / "report.md"),
        "constraints": rel(root / "trust_region_constraints.csv"),
        "decisions": rel(root / "trust_region_decisions.csv"),
        "algorithm": rel(root / "algorithm.json"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_average_trust_region_bounds_smoke() -> dict[str, Any]:
    root = repo_path("results/average_trust_region_bounds_smoke")
    summary = read_json(root / "summary.json")
    matrix = read_csv(root / "trust_region_bounds_smoke_matrix.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "assertion_count": int(summary.get("assertion_count", len(matrix))),
        "passed_assertion_count": int(summary.get("passed_assertion_count", 0)),
        "failed_assertion_count": int(summary.get("failed_assertion_count", 0)),
        "assertion_rows": [clean_row(row) for _, row in matrix.iterrows()],
        "report": rel(root / "report.md"),
        "matrix": rel(root / "trust_region_bounds_smoke_matrix.csv"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_average_connectivity_diagnostic() -> dict[str, Any]:
    root = repo_path("results/average_connectivity_diagnostic")
    summary = read_json(root / "summary.json")
    diagnostics = read_csv(root / "path_diagnostics.csv")
    by_case = {str(row["case"]): clean_row(row) for _, row in diagnostics.iterrows()}
    dense_source = by_case.get("dense_instruct_coder_source_path", {})
    dense_lambda = by_case.get("dense_base_anchored_lambda_family", {})
    qwen3_source = by_case.get("qwen3_moe_instruct_coder_source_path", {})
    qwen3_base = by_case.get("qwen3_moe_base_coder_source_path", {})
    qwen3_complementary = by_case.get("qwen3_moe_thinking_coder_complementary_path", {})
    return {
        "summary": summary,
        "status": summary.get("status"),
        "case_count": int(summary.get("case_count", len(diagnostics))),
        "path_rejected_count": int(summary.get("path_rejected_count", 0)),
        "midpoint_rejected_count": int(summary.get("midpoint_rejected_count", 0)),
        "endpoint_frontier_win_count": int(summary.get("endpoint_frontier_win_count", 0)),
        "complementarity_observed_count": int(summary.get("complementarity_observed_count", 0)),
        "dense_source_midpoint_gap": maybe_float(summary.get("dense_source_midpoint_gap")),
        "dense_source_endpoint_gap": maybe_float(summary.get("dense_source_endpoint_gap")),
        "dense_lambda_endpoint_gap": maybe_float(dense_lambda.get("endpoint_frontier_gap")),
        "dense_lambda_midpoint_gap": maybe_float(dense_lambda.get("midpoint_gap")),
        "qwen3_instruct_coder_gap": maybe_float(summary.get("qwen3_instruct_coder_gap")),
        "qwen3_instruct_coder_midpoint_gap": maybe_float(
            summary.get("qwen3_instruct_coder_midpoint_gap")
        ),
        "qwen3_base_coder_gap": maybe_float(qwen3_base.get("endpoint_frontier_gap")),
        "qwen3_complementary_gap": maybe_float(qwen3_complementary.get("endpoint_frontier_gap")),
        "dense_source_decision": dense_source.get("decision"),
        "dense_lambda_decision": dense_lambda.get("decision"),
        "qwen3_source_decision": qwen3_source.get("decision"),
        "rows": [clean_row(row) for _, row in diagnostics.iterrows()],
        "report": rel(root / "report.md"),
        "path_diagnostics": rel(root / "path_diagnostics.csv"),
        "acceptance_rules": rel(root / "acceptance_rules.json"),
        "figure": rel(root / "connectivity_gaps.png"),
        "summary_path": rel(root / "summary.json"),
    }


def summarize_average_invariant_audit() -> dict[str, Any]:
    root = repo_path("results/average_invariant_audit")
    summary = read_json(root / "summary.json")
    invariants = read_csv(root / "invariant_table.csv")
    method_matrix = read_csv(root / "method_invariant_matrix.csv")
    return {
        "summary": summary,
        "status": summary.get("status"),
        "invariant_count": int(summary.get("invariant_count", len(invariants))),
        "method_family_count": int(summary.get("method_family_count", len(method_matrix))),
        "hard_gate_blocker_count": int(summary.get("hard_gate_blocker_count", 0)),
        "default_accepted_method_count": int(summary.get("default_accepted_method_count", 0)),
        "default_rejected_method_count": int(summary.get("default_rejected_method_count", 0)),
        "same_shape_contract_pass": bool(summary.get("same_shape_contract_pass", False)),
        "router_allowed_layers": maybe_int(summary.get("router_allowed_layers")),
        "router_layer_count": maybe_int(summary.get("router_layer_count")),
        "expert_identity_fraction": maybe_float(summary.get("expert_identity_fraction")),
        "high_route_geometry_risk_expert_count": maybe_int(
            summary.get("high_route_geometry_risk_expert_count")
        ),
        "selected_candidate_id": summary.get("selected_candidate_id"),
        "selected_nonbase_mass_retention": maybe_float(summary.get("selected_nonbase_mass_retention")),
        "selected_max_predicted_relative_delta": maybe_float(
            summary.get("selected_max_predicted_relative_delta")
        ),
        "final_selection_status": summary.get("final_selection_status"),
        "status_counts": summary.get("status_counts", {}),
        "invariant_rows": [clean_row(row) for _, row in invariants.iterrows()],
        "method_rows": [clean_row(row) for _, row in method_matrix.iterrows()],
        "report": rel(root / "report.md"),
        "invariants": rel(root / "invariant_table.csv"),
        "method_matrix": rel(root / "method_invariant_matrix.csv"),
        "algorithm_spec": rel(root / "algorithm_spec.json"),
        "literature_sources": rel(root / "literature_sources.json"),
        "figure": rel(root / "invariant_status.png"),
        "summary_path": rel(root / "summary.json"),
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
            "item": "Dense curvature-displacement mechanism probe",
            "status": "complete",
            "evidence": "results/fp_curvature_law/report.md compares diagonal-Fisher second-order midpoint predictions against real Qwen instruct/coder interpolation loss.",
        },
        {
            "item": "Unified merge-family selector",
            "status": "complete",
            "evidence": "results/fp_merge_compare_dense/report.md evaluates a finite family containing linear average, task arithmetic, sign-elect, and magnitude-weighted variants, then selects by held-out worst-task NLL.",
        },
        {
            "item": "Dense exact-answer generation smoke",
            "status": "complete",
            "evidence": "results/fp_gen_eval_dense/report.md evaluates base, endpoints, linear average, and unified lambda=0 on built-in math/code-output generation tasks without executing model-generated code.",
        },
        {
            "item": "Qwen3 MoE generation-level downstream matrix",
            "status": "complete",
            "evidence": "results/fp_downstream_matrix/report.md compares official Qwen3 MoE parents, naive averages, and router-calibrated averages on MMLU/GSM8K/HumanEval generation tasks; it is auxiliary evidence, not the final vLLM selector.",
        },
        {
            "item": "Qwen3 MoE generation-level mechanism attribution",
            "status": "complete",
            "evidence": "results/fp_downstream_attribution/report.md attributes the generation matrix into naive-average regression, router-calibration recovery, and remaining source-frontier gap by task.",
        },
        {
            "item": "Qwen3 MoE generation confidence audit",
            "status": "complete",
            "evidence": "results/fp_downstream_confidence_audit/report.md adds Wilson aggregate uncertainty bounds to the generation matrix and shows router calibration is directional but not yet a confident source-frontier win.",
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
            "item": "Qwen dense sparse-method candidate",
            "status": "complete",
            "evidence": "results/qwen_dense_sparse_method_candidate/report.md and results/qwen_dense_attention_sparse_method_candidate/report.md compare broad attention+MLP sparse rules against an attention-only sparse rule under real vLLM eval.",
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
            "item": "Probe-gated unified average plan",
            "status": "complete",
            "evidence": "results/probe_gated_unified_average_plan/report.md turns Dense vLLM ablations and toy MoE mechanism contrasts into a same-shape intervention gate rather than a static method ranking.",
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
            "item": "Average method gate matrix",
            "status": "complete",
            "evidence": "results/average_method_gate_matrix/report.md turns common Dense/MoE averaging method families into current-evidence accept/reject/conditional gates.",
        },
        {
            "item": "Average trust-region bounds",
            "status": "complete",
            "evidence": "results/average_trust_region_bounds/report.md converts Dense curvature failure, held-out lambda paths, MoE source-line barriers, router top-k margins, and routed expert caps into executable average-movement bounds.",
        },
        {
            "item": "Average connectivity diagnostic",
            "status": "complete",
            "evidence": "results/average_connectivity_diagnostic/report.md unifies Dense/MoE endpoint-frontier, midpoint, barrier, complementarity, and local-quadratic gates.",
        },
        {
            "item": "Average invariant audit",
            "status": "complete",
            "evidence": "results/average_invariant_audit/report.md converts model-averaging literature and current Dense/MoE probes into executable acceptance invariants and method gates.",
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
            "item": "Dense sparse-method writer smoke",
            "status": "complete",
            "evidence": "results/dense_sparse_method_writer_smoke/report.md verifies coordinate-wise TIES-style trim/sign-elect/merge inside the same-shape checkpoint writer.",
        },
        {
            "item": "MoE tensor-rule writer materialization",
            "status": "complete",
            "evidence": "results/moe_tensor_rule_writer_smoke/report.md writes a tiny MoE-like safetensors checkpoint and verifies tensor-rule, freeze-router, router-bias additive deltas, full-tensor router deltas, and non-floating tensor behavior numerically.",
        },
        {
            "item": "MoE router delta calibration smoke",
            "status": "complete",
            "evidence": "results/moe_router_delta_calibration_smoke/report.md trains a same-shape router safetensors delta from hidden/router-logit cache, improving route KL and top-1 agreement under global/per-router cap-table relative-norm caps.",
        },
        {
            "item": "MoE router calibration cache smoke",
            "status": "complete",
            "evidence": "results/moe_router_calibration_cache_smoke/report.md captures student router hidden states and teacher router logits from forward hooks, then verifies the cache by training a same-shape router delta.",
        },
        {
            "item": "MoE combined writer smoke",
            "status": "complete",
            "evidence": "results/moe_combined_writer_smoke/report.md verifies expert tensor rules, source expert alias remap, freeze-router, and router-bias additive deltas in one same-shape writer call.",
        },
        {
            "item": "MoE packed-expert writer smoke",
            "status": "complete",
            "evidence": "results/moe_packed_expert_writer_smoke/report.md verifies first-dimension packed expert slice weights and source-expert remaps for Qwen-style packed MoE tensors.",
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
            "item": "MoE packed route-weight recipe smoke",
            "status": "complete",
            "evidence": "results/moe_packed_route_weight_recipe_smoke/report.md verifies route/expert weights can emit Qwen-style packed_expert_rules.csv with source-expert remap columns.",
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
            "item": "First-principles MoE mechanism probe",
            "status": "complete",
            "evidence": "results/fp_moe_mechanism/report.md isolates function-preserving expert permutation, expert alignment, router calibration, and Fisher ablations with real forward/backward passes.",
        },
        {
            "item": "Real MoE expert-gauge self-merge probe",
            "status": "complete",
            "evidence": "results/fp_moe_real_probe/report.md runs a function-preserving expert/router permutation on a real packed OLMoE checkpoint and shows same-name averaging fails unless expert identity is recovered.",
        },
        {
            "item": "MoE probe-gated selector",
            "status": "complete",
            "evidence": "results/moe_probe_gated_selector/report.md combines real OLMoE gauge evidence, Qwen3 expert correspondence, and toy route/capacity selection into a same-shape MoE average gate.",
        },
        {
            "item": "Qwen3 MoE unified average preflight",
            "status": "complete",
            "evidence": "results/moe_unified_preflight_qwen3_30b/report.md verifies Qwen3-30B Instruct/Coder same-shape config, router tensor contract, routed expert layout, expert identity gate, and the emitted real routing probe command.",
        },
        {
            "item": "Qwen3 MoE real routing readiness",
            "status": "complete",
            "evidence": "results/moe_routing_readiness/qwen3_30b_instruct_vs_coder/report.md analyzes the real Qwen3-30B Instruct/Coder route overlap and expert load probe, showing direct router averaging is high risk and needs calibration or freeze.",
        },
        {
            "item": "Qwen3 MoE route-guarded unified candidate",
            "status": "complete",
            "evidence": "results/qwen3_moe_unified_route_guarded_candidate/report.md converts the real Qwen3 route/load probe into source-route-conditioned same-shape tensor rules and a validated writer dry-run command.",
        },
        {
            "item": "Qwen3 MoE mechanism-gated vLLM eval gate",
            "status": "complete",
            "evidence": "results/qwen3_moe_mechanism_eval_gate/report.md turns two source endpoints and all registered same-shape Qwen3 MoE candidates into mechanism tests, a one-model-at-a-time vLLM run script, and endpoint-fallback selection rules.",
        },
        {
            "item": "Qwen3 MoE statistically powered vLLM eval budget",
            "status": "complete",
            "evidence": "results/qwen3_moe_eval_budget_plan/report.md raises the Qwen3 source/candidate vLLM run from a 64-example smoke floor to a Wilson/paired-test budgeted eval script; results/qwen3_moe_eval_budget_queue_smoke/report.md verifies the default final queue excludes ablation-only candidates.",
        },
        {
            "item": "Qwen3 MoE adaptive vLLM eval scheduler",
            "status": "complete",
            "evidence": "results/qwen3_moe_adaptive_eval_schedule/report.md turns the fixed Qwen3 MoE budget into a sequential source-control, mechanism-targeted probe-task, and full-budget escalation schedule; results/qwen3_moe_adaptive_eval_schedule_smoke/report.md covers source-missing, probe-selected, promising-escalation, full-ready, dominated-prune, coverage-selection, and task-selection branches.",
        },
        {
            "item": "Qwen3 MoE eval task manifest preflight",
            "status": "complete",
            "evidence": "results/qwen3_moe_eval_manifest_preflight/report.md checks that all budgeted source/candidate evals share one canonical task manifest and that the manifest contains the required task/example keys before vLLM runs.",
        },
        {
            "item": "Qwen3 MoE mechanism leverage map",
            "status": "complete",
            "evidence": "results/qwen3_moe_mechanism_levers/report.md ranks MoE-specific failure mechanisms, next experiments, and importance-guided layer/chunk calibration slots from real Qwen3 probes, including expert geometry and subspace conflict probes.",
        },
        {
            "item": "Qwen3 MoE expert geometry probe",
            "status": "complete",
            "evidence": "results/qwen3_moe_expert_geometry_probe/report.md reads 18,432 routed expert tensors from real Qwen3 Instruct/Coder safetensors and joins internal geometry risk with route/load context.",
        },
        {
            "item": "Qwen3 MoE expert subspace conflict probe",
            "status": "complete",
            "evidence": "results/qwen3_moe_expert_subspace_conflict_probe/report.md converts real expert channel/chunk geometry into subspace conflict gates and a candidate scale plan for uncovered high-risk experts.",
        },
        {
            "item": "Qwen3 MoE layer/chunk coefficient candidate",
            "status": "complete",
            "evidence": "results/qwen3_moe_layer_chunk_candidate/report.md converts the mechanism leverage layer scores into writer-ready same-shape tensor rules; results/qwen3_moe_layer_chunk_delta_audit/report.md verifies the materialized same-shape checkpoint.",
        },
        {
            "item": "Qwen3 MoE unified downstream result selector",
            "status": "complete",
            "evidence": "results/qwen3_moe_unified_result_selection/report.md gates the unified same-shape average against both Qwen3 source endpoints after matched vLLM eval; results/qwen3_moe_unified_result_selection_smoke/report.md covers candidate-win, source-dominance, task-regression, and no-gain branches.",
        },
        {
            "item": "Qwen3 MoE final candidate selector",
            "status": "complete",
            "evidence": "results/qwen3_moe_final_candidate_selection/report.md ranks all registered same-shape Qwen3 MoE candidates against both source endpoints after eval-bundle audit, with source-dominance, task-regression, score-confidence, paired-prediction, checkpoint-audit, and provisional-selection gates.",
        },
        {
            "item": "Qwen3 MoE candidate trust-region gate",
            "status": "complete",
            "evidence": "results/qwen3_moe_candidate_trust_region_gate/report.md marks old high-risk candidates as ablation-only and exposes only strict routed-expert trust-region candidates to final default selection.",
        },
        {
            "item": "Unified Dense/MoE average optimizer",
            "status": "complete",
            "evidence": "results/unified_average_optimizer/report.md converts Dense barrier probes, Dense/Qwen3 MoE straight-line connectivity, MoE gauge probes, Qwen3 expert identity, router movement, router margin fragility, router-only NLL calibration evidence, unified mechanism caps, router-calibration gating, and final candidate-selection gates into one same-shape operation policy.",
        },
        {
            "item": "Qwen3 MoE vLLM eval bundle audit",
            "status": "complete",
            "evidence": "results/qwen3_moe_eval_bundle_audit/report.md checks every Qwen3 source/candidate eval output for model-id, task-manifest sha, task, example-count, prediction, primary-score, and paired prediction-key consistency before selector use; results/qwen3_moe_eval_bundle_audit_smoke/report.md covers valid, stale-model, missing-task, low-example, key-mismatch, and manifest-mismatch bundles.",
        },
        {
            "item": "Qwen3 MoE mechanism effect attribution",
            "status": "complete",
            "evidence": "results/qwen3_moe_mechanism_effect_attribution/report.md decomposes the Qwen3 MoE source-frontier -> route-guarded -> audit-gated -> trust-region -> expert-only -> tail-trimmed -> searched-cap -> layer/chunk -> unified-mechanism chain into structural and downstream score deltas, gated by the eval-bundle audit.",
        },
        {
            "item": "Qwen3 MoE downstream feedback optimizer",
            "status": "complete",
            "evidence": "results/qwen3_moe_feedback_optimizer/report.md converts source-frontier task regressions from vLLM eval into bounded routed-expert rule updates; results/qwen3_moe_feedback_optimizer_smoke/report.md verifies code-regression restoration, non-code source-regression shrinkage, hard-cap enforcement, no-update awaiting-eval behavior, and eval-bundle-to-feedback integration.",
        },
        {
            "item": "Qwen3 MoE mechanistic unified candidate",
            "status": "complete",
            "evidence": "results/qwen3_moe_mechanistic_unified_candidate/report.md solves per-expert nonbase scale from benefit, curvature, and interference proxies, using real route mass, expert geometry, subspace conflict, delta pressure, and feedback priors; results/qwen3_moe_mechanistic_evidence_audit/report.md checks the B/H/I gradient, hard-cap binding, and internal-feature scale response; results/qwen3_moe_mechanistic_sensitivity/report.md reruns feature-family counterfactual full-score ablations to identify which internal signals protect the complete B/H/I objective; results/qwen3_moe_router_expert_coupling/report.md joins router top-k boundary fragility with expert scales to verify router-boundary risk becomes expert trust-region shrink; results/qwen3_moe_router_coupled_candidate/report.md materializes that coupling into a writer-ready ablation-only same-shape candidate; results/qwen3_moe_mechanistic_unified_candidate_smoke/report.md verifies monotonic mechanism behavior, hard-cap enforcement, and feedback shrink gating.",
        },
        {
            "item": "Qwen3 MoE post-vLLM eval refresh pipeline",
            "status": "complete",
            "evidence": "results/qwen3_moe_post_eval_refresh/report.md runs eval-bundle audit, unified/final selection, mechanism attribution, downstream feedback optimization, mechanistic unified candidate generation, mechanistic evidence audit, mechanistic sensitivity attribution, router-expert coupling attribution, router-coupled ablation candidate generation, unified average optimizer refresh, smoke checks, and collect_results in a fixed post-eval order after remote vLLM outputs land.",
        },
        {
            "item": "Qwen3 MoE searched cap-law materialized candidate",
            "status": "complete",
            "evidence": "results/qwen3_moe_searched_no_gt065_delta_audit/report.md verifies the materialized searched 0.65 cap-law checkpoint and adds it to the Qwen3 MoE eval gate.",
        },
        {
            "item": "Qwen3 MoE router move gate",
            "status": "complete",
            "evidence": "results/qwen3_moe_router_move_gate/report.md combines router tensor deltas with real routing readiness and rejects direct router-weight movement for all 48 layers.",
        },
        {
            "item": "Qwen3 MoE router margin fragility probe",
            "status": "complete",
            "evidence": "results/qwen3_moe_router_margin_fragility/report.md ranks router layers and prompt categories by top-k boundary fragility from real Qwen3 route margins, overlap, and router movement.",
        },
        {
            "item": "Qwen3 MoE router calibration NLL probe",
            "status": "complete",
            "evidence": "results/qwen3_moe_router_calibration_nll_probe/report.md formalizes the real Qwen3 router-only training probe, showing the averaged MoE improves when only router dispatch is recalibrated while keeping experts frozen.",
        },
        {
            "item": "Qwen3 MoE router calibration job",
            "status": "complete",
            "evidence": "results/qwen3_moe_router_calibration_job/report.md turns the rejected direct-router-move result into a margin-capped route-KD router-calibration sweep job and locks source/baseline/candidate vLLM evals to one task manifest.",
        },
        {
            "item": "Qwen3 MoE router calibration result selector",
            "status": "complete",
            "evidence": "results/qwen3_moe_router_calibration_selection/report.md accepts a router-calibrated cap only when matched vLLM eval, router-only tensor audit, top-k margin cap compliance, and source/baseline dominance gates pass.",
        },
        {
            "item": "Qwen3 MoE trust-region cap-law search",
            "status": "complete",
            "evidence": "results/qwen3_moe_trust_region_cap_search/report.md searches interpretable expert cap laws over real Qwen3 route-mass, risk-flag, and safetensors-delta probes and emits writer-ready next-candidate rules.",
        },
        {
            "item": "Qwen3 MoE unified mechanism candidate",
            "status": "complete",
            "evidence": "results/qwen3_moe_unified_mechanism_candidate/report.md turns route mass, router fragility, load, source-conflict, delta, expert geometry, and subspace-conflict probes into one same-shape constrained optimizer and writer-ready candidate.",
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
        "fp_curvature_law": summarize_fp_curvature_law(),
        "fp_merge_compare_dense": summarize_fp_merge_compare(),
        "fp_gen_eval_dense": summarize_fp_gen_eval_dense(),
        "fp_downstream_matrix": summarize_fp_downstream_matrix(),
        "fp_downstream_attribution": summarize_fp_downstream_attribution(),
        "fp_downstream_confidence_audit": summarize_fp_downstream_confidence_audit(),
        "qwen_probe_smoke": summarize_qwen_probe_smoke(),
        "average_decision_report": summarize_average_decision_report(),
        "model_averaging_literature_review": summarize_model_averaging_literature_review(),
        "average_method_gate_matrix": summarize_average_method_gate_matrix(),
        "average_trust_region_bounds": summarize_average_trust_region_bounds(),
        "average_trust_region_bounds_smoke": summarize_average_trust_region_bounds_smoke(),
        "average_connectivity_diagnostic": summarize_average_connectivity_diagnostic(),
        "average_invariant_audit": summarize_average_invariant_audit(),
        "qwen_target_model_registry": summarize_qwen_target_model_registry(),
        "moe_routing_probe_smoke": summarize_moe_routing_probe_smoke(),
        "moe_average_plan": summarize_moe_average_plan(),
        "same_shape_writer_smoke": summarize_same_shape_writer_smoke(),
        "dense_sparse_method_writer_smoke": summarize_dense_sparse_method_writer_smoke(),
        "moe_tensor_rule_writer_smoke": summarize_moe_tensor_rule_writer_smoke(),
        "moe_router_delta_calibration_smoke": summarize_moe_router_delta_calibration_smoke(),
        "moe_router_calibration_cache_smoke": summarize_moe_router_calibration_cache_smoke(),
        "moe_combined_writer_smoke": summarize_moe_combined_writer_smoke(),
        "moe_packed_expert_writer_smoke": summarize_moe_packed_expert_writer_smoke(),
        "checkpoint_topology_inspect": summarize_checkpoint_topology(),
        "average_candidate_recipes": summarize_average_candidate_recipes(),
        "moe_route_weight_recipes": summarize_moe_route_weight_recipes(),
        "moe_packed_route_weight_recipe_smoke": summarize_moe_packed_route_weight_recipe_smoke(),
        "moe_router_bias_plan": summarize_moe_router_bias_plan(),
        "moe_confidence_blended_router_bias_plan": summarize_moe_confidence_blended_router_bias_plan(),
        "toy_moe_expert_weight_recipes": summarize_toy_moe_expert_weight_recipes(),
        "toy_moe_output_projection_recipes": summarize_toy_moe_output_projection_recipes(),
        "toy_moe_confidence_blended_recipes": summarize_toy_moe_confidence_blended_recipes(),
        "moe_confidence_blended_combined_recipe": summarize_moe_confidence_blended_combined_recipe(),
        "moe_routing_readiness": summarize_moe_routing_readiness(),
        "toy_moe_merge": summarize_toy_moe_merge(),
        "fp_moe_mechanism": summarize_fp_moe_mechanism(),
        "fp_moe_real_probe": summarize_fp_moe_real_probe(),
        "moe_probe_gated_selector": summarize_moe_probe_gated_selector(),
        "moe_unified_preflight_qwen3_30b": summarize_moe_unified_preflight(),
        "qwen3_moe_routing_probe": summarize_qwen3_moe_routing_probe(),
        "qwen3_moe_routing_readiness": summarize_qwen3_moe_routing_readiness(),
        "qwen3_moe_route_guarded_candidate": summarize_qwen3_moe_route_guarded_candidate(),
        "qwen3_moe_materialized_delta_audit": summarize_qwen3_moe_materialized_delta_audit(),
        "qwen3_moe_audit_gated_candidate": summarize_qwen3_moe_audit_gated_candidate(),
        "qwen3_moe_audit_gated_delta_audit": summarize_qwen3_moe_audit_gated_delta_audit(),
        "qwen3_moe_trust_region_candidate": summarize_qwen3_moe_trust_region_candidate(),
        "qwen3_moe_trust_region_delta_audit": summarize_qwen3_moe_trust_region_delta_audit(),
        "qwen3_moe_trust_region_delta_validation": summarize_qwen3_moe_trust_region_delta_validation(),
        "qwen3_moe_expert_only_trust_region_candidate": (
            summarize_qwen3_moe_expert_only_trust_region_candidate()
        ),
        "qwen3_moe_expert_only_delta_audit": summarize_qwen3_moe_expert_only_delta_audit(),
        "qwen3_moe_tail_trimmed_expert_only_candidate": (
            summarize_qwen3_moe_tail_trimmed_expert_only_candidate()
        ),
        "qwen3_moe_tail_trimmed_delta_audit": summarize_qwen3_moe_tail_trimmed_delta_audit(),
        "qwen3_moe_searched_no_gt065_delta_audit": summarize_qwen3_moe_searched_no_gt065_delta_audit(),
        "qwen3_moe_layer_chunk_delta_audit": summarize_qwen3_moe_layer_chunk_delta_audit(),
        "qwen3_moe_delta_frontier": summarize_qwen3_moe_delta_frontier(),
        "qwen3_moe_mechanism_eval_gate": summarize_qwen3_moe_mechanism_eval_gate(),
        "qwen3_moe_eval_budget_plan": summarize_qwen3_moe_eval_budget_plan(),
        "qwen3_moe_eval_budget_queue_smoke": summarize_qwen3_moe_eval_budget_queue_smoke(),
        "qwen3_moe_adaptive_eval_schedule": summarize_qwen3_moe_adaptive_eval_schedule(),
        "qwen3_moe_adaptive_eval_schedule_smoke": summarize_qwen3_moe_adaptive_eval_schedule_smoke(),
        "qwen3_moe_eval_manifest_preflight": summarize_qwen3_moe_eval_manifest_preflight(),
        "qwen3_moe_mechanism_levers": summarize_qwen3_moe_mechanism_levers(),
        "qwen3_moe_expert_geometry_probe": summarize_qwen3_moe_expert_geometry_probe(),
        "qwen3_moe_expert_subspace_conflict_probe": summarize_qwen3_moe_expert_subspace_conflict_probe(),
        "qwen3_moe_layer_chunk_candidate": summarize_qwen3_moe_layer_chunk_candidate(),
        "qwen3_moe_unified_result_selection": summarize_qwen3_moe_unified_result_selection(),
        "qwen3_moe_unified_result_selection_smoke": summarize_qwen3_moe_unified_result_selection_smoke(),
        "qwen3_moe_candidate_trust_region_gate": summarize_qwen3_moe_candidate_trust_region_gate(),
        "qwen3_moe_final_candidate_selection": summarize_qwen3_moe_final_candidate_selection(),
        "qwen3_moe_final_candidate_selection_smoke": summarize_qwen3_moe_final_candidate_selection_smoke(),
        "unified_average_optimizer": summarize_unified_average_optimizer(),
        "unified_average_optimizer_ledger_smoke": summarize_unified_average_optimizer_ledger_smoke(),
        "qwen3_moe_eval_bundle_audit": summarize_qwen3_moe_eval_bundle_audit(),
        "qwen3_moe_eval_bundle_audit_smoke": summarize_qwen3_moe_eval_bundle_audit_smoke(),
        "qwen3_moe_mechanism_effect_attribution": summarize_qwen3_moe_mechanism_effect_attribution(),
        "qwen3_moe_mechanism_effect_attribution_smoke": (
            summarize_qwen3_moe_mechanism_effect_attribution_smoke()
        ),
        "qwen3_moe_feedback_optimizer": summarize_qwen3_moe_feedback_optimizer(),
        "qwen3_moe_feedback_optimizer_smoke": summarize_qwen3_moe_feedback_optimizer_smoke(),
        "qwen3_moe_mechanistic_unified_candidate": summarize_qwen3_moe_mechanistic_unified_candidate(),
        "qwen3_moe_mechanistic_unified_candidate_smoke": (
            summarize_qwen3_moe_mechanistic_unified_candidate_smoke()
        ),
        "qwen3_moe_mechanistic_evidence_audit": summarize_qwen3_moe_mechanistic_evidence_audit(),
        "qwen3_moe_mechanistic_sensitivity": summarize_qwen3_moe_mechanistic_sensitivity(),
        "qwen3_moe_router_expert_coupling": summarize_qwen3_moe_router_expert_coupling(),
        "qwen3_moe_router_coupled_candidate": summarize_qwen3_moe_router_coupled_candidate(),
        "qwen3_moe_router_coupled_retention_frontier": (
            summarize_qwen3_moe_router_coupled_retention_frontier()
        ),
        "qwen3_moe_post_eval_refresh": summarize_qwen3_moe_post_eval_refresh(),
        "qwen3_moe_post_eval_refresh_plan": summarize_qwen3_moe_post_eval_refresh_plan(),
        "qwen3_moe_router_move_gate": summarize_qwen3_moe_router_move_gate(),
        "qwen3_moe_router_margin_fragility": summarize_qwen3_moe_router_margin_fragility(),
        "qwen3_moe_router_calibration_nll_probe": summarize_qwen3_moe_router_calibration_nll_probe(),
        "qwen3_moe_router_calibration_job": summarize_qwen3_moe_router_calibration_job(),
        "qwen3_moe_router_calibration_selection": summarize_qwen3_moe_router_calibration_selection(),
        "qwen3_moe_router_calibration_row_validation_negative_smoke": (
            summarize_qwen3_moe_router_calibration_row_validation_negative_smoke()
        ),
        "qwen3_moe_router_calibration_source_dominance_negative_smoke": (
            summarize_qwen3_moe_router_calibration_source_dominance_negative_smoke()
        ),
        "qwen3_moe_router_calibration_no_gain_negative_smoke": (
            summarize_qwen3_moe_router_calibration_no_gain_negative_smoke()
        ),
        "qwen3_moe_router_calibration_task_regression_negative_smoke": (
            summarize_qwen3_moe_router_calibration_task_regression_negative_smoke()
        ),
        "qwen3_moe_router_calibration_selector_matrix_smoke": (
            summarize_qwen3_moe_router_calibration_selector_matrix_smoke()
        ),
        "qwen3_moe_trust_region_cap_search": summarize_qwen3_moe_trust_region_cap_search(),
        "qwen3_moe_unified_mechanism_candidate": summarize_qwen3_moe_unified_mechanism_candidate(),
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
        "qwen_dense_sparse_method_candidate": summarize_qwen_dense_sparse_method_candidate(
            "results/qwen_dense_sparse_method_candidate"
        ),
        "qwen_dense_attention_sparse_method_candidate": summarize_qwen_dense_sparse_method_candidate(
            "results/qwen_dense_attention_sparse_method_candidate"
        ),
        "checkpoint_materialization_readiness": summarize_checkpoint_materialization_readiness(),
        "moe_materialization_pipeline_plan": summarize_moe_materialization_pipeline_plan(),
        "probe_gated_unified_average_plan": summarize_probe_gated_unified_average_plan(),
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
            "python scripts/fp_curvature_law.py --out results/fp_curvature_law",
            "python scripts/fp_merge_compare.py --instruct Qwen/Qwen2.5-0.5B-Instruct --coder Qwen/Qwen2.5-Coder-0.5B-Instruct --base Qwen/Qwen2.5-0.5B --out results/fp_merge_compare_dense --n-general 4 --n-code 4 --seqlen 128 --grid-profile linear --device cpu",
            "python scripts/fp_gen_eval.py --instruct Qwen/Qwen2.5-0.5B-Instruct --coder Qwen/Qwen2.5-Coder-0.5B-Instruct --base Qwen/Qwen2.5-0.5B --out results/fp_gen_eval_dense --n-math 2 --n-code 2 --max-new-tokens 8 --device cpu --methods base,instruct,coder,linear,unified",
            "python scripts/fp_moe_mechanism.py --out results/fp_moe_mechanism --base-steps 700 --ft-steps 500",
            "python scripts/fp_moe_real_probe.py --mode gauge_selfmerge --model-a allenai/OLMoE-1B-7B-0924-Instruct --out results/fp_moe_real_probe --n-probe 4 --seqlen 128",
            "python scripts/build_moe_probe_gated_selector.py",
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
            "python scripts/build_average_method_gate_matrix.py --output-dir results/average_method_gate_matrix",
            "python scripts/build_average_trust_region_bounds.py --output-dir results/average_trust_region_bounds",
            "python scripts/build_connectivity_diagnostic.py --output-dir results/average_connectivity_diagnostic",
            "python scripts/build_average_invariant_audit.py --output-dir results/average_invariant_audit",
            "python scripts/smoke_moe_routing_probe_contract.py",
            "python scripts/smoke_vllm_downstream_eval_contract.py --output-dir results/vllm_downstream_eval_smoke",
            "PYTHONPATH=src python scripts/build_vllm_checkpoint_eval_plan.py --output-dir results/vllm_checkpoint_eval_plan",
            "PYTHONPATH=src python scripts/build_vllm_source_merge_comparison.py --output-dir results/vllm_source_merge_comparison",
            "PYTHONPATH=src python scripts/build_probe_guided_dense_average_candidate.py --output-dir results/probe_guided_dense_average_candidate",
            "PYTHONPATH=src python scripts/build_qwen_dense_module_guarded_candidate.py --output-dir results/qwen_dense_module_guarded_candidate --variant module_guarded",
            "PYTHONPATH=src python scripts/build_qwen_dense_module_guarded_candidate.py --output-dir results/qwen_dense_norm_guarded_candidate --variant norm_only",
            "PYTHONPATH=src python scripts/build_qwen_dense_module_guarded_candidate.py --output-dir results/qwen_dense_selective_norm_guarded_candidate --variant selective_norm",
            "PYTHONPATH=src python scripts/build_qwen_dense_sparse_method_candidate.py --output-dir results/qwen_dense_sparse_method_candidate",
            "PYTHONPATH=src python scripts/build_qwen_dense_sparse_method_candidate.py --output-dir results/qwen_dense_attention_sparse_method_candidate --candidate-id qwen_0_5b_attention_sparse_method_bridge --groups attention",
            "PYTHONPATH=src python scripts/build_average_decision_report.py",
            "python scripts/build_qwen_target_model_registry.py",
            "PYTHONPATH=src python scripts/build_moe_average_plan.py",
            "python scripts/write_same_shape_average_checkpoint.py --base BASE --source expert=EXPERT --dry-run --output-dir results/same_shape_writer_smoke",
            "PYTHONPATH=scripts python scripts/smoke_dense_sparse_method_writer.py --output-dir results/dense_sparse_method_writer_smoke",
            "python scripts/smoke_moe_tensor_rule_writer.py --output-dir results/moe_tensor_rule_writer_smoke",
            "python scripts/train_moe_router_delta_calibration.py --smoke-cap-table --output-dir results/moe_router_delta_calibration_smoke",
            "PYTHONPATH=scripts python scripts/collect_moe_router_calibration_cache.py --smoke --output-dir results/moe_router_calibration_cache_smoke",
            "PYTHONPATH=src python scripts/smoke_moe_combined_writer.py --output-dir results/moe_combined_writer_smoke",
            "PYTHONPATH=scripts python scripts/smoke_moe_packed_expert_writer.py --output-dir results/moe_packed_expert_writer_smoke",
            "python scripts/inspect_checkpoint_topology.py --model NAME=MODEL_PATH --output-dir results/checkpoint_topology_inspect",
            "PYTHONPATH=src python scripts/build_average_candidate_recipes.py",
            "PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code",
            "PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code --source general --source code",
            "PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --output-dir results/moe_packed_route_weight_recipe_smoke --expert-weight-csv results/moe_packed_route_weight_recipe_smoke/expert_weights.csv --source general --source code --topology-summary results/checkpoint_topology_inspect/summary.json --checkpoint-output-dir results/checkpoints/moe_packed_route_weight_candidate",
            "PYTHONPATH=src python scripts/build_moe_router_bias_plan.py --router-dir results/toy_moe_merge --method unified_moe_average --router-bias-template '{router}.bias'",
            "PYTHONPATH=src python scripts/build_moe_router_bias_plan.py --router-dir results/toy_moe_merge --method unified_confidence_blended_moe_average --output-dir results/moe_confidence_blended_router_bias_plan --router-bias-template '{router}.bias'",
            "PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --output-dir results/toy_moe_expert_weight_recipes --expert-weight-csv results/toy_moe_merge/expert_search_weights_by_expert.csv --source general --source code --checkpoint-output-dir results/checkpoints/toy_moe_expert_weight_candidate --topology-summary ''",
            "PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --output-dir results/toy_moe_output_projection_recipes --expert-weight-csv results/toy_moe_merge/expert_output_projection_weights_by_expert.csv --expert-weight-category combined --source general --source code --checkpoint-output-dir results/checkpoints/toy_moe_output_projection_candidate --topology-summary ''",
            "PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --output-dir results/toy_moe_confidence_blended_recipes --expert-weight-csv results/toy_moe_merge/confidence_blended_expert_weights_by_expert.csv --source general --source code --checkpoint-output-dir results/checkpoints/toy_moe_confidence_blended_candidate --topology-summary ''",
            "PYTHONPATH=src python scripts/build_moe_combined_materialization_recipe.py",
            "PYTHONPATH=src python scripts/build_checkpoint_materialization_readiness.py --output-dir results/checkpoint_materialization_readiness",
            "PYTHONPATH=src python scripts/build_moe_materialization_pipeline_plan.py --output-dir results/moe_materialization_pipeline_plan",
            "PYTHONPATH=src python scripts/build_probe_gated_unified_average_plan.py --output-dir results/probe_gated_unified_average_plan",
            "python scripts/audit_materialized_checkpoint_delta.py --base BASE --candidate CANDIDATE --output-dir results/qwen3_moe_materialized_delta_audit",
            "python scripts/build_qwen3_moe_audit_gated_candidate.py --output-dir results/qwen3_moe_audit_gated_candidate",
            "python scripts/audit_materialized_checkpoint_delta.py --base BASE --candidate AUDIT_GATED_CANDIDATE --output-dir results/qwen3_moe_audit_gated_delta_audit",
            "python scripts/build_qwen3_moe_trust_region_candidate.py --output-dir results/qwen3_moe_trust_region_candidate",
            "python scripts/audit_materialized_checkpoint_delta.py --base BASE --candidate TRUST_REGION_CANDIDATE --output-dir results/qwen3_moe_trust_region_delta_audit",
            "python scripts/validate_qwen3_moe_trust_region_delta.py --output-dir results/qwen3_moe_trust_region_delta_validation",
            "python scripts/build_qwen3_moe_expert_only_trust_region_candidate.py --output-dir results/qwen3_moe_expert_only_trust_region_candidate",
            "python scripts/audit_materialized_checkpoint_delta.py --base BASE --candidate EXPERT_ONLY_CANDIDATE --output-dir results/qwen3_moe_expert_only_delta_audit",
            "python scripts/build_qwen3_moe_tail_trimmed_expert_only_candidate.py --output-dir results/qwen3_moe_tail_trimmed_expert_only_candidate",
            "python scripts/audit_materialized_checkpoint_delta.py --base BASE --candidate TAIL_TRIMMED_CANDIDATE --output-dir results/qwen3_moe_tail_trimmed_delta_audit",
            "bash results/qwen3_moe_trust_region_cap_search/searched_no_gt065_max_retention_writer_command.txt",
            "python scripts/audit_materialized_checkpoint_delta.py --base BASE --candidate results/checkpoints/qwen3_moe_searched_no_gt065_max_retention_candidate --output-dir results/qwen3_moe_searched_no_gt065_delta_audit",
            "python scripts/build_qwen3_moe_delta_frontier.py --output-dir results/qwen3_moe_delta_frontier",
            "python scripts/plan_qwen3_moe_eval_budget.py --output-dir results/qwen3_moe_eval_budget_plan",
            "python scripts/smoke_qwen3_moe_eval_budget_queue.py --output-dir results/qwen3_moe_eval_budget_queue_smoke",
            "python scripts/schedule_qwen3_moe_adaptive_eval.py --output-dir results/qwen3_moe_adaptive_eval_schedule",
            "python scripts/schedule_qwen3_moe_adaptive_eval.py --smoke-matrix --output-dir results/qwen3_moe_adaptive_eval_schedule_smoke",
            "python scripts/probe_qwen3_moe_expert_geometry.py --output-dir results/qwen3_moe_expert_geometry_probe",
            "python scripts/analyze_qwen3_moe_expert_subspace_conflicts.py --output-dir results/qwen3_moe_expert_subspace_conflict_probe --validate-dry-run",
            "python scripts/analyze_qwen3_moe_mechanism_levers.py --output-dir results/qwen3_moe_mechanism_levers",
            "python scripts/build_qwen3_moe_layer_chunk_candidate.py --output-dir results/qwen3_moe_layer_chunk_candidate",
            "bash results/qwen3_moe_layer_chunk_candidate/writer_command.txt",
            "python scripts/audit_materialized_checkpoint_delta.py --base BASE --candidate results/checkpoints/qwen3_moe_layer_chunk_candidate --output-dir results/qwen3_moe_layer_chunk_delta_audit",
            "python scripts/build_qwen3_moe_router_move_gate.py --output-dir results/qwen3_moe_router_move_gate",
            "python scripts/build_qwen3_moe_router_margin_fragility.py --output-dir results/qwen3_moe_router_margin_fragility",
            "python scripts/build_qwen3_moe_router_calibration_nll_probe.py --output-dir results/qwen3_moe_router_calibration_nll_probe",
            "python scripts/build_qwen3_moe_router_calibration_job.py --output-dir results/qwen3_moe_router_calibration_job",
            "python scripts/select_qwen3_moe_router_calibration_result.py --output-dir results/qwen3_moe_router_calibration_selection",
            "python scripts/select_qwen3_moe_router_calibration_result.py --smoke --output-dir results/qwen3_moe_router_calibration_selection_smoke",
            "python scripts/select_qwen3_moe_router_calibration_result.py --row-validation-negative-smoke --output-dir results/qwen3_moe_router_calibration_selection_row_validation_negative_smoke",
            "python scripts/select_qwen3_moe_router_calibration_result.py --source-dominance-negative-smoke --output-dir results/qwen3_moe_router_calibration_selection_source_dominance_negative_smoke",
            "python scripts/select_qwen3_moe_router_calibration_result.py --no-downstream-gain-negative-smoke --output-dir results/qwen3_moe_router_calibration_selection_no_gain_negative_smoke",
            "python scripts/select_qwen3_moe_router_calibration_result.py --task-regression-negative-smoke --output-dir results/qwen3_moe_router_calibration_selection_task_regression_negative_smoke",
            "python scripts/smoke_qwen3_moe_router_calibration_selector_matrix.py --output-dir results/qwen3_moe_router_calibration_selector_matrix_smoke",
            "python scripts/build_qwen3_moe_unified_mechanism_candidate.py --output-dir results/qwen3_moe_unified_mechanism_candidate",
            "bash results/qwen3_moe_unified_mechanism_candidate/writer_command.txt",
            "python scripts/audit_materialized_checkpoint_delta.py --base BASE --candidate results/checkpoints/qwen3_moe_unified_mechanism_candidate --output-dir results/qwen3_moe_unified_mechanism_delta_audit",
            "python scripts/select_qwen3_moe_unified_result.py --output-dir results/qwen3_moe_unified_result_selection",
            "python scripts/select_qwen3_moe_unified_result.py --smoke-matrix --output-dir results/qwen3_moe_unified_result_selection_smoke",
            "python scripts/build_qwen3_moe_candidate_trust_region_gate.py --output-dir results/qwen3_moe_candidate_trust_region_gate",
            "python scripts/select_qwen3_moe_final_candidate.py --output-dir results/qwen3_moe_final_candidate_selection",
            "python scripts/select_qwen3_moe_final_candidate.py --smoke-matrix --output-dir results/qwen3_moe_final_candidate_selection_smoke",
            "python scripts/build_unified_average_optimizer.py --output-dir results/unified_average_optimizer",
            "python scripts/smoke_unified_average_optimizer_ledger.py --output-dir results/unified_average_optimizer_ledger_smoke",
            "python scripts/audit_qwen3_moe_eval_bundle.py --output-dir results/qwen3_moe_eval_bundle_audit",
            "python scripts/audit_qwen3_moe_eval_bundle.py --smoke-matrix --output-dir results/qwen3_moe_eval_bundle_audit_smoke",
            "python scripts/attribute_qwen3_moe_mechanism_effects.py --output-dir results/qwen3_moe_mechanism_effect_attribution",
            "python scripts/attribute_qwen3_moe_mechanism_effects.py --smoke-matrix --output-dir results/qwen3_moe_mechanism_effect_attribution_smoke",
            "python scripts/build_qwen3_moe_feedback_optimizer.py --output-dir results/qwen3_moe_feedback_optimizer",
            "python scripts/build_qwen3_moe_feedback_optimizer.py --smoke-matrix --output-dir results/qwen3_moe_feedback_optimizer_smoke",
            "python scripts/build_qwen3_moe_mechanistic_unified_candidate.py --output-dir results/qwen3_moe_mechanistic_unified_candidate",
            "python scripts/build_qwen3_moe_mechanistic_unified_candidate.py --smoke-matrix --output-dir results/qwen3_moe_mechanistic_unified_candidate_smoke",
            "python scripts/audit_qwen3_moe_mechanistic_evidence.py --output-dir results/qwen3_moe_mechanistic_evidence_audit",
            "python scripts/analyze_qwen3_moe_mechanistic_sensitivity.py --output-dir results/qwen3_moe_mechanistic_sensitivity",
            "python scripts/analyze_qwen3_moe_router_expert_coupling.py --output-dir results/qwen3_moe_router_expert_coupling",
            "python scripts/build_qwen3_moe_router_coupled_candidate.py --output-dir results/qwen3_moe_router_coupled_candidate",
            "python scripts/refresh_qwen3_moe_post_eval.py --plan-only --include-smoke --output-dir results/qwen3_moe_post_eval_refresh_plan",
            "python scripts/refresh_qwen3_moe_post_eval.py --include-smoke --output-dir results/qwen3_moe_post_eval_refresh",
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
    fp_curvature = exp["fp_curvature_law"]
    fp_merge_compare = exp["fp_merge_compare_dense"]
    fp_gen_eval = exp["fp_gen_eval_dense"]
    fp_downstream_matrix = exp["fp_downstream_matrix"]
    fp_downstream_attribution = exp["fp_downstream_attribution"]
    fp_downstream_confidence = exp["fp_downstream_confidence_audit"]
    average_decision = exp["average_decision_report"]
    literature_review = exp["model_averaging_literature_review"]
    average_method_gate_matrix = exp["average_method_gate_matrix"]
    average_trust_region_bounds = exp["average_trust_region_bounds"]
    average_trust_region_bounds_smoke = exp["average_trust_region_bounds_smoke"]
    average_connectivity = exp["average_connectivity_diagnostic"]
    average_invariant_audit = exp["average_invariant_audit"]
    qwen_registry = exp["qwen_target_model_registry"]
    routing_probe_smoke = exp["moe_routing_probe_smoke"]
    moe_average_plan = exp["moe_average_plan"]
    writer_smoke = exp["same_shape_writer_smoke"]
    dense_sparse_writer_smoke = exp["dense_sparse_method_writer_smoke"]
    moe_tensor_rule_writer_smoke = exp["moe_tensor_rule_writer_smoke"]
    moe_router_delta_calibration_smoke = exp["moe_router_delta_calibration_smoke"]
    moe_router_calibration_cache_smoke = exp["moe_router_calibration_cache_smoke"]
    moe_combined_writer_smoke = exp["moe_combined_writer_smoke"]
    moe_packed_expert_writer_smoke = exp["moe_packed_expert_writer_smoke"]
    topology = exp["checkpoint_topology_inspect"]
    moe_models = [model for model in topology["models"] if model.get("config", {}).get("is_moe_config")]
    recipes = exp["average_candidate_recipes"]
    route_weight_recipes = exp["moe_route_weight_recipes"]
    packed_route_recipe_smoke = exp["moe_packed_route_weight_recipe_smoke"]
    router_bias_plan = exp["moe_router_bias_plan"]
    confidence_blended_router_bias_plan = exp["moe_confidence_blended_router_bias_plan"]
    toy_expert_weight_recipes = exp["toy_moe_expert_weight_recipes"]
    toy_output_projection_recipes = exp["toy_moe_output_projection_recipes"]
    toy_confidence_blended_recipes = exp["toy_moe_confidence_blended_recipes"]
    confidence_blended_combined_recipe = exp["moe_confidence_blended_combined_recipe"]
    routing_readiness = exp["moe_routing_readiness"]
    toy_moe = exp["toy_moe_merge"]
    fp_moe = exp["fp_moe_mechanism"]
    fp_moe_real = exp["fp_moe_real_probe"]
    moe_selector = exp["moe_probe_gated_selector"]
    moe_unified_preflight = exp["moe_unified_preflight_qwen3_30b"]
    qwen3_moe_routing_probe = exp["qwen3_moe_routing_probe"]
    qwen3_moe_routing_readiness = exp["qwen3_moe_routing_readiness"]
    qwen3_moe_route_guarded_candidate = exp["qwen3_moe_route_guarded_candidate"]
    qwen3_moe_materialized_delta_audit = exp["qwen3_moe_materialized_delta_audit"]
    qwen3_moe_audit_gated_candidate = exp["qwen3_moe_audit_gated_candidate"]
    qwen3_moe_audit_gated_delta_audit = exp["qwen3_moe_audit_gated_delta_audit"]
    qwen3_moe_trust_region_candidate = exp["qwen3_moe_trust_region_candidate"]
    qwen3_moe_trust_region_delta_audit = exp["qwen3_moe_trust_region_delta_audit"]
    qwen3_moe_trust_region_delta_validation = exp["qwen3_moe_trust_region_delta_validation"]
    qwen3_moe_expert_only_trust_region_candidate = exp["qwen3_moe_expert_only_trust_region_candidate"]
    qwen3_moe_expert_only_delta_audit = exp["qwen3_moe_expert_only_delta_audit"]
    qwen3_moe_tail_trimmed_expert_only_candidate = exp["qwen3_moe_tail_trimmed_expert_only_candidate"]
    qwen3_moe_tail_trimmed_delta_audit = exp["qwen3_moe_tail_trimmed_delta_audit"]
    qwen3_moe_searched_no_gt065_delta_audit = exp["qwen3_moe_searched_no_gt065_delta_audit"]
    qwen3_moe_layer_chunk_delta_audit = exp["qwen3_moe_layer_chunk_delta_audit"]
    qwen3_moe_delta_frontier = exp["qwen3_moe_delta_frontier"]
    qwen3_moe_mechanism_eval_gate = exp["qwen3_moe_mechanism_eval_gate"]
    qwen3_moe_eval_budget_plan = exp["qwen3_moe_eval_budget_plan"]
    qwen3_moe_eval_budget_queue_smoke = exp["qwen3_moe_eval_budget_queue_smoke"]
    qwen3_moe_adaptive_eval_schedule = exp["qwen3_moe_adaptive_eval_schedule"]
    qwen3_moe_adaptive_eval_schedule_smoke = exp["qwen3_moe_adaptive_eval_schedule_smoke"]
    qwen3_moe_eval_manifest_preflight = exp["qwen3_moe_eval_manifest_preflight"]
    qwen3_moe_mechanism_levers = exp["qwen3_moe_mechanism_levers"]
    qwen3_moe_expert_geometry_probe = exp["qwen3_moe_expert_geometry_probe"]
    qwen3_moe_expert_subspace_conflict_probe = exp["qwen3_moe_expert_subspace_conflict_probe"]
    qwen3_moe_layer_chunk_candidate = exp["qwen3_moe_layer_chunk_candidate"]
    qwen3_moe_unified_result_selection = exp["qwen3_moe_unified_result_selection"]
    qwen3_moe_unified_result_selection_smoke = exp["qwen3_moe_unified_result_selection_smoke"]
    qwen3_moe_candidate_trust_region_gate = exp["qwen3_moe_candidate_trust_region_gate"]
    qwen3_moe_final_candidate_selection = exp["qwen3_moe_final_candidate_selection"]
    qwen3_moe_final_candidate_selection_smoke = exp["qwen3_moe_final_candidate_selection_smoke"]
    unified_average_optimizer = exp["unified_average_optimizer"]
    unified_average_optimizer_ledger_smoke = exp["unified_average_optimizer_ledger_smoke"]
    qwen3_moe_eval_bundle_audit = exp["qwen3_moe_eval_bundle_audit"]
    qwen3_moe_eval_bundle_audit_smoke = exp["qwen3_moe_eval_bundle_audit_smoke"]
    qwen3_moe_mechanism_effect_attribution = exp["qwen3_moe_mechanism_effect_attribution"]
    qwen3_moe_mechanism_effect_attribution_smoke = exp[
        "qwen3_moe_mechanism_effect_attribution_smoke"
    ]
    qwen3_moe_feedback_optimizer = exp["qwen3_moe_feedback_optimizer"]
    qwen3_moe_feedback_optimizer_smoke = exp["qwen3_moe_feedback_optimizer_smoke"]
    qwen3_moe_mechanistic_unified_candidate = exp["qwen3_moe_mechanistic_unified_candidate"]
    qwen3_moe_mechanistic_unified_candidate_smoke = exp[
        "qwen3_moe_mechanistic_unified_candidate_smoke"
    ]
    qwen3_moe_mechanistic_evidence_audit = exp["qwen3_moe_mechanistic_evidence_audit"]
    qwen3_moe_mechanistic_sensitivity = exp["qwen3_moe_mechanistic_sensitivity"]
    qwen3_moe_router_expert_coupling = exp["qwen3_moe_router_expert_coupling"]
    qwen3_moe_router_coupled_candidate = exp["qwen3_moe_router_coupled_candidate"]
    qwen3_moe_router_coupled_retention_frontier = exp[
        "qwen3_moe_router_coupled_retention_frontier"
    ]
    qwen3_moe_post_eval_refresh = exp["qwen3_moe_post_eval_refresh"]
    qwen3_moe_post_eval_refresh_plan = exp["qwen3_moe_post_eval_refresh_plan"]
    qwen3_moe_router_move_gate = exp["qwen3_moe_router_move_gate"]
    qwen3_moe_router_margin_fragility = exp["qwen3_moe_router_margin_fragility"]
    qwen3_moe_router_calibration_nll_probe = exp["qwen3_moe_router_calibration_nll_probe"]
    qwen3_moe_router_calibration_job = exp["qwen3_moe_router_calibration_job"]
    qwen3_moe_router_calibration_selection = exp["qwen3_moe_router_calibration_selection"]
    qwen3_moe_router_calibration_row_validation_negative_smoke = exp[
        "qwen3_moe_router_calibration_row_validation_negative_smoke"
    ]
    qwen3_moe_router_calibration_row_validation_negative_rows = (
        qwen3_moe_router_calibration_row_validation_negative_smoke.get("candidate_rows") or [{}]
    )
    qwen3_moe_router_calibration_source_dominance_negative_smoke = exp[
        "qwen3_moe_router_calibration_source_dominance_negative_smoke"
    ]
    qwen3_moe_router_calibration_source_dominance_rows = (
        qwen3_moe_router_calibration_source_dominance_negative_smoke.get("candidate_rows") or [{}]
    )
    qwen3_moe_router_calibration_no_gain_negative_smoke = exp[
        "qwen3_moe_router_calibration_no_gain_negative_smoke"
    ]
    qwen3_moe_router_calibration_no_gain_rows = (
        qwen3_moe_router_calibration_no_gain_negative_smoke.get("candidate_rows") or [{}]
    )
    qwen3_moe_router_calibration_task_regression_negative_smoke = exp[
        "qwen3_moe_router_calibration_task_regression_negative_smoke"
    ]
    qwen3_moe_router_calibration_task_regression_rows = (
        qwen3_moe_router_calibration_task_regression_negative_smoke.get("candidate_rows") or [{}]
    )
    qwen3_moe_router_calibration_selector_matrix_smoke = exp[
        "qwen3_moe_router_calibration_selector_matrix_smoke"
    ]
    def matching_decision_reason(rows: list[dict[str, Any]], marker: str | None = None) -> Any:
        for row in rows:
            reason = row.get("decision_reason")
            if marker is None or (reason is not None and marker in str(reason)):
                return reason
        return rows[0].get("decision_reason") if rows else None

    qwen3_moe_trust_region_cap_search = exp["qwen3_moe_trust_region_cap_search"]
    qwen3_moe_unified_mechanism_candidate = exp["qwen3_moe_unified_mechanism_candidate"]
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
    qwen_dense_sparse_method = exp["qwen_dense_sparse_method_candidate"]
    qwen_dense_attention_sparse_method = exp["qwen_dense_attention_sparse_method_candidate"]
    materialization_readiness = exp["checkpoint_materialization_readiness"]
    moe_pipeline_plan = exp["moe_materialization_pipeline_plan"]
    probe_gated_unified = exp["probe_gated_unified_average_plan"]
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
                "| dense curvature law | general actual / predicted degradation | "
                f"{fmt(fp_curvature['curvature_law'].get('ratio_general'))} |"
            ),
            (
                "| dense curvature law | code actual / predicted degradation | "
                f"{fmt(fp_curvature['curvature_law'].get('ratio_code'))} |"
            ),
            (
                "| dense curvature law | uniform / Fisher worst NLL | "
                f"{fmt(fp_curvature['uniform'].get('worst'))} / {fmt(fp_curvature['fisher'].get('worst'))} |"
            ),
            (
                "| dense curvature law | top interference tensor | "
                f"{fp_curvature['top_interference_tensor'].get('name')} |"
            ),
            (
                "| dense unified selector | selected lambda / test worst NLL | "
                f"{fmt(fp_merge_compare['selected_config'].get('lam'), 2)} / {fmt(fp_merge_compare['unified'].get('worst'))} |"
            ),
            (
                "| dense unified selector | linear / TIES worst delta vs unified | "
                f"{fmt(fp_merge_compare['linear_worst_delta_vs_unified'])} / {fmt(fp_merge_compare['ties_worst_delta_vs_unified'])} |"
            ),
            (
                "| dense unified selector | unified minus best endpoint worst NLL | "
                f"{fmt(fp_merge_compare['unified_worst_minus_best_endpoint'])} |"
            ),
            (
                "| dense generation smoke | best method / linear avg accuracy | "
                f"{fp_gen_eval['best_method']} / {fmt(fp_gen_eval['linear'].get('avg_accuracy'))} |"
            ),
            (
                "| dense generation smoke | unified avg delta vs linear | "
                f"{fmt(fp_gen_eval['unified_avg_delta_vs_linear'])} |"
            ),
            (
                "| dense generation smoke | coder worst delta vs unified | "
                f"{fmt(fp_gen_eval['coder_worst_delta_vs_unified'])} |"
            ),
            (
                "| Qwen3 MoE downstream generation matrix | best avg model / avg | "
                f"{fp_downstream_matrix['best_avg_model']} / {fmt(fp_downstream_matrix['best_avg'])} |"
            ),
            (
                "| Qwen3 MoE downstream generation matrix | Instruct+Coder avg -> +router-cal avg | "
                f"{fmt(fp_downstream_matrix['pair_merge_avg'])} -> {fmt(fp_downstream_matrix['pair_routercal_avg'])} |"
            ),
            (
                "| Qwen3 MoE downstream generation matrix | router-cal avg gain / HumanEval gain / gap to best parent | "
                f"{fmt(fp_downstream_matrix['pair_routercal_avg_gain'])} / "
                f"{fmt(fp_downstream_matrix['pair_routercal_humaneval_gain'])} / "
                f"{fmt(fp_downstream_matrix['pair_routercal_gap_to_best_parent_avg'])} |"
            ),
            (
                "| Qwen3 MoE downstream attribution | avg drop / router-cal recovery fraction / gap | "
                f"{fmt(fp_downstream_attribution['avg_naive_drop_vs_pair_frontier'])} / "
                f"{fmt(fp_downstream_attribution['avg_routercal_recovery_fraction'])} / "
                f"{fmt(fp_downstream_attribution['avg_routercal_gap_vs_pair_frontier'])} |"
            ),
            (
                "| Qwen3 MoE downstream attribution | HumanEval recovery / scores beating pair frontier | "
                f"{fmt(fp_downstream_attribution['humaneval_routercal_recovery_fraction'])} / "
                f"{fp_downstream_attribution['routercal_beats_pair_frontier_count']}"
                f"/{fp_downstream_attribution['score_count']} |"
            ),
            (
                "| Qwen3 MoE downstream confidence | positive / confident-positive tasks vs naive | "
                f"{fp_downstream_confidence['routercal_positive_task_count_vs_naive']}"
                f"/{fp_downstream_confidence['task_count']} / "
                f"{fp_downstream_confidence['routercal_confident_positive_task_count_vs_naive']}"
                f"/{fp_downstream_confidence['task_count']} |"
            ),
            (
                "| Qwen3 MoE downstream confidence | confident source-frontier wins / avg gain interval | "
                f"{fp_downstream_confidence['routercal_confident_beats_pair_frontier_task_count']}"
                f"/{fp_downstream_confidence['task_count']} / "
                f"[{fmt(fp_downstream_confidence['routercal_avg_diff_lower_vs_naive'])}, "
                f"{fmt(fp_downstream_confidence['routercal_avg_diff_upper_vs_naive'])}] |"
            ),
            (
                "| first-principles MoE mechanism | gauge-equivalent B MSE | "
                f"{fp_moe['gauge_equivalence_mse']:.2e} |"
            ),
            (
                "| first-principles MoE mechanism | router agreement raw to aligned | "
                f"{fmt(fp_moe['router_agreement_raw'])} to {fmt(fp_moe['router_agreement_aligned'])} |"
            ),
            (
                "| first-principles MoE mechanism | same-name to aligned worst loss | "
                f"{fmt(fp_moe['uniform_same_name']['worst'])} to {fmt(fp_moe['uniform_aligned']['worst'])} |"
            ),
            (
                "| first-principles MoE mechanism | aligned + router-calibrated worst loss | "
                f"{fmt(fp_moe['uniform_aligned_routercal']['worst'])} |"
            ),
            (
                "| first-principles MoE mechanism | Fisher worst-loss reduction after alignment | "
                f"{fmt(fp_moe['route_conditioned_fisher'].get('worst_loss_reduction'))} |"
            ),
            (
                "| MoE probe-gated selector | global gauge rule | "
                f"{moe_selector['global_moe_gauge_decision']} |"
            ),
            (
                "| MoE probe-gated selector | Qwen3 expert identity decision | "
                f"{moe_selector['qwen3_expert_identity_decision']} |"
            ),
            (
                "| MoE probe-gated selector | next blocking probe | "
                f"{moe_selector['next_blocking_probe']} |"
            ),
            (
                "| Qwen3 MoE unified preflight | status | "
                f"{moe_unified_preflight['status']} |"
            ),
            (
                "| Qwen3 MoE unified preflight | same-shape / expert identity / CUDA | "
                f"{moe_unified_preflight['same_shape_contract_pass']} / "
                f"{moe_unified_preflight['expert_identity_status']} / "
                f"{moe_unified_preflight['cuda_available']} |"
            ),
            (
                "| Qwen3 MoE unified preflight | router / routed expert tensors | "
                f"{moe_unified_preflight['router_tensor_rows']} / "
                f"{moe_unified_preflight['routed_expert_tensor_rows']} |"
            ),
            (
                "| Qwen3 MoE routing probe | prompts / routers / overlap rows | "
                f"{qwen3_moe_routing_probe['prompt_count']} / "
                f"{qwen3_moe_routing_probe['router_count']} / "
                f"{qwen3_moe_routing_probe['route_overlap_rows']} |"
            ),
            (
                "| Qwen3 MoE routing probe | mean/min top-k Jaccard | "
                f"{fmt(qwen3_moe_routing_probe['mean_topk_jaccard'])} / "
                f"{fmt(qwen3_moe_routing_probe['min_topk_jaccard'])} |"
            ),
            (
                "| Qwen3 MoE routing probe | mean/min top1 agreement | "
                f"{fmt(qwen3_moe_routing_probe['mean_top1_agreement'])} / "
                f"{fmt(qwen3_moe_routing_probe['min_top1_agreement'])} |"
            ),
            (
                "| Qwen3 MoE routing readiness | status | "
                f"{qwen3_moe_routing_readiness['readiness_status']} |"
            ),
            (
                "| Qwen3 MoE routing readiness | calibrate / small-lambda / passed / freeze rows | "
                f"{qwen3_moe_routing_readiness['router_action_counts'].get('calibrate_router_before_average', 0)} / "
                f"{qwen3_moe_routing_readiness['router_action_counts'].get('small_lambda_router_with_overlap_guard', 0)} / "
                f"{qwen3_moe_routing_readiness['router_action_counts'].get('router_probe_passed_for_small_lambda', 0)} / "
                f"{qwen3_moe_routing_readiness['router_action_counts'].get('freeze_router_and_check_load_balance', 0)} |"
            ),
            (
                "| Qwen3 MoE route-guarded candidate | status / frozen router | "
                f"{qwen3_moe_route_guarded_candidate['recipe_status']} / "
                f"{qwen3_moe_route_guarded_candidate['freeze_router']} |"
            ),
            (
                "| Qwen3 MoE route-guarded candidate | expert rules / route rows used / skipped | "
                f"{qwen3_moe_route_guarded_candidate['expert_rule_count']} / "
                f"{qwen3_moe_route_guarded_candidate['source_route_conditioned_used_rows']} / "
                f"{qwen3_moe_route_guarded_candidate['source_route_conditioned_skipped_rows']} |"
            ),
            (
                "| Qwen3 MoE route-guarded candidate | manifest expert / attention / router hits | "
                f"{qwen3_moe_route_guarded_candidate['writer_dry_run_expert_tensor_rule_hits']} / "
                f"{qwen3_moe_route_guarded_candidate['writer_dry_run_shared_attention_hits']} / "
                f"{qwen3_moe_route_guarded_candidate['writer_dry_run_freeze_router_hits']} |"
            ),
            (
                "| Qwen3 MoE materialized delta audit | status / changed tensors / router changed | "
                f"{qwen3_moe_materialized_delta_audit['status']} / "
                f"{qwen3_moe_materialized_delta_audit['changed_tensors']} / "
                f"{qwen3_moe_materialized_delta_audit['router_changed_tensors']}"
                f"/{qwen3_moe_materialized_delta_audit['router_tensors']} |"
            ),
            (
                "| Qwen3 MoE materialized delta audit | changed numel frac / relative delta norm / max abs delta | "
                f"{fmt(qwen3_moe_materialized_delta_audit['changed_numel_fraction'])} / "
                f"{fmt(qwen3_moe_materialized_delta_audit['relative_delta_norm'])} / "
                f"{fmt(qwen3_moe_materialized_delta_audit['max_abs_delta'])} |"
            ),
            (
                "| Qwen3 MoE audit-gated candidate | status / scaled rules / dry-run router hits | "
                f"{qwen3_moe_audit_gated_candidate['status']} / "
                f"{qwen3_moe_audit_gated_candidate['scaled_expert_rule_count']} / "
                f"{qwen3_moe_audit_gated_candidate['writer_dry_run_freeze_router_hits']} |"
            ),
            (
                "| Qwen3 MoE audit-gated candidate | mean nonbase weight / max audited rel-delta / min scale | "
                f"{fmt(qwen3_moe_audit_gated_candidate['mean_original_effective_nonbase_weight'])}"
                f"->{fmt(qwen3_moe_audit_gated_candidate['mean_effective_nonbase_weight'])} / "
                f"{fmt(qwen3_moe_audit_gated_candidate['max_audit_relative_delta_before_cap'])} / "
                f"{fmt(qwen3_moe_audit_gated_candidate['min_audit_delta_scale'])} |"
            ),
            (
                "| Qwen3 MoE audit-gated delta audit | status / total relative norm / router changed | "
                f"{qwen3_moe_audit_gated_delta_audit['status']} / "
                f"{fmt(qwen3_moe_audit_gated_delta_audit['relative_delta_norm'])} / "
                f"{qwen3_moe_audit_gated_delta_audit['router_changed_tensors']}"
                f"/{qwen3_moe_audit_gated_delta_audit['router_tensors']} |"
            ),
            (
                "| Qwen3 MoE audit-gated delta audit | max routed rel-delta / routed tensors >1.0 / >0.75 | "
                f"{fmt(qwen3_moe_audit_gated_delta_audit['max_routed_tensor_relative_delta'])} / "
                f"{qwen3_moe_audit_gated_delta_audit['routed_tensors_relative_delta_gt_1']} / "
                f"{qwen3_moe_audit_gated_delta_audit['routed_tensors_relative_delta_gt_075']} |"
            ),
            (
                "| Qwen3 MoE trust-region candidate | status / scaled rules / beyond delta cap | "
                f"{qwen3_moe_trust_region_candidate['status']} / "
                f"{qwen3_moe_trust_region_candidate['scaled_expert_rule_count']} / "
                f"{qwen3_moe_trust_region_candidate['scaled_beyond_delta_cap_count']} |"
            ),
            (
                "| Qwen3 MoE trust-region candidate | estimated total rel-norm / max routed rel-delta / >0.75 | "
                f"{fmt(qwen3_moe_trust_region_candidate['estimated_relative_delta_norm'])} / "
                f"{fmt(qwen3_moe_trust_region_candidate['estimated_routed_tensor_max_relative_delta'])} / "
                f"{qwen3_moe_trust_region_candidate['estimated_routed_tensors_relative_delta_gt_075']} |"
            ),
            (
                "| Qwen3 MoE trust-region candidate | dry-run expert / attention / router hits | "
                f"{qwen3_moe_trust_region_candidate['writer_dry_run_expert_tensor_rule_hits']} / "
                f"{qwen3_moe_trust_region_candidate['writer_dry_run_shared_attention_hits']} / "
                f"{qwen3_moe_trust_region_candidate['writer_dry_run_freeze_router_hits']} |"
            ),
            (
                "| Qwen3 MoE trust-region delta audit | status / total relative norm / router changed | "
                f"{qwen3_moe_trust_region_delta_audit['status']} / "
                f"{fmt(qwen3_moe_trust_region_delta_audit['relative_delta_norm'])} / "
                f"{qwen3_moe_trust_region_delta_audit['router_changed_tensors']}"
                f"/{qwen3_moe_trust_region_delta_audit['router_tensors']} |"
            ),
            (
                "| Qwen3 MoE trust-region delta audit | max routed rel-delta / routed tensors >1.0 / >0.75 | "
                f"{fmt(qwen3_moe_trust_region_delta_audit['max_routed_tensor_relative_delta'])} / "
                f"{qwen3_moe_trust_region_delta_audit['routed_tensors_relative_delta_gt_1']} / "
                f"{qwen3_moe_trust_region_delta_audit['routed_tensors_relative_delta_gt_075']} |"
            ),
            (
                "| Qwen3 MoE trust-region delta validation | status / max abs pred error / p99 pred error | "
                f"{qwen3_moe_trust_region_delta_validation['status']} / "
                f"{fmt(qwen3_moe_trust_region_delta_validation['max_abs_relative_error'])} / "
                f"{fmt(qwen3_moe_trust_region_delta_validation['p99_abs_relative_error'])} |"
            ),
            (
                "| Qwen3 MoE trust-region delta validation | tensors above tolerance / actual >0.75 / rounding slop | "
                f"{qwen3_moe_trust_region_delta_validation['tensors_above_relative_tolerance']} / "
                f"{qwen3_moe_trust_region_delta_validation['routed_actual_relative_delta_gt_075']} / "
                f"{qwen3_moe_trust_region_delta_validation['routed_above_075_rounding_slop']} |"
            ),
            (
                "| Qwen3 MoE expert-only ablation | status / attention rule / dry-run router hits | "
                f"{qwen3_moe_expert_only_trust_region_candidate['status']} / "
                f"{qwen3_moe_expert_only_trust_region_candidate['attention_rule']} / "
                f"{qwen3_moe_expert_only_trust_region_candidate['writer_dry_run_freeze_router_hits']} |"
            ),
            (
                "| Qwen3 MoE expert-only ablation | materialized / shards / dry-run router hits | "
                f"{qwen3_moe_expert_only_trust_region_candidate['writer_checkpoint_materialized']} / "
                f"{qwen3_moe_expert_only_trust_region_candidate['writer_materialized_shards']} / "
                f"{qwen3_moe_expert_only_trust_region_candidate['writer_dry_run_freeze_router_hits']} |"
            ),
            (
                "| Qwen3 MoE expert-only ablation | estimated rel-norm / reduction / attention energy frac | "
                f"{fmt(qwen3_moe_expert_only_trust_region_candidate['estimated_expert_only_relative_delta_norm'])} / "
                f"{fmt(qwen3_moe_expert_only_trust_region_candidate['estimated_relative_delta_norm_reduction'])} / "
                f"{fmt(qwen3_moe_expert_only_trust_region_candidate['attention_delta_energy_fraction'])} |"
            ),
            (
                "| Qwen3 MoE expert-only ablation | frozen tensors / expert / attention hits | "
                f"{qwen3_moe_expert_only_trust_region_candidate['writer_dry_run_frozen_tensors']} / "
                f"{qwen3_moe_expert_only_trust_region_candidate['writer_dry_run_expert_tensor_rule_hits']} / "
                f"{qwen3_moe_expert_only_trust_region_candidate['writer_dry_run_shared_attention_hits']} |"
            ),
            (
                "| Qwen3 MoE expert-only delta audit | status / total relative norm / router changed | "
                f"{qwen3_moe_expert_only_delta_audit['status']} / "
                f"{fmt(qwen3_moe_expert_only_delta_audit['relative_delta_norm'])} / "
                f"{qwen3_moe_expert_only_delta_audit['router_changed_tensors']}"
                f"/{qwen3_moe_expert_only_delta_audit['router_tensors']} |"
            ),
            (
                "| Qwen3 MoE expert-only delta audit | max routed rel-delta / routed tensors >1.0 / >0.75 | "
                f"{fmt(qwen3_moe_expert_only_delta_audit['max_routed_tensor_relative_delta'])} / "
                f"{qwen3_moe_expert_only_delta_audit['routed_tensors_relative_delta_gt_1']} / "
                f"{qwen3_moe_expert_only_delta_audit['routed_tensors_relative_delta_gt_075']} |"
            ),
            (
                "| Qwen3 MoE tail-trimmed expert-only candidate | status / target cap / scaled groups | "
                f"{qwen3_moe_tail_trimmed_expert_only_candidate['status']} / "
                f"{fmt(qwen3_moe_tail_trimmed_expert_only_candidate['target_cap'])} / "
                f"{qwen3_moe_tail_trimmed_expert_only_candidate['scaled_expert_group_count']} |"
            ),
            (
                "| Qwen3 MoE tail-trimmed expert-only candidate | estimated rel-norm / routed max / >0.65 | "
                f"{fmt(qwen3_moe_tail_trimmed_expert_only_candidate['estimated_total_relative_delta_norm'])} / "
                f"{fmt(qwen3_moe_tail_trimmed_expert_only_candidate['estimated_routed_max_relative_delta'])} / "
                f"{qwen3_moe_tail_trimmed_expert_only_candidate['estimated_routed_tensors_gt_065']} |"
            ),
            (
                "| Qwen3 MoE tail-trimmed delta audit | status / total relative norm / router changed | "
                f"{qwen3_moe_tail_trimmed_delta_audit['status']} / "
                f"{fmt(qwen3_moe_tail_trimmed_delta_audit['relative_delta_norm'])} / "
                f"{qwen3_moe_tail_trimmed_delta_audit['router_changed_tensors']}"
                f"/{qwen3_moe_tail_trimmed_delta_audit['router_tensors']} |"
            ),
            (
                "| Qwen3 MoE tail-trimmed delta audit | max routed rel-delta / routed tensors >1.0 / >0.75 | "
                f"{fmt(qwen3_moe_tail_trimmed_delta_audit['max_routed_tensor_relative_delta'])} / "
                f"{qwen3_moe_tail_trimmed_delta_audit['routed_tensors_relative_delta_gt_1']} / "
                f"{qwen3_moe_tail_trimmed_delta_audit['routed_tensors_relative_delta_gt_075']} |"
            ),
            (
                "| Qwen3 MoE searched cap-law delta audit | status / total relative norm / router changed | "
                f"{qwen3_moe_searched_no_gt065_delta_audit['status']} / "
                f"{fmt(qwen3_moe_searched_no_gt065_delta_audit['relative_delta_norm'])} / "
                f"{qwen3_moe_searched_no_gt065_delta_audit['router_changed_tensors']}"
                f"/{qwen3_moe_searched_no_gt065_delta_audit['router_tensors']} |"
            ),
            (
                "| Qwen3 MoE searched cap-law delta audit | max routed rel-delta / >0.75 / >0.65 / >0.6505 | "
                f"{fmt(qwen3_moe_searched_no_gt065_delta_audit['max_routed_tensor_relative_delta'])} / "
                f"{qwen3_moe_searched_no_gt065_delta_audit['routed_tensors_relative_delta_gt_075']} / "
                f"{qwen3_moe_searched_no_gt065_delta_audit['routed_tensors_relative_delta_gt_065']} / "
                f"{qwen3_moe_searched_no_gt065_delta_audit['routed_tensors_relative_delta_gt_06505']} |"
            ),
            (
                "| Qwen3 MoE layer/chunk delta audit | status / total relative norm / router changed | "
                f"{qwen3_moe_layer_chunk_delta_audit['status']} / "
                f"{fmt(qwen3_moe_layer_chunk_delta_audit['relative_delta_norm'])} / "
                f"{qwen3_moe_layer_chunk_delta_audit['router_changed_tensors']}"
                f"/{qwen3_moe_layer_chunk_delta_audit['router_tensors']} |"
            ),
            (
                "| Qwen3 MoE layer/chunk delta audit | max routed rel-delta / >0.75 / >0.65 / >0.6505 | "
                f"{fmt(qwen3_moe_layer_chunk_delta_audit['max_routed_tensor_relative_delta'])} / "
                f"{qwen3_moe_layer_chunk_delta_audit['routed_tensors_relative_delta_gt_075']} / "
                f"{qwen3_moe_layer_chunk_delta_audit['routed_tensors_relative_delta_gt_065']} / "
                f"{qwen3_moe_layer_chunk_delta_audit['routed_tensors_relative_delta_gt_06505']} |"
            ),
            (
                "| Qwen3 MoE delta frontier | best safety candidate / next required gate | "
                f"{qwen3_moe_delta_frontier['best_delta_safety_candidate']} / "
                f"{qwen3_moe_delta_frontier['next_required_gate']} |"
            ),
            (
                "| Qwen3 MoE delta frontier | structural dominated / mechanistic nearest / distance | "
                f"{qwen3_moe_delta_frontier['structural_dominated_candidate_count']} / "
                f"{qwen3_moe_delta_frontier['mechanistic_nearest_structural_candidate']} / "
                f"{fmt(qwen3_moe_delta_frontier['mechanistic_nearest_structural_distance'])} |"
            ),
            (
                "| Qwen3 MoE delta frontier | audit->trust routed >0.75 reduction / trust->expert-only routed >0.75 reduction | "
                f"{qwen3_moe_delta_frontier['audit_to_trust_routed_gt_075_reduction']} / "
                f"{qwen3_moe_delta_frontier['trust_to_expert_only_routed_gt_075_reduction']} |"
            ),
            (
                "| Qwen3 MoE delta frontier | trust vs expert-only total rel-norm / attention norm reduction | "
                f"{fmt(qwen3_moe_delta_frontier['trust_region_total_relative_delta_norm'])}"
                f"->{fmt(qwen3_moe_delta_frontier['expert_only_total_relative_delta_norm'])} / "
                f"{fmt(qwen3_moe_delta_frontier['trust_to_expert_only_attention_norm_reduction'])} |"
            ),
            (
                "| Qwen3 MoE delta frontier | expert-only->tail-trimmed rel-norm reduction / routed >0.65 reduction | "
                f"{fmt(qwen3_moe_delta_frontier['expert_only_to_tail_trimmed_relative_norm_reduction'])} / "
                f"{qwen3_moe_delta_frontier['expert_only_to_tail_trimmed_routed_gt_065_reduction']} |"
            ),
            (
                "| Qwen3 MoE delta frontier | tail-trimmed vs searched rel-norm delta / >0.6505 counts | "
                f"{fmt(qwen3_moe_delta_frontier['tail_trimmed_to_searched_no_gt065_relative_norm_delta'])} / "
                f"{qwen3_moe_delta_frontier['tail_trimmed_routed_gt_0_6505']}"
                f"->{qwen3_moe_delta_frontier['searched_no_gt065_routed_gt_0_6505']} |"
            ),
            (
                "| Qwen3 MoE delta frontier | searched->layer/chunk rel-norm reduction / >0.65 reduction / >0.6505 | "
                f"{fmt(qwen3_moe_delta_frontier['searched_no_gt065_to_layer_chunk_relative_norm_reduction'])} / "
                f"{qwen3_moe_delta_frontier['searched_no_gt065_to_layer_chunk_routed_gt_065_reduction']} / "
                f"{qwen3_moe_delta_frontier['layer_chunk_routed_gt_0_6505']} |"
            ),
            (
                "| Qwen3 MoE delta frontier | unified matches searched / unified rel-norm / router changed | "
                f"{qwen3_moe_delta_frontier['unified_mechanism_matches_searched_no_gt065_delta']} / "
                f"{fmt(qwen3_moe_delta_frontier['unified_mechanism_total_relative_delta_norm'])} / "
                f"{qwen3_moe_delta_frontier['unified_mechanism_router_changed_tensors']} |"
            ),
            (
                "| Qwen3 MoE delta frontier | layer/chunk->unified rel-norm reduction / >0.65 reduction / unified >0.6505 | "
                f"{fmt(qwen3_moe_delta_frontier['layer_chunk_to_unified_relative_norm_reduction'])} / "
                f"{qwen3_moe_delta_frontier['layer_chunk_to_unified_routed_gt_065_reduction']} / "
                f"{qwen3_moe_delta_frontier['unified_mechanism_routed_gt_0_6505']} |"
            ),
            (
                "| Qwen3 MoE delta frontier | unified->mechanistic rel-norm reduction / >0.65 delta / mechanistic >0.6505 | "
                f"{fmt(qwen3_moe_delta_frontier['unified_to_mechanistic_relative_norm_reduction'])} / "
                f"{qwen3_moe_delta_frontier['unified_to_mechanistic_routed_gt_065_reduction']} / "
                f"{qwen3_moe_delta_frontier['mechanistic_unified_routed_gt_0_6505']} |"
            ),
            (
                "| Qwen3 MoE delta frontier | mechanistic->subspace rel-norm delta / >0.65 reduction / subspace >0.6505 | "
                f"{fmt(qwen3_moe_delta_frontier['mechanistic_to_subspace_relative_norm_delta'])} / "
                f"{qwen3_moe_delta_frontier['mechanistic_to_subspace_routed_gt_065_reduction']} / "
                f"{qwen3_moe_delta_frontier['subspace_scaled_routed_gt_0_6505']} |"
            ),
            (
                "| Qwen3 MoE mechanism eval gate | status / selection / selected | "
                f"{qwen3_moe_mechanism_eval_gate['status']} / "
                f"{qwen3_moe_mechanism_eval_gate['selection_status']} / "
                f"{qwen3_moe_mechanism_eval_gate['selected_method']} |"
            ),
            (
                "| Qwen3 MoE mechanism eval gate | ready / completed / awaiting tests | "
                f"{qwen3_moe_mechanism_eval_gate['ready_to_host_count']} / "
                f"{qwen3_moe_mechanism_eval_gate['completed_eval_count']} / "
                f"{qwen3_moe_mechanism_eval_gate['awaiting_tests']} |"
            ),
            (
                "| Qwen3 MoE mechanism eval gate | local GPU / best delta-safety candidate | "
                f"{qwen3_moe_mechanism_eval_gate['local_gpu_status']} / "
                f"{qwen3_moe_mechanism_eval_gate['best_delta_safety_candidate']} |"
            ),
            (
                "| Qwen3 MoE mechanism eval gate | unified serve / audit / optimizer test | "
                f"{qwen3_moe_mechanism_eval_gate['unified_candidate_serve_status']} / "
                f"{qwen3_moe_mechanism_eval_gate['unified_candidate_audit_passed']} / "
                f"{qwen3_moe_mechanism_eval_gate['unified_mechanism_optimizer_status']} |"
            ),
            (
                "| Qwen3 MoE eval budget plan | status / current -> recommended examples | "
                f"{qwen3_moe_eval_budget_plan['status']} / "
                f"{qwen3_moe_eval_budget_plan['current_gate_examples']} -> "
                f"{qwen3_moe_eval_budget_plan['recommended_max_examples']} |"
            ),
            (
                "| Qwen3 MoE eval budget plan | planned / ready / pending methods | "
                f"{qwen3_moe_eval_budget_plan['method_count']} / "
                f"{qwen3_moe_eval_budget_plan['ready_to_host_method_count']} / "
                f"{qwen3_moe_eval_budget_plan['pending_materialization_method_count']} |"
            ),
            (
                "| Qwen3 MoE eval budget plan | current / recommended / extra prompt budget | "
                f"{qwen3_moe_eval_budget_plan['total_current_prompt_budget']} / "
                f"{qwen3_moe_eval_budget_plan['total_recommended_prompt_budget']} / "
                f"{qwen3_moe_eval_budget_plan['total_additional_prompt_budget']} |"
            ),
            (
                "| Qwen3 MoE eval budget plan | ready current / recommended / extra prompt budget | "
                f"{qwen3_moe_eval_budget_plan['ready_to_host_current_prompt_budget']} / "
                f"{qwen3_moe_eval_budget_plan['ready_to_host_recommended_prompt_budget']} / "
                f"{qwen3_moe_eval_budget_plan['ready_to_host_additional_prompt_budget']} |"
            ),
            (
                "| Qwen3 MoE eval budget plan | default queue / final methods / final prompts | "
                f"{qwen3_moe_eval_budget_plan['default_runner_request']} / "
                f"{qwen3_moe_eval_budget_plan['final_core_method_count']} / "
                f"{qwen3_moe_eval_budget_plan['final_core_recommended_prompt_budget']} |"
            ),
            (
                "| Qwen3 MoE eval budget plan | mechanism ablation methods / prompts | "
                f"{qwen3_moe_eval_budget_plan['mechanism_ablation_method_count']} / "
                f"{qwen3_moe_eval_budget_plan['mechanism_ablation_recommended_prompt_budget']} |"
            ),
            (
                "| Qwen3 MoE eval budget plan | Wilson n / paired n / capped tasks | "
                f"{qwen3_moe_eval_budget_plan['wilson_required_examples']} / "
                f"{qwen3_moe_eval_budget_plan['paired_required_examples']} / "
                f"{qwen3_moe_eval_budget_plan['dataset_capped_tasks']} |"
            ),
            (
                "| Qwen3 MoE eval budget plan | task manifest aligned / canonical manifest | "
                f"{qwen3_moe_eval_budget_plan['task_manifest_aligned_method_count']}"
                f"/{qwen3_moe_eval_budget_plan['method_count']} / "
                f"{qwen3_moe_eval_budget_plan['canonical_task_manifest']} |"
            ),
            (
                "| Qwen3 MoE eval budget queue smoke | status / assertions | "
                f"{qwen3_moe_eval_budget_queue_smoke['status']} / "
                f"{qwen3_moe_eval_budget_queue_smoke['passed_assertion_count']}"
                f"/{qwen3_moe_eval_budget_queue_smoke['assertion_count']} |"
            ),
            (
                "| Qwen3 MoE eval budget queue smoke | final / mechanism / router methods | "
                f"{qwen3_moe_eval_budget_queue_smoke['final_queue_method_count']} / "
                f"{qwen3_moe_eval_budget_queue_smoke['mechanism_queue_method_count']} / "
                f"{qwen3_moe_eval_budget_queue_smoke['router_queue_method_count']} |"
            ),
            (
                "| Qwen3 MoE adaptive eval schedule | status / top action / top method | "
                f"{qwen3_moe_adaptive_eval_schedule['status']} / "
                f"{qwen3_moe_adaptive_eval_schedule['top_eval_action']} / "
                f"{qwen3_moe_adaptive_eval_schedule['top_method']} |"
            ),
            (
                "| Qwen3 MoE adaptive eval schedule | source controls / round1 probes / probe->full examples | "
                f"{qwen3_moe_adaptive_eval_schedule['source_controls_complete']} / "
                f"{qwen3_moe_adaptive_eval_schedule['round1_probe_candidate_count']} / "
                f"{qwen3_moe_adaptive_eval_schedule['probe_examples']} -> "
                f"{qwen3_moe_adaptive_eval_schedule['full_examples']} |"
            ),
            (
                "| Qwen3 MoE adaptive eval schedule | runnable methods / prompt budget / round1 probe prompts | "
                f"{qwen3_moe_adaptive_eval_schedule['runnable_method_count']} / "
                f"{qwen3_moe_adaptive_eval_schedule['runnable_prompt_budget']} / "
                f"{qwen3_moe_adaptive_eval_schedule['round1_probe_task_budget']} |"
            ),
            (
                "| Qwen3 MoE adaptive eval schedule | round1 policy / covered mechanism tests | "
                f"{qwen3_moe_adaptive_eval_schedule['round1_selection_policy']} / "
                f"{qwen3_moe_adaptive_eval_schedule['round1_covered_mechanism_test_count']} |"
            ),
            (
                "| Qwen3 MoE adaptive eval schedule | structural frontier / best structural method / score | "
                f"{qwen3_moe_adaptive_eval_schedule['structural_frontier_available']} / "
                f"{qwen3_moe_adaptive_eval_schedule['best_structural_method']} / "
                f"{fmt(qwen3_moe_adaptive_eval_schedule['best_structural_safety_score'])} |"
            ),
            (
                "| Qwen3 MoE adaptive eval schedule | structural dominance / frontier members / dominated methods | "
                f"{qwen3_moe_adaptive_eval_schedule['structural_dominance_available']} / "
                f"{qwen3_moe_adaptive_eval_schedule['structural_frontier_member_count']} / "
                f"{qwen3_moe_adaptive_eval_schedule['structurally_dominated_method_count']} |"
            ),
            (
                "| Qwen3 MoE adaptive eval schedule | paired gate status counts / alpha | "
                f"{qwen3_moe_adaptive_eval_schedule['paired_gate_status_counts']} / "
                f"{qwen3_moe_adaptive_eval_schedule['paired_alpha']:.3f} |"
            ),
            (
                "| Qwen3 MoE adaptive eval schedule smoke | status / assertions | "
                f"{qwen3_moe_adaptive_eval_schedule_smoke['status']} / "
                f"{qwen3_moe_adaptive_eval_schedule_smoke['passed_assertion_count']}"
                f"/{qwen3_moe_adaptive_eval_schedule_smoke['assertion_count']} |"
            ),
            (
                "| Qwen3 MoE eval manifest preflight | status / tasks sufficient / methods aligned | "
                f"{qwen3_moe_eval_manifest_preflight['status']} / "
                f"{qwen3_moe_eval_manifest_preflight['task_sufficient_count']}"
                f"/{qwen3_moe_eval_manifest_preflight['task_count']} / "
                f"{qwen3_moe_eval_manifest_preflight['task_manifest_aligned_method_count']}"
                f"/{qwen3_moe_eval_manifest_preflight['method_count']} |"
            ),
            (
                "| Qwen3 MoE eval budget plan | router active / ready / pending / plan-pruned caps | "
                f"{qwen3_moe_eval_budget_plan['router_calibration_active_candidate_count']} / "
                f"{qwen3_moe_eval_budget_plan['router_calibration_active_ready_count']} / "
                f"{qwen3_moe_eval_budget_plan['router_calibration_active_pending_count']} / "
                f"{qwen3_moe_eval_budget_plan['router_calibration_plan_pruned_candidate_count']} |"
            ),
            (
                "| Qwen3 MoE mechanism levers | top lever / priority / next test | "
                f"{qwen3_moe_mechanism_levers['top_lever']} / "
                f"{qwen3_moe_mechanism_levers['top_lever_priority']:.2f} / "
                f"{qwen3_moe_mechanism_levers['top_lever_next_test']} |"
            ),
            (
                "| Qwen3 MoE mechanism levers | fine calibration layers / top layer score | "
                f"{qwen3_moe_mechanism_levers['fine_calibration_layers']} / "
                f"{qwen3_moe_mechanism_levers['top_chunk_layer']}:"
                f"{qwen3_moe_mechanism_levers['top_chunk_score']:.3f} |"
            ),
            (
                "| Qwen3 MoE mechanism levers | expert geometry used / top geometry layer | "
                f"{qwen3_moe_mechanism_levers['expert_geometry_probe_used']} / "
                f"{qwen3_moe_mechanism_levers['top_expert_geometry_layer']}:"
                f"{fmt(qwen3_moe_mechanism_levers['top_expert_geometry_layer_risk'])} |"
            ),
            (
                "| Qwen3 MoE mechanism levers | expert subspace used / high / extra-scaled / top layer | "
                f"{qwen3_moe_mechanism_levers['expert_subspace_probe_used']} / "
                f"{qwen3_moe_mechanism_levers['high_subspace_conflict_expert_count']} / "
                f"{qwen3_moe_mechanism_levers['subspace_extra_scaled_expert_count']} / "
                f"{qwen3_moe_mechanism_levers['top_subspace_conflict_layer']} |"
            ),
            (
                "| Qwen3 MoE expert geometry probe | projection tensors / experts / layers | "
                f"{qwen3_moe_expert_geometry_probe['projection_tensor_count']} / "
                f"{qwen3_moe_expert_geometry_probe['expert_count']} / "
                f"{qwen3_moe_expert_geometry_probe['layer_count']} |"
            ),
            (
                "| Qwen3 MoE expert geometry probe | mean-p05 cosine / mean-p95 rel-delta | "
                f"{fmt(qwen3_moe_expert_geometry_probe['mean_projection_cosine'])}-"
                f"{fmt(qwen3_moe_expert_geometry_probe['p05_projection_cosine'])} / "
                f"{fmt(qwen3_moe_expert_geometry_probe['mean_projection_relative_delta'])}-"
                f"{fmt(qwen3_moe_expert_geometry_probe['p95_projection_relative_delta'])} |"
            ),
            (
                "| Qwen3 MoE expert geometry probe | high internal / route+geometry risk experts | "
                f"{qwen3_moe_expert_geometry_probe['high_internal_geometry_risk_expert_count']} / "
                f"{qwen3_moe_expert_geometry_probe['high_route_geometry_risk_expert_count']} |"
            ),
            (
                "| Qwen3 MoE expert geometry probe | top layer / top expert risk | "
                f"{qwen3_moe_expert_geometry_probe['top_layer_by_route_geometry_risk']} / "
                f"{qwen3_moe_expert_geometry_probe['top_expert_layer']}:"
                f"{qwen3_moe_expert_geometry_probe['top_expert_id']} "
                f"({fmt(qwen3_moe_expert_geometry_probe['top_expert_route_geometry_risk_score'])}) |"
            ),
            (
                "| Qwen3 MoE expert subspace conflict probe | projections / high / route-high / extra-scaled | "
                f"{qwen3_moe_expert_subspace_conflict_probe['projection_tensor_count']} / "
                f"{qwen3_moe_expert_subspace_conflict_probe['high_subspace_conflict_expert_count']} / "
                f"{qwen3_moe_expert_subspace_conflict_probe['route_important_high_subspace_conflict_expert_count']} / "
                f"{qwen3_moe_expert_subspace_conflict_probe['subspace_extra_scaled_expert_count']} |"
            ),
            (
                "| Qwen3 MoE expert subspace conflict probe | top layer / max conflict / coder reduction / next action | "
                f"{qwen3_moe_expert_subspace_conflict_probe['top_subspace_conflict_layer']} / "
                f"{fmt(qwen3_moe_expert_subspace_conflict_probe['max_subspace_conflict_score'])} / "
                f"{fmt(qwen3_moe_expert_subspace_conflict_probe['total_coder_weight_reduction'], 6)} / "
                f"{qwen3_moe_expert_subspace_conflict_probe['next_action']} |"
            ),
            (
                "| Qwen3 MoE expert subspace conflict probe | dry-run / floating / tensor-rule hits / frozen-router hits | "
                f"{qwen3_moe_expert_subspace_conflict_probe['dry_run_validated']} / "
                f"{qwen3_moe_expert_subspace_conflict_probe['dry_run_floating_tensors']} / "
                f"{qwen3_moe_expert_subspace_conflict_probe['dry_run_tensor_rule_hit_count']} / "
                f"{qwen3_moe_expert_subspace_conflict_probe['dry_run_freeze_router_hits']} |"
            ),
            (
                "| Qwen3 MoE layer/chunk candidate | schedule / feasible schedules / changed groups | "
                f"{qwen3_moe_layer_chunk_candidate['selected_schedule_id']} / "
                f"{qwen3_moe_layer_chunk_candidate['feasible_schedule_count']}"
                f"/{qwen3_moe_layer_chunk_candidate['schedule_count']} / "
                f"{qwen3_moe_layer_chunk_candidate['selected_changed_group_count']} |"
            ),
            (
                "| Qwen3 MoE layer/chunk candidate | retention / risk delta reduction / max rel-delta | "
                f"{fmt(qwen3_moe_layer_chunk_candidate['selected_route_mass_weighted_coder_retention'])} / "
                f"{fmt(qwen3_moe_layer_chunk_candidate['selected_risk_weighted_delta_reduction'])} / "
                f"{fmt(qwen3_moe_layer_chunk_candidate['selected_max_predicted_relative_delta'])} |"
            ),
            (
                "| Qwen3 MoE layer/chunk candidate | dry-run / floating / frozen / tensor-rule hits | "
                f"{qwen3_moe_layer_chunk_candidate['dry_run_validated']} / "
                f"{qwen3_moe_layer_chunk_candidate['dry_run_floating_tensors']} / "
                f"{qwen3_moe_layer_chunk_candidate['dry_run_frozen_tensors']} / "
                f"{qwen3_moe_layer_chunk_candidate['dry_run_tensor_rule_hit_count']} |"
            ),
            (
                "| Qwen3 MoE unified result selector | status / selected / reason | "
                f"{qwen3_moe_unified_result_selection['status']} / "
                f"{qwen3_moe_unified_result_selection['selected_method']} / "
                f"{qwen3_moe_unified_result_selection['selection_reason']} |"
            ),
            (
                "| Qwen3 MoE unified result selector | source complete / unified complete / eligible | "
                f"{qwen3_moe_unified_result_selection['sources_complete']} / "
                f"{qwen3_moe_unified_result_selection['unified_completed']} / "
                f"{qwen3_moe_unified_result_selection['eligible_candidate_count']}"
                f"/{qwen3_moe_unified_result_selection['candidate_count']} |"
            ),
            (
                "| Qwen3 MoE unified result selector smoke | status / passed cases | "
                f"{qwen3_moe_unified_result_selection_smoke['status']} / "
                f"{qwen3_moe_unified_result_selection_smoke['passed_case_count']}"
                f"/{qwen3_moe_unified_result_selection_smoke['case_count']} |"
            ),
            (
                "| Qwen3 MoE candidate trust-region gate | status / final-selectable / ablation-only | "
                f"{qwen3_moe_candidate_trust_region_gate['status']} / "
                f"{qwen3_moe_candidate_trust_region_gate['final_selectable_candidate_count']}"
                f"/{qwen3_moe_candidate_trust_region_gate['candidate_count']} / "
                f"{qwen3_moe_candidate_trust_region_gate['ablation_only_candidate_count']} |"
            ),
            (
                "| Qwen3 MoE candidate trust-region gate | strict cap / selected methods | "
                f"{fmt(qwen3_moe_candidate_trust_region_gate['strict_routed_max_relative_delta'])} / "
                f"{qwen3_moe_candidate_trust_region_gate['final_selectable_methods']} |"
            ),
            (
                "| Qwen3 MoE final candidate selector | status / selected / eligible | "
                f"{qwen3_moe_final_candidate_selection['status']} / "
                f"{qwen3_moe_final_candidate_selection['selected_method']} / "
                f"{qwen3_moe_final_candidate_selection['eligible_candidate_count']}"
                f"/{qwen3_moe_final_candidate_selection['candidate_count']} |"
            ),
            (
                "| Qwen3 MoE final candidate selector | usable / complete / best source | "
                f"{qwen3_moe_final_candidate_selection['usable_candidate_count']}"
                f"/{qwen3_moe_final_candidate_selection['candidate_count']} / "
                f"{qwen3_moe_final_candidate_selection['candidates_complete']} / "
                f"{qwen3_moe_final_candidate_selection['best_source_by_avg']} |"
            ),
            (
                "| Qwen3 MoE final candidate selector | uncertainty / paired gates / paired alpha | "
                f"{qwen3_moe_final_candidate_selection['uncertainty_gate']} / "
                f"{qwen3_moe_final_candidate_selection['paired_prediction_gate']} / "
                f"{qwen3_moe_final_candidate_selection['paired_alpha']:.3f} |"
            ),
            (
                "| Qwen3 MoE final candidate selector | structural frontier / dominated / safety / tie tolerance | "
                f"{qwen3_moe_final_candidate_selection['selected_structural_frontier_member']} / "
                f"{qwen3_moe_final_candidate_selection['selected_structurally_dominated']} / "
                f"{fmt(qwen3_moe_final_candidate_selection['selected_structural_safety_score'])} / "
                f"{fmt(qwen3_moe_final_candidate_selection['selection_score_tie_tolerance'])} |"
            ),
            (
                "| Qwen3 MoE final candidate selector | rank mode / confidence band / band size / point leader | "
                f"{qwen3_moe_final_candidate_selection['selection_rank_mode']} / "
                f"{qwen3_moe_final_candidate_selection['confidence_tie_band']} / "
                f"{qwen3_moe_final_candidate_selection['selection_rank_band_size']} / "
                f"{qwen3_moe_final_candidate_selection['selection_point_leader_method']} |"
            ),
            (
                "| Qwen3 MoE final candidate selector smoke | status / passed cases | "
                f"{qwen3_moe_final_candidate_selection_smoke['status']} / "
                f"{qwen3_moe_final_candidate_selection_smoke['passed_case_count']}"
                f"/{qwen3_moe_final_candidate_selection_smoke['case_count']} |"
            ),
            (
                "| unified average optimizer | status / dense / MoE | "
                f"{unified_average_optimizer['status']} / "
                f"{unified_average_optimizer['dense_decision']} / "
                f"{unified_average_optimizer['moe_decision']} |"
            ),
            (
                "| unified average optimizer | hypotheses / queue / top experiment | "
                f"{unified_average_optimizer['hypothesis_count']} / "
                f"{unified_average_optimizer['next_experiment_count']} / "
                f"{(unified_average_optimizer.get('top_next_experiment') or {}).get('experiment')} |"
            ),
            (
                "| unified average optimizer | evidence ledger / verdicts | "
                f"{unified_average_optimizer['evidence_ledger_count']} / "
                f"{unified_average_optimizer['evidence_verdict_counts']} |"
            ),
            (
                "| unified average optimizer | contract status / passed / blocked | "
                f"{unified_average_optimizer['contract_status']} / "
                f"{unified_average_optimizer['contract_passed_requirement_count']}"
                f"/{unified_average_optimizer['contract_requirement_count']} / "
                f"{unified_average_optimizer['contract_blocking_requirements']} |"
            ),
            (
                "| unified average optimizer ledger smoke | status / passed cases / assertions | "
                f"{unified_average_optimizer_ledger_smoke['status']} / "
                f"{unified_average_optimizer_ledger_smoke['passed_case_count']}"
                f"/{unified_average_optimizer_ledger_smoke['case_count']} / "
                f"{unified_average_optimizer_ledger_smoke['passed_assertion_count']}"
                f"/{unified_average_optimizer_ledger_smoke['assertion_count']} |"
            ),
            (
                "| unified average optimizer | dense linear / unified / endpoint worst NLL | "
                f"{unified_average_optimizer['dense_linear_worst_nll']:.3f} / "
                f"{unified_average_optimizer['dense_unified_worst_nll']:.3f} / "
                f"{unified_average_optimizer['dense_best_endpoint_worst_nll']:.3f} |"
            ),
            (
                "| unified average optimizer | dense lambda midpoint / best-family worst NLL | "
                f"{unified_average_optimizer['dense_lambda_linear_worst_nll']:.3f} / "
                f"{unified_average_optimizer['dense_lambda_best_worst_nll']:.3f} |"
            ),
            (
                "| unified average optimizer | real MoE gauge / router / Qwen3 final | "
                f"{unified_average_optimizer['real_gauge_naive_degradation']:.3f} -> "
                f"{unified_average_optimizer['real_gauge_aligned_degradation']:.3f} / "
                f"{unified_average_optimizer['router_action']} / "
                f"{unified_average_optimizer['qwen3_final_selection_status']} "
                f"({unified_average_optimizer['qwen3_eligible_candidates']}"
                f"/{unified_average_optimizer['qwen3_candidate_count']}) |"
            ),
            (
                "| unified average optimizer | Qwen3 unified candidate / subspace-delta / rule status | "
                f"{unified_average_optimizer['qwen3_unified_candidate_id']} / "
                f"{fmt(unified_average_optimizer['qwen3_unified_subspace_weighted_predicted_relative_delta'])} / "
                f"{unified_average_optimizer['qwen3_unified_materialized_rule_status']} |"
            ),
            (
                "| unified average optimizer | Qwen3 unified audit norm / >0.65 / manifest max diff | "
                f"{unified_average_optimizer['qwen3_unified_relative_delta_norm']:.3f} / "
                f"{unified_average_optimizer['qwen3_unified_routed_gt_065']} / "
                f"{fmt(unified_average_optimizer['qwen3_unified_max_manifest_weight_abs_diff'])} |"
            ),
            (
                "| unified average optimizer | final selector confidence band / rank mode / band size | "
                f"{unified_average_optimizer['qwen3_final_confidence_tie_band']} / "
                f"{unified_average_optimizer['qwen3_final_selection_rank_mode']} / "
                f"{unified_average_optimizer['qwen3_final_selection_rank_band_size']} |"
            ),
            (
                "| unified average optimizer | router margin high layers / top / min safe-lambda | "
                f"{unified_average_optimizer['qwen3_router_margin_high_fragility_layers']}"
                f"/{unified_average_optimizer['qwen3_router_margin_layer_count']} / "
                f"L{unified_average_optimizer['qwen3_router_margin_top_layer']} "
                f"{unified_average_optimizer['qwen3_router_margin_top_score']:.3f} / "
                f"{unified_average_optimizer['qwen3_router_margin_min_safe_lambda_proxy']:.3f} |"
            ),
            (
                "| unified average optimizer | Qwen3 MoE straight-line interior gap / general barrier | "
                f"{unified_average_optimizer['qwen3_interpolation_interior_gap_nll']:.3f} / "
                f"{unified_average_optimizer['qwen3_interpolation_general_barrier_nll']:.3f} |"
            ),
            (
                "| unified average optimizer | Qwen3 Base->Coder interior gap / complementary win | "
                f"{unified_average_optimizer['qwen3_base_coder_interior_gap_nll']:.3f} / "
                f"{unified_average_optimizer['qwen3_complementary_merge_beats_sources']} |"
            ),
            (
                "| unified average optimizer | layer/chunk->unified norm / >0.65 reduction | "
                f"{unified_average_optimizer['qwen3_layer_chunk_to_unified_relative_norm_reduction']:.3f} / "
                f"{unified_average_optimizer['qwen3_layer_chunk_to_unified_routed_gt_065_reduction']} |"
            ),
            (
                "| unified average optimizer | router calibration status / eligible | "
                f"{unified_average_optimizer['qwen3_router_calibration_status']} / "
                f"{unified_average_optimizer['qwen3_router_calibration_eligible_candidates']}"
                f"/{unified_average_optimizer['qwen3_router_calibration_candidate_count']} |"
            ),
            (
                "| unified average optimizer | router NLL probe worst reduction / code gap | "
                f"{unified_average_optimizer['qwen3_router_calibration_nll_worst_reduction']:.3f} / "
                f"{unified_average_optimizer['qwen3_router_calibration_nll_code_gap_to_best_source']:.3f} |"
            ),
            (
                "| Qwen3 MoE eval bundle audit | status / usable / invalid complete | "
                f"{qwen3_moe_eval_bundle_audit['status']} / "
                f"{qwen3_moe_eval_bundle_audit['usable_for_selection_count']}"
                f"/{qwen3_moe_eval_bundle_audit['method_count']} / "
                f"{qwen3_moe_eval_bundle_audit['invalid_complete_count']} |"
            ),
            (
                "| Qwen3 MoE eval bundle audit | source usable / candidate usable / unified usable | "
                f"{qwen3_moe_eval_bundle_audit['source_usable_count']}"
                f"/{qwen3_moe_eval_bundle_audit['source_count']} / "
                f"{qwen3_moe_eval_bundle_audit['candidate_usable_count']}"
                f"/{qwen3_moe_eval_bundle_audit['candidate_count']} / "
                f"{qwen3_moe_eval_bundle_audit['unified_usable']} |"
            ),
            (
                "| Qwen3 MoE eval bundle audit | pairable sources / failed methods | "
                f"{qwen3_moe_eval_bundle_audit['pairability_complete_source_count']} / "
                f"{qwen3_moe_eval_bundle_audit['pairability_failed_method_count']} |"
            ),
            (
                "| Qwen3 MoE eval bundle audit smoke | status / passed cases | "
                f"{qwen3_moe_eval_bundle_audit_smoke['status']} / "
                f"{qwen3_moe_eval_bundle_audit_smoke['passed_case_count']}"
                f"/{qwen3_moe_eval_bundle_audit_smoke['case_count']} |"
            ),
            (
                "| Qwen3 MoE mechanism attribution | status / scored / regressions | "
                f"{qwen3_moe_mechanism_effect_attribution['status']} / "
                f"{qwen3_moe_mechanism_effect_attribution['scored_transition_count']}"
                f"/{qwen3_moe_mechanism_effect_attribution['transition_count']} / "
                f"{qwen3_moe_mechanism_effect_attribution['regressing_transition_count']} |"
            ),
            (
                "| Qwen3 MoE mechanism attribution | best avg / best worst transition | "
                f"{qwen3_moe_mechanism_effect_attribution['best_avg_transition']} / "
                f"{qwen3_moe_mechanism_effect_attribution['best_worst_transition']} |"
            ),
            (
                "| Qwen3 MoE mechanism attribution smoke | status / passed cases | "
                f"{qwen3_moe_mechanism_effect_attribution_smoke['status']} / "
                f"{qwen3_moe_mechanism_effect_attribution_smoke['passed_case_count']}"
                f"/{qwen3_moe_mechanism_effect_attribution_smoke['case_count']} |"
            ),
            (
                "| Qwen3 MoE feedback optimizer | status / scored tasks / regressions / changed groups | "
                f"{qwen3_moe_feedback_optimizer['status']} / "
                f"{qwen3_moe_feedback_optimizer['scored_task_count']}"
                f"/{qwen3_moe_feedback_optimizer['task_count']} / "
                f"{qwen3_moe_feedback_optimizer['regression_task_count']} / "
                f"{qwen3_moe_feedback_optimizer['changed_group_count']} |"
            ),
            (
                "| Qwen3 MoE feedback optimizer | candidate / base selection / frontier-dominated | "
                f"{qwen3_moe_feedback_optimizer['candidate_method']} / "
                f"{qwen3_moe_feedback_optimizer['feedback_base_selection_status']} / "
                f"{qwen3_moe_feedback_optimizer['feedback_base_structural_frontier_member']}"
                f"-{qwen3_moe_feedback_optimizer['feedback_base_structurally_dominated']} |"
            ),
            (
                "| Qwen3 MoE feedback optimizer | feedback base candidates considered | "
                f"{qwen3_moe_feedback_optimizer['feedback_base_candidate_count']} |"
            ),
            (
                "| Qwen3 MoE feedback optimizer | materialization gate | "
                f"{qwen3_moe_feedback_optimizer['materialization_gate']} |"
            ),
            (
                "| Qwen3 MoE feedback optimizer | nonbase ratio / max expected delta / hard-cap violations | "
                f"{qwen3_moe_feedback_optimizer['route_mass_weighted_nonbase_ratio']:.3f} / "
                f"{qwen3_moe_feedback_optimizer['max_feedback_expected_relative_delta']:.3f} / "
                f"{qwen3_moe_feedback_optimizer['groups_over_hard_cap_after_feedback']} |"
            ),
            (
                "| Qwen3 MoE feedback optimizer smoke | status / passed cases | "
                f"{qwen3_moe_feedback_optimizer_smoke['status']} / "
                f"{qwen3_moe_feedback_optimizer_smoke['passed_case_count']}"
                f"/{qwen3_moe_feedback_optimizer_smoke['case_count']} |"
            ),
            (
                "| Qwen3 MoE mechanistic unified candidate | selected / candidates / feedback | "
                f"{qwen3_moe_mechanistic_unified_candidate['selected_candidate_id']} / "
                f"{qwen3_moe_mechanistic_unified_candidate['candidate_count']} / "
                f"{qwen3_moe_mechanistic_unified_candidate['feedback_status']} |"
            ),
            (
                "| Qwen3 MoE mechanistic unified candidate | nominal cap / effective cap / write margin | "
                f"{fmt(qwen3_moe_mechanistic_unified_candidate['nominal_hard_cap'])} / "
                f"{fmt(qwen3_moe_mechanistic_unified_candidate['effective_hard_cap'])} / "
                f"{fmt(qwen3_moe_mechanistic_unified_candidate['materialization_safety_margin'])} |"
            ),
            (
                "| Qwen3 MoE mechanistic unified candidate | retention / max rel-delta / hard-cap violations | "
                f"{fmt(qwen3_moe_mechanistic_unified_candidate['selected_nonbase_mass_retention'])} / "
                f"{fmt(qwen3_moe_mechanistic_unified_candidate['selected_max_predicted_relative_delta'])} / "
                f"{qwen3_moe_mechanistic_unified_candidate['selected_hard_cap_violation_count']} |"
            ),
            (
                "| Qwen3 MoE mechanistic unified candidate | risk-delta / benefit-scale / loss proxy | "
                f"{fmt(qwen3_moe_mechanistic_unified_candidate['selected_risk_weighted_predicted_delta'])} / "
                f"{fmt(qwen3_moe_mechanistic_unified_candidate['selected_benefit_weighted_scale'])} / "
                f"{fmt(qwen3_moe_mechanistic_unified_candidate['selected_mean_mechanistic_loss_proxy'])} |"
            ),
            (
                "| Qwen3 MoE mechanistic unified candidate | writer manifest / dry-run / tensor-rule hits / freeze-router hits | "
                f"{qwen3_moe_mechanistic_unified_candidate['writer_manifest_validated']} / "
                f"{qwen3_moe_mechanistic_unified_candidate['writer_manifest_dry_run']} / "
                f"{qwen3_moe_mechanistic_unified_candidate['dry_run_tensor_rule_hit_count']} / "
                f"{qwen3_moe_mechanistic_unified_candidate['dry_run_freeze_router_hits']} |"
            ),
            (
                "| Qwen3 MoE mechanistic evidence audit | gradient agree / objective improved / hard-cap bound | "
                f"{fmt(qwen3_moe_mechanistic_evidence_audit['gradient_sign_agreement_rate'])} / "
                f"{fmt(qwen3_moe_mechanistic_evidence_audit['objective_proxy_improved_group_fraction'])} / "
                f"{qwen3_moe_mechanistic_evidence_audit['hard_cap_bound_group_count']} |"
            ),
            (
                "| Qwen3 MoE mechanistic evidence audit | dominant binding / suppressing features | "
                f"{qwen3_moe_mechanistic_evidence_audit['dominant_binding']} / "
                f"{', '.join(qwen3_moe_mechanistic_evidence_audit['most_scale_suppressing_features'][:3])} |"
            ),
            (
                "| Qwen3 MoE mechanistic sensitivity | strongest objective / delta / reselected | "
                f"{qwen3_moe_mechanistic_sensitivity['strongest_objective_ablation']} / "
                f"{fmt(qwen3_moe_mechanistic_sensitivity['strongest_objective_delta'])} / "
                f"{qwen3_moe_mechanistic_sensitivity['strongest_objective_reselected_candidate_id']} |"
            ),
            (
                "| Qwen3 MoE mechanistic sensitivity | strongest scale / shift / top shrink feature | "
                f"{qwen3_moe_mechanistic_sensitivity['strongest_scale_ablation']} / "
                f"{fmt(qwen3_moe_mechanistic_sensitivity['strongest_scale_shift'], 4)} / "
                f"{qwen3_moe_mechanistic_sensitivity['top_shrink_feature']} "
                f"({fmt(qwen3_moe_mechanistic_sensitivity['top_shrink_feature_corr'], 3)}) |"
            ),
            (
                "| Qwen3 MoE router-expert coupling | gate / fragility->feature / fragility->shrink | "
                f"{qwen3_moe_router_expert_coupling['gate']} / "
                f"{fmt(qwen3_moe_router_expert_coupling['fragility_router_feature_corr'])} / "
                f"{fmt(qwen3_moe_router_expert_coupling['fragility_scale_shrink_corr'])} |"
            ),
            (
                "| Qwen3 MoE router-expert coupling | high-low shrink lift / top layer / high-low scale | "
                f"{fmt(qwen3_moe_router_expert_coupling['high_vs_low_weighted_shrink_lift'], 4)} / "
                f"L{qwen3_moe_router_expert_coupling['top_coupled_layer_id']} / "
                f"{fmt(qwen3_moe_router_expert_coupling['high_fragility_weighted_scale'])}-"
                f"{fmt(qwen3_moe_router_expert_coupling['low_fragility_weighted_scale'])} |"
            ),
            (
                "| Qwen3 MoE router-coupled candidate | gate / selected / changed groups | "
                f"{qwen3_moe_router_coupled_candidate['selection_gate']} / "
                f"{qwen3_moe_router_coupled_candidate['selected_candidate_id']} / "
                f"{qwen3_moe_router_coupled_candidate['selected_changed_group_count']} |"
            ),
            (
                "| Qwen3 MoE router-coupled candidate | retention delta / coupled delta reduction / risk reduction | "
                f"{fmt(qwen3_moe_router_coupled_candidate['selected_retention_delta_vs_mechanistic'], 4)} / "
                f"{fmt(qwen3_moe_router_coupled_candidate['selected_router_coupled_delta_reduction_vs_mechanistic'], 4)} / "
                f"{fmt(qwen3_moe_router_coupled_candidate['selected_risk_delta_reduction_vs_mechanistic'], 4)} |"
            ),
            (
                "| Qwen3 MoE router-coupled retention frontier | gate / constrained / stress | "
                f"{qwen3_moe_router_coupled_retention_frontier['gate']} / "
                f"{qwen3_moe_router_coupled_retention_frontier['constrained_candidate_id']} / "
                f"{qwen3_moe_router_coupled_retention_frontier['stress_candidate_id']} |"
            ),
            (
                "| Qwen3 MoE router-coupled retention frontier | pass default / effect fraction / action | "
                f"{qwen3_moe_router_coupled_retention_frontier['default_gate_candidate_count']}"
                f"/{qwen3_moe_router_coupled_retention_frontier['candidate_count']} / "
                f"{fmt(qwen3_moe_router_coupled_retention_frontier['constrained_effect_fraction_vs_stress'], 4)} / "
                f"{qwen3_moe_router_coupled_retention_frontier['recommended_unified_action']} |"
            ),
            (
                "| Qwen3 MoE mechanistic unified smoke | status / passed cases | "
                f"{qwen3_moe_mechanistic_unified_candidate_smoke['status']} / "
                f"{qwen3_moe_mechanistic_unified_candidate_smoke['passed_case_count']}"
                f"/{qwen3_moe_mechanistic_unified_candidate_smoke['case_count']} |"
            ),
            (
                "| Qwen3 MoE post-eval refresh | status / passed steps / audit usable | "
                f"{qwen3_moe_post_eval_refresh['status']} / "
                f"{qwen3_moe_post_eval_refresh['passed_step_count']}"
                f"/{qwen3_moe_post_eval_refresh['step_count']} / "
                f"{qwen3_moe_post_eval_refresh['audit_usable_for_selection']}"
                f"/{qwen3_moe_post_eval_refresh['audit_method_count']} |"
            ),
            (
                "| Qwen3 MoE post-eval refresh | selection / final selection / attribution scored / plan steps | "
                f"{qwen3_moe_post_eval_refresh['selection_status']} / "
                f"{qwen3_moe_post_eval_refresh['final_selection_status']} / "
                f"{qwen3_moe_post_eval_refresh['attribution_scored_transition_count']}"
                f"/{qwen3_moe_post_eval_refresh['attribution_transition_count']} / "
                f"{qwen3_moe_post_eval_refresh_plan['planned_step_count']}"
                f"/{qwen3_moe_post_eval_refresh_plan['step_count']} |"
            ),
            (
                "| Qwen3 MoE post-eval refresh | feedback status / scored tasks / changed groups | "
                f"{qwen3_moe_post_eval_refresh['feedback_status']} / "
                f"{qwen3_moe_post_eval_refresh['feedback_scored_task_count']}"
                f"/{qwen3_moe_post_eval_refresh['feedback_task_count']} / "
                f"{qwen3_moe_post_eval_refresh['feedback_changed_group_count']} |"
            ),
            (
                "| Qwen3 MoE post-eval refresh | mechanistic status / retention / hard-cap violations | "
                f"{qwen3_moe_post_eval_refresh['mechanistic_status']} / "
                f"{fmt(qwen3_moe_post_eval_refresh['mechanistic_retention'])} / "
                f"{qwen3_moe_post_eval_refresh['mechanistic_hard_cap_violations']} |"
            ),
            (
                "| Qwen3 MoE post-eval refresh | sensitivity objective / scale | "
                f"{qwen3_moe_post_eval_refresh['mechanistic_sensitivity_strongest_objective_ablation']} "
                f"{fmt(qwen3_moe_post_eval_refresh['mechanistic_sensitivity_strongest_objective_delta'])} / "
                f"{qwen3_moe_post_eval_refresh['mechanistic_sensitivity_strongest_scale_ablation']} "
                f"{fmt(qwen3_moe_post_eval_refresh['mechanistic_sensitivity_scale_shift'], 4)} |"
            ),
            (
                "| Qwen3 MoE post-eval refresh | router-expert coupling | "
                f"{qwen3_moe_post_eval_refresh['router_expert_coupling_gate']} / "
                f"{fmt(qwen3_moe_post_eval_refresh['router_expert_coupling_fragility_router_feature_corr'])} / "
                f"{fmt(qwen3_moe_post_eval_refresh['router_expert_coupling_fragility_scale_shrink_corr'])} / "
                f"L{qwen3_moe_post_eval_refresh['router_expert_coupling_top_layer']} |"
            ),
            (
                "| Qwen3 MoE post-eval refresh | router-coupled candidate | "
                f"{qwen3_moe_post_eval_refresh['router_coupled_candidate_selection_gate']} / "
                f"{qwen3_moe_post_eval_refresh['router_coupled_candidate_selected']} / "
                f"{fmt(qwen3_moe_post_eval_refresh['router_coupled_candidate_retention_delta'], 4)} / "
                f"{fmt(qwen3_moe_post_eval_refresh['router_coupled_candidate_delta_reduction'], 4)} |"
            ),
            (
                "| Qwen3 MoE router move gate | status / action / allowed layers | "
                f"{qwen3_moe_router_move_gate['status']} / "
                f"{qwen3_moe_router_move_gate['recommended_unified_router_action']} / "
                f"{qwen3_moe_router_move_gate['allowed_router_layer_count']}"
                f"/{qwen3_moe_router_move_gate['router_layer_count']} |"
            ),
            (
                "| Qwen3 MoE router move gate | unsafe / calibrate / freeze rows | "
                f"{qwen3_moe_router_move_gate['unsafe_readiness_rows']} / "
                f"{qwen3_moe_router_move_gate['calibrate_readiness_rows']} / "
                f"{qwen3_moe_router_move_gate['freeze_readiness_rows']} |"
            ),
            (
                "| Qwen3 MoE router move gate | router rel-norm / mean-min top-k Jaccard / min top1 | "
                f"{fmt(qwen3_moe_router_move_gate['total_router_relative_delta_norm'])} / "
                f"{fmt(qwen3_moe_router_move_gate['mean_topk_jaccard'])}"
                f"-{fmt(qwen3_moe_router_move_gate['min_topk_jaccard'])} / "
                f"{fmt(qwen3_moe_router_move_gate['min_top1_agreement'])} |"
            ),
            (
                "| Qwen3 MoE router margin fragility | status / high-fragility layers / top layer | "
                f"{qwen3_moe_router_margin_fragility['status']} / "
                f"{qwen3_moe_router_margin_fragility['high_fragility_layer_count']}"
                f"/{qwen3_moe_router_margin_fragility['router_layer_count']} / "
                f"L{qwen3_moe_router_margin_fragility['top_fragile_layer']} |"
            ),
            (
                "| Qwen3 MoE router margin fragility | top score / min safe-lambda proxy / top category | "
                f"{fmt(qwen3_moe_router_margin_fragility['top_fragility_score'])} / "
                f"{fmt(qwen3_moe_router_margin_fragility['min_safe_lambda_proxy'])} / "
                f"{qwen3_moe_router_margin_fragility['top_fragile_category']} |"
            ),
            (
                "| Qwen3 MoE router calibration NLL probe | status / worst / avg reduction | "
                f"{qwen3_moe_router_calibration_nll_probe['status']} / "
                f"{fmt(qwen3_moe_router_calibration_nll_probe['worst_nll_reduction_vs_linear'])} / "
                f"{fmt(qwen3_moe_router_calibration_nll_probe['avg_nll_reduction_vs_linear'])} |"
            ),
            (
                "| Qwen3 MoE router calibration NLL probe | code gap / worst gap to best source | "
                f"{fmt(qwen3_moe_router_calibration_nll_probe['routercal_code_gap_to_best_source'])} / "
                f"{fmt(qwen3_moe_router_calibration_nll_probe['routercal_worst_gap_to_best_source'])} |"
            ),
            (
                "| Qwen3 MoE router calibration job | status / local GPU / candidates / stages | "
                f"{qwen3_moe_router_calibration_job['status']} / "
                f"{qwen3_moe_router_calibration_job['local_gpu_status']} / "
                f"{qwen3_moe_router_calibration_job['candidate_count']} / "
                f"{qwen3_moe_router_calibration_job['stage_count']} |"
            ),
            (
                "| Qwen3 MoE router calibration job | source controls / ready | "
                f"{qwen3_moe_router_calibration_job['source_control_count']} / "
                f"{qwen3_moe_router_calibration_job['source_controls_ready']} |"
            ),
            (
                "| Qwen3 MoE router calibration job | task manifest / create-if-missing | "
                f"{qwen3_moe_router_calibration_job['task_manifest']} / "
                f"{qwen3_moe_router_calibration_job['create_task_manifest_if_missing']} |"
            ),
            (
                "| Qwen3 MoE router calibration job | margin safe-lambda / planned-pass caps | "
                f"{fmt(qwen3_moe_router_calibration_job['router_margin_safe_lambda_proxy'])} / "
                f"{qwen3_moe_router_calibration_job['router_margin_planned_pass_count']}"
                f"/{qwen3_moe_router_calibration_job['candidate_count']} |"
            ),
            (
                "| Qwen3 MoE router calibration job | default-run caps | "
                f"{qwen3_moe_router_calibration_job['default_run_candidate_count']}"
                f"/{qwen3_moe_router_calibration_job['candidate_count']} |"
            ),
            (
                "| Qwen3 MoE router calibration job | margin-profile enabled / cap rows / min-mean-max | "
                f"{qwen3_moe_router_calibration_job['router_margin_profile_enabled']} / "
                f"{qwen3_moe_router_calibration_job['router_margin_profile_cap_rows']} / "
                f"{fmt(qwen3_moe_router_calibration_job['router_margin_profile_min_cap'])}-"
                f"{fmt(qwen3_moe_router_calibration_job['router_margin_profile_mean_cap'])}-"
                f"{fmt(qwen3_moe_router_calibration_job['router_margin_profile_max_cap'])} |"
            ),
            (
                "| Qwen3 MoE router calibration job | inputs student / teacher / prompts | "
                f"{qwen3_moe_router_calibration_job['student_exists']} / "
                f"{qwen3_moe_router_calibration_job['teacher_exists']} / "
                f"{qwen3_moe_router_calibration_job['prompts_exists']} |"
            ),
            (
                "| Qwen3 MoE router calibration selector | status / selected / eligible | "
                f"{qwen3_moe_router_calibration_selection['status']} / "
                f"{qwen3_moe_router_calibration_selection['selected_method']} / "
                f"{qwen3_moe_router_calibration_selection['eligible_candidate_count']}"
                f"/{qwen3_moe_router_calibration_selection['candidate_count']} |"
            ),
            (
                "| Qwen3 MoE router calibration selector | source required-complete / baseline eval / candidate eval / audit | "
                f"{qwen3_moe_router_calibration_selection['source_eval_required']}-"
                f"{qwen3_moe_router_calibration_selection['source_eval_completed']} / "
                f"{qwen3_moe_router_calibration_selection['baseline_eval_completed']} / "
                f"{qwen3_moe_router_calibration_selection['candidate_eval_completed']} / "
                f"{qwen3_moe_router_calibration_selection['audit_completed']} |"
            ),
            (
                "| Qwen3 MoE router calibration selector | training / hard route-load / group validation | "
                f"{qwen3_moe_router_calibration_selection['training_completed']} / "
                f"{qwen3_moe_router_calibration_selection['capacity_metrics_completed']} / "
                f"{qwen3_moe_router_calibration_selection['group_validation_completed']} |"
            ),
            (
                "| Qwen3 MoE router calibration selector | margin gate / safe-lambda / high layers | "
                f"{qwen3_moe_router_calibration_selection['router_margin_gate_completed']} / "
                f"{fmt(qwen3_moe_router_calibration_selection['router_margin_min_safe_lambda_proxy'])} / "
                f"{qwen3_moe_router_calibration_selection['router_margin_high_fragility_layer_count']}"
                f"/{qwen3_moe_router_calibration_selection['router_margin_layer_count']} |"
            ),
            (
                "| Qwen3 MoE router calibration selector | active / plan-pruned candidates | "
                f"{qwen3_moe_router_calibration_selection['active_candidate_count']} / "
                f"{qwen3_moe_router_calibration_selection['plan_pruned_candidate_count']} |"
            ),
            (
                "| Qwen3 MoE router row-validation negative smoke | status / eligible / group validation | "
                f"{qwen3_moe_router_calibration_row_validation_negative_smoke['status']} / "
                f"{qwen3_moe_router_calibration_row_validation_negative_smoke['eligible_candidate_count']}"
                f"/{qwen3_moe_router_calibration_row_validation_negative_smoke['candidate_count']} / "
                f"{qwen3_moe_router_calibration_row_validation_negative_smoke['group_validation_completed']} |"
            ),
            (
                "| Qwen3 MoE router row-validation negative smoke | first decision reason | "
                f"{matching_decision_reason(qwen3_moe_router_calibration_row_validation_negative_rows, 'router_validation_not_group_heldout')} |"
            ),
            (
                "| Qwen3 MoE router source-dominance negative smoke | status / selected / eligible | "
                f"{qwen3_moe_router_calibration_source_dominance_negative_smoke['status']} / "
                f"{qwen3_moe_router_calibration_source_dominance_negative_smoke['selected_method']} / "
                f"{qwen3_moe_router_calibration_source_dominance_negative_smoke['eligible_candidate_count']}"
                f"/{qwen3_moe_router_calibration_source_dominance_negative_smoke['candidate_count']} |"
            ),
            (
                "| Qwen3 MoE router source-dominance negative smoke | first decision reason | "
                f"{matching_decision_reason(qwen3_moe_router_calibration_source_dominance_rows, 'source_endpoint_dominates')} |"
            ),
            (
                "| Qwen3 MoE router no-gain negative smoke | status / selected / eligible | "
                f"{qwen3_moe_router_calibration_no_gain_negative_smoke['status']} / "
                f"{qwen3_moe_router_calibration_no_gain_negative_smoke['selected_method']} / "
                f"{qwen3_moe_router_calibration_no_gain_negative_smoke['eligible_candidate_count']}"
                f"/{qwen3_moe_router_calibration_no_gain_negative_smoke['candidate_count']} |"
            ),
            (
                "| Qwen3 MoE router no-gain negative smoke | first decision reason | "
                f"{matching_decision_reason(qwen3_moe_router_calibration_no_gain_rows, 'no_downstream_gain')} |"
            ),
            (
                "| Qwen3 MoE router task-regression negative smoke | status / selected / eligible | "
                f"{qwen3_moe_router_calibration_task_regression_negative_smoke['status']} / "
                f"{qwen3_moe_router_calibration_task_regression_negative_smoke['selected_method']} / "
                f"{qwen3_moe_router_calibration_task_regression_negative_smoke['eligible_candidate_count']}"
                f"/{qwen3_moe_router_calibration_task_regression_negative_smoke['candidate_count']} |"
            ),
            (
                "| Qwen3 MoE router task-regression negative smoke | first decision reason | "
                f"{matching_decision_reason(qwen3_moe_router_calibration_task_regression_rows, 'task_score_regression')} |"
            ),
            (
                "| Qwen3 MoE router selector matrix smoke | status / passed cases | "
                f"{qwen3_moe_router_calibration_selector_matrix_smoke['status']} / "
                f"{qwen3_moe_router_calibration_selector_matrix_smoke['passed_case_count']}"
                f"/{qwen3_moe_router_calibration_selector_matrix_smoke['case_count']} |"
            ),
            (
                "| Qwen3 MoE cap-law search | searched / frontier / expert groups | "
                f"{qwen3_moe_trust_region_cap_search['searched_law_count']} / "
                f"{qwen3_moe_trust_region_cap_search['pareto_frontier_count']} / "
                f"{qwen3_moe_trust_region_cap_search['expert_group_count']} |"
            ),
            (
                "| Qwen3 MoE cap-law search | current trust vs uniform 0.65 retention | "
                f"{fmt(qwen3_moe_trust_region_cap_search['current_trust_retention'])} / "
                f"{fmt(qwen3_moe_trust_region_cap_search['uniform_065_retention'])} |"
            ),
            (
                "| Qwen3 MoE cap-law search | current trust vs uniform 0.65 >0.65 groups | "
                f"{qwen3_moe_trust_region_cap_search['current_trust_routed_gt_065_groups']} / "
                f"{qwen3_moe_trust_region_cap_search['uniform_065_routed_gt_065_groups']} |"
            ),
            (
                "| Qwen3 MoE cap-law search | extra risk penalties threshold-efficient | "
                f"{qwen3_moe_trust_region_cap_search['current_extra_risk_penalties_delta_threshold_efficient']} |"
            ),
            (
                "| Qwen3 MoE cap-law search | validated dry-run rules / expert hits / router hits | "
                f"{qwen3_moe_trust_region_cap_search['dry_run_validated_rule_count']} / "
                f"{qwen3_moe_trust_region_cap_search['max_dry_run_expert_rule_hits']} / "
                f"{qwen3_moe_trust_region_cap_search['max_dry_run_freeze_router_hits']} |"
            ),
            (
                "| Qwen3 MoE unified mechanism candidate | selected / family / candidates | "
                f"{qwen3_moe_unified_mechanism_candidate['selected_candidate_id']} / "
                f"{qwen3_moe_unified_mechanism_candidate['selected_candidate_family']} / "
                f"{qwen3_moe_unified_mechanism_candidate['candidate_count']} |"
            ),
            (
                "| Qwen3 MoE unified mechanism candidate | retention / max rel-delta / hard-cap violations | "
                f"{fmt(qwen3_moe_unified_mechanism_candidate['selected_nonbase_mass_retention'])} / "
                f"{fmt(qwen3_moe_unified_mechanism_candidate['selected_max_predicted_relative_delta'])} / "
                f"{qwen3_moe_unified_mechanism_candidate['selected_routed_gt_hard_cap_groups']} |"
            ),
            (
                "| Qwen3 MoE unified mechanism candidate | risk-delta / geometry-delta / subspace-delta | "
                f"{fmt(qwen3_moe_unified_mechanism_candidate['selected_risk_weighted_predicted_relative_delta'])} / "
                f"{fmt(qwen3_moe_unified_mechanism_candidate['selected_geometry_weighted_predicted_relative_delta'])} / "
                f"{fmt(qwen3_moe_unified_mechanism_candidate['selected_subspace_weighted_predicted_relative_delta'])} |"
            ),
            (
                "| Qwen3 MoE unified mechanism candidate | geometry used / subspace used / high-subspace scale | "
                f"{qwen3_moe_unified_mechanism_candidate['expert_geometry_probe_used']} / "
                f"{qwen3_moe_unified_mechanism_candidate['subspace_conflict_probe_used']} / "
                f"{fmt(qwen3_moe_unified_mechanism_candidate['selected_high_subspace_mean_scale'])} |"
            ),
            (
                "| Qwen3 MoE unified mechanism candidate | router / attention policy | "
                f"{qwen3_moe_unified_mechanism_candidate['router_policy']} / "
                f"{qwen3_moe_unified_mechanism_candidate['shared_attention_policy']} |"
            ),
            (
                "| Qwen3 MoE unified mechanism candidate | materialized rules / manifest match / max diff | "
                f"{qwen3_moe_unified_mechanism_candidate['materialized_checkpoint_rule_status']} / "
                f"{qwen3_moe_unified_mechanism_candidate['matches_materialized_checkpoint_manifest']} / "
                f"{fmt(qwen3_moe_unified_mechanism_candidate['max_materialized_checkpoint_weight_abs_diff'])} |"
            ),
            (
                "| Qwen3 MoE unified mechanism candidate | matches validated no-gt-0.65 rules / max diff | "
                f"{qwen3_moe_unified_mechanism_candidate['matches_validated_reference_rules']} / "
                f"{fmt(qwen3_moe_unified_mechanism_candidate['max_reference_weight_abs_diff'])} |"
            ),
            (
                "| real MoE gauge self-merge | baseline / same-name / aligned NLL | "
                f"{fmt(fp_moe_real['baseline_nll'])} / "
                f"{fmt(fp_moe_real['naive_same_name_average_nll'])} / "
                f"{fmt(fp_moe_real['aligned_average_nll'])} |"
            ),
            (
                "| real MoE gauge self-merge | same-name degradation vs baseline | "
                f"{fmt(fp_moe_real['naive_degradation_vs_baseline'])} |"
            ),
            (
                "| real MoE gauge self-merge | recovered expert permutations | "
                f"{fp_moe_real['layers_perm_recovered']} / {fp_moe_real['n_moe_layers']} |"
            ),
            (
                "| real Qwen3 MoE correspondence | identity-optimal layers / mean diag cosine | "
                f"{fmt(fp_moe_real['qwen3_frac_layers_identity_optimal'])} / "
                f"{fmt(fp_moe_real['qwen3_mean_diag_cos'])} |"
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
                "| vLLM checkpoint eval plan | unified serve / eval output | "
                f"{vllm_checkpoint_eval_plan['unified_candidate_serve_status']} / "
                f"{vllm_checkpoint_eval_plan['unified_candidate_eval_output_dir']} |"
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
                "| Qwen dense broad sparse-method candidate | selected tensors / applied sparse rules / vLLM avg / delta vs global | "
                f"{qwen_dense_sparse_method['selected_tensor_count']} / "
                f"{qwen_dense_sparse_method['dry_run_tensor_method_applied_count']} / "
                f"{fmt(qwen_dense_sparse_method['vllm_avg_primary_score'])} / "
                f"{fmt(qwen_dense_sparse_method['vllm_delta_vs_base_bridge_avg_primary'])} |"
            ),
            (
                "| Qwen dense attention sparse-method candidate | selected tensors / applied sparse rules / vLLM avg / delta vs global | "
                f"{qwen_dense_attention_sparse_method['selected_tensor_count']} / "
                f"{qwen_dense_attention_sparse_method['dry_run_tensor_method_applied_count']} / "
                f"{fmt(qwen_dense_attention_sparse_method['vllm_avg_primary_score'])} / "
                f"{fmt(qwen_dense_attention_sparse_method['vllm_delta_vs_base_bridge_avg_primary'])} |"
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
                "| checkpoint materialization readiness | unified writer / vLLM / end-to-end | "
                f"{materialization_readiness['unified_candidate_writer_status']} / "
                f"{materialization_readiness['unified_candidate_vllm_plan_status']} / "
                f"{materialization_readiness['unified_candidate_end_to_end_status']} |"
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
                "| probe-gated unified average | dense default action | "
                f"{probe_gated_unified['dense_default_action']} |"
            ),
            (
                "| probe-gated unified average | dense bridge delta / module-guard delta | "
                f"{probe_gated_unified['dense_bridge_avg_primary_delta_vs_uniform']:.3f} / "
                f"{probe_gated_unified['dense_module_guard_delta_vs_bridge']:.3f} |"
            ),
            (
                "| probe-gated unified average | MoE default action | "
                f"{probe_gated_unified['moe_default_action']} |"
            ),
            (
                "| probe-gated unified average | MoE expert gain / overflow delta / real blocker | "
                f"{probe_gated_unified['moe_expert_identity_soft_worst_acc_gain']:.3f} / "
                f"{probe_gated_unified['moe_capacity_topk_overflow_delta']:.3f} / "
                f"{probe_gated_unified['real_qwen_moe_blocker']} |"
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
                "| average method gate matrix | accepted / rejected-default / conditional / active / required | "
                f"{average_method_gate_matrix['accepted_by_default_count']} / "
                f"{average_method_gate_matrix['default_rejected_count']} / "
                f"{average_method_gate_matrix['conditional_count']} / "
                f"{average_method_gate_matrix['active_lever_count']} / "
                f"{average_method_gate_matrix['required_precondition_count']} |"
            ),
            (
                "| average method gate matrix | dense midpoint / best-family / Qwen3 interior gap | "
                f"{fmt(average_method_gate_matrix['dense_lambda_linear_worst_nll'])} / "
                f"{fmt(average_method_gate_matrix['dense_lambda_best_worst_nll'])} / "
                f"{fmt(average_method_gate_matrix['qwen3_interpolation_interior_gap_nll'])} |"
            ),
            (
                "| average trust-region bounds | status / constraints / passed-rejected-waiting | "
                f"{average_trust_region_bounds['status']} / "
                f"{average_trust_region_bounds['constraint_count']} / "
                f"{average_trust_region_bounds['passed_count']}-"
                f"{average_trust_region_bounds['rejected_count']}-"
                f"{average_trust_region_bounds['waiting_count']} |"
            ),
            (
                "| average trust-region bounds | Dense lambda bound / safe uniform lambda / router safe lambda | "
                f"{fmt(average_trust_region_bounds['dense_local_task_vector_lambda_bound'])} / "
                f"{fmt(average_trust_region_bounds['dense_safe_uniform_lambda'])} / "
                f"{fmt(average_trust_region_bounds['moe_router_safe_lambda_proxy'])} |"
            ),
            (
                "| average trust-region bounds | router midpoint over bound / mechanistic cap / selected max delta | "
                f"{fmt(average_trust_region_bounds['moe_direct_router_average_over_safe_bound'])} / "
                f"{fmt(average_trust_region_bounds['mechanistic_effective_expert_delta_cap'])} / "
                f"{fmt(average_trust_region_bounds['mechanistic_selected_max_predicted_relative_delta'])} |"
            ),
            (
                "| average trust-region bounds smoke | status / assertions | "
                f"{average_trust_region_bounds_smoke['status']} / "
                f"{average_trust_region_bounds_smoke['passed_assertion_count']}"
                f"/{average_trust_region_bounds_smoke['assertion_count']} |"
            ),
            (
                "| average connectivity diagnostic | path rejected / midpoint rejected / frontier wins | "
                f"{average_connectivity['path_rejected_count']}/{average_connectivity['case_count']} / "
                f"{average_connectivity['midpoint_rejected_count']}/{average_connectivity['case_count']} / "
                f"{average_connectivity['endpoint_frontier_win_count']} |"
            ),
            (
                "| average connectivity diagnostic | Dense midpoint gap / Dense anchor gap / Qwen3 MoE gap | "
                f"{fmt(average_connectivity['dense_source_midpoint_gap'])} / "
                f"{fmt(average_connectivity['dense_lambda_endpoint_gap'])} / "
                f"{fmt(average_connectivity['qwen3_instruct_coder_gap'])} |"
            ),
            (
                "| average invariant audit | invariants / hard blockers / default accepted methods | "
                f"{average_invariant_audit['invariant_count']} / "
                f"{average_invariant_audit['hard_gate_blocker_count']} / "
                f"{average_invariant_audit['default_accepted_method_count']} |"
            ),
            (
                "| average invariant audit | same-shape / router allowed layers / final selector | "
                f"{average_invariant_audit['same_shape_contract_pass']} / "
                f"{average_invariant_audit['router_allowed_layers']}"
                f"/{average_invariant_audit['router_layer_count']} / "
                f"{average_invariant_audit['final_selection_status']} |"
            ),
            (
                "| average invariant audit | selected candidate / retention / predicted max delta | "
                f"{average_invariant_audit['selected_candidate_id']} / "
                f"{fmt(average_invariant_audit['selected_nonbase_mass_retention'])} / "
                f"{fmt(average_invariant_audit['selected_max_predicted_relative_delta'])} |"
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
                "| dense sparse-method writer smoke | status | "
                f"{dense_sparse_writer_smoke['status']} |"
            ),
            (
                "| dense sparse-method writer smoke | checked / failed tensors / method rules | "
                f"{dense_sparse_writer_smoke['checked_tensors']} / "
                f"{dense_sparse_writer_smoke['failed_tensors']} / "
                f"{dense_sparse_writer_smoke['tensor_method_rule_count']} |"
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
                "| MoE tensor-rule writer smoke | safetensors tensor delta tensors / values | "
                f"{moe_tensor_rule_writer_smoke['tensor_delta_safetensors_tensors']} / "
                f"{moe_tensor_rule_writer_smoke['tensor_delta_safetensors_values']} |"
            ),
            (
                "| MoE router delta calibration smoke | status / routers / delta tensors | "
                f"{moe_router_delta_calibration_smoke['status']} / "
                f"{moe_router_delta_calibration_smoke['router_count']} / "
                f"{moe_router_delta_calibration_smoke['delta_tensor_count']} |"
            ),
            (
                "| MoE router delta calibration smoke | route KL initial-final / top1 initial-final / max rel delta | "
                f"{fmt(moe_router_delta_calibration_smoke['mean_initial_route_kl'], 4)}-"
                f"{fmt(moe_router_delta_calibration_smoke['mean_final_route_kl'], 4)} / "
                f"{fmt(moe_router_delta_calibration_smoke['mean_initial_top1_agreement'], 4)}-"
                f"{fmt(moe_router_delta_calibration_smoke['mean_final_top1_agreement'], 4)} / "
                f"{fmt(moe_router_delta_calibration_smoke['max_final_relative_delta_norm'], 4)} |"
            ),
            (
                "| MoE router delta calibration smoke | cap mode / min-mean-max cap / max utilization | "
                f"{moe_router_delta_calibration_smoke['router_cap_mode']} / "
                f"{fmt(moe_router_delta_calibration_smoke['min_router_relative_norm_cap'], 4)}-"
                f"{fmt(moe_router_delta_calibration_smoke['mean_router_relative_norm_cap'], 4)}-"
                f"{fmt(moe_router_delta_calibration_smoke['max_router_relative_norm_cap'], 4)} / "
                f"{fmt(moe_router_delta_calibration_smoke['max_cap_utilization'], 4)} |"
            ),
            (
                "| MoE router delta calibration smoke | selection policy-split / selected epoch / score | "
                f"{moe_router_delta_calibration_smoke['selection_policy']}-"
                f"{moe_router_delta_calibration_smoke['selection_split']} / "
                f"{fmt(moe_router_delta_calibration_smoke['mean_selected_epoch'], 2)} / "
                f"{fmt(moe_router_delta_calibration_smoke['mean_selection_score'], 4)} |"
            ),
            (
                "| MoE router delta calibration smoke | train/selection samples / validation fraction | "
                f"{fmt(moe_router_delta_calibration_smoke['mean_train_samples'], 1)}/"
                f"{fmt(moe_router_delta_calibration_smoke['mean_selection_samples'], 1)} / "
                f"{fmt(moe_router_delta_calibration_smoke['mean_validation_fraction'], 3)} |"
            ),
            (
                "| MoE router delta calibration smoke | train/validation groups | "
                f"{fmt(moe_router_delta_calibration_smoke['mean_train_group_count'], 1)}/"
                f"{fmt(moe_router_delta_calibration_smoke['mean_validation_group_count'], 1)} |"
            ),
            (
                "| MoE router delta calibration smoke | train-validation KL / top1 gap | "
                f"{fmt(moe_router_delta_calibration_smoke['mean_train_final_route_kl'], 4)}-"
                f"{fmt(moe_router_delta_calibration_smoke['mean_final_route_kl'], 4)} / "
                f"{fmt(moe_router_delta_calibration_smoke['mean_train_final_top1_agreement'], 4)}-"
                f"{fmt(moe_router_delta_calibration_smoke['mean_final_top1_agreement'], 4)} / "
                f"{fmt(moe_router_delta_calibration_smoke['max_route_kl_generalization_gap'], 4)}/"
                f"{fmt(moe_router_delta_calibration_smoke['max_top1_generalization_drop'], 4)} |"
            ),
            (
                "| MoE router delta calibration smoke | hard top1/top-k overflow initial-final / increase | "
                f"{fmt(moe_router_delta_calibration_smoke['max_initial_top1_capacity_overflow_fraction'], 4)}-"
                f"{fmt(moe_router_delta_calibration_smoke['max_final_top1_capacity_overflow_fraction'], 4)} / "
                f"{fmt(moe_router_delta_calibration_smoke['max_initial_topk_capacity_overflow_fraction'], 4)}-"
                f"{fmt(moe_router_delta_calibration_smoke['max_final_topk_capacity_overflow_fraction'], 4)} / "
                f"{fmt(moe_router_delta_calibration_smoke['max_router_top1_capacity_overflow_increase'], 4)}/"
                f"{fmt(moe_router_delta_calibration_smoke['max_router_topk_capacity_overflow_increase'], 4)} |"
            ),
            (
                "| MoE router delta calibration smoke | hard top1/top-k max load initial-final | "
                f"{fmt(moe_router_delta_calibration_smoke['max_initial_top1_load_fraction'], 4)}-"
                f"{fmt(moe_router_delta_calibration_smoke['max_final_top1_load_fraction'], 4)} / "
                f"{fmt(moe_router_delta_calibration_smoke['max_initial_topk_load_fraction'], 4)}-"
                f"{fmt(moe_router_delta_calibration_smoke['max_final_topk_load_fraction'], 4)} |"
            ),
            (
                "| MoE router calibration cache smoke | status / ready routers / cache rows | "
                f"{moe_router_calibration_cache_smoke['status']} / "
                f"{moe_router_calibration_cache_smoke['cache_ready_router_count']}"
                f"/{moe_router_calibration_cache_smoke['common_router_count']} / "
                f"{moe_router_calibration_cache_smoke['total_cache_rows']} |"
            ),
            (
                "| MoE router calibration cache smoke | materialization status / checked / failed | "
                f"{moe_router_calibration_cache_smoke['materialization_status']} / "
                f"{moe_router_calibration_cache_smoke['materialization_checked_tensors']} / "
                f"{moe_router_calibration_cache_smoke['materialization_failed_tensors']} |"
            ),
            (
                "| MoE router calibration cache smoke | cache KL / trained KL initial-final / trained top1 initial-final | "
                f"{fmt(moe_router_calibration_cache_smoke['mean_student_teacher_route_kl'], 4)} / "
                f"{fmt(moe_router_calibration_cache_smoke['calibration_mean_initial_route_kl'], 4)}-"
                f"{fmt(moe_router_calibration_cache_smoke['calibration_mean_final_route_kl'], 4)} / "
                f"{fmt(moe_router_calibration_cache_smoke['calibration_mean_initial_top1_agreement'], 4)}-"
                f"{fmt(moe_router_calibration_cache_smoke['calibration_mean_final_top1_agreement'], 4)} |"
            ),
            (
                "| MoE router calibration cache smoke | selection split / samples / groups | "
                f"{moe_router_calibration_cache_smoke['calibration_selection_policy']}-"
                f"{moe_router_calibration_cache_smoke['calibration_selection_split']} / "
                f"{fmt(moe_router_calibration_cache_smoke['calibration_mean_train_samples'], 1)}/"
                f"{fmt(moe_router_calibration_cache_smoke['calibration_mean_selection_samples'], 1)} / "
                f"{fmt(moe_router_calibration_cache_smoke['calibration_mean_train_group_count'], 1)}/"
                f"{fmt(moe_router_calibration_cache_smoke['calibration_mean_validation_group_count'], 1)} |"
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
                "| MoE packed-expert writer smoke | status | "
                f"{moe_packed_expert_writer_smoke['status']} |"
            ),
            (
                "| MoE packed-expert writer smoke | checked / failed tensors | "
                f"{moe_packed_expert_writer_smoke['checked_tensors']} / "
                f"{moe_packed_expert_writer_smoke['failed_tensors']} |"
            ),
            (
                "| MoE packed-expert writer smoke | packed rule tensors / slices / values | "
                f"{moe_packed_expert_writer_smoke['packed_expert_rule_tensors']} / "
                f"{moe_packed_expert_writer_smoke['packed_expert_rule_slices']} / "
                f"{moe_packed_expert_writer_smoke['packed_expert_rule_values']} |"
            ),
            (
                "| checkpoint topology | inspected MoE configs | "
                f"{len(moe_models)} |"
            ),
            (
                "| checkpoint topology | primary real MoE source | "
                f"{topology['primary_model']} / weights={topology['primary_weights_available']} |"
            ),
            (
                "| checkpoint topology | experts config / packed weights / routed expert bytes | "
                f"{topology['primary_num_experts']} / "
                f"{topology['primary_num_experts_with_weights']} / "
                f"{topology['primary_routed_expert_bytes']} |"
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
                "| MoE packed route-weight recipe smoke | packed rules / tensors / slices | "
                f"{packed_route_recipe_smoke['packed_expert_rule_count']} / "
                f"{packed_route_recipe_smoke['packed_expert_rule_tensor_count']} / "
                f"{packed_route_recipe_smoke['packed_expert_rule_slice_count']} |"
            ),
            (
                "| MoE packed route-weight recipe smoke | writer command uses packed CSV | "
                f"{packed_route_recipe_smoke['writer_command_has_packed_rule']} |"
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
