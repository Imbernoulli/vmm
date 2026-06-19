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
    if path.suffix in {".csv", ".json"}:
        return "data"
    if path.suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        return "figure"
    return "artifact"


def collect_artifacts() -> list[dict[str, Any]]:
    roots = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "RESEARCH_REPORT.md",
        REPO_ROOT / "PAPER.md",
        REPO_ROOT / "proposal.md",
        REPO_ROOT / "src",
        REPO_ROOT / "scripts",
        REPO_ROOT / "results",
    ]
    suffixes = {".py", ".md", ".csv", ".json", ".png", ".html"}
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
