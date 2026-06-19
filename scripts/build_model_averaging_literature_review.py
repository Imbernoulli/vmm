#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


SOURCES: list[dict[str, str]] = [
    {
        "short_name": "Fisher merging",
        "year": "2021",
        "category": "dense_importance_weighting",
        "title": "Merging Models with Fisher-Weighted Averaging",
        "url": "https://arxiv.org/abs/2111.09832",
        "use_in_this_repo": "Use diagonal Fisher or NLL sensitivity to downweight parameters whose source model strongly relies on them.",
    },
    {
        "short_name": "Model soups",
        "year": "2022",
        "category": "dense_weight_average",
        "title": "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time",
        "url": "https://arxiv.org/abs/2203.05482",
        "use_in_this_repo": "Treat uniform and greedy soups as low-barrier baselines, not as proof that arbitrary domain experts can be averaged.",
    },
    {
        "short_name": "Task arithmetic",
        "year": "2022",
        "category": "task_vector",
        "title": "Editing Models with Task Arithmetic",
        "url": "https://arxiv.org/abs/2212.04089",
        "use_in_this_repo": "Represent Qwen branches as same-anchor deltas and search coefficients on task-vector planes.",
    },
    {
        "short_name": "Git Re-Basin",
        "year": "2022",
        "category": "connectivity_alignment",
        "title": "Git Re-Basin: Merging Models modulo Permutation Symmetries",
        "url": "https://arxiv.org/abs/2209.04836",
        "use_in_this_repo": "Use barrier and alignment probes before trusting direct weight averages across differently trained checkpoints.",
    },
    {
        "short_name": "ZipIt",
        "year": "2023",
        "category": "feature_alignment",
        "title": "ZipIt! Merging Models from Different Tasks without Training",
        "url": "https://arxiv.org/abs/2305.03053",
        "use_in_this_repo": "Keep feature/permutation alignment as an escalation path when same-base assumptions fail.",
    },
    {
        "short_name": "TIES",
        "year": "2023",
        "category": "dense_conflict_resolution",
        "title": "TIES-Merging: Resolving Interference When Merging Models",
        "url": "https://arxiv.org/abs/2306.01708",
        "use_in_this_repo": "Map sign disagreement and small-delta redundancy into coordinate-level tensor rules.",
    },
    {
        "short_name": "AdaMerging",
        "year": "2023",
        "category": "coefficient_learning",
        "title": "AdaMerging: Adaptive Model Merging for Multi-Task Learning",
        "url": "https://arxiv.org/abs/2310.02575",
        "use_in_this_repo": "Learn task-wise or layer-wise coefficients with unlabeled calibration data when grid search is too coarse.",
    },
    {
        "short_name": "DARE",
        "year": "2023",
        "category": "delta_sparsification",
        "title": "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch",
        "url": "https://arxiv.org/abs/2311.03099",
        "use_in_this_repo": "Only consider random drop/rescale when delta magnitude probes show high redundancy.",
    },
    {
        "short_name": "DELLA",
        "year": "2024",
        "category": "delta_sparsification",
        "title": "DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based Sampling",
        "url": "https://arxiv.org/abs/2406.11617",
        "use_in_this_repo": "Prefer magnitude-aware retention curves over blind DARE when important deltas are concentrated.",
    },
    {
        "short_name": "Model merging survey",
        "year": "2024",
        "category": "survey",
        "title": "Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities",
        "url": "https://arxiv.org/abs/2408.07666",
        "use_in_this_repo": "Use as taxonomy coverage: averaging, task arithmetic, data-informed merging, alignment, and applications.",
    },
    {
        "short_name": "DAM",
        "year": "2024",
        "category": "coefficient_learning",
        "title": "Merging in a Bottle: Differentiable Adaptive Merging and the Path from Averaging to Automation",
        "url": "https://arxiv.org/abs/2410.08371",
        "use_in_this_repo": "Frame coefficient tuning as differentiable optimization when held-out NLL is available.",
    },
    {
        "short_name": "WEMoE",
        "year": "2024",
        "category": "dynamic_moe_upper_bound",
        "title": "Efficient and Effective Weight-Ensembling Mixture of Experts for Multi-Task Model Merging",
        "url": "https://arxiv.org/abs/2410.21804",
        "use_in_this_repo": "Use dynamic expert selection as an upper bound, then compress back to same-shape average when required.",
    },
    {
        "short_name": "MergeME",
        "year": "2025",
        "category": "moe_model_merging",
        "title": "MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs",
        "url": "https://arxiv.org/abs/2502.00997",
        "use_in_this_repo": "Treat parameter interference and routing heuristics as first-class MoE merge problems.",
    },
    {
        "short_name": "STAR",
        "year": "2025",
        "category": "spectral_delta_sparsification",
        "title": "STAR: Spectral Truncation and Rescale for Model Merging",
        "url": "https://arxiv.org/abs/2502.10339",
        "use_in_this_repo": "Probe singular spectra before deciding whether small spectral components should be dropped or rescaled.",
    },
    {
        "short_name": "Qwen3",
        "year": "2025",
        "category": "target_model_family",
        "title": "Qwen3 Technical Report",
        "url": "https://arxiv.org/abs/2505.09388",
        "use_in_this_repo": "Use Qwen dense and MoE model family as the main experimental target for same-shape averaging.",
    },
    {
        "short_name": "Sub-MoE",
        "year": "2025",
        "category": "moe_expert_compression",
        "title": "Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging",
        "url": "https://arxiv.org/abs/2506.23266",
        "use_in_this_repo": "Use expert output similarity, clustering, and route frequency as evidence before merging experts.",
    },
    {
        "short_name": "FroM",
        "year": "2025",
        "category": "data_free_adaptive_merging",
        "title": "FroM: Frobenius Norm-Based Data-Free Adaptive Model Merging",
        "url": "https://arxiv.org/abs/2506.02478",
        "use_in_this_repo": "Use parameter norms as a data-free fallback when activation or Fisher probes are unavailable.",
    },
    {
        "short_name": "RegMean++",
        "year": "2025",
        "category": "activation_regression",
        "title": "RegMean++: Enhancing Effectiveness and Generalization of Regression Mean for Model Merging",
        "url": "https://arxiv.org/abs/2508.03121",
        "use_in_this_repo": "Use activation covariance and cross-layer dependency probes for off-plane linear-layer merges.",
    },
    {
        "short_name": "Merge scaling laws",
        "year": "2025",
        "category": "scaling_laws",
        "title": "Model Merging Scaling Laws in Large Language Models",
        "url": "https://arxiv.org/abs/2509.24244",
        "use_in_this_repo": "Estimate diminishing returns from adding more experts before paying for large merge sweeps.",
    },
    {
        "short_name": "MergeMoE",
        "year": "2025",
        "category": "moe_expert_compression",
        "title": "MergeMoE: Efficient Compression of MoE Models via Expert Output Merging",
        "url": "https://arxiv.org/abs/2510.14436",
        "use_in_this_repo": "Prefer output-space expert matching over raw parameter averaging for routed experts.",
    },
    {
        "short_name": "HARC",
        "year": "2026",
        "category": "moe_router_calibration",
        "title": "When Model Merging Breaks Routing: Training-Free Calibration for MoE",
        "url": "https://arxiv.org/abs/2606.03391",
        "use_in_this_repo": "Gate MoE averages on routing breakdown metrics and calibrate routers before accepting a merge.",
    },
]


METHODS: list[dict[str, str]] = [
    {
        "method_family": "Uniform / linear average",
        "dense_use": "Baseline for same-base checkpoints; useful when lambda path has low barrier.",
        "moe_use": "Negative baseline unless router/expert probes prove route stability.",
        "primary_probe": "endpoint score; lambda sweep; midpoint barrier; worst-task score",
        "failure_signal": "merged NLL higher than both endpoints; worst-task retention collapse",
        "recommended_action": "Only materialize if validation grid beats endpoints or is not endpoint-only.",
        "sources": "Model soups; Task arithmetic",
    },
    {
        "method_family": "Task arithmetic / coefficient search",
        "dense_use": "Search task-vector coefficients on same-anchor deltas.",
        "moe_use": "Apply separately to shared modules, router, and expert groups.",
        "primary_probe": "alpha/beta grid; layer cosine; held-in retention",
        "failure_signal": "best grid is endpoint; high plane barrier; layer-wise conflict",
        "recommended_action": "Move from global weights to layer/module/expert-specific weights.",
        "sources": "Task arithmetic; AdaMerging; DAM",
    },
    {
        "method_family": "Sign / sparsity conflict methods",
        "dense_use": "Use TIES, DARE, DELLA, or STAR when deltas are redundant or sign-conflicting.",
        "moe_use": "Use on shared and expert FFN deltas after expert matching, not on router blindly.",
        "primary_probe": "sign conflict; weighted conflict; delta magnitude distribution; singular spectrum",
        "failure_signal": "large important deltas dropped; conflict concentrated in norms/lm_head/router",
        "recommended_action": "Convert conflict signals into tensor rules and preserve critical groups.",
        "sources": "TIES; DARE; DELLA; STAR",
    },
    {
        "method_family": "Importance / activation-aware average",
        "dense_use": "Use Fisher, RegMean, or RegMean++ when calibration activations are available.",
        "moe_use": "Estimate expert sensitivity with route-conditioned NLL and activation covariance.",
        "primary_probe": "diagonal Fisher; activation covariance; NLL sensitivity",
        "failure_signal": "calibration data does not match target tasks; off-plane residual is large",
        "recommended_action": "Report as structured average with plane residual, not as raw on-plane average.",
        "sources": "Fisher merging; RegMean++; FroM",
    },
    {
        "method_family": "Alignment before averaging",
        "dense_use": "Needed when checkpoints are not same initialization or barrier remains high.",
        "moe_use": "Needed when expert indices or feature spaces are permuted.",
        "primary_probe": "permutation residual; feature CKA; expert output cosine",
        "failure_signal": "same score endpoints but high interpolation barrier",
        "recommended_action": "Run feature/expert matching before computing any average.",
        "sources": "Git Re-Basin; ZipIt; Sub-MoE; MergeMoE",
    },
    {
        "method_family": "Router-aware MoE average",
        "dense_use": "Not applicable.",
        "moe_use": "Freeze or calibrate router; merge shared/expert tensors with separate rules.",
        "primary_probe": "route overlap; router entropy; max expert fraction; top-k margin",
        "failure_signal": "routing breakdown; route collapse; low top-k agreement after merge",
        "recommended_action": "Keep router frozen, calibrate router, or reject candidate before writing checkpoint.",
        "sources": "MergeME; HARC; Qwen3; WEMoE",
    },
]


PROBES: list[dict[str, str]] = [
    {
        "probe": "endpoint and held-in retention",
        "what_it_measures": "Whether each source model's native ability survives the average.",
        "dense_decision": "Reject averages that improve mean score while sacrificing one expert.",
        "moe_decision": "Use per-domain retention before trusting route-aware rules.",
        "artifact_target": "method_metrics.csv; decision_table.csv",
    },
    {
        "probe": "lambda path and midpoint barrier",
        "what_it_measures": "Linear connectivity between endpoints or between anchor and source.",
        "dense_decision": "Low barrier supports soups/task arithmetic; high barrier triggers alignment or layer-wise coefficients.",
        "moe_decision": "Run shared-only, router-frozen, and all-weight paths separately.",
        "artifact_target": "path_metrics.csv; qwen path sweep; alpha/beta grids",
    },
    {
        "probe": "delta cosine and sign conflict",
        "what_it_measures": "Parameter-level direction agreement and destructive sign disagreement.",
        "dense_decision": "Choose TIES/DARE/DELLA/STAR or freeze conflict-heavy groups.",
        "moe_decision": "Compute per shared module and per matched expert, never only globally.",
        "artifact_target": "delta_summary.csv; interference.csv",
    },
    {
        "probe": "activation/Fisher sensitivity",
        "what_it_measures": "Which parameters or linear layers are important on calibration data.",
        "dense_decision": "Use Fisher/RegMean/AdaMerging style coefficients.",
        "moe_decision": "Compute route-conditioned sensitivity for experts and router.",
        "artifact_target": "future activation covariance and Fisher summaries",
    },
    {
        "probe": "router entropy and route overlap",
        "what_it_measures": "Whether MoE routing still dispatches tokens to appropriate experts after merging.",
        "dense_decision": "Not applicable.",
        "moe_decision": "Freeze/calibrate router or reject all-weight average if overlap collapses.",
        "artifact_target": "router_summary.csv; route_overlap.csv; router_readiness.csv",
    },
    {
        "probe": "expert output similarity",
        "what_it_measures": "Whether expert index e in two checkpoints represents the same function.",
        "dense_decision": "Use analogous feature alignment only if initialization differs.",
        "moe_decision": "Build expert remap aliases before averaging expert tensors.",
        "artifact_target": "expert_match.csv; source_tensor_aliases.txt",
    },
]


MOE_OPTIMIZATIONS: list[dict[str, str]] = [
    {
        "stage": "0_topology_gate",
        "question": "Do all inputs have the same config, tokenizer, router shape, expert count, and tensor names?",
        "required_probe": "config/header inspection",
        "accept_rule": "same shape or documented source tensor aliases only",
        "writer_action": "Proceed to dry-run validation.",
    },
    {
        "stage": "1_router_gate",
        "question": "Does simple averaging break routing?",
        "required_probe": "router entropy, top-k agreement, route overlap, max expert fraction",
        "accept_rule": "No collapse, no large drift, no fragile top-k boundary.",
        "writer_action": "Freeze router by default; allow small router delta only after readiness passes.",
    },
    {
        "stage": "2_expert_alignment",
        "question": "Are source expert indices semantically aligned?",
        "required_probe": "expert output cosine, route coactivation, task profile similarity",
        "accept_rule": "Matched experts above cosine threshold; manual review for low matches.",
        "writer_action": "Pass source_tensor_aliases.txt to same-shape writer.",
    },
    {
        "stage": "3_expert_weighting",
        "question": "Which source should dominate each expert tensor?",
        "required_probe": "route frequency, NLL sensitivity, expert delta conflict",
        "accept_rule": "Weights reflect task route mass and do not damage general retention.",
        "writer_action": "Emit tensor_rules.txt with per-expert source weights.",
    },
    {
        "stage": "4_shared_module_merge",
        "question": "Can shared attention/norm/MLP be averaged globally?",
        "required_probe": "layer cosine, sign conflict, Fisher/activation sensitivity",
        "accept_rule": "Use module-specific weights when conflicts concentrate.",
        "writer_action": "Emit tensor rules for shared modules; freeze risky lm_head/norm if needed.",
    },
    {
        "stage": "5_candidate_acceptance",
        "question": "Does the materialized checkpoint beat baselines on held-out tasks?",
        "required_probe": "held-in retention, worst score, format safety, cost",
        "accept_rule": "Beat all-weight average and avoid endpoint-only pseudo-success.",
        "writer_action": "Promote candidate only after held-out eval.",
    },
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(output_dir: Path, summary: dict[str, Any]) -> str:
    lines = [
        "# Model Averaging Literature and Probe Matrix",
        "",
        "这份报告把 Dense model averaging、线性插值、task-vector merging 和 MoE 专用 merging 的论文证据整理成一张工程矩阵。重点不是机械列方法，而是回答：看到哪些指标时，应该使用哪类平均策略；哪些情况下应该先拒绝或校准，而不是直接写 checkpoint。",
        "",
        "## 结论",
        "",
        "1. Dense 模型可以从 linear average、greedy soup、task arithmetic 开始，但前提是同源、低 barrier、端点能力相近；否则 `0.5/0.5` 只是负 baseline。",
        "2. TIES、DARE、DELLA、STAR 这些方法本质上是在处理 delta 冗余、符号冲突或谱空间冲突；它们需要 delta magnitude、sign conflict、singular spectrum 等 probe 支撑。",
        "3. Fisher、RegMean、AdaMerging、DAM 这类方法把 average 变成重要性加权或 coefficient learning；它们更像 probe-guided average，而不是固定菜谱。",
        "4. MoE 的核心失败模式是 router/expert 共同失配：router 可能 breakdown，expert index 可能不再对应同一功能，专家专长会让全参数同权平均更脆弱。",
        "5. 对 Qwen3-30B-A3B / Qwen3-Coder-30B-A3B 这类同构 MoE，最保守路径是 topology gate -> router gate -> expert matching -> route-frequency tensor rules -> held-out eval。",
        "",
        "## 关键计数",
        "",
        f"- Sources reviewed: `{summary['source_count']}`",
        f"- Method families: `{summary['method_family_count']}`",
        f"- Probe groups: `{summary['probe_count']}`",
        f"- MoE optimization stages: `{summary['moe_stage_count']}`",
        "",
        "## 方法矩阵",
        "",
        "| method family | dense use | MoE use | primary probe | recommended action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in METHODS:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method_family"],
                    row["dense_use"],
                    row["moe_use"],
                    row["primary_probe"],
                    row["recommended_action"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Probe 矩阵",
            "",
            "| probe | measures | dense decision | MoE decision | artifact target |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in PROBES:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["probe"],
                    row["what_it_measures"],
                    row["dense_decision"],
                    row["moe_decision"],
                    row["artifact_target"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## MoE 优化路线",
            "",
            "| stage | question | required probe | accept rule | writer action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in MOE_OPTIMIZATIONS:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["stage"],
                    row["question"],
                    row["required_probe"],
                    row["accept_rule"],
                    row["writer_action"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 对当前仓库的直接影响",
            "",
            "- `results/average_decision_report/report.md` 负责把 Dense/Qwen merge grid、barrier 和 conflict probe 变成是否 materialize 的决策。",
            "- `results/moe_routing_readiness/report.md` 是 MoE 的 router gate；没有真实 routing probe 时，MoE route-weight recipe 必须保持 `waiting_for_routing_probe`。",
            "- `results/toy_moe_expert_remap_plan/source_tensor_aliases.txt` 对应 MoE 优化路线里的 expert alignment stage：它不改变输出结构，只改变 source tensor 的读取坐标。",
            "- 下一步真实 Qwen3 MoE 实验应优先补 `Qwen3-30B-A3B-Base`、`Qwen3-30B-A3B`、`Qwen3-Coder-30B-A3B-Instruct` 的 route traces、expert output similarity 和 held-out NLL，而不是先做全权重平均。",
            "",
            "## 文件",
            "",
            f"- `{rel(output_dir / 'method_matrix.csv')}`",
            f"- `{rel(output_dir / 'probe_matrix.csv')}`",
            f"- `{rel(output_dir / 'moe_optimization_matrix.csv')}`",
            f"- `{rel(output_dir / 'source_matrix.csv')}`",
            f"- `{rel(output_dir / 'summary.json')}`",
            "",
            "## Sources",
            "",
        ]
    )
    for row in SOURCES:
        lines.append(f"- {row['short_name']} ({row['year']}): [{row['title']}]({row['url']})")
    return "\n".join(lines) + "\n"


def build_summary(output_dir: Path) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(SOURCES),
        "method_family_count": len(METHODS),
        "probe_count": len(PROBES),
        "moe_stage_count": len(MOE_OPTIMIZATIONS),
        "dense_recommendation": "Treat uniform averaging as a low-barrier baseline; escalate to conflict-aware, importance-aware, or coefficient-learned averages only when probes justify them.",
        "moe_recommendation": "Do not merge router and experts blindly; run topology, routing, expert matching, and route-weight probes before materializing same-shape checkpoints.",
        "outputs": {
            "report": rel(output_dir / "report.md"),
            "method_matrix": rel(output_dir / "method_matrix.csv"),
            "probe_matrix": rel(output_dir / "probe_matrix.csv"),
            "moe_optimization_matrix": rel(output_dir / "moe_optimization_matrix.csv"),
            "source_matrix": rel(output_dir / "source_matrix.csv"),
            "summary": rel(output_dir / "summary.json"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a literature-grounded model averaging probe matrix.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/model_averaging_literature_review"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "method_matrix.csv", METHODS)
    write_csv(output_dir / "probe_matrix.csv", PROBES)
    write_csv(output_dir / "moe_optimization_matrix.csv", MOE_OPTIMIZATIONS)
    write_csv(output_dir / "source_matrix.csv", SOURCES)
    summary = build_summary(output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(output_dir, summary), encoding="utf-8")
    print(f"Wrote model averaging literature review to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
