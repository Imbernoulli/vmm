#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

LITERATURE_PRIORS = [
    {
        "key": "mode_connectivity",
        "source": "https://arxiv.org/abs/1802.10026",
        "mechanism": "A weight average is trusted only when the probed path stays in a low-loss basin.",
    },
    {
        "key": "model_soups",
        "source": "https://arxiv.org/abs/2203.05482",
        "mechanism": "Same-basin finetunes can average well, but endpoint fallback is part of the recipe.",
    },
    {
        "key": "git_rebasin",
        "source": "https://arxiv.org/abs/2209.04836",
        "mechanism": "Permutation symmetry must be canonicalized before weight-space merging.",
    },
    {
        "key": "ties",
        "source": "https://arxiv.org/abs/2306.01708",
        "mechanism": "Coordinate sign conflict is a real dense failure signal, but it still needs held-out gating.",
    },
    {
        "key": "dare",
        "source": "https://arxiv.org/abs/2311.03099",
        "mechanism": "Delta pruning/rescaling is useful only when the retained delta is not too large or noisy.",
    },
    {
        "key": "mergeme",
        "source": "https://arxiv.org/abs/2502.00997",
        "mechanism": "MoE merging must handle parameter interference and routing, not just average experts.",
    },
    {
        "key": "sub_moe",
        "source": "https://arxiv.org/abs/2506.23266",
        "mechanism": "Expert output similarity/subspace structure is a better merge signal than tensor names alone.",
    },
    {
        "key": "mergemoe",
        "source": "https://arxiv.org/abs/2510.14436",
        "mechanism": "MoE expert merging can be formulated through output-space matching and optimization.",
    },
    {
        "key": "namex",
        "source": "https://arxiv.org/abs/2510.16138",
        "mechanism": "Expert weights should reflect cooperation/competition rather than a fixed uniform prior.",
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


def dense_feature_rows(curvature: dict[str, Any], dense_selector: dict[str, Any], gen_eval: dict[str, Any]) -> list[dict[str, Any]]:
    curve = curvature.get("curvature_law", {})
    merges = curvature.get("merges", {})
    uniform = merges.get("uniform", {})
    fisher = merges.get("fisher", {})
    dense_results = dense_selector.get("results", {})
    unified = dense_results.get("unified", {})
    linear = dense_results.get("linear", {})
    ties = dense_results.get("ties_0.5", {})
    gen_results = gen_eval.get("results", {})
    return [
        {
            "domain": "dense",
            "probe": "curvature_law_general",
            "value": curve.get("ratio_general"),
            "threshold": 5.0,
            "decision_signal": "high_nonlocal_barrier",
            "evidence": (
                f"actual/predicted general degradation = {fmt(curve.get('ratio_general'))}; "
                f"uniform worst NLL = {fmt(uniform.get('worst'))}; fisher worst NLL = {fmt(fisher.get('worst'))}"
            ),
        },
        {
            "domain": "dense",
            "probe": "curvature_law_code",
            "value": curve.get("ratio_code"),
            "threshold": 5.0,
            "decision_signal": "high_nonlocal_barrier",
            "evidence": f"actual/predicted code degradation = {fmt(curve.get('ratio_code'))}",
        },
        {
            "domain": "dense",
            "probe": "heldout_unified_selector",
            "value": unified.get("worst"),
            "threshold": linear.get("worst"),
            "decision_signal": "allow_endpoint_or_anchor_fallback",
            "evidence": (
                f"unified test worst NLL = {fmt(unified.get('worst'))}; "
                f"linear = {fmt(linear.get('worst'))}; TIES = {fmt(ties.get('worst'))}; "
                f"best endpoint = {fmt(dense_selector.get('best_endpoint_worst'))}"
            ),
        },
        {
            "domain": "dense",
            "probe": "generation_smoke",
            "value": (gen_results.get("linear") or {}).get("avg_accuracy"),
            "threshold": (gen_results.get("unified") or {}).get("avg_accuracy"),
            "decision_signal": "linear_generation_regression",
            "evidence": (
                f"linear avg accuracy = {fmt((gen_results.get('linear') or {}).get('avg_accuracy'))}; "
                f"unified avg accuracy = {fmt((gen_results.get('unified') or {}).get('avg_accuracy'))}; "
                f"best smoke method = {gen_eval.get('best_method')}"
            ),
        },
    ]


def moe_feature_rows(
    mechanism: dict[str, Any],
    real_gauge: dict[str, Any],
    moe_selector: dict[str, Any],
    router_gate: dict[str, Any],
    final_selection: dict[str, Any],
) -> list[dict[str, Any]]:
    selection = final_selection.get("current_selection") or {}
    return [
        {
            "domain": "moe",
            "probe": "controlled_expert_gauge",
            "value": mechanism.get("gauge_equivalence_mse"),
            "threshold": 1e-9,
            "decision_signal": "expert_permutation_is_function_preserving",
            "evidence": (
                f"gauge-equivalent B MSE = {fmt(mechanism.get('gauge_equivalence_mse'), 8)}; "
                f"same-name worst = {fmt((mechanism.get('results') or {}).get('uniform_same_name', {}).get('worst'))}; "
                f"aligned worst = {fmt((mechanism.get('results') or {}).get('uniform_aligned', {}).get('worst'))}"
            ),
        },
        {
            "domain": "moe",
            "probe": "real_olmoe_gauge_selfmerge",
            "value": real_gauge.get("naive_degradation_vs_baseline"),
            "threshold": 1.0,
            "decision_signal": "reject_same_name_average_without_alignment",
            "evidence": (
                f"baseline NLL = {fmt(real_gauge.get('baseline_nll'))}; "
                f"same-name average NLL = {fmt(real_gauge.get('naive_sameNAME_average_nll'))}; "
                f"aligned average NLL = {fmt(real_gauge.get('aligned_average_nll'))}; "
                f"layers recovered = {real_gauge.get('layers_perm_recovered')}/{real_gauge.get('n_moe_layers')}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_expert_identity",
            "value": moe_selector.get("qwen3_identity_fraction"),
            "threshold": 1.0,
            "decision_signal": "identity_alignment_is_allowed_for_this_pair",
            "evidence": (
                f"identity-optimal layer fraction = {fmt(moe_selector.get('qwen3_identity_fraction'))}; "
                f"argmax identity fraction = {fmt(moe_selector.get('qwen3_argmax_identity_fraction'))}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_router_move_gate",
            "value": router_gate.get("allowed_router_layer_count"),
            "threshold": router_gate.get("router_layer_count"),
            "decision_signal": "freeze_router_or_train_route_kd_delta",
            "evidence": (
                f"allowed router layers = {router_gate.get('allowed_router_layer_count')}/"
                f"{router_gate.get('router_layer_count')}; "
                f"top-k Jaccard mean/min = {fmt(router_gate.get('mean_topk_jaccard'))}/"
                f"{fmt(router_gate.get('min_topk_jaccard'))}; "
                f"top1 agreement mean/min = {fmt(router_gate.get('mean_top1_agreement'))}/"
                f"{fmt(router_gate.get('min_top1_agreement'))}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_final_candidate_selection",
            "value": selection.get("eligible_candidate_count"),
            "threshold": selection.get("candidate_count"),
            "decision_signal": "await_matched_vllm_before_accepting_average",
            "evidence": (
                f"status = {selection.get('status')}; eligible candidates = "
                f"{selection.get('eligible_candidate_count')}/{selection.get('candidate_count')}; "
                f"reason = {selection.get('reason')}"
            ),
        },
    ]


def build_decisions(features: pd.DataFrame, dense_selector: dict[str, Any], router_gate: dict[str, Any]) -> pd.DataFrame:
    dense_config = ((dense_selector.get("results") or {}).get("unified") or {}).get("config") or {}
    rows = [
        {
            "stage": "dense_connectivity_gate",
            "operation": "do_not_use_linear_midpoint_by_default",
            "condition": "actual Fisher-prediction error ratio is far above local-quadratic range",
            "selected_action": f"search base-anchored coefficient family; current config = {json.dumps(dense_config, sort_keys=True)}",
            "why_it_should_improve": "It prevents a fixed 0.5 midpoint from crossing a measured high-loss barrier.",
            "same_shape_invariant": "same tokenizer, model class, tensor names, and tensor shapes",
        },
        {
            "stage": "dense_sparse_coordinate_gate",
            "operation": "make TIES/DARE-style sparsity conditional",
            "condition": "coordinate conflict rules must beat the anchor on held-out worst-task loss",
            "selected_action": "only materialize sparse conflict rules when held-out and vLLM gates pass",
            "why_it_should_improve": "It keeps sign-conflict probes as diagnostics without letting them delete useful dense capacity.",
            "same_shape_invariant": "same dense architecture; only values change",
        },
        {
            "stage": "moe_expert_identity_gate",
            "operation": "canonicalize expert gauge before averaging",
            "condition": "MoE expert/router row permutation is function preserving but breaks same-name average",
            "selected_action": "run layer-wise expert alignment; for Qwen3 Instruct/Coder the mapping is currently identity",
            "why_it_should_improve": "It removes a discrete symmetry error before any continuous weight interpolation is attempted.",
            "same_shape_invariant": "expert count and router shape are unchanged; only source tensor aliases are remapped",
        },
        {
            "stage": "moe_router_gate",
            "operation": "freeze direct router movement",
            "condition": "no Qwen3 router layer passes the all-category movement guard",
            "selected_action": router_gate.get("recommended_unified_router_action", "freeze_router"),
            "why_it_should_improve": "It avoids averaging a discrete top-k dispatch boundary that has high measured source disagreement.",
            "same_shape_invariant": "router tensor shape is unchanged; optional route-KD writes a capped same-shape delta",
        },
        {
            "stage": "moe_candidate_gate",
            "operation": "select only after audited downstream eval",
            "condition": "structural probes can rank risk, but source dominance and task regression need vLLM scores",
            "selected_action": "keep all seven Qwen3 candidates provisional until eval-bundle audit passes",
            "why_it_should_improve": "It prevents structural cleanliness from being mistaken for actual downstream dominance.",
            "same_shape_invariant": "all candidates remain same-shape Qwen3 MoE checkpoints",
        },
    ]
    return pd.DataFrame(rows)


def build_algorithm(decisions: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "mechanism_gated_unified_average",
        "contract": "The output model must keep the same config, tokenizer, model class, tensor names, and tensor shapes as the chosen target.",
        "steps": [
            {
                "step": int(index + 1),
                "stage": row["stage"],
                "operation": row["operation"],
                "condition": row["condition"],
                "selected_action": row["selected_action"],
            }
            for index, row in decisions.iterrows()
        ],
        "literature_priors": LITERATURE_PRIORS,
    }


def build_report(summary: dict[str, Any], features: pd.DataFrame, decisions: pd.DataFrame) -> str:
    dense = summary["dense"]
    moe = summary["moe"]
    lines = [
        "# Unified Average Optimizer",
        "",
        "这个脚本把 Dense 和 MoE 的 probe 结果统一成同一个操作选择器：先测几何和对称性，再决定能不能平均、平均多少、哪些结构必须冻结或校准。它不是按算法名投票，而是让每个平均动作都绑定一个可测机制。",
        "",
        "## Current Decision",
        "",
        f"- Dense: `{dense['decision']}`；linear worst NLL `{fmt(dense['linear_worst_nll'])}`，unified worst NLL `{fmt(dense['unified_worst_nll'])}`。",
        f"- MoE: `{moe['decision']}`；真实 OLMoE same-name average degradation `{fmt(moe['real_gauge_naive_degradation'])}`，Qwen3 router action `{moe['router_action']}`。",
        f"- Qwen3 final selection: `{moe['qwen3_final_selection_status']}`，eligible `{moe['qwen3_eligible_candidates']}/{moe['qwen3_candidate_count']}`。",
        "",
        "## Mechanism Features",
        "",
        "| domain | probe | signal | value | threshold | evidence |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for _, row in features.iterrows():
        lines.append(
            f"| `{row['domain']}` | `{row['probe']}` | `{row['decision_signal']}` | "
            f"{fmt(row['value'])} | {fmt(row['threshold'])} | {row['evidence']} |"
        )
    lines.extend(["", "## Operations", "", "| stage | operation | selected action | why |", "| --- | --- | --- | --- |"])
    for _, row in decisions.iterrows():
        lines.append(
            f"| `{row['stage']}` | `{row['operation']}` | {row['selected_action']} | {row['why_it_should_improve']} |"
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
            f"- `{summary['outputs']['features']}`",
            f"- `{summary['outputs']['decisions']}`",
            f"- `{summary['outputs']['algorithm']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    curvature = read_json(args.dense_curvature)
    dense_selector = read_json(args.dense_selector)
    gen_eval = read_json(args.dense_generation)
    mechanism = read_json(args.moe_mechanism)
    real_gauge = read_json(args.real_moe_gauge)
    moe_selector = read_json(args.moe_selector)
    router_gate = read_json(args.qwen3_router_gate)
    final_selection = read_json(args.qwen3_final_selection)

    feature_rows = dense_feature_rows(curvature, dense_selector, gen_eval)
    feature_rows.extend(moe_feature_rows(mechanism, real_gauge, moe_selector, router_gate, final_selection))
    features = pd.DataFrame(feature_rows)
    decisions = build_decisions(features, dense_selector, router_gate)
    algorithm = build_algorithm(decisions)

    dense_results = dense_selector.get("results") or {}
    final_current = final_selection.get("current_selection") or {}
    summary = {
        "schema_version": 1,
        "status": "built_waiting_for_qwen3_vllm_eval"
        if final_current.get("status") == "awaiting_source_eval"
        else "built",
        "dense": {
            "decision": "avoid_linear_midpoint_use_probe_selected_anchor_or_low_lambda",
            "curvature_ratio_general": fnum((curvature.get("curvature_law") or {}).get("ratio_general")),
            "curvature_ratio_code": fnum((curvature.get("curvature_law") or {}).get("ratio_code")),
            "linear_worst_nll": fnum((dense_results.get("linear") or {}).get("worst")),
            "unified_worst_nll": fnum((dense_results.get("unified") or {}).get("worst")),
            "best_endpoint_worst_nll": fnum(dense_selector.get("best_endpoint_worst")),
            "generation_best_method": gen_eval.get("best_method"),
        },
        "moe": {
            "decision": "align_experts_freeze_router_then_gate_candidate_by_vllm",
            "controlled_gauge_mse": fnum(mechanism.get("gauge_equivalence_mse")),
            "real_gauge_naive_degradation": fnum(real_gauge.get("naive_degradation_vs_baseline")),
            "real_gauge_aligned_degradation": fnum(real_gauge.get("aligned_degradation_vs_baseline")),
            "qwen3_identity_fraction": fnum(moe_selector.get("qwen3_identity_fraction")),
            "router_action": router_gate.get("recommended_unified_router_action"),
            "qwen3_final_selection_status": final_current.get("status"),
            "qwen3_eligible_candidates": final_current.get("eligible_candidate_count"),
            "qwen3_candidate_count": final_current.get("candidate_count"),
        },
        "outputs": {},
    }

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "mechanism_features.csv"
    decisions_path = output_dir / "operation_decisions.csv"
    algorithm_path = output_dir / "algorithm.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary["outputs"] = {
        "features": rel(features_path),
        "decisions": rel(decisions_path),
        "algorithm": rel(algorithm_path),
        "summary": rel(summary_path),
        "report": rel(report_path),
    }

    features.to_csv(features_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    algorithm_path.write_text(json.dumps(json_safe(algorithm), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, features, decisions), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mechanism-gated Dense/MoE unified average optimizer.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/unified_average_optimizer"))
    parser.add_argument("--dense-curvature", type=Path, default=Path("results/fp_curvature_law/summary.json"))
    parser.add_argument("--dense-selector", type=Path, default=Path("results/fp_merge_compare_dense/summary.json"))
    parser.add_argument("--dense-generation", type=Path, default=Path("results/fp_gen_eval_dense/summary.json"))
    parser.add_argument("--moe-mechanism", type=Path, default=Path("results/fp_moe_mechanism/summary.json"))
    parser.add_argument("--real-moe-gauge", type=Path, default=Path("results/fp_moe_real_probe/summary.json"))
    parser.add_argument("--moe-selector", type=Path, default=Path("results/moe_probe_gated_selector/summary.json"))
    parser.add_argument("--qwen3-router-gate", type=Path, default=Path("results/qwen3_moe_router_move_gate/summary.json"))
    parser.add_argument(
        "--qwen3-final-selection",
        type=Path,
        default=Path("results/qwen3_moe_final_candidate_selection/summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args)
    print(f"Wrote unified average optimizer to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; dense={summary['dense']['decision']}; moe={summary['moe']['decision']}"
    )


if __name__ == "__main__":
    main()
