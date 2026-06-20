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
        "key": "expert_merging",
        "source": "https://arxiv.org/abs/2509.25712",
        "mechanism": "Layer and chunk coefficients should be guided by unlabeled hidden/logit alignment rather than a fixed global merge weight.",
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
    {
        "key": "harc",
        "source": "https://arxiv.org/abs/2606.03391",
        "mechanism": "MoE router movement must be gated by top-k boundary stability, not treated like an ordinary dense tensor.",
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


def dense_feature_rows(
    curvature: dict[str, Any],
    dense_selector: dict[str, Any],
    dense_lambda: dict[str, Any],
    gen_eval: dict[str, Any],
) -> list[dict[str, Any]]:
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
            "probe": "dense_lambda_connectivity",
            "value": dense_lambda.get("linear_worst"),
            "threshold": dense_lambda.get("unified_best_worst"),
            "decision_signal": "straight_line_midpoint_rejected",
            "evidence": (
                f"linear midpoint worst NLL = {fmt(dense_lambda.get('linear_worst'))}; "
                f"best lambda-family worst NLL = {fmt(dense_lambda.get('unified_best_worst'))}; "
                f"best config = {json.dumps(dense_lambda.get('unified_best_config') or {}, sort_keys=True)}; "
                f"best source endpoint worst NLL = {fmt(dense_lambda.get('best_endpoint_worst'))}"
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
    unified_candidate: dict[str, Any],
    unified_audit: dict[str, Any],
    delta_frontier: dict[str, Any],
    router_calibration_nll_probe: dict[str, Any],
    router_calibration: dict[str, Any],
    router_margin_fragility: dict[str, Any],
    qwen3_moe_interpolation: dict[str, Any],
    qwen3_moe_complementary: dict[str, Any],
    qwen3_moe_base_coder: dict[str, Any],
    qwen3_downstream_matrix: dict[str, Any],
    qwen3_downstream_attribution: dict[str, Any],
    qwen3_downstream_confidence: dict[str, Any],
) -> list[dict[str, Any]]:
    selection = final_selection.get("current_selection") or {}
    router_selection = router_calibration.get("current_selection") or {}
    interpolation_gap = None
    if qwen3_moe_interpolation.get("best_interior_worst") is not None and qwen3_moe_interpolation.get(
        "endpoint_best_worst"
    ) is not None:
        interpolation_gap = (
            float(qwen3_moe_interpolation["best_interior_worst"])
            - float(qwen3_moe_interpolation["endpoint_best_worst"])
        )
    complementary_gap = None
    if qwen3_moe_complementary.get("best_merge_avg_nll") is not None and qwen3_moe_complementary.get(
        "best_source_avg_nll"
    ) is not None:
        complementary_gap = (
            float(qwen3_moe_complementary["best_merge_avg_nll"])
            - float(qwen3_moe_complementary["best_source_avg_nll"])
        )
    base_coder_gap = None
    if qwen3_moe_base_coder.get("best_interior_worst") is not None and qwen3_moe_base_coder.get(
        "endpoint_best_worst"
    ) is not None:
        base_coder_gap = (
            float(qwen3_moe_base_coder["best_interior_worst"])
            - float(qwen3_moe_base_coder["endpoint_best_worst"])
        )
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
            "probe": "qwen3_router_margin_fragility",
            "value": router_margin_fragility.get("top_fragility_score"),
            "threshold": router_margin_fragility.get("high_fragility_threshold"),
            "decision_signal": "topk_boundary_lambda_cap_rejects_direct_router_average",
            "evidence": (
                f"high-fragility layers = {router_margin_fragility.get('high_fragility_layer_count')}/"
                f"{router_margin_fragility.get('router_layer_count')}; "
                f"top layer = L{router_margin_fragility.get('top_fragile_layer')} "
                f"score {fmt(router_margin_fragility.get('top_fragility_score'))}; "
                f"top category = {router_margin_fragility.get('top_fragile_category')} "
                f"score {fmt(router_margin_fragility.get('top_category_fragility_score'))}; "
                f"min safe-lambda proxy = {fmt(router_margin_fragility.get('min_safe_lambda_proxy'))}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_straight_line_connectivity",
            "value": interpolation_gap,
            "threshold": 0.0,
            "decision_signal": "reject_source_to_source_linear_interpolation",
            "evidence": (
                f"best interior worst NLL = {fmt(qwen3_moe_interpolation.get('best_interior_worst'))}; "
                f"best endpoint worst NLL = {fmt(qwen3_moe_interpolation.get('endpoint_best_worst'))}; "
                f"interior gap = {fmt(interpolation_gap)}; "
                f"general barrier = {fmt(qwen3_moe_interpolation.get('barrier_general'))}; "
                f"task-vector cosine vs base = "
                f"{fmt((qwen3_moe_interpolation.get('geometry') or {}).get('cos(tauI,tauC)'))}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_complementary_pair_connectivity",
            "value": complementary_gap,
            "threshold": 0.0,
            "decision_signal": "do_not_assume_specialist_complementarity_is_averageable",
            "evidence": (
                f"best merge avg NLL = {fmt(qwen3_moe_complementary.get('best_merge_avg_nll'))}; "
                f"best source avg NLL = {fmt(qwen3_moe_complementary.get('best_source_avg_nll'))}; "
                f"merge-source gap = {fmt(complementary_gap)}; "
                f"best merge t = {fmt(qwen3_moe_complementary.get('best_merge_t'))}; "
                f"merge beats both sources = {qwen3_moe_complementary.get('merge_beats_both_sources_on_avg')}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_base_to_coder_connectivity",
            "value": base_coder_gap,
            "threshold": 0.0,
            "decision_signal": "source_delta_from_base_is_not_safe_without_gate",
            "evidence": (
                f"best interior worst NLL = {fmt(qwen3_moe_base_coder.get('best_interior_worst'))}; "
                f"best endpoint worst NLL = {fmt(qwen3_moe_base_coder.get('endpoint_best_worst'))}; "
                f"interior gap = {fmt(base_coder_gap)}; "
                f"general barrier = {fmt(qwen3_moe_base_coder.get('barrier_general'))}; "
                f"task-vector norm = {fmt((qwen3_moe_base_coder.get('geometry') or {}).get('norm_tauC'))}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_unified_mechanism_optimizer",
            "value": unified_candidate.get("selected_risk_weighted_predicted_relative_delta"),
            "threshold": unified_candidate.get("hard_cap"),
            "decision_signal": "use_router_evidence_geometry_risk_caps",
            "evidence": (
                f"selected = {unified_candidate.get('selected_candidate_id')}; "
                f"family = {unified_candidate.get('selected_candidate_family')}; "
                f"retention = {fmt(unified_candidate.get('selected_nonbase_mass_retention'))}; "
                f"risk-weighted predicted rel delta = "
                f"{fmt(unified_candidate.get('selected_risk_weighted_predicted_relative_delta'))}; "
                f"geometry-weighted predicted rel delta = "
                f"{fmt(unified_candidate.get('selected_geometry_weighted_predicted_relative_delta'))}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_unified_materialized_audit",
            "value": unified_audit.get("relative_delta_norm"),
            "threshold": delta_frontier.get("layer_chunk_total_relative_delta_norm"),
            "decision_signal": "materialized_same_shape_tail_reduction",
            "evidence": (
                f"audit status = {unified_audit.get('status')}; "
                f"total relative norm = {fmt(unified_audit.get('relative_delta_norm'))}; "
                f"router changed = {unified_audit.get('router_changed_tensors')}/"
                f"{unified_audit.get('router_tensors')}; "
                f"layer/chunk->unified norm reduction = "
                f"{fmt(delta_frontier.get('layer_chunk_to_unified_relative_norm_reduction'))}; "
                f"routed >0.65 reduction = {delta_frontier.get('layer_chunk_to_unified_routed_gt_065_reduction')}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_subspace_scaled_materialized_audit",
            "value": delta_frontier.get("subspace_scaled_total_relative_delta_norm"),
            "threshold": delta_frontier.get("unified_mechanism_total_relative_delta_norm"),
            "decision_signal": "test_uncovered_subspace_conflict_shrink",
            "evidence": (
                f"subspace total relative norm = "
                f"{fmt(delta_frontier.get('subspace_scaled_total_relative_delta_norm'))}; "
                f"unified->subspace norm reduction = "
                f"{fmt(delta_frontier.get('unified_to_subspace_relative_norm_reduction'), 6)}; "
                f"subspace routed >0.65 = {delta_frontier.get('subspace_scaled_routed_gt_0_65')}; "
                f"router changed = {delta_frontier.get('subspace_scaled_router_changed_tensors')}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_router_calibration_nll_probe",
            "value": router_calibration_nll_probe.get("worst_nll_reduction_vs_linear"),
            "threshold": 0.0,
            "decision_signal": "router_dispatch_is_real_optimization_lever",
            "evidence": (
                f"status = {router_calibration_nll_probe.get('status')}; "
                f"linear worst NLL = "
                f"{fmt((router_calibration_nll_probe.get('linear_merge') or {}).get('worst_nll'))}; "
                f"router-cal worst NLL = "
                f"{fmt((router_calibration_nll_probe.get('router_calibrated') or {}).get('worst_nll'))}; "
                f"worst reduction = {fmt(router_calibration_nll_probe.get('worst_nll_reduction_vs_linear'))}; "
                f"code gap to best source = {fmt(router_calibration_nll_probe.get('routercal_code_gap_to_best_source'))}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_generation_downstream_routercal_matrix",
            "value": qwen3_downstream_matrix.get("pair_routercal_avg_gain"),
            "threshold": 0.0,
            "decision_signal": "router_calibration_recovers_generation_interference_but_not_endpoint_dominance",
            "evidence": (
                f"status = {qwen3_downstream_matrix.get('status')}; "
                f"Instruct+Coder avg = {fmt(qwen3_downstream_matrix.get('pair_merge_avg'))}; "
                f"+router-cal avg = {fmt(qwen3_downstream_matrix.get('pair_routercal_avg'))}; "
                f"avg gain = {fmt(qwen3_downstream_matrix.get('pair_routercal_avg_gain'))}; "
                f"HumanEval gain = {fmt(qwen3_downstream_matrix.get('pair_routercal_humaneval_gain'))}; "
                f"gap to best parent avg = {fmt(qwen3_downstream_matrix.get('pair_routercal_gap_to_best_parent_avg'))}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_generation_routercal_effect_attribution",
            "value": qwen3_downstream_attribution.get("avg_routercal_recovery_fraction"),
            "threshold": 1.0,
            "decision_signal": "router_calibration_is_repair_not_acceptance_rule",
            "evidence": (
                f"status = {qwen3_downstream_attribution.get('status')}; "
                f"avg naive drop = {fmt(qwen3_downstream_attribution.get('avg_naive_drop_vs_pair_frontier'))}; "
                f"avg recovery fraction = {fmt(qwen3_downstream_attribution.get('avg_routercal_recovery_fraction'))}; "
                f"HumanEval recovery = {fmt(qwen3_downstream_attribution.get('humaneval_routercal_recovery_fraction'))}; "
                f"beats pair frontier = {qwen3_downstream_attribution.get('routercal_beats_pair_frontier_count')}/"
                f"{qwen3_downstream_attribution.get('score_count')}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_generation_confidence_audit",
            "value": qwen3_downstream_confidence.get("routercal_confident_positive_task_count_vs_naive"),
            "threshold": qwen3_downstream_confidence.get("task_count"),
            "decision_signal": "generation_gain_directional_not_confident_or_source_dominant",
            "evidence": (
                f"status = {qwen3_downstream_confidence.get('status')}; "
                f"positive tasks vs naive = "
                f"{qwen3_downstream_confidence.get('routercal_positive_task_count_vs_naive')}/"
                f"{qwen3_downstream_confidence.get('task_count')}; "
                f"confident positive tasks = "
                f"{qwen3_downstream_confidence.get('routercal_confident_positive_task_count_vs_naive')}/"
                f"{qwen3_downstream_confidence.get('task_count')}; "
                f"confident source-frontier wins = "
                f"{qwen3_downstream_confidence.get('routercal_confident_beats_pair_frontier_task_count')}/"
                f"{qwen3_downstream_confidence.get('task_count')}; "
                f"avg gain interval = ["
                f"{fmt(qwen3_downstream_confidence.get('routercal_avg_diff_lower_vs_naive'))}, "
                f"{fmt(qwen3_downstream_confidence.get('routercal_avg_diff_upper_vs_naive'))}]"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_router_calibration_gate",
            "value": router_selection.get("eligible_candidate_count"),
            "threshold": router_selection.get("candidate_count"),
            "decision_signal": "do_not_accept_router_delta_without_baseline_eval",
            "evidence": (
                f"status = {router_selection.get('status')}; "
                f"eligible router-cal candidates = {router_selection.get('eligible_candidate_count')}/"
                f"{router_selection.get('candidate_count')}; "
                f"reason = {router_selection.get('reason')}"
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


def build_decisions(
    features: pd.DataFrame,
    dense_selector: dict[str, Any],
    router_gate: dict[str, Any],
    unified_candidate: dict[str, Any],
    delta_frontier: dict[str, Any],
    router_calibration_nll_probe: dict[str, Any],
    router_calibration: dict[str, Any],
    router_margin_fragility: dict[str, Any],
    qwen3_moe_interpolation: dict[str, Any],
    qwen3_moe_complementary: dict[str, Any],
    qwen3_moe_base_coder: dict[str, Any],
    qwen3_downstream_matrix: dict[str, Any],
    qwen3_downstream_attribution: dict[str, Any],
    qwen3_downstream_confidence: dict[str, Any],
) -> pd.DataFrame:
    dense_config = ((dense_selector.get("results") or {}).get("unified") or {}).get("config") or {}
    router_selection = router_calibration.get("current_selection") or {}
    interpolation_gap = None
    if qwen3_moe_interpolation.get("best_interior_worst") is not None and qwen3_moe_interpolation.get(
        "endpoint_best_worst"
    ) is not None:
        interpolation_gap = (
            float(qwen3_moe_interpolation["best_interior_worst"])
            - float(qwen3_moe_interpolation["endpoint_best_worst"])
        )
    complementary_gap = None
    if qwen3_moe_complementary.get("best_merge_avg_nll") is not None and qwen3_moe_complementary.get(
        "best_source_avg_nll"
    ) is not None:
        complementary_gap = (
            float(qwen3_moe_complementary["best_merge_avg_nll"])
            - float(qwen3_moe_complementary["best_source_avg_nll"])
        )
    base_coder_gap = None
    if qwen3_moe_base_coder.get("best_interior_worst") is not None and qwen3_moe_base_coder.get(
        "endpoint_best_worst"
    ) is not None:
        base_coder_gap = (
            float(qwen3_moe_base_coder["best_interior_worst"])
            - float(qwen3_moe_base_coder["endpoint_best_worst"])
        )
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
            "stage": "moe_router_margin_cap_gate",
            "operation": "bound router movement by observed top-k margins",
            "condition": (
                "router interpolation can change a token's expert assignment when the router-logit movement "
                "exceeds that token's top-k margin; current min safe-lambda proxy is "
                f"{fmt(router_margin_fragility.get('min_safe_lambda_proxy'))}, with "
                f"{router_margin_fragility.get('high_fragility_layer_count')}/"
                f"{router_margin_fragility.get('router_layer_count')} high-fragility layers"
            ),
            "selected_action": (
                f"{router_margin_fragility.get('recommended_unified_router_action', 'freeze_router')}; "
                f"direct router average rejected by {router_margin_fragility.get('status')}"
            ),
            "why_it_should_improve": "It prevents a small weight-space router step from crossing a discrete dispatch boundary and sending tokens to untrained expert combinations.",
            "same_shape_invariant": "router tensors remain same-shape; any later route-KD delta must pass the same margin, load, and downstream gates",
        },
        {
            "stage": "moe_straight_line_connectivity_gate",
            "operation": "reject_unconditional_source_to_source_linear_interpolation",
            "condition": (
                f"Qwen3 Instruct/Coder best interior worst-NLL is above the best endpoint by "
                f"{fmt(interpolation_gap)}, and the complementary Thinking/Coder path does not beat its best source "
                f"(avg gap {fmt(complementary_gap)}); the Base/Coder line also keeps the best interior above "
                f"the best endpoint by {fmt(base_coder_gap)}"
            ),
            "selected_action": "use route/evidence/geometry-constrained same-shape candidates instead of a source-to-source midpoint",
            "why_it_should_improve": "It treats model connectivity as measured evidence: a smooth-looking line is not accepted unless an interior point beats the source frontier.",
            "same_shape_invariant": "candidate generation still writes the same tensor names and shapes; the gate only decides whether straight-line interpolation is allowed",
        },
        {
            "stage": "moe_expert_delta_optimizer",
            "operation": "apply retention-constrained router/evidence/geometry caps",
            "condition": "expert identity is aligned, direct router movement is rejected, and real expert geometry exposes nonuniform risk",
            "selected_action": (
                f"{unified_candidate.get('selected_candidate_id')} with hard cap "
                f"{fmt(unified_candidate.get('hard_cap'))}; "
                f"layer/chunk->unified routed >0.65 reduction = "
                f"{delta_frontier.get('layer_chunk_to_unified_routed_gt_065_reduction')}; "
                f"unified->subspace extra norm reduction = "
                f"{fmt(delta_frontier.get('unified_to_subspace_relative_norm_reduction'), 6)}"
            ),
            "why_it_should_improve": "It keeps useful Coder-route mass while shrinking high-risk routed expert deltas instead of using one global coefficient.",
            "same_shape_invariant": "only routed expert tensor values change; router, attention, embeddings, norms, names, and shapes stay fixed",
        },
        {
            "stage": "moe_router_calibration_gate",
            "operation": "treat router calibration as a separately audited ablation",
            "condition": "router-only NLL probe improves the linear MoE merge, but Qwen3 baseline/source vLLM eval is not complete",
            "selected_action": (
                f"nll_probe_worst_reduction={fmt(router_calibration_nll_probe.get('worst_nll_reduction_vs_linear'))}; "
                f"generation_avg_gain={fmt(qwen3_downstream_matrix.get('pair_routercal_avg_gain'))}; "
                f"generation_recovery_fraction={fmt(qwen3_downstream_attribution.get('avg_routercal_recovery_fraction'))}; "
                f"confidence_positive_tasks="
                f"{qwen3_downstream_confidence.get('routercal_confident_positive_task_count_vs_naive')}/"
                f"{qwen3_downstream_confidence.get('task_count')}; "
                f"confidence_source_frontier_wins="
                f"{qwen3_downstream_confidence.get('routercal_confident_beats_pair_frontier_task_count')}/"
                f"{qwen3_downstream_confidence.get('task_count')}; "
                f"{router_selection.get('status')}: {router_selection.get('reason')}"
            ),
            "why_it_should_improve": "It keeps router calibration as an active MoE-specific lever while still requiring source-dominance and task-regression gates before acceptance.",
            "same_shape_invariant": "any accepted router delta must keep the same router tensors and pass router-only audit caps",
        },
        {
            "stage": "moe_candidate_gate",
            "operation": "select only after audited downstream eval",
            "condition": "structural probes can rank risk, but source dominance and task regression need vLLM scores",
            "selected_action": "keep all registered Qwen3 candidates provisional until eval-bundle audit passes",
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
        "mechanism_equations": {
            "dense_second_order_gate": "accept a straight-line average only when the measured path loss does not exceed the endpoint frontier; local Fisher/RegMean curvature is treated as a proxy, not a proof",
            "moe_router_margin_gate": "for router logits z_lambda = z_A + lambda * (z_B - z_A), a top-k assignment can flip when lambda * ||Delta z|| reaches the observed top-k margin; therefore small empirical margins imply a near-zero safe router lambda",
            "same_shape_constraint": "all accepted operations must preserve model config, tokenizer, class, tensor names, tensor shapes, and MoE expert/router cardinalities",
        },
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
        f"- Dense lambda connectivity: midpoint worst NLL `{fmt(dense['lambda_linear_worst_nll'])}`，best lambda-family worst NLL `{fmt(dense['lambda_best_worst_nll'])}`。",
        f"- MoE: `{moe['decision']}`；真实 OLMoE same-name average degradation `{fmt(moe['real_gauge_naive_degradation'])}`，Qwen3 router action `{moe['router_action']}`。",
        f"- Qwen3 MoE straight-line connectivity: best interior worst NLL `{fmt(moe['qwen3_interpolation_best_interior_worst_nll'])}` vs best endpoint `{fmt(moe['qwen3_interpolation_endpoint_best_worst_nll'])}`；interior gap `{fmt(moe['qwen3_interpolation_interior_gap_nll'])}`，general barrier `{fmt(moe['qwen3_interpolation_general_barrier_nll'])}`。",
        f"- Qwen3 complementary path: best merge avg NLL `{fmt(moe['qwen3_complementary_best_merge_avg_nll'])}` vs best source avg `{fmt(moe['qwen3_complementary_best_source_avg_nll'])}`；merge beats sources `{moe['qwen3_complementary_merge_beats_sources']}`。",
        f"- Qwen3 Base->Coder path: best interior worst NLL `{fmt(moe['qwen3_base_coder_best_interior_worst_nll'])}` vs best endpoint `{fmt(moe['qwen3_base_coder_endpoint_best_worst_nll'])}`；interior gap `{fmt(moe['qwen3_base_coder_interior_gap_nll'])}`，general barrier `{fmt(moe['qwen3_base_coder_general_barrier_nll'])}`。",
        f"- Qwen3 unified mechanism: `{moe['qwen3_unified_candidate_id']}`；audit relative norm `{fmt(moe['qwen3_unified_relative_delta_norm'])}`，routed >0.65 `{moe['qwen3_unified_routed_gt_065']}`。",
        f"- Qwen3 subspace-scaled ablation: audit relative norm `{fmt(moe['qwen3_subspace_total_relative_delta_norm'])}`，unified->subspace norm reduction `{fmt(moe['qwen3_unified_to_subspace_relative_norm_reduction'], 6)}`，routed >0.65 `{moe['qwen3_subspace_routed_gt_065']}`。",
        f"- Qwen3 router margin fragility: high layers `{moe['qwen3_router_margin_high_fragility_layers']}/{moe['qwen3_router_margin_layer_count']}`，top `L{moe['qwen3_router_margin_top_layer']}` score `{fmt(moe['qwen3_router_margin_top_score'])}`，min safe-lambda proxy `{fmt(moe['qwen3_router_margin_min_safe_lambda_proxy'])}`。",
        f"- Qwen3 router NLL probe: worst-NLL reduction `{fmt(moe['qwen3_router_calibration_nll_worst_reduction'])}`，code gap to best source `{fmt(moe['qwen3_router_calibration_nll_code_gap_to_best_source'])}`。",
        f"- Qwen3 generation matrix: Instruct+Coder avg `{fmt(moe['qwen3_generation_pair_merge_avg'])}` -> router-cal avg `{fmt(moe['qwen3_generation_pair_routercal_avg'])}`；avg gain `{fmt(moe['qwen3_generation_pair_routercal_avg_gain'])}`，gap to best parent `{fmt(moe['qwen3_generation_pair_routercal_gap_to_best_parent_avg'])}`。",
        f"- Qwen3 generation attribution: router-cal recovers `{fmt(moe['qwen3_generation_avg_routercal_recovery_fraction'])}` of avg naive drop and beats pair frontier on `{moe['qwen3_generation_routercal_beats_pair_frontier_count']}/{moe['qwen3_generation_attribution_score_count']}` scores。",
        f"- Qwen3 generation confidence: positive tasks vs naive `{moe['qwen3_generation_routercal_positive_tasks_vs_naive']}/{moe['qwen3_generation_confidence_task_count']}`，confident positives `{moe['qwen3_generation_routercal_confident_positive_tasks']}/{moe['qwen3_generation_confidence_task_count']}`，confident source-frontier wins `{moe['qwen3_generation_routercal_confident_source_frontier_wins']}/{moe['qwen3_generation_confidence_task_count']}`；avg gain interval `[{fmt(moe['qwen3_generation_routercal_avg_gain_lower'])}, {fmt(moe['qwen3_generation_routercal_avg_gain_upper'])}]`。",
        f"- Qwen3 router calibration: `{moe['qwen3_router_calibration_status']}`。",
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
    dense_lambda = read_json(args.dense_lambda)
    gen_eval = read_json(args.dense_generation)
    mechanism = read_json(args.moe_mechanism)
    real_gauge = read_json(args.real_moe_gauge)
    moe_selector = read_json(args.moe_selector)
    router_gate = read_json(args.qwen3_router_gate)
    final_selection = read_json(args.qwen3_final_selection)
    unified_candidate = read_json(args.qwen3_unified_candidate)
    unified_audit = read_json(args.qwen3_unified_audit)
    delta_frontier = read_json(args.qwen3_delta_frontier)
    router_calibration_nll_probe = read_json(args.qwen3_router_calibration_nll_probe)
    router_calibration = read_json(args.qwen3_router_calibration)
    router_margin_fragility = read_json(args.qwen3_router_margin_fragility)
    qwen3_moe_interpolation = read_json(args.qwen3_moe_interpolation)
    qwen3_moe_complementary = read_json(args.qwen3_moe_complementary)
    qwen3_moe_base_coder = read_json(args.qwen3_moe_base_coder)
    qwen3_downstream_matrix = read_json(args.qwen3_downstream_matrix)
    qwen3_downstream_attribution = read_json(args.qwen3_downstream_attribution)
    qwen3_downstream_confidence = read_json(args.qwen3_downstream_confidence)

    feature_rows = dense_feature_rows(curvature, dense_selector, dense_lambda, gen_eval)
    feature_rows.extend(
        moe_feature_rows(
            mechanism,
            real_gauge,
            moe_selector,
            router_gate,
            final_selection,
            unified_candidate,
            unified_audit,
            delta_frontier,
            router_calibration_nll_probe,
            router_calibration,
            router_margin_fragility,
            qwen3_moe_interpolation,
            qwen3_moe_complementary,
            qwen3_moe_base_coder,
            qwen3_downstream_matrix,
            qwen3_downstream_attribution,
            qwen3_downstream_confidence,
        )
    )
    features = pd.DataFrame(feature_rows)
    decisions = build_decisions(
        features,
        dense_selector,
        router_gate,
        unified_candidate,
        delta_frontier,
        router_calibration_nll_probe,
        router_calibration,
        router_margin_fragility,
        qwen3_moe_interpolation,
        qwen3_moe_complementary,
        qwen3_moe_base_coder,
        qwen3_downstream_matrix,
        qwen3_downstream_attribution,
        qwen3_downstream_confidence,
    )
    algorithm = build_algorithm(decisions)

    dense_results = dense_selector.get("results") or {}
    final_current = final_selection.get("current_selection") or {}
    router_current = router_calibration.get("current_selection") or {}
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
            "lambda_linear_worst_nll": fnum(dense_lambda.get("linear_worst")),
            "lambda_best_worst_nll": fnum(dense_lambda.get("unified_best_worst")),
            "lambda_best_config": dense_lambda.get("unified_best_config"),
            "generation_best_method": gen_eval.get("best_method"),
        },
        "moe": {
            "decision": "align_experts_freeze_router_then_gate_candidate_by_vllm",
            "controlled_gauge_mse": fnum(mechanism.get("gauge_equivalence_mse")),
            "real_gauge_naive_degradation": fnum(real_gauge.get("naive_degradation_vs_baseline")),
            "real_gauge_aligned_degradation": fnum(real_gauge.get("aligned_degradation_vs_baseline")),
            "qwen3_identity_fraction": fnum(moe_selector.get("qwen3_identity_fraction")),
            "router_action": router_gate.get("recommended_unified_router_action"),
            "qwen3_unified_candidate_id": unified_candidate.get("selected_candidate_id"),
            "qwen3_unified_candidate_family": unified_candidate.get("selected_candidate_family"),
            "qwen3_unified_candidate_count": unified_candidate.get("candidate_count"),
            "qwen3_unified_nonbase_mass_retention": fnum(
                unified_candidate.get("selected_nonbase_mass_retention")
            ),
            "qwen3_unified_risk_weighted_predicted_relative_delta": fnum(
                unified_candidate.get("selected_risk_weighted_predicted_relative_delta")
            ),
            "qwen3_unified_geometry_weighted_predicted_relative_delta": fnum(
                unified_candidate.get("selected_geometry_weighted_predicted_relative_delta")
            ),
            "qwen3_unified_audit_status": unified_audit.get("status"),
            "qwen3_unified_relative_delta_norm": fnum(unified_audit.get("relative_delta_norm")),
            "qwen3_unified_router_changed_tensors": unified_audit.get("router_changed_tensors"),
            "qwen3_unified_router_tensors": unified_audit.get("router_tensors"),
            "qwen3_unified_routed_gt_065": delta_frontier.get("unified_mechanism_routed_gt_0_65"),
            "qwen3_router_margin_status": router_margin_fragility.get("status"),
            "qwen3_router_margin_high_fragility_layers": router_margin_fragility.get(
                "high_fragility_layer_count"
            ),
            "qwen3_router_margin_layer_count": router_margin_fragility.get("router_layer_count"),
            "qwen3_router_margin_top_layer": router_margin_fragility.get("top_fragile_layer"),
            "qwen3_router_margin_top_score": fnum(router_margin_fragility.get("top_fragility_score")),
            "qwen3_router_margin_top_category": router_margin_fragility.get("top_fragile_category"),
            "qwen3_router_margin_top_category_score": fnum(
                router_margin_fragility.get("top_category_fragility_score")
            ),
            "qwen3_router_margin_min_safe_lambda_proxy": fnum(
                router_margin_fragility.get("min_safe_lambda_proxy")
            ),
            "qwen3_router_calibration_nll_status": router_calibration_nll_probe.get("status"),
            "qwen3_router_calibration_nll_worst_reduction": fnum(
                router_calibration_nll_probe.get("worst_nll_reduction_vs_linear")
            ),
            "qwen3_router_calibration_nll_avg_reduction": fnum(
                router_calibration_nll_probe.get("avg_nll_reduction_vs_linear")
            ),
            "qwen3_router_calibration_nll_code_gap_to_best_source": fnum(
                router_calibration_nll_probe.get("routercal_code_gap_to_best_source")
            ),
            "qwen3_router_calibration_nll_worst_gap_to_best_source": fnum(
                router_calibration_nll_probe.get("routercal_worst_gap_to_best_source")
            ),
            "qwen3_generation_matrix_status": qwen3_downstream_matrix.get("status"),
            "qwen3_generation_best_avg_model": qwen3_downstream_matrix.get("best_avg_model"),
            "qwen3_generation_best_avg": fnum(qwen3_downstream_matrix.get("best_avg")),
            "qwen3_generation_pair_merge_avg": fnum(qwen3_downstream_matrix.get("pair_merge_avg")),
            "qwen3_generation_pair_routercal_avg": fnum(
                qwen3_downstream_matrix.get("pair_routercal_avg")
            ),
            "qwen3_generation_pair_routercal_avg_gain": fnum(
                qwen3_downstream_matrix.get("pair_routercal_avg_gain")
            ),
            "qwen3_generation_pair_routercal_humaneval_gain": fnum(
                qwen3_downstream_matrix.get("pair_routercal_humaneval_gain")
            ),
            "qwen3_generation_pair_routercal_gap_to_best_parent_avg": fnum(
                qwen3_downstream_matrix.get("pair_routercal_gap_to_best_parent_avg")
            ),
            "qwen3_generation_attribution_status": qwen3_downstream_attribution.get("status"),
            "qwen3_generation_attribution_score_count": qwen3_downstream_attribution.get("score_count"),
            "qwen3_generation_avg_naive_drop_vs_pair_frontier": fnum(
                qwen3_downstream_attribution.get("avg_naive_drop_vs_pair_frontier")
            ),
            "qwen3_generation_avg_routercal_recovery_fraction": fnum(
                qwen3_downstream_attribution.get("avg_routercal_recovery_fraction")
            ),
            "qwen3_generation_humaneval_routercal_recovery_fraction": fnum(
                qwen3_downstream_attribution.get("humaneval_routercal_recovery_fraction")
            ),
            "qwen3_generation_routercal_beats_pair_frontier_count": qwen3_downstream_attribution.get(
                "routercal_beats_pair_frontier_count"
            ),
            "qwen3_generation_confidence_status": qwen3_downstream_confidence.get("status"),
            "qwen3_generation_confidence_task_count": qwen3_downstream_confidence.get("task_count"),
            "qwen3_generation_routercal_positive_tasks_vs_naive": qwen3_downstream_confidence.get(
                "routercal_positive_task_count_vs_naive"
            ),
            "qwen3_generation_routercal_confident_positive_tasks": qwen3_downstream_confidence.get(
                "routercal_confident_positive_task_count_vs_naive"
            ),
            "qwen3_generation_routercal_confident_source_frontier_wins": qwen3_downstream_confidence.get(
                "routercal_confident_beats_pair_frontier_task_count"
            ),
            "qwen3_generation_routercal_avg_gain_lower": fnum(
                qwen3_downstream_confidence.get("routercal_avg_diff_lower_vs_naive")
            ),
            "qwen3_generation_routercal_avg_gain_upper": fnum(
                qwen3_downstream_confidence.get("routercal_avg_diff_upper_vs_naive")
            ),
            "qwen3_generation_routercal_avg_source_frontier_gap_lower": fnum(
                qwen3_downstream_confidence.get("routercal_avg_gap_lower_vs_pair_frontier")
            ),
            "qwen3_generation_routercal_avg_source_frontier_gap_upper": fnum(
                qwen3_downstream_confidence.get("routercal_avg_gap_upper_vs_pair_frontier")
            ),
            "qwen3_layer_chunk_to_unified_relative_norm_reduction": fnum(
                delta_frontier.get("layer_chunk_to_unified_relative_norm_reduction")
            ),
            "qwen3_layer_chunk_to_unified_routed_gt_065_reduction": delta_frontier.get(
                "layer_chunk_to_unified_routed_gt_065_reduction"
            ),
            "qwen3_subspace_total_relative_delta_norm": fnum(
                delta_frontier.get("subspace_scaled_total_relative_delta_norm")
            ),
            "qwen3_subspace_router_changed_tensors": delta_frontier.get(
                "subspace_scaled_router_changed_tensors"
            ),
            "qwen3_subspace_routed_gt_065": delta_frontier.get("subspace_scaled_routed_gt_0_65"),
            "qwen3_subspace_routed_gt_06505": delta_frontier.get("subspace_scaled_routed_gt_0_6505"),
            "qwen3_unified_to_subspace_relative_norm_reduction": fnum(
                delta_frontier.get("unified_to_subspace_relative_norm_reduction")
            ),
            "qwen3_unified_to_subspace_routed_gt_065_reduction": delta_frontier.get(
                "unified_to_subspace_routed_gt_065_reduction"
            ),
            "qwen3_router_calibration_status": router_current.get("status"),
            "qwen3_router_calibration_eligible_candidates": router_current.get(
                "eligible_candidate_count"
            ),
            "qwen3_router_calibration_candidate_count": router_current.get("candidate_count"),
            "qwen3_interpolation_best_interior_worst_nll": fnum(
                qwen3_moe_interpolation.get("best_interior_worst")
            ),
            "qwen3_interpolation_endpoint_best_worst_nll": fnum(
                qwen3_moe_interpolation.get("endpoint_best_worst")
            ),
            "qwen3_interpolation_interior_gap_nll": fnum(
                None
                if qwen3_moe_interpolation.get("best_interior_worst") is None
                or qwen3_moe_interpolation.get("endpoint_best_worst") is None
                else float(qwen3_moe_interpolation["best_interior_worst"])
                - float(qwen3_moe_interpolation["endpoint_best_worst"])
            ),
            "qwen3_interpolation_general_barrier_nll": fnum(qwen3_moe_interpolation.get("barrier_general")),
            "qwen3_complementary_best_merge_avg_nll": fnum(qwen3_moe_complementary.get("best_merge_avg_nll")),
            "qwen3_complementary_best_source_avg_nll": fnum(qwen3_moe_complementary.get("best_source_avg_nll")),
            "qwen3_complementary_merge_beats_sources": qwen3_moe_complementary.get(
                "merge_beats_both_sources_on_avg"
            ),
            "qwen3_base_coder_best_interior_worst_nll": fnum(qwen3_moe_base_coder.get("best_interior_worst")),
            "qwen3_base_coder_endpoint_best_worst_nll": fnum(qwen3_moe_base_coder.get("endpoint_best_worst")),
            "qwen3_base_coder_interior_gap_nll": fnum(
                None
                if qwen3_moe_base_coder.get("best_interior_worst") is None
                or qwen3_moe_base_coder.get("endpoint_best_worst") is None
                else float(qwen3_moe_base_coder["best_interior_worst"])
                - float(qwen3_moe_base_coder["endpoint_best_worst"])
            ),
            "qwen3_base_coder_general_barrier_nll": fnum(qwen3_moe_base_coder.get("barrier_general")),
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
    parser.add_argument("--dense-lambda", type=Path, default=Path("results/fp_dense_lambda/summary.json"))
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
    parser.add_argument(
        "--qwen3-unified-candidate",
        type=Path,
        default=Path("results/qwen3_moe_unified_mechanism_candidate/summary.json"),
    )
    parser.add_argument(
        "--qwen3-unified-audit",
        type=Path,
        default=Path("results/qwen3_moe_unified_mechanism_delta_audit/summary.json"),
    )
    parser.add_argument(
        "--qwen3-delta-frontier",
        type=Path,
        default=Path("results/qwen3_moe_delta_frontier/summary.json"),
    )
    parser.add_argument(
        "--qwen3-router-calibration-nll-probe",
        type=Path,
        default=Path("results/qwen3_moe_router_calibration_nll_probe/summary.json"),
    )
    parser.add_argument(
        "--qwen3-router-calibration",
        type=Path,
        default=Path("results/qwen3_moe_router_calibration_selection/summary.json"),
    )
    parser.add_argument(
        "--qwen3-downstream-matrix",
        type=Path,
        default=Path("results/fp_downstream_matrix/summary.json"),
    )
    parser.add_argument(
        "--qwen3-downstream-attribution",
        type=Path,
        default=Path("results/fp_downstream_attribution/summary.json"),
    )
    parser.add_argument(
        "--qwen3-downstream-confidence",
        type=Path,
        default=Path("results/fp_downstream_confidence_audit/summary.json"),
    )
    parser.add_argument(
        "--qwen3-router-margin-fragility",
        type=Path,
        default=Path("results/qwen3_moe_router_margin_fragility/summary.json"),
    )
    parser.add_argument(
        "--qwen3-moe-interpolation",
        type=Path,
        default=Path("results/fp_moe_barrier/summary.json"),
    )
    parser.add_argument(
        "--qwen3-moe-complementary",
        type=Path,
        default=Path("results/fp_moe_complementary/summary.json"),
    )
    parser.add_argument(
        "--qwen3-moe-base-coder",
        type=Path,
        default=Path("results/fp_moe_forgetting_base_coder/summary.json"),
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
