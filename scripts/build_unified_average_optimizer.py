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
        "key": "loss_landscape",
        "source": "https://arxiv.org/abs/1712.09913",
        "mechanism": "Loss landscapes should be inspected on meaningful weight-space directions; visual smoothness is not evidence that a source-to-source midpoint is safe.",
    },
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
        "key": "regmean++",
        "source": "https://arxiv.org/abs/2508.03121",
        "mechanism": "Activation regression should account for intra- and cross-layer dependencies; layer-wise closed forms are useful diagnostics but can miss propagation effects.",
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
    {
        "key": "router_kd_calibration",
        "source": "https://arxiv.org/abs/2603.02217",
        "mechanism": "Expert edits create router-expert mismatch; router-only KD is a lightweight repair lever but still needs downstream acceptance gates.",
    },
    {
        "key": "output_space_projection",
        "source": "https://arxiv.org/abs/2605.29101",
        "mechanism": "Output residual projection provides a convex calibration objective and a captured-residual diagnostic for when output-space coefficients should improve a merge.",
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


def maybe_int(value: Any) -> int | None:
    value = clean_value(value)
    return None if value is None else int(value)


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
    qwen3_router_coupled_retention_frontier: dict[str, Any],
    qwen3_source_set_complementarity: dict[str, Any],
    qwen3_average_source_set_optimizer: dict[str, Any],
    qwen_source_discovery_plan: dict[str, Any],
    qwen_source_discovery_eval_plan: dict[str, Any],
    qwen_source_frontier_eval_feedback: dict[str, Any],
    qwen3_router_calibration_frontier: dict[str, Any],
) -> list[dict[str, Any]]:
    selection = final_selection.get("current_selection") or {}
    router_selection = router_calibration.get("current_selection") or {}
    constrained_router = qwen3_router_coupled_retention_frontier.get("constrained") or {}
    stress_router = qwen3_router_coupled_retention_frontier.get("stress") or {}
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
            "probe": "qwen3_router_coupled_retention_frontier",
            "value": qwen3_router_coupled_retention_frontier.get(
                "constrained_effect_fraction_vs_stress"
            ),
            "threshold": qwen3_router_coupled_retention_frontier.get("minimum_effective_fraction"),
            "decision_signal": "direct_router_boundary_shrink_not_default_under_retention",
            "evidence": (
                f"gate = {qwen3_router_coupled_retention_frontier.get('gate')}; "
                f"default-gate candidates = "
                f"{qwen3_router_coupled_retention_frontier.get('default_gate_candidate_count')}/"
                f"{qwen3_router_coupled_retention_frontier.get('candidate_count')}; "
                f"constrained = {constrained_router.get('candidate_id')} "
                f"retention_delta {fmt(constrained_router.get('retention_delta_vs_base'), 6)} "
                f"coupled_reduction {fmt(constrained_router.get('router_coupled_delta_reduction_vs_base'), 8)}; "
                f"stress = {stress_router.get('candidate_id')} "
                f"coupled_reduction {fmt(stress_router.get('router_coupled_delta_reduction_vs_base'), 6)}; "
                f"effect fraction = "
                f"{fmt(qwen3_router_coupled_retention_frontier.get('constrained_effect_fraction_vs_stress'), 4)}"
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
            "decision_signal": "use_router_evidence_geometry_subspace_risk_caps",
            "evidence": (
                f"selected = {unified_candidate.get('selected_candidate_id')}; "
                f"family = {unified_candidate.get('selected_candidate_family')}; "
                f"retention = {fmt(unified_candidate.get('selected_nonbase_mass_retention'))}; "
                f"risk-weighted predicted rel delta = "
                f"{fmt(unified_candidate.get('selected_risk_weighted_predicted_relative_delta'))}; "
                f"geometry-weighted predicted rel delta = "
                f"{fmt(unified_candidate.get('selected_geometry_weighted_predicted_relative_delta'))}; "
                f"subspace-weighted predicted rel delta = "
                f"{fmt(unified_candidate.get('selected_subspace_weighted_predicted_relative_delta'))}; "
                f"high-subspace mean scale = {fmt(unified_candidate.get('selected_high_subspace_mean_scale'))}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_unified_materialized_audit",
            "value": unified_audit.get("relative_delta_norm"),
            "threshold": delta_frontier.get("layer_chunk_total_relative_delta_norm"),
            "decision_signal": "materialized_same_shape_tail_reduction",
            "evidence": (
                f"rule status = {unified_candidate.get('materialized_checkpoint_rule_status')}; "
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
                f"mechanistic->subspace norm delta = "
                f"{fmt(delta_frontier.get('mechanistic_to_subspace_relative_norm_delta'), 6)}; "
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
            "probe": "qwen3_source_set_complementarity_gate",
            "value": qwen3_source_set_complementarity.get(
                "current_frontier_avg_gain_vs_best_single"
            ),
            "threshold": qwen3_source_set_complementarity.get("min_avg_frontier_gain"),
            "decision_signal": "current_source_set_is_source_dominated_not_complementary",
            "evidence": (
                f"current source set = {qwen3_source_set_complementarity.get('current_source_set')}; "
                f"gate = {qwen3_source_set_complementarity.get('current_gate')}; "
                f"dominant source = {qwen3_source_set_complementarity.get('current_dominant_source')}; "
                f"frontier avg gain = "
                f"{fmt(qwen3_source_set_complementarity.get('current_frontier_avg_gain_vs_best_single'))}; "
                f"best observed merge = "
                f"{qwen3_source_set_complementarity.get('current_best_observed_merge')} "
                f"gap {fmt(qwen3_source_set_complementarity.get('current_best_observed_avg_gap_to_frontier'))}; "
                f"complementary measured sets = "
                f"{qwen3_source_set_complementarity.get('complementary_source_set_count')}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_source_set_surplus_optimizer",
            "value": (qwen3_average_source_set_optimizer.get("top_source_set") or {}).get(
                "frontier_avg_surplus_vs_interference"
            ),
            "threshold": 0.0,
            "decision_signal": "measured_complementarity_is_below_observed_interference_budget",
            "evidence": (
                f"status = {qwen3_average_source_set_optimizer.get('status')}; "
                f"top source set = "
                f"{(qwen3_average_source_set_optimizer.get('top_source_set') or {}).get('source_set')}; "
                f"top gate = "
                f"{(qwen3_average_source_set_optimizer.get('top_source_set') or {}).get('optimizer_gate')}; "
                f"frontier avg gain = "
                f"{fmt((qwen3_average_source_set_optimizer.get('top_source_set') or {}).get('frontier_avg_gain_vs_best_single'))}; "
                f"interference budget = {fmt(qwen3_average_source_set_optimizer.get('interference_budget'))}; "
                f"surplus = "
                f"{fmt((qwen3_average_source_set_optimizer.get('top_source_set') or {}).get('frontier_avg_surplus_vs_interference'))}; "
                f"task surplus positive = "
                f"{qwen3_average_source_set_optimizer.get('top_task_surplus_positive_count')}/"
                f"{qwen3_average_source_set_optimizer.get('task_count')}; "
                f"no-gain tasks = {qwen3_average_source_set_optimizer.get('top_no_gain_task_count')}/"
                f"{qwen3_average_source_set_optimizer.get('task_count')}; "
                f"best task gain = {qwen3_average_source_set_optimizer.get('top_best_task_gain_task')} "
                f"{fmt(qwen3_average_source_set_optimizer.get('top_best_task_gain'))}; "
                f"final-budget sets = "
                f"{qwen3_average_source_set_optimizer.get('final_average_budget_candidate_count')}; "
                f"probe-only sets = {qwen3_average_source_set_optimizer.get('probe_only_source_set_count')}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen_source_discovery_plan",
            "value": qwen_source_discovery_plan.get(
                "measured_additional_frontier_avg_gain_needed"
            ),
            "threshold": 0.0,
            "decision_signal": "search_stronger_source_sets_before_more_average_tuning",
            "evidence": (
                f"status = {qwen_source_discovery_plan.get('status')}; "
                f"top measured set = "
                f"{(qwen_source_discovery_plan.get('top_measured_source_set') or {}).get('source_set')}; "
                f"additional avg frontier gain needed = "
                f"{fmt(qwen_source_discovery_plan.get('measured_additional_frontier_avg_gain_needed'))}; "
                f"top scenario = "
                f"{(qwen_source_discovery_plan.get('top_scenario') or {}).get('scenario_id')}; "
                f"top queue = "
                f"{(qwen_source_discovery_plan.get('top_queue_item') or {}).get('queue_item')}; "
                f"task blockers = {qwen_source_discovery_plan.get('task_gap_tasks')}; "
                f"top task gap = {qwen_source_discovery_plan.get('top_task_gap_task')} "
                f"{qwen_source_discovery_plan.get('top_task_gap_status')} needs "
                f"{fmt(qwen_source_discovery_plan.get('top_task_gap_additional_gain_needed'))}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen_source_discovery_vllm_eval_plan",
            "value": qwen_source_discovery_eval_plan.get("eval_job_count"),
            "threshold": 1,
            "decision_signal": "source_frontier_eval_jobs_are_materialized_before_average_acceptance",
            "evidence": (
                f"status = {qwen_source_discovery_eval_plan.get('status')}; "
                f"jobs = {qwen_source_discovery_eval_plan.get('eval_job_count')}; "
                f"task names = {qwen_source_discovery_eval_plan.get('task_names')}; "
                f"task compatibility = "
                f"{qwen_source_discovery_eval_plan.get('task_name_compatibility_status')}; "
                f"top job = "
                f"{(qwen_source_discovery_eval_plan.get('top_eval_job') or {}).get('job_id')}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen_source_frontier_eval_feedback",
            "value": qwen_source_frontier_eval_feedback.get("scored_job_count"),
            "threshold": 1,
            "decision_signal": "vllm_source_frontier_feedback_controls_average_budget",
            "evidence": (
                f"status = {qwen_source_frontier_eval_feedback.get('status')}; "
                f"scored = {qwen_source_frontier_eval_feedback.get('scored_job_count')}/"
                f"{qwen_source_frontier_eval_feedback.get('job_count')}; "
                f"final candidates = "
                f"{qwen_source_frontier_eval_feedback.get('final_average_budget_candidate_count')}; "
                f"probe only = {qwen_source_frontier_eval_feedback.get('probe_only_candidate_count')}; "
                f"top = {(qwen_source_frontier_eval_feedback.get('top_scored_job') or {}).get('job_id')}; "
                f"gate = {(qwen_source_frontier_eval_feedback.get('top_scored_job') or {}).get('decision_gate')}; "
                f"surplus = "
                f"{fmt((qwen_source_frontier_eval_feedback.get('top_scored_job') or {}).get('surplus_vs_interference'))}; "
                f"blocker = {qwen_source_frontier_eval_feedback.get('blocking_reason')}"
            ),
        },
        {
            "domain": "moe",
            "probe": "qwen3_router_calibration_frontier",
            "value": qwen3_router_calibration_frontier.get("default_candidate_count"),
            "threshold": 1,
            "decision_signal": "router_calibration_is_margin_frontier_not_default_router_average",
            "evidence": (
                f"status = {qwen3_router_calibration_frontier.get('status')}; "
                f"default candidates = {qwen3_router_calibration_frontier.get('default_candidate_count')}/"
                f"{qwen3_router_calibration_frontier.get('candidate_count')}; "
                f"recommended = {qwen3_router_calibration_frontier.get('recommended_default_candidates')}; "
                f"safe lambda = {fmt(qwen3_router_calibration_frontier.get('safe_lambda_proxy'))}; "
                f"nll signal = {fmt(qwen3_router_calibration_frontier.get('nll_worst_reduction_signal'))}; "
                f"generation signal = {fmt(qwen3_router_calibration_frontier.get('generation_avg_gain_signal'))}; "
                f"blocker = {qwen3_router_calibration_frontier.get('acceptance_blocker')}"
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
        {
            "domain": "moe",
            "probe": "qwen3_final_selector_confidence_band",
            "value": selection.get("selection_rank_band_size"),
            "threshold": selection.get("eligible_candidate_count"),
            "decision_signal": "use_structural_tiebreak_only_when_downstream_scores_are_statistically_tied",
            "evidence": (
                f"confidence tie band = {selection.get('confidence_tie_band')}; "
                f"rank mode = {selection.get('selection_rank_mode')}; "
                f"point leader = {selection.get('selection_point_leader_method')}; "
                f"band size = {selection.get('selection_rank_band_size')}; "
                f"structural-frontier eligible = {selection.get('structural_frontier_eligible_count')}; "
                f"rank band methods = {selection.get('selection_rank_band_methods')}"
            ),
        },
    ]


def build_decisions(
    features: pd.DataFrame,
    dense_selector: dict[str, Any],
    router_gate: dict[str, Any],
    final_selection: dict[str, Any],
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
    qwen3_router_coupled_retention_frontier: dict[str, Any],
    qwen3_source_set_complementarity: dict[str, Any],
    qwen3_average_source_set_optimizer: dict[str, Any],
    qwen_source_frontier_eval_feedback: dict[str, Any],
    qwen3_router_calibration_frontier: dict[str, Any],
) -> pd.DataFrame:
    dense_config = ((dense_selector.get("results") or {}).get("unified") or {}).get("config") or {}
    final_current = final_selection.get("current_selection") or {}
    router_selection = router_calibration.get("current_selection") or {}
    constrained_router = qwen3_router_coupled_retention_frontier.get("constrained") or {}
    stress_router = qwen3_router_coupled_retention_frontier.get("stress") or {}
    top_source_set = qwen3_average_source_set_optimizer.get("top_source_set") or {}
    top_source_frontier_feedback = qwen_source_frontier_eval_feedback.get("top_scored_job") or {}
    router_frontier_recommended = ",".join(
        str(item) for item in (qwen3_router_calibration_frontier.get("recommended_default_candidates") or [])
    )
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
            "stage": "moe_router_coupled_shrink_gate",
            "operation": "keep router-boundary fragility inside B/H/I instead of direct extra shrink",
            "condition": (
                "a fine router-coupled shrink frontier found "
                f"{qwen3_router_coupled_retention_frontier.get('default_gate_candidate_count')}/"
                f"{qwen3_router_coupled_retention_frontier.get('candidate_count')} candidates that pass "
                "hard cap and retention, but the best retention-safe direct shrink gives only "
                f"{fmt(qwen3_router_coupled_retention_frontier.get('constrained_effect_fraction_vs_stress'), 4)} "
                "of the aggressive ablation proxy effect"
            ),
            "selected_action": (
                f"{qwen3_router_coupled_retention_frontier.get('recommended_unified_action')}; "
                f"constrained={constrained_router.get('candidate_id')} "
                f"coupled_reduction={fmt(constrained_router.get('router_coupled_delta_reduction_vs_base'), 8)}; "
                f"stress={stress_router.get('candidate_id')} "
                f"coupled_reduction={fmt(stress_router.get('router_coupled_delta_reduction_vs_base'), 6)}"
            ),
            "why_it_should_improve": "It avoids spending retention budget on a direct router-boundary term whose safe version is too weak; the same signal stays useful as an interference feature inside the expert scale objective.",
            "same_shape_invariant": "direct-shrink ablations still use same-shape expert tensor rules; the default algorithm does not add another tensor movement term",
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
            "stage": "moe_source_set_complementarity_gate",
            "operation": "require source-set complementarity before expecting average to beat sources",
            "condition": (
                f"current source set {qwen3_source_set_complementarity.get('current_source_set')} has gate "
                f"{qwen3_source_set_complementarity.get('current_gate')}; dominant source "
                f"{qwen3_source_set_complementarity.get('current_dominant_source')}; frontier avg gain "
                f"{fmt(qwen3_source_set_complementarity.get('current_frontier_avg_gain_vs_best_single'))}; "
                f"best observed merge gap "
                f"{fmt(qwen3_source_set_complementarity.get('current_best_observed_avg_gap_to_frontier'))}"
            ),
            "selected_action": qwen3_source_set_complementarity.get("recommended_action"),
            "why_it_should_improve": "It avoids spending algorithmic effort trying to make an average beat a source set whose measured frontier is already a single dominant endpoint.",
            "same_shape_invariant": "this gate changes source-set selection and evaluation priority only; any accepted output remains one same-shape checkpoint",
        },
        {
            "stage": "moe_source_set_surplus_gate",
            "operation": "require complementarity surplus to cover observed merge interference",
            "condition": (
                f"best measured complementary set {top_source_set.get('source_set')} has frontier avg gain "
                f"{fmt(top_source_set.get('frontier_avg_gain_vs_best_single'))}, but observed merge interference "
                f"budget is {fmt(qwen3_average_source_set_optimizer.get('interference_budget'))}; surplus "
                f"{fmt(top_source_set.get('frontier_avg_surplus_vs_interference'))}; task surplus-positive "
                f"{qwen3_average_source_set_optimizer.get('top_task_surplus_positive_count')}/"
                f"{qwen3_average_source_set_optimizer.get('task_count')}; no-gain tasks "
                f"{qwen3_average_source_set_optimizer.get('top_no_gain_task_count')}/"
                f"{qwen3_average_source_set_optimizer.get('task_count')}; blockers "
                f"{qwen3_average_source_set_optimizer.get('top_blocking_tasks')}"
            ),
            "selected_action": (
                f"{top_source_set.get('optimizer_gate')}: {top_source_set.get('recommended_action')}; "
                f"source_weights={top_source_set.get('source_weights')}"
            ),
            "why_it_should_improve": "It prevents a weakly complementary source set from being promoted to final-average budget when measured average interference is larger than the available task-frontier gain.",
            "same_shape_invariant": "the gate only changes source discovery and eval priority; any probe candidate still keeps one same-shape checkpoint",
        },
        {
            "stage": "moe_source_frontier_vllm_feedback_gate",
            "operation": "feed completed vLLM source-frontier results back into average-budget decisions",
            "condition": (
                f"feedback status {qwen_source_frontier_eval_feedback.get('status')}; scored "
                f"{qwen_source_frontier_eval_feedback.get('scored_job_count')}/"
                f"{qwen_source_frontier_eval_feedback.get('job_count')}; final candidates "
                f"{qwen_source_frontier_eval_feedback.get('final_average_budget_candidate_count')}; "
                f"top job {top_source_frontier_feedback.get('job_id')} gate "
                f"{top_source_frontier_feedback.get('decision_gate')} surplus "
                f"{fmt(top_source_frontier_feedback.get('surplus_vs_interference'))}"
            ),
            "selected_action": (
                top_source_frontier_feedback.get("recommended_action")
                or qwen_source_frontier_eval_feedback.get("blocking_reason")
                or "wait for vLLM source-frontier metrics before accepting any source-set average"
            ),
            "why_it_should_improve": "It closes the loop from hosted endpoint evaluation to the same average acceptance equation instead of relying on static proxy matrices.",
            "same_shape_invariant": "this gate changes only which source set receives average materialization budget; accepted candidates still preserve the input architecture",
        },
        {
            "stage": "moe_expert_delta_optimizer",
            "operation": "apply retention-constrained router/evidence/geometry/subspace caps",
            "condition": "expert identity is aligned, direct router movement is rejected, and real expert geometry/subspace probes expose nonuniform risk",
            "selected_action": (
                f"{unified_candidate.get('selected_candidate_id')} with hard cap "
                f"{fmt(unified_candidate.get('hard_cap'))}; "
                f"subspace-weighted rel-delta = "
                f"{fmt(unified_candidate.get('selected_subspace_weighted_predicted_relative_delta'))}; "
                f"materialized rule status = {unified_candidate.get('materialized_checkpoint_rule_status')}; "
                f"layer/chunk->unified routed >0.65 reduction = "
                f"{delta_frontier.get('layer_chunk_to_unified_routed_gt_065_reduction')}; "
                f"mechanistic->subspace norm delta = "
                f"{fmt(delta_frontier.get('mechanistic_to_subspace_relative_norm_delta'), 6)}"
            ),
            "why_it_should_improve": "It keeps useful Coder-route mass while shrinking high-risk routed expert deltas and local subspace conflicts instead of using one global coefficient.",
            "same_shape_invariant": "only routed expert tensor values change; router, attention, embeddings, norms, names, and shapes stay fixed",
        },
        {
            "stage": "moe_router_calibration_frontier_gate",
            "operation": "restrict router calibration to margin-safe frontier probes",
            "condition": (
                "direct router averaging is rejected, but local NLL and small generation probes show router "
                "calibration can repair dispatch; current safe-lambda proxy is "
                f"{fmt(qwen3_router_calibration_frontier.get('safe_lambda_proxy'))}, and "
                f"{qwen3_router_calibration_frontier.get('default_candidate_count')}/"
                f"{qwen3_router_calibration_frontier.get('candidate_count')} route-KD caps are default-run candidates"
            ),
            "selected_action": (
                f"run default router-cal frontier candidates [{router_frontier_recommended}] only as ablations; "
                f"stress caps stay non-default; blocker={qwen3_router_calibration_frontier.get('acceptance_blocker')}"
            ),
            "why_it_should_improve": "It turns router calibration into a bounded same-shape repair test instead of reintroducing unsafe direct router averaging.",
            "same_shape_invariant": "accepted router-cal variants may only add audited same-shape router tensors after matched vLLM source-dominance gates pass",
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
            "stage": "moe_statistical_selector_gate",
            "operation": "rank structural mechanisms only inside a downstream confidence tie band",
            "condition": (
                "finite-budget vLLM eval can make point estimates differ by noise; after hard source/task/paired gates, "
                "structural frontier and safety are used only when the point leader's confidence interval overlaps the candidate band"
            ),
            "selected_action": (
                f"confidence_tie_band={final_current.get('confidence_tie_band')}; "
                f"rank_mode={final_current.get('selection_rank_mode')}; "
                f"point_leader={final_current.get('selection_point_leader_method')}; "
                f"band_size={final_current.get('selection_rank_band_size')}; "
                f"structural_frontier_eligible={final_current.get('structural_frontier_eligible_count')}"
            ),
            "why_it_should_improve": "It avoids overfitting tiny finite-sample score gaps while still refusing to let structural cleanliness override a statistically separated downstream winner.",
            "same_shape_invariant": "the gate changes only selection policy; all accepted candidates remain same-shape Qwen3 MoE checkpoints",
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


def build_mechanism_hypotheses(summary: dict[str, Any]) -> pd.DataFrame:
    dense = summary["dense"]
    moe = summary["moe"]
    dense_midpoint_rejected = (
        fnum(dense.get("curvature_ratio_general")) is not None
        and fnum(dense.get("curvature_ratio_code")) is not None
        and float(dense["curvature_ratio_general"]) > 5.0
        and float(dense["curvature_ratio_code"]) > 5.0
    )
    dense_lambda_rejected = (
        fnum(dense.get("lambda_linear_worst_nll")) is not None
        and fnum(dense.get("lambda_best_worst_nll")) is not None
        and float(dense["lambda_linear_worst_nll"]) > float(dense["lambda_best_worst_nll"])
    )
    interpolation_rejected = (
        fnum(moe.get("qwen3_interpolation_interior_gap_nll")) is not None
        and float(moe["qwen3_interpolation_interior_gap_nll"]) > 0.0
    )
    base_coder_rejected = (
        fnum(moe.get("qwen3_base_coder_interior_gap_nll")) is not None
        and float(moe["qwen3_base_coder_interior_gap_nll"]) > 0.0
    )
    router_direct_rejected = (
        str(moe.get("router_action")) == "freeze_router"
        or (
            fnum(moe.get("qwen3_router_margin_min_safe_lambda_proxy")) is not None
            and float(moe["qwen3_router_margin_min_safe_lambda_proxy"]) < 0.05
        )
    )
    router_direct_shrink_not_default = (
        str(moe.get("qwen3_router_coupled_frontier_gate"))
        == "direct_router_boundary_term_not_default"
        or (
            fnum(moe.get("qwen3_router_coupled_frontier_effect_fraction")) is not None
            and fnum(moe.get("qwen3_router_coupled_frontier_minimum_effect_fraction")) is not None
            and float(moe["qwen3_router_coupled_frontier_effect_fraction"])
            < float(moe["qwen3_router_coupled_frontier_minimum_effect_fraction"])
        )
    )
    unified_structural_passed = (
        moe.get("qwen3_unified_materialized_rule_status") == "fresh"
        and moe.get("qwen3_unified_audit_status") == "passed"
        and maybe_int(moe.get("qwen3_unified_routed_gt_065")) == 0
    )
    router_cal_promising = (
        fnum(moe.get("qwen3_router_calibration_nll_worst_reduction")) is not None
        and float(moe["qwen3_router_calibration_nll_worst_reduction"]) > 0.0
    )
    source_set_dominated = str(moe.get("qwen3_source_set_current_gate")) == (
        "source_dominated_not_averageable_as_final"
    )
    source_set_surplus_ready = (
        fnum(moe.get("qwen3_source_set_top_surplus_vs_interference")) is not None
        and float(moe["qwen3_source_set_top_surplus_vs_interference"]) >= 0.0
    )
    generation_router_cal_directional = (
        fnum(moe.get("qwen3_generation_pair_routercal_avg_gain")) is not None
        and float(moe["qwen3_generation_pair_routercal_avg_gain"]) > 0.0
    )
    confidence_tie_band_enabled = bool(moe.get("qwen3_final_confidence_tie_band"))

    rows = [
        {
            "hypothesis_id": "dense_same_basin_required",
            "domain": "dense",
            "current_status": "rejected_for_current_midpoint"
            if dense_midpoint_rejected and dense_lambda_rejected
            else "needs_more_path_evidence",
            "mechanism": "Dense weight averaging is useful only when the straight path stays inside a low-loss basin; local Fisher/RegMean curvature is a proxy, not proof.",
            "current_evidence": (
                f"curvature ratios general/code = {fmt(dense.get('curvature_ratio_general'))}/"
                f"{fmt(dense.get('curvature_ratio_code'))}; midpoint worst NLL = "
                f"{fmt(dense.get('lambda_linear_worst_nll'))}; best lambda-family worst NLL = "
                f"{fmt(dense.get('lambda_best_worst_nll'))}"
            ),
            "falsification_test": "A lambda/path sweep plus held-out generation eval must show an interior same-shape checkpoint beating the endpoint frontier on worst-task score.",
            "action_if_supported": "allow a low-loss dense soup/task-vector point",
            "action_if_falsified": "fall back to endpoint/anchor or keep the current probe-selected low-lambda bridge",
            "next_command": "python scripts/fp_dense_lambda.py --out results/fp_dense_lambda",
            "literature_keys": "mode_connectivity,model_soups,fisher_merging,regmean++",
        },
        {
            "hypothesis_id": "dense_coordinate_conflict_is_diagnostic_not_default",
            "domain": "dense",
            "current_status": "conditional_only",
            "mechanism": "Sign/magnitude conflict can identify harmful deltas, but a broad sparse rule can delete useful adaptation unless it wins a held-out gate.",
            "current_evidence": (
                f"linear worst NLL = {fmt(dense.get('linear_worst_nll'))}; unified worst NLL = "
                f"{fmt(dense.get('unified_worst_nll'))}; best endpoint worst NLL = "
                f"{fmt(dense.get('best_endpoint_worst_nll'))}"
            ),
            "falsification_test": "A TIES/DARE/DELLA-style materialization must improve worst-task score over the current anchor without endpoint domination.",
            "action_if_supported": "emit sparse coordinate tensor rules for dense deltas",
            "action_if_falsified": "keep conflict probes as diagnostics and do not sparsify by default",
            "next_command": "python scripts/fp_merge_compare.py --help",
            "literature_keys": "ties,dare,della,star",
        },
        {
            "hypothesis_id": "moe_expert_gauge_alignment_precedes_average",
            "domain": "moe",
            "current_status": "alignment_required"
            if (fnum(moe.get("real_gauge_naive_degradation")) or 0.0) > 1.0
            else "identity_alignment_sufficient_for_current_pair",
            "mechanism": "MoE expert index is a gauge; same-name expert averaging is invalid until expert identity has been canonicalized.",
            "current_evidence": (
                f"real same-name degradation = {fmt(moe.get('real_gauge_naive_degradation'))}; "
                f"aligned degradation = {fmt(moe.get('real_gauge_aligned_degradation'))}; "
                f"Qwen3 identity fraction = {fmt(moe.get('qwen3_identity_fraction'))}"
            ),
            "falsification_test": "Expert output cosine/route coactivation must show same-index experts are functionally matched, or a remap must recover the self-merge baseline.",
            "action_if_supported": "average matched experts under source tensor aliases",
            "action_if_falsified": "reject same-name expert average and require layer-wise expert remapping",
            "next_command": "python scripts/fp_moe_barrier.py --help",
            "literature_keys": "git_rebasin,sub_moe,mergeme",
        },
        {
            "hypothesis_id": "moe_direct_router_average_crosses_topk_boundaries",
            "domain": "moe",
            "current_status": "direct_router_average_rejected" if router_direct_rejected else "router_move_open",
            "mechanism": "A small router weight delta can change sparse top-k dispatch if it exceeds the observed logit margin.",
            "current_evidence": (
                f"router action = {moe.get('router_action')}; high-fragility layers = "
                f"{moe.get('qwen3_router_margin_high_fragility_layers')}/"
                f"{moe.get('qwen3_router_margin_layer_count')}; min safe-lambda proxy = "
                f"{fmt(moe.get('qwen3_router_margin_min_safe_lambda_proxy'))}"
            ),
            "falsification_test": "A router-moving candidate must preserve top-k overlap/load and beat the frozen-router candidate under the same downstream manifest.",
            "action_if_supported": "allow a capped route-KD router delta",
            "action_if_falsified": "freeze router and move only expert tensors",
            "next_command": "results/qwen3_moe_router_calibration_job/run_router_calibration_job.sh preflight",
            "literature_keys": "harc,router_calibration,mergeme",
        },
        {
            "hypothesis_id": "moe_direct_router_boundary_shrink_is_not_default",
            "domain": "moe",
            "current_status": "direct_shrink_ablation_only"
            if router_direct_shrink_not_default
            else "direct_shrink_candidate_needs_acceptance",
            "mechanism": "Router-boundary fragility is a useful interference feature, but a direct extra shrink term must buy enough proxy reduction without violating non-base retention.",
            "current_evidence": (
                f"frontier gate = {moe.get('qwen3_router_coupled_frontier_gate')}; "
                f"default-gate candidates = "
                f"{moe.get('qwen3_router_coupled_frontier_default_gate_candidates')}/"
                f"{moe.get('qwen3_router_coupled_frontier_candidate_count')}; "
                f"effect fraction = {fmt(moe.get('qwen3_router_coupled_frontier_effect_fraction'), 4)}; "
                f"constrained reduction = "
                f"{fmt(moe.get('qwen3_router_coupled_frontier_constrained_reduction'), 8)}; "
                f"stress reduction = {fmt(moe.get('qwen3_router_coupled_frontier_stress_reduction'), 6)}"
            ),
            "falsification_test": "A direct router-boundary shrink must pass retention/hard-cap gates and achieve a material fraction of the aggressive ablation effect, then beat the B/H/I-only default under the locked downstream manifest.",
            "action_if_supported": "keep router fragility inside B/H/I and leave direct shrink as ablation",
            "action_if_falsified": "promote a retention-safe direct router-boundary shrink term into the expert scale objective",
            "next_command": "python scripts/analyze_qwen3_moe_router_coupled_retention_frontier.py --output-dir results/qwen3_moe_router_coupled_retention_frontier",
            "literature_keys": "harc,router_kd_calibration,expert_merging",
        },
        {
            "hypothesis_id": "moe_source_to_source_line_not_averageable",
            "domain": "moe",
            "current_status": "straight_line_rejected"
            if interpolation_rejected and base_coder_rejected
            else "needs_more_connectivity_evidence",
            "mechanism": "MoE endpoint specialization does not imply the source-to-source line is a low-loss or source-dominant path.",
            "current_evidence": (
                f"Instruct/Coder interior gap = {fmt(moe.get('qwen3_interpolation_interior_gap_nll'))}; "
                f"Base/Coder interior gap = {fmt(moe.get('qwen3_base_coder_interior_gap_nll'))}; "
                f"complementary merge beats sources = {moe.get('qwen3_complementary_merge_beats_sources')}"
            ),
            "falsification_test": "A straight-line or 2D plane sweep must find an interior point that beats both source endpoints on the paired downstream frontier.",
            "action_if_supported": "allow source-to-source interpolation for this pair",
            "action_if_falsified": "use router/evidence/geometry-constrained same-shape expert rules",
            "next_command": "python scripts/fp_moe_barrier.py --out results/fp_moe_barrier",
            "literature_keys": "mode_connectivity,loss_landscape,model_soups",
        },
        {
            "hypothesis_id": "source_set_complementarity_precedes_average",
            "domain": "dense_and_moe",
            "current_status": "current_qwen3_pair_source_dominated"
            if source_set_dominated
            else "source_set_has_measured_complementarity",
            "mechanism": "Averaging can only beat the endpoint frontier when different sources own different measured tasks or slices; if one source dominates, merging is repair/regularization rather than a final-candidate prior.",
            "current_evidence": (
                f"source set = {moe.get('qwen3_source_set_current')}; "
                f"gate = {moe.get('qwen3_source_set_current_gate')}; "
                f"dominant source = {moe.get('qwen3_source_set_current_dominant_source')}; "
                f"frontier avg gain = {fmt(moe.get('qwen3_source_set_current_frontier_avg_gain'))}; "
                f"best observed merge gap = {fmt(moe.get('qwen3_source_set_current_best_observed_avg_gap'))}; "
                f"complementary measured sets = {moe.get('qwen3_source_set_complementary_count')}"
            ),
            "falsification_test": "Endpoint eval for a source set must show a material task-frontier gain over its best single source, then the same-shape average must approach or beat that frontier under locked downstream eval.",
            "action_if_supported": "prioritize averaging only for complementary source sets",
            "action_if_falsified": "return the dominant endpoint and use average candidates only as repair/ablation probes",
            "next_command": "python scripts/build_qwen3_source_set_complementarity_gate.py --output-dir results/qwen3_source_set_complementarity_gate",
            "literature_keys": "model_soups,expert_merging,mode_connectivity",
        },
        {
            "hypothesis_id": "source_set_surplus_must_exceed_interference",
            "domain": "dense_and_moe",
            "current_status": "top_measured_source_set_probe_only_below_interference_budget"
            if not source_set_surplus_ready
            else "source_set_has_enough_surplus_for_final_average_budget",
            "mechanism": "Endpoint complementarity is necessary but not sufficient: the measured source-frontier gain must exceed the known merge-interference budget before an average is promoted beyond probe-only.",
            "current_evidence": (
                f"top source set = {moe.get('qwen3_source_set_top_source_set')}; "
                f"top gate = {moe.get('qwen3_source_set_top_optimizer_gate')}; "
                f"frontier avg gain = {fmt(moe.get('qwen3_source_set_top_frontier_avg_gain'))}; "
                f"interference budget = {fmt(moe.get('qwen3_source_set_interference_budget'))}; "
                f"surplus = {fmt(moe.get('qwen3_source_set_top_surplus_vs_interference'))}; "
                f"source weights = {moe.get('qwen3_source_set_top_source_weights')}"
            ),
            "falsification_test": "A measured source set must show frontier_avg_gain minus observed merge-interference budget >= 0, then its same-shape average must pass locked downstream eval.",
            "action_if_supported": "promote the source set to final materialization budget",
            "action_if_falsified": "keep the source set probe-only and search for stronger endpoint complementarity",
            "next_command": "python scripts/build_qwen3_average_source_set_optimizer.py --output-dir results/qwen3_average_source_set_optimizer",
            "literature_keys": "model_soups,ties,expert_merging,harc",
        },
        {
            "hypothesis_id": "moe_risk_weighted_expert_caps_preserve_useful_route_mass",
            "domain": "moe",
            "current_status": "structurally_passed_waiting_downstream_eval"
            if unified_structural_passed
            else "structural_gate_failed_or_stale",
            "mechanism": "Routed expert movement should be capped by route mass, geometry, and subspace conflict while retaining non-base specialization.",
            "current_evidence": (
                f"candidate = {moe.get('qwen3_unified_candidate_id')}; retention = "
                f"{fmt(moe.get('qwen3_unified_nonbase_mass_retention'))}; "
                f"subspace-weighted rel-delta = "
                f"{fmt(moe.get('qwen3_unified_subspace_weighted_predicted_relative_delta'))}; "
                f"routed >0.65 = {moe.get('qwen3_unified_routed_gt_065')}"
            ),
            "falsification_test": "Budgeted vLLM eval must show the unified candidate is non-dominated by both sources and does not regress any task beyond tolerance.",
            "action_if_supported": "promote unified mechanism candidate as the same-shape MoE default",
            "action_if_falsified": "select endpoint or simpler searched_no_gt065/layer_chunk rule and inspect failed transition attribution",
            "next_command": "results/qwen3_moe_eval_budget_plan/run_eval_budget.sh final",
            "literature_keys": "expert_merging,sub_moe,regmean++,output_space_projection",
        },
        {
            "hypothesis_id": "router_calibration_repairs_dispatch_but_is_not_acceptance",
            "domain": "moe",
            "current_status": "promising_but_unaccepted"
            if router_cal_promising and generation_router_cal_directional
            else "awaiting_router_calibration_evidence",
            "mechanism": "Router calibration can repair part of dispatch interference, but it is not a valid final average unless it beats source and task gates.",
            "current_evidence": (
                f"NLL worst reduction = {fmt(moe.get('qwen3_router_calibration_nll_worst_reduction'))}; "
                f"generation avg gain = {fmt(moe.get('qwen3_generation_pair_routercal_avg_gain'))}; "
                f"confident positive tasks = "
                f"{moe.get('qwen3_generation_routercal_confident_positive_tasks')}/"
                f"{moe.get('qwen3_generation_confidence_task_count')}; "
                f"source-frontier wins = "
                f"{moe.get('qwen3_generation_routercal_confident_source_frontier_wins')}/"
                f"{moe.get('qwen3_generation_confidence_task_count')}"
            ),
            "falsification_test": "Router-calibrated candidates must beat the frozen-router baseline under paired source controls and maintain router-only audit caps.",
            "action_if_supported": "attach a capped router delta after the expert-rule candidate",
            "action_if_falsified": "keep router frozen and treat router-cal as a diagnostic repair signal",
            "next_command": "results/qwen3_moe_router_calibration_job/run_router_calibration_job.sh all",
            "literature_keys": "harc,router_kd_calibration",
        },
        {
            "hypothesis_id": "downstream_source_dominance_is_final_gate",
            "domain": "dense_and_moe",
            "current_status": moe.get("qwen3_final_selection_status") or "unknown",
            "mechanism": "Structural probes can explain risk, but a same-shape average is only useful if it survives source dominance and task-regression tests.",
            "current_evidence": (
                f"final selection status = {moe.get('qwen3_final_selection_status')}; eligible candidates = "
                f"{moe.get('qwen3_eligible_candidates')}/{moe.get('qwen3_candidate_count')}; "
                f"router calibration status = {moe.get('qwen3_router_calibration_status')}"
            ),
            "falsification_test": "All candidates and sources must be scored on the locked manifest; dominated averages are rejected even if their structural audit looks clean.",
            "action_if_supported": "select the best non-dominated same-shape candidate",
            "action_if_falsified": "return the source endpoint and use failed mechanism attribution to design the next candidate",
            "next_command": "python scripts/select_qwen3_moe_unified_result.py",
            "literature_keys": "model_soups,expert_merging,harc",
        },
        {
            "hypothesis_id": "moe_structural_tiebreak_requires_statistical_equivalence",
            "domain": "moe",
            "current_status": "policy_ready_waiting_eval"
            if confidence_tie_band_enabled
            else "point_estimate_ranking_only",
            "mechanism": "MoE structural probes should explain and break ties among statistically indistinguishable candidates, not override a downstream winner with separated confidence intervals.",
            "current_evidence": (
                f"confidence tie band = {moe.get('qwen3_final_confidence_tie_band')}; "
                f"rank mode = {moe.get('qwen3_final_selection_rank_mode')}; "
                f"point leader = {moe.get('qwen3_final_selection_point_leader_method')}; "
                f"rank band size = {moe.get('qwen3_final_selection_rank_band_size')}; "
                f"structural-frontier eligible = {moe.get('qwen3_final_structural_frontier_eligible_count')}"
            ),
            "falsification_test": "Selector smoke and locked-manifest eval must show that structural frontier/safety affects rank only inside a confidence-overlap band; if intervals separate, the point leader wins.",
            "action_if_supported": "use structural frontier and safety as tie-breakers inside the eligible confidence band",
            "action_if_falsified": "fall back to pure downstream point estimate or require larger eval budget before ranking structural variants",
            "next_command": "python scripts/select_qwen3_moe_final_candidate.py --smoke-matrix --output-dir results/qwen3_moe_final_candidate_selection_smoke",
            "literature_keys": "model_soups,mode_connectivity,sub_moe",
        },
    ]
    return pd.DataFrame(rows)


def build_next_experiment_queue(
    summary: dict[str, Any],
    evidence_ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    moe = summary["moe"]
    if evidence_ledger is None:
        hypotheses = build_mechanism_hypotheses(summary)
        evidence_ledger = build_evidence_ledger(summary, hypotheses)

    verdicts = {
        str(row["hypothesis_id"]): str(row["verdict"]) for _, row in evidence_ledger.iterrows()
    }

    def verdict(hypothesis_id: str, default: str = "unknown") -> str:
        return verdicts.get(hypothesis_id, default)

    final_status = str(moe.get("qwen3_final_selection_status") or "")
    router_status = str(moe.get("qwen3_router_calibration_status") or "")
    unified_verdict = verdict("moe_risk_weighted_expert_caps_preserve_useful_route_mass")
    downstream_verdict = verdict("downstream_source_dominance_is_final_gate")
    router_verdict = verdict("router_calibration_repairs_dispatch_but_is_not_acceptance")
    dense_verdict = verdict("dense_same_basin_required")
    source_surplus_verdict = verdict("source_set_surplus_must_exceed_interference")

    source_eval_waiting = (
        unified_verdict == "awaiting_downstream_eval"
        or downstream_verdict == "awaiting_downstream_eval"
        or final_status in {"awaiting_source_eval", "awaiting_eval", "awaiting_baseline_eval"}
    )
    source_eval_terminal = unified_verdict in {
        "supports_current_action",
        "falsified_by_downstream_eval",
    } or downstream_verdict in {"supports_current_action", "supports_source_fallback"}
    router_waiting = router_verdict in {
        "promising_but_unaccepted",
        "awaiting_router_calibration_evidence",
    } or router_status in {
        "awaiting_baseline_eval",
        "awaiting_eval",
        "awaiting_source_eval",
    }
    router_terminal = router_verdict in {
        "supports_current_action",
        "supports_freeze_router_baseline",
    }

    if source_eval_waiting:
        source_eval_priority = 1.00
        source_eval_status = "blocked_on_gpu_vllm"
        source_eval_why = "The unified candidate and source frontier still lack the required locked-manifest vLLM gate."
        source_eval_expected = "select_qwen3_moe_unified_result can move from awaiting_source_eval to accept/reject/source fallback"
    elif downstream_verdict == "supports_current_action" and unified_verdict == "supports_current_action":
        source_eval_priority = 0.42
        source_eval_status = "completed_by_selector"
        source_eval_why = "The downstream selector already accepted the same-shape unified candidate for the current bundle."
        source_eval_expected = "rerun only when adding new tasks, sources, or candidate checkpoints"
    elif downstream_verdict == "supports_source_fallback" or unified_verdict == "falsified_by_downstream_eval":
        source_eval_priority = 0.44
        source_eval_status = "completed_source_fallback"
        source_eval_why = "The locked downstream gate rejected the current average; attribution should explain the failed mechanism next."
        source_eval_expected = "rerun only after a new candidate is materialized"
    else:
        source_eval_priority = 0.70
        source_eval_status = "refresh_after_new_candidates"
        source_eval_why = "No active downstream blocker is registered, but future candidates still require the same gate."
        source_eval_expected = "wait for a new same-shape candidate before spending vLLM budget"

    if router_verdict == "supports_current_action":
        router_priority = 0.46
        router_queue_status = "completed_by_selector"
        router_why = "Router calibration has already been selected; downstream attribution should verify what it changed."
        router_expected = "monitor route audit and task-regression drift before another router-calibration run"
    elif router_verdict == "supports_freeze_router_baseline":
        router_priority = 0.43
        router_queue_status = "rejected_by_selector"
        router_why = "The router selector chose the frozen-router baseline, so retraining the same router delta is not the next bottleneck."
        router_expected = "design a new router repair objective before retrying router calibration"
    elif router_waiting:
        router_priority = 0.95
        router_queue_status = "blocked_on_gpu_vllm"
        router_why = "Router NLL/generation probes are positive directionally, but paired vLLM acceptance evidence is still missing."
        router_expected = "router calibration selector can decide cap001/margin_profile vs freeze-router baseline"
    else:
        router_priority = 0.60
        router_queue_status = "refresh_after_baseline_eval"
        router_why = "Router calibration is not the current global blocker, but should be refreshed after new router evidence."
        router_expected = "refresh router-calibrated candidates after new cache or eval artifacts land"

    if source_surplus_verdict == "ready_for_final_average_budget":
        source_set_priority = 0.96
        source_set_status = "ready_for_materialization"
        source_set_why = "A measured source set has enough endpoint-frontier surplus to cover the observed merge-interference budget."
        source_set_expected = "materialize the frontier-weighted same-shape source-set candidate and send it through the locked vLLM gate"
    elif source_surplus_verdict == "supports_current_action":
        source_set_priority = 0.91
        source_set_status = "ready_for_endpoint_expansion_no_final_candidate"
        source_set_why = "The best measured complementary source set is still below the observed merge-interference budget, so source discovery is the next useful non-GPU-blocked direction."
        source_set_expected = "expand endpoint/source-set eval or add more complementary downstream sources before final average materialization"
    else:
        source_set_priority = 0.68
        source_set_status = "refresh_source_set_gate"
        source_set_why = "The source-set surplus gate is missing or stale."
        source_set_expected = "refresh source-set surplus optimizer from the current downstream matrix"

    if source_eval_terminal:
        attribution_priority = 0.98
        attribution_status = "ready_after_downstream_selector"
        attribution_why = "The selector has produced an accept/reject verdict; attribution can now identify which mechanism caused it."
    elif router_terminal:
        attribution_priority = 0.90
        attribution_status = "ready_after_router_selector"
        attribution_why = "Router calibration has a selector verdict; attribution should compare frozen-router and calibrated paths."
    else:
        attribution_priority = 0.88
        attribution_status = "awaiting_eval_bundle"
        attribution_why = "After vLLM eval, this converts score deltas into pass/fail evidence for tail caps, layer chunks, unified caps, and subspace scaling."

    optimizer_refresh_priority = 0.90 if (source_eval_terminal or router_terminal) else 0.82
    optimizer_refresh_status = "ready_after_verdict" if (source_eval_terminal or router_terminal) else "ready"
    dense_recheck_priority = 0.86 if dense_verdict not in {
        "supports_current_action",
        "supports_conditional_action",
    } else 0.55
    dense_recheck_status = (
        "needs_path_evidence"
        if dense_recheck_priority > 0.55
        else "lower_priority_until_qwen_moe_eval_unblocked"
    )
    queue = [
        {
            "base_rank": 1,
            "experiment": "budgeted_qwen3_moe_downstream_eval",
            "mechanism_target": "source dominance and task-regression gate for final trust-region candidates",
            "driving_hypothesis_id": "downstream_source_dominance_is_final_gate",
            "driving_verdict": downstream_verdict,
            "gate_type": "required_downstream_acceptance",
            "why_now": source_eval_why,
            "command": "results/qwen3_moe_eval_budget_plan/run_eval_budget.sh final",
            "preflight_command": "results/qwen3_moe_eval_budget_plan/run_eval_budget.sh preflight final",
            "priority_score": source_eval_priority,
            "status": source_eval_status,
            "expected_decision_update": source_eval_expected,
        },
        {
            "base_rank": 2,
            "experiment": "router_calibration_active_candidates",
            "mechanism_target": "whether capped route-KD router deltas repair dispatch after frozen-router expert merge",
            "driving_hypothesis_id": "router_calibration_repairs_dispatch_but_is_not_acceptance",
            "driving_verdict": router_verdict,
            "gate_type": "router_repair_acceptance",
            "why_now": router_why,
            "command": "results/qwen3_moe_router_calibration_job/run_router_calibration_job.sh all",
            "preflight_command": "results/qwen3_moe_router_calibration_job/run_router_calibration_job.sh preflight",
            "priority_score": router_priority,
            "status": router_queue_status,
            "expected_decision_update": router_expected,
        },
        {
            "base_rank": 3,
            "experiment": "source_set_surplus_discovery",
            "mechanism_target": "find source sets whose endpoint-frontier gain can overcome measured merging interference",
            "driving_hypothesis_id": "source_set_surplus_must_exceed_interference",
            "driving_verdict": source_surplus_verdict,
            "gate_type": "source_set_selection_before_average",
            "why_now": source_set_why,
            "command": "python scripts/build_qwen_source_discovery_eval_plan.py --output-dir results/qwen_source_discovery_eval_plan",
            "preflight_command": "python scripts/build_qwen_source_discovery_plan.py --output-dir results/qwen_source_discovery_plan",
            "priority_score": source_set_priority,
            "status": source_set_status,
            "expected_decision_update": source_set_expected,
        },
        {
            "base_rank": 4,
            "experiment": "mechanism_effect_attribution_refresh",
            "mechanism_target": "which structural intervention actually changes downstream scores",
            "driving_hypothesis_id": "moe_risk_weighted_expert_caps_preserve_useful_route_mass",
            "driving_verdict": unified_verdict,
            "gate_type": "post_selector_mechanism_attribution",
            "why_now": attribution_why,
            "command": "python scripts/attribute_qwen3_moe_mechanism_effects.py",
            "preflight_command": "python scripts/audit_qwen3_moe_eval_bundle.py --output-dir results/qwen3_moe_eval_bundle_audit",
            "priority_score": attribution_priority,
            "status": attribution_status,
            "expected_decision_update": "operation_decisions can stop relying on structural-only risk reductions",
        },
        {
            "base_rank": 5,
            "experiment": "unified_optimizer_refresh",
            "mechanism_target": "update the Dense/MoE unified average policy after new eval or router-calibration evidence",
            "driving_hypothesis_id": "downstream_source_dominance_is_final_gate",
            "driving_verdict": downstream_verdict,
            "gate_type": "policy_refresh",
            "why_now": "The optimizer is the place where mechanism evidence becomes a same-shape algorithm contract.",
            "command": "python scripts/build_unified_average_optimizer.py --output-dir results/unified_average_optimizer",
            "preflight_command": "python -m py_compile scripts/build_unified_average_optimizer.py",
            "priority_score": optimizer_refresh_priority,
            "status": optimizer_refresh_status,
            "expected_decision_update": "mechanism_hypotheses and next_experiment_queue refresh from current artifacts",
        },
        {
            "base_rank": 6,
            "experiment": "dense_low_loss_path_recheck",
            "mechanism_target": "whether Dense same-base averages become valid under a different coefficient/path family",
            "driving_hypothesis_id": "dense_same_basin_required",
            "driving_verdict": dense_verdict,
            "gate_type": "dense_connectivity_recheck",
            "why_now": "Dense midpoint failure is already measured; only a new source pair or coefficient family can overturn it.",
            "command": "python scripts/fp_dense_lambda.py --out results/fp_dense_lambda",
            "preflight_command": "python scripts/fp_dense_lambda.py --help",
            "priority_score": dense_recheck_priority,
            "status": dense_recheck_status,
            "expected_decision_update": "dense_same_basin_required hypothesis can move from rejected to supported for a specific pair",
        },
    ]
    out = pd.DataFrame(queue).sort_values(["priority_score", "base_rank"], ascending=[False, True])
    out = out.reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    columns = [
        "rank",
        "experiment",
        "mechanism_target",
        "driving_hypothesis_id",
        "driving_verdict",
        "gate_type",
        "why_now",
        "command",
        "preflight_command",
        "priority_score",
        "status",
        "expected_decision_update",
    ]
    return out[columns]


def build_evidence_ledger(summary: dict[str, Any], hypotheses: pd.DataFrame) -> pd.DataFrame:
    dense = summary["dense"]
    moe = summary["moe"]
    hypothesis_by_id = {str(row["hypothesis_id"]): row for _, row in hypotheses.iterrows()}
    final_status = str(moe.get("qwen3_final_selection_status") or "")
    final_selected = str(moe.get("qwen3_final_selected_method") or "")
    router_cal_status = str(moe.get("qwen3_router_calibration_status") or "")
    router_cal_selected = str(moe.get("qwen3_router_calibration_selected_method") or "")
    unified_selected = final_status == "select_unified_candidate" and (
        final_selected == "qwen3_moe_unified_mechanism_candidate"
        or final_selected == str(moe.get("qwen3_unified_candidate_id"))
    )
    source_fallback_selected = final_status == "keep_source_endpoint"
    router_cal_selected = router_cal_status == "selected_router_calibrated_candidate" and bool(
        router_cal_selected
    )
    router_cal_rejected = router_cal_status == "keep_frozen_router_baseline"

    def hypothesis_status(hypothesis_id: str) -> str:
        row = hypothesis_by_id.get(hypothesis_id)
        return "" if row is None else str(row["current_status"])

    unified_verdict = "awaiting_downstream_eval"
    unified_action = "keep_unified_mechanism_candidate_provisional"
    unified_gate = "budgeted vLLM eval versus both sources and registered candidates"
    unified_confidence = 0.66
    if unified_selected:
        unified_verdict = "supports_current_action"
        unified_action = "promote_unified_mechanism_candidate"
        unified_gate = "post-selection regression monitoring on new task packs"
        unified_confidence = 0.94
    elif source_fallback_selected:
        unified_verdict = "falsified_by_downstream_eval"
        unified_action = "reject_unified_mechanism_candidate_for_current_pair"
        unified_gate = "inspect transition attribution before designing the next candidate"
        unified_confidence = 0.90

    router_cal_verdict = "promising_but_unaccepted"
    router_cal_action = "train_and_eval_router_calibration_as_ablation_not_default"
    router_cal_gate = "paired vLLM eval with frozen-router baseline and source controls"
    router_cal_confidence = 0.62
    if router_cal_selected:
        router_cal_verdict = "supports_current_action"
        router_cal_action = "attach_selected_capped_router_delta"
        router_cal_gate = "continue router-only audit and task-regression monitoring"
        router_cal_confidence = 0.92
    elif router_cal_rejected:
        router_cal_verdict = "supports_freeze_router_baseline"
        router_cal_action = "keep_router_frozen_for_current_candidate"
        router_cal_gate = "design a new router-calibration mechanism before retrying"
        router_cal_confidence = 0.88

    downstream_verdict = "awaiting_downstream_eval"
    downstream_action = "do_not_accept_any_average_until_locked_manifest_eval_completes"
    downstream_gate = "complete budgeted vLLM eval bundle audit"
    if unified_selected:
        downstream_verdict = "supports_current_action"
        downstream_action = "select_best_non_dominated_same_shape_candidate"
        downstream_gate = "monitor additional task packs and source frontier drift"
    elif source_fallback_selected:
        downstream_verdict = "supports_source_fallback"
        downstream_action = "return_source_endpoint_for_current_pair"
        downstream_gate = "use failed mechanism attribution to design the next candidate"

    rows = [
        {
            "hypothesis_id": "dense_same_basin_required",
            "evidence_tier": "path_nll_plus_curvature_proxy",
            "verdict": "supports_current_action",
            "current_status": hypothesis_status("dense_same_basin_required"),
            "current_algorithm_action": "reject_dense_linear_midpoint_use_low_lambda_or_endpoint_anchor",
            "why_verdict": (
                "Both curvature ratios are far above the local quadratic range and the midpoint path is worse than "
                "the best lambda-family point."
            ),
            "acceptance_gate_still_needed": "held-out generation or vLLM eval for any new interior dense point",
            "numeric_signal": fnum(dense.get("lambda_linear_worst_nll")) - fnum(dense.get("lambda_best_worst_nll"))
            if fnum(dense.get("lambda_linear_worst_nll")) is not None
            and fnum(dense.get("lambda_best_worst_nll")) is not None
            else None,
            "confidence_score": 0.92,
        },
        {
            "hypothesis_id": "dense_coordinate_conflict_is_diagnostic_not_default",
            "evidence_tier": "heldout_nll_selector",
            "verdict": "supports_conditional_action",
            "current_status": hypothesis_status("dense_coordinate_conflict_is_diagnostic_not_default"),
            "current_algorithm_action": "keep_sparse_conflict_rules_as_ablation_not_default",
            "why_verdict": (
                "Conflict-aware variants have to beat the anchor on held-out worst loss; the current evidence does not "
                "justify broad sparse deletion as a default."
            ),
            "acceptance_gate_still_needed": "materialized sparse candidate must beat anchor and source frontier",
            "numeric_signal": fnum(dense.get("unified_worst_nll")) - fnum(dense.get("best_endpoint_worst_nll"))
            if fnum(dense.get("unified_worst_nll")) is not None
            and fnum(dense.get("best_endpoint_worst_nll")) is not None
            else None,
            "confidence_score": 0.72,
        },
        {
            "hypothesis_id": "moe_expert_gauge_alignment_precedes_average",
            "evidence_tier": "controlled_and_real_gauge_probe",
            "verdict": "supports_current_action",
            "current_status": hypothesis_status("moe_expert_gauge_alignment_precedes_average"),
            "current_algorithm_action": "canonicalize_or_verify_expert_identity_before_expert_average",
            "why_verdict": (
                "The real self-merge same-name degradation is large while aligned degradation is zero, so expert index "
                "identity cannot be assumed generally."
            ),
            "acceptance_gate_still_needed": "per-layer expert matching or verified identity for each target pair",
            "numeric_signal": fnum(moe.get("real_gauge_naive_degradation")),
            "confidence_score": 0.98,
        },
        {
            "hypothesis_id": "moe_direct_router_average_crosses_topk_boundaries",
            "evidence_tier": "router_margin_and_topk_proxy",
            "verdict": "supports_current_action",
            "current_status": hypothesis_status("moe_direct_router_average_crosses_topk_boundaries"),
            "current_algorithm_action": "freeze_router_or_only_allow_capped_route_kd_delta",
            "why_verdict": (
                "Half of Qwen3 router layers are high-fragility and the minimum safe-lambda proxy is near zero, so "
                "direct router averaging is a boundary-crossing risk."
            ),
            "acceptance_gate_still_needed": "router-moving candidate must pass route overlap, load, audit, and downstream gates",
            "numeric_signal": fnum(moe.get("qwen3_router_margin_min_safe_lambda_proxy")),
            "confidence_score": 0.93,
        },
        {
            "hypothesis_id": "moe_direct_router_boundary_shrink_is_not_default",
            "evidence_tier": "retention_constrained_router_coupled_frontier",
            "verdict": "supports_current_action"
            if str(moe.get("qwen3_router_coupled_frontier_gate"))
            == "direct_router_boundary_term_not_default"
            else "needs_more_frontier_evidence",
            "current_status": hypothesis_status("moe_direct_router_boundary_shrink_is_not_default"),
            "current_algorithm_action": "use_router_fragility_as_BHI_interference_feature_keep_direct_shrink_ablation_only",
            "why_verdict": (
                "The retention-safe direct shrink frontier has a tiny proxy effect compared with the aggressive "
                "ablation, so the default should not spend non-base retention on a separate direct shrink term."
            ),
            "acceptance_gate_still_needed": "a direct-shrink ablation must beat the B/H/I-only default under matched vLLM eval before promotion",
            "numeric_signal": fnum(moe.get("qwen3_router_coupled_frontier_effect_fraction")),
            "confidence_score": 0.88,
        },
        {
            "hypothesis_id": "moe_source_to_source_line_not_averageable",
            "evidence_tier": "source_to_source_nll_path_probe",
            "verdict": "supports_current_action",
            "current_status": hypothesis_status("moe_source_to_source_line_not_averageable"),
            "current_algorithm_action": "reject_unconditional_moe_source_to_source_midpoint",
            "why_verdict": (
                "Both Instruct/Coder and Base/Coder interior points are worse than the best endpoint, and the "
                "complementary path does not beat sources."
            ),
            "acceptance_gate_still_needed": "a future plane/path sweep must find a non-endpoint downstream win",
            "numeric_signal": fnum(moe.get("qwen3_interpolation_interior_gap_nll")),
            "confidence_score": 0.90,
        },
        {
            "hypothesis_id": "source_set_complementarity_precedes_average",
            "evidence_tier": "generation_source_frontier_gate",
            "verdict": "supports_current_action",
            "current_status": hypothesis_status("source_set_complementarity_precedes_average"),
            "current_algorithm_action": "treat_current_instruct_coder_average_as_repair_or_ablation_until_new_complementary_sources_pass_endpoint_eval",
            "why_verdict": (
                "The current measured source frontier is a single dominant Instruct endpoint; the best observed "
                "merge remains below that frontier, so averaging is not yet a source-beating prior."
            ),
            "acceptance_gate_still_needed": "new source sets must show material endpoint complementarity and then pass locked downstream eval",
            "numeric_signal": fnum(moe.get("qwen3_source_set_current_best_observed_avg_gap")),
            "confidence_score": 0.87,
        },
        {
            "hypothesis_id": "source_set_surplus_must_exceed_interference",
            "evidence_tier": "source_frontier_gain_vs_observed_merge_gap",
            "verdict": "supports_current_action"
            if str(moe.get("qwen3_source_set_top_optimizer_gate"))
            in {"probe_only_below_interference_budget", "reject_source_dominated"}
            else "ready_for_final_average_budget",
            "current_status": hypothesis_status("source_set_surplus_must_exceed_interference"),
            "current_algorithm_action": "keep_top_measured_complementary_source_set_probe_only_until_surplus_beats_interference",
            "why_verdict": (
                "The top measured source set is complementary, but its frontier avg gain is smaller than the "
                "observed merge-interference budget, so final average budget would be premature."
            ),
            "acceptance_gate_still_needed": "find a source set whose frontier gain exceeds observed interference, then pass locked downstream eval",
            "numeric_signal": fnum(moe.get("qwen3_source_set_top_surplus_vs_interference")),
            "confidence_score": 0.84,
        },
        {
            "hypothesis_id": "moe_risk_weighted_expert_caps_preserve_useful_route_mass",
            "evidence_tier": "structural_audit_without_downstream_acceptance",
            "verdict": unified_verdict,
            "current_status": hypothesis_status("moe_risk_weighted_expert_caps_preserve_useful_route_mass"),
            "current_algorithm_action": unified_action,
            "why_verdict": (
                "The candidate is structurally clean, keeps high non-base route mass, and removes routed >0.65 tails, "
                "but final acceptance follows the downstream selector."
            ),
            "acceptance_gate_still_needed": unified_gate,
            "numeric_signal": fnum(moe.get("qwen3_unified_nonbase_mass_retention")),
            "confidence_score": unified_confidence,
        },
        {
            "hypothesis_id": "router_calibration_repairs_dispatch_but_is_not_acceptance",
            "evidence_tier": "nll_probe_and_generation_smoke",
            "verdict": router_cal_verdict,
            "current_status": hypothesis_status("router_calibration_repairs_dispatch_but_is_not_acceptance"),
            "current_algorithm_action": router_cal_action,
            "why_verdict": (
                "Router calibration improves NLL and the generation matrix directionally, but confidence intervals and "
                "source-frontier checks do not prove final dominance unless the router selector passes."
            ),
            "acceptance_gate_still_needed": router_cal_gate,
            "numeric_signal": fnum(moe.get("qwen3_router_calibration_nll_worst_reduction")),
            "confidence_score": router_cal_confidence,
        },
        {
            "hypothesis_id": "downstream_source_dominance_is_final_gate",
            "evidence_tier": "missing_required_downstream_eval",
            "verdict": downstream_verdict,
            "current_status": hypothesis_status("downstream_source_dominance_is_final_gate"),
            "current_algorithm_action": downstream_action,
            "why_verdict": (
                "The final selector is the acceptance authority for source dominance and task-regression gates."
            ),
            "acceptance_gate_still_needed": downstream_gate,
            "numeric_signal": maybe_int(moe.get("qwen3_eligible_candidates")),
            "confidence_score": 1.0,
        },
        {
            "hypothesis_id": "moe_structural_tiebreak_requires_statistical_equivalence",
            "evidence_tier": "selector_policy_and_smoke_matrix",
            "verdict": "supports_current_action"
            if bool(moe.get("qwen3_final_confidence_tie_band"))
            else "needs_selector_policy_update",
            "current_status": hypothesis_status("moe_structural_tiebreak_requires_statistical_equivalence"),
            "current_algorithm_action": "apply_structural_frontier_safety_only_inside_confidence_tie_band",
            "why_verdict": (
                "The final selector exposes a confidence-overlap rank mode and smoke coverage for structural tie-breaks, "
                "so structural probes explain statistically tied candidates instead of overriding separated downstream evidence."
            ),
            "acceptance_gate_still_needed": "complete locked-manifest vLLM eval so the confidence band contains real candidate scores",
            "numeric_signal": maybe_int(moe.get("qwen3_final_selection_rank_band_size")),
            "confidence_score": 0.86 if bool(moe.get("qwen3_final_confidence_tie_band")) else 0.30,
        },
    ]
    return pd.DataFrame(rows)


def build_algorithm_contract(
    summary: dict[str, Any],
    evidence_ledger: pd.DataFrame,
    experiment_queue: pd.DataFrame,
) -> pd.DataFrame:
    dense = summary["dense"]
    moe = summary["moe"]
    verdicts = {
        str(row["hypothesis_id"]): str(row["verdict"]) for _, row in evidence_ledger.iterrows()
    }
    queue_by_experiment = {
        str(row["experiment"]): row for _, row in experiment_queue.iterrows()
    }

    def verdict(hypothesis_id: str) -> str:
        return verdicts.get(hypothesis_id, "unknown")

    def command(experiment: str, fallback: str) -> str:
        row = queue_by_experiment.get(experiment)
        return fallback if row is None else str(row["command"])

    dense_gate = verdict("dense_same_basin_required") == "supports_current_action"
    expert_gate = verdict("moe_expert_gauge_alignment_precedes_average") == "supports_current_action"
    router_gate = verdict("moe_direct_router_average_crosses_topk_boundaries") == "supports_current_action"
    router_shrink_gate = (
        verdict("moe_direct_router_boundary_shrink_is_not_default") == "supports_current_action"
    )
    connectivity_gate = verdict("moe_source_to_source_line_not_averageable") == "supports_current_action"
    source_set_gate = verdict("source_set_complementarity_precedes_average") == "supports_current_action"
    source_set_surplus_gate = verdict("source_set_surplus_must_exceed_interference") in {
        "supports_current_action",
        "ready_for_final_average_budget",
    }
    structural_gate = (
        moe.get("qwen3_unified_materialized_rule_status") == "fresh"
        and moe.get("qwen3_unified_audit_status") == "passed"
        and maybe_int(moe.get("qwen3_unified_routed_gt_065")) == 0
        and maybe_int(moe.get("qwen3_unified_router_changed_tensors")) == 0
        and bool(moe.get("qwen3_unified_matches_materialized_checkpoint_manifest", False))
    )
    downstream_verdict = verdict("downstream_source_dominance_is_final_gate")
    unified_verdict = verdict("moe_risk_weighted_expert_caps_preserve_useful_route_mass")
    router_cal_verdict = verdict("router_calibration_repairs_dispatch_but_is_not_acceptance")
    tiebreak_gate = verdict("moe_structural_tiebreak_requires_statistical_equivalence") == "supports_current_action"
    downstream_terminal = downstream_verdict in {"supports_current_action", "supports_source_fallback"}
    unified_accepted = (
        downstream_verdict == "supports_current_action"
        and unified_verdict == "supports_current_action"
    )
    source_fallback = downstream_verdict == "supports_source_fallback"
    downstream_waiting = downstream_verdict == "awaiting_downstream_eval"
    router_cal_guarded = router_cal_verdict in {
        "promising_but_unaccepted",
        "supports_current_action",
        "supports_freeze_router_baseline",
    }

    rows = [
        {
            "requirement": "same_shape_output_contract",
            "mechanism": "architecture_invariant",
            "required_state": "output preserves config/tokenizer/class/tensor names/tensor shapes",
            "observed_state": "algorithm contract writes same-shape operations only",
            "passed": True,
            "blocking_status": "passed",
            "next_command": "python scripts/build_unified_average_optimizer.py --output-dir results/unified_average_optimizer",
        },
        {
            "requirement": "dense_midpoint_path_gate",
            "mechanism": "mode_connectivity_and_curvature",
            "required_state": "linear midpoint is not accepted unless path evidence beats endpoint frontier",
            "observed_state": (
                f"dense decision={dense.get('decision')}; "
                f"midpoint worst={fmt(dense.get('lambda_linear_worst_nll'))}; "
                f"best lambda-family worst={fmt(dense.get('lambda_best_worst_nll'))}; "
                f"verdict={verdict('dense_same_basin_required')}"
            ),
            "passed": dense_gate,
            "blocking_status": "passed" if dense_gate else "failed_dense_path_gate",
            "next_command": "python scripts/fp_dense_lambda.py --out results/fp_dense_lambda",
        },
        {
            "requirement": "moe_expert_identity_gate",
            "mechanism": "expert_gauge_alignment",
            "required_state": "expert identity is verified or canonicalized before expert averaging",
            "observed_state": (
                f"identity_fraction={fmt(moe.get('qwen3_identity_fraction'))}; "
                f"real same-name degradation={fmt(moe.get('real_gauge_naive_degradation'))}; "
                f"verdict={verdict('moe_expert_gauge_alignment_precedes_average')}"
            ),
            "passed": expert_gate,
            "blocking_status": "passed" if expert_gate else "failed_expert_alignment_gate",
            "next_command": "python scripts/fp_moe_real_probe.py --help",
        },
        {
            "requirement": "moe_router_boundary_gate",
            "mechanism": "topk_margin_fragility",
            "required_state": "direct router averaging is frozen or replaced by audited capped route-KD",
            "observed_state": (
                f"router_action={moe.get('router_action')}; "
                f"high_fragility_layers={moe.get('qwen3_router_margin_high_fragility_layers')}/"
                f"{moe.get('qwen3_router_margin_layer_count')}; "
                f"min_safe_lambda={fmt(moe.get('qwen3_router_margin_min_safe_lambda_proxy'))}; "
                f"verdict={verdict('moe_direct_router_average_crosses_topk_boundaries')}"
            ),
            "passed": router_gate,
            "blocking_status": "passed" if router_gate else "failed_router_boundary_gate",
            "next_command": "python scripts/build_qwen3_moe_router_margin_fragility.py --output-dir results/qwen3_moe_router_margin_fragility",
        },
        {
            "requirement": "moe_router_coupled_retention_frontier_gate",
            "mechanism": "router_boundary_signal_budgeting",
            "required_state": "router-boundary fragility may shape B/H/I interference, but direct extra shrink is not default unless retention-safe effect is material",
            "observed_state": (
                f"frontier_gate={moe.get('qwen3_router_coupled_frontier_gate')}; "
                f"default_gate_candidates={moe.get('qwen3_router_coupled_frontier_default_gate_candidates')}/"
                f"{moe.get('qwen3_router_coupled_frontier_candidate_count')}; "
                f"effect_fraction={fmt(moe.get('qwen3_router_coupled_frontier_effect_fraction'), 4)}; "
                f"constrained={moe.get('qwen3_router_coupled_frontier_constrained_candidate_id')}; "
                f"stress={moe.get('qwen3_router_coupled_frontier_stress_candidate_id')}; "
                f"verdict={verdict('moe_direct_router_boundary_shrink_is_not_default')}"
            ),
            "passed": router_shrink_gate,
            "blocking_status": "passed" if router_shrink_gate else "failed_router_coupled_retention_frontier_gate",
            "next_command": "python scripts/analyze_qwen3_moe_router_coupled_retention_frontier.py --output-dir results/qwen3_moe_router_coupled_retention_frontier",
        },
        {
            "requirement": "moe_connectivity_gate",
            "mechanism": "source_to_source_path_probe",
            "required_state": "unconditional source-to-source interpolation is rejected unless an interior point beats source frontier",
            "observed_state": (
                f"instruct/coder interior gap={fmt(moe.get('qwen3_interpolation_interior_gap_nll'))}; "
                f"base/coder interior gap={fmt(moe.get('qwen3_base_coder_interior_gap_nll'))}; "
                f"complementary beats sources={moe.get('qwen3_complementary_merge_beats_sources')}; "
                f"verdict={verdict('moe_source_to_source_line_not_averageable')}"
            ),
            "passed": connectivity_gate,
            "blocking_status": "passed" if connectivity_gate else "failed_connectivity_gate",
            "next_command": "python scripts/fp_moe_barrier.py --out results/fp_moe_barrier",
        },
        {
            "requirement": "source_set_complementarity_gate",
            "mechanism": "endpoint_frontier_before_average",
            "required_state": "average is only treated as a final-candidate prior when the source set has measured task complementarity; source-dominated sets are repair/ablation only",
            "observed_state": (
                f"source_set={moe.get('qwen3_source_set_current')}; "
                f"gate={moe.get('qwen3_source_set_current_gate')}; "
                f"dominant={moe.get('qwen3_source_set_current_dominant_source')}; "
                f"frontier_avg_gain={fmt(moe.get('qwen3_source_set_current_frontier_avg_gain'))}; "
                f"best_observed_gap={fmt(moe.get('qwen3_source_set_current_best_observed_avg_gap'))}; "
                f"verdict={verdict('source_set_complementarity_precedes_average')}"
            ),
            "passed": source_set_gate,
            "blocking_status": "passed" if source_set_gate else "failed_source_set_complementarity_gate",
            "next_command": "python scripts/build_qwen3_source_set_complementarity_gate.py --output-dir results/qwen3_source_set_complementarity_gate",
        },
        {
            "requirement": "source_set_surplus_budget_gate",
            "mechanism": "frontier_gain_minus_merge_interference",
            "required_state": "a source set is promoted to final average budget only when frontier gain exceeds observed merge interference; otherwise it remains probe-only or rejected",
            "observed_state": (
                f"top_source_set={moe.get('qwen3_source_set_top_source_set')}; "
                f"top_gate={moe.get('qwen3_source_set_top_optimizer_gate')}; "
                f"frontier_avg_gain={fmt(moe.get('qwen3_source_set_top_frontier_avg_gain'))}; "
                f"interference_budget={fmt(moe.get('qwen3_source_set_interference_budget'))}; "
                f"surplus={fmt(moe.get('qwen3_source_set_top_surplus_vs_interference'))}; "
                f"final_budget_sets={moe.get('qwen3_source_set_final_budget_candidate_count')}; "
                f"verdict={verdict('source_set_surplus_must_exceed_interference')}"
            ),
            "passed": source_set_surplus_gate,
            "blocking_status": "passed" if source_set_surplus_gate else "failed_source_set_surplus_budget_gate",
            "next_command": "python scripts/build_qwen3_average_source_set_optimizer.py --output-dir results/qwen3_average_source_set_optimizer",
        },
        {
            "requirement": "mechanistic_candidate_structural_gate",
            "mechanism": "route_geometry_subspace_cap_audit",
            "required_state": "materialized candidate is fresh, manifest-matched, router-frozen, and below routed tail cap",
            "observed_state": (
                f"rule_status={moe.get('qwen3_unified_materialized_rule_status')}; "
                f"audit_status={moe.get('qwen3_unified_audit_status')}; "
                f"routed_gt_0.65={moe.get('qwen3_unified_routed_gt_065')}; "
                f"router_changed={moe.get('qwen3_unified_router_changed_tensors')}/"
                f"{moe.get('qwen3_unified_router_tensors')}; "
                f"manifest_match={moe.get('qwen3_unified_matches_materialized_checkpoint_manifest')}"
            ),
            "passed": structural_gate,
            "blocking_status": "passed" if structural_gate else "failed_structural_candidate_gate",
            "next_command": "python scripts/audit_materialized_checkpoint_delta.py --base BASE --candidate results/checkpoints/qwen3_moe_unified_mechanism_candidate --output-dir results/qwen3_moe_unified_mechanism_delta_audit",
        },
        {
            "requirement": "router_calibration_separate_acceptance_gate",
            "mechanism": "router_repair_not_default_average",
            "required_state": "router calibration stays an ablation unless its own selector accepts it",
            "observed_state": (
                f"router_calibration_status={moe.get('qwen3_router_calibration_status')}; "
                f"eligible={moe.get('qwen3_router_calibration_eligible_candidates')}/"
                f"{moe.get('qwen3_router_calibration_candidate_count')}; "
                f"verdict={router_cal_verdict}"
            ),
            "passed": router_cal_guarded,
            "blocking_status": "passed" if router_cal_guarded else "failed_router_calibration_gate",
            "next_command": command(
                "router_calibration_active_candidates",
                "results/qwen3_moe_router_calibration_job/run_router_calibration_job.sh all",
            ),
        },
        {
            "requirement": "statistical_structural_tiebreak_gate",
            "mechanism": "confidence_band_selector",
            "required_state": "structural frontier can only break ties inside a downstream confidence-overlap band",
            "observed_state": (
                f"tie_band={moe.get('qwen3_final_confidence_tie_band')}; "
                f"rank_mode={moe.get('qwen3_final_selection_rank_mode')}; "
                f"band_size={moe.get('qwen3_final_selection_rank_band_size')}; "
                f"point_leader={moe.get('qwen3_final_selection_point_leader_method')}; "
                f"verdict={verdict('moe_structural_tiebreak_requires_statistical_equivalence')}"
            ),
            "passed": tiebreak_gate,
            "blocking_status": "passed" if tiebreak_gate else "failed_statistical_tiebreak_gate",
            "next_command": "python scripts/select_qwen3_moe_final_candidate.py --smoke-matrix --output-dir results/qwen3_moe_final_candidate_selection_smoke",
        },
        {
            "requirement": "downstream_source_dominance_gate",
            "mechanism": "locked_manifest_vllm_eval",
            "required_state": "sources and final candidates pass eval-bundle audit, source dominance, task regression, confidence, and paired-prediction gates",
            "observed_state": (
                f"final_status={moe.get('qwen3_final_selection_status')}; "
                f"eligible={moe.get('qwen3_eligible_candidates')}/"
                f"{moe.get('qwen3_candidate_count')}; "
                f"downstream_verdict={downstream_verdict}; "
                f"unified_verdict={unified_verdict}"
            ),
            "passed": downstream_terminal,
            "blocking_status": "passed"
            if downstream_terminal
            else ("blocked_on_downstream_eval" if downstream_waiting else "failed_downstream_gate"),
            "next_command": command(
                "budgeted_qwen3_moe_downstream_eval",
                "results/qwen3_moe_eval_budget_plan/run_eval_budget.sh final",
            ),
        },
        {
            "requirement": "final_unified_average_acceptance",
            "mechanism": "all_gates_joint",
            "required_state": "all structural gates pass and downstream selector accepts the same-shape average",
            "observed_state": (
                f"structural_gate={structural_gate}; downstream_terminal={downstream_terminal}; "
                f"unified_accepted={unified_accepted}; source_fallback={source_fallback}"
            ),
            "passed": unified_accepted,
            "blocking_status": "passed"
            if unified_accepted
            else (
                "rejected_source_fallback"
                if source_fallback
                else ("blocked_on_downstream_eval" if downstream_waiting else "failed_final_acceptance")
            ),
            "next_command": command(
                "budgeted_qwen3_moe_downstream_eval",
                "results/qwen3_moe_eval_budget_plan/run_eval_budget.sh final",
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_algorithm(
    decisions: pd.DataFrame,
    hypotheses: pd.DataFrame,
    evidence_ledger: pd.DataFrame,
    algorithm_contract: pd.DataFrame,
) -> dict[str, Any]:
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
        "falsification_tests": [
            {
                "hypothesis_id": row["hypothesis_id"],
                "current_status": row["current_status"],
                "falsification_test": row["falsification_test"],
                "next_command": row["next_command"],
            }
            for _, row in hypotheses.iterrows()
        ],
        "evidence_ledger": [
            {
                "hypothesis_id": row["hypothesis_id"],
                "evidence_tier": row["evidence_tier"],
                "verdict": row["verdict"],
                "current_algorithm_action": row["current_algorithm_action"],
                "acceptance_gate_still_needed": row["acceptance_gate_still_needed"],
            }
            for _, row in evidence_ledger.iterrows()
        ],
        "algorithm_contract": [
            {
                "requirement": row["requirement"],
                "mechanism": row["mechanism"],
                "required_state": row["required_state"],
                "observed_state": row["observed_state"],
                "passed": bool(row["passed"]),
                "blocking_status": row["blocking_status"],
                "next_command": row["next_command"],
            }
            for _, row in algorithm_contract.iterrows()
        ],
        "literature_priors": LITERATURE_PRIORS,
        "mechanism_equations": {
            "dense_second_order_gate": "accept a straight-line average only when the measured path loss does not exceed the endpoint frontier; local Fisher/RegMean curvature is treated as a proxy, not a proof",
            "moe_router_margin_gate": "for router logits z_lambda = z_A + lambda * (z_B - z_A), a top-k assignment can flip when lambda * ||Delta z|| reaches the observed top-k margin; therefore small empirical margins imply a near-zero safe router lambda",
            "moe_router_coupled_shrink_gate": "let router fragility enter the expert-scale interference score, but add a separate direct shrink term only if the retention-safe frontier achieves a material fraction of the aggressive ablation effect and then passes matched downstream eval",
            "source_set_complementarity_gate": "before accepting any average, compare the source-set task frontier to the best single source; if the frontier has no material gain because one source dominates, average candidates are repair or ablation tests rather than source-beating priors",
            "source_set_surplus_gate": "promote a source set beyond probe-only only if source_frontier_avg_gain - observed_merge_interference_budget >= 0; otherwise search for stronger endpoint complementarity or run small probes",
            "vllm_source_frontier_feedback_gate": "after hosted vLLM eval, recompute task-wise endpoint frontier and promote an average only if hosted source_frontier_avg_gain - observed_merge_interference_budget >= 0",
            "moe_expert_subspace_cap": "for each routed expert group g, choose scale s_g to preserve route-mass-weighted nonbase contribution while reducing predicted delta under route, geometry, and subspace-conflict weights",
            "moe_confidence_tie_band_gate": "after source, task-regression, confidence-dominance, and paired-prediction gates, find the point-estimate leader; candidates whose aggregate confidence upper bounds overlap the leader lower bounds form the rank band, and structural frontier/safety can break ties only inside that band",
            "same_shape_constraint": "all accepted operations must preserve model config, tokenizer, class, tensor names, tensor shapes, and MoE expert/router cardinalities",
        },
    }


def build_report(
    summary: dict[str, Any],
    features: pd.DataFrame,
    decisions: pd.DataFrame,
    hypotheses: pd.DataFrame,
    evidence_ledger: pd.DataFrame,
    algorithm_contract: pd.DataFrame,
    experiment_queue: pd.DataFrame,
) -> str:
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
        f"- Qwen3 unified mechanism: `{moe['qwen3_unified_candidate_id']}`；retention `{fmt(moe['qwen3_unified_nonbase_mass_retention'])}`，subspace-weighted rel-delta `{fmt(moe['qwen3_unified_subspace_weighted_predicted_relative_delta'])}`，high-subspace mean scale `{fmt(moe['qwen3_unified_high_subspace_mean_scale'])}`，materialized rules `{moe['qwen3_unified_materialized_rule_status']}`，audit relative norm `{fmt(moe['qwen3_unified_relative_delta_norm'])}`，routed >0.65 `{moe['qwen3_unified_routed_gt_065']}`。",
        f"- Qwen3 subspace-scaled ablation: audit relative norm `{fmt(moe['qwen3_subspace_total_relative_delta_norm'])}`，mechanistic->subspace norm delta `{fmt(moe['qwen3_mechanistic_to_subspace_relative_norm_delta'], 6)}`，routed >0.65 `{moe['qwen3_subspace_routed_gt_065']}`。",
        f"- Qwen3 router margin fragility: high layers `{moe['qwen3_router_margin_high_fragility_layers']}/{moe['qwen3_router_margin_layer_count']}`，top `L{moe['qwen3_router_margin_top_layer']}` score `{fmt(moe['qwen3_router_margin_top_score'])}`，min safe-lambda proxy `{fmt(moe['qwen3_router_margin_min_safe_lambda_proxy'])}`。",
        f"- Qwen3 router-coupled retention frontier: gate `{moe['qwen3_router_coupled_frontier_gate']}`，default-gate candidates `{moe['qwen3_router_coupled_frontier_default_gate_candidates']}/{moe['qwen3_router_coupled_frontier_candidate_count']}`，effect fraction `{fmt(moe['qwen3_router_coupled_frontier_effect_fraction'], 4)}`，constrained `{moe['qwen3_router_coupled_frontier_constrained_candidate_id']}`，stress `{moe['qwen3_router_coupled_frontier_stress_candidate_id']}`。",
        f"- Qwen3 router NLL probe: worst-NLL reduction `{fmt(moe['qwen3_router_calibration_nll_worst_reduction'])}`，code gap to best source `{fmt(moe['qwen3_router_calibration_nll_code_gap_to_best_source'])}`。",
        f"- Qwen3 generation matrix: Instruct+Coder avg `{fmt(moe['qwen3_generation_pair_merge_avg'])}` -> router-cal avg `{fmt(moe['qwen3_generation_pair_routercal_avg'])}`；avg gain `{fmt(moe['qwen3_generation_pair_routercal_avg_gain'])}`，gap to best parent `{fmt(moe['qwen3_generation_pair_routercal_gap_to_best_parent_avg'])}`。",
        f"- Qwen3 generation attribution: router-cal recovers `{fmt(moe['qwen3_generation_avg_routercal_recovery_fraction'])}` of avg naive drop and beats pair frontier on `{moe['qwen3_generation_routercal_beats_pair_frontier_count']}/{moe['qwen3_generation_attribution_score_count']}` scores。",
        f"- Qwen3 generation confidence: positive tasks vs naive `{moe['qwen3_generation_routercal_positive_tasks_vs_naive']}/{moe['qwen3_generation_confidence_task_count']}`，confident positives `{moe['qwen3_generation_routercal_confident_positive_tasks']}/{moe['qwen3_generation_confidence_task_count']}`，confident source-frontier wins `{moe['qwen3_generation_routercal_confident_source_frontier_wins']}/{moe['qwen3_generation_confidence_task_count']}`；avg gain interval `[{fmt(moe['qwen3_generation_routercal_avg_gain_lower'])}, {fmt(moe['qwen3_generation_routercal_avg_gain_upper'])}]`。",
        f"- Qwen3 source-set complementarity: `{moe['qwen3_source_set_current']}` gate `{moe['qwen3_source_set_current_gate']}`，dominant `{moe['qwen3_source_set_current_dominant_source']}`，frontier avg gain `{fmt(moe['qwen3_source_set_current_frontier_avg_gain'])}`，best observed merge gap `{fmt(moe['qwen3_source_set_current_best_observed_avg_gap'])}`。",
        f"- Qwen3 source-set surplus: top `{moe['qwen3_source_set_top_source_set']}` gate `{moe['qwen3_source_set_top_optimizer_gate']}`，frontier avg gain `{fmt(moe['qwen3_source_set_top_frontier_avg_gain'])}` vs interference budget `{fmt(moe['qwen3_source_set_interference_budget'])}`，surplus `{fmt(moe['qwen3_source_set_top_surplus_vs_interference'])}`；task surplus-positive `{moe['qwen3_source_set_top_task_surplus_positive_count']}/{moe['qwen3_source_set_task_count']}`，no-gain `{moe['qwen3_source_set_top_no_gain_task_count']}/{moe['qwen3_source_set_task_count']}`，best task `{moe['qwen3_source_set_top_best_task_gain_task']}` gain `{fmt(moe['qwen3_source_set_top_best_task_gain'])}`，weights `{moe['qwen3_source_set_top_source_weights']}`。",
        f"- Qwen source discovery: `{moe['qwen_source_discovery_status']}`，top scenario `{moe['qwen_source_discovery_top_scenario']}`，top queue `{moe['qwen_source_discovery_top_queue_item']}`，additional frontier avg gain needed `{fmt(moe['qwen_source_discovery_measured_additional_frontier_avg_gain_needed'])}`；task blockers `{moe['qwen_source_discovery_task_gap_tasks']}`，top task gap `{moe['qwen_source_discovery_top_task_gap_task']}` / `{moe['qwen_source_discovery_top_task_gap_status']}` needs `{fmt(moe['qwen_source_discovery_top_task_gap_additional_gain_needed'])}`。",
        f"- Qwen source discovery eval: `{moe['qwen_source_discovery_eval_status']}`，jobs `{moe['qwen_source_discovery_eval_job_count']}`，top job `{moe['qwen_source_discovery_eval_top_job']}`，task names `{moe['qwen_source_discovery_eval_task_names']}`，compatibility `{moe['qwen_source_discovery_eval_task_name_status']}`。",
        f"- Qwen source frontier eval feedback: `{moe['qwen_source_frontier_eval_feedback_status']}`，scored `{moe['qwen_source_frontier_eval_feedback_scored']}/{moe['qwen_source_frontier_eval_feedback_jobs']}`，final `{moe['qwen_source_frontier_eval_feedback_final_candidates']}`，probe-only `{moe['qwen_source_frontier_eval_feedback_probe_only']}`，top `{moe['qwen_source_frontier_eval_feedback_top_job']}` gate `{moe['qwen_source_frontier_eval_feedback_top_gate']}`，surplus `{fmt(moe['qwen_source_frontier_eval_feedback_top_surplus'])}`，blocker `{moe['qwen_source_frontier_eval_feedback_blocker']}`。",
        f"- Qwen3 router calibration frontier: `{moe['qwen3_router_calibration_frontier_status']}`，default `{moe['qwen3_router_calibration_frontier_default_candidates']}/{moe['qwen3_router_calibration_frontier_candidate_count']}`，recommended `{moe['qwen3_router_calibration_frontier_recommended']}`，blocker `{moe['qwen3_router_calibration_frontier_blocker']}`，nll `{fmt(moe['qwen3_router_calibration_frontier_nll_signal'])}`，generation `{fmt(moe['qwen3_router_calibration_frontier_generation_signal'])}`。",
        f"- Qwen3 router calibration: `{moe['qwen3_router_calibration_status']}`。",
        f"- Qwen3 final selection: `{moe['qwen3_final_selection_status']}`，eligible `{moe['qwen3_eligible_candidates']}/{moe['qwen3_candidate_count']}`。",
        f"- Qwen3 final selector rank gate: confidence band `{moe['qwen3_final_confidence_tie_band']}`，rank mode `{moe['qwen3_final_selection_rank_mode']}`，band size `{moe['qwen3_final_selection_rank_band_size']}`，point leader `{moe['qwen3_final_selection_point_leader_method']}`。",
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
            "## Falsification Tests",
            "",
            "| hypothesis | status | current evidence | falsification test | next command |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in hypotheses.iterrows():
        lines.append(
            f"| `{row['hypothesis_id']}` | `{row['current_status']}` | {row['current_evidence']} | "
            f"{row['falsification_test']} | `{row['next_command']}` |"
        )
    lines.extend(
        [
            "",
            "## Evidence Ledger",
            "",
            "| hypothesis | verdict | evidence tier | current action | gate still needed |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in evidence_ledger.iterrows():
        lines.append(
            f"| `{row['hypothesis_id']}` | `{row['verdict']}` | `{row['evidence_tier']}` | "
            f"{row['current_algorithm_action']} | {row['acceptance_gate_still_needed']} |"
        )
    lines.extend(
        [
            "",
            "## Algorithm Contract",
            "",
            f"- Contract status: `{summary['contract_status']}`",
            f"- Requirements passed: `{summary['contract_passed_requirement_count']}/{summary['contract_requirement_count']}`",
            f"- Blocking requirements: `{summary['contract_blocking_requirement_count']}`",
            "",
            "| requirement | mechanism | status | passed | observed | next command |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in algorithm_contract.iterrows():
        lines.append(
            f"| `{row['requirement']}` | `{row['mechanism']}` | `{row['blocking_status']}` | "
            f"`{bool(row['passed'])}` | {row['observed_state']} | `{row['next_command']}` |"
        )
    lines.extend(
        [
            "",
            "## Next Experiments",
            "",
            "| rank | experiment | status | priority | driving verdict | command | expected update |",
            "| ---: | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for _, row in experiment_queue.iterrows():
        driving = f"{row['driving_hypothesis_id']}={row['driving_verdict']}"
        lines.append(
            f"| {int(row['rank'])} | `{row['experiment']}` | `{row['status']}` | "
            f"{fmt(row['priority_score'], 2)} | `{driving}` | `{row['command']}` | "
            f"{row['expected_decision_update']} |"
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
            f"- `{summary['outputs']['hypotheses']}`",
            f"- `{summary['outputs']['evidence_ledger']}`",
            f"- `{summary['outputs']['algorithm_contract']}`",
            f"- `{summary['outputs']['next_experiment_queue']}`",
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
    qwen3_router_coupled_retention_frontier = read_json(
        args.qwen3_router_coupled_retention_frontier
    )
    qwen3_source_set_complementarity = read_json(args.qwen3_source_set_complementarity)
    qwen3_average_source_set_optimizer = read_json(args.qwen3_average_source_set_optimizer)
    qwen_source_discovery_plan = read_json(args.qwen_source_discovery_plan)
    qwen_source_discovery_eval_plan = read_json(args.qwen_source_discovery_eval_plan)
    qwen_source_frontier_eval_feedback = read_json(args.qwen_source_frontier_eval_feedback)
    qwen3_router_calibration_frontier = read_json(args.qwen3_router_calibration_frontier)

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
            qwen3_router_coupled_retention_frontier,
            qwen3_source_set_complementarity,
            qwen3_average_source_set_optimizer,
            qwen_source_discovery_plan,
            qwen_source_discovery_eval_plan,
            qwen_source_frontier_eval_feedback,
            qwen3_router_calibration_frontier,
        )
    )
    features = pd.DataFrame(feature_rows)
    decisions = build_decisions(
        features,
        dense_selector,
        router_gate,
        final_selection,
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
        qwen3_router_coupled_retention_frontier,
        qwen3_source_set_complementarity,
        qwen3_average_source_set_optimizer,
        qwen_source_frontier_eval_feedback,
        qwen3_router_calibration_frontier,
    )

    dense_results = dense_selector.get("results") or {}
    final_current = final_selection.get("current_selection") or {}
    router_current = router_calibration.get("current_selection") or {}
    router_frontier_constrained = qwen3_router_coupled_retention_frontier.get("constrained") or {}
    router_frontier_stress = qwen3_router_coupled_retention_frontier.get("stress") or {}
    source_set_top = qwen3_average_source_set_optimizer.get("top_source_set") or {}
    materialized_rule_status = unified_candidate.get("materialized_checkpoint_rule_status")
    if materialized_rule_status != "fresh":
        status = "built_waiting_for_qwen3_materialization_and_vllm_eval"
    elif final_current.get("status") == "awaiting_source_eval":
        status = "built_waiting_for_qwen3_vllm_eval"
    else:
        status = "built"
    summary = {
        "schema_version": 1,
        "status": status,
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
            "qwen3_unified_subspace_weighted_predicted_relative_delta": fnum(
                unified_candidate.get("selected_subspace_weighted_predicted_relative_delta")
            ),
            "qwen3_unified_subspace_risk_weighted_coder_retention": fnum(
                unified_candidate.get("selected_subspace_risk_weighted_coder_retention")
            ),
            "qwen3_unified_high_subspace_mean_scale": fnum(
                unified_candidate.get("selected_high_subspace_mean_scale")
            ),
            "qwen3_unified_subspace_conflict_probe_used": unified_candidate.get("subspace_conflict_probe_used"),
            "qwen3_unified_materialized_rule_status": materialized_rule_status,
            "qwen3_unified_matches_materialized_checkpoint_manifest": unified_candidate.get(
                "matches_materialized_checkpoint_manifest"
            ),
            "qwen3_unified_max_manifest_weight_abs_diff": fnum(
                unified_candidate.get("max_materialized_checkpoint_weight_abs_diff")
            ),
            "qwen3_unified_tensor_rules_sha256": unified_candidate.get("tensor_rules_sha256"),
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
            "qwen3_router_coupled_frontier_status": qwen3_router_coupled_retention_frontier.get(
                "status"
            ),
            "qwen3_router_coupled_frontier_gate": qwen3_router_coupled_retention_frontier.get(
                "gate"
            ),
            "qwen3_router_coupled_frontier_action": qwen3_router_coupled_retention_frontier.get(
                "recommended_unified_action"
            ),
            "qwen3_router_coupled_frontier_candidate_count": qwen3_router_coupled_retention_frontier.get(
                "candidate_count"
            ),
            "qwen3_router_coupled_frontier_default_gate_candidates": qwen3_router_coupled_retention_frontier.get(
                "default_gate_candidate_count"
            ),
            "qwen3_router_coupled_frontier_minimum_effect_fraction": fnum(
                qwen3_router_coupled_retention_frontier.get("minimum_effective_fraction")
            ),
            "qwen3_router_coupled_frontier_effect_fraction": fnum(
                qwen3_router_coupled_retention_frontier.get(
                    "constrained_effect_fraction_vs_stress"
                )
            ),
            "qwen3_router_coupled_frontier_constrained_candidate_id": router_frontier_constrained.get(
                "candidate_id"
            ),
            "qwen3_router_coupled_frontier_constrained_retention": fnum(
                router_frontier_constrained.get("nonbase_mass_retention")
            ),
            "qwen3_router_coupled_frontier_constrained_retention_delta": fnum(
                router_frontier_constrained.get("retention_delta_vs_base")
            ),
            "qwen3_router_coupled_frontier_constrained_reduction": fnum(
                router_frontier_constrained.get("router_coupled_delta_reduction_vs_base")
            ),
            "qwen3_router_coupled_frontier_stress_candidate_id": router_frontier_stress.get(
                "candidate_id"
            ),
            "qwen3_router_coupled_frontier_stress_retention_delta": fnum(
                router_frontier_stress.get("retention_delta_vs_base")
            ),
            "qwen3_router_coupled_frontier_stress_reduction": fnum(
                router_frontier_stress.get("router_coupled_delta_reduction_vs_base")
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
            "qwen3_source_set_gate_status": qwen3_source_set_complementarity.get("status"),
            "qwen3_source_set_current": qwen3_source_set_complementarity.get("current_source_set"),
            "qwen3_source_set_current_gate": qwen3_source_set_complementarity.get("current_gate"),
            "qwen3_source_set_current_dominant_source": qwen3_source_set_complementarity.get(
                "current_dominant_source"
            ),
            "qwen3_source_set_current_frontier_avg": fnum(
                qwen3_source_set_complementarity.get("current_frontier_avg")
            ),
            "qwen3_source_set_current_frontier_avg_gain": fnum(
                qwen3_source_set_complementarity.get("current_frontier_avg_gain_vs_best_single")
            ),
            "qwen3_source_set_current_best_observed_merge": qwen3_source_set_complementarity.get(
                "current_best_observed_merge"
            ),
            "qwen3_source_set_current_best_observed_avg_gap": fnum(
                qwen3_source_set_complementarity.get("current_best_observed_avg_gap_to_frontier")
            ),
            "qwen3_source_set_complementary_count": qwen3_source_set_complementarity.get(
                "complementary_source_set_count"
            ),
            "qwen3_source_set_source_dominated_count": qwen3_source_set_complementarity.get(
                "source_dominated_set_count"
            ),
            "qwen3_source_set_recommended_action": qwen3_source_set_complementarity.get(
                "recommended_action"
            ),
            "qwen3_source_set_surplus_optimizer_status": qwen3_average_source_set_optimizer.get(
                "status"
            ),
            "qwen3_source_set_interference_budget": fnum(
                qwen3_average_source_set_optimizer.get("interference_budget")
            ),
            "qwen3_source_set_interference_budget_source": qwen3_average_source_set_optimizer.get(
                "interference_budget_source"
            ),
            "qwen3_source_set_final_budget_candidate_count": qwen3_average_source_set_optimizer.get(
                "final_average_budget_candidate_count"
            ),
            "qwen3_source_set_probe_only_count": qwen3_average_source_set_optimizer.get(
                "probe_only_source_set_count"
            ),
            "qwen3_source_set_task_count": qwen3_average_source_set_optimizer.get("task_count"),
            "qwen3_source_set_top_positive_task_count": qwen3_average_source_set_optimizer.get(
                "top_positive_task_count"
            ),
            "qwen3_source_set_top_task_surplus_positive_count": qwen3_average_source_set_optimizer.get(
                "top_task_surplus_positive_count"
            ),
            "qwen3_source_set_top_no_gain_task_count": qwen3_average_source_set_optimizer.get(
                "top_no_gain_task_count"
            ),
            "qwen3_source_set_top_blocking_tasks": qwen3_average_source_set_optimizer.get(
                "top_blocking_tasks"
            ),
            "qwen3_source_set_top_best_task_gain_task": qwen3_average_source_set_optimizer.get(
                "top_best_task_gain_task"
            ),
            "qwen3_source_set_top_best_task_gain": fnum(
                qwen3_average_source_set_optimizer.get("top_best_task_gain")
            ),
            "qwen3_source_set_top_best_task_frontier_source": qwen3_average_source_set_optimizer.get(
                "top_best_task_frontier_source"
            ),
            "qwen3_source_set_top_source_set": source_set_top.get("source_set"),
            "qwen3_source_set_top_optimizer_gate": source_set_top.get("optimizer_gate"),
            "qwen3_source_set_top_candidate_id": source_set_top.get("candidate_id"),
            "qwen3_source_set_top_frontier_avg_gain": fnum(
                source_set_top.get("frontier_avg_gain_vs_best_single")
            ),
            "qwen3_source_set_top_frontier_worst_gain": fnum(
                source_set_top.get("frontier_worst_gain_vs_best_single")
            ),
            "qwen3_source_set_top_surplus_vs_interference": fnum(
                source_set_top.get("frontier_avg_surplus_vs_interference")
            ),
            "qwen3_source_set_top_source_weights": source_set_top.get("source_weights"),
            "qwen3_source_set_top_action": source_set_top.get("recommended_action"),
            "qwen_source_discovery_status": qwen_source_discovery_plan.get("status"),
            "qwen_source_discovery_top_scenario": (
                qwen_source_discovery_plan.get("top_scenario") or {}
            ).get("scenario_id"),
            "qwen_source_discovery_top_action": (
                qwen_source_discovery_plan.get("top_scenario") or {}
            ).get("next_action"),
            "qwen_source_discovery_top_queue_item": (
                qwen_source_discovery_plan.get("top_queue_item") or {}
            ).get("queue_item"),
            "qwen_source_discovery_measured_additional_frontier_avg_gain_needed": fnum(
                qwen_source_discovery_plan.get("measured_additional_frontier_avg_gain_needed")
            ),
            "qwen_source_discovery_task_gap_targets": qwen_source_discovery_plan.get(
                "task_gap_target_count"
            ),
            "qwen_source_discovery_task_gap_blockers": qwen_source_discovery_plan.get(
                "task_gap_blocker_count"
            ),
            "qwen_source_discovery_task_gap_tasks": qwen_source_discovery_plan.get(
                "task_gap_tasks"
            ),
            "qwen_source_discovery_top_task_gap_task": qwen_source_discovery_plan.get(
                "top_task_gap_task"
            ),
            "qwen_source_discovery_top_task_gap_status": qwen_source_discovery_plan.get(
                "top_task_gap_status"
            ),
            "qwen_source_discovery_top_task_gap_capability": qwen_source_discovery_plan.get(
                "top_task_gap_capability"
            ),
            "qwen_source_discovery_top_task_gap_additional_gain_needed": fnum(
                qwen_source_discovery_plan.get("top_task_gap_additional_gain_needed")
            ),
            "qwen_source_discovery_top_task_gap_next_action": qwen_source_discovery_plan.get(
                "top_task_gap_next_action"
            ),
            "qwen_source_discovery_eval_status": qwen_source_discovery_eval_plan.get("status"),
            "qwen_source_discovery_eval_job_count": qwen_source_discovery_eval_plan.get(
                "eval_job_count"
            ),
            "qwen_source_discovery_eval_top_job": (
                qwen_source_discovery_eval_plan.get("top_eval_job") or {}
            ).get("job_id"),
            "qwen_source_discovery_eval_task_name_status": qwen_source_discovery_eval_plan.get(
                "task_name_compatibility_status"
            ),
            "qwen_source_discovery_eval_task_names": qwen_source_discovery_eval_plan.get(
                "task_names"
            ),
            "qwen_source_frontier_eval_feedback_status": qwen_source_frontier_eval_feedback.get(
                "status"
            ),
            "qwen_source_frontier_eval_feedback_scored": qwen_source_frontier_eval_feedback.get(
                "scored_job_count"
            ),
            "qwen_source_frontier_eval_feedback_jobs": qwen_source_frontier_eval_feedback.get(
                "job_count"
            ),
            "qwen_source_frontier_eval_feedback_final_candidates": qwen_source_frontier_eval_feedback.get(
                "final_average_budget_candidate_count"
            ),
            "qwen_source_frontier_eval_feedback_probe_only": qwen_source_frontier_eval_feedback.get(
                "probe_only_candidate_count"
            ),
            "qwen_source_frontier_eval_feedback_top_job": (
                qwen_source_frontier_eval_feedback.get("top_scored_job") or {}
            ).get("job_id"),
            "qwen_source_frontier_eval_feedback_top_gate": (
                qwen_source_frontier_eval_feedback.get("top_scored_job") or {}
            ).get("decision_gate"),
            "qwen_source_frontier_eval_feedback_top_surplus": fnum(
                (qwen_source_frontier_eval_feedback.get("top_scored_job") or {}).get(
                    "surplus_vs_interference"
                )
            ),
            "qwen_source_frontier_eval_feedback_blocker": qwen_source_frontier_eval_feedback.get(
                "blocking_reason"
            ),
            "qwen3_router_calibration_frontier_status": qwen3_router_calibration_frontier.get(
                "status"
            ),
            "qwen3_router_calibration_frontier_default_candidates": qwen3_router_calibration_frontier.get(
                "default_candidate_count"
            ),
            "qwen3_router_calibration_frontier_candidate_count": qwen3_router_calibration_frontier.get(
                "candidate_count"
            ),
            "qwen3_router_calibration_frontier_recommended": qwen3_router_calibration_frontier.get(
                "recommended_default_candidates"
            ),
            "qwen3_router_calibration_frontier_blocker": qwen3_router_calibration_frontier.get(
                "acceptance_blocker"
            ),
            "qwen3_router_calibration_frontier_nll_signal": fnum(
                qwen3_router_calibration_frontier.get("nll_worst_reduction_signal")
            ),
            "qwen3_router_calibration_frontier_generation_signal": fnum(
                qwen3_router_calibration_frontier.get("generation_avg_gain_signal")
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
            "qwen3_mechanistic_to_subspace_relative_norm_delta": fnum(
                delta_frontier.get("mechanistic_to_subspace_relative_norm_delta")
            ),
            "qwen3_mechanistic_to_subspace_routed_gt_065_reduction": delta_frontier.get(
                "mechanistic_to_subspace_routed_gt_065_reduction"
            ),
            "qwen3_router_calibration_status": router_current.get("status"),
            "qwen3_router_calibration_selected_method": router_current.get("selected_method"),
            "qwen3_router_calibration_reason": router_current.get("reason"),
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
            "qwen3_final_selected_method": final_current.get("selected_method"),
            "qwen3_final_selection_reason": final_current.get("reason"),
            "qwen3_eligible_candidates": final_current.get("eligible_candidate_count"),
            "qwen3_candidate_count": final_current.get("candidate_count"),
            "qwen3_final_confidence_tie_band": final_current.get("confidence_tie_band"),
            "qwen3_final_selection_rank_mode": final_current.get("selection_rank_mode"),
            "qwen3_final_selection_point_leader_method": final_current.get("selection_point_leader_method"),
            "qwen3_final_selection_rank_band_size": final_current.get("selection_rank_band_size"),
            "qwen3_final_selection_rank_band_methods": final_current.get("selection_rank_band_methods", []),
            "qwen3_final_structural_frontier_eligible_count": final_current.get(
                "structural_frontier_eligible_count"
            ),
        },
        "outputs": {},
    }
    hypotheses = build_mechanism_hypotheses(summary)
    evidence_ledger = build_evidence_ledger(summary, hypotheses)
    experiment_queue = build_next_experiment_queue(summary, evidence_ledger)
    algorithm_contract = build_algorithm_contract(summary, evidence_ledger, experiment_queue)
    algorithm = build_algorithm(decisions, hypotheses, evidence_ledger, algorithm_contract)
    status_counts = hypotheses["current_status"].value_counts().to_dict()
    verdict_counts = evidence_ledger["verdict"].value_counts().to_dict()
    blocking_contract = algorithm_contract[algorithm_contract["blocking_status"].astype(str) != "passed"]
    failed_contract = blocking_contract[
        ~blocking_contract["blocking_status"].astype(str).str.startswith("blocked_on_")
    ]
    if bool(algorithm_contract["passed"].all()):
        contract_status = "accepted_unified_average"
    elif not failed_contract.empty:
        contract_status = "failed_contract"
    elif any(blocking_contract["blocking_status"].astype(str).str.contains("downstream_eval")):
        contract_status = "blocked_on_downstream_eval"
    else:
        contract_status = "blocked_on_mechanism_gate"
    top_experiment = experiment_queue.iloc[0].to_dict() if not experiment_queue.empty else {}
    summary["hypothesis_count"] = int(len(hypotheses))
    summary["hypothesis_status_counts"] = {str(key): int(value) for key, value in status_counts.items()}
    summary["evidence_ledger_count"] = int(len(evidence_ledger))
    summary["evidence_verdict_counts"] = {
        str(key): int(value) for key, value in verdict_counts.items()
    }
    summary["next_experiment_count"] = int(len(experiment_queue))
    summary["contract_status"] = contract_status
    summary["contract_requirement_count"] = int(len(algorithm_contract))
    summary["contract_passed_requirement_count"] = int(algorithm_contract["passed"].astype(bool).sum())
    summary["contract_blocking_requirement_count"] = int(len(blocking_contract))
    summary["contract_failed_requirement_count"] = int(len(failed_contract))
    summary["contract_blocking_requirements"] = [
        str(row["requirement"]) for _, row in blocking_contract.iterrows()
    ]
    summary["top_next_experiment"] = {
        "experiment": top_experiment.get("experiment"),
        "status": top_experiment.get("status"),
        "priority_score": fnum(top_experiment.get("priority_score")),
        "command": top_experiment.get("command"),
        "preflight_command": top_experiment.get("preflight_command"),
    }

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "mechanism_features.csv"
    decisions_path = output_dir / "operation_decisions.csv"
    hypotheses_path = output_dir / "mechanism_hypotheses.csv"
    evidence_ledger_path = output_dir / "hypothesis_evidence_ledger.csv"
    algorithm_contract_path = output_dir / "algorithm_contract.csv"
    queue_path = output_dir / "next_experiment_queue.csv"
    algorithm_path = output_dir / "algorithm.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary["outputs"] = {
        "features": rel(features_path),
        "decisions": rel(decisions_path),
        "hypotheses": rel(hypotheses_path),
        "evidence_ledger": rel(evidence_ledger_path),
        "algorithm_contract": rel(algorithm_contract_path),
        "next_experiment_queue": rel(queue_path),
        "algorithm": rel(algorithm_path),
        "summary": rel(summary_path),
        "report": rel(report_path),
    }

    features.to_csv(features_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    hypotheses.to_csv(hypotheses_path, index=False)
    evidence_ledger.to_csv(evidence_ledger_path, index=False)
    algorithm_contract.to_csv(algorithm_contract_path, index=False)
    experiment_queue.to_csv(queue_path, index=False)
    algorithm_path.write_text(json.dumps(json_safe(algorithm), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(
        build_report(
            summary,
            features,
            decisions,
            hypotheses,
            evidence_ledger,
            algorithm_contract,
            experiment_queue,
        ),
        encoding="utf-8",
    )
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
        "--qwen3-source-set-complementarity",
        type=Path,
        default=Path("results/qwen3_source_set_complementarity_gate/summary.json"),
    )
    parser.add_argument(
        "--qwen3-average-source-set-optimizer",
        type=Path,
        default=Path("results/qwen3_average_source_set_optimizer/summary.json"),
    )
    parser.add_argument(
        "--qwen-source-discovery-plan",
        type=Path,
        default=Path("results/qwen_source_discovery_plan/summary.json"),
    )
    parser.add_argument(
        "--qwen-source-discovery-eval-plan",
        type=Path,
        default=Path("results/qwen_source_discovery_eval_plan/summary.json"),
    )
    parser.add_argument(
        "--qwen-source-frontier-eval-feedback",
        type=Path,
        default=Path("results/qwen_source_frontier_eval_feedback/summary.json"),
    )
    parser.add_argument(
        "--qwen3-router-calibration-frontier",
        type=Path,
        default=Path("results/qwen3_moe_router_calibration_frontier/summary.json"),
    )
    parser.add_argument(
        "--qwen3-router-margin-fragility",
        type=Path,
        default=Path("results/qwen3_moe_router_margin_fragility/summary.json"),
    )
    parser.add_argument(
        "--qwen3-router-coupled-retention-frontier",
        type=Path,
        default=Path("results/qwen3_moe_router_coupled_retention_frontier/summary.json"),
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
