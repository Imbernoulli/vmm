#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


LITERATURE_SOURCES: list[dict[str, str]] = [
    {
        "key": "fisher_merging",
        "title": "Merging Models with Fisher-Weighted Averaging",
        "url": "https://arxiv.org/abs/2111.09832",
        "mechanism": "Local posterior curvature can weight averages, but it is only a local argument.",
    },
    {
        "key": "model_soups",
        "title": "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time",
        "url": "https://arxiv.org/abs/2203.05482",
        "mechanism": "Weight averaging is plausible inside one low-error basin; it is not a guarantee for divergent experts.",
    },
    {
        "key": "git_rebasin",
        "title": "Git Re-Basin: Merging Models modulo Permutation Symmetries",
        "url": "https://arxiv.org/abs/2209.04836",
        "mechanism": "Permutation or expert identity must be resolved before same-name parameters are averaged.",
    },
    {
        "key": "ties",
        "title": "TIES-Merging: Resolving Interference When Merging Models",
        "url": "https://arxiv.org/abs/2306.01708",
        "mechanism": "Small redundant deltas and sign disagreement are coordinate-level interference signals.",
    },
    {
        "key": "dare",
        "title": "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch",
        "url": "https://arxiv.org/abs/2311.03099",
        "mechanism": "Drop-and-rescale is justified only when delta probes show strong redundancy.",
    },
    {
        "key": "della",
        "title": "DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based Sampling",
        "url": "https://arxiv.org/abs/2406.11617",
        "mechanism": "Magnitude-aware delta retention is safer than blind random dropping.",
    },
    {
        "key": "wemoe",
        "title": "Efficient and Effective Weight-Ensembling Mixture of Experts for Multi-Task Model Merging",
        "url": "https://arxiv.org/abs/2410.21804",
        "mechanism": "Dynamic expert selection can reduce interference, but it changes the execution structure.",
    },
    {
        "key": "mergeme",
        "title": "MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs",
        "url": "https://arxiv.org/abs/2502.00997",
        "mechanism": "MoE merging must handle parameter interference and routing heuristics explicitly.",
    },
    {
        "key": "sub_moe",
        "title": "Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging",
        "url": "https://arxiv.org/abs/2506.23266",
        "mechanism": "Expert output similarity and subspace structure are better evidence than raw weight names.",
    },
    {
        "key": "expert_merging",
        "title": "Expert Merging: Model Merging with Unsupervised Expert Alignment and Importance-Guided Layer Chunking",
        "url": "https://arxiv.org/abs/2509.25712",
        "mechanism": "Layer heterogeneity and unlabeled hidden/logit alignment motivate layer/chunk coefficients.",
    },
    {
        "key": "harc",
        "title": "When Model Merging Breaks Routing: Training-Free Calibration for MoE",
        "url": "https://arxiv.org/abs/2606.03391",
        "mechanism": "Top-k router perturbations can break dispatch, so router movement needs its own calibration gate.",
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
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def maybe_float(value: Any) -> float | None:
    value = clean_value(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def maybe_int(value: Any) -> int | None:
    value = clean_value(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 4) -> str:
    value = maybe_float(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def clean_row(row: pd.Series) -> dict[str, Any]:
    return {str(key): clean_value(value) for key, value in row.items()}


def collect_context(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "connectivity": read_json(args.connectivity_summary),
        "connectivity_rows": read_csv(args.connectivity_rows),
        "preflight": read_json(args.preflight_summary),
        "router_move": read_json(args.router_move_summary),
        "expert_geometry": read_json(args.expert_geometry_summary),
        "unified_candidate": read_json(args.unified_candidate_summary),
        "unified_delta_audit": read_json(args.unified_delta_audit_summary),
        "router_calibration_nll": read_json(args.router_calibration_nll_summary),
        "final_selection": read_json(args.final_selection_summary),
        "eval_bundle_audit": read_json(args.eval_bundle_audit_summary),
        "method_gate": read_json(args.method_gate_summary),
    }


def build_invariants(ctx: dict[str, Any]) -> pd.DataFrame:
    connectivity = ctx["connectivity"]
    preflight = ctx["preflight"]
    identity = preflight.get("expert_identity_gate") or {}
    router = ctx["router_move"]
    geometry = ctx["expert_geometry"]
    unified = ctx["unified_candidate"]
    delta_audit = ctx["unified_delta_audit"]
    router_cal = ctx["router_calibration_nll"]
    final_selection = ctx["final_selection"].get("current_selection") or {}
    eval_audit = ctx["eval_bundle_audit"]

    rows: list[dict[str, Any]] = [
        {
            "invariant_id": "same_shape_contract",
            "domain": "dense+moe",
            "hard_gate": True,
            "status": "pass" if preflight.get("same_shape_contract_pass") else "reject",
            "current_value": f"same_shape={preflight.get('same_shape_contract_pass')}",
            "requirement": "The output checkpoint must keep the input model architecture, tensor names, and config contract.",
            "failure_mode": "A method can score well by changing topology, but it is not the requested average target.",
            "algorithm_action": "Reject topology-expanding methods as final outputs; use them only as upper-bound probes.",
            "primary_evidence": rel("results/moe_unified_preflight_qwen3_30b/summary.json"),
            "literature_keys": "wemoe,sub_moe",
            "proof_role": "Defines the feasible set.",
        },
        {
            "invariant_id": "endpoint_frontier",
            "domain": "dense+moe",
            "hard_gate": True,
            "status": "reject_default_average"
            if int(connectivity.get("path_rejected_count", 0)) > 0
            else "pass",
            "current_value": (
                f"path_rejected={connectivity.get('path_rejected_count')}/"
                f"{connectivity.get('case_count')}; frontier_wins={connectivity.get('endpoint_frontier_win_count')}"
            ),
            "requirement": "A candidate average must beat the best source endpoint frontier on the measured objective.",
            "failure_mode": "A smooth-looking line or named method can still be dominated by one source model.",
            "algorithm_action": "Always include source endpoints and allow endpoint/anchor fallback.",
            "primary_evidence": rel("results/average_connectivity_diagnostic/summary.json"),
            "literature_keys": "model_soups,git_rebasin",
            "proof_role": "Prevents accepting a dominated merge.",
        },
        {
            "invariant_id": "fixed_midpoint_safety",
            "domain": "dense+moe",
            "hard_gate": True,
            "status": "reject_default_average"
            if int(connectivity.get("midpoint_rejected_count", 0)) > 0
            else "pass",
            "current_value": (
                f"midpoint_rejected={connectivity.get('midpoint_rejected_count')}/"
                f"{connectivity.get('case_count')}; dense_midpoint_gap={fmt(connectivity.get('dense_source_midpoint_gap'))}; "
                f"qwen3_midpoint_gap={fmt(connectivity.get('qwen3_instruct_coder_midpoint_gap'))}"
            ),
            "requirement": "The 0.5/0.5 point is a separate candidate and must pass its own frontier check.",
            "failure_mode": "Coefficient search may find an anchor fallback while the midpoint remains bad.",
            "algorithm_action": "Keep uniform average as a negative baseline unless midpoint gap is within tolerance.",
            "primary_evidence": rel("results/average_connectivity_diagnostic/path_diagnostics.csv"),
            "literature_keys": "model_soups,task_arithmetic",
            "proof_role": "Separates averaging from coefficient/anchor selection.",
        },
        {
            "invariant_id": "local_quadratic_validity",
            "domain": "dense",
            "hard_gate": False,
            "status": "reject_as_sufficient_gate",
            "current_value": "Fisher actual/predicted degradation ratios are far above a small local-error regime.",
            "requirement": "Fisher or RegMean-style local approximations must be calibrated against actual path loss.",
            "failure_mode": "A local second-order model underestimates nonlocal midpoint degradation.",
            "algorithm_action": "Use local curvature as a feature, not as a proof of merge safety.",
            "primary_evidence": rel("results/fp_curvature_law/report.md"),
            "literature_keys": "fisher_merging",
            "proof_role": "Blocks curvature-only acceptance.",
        },
        {
            "invariant_id": "expert_identity_alignment",
            "domain": "moe",
            "hard_gate": True,
            "status": "pass" if identity.get("pass") else "reject",
            "current_value": (
                f"identity_layers={identity.get('n_layers')}/{identity.get('expected_layers')}; "
                f"identity_fraction={fmt(identity.get('frac_layers_identity_optimal'))}"
            ),
            "requirement": "Routed experts can only be same-name averaged after identity/alignment is established.",
            "failure_mode": "Function-preserving expert permutations can make same-name averaging arbitrarily wrong.",
            "algorithm_action": "Use identity slices for Qwen3; require remap/alignment for other MoEs.",
            "primary_evidence": rel("results/moe_unified_preflight_qwen3_30b/summary.json"),
            "literature_keys": "git_rebasin,mergeme",
            "proof_role": "Makes expert tensor names meaningful.",
        },
        {
            "invariant_id": "router_stability",
            "domain": "moe",
            "hard_gate": True,
            "status": "reject_router_movement"
            if int(router.get("allowed_router_layer_count", 0)) == 0
            else "conditional",
            "current_value": (
                f"allowed_router_layers={router.get('allowed_router_layer_count')}/"
                f"{router.get('router_layer_count')}; min_top1={fmt(router.get('min_top1_agreement'))}; "
                f"min_topk_jaccard={fmt(router.get('min_topk_jaccard'))}"
            ),
            "requirement": "Direct router weight movement must preserve top-k dispatch over observed route categories.",
            "failure_mode": "Small router perturbations change discrete expert assignment and amplify expert mismatch.",
            "algorithm_action": "Freeze router by default; test router-KD/HARC-style calibration as a separate intervention.",
            "primary_evidence": rel("results/qwen3_moe_router_move_gate/summary.json"),
            "literature_keys": "harc",
            "proof_role": "Prevents routing breakdown.",
        },
        {
            "invariant_id": "expert_internal_geometry",
            "domain": "moe",
            "hard_gate": False,
            "status": "conditional_geometry_shrink",
            "current_value": (
                f"high_internal={geometry.get('high_internal_geometry_risk_expert_count')}; "
                f"high_route_geometry={geometry.get('high_route_geometry_risk_expert_count')}; "
                f"mean_cos={fmt(geometry.get('mean_expert_combined_cosine'))}"
            ),
            "requirement": "A single global expert coefficient is unsafe when expert geometry varies by layer and route mass.",
            "failure_mode": "High route+geometry-risk experts dominate regressions under uniform source weights.",
            "algorithm_action": "Use route/geometry-weighted caps and layer/chunk coefficients.",
            "primary_evidence": rel("results/qwen3_moe_expert_geometry_probe/summary.json"),
            "literature_keys": "expert_merging,sub_moe",
            "proof_role": "Motivates nonuniform expert coefficients.",
        },
        {
            "invariant_id": "routed_delta_trust_region",
            "domain": "moe",
            "hard_gate": True,
            "status": "pass"
            if int(unified.get("selected_routed_gt_hard_cap_groups", 1)) == 0
            else "reject",
            "current_value": (
                f"selected_max_predicted_delta={fmt(unified.get('selected_max_predicted_relative_delta'))}; "
                f"groups_gt_hard_cap={unified.get('selected_routed_gt_hard_cap_groups')}; "
                f"delta_audit_status={delta_audit.get('status')}"
            ),
            "requirement": "Routed expert deltas must satisfy a hard trust-region cap before vLLM selection.",
            "failure_mode": "Large routed expert movement can look useful structurally while causing task regressions.",
            "algorithm_action": "Only materialize candidates that pass cap and same-shape delta audit.",
            "primary_evidence": rel("results/qwen3_moe_unified_mechanism_candidate/summary.json"),
            "literature_keys": "ties,della,expert_merging",
            "proof_role": "Bounds structural intervention size.",
        },
        {
            "invariant_id": "router_calibration_not_acceptance",
            "domain": "moe",
            "hard_gate": False,
            "status": "mechanism_supported_not_sufficient",
            "current_value": (
                f"worst_nll_reduction={fmt(router_cal.get('worst_nll_reduction_vs_linear'))}; "
                f"worst_gap_to_best_source={fmt(router_cal.get('routercal_worst_gap_to_best_source'))}"
            ),
            "requirement": "Router-only improvement supports the mechanism but cannot accept the merge without endpoint dominance.",
            "failure_mode": "A calibrated average can improve over linear average while still losing to a source.",
            "algorithm_action": "Queue router-calibrated candidates for matched vLLM evaluation and source-dominance selection.",
            "primary_evidence": rel("results/qwen3_moe_router_calibration_nll_probe/summary.json"),
            "literature_keys": "harc",
            "proof_role": "Distinguishes mechanism evidence from final acceptance.",
        },
        {
            "invariant_id": "matched_downstream_dominance",
            "domain": "dense+moe",
            "hard_gate": True,
            "status": "awaiting_eval"
            if final_selection.get("status") == "awaiting_source_eval"
            else str(final_selection.get("status")),
            "current_value": (
                f"usable_candidates={final_selection.get('usable_candidate_count')}/"
                f"{final_selection.get('candidate_count')}; "
                f"usable_eval_bundles={eval_audit.get('usable_for_selection_count')}/"
                f"{eval_audit.get('method_count')}"
            ),
            "requirement": "Final acceptance requires matched vLLM eval, no source dominance, no task regression, and paired prediction gates.",
            "failure_mode": "Structural probes are not downstream task performance.",
            "algorithm_action": "Do not claim a unified average wins until source and candidate eval bundles pass audit.",
            "primary_evidence": rel("results/qwen3_moe_final_candidate_selection/summary.json"),
            "literature_keys": "model_soups,expert_merging,harc",
            "proof_role": "Provides the final measured-regret gate.",
        },
    ]
    return pd.DataFrame(rows)


def build_method_matrix(invariants: pd.DataFrame, ctx: dict[str, Any]) -> pd.DataFrame:
    status_by_id = {str(row["invariant_id"]): str(row["status"]) for _, row in invariants.iterrows()}
    method_gate = ctx["method_gate"]
    accepted = int(method_gate.get("accepted_by_default_count", 0) or 0)
    return pd.DataFrame(
        [
            {
                "method_family": "Uniform / linear average",
                "current_gate": "reject_as_default",
                "violated_or_required_invariants": "endpoint_frontier;fixed_midpoint_safety",
                "why": "Current Dense and Qwen3 MoE midpoint/path evidence is dominated by source endpoints.",
                "use_if": "Only if the midpoint itself passes endpoint-frontier and downstream gates.",
                "same_shape_output": True,
            },
            {
                "method_family": "Model soups / greedy soup",
                "current_gate": "conditional",
                "violated_or_required_invariants": "endpoint_frontier;matched_downstream_dominance",
                "why": "Soup logic is valid inside one low-error basin, but current source paths are not proven to be in that basin.",
                "use_if": "Greedy validation includes endpoints and rejects any candidate that hurts the frontier.",
                "same_shape_output": True,
            },
            {
                "method_family": "Task arithmetic / coefficient search",
                "current_gate": "conditional_endpoint_fallback",
                "violated_or_required_invariants": "endpoint_frontier;fixed_midpoint_safety",
                "why": "Coefficient search may find an anchor/base fallback while the raw average remains unsafe.",
                "use_if": "Search over layer/module/expert coefficients and permit endpoint or anchor selection.",
                "same_shape_output": True,
            },
            {
                "method_family": "TIES / DARE / DELLA / STAR",
                "current_gate": "conditional_no_router_blindness",
                "violated_or_required_invariants": "router_stability;expert_internal_geometry;routed_delta_trust_region",
                "why": "Coordinate conflict methods do not solve MoE dispatch, expert identity, or route-load risk by themselves.",
                "use_if": "Apply after expert identity/remap, with router frozen/calibrated and routed delta caps.",
                "same_shape_output": True,
            },
            {
                "method_family": "Fisher / RegMean / RegMean++",
                "current_gate": "conditional_not_sufficient",
                "violated_or_required_invariants": "local_quadratic_validity;matched_downstream_dominance",
                "why": "Local or layerwise regression arguments can underestimate nonlocal merge barriers.",
                "use_if": "Use as a feature or tensor rule, then validate on held-out NLL and downstream eval.",
                "same_shape_output": True,
            },
            {
                "method_family": "WEMoE / dynamic MoE upscaling",
                "current_gate": "disallowed_as_final_output",
                "violated_or_required_invariants": "same_shape_contract",
                "why": "Dynamic expert modules can reduce interference but change the model structure requested here.",
                "use_if": "Use as an upper bound or teacher, then distill/compress back into same-shape rules.",
                "same_shape_output": False,
            },
            {
                "method_family": "Expert Merging++ / layer chunking",
                "current_gate": "active_structural_candidate",
                "violated_or_required_invariants": "expert_internal_geometry;routed_delta_trust_region;matched_downstream_dominance",
                "why": "Current Qwen3 geometry shows layer/expert heterogeneity; layer/chunk coefficients directly target it.",
                "use_if": "Keep same-shape tensor rules and require final vLLM source-frontier dominance.",
                "same_shape_output": True,
            },
            {
                "method_family": "Sub-MoE / expert-output clustering",
                "current_gate": "probe_only_under_same_shape_contract",
                "violated_or_required_invariants": "same_shape_contract;expert_identity_alignment",
                "why": "Expert compression can change expert count, but output similarity is useful as a probe.",
                "use_if": "Use output/subspace similarity to decide remaps or caps without changing tensor topology.",
                "same_shape_output": False,
            },
            {
                "method_family": "HARC / router-only calibration",
                "current_gate": "active_separate_intervention",
                "violated_or_required_invariants": "router_stability;matched_downstream_dominance",
                "why": "Router-only NLL probe improves linear MoE but still does not dominate the best source.",
                "use_if": "Train/calibrate router deltas under caps, audit them, then evaluate with source endpoints.",
                "same_shape_output": True,
            },
            {
                "method_family": "Unified same-shape mechanism optimizer",
                "current_gate": "structurally_ready_awaiting_eval",
                "violated_or_required_invariants": "matched_downstream_dominance",
                "why": (
                    "It satisfies same-shape, identity, frozen-router, geometry, and delta-cap gates, "
                    "but final acceptance is still downstream."
                ),
                "use_if": "Accept only after eval-bundle audit and final selector beat source frontier.",
                "same_shape_output": True,
            },
        ]
    ).assign(global_default_accept_count=accepted, router_status=status_by_id.get("router_stability"))


def build_algorithm_spec(invariants: pd.DataFrame, ctx: dict[str, Any]) -> dict[str, Any]:
    unified = ctx["unified_candidate"]
    final_selection = ctx["final_selection"].get("current_selection") or {}
    hard_failures = [
        str(row["invariant_id"])
        for _, row in invariants.iterrows()
        if bool(row["hard_gate"]) and str(row["status"]).startswith(("reject", "awaiting"))
    ]
    return {
        "schema_version": 1,
        "name": "same_shape_guarded_average",
        "objective": (
            "Maximize retained nonbase/source capability under same-shape, endpoint-frontier, "
            "expert-identity, router-stability, routed-delta, and downstream-dominance constraints."
        ),
        "moe_parameter_objective": {
            "decision_variable": "per routed expert group nonbase coefficient c_g in [0, 1]",
            "proxy_maximize": "sum_g route_mass_g * source_retention_g * c_g",
            "hard_constraints": [
                "router tensors are frozen unless router-stability or calibration gate passes",
                "predicted routed expert relative delta <= hard cap",
                "same-shape tensor names/config are unchanged",
                "source endpoints remain selectable",
            ],
            "risk_terms": [
                "router fragility",
                "low route evidence",
                "load/capacity pressure",
                "expert internal geometry risk",
                "layer/chunk sensitivity",
                "source conflict",
            ],
            "current_selected_candidate": unified.get("selected_candidate_id"),
            "current_selected_family": unified.get("selected_candidate_family"),
            "current_retention": maybe_float(unified.get("selected_nonbase_mass_retention")),
            "current_max_predicted_relative_delta": maybe_float(
                unified.get("selected_max_predicted_relative_delta")
            ),
        },
        "gate_order": [
            "same_shape_contract",
            "endpoint_frontier",
            "expert_identity_alignment",
            "router_stability",
            "expert_internal_geometry",
            "routed_delta_trust_region",
            "router_calibration_not_acceptance",
            "matched_downstream_dominance",
        ],
        "bounded_regret_statement": (
            "The optimizer is not a proof that averaging always wins. Its defensible guarantee is selector-level: "
            "because endpoints are always in the candidate set and final acceptance requires audited matched eval "
            "with source-dominance, task-regression, uncertainty, and paired-prediction gates, the deployed choice "
            "falls back to the best source whenever the same-shape average lacks measured evidence."
        ),
        "current_blockers": hard_failures,
        "final_selection_status": final_selection.get("status"),
    }


def write_figure(path: Path, invariants: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [
        "pass",
        "conditional_geometry_shrink",
        "mechanism_supported_not_sufficient",
        "reject_default_average",
        "reject_as_sufficient_gate",
        "reject_router_movement",
        "awaiting_eval",
    ]
    colors = {
        "pass": "#15803d",
        "conditional_geometry_shrink": "#ca8a04",
        "mechanism_supported_not_sufficient": "#0f766e",
        "reject_default_average": "#b91c1c",
        "reject_as_sufficient_gate": "#b91c1c",
        "reject_router_movement": "#b91c1c",
        "awaiting_eval": "#4f46e5",
    }
    labels = {
        "pass": "pass",
        "conditional_geometry_shrink": "conditional geometry shrink",
        "mechanism_supported_not_sufficient": "mechanism supported, not sufficient",
        "reject_default_average": "reject default average",
        "reject_as_sufficient_gate": "reject as sufficient gate",
        "reject_router_movement": "reject router movement",
        "awaiting_eval": "awaiting eval",
    }
    counts = invariants["status"].value_counts().to_dict()
    statuses = [status for status in order if counts.get(status, 0)]
    values = [counts[status] for status in statuses]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y_labels = [labels.get(status, status.replace("_", " ")) for status in statuses]
    ax.barh(y_labels, values, color=[colors.get(status, "#6b7280") for status in statuses])
    ax.set_xlabel("invariant count")
    ax.set_title("Average invariant audit status")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for index, value in enumerate(values):
        ax.text(value + 0.05, index, str(value), va="center", ha="left", fontsize=10)
    ax.set_xlim(0, max(values) + 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(
    summary: dict[str, Any],
    invariants: pd.DataFrame,
    methods: pd.DataFrame,
    algorithm: dict[str, Any],
) -> str:
    lines = [
        "# Average Invariant Audit",
        "",
        "This audit turns model-averaging literature and current Dense/MoE probes into executable invariants. It is not a method leaderboard: it explains which assumptions must hold before an average can be accepted, and which assumptions currently fail.",
        "",
        "## Result",
        "",
        f"- Invariants: `{summary['invariant_count']}`",
        f"- Hard gates not yet accepting average: `{summary['hard_gate_blocker_count']}`",
        f"- Default-accepted method families: `{summary['default_accepted_method_count']}`",
        f"- Current algorithm: `{algorithm['name']}`",
        f"- Final selection status: `{algorithm['final_selection_status']}`",
        "",
        "## Invariants",
        "",
        "| invariant | domain | hard | status | current value | algorithm action |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in invariants.iterrows():
        lines.append(
            f"| `{row['invariant_id']}` | `{row['domain']}` | `{row['hard_gate']}` | "
            f"`{row['status']}` | {row['current_value']} | {row['algorithm_action']} |"
        )
    lines.extend(
        [
            "",
            "## Method Matrix",
            "",
            "| method family | current gate | required invariants | why | use if |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in methods.iterrows():
        lines.append(
            f"| `{row['method_family']}` | `{row['current_gate']}` | "
            f"{row['violated_or_required_invariants']} | {row['why']} | {row['use_if']} |"
        )
    lines.extend(
        [
            "",
            "## Selector-Level Statement",
            "",
            algorithm["bounded_regret_statement"],
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['invariants']}`",
            f"- `{summary['outputs']['method_matrix']}`",
            f"- `{summary['outputs']['algorithm_spec']}`",
            f"- `{summary['outputs']['literature_sources']}`",
            f"- `{summary['outputs']['figure']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    ctx = collect_context(args)
    invariants = build_invariants(ctx)
    methods = build_method_matrix(invariants, ctx)
    algorithm = build_algorithm_spec(invariants, ctx)

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    invariants_path = output_dir / "invariant_table.csv"
    methods_path = output_dir / "method_invariant_matrix.csv"
    algorithm_path = output_dir / "algorithm_spec.json"
    sources_path = output_dir / "literature_sources.json"
    figure_path = output_dir / "invariant_status.png"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    hard_blockers = invariants[
        invariants["hard_gate"].astype(bool)
        & invariants["status"].astype(str).str.startswith(("reject", "awaiting"))
    ]
    status_counts = {str(key): int(value) for key, value in invariants["status"].value_counts().items()}
    default_accepted = int(ctx["method_gate"].get("accepted_by_default_count", 0) or 0)
    summary = {
        "schema_version": 1,
        "status": "average_requires_guarded_selector_and_eval",
        "invariant_count": int(len(invariants)),
        "method_family_count": int(len(methods)),
        "hard_gate_blocker_count": int(len(hard_blockers)),
        "status_counts": status_counts,
        "default_accepted_method_count": default_accepted,
        "default_rejected_method_count": int(ctx["method_gate"].get("default_rejected_count", 0) or 0),
        "same_shape_contract_pass": bool(ctx["preflight"].get("same_shape_contract_pass", False)),
        "router_allowed_layers": maybe_int(ctx["router_move"].get("allowed_router_layer_count")),
        "router_layer_count": maybe_int(ctx["router_move"].get("router_layer_count")),
        "expert_identity_fraction": maybe_float(
            (ctx["preflight"].get("expert_identity_gate") or {}).get("frac_layers_identity_optimal")
        ),
        "high_route_geometry_risk_expert_count": maybe_int(
            ctx["expert_geometry"].get("high_route_geometry_risk_expert_count")
        ),
        "selected_candidate_id": ctx["unified_candidate"].get("selected_candidate_id"),
        "selected_nonbase_mass_retention": maybe_float(
            ctx["unified_candidate"].get("selected_nonbase_mass_retention")
        ),
        "selected_max_predicted_relative_delta": maybe_float(
            ctx["unified_candidate"].get("selected_max_predicted_relative_delta")
        ),
        "final_selection_status": (ctx["final_selection"].get("current_selection") or {}).get("status"),
        "outputs": {
            "invariants": rel(invariants_path),
            "method_matrix": rel(methods_path),
            "algorithm_spec": rel(algorithm_path),
            "literature_sources": rel(sources_path),
            "figure": rel(figure_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }

    invariants.to_csv(invariants_path, index=False)
    methods.to_csv(methods_path, index=False)
    algorithm_path.write_text(json.dumps(algorithm, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sources_path.write_text(json.dumps(LITERATURE_SOURCES, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_figure(figure_path, invariants)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, invariants, methods, algorithm), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a theory/probe invariant audit for Dense/MoE averaging.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/average_invariant_audit"))
    parser.add_argument(
        "--connectivity-summary",
        type=Path,
        default=Path("results/average_connectivity_diagnostic/summary.json"),
    )
    parser.add_argument(
        "--connectivity-rows",
        type=Path,
        default=Path("results/average_connectivity_diagnostic/path_diagnostics.csv"),
    )
    parser.add_argument(
        "--preflight-summary",
        type=Path,
        default=Path("results/moe_unified_preflight_qwen3_30b/summary.json"),
    )
    parser.add_argument(
        "--router-move-summary",
        type=Path,
        default=Path("results/qwen3_moe_router_move_gate/summary.json"),
    )
    parser.add_argument(
        "--expert-geometry-summary",
        type=Path,
        default=Path("results/qwen3_moe_expert_geometry_probe/summary.json"),
    )
    parser.add_argument(
        "--unified-candidate-summary",
        type=Path,
        default=Path("results/qwen3_moe_unified_mechanism_candidate/summary.json"),
    )
    parser.add_argument(
        "--unified-delta-audit-summary",
        type=Path,
        default=Path("results/qwen3_moe_unified_mechanism_delta_audit/summary.json"),
    )
    parser.add_argument(
        "--router-calibration-nll-summary",
        type=Path,
        default=Path("results/qwen3_moe_router_calibration_nll_probe/summary.json"),
    )
    parser.add_argument(
        "--final-selection-summary",
        type=Path,
        default=Path("results/qwen3_moe_final_candidate_selection/summary.json"),
    )
    parser.add_argument(
        "--eval-bundle-audit-summary",
        type=Path,
        default=Path("results/qwen3_moe_eval_bundle_audit/summary.json"),
    )
    parser.add_argument(
        "--method-gate-summary",
        type=Path,
        default=Path("results/average_method_gate_matrix/summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(
        "Wrote invariant audit: "
        f"{summary['status']}; hard_blockers={summary['hard_gate_blocker_count']}; "
        f"default_accepted={summary['default_accepted_method_count']}"
    )


if __name__ == "__main__":
    main()
