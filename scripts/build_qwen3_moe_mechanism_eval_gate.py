#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_METHODS = [
    "source_qwen3_30b_instruct",
    "source_qwen3_30b_coder",
]

CANDIDATE_METHODS = [
    "qwen3_moe_unified_route_guarded_candidate",
    "qwen3_moe_audit_gated_candidate",
    "qwen3_moe_trust_region_candidate",
    "qwen3_moe_expert_only_trust_region_candidate",
    "qwen3_moe_tail_trimmed_expert_only_candidate",
    "qwen3_moe_searched_no_gt065_max_retention_candidate",
    "qwen3_moe_layer_chunk_candidate",
    "qwen3_moe_unified_mechanism_candidate",
    "qwen3_moe_mechanistic_unified_candidate",
    "qwen3_moe_subspace_scaled_candidate",
    "qwen3_moe_router_coupled_candidate",
    "qwen3_moe_harc_router_candidate",
]

METHOD_ORDER = SOURCE_METHODS + CANDIDATE_METHODS

METHOD_META: dict[str, dict[str, str]] = {
    "source_qwen3_30b_instruct": {
        "role": "source",
        "short_name": "instruct_source",
        "mechanism": "general/instruction endpoint",
        "question": "Instruct source sets the general, safety, and instruction-following control floor.",
        "required_controls": "coder source; all candidates",
    },
    "source_qwen3_30b_coder": {
        "role": "source",
        "short_name": "coder_source",
        "mechanism": "code endpoint",
        "question": "Coder source sets the code-specialized control floor.",
        "required_controls": "instruct source; all candidates",
    },
    "qwen3_moe_unified_route_guarded_candidate": {
        "role": "candidate",
        "short_name": "route_guarded",
        "mechanism": "freeze router + route-conditioned expert weights + small attention step",
        "question": "Does preserving the original router while moving route-relevant experts keep both source abilities?",
        "required_controls": "both sources; audit-gated",
    },
    "qwen3_moe_audit_gated_candidate": {
        "role": "candidate",
        "short_name": "audit_gated",
        "mechanism": "route-conditioned experts + file-level relative-delta cap",
        "question": "Are the largest routed-expert deltas harmful enough that clipping them improves downstream behavior?",
        "required_controls": "route-guarded; trust-region",
    },
    "qwen3_moe_trust_region_candidate": {
        "role": "candidate",
        "short_name": "trust_region",
        "mechanism": "route/load/category/router-fragility trust-region caps",
        "question": "Do internal MoE risk signals predict which expert deltas need a smaller trust region?",
        "required_controls": "audit-gated; expert-only",
    },
    "qwen3_moe_expert_only_trust_region_candidate": {
        "role": "candidate",
        "short_name": "expert_only",
        "mechanism": "trust-region experts + frozen shared attention + frozen router",
        "question": "Is the shared attention Coder delta useful, or is expert-only movement the safer unified rule?",
        "required_controls": "trust-region; tail-trimmed",
    },
    "qwen3_moe_tail_trimmed_expert_only_candidate": {
        "role": "candidate",
        "short_name": "tail_trimmed",
        "mechanism": "expert-only + second-stage routed-expert tail cap at 0.65",
        "question": "Does removing the remaining high-tail expert deltas preserve utility while lowering risk?",
        "required_controls": "expert-only; both sources",
    },
    "qwen3_moe_searched_no_gt065_max_retention_candidate": {
        "role": "candidate",
        "short_name": "searched_no_gt065",
        "mechanism": "freeze router/attention + source-route expert weights + searched uniform 0.65 cap",
        "question": "Can a simple searched 0.65 cap replace the hand-built route/load/category risk penalties?",
        "required_controls": "tail-trimmed; expert-only; both sources",
    },
    "qwen3_moe_layer_chunk_candidate": {
        "role": "candidate",
        "short_name": "layer_chunk",
        "mechanism": "freeze router/attention + source-route expert weights + importance-guided layer/chunk coefficients",
        "question": "Do layer sensitivity coefficients reduce harmful movement without removing useful Coder specialization?",
        "required_controls": "searched_no_gt065; tail-trimmed; both sources",
    },
    "qwen3_moe_unified_mechanism_candidate": {
        "role": "candidate",
        "short_name": "unified_mechanism",
        "mechanism": "mechanism-optimized same-shape MoE average with frozen router/attention and router/evidence/geometry-risk expert caps",
        "question": "Does the unified mechanism optimizer produce a candidate that survives source dominance and downstream task gates?",
        "required_controls": "both sources; searched_no_gt065; layer_chunk; router calibration selector",
    },
    "qwen3_moe_mechanistic_unified_candidate": {
        "role": "candidate",
        "short_name": "mechanistic_unified",
        "mechanism": "damped per-expert benefit/curvature/interference optimizer with frozen router/attention",
        "question": "Does a first-principles marginal-utility scale law improve the unified average beyond hand-weighted mechanism caps?",
        "required_controls": "both sources; unified_mechanism; feedback optimizer; expert geometry; expert subspace conflict probe",
    },
    "qwen3_moe_subspace_scaled_candidate": {
        "role": "candidate",
        "short_name": "subspace_scaled",
        "mechanism": "unified mechanism candidate plus extra shrink for uncovered expert channel/chunk subspace conflicts",
        "question": "Do the remaining uncovered subspace-conflict experts explain downstream regressions beyond the unified mechanism candidate?",
        "required_controls": "both sources; unified_mechanism; layer_chunk; expert subspace conflict probe",
    },
    "qwen3_moe_router_coupled_candidate": {
        "role": "candidate",
        "short_name": "router_coupled",
        "mechanism": "mechanistic unified candidate plus extra shrink for high router-boundary-fragility expert groups",
        "question": "Does extra router-boundary conservative expert shrink improve downstream robustness beyond mechanistic_unified?",
        "required_controls": "both sources; mechanistic_unified; router-expert coupling probe; expert subspace conflict probe",
    },
    "qwen3_moe_harc_router_candidate": {
        "role": "candidate",
        "short_name": "harc_router",
        "mechanism": "unified mechanism candidate plus matrix-free HARC router calibration delta",
        "question": "Does second-order router calibration improve dispatch beyond the frozen-router unified mechanism candidate?",
        "required_controls": "both sources; unified_mechanism; router-only baseline; HARC stats/solver/materialization checks",
    },
}

MECHANISM_TESTS = [
    {
        "test": "source_control_floor",
        "from_method": "source_qwen3_30b_instruct",
        "to_method": "source_qwen3_30b_coder",
        "mechanism_question": "How different are the source endpoints on the same downstream tasks?",
        "why_it_matters": "A merged checkpoint is only meaningful if it is compared with both endpoint trade-offs, not just one source.",
        "pass_signal": "Sources expose complementary strengths; candidate ranking uses Pareto dominance across both.",
        "fail_signal": "One source dominates the other and all candidates; the unified selector should return that endpoint.",
    },
    {
        "test": "tail_delta_cap",
        "from_method": "qwen3_moe_unified_route_guarded_candidate",
        "to_method": "qwen3_moe_audit_gated_candidate",
        "mechanism_question": "Does clipping extreme routed-expert deltas help?",
        "why_it_matters": "This isolates whether file-level delta outliers are a real behavioral risk or only a cosmetic norm metric.",
        "pass_signal": "Audit-gated keeps or improves avg/worst/task scores after removing high relative-delta tails.",
        "fail_signal": "Audit-gated loses task scores; the cap is too tight or the high-delta experts carry useful ability.",
    },
    {
        "test": "route_load_trust_region",
        "from_method": "qwen3_moe_audit_gated_candidate",
        "to_method": "qwen3_moe_trust_region_candidate",
        "mechanism_question": "Do route/load/category/fragility probes identify the expert groups that need a tighter cap?",
        "why_it_matters": "This is the core internal-parameter probe: it tests whether MoE-specific signals explain performance, not only norm size.",
        "pass_signal": "Trust-region improves or preserves downstream scores while reducing routed tail risk.",
        "fail_signal": "Trust-region drops scores versus audit-gated; the internal risk gate is over-conservative or mis-specified.",
    },
    {
        "test": "shared_attention_ablation",
        "from_method": "qwen3_moe_trust_region_candidate",
        "to_method": "qwen3_moe_expert_only_trust_region_candidate",
        "mechanism_question": "Should the unified MoE rule move shared attention, or keep it fixed?",
        "why_it_matters": "Delta audit alone cannot answer this because attention removal barely changes routed-tail risk; only downstream eval decides utility.",
        "pass_signal": "Expert-only matches or beats trust-region; freeze shared attention in the unified rule.",
        "fail_signal": "Trust-region beats expert-only; retain a small shared-attention step and tune its coefficient.",
    },
    {
        "test": "second_stage_tail_trim",
        "from_method": "qwen3_moe_expert_only_trust_region_candidate",
        "to_method": "qwen3_moe_tail_trimmed_expert_only_candidate",
        "mechanism_question": "Does the stricter 0.65 routed-expert tail cap remove risk without removing ability?",
        "why_it_matters": "This decides whether the next unified default should use a 0.65 tail trim or stop at the trust-region cap.",
        "pass_signal": "Tail-trimmed keeps or improves downstream scores while eliminating >0.75 routed tails.",
        "fail_signal": "Tail-trimmed loses task scores; the remaining tail contained useful specialization.",
    },
    {
        "test": "risk_penalty_simplification",
        "from_method": "qwen3_moe_tail_trimmed_expert_only_candidate",
        "to_method": "qwen3_moe_searched_no_gt065_max_retention_candidate",
        "mechanism_question": "Are hand-built risk penalties necessary after a uniform 0.65 expert cap is enforced?",
        "why_it_matters": "The cap-law search found that a simple 0.65 cap removes the high tail with slightly higher route-mass retention; downstream eval must decide whether the simpler rule keeps ability.",
        "pass_signal": "Searched no-gt-0.65 matches or beats tail-trimmed; simplify the unified expert cap law.",
        "fail_signal": "Tail-trimmed beats searched no-gt-0.65; keep route/load/category risk penalties despite the internal proxy result.",
    },
    {
        "test": "layer_chunk_sensitivity",
        "from_method": "qwen3_moe_searched_no_gt065_max_retention_candidate",
        "to_method": "qwen3_moe_layer_chunk_candidate",
        "mechanism_question": "Do importance-guided layer/chunk coefficients improve the unified MoE rule beyond a uniform expert cap?",
        "why_it_matters": "The layer/chunk candidate has lower structural delta and preserves high route-mass Coder contribution, but only downstream tasks can tell whether shrinking high-sensitive layers removed useful specialization.",
        "pass_signal": "Layer/chunk matches or beats searched no-gt-0.65 on avg/worst/task scores; keep layer-sensitive coefficients in the unified optimizer.",
        "fail_signal": "Searched no-gt-0.65 beats layer/chunk; the extra layer sensitivity shrink is over-conservative or mis-targeted.",
    },
    {
        "test": "candidate_vs_sources",
        "from_method": "source_qwen3_30b_instruct",
        "to_method": "qwen3_moe_unified_mechanism_candidate",
        "mechanism_question": "Does any same-shape candidate avoid Pareto domination by the two source endpoints?",
        "why_it_matters": "If every candidate is dominated by an endpoint, the correct same-shape output is an endpoint/no-average, not a worse average.",
        "pass_signal": "At least one candidate is non-dominated by both sources and wins on avg/worst/task trade-off.",
        "fail_signal": "All candidates are dominated; select endpoint and use probes to design the next intervention.",
    },
    {
        "test": "unified_mechanism_optimizer",
        "from_method": "qwen3_moe_layer_chunk_candidate",
        "to_method": "qwen3_moe_unified_mechanism_candidate",
        "mechanism_question": "Does the router/evidence/geometry-risk optimizer improve downstream behavior beyond the layer/chunk candidate?",
        "why_it_matters": "The unified method is now a distinct materialized checkpoint; norm safety alone cannot prove the extra risk-weighted shrink is useful.",
        "pass_signal": "Unified matches or beats layer/chunk without source dominance or task regression.",
        "fail_signal": "Layer/chunk or an endpoint dominates, so the extra unified risk shrink should be rejected.",
    },
    {
        "test": "expert_subspace_conflict_ablation",
        "from_method": "qwen3_moe_mechanistic_unified_candidate",
        "to_method": "qwen3_moe_subspace_scaled_candidate",
        "mechanism_question": "Do uncovered high subspace-conflict experts need additional non-base shrink after the unified mechanism cap?",
        "why_it_matters": "The subspace probe found only a small set of uncovered channel/chunk conflicts; this isolates whether those localized conflicts matter behaviorally.",
        "pass_signal": "Subspace-scaled matches or beats mechanistic_unified on avg/worst/task scores without source dominance.",
        "fail_signal": "Mechanistic_unified beats subspace-scaled; the remaining subspace conflicts were not behaviorally harmful or the shrink removed useful specialization.",
    },
    {
        "test": "mechanistic_unified_optimizer",
        "from_method": "qwen3_moe_unified_mechanism_candidate",
        "to_method": "qwen3_moe_mechanistic_unified_candidate",
        "mechanism_question": "Does the benefit/curvature/interference objective explain a better scale law than the current risk-weighted cap search?",
        "why_it_matters": "This tests the proposed unified algorithm as a scientific mechanism: high-benefit groups should be preserved while high-curvature/interference groups are damped.",
        "pass_signal": "Mechanistic unified matches or beats unified_mechanism without source dominance or task regression.",
        "fail_signal": "Unified_mechanism beats mechanistic_unified; the marginal-utility proxy is overfit or missing an internal signal.",
    },
    {
        "test": "router_coupled_boundary_ablation",
        "from_method": "qwen3_moe_mechanistic_unified_candidate",
        "to_method": "qwen3_moe_router_coupled_candidate",
        "mechanism_question": "Does the layer-level router-boundary fragility signal justify extra expert shrink after the B/H/I scale law?",
        "why_it_matters": "The coupling probe shows router fragility already correlates with expert shrink; this ablation tests whether adding a direct router-boundary term gives real downstream robustness or only sacrifices retention.",
        "pass_signal": "Router-coupled matches or beats mechanistic_unified on avg/worst/task scores despite lower nonbase retention; keep a direct router-boundary term.",
        "fail_signal": "Mechanistic_unified beats router-coupled; B/H/I already captured the useful router-boundary risk and the extra shrink is over-conservative.",
    },
    {
        "test": "harc_router_calibration_ablation",
        "from_method": "qwen3_moe_unified_mechanism_candidate",
        "to_method": "qwen3_moe_harc_router_candidate",
        "mechanism_question": "Does HARC-style second-order router calibration recover downstream score beyond the frozen-router unified mechanism candidate?",
        "why_it_matters": "The current safe default freezes routers because direct router averaging breaks routing; HARC should only become useful if a curvature-calibrated router delta improves downstream behavior under the same task manifest.",
        "pass_signal": "HARC router candidate matches or beats unified_mechanism on avg/worst/task scores without source dominance or task regression; promote HARC router calibration from ablation to a gated candidate family.",
        "fail_signal": "Unified_mechanism or a source endpoint beats HARC router candidate; keep router freeze as the default and treat router calibration as overfit or under-supported.",
    },
]

LITERATURE_SOURCES = [
    {
        "key": "loss_landscape_visualization",
        "title": "Visualizing the Loss Landscape of Neural Nets",
        "url": "https://arxiv.org/abs/1712.09913",
        "mechanism_used_here": "Treat loss surfaces as 2D slices through weight space; for this project the axes are task vectors or source deltas rather than random directions.",
    },
    {
        "key": "mode_connectivity",
        "title": "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs",
        "url": "https://arxiv.org/abs/1802.10026",
        "mechanism_used_here": "If two checkpoints are connected by a low-loss path, averaging may work; if the straight line crosses a barrier, the selector should shrink or reject the merge.",
    },
    {
        "key": "essentially_no_barriers",
        "title": "Essentially No Barriers in Neural Network Energy Landscape",
        "url": "https://arxiv.org/abs/1803.00885",
        "mechanism_used_here": "Nonlinear connectivity can exist even when straight-line averaging is poor, motivating path probes rather than only midpoint scores.",
    },
    {
        "key": "model_soups",
        "title": "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time",
        "url": "https://arxiv.org/abs/2203.05482",
        "mechanism_used_here": "Weight averaging is plausible when fine-tuned models sit in one low-error basin; the gate must test that assumption instead of assuming it.",
    },
    {
        "key": "fisher_merging",
        "title": "Merging Models with Fisher-Weighted Averaging",
        "url": "https://arxiv.org/abs/2111.09832",
        "mechanism_used_here": "Fisher/Laplace weighting gives a local quadratic explanation, but our Qwen dense probe shows local curvature can underpredict nonlocal barriers.",
    },
    {
        "key": "ties",
        "title": "TIES-Merging: Resolving Interference When Merging Models",
        "url": "https://arxiv.org/abs/2306.01708",
        "mechanism_used_here": "Sign and magnitude conflicts are useful probes, but sparse conflict rules still need held-out/vLLM gates before touching broad LLM modules.",
    },
    {
        "key": "git_rebasin",
        "title": "Git Re-Basin: Merging Models modulo Permutation Symmetries",
        "url": "https://arxiv.org/abs/2209.04836",
        "mechanism_used_here": "Permutation symmetries explain why expert identity alignment must precede same-name averaging.",
    },
    {
        "key": "mergeme",
        "title": "MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs",
        "url": "https://arxiv.org/abs/2502.00997",
        "mechanism_used_here": "MoE merging needs explicit handling of parameter interference and routing, not only uniform expert averaging.",
    },
    {
        "key": "harc",
        "title": "When Model Merging Breaks Routing: Training-Free Calibration for MoE",
        "url": "https://arxiv.org/abs/2606.03391",
        "mechanism_used_here": "Router perturbations can cause routing breakdown; the current Qwen3 rule freezes router first and leaves router calibration as a separate ablation.",
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


def read_json_if_exists(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
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
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return clean_value(value)


def maybe_float(value: Any) -> float | None:
    value = clean_value(value)
    return None if value is None else float(value)


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def local_gpu_status() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "status": "nvidia_smi_missing", "detail": "nvidia-smi is not on PATH"}
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return {"available": False, "status": type(exc).__name__, "detail": str(exc)}
    if result.returncode != 0:
        return {
            "available": False,
            "status": "nvidia_smi_failed",
            "detail": (result.stderr or result.stdout).strip(),
        }
    gpu_lines = [line for line in result.stdout.splitlines() if line.strip().startswith("GPU ")]
    return {
        "available": bool(gpu_lines),
        "status": "available" if gpu_lines else "no_gpus_reported",
        "gpu_count": len(gpu_lines),
        "detail": result.stdout.strip(),
    }


def primary_metric(row: pd.Series | dict[str, Any]) -> tuple[str, float]:
    for key in ("strict_exact", "accuracy", "policy_accuracy", "compile_rate"):
        value = row.get(key)
        if clean_value(value) is not None:
            return key, float(value)
    return "score", 0.0


def read_eval_state(eval_output_dir: str | Path) -> dict[str, Any]:
    root = repo_path(eval_output_dir)
    summary = read_json_if_exists(root / "summary.json")
    metrics = read_csv_if_exists(root / "metrics.csv")
    model_summary = read_csv_if_exists(root / "model_summary.csv")
    status = str(summary.get("status", "not_run"))
    if not model_summary.empty:
        first = model_summary.iloc[0]
        avg_primary = maybe_float(first.get("avg_primary_score"))
        worst_primary = maybe_float(first.get("worst_primary_score"))
    else:
        summary_rows = summary.get("model_summary") or []
        first = summary_rows[0] if summary_rows else {}
        avg_primary = maybe_float(first.get("avg_primary_score"))
        worst_primary = maybe_float(first.get("worst_primary_score"))

    task_scores: dict[str, float] = {}
    if not metrics.empty:
        for _, row in metrics.iterrows():
            task = str(row.get("task", "unknown"))
            _, score = primary_metric(row)
            task_scores[task] = score

    state: dict[str, Any] = {
        "eval_status": status,
        "eval_completed": status == "complete",
        "avg_primary_score": avg_primary,
        "worst_primary_score": worst_primary,
        "task_scores": task_scores,
        "metrics_path": rel(root / "metrics.csv") if (root / "metrics.csv").exists() else None,
        "summary_path": rel(root / "summary.json") if (root / "summary.json").exists() else None,
    }
    for task, score in task_scores.items():
        state[f"task_{task}_score"] = score
    return state


def load_ordered_plan(plan_path: Path) -> pd.DataFrame:
    plan = pd.read_csv(plan_path)
    rows = []
    for method in METHOD_ORDER:
        selected = plan[plan["method"] == method]
        if selected.empty:
            raise ValueError(f"{plan_path} is missing required method: {method}")
        rows.append(selected.iloc[0].to_dict())
    return pd.DataFrame(rows)


def command_join(tokens: list[str]) -> str:
    return " ".join(shlex.quote(str(token)) for token in tokens)


def bool_from_summary(value: Any) -> bool:
    value = clean_value(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def add_harc_router_candidate_plan_row(
    plan: pd.DataFrame,
    *,
    summary_path: Path,
    port: int,
) -> pd.DataFrame:
    method = "qwen3_moe_harc_router_candidate"
    if method in set(str(item) for item in plan.get("method", pd.Series(dtype=object)).tolist()):
        return plan

    summary = read_json_if_exists(summary_path)
    reference = plan[plan["method"] == "qwen3_moe_unified_mechanism_candidate"]
    if reference.empty:
        reference = plan[plan["method"].astype(str).str.startswith("source_")]
    if reference.empty:
        raise ValueError("Cannot build HARC router candidate plan row without a reference Qwen3 row")
    ref = reference.iloc[0].to_dict()
    checkpoint_path = str(
        summary.get("candidate_checkpoint_dir")
        or "results/checkpoints/qwen3_moe_harc_router_candidate"
    )
    checkpoint_exists = bool_from_summary(summary.get("candidate_checkpoint_exists")) and repo_path(checkpoint_path).exists()
    if checkpoint_exists:
        serve_status = "ready_to_host"
    elif summary.get("status") == "harc_router_candidate_waiting_for_solver_delta":
        serve_status = "checkpoint_missing_until_harc_solver_delta"
    else:
        serve_status = "checkpoint_missing_until_materialized"

    served_model_id = "candidate_qwen3_moe_harc_router_candidate"
    base_url = f"http://127.0.0.1:{port}/v1"
    eval_output_dir = "results/vllm_checkpoint_eval/qwen3_moe_harc_router_candidate"
    tasks = str(ref.get("tasks") or "gsm8k,mmlu,safety,humaneval_compile")
    example_source = str(ref.get("example_source") or "datasets")
    max_examples = int(ref.get("max_examples") or 64)
    dtype = str(ref.get("dtype") or "bfloat16")
    tensor_parallel = int(ref.get("tensor_parallel_size") or 4)
    gpu = str(ref.get("gpu") or "0,1,2,3")
    serve_command = command_join(
        [
            f"CUDA_VISIBLE_DEVICES={gpu}",
            "vllm",
            "serve",
            checkpoint_path,
            "--served-model-name",
            served_model_id,
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--dtype",
            dtype,
            "--tensor-parallel-size",
            tensor_parallel,
        ]
    )
    eval_command = command_join(
        [
            "python",
            "scripts/run_vllm_downstream_eval.py",
            "--base-url",
            base_url,
            "--models",
            served_model_id,
            "--tasks",
            tasks,
            "--example-source",
            example_source,
            "--max-examples",
            max_examples,
            "--output-dir",
            eval_output_dir,
            "--task-manifest",
            "results/qwen3_moe_mechanism_eval_gate/task_manifest.json",
            "--create-task-manifest-if-missing",
        ]
    )
    row = {
        "eval_order": int(pd.to_numeric(plan.get("eval_order"), errors="coerce").max()) + 1,
        "candidate_source": rel(summary_path),
        "method": method,
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": checkpoint_exists,
        "serve_status": serve_status,
        "served_model_id": served_model_id,
        "host": "127.0.0.1",
        "port": port,
        "base_url": base_url,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel,
        "gpu": gpu,
        "tasks": tasks,
        "example_source": example_source,
        "max_examples": max_examples,
        "eval_output_dir": eval_output_dir,
        "eval_status": "not_run",
        "eval_completed": False,
        "eval_avg_primary_score": None,
        "eval_worst_primary_score": None,
        "serve_command": serve_command,
        "eval_command": eval_command,
        "notes": (
            "HARC router calibration ablation: start from the unified mechanism candidate, then apply "
            "a matrix-free second-order router delta. Current materialization status is "
            f"{summary.get('status', 'missing_summary')}; this row becomes hostable only after the "
            "router cache, solver delta, and materialization checks succeed."
        ),
    }
    for column in plan.columns:
        row.setdefault(column, None)
    augmented = plan.copy()
    augmented.loc[len(augmented), list(plan.columns)] = [row.get(column) for column in plan.columns]
    return augmented


def load_augmented_ordered_plan(args: argparse.Namespace) -> pd.DataFrame:
    plan = pd.read_csv(repo_path(args.vllm_plan))
    plan = add_harc_router_candidate_plan_row(
        plan,
        summary_path=repo_path(args.harc_router_candidate_summary),
        port=args.harc_router_candidate_port,
    )
    rows = []
    for method in METHOD_ORDER:
        selected = plan[plan["method"] == method]
        if selected.empty:
            raise ValueError(f"{repo_path(args.vllm_plan)} is missing required method after augmentation: {method}")
        rows.append(selected.iloc[0].to_dict())
    return pd.DataFrame(rows)


def load_delta_frontier(delta_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates = read_csv_if_exists(delta_dir / "candidate_delta_frontier.csv")
    pairwise = read_csv_if_exists(delta_dir / "pairwise_delta_reductions.csv")
    summary = read_json_if_exists(delta_dir / "summary.json")
    return candidates, pairwise, summary


def build_eval_gate_plan(
    plan: pd.DataFrame,
    delta_candidates: pd.DataFrame,
    readiness: pd.DataFrame,
) -> pd.DataFrame:
    delta_by_method = {
        str(row["method"]): row.to_dict()
        for _, row in delta_candidates.iterrows()
        if "method" in row and clean_value(row.get("method")) is not None
    }
    readiness_by_method = {
        str(row["candidate"]): row.to_dict()
        for _, row in readiness.iterrows()
        if "candidate" in row and clean_value(row.get("candidate")) is not None
    }

    rows: list[dict[str, Any]] = []
    for idx, row in plan.iterrows():
        method = str(row["method"])
        meta = METHOD_META[method]
        eval_state = read_eval_state(row["eval_output_dir"])
        delta = delta_by_method.get(method, {})
        ready = readiness_by_method.get(method, {})
        role = meta["role"]
        audit_status = delta.get("status")
        row_out = {
            "gate_order": idx,
            "method": method,
            "short_name": meta["short_name"],
            "role": role,
            "mechanism": meta["mechanism"],
            "mechanism_question": meta["question"],
            "required_controls": meta["required_controls"],
            "checkpoint_path": row.get("checkpoint_path"),
            "checkpoint_exists": bool(row.get("checkpoint_exists")),
            "serve_status": row.get("serve_status"),
            "served_model_id": row.get("served_model_id"),
            "base_url": row.get("base_url"),
            "port": row.get("port"),
            "tasks": row.get("tasks"),
            "example_source": row.get("example_source"),
            "max_examples": row.get("max_examples"),
            "eval_output_dir": row.get("eval_output_dir"),
            "eval_status": eval_state["eval_status"],
            "eval_completed": bool(eval_state["eval_completed"]),
            "avg_primary_score": eval_state["avg_primary_score"],
            "worst_primary_score": eval_state["worst_primary_score"],
            "task_gsm8k_score": eval_state.get("task_gsm8k_score"),
            "task_mmlu_score": eval_state.get("task_mmlu_score"),
            "task_safety_score": eval_state.get("task_safety_score"),
            "task_humaneval_compile_score": eval_state.get("task_humaneval_compile_score"),
            "audit_status": audit_status if role == "candidate" else "source",
            "audit_passed": bool(role == "source" or audit_status == "passed"),
            "total_relative_delta_norm": maybe_float(delta.get("total_relative_delta_norm")),
            "routed_relative_delta_norm": maybe_float(delta.get("routed_relative_delta_norm")),
            "routed_max_tensor_relative_delta": maybe_float(delta.get("routed_max_tensor_relative_delta")),
            "routed_tensors_gt_1_0": clean_value(delta.get("routed_tensors_gt_1_0")),
            "routed_tensors_gt_0_75": clean_value(delta.get("routed_tensors_gt_0_75")),
            "routed_tensors_gt_0_65": clean_value(delta.get("routed_tensors_gt_0_65")),
            "attention_relative_delta_norm": maybe_float(delta.get("attention_relative_delta_norm")),
            "attention_changed_tensors": clean_value(delta.get("attention_changed_tensors")),
            "router_changed_tensors": clean_value(delta.get("router_changed_tensors")),
            "end_to_end_status": ready.get("end_to_end_status"),
            "serve_command": row.get("serve_command"),
            "eval_command": row.get("eval_command"),
            "notes": row.get("notes"),
        }
        rows.append(row_out)
    return pd.DataFrame(rows)


def score_columns(df: pd.DataFrame) -> list[str]:
    columns = ["avg_primary_score", "worst_primary_score"]
    columns.extend(
        col
        for col in [
            "task_gsm8k_score",
            "task_mmlu_score",
            "task_safety_score",
            "task_humaneval_compile_score",
        ]
        if col in df.columns
    )
    return columns


def dominates(left: pd.Series, right: pd.Series, columns: list[str], eps: float = 1e-9) -> bool:
    pairs: list[tuple[float, float]] = []
    for col in columns:
        l_val = clean_value(left.get(col))
        r_val = clean_value(right.get(col))
        if l_val is None or r_val is None:
            continue
        pairs.append((float(l_val), float(r_val)))
    if not pairs:
        return False
    return all(left_val >= right_val - eps for left_val, right_val in pairs) and any(
        left_val > right_val + eps for left_val, right_val in pairs
    )


def build_selection(gate: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    columns = score_columns(gate)
    sources = gate[gate["role"] == "source"].copy()
    candidates = gate[gate["role"] == "candidate"].copy()
    sources_complete = bool(len(sources) == len(SOURCE_METHODS) and sources["eval_completed"].all())
    candidates_complete = bool(len(candidates) == len(CANDIDATE_METHODS) and candidates["eval_completed"].all())

    if sources_complete:
        best_source_avg = sources.sort_values(["avg_primary_score", "worst_primary_score"], ascending=False).iloc[0]
        best_source_worst = sources.sort_values(["worst_primary_score", "avg_primary_score"], ascending=False).iloc[0]
    else:
        best_source_avg = pd.Series(dtype=object)
        best_source_worst = pd.Series(dtype=object)

    for _, row in gate.iterrows():
        method = str(row["method"])
        dominated_by = []
        if bool(row.get("eval_completed")) and sources_complete and row.get("role") == "candidate":
            for _, source in sources.iterrows():
                if dominates(source, row, columns):
                    dominated_by.append(str(source["method"]))
        eligible = (
            row.get("role") == "candidate"
            and bool(row.get("eval_completed"))
            and bool(row.get("audit_passed"))
            and sources_complete
            and not dominated_by
        )
        rows.append(
            {
                "method": method,
                "role": row.get("role"),
                "eval_completed": bool(row.get("eval_completed")),
                "audit_passed": bool(row.get("audit_passed")),
                "avg_primary_score": row.get("avg_primary_score"),
                "worst_primary_score": row.get("worst_primary_score"),
                "task_gsm8k_score": row.get("task_gsm8k_score"),
                "task_mmlu_score": row.get("task_mmlu_score"),
                "task_safety_score": row.get("task_safety_score"),
                "task_humaneval_compile_score": row.get("task_humaneval_compile_score"),
                "total_relative_delta_norm": row.get("total_relative_delta_norm"),
                "routed_tensors_gt_0_75": row.get("routed_tensors_gt_0_75"),
                "dominated_by_source": ",".join(dominated_by),
                "selection_eligible": eligible,
            }
        )

    selection_df = pd.DataFrame(rows)
    eligible_df = selection_df[selection_df["selection_eligible"]].copy()
    if not sources_complete:
        selection = {
            "status": "awaiting_source_eval",
            "selected_method": None,
            "reason": "Both Qwen3 source endpoints must be evaluated before candidate selection is meaningful.",
        }
    elif not candidates_complete:
        selection = {
            "status": "awaiting_candidate_eval",
            "selected_method": None,
            "reason": "All Qwen3 MoE candidates need the same downstream task run before choosing a unified rule.",
            "completed_candidate_count": int(candidates["eval_completed"].sum()),
            "candidate_count": int(len(candidates)),
        }
    elif eligible_df.empty:
        if sources_complete and not sources.empty:
            endpoint = best_source_avg
            selected = str(endpoint["method"])
        else:
            selected = None
        selection = {
            "status": "select_endpoint_no_average",
            "selected_method": selected,
            "reason": "Every completed same-shape candidate is dominated by at least one source endpoint or failed audit gates.",
        }
    else:
        eligible_df = eligible_df.sort_values(
            ["avg_primary_score", "worst_primary_score", "total_relative_delta_norm"],
            ascending=[False, False, True],
        )
        selected_row = eligible_df.iloc[0]
        selection = {
            "status": "selected_candidate",
            "selected_method": str(selected_row["method"]),
            "reason": "Selected the non-dominated audited candidate with the best avg/worst downstream score and lower delta norm as tie-breaker.",
        }

    selection.update(
        {
            "sources_complete": sources_complete,
            "candidates_complete": candidates_complete,
            "best_source_by_avg": None if best_source_avg.empty else str(best_source_avg.get("method")),
            "best_source_by_worst": None if best_source_worst.empty else str(best_source_worst.get("method")),
            "score_columns": columns,
        }
    )
    return selection_df, selection


def build_mechanism_tests(gate: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    gate_by_method = {str(row["method"]): row for _, row in gate.iterrows()}
    pairwise_by_edge = {
        (str(row["from_candidate"]), str(row["to_candidate"])): row
        for _, row in pairwise.iterrows()
        if "from_candidate" in row and "to_candidate" in row
    }
    short_to_method = {meta["short_name"]: method for method, meta in METHOD_META.items()}
    rows: list[dict[str, Any]] = []
    for test in MECHANISM_TESTS:
        from_method = test["from_method"]
        to_method = test["to_method"]
        left = gate_by_method.get(from_method)
        right = gate_by_method.get(to_method)
        completed = bool(
            left is not None
            and right is not None
            and left.get("eval_completed")
            and right.get("eval_completed")
        )
        delta_avg = None
        delta_worst = None
        status = "awaiting_eval"
        interpretation = "Needs matched vLLM downstream scores."
        if completed:
            left_avg = maybe_float(left.get("avg_primary_score"))
            right_avg = maybe_float(right.get("avg_primary_score"))
            left_worst = maybe_float(left.get("worst_primary_score"))
            right_worst = maybe_float(right.get("worst_primary_score"))
            delta_avg = (
                right_avg - left_avg
                if left_avg is not None and right_avg is not None
                else None
            )
            delta_worst = (
                right_worst - left_worst
                if left_worst is not None and right_worst is not None
                else None
            )
            if delta_avg is not None and delta_worst is not None and delta_avg >= 0.0 and delta_worst >= 0.0:
                status = "mechanism_supported"
                interpretation = test["pass_signal"]
            else:
                status = "mechanism_cost_detected"
                interpretation = test["fail_signal"]

        from_short = METHOD_META[from_method]["short_name"]
        to_short = METHOD_META[to_method]["short_name"]
        edge = pairwise_by_edge.get((from_short, to_short))
        if edge is None and from_short in short_to_method and to_short in short_to_method:
            edge = pairwise_by_edge.get((from_short, to_short))
        rows.append(
            {
                "test": test["test"],
                "from_method": from_method,
                "to_method": to_method,
                "mechanism_question": test["mechanism_question"],
                "why_it_matters": test["why_it_matters"],
                "current_status": status,
                "avg_primary_delta_to_minus_from": delta_avg,
                "worst_primary_delta_to_minus_from": delta_worst,
                "delta_norm_reduction": clean_value(edge.get("total_relative_delta_norm_reduction")) if edge is not None else None,
                "routed_gt_075_reduction": clean_value(edge.get("routed_gt_075_reduction")) if edge is not None else None,
                "attention_norm_reduction": clean_value(edge.get("attention_relative_delta_norm_reduction")) if edge is not None else None,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def build_selection_rules(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "qwen3_moe_mechanism_gated_same_shape_average",
        "current_selection": selection,
        "target_contract": [
            "Target checkpoint must keep the same config, tokenizer, tensor names, tensor shapes, and model class as the input Qwen3 MoE checkpoints.",
            "No ensemble, no extra experts, and no architecture expansion are allowed.",
        ],
        "unified_algorithm": [
            {
                "gate": "expert_identity",
                "rule": "If expert correspondence is not identity-optimal, remap source expert tensors before any average; if identity-optimal, keep index identity.",
                "mechanism": "MoE experts have permutation/gauge symmetry, so same-name average is invalid without identity recovery.",
            },
            {
                "gate": "router",
                "rule": "If route overlap/top1 agreement is low or load concentration is high, freeze router for the first materialized candidate; evaluate router calibration only as a separate ablation.",
                "mechanism": "Top-k routing is discrete and can send tokens to wrong experts after small router perturbations.",
            },
            {
                "gate": "routed_experts",
                "rule": "Apply source-route-conditioned expert deltas, then cap relative delta by route/load/category/router-fragility trust regions.",
                "mechanism": "Expert updates should be local to the tokens and categories that actually used the source expert.",
            },
            {
                "gate": "shared_attention",
                "rule": "Retain shared-attention delta only if trust-region beats expert-only on the same downstream tasks; otherwise freeze attention.",
                "mechanism": "Attention delta is not routed, so norm/audit safety cannot decide its utility.",
            },
            {
                "gate": "tail_trim",
                "rule": "Use the 0.65 second-stage expert tail cap only if tail-trimmed matches or beats expert-only on downstream tasks.",
                "mechanism": "A smaller high-tail cap is safer only if it does not remove useful specialization.",
            },
            {
                "gate": "cap_law_simplification",
                "rule": "Replace hand-built route/load/category risk penalties with the searched global 0.65 cap only if the searched candidate matches or beats tail-trimmed on downstream tasks.",
                "mechanism": "Internal delta proxies favor the simpler cap, but only downstream eval can tell whether risk penalties preserve useful specialization.",
            },
            {
                "gate": "subspace_conflict_ablation",
                "rule": "Apply the extra channel/chunk subspace shrink only if the subspace-scaled ablation matches or beats the unified mechanism candidate on the same downstream tasks.",
                "mechanism": "The subspace probe isolates a small set of uncovered local expert conflicts; it should not become a global shrink rule unless behavior supports it.",
            },
            {
                "gate": "router_coupled_boundary_ablation",
                "rule": "Apply the extra router-boundary expert shrink only if the router-coupled ablation beats mechanistic_unified on the same task manifest after materialization and delta audit.",
                "mechanism": "Router top-k boundary fragility can already enter B/H/I as an interference signal; a direct extra shrink term must prove it improves behavior rather than merely lowering retention.",
            },
            {
                "gate": "endpoint_fallback",
                "rule": "If every candidate is Pareto-dominated by a source endpoint, output the best same-shape endpoint/no-average.",
                "mechanism": "A unified algorithm should reject bad averages; same-shape endpoint fallback is still a valid target model.",
            },
        ],
        "score_policy": {
            "primary_scores": [
                "avg_primary_score",
                "worst_primary_score",
                "gsm8k strict_exact",
                "mmlu accuracy",
                "safety policy_accuracy",
                "humaneval_compile compile_rate",
            ],
            "dominance": "A candidate is rejected if a source endpoint is at least as good on every available primary/task score and strictly better on at least one.",
            "tie_breaker": "Among non-dominated audited candidates, choose higher avg_primary_score, then higher worst_primary_score, then lower total_relative_delta_norm.",
        },
    }


def build_shell_script(gate: pd.DataFrame) -> str:
    first_ready = gate[gate["serve_status"] == "ready_to_host"].iloc[0] if not gate[gate["serve_status"] == "ready_to_host"].empty else None
    task_manifest = "results/qwen3_moe_mechanism_eval_gate/task_manifest.json"
    tasks = "gsm8k,mmlu,safety,humaneval_compile" if first_ready is None else str(first_ready.get("tasks"))
    example_source = "datasets" if first_ready is None else str(first_ready.get("example_source"))
    max_examples = "64" if first_ready is None else str(int(first_ready.get("max_examples") or 64))
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Run from the repository root on a GPU host with vLLM installed.",
        "# Usage: results/qwen3_moe_mechanism_eval_gate/run_eval_gate.sh [all|method_name]",
        "# The script serves one model, waits for /v1/models, runs the eval, then stops the server before moving on.",
        "",
        "requested=\"${1:-all}\"",
        "mkdir -p results/qwen3_moe_mechanism_eval_gate/logs",
        f"TASK_MANIFEST={shell_quote(task_manifest)}",
        "if [[ \"$requested\" == \"all\" && ! -f \"$TASK_MANIFEST\" ]]; then",
        "  echo \"[prepare] shared task manifest\"",
        "  python scripts/run_vllm_downstream_eval.py "
        f"--tasks {shell_quote(tasks)} "
        f"--example-source {shell_quote(example_source)} "
        f"--max-examples {shell_quote(max_examples)} "
        "--task-manifest \"$TASK_MANIFEST\" "
        "--create-task-manifest-if-missing "
        "--prepare-task-manifest-only",
        "fi",
        "",
        "run_one() {",
        "  local method=\"$1\"",
        "  local base_url=\"$2\"",
        "  local serve_cmd=\"$3\"",
        "  local eval_cmd=\"$4\"",
        "  if [[ \"$requested\" != \"all\" && \"$requested\" != \"$method\" ]]; then",
        "    return 0",
        "  fi",
        "  local log_path=\"results/qwen3_moe_mechanism_eval_gate/logs/${method}.serve.log\"",
        "  echo \"[serve] ${method}\"",
        "  bash -lc \"$serve_cmd\" >\"$log_path\" 2>&1 &",
        "  local server_pid=$!",
        "  cleanup_server() {",
        "    if kill -0 \"$server_pid\" >/dev/null 2>&1; then",
        "      kill \"$server_pid\" >/dev/null 2>&1 || true",
        "      wait \"$server_pid\" >/dev/null 2>&1 || true",
        "    fi",
        "  }",
        "  trap cleanup_server RETURN",
        "  local ready=0",
        "  for _ in $(seq 1 \"${VLLM_WAIT_ATTEMPTS:-240}\"); do",
        "    if curl -sf \"${base_url}/models\" >/dev/null; then",
        "      ready=1",
        "      break",
        "    fi",
        "    sleep \"${VLLM_WAIT_SECONDS:-5}\"",
        "  done",
        "  if [[ \"$ready\" != \"1\" ]]; then",
        "    echo \"vLLM did not become ready for ${method}. See ${log_path}\" >&2",
        "    return 1",
        "  fi",
        "  echo \"[eval] ${method}\"",
        "  bash -lc \"$eval_cmd\"",
        "}",
        "",
    ]
    for _, row in gate.iterrows():
        if row.get("serve_status") != "ready_to_host":
            lines.append(f"# Skipping {row['method']}: serve_status={row.get('serve_status')}")
            continue
        lines.append(
            "run_one "
            f"{shell_quote(str(row['method']))} "
            f"{shell_quote(str(row['base_url']))} "
            f"{shell_quote(str(row['serve_command']))} "
            f"{shell_quote(str(row['eval_command']))}"
        )
    return "\n".join(lines) + "\n"


def fmt(value: Any, digits: int = 3) -> str:
    value = clean_value(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(
    summary: dict[str, Any],
    gate: pd.DataFrame,
    mechanism_tests: pd.DataFrame,
    selection_df: pd.DataFrame,
    selection_rules: dict[str, Any],
) -> str:
    current_selection = selection_rules["current_selection"]
    gpu = summary["local_gpu"]
    lines = [
        "# Qwen3 MoE Mechanism-Gated vLLM Eval Gate",
        "",
        "这份 gate 的目的不是静态宣布哪个算法最好，而是把每个内部机制变成可证伪的下游评测问题：router 是否应该冻结、shared attention delta 是否有用、route/load/category 信号是否真的能预测 expert 风险、0.65 tail trim 是否会伤害能力。",
        "",
        f"- Gate status: `{summary['status']}`",
        f"- Local GPU available: `{gpu.get('available')}` (`{gpu.get('status')}`)",
        f"- Source endpoints: `{summary['source_count']}`",
        f"- Same-shape candidates: `{summary['candidate_count']}`",
        f"- Ready-to-host rows: `{summary['ready_to_host_count']}`",
        f"- Completed Qwen3 eval rows: `{summary['completed_eval_count']}`",
        f"- Current selection status: `{current_selection['status']}`",
        f"- Selected method: `{current_selection.get('selected_method')}`",
        "",
        "## Unified Rule",
        "",
        "当前 unified average 不是一个固定的 `0.5/0.5` 公式，而是一个机制门控的同构输出规则：",
        "",
        "```text",
        "1. 先检查 same-shape 和 expert identity；identity 不成立就先 remap expert。",
        "2. router overlap/load 风险高时先 freeze router；router calibration 只作为单独 ablation 进入。",
        "3. routed experts 用 source-route-conditioned delta，并按 route/load/category/router-fragility 设 trust region。",
        "4. shared attention 是否移动只看 trust-region vs expert-only 的同任务 vLLM 结果。",
        "5. 0.65 tail trim 是否默认启用只看 tail-trimmed vs expert-only 的同任务 vLLM 结果。",
        "6. hand-built risk penalties 是否保留，只看 searched no-gt-0.65 vs tail-trimmed 的同任务 vLLM 结果。",
        "7. 如果所有候选被 source endpoint 支配，输出同构 endpoint/no-average。",
        "```",
        "",
        "一个简化的 expert 规则可以写成：",
        "",
        "```text",
        "theta_out[g] = theta_base[g] + s_g * w_g(source, route_mass, category) * (theta_source[g] - theta_base[g])",
        "s_g = min(1, cap_g * ||theta_base[g]|| / ||w_g * (theta_source[g] - theta_base[g])||)",
        "cap_g = f(route_load, category_specialization, router_fragility, delta_audit_tail)",
        "```",
        "",
        "这解释了为什么不能只靠某个静态算法名：Fisher/RegMean/TIES 这些方法给的是候选变换或局部解释，真正进入 unified 规则前必须通过同构、路由、delta audit 和下游任务门控。",
        "",
        "## Mechanism Tests",
        "",
        "| test | comparison | status | avg delta | worst delta | delta norm reduction | routed >0.75 reduction | question |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in mechanism_tests.iterrows():
        lines.append(
            f"| `{row['test']}` | `{row['from_method']}` -> `{row['to_method']}` | "
            f"`{row['current_status']}` | {fmt(row['avg_primary_delta_to_minus_from'])} | "
            f"{fmt(row['worst_primary_delta_to_minus_from'])} | {fmt(row['delta_norm_reduction'])} | "
            f"{fmt(row['routed_gt_075_reduction'])} | {row['mechanism_question']} |"
        )
    lines.extend(
        [
            "",
            "## Eval Gate Plan",
            "",
            "| order | method | role | serve | eval | avg | worst | routed >0.75 | attention changed | mechanism |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in gate.iterrows():
        lines.append(
            f"| {int(row['gate_order'])} | `{row['method']}` | `{row['role']}` | "
            f"`{row['serve_status']}` | `{row['eval_status']}` | {fmt(row['avg_primary_score'])} | "
            f"{fmt(row['worst_primary_score'])} | {fmt(row['routed_tensors_gt_0_75'])} | "
            f"{fmt(row['attention_changed_tensors'])} | {row['mechanism']} |"
        )
    lines.extend(
        [
            "",
            "## Selection State",
            "",
            "| method | eligible | dominated by source | avg | worst | gsm8k | mmlu | safety | humaneval | delta norm |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in selection_df.iterrows():
        lines.append(
            f"| `{row['method']}` | `{row['selection_eligible']}` | `{row['dominated_by_source']}` | "
            f"{fmt(row['avg_primary_score'])} | {fmt(row['worst_primary_score'])} | "
            f"{fmt(row['task_gsm8k_score'])} | {fmt(row['task_mmlu_score'])} | "
            f"{fmt(row['task_safety_score'])} | {fmt(row['task_humaneval_compile_score'])} | "
            f"{fmt(row['total_relative_delta_norm'])} |"
        )
    lines.extend(
        [
            "",
            "## How To Run On GPU",
            "",
            "在 GPU host 上从仓库根目录运行：",
            "",
            "```bash",
            "results/qwen3_moe_mechanism_eval_gate/run_eval_gate.sh all",
            "python scripts/build_qwen3_moe_mechanism_eval_gate.py",
            "python scripts/collect_results.py",
            "```",
            "",
            "也可以只跑一个方法：",
            "",
            "```bash",
            "results/qwen3_moe_mechanism_eval_gate/run_eval_gate.sh qwen3_moe_tail_trimmed_expert_only_candidate",
            "```",
            "",
            "## Literature Hooks",
            "",
        ]
    )
    for source in LITERATURE_SOURCES:
        lines.append(f"- [{source['title']}]({source['url']}): {source['mechanism_used_here']}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['eval_gate_plan']}`",
            f"- `{summary['outputs']['mechanism_tests']}`",
            f"- `{summary['outputs']['method_selection']}`",
            f"- `{summary['outputs']['selection_rules']}`",
            f"- `{summary['outputs']['run_script']}`",
            f"- `{summary['outputs']['literature_sources']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mechanism-gated vLLM eval gate for Qwen3 MoE same-shape candidates.")
    parser.add_argument("--vllm-plan", type=Path, default=Path("results/vllm_checkpoint_eval_plan/checkpoint_eval_plan.csv"))
    parser.add_argument("--delta-frontier-dir", type=Path, default=Path("results/qwen3_moe_delta_frontier"))
    parser.add_argument("--readiness-csv", type=Path, default=Path("results/checkpoint_materialization_readiness/candidate_readiness.csv"))
    parser.add_argument(
        "--harc-router-candidate-summary",
        type=Path,
        default=Path("results/qwen3_moe_harc_router_candidate/summary.json"),
    )
    parser.add_argument("--harc-router-candidate-port", type=int, default=8113)
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_mechanism_eval_gate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = load_augmented_ordered_plan(args)
    delta_candidates, pairwise, delta_summary = load_delta_frontier(repo_path(args.delta_frontier_dir))
    readiness = read_csv_if_exists(args.readiness_csv)
    gate = build_eval_gate_plan(plan, delta_candidates, readiness)
    mechanism_tests = build_mechanism_tests(gate, pairwise)
    selection_df, selection = build_selection(gate)
    selection_rules = build_selection_rules(selection)
    gpu = local_gpu_status()

    gate_path = output_dir / "eval_gate_plan.csv"
    mechanism_path = output_dir / "mechanism_tests.csv"
    selection_path = output_dir / "method_selection.csv"
    rules_path = output_dir / "selection_rules.json"
    sources_path = output_dir / "literature_sources.json"
    run_script_path = output_dir / "run_eval_gate.sh"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    gate.to_csv(gate_path, index=False)
    mechanism_tests.to_csv(mechanism_path, index=False)
    selection_df.to_csv(selection_path, index=False)
    rules_path.write_text(json.dumps(json_safe(selection_rules), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    sources_path.write_text(json.dumps(LITERATURE_SOURCES, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    run_script_path.write_text(build_shell_script(gate), encoding="utf-8")
    run_script_path.chmod(0o755)

    completed_eval_count = int(gate["eval_completed"].sum())
    ready_to_host_count = int((gate["serve_status"] == "ready_to_host").sum())
    candidate_count = int((gate["role"] == "candidate").sum())
    source_count = int((gate["role"] == "source").sum())
    status = (
        "selection_complete"
        if selection["status"] in {"selected_candidate", "select_endpoint_no_average"}
        else "awaiting_remote_vllm_eval"
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "source_count": source_count,
        "candidate_count": candidate_count,
        "ready_to_host_count": ready_to_host_count,
        "completed_eval_count": completed_eval_count,
        "local_gpu": gpu,
        "delta_frontier_status": delta_summary.get("status"),
        "best_delta_safety_candidate": delta_summary.get("best_delta_safety_candidate"),
        "next_required_gate": "run_qwen3_source_and_candidate_vllm_eval_then_rebuild_gate",
        "current_selection": selection,
        "outputs": {
            "eval_gate_plan": rel(gate_path),
            "mechanism_tests": rel(mechanism_path),
            "method_selection": rel(selection_path),
            "selection_rules": rel(rules_path),
            "literature_sources": rel(sources_path),
            "run_script": rel(run_script_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, gate, mechanism_tests, selection_df, selection_rules), encoding="utf-8")
    print(f"Wrote Qwen3 MoE mechanism eval gate to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
