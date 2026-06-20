#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


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


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def command_text(command: list[str]) -> str:
    return " ".join(command)


def run_step(step: dict[str, Any], *, plan_only: bool) -> dict[str, Any]:
    started = time.time()
    result = {
        "step": step["step"],
        "kind": step["kind"],
        "command": command_text(step["command"]),
        "status": "planned" if plan_only else "running",
        "returncode": None,
        "duration_sec": 0.0,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if plan_only:
        return result
    completed = subprocess.run(
        step["command"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    result.update(
        {
            "status": "passed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "duration_sec": time.time() - started,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    )
    return result


def build_steps(args: argparse.Namespace) -> list[dict[str, Any]]:
    py = "python"
    steps = [
        {
            "step": "audit_eval_bundles",
            "kind": "gate",
            "command": [
                py,
                "scripts/audit_qwen3_moe_eval_bundle.py",
                "--gate-dir",
                str(args.gate_dir),
                "--output-dir",
                str(args.audit_dir),
            ],
        },
        {
            "step": "select_unified_result",
            "kind": "selector",
            "command": [
                py,
                "scripts/select_qwen3_moe_unified_result.py",
                "--gate-dir",
                str(args.gate_dir),
                "--output-dir",
                str(args.selection_dir),
            ],
        },
        {
            "step": "select_final_candidate",
            "kind": "selector",
            "command": [
                py,
                "scripts/select_qwen3_moe_final_candidate.py",
                "--gate-dir",
                str(args.gate_dir),
                "--audit-dir",
                str(args.audit_dir),
                "--output-dir",
                str(args.final_selection_dir),
            ],
        },
        {
            "step": "attribute_mechanism_effects",
            "kind": "attribution",
            "command": [
                py,
                "scripts/attribute_qwen3_moe_mechanism_effects.py",
                "--gate-dir",
                str(args.gate_dir),
                "--audit-dir",
                str(args.audit_dir),
                "--output-dir",
                str(args.attribution_dir),
            ],
        },
        {
            "step": "build_feedback_optimizer",
            "kind": "optimizer",
            "command": [
                py,
                "scripts/build_qwen3_moe_feedback_optimizer.py",
                "--gate-dir",
                str(args.gate_dir),
                "--audit-dir",
                str(args.audit_dir),
                "--output-dir",
                str(args.feedback_dir),
            ],
        },
        {
            "step": "build_mechanistic_unified_candidate",
            "kind": "optimizer",
            "command": [
                py,
                "scripts/build_qwen3_moe_mechanistic_unified_candidate.py",
                "--output-dir",
                str(args.mechanistic_dir),
            ],
        },
        {
            "step": "audit_mechanistic_evidence",
            "kind": "attribution",
            "command": [
                py,
                "scripts/audit_qwen3_moe_mechanistic_evidence.py",
                "--output-dir",
                str(args.mechanistic_evidence_dir),
            ],
        },
        {
            "step": "build_unified_average_optimizer",
            "kind": "optimizer",
            "command": [
                py,
                "scripts/build_unified_average_optimizer.py",
                "--output-dir",
                str(args.unified_optimizer_dir),
            ],
        },
    ]
    if args.include_smoke:
        steps.extend(
            [
                {
                    "step": "audit_eval_bundles_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/audit_qwen3_moe_eval_bundle.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.audit_smoke_dir),
                    ],
                },
                {
                    "step": "select_unified_result_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/select_qwen3_moe_unified_result.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.selection_smoke_dir),
                    ],
                },
                {
                    "step": "select_final_candidate_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/select_qwen3_moe_final_candidate.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.final_selection_smoke_dir),
                    ],
                },
                {
                    "step": "attribute_mechanism_effects_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/attribute_qwen3_moe_mechanism_effects.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.attribution_smoke_dir),
                    ],
                },
                {
                    "step": "build_feedback_optimizer_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/build_qwen3_moe_feedback_optimizer.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.feedback_smoke_dir),
                    ],
                },
                {
                    "step": "build_mechanistic_unified_candidate_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/build_qwen3_moe_mechanistic_unified_candidate.py",
                        "--smoke-matrix",
                        "--output-dir",
                        str(args.mechanistic_smoke_dir),
                    ],
                },
                {
                    "step": "unified_average_optimizer_ledger_smoke",
                    "kind": "smoke",
                    "command": [
                        py,
                        "scripts/smoke_unified_average_optimizer_ledger.py",
                        "--summary",
                        str(args.unified_optimizer_dir / "summary.json"),
                        "--output-dir",
                        str(args.unified_optimizer_smoke_dir),
                    ],
                },
            ]
        )
    if not args.skip_collect:
        steps.append(
            {
                "step": "collect_results",
                "kind": "summary",
                "command": [py, "scripts/collect_results.py"],
            }
        )
    return steps


def downstream_status(args: argparse.Namespace) -> dict[str, Any]:
    audit = read_json(repo_path(args.audit_dir) / "summary.json")
    selection = read_json(repo_path(args.selection_dir) / "summary.json")
    final_selection = read_json(repo_path(args.final_selection_dir) / "summary.json")
    attribution = read_json(repo_path(args.attribution_dir) / "summary.json")
    feedback = read_json(repo_path(args.feedback_dir) / "summary.json")
    mechanistic = read_json(repo_path(args.mechanistic_dir) / "summary.json")
    mechanistic_evidence = read_json(repo_path(args.mechanistic_evidence_dir) / "summary.json")
    unified_optimizer = read_json(repo_path(args.unified_optimizer_dir) / "summary.json")
    unified_optimizer_smoke = read_json(repo_path(args.unified_optimizer_smoke_dir) / "summary.json")
    final_current = final_selection.get("current_selection") or {}
    optimizer_moe = unified_optimizer.get("moe") or {}
    optimizer_top = unified_optimizer.get("top_next_experiment") or {}
    return {
        "audit_status": audit.get("status"),
        "audit_usable_for_selection": audit.get("usable_for_selection_count"),
        "audit_method_count": audit.get("method_count"),
        "selection_status": selection.get("status"),
        "selected_method": (selection.get("current_selection") or {}).get("selected_method"),
        "selection_reason": (selection.get("current_selection") or {}).get("reason"),
        "final_selection_status": final_selection.get("status"),
        "final_selected_method": final_current.get("selected_method"),
        "final_selection_reason": final_current.get("reason"),
        "final_eligible_candidate_count": final_current.get("eligible_candidate_count"),
        "final_candidate_count": final_current.get("candidate_count"),
        "attribution_status": attribution.get("status"),
        "attribution_scored_transition_count": attribution.get("scored_transition_count"),
        "attribution_transition_count": attribution.get("transition_count"),
        "feedback_status": feedback.get("status"),
        "feedback_scored_task_count": feedback.get("scored_task_count"),
        "feedback_task_count": feedback.get("task_count"),
        "feedback_regression_task_count": feedback.get("regression_task_count"),
        "feedback_changed_group_count": feedback.get("changed_group_count"),
        "mechanistic_status": mechanistic.get("status"),
        "mechanistic_selected_candidate": mechanistic.get("selected_candidate_id"),
        "mechanistic_retention": mechanistic.get("selected_nonbase_mass_retention"),
        "mechanistic_hard_cap_violations": mechanistic.get("selected_hard_cap_violation_count"),
        "mechanistic_evidence_status": mechanistic_evidence.get("status"),
        "mechanistic_evidence_gradient_agreement": mechanistic_evidence.get("gradient_sign_agreement_rate"),
        "mechanistic_evidence_objective_improved_fraction": mechanistic_evidence.get(
            "objective_proxy_improved_group_fraction"
        ),
        "mechanistic_evidence_hard_cap_bound_group_count": mechanistic_evidence.get(
            "hard_cap_bound_group_count"
        ),
        "unified_optimizer_status": unified_optimizer.get("status"),
        "unified_optimizer_top_experiment": optimizer_top.get("experiment"),
        "unified_optimizer_top_experiment_status": optimizer_top.get("status"),
        "unified_optimizer_final_confidence_tie_band": optimizer_moe.get("qwen3_final_confidence_tie_band"),
        "unified_optimizer_final_rank_mode": optimizer_moe.get("qwen3_final_selection_rank_mode"),
        "unified_optimizer_final_rank_band_size": optimizer_moe.get("qwen3_final_selection_rank_band_size"),
        "unified_optimizer_smoke_status": unified_optimizer_smoke.get("status"),
        "unified_optimizer_smoke_passed": unified_optimizer_smoke.get("passed_case_count"),
        "unified_optimizer_smoke_cases": unified_optimizer_smoke.get("case_count"),
    }


def build_report(summary: dict[str, Any]) -> str:
    downstream = summary.get("downstream") or {}
    lines = [
        "# Qwen3 MoE Post-Eval Refresh",
        "",
        "这个脚本在远端 vLLM eval 落盘后按固定顺序刷新 eval bundle audit、unified/final selector、mechanism attribution、feedback/mechanistic optimizer、unified average optimizer 和总汇总，避免手工漏跑或用到旧结果。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Plan only: `{summary['plan_only']}`",
        f"- Steps passed: `{summary['passed_step_count']}/{summary['step_count']}`",
        f"- Audit: `{downstream.get('audit_status', 'n/a')}` (`{downstream.get('audit_usable_for_selection', 'n/a')}/{downstream.get('audit_method_count', 'n/a')}` usable)",
        f"- Selection: `{downstream.get('selection_status', 'n/a')}` -> `{downstream.get('selected_method', 'n/a')}`",
        f"- Final selection: `{downstream.get('final_selection_status', 'n/a')}` -> `{downstream.get('final_selected_method', 'n/a')}` (`{downstream.get('final_eligible_candidate_count', 'n/a')}/{downstream.get('final_candidate_count', 'n/a')}` eligible)",
        f"- Attribution: `{downstream.get('attribution_status', 'n/a')}` (`{downstream.get('attribution_scored_transition_count', 'n/a')}/{downstream.get('attribution_transition_count', 'n/a')}` scored)",
        f"- Feedback optimizer: `{downstream.get('feedback_status', 'n/a')}` (`{downstream.get('feedback_scored_task_count', 'n/a')}/{downstream.get('feedback_task_count', 'n/a')}` scored, `{downstream.get('feedback_changed_group_count', 'n/a')}` changed groups)",
        f"- Mechanistic unified: `{downstream.get('mechanistic_status', 'n/a')}` -> `{downstream.get('mechanistic_selected_candidate', 'n/a')}` (`retention={downstream.get('mechanistic_retention', 'n/a')}`, `violations={downstream.get('mechanistic_hard_cap_violations', 'n/a')}`)",
        f"- Mechanistic evidence: `{downstream.get('mechanistic_evidence_status', 'n/a')}` (`gradient_agreement={downstream.get('mechanistic_evidence_gradient_agreement', 'n/a')}`, `objective_improved={downstream.get('mechanistic_evidence_objective_improved_fraction', 'n/a')}`)",
        f"- Unified average optimizer: `{downstream.get('unified_optimizer_status', 'n/a')}` (top next experiment `{downstream.get('unified_optimizer_top_experiment', 'n/a')}` / `{downstream.get('unified_optimizer_top_experiment_status', 'n/a')}`)",
        f"- Unified selector rank gate in optimizer: confidence band `{downstream.get('unified_optimizer_final_confidence_tie_band', 'n/a')}`, rank mode `{downstream.get('unified_optimizer_final_rank_mode', 'n/a')}`, band size `{downstream.get('unified_optimizer_final_rank_band_size', 'n/a')}`",
        f"- Unified optimizer ledger smoke: `{downstream.get('unified_optimizer_smoke_status', 'n/a')}` (`{downstream.get('unified_optimizer_smoke_passed', 'n/a')}/{downstream.get('unified_optimizer_smoke_cases', 'n/a')}` cases)",
        "",
        "| step | kind | status | returncode | seconds |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in summary["steps"]:
        lines.append(
            f"| `{row['step']}` | `{row['kind']}` | `{row['status']}` | "
            f"{row['returncode']} | {float(row['duration_sec']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
        ]
    )
    for row in summary["steps"]:
        lines.append(f"- `{row['command']}`")
    return "\n".join(lines) + "\n"


def write_outputs(output_dir: Path, step_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = [row for row in step_rows if row["status"] == "failed"]
    passed = [row for row in step_rows if row["status"] == "passed"]
    planned = [row for row in step_rows if row["status"] == "planned"]
    status = "planned" if args.plan_only else ("failed" if failed else "passed")
    summary = {
        "schema_version": 1,
        "status": status,
        "plan_only": bool(args.plan_only),
        "step_count": int(len(step_rows)),
        "passed_step_count": int(len(passed)),
        "planned_step_count": int(len(planned)),
        "failed_step_count": int(len(failed)),
        "downstream": downstream_status(args) if not args.plan_only else {},
        "steps": step_rows,
        "outputs": {
            "steps": rel(output_dir / "steps.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    pd.DataFrame(step_rows).to_csv(output_dir / "steps.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Qwen3 MoE post-vLLM eval gates in the correct order.")
    parser.add_argument("--gate-dir", type=Path, default=Path("results/qwen3_moe_mechanism_eval_gate"))
    parser.add_argument("--audit-dir", type=Path, default=Path("results/qwen3_moe_eval_bundle_audit"))
    parser.add_argument("--selection-dir", type=Path, default=Path("results/qwen3_moe_unified_result_selection"))
    parser.add_argument("--final-selection-dir", type=Path, default=Path("results/qwen3_moe_final_candidate_selection"))
    parser.add_argument("--attribution-dir", type=Path, default=Path("results/qwen3_moe_mechanism_effect_attribution"))
    parser.add_argument("--feedback-dir", type=Path, default=Path("results/qwen3_moe_feedback_optimizer"))
    parser.add_argument(
        "--mechanistic-dir",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_unified_candidate"),
    )
    parser.add_argument(
        "--mechanistic-evidence-dir",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_evidence_audit"),
    )
    parser.add_argument(
        "--unified-optimizer-dir",
        type=Path,
        default=Path("results/unified_average_optimizer"),
    )
    parser.add_argument("--audit-smoke-dir", type=Path, default=Path("results/qwen3_moe_eval_bundle_audit_smoke"))
    parser.add_argument("--selection-smoke-dir", type=Path, default=Path("results/qwen3_moe_unified_result_selection_smoke"))
    parser.add_argument(
        "--final-selection-smoke-dir",
        type=Path,
        default=Path("results/qwen3_moe_final_candidate_selection_smoke"),
    )
    parser.add_argument(
        "--attribution-smoke-dir",
        type=Path,
        default=Path("results/qwen3_moe_mechanism_effect_attribution_smoke"),
    )
    parser.add_argument(
        "--feedback-smoke-dir",
        type=Path,
        default=Path("results/qwen3_moe_feedback_optimizer_smoke"),
    )
    parser.add_argument(
        "--mechanistic-smoke-dir",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_unified_candidate_smoke"),
    )
    parser.add_argument(
        "--unified-optimizer-smoke-dir",
        type=Path,
        default=Path("results/unified_average_optimizer_ledger_smoke"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_post_eval_refresh"))
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = build_steps(args)
    rows = []
    for step in steps:
        result = run_step(step, plan_only=args.plan_only)
        rows.append(result)
        if result["status"] == "failed":
            break
    summary = write_outputs(args.output_dir, rows, args)
    print(f"Wrote Qwen3 MoE post-eval refresh to {repo_path(args.output_dir).resolve()}")
    print(f"Status: {summary['status']}; steps={summary['passed_step_count']}/{summary['step_count']}")
    if summary["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
