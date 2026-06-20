#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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


def bool_value(value: Any) -> bool:
    value = clean_value(value)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def method_set(table: pd.DataFrame, mask: pd.Series) -> set[str]:
    if table.empty:
        return set()
    return {str(method) for method in table[mask]["method"].tolist()}


def queue_rows(method_budget: pd.DataFrame, queue: str) -> pd.DataFrame:
    if method_budget.empty or "eval_queue" not in method_budget:
        return pd.DataFrame()
    return method_budget[method_budget["eval_queue"].astype(str) == queue]


def build_report(summary: dict[str, Any], matrix: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE Eval Budget Queue Smoke",
        "",
        "This smoke verifies that the budgeted vLLM runner defaults to the final-selection queue, while ablation-only candidates stay out of the default run.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Assertions: `{summary['passed_assertion_count']}/{summary['assertion_count']}`",
        f"- Final queue methods: `{summary['final_queue_method_count']}`",
        f"- Mechanism queue methods: `{summary['mechanism_queue_method_count']}`",
        "",
        "| assertion | expected | actual | passed |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in matrix.iterrows():
        lines.append(
            f"| `{row['assertion']}` | `{row['expected']}` | `{row['actual']}` | `{bool(row['passed'])}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['matrix']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Qwen3 MoE eval budget queue semantics.")
    parser.add_argument("--eval-budget-dir", type=Path, default=Path("results/qwen3_moe_eval_budget_plan"))
    parser.add_argument(
        "--candidate-trust-gate",
        type=Path,
        default=Path("results/qwen3_moe_candidate_trust_region_gate/candidate_trust_region_gate.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen3_moe_eval_budget_queue_smoke"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budget_dir = repo_path(args.eval_budget_dir)
    summary = read_json(budget_dir / "summary.json")
    method_budget = read_csv(budget_dir / "method_budget.csv")
    alignment = read_csv(budget_dir / "task_manifest_alignment.csv")
    trust_gate = read_csv(args.candidate_trust_gate)
    run_script = (budget_dir / "run_eval_budget.sh").read_text(encoding="utf-8")

    sources = method_set(method_budget, method_budget["role"].astype(str) == "source")
    trust_final = method_set(
        trust_gate,
        (trust_gate["role"].astype(str) == "candidate")
        & trust_gate["final_selectable_by_trust_region"].map(bool_value),
    )
    trust_ablation = method_set(
        trust_gate,
        (trust_gate["role"].astype(str) == "candidate")
        & (trust_gate["trust_region_category"].astype(str) == "ablation_only"),
    )
    final_rows = queue_rows(method_budget, "final_selection_core")
    mechanism_rows = queue_rows(method_budget, "mechanism_ablation")
    router_rows = queue_rows(method_budget, "router_calibration_pending")
    final_methods = set(final_rows["method"].astype(str).tolist()) if not final_rows.empty else set()
    mechanism_methods = set(mechanism_rows["method"].astype(str).tolist()) if not mechanism_rows.empty else set()
    final_prompt_sum = int(final_rows["recommended_prompt_budget"].sum()) if not final_rows.empty else 0
    mechanism_prompt_sum = int(mechanism_rows["recommended_prompt_budget"].sum()) if not mechanism_rows.empty else 0
    aligned = int(alignment["task_manifest_aligned"].map(bool_value).sum()) if not alignment.empty else 0

    rows = [
        {
            "assertion": "default_runner_request_is_final",
            "expected": "final",
            "actual": summary.get("default_runner_request"),
            "passed": summary.get("default_runner_request") == "final",
        },
        {
            "assertion": "runner_shell_defaults_to_final",
            "expected": 'requested="${1:-final}"',
            "actual": "present" if 'requested="${1:-final}"' in run_script else "missing",
            "passed": 'requested="${1:-final}"' in run_script,
        },
        {
            "assertion": "runner_preflight_defaults_to_final_scope",
            "expected": 'preflight_scope="${2:-final}"',
            "actual": "present" if 'preflight_scope="${2:-final}"' in run_script else "missing",
            "passed": 'preflight_scope="${2:-final}"' in run_script,
        },
        {
            "assertion": "final_queue_methods_match_sources_plus_trust_gate",
            "expected": ",".join(sorted(sources | trust_final)),
            "actual": ",".join(sorted(final_methods)),
            "passed": final_methods == sources | trust_final,
        },
        {
            "assertion": "mechanism_queue_methods_match_ablation_only_candidates",
            "expected": ",".join(sorted(trust_ablation)),
            "actual": ",".join(sorted(mechanism_methods)),
            "passed": mechanism_methods == trust_ablation,
        },
        {
            "assertion": "summary_final_core_method_count_matches_table",
            "expected": len(final_rows),
            "actual": summary.get("final_core_method_count"),
            "passed": int(summary.get("final_core_method_count", -1)) == len(final_rows),
        },
        {
            "assertion": "summary_final_prompt_budget_matches_table",
            "expected": final_prompt_sum,
            "actual": summary.get("final_core_recommended_prompt_budget"),
            "passed": int(summary.get("final_core_recommended_prompt_budget", -1)) == final_prompt_sum,
        },
        {
            "assertion": "summary_mechanism_count_matches_table",
            "expected": len(mechanism_rows),
            "actual": summary.get("mechanism_ablation_method_count"),
            "passed": int(summary.get("mechanism_ablation_method_count", -1)) == len(mechanism_rows),
        },
        {
            "assertion": "summary_mechanism_prompt_budget_matches_table",
            "expected": mechanism_prompt_sum,
            "actual": summary.get("mechanism_ablation_recommended_prompt_budget"),
            "passed": int(summary.get("mechanism_ablation_recommended_prompt_budget", -1)) == mechanism_prompt_sum,
        },
        {
            "assertion": "router_queue_not_ready_by_default",
            "expected": "no ready router rows",
            "actual": int((router_rows["serve_status"].astype(str) == "ready_to_host").sum()) if not router_rows.empty else 0,
            "passed": bool(router_rows.empty)
            or int((router_rows["serve_status"].astype(str) == "ready_to_host").sum()) == 0,
        },
        {
            "assertion": "task_manifest_alignment_covers_all_budget_rows",
            "expected": len(method_budget),
            "actual": aligned,
            "passed": aligned == len(method_budget) and len(method_budget) > 0,
        },
    ]
    matrix = pd.DataFrame(rows)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "eval_budget_queue_smoke_matrix.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    matrix.to_csv(matrix_path, index=False)
    passed = int(matrix["passed"].astype(bool).sum())
    smoke_summary = {
        "schema_version": 1,
        "status": "passed" if passed == len(matrix) else "failed",
        "assertion_count": int(len(matrix)),
        "passed_assertion_count": passed,
        "failed_assertion_count": int(len(matrix) - passed),
        "final_queue_method_count": int(len(final_rows)),
        "mechanism_queue_method_count": int(len(mechanism_rows)),
        "router_queue_method_count": int(len(router_rows)),
        "final_queue_methods": sorted(final_methods),
        "mechanism_queue_methods": sorted(mechanism_methods),
        "outputs": {
            "matrix": rel(matrix_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    summary_path.write_text(json.dumps(json_safe(smoke_summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(smoke_summary, matrix), encoding="utf-8")
    print(f"Wrote Qwen3 MoE eval budget queue smoke to {output_dir.resolve()}")
    print(f"Status: {smoke_summary['status']}; assertions {passed}/{len(matrix)}")
    if passed != len(matrix):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
