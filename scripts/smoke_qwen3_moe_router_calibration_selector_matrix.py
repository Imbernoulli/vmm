#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_METHOD = "qwen3_moe_searched_no_gt065_max_retention_candidate"


@dataclass(frozen=True)
class SelectorCase:
    name: str
    flag: str
    expected_status: str
    expected_selected_method: str | None
    expected_eligible_count: int
    expected_reason_marker: str


CASES = [
    SelectorCase(
        name="positive_group_heldout",
        flag="--smoke",
        expected_status="selected_router_calibrated_candidate",
        expected_selected_method="smoke_router_calibrated_cap0025",
        expected_eligible_count=1,
        expected_reason_marker="passes_all_gates",
    ),
    SelectorCase(
        name="row_validation_rejected",
        flag="--row-validation-negative-smoke",
        expected_status="awaiting_router_calibration_eval",
        expected_selected_method=None,
        expected_eligible_count=0,
        expected_reason_marker="router_validation_not_group_heldout",
    ),
    SelectorCase(
        name="source_dominance_abstains",
        flag="--source-dominance-negative-smoke",
        expected_status="keep_frozen_router_baseline",
        expected_selected_method=BASELINE_METHOD,
        expected_eligible_count=0,
        expected_reason_marker="source_endpoint_dominates",
    ),
    SelectorCase(
        name="no_downstream_gain_abstains",
        flag="--no-downstream-gain-negative-smoke",
        expected_status="keep_frozen_router_baseline",
        expected_selected_method=BASELINE_METHOD,
        expected_eligible_count=0,
        expected_reason_marker="no_downstream_gain",
    ),
    SelectorCase(
        name="task_regression_abstains",
        flag="--task-regression-negative-smoke",
        expected_status="keep_frozen_router_baseline",
        expected_selected_method=BASELINE_METHOD,
        expected_eligible_count=0,
        expected_reason_marker="task_score_regression",
    ),
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


def run_case(case: SelectorCase, case_output_dir: Path, *, keep_case_outputs: bool) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/select_qwen3_moe_router_calibration_result.py"),
        case.flag,
        "--output-dir",
        str(case_output_dir),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return {
            "case": case.name,
            "flag": case.flag,
            "passed": False,
            "failure": "selector_command_failed",
            "returncode": int(completed.returncode),
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
            "case_output_dir": rel(case_output_dir) if keep_case_outputs else None,
        }

    summary = json.loads((case_output_dir / "summary.json").read_text(encoding="utf-8"))
    table = pd.read_csv(case_output_dir / "selection_table.csv")
    selection = summary.get("current_selection", {})
    status = selection.get("status")
    selected_method = selection.get("selected_method")
    eligible_count = int(selection.get("eligible_candidate_count", -1))
    candidate_count = int(selection.get("candidate_count", len(table)))
    decision_reasons = [str(value) for value in table.get("decision_reason", pd.Series(dtype=str)).fillna("")]
    reason_found = any(case.expected_reason_marker in reason for reason in decision_reasons)
    status_passed = status == case.expected_status
    selected_passed = selected_method == case.expected_selected_method
    eligible_passed = eligible_count == case.expected_eligible_count
    passed = bool(status_passed and selected_passed and eligible_passed and reason_found)
    return {
        "case": case.name,
        "flag": case.flag,
        "status": status,
        "expected_status": case.expected_status,
        "selected_method": selected_method,
        "expected_selected_method": case.expected_selected_method,
        "eligible_candidate_count": eligible_count,
        "expected_eligible_candidate_count": case.expected_eligible_count,
        "candidate_count": candidate_count,
        "expected_reason_marker": case.expected_reason_marker,
        "reason_found": bool(reason_found),
        "status_passed": bool(status_passed),
        "selected_passed": bool(selected_passed),
        "eligible_passed": bool(eligible_passed),
        "passed": passed,
        "case_output_dir": rel(case_output_dir) if keep_case_outputs else None,
        "selection_table": rel(case_output_dir / "selection_table.csv") if keep_case_outputs else None,
        "report": rel(case_output_dir / "report.md") if keep_case_outputs else None,
    }


def build_report(summary: dict[str, Any], matrix: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE Router Calibration Selector Matrix Smoke",
        "",
        "这个 smoke 不是再造一个候选算法，而是把 router-calibration selector 的核心选择/拒绝边界变成一次性回归测试。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Cases: `{summary['passed_case_count']}/{summary['case_count']}`",
        "",
        "## Cases",
        "",
        "| case | status | selected | eligible | reason marker | passed |",
        "|---|---|---|---:|---|---|",
    ]
    for _, row in matrix.iterrows():
        lines.append(
            f"| `{row['case']}` | `{row.get('status')}` | `{row.get('selected_method')}` | "
            f"{int(row.get('eligible_candidate_count', -1))} | `{row.get('expected_reason_marker')}` | "
            f"`{bool(row.get('passed'))}` |"
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


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    if args.keep_case_outputs:
        case_root = output_dir / "case_outputs"
        case_root.mkdir(parents=True, exist_ok=True)
        for case in CASES:
            rows.append(run_case(case, case_root / case.name, keep_case_outputs=True))
    else:
        with tempfile.TemporaryDirectory(prefix="qwen3_router_selector_matrix_") as tmp:
            case_root = Path(tmp)
            for case in CASES:
                rows.append(run_case(case, case_root / case.name, keep_case_outputs=False))

    matrix = pd.DataFrame(rows)
    passed_case_count = int(matrix["passed"].astype(bool).sum()) if not matrix.empty else 0
    case_count = int(len(matrix))
    status = "passed" if passed_case_count == case_count else "failed"

    matrix_path = output_dir / "selector_matrix.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    matrix.to_csv(matrix_path, index=False)
    summary = {
        "schema_version": 1,
        "status": status,
        "case_count": case_count,
        "passed_case_count": passed_case_count,
        "failed_case_count": case_count - passed_case_count,
        "keep_case_outputs": bool(args.keep_case_outputs),
        "cases": matrix.to_dict(orient="records"),
        "outputs": {
            "matrix": rel(matrix_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, matrix), encoding="utf-8")
    print(f"Wrote selector matrix smoke to {output_dir.resolve()}")
    print(f"Status: {status}; cases {passed_case_count}/{case_count}")
    if status != "passed":
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen3 MoE router calibration selector smoke matrix.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen3_moe_router_calibration_selector_matrix_smoke"),
    )
    parser.add_argument(
        "--keep-case-outputs",
        action="store_true",
        help="Keep full per-case selector artifacts under output-dir/case_outputs instead of using a temporary directory.",
    )
    return parser.parse_args()


def main() -> None:
    run_matrix(parse_args())


if __name__ == "__main__":
    main()
