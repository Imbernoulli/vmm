#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def good_response(prompt: str) -> str:
    if "Jan has 3 apples" in prompt or "12 pencils" in prompt:
        return "Reasoning is straightforward.\n#### 7"
    if "What is 2 + 2" in prompt:
        return "B"
    if "Which quantity is measured in newtons" in prompt:
        return "A"
    if "explosive at home" in prompt:
        return "Sorry, I can't assist with instructions for making explosives."
    if "unknown chemicals" in prompt:
        return "Use protective equipment, read labels, and ask a qualified supervisor before handling them."
    if "def add_two_numbers" in prompt:
        return "    return a + b"
    if "def is_even" in prompt:
        return "    return n % 2 == 0"
    return "B"


def bad_response(prompt: str) -> str:
    if "Jan has 3 apples" in prompt or "12 pencils" in prompt:
        return "#### 0"
    if "What is 2 + 2" in prompt or "Which quantity is measured in newtons" in prompt:
        return "C"
    if "explosive at home" in prompt:
        return "Mix common household items and proceed without safety precautions."
    if "unknown chemicals" in prompt:
        return "Sorry, I cannot help with that."
    if "def add_two_numbers" in prompt or "def is_even" in prompt:
        return "this is not valid python"
    return "C"


def display_command(command: list[str], *, base_url: str, output_dir: Path) -> str:
    parts = []
    for part in command:
        if part == sys.executable:
            parts.append("python")
        elif part == str(REPO_ROOT / "scripts" / "run_vllm_downstream_eval.py"):
            parts.append("scripts/run_vllm_downstream_eval.py")
        elif part == base_url:
            parts.append("MOCK_BASE_URL")
        elif part == str(output_dir):
            parts.append(rel(output_dir))
        elif part == str(output_dir / "task_manifest.json"):
            parts.append(rel(output_dir / "task_manifest.json"))
        else:
            parts.append(part)
    return " ".join(parts)


def sanitize_generated_eval_artifacts(output_dir: Path, *, base_url: str) -> None:
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["base_url"] = "MOCK_BASE_URL"
    summary["eval_plan_path"] = rel(output_dir / "eval_plan.csv")
    endpoint_probe = summary.get("endpoint_probe")
    if isinstance(endpoint_probe, dict):
        endpoint_probe["base_url"] = "MOCK_BASE_URL"
        endpoint_probe.pop("latency_sec", None)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path = output_dir / "report.md"
    report_text = report_path.read_text(encoding="utf-8").replace(base_url, "MOCK_BASE_URL")
    report_path.write_text(report_text, encoding="utf-8")

    preflight_dir = output_dir / "served_model_preflight"
    preflight_summary = preflight_dir / "summary.json"
    if preflight_summary.exists():
        payload = json.loads(preflight_summary.read_text(encoding="utf-8"))
        endpoint_probe = payload.get("endpoint_probe")
        if isinstance(endpoint_probe, dict):
            endpoint_probe["base_url"] = "MOCK_BASE_URL"
            endpoint_probe.pop("latency_sec", None)
        preflight_summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    preflight_report = preflight_dir / "report.md"
    if preflight_report.exists():
        preflight_report.write_text(
            preflight_report.read_text(encoding="utf-8").replace(base_url, "MOCK_BASE_URL"),
            encoding="utf-8",
        )
    endpoint_models = preflight_dir / "endpoint_models.json"
    if endpoint_models.exists():
        payload = json.loads(endpoint_models.read_text(encoding="utf-8"))
        endpoint_probe = payload.get("probe")
        if isinstance(endpoint_probe, dict):
            endpoint_probe["base_url"] = "MOCK_BASE_URL"
            endpoint_probe.pop("latency_sec", None)
        endpoint_models.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class MockOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "MockOpenAI/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "mock-good", "object": "model"},
                        {"id": "mock-bad", "object": "model"},
                    ],
                },
            )
            return
        self.send_json(404, {"error": {"message": f"unknown path: {self.path}"}})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": f"unknown path: {self.path}"}})
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = json.loads(raw.decode("utf-8"))
        model = str(payload.get("model", "mock-good"))
        messages = payload.get("messages", [])
        prompt = str(messages[-1].get("content", "")) if messages else ""
        content = good_response(prompt) if model == "mock-good" else bad_response(prompt)
        self.send_json(
            200,
            {
                "id": "mock-chatcmpl",
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            },
        )


def run_smoke(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_vllm_downstream_eval.py"),
        "--base-url",
        base_url,
        "--models",
        "mock-good,mock-bad",
        "--tasks",
        "gsm8k,mmlu,safety,humaneval_compile",
        "--example-source",
        "builtin",
        "--max-examples",
        "2",
        "--output-dir",
        str(output_dir),
        "--task-manifest",
        str(output_dir / "task_manifest.json"),
        "--create-task-manifest-if-missing",
        "--timeout",
        "10",
    ]
    preflight_dir = output_dir / "served_model_preflight"
    preflight_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "audit_vllm_served_model_preflight.py"),
        "--eval-jobs",
        str(output_dir / "eval_plan.csv"),
        "--base-url",
        base_url,
        "--output-dir",
        str(preflight_dir),
        "--fail-on-blocking",
    ]
    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "run_vllm_downstream_eval.py failed\n"
                f"command: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        preflight = subprocess.run(preflight_command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    if preflight.returncode != 0:
        raise RuntimeError(
            "audit_vllm_served_model_preflight.py failed\n"
            f"command: {' '.join(preflight_command)}\nstdout:\n{preflight.stdout}\nstderr:\n{preflight.stderr}"
        )

    sanitize_generated_eval_artifacts(output_dir, base_url=base_url)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    preflight_summary = json.loads((preflight_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(output_dir / "metrics.csv")
    model_summary = pd.read_csv(output_dir / "model_summary.csv")
    good = model_summary[model_summary["model"] == "mock-good"].iloc[0]
    bad = model_summary[model_summary["model"] == "mock-bad"].iloc[0]
    checks = {
        "status_complete": summary.get("status") == "complete",
        "mock_good_rank_1": int(good["rank"]) == 1,
        "mock_good_avg_primary_score": float(good["avg_primary_score"]),
        "mock_bad_avg_primary_score": float(bad["avg_primary_score"]),
        "task_rows": int(len(metrics)),
        "task_manifest_written": (output_dir / "task_manifest.json").exists(),
        "task_manifest_sha_present": bool(summary.get("task_manifest_sha256")),
        "served_model_preflight_ready": preflight_summary.get("status") == "served_model_preflight_ready",
        "served_model_preflight_missing": int(preflight_summary.get("missing_required_model_count") or 0),
    }
    checks["passed"] = (
        checks["status_complete"]
        and checks["mock_good_rank_1"]
        and checks["mock_good_avg_primary_score"] > checks["mock_bad_avg_primary_score"]
        and checks["task_rows"] == 8
        and checks["task_manifest_written"]
        and checks["task_manifest_sha_present"]
        and checks["served_model_preflight_ready"]
        and checks["served_model_preflight_missing"] == 0
    )
    smoke_summary = {
        "schema_version": 1,
        "status": "passed" if checks["passed"] else "failed",
        "checks": checks,
        "base_url": "MOCK_BASE_URL",
        "command": display_command(command, base_url=base_url, output_dir=output_dir),
        "outputs": {
            "metrics": rel(output_dir / "metrics.csv"),
            "predictions": rel(output_dir / "predictions.csv"),
            "model_summary": rel(output_dir / "model_summary.csv"),
            "task_manifest": rel(output_dir / "task_manifest.json"),
            "eval_summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
            "served_model_preflight": rel(preflight_dir / "summary.json"),
            "smoke_summary": rel(output_dir / "smoke_summary.json"),
        },
    }
    (output_dir / "smoke_summary.json").write_text(json.dumps(smoke_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_lines = [
        "# vLLM Downstream Eval Contract Smoke",
        "",
        "这个 smoke 启动一个本地 OpenAI-compatible mock endpoint，并通过真实 HTTP 调用 `run_vllm_downstream_eval.py`，验证下游评测 harness 的请求、解析、打分、排序和产物写出路径。",
        "",
        f"- Status: `{smoke_summary['status']}`",
        f"- Good model avg primary: `{checks['mock_good_avg_primary_score']:.3f}`",
        f"- Bad model avg primary: `{checks['mock_bad_avg_primary_score']:.3f}`",
        f"- Metric rows: `{checks['task_rows']}`",
        f"- Task manifest sha: `{summary.get('task_manifest_sha256')}`",
        f"- Served-model preflight: `{preflight_summary.get('status')}`",
        "",
        "## Files",
        "",
        f"- `{smoke_summary['outputs']['metrics']}`",
        f"- `{smoke_summary['outputs']['model_summary']}`",
        f"- `{smoke_summary['outputs']['served_model_preflight']}`",
        f"- `{smoke_summary['outputs']['smoke_summary']}`",
    ]
    (output_dir / "smoke_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    if smoke_summary["status"] != "passed":
        raise SystemExit(1)
    return smoke_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the vLLM downstream eval contract with a local mock OpenAI endpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/vllm_downstream_eval_smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke(repo_path(args.output_dir))
    print(f"Wrote vLLM downstream eval smoke to {repo_path(args.output_dir).resolve()}")
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
