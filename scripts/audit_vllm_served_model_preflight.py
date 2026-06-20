#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
import urllib.request
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


def read_csv(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def split_csv(value: Any) -> list[str]:
    value = clean_value(value)
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def get_json(url: str, api_key: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_endpoint_models(base_url: str, api_key: str, timeout: float) -> tuple[dict[str, Any], list[str]]:
    started = time.time()
    url = f"{base_url.rstrip('/')}/models"
    try:
        payload = get_json(url, api_key, timeout)
        models = []
        for item in payload.get("data") or []:
            model_id = item.get("id") if isinstance(item, dict) else None
            if model_id:
                models.append(str(model_id))
        return (
            {
                "status": "ok",
                "base_url": base_url,
                "latency_sec": time.time() - started,
                "model_count": len(models),
                "raw_response": payload,
            },
            sorted(set(models)),
        )
    except Exception as exc:
        return (
            {
                "status": "unavailable",
                "base_url": base_url,
                "latency_sec": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            [],
        )


def load_endpoint_models(args: argparse.Namespace) -> tuple[dict[str, Any], list[str], bool]:
    if args.endpoint_models_json:
        payload = read_json(args.endpoint_models_json)
        if isinstance(payload, list):
            models = sorted({str(item) for item in payload})
        elif "data" in payload:
            models = sorted(
                {
                    str(item.get("id"))
                    for item in payload.get("data") or []
                    if isinstance(item, dict) and item.get("id")
                }
            )
        else:
            raw = payload.get("models") if isinstance(payload.get("models"), list) else payload
            models = sorted({str(item) for item in raw} if isinstance(raw, list) else set())
        return (
            {
                "status": "loaded_from_file",
                "path": rel(args.endpoint_models_json),
                "model_count": len(models),
            },
            models,
            True,
        )
    if args.base_url:
        probe, models = fetch_endpoint_models(args.base_url, args.api_key, args.timeout)
        return probe, models, probe.get("status") == "ok"
    return (
        {
            "status": "not_requested",
            "message": "Pass --base-url or --endpoint-models-json to verify served model ids.",
        },
        [],
        False,
    )


def required_model_rows(eval_jobs: pd.DataFrame, endpoint_models: set[str], has_endpoint_list: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, job in eval_jobs.iterrows():
        if "served_models" in eval_jobs.columns:
            models = split_csv(job.get("served_models"))
        elif "served_model_id" in eval_jobs.columns:
            models = split_csv(job.get("served_model_id"))
        else:
            models = []
        for model in models:
            rows.append(
                {
                    "job_id": job.get("job_id", job.get("method", f"row_{idx}")),
                    "scenario_id": job.get("scenario_id", ""),
                    "rank": job.get("rank", job.get("eval_order", idx)),
                    "served_model": model,
                    "endpoint_model_list_available": has_endpoint_list,
                    "present_on_endpoint": model in endpoint_models if has_endpoint_list else False,
                    "tasks": job.get("tasks", ""),
                    "task_manifest": job.get("task_manifest", ""),
                    "smoke_task_manifest": job.get("smoke_task_manifest", ""),
                    "output_dir": job.get("output_dir", ""),
                }
            )
    return pd.DataFrame(rows)


def manifest_rows(eval_jobs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, job in eval_jobs.iterrows():
        tasks = split_csv(job.get("tasks"))
        for kind, column in (("production", "task_manifest"), ("smoke", "smoke_task_manifest")):
            manifest = clean_value(job.get(column)) if column in eval_jobs.columns else None
            if not manifest:
                continue
            path = repo_path(manifest)
            payload = read_json(path)
            examples = payload.get("examples") or {}
            missing_tasks = [task for task in tasks if task not in examples]
            counts = {
                str(task): len(rows)
                for task, rows in examples.items()
                if isinstance(rows, list) and (not tasks or task in tasks)
            }
            required = clean_value(job.get("max_examples"))
            try:
                required_count = int(required) if required is not None else 0
            except (TypeError, ValueError):
                required_count = 0
            if kind == "smoke":
                required_count = 0
            insufficient = [
                task
                for task in tasks
                if task in examples and isinstance(examples[task], list) and len(examples[task]) < required_count
            ]
            rows.append(
                {
                    "job_id": job.get("job_id", job.get("method", f"row_{idx}")),
                    "scenario_id": job.get("scenario_id", ""),
                    "manifest_kind": kind,
                    "task_manifest": manifest,
                    "exists": path.exists(),
                    "tasks": ",".join(tasks),
                    "task_count": len(tasks),
                    "missing_tasks": ",".join(missing_tasks),
                    "insufficient_tasks": ",".join(insufficient),
                    "required_examples_per_task": required_count,
                    "manifest_counts": json.dumps(counts, sort_keys=True),
                    "ready": path.exists() and not missing_tasks and not insufficient,
                }
            )
    return pd.DataFrame(rows)


def choose_status(
    *,
    required: pd.DataFrame,
    manifests: pd.DataFrame,
    endpoint_probe: dict[str, Any],
    has_endpoint_list: bool,
) -> str:
    if not has_endpoint_list:
        if endpoint_probe.get("status") == "unavailable":
            return "endpoint_unavailable"
        return "static_preflight_ready_waiting_for_endpoint_model_list"
    missing_model_count = int((~required["present_on_endpoint"].astype(bool)).sum()) if not required.empty else 0
    if missing_model_count:
        return "served_model_preflight_failed_missing_models"
    if not manifests.empty and not manifests["ready"].astype(bool).all():
        return "served_model_preflight_waiting_for_task_manifests"
    return "served_model_preflight_ready"


def build_report(summary: dict[str, Any], required: pd.DataFrame, manifests: pd.DataFrame) -> str:
    lines = [
        "# vLLM Served Model Preflight",
        "",
        "这个 preflight 在真正跑 downstream eval 前检查两件事：计划中的 served model id 是否出现在 vLLM `/models`，以及每个 job 的 task manifest 是否已经准备好并覆盖对应任务。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Endpoint probe: `{summary['endpoint_probe_status']}`",
        f"- Required served models: `{summary['unique_required_model_count']}`",
        f"- Endpoint served models: `{summary['endpoint_model_count']}`",
        f"- Missing required models: `{summary['missing_required_model_count']}`",
        f"- Manifest ready: `{summary['ready_manifest_count']}/{summary['manifest_check_count']}`",
        "",
        "## Missing Models",
        "",
    ]
    missing = required[~required["present_on_endpoint"].astype(bool)] if not required.empty else pd.DataFrame()
    if summary["endpoint_model_list_available"] and not missing.empty:
        lines.extend(["| job | served model | tasks |", "| --- | --- | --- |"])
        for _, row in missing.iterrows():
            lines.append(f"| `{row['job_id']}` | `{row['served_model']}` | `{row['tasks']}` |")
    elif summary["endpoint_model_list_available"]:
        lines.append("No required served models are missing from the endpoint model list.")
    else:
        lines.append("Endpoint model list was not available; pass `--base-url` against the running vLLM server.")
    lines.extend(
        [
            "",
            "## Manifest Checks",
            "",
            "| job | kind | exists | ready | missing tasks | insufficient tasks |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in manifests.iterrows():
        lines.append(
            f"| `{row['job_id']}` | `{row['manifest_kind']}` | `{bool(row['exists'])}` | "
            f"`{bool(row['ready'])}` | `{row['missing_tasks']}` | `{row['insufficient_tasks']}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['required_models']}`",
            f"- `{summary['outputs']['manifest_checks']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_jobs = read_csv(args.eval_jobs)
    endpoint_probe, endpoint_models, has_endpoint_list = load_endpoint_models(args)
    endpoint_model_set = set(endpoint_models)
    required = required_model_rows(eval_jobs, endpoint_model_set, has_endpoint_list)
    manifests = manifest_rows(eval_jobs)

    unique_required = sorted(set(required["served_model"].astype(str))) if not required.empty else []
    missing_required = (
        sorted(set(required.loc[~required["present_on_endpoint"].astype(bool), "served_model"].astype(str)))
        if has_endpoint_list and not required.empty
        else []
    )
    ready_manifest_count = int(manifests["ready"].astype(bool).sum()) if not manifests.empty else 0
    status = choose_status(
        required=required,
        manifests=manifests,
        endpoint_probe=endpoint_probe,
        has_endpoint_list=has_endpoint_list,
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "eval_jobs": rel(args.eval_jobs),
        "endpoint_model_list_available": has_endpoint_list,
        "endpoint_probe_status": endpoint_probe.get("status"),
        "endpoint_model_count": len(endpoint_models),
        "unique_required_model_count": len(unique_required),
        "required_model_row_count": int(len(required)),
        "missing_required_model_count": len(missing_required),
        "missing_required_models": missing_required,
        "manifest_check_count": int(len(manifests)),
        "ready_manifest_count": ready_manifest_count,
        "blocking_reason": (
            "Start the vLLM server and rerun this preflight with --base-url."
            if endpoint_probe.get("status") in {"not_requested", "unavailable"}
            else "Serve every missing model id before running downstream eval."
            if missing_required
            else "Prepare task manifests before running downstream eval."
            if ready_manifest_count < len(manifests)
            else ""
        ),
        "endpoint_probe": endpoint_probe,
        "endpoint_models": endpoint_models,
        "outputs": {
            "required_models": rel(output_dir / "required_served_models.csv"),
            "manifest_checks": rel(output_dir / "task_manifest_checks.csv"),
            "endpoint_models": rel(output_dir / "endpoint_models.json"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    required.to_csv(output_dir / "required_served_models.csv", index=False)
    manifests.to_csv(output_dir / "task_manifest_checks.csv", index=False)
    (output_dir / "endpoint_models.json").write_text(
        json.dumps(json_safe({"probe": endpoint_probe, "models": endpoint_models}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(build_report(summary, required, manifests), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight planned vLLM downstream eval jobs against served model ids and task manifests.")
    parser.add_argument("--eval-jobs", type=Path, default=Path("results/qwen_source_discovery_eval_plan/vllm_eval_jobs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen_source_discovery_served_model_preflight"))
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--endpoint-models-json", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--fail-on-blocking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args)
    print(f"Wrote vLLM served model preflight to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; required={summary['unique_required_model_count']}; "
        f"missing={summary['missing_required_model_count']}; "
        f"manifests={summary['ready_manifest_count']}/{summary['manifest_check_count']}"
    )
    if args.fail_on_blocking and summary["status"] != "served_model_preflight_ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
