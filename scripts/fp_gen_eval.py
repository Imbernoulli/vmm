"""First-principles generation eval for in-memory same-shape merges.

This is a small exact-answer generation smoke test, not a replacement for the
larger vLLM benchmark harness.  It exists to check whether an NLL-selected merge
also behaves sensibly when asked to generate short answers.

The tasks are built in so the script is reproducible without dataset downloads:

  - math_exact: GSM-style arithmetic, answer with one final number
  - code_output_exact: predict the printed output of small Python snippets

The code task does not execute model-generated code.  It checks the final text
answer only, so the harness is safe to run in ordinary CI/local shells.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fp_merge_compare import apply_merge, load_cpu_params, set_endpoint  # noqa: E402


MATH_TASKS = [
    {
        "id": "math_01",
        "prompt": "Question: Mina has 18 apples. She gives 5 to a friend and buys 7 more. Answer with only the final number.\nAnswer:",
        "answer": "20",
    },
    {
        "id": "math_02",
        "prompt": "Question: A box has 6 rows of 4 markers. Three markers are missing. Answer with only the final number.\nAnswer:",
        "answer": "21",
    },
    {
        "id": "math_03",
        "prompt": "Question: Tom reads 12 pages on Monday and twice as many on Tuesday. How many pages did he read total? Answer with only the final number.\nAnswer:",
        "answer": "36",
    },
    {
        "id": "math_04",
        "prompt": "Question: A taxi costs 8 dollars plus 3 dollars per mile. What is the cost for 5 miles? Answer with only the final number.\nAnswer:",
        "answer": "23",
    },
]

CODE_OUTPUT_TASKS = [
    {
        "id": "code_01",
        "prompt": "What does this Python code print? Answer with only the printed value.\n\nx = 3\nprint(x * x + 1)\n\nAnswer:",
        "answer": "10",
    },
    {
        "id": "code_02",
        "prompt": "What does this Python code print? Answer with only the printed value.\n\nitems = ['a', 'bb', 'ccc']\nprint(len(items[-1]))\n\nAnswer:",
        "answer": "3",
    },
    {
        "id": "code_03",
        "prompt": "What does this Python code print? Answer with only the printed value.\n\nvalue = 0\nfor n in [2, 4, 6]:\n    value += n\nprint(value)\n\nAnswer:",
        "answer": "12",
    },
    {
        "id": "code_04",
        "prompt": "What does this Python code print? Answer with only the printed value.\n\ndef f(x):\n    return x + 2\nprint(f(5))\n\nAnswer:",
        "answer": "7",
    },
]


def log(message: str) -> None:
    print(message, flush=True)


def cycle(items: list[dict[str, str]], n: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    while len(out) < n:
        out.extend(items)
    return out[:n]


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = text.splitlines()[0] if text else ""
    text = text.strip().strip("`").strip()
    if "####" in text:
        text = text.split("####")[-1]
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[0].replace(",", "").rstrip(".")
    return re.sub(r"\s+", " ", text.lower()).strip(" .")


@torch.no_grad()
def generate_answer(model, tok, device: str, prompt: str, max_new_tokens: int) -> str:
    inputs = tok(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def score_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


@torch.no_grad()
def eval_tasks(model, tok, device: str, *, n_math: int, n_code: int, max_new_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_sets = [
        ("math_exact", cycle(MATH_TASKS, n_math)),
        ("code_output_exact", cycle(CODE_OUTPUT_TASKS, n_code)),
    ]
    for task_name, tasks in task_sets:
        for task in tasks:
            raw = generate_answer(model, tok, device, task["prompt"], max_new_tokens=max_new_tokens)
            pred = normalize_answer(raw)
            gold = normalize_answer(task["answer"])
            rows.append(
                {
                    "task": task_name,
                    "id": task["id"],
                    "gold": gold,
                    "prediction": pred,
                    "raw_generation": raw.replace("\n", "\\n")[:240],
                    "correct": pred == gold,
                }
            )
    by_task: dict[str, Any] = {}
    for task_name in ("math_exact", "code_output_exact"):
        by_task[task_name] = score_rows([row for row in rows if row["task"] == task_name])
    worst = min(metrics["accuracy"] for metrics in by_task.values()) if by_task else 0.0
    avg = sum(metrics["accuracy"] for metrics in by_task.values()) / max(len(by_task), 1)
    return rows, {"by_task": by_task, "avg_accuracy": avg, "worst_accuracy": worst}


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    valid = {"base", "instruct", "coder", "linear", "unified"}
    bad = sorted(set(methods) - valid)
    if bad:
        raise ValueError(f"Unknown methods {bad}; valid methods are {sorted(valid)}")
    return methods


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Generation Exact-Answer Merge Eval",
        "",
        "这个实验是一个小型生成式 smoke test：不用外部 dataset，也不执行模型生成的代码；只检查数学题和代码输出题的最终答案。",
        "",
        "## Result",
        "",
        "| method | math | code_output | avg | worst |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method, metrics in summary["results"].items():
        math_acc = metrics["by_task"]["math_exact"]["accuracy"]
        code_acc = metrics["by_task"]["code_output_exact"]["accuracy"]
        lines.append(
            f"| `{method}` | {math_acc:.3f} | {code_acc:.3f} | {metrics['avg_accuracy']:.3f} | {metrics['worst_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
            "## Files",
            "",
            "- `summary.json`",
            "- `predictions.csv`",
            "- `method_metrics.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--coder", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-math", type=int, default=4)
    ap.add_argument("--n-code", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--unified-config", default=None, help="JSON dict for the unified merge config")
    ap.add_argument("--methods", default="base,instruct,coder,linear,unified")
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.instruct, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log(f"loading Instruct ({device}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.instruct,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()
    A = OrderedDict((name, param.detach().cpu().clone()) for name, param in model.named_parameters())
    log("loading Coder (CPU) ...")
    B = load_cpu_params(args.coder)
    log("loading Base (CPU) ...")
    base = load_cpu_params(args.base)

    unified_cfg = json.loads(args.unified_config) if args.unified_config else {
        "density": 1.0,
        "lam": 0.0,
        "sign_resolve": False,
        "importance": "uniform",
        "router": "average",
    }
    linear_cfg = {"density": 1.0, "lam": 1.0, "sign_resolve": False, "importance": "uniform", "router": "average"}

    def set_model(method: str) -> None:
        if method == "base":
            set_endpoint(model, base)
        elif method == "instruct":
            set_endpoint(model, A)
        elif method == "coder":
            set_endpoint(model, B)
        elif method == "linear":
            apply_merge(model, base, A, B, **linear_cfg)
        elif method == "unified":
            apply_merge(model, base, A, B, **unified_cfg)
        else:
            raise ValueError(method)

    results: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for method in parse_methods(args.methods):
        log(f"\n=== {method} ===")
        set_model(method)
        rows, metrics = eval_tasks(
            model,
            tok,
            device,
            n_math=args.n_math,
            n_code=args.n_code,
            max_new_tokens=args.max_new_tokens,
        )
        for row in rows:
            row["method"] = method
            prediction_rows.append(row)
        results[method] = metrics
        log(
            f"  math={metrics['by_task']['math_exact']['accuracy']:.3f} "
            f"code={metrics['by_task']['code_output_exact']['accuracy']:.3f} "
            f"avg={metrics['avg_accuracy']:.3f} worst={metrics['worst_accuracy']:.3f}"
        )

    method_metrics = [
        {
            "method": method,
            "math_exact": metrics["by_task"]["math_exact"]["accuracy"],
            "code_output_exact": metrics["by_task"]["code_output_exact"]["accuracy"],
            "avg_accuracy": metrics["avg_accuracy"],
            "worst_accuracy": metrics["worst_accuracy"],
        }
        for method, metrics in results.items()
    ]
    method_metrics = sorted(method_metrics, key=lambda row: (row["avg_accuracy"], row["worst_accuracy"]), reverse=True)
    best_method = method_metrics[0]["method"] if method_metrics else None
    linear_avg = results.get("linear", {}).get("avg_accuracy")
    unified_avg = results.get("unified", {}).get("avg_accuracy")
    interpretation = (
        "The exact-answer generation smoke follows the same boundary as the NLL selector: "
        "it is a fast behavioral check, not proof that a merge beats all endpoints. "
    )
    if linear_avg is not None and unified_avg is not None:
        if unified_avg > linear_avg:
            interpretation += "The unified candidate beats the linear midpoint on this smoke slice."
        elif unified_avg < linear_avg:
            interpretation += "The linear midpoint beats the unified candidate on this smoke slice, so the selector needs a larger generation gate before materialization."
        else:
            interpretation += "The unified candidate ties the linear midpoint on this smoke slice."

    summary = {
        "schema_version": 1,
        "models": {"instruct": args.instruct, "coder": args.coder, "base": args.base},
        "task_counts": {"math_exact": args.n_math, "code_output_exact": args.n_code},
        "methods": parse_methods(args.methods),
        "unified_config": unified_cfg,
        "best_method": best_method,
        "results": results,
        "method_metrics": method_metrics,
        "interpretation": interpretation,
        "outputs": {
            "summary": "summary.json",
            "predictions": "predictions.csv",
            "method_metrics": "method_metrics.csv",
            "report": "report.md",
        },
    }
    pd.DataFrame(prediction_rows).to_csv(outdir / "predictions.csv", index=False)
    pd.DataFrame(method_metrics).to_csv(outdir / "method_metrics.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (outdir / "report.md").write_text(build_report(summary), encoding="utf-8")
    log(f"\nwrote {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
