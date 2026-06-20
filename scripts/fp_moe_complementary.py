"""Task B: a genuinely complementary MoE pair -> can the merge beat BOTH sources?

Instruct dominated Coder on our probes, so Instruct+Coder can't show a "combine"
gain.  Here we use a reasoning specialist (Qwen3-30B-A3B-Thinking-2507) and a
code specialist (Qwen3-Coder-30B-A3B-Instruct), same base, and two tasks where
each should win one:
  - math      : GSM8K answer (reasoning) NLL  -> Thinking better
  - code      : HumanEval completion NLL       -> Coder better

If the smooth-barrier MoE merge inherits both strengths, its AVERAGE NLL over the
two tasks beats each source's average (model-soup multitask win) even though it
loses to the stronger source on that source's home task.  We also scan the
interpolation to find the best multitask point.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def log(m):
    print(m, flush=True)


def load_cpu(mid):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
    sd = OrderedDict((n, p.detach().cpu().clone()) for n, p in m.named_parameters())
    del m
    return sd


def build_math(tok, n, max_len=512):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n * 2))
    out = []
    for ex in ds:
        q = tok("Question: " + ex["question"] + "\nAnswer:", return_tensors="pt").input_ids[0]
        a = tok(" " + ex["answer"], return_tensors="pt").input_ids[0]
        ids = torch.cat([q, a])[:max_len]
        m = torch.zeros(len(ids), dtype=torch.bool); m[min(len(q), len(ids)):] = True
        if m.sum() >= 2:
            out.append((ids, m))
        if len(out) >= n:
            break
    return out


def build_code(tok, n, max_len=512):
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    out = []
    for ex in ds:
        p = tok(ex["prompt"], return_tensors="pt").input_ids[0]
        c = tok(ex["canonical_solution"], return_tensors="pt").input_ids[0]
        ids = torch.cat([p, c])[:max_len]
        m = torch.zeros(len(ids), dtype=torch.bool); m[min(len(p), len(ids)):] = True
        if m.sum() >= 2:
            out.append((ids, m))
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def nll(model, batches, device):
    tot = ntok = 0
    lf = torch.nn.CrossEntropyLoss(reduction="none")
    for ids, mask in batches:
        ids = ids.unsqueeze(0).to(device)
        ce = lf(model(ids).logits[0][:-1].float(), ids[0, 1:])
        mm = mask[1:].to(device); tot += float((ce * mm).sum()); ntok += int(mm.sum())
    return tot / max(ntok, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math-model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--code-model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--out", default="results/fp_moe_complementary")
    ap.add_argument("--n-math", type=int, default=32)
    ap.add_argument("--n-code", type=int, default=40)
    ap.add_argument("--n-interp", type=int, default=5)
    args = ap.parse_args()

    device = "cuda"
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.math_model)
    math = build_math(tok, args.n_math)
    code = build_code(tok, args.n_code)
    log(f"math {len(math)}  code {len(code)}")

    log("loading math-model (GPU) ...")
    model = AutoModelForCausalLM.from_pretrained(args.math_model, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    A = OrderedDict((n, p.detach().cpu().clone()) for n, p in model.named_parameters())
    log("loading code-model (CPU) ...")
    B = load_cpu(args.code_model)

    @torch.no_grad()
    def set_interp(t):
        for n, p in model.named_parameters():
            p.copy_(((1 - t) * A[n].float() + t * B[n].float()).to(p.dtype))

    def ev(label):
        m = nll(model, math, device); c = nll(model, code, device)
        log(f"  [{label}] math={m:.4f} code={c:.4f} avg={0.5*(m+c):.4f} worst={max(m,c):.4f}")
        return {"math": m, "code": c, "avg": 0.5 * (m + c), "worst": max(m, c)}

    res = {}
    set_interp(0.0); res["math_model(Thinking)"] = ev("Thinking")
    set_interp(1.0); res["code_model(Coder)"] = ev("Coder")
    path = []
    for i in range(args.n_interp):
        t = i / (args.n_interp - 1)
        set_interp(t); r = ev(f"t={t:.2f}"); path.append({"t": t, **r})

    src_avg_a = res["math_model(Thinking)"]["avg"]
    src_avg_b = res["code_model(Coder)"]["avg"]
    best_src_avg = min(src_avg_a, src_avg_b)
    best_merge = min(path, key=lambda r: r["avg"])
    summary = {
        "math_model": args.math_model, "code_model": args.code_model,
        "endpoints": {"Thinking": res["math_model(Thinking)"], "Coder": res["code_model(Coder)"]},
        "interpolation": path,
        "best_source_avg_nll": best_src_avg,
        "best_merge_avg_nll": best_merge["avg"],
        "best_merge_t": best_merge["t"],
        "merge_beats_both_sources_on_avg": best_merge["avg"] < best_src_avg,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {outdir/'summary.json'}")
    log("\n=== VERDICT (avg NLL over math+code, lower=better) ===")
    log(f"  Thinking avg={src_avg_a:.4f}  Coder avg={src_avg_b:.4f}  best source avg={best_src_avg:.4f}")
    log(f"  best merge (t={best_merge['t']:.2f}) avg={best_merge['avg']:.4f}")
    if best_merge["avg"] < best_src_avg:
        log(f"  >>> MERGE BEATS BOTH SOURCES on multitask avg by {best_src_avg-best_merge['avg']:.4f} NLL <<<")
    else:
        log(f"  merge does not beat best source avg (gap {best_merge['avg']-best_src_avg:+.4f})")


if __name__ == "__main__":
    main()
