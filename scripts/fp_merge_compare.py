"""First-principles experiment 5: a UNIFIED merge family, selected by connectivity.

Instead of asking "which named method (linear / TIES / Fisher / task-arithmetic)
wins in this regime", we put them all in ONE parameterized family applied to
*aligned* coordinates, and let a held-out connectivity probe pick the coefficients:

  theta* = base + lambda * combine(a, b; density, sign_resolve, importance)
     a = theta_A - base,  b = theta_B - base        (task vectors, aligned)
     combine = importance-weighted, sign-elected, magnitude-trimmed disjoint merge
     router (MoE): {average | keep_A | keep_base}   -- treat gate as discrete

Special cases the family contains:
  linear avg         : density=1, sign_resolve=False, importance=uniform, lambda=1
  TIES               : density<1, sign_resolve=True,  importance=uniform
  task arithmetic    : density=1, sign_resolve=False, lambda free
  magnitude-importance: importance=mag

We select (density, lambda, sign_resolve, importance, router) on a held-out split
by worst-task NLL, then report on a disjoint test split.  This operationalises
"measure connectivity, don't assume it": the geometry picks the method.

Works for dense (Qwen2.5-0.5B instruct/coder/base) and MoE (Qwen3-30B-A3B
instruct/coder/base) -- same code, model ids differ.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path

import torch
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROUTER_RE = re.compile(r"\.mlp\.gate\.weight$")


def log(m):
    print(m, flush=True)


def load_cpu_params(model_id, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True)
    sd = OrderedDict((n, p.detach().to("cpu").clone()) for n, p in m.named_parameters())
    del m
    return sd


# ---------------- data ----------------
GENERAL_TEXTS = [
    "Large language models are often adapted from a common base model. "
    "A useful merged checkpoint must preserve broad instruction following while avoiding a high loss barrier.",
    "Weight averaging is reliable only when the endpoints lie in a compatible basin. "
    "When the interpolation path crosses a ridge, midpoint averaging can damage both tasks.",
    "A validation probe should measure the merged model directly instead of assuming that a named merge rule is safe.",
    "The scientific question is not which recipe wins on a table, but which mechanism explains the failure mode.",
    "Model connectivity links the geometry of fine tuned checkpoints to the performance of weight-space interpolation.",
    "Small held out slices are not a final benchmark, but they are useful for rejecting unsafe checkpoint candidates.",
    "Dense transformer branches can share most parameters while still disagreeing on a small set of important directions.",
    "A same-shape merged model must keep the tokenizer, config, tensor names, tensor shapes, and model class unchanged.",
    "The source checkpoints can specialize in different skills even when they started from the same pretrained anchor.",
    "If a merge is selected by a probe, the result still has to be checked on a disjoint split and then by hosted eval.",
    "A unified method should contain simple averaging as a special case, but should not be forced to choose it.",
]

CODE_SNIPPETS = [
    (
        "def add_numbers(values):\n"
        "    total = 0\n"
        "    for value in values:\n"
        "        total += value\n",
        "    return total\n",
    ),
    (
        "def normalize_name(text):\n"
        "    text = text.strip().lower()\n"
        "    return '-'.join(text.split())\n",
        "",
    ),
    (
        "def is_palindrome(text):\n"
        "    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n",
        "    return cleaned == cleaned[::-1]\n",
    ),
    (
        "def chunked(items, size):\n"
        "    for start in range(0, len(items), size):\n",
        "        yield items[start:start + size]\n",
    ),
    (
        "def safe_divide(a, b):\n"
        "    if b == 0:\n"
        "        return None\n",
        "    return a / b\n",
    ),
    (
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.value = 0\n"
        "    def inc(self):\n",
        "        self.value += 1\n",
    ),
    (
        "def flatten(nested):\n"
        "    out = []\n"
        "    for row in nested:\n",
        "        out.extend(row)\n"
        "    return out\n",
    ),
    (
        "def moving_average(values, window):\n"
        "    result = []\n"
        "    for idx in range(len(values)):\n",
        "        lo = max(0, idx - window + 1)\n"
        "        result.append(sum(values[lo:idx + 1]) / (idx - lo + 1))\n"
        "    return result\n",
    ),
]


def _cycle(items, n):
    out = []
    while len(out) < n:
        out.extend(items)
    return out[:n]


def build_general(tok, n_seq, seqlen):
    out = []
    for text in _cycle(GENERAL_TEXTS, n_seq):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=seqlen).input_ids[0]
        if ids.numel() >= 4:
            out.append((ids, None))
    return out


def build_code(tok, n, max_len=256):
    out = []
    for prompt, completion in _cycle(CODE_SNIPPETS, n):
        p = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len).input_ids[0]
        c = tok(completion or "\n", return_tensors="pt", truncation=True, max_length=max_len).input_ids[0]
        ids = torch.cat([p, c])[:max_len]
        m = torch.zeros(len(ids), dtype=torch.bool)
        m[min(len(p), len(ids)) :] = True
        if m.sum() >= 1 and ids.numel() >= 4:
            out.append((ids, m))
    return out


@torch.no_grad()
def nll(model, batches, device):
    tot, ntok = 0.0, 0
    lf = torch.nn.CrossEntropyLoss(reduction="none")
    for ids, mask in batches:
        ids = ids.unsqueeze(0).to(device)
        lg = model(ids).logits[0]
        ce = lf(lg[:-1].float(), ids[0, 1:])
        if mask is None:
            tot += float(ce.sum()); ntok += ce.numel()
        else:
            m = mask[1:].to(device); tot += float((ce * m).sum()); ntok += int(m.sum())
    return tot / max(ntok, 1)


# ---------------- the unified merge family ----------------
def trim(x, density):
    if density >= 1.0:
        return x
    k = max(1, int(math.ceil(x.numel() * density)))
    if k >= x.numel():
        return x
    thr = torch.topk(x.abs().reshape(-1), k, largest=True).values[-1]
    return torch.where(x.abs() >= thr, x, torch.zeros_like(x))


def combine(a, b, *, density, sign_resolve, importance):
    """combine two task-vector tensors into one delta (all fp32)."""
    if density < 1.0:
        a = trim(a, density); b = trim(b, density)
    if importance == "mag":
        wa, wb = a.abs(), b.abs()
    else:  # uniform
        wa, wb = torch.ones_like(a), torch.ones_like(b)
    if sign_resolve:
        s = torch.sign(wa * a + wb * b)
        ka = ((torch.sign(a) == s) & (a != 0)).float()
        kb = ((torch.sign(b) == s) & (b != 0)).float()
        num = wa * a * ka + wb * b * kb
        den = wa * ka + wb * kb
    else:
        num = wa * a + wb * b
        den = wa + wb
    return torch.where(den > 0, num / den, torch.zeros_like(num))


@torch.no_grad()
def apply_merge(model, base, A, B, *, density, lam, sign_resolve, importance, router):
    """Write base + lam*combine(a,b) into model params; router policy overrides gate."""
    for n, p in model.named_parameters():
        if n not in base:
            continue
        if router != "average" and ROUTER_RE.search(n):
            src = A if router == "keep_A" else base
            p.copy_(src[n].to(p.dtype))
            continue
        a = A[n].float() - base[n].float()
        b = B[n].float() - base[n].float()
        delta = combine(a, b, density=density, sign_resolve=sign_resolve, importance=importance)
        p.copy_((base[n].float() + lam * delta).to(p.dtype))


@torch.no_grad()
def set_endpoint(model, sd):
    for n, p in model.named_parameters():
        if n in sd:
            p.copy_(sd[n].to(p.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--coder", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-general", type=int, default=12)
    ap.add_argument("--n-code", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--is-moe", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument(
        "--grid-profile",
        choices=["linear", "quick", "full"],
        default="linear",
        help="linear avoids expensive top-k sparsification; quick adds one sparse density; full is larger.",
    )
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.instruct, local_files_only=True)
    gen = build_general(tok, args.n_general, args.seqlen)
    code = build_code(tok, args.n_code)
    # held-out (selection) vs test (report) split
    gh, gt = gen[: len(gen) // 2], gen[len(gen) // 2 :]
    ch, ct = code[: len(code) // 2], code[len(code) // 2 :]
    log(f"general {len(gh)}+{len(gt)}  code {len(ch)}+{len(ct)}")

    log(f"loading Instruct ({device}) ...")
    model = AutoModelForCausalLM.from_pretrained(args.instruct, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    A = OrderedDict((n, p.detach().cpu().clone()) for n, p in model.named_parameters())
    log("loading Coder (CPU) ...")
    B = load_cpu_params(args.coder)
    log("loading Base (CPU) ...")
    base = load_cpu_params(args.base)

    def evalset(hodl):
        g = nll(model, gh if hodl else gt, device)
        c = nll(model, ch if hodl else ct, device)
        return {"general": g, "code": c, "worst": max(g, c), "avg": 0.5 * (g + c)}

    results = {}
    # endpoints
    set_endpoint(model, A); results["instruct"] = evalset(False)
    set_endpoint(model, B); results["coder"] = evalset(False)
    log(f"  instruct {results['instruct']}  coder {results['coder']}")
    best_endpoint_worst = min(results["instruct"]["worst"], results["coder"]["worst"])

    # named baselines (report on test)
    def run(name, **kw):
        apply_merge(model, base, A, B, **kw)
        r = evalset(False)
        results[name] = {**r, "config": kw}
        log(f"  [{name}] worst={r['worst']:.4f} avg={r['avg']:.4f} (gen={r['general']:.3f} code={r['code']:.3f})")
        return r

    run("linear", density=1.0, lam=1.0, sign_resolve=False, importance="uniform", router="average")
    run("task_arith_0.5", density=1.0, lam=0.5, sign_resolve=False, importance="uniform", router="average")
    run("ties_0.5", density=0.5, lam=1.0, sign_resolve=True, importance="uniform", router="average")

    # ---- unified: select (density, lam, sign_resolve, importance, router) on HELD-OUT ----
    grid = []
    if args.grid_profile == "linear":
        densities = (1.0,)
        lambdas = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
        sign_resolves = (False, True)
        importances = ("uniform", "mag")
    elif args.grid_profile == "quick":
        densities = (1.0, 0.5)
        lambdas = (0.5, 0.8, 1.0)
        sign_resolves = (False, True)
        importances = ("uniform", "mag")
    else:
        densities = (1.0, 0.7, 0.5, 0.3)
        lambdas = (0.25, 0.5, 0.8, 1.0, 1.25)
        sign_resolves = (False, True)
        importances = ("uniform", "mag")

    for density in densities:
        for lam in lambdas:
            for sr in (True, False):
                if sr not in sign_resolves:
                    continue
                for imp in importances:
                    routers = ("average", "keep_A") if args.is_moe else ("average",)
                    for rt in routers:
                        grid.append(dict(density=density, lam=lam, sign_resolve=sr, importance=imp, router=rt))
    log(f"unified selection over {len(grid)} configs on held-out ...")
    best, best_cfg = None, None
    sel_trace = []
    for cfg in grid:
        apply_merge(model, base, A, B, **cfg)
        r = evalset(True)  # held-out
        sel_trace.append({**cfg, "ho_worst": r["worst"], "ho_avg": r["avg"]})
        score = r["worst"]
        if best is None or score < best:
            best, best_cfg = score, cfg
    log(f"  selected: {best_cfg}  (held-out worst={best:.4f})")
    apply_merge(model, base, A, B, **best_cfg)
    rtest = evalset(False)
    results["unified"] = {**rtest, "config": best_cfg, "ho_worst": best}
    log(f"  [unified] TEST worst={rtest['worst']:.4f} avg={rtest['avg']:.4f} (gen={rtest['general']:.3f} code={rtest['code']:.3f})")

    summary = {
        "schema_version": 1,
        "models": {"instruct": args.instruct, "coder": args.coder, "base": args.base},
        "is_moe": args.is_moe,
        "grid_profile": args.grid_profile,
        "best_endpoint_worst": best_endpoint_worst,
        "candidate_count": len(grid),
        "results": results,
        "selection_trace": sel_trace,
        "interpretation": {
            "unified_family": "base + lambda * coordinate_rule(delta_instruct, delta_coder), with named methods as special cases",
            "selection_rule": "choose the candidate with lowest held-out worst-task NLL, then report it on a disjoint test split",
            "why_this_is_unified": "the method selector is a probe over mechanisms: barrier/coefficient, sign disagreement, magnitude importance, and optionally router policy",
            "finite_candidate_note": "inside the candidate set, validation risk selection targets the empirical best candidate; residual risk is validation noise, not a static algorithm assumption",
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(sel_trace).sort_values(["ho_worst", "ho_avg"]).to_csv(outdir / "selection_trace.csv", index=False)
    pd.DataFrame(
        [
            {"method": name, **{k: v for k, v in row.items() if k != "config"}, "config": json.dumps(row.get("config", {}), sort_keys=True)}
            for name, row in results.items()
        ]
    ).sort_values(["worst", "avg"]).to_csv(outdir / "method_metrics.csv", index=False)
    report_lines = [
        "# Unified Merge Family Probe",
        "",
        "这个实验不是预先判断哪个命名算法最好，而是把多个方法写成同一个候选族，再用 held-out worst-task NLL 选择候选，最后在 disjoint test split 上复验。",
        "",
        "## Result",
        "",
        f"- candidate count: `{len(grid)}`",
        f"- grid profile: `{args.grid_profile}`",
        f"- selected config: `{json.dumps(best_cfg, sort_keys=True)}`",
        f"- selected held-out worst NLL: `{best:.4f}`",
        f"- unified test worst NLL: `{rtest['worst']:.4f}`",
        f"- best endpoint worst NLL: `{best_endpoint_worst:.4f}`",
        "",
        "## Mechanism",
        "",
        "`linear average`、`task arithmetic`、sign-elect、magnitude-weighted merge 都是 `base + lambda * combine(delta_A, delta_B)` 的特例。选择器测的是：midpoint 是否跨 barrier、任务向量是否需要缩放、同坐标 sign 冲突是否应该被过滤、以及重要性是否集中在大幅度坐标上。",
        "",
        "在有限候选族内，held-out 选择等价于选经验风险最低的候选；它不能保证击败所有未知算法，但能避免把某个固定方法当成先验真理。真正上线前仍需要 vLLM hosted downstream eval。",
        "",
        "## Files",
        "",
        "- `summary.json`",
        "- `method_metrics.csv`",
        "- `selection_trace.csv`",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n")
    log(f"\nwrote {outdir/'summary.json'}")
    log("\n=== VERDICT (worst-task NLL, lower=better) ===")
    for k in ["instruct", "coder", "linear", "task_arith_0.5", "ties_0.5", "unified"]:
        if k in results:
            log(f"   {k:16s} worst={results[k]['worst']:.4f} avg={results[k]['avg']:.4f}")
    log(f"   best single endpoint worst={best_endpoint_worst:.4f}")


if __name__ == "__main__":
    main()
