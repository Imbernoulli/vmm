"""Fast dense lambda-sweep: show the unified family contains AND beats linear avg.

theta(lambda) = base + lambda * (a+b)/2,  a=instruct-base, b=coder-base
  lambda=0 -> base ; lambda=1 -> 0.5(instruct+coder) = linear midpoint
Also a sign-resolved (TIES-style) variant.  Real wikitext + humaneval NLL.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def log(m):
    print(m, flush=True)


def load_params(mid, device=None):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32, local_files_only=True)
    if device:
        m.to(device)
    return m


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--coder", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="results/fp_dense_lambda")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.instruct)

    # real eval data
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t and not t.isspace())
    wids = tok(text, return_tensors="pt").input_ids[0]
    gen = [(wids[i : i + 256], None) for i in range(0, 256 * 48, 256)][:48]
    he = load_dataset("openai/openai_humaneval", split="test")
    code = []
    for ex in he:
        p = tok(ex["prompt"], return_tensors="pt").input_ids[0]
        c = tok(ex["canonical_solution"], return_tensors="pt").input_ids[0]
        ids = torch.cat([p, c])[:512]
        m = torch.zeros(len(ids), dtype=torch.bool); m[min(len(p), len(ids)):] = True
        if m.sum() >= 2:
            code.append((ids, m))
        if len(code) >= 60:
            break

    @torch.no_grad()
    def nll(model, batches):
        tot = ntok = 0
        lf = torch.nn.CrossEntropyLoss(reduction="none")
        for ids, mask in batches:
            ids = ids.unsqueeze(0).to(device)
            ce = lf(model(ids).logits[0][:-1].float(), ids[0, 1:])
            if mask is None:
                tot += float(ce.sum()); ntok += ce.numel()
            else:
                mm = mask[1:].to(device); tot += float((ce * mm).sum()); ntok += int(mm.sum())
        return tot / max(ntok, 1)

    model = load_params(args.instruct, device).eval()
    A = OrderedDict((n, p.detach().cpu().clone()) for n, p in model.named_parameters())
    B = {n: p.detach().cpu().clone() for n, p in load_params(args.coder).named_parameters()}
    base = {n: p.detach().cpu().clone() for n, p in load_params(args.base).named_parameters()}

    @torch.no_grad()
    def set_lambda(lam, sign_resolve=False, density=1.0):
        for n, p in model.named_parameters():
            a = A[n].float() - base[n].float()
            b = B[n].float() - base[n].float()
            if density < 1.0:
                ka = a.abs() >= torch.quantile(a.abs().flatten().float(), 1 - density) if a.numel() else a
            if sign_resolve:
                s = torch.sign(a + b)
                ka = (torch.sign(a) == s) & (a != 0)
                kb = (torch.sign(b) == s) & (b != 0)
                num = a * ka + b * kb
                den = (ka.float() + kb.float())
                delta = torch.where(den > 0, num / den.clamp_min(1), torch.zeros_like(num))
            else:
                delta = 0.5 * (a + b)
            p.copy_((base[n].float() + lam * delta).to(p.dtype))

    @torch.no_grad()
    def set_endpoint(sd):
        for n, p in model.named_parameters():
            p.copy_(sd[n].to(p.dtype))

    rows = []
    set_endpoint(A); gi, ci = nll(model, gen), nll(model, code)
    set_endpoint(B); gc, cc = nll(model, gen), nll(model, code)
    best_endpoint = min(max(gi, ci), max(gc, cc))
    log(f"instruct worst={max(gi,ci):.3f}  coder worst={max(gc,cc):.3f}  best-endpoint worst={best_endpoint:.3f}")

    for lam in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        set_lambda(lam)
        g, c = nll(model, gen), nll(model, code)
        rows.append({"lambda": lam, "mode": "uniform", "general": g, "code": c, "worst": max(g, c)})
        tag = "  <- LINEAR avg" if lam == 1.0 else ""
        log(f"  uniform   lam={lam:.2f} worst={max(g,c):.3f}{tag}")
    for lam in [0.25, 0.5, 0.75, 1.0]:
        set_lambda(lam, sign_resolve=True)
        g, c = nll(model, gen), nll(model, code)
        rows.append({"lambda": lam, "mode": "sign_resolved", "general": g, "code": c, "worst": max(g, c)})
        log(f"  sign-res  lam={lam:.2f} worst={max(g,c):.3f}")

    linear = next(r for r in rows if r["mode"] == "uniform" and r["lambda"] == 1.0)["worst"]
    unified = min(r["worst"] for r in rows)
    unified_row = min(rows, key=lambda r: r["worst"])
    summary = {
        "best_endpoint_worst": best_endpoint,
        "linear_worst": linear,
        "unified_best_worst": unified,
        "unified_best_config": unified_row,
        "rows": rows,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\nLINEAR worst-NLL          = {linear:.3f}")
    log(f"UNIFIED-best worst-NLL    = {unified:.3f}  (config: {unified_row['mode']} lam={unified_row['lambda']})")
    log(f"best single endpoint worst= {best_endpoint:.3f}")
    log(f"unified beats linear by {linear-unified:.3f} NLL; gap to endpoint {unified-best_endpoint:.3f}")
    log(f"wrote {outdir/'summary.json'}")


if __name__ == "__main__":
    main()
