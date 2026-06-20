"""First-principles experiment 4: interpolation barrier of a REAL MoE LLM pair.

The 30B-MoE analogue of fp_curvature_law.  We showed (a) dense far-apart experts
have a large anharmonic barrier no linear merge crosses, and (b) real same-base
MoE experts stay index-aligned (no permutation needed).  So the open question for
a real MoE merge is purely geometric: along the Instruct<->Coder weight path, is
there a barrier, and does ANY linear-in-weight merge beat the better endpoint?

We hold full state dicts on CPU (RAM is large), interpolate per floating param,
load into one GPU model, and measure code (HumanEval completion) and general
(wikitext) NLL.  We also report task-vector geometry vs the shared base.
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


def load_cpu_params(model_id, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True)
    sd = OrderedDict((n, p.detach().to("cpu").clone()) for n, p in m.named_parameters())
    del m
    return sd


def build_wikitext(tok, n_seq, seqlen):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t and not t.isspace())
    ids = tok(text, return_tensors="pt").input_ids[0]
    out = []
    for i in range(0, len(ids) - seqlen, seqlen):
        out.append((ids[i : i + seqlen], None))
        if len(out) >= n_seq:
            break
    return out


def build_humaneval(tok, n, max_len=640):
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    out = []
    for ex in ds:
        p = tok(ex["prompt"], return_tensors="pt").input_ids[0]
        c = tok(ex["canonical_solution"], return_tensors="pt").input_ids[0]
        ids = torch.cat([p, c])[:max_len]
        m = torch.zeros(len(ids), dtype=torch.bool)
        m[min(len(p), len(ids)) :] = True
        if m.sum() >= 2:
            out.append((ids, m))
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def nll(model, batches, device):
    tot, ntok = 0.0, 0
    lf = torch.nn.CrossEntropyLoss(reduction="none")
    for ids, mask in batches:
        ids = ids.unsqueeze(0).to(device)
        lg = model(ids).logits[0]
        tgt = ids[0, 1:]
        ce = lf(lg[:-1].float(), tgt)
        if mask is None:
            tot += float(ce.sum())
            ntok += ce.numel()
        else:
            m = mask[1:].to(device)
            tot += float((ce * m).sum())
            ntok += int(m.sum())
    return tot / max(ntok, 1)


@torch.no_grad()
def set_interp(model, sd_a, sd_b, t):
    """model floating params <- (1-t)*a + t*b."""
    for n, p in model.named_parameters():
        if n in sd_a:
            p.copy_(((1.0 - t) * sd_a[n].float() + t * sd_b[n].float()).to(p.dtype))


def flat_norm_cos(sd_a, sd_b, sd_base, names):
    """task-vector geometry vs base (chunked to bound memory)."""
    dot = na = nb = 0.0
    for n in names:
        a = (sd_a[n].float() - sd_base[n].float()).reshape(-1)
        b = (sd_b[n].float() - sd_base[n].float()).reshape(-1)
        dot += float(a @ b)
        na += float(a @ a)
        nb += float(b @ b)
    import math

    return {
        "cos(tauI,tauC)": dot / (math.sqrt(na) * math.sqrt(nb) + 1e-12),
        "norm_tauI": math.sqrt(na),
        "norm_tauC": math.sqrt(nb),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--coder", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--base", default="Qwen/Qwen3-30B-A3B-Base")
    ap.add_argument("--out", default="results/fp_moe_barrier")
    ap.add_argument("--n-general", type=int, default=24)
    ap.add_argument("--n-code", type=int, default=40)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--n-interp", type=int, default=7)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.instruct)
    log("building eval data ...")
    gen = build_wikitext(tok, args.n_general, args.seqlen)
    code = build_humaneval(tok, args.n_code)
    log(f"  general={len(gen)} code={len(code)}")

    log("loading Instruct onto GPU ...")
    model = AutoModelForCausalLM.from_pretrained(args.instruct, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    sd_I = OrderedDict((n, p.detach().to("cpu").clone()) for n, p in model.named_parameters())
    log("loading Coder params to CPU ...")
    sd_C = load_cpu_params(args.coder)
    names = list(sd_I.keys())
    assert list(sd_C.keys()) == names, "param mismatch"

    def ev(label):
        g = nll(model, gen, device)
        c = nll(model, code, device)
        log(f"  [{label}] general={g:.4f} code={c:.4f} worst={max(g,c):.4f}")
        return {"general": g, "code": c, "worst": max(g, c), "avg": 0.5 * (g + c)}

    res = {}
    # endpoints
    set_interp(model, sd_I, sd_C, 0.0)
    res["instruct"] = ev("instruct")
    set_interp(model, sd_I, sd_C, 1.0)
    res["coder"] = ev("coder")

    # interpolation path
    path = []
    for i in range(args.n_interp):
        t = i / (args.n_interp - 1)
        set_interp(model, sd_I, sd_C, t)
        r = ev(f"t={t:.2f}")
        path.append({"t": t, **r})

    # geometry vs base
    geom = None
    try:
        log("loading base params to CPU for task-vector geometry ...")
        sd_base = load_cpu_params(args.base)
        geom = flat_norm_cos(sd_I, sd_C, sd_base, names)
        log(f"  geometry: {geom}")
    except Exception as e:
        log(f"  geometry skipped: {e}")

    summary = {
        "models": {"instruct": args.instruct, "coder": args.coder, "base": args.base},
        "endpoints": {"instruct": res["instruct"], "coder": res["coder"]},
        "interpolation": path,
        "geometry": geom,
        "barrier_general": max(p["general"] for p in path) - max(path[0]["general"], path[-1]["general"]),
        "barrier_code": max(p["code"] for p in path) - max(path[0]["code"], path[-1]["code"]),
        "best_interior_worst": min(p["worst"] for p in path[1:-1]),
        "endpoint_best_worst": min(path[0]["worst"], path[-1]["worst"]),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {outdir/'summary.json'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = [p["t"] for p in path]
        plt.figure(figsize=(7, 5))
        plt.plot(ts, [p["general"] for p in path], "o-", label="general (wikitext)")
        plt.plot(ts, [p["code"] for p in path], "s-", label="code (humaneval)")
        plt.plot(ts, [p["worst"] for p in path], "k--", label="worst-task")
        plt.axvline(0.5, color="gray", ls=":", alpha=0.6)
        plt.xlabel("t  (0=Instruct, 1=Coder)")
        plt.ylabel("NLL")
        plt.title("Qwen3-30B-A3B MoE: Instruct<->Coder interpolation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "moe_barrier.png", dpi=130)
        log(f"wrote {outdir/'moe_barrier.png'}")
    except Exception as e:
        log(f"figure failed: {e}")

    log("\n=== VERDICT ===")
    log(f"barrier(general)={summary['barrier_general']:.3f} barrier(code)={summary['barrier_code']:.3f}")
    log(f"best interior worst-NLL={summary['best_interior_worst']:.3f} vs best endpoint worst-NLL={summary['endpoint_best_worst']:.3f}")


if __name__ == "__main__":
    main()
