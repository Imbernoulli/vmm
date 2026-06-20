"""Task A: does calibrating the MERGED router improve a real MoE merge?

Finding 5 showed two real same-base MoEs (Qwen3-30B Instruct/Coder) route the
same tokens differently (~55% disagreement, worst in middle layers), yet the
weight-merge stays smooth because of expert alignment + top-8/128 redundancy.
The predicted remaining lever: after averaging experts, the averaged router is a
compromise that loses dispatch precision -> re-fit ONLY the router (gate.weight)
to the merged experts on a small mixed set, experts frozen, architecture intact.

We compare general (wikitext) + code (humaneval) NLL for:
  instruct, coder, linear-merge 0.5(I+C), linear-merge + router-calibration.
Only the 48 `mlp.gate.weight` tensors are trained; everything else is frozen.
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


def load_cpu_params(mid):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
    sd = OrderedDict((n, p.detach().cpu().clone()) for n, p in m.named_parameters())
    del m
    return sd


def build_data(tok, seqlen, n):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t and not t.isspace())
    wids = tok(text, return_tensors="pt").input_ids[0]
    gen = [(wids[i : i + seqlen], None) for i in range(0, seqlen * (2 * n + 4), seqlen)][: 2 * n]
    he = load_dataset("openai/openai_humaneval", split="test")
    code = []
    for ex in he:
        p = tok(ex["prompt"], return_tensors="pt").input_ids[0]
        c = tok(ex["canonical_solution"], return_tensors="pt").input_ids[0]
        ids = torch.cat([p, c])[:512]
        m = torch.zeros(len(ids), dtype=torch.bool); m[min(len(p), len(ids)):] = True
        if m.sum() >= 2:
            code.append((ids, m))
        if len(code) >= 2 * n:
            break
    return gen, code


@torch.no_grad()
def nll(model, batches, device):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--coder", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--out", default="results/fp_moe_router_calibrate")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()

    device = "cuda"
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.instruct)
    gen, code = build_data(tok, args.seqlen, args.n)
    gh, gt = gen[: len(gen) // 2], gen[len(gen) // 2 :]   # cal / eval split
    ch, ct = code[: len(code) // 2], code[len(code) // 2 :]
    log(f"cal: {len(gh)} gen + {len(ch)} code ; eval: {len(gt)} gen + {len(ct)} code")

    log("loading Instruct (GPU) ...")
    model = AutoModelForCausalLM.from_pretrained(args.instruct, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    A = OrderedDict((n, p.detach().cpu().clone()) for n, p in model.named_parameters())
    log("loading Coder (CPU) ...")
    B = load_cpu_params(args.coder)

    def set_params(sd):
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(sd[n].to(p.dtype))

    def set_linear():
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_((0.5 * (A[n].float() + B[n].float())).to(p.dtype))

    def ev():
        g = nll(model, gt, device); c = nll(model, ct, device)
        return {"general": g, "code": c, "worst": max(g, c), "avg": 0.5 * (g + c)}

    results = {}
    set_params(A); results["instruct"] = ev(); log(f"  instruct {results['instruct']}")
    set_params(B); results["coder"] = ev(); log(f"  coder {results['coder']}")
    set_linear(); results["linear_merge"] = ev(); log(f"  linear_merge {results['linear_merge']}")

    # ---- router calibration: train only *.mlp.gate.weight ----
    router_params = []
    for n, p in model.named_parameters():
        train = n.endswith(".mlp.gate.weight")
        p.requires_grad_(train)
        if train:
            router_params.append(p)
    log(f"calibrating {len(router_params)} router tensors "
        f"({sum(p.numel() for p in router_params)/1e6:.1f}M params), experts frozen ...")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.Adam(router_params, lr=args.lr)
    cal = [(b, None) for b in [x[0] for x in gh]] + ch  # mixed general(full) + code(masked)
    lf = torch.nn.CrossEntropyLoss(reduction="none")
    step = 0
    while step < args.steps:
        for ids, mask in cal:
            ids_b = ids.unsqueeze(0).to(device)
            out = model(ids_b).logits[0]
            ce = lf(out[:-1].float(), ids_b[0, 1:])
            if mask is None:
                loss = ce.mean()
            else:
                mm = mask[1:].to(device); loss = (ce * mm).sum() / mm.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 20 == 0:
                log(f"  step {step}/{args.steps} loss={float(loss):.4f}")
            if step >= args.steps:
                break
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    results["linear_merge_routercal"] = ev()
    log(f"  linear_merge_routercal {results['linear_merge_routercal']}")

    summary = {"models": {"instruct": args.instruct, "coder": args.coder}, "results": results,
               "steps": args.steps, "lr": args.lr}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {outdir/'summary.json'}")
    log("\n=== VERDICT (NLL, lower=better) ===")
    for k in ["instruct", "coder", "linear_merge", "linear_merge_routercal"]:
        r = results[k]
        log(f"  {k:24s} general={r['general']:.4f} code={r['code']:.4f} worst={r['worst']:.4f} avg={r['avg']:.4f}")
    base_worst = results["linear_merge"]["worst"]; cal_worst = results["linear_merge_routercal"]["worst"]
    log(f"\nrouter calibration changed worst-NLL by {cal_worst-base_worst:+.4f} "
        f"({'better' if cal_worst<base_worst else 'worse'})")


if __name__ == "__main__":
    main()
