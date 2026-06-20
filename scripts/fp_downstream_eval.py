"""Auxiliary generation downstream eval for one Qwen3 MoE model path.

This is the first-principles transformers-generation matrix runner. It is useful
for mechanism evidence, but it is not the final matched vLLM selector path.

Tasks (all real, plain few-shot prompting so base/instruct/coder/merge compare
apples-to-apples):
  - gsm8k     : 4-shot, greedy generate, numeric exact-match
  - humaneval : greedy complete, pass@1 via guarded subprocess (timeout, tempdir)
  - mmlu      : 5-shot-free letter log-likelihood accuracy (argmax over A/B/C/D)

Usage: fp_downstream_eval.py --model <path|id> --out <dir> [--n-gsm8k N ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def log(m):
    print(m, flush=True)


GSM8K_FEWSHOT = """Question: Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?
Answer: In April she sold 48. In May she sold 48/2 = 24. Total 48+24 = 72. #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Per minute 12/60 = 0.2. For 50 minutes 50*0.2 = 10. #### 10

Question: Betty is saving for a $100 wallet. She has half the money. Her parents give $15 and grandparents twice as much as parents. How much more does she need?
Answer: Half is 50. Grandparents give 2*15=30. She has 50+15+30=95. Needs 100-95=5. #### 5

Question: James writes a 3-page letter to 2 friends twice a week. How many pages does he write a year?
Answer: Each time 3*2=6 pages. Twice a week 6*2=12. Per year 12*52=624. #### 624

"""


def extract_num(text):
    if "####" in text:
        text = text.split("####")[1]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[0].replace(",", "").rstrip(".") if nums else None


@torch.no_grad()
def eval_gsm8k(model, tok, device, n):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    ok = 0
    for ex in ds:
        ids = tok(GSM8K_FEWSHOT + f"Question: {ex['question']}\nAnswer:", return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split("Question:")[0]
        ok += int((extract_num(gen) or "X") == (extract_num(ex["answer"]) or "Y"))
    return ok / max(n, 1)


def run_prog(program, timeout=6):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(program); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout,
                           env={"PYTHONPATH": "", "PATH": "/usr/bin:/bin"})
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except OSError: pass


@torch.no_grad()
def eval_humaneval(model, tok, device, n):
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test").select(range(n))
    passed = 0
    for ex in ds:
        ids = tok(ex["prompt"], return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, max_new_tokens=384, do_sample=False, pad_token_id=tok.eos_token_id)
        comp = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        for stop in ["\ndef ", "\nclass ", "\nif __name__", "\n```", "\nprint(", "\n#"]:
            if stop in comp:
                comp = comp.split(stop)[0]
        prog = ex["prompt"] + comp + "\n" + ex["test"] + f"\ncheck({ex['entry_point']})\n"
        passed += int(run_prog(prog))
    return passed / max(n, 1)


@torch.no_grad()
def eval_mmlu(model, tok, device, n):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    idxs = list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    letters = ["A", "B", "C", "D"]
    # token ids for " A".." D" (leading space, as after "Answer:")
    lt_ids = [tok(" " + L, add_special_tokens=False).input_ids[-1] for L in letters]
    ok = 0
    for i in idxs:
        ex = ds[i]
        q = ex["question"]; ch = ex["choices"]; gold = ex["answer"]
        prompt = q + "\n" + "\n".join(f"{letters[j]}. {ch[j]}" for j in range(4)) + "\nAnswer:"
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        logits = model(ids).logits[0, -1].float()
        pred = int(torch.tensor([logits[t] for t in lt_ids]).argmax())
        ok += int(pred == gold)
    return ok / max(len(idxs), 1)


def build_merged(mids, device, router_cal, cal_steps):
    """In-memory uniform average of same-lineage models (+ optional router cal)."""
    from collections import OrderedDict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    w = 1.0 / len(mids)
    # load skeleton (= first model) on GPU, scale by w, then accumulate the rest (each model loaded once)
    model = AutoModelForCausalLM.from_pretrained(mids[0], torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    tok = AutoTokenizer.from_pretrained(mids[0], local_files_only=True)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(w)
        for mid in mids[1:]:
            m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, local_files_only=True)
            src = dict(m.named_parameters())
            for n, p in model.named_parameters():
                p.add_(w * src[n].to(p.device, p.dtype))
            del m, src
            log(f"  merged in {mid}")
    if router_cal:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"] if t and not t.isspace())
        wids = tok(text, return_tensors="pt").input_ids[0]
        gen = [wids[i:i + 256] for i in range(0, 256 * 30, 256)][:24]
        he = load_dataset("openai/openai_humaneval", split="test")
        code = []
        for ex in he:
            pp = tok(ex["prompt"], return_tensors="pt").input_ids[0]
            cc = tok(ex["canonical_solution"], return_tensors="pt").input_ids[0]
            ids = torch.cat([pp, cc])[:512]
            mask = torch.zeros(len(ids), dtype=torch.bool); mask[min(len(pp), len(ids)):] = True
            code.append((ids, mask))
            if len(code) >= 24:
                break
        rparams = []
        for n, p in model.named_parameters():
            tr = n.endswith(".mlp.gate.weight"); p.requires_grad_(tr)
            if tr: rparams.append(p)
        log(f"  router-cal {len(rparams)} routers ...")
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        opt = torch.optim.Adam(rparams, lr=2e-4)
        lf = torch.nn.CrossEntropyLoss(reduction="none")
        cal = [(g, None) for g in gen] + code
        step = 0
        while step < cal_steps:
            for ids, mask in cal:
                ib = ids.unsqueeze(0).to(device)
                ce = lf(model(ib).logits[0][:-1].float(), ib[0, 1:])
                loss = ce.mean() if mask is None else (ce * mask[1:].to(device)).sum() / mask[1:].sum().clamp_min(1)
                opt.zero_grad(); loss.backward(); opt.step(); step += 1
                if step >= cal_steps:
                    break
        model.eval(); model.gradient_checkpointing_disable(); model.config.use_cache = True
        for p in model.parameters():
            p.requires_grad_(False)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--merge", default=None, help="comma-separated model ids to uniform-average")
    ap.add_argument("--router-cal", action="store_true")
    ap.add_argument("--cal-steps", type=int, default=80)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm8k", type=int, default=40)
    ap.add_argument("--n-humaneval", type=int, default=40)
    ap.add_argument("--n-mmlu", type=int, default=120)
    ap.add_argument("--tasks", default="mmlu,gsm8k,humaneval")
    args = ap.parse_args()

    device = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.merge:
        mids = [m.strip() for m in args.merge.split(",") if m.strip()]
        name = args.name or ("merge_" + "+".join(Path(m).name.split("-")[-1] for m in mids) + ("_routercal" if args.router_cal else ""))
        log(f"[{name}] building merge of {mids} router_cal={args.router_cal} ...")
        model, tok = build_merged(mids, device, args.router_cal, args.cal_steps)
        model.eval()
    else:
        name = args.name or Path(args.model).name
        log(f"[{name}] loading {args.model} ...")
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    res = {"model": args.model or ("merge:" + (args.merge or "")), "router_cal": bool(args.router_cal), "name": name}
    tasks = args.tasks.split(",")
    if "mmlu" in tasks:
        res["mmlu"] = eval_mmlu(model, tok, device, args.n_mmlu); log(f"[{name}] mmlu={res['mmlu']:.3f}")
    if "gsm8k" in tasks:
        res["gsm8k"] = eval_gsm8k(model, tok, device, args.n_gsm8k); log(f"[{name}] gsm8k={res['gsm8k']:.3f}")
    if "humaneval" in tasks:
        res["humaneval"] = eval_humaneval(model, tok, device, args.n_humaneval); log(f"[{name}] humaneval={res['humaneval']:.3f}")
    accs = [res[t] for t in ("mmlu", "gsm8k", "humaneval") if t in res]
    res["avg"] = sum(accs) / len(accs)
    res["worst"] = min(accs)

    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{name}.json").write_text(json.dumps(res, indent=2))
    log(f"[{name}] DONE  mmlu={res.get('mmlu')} gsm8k={res.get('gsm8k')} humaneval={res.get('humaneval')} avg={res['avg']:.3f}")


if __name__ == "__main__":
    main()
