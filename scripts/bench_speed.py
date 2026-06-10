#!/usr/bin/env python3
"""Same-model, same-input speed comparison: Concat vs APE vs Ours-bank (parallel encoding).

All three run on Qwen3-8B + SDPA, on the SAME synthetic passage set, so the only thing varying is
HOW the passages are encoded/read:
  concat    : N passages in one sequence, full causal attention (prefill = O((N·L)^2))
  ape       : N passages each encoded independently from pos 0 into a bank; query reads whole bank
              with APE realignment (temperature/scale<1). prefill = N × O(L^2), parallel.
  ours      : identical independent-encoding bank, query reads whole bank, temp=scale=1 (the trained
              CrossKV read). Same compute graph as APE minus the scalar realignment.

We sweep N (#passages) and report prefill latency + decode latency (ms), median of several runs
after warmup. This answers: is our parallel-encoding read about as fast as APE, and how do both
compare to Concat as context grows.
"""
import argparse, os, sys, time, statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.eval_multiquery import build_prompt, context_mask_for, decode_texts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--n-paras", default="4,8,16,32,64")
    p.add_argument("--para-len", type=int, default=120, help="approx tokens per passage")
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--seg-cap", type=int, default=768)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--gpu", type=int, default=5)
    return p.parse_args()


def synth_passages(tok, n, para_len, rng_words):
    """n synthetic passages of ~para_len tokens each (deterministic filler + an index marker)."""
    out = []
    for i in range(n):
        body = " ".join(rng_words[(i * 37 + j) % len(rng_words)] for j in range(para_len))
        out.append(f"Passage {i}: {body}")
    return out


def _sync():
    torch.cuda.synchronize()


@torch.no_grad()
def time_concat(model, tok, chunks, question, device, max_new, seg_cap):
    # build one long sequence
    ctx = "".join(tok.decode(tok(p, add_special_tokens=False, truncation=True,
                  max_length=seg_cap).input_ids) for p in chunks)
    full = build_prompt(tok, ctx, question)
    ids = tok(full, return_tensors="pt", truncation=False).input_ids.to(device)
    _sync(); t0 = time.time()
    out = model(input_ids=ids, use_cache=True)
    pkv = out.past_key_values
    _sync(); t_prefill = time.time() - t0
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    _sync(); t0 = time.time()
    for _ in range(max_new - 1):
        out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    _sync(); t_decode = time.time() - t0
    return t_prefill, t_decode


@torch.no_grad()
def time_bank(model, tok, mgr, chunks, question, device, max_new, seg_cap, temp, scale):
    """Parallel-encoding bank read (ours: temp=scale=1; APE: temp=scale=0.9)."""
    cp = [build_prompt(tok, c, question) for c in chunks]
    enc = tok(cp, return_tensors="pt", padding=True, truncation=True, max_length=seg_cap)
    cids = enc["input_ids"].to(device); cattn = enc["attention_mask"].to(device)
    cm = context_mask_for(tok, cp, chunks, cids.shape[1], cattn, 4096).to(device)
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(temp, scale)
    # ---- prefill = capture (independent encoding) ----
    _sync(); t0 = time.time()
    cap_bs = 8
    mgr.start_capture(cm)
    for s in range(0, len(chunks), cap_bs):
        mgr.context_mask = cm[s:s + cap_bs].bool(); mgr.set_valid(cattn[s:s + cap_bs])
        model(input_ids=cids[s:s + cap_bs], attention_mask=cattn[s:s + cap_bs], use_cache=False)
    _sync(); t_prefill = time.time() - t0
    # ---- decode = query reads whole bank ----
    off = int(cattn.sum(1).max().item())
    qp = build_prompt(tok, "", question)
    enc = tok([qp], return_tensors="pt"); qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
    mgr.start_use()
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
    nxt_pos = off + qattn.sum(1)
    cur = qattn.clone(); nxt = qids.clone(); pkv = None
    _sync(); t0 = time.time()
    for step in range(max_new):
        mgr.set_valid(cur)
        mgr.set_query_rows(cur.bool() if step == 0 else torch.ones(1, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(1, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        cur = torch.cat([cur, torch.ones(1, 1, dtype=cur.dtype, device=device)], 1)
    _sync(); t_decode = time.time() - t0
    return t_prefill, t_decode


def median_time(fn, runs, warmup):
    for _ in range(warmup):
        fn(); torch.cuda.empty_cache()
    ps, ds = [], []
    for _ in range(runs):
        p, d = fn(); ps.append(p); ds.append(d); torch.cuda.empty_cache()
    return statistics.median(ps) * 1000, statistics.median(ds) * 1000


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu); device = f"cuda:{args.gpu}"
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    mgr = BatchCrossCache(list(range(len(model.model.layers))))
    mgr.register(model)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()
    question = "What is the capital city mentioned in passage 0?"

    print(f"{'N':>4} | {'concat pre/dec':>16} | {'ape pre/dec':>16} | {'ours pre/dec':>16}  (ms)", flush=True)
    for N in [int(x) for x in args.n_paras.split(",")]:
        chunks = synth_passages(tok, N, args.para_len, words)
        try:
            cp, cd = median_time(lambda: time_concat(model, tok, chunks, question, device, args.max_new, args.seg_cap), args.runs, args.warmup)
            c_str = f"{cp:6.0f}/{cd:5.0f}"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); c_str = "  OOM/  OOM"
        ap, ad = median_time(lambda: time_bank(model, tok, mgr, chunks, question, device, args.max_new, args.seg_cap, 0.9, 0.9), args.runs, args.warmup)
        op, od = median_time(lambda: time_bank(model, tok, mgr, chunks, question, device, args.max_new, args.seg_cap, 1.0, 1.0), args.runs, args.warmup)
        print(f"{N:>4} | {c_str:>16} | {ap:6.0f}/{ad:5.0f} | {op:6.0f}/{od:5.0f}", flush=True)


if __name__ == "__main__":
    main()
