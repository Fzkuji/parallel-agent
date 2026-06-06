#!/usr/bin/env python3
"""Synthetic inference-efficiency benchmark for the multi-query cross-cache.

Pure speed measurement (no accuracy, no real data). For each method x G x
context-length, we feed RANDOM tokens of a FIXED length and measure two phases
separately so they stay comparable:

  - PREFILL: time to encode the context(s) and be ready to generate.
  - DECODE : time to generate a FIXED number of tokens (no EOS, forced loop) ->
             reported per token, so generation-length differences can't confound.

Methods (inference only, no backward):
  independent : each query encodes its OWN context (len L), decodes alone
  concat      : each query encodes the FULL union (G*L) in-context  (oracle cost)
  mq_readall  : two-phase cross-cache, each query reads the whole bank
  mq_topk     : cross-cache + selective top-k (maxsim selection cost INCLUDED)
  ape         : cross-cache read-all + APE realignment (manual attention path)

Setup follows APE (ICLR'25): generation length fixed (default 256), context
length swept, batch/parallelism swept. Single GPU, warmup + median over repeats.
Emits CSV to --out.
"""
import argparse, csv, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from transformers import AutoModelForCausalLM, AutoConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--methods", default="independent,concat,mq_readall,mq_topk,ape")
    p.add_argument("--G", default="2,4,8,16", help="parallelism (number of queries) to sweep")
    p.add_argument("--ctx-len", default="1024,2048,4096,8192", help="per-query context length to sweep")
    p.add_argument("--query-len", type=int, default=16)
    p.add_argument("--gen-len", type=int, default=256, help="fixed generated tokens (no EOS), APE-style")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--out", default="/tmp/bench_efficiency.csv")
    return p.parse_args()


def _rand(G, L, vocab, device):
    return torch.randint(1, vocab, (G, L), device=device)


def _sync():
    torch.cuda.synchronize()


def _median(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


@torch.no_grad()
def run_independent(model, mgr, G, L, ql, gen, vocab, device, concat=False):
    """Standard in-context baseline. concat=True -> each query holds the G*L union."""
    mgr.set_enabled(False)
    Lc = G * L if concat else L
    ids = torch.cat([_rand(G, Lc, vocab, device), _rand(G, ql, vocab, device)], dim=1)
    attn = torch.ones_like(ids)
    _sync(); t0 = time.perf_counter()
    out = model(input_ids=ids, attention_mask=attn, use_cache=True)
    pkv = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1)
    _sync(); t_pre = time.perf_counter() - t0
    cur = attn
    _sync(); t1 = time.perf_counter()
    for _ in range(gen):
        cur = torch.cat([cur, torch.ones(G, 1, dtype=cur.dtype, device=device)], 1)
        out = model(input_ids=nxt, attention_mask=cur, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1)
    _sync(); t_dec = time.perf_counter() - t1
    return t_pre, t_dec


@torch.no_grad()
def run_crosscache(model, mgr, G, L, ql, gen, vocab, device, topk=0, realign=(1.0, 1.0)):
    """Two-phase cross-cache. topk>0 -> selective (maxsim selection cost included)."""
    nl = model.config.num_hidden_layers
    cids = _rand(G, L, vocab, device); cattn = torch.ones_like(cids)
    cm = torch.ones(G, L, dtype=torch.bool, device=device)
    _sync(); t0 = time.perf_counter()
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(1.0, 1.0)
    mgr.start_capture(cm); mgr.set_valid(cattn)
    out1 = model(input_ids=cids, attention_mask=cattn, use_cache=False, output_hidden_states=(topk > 0))
    off = L
    qids = _rand(G, ql, vocab, device); qattn = torch.ones_like(qids)
    if 0 < topk < G:                                   # selective: maxsim top-k (cost counted in prefill)
        chs = out1.hidden_states[-1]
        mgr.set_enabled(False)
        qhs = model(input_ids=qids, attention_mask=qattn, output_hidden_states=True).hidden_states[-1]
        mgr.set_enabled(True)
        qn = torch.nn.functional.normalize(qhs.float(), dim=-1)
        cn = torch.nn.functional.normalize(chs.float(), dim=-1)
        R = torch.einsum("itd,jsd->ijts", qn, cn).max(dim=3).values.mean(dim=2)   # [G,G] maxsim
        topi = R.topk(topk, dim=1).indices
        allowed = torch.zeros(G, G, dtype=torch.bool, device=device)
        allowed.scatter_(1, topi, True); mgr.set_allowed(allowed)
    mgr.set_realign(*realign); mgr.start_use()
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
    nxt_pos = off + qattn.sum(1)
    cur = qattn.clone(); pkv = None; nxt = qids
    out = None
    for step in range(1 + gen):                        # step 0 = prefill the query prompt; rest = decode
        if step == 1:
            _sync(); t_pre = time.perf_counter() - t0; t1 = time.perf_counter()
        mgr.set_valid(cur)
        mgr.set_query_rows(cur.bool() if step == 0 else torch.ones(G, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(G, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        nxt = out.logits[:, -1:].argmax(-1)
        cur = torch.cat([cur, torch.ones(G, 1, dtype=cur.dtype, device=device)], 1)
    _sync(); t_dec = time.perf_counter() - t1
    return t_pre, t_dec


def bench_once(method, model, mgr, G, L, args, vocab, device):
    if method == "independent":
        return run_independent(model, mgr, G, L, args.query_len, args.gen_len, vocab, device)
    if method == "concat":
        return run_independent(model, mgr, G, L, args.query_len, args.gen_len, vocab, device, concat=True)
    if method == "mq_readall":
        return run_crosscache(model, mgr, G, L, args.query_len, args.gen_len, vocab, device)
    if method == "mq_topk":
        return run_crosscache(model, mgr, G, L, args.query_len, args.gen_len, vocab, device, topk=max(1, G // 2))
    if method == "ape":
        return run_crosscache(model, mgr, G, L, args.query_len, args.gen_len, vocab, device, realign=(0.9, 0.9))
    raise ValueError(method)


def main():
    args = parse_args()
    device = "cuda"
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    vocab = cfg.vocab_size
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device).eval()
    nl = model.config.num_hidden_layers
    mgr = BatchCrossCache(list(range(nl))); mgr.register(model)

    Gs = [int(x) for x in args.G.split(",")]
    Ls = [int(x) for x in args.ctx_len.split(",")]
    methods = args.methods.split(",")
    rows = []
    fout = open(args.out, "w", newline="")
    cw = csv.writer(fout)
    cw.writerow(["method", "G", "ctx_len", "prefill_ms_per_q", "decode_ms_per_tok_per_q", "peak_gb"])
    print(f"{'method':<12}{'G':>4}{'ctxL':>7}{'pre/q(ms)':>11}{'dec/tok/q(ms)':>15}{'peakGB':>8}", flush=True)
    for L in Ls:
        for G in Gs:
            for m in methods:
                try:
                    for _ in range(args.warmup):
                        bench_once(m, model, mgr, G, L, args, vocab, device)
                    torch.cuda.reset_peak_memory_stats()
                    pres, decs = [], []
                    for _ in range(args.iters):
                        tp, td = bench_once(m, model, mgr, G, L, args, vocab, device)
                        pres.append(tp); decs.append(td)
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    pre_q = _median(pres) / G * 1e3                     # ms per question
                    dec_q = _median(decs) / G / args.gen_len * 1e3      # ms per token per question
                    row = [m, G, L, round(pre_q, 3), round(dec_q, 4), round(peak, 2)]
                    rows.append(row); cw.writerow(row); fout.flush()
                    print(f"{m:<12}{G:>4}{L:>7}{pre_q:>11.3f}{dec_q:>15.4f}{peak:>8.2f}", flush=True)
                except RuntimeError as e:
                    msg = "OOM" if "out of memory" in str(e).lower() else "ERR"
                    print(f"{m:<12}{G:>4}{L:>7}{'':>11}{msg:>15}", flush=True)
                    torch.cuda.empty_cache()
    fout.close()
    print(f"\nwrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
