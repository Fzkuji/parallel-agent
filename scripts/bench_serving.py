#!/usr/bin/env python3
"""Multi-query SERVING efficiency: amortize the pool encoding across a query stream.

The shared evidence pool is encoded ONCE into a bank. Every incoming query -- whether
present at the start or arriving later (dynamic insertion) -- is answered by reading
that bank, paying only its own short prefill + bank-read. Concat instead re-encodes
the whole pool for EVERY query. So for a stream of Q queries over a pool of G contexts:

    CrossKV total = t_bank (once) + Q * t_read
    Concat  total = Q * t_concat        (t_concat re-encodes the pool each time)

As Q grows, CrossKV amortizes the one-time encoding; Concat pays it Q times. This is the
serving regime APE (single query) and Concat structurally cannot match: build once, serve
an open-ended, changing set of queries -- concurrent reads, async exits, mid-flight joins.

Synthetic fixed-length inputs; single GPU; warmup + median. Reports t_bank / t_read /
t_concat (ms) and the break-even and steady-state speedup over a query stream.
"""
import argparse, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from transformers.models.qwen2.modeling_qwen2 import repeat_kv  # noqa: F401  (mechanism import parity)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--G", default="4,8,16", help="pool size = number of contexts")
    p.add_argument("--ctx-len", type=int, default=2048, help="tokens per context")
    p.add_argument("--query-len", type=int, default=16)
    p.add_argument("--gen-len", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--stream", default="1,2,4,8,16,32,64", help="query-stream sizes Q to report amortization for")
    return p.parse_args()


def _sync():
    torch.cuda.synchronize()


def _med(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


@torch.no_grad()
def time_bank(model, mgr, G, L, vocab, device):
    """Encode G contexts once into the shared bank."""
    cids = torch.randint(1, vocab, (G, L), device=device)
    cattn = torch.ones_like(cids)
    cm = torch.ones(G, L, dtype=torch.bool, device=device)
    _sync(); t0 = time.perf_counter()
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(1.0, 1.0)
    mgr.start_capture(cm); mgr.set_valid(cattn)
    model(input_ids=cids, attention_mask=cattn, use_cache=False)
    _sync()
    return time.perf_counter() - t0, int(cattn.sum(1).max().item())


@torch.no_grad()
def time_read(model, mgr, off, ql, gen, vocab, device, nq=1):
    """Answer nq queries that READ the pre-built bank (no re-encoding)."""
    qids = torch.randint(1, vocab, (nq, ql), device=device); qattn = torch.ones_like(qids)
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
    nxt_pos = off + qattn.sum(1)
    mgr.start_use()
    cur = qattn.clone(); pkv = None; nxt = qids
    _sync(); t0 = time.perf_counter()
    for step in range(1 + gen):
        mgr.set_valid(cur)
        mgr.set_query_rows(cur.bool() if step == 0 else torch.ones(nq, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(nq, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        nxt = out.logits[:, -1:].argmax(-1)
        cur = torch.cat([cur, torch.ones(nq, 1, dtype=cur.dtype, device=device)], 1)
    _sync()
    return time.perf_counter() - t0


@torch.no_grad()
def time_concat(model, mgr, G, L, ql, gen, vocab, device):
    """Concat: ONE query re-encodes the whole pool (G*L) in-context, then decodes."""
    mgr.set_enabled(False)
    ids = torch.randint(1, vocab, (1, G * L + ql), device=device); attn = torch.ones_like(ids)
    _sync(); t0 = time.perf_counter()
    out = model(input_ids=ids, attention_mask=attn, use_cache=True)
    pkv = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1); cur = attn
    for _ in range(gen):
        cur = torch.cat([cur, torch.ones(1, 1, dtype=cur.dtype, device=device)], 1)
        out = model(input_ids=nxt, attention_mask=cur, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1)
    _sync()
    mgr.set_enabled(True)
    return time.perf_counter() - t0


def main():
    args = parse_args()
    device = "cuda"
    vocab = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True).vocab_size
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device).eval()
    mgr = BatchCrossCache(list(range(model.config.num_hidden_layers))); mgr.register(model)
    Qs = [int(x) for x in args.stream.split(",")]

    print(f"{'G':>4}{'t_bank(ms)':>12}{'t_read(ms)':>12}{'t_concat(ms)':>14}  amortized total over a Q-stream (s)")
    for G in [int(x) for x in args.G.split(",")]:
        L = args.ctx_len
        try:
            for _ in range(args.warmup):
                _, off = time_bank(model, mgr, G, L, vocab, device)
                time_read(model, mgr, off, args.query_len, args.gen_len, vocab, device)
                time_concat(model, mgr, G, L, args.query_len, args.gen_len, vocab, device)
            tb, tr, tc = [], [], []
            for _ in range(args.iters):
                b, off = time_bank(model, mgr, G, L, vocab, device); tb.append(b)
                tr.append(time_read(model, mgr, off, args.query_len, args.gen_len, vocab, device))
                tc.append(time_concat(model, mgr, G, L, args.query_len, args.gen_len, vocab, device))
            B, R, C = _med(tb), _med(tr), _med(tc)
            # amortized total time for a stream of Q queries
            streams = []
            for Q in Qs:
                cross = B + Q * R
                concat = Q * C
                streams.append(f"Q={Q}:{concat / cross:.1f}x")
            print(f"{G:>4}{B*1e3:>12.1f}{R*1e3:>12.1f}{C*1e3:>14.1f}  " + " ".join(streams), flush=True)
        except RuntimeError as e:
            msg = "OOM" if "out of memory" in str(e).lower() else "ERR"
            print(f"{G:>4}  {msg}", flush=True); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
