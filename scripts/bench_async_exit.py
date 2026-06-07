#!/usr/bin/env python3
"""Async exit: when a query in a parallel multi-query batch finishes, drop it so the batch shrinks.

G queries decode in parallel against the shared bank; their answers have different lengths.
The naive loop keeps a finished query in the batch (padding it) and wastes its compute until the
slowest query ends. Async exit removes finished rows, so later decode steps process fewer
sequences. This is a serving win unique to the multi-query regime (a single-query method has
nothing to shrink). We measure decode wall-clock with vs without shrinking, over a spread of
answer lengths, against the shared bank.

Synthetic; single GPU; warmup + median.
"""
import argparse, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--G", type=int, default=16)
    p.add_argument("--ctx-len", type=int, default=1024)
    p.add_argument("--query-len", type=int, default=16)
    p.add_argument("--max-gen", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    return p.parse_args()


def _sync():
    torch.cuda.synchronize()


def _med(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


@torch.no_grad()
def build_bank(model, mgr, G, L, vocab, device):
    cids = torch.randint(1, vocab, (G, L), device=device); cattn = torch.ones_like(cids)
    cm = torch.ones(G, L, dtype=torch.bool, device=device)
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(1.0, 1.0)
    mgr.start_capture(cm); mgr.set_valid(cattn)
    model(input_ids=cids, attention_mask=cattn, use_cache=False)
    return int(cattn.sum(1).max().item())


@torch.no_grad()
def decode(model, mgr, off, G, ql, lengths, vocab, device, shrink):
    """Decode G queries reading the bank; each query i stops at lengths[i]. If shrink, drop finished rows."""
    qids = torch.randint(1, vocab, (G, ql), device=device); qattn = torch.ones_like(qids)
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
    nxt_pos = off + qattn.sum(1)
    mgr.start_use()
    cur = qattn.clone(); pkv = None; nxt = qids
    alive = torch.arange(G, device=device)                 # original indices still decoding
    done_at = torch.tensor(lengths, device=device)
    _sync(); t0 = time.perf_counter()
    for step in range(max(lengths)):
        n = nxt.shape[0]
        mgr.set_valid(cur); mgr.set_query_rows(cur.bool() if step == 0 else torch.ones(n, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(n, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        nxt = out.logits[:, -1:].argmax(-1)
        cur = torch.cat([cur, torch.ones(n, 1, dtype=cur.dtype, device=device)], 1)
        if shrink:
            keep = done_at[alive] > (step + 1)             # rows that still need more tokens
            if keep.sum() < n:
                alive = alive[keep]; nxt = nxt[keep]; cur = cur[keep]; nxt_pos = nxt_pos[keep]
                pkv = _gather_cache(pkv, keep)
            if alive.numel() == 0:
                break
    _sync()
    return time.perf_counter() - t0


def _gather_cache(pkv, keep):
    idx = keep.nonzero(as_tuple=True)[0]
    if hasattr(pkv, "reorder_cache"):
        try:
            pkv.reorder_cache(idx); return pkv
        except Exception:
            pass
    if hasattr(pkv, "layers"):                          # transformers >=4.54 cache refactor
        for layer in pkv.layers:
            if getattr(layer, "keys", None) is not None:
                layer.keys = layer.keys.index_select(0, idx)
                layer.values = layer.values.index_select(0, idx)
        return pkv
    for i in range(len(pkv.key_cache)):                 # legacy API
        if pkv.key_cache[i] is not None:
            pkv.key_cache[i] = pkv.key_cache[i].index_select(0, idx)
            pkv.value_cache[i] = pkv.value_cache[i].index_select(0, idx)
    return pkv


def main():
    args = parse_args()
    device = "cuda"
    vocab = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True).vocab_size
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa").to(device).eval()
    mgr = BatchCrossCache(list(range(model.config.num_hidden_layers))); mgr.register(model)
    G = args.G
    import random; rng = random.Random(0)
    # varied answer lengths: a few long, most short (realistic QA spread)
    lengths = sorted([rng.randint(2, args.max_gen) for _ in range(G)])

    def run(shrink):
        ts = []
        for _ in range(args.warmup):
            off = build_bank(model, mgr, G, args.ctx_len, vocab, device)
            decode(model, mgr, off, G, args.query_len, lengths, vocab, device, shrink)
        for _ in range(args.iters):
            off = build_bank(model, mgr, G, args.ctx_len, vocab, device)
            ts.append(decode(model, mgr, off, G, args.query_len, lengths, vocab, device, shrink))
        return _med(ts)

    naive = run(False); shrunk = run(True)
    print(f"G={G} answer lengths min/med/max = {lengths[0]}/{lengths[G//2]}/{lengths[-1]}", flush=True)
    print(f"decode (ms):  naive={naive*1e3:.1f}   async-exit(shrink)={shrunk*1e3:.1f}   speedup={naive/shrunk:.2f}x", flush=True)


if __name__ == "__main__":
    main()
