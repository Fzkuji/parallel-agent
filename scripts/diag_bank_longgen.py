#!/usr/bin/env python3
"""Diagnose: does bank-read degenerate into token repetition under LONG free generation?

Cross model (Qwen2.5 = no thinking, no QK-norm | Qwen3 = thinking, QK-norm) x generation length
(short ~30 tok answer | long ~256 tok). If only Qwen3-long degenerates -> thinking/QK-norm specific.
If both models degenerate at long generation -> it's a general bank-read decode bug (position /
LSE-merge accumulation), independent of thinking. This separation is the paper's mechanism analysis.

Metric: max_run = longest run of an identical token in the generation (high -> degenerated into
repetition). Also prints a snippet.
"""
import argparse, os, sys, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.bench_distract import build_examples, make_paras
from scripts.eval_multiquery import context_mask_for
from src.prompts import build_single_prompt
from src.templates import build_chat_prompt
from src.models import Question


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n-items", type=int, default=8)
    p.add_argument("--n-paras", type=int, default=4)
    p.add_argument("--seg-cap", type=int, default=768)
    p.add_argument("--long-new", type=int, default=256)
    p.add_argument("--short-new", type=int, default=32)
    p.add_argument("--gpu", type=int, default=5)
    return p.parse_args()


def prompt(tok, context, question):
    q = Question(qid="q", text=question, priority=1.0, answer_tokens=12, type_hint=None, references=[])
    sp, up = build_single_prompt(context, q, dataset="hotpot")
    return build_chat_prompt(tok, up, system_prompt=sp, enable_thinking=False)


def max_run(ids):
    """longest run of an identical token id (degeneration signal)."""
    best = cur = 1
    for i in range(1, len(ids)):
        cur = cur + 1 if ids[i] == ids[i - 1] else 1
        best = max(best, cur)
    return best


@torch.no_grad()
def bank_gen(model, tok, mgr, chunks, question, device, max_new, seg_cap):
    cp = [prompt(tok, c, question) for c in chunks]
    enc = tok(cp, return_tensors="pt", padding=True, truncation=True, max_length=seg_cap)
    cids = enc["input_ids"].to(device); cattn = enc["attention_mask"].to(device)
    cm = context_mask_for(tok, cp, chunks, cids.shape[1], cattn, 1600).to(device)
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(1.0, 1.0)
    mgr.start_capture(cm)
    for s in range(0, len(chunks), 8):
        mgr.context_mask = cm[s:s + 8].bool(); mgr.set_valid(cattn[s:s + 8])
        model(input_ids=cids[s:s + 8], attention_mask=cattn[s:s + 8], use_cache=False)
    off = int(cattn.sum(1).max().item())
    qp = prompt(tok, "", question)
    enc = tok([qp], return_tensors="pt", add_special_tokens=False)
    qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    mgr.start_use()
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
    nxt_pos = off + qattn.sum(1)
    gen = qids.clone(); cur = qattn.clone()
    fin = torch.zeros(1, dtype=torch.bool, device=device); pkv = None; nxt = gen
    P = qids.shape[1]
    out_ids = []
    for step in range(max_new):
        mgr.set_valid(cur)
        mgr.set_query_rows(cur.bool() if step == 0 else torch.ones(1, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(1, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        t = out.logits[:, -1].argmax(-1); t = torch.where(fin, torch.full_like(t, pad), t)
        fin = fin | (t == eos)
        out_ids.append(int(t))
        cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break
    return out_ids


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu); device = f"cuda:{args.gpu}"
    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    mgr = BatchCrossCache(list(range(len(model.model.layers)))); mgr.register(model)
    parsed, pool = build_examples(
        "/mnt/data/zichuanfu/.cache/huggingface/datasets/RUC-NLPIR___flash_rag_datasets",
        "2wikimultihopqa", args.n_items, 42)
    items = [{"paras": make_paras(ex, args.n_paras, pool, rng), "q": ex["question"]}
             for ex in parsed[:args.n_items]]

    for label, mn in [("short", args.short_new), ("long", args.long_new)]:
        runs = []
        for it in items:
            ids = bank_gen(model, tok, mgr, it["paras"], it["q"], device, mn, args.seg_cap)
            runs.append(max_run(ids) if ids else 0)
            torch.cuda.empty_cache()
        avg_run = sum(runs) / len(runs)
        degen = sum(1 for r in runs if r >= 5) / len(runs)
        # show one long-gen sample
        samp = tok.decode(bank_gen(model, tok, mgr, items[0]["paras"], items[0]["q"], device, mn, args.seg_cap))
        print(f"[{args.tag:10s} {label:5s} mn={mn:3d}] avg_max_run={avg_run:.1f}  "
              f"degen_frac={degen:.2f}  sample={samp[:70]!r}", flush=True)


if __name__ == "__main__":
    main()
