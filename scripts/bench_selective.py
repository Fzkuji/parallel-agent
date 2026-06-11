#!/usr/bin/env python3
"""Selective read on the bench_standard pipeline (same data, same SubEM -> directly comparable).

Capture the bank ONCE, decode full-read recording per-context attention mass (start_relevance),
then re-decode at each read fraction via set_allowed(top-k by attention mass). The question:
at high distractor density (np16/np32) does dropping low-relevance contexts close the gap to
concat that full-read leaves open (np32: ours 30 vs concat 46)?
"""
import argparse, os, sys, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.eval_multiquery import context_mask_for, decode_texts
from scripts.bench_distract import build_examples, make_paras
from scripts.bench_standard import bp, subem, extract_answer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", default=None)
    p.add_argument("--dataset", default="2wikimultihopqa",
                   choices=["2wikimultihopqa", "hotpotqa", "musique"])
    p.add_argument("--flashrag-root",
                   default="/mnt/data/zichuanfu/.cache/huggingface/datasets/RUC-NLPIR___flash_rag_datasets")
    p.add_argument("--num-q", type=int, default=50)
    p.add_argument("--n-paras", type=int, default=8)
    p.add_argument("--fracs", default="0.5,0.25")
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--seg-cap", type=int, default=768)
    p.add_argument("--max-plen", type=int, default=1600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=5)
    return p.parse_args()


@torch.no_grad()
def capture(model, tok, mgr, chunks, question, device, seg_cap, max_plen):
    cp = [bp(tok, c, question, False) for c in chunks]
    enc = tok(cp, return_tensors="pt", padding=True, truncation=True,
              max_length=min(max_plen, seg_cap))
    cids = enc["input_ids"].to(device); cattn = enc["attention_mask"].to(device)
    cm = context_mask_for(tok, cp, chunks, cids.shape[1], cattn, max_plen).to(device)
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(1.0, 1.0)
    mgr.start_capture(cm)
    for s in range(0, len(chunks), 8):
        mgr.context_mask = cm[s:s + 8].bool(); mgr.set_valid(cattn[s:s + 8])
        model(input_ids=cids[s:s + 8], attention_mask=cattn[s:s + 8], use_cache=False)
    return int(cattn.sum(1).max().item())


@torch.no_grad()
def decode(model, tok, mgr, question, device, off, max_new, rec_c=0):
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    qp = bp(tok, "", question, False)
    enc = tok([qp], return_tensors="pt", add_special_tokens=False)
    qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
    mgr.start_use()
    if rec_c:
        mgr.start_relevance(rec_c)
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
    nxt_pos = off + qattn.sum(1)
    gen = qids.clone(); cur = qattn.clone()
    fin = torch.zeros(1, dtype=torch.bool, device=device); pkv = None; nxt = gen
    P = qids.shape[1]
    for step in range(max_new):
        mgr.set_valid(cur)
        mgr.set_query_rows(cur.bool() if step == 0 else torch.ones(1, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(1, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid,
                    past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        t = out.logits[:, -1].argmax(-1); t = torch.where(fin, torch.full_like(t, pad), t)
        fin = fin | (t == eos)
        gen = torch.cat([gen, t.unsqueeze(1)], 1)
        cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break
    rel = mgr.relevance() if rec_c else None
    return extract_answer(decode_texts(tok, gen, P, eos, pad)[0]), rel


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu); device = f"cuda:{args.gpu}"
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload().eval()
        print("loaded LoRA:", args.lora_path, flush=True)
    mgr = BatchCrossCache(list(range(len(model.model.layers)))); mgr.register(model)

    parsed, pool = build_examples(args.flashrag_root, args.dataset, args.num_q, args.seed)
    items = [{"paras": make_paras(ex, args.n_paras, pool, rng),
              "question": ex["question"], "answers": ex["answer"]}
             for ex in parsed[:args.num_q]]
    sweep = [float(x) for x in args.fracs.split(",") if x.strip()]
    acc = {"full": 0.0}; accf = {f: 0.0 for f in sweep}
    for it in items:
        C = len(it["paras"])
        off = capture(model, tok, mgr, it["paras"], it["question"], device, args.seg_cap, args.max_plen)
        pred, rel = decode(model, tok, mgr, it["question"], device, off, args.max_new, rec_c=C)
        acc["full"] += subem(pred, it["answers"])
        for f in sweep:
            k = max(1, round(f * C))
            a = torch.zeros(1, C, dtype=torch.bool, device=device)
            a.scatter_(1, rel.to(device).topk(k, dim=1).indices, True)
            mgr.set_allowed(a)
            pred2, _ = decode(model, tok, mgr, it["question"], device, off, args.max_new)
            mgr.set_allowed(None)
            accf[f] += subem(pred2, it["answers"])
        mgr.bank = {}
        torch.cuda.empty_cache()
    n = len(items)
    parts = [f"full={100*acc['full']/n:.1f}"] + [f"frac{f}={100*accf[f]/n:.1f}" for f in sweep]
    print(f"== {args.dataset} np{args.n_paras} n={n} seed={args.seed} ==  " + "  ".join(parts), flush=True)


if __name__ == "__main__":
    main()
