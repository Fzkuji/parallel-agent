#!/usr/bin/env python3
"""Standard single-turn QA accuracy: the BASELINE table that everything else is measured against.

Same passage set (gold + distractors), same question, asked directly. We cross two axes:
  encoding : concat (full attention, the standard upper bound) | ape | ours (independent bank-read)
  thinking : think (Qwen3 <think>...; max_new large) | no-think (direct answer; max_new small)

Reports SubEM. The two diffs that matter:
  think vs no-think  (does reasoning help, per encoding)
  concat vs ours     (cost of independent encoding, per thinking mode)

Why this exists: bank_read on a thinking model with small max_new spends every token reasoning and
never reaches the answer -> SubEM floor (~8%). Separating think/no-think and giving think enough
tokens removes that artifact and gives an honest accuracy baseline.
"""
import argparse, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.eval_multiquery import context_mask_for, decode_texts
from scripts.bench_distract import build_examples, make_paras
from src.templates import build_chat_prompt
from src.prompts import build_single_prompt
from src.models import Question
from src.evaluation.basic import normalize_answer
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", default=None)
    p.add_argument("--dataset", default="2wikimultihopqa",
                   choices=["2wikimultihopqa", "hotpotqa", "musique"])
    p.add_argument("--flashrag-root",
                   default="/mnt/data/zichuanfu/.cache/huggingface/datasets/RUC-NLPIR___flash_rag_datasets")
    p.add_argument("--num-q", type=int, default=50)
    p.add_argument("--n-paras", type=int, default=4)
    p.add_argument("--encodings", default="concat,ape,ours")
    p.add_argument("--think-modes", default="think,nothink")
    p.add_argument("--max-new-think", type=int, default=256)
    p.add_argument("--max-new-nothink", type=int, default=32)
    p.add_argument("--seg-cap", type=int, default=768)
    p.add_argument("--max-plen", type=int, default=1600)
    p.add_argument("--ape-temp", type=float, default=0.9)
    p.add_argument("--ape-scale", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=5)
    return p.parse_args()


def subem(pred, answers):
    p = normalize_answer(pred)
    return 1.0 if any(normalize_answer(a) and normalize_answer(a) in p for a in answers) else 0.0


def bp(tok, context, question, think):
    """build a chat prompt; enable_thinking toggles Qwen3 <think>."""
    q = Question(qid="q", text=question, priority=1.0, answer_tokens=12, type_hint=None, references=[])
    sp, up = build_single_prompt(context, q, dataset="hotpot")
    return build_chat_prompt(tok, up, system_prompt=sp, enable_thinking=think)


def extract_answer(text):
    """strip a trailing <think>...</think> if present, keep what follows (the actual answer)."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


@torch.no_grad()
def gen_concat(model, tok, chunks, question, device, max_new, seg_cap, think):
    ctx = "".join(tok.decode(tok(p, add_special_tokens=False, truncation=True,
                  max_length=seg_cap).input_ids) + "\n" for p in chunks)
    full = bp(tok, ctx, question, think)
    ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)[0]
    return extract_answer(tok.decode(out[ids.shape[1]:], skip_special_tokens=True))


@torch.no_grad()
def gen_bank(model, tok, mgr, chunks, question, device, max_new, seg_cap, max_plen, think, temp, scale):
    """ours/ape: independent-encoding bank read."""
    cp = [bp(tok, c, question, think) for c in chunks]
    enc = tok(cp, return_tensors="pt", padding=True, truncation=True, max_length=min(max_plen, seg_cap))
    cids = enc["input_ids"].to(device); cattn = enc["attention_mask"].to(device)
    cm = context_mask_for(tok, cp, chunks, cids.shape[1], cattn, max_plen).to(device)
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(temp, scale)
    mgr.start_capture(cm)
    cap_bs = 8
    for s in range(0, len(chunks), cap_bs):
        mgr.context_mask = cm[s:s + cap_bs].bool(); mgr.set_valid(cattn[s:s + cap_bs])
        model(input_ids=cids[s:s + cap_bs], attention_mask=cattn[s:s + cap_bs], use_cache=False)
    off = int(cattn.sum(1).max().item())
    qp = bp(tok, "", question, think)
    enc = tok([qp], return_tensors="pt", add_special_tokens=False)
    qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    mgr.start_use()
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
    nxt_pos = off + qattn.sum(1)
    gen = qids.clone(); cur = qattn.clone()
    fin = torch.zeros(1, dtype=torch.bool, device=device); pkv = None; nxt = gen
    P = qids.shape[1]
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
        gen = torch.cat([gen, t.unsqueeze(1)], 1)
        cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break
    return extract_answer(decode_texts(tok, gen, P, eos, pad)[0])


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
        for lp in args.lora_path.split(","):
            model = PeftModel.from_pretrained(model, lp).merge_and_unload().eval()
            print("loaded LoRA:", lp, flush=True)
    mgr = BatchCrossCache(list(range(len(model.model.layers))))
    mgr.register(model)

    parsed, pool = build_examples(args.flashrag_root, args.dataset, args.num_q, args.seed)
    items = []
    for ex in parsed[:args.num_q]:
        items.append({"paras": make_paras(ex, args.n_paras, pool, rng),
                      "question": ex["question"], "answers": ex["answer"]})

    encs = args.encodings.split(","); tms = args.think_modes.split(",")
    print(f"== {args.dataset}  n={len(items)}  n_paras={args.n_paras} ==", flush=True)
    print(f"{'encoding':>8} | {'think':>6} | {'nothink':>7}", flush=True)
    results = {}
    for enc in encs:
        row = {}
        for tm in tms:
            think = (tm == "think")
            mn = args.max_new_think if think else args.max_new_nothink
            mgr.set_enabled(enc != "concat")
            tot = 0.0
            for it in items:
                if enc == "concat":
                    pred = gen_concat(model, tok, it["paras"], it["question"], device, mn, args.seg_cap, think)
                else:
                    temp, scale = (args.ape_temp, args.ape_scale) if enc == "ape" else (1.0, 1.0)
                    pred = gen_bank(model, tok, mgr, it["paras"], it["question"], device, mn,
                                    args.seg_cap, args.max_plen, think, temp, scale)
                tot += subem(pred, it["answers"])
                torch.cuda.empty_cache()
            row[tm] = 100 * tot / len(items)
        results[enc] = row
        print(f"{enc:>8} | {row.get('think', float('nan')):6.1f} | {row.get('nothink', float('nan')):7.1f}", flush=True)


if __name__ == "__main__":
    main()
