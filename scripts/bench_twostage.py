#!/usr/bin/env python3
"""End-to-end two-stage read.

Stage 1 (parallel, cacheable): encode all N passages independently into the bank, each prefixed
with a "Passage i:" header; the select-trained model generates "<select>i,j</select>" while
reading the bank (supervised by gold supporting_facts at training time).
Stage 2 (tiny joint read): jointly re-encode ONLY the selected passages with plain concat and
answer. Cost = parallel prefill + a short joint prefill over 2-4 passages — stays in the
parallel-encoding cost class while restoring cross-passage interactions for the final read.

Reports SubEM (think/nothink for stage 2) + selection recall/precision/avg-k against gold.
"""
import argparse, os, sys, random, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.bench_distract import build_examples
from scripts.bench_standard import bp, subem, extract_answer, gen_concat
from scripts.eval_multiquery import context_mask_for


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", required=True, help="select-trained adapter(s), comma separated")
    p.add_argument("--dataset", default="2wikimultihopqa",
                   choices=["2wikimultihopqa", "hotpotqa", "musique"])
    p.add_argument("--flashrag-root",
                   default="/mnt/data/zichuanfu/.cache/huggingface/datasets/RUC-NLPIR___flash_rag_datasets")
    p.add_argument("--num-q", type=int, default=50)
    p.add_argument("--n-paras", type=int, default=4)
    p.add_argument("--think-modes", default="think,nothink", help="stage-2 answer modes")
    p.add_argument("--max-new-think", type=int, default=512)
    p.add_argument("--max-new-nothink", type=int, default=32)
    p.add_argument("--sel-max-new", type=int, default=24)
    p.add_argument("--seg-cap", type=int, default=768)
    p.add_argument("--max-plen", type=int, default=1600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
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
def gen_select(model, tok, mgr, question, device, off, max_new):
    """greedy decode from the bank until </select>."""
    qp = bp(tok, "", question, True)
    enc = tok([qp], return_tensors="pt", add_special_tokens=False)
    qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    mgr.start_use()
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
    nxt_pos = off + qattn.sum(1)
    cur = qattn.clone(); nxt = qids.clone(); pkv = None
    out_ids = []
    for step in range(max_new):
        mgr.set_valid(cur)
        mgr.set_query_rows(cur.bool() if step == 0 else torch.ones(1, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(1, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid,
                    past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        t = out.logits[:, -1].argmax(-1)
        if int(t) == eos:
            break
        out_ids.append(int(t))
        cur = torch.cat([cur, torch.ones(1, 1, dtype=cur.dtype, device=device)], 1)
        nxt = t.unsqueeze(1)
        text = tok.decode(out_ids)
        if "</select>" in text:
            break
    return tok.decode(out_ids)


def parse_select(text, n):
    m = re.search(r"<select>([\d,\s]+)</select>", text)
    if not m:
        m = re.search(r"<select>([\d,\s]+)", text)
    if not m:
        return None
    idx = sorted({int(x) for x in re.findall(r"\d+", m.group(1)) if 1 <= int(x) <= n})
    return idx or None


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu); device = f"cuda:{args.gpu}"
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    from peft import PeftModel
    for lp in args.lora_path.split(","):
        model = PeftModel.from_pretrained(model, lp).merge_and_unload().eval()
        print("loaded LoRA:", lp, flush=True)
    mgr = BatchCrossCache(list(range(len(model.model.layers)))); mgr.register(model)

    parsed, pool = build_examples(args.flashrag_root, args.dataset, args.num_q, args.seed)
    items = []
    for ex in parsed[:args.num_q]:
        gold = list(ex["gold"])
        need = max(0, args.n_paras - len(gold))
        distr = rng.sample(pool, min(need, len(pool)))
        paras = gold + distr
        idx = list(range(len(paras))); rng.shuffle(idx)
        paras = [paras[i] for i in idx]
        gold_pos = {pi for pi, oi in enumerate(idx) if oi < len(gold)}  # 0-based positions of gold
        items.append({"paras": paras, "gold_pos": gold_pos,
                      "question": ex["question"], "answers": ex["answer"]})

    tms = [t for t in args.think_modes.split(",") if t]
    acc = {t: 0.0 for t in tms}
    rec_full = 0; prec_n = 0.0; k_sum = 0; parse_fail = 0
    for it in items:
        headers = [f"Passage {i+1}: {p}" for i, p in enumerate(it["paras"])]
        off = capture(model, tok, mgr, headers, it["question"], device, args.seg_cap, args.max_plen)
        sel_text = gen_select(model, tok, mgr, it["question"], device, off, args.sel_max_new)
        idx = parse_select(sel_text, len(it["paras"]))
        mgr.bank = {}
        if idx is None:
            parse_fail += 1
            idx = list(range(1, len(it["paras"]) + 1))  # fallback: keep all
        sel0 = {i - 1 for i in idx}
        if it["gold_pos"] <= sel0:
            rec_full += 1
        prec_n += len(it["gold_pos"] & sel0) / max(1, len(sel0))
        k_sum += len(sel0)
        selected = [it["paras"][i] for i in sorted(sel0)]
        mgr.set_enabled(False)
        for t in tms:
            think = (t == "think")
            mn = args.max_new_think if think else args.max_new_nothink
            pred = gen_concat(model, tok, selected, it["question"], device, mn, args.seg_cap, think)
            acc[t] += subem(pred, it["answers"])
        mgr.set_enabled(True)
        torch.cuda.empty_cache()
    n = len(items)
    parts = [f"{t}={100*acc[t]/n:.1f}" for t in tms]
    print(f"== twostage {args.dataset} np{args.n_paras} n={n} ==  " + "  ".join(parts) +
          f"  sel_full_recall={100*rec_full/n:.1f}  sel_prec={100*prec_n/n:.1f}  "
          f"avg_k={k_sum/n:.2f}  parse_fail={parse_fail}", flush=True)


if __name__ == "__main__":
    main()
