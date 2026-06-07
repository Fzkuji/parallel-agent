#!/usr/bin/env python3
"""Route-2 (RAFT/best-of-N) go/no-go probe, NO training.

For each LongBench question, the trained bank reader reads ALL passages (the deployed
`ours` arm, with distractors) and we SAMPLE N answers at temperature T. We report:
  greedy    = qa_f1 of the greedy (argmax) bank read           (== ours)
  best_of_N = mean over questions of max-over-N qa_f1          (the RAFT ceiling)
  oracle    = qa_f1 of greedy full-attention over gold passages (the bar to beat)

GATE: if best_of_N > oracle on a dataset, RAFT can plausibly push greedy toward that
ceiling (SFT-on-winners recovers part of the greedy->best-of-N gap) and the route can
exceed oracle. If best_of_N <= oracle, the encoding deficit exceeds the sampling gap and
RAFT cannot beat oracle on that dataset -> pivot to Route 3.

Reuses bench_distract.bank_read's capture, and bench_longbench's metric/splits/oracle.
"""
import argparse, os, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.eval_multiquery import build_prompt, context_mask_for, decode_texts
from scripts.bench_longbench import best_f1, split_passages, oracle_passages, LB_PROMPT
from src.inference import extract_box_answer


@torch.no_grad()
def capture_bank(model, tok, mgr, chunks, question, device, max_plen):
    cp = [build_prompt(tok, c, question) for c in chunks]
    enc = tok(cp, return_tensors="pt", padding=True, truncation=True, max_length=max_plen)
    cids = enc["input_ids"].to(device); cattn = enc["attention_mask"].to(device)
    cm = context_mask_for(tok, cp, chunks, cids.shape[1], cattn, max_plen).to(device)
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(1.0, 1.0)
    mgr.start_capture(cm); mgr.set_valid(cattn)
    model(input_ids=cids, attention_mask=cattn, use_cache=False)
    return int(cattn.sum(1).max().item())


@torch.no_grad()
def sample_one(model, tok, mgr, question, device, off, max_plen, max_new, temp):
    """One sampled bank-read decode (temp>0 -> multinomial; temp==0 -> greedy)."""
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    qp = build_prompt(tok, "", question)
    enc = tok([qp], return_tensors="pt", padding=True, truncation=True, max_length=max_plen)
    qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
    mgr.start_use()
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
    nxt_pos = off + qattn.sum(1)
    gen = qids.clone(); cur = qattn.clone()
    fin = torch.zeros(1, dtype=torch.bool, device=device); pkv = None; nxt = gen
    P = qids.shape[1]
    for step in range(max_new):
        mgr.set_valid(cur)
        mgr.set_query_rows(cur.bool() if step == 0
                           else torch.ones(1, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(1, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid,
                    past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        logits = out.logits[:, -1]
        if temp <= 0:
            t = logits.argmax(-1)
        else:
            t = torch.multinomial((logits / temp).softmax(-1), 1).squeeze(-1)
        t = torch.where(fin, torch.full_like(t, pad), t)
        fin = fin | (t == eos)
        gen = torch.cat([gen, t.unsqueeze(1)], 1)
        cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break
    return decode_texts(tok, gen, P, eos, pad)[0]


@torch.no_grad()
def oracle_f1(model, tok, gold_ctx, question, device, max_new):
    prompt = LB_PROMPT.format(context=gold_ctx, input=question)
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=16000).to(device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", required=True)
    p.add_argument("--tasks", default="2wikimqa,hotpotqa,musique")
    p.add_argument("--data-dir", default="/mnt/data/zichuanfu/longbench_export")
    p.add_argument("--num-q", type=int, default=50)
    p.add_argument("--raft-n", type=int, default=16)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--max-prompt-length", type=int, default=16000)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"; tok.truncation_side = "left"
    device = "cuda"
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device).eval()
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.lora_path).merge_and_unload().eval()
    nl = model.config.num_hidden_layers
    mgr = BatchCrossCache(list(range(nl))); mgr.register(model)

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        path = os.path.join(args.data_dir, f"{task}.jsonl")
        data = [json.loads(l) for l in open(path) if l.strip()][: args.num_q]
        g_sum = bo_sum = orc_sum = 0.0; n = 0
        for ex in data:
            passages = split_passages(ex["context"]); answers = ex["answers"]; q = ex["input"]
            # ours bank-read (all passages, with distractors): greedy + N samples
            off = capture_bank(model, tok, mgr, passages, q, device, args.max_prompt_length)
            greedy = extract_box_answer(sample_one(model, tok, mgr, q, device, off,
                                                   args.max_prompt_length, args.max_new, 0.0))[0]
            g_f1 = best_f1(greedy, answers)
            best = g_f1
            for _ in range(args.raft_n):
                s = extract_box_answer(sample_one(model, tok, mgr, q, device, off,
                                                  args.max_prompt_length, args.max_new, args.temp))[0]
                best = max(best, best_f1(s, answers))
            # oracle (full attention, gold passages only)
            gold = "\n\n".join(oracle_passages(passages, answers))
            mgr.set_enabled(False)
            orc = extract_box_answer(oracle_f1(model, tok, gold, q, device, args.max_new))[0]
            mgr.set_enabled(True)
            o_f1 = best_f1(orc, answers)
            g_sum += g_f1; bo_sum += best; orc_sum += o_f1; n += 1
            if n % 25 == 0:
                print(f"  [{task} {n}/{len(data)}] greedy={100*g_sum/n:.1f} "
                      f"best_of_{args.raft_n}={100*bo_sum/n:.1f} oracle={100*orc_sum/n:.1f}", flush=True)
        bo = 100 * bo_sum / n; orc = 100 * orc_sum / n; gd = 100 * g_sum / n
        gate = "PASS (RAFT viable)" if bo > orc else "FAIL (encoding deficit > sampling gap)"
        print(f"{task}: greedy={gd:.2f} best_of_{args.raft_n}={bo:.2f} oracle={orc:.2f} -> {gate}", flush=True)


if __name__ == "__main__":
    main()
