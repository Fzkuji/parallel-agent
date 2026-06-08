"""Stage-0b DECISIVE probe: can generation-time thinking recover the cross-passage bridge
that independent encoding dropped from the bank KV?

For each gold-only sample (Qwen3-8B, fixed hook):
  (1) teacher think = full-attention concat-gold generates a <think>...</think> trace (it
      solves the 2nd hop in-context; we keep only traces whose answer hits gold).
  (2) FORCED-THINK bank-read: capture the gold passages INDEPENDENTLY into the bank, then
      feed the teacher's correct <think> as a prefix to the query and let the bank-read
      reader emit only the final answer (reading the independent bank).
  (3) NO-THINK bank-read: same bank, answer directly.

If forced-think >> no-think -> the bridge is RE-DERIVABLE from the bank once the reasoning
is supplied; thinking is the right lever, proceed to distill+GRPO.
If forced-think ~ no-think -> the bridge info is ABSENT/distorted in the KV; thinking can't
conjure it -> KILL the thinking project, pivot to a capture-time encoding fix.
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
def teacher_think(model, tok, gold_ctx, question, device, max_new):
    """Full-attention concat-gold: generate a <think>...</think><answer> trace."""
    prompt = LB_PROMPT.format(context=gold_ctx, input=question)
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=16000).to(device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    g = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    think = g.split("</think>")[0] + "</think>" if "</think>" in g else None
    ans, _ = extract_box_answer(g)
    return think, ans, g


@torch.no_grad()
def bank_capture(model, tok, mgr, chunks, question, device, max_plen):
    cp = [build_prompt(tok, c, question) for c in chunks]
    enc = tok(cp, return_tensors="pt", padding=True, truncation=True, max_length=max_plen)
    cids = enc["input_ids"].to(device); cattn = enc["attention_mask"].to(device)
    cm = context_mask_for(tok, cp, chunks, cids.shape[1], cattn, max_plen).to(device)
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(1.0, 1.0)
    mgr.start_capture(cm); mgr.set_valid(cattn)
    model(input_ids=cids, attention_mask=cattn, use_cache=False)
    return int(cattn.sum(1).max().item())


@torch.no_grad()
def bank_answer(model, tok, mgr, question, device, off, max_plen, max_new, think_prefix=None):
    """Bank-read decode. If think_prefix given, seed the query with the teacher's <think> so
    the reader answers AFTER the supplied reasoning."""
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    qp = build_prompt(tok, "", question)
    if think_prefix:
        qp = qp + think_prefix + "\n<answer>"   # seed reasoning + open the answer tag
    enc = tok([qp], return_tensors="pt", truncation=True, max_length=max_plen, add_special_tokens=False)
    qids = enc["input_ids"].to(device); qattn = torch.ones_like(qids)
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
        t = out.logits[:, -1].argmax(-1)
        t = torch.where(fin, torch.full_like(t, pad), t)
        fin = fin | (t == eos)
        gen = torch.cat([gen, t.unsqueeze(1)], 1)
        cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break
    txt = decode_texts(tok, gen, P, eos, pad)[0]
    return txt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/mnt/data/zichuanfu/models/Qwen3-8B")
    p.add_argument("--tasks", default="hotpotqa,2wikimqa")
    p.add_argument("--data-dir", default="/mnt/data/zichuanfu/longbench_export")
    p.add_argument("--num-q", type=int, default=30)
    p.add_argument("--teacher-max-new", type=int, default=768)
    p.add_argument("--ans-max-new", type=int, default=48)
    p.add_argument("--max-prompt-length", type=int, default=4096)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"; tok.truncation_side = "left"
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map="auto").eval()
    nl = model.config.num_hidden_layers
    mgr = BatchCrossCache(list(range(nl))); mgr.register(model)

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        path = os.path.join(args.data_dir, f"{task}.jsonl")
        data = [json.loads(l) for l in open(path) if l.strip()][: args.num_q]
        sums = {"oracle_think": 0.0, "nothink": 0.0, "forced": 0.0}; n = 0; n_teach_ok = 0
        for ex in data:
            ps = split_passages(ex["context"]); answers = ex["answers"]; q = ex["input"]
            gold = oracle_passages(ps, answers)
            gold_ctx = "\n\n".join(gold)
            think, t_ans, _ = teacher_think(model, tok, gold_ctx, q, device, args.teacher_max_new)
            t_f1 = best_f1(t_ans, answers)
            sums["oracle_think"] += t_f1
            teacher_ok = (think is not None) and (t_f1 >= 0.5)
            n_teach_ok += int(teacher_ok)
            mgr.set_enabled(True)
            off = bank_capture(model, tok, mgr, gold, q, device, args.max_prompt_length)
            nt = bank_answer(model, tok, mgr, q, device, off, args.max_prompt_length, args.ans_max_new)
            sums["nothink"] += best_f1(extract_box_answer(nt)[0], answers)
            # forced-think only meaningful when the teacher think is correct
            if teacher_ok:
                off2 = bank_capture(model, tok, mgr, gold, q, device, args.max_prompt_length)
                ft = bank_answer(model, tok, mgr, q, device, off2, args.max_prompt_length,
                                 args.ans_max_new, think_prefix=think)
                sums["forced"] += best_f1(extract_box_answer(ft)[0], answers)
            mgr.set_enabled(True)
            n += 1
            if n % 10 == 0:
                print(f"  [{task} {n}] oracle_think={100*sums['oracle_think']/n:.1f} "
                      f"nothink={100*sums['nothink']/n:.1f} "
                      f"forced(on {n_teach_ok} ok)={100*sums['forced']/max(1,n_teach_ok):.1f}", flush=True)
        print(f"{task}: oracle_think={100*sums['oracle_think']/n:.2f} "
              f"bank_nothink={100*sums['nothink']/n:.2f} "
              f"bank_forced_think={100*sums['forced']/max(1,n_teach_ok):.2f} "
              f"(teacher_ok={n_teach_ok}/{n})", flush=True)


if __name__ == "__main__":
    main()
