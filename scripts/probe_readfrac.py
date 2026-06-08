"""Stage-C read-fraction go/no-go probe (training-free). For each question: capture the bank,
greedy-decode once recording per-context attention mass, then re-decode reading only the top-k
contexts (k = full, 0.5C, 0.3C) via set_allowed. If accuracy at 0.5C/0.3C stays within ~1pt of
full read, the read-fraction Pareto reward is viable (the model already reaches the answer from a
query-selected sub-bank) and we train it; else ship it OFF."""
import argparse, os, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.eval_multiquery import build_prompt, context_mask_for, decode_texts
from scripts.bench_longbench import best_f1, split_passages
from src.inference import extract_box_answer
from src.templates import set_think_tokens


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
def decode(model, tok, mgr, question, device, off, max_plen, max_new, rec_c=0):
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    qp = build_prompt(tok, "", question)
    enc = tok([qp], return_tensors="pt", truncation=True, max_length=max_plen)
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
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid, past_key_values=pkv, use_cache=True)
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
    rel = mgr.relevance() if rec_c else None
    return decode_texts(tok, gen, P, eos, pad)[0], rel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", required=True)
    p.add_argument("--tasks", default="2wikimqa,hotpotqa")
    p.add_argument("--data-dir", default="/mnt/data/zichuanfu/longbench_export")
    p.add_argument("--num-q", type=int, default=50)
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--max-prompt-length", type=int, default=12000)
    p.add_argument("--think", action="store_true")
    args = p.parse_args()
    set_think_tokens(args.think)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"; tok.truncation_side = "left"
    device = "cuda"
    base = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                                trust_remote_code=True, attn_implementation="sdpa").to(device).eval()
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.lora_path).merge_and_unload().eval()
    nl = model.config.num_hidden_layers
    mgr = BatchCrossCache(list(range(nl))); mgr.register(model)

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        data = [json.loads(l) for l in open(os.path.join(args.data_dir, f"{task}.jsonl")) if l.strip()][:args.num_q]
        acc = {"full": 0.0, "0.5C": 0.0, "0.3C": 0.0}; frac = {"0.5C": 0.0, "0.3C": 0.0}; n = 0
        for ex in data:
            ps = split_passages(ex["context"]); C = len(ps); ans = ex["answers"]; q = ex["input"]
            off = capture_bank(model, tok, mgr, ps, q, device, args.max_prompt_length)
            full, rel = decode(model, tok, mgr, q, device, off, args.max_prompt_length, args.max_new, rec_c=C)
            acc["full"] += best_f1(extract_box_answer(full)[0], ans)
            for tag, kf in [("0.5C", 0.5), ("0.3C", 0.3)]:
                k = max(1, round(kf * C))
                a = torch.zeros(1, C, dtype=torch.bool, device=device)
                a.scatter_(1, rel.to(device).topk(k, dim=1).indices, True)
                mgr.set_allowed(a)
                out, _ = decode(model, tok, mgr, q, device, off, args.max_prompt_length, args.max_new)
                mgr.set_allowed(None)
                acc[tag] += best_f1(extract_box_answer(out)[0], ans)
                frac[tag] += k / C
            mgr.bank = {}; n += 1
        print(f"{task} (n={n}): full={100*acc['full']/n:.1f}  "
              f"0.5C={100*acc['0.5C']/n:.1f}(read {frac['0.5C']/n:.2f})  "
              f"0.3C={100*acc['0.3C']/n:.1f}(read {frac['0.3C']/n:.2f})", flush=True)


if __name__ == "__main__":
    main()
