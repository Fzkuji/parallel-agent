#!/usr/bin/env python3
"""Probe a (large) teacher's TRUE gold-only-concat ability on LongBench multi-hop QA,
using the OFFICIAL LongBench prompt (no answer-tag system prompt), to decide whether it
is strong enough (>7B oracle ~66.6) to serve as a distillation teacher that can lift the
bank-read student ABOVE the 7B oracle. The earlier 12-pt score was a format artifact of
the answer-tag prompt; this probe uses the plain LongBench template + larger max_new."""
import argparse, os, re, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.bench_longbench import (
    best_f1, split_passages, oracle_passages, LB_PROMPT,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--tasks", default="2wikimqa,hotpotqa,musique")
    p.add_argument("--data-dir", default="/mnt/data/zichuanfu/longbench_export")
    p.add_argument("--num-q", type=int, default=100)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--gold-only", action="store_true", help="oracle passages only (else full context)")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"; tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", device_map="auto").eval()

    def gen(context, q):
        prompt = LB_PROMPT.format(context=context, input=q)
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=32000).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        path = os.path.join(args.data_dir, f"{task}.jsonl")
        data = [json.loads(l) for l in open(path) if l.strip()][: args.num_q]
        sc = 0.0; n = 0
        for ex in data:
            passages = split_passages(ex["context"])
            ctx = "\n\n".join(oracle_passages(passages, ex["answers"]) if args.gold_only
                              else passages)
            pred = gen(ctx, ex["input"])
            sc += best_f1(pred, ex["answers"]); n += 1
            if n % 25 == 0:
                print(f"  [{task} {n}/{len(data)}] qa_f1={100*sc/n:.1f}", flush=True)
        print(f"{task}: qa_f1={100*sc/n:.2f}  (gold_only={args.gold_only})", flush=True)


if __name__ == "__main__":
    main()
