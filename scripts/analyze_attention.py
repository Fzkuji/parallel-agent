#!/usr/bin/env python3
"""Mechanism analysis: does training make the query attend the RIGHT sibling context?

For distributed-evidence multi-query groups, each query's gold evidence lies in some
sibling context(s). We record the query->bank attention mass per source context (first
decode step, all layers) and measure the fraction landing on the gold-bearing context.
Compare base vs trained -> evidence that training learns cross-context retrieval.
"""
import argparse, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.eval_multiquery import build_prompt, context_mask_for
from src.datasets import load_multiquery_hotpot_groups


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", default=None)
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--num-groups", type=int, default=40)
    p.add_argument("--max-prompt-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def gold_context_of(items):
    """which sibling context holds each query's gold? heuristic: the context whose paragraphs
    contain the answer string; fall back to 'has_supporting' own slice."""
    gold = []
    for i, it in enumerate(items):
        ans = it["references"][0].lower()
        hit = [j for j, jt in enumerate(items) if ans in jt["context"].lower()]
        gold.append(set(hit) if hit else {i})
    return gold


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"; tok.truncation_side = "left"
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device).eval()
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload().eval()
    nl = model.config.num_hidden_layers
    mgr = BatchCrossCache(list(range(nl))); mgr.register(model)

    groups = load_multiquery_hotpot_groups(
        split="validation", n_agents=args.n_agents, paragraphs_per_agent=6,
        max_groups=args.num_groups, seed=args.seed, only_bridge=True,
        require_min_supporting=2, cross_question_distractor_pool=True)

    gold_frac = []; self_frac = []
    for g in groups:
        items = g["items"]; G = len(items)
        cp = [build_prompt(tok, it["context"], it["question"]) for it in items]
        enc = tok(cp, return_tensors="pt", padding=True, truncation=True, max_length=args.max_prompt_length)
        cids = enc["input_ids"].to(device); cattn = enc["attention_mask"].to(device)
        cm = context_mask_for(tok, cp, [it["context"] for it in items], cids.shape[1], cattn, args.max_prompt_length).to(device)
        mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(1.0, 1.0)
        mgr.start_capture(cm); mgr.set_valid(cattn)
        model(input_ids=cids, attention_mask=cattn, use_cache=False)
        off = int(cattn.sum(1).max().item())
        qp = [build_prompt(tok, "", it["question"]) for it in items]
        enc = tok(qp, return_tensors="pt", padding=True, truncation=True, max_length=args.max_prompt_length)
        qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
        pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
        mgr.set_valid(qattn); mgr.set_query_rows(qattn.bool()); mgr.start_use()
        mgr.record_attn = True; mgr.attn_record = []
        model(input_ids=qids, attention_mask=qattn, position_ids=pos, use_cache=False)
        mgr.record_attn = False
        att = torch.stack(mgr.attn_record).mean(0)              # [G,G] avg over layers; mass per source ctx
        att = att / att.sum(dim=1, keepdim=True).clamp(min=1e-9)
        gold = gold_context_of(items)
        for i in range(G):
            gold_frac.append(sum(att[i, j].item() for j in gold[i]))
            self_frac.append(att[i, i].item())
    import statistics as st
    tag = "trained" if args.lora_path else "base"
    print(f"[{tag}] groups={len(groups)} n={len(gold_frac)}  "
          f"attn_on_gold_ctx={st.mean(gold_frac):.3f}  attn_on_own_slice={st.mean(self_frac):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
