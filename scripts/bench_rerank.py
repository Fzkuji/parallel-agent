#!/usr/bin/env python3
"""Likelihood reranking over sampled candidates: pass@8 is 92 while greedy is 74 — the policy's
distribution almost always contains the right answer; selection is the bottleneck. For each
question: capture the bank once, sample N completions, extract distinct candidate answers, then
score each candidate by the mean token logprob of the canonical answer continuation
"<answer>X</answer>" under the SAME bank-read forward, and pick the argmax.
"""
import argparse, os, sys, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.bench_distract import build_examples, make_paras
from scripts.bench_standard import bp, subem, extract_answer
from scripts.bench_selective import capture
from scripts.train_grpo_bank import sample_group, left_pad
from src.evaluation.basic import normalize_answer


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
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--think", action="store_true")
    p.add_argument("--max-new", type=int, default=320)
    p.add_argument("--seg-cap", type=int, default=768)
    p.add_argument("--max-plen", type=int, default=1600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def score_candidates(model, tok, mgr, question, cands, device, off):
    """mean token logprob of '<answer>X</answer>' under the bank-read, per candidate."""
    qp = bp(tok, "", question, False)
    qids = tok(qp, add_special_tokens=False)["input_ids"]
    ids_l, lab_l = [], []
    for c in cands:
        aids = tok(f"<answer>{c}</answer>", add_special_tokens=False)["input_ids"]
        ids_l.append(qids + aids)
        lab_l.append([-100] * len(qids) + aids)
    pad = tok.pad_token_id or tok.eos_token_id
    ids, attn = left_pad(ids_l, pad, device)
    labels, _ = left_pad(lab_l, -100, device)
    labels = labels.masked_fill(attn == 0, -100)
    pos = (attn.long().cumsum(1) - 1).clamp(min=0) + off
    mgr.set_valid(attn); mgr.set_query_rows(attn.bool()); mgr.start_use()
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
    logp = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
    lab = labels[:, 1:]
    mask = lab != -100
    g = torch.gather(logp, 2, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return ((g * mask).sum(1) / mask.sum(1).clamp(min=1)).tolist()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu); device = f"cuda:{args.gpu}"
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
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
    mgr = BatchCrossCache(list(range(len(model.model.layers)))); mgr.register(model)

    parsed, pool = build_examples(args.flashrag_root, args.dataset, args.num_q, args.seed)
    items = [{"paras": make_paras(ex, args.n_paras, pool, rng),
              "question": ex["question"], "answers": ex["answer"]}
             for ex in parsed[:args.num_q]]

    tot_rr = 0.0; tot_any = 0.0
    for it in items:
        off = capture(model, tok, mgr, it["paras"], it["question"], device, args.seg_cap, args.max_plen)
        with torch.no_grad():
            texts, _, _ = sample_group(model, tok, mgr, it["question"], device, off,
                                       args.n_samples, args.max_new, args.temp, args.think)
        cands, seen = [], set()
        for t in texts:
            a = extract_answer(t).strip()
            a = a.split("</answer>")[0].replace("<answer>", "").strip() or a
            k = normalize_answer(a)
            if a and k and k not in seen:
                seen.add(k); cands.append(a[:80])
        if not cands:
            mgr.bank = {}; continue
        scores = score_candidates(model, tok, mgr, it["question"], cands, device, off)
        pick = cands[max(range(len(cands)), key=lambda i: scores[i])]
        tot_rr += subem(pick, it["answers"])
        tot_any += 1.0 if any(subem(c, it["answers"]) for c in cands) else 0.0
        mgr.bank = {}
        torch.cuda.empty_cache()
    n = len(items)
    print(f"== {args.dataset} np{args.n_paras} n={n} N={args.n_samples} T={args.temp} "
          f"think={args.think} ==  rerank={100*tot_rr/n:.1f}  ceiling(any)={100*tot_any/n:.1f}", flush=True)


if __name__ == "__main__":
    main()
