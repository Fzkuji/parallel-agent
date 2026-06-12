#!/usr/bin/env python3
"""Self-consistency over the bank-read decode: sample N completions (temperature), majority-vote
the extracted answers. Decode is cheap under parallel encoding (bank prefill amortized across all
N samples in one batch), so this is an inference-time accuracy lever orthogonal to training.
"""
import argparse, os, sys, random
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.bench_distract import build_examples, make_paras
from scripts.bench_standard import bp, subem, extract_answer
from scripts.bench_selective import capture
from scripts.train_grpo_bank import sample_group
from src.evaluation.basic import normalize_answer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", default=None, help="comma-separated adapters merged in order")
    p.add_argument("--dataset", default="2wikimultihopqa",
                   choices=["2wikimultihopqa", "hotpotqa", "musique"])
    p.add_argument("--flashrag-root",
                   default="/mnt/data/zichuanfu/.cache/huggingface/datasets/RUC-NLPIR___flash_rag_datasets")
    p.add_argument("--num-q", type=int, default=50)
    p.add_argument("--n-paras", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--think", action="store_true")
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--seg-cap", type=int, default=768)
    p.add_argument("--max-plen", type=int, default=1600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


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

    tot_vote = 0.0; tot_any = 0.0
    for it in items:
        off = capture(model, tok, mgr, it["paras"], it["question"], device, args.seg_cap, args.max_plen)
        with torch.no_grad():
            texts, _, _ = sample_group(model, tok, mgr, it["question"], device, off,
                                       args.n_samples, args.max_new, args.temp, args.think)
        preds = [normalize_answer(extract_answer(t)) for t in texts]
        preds = [p for p in preds if p]
        vote = Counter(preds).most_common(1)[0][0] if preds else ""
        tot_vote += subem(vote, it["answers"])
        tot_any += 1.0 if any(subem(p, it["answers"]) for p in preds) else 0.0
        mgr.bank = {}
        torch.cuda.empty_cache()
    n = len(items)
    print(f"== {args.dataset} np{args.n_paras} n={n} N={args.n_samples} T={args.temp} "
          f"think={args.think} ==  vote={100*tot_vote/n:.1f}  any(pass@N)={100*tot_any/n:.1f}", flush=True)


if __name__ == "__main__":
    main()
