#!/usr/bin/env python3
"""Per-passage relevance scoring for two-stage selection — diagnosis-driven redesign.

The generative <select> head shows a first/last positional bias at 16 segments (middle passages
nearly never selected; pure recall 0/30): with all segments sharing the same position range, the
query's attention competition drowns middle segments. Fix under test: score each passage IN
ISOLATION (set_allowed = that one segment), so there is no cross-segment competition at all.
Score = logP(" yes") on a relevance probe. Reports recall@k of the gold set for k=2..8.
"""
import argparse, os, sys, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.bench_distract import build_examples
from scripts.bench_standard import bp
from scripts.bench_twostage import capture
from scripts.train_grpo_bank import left_pad


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", default=None)
    p.add_argument("--dataset", default="2wikimultihopqa")
    p.add_argument("--flashrag-root",
                   default="/mnt/data/zichuanfu/.cache/huggingface/datasets/RUC-NLPIR___flash_rag_datasets")
    p.add_argument("--num-q", type=int, default=50)
    p.add_argument("--n-paras", type=int, default=16)
    p.add_argument("--seg-cap", type=int, default=768)
    p.add_argument("--max-plen", type=int, default=1600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--shard", default="0/1", help="i/N data-parallel shard for 8-GPU single-experiment runs")
    p.add_argument("--rank-dump", default=None, help="JSONL: per-gold-passage rank (diagnose under-train vs conditional-relevance)")
    return p.parse_args()


@torch.no_grad()
def passage_scores(model, tok, mgr, question, C, device, off):
    """each row of the query batch reads exactly ONE segment; score logP(' yes')."""
    probe = bp(tok, "", question + " Does the passage shown to you contain information needed "
                                   "to answer this question? Reply yes or no.", False)
    yes_ids = tok(" yes", add_special_tokens=False)["input_ids"]
    qids = tok(probe, add_special_tokens=False)["input_ids"]
    ids_l = [qids + yes_ids for _ in range(C)]
    lab_l = [[-100] * len(qids) + yes_ids for _ in range(C)]
    pad = tok.pad_token_id or tok.eos_token_id
    ids, attn = left_pad(ids_l, pad, device)
    labels, _ = left_pad(lab_l, -100, device)
    labels = labels.masked_fill(attn == 0, -100)
    allowed = torch.eye(C, dtype=torch.bool, device=device)
    mgr.set_allowed(allowed)
    pos = (attn.long().cumsum(1) - 1).clamp(min=0) + off
    mgr.set_valid(attn); mgr.set_query_rows(attn.bool()); mgr.start_use()
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
    logp = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
    lab = labels[:, 1:]
    mask = lab != -100
    g = torch.gather(logp, 2, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mgr.set_allowed(None)
    return ((g * mask).sum(1) / mask.sum(1).clamp(min=1))


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
    mgr = BatchCrossCache(list(range(len(model.model.layers)))); mgr.register(model)

    parsed, pool = build_examples(args.flashrag_root, args.dataset, args.num_q, args.seed)
    si, sn = (int(x) for x in args.shard.split("/"))
    items = []
    for ex in parsed[:args.num_q]:
        gold = list(ex["gold"])
        need = max(0, args.n_paras - len(gold))
        distr = rng.sample(pool, min(need, len(pool)))
        paras = gold + distr
        idx = list(range(len(paras))); rng.shuffle(idx)
        paras = [paras[i] for i in idx]
        gold_pos = {pi for pi, oi in enumerate(idx) if oi < len(gold)}
        items.append({"paras": paras, "gold_pos": gold_pos, "question": ex["question"]})
    items = items[si::sn]

    ks = [2, 3, 4, 6, 8]
    rec = {k: 0 for k in ks}
    # gold-rank histogram: where do gold passages land when scored? best-rank vs worst-rank per Q
    gold_ranks = []  # every gold passage's rank (0=top)
    worst_ranks = []  # the WORST-ranked gold per question (the 2nd-hop bottleneck)
    import json as _json
    df = open(args.rank_dump, "w") if args.rank_dump else None
    for it in items:
        C = len(it["paras"])
        off = capture(model, tok, mgr, it["paras"], it["question"], device, args.seg_cap, args.max_plen)
        sc = passage_scores(model, tok, mgr, it["question"], C, device, off)
        order = sc.argsort(descending=True).tolist()
        rank_of = {p: r for r, p in enumerate(order)}
        gr = sorted(rank_of[g] for g in it["gold_pos"])
        gold_ranks.extend(gr); worst_ranks.append(gr[-1] if gr else C)
        for k in ks:
            if it["gold_pos"] <= set(order[:k]):
                rec[k] += 1
        if df:
            df.write(_json.dumps({"C": C, "n_gold": len(it["gold_pos"]), "gold_ranks": gr,
                                  "q": it["question"][:80]}) + "\n")
        mgr.bank = {}
        torch.cuda.empty_cache()
    if df:
        df.close()
    n = len(items)
    parts = [f"recall@{k}={100*rec[k]/n:.1f}" for k in ks]
    # how the worst gold per Q distributes — the killer metric for under-train vs conditional
    wr = sorted(worst_ranks)
    buckets = {"top4": sum(1 for r in worst_ranks if r < 4),
               "top8": sum(1 for r in worst_ranks if r < 8),
               "top16": sum(1 for r in worst_ranks if r < 16),
               "beyond16": sum(1 for r in worst_ranks if r >= 16)}
    print(f"== passage-score {args.dataset} np{args.n_paras} n={n} shard={args.shard} ==  "
          + "  ".join(parts), flush=True)
    print(f"   worst-gold-rank buckets {buckets}  (beyond16 = unrecoverable-by-topk gold)", flush=True)


if __name__ == "__main__":
    main()
