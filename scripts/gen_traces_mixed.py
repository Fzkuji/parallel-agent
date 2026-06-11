#!/usr/bin/env python3
"""Mixed-source thinking trajectories: 2wiki train (native multi-doc) + SQuAD chunked (single doc
split into K segments — the bank doesn't care whether segments come from many docs or one doc
sliced). Teacher = Qwen3 full-attention over the gold context with thinking; keep only
teacher-correct. Each row's group_items = gold chunks (has_supporting=True) + cross-question
distractor chunks (False), so train_think_distill --capture-all trains the student on DIRTY banks
(density-matched to eval) while the target trajectory comes from the clean teacher.

Output rows are drop-in compatible with train_think_distill.py.
"""
import argparse, os, sys, json, random, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.bench_distract import _load_arrow, _para_text
from scripts.bench_longbench import best_f1
from src.inference import extract_box_answer
from scripts.train_multiquery_lora import build_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-path", default="/mnt/data/zichuanfu/models/Qwen3-8B")
    p.add_argument("--flashrag-root",
                   default="/mnt/data/zichuanfu/.cache/huggingface/datasets/RUC-NLPIR___flash_rag_datasets")
    p.add_argument("--n-2wiki", type=int, default=6000)
    p.add_argument("--n-squad", type=int, default=4000)
    p.add_argument("--squad-chunks", type=int, default=3)
    p.add_argument("--max-distract", type=int, default=6,
                   help="random 0..K cross-question distractor chunks stored per row")
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


def chunk_text(text, k):
    ss = sents(text)
    if not ss:
        return [text]
    k = min(k, len(ss))
    per = (len(ss) + k - 1) // k
    return [" ".join(ss[i:i + per]) for i in range(0, len(ss), per)]


def load_2wiki_train(root, n, rng):
    rows = _load_arrow(root, "2wikimultihopqa", "train")
    rng.shuffle(rows)
    items, pool = [], []
    for r in rows:
        md = r["metadata"]
        if isinstance(md, str):
            md = json.loads(md)
        ctx = md.get("context"); sf = md.get("supporting_facts")
        if not isinstance(ctx, dict) or "title" not in ctx or not sf:
            continue
        titles = ctx["title"]
        texts = ctx.get("content") or ctx.get("text") or ctx.get("sentences")
        if texts is None or len(texts) != len(titles):
            continue
        gold_titles = set(sf.get("title", []))
        gold, others = [], []
        for t, tx in zip(titles, texts):
            s = _para_text(tx).strip()
            if not s:
                continue
            (gold if t in gold_titles else others).append(s)
        pool.extend(others)
        if len(gold) < 2:
            continue
        ans = r["golden_answers"]
        ans = ans if isinstance(ans, list) else [ans]
        items.append({"question": r["question"], "answers": ans, "gold": gold, "src": "2wiki"})
        if len(items) >= n and len(pool) > 5000:
            break
    return items, pool


def load_squad_chunked(n, k, rng):
    from datasets import load_dataset
    ds = load_dataset("squad", split="train")
    idx = list(range(len(ds))); rng.shuffle(idx)
    items, pool = [], []
    for i in idx:
        ex = ds[i]
        chunks = chunk_text(ex["context"], k)
        if len(chunks) < 2:
            continue
        pool.extend(chunks)
        items.append({"question": ex["question"], "answers": list(ex["answers"]["text"]),
                      "gold": chunks, "src": "squad"})
        if len(items) >= n:
            break
    return items, pool


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    w2, pool_w2 = load_2wiki_train(args.flashrag_root, args.n_2wiki, rng)
    sq, pool_sq = load_squad_chunked(args.n_squad, args.squad_chunks, rng)
    print(f"loaded 2wiki={len(w2)} squad={len(sq)}  pools: {len(pool_w2)}/{len(pool_sq)}", flush=True)
    items = w2 + sq
    rng.shuffle(items)
    pools = {"2wiki": pool_w2, "squad": pool_sq}

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher_path)
    llm = LLM(model=args.teacher_path, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=8192, enforce_eager=True)
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_new)

    prompts = [build_prompt(tok, "\n\n".join(it["gold"]), it["question"]) for it in items]
    outs = llm.generate(prompts, sp)

    kept = {"2wiki": 0, "squad": 0}
    with open(args.out, "w") as df:
        for it, o in zip(items, outs):
            traj = o.outputs[0].text
            ans, _ = extract_box_answer(traj)
            ok = best_f1(ans, it["answers"]) >= 0.5 and "</think>" in traj and "<answer>" in traj
            if not ok:
                continue
            ta = traj[traj.find("<think>"):] if "<think>" in traj else traj
            gi = [{"context": c, "question": "", "references": [""], "has_supporting": True}
                  for c in it["gold"]]
            nd = rng.randint(0, args.max_distract)
            own = set(it["gold"])
            distr = [c for c in rng.sample(pools[it["src"]], min(nd * 3 + 1, len(pools[it["src"]])))
                     if c not in own][:nd]
            gi += [{"context": c, "question": "", "references": [""], "has_supporting": False}
                   for c in distr]
            rng.shuffle(gi)
            df.write(json.dumps({"question": it["question"], "references": it["answers"],
                                 "src": it["src"], "group_items": gi,
                                 "think_answer": ta.strip()}) + "\n")
            kept[it["src"]] += 1
    print(f"KEPT 2wiki={kept['2wiki']} squad={kept['squad']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
