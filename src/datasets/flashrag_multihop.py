"""Multi-dataset multi-hop rollout loader (2wiki + hotpot + musique flashrag TRAIN splits) for
stage-C GRPO. Emits ONE item per paragraph (gold + same-domain distractors), matching the eval
bank_read per-paragraph bank, so GRPO sees all three distributions (fixes the hotpot-only
cross-distribution failure). Trains on split='train' ONLY — never the LongBench eval jsonl or
the flashrag dev split.

Schema gotchas (verified on server): hotpot train is 2-sharded (train-00000/00001-of-00002) so
the glob must use 'train*'; 2wiki context uses 'content', hotpot uses 'sentences'; musique has
no context/supporting_facts, only question_decomposition[].support_paragraph.
"""
import glob, json, random
import pyarrow as pa, pyarrow.ipc as ipc

FLASHRAG = "/mnt/data/zichuanfu/.cache/huggingface/datasets/RUC-NLPIR___flash_rag_datasets"


def _load_arrow(root, name, split):
    fs = sorted(glob.glob(f"{root}/{name}/0.0.0/*/flash_rag_datasets-{split}*.arrow"))  # *=hotpot shards
    rows = []
    for f in fs:
        with pa.memory_map(f, "r") as s:
            try:
                t = ipc.open_stream(s).read_all()
            except Exception:
                s.seek(0); t = ipc.open_file(s).read_all()
        rows += t.to_pylist()
    if not rows:
        raise FileNotFoundError(f"{name}/{split} under {root}")
    return rows


def _md(r):
    m = r["metadata"]; return json.loads(m) if isinstance(m, str) else m


def _fmt(title, text):
    return f"[{title}] {text}"


def _parse_row(name, r):
    md = _md(r); ans = r.get("golden_answers"); ans = ans if isinstance(ans, list) else [ans]
    q = (r.get("question") or "").strip()
    if name.startswith("musique"):
        gold = []
        for h in md.get("question_decomposition") or []:
            sp = h.get("support_paragraph") or {}
            tx = (sp.get("paragraph_text") or "").strip()
            if tx:
                gold.append(_fmt(sp.get("title", ""), tx))
        return (q, gold[:2], [], ans) if (q and len(gold) >= 2 and ans and ans[0]) else None
    ctx = md.get("context"); sf = md.get("supporting_facts")
    if not isinstance(ctx, dict) or "title" not in ctx or not sf:
        return None
    titles = ctx["title"]
    texts = ctx.get("content") or ctx.get("text") or ctx.get("sentences")  # 2wiki=content, hotpot=sentences
    if texts is None or len(texts) != len(titles):
        return None
    goldset = set(sf.get("title", []))
    gold, distr = [], []
    for t, tx in zip(titles, texts):
        s = (" ".join(tx) if isinstance(tx, (list, tuple)) else str(tx)).strip()
        if not s:
            continue
        (gold if t in goldset else distr).append(_fmt(t, s))
    return (q, gold[:2], distr, ans) if (q and len(gold) >= 2 and ans and ans[0]) else None


def load_flashrag_multihop_groups(datasets=("2wikimultihopqa", "hotpotqa", "musique"),
                                  split="train", n_paras=8, max_groups=1500, seed=42, root=FLASHRAG):
    """ONE query per group; items = one item PER PARAGRAPH (gold + distractors). Distractors are a
    same-dataset global pool (hard same-domain negatives). The 3 datasets are round-robin
    interleaved so every optimizer window sees all distributions."""
    rng = random.Random(seed)
    per_ds = {}
    for name in datasets:
        rows = _load_arrow(root, name, split); rng.shuffle(rows)
        parsed, pool = [], []
        for r in rows:
            p = _parse_row(name, r)
            if p is None:
                continue
            q, gold, distr, ans = p
            parsed.append({"q": q, "gold": gold, "ans": ans})
            pool.extend(distr if distr else gold)  # musique: pool = other questions' gold (real hard neg)
            if len(pool) > 8000:
                pool = pool[:8000]
        groups = []
        cap = max(1, max_groups // len(datasets))
        for ex in parsed[:cap]:
            need = max(0, n_paras - len(ex["gold"]))
            gset = set(ex["gold"])
            distr = [c for c in rng.sample(pool, min(need * 2, len(pool))) if c not in gset][:need] if pool else []
            paras = ex["gold"] + distr
            rng.shuffle(paras)
            items = [{"context": p, "question": ex["q"], "references": ex["ans"],
                      "has_supporting": p in gset} for p in paras]
            groups.append({"items": items, "question": ex["q"], "references": ex["ans"],
                           "gold": ex["gold"], "dataset": name})
        per_ds[name] = groups
    out, idx = [], {k: 0 for k in per_ds}
    while any(idx[k] < len(per_ds[k]) for k in per_ds):
        for k in per_ds:
            if idx[k] < len(per_ds[k]):
                out.append(per_ds[k][idx[k]]); idx[k] += 1
    return out
