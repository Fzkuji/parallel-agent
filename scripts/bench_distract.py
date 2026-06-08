#!/usr/bin/env python3
"""Distractor-accumulation degradation curve: Concat vs APE vs CrossKV (ours).

Modern long-context RAG protocol (Long-Context LLMs Meet RAG, ICLR'25; Databricks'24):
fix the gold evidence, then INJECT more and more hard-negative distractor paragraphs
(borrowed from OTHER questions in the same dataset) so the context length grows. As the
distractor pile grows, a model that reads everything in one sequence degrades ("more
retrieved passages first help then hurt"). We sweep n_paras (=> length) and compare THREE
ways to feed the SAME paragraph set:

  Concat   (baseline) : all paragraphs in ONE long sequence (standard long-context RAG).
                        Lost-in-the-middle + window overflow -> accuracy collapses.
  APE                 : each paragraph encoded INDEPENDENTLY from position 0 into a shared
                        bank; the query reads the WHOLE bank with APE realignment
                        (temperature + LSE scaling). Training-free; uniform read, so hard
                        negatives still dilute the answer.
  CrossKV  (ours)     : same independent encoding, but a LoRA-trained query dynamically
                        SELECTS the relevant paragraphs each step and filters hard negatives
                        (parameter-free adaptive read). Length-invariant -> flattest curve.

Data: 2wikimultihopqa (flashrag local arrow), gold = supporting_facts paragraphs (~2),
distractors injected from a global pool of all non-gold paragraphs. Metric: SubEM
(substring containment, the HELMET/ParallelComp standard) + EM for reference.
"""
import argparse, glob, json, os, random, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.eval_multiquery import (
    independent, build_prompt, context_mask_for, _maxsim_scores,
    adaptive_allowed, decode_texts,
)
from src.evaluation.basic import compute_em, normalize_answer
from src.inference import extract_answer

FLASHRAG = "/mnt/data/zichuanfu/.cache/huggingface/datasets/RUC-NLPIR___flash_rag_datasets"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", default=None, help="CrossKV LoRA; APE/Concat ignore it")
    p.add_argument("--dataset", default="2wikimultihopqa",
                   choices=["2wikimultihopqa", "hotpotqa"])
    p.add_argument("--flashrag-root", default=FLASHRAG)
    p.add_argument("--n-paras", default="10,20,40,80,160",
                   help="total paragraphs per question (=> context length) to sweep")
    p.add_argument("--num-q", type=int, default=100)
    p.add_argument("--arms", default="concat,ape,ours",
                   help="comma list subset of {concat,ape,ours}")
    p.add_argument("--ape-temp", type=float, default=0.9, help="APE bank temperature (<1 sharpens)")
    p.add_argument("--ape-scale", type=float, default=0.9, help="APE bank LSE scaling (<1 down-weights)")
    p.add_argument("--adaptive", action="store_true",
                   help="ours: parameter-free per-step adaptive paragraph selection (filter distractors)")
    p.add_argument("--max-new", type=int, default=24)
    p.add_argument("--max-prompt-length", type=int, default=131072)
    p.add_argument("--chunk-plen", type=int, default=1024, help="per-paragraph encode truncation length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="/tmp/distract")
    return p.parse_args()


# ----------------------------------------------------------------------------- data
def _load_arrow(root, name, split):
    import pyarrow as pa, pyarrow.ipc as ipc
    fs = glob.glob(f"{root}/{name}/0.0.0/*/flash_rag_datasets-{split}.arrow")
    if not fs:
        raise FileNotFoundError(f"no {split} arrow for {name} under {root}")
    with pa.memory_map(fs[0], "r") as src:
        try:
            return ipc.open_stream(src).read_all().to_pylist()
        except Exception:
            src.seek(0)
            return ipc.open_file(src).read_all().to_pylist()


def _para_text(piece):
    """A context paragraph is a list of sentences (2wiki: content[i], hotpot: text[i])."""
    return " ".join(piece) if isinstance(piece, (list, tuple)) else str(piece)


def build_examples(root, name, num_q, seed):
    """Each example: gold paragraphs (supporting) + a GLOBAL distractor pool to inject from.
    The pool is every non-gold paragraph across the dataset (real hard negatives, same domain)."""
    split = "dev"
    rows = _load_arrow(root, name, split)
    rng = random.Random(seed)
    rng.shuffle(rows)
    pool = []          # global distractor paragraphs (strings)
    parsed = []
    for r in rows:
        md = r["metadata"]
        if isinstance(md, str):
            md = json.loads(md)
        ctx = md.get("context")
        sf = md.get("supporting_facts")
        if not isinstance(ctx, dict) or "title" not in ctx or not sf:
            continue
        titles = ctx["title"]
        texts = ctx.get("content") or ctx.get("text")
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
        if gold:
            ans = r["golden_answers"]
            ans = ans if isinstance(ans, list) else [ans]
            parsed.append({"gold": gold, "question": r["question"], "answer": ans})
    rng.shuffle(parsed)
    rng.shuffle(pool)
    return parsed[: num_q * 2], pool          # extra parsed in case some get skipped


def make_paras(ex, n_paras, pool, rng):
    """gold + injected distractors up to n_paras, shuffled (gold not always last)."""
    gold = ex["gold"]
    need = max(0, n_paras - len(gold))
    distr = rng.sample(pool, min(need, len(pool)))
    paras = gold + distr
    rng.shuffle(paras)
    return paras


# ----------------------------------------------------------------------------- metric
def compute_subem(prediction, references):
    """Substring EM: 1.0 if any normalized gold answer is contained in the normalized
    prediction. The HELMET / ParallelComp standard for long-context QA (more robust than
    strict EM for verbose multi-hop answers)."""
    p = normalize_answer(prediction)
    for ref in references:
        r = normalize_answer(ref)
        if r and r in p:
            return 1.0
    return 0.0


# ----------------------------------------------------------------------------- ours/APE read
@torch.no_grad()
@torch.no_grad()
def _relevance_decode(model, tok, mgr, qids, qattn, device, off, n_steps, eos, pad):
    """Short greedy decode that lets mgr.record per-passage bank attention (start_relevance must
    be active). Used only to RANK passages for read-fraction top-k; output discarded."""
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
    nxt_pos = off + qattn.sum(1)
    cur = qattn.clone(); fin = torch.zeros(1, dtype=torch.bool, device=device); pkv = None; nxt = qids.clone()
    for step in range(n_steps):
        mgr.set_valid(cur)
        mgr.set_query_rows(cur.bool() if step == 0 else torch.ones(1, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(1, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        t = out.logits[:, -1].argmax(-1); t = torch.where(fin, torch.full_like(t, pad), t)
        fin = fin | (t == eos)
        cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break


def bank_read(model, tok, mgr, chunks, question, device, max_new, max_plen,
              temp=1.0, scale=1.0, adaptive=False, topk_frac=1.0):
    """A SINGLE query reads a bank of independently-encoded paragraphs.
    temp/scale<1 -> APE realignment (training-free). adaptive -> parameter-free per-query
    paragraph selection. topk_frac<1 -> READ-FRACTION: a cheap first decode records per-passage
    attention, then only the top-(topk_frac) most-attended passages are read (Pareto: read fewer,
    often MORE accurate by filtering distractors). With a LoRA-merged model and temp=scale=1,
    this is the trained CrossKV read."""
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    cp = [build_prompt(tok, c, question) for c in chunks]
    enc = tok(cp, return_tensors="pt", padding=True, truncation=True, max_length=max_plen)
    cids = enc["input_ids"].to(device); cattn = enc["attention_mask"].to(device)
    cm = context_mask_for(tok, cp, chunks, cids.shape[1], cattn, max_plen).to(device)
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None)
    mgr.set_realign(temp, scale)
    mgr.start_capture(cm); mgr.set_valid(cattn)
    out1 = model(input_ids=cids, attention_mask=cattn, use_cache=False,
                 output_hidden_states=adaptive)
    off = int(cattn.sum(1).max().item()); K = len(chunks)
    qp = build_prompt(tok, "", question)
    enc = tok([qp], return_tensors="pt", padding=True, truncation=True, max_length=max_plen)
    qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
    if adaptive and K > 1:
        chs = out1.hidden_states[-1]
        mgr.set_enabled(False)
        qhs = model(input_ids=qids, attention_mask=qattn, output_hidden_states=True).hidden_states[-1]
        mgr.set_enabled(True)
        R = _maxsim_scores(qhs, qattn, chs, cm)[:1]   # [1,K]
        mgr.set_allowed(adaptive_allowed(R))          # parameter-free selection
    if topk_frac < 1.0 and K > 1:
        # READ-FRACTION via ATTENTION MASS: a short relevance decode records per-passage bank
        # attention, then read only the top-(topk_frac) most-attended passages (Pareto: read
        # fewer, often more accurate by filtering distractors).
        import torch as _t
        mgr.start_use(); mgr.set_allowed(None); mgr.start_relevance(K)
        _relevance_decode(model, tok, mgr, qids, qattn, device, off, n_steps=16, eos=eos, pad=pad)
        rel = mgr.relevance()
        if rel is not None:
            kk = max(1, round(topk_frac * K))
            a = _t.zeros(1, K, dtype=_t.bool, device=device)
            a.scatter_(1, rel.to(device).topk(kk, dim=1).indices, True)
            mgr.set_allowed(a)
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
        t = out.logits[:, -1].argmax(-1); t = torch.where(fin, torch.full_like(t, pad), t)
        fin = fin | (t == eos)
        gen = torch.cat([gen, t.unsqueeze(1)], 1)
        cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break
    n_read = int(mgr.allowed.sum().item()) if (adaptive and mgr.allowed is not None) else K
    return decode_texts(tok, gen, P, eos, pad)[0], n_read


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"; tok.truncation_side = "left"
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device).eval()
    if args.lora_path and "ours" in arms:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload().eval()
    nl = model.config.num_hidden_layers
    mgr = BatchCrossCache(list(range(nl))); mgr.register(model)

    parsed, pool = build_examples(args.flashrag_root, args.dataset, args.num_q, args.seed)
    sweep = [int(x) for x in args.n_paras.split(",")]
    results = {a: {} for a in arms}

    hdr = f"{'nParas':>7}{'~tok':>8}"
    for a in arms:
        hdr += f"{a:>10}"
    if "ours" in arms and args.adaptive:
        hdr += f"{'read':>7}"
    print(hdr, flush=True)

    for n_paras in sweep:
        rng = random.Random(args.seed + n_paras)
        acc = {a: 0.0 for a in arms}; oom = {a: 0 for a in arms}
        ntot = 0; toklen = 0; read_sum = 0.0
        for ex in parsed:
            if ntot >= args.num_q:
                break
            paras = make_paras(ex, n_paras, pool, rng)
            full = "\n\n".join(paras)
            if ntot == 0:
                toklen = len(tok(full)["input_ids"])
            ok = True
            for a in arms:
                try:
                    if a == "concat":
                        out = independent(model, tok, mgr,
                                          [{"context": full, "question": ex["question"],
                                            "references": ex["answer"], "has_supporting": True}],
                                          device, args.max_new, args.max_prompt_length)[0]
                        nr = None
                    elif a == "ape":
                        out, nr = bank_read(model, tok, mgr, paras, ex["question"], device,
                                            args.max_new, args.max_prompt_length,
                                            temp=args.ape_temp, scale=args.ape_scale, adaptive=False)
                    else:  # ours
                        out, nr = bank_read(model, tok, mgr, paras, ex["question"], device,
                                            args.max_new, args.max_prompt_length,
                                            temp=1.0, scale=1.0, adaptive=args.adaptive)
                    ans, _ = extract_answer(out, "hotpot")
                    acc[a] += compute_subem(ans, ex["answer"])
                    if a == "ours" and nr is not None:
                        read_sum += nr
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); oom[a] += 1; ok = False
            if ok:
                ntot += 1
        line = f"{n_paras:>7}{toklen:>8}"
        for a in arms:
            n = ntot - oom[a]
            v = (100.0 * acc[a] / n) if n else float("nan")
            results[a][n_paras] = {"subem": v, "n": n, "oom": oom[a]}
            line += (f"{v:>10.1f}" if n else f"{'OOMx'+str(oom[a]):>10}")
        if "ours" in arms and args.adaptive and ntot:
            line += f"{read_sum/ntot:>7.1f}"
        print(line, flush=True)

    out_path = os.path.join(args.output_dir, f"distract_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
