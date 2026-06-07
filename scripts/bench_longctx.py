#!/usr/bin/env python3
"""Long-context single-question QA: parallel split-encoding vs the vanilla single-sequence baseline.

A normal QA question whose evidence sits in a long document (gold paragraphs +
many distractors). We compare ONLY two ways to feed the SAME content:

  Vanilla (baseline): the standard model reads ALL paragraphs in ONE sequence (standard long-context).
  Ours   (parallel) : paragraphs split into K chunks, each encoded INDEPENDENTLY
                      from position 0 into a shared bank; the single query reads
                      the whole bank. No chunk is buried deep -> no lost-in-the-middle.

We sweep the number of paragraphs (=> context length). Hypothesis: as length grows,
Vanilla degrades (lost-in-the-middle) while Ours stays flatter. APE cannot do this
(no per-query training / selective reading); we can. (No Independent baseline: splitting
would hide evidence from some chunks, which is not the comparison we care about.)
"""
import argparse, os, sys, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.eval_multiquery import independent, multiquery, build_prompt, context_mask_for, _maxsim_scores, decode_texts
from src.datasets.hotpot_distributed import _parse_hotpot, _format_paragraph
from src.evaluation.basic import compute_em
from src.inference import extract_answer
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", default=None)
    p.add_argument("--n-chunks", type=int, default=8, help="K: split the document into K parallel chunks")
    p.add_argument("--para-level", action="store_true",
                   help="encode EACH paragraph as its own chunk and select top-k PARAGRAPHS "
                        "(fine-grained selection: read only the few relevant paras, length-invariant)")
    p.add_argument("--niah-single", action="store_true",
                   help="single-hop synthetic NIAH (1 needle fact + distractors) instead of multi-hop HotpotQA")
    p.add_argument("--n-paras", default="10,20,40,80", help="total paragraphs (=> length) to sweep")
    p.add_argument("--topk", type=int, default=0, help=">0: query reads only top-k chunks/paras (selective)")
    p.add_argument("--num-q", type=int, default=80)
    p.add_argument("--max-prompt-length", type=int, default=32768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="/tmp/longctx")
    return p.parse_args()


def build_examples(n_paras, num_q, seed):
    raw = list(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"))
    rng = random.Random(seed); rng.shuffle(raw)
    parsed, pool = _parse_hotpot(raw, only_bridge=True, require_min_supporting=2)
    pool = list(pool)
    exs = []
    for q in parsed:
        if len(exs) >= num_q:
            break
        gold = [_format_paragraph(t, p) for (t, p) in q["supporting"][:2]]
        need = n_paras - len(gold)
        paras = gold + rng.sample(pool, min(need, len(pool)))
        rng.shuffle(paras)
        exs.append({"paras": paras, "question": q["question"], "answer": q["answer"]})
    return exs


def build_examples_niah(n_paras, num_q, seed):
    """Single-hop NIAH: 1 needle paragraph (a self-contained fact) among n_paras-1 wiki distractors.
    Single-hop -> splitting does NOT break reasoning; the only challenge is RETRIEVING the needle,
    where Vanilla suffers lost-in-the-middle and our position-0 chunks do not."""
    raw = list(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"))
    rng = random.Random(seed); rng.shuffle(raw)
    _parsed, pool = _parse_hotpot(raw, only_bridge=True, require_min_supporting=2)
    pool = list(pool)
    exs = []
    for i in range(num_q):
        key = f"Project-{rng.randint(1000, 9999)}-{chr(65 + i % 26)}"
        val = str(rng.randint(100000, 999999))
        needle = f"Important: the access code for {key} is {val}. Remember this access code."
        paras = [needle] + rng.sample(pool, min(n_paras - 1, len(pool)))
        rng.shuffle(paras)
        exs.append({"paras": paras,
                    "question": f"What is the access code for {key}? Answer with the number only.",
                    "answer": val})
    return exs


@torch.no_grad()
def ours_read(model, tok, mgr, chunks, question, device, max_new, max_plen, topk=0):
    """A SINGLE query reads a bank of K independently-encoded chunks (the real method, no query
    duplication -> K-times less memory than reusing multiquery with K identical queries)."""
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    cp = [build_prompt(tok, c, question) for c in chunks]
    enc = tok(cp, return_tensors="pt", padding=True, truncation=True, max_length=max_plen)
    cids = enc["input_ids"].to(device); cattn = enc["attention_mask"].to(device)
    cm = context_mask_for(tok, cp, chunks, cids.shape[1], cattn, max_plen).to(device)
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(1.0, 1.0)
    mgr.start_capture(cm); mgr.set_valid(cattn)
    out1 = model(input_ids=cids, attention_mask=cattn, use_cache=False, output_hidden_states=(topk > 0))
    off = int(cattn.sum(1).max().item()); K = len(chunks)
    qp = build_prompt(tok, "", question)
    enc = tok([qp], return_tensors="pt", padding=True, truncation=True, max_length=max_plen)
    qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
    if 0 < topk < K:
        chs = out1.hidden_states[-1]
        mgr.set_enabled(False)
        qhs = model(input_ids=qids, attention_mask=qattn, output_hidden_states=True).hidden_states[-1]
        mgr.set_enabled(True)
        R = _maxsim_scores(qhs, qattn, chs, cm)
        allowed = torch.zeros(1, K, dtype=torch.bool, device=device)
        allowed.scatter_(1, R.topk(topk, dim=1).indices, True); mgr.set_allowed(allowed)
    mgr.start_use()
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
    nxt_pos = off + qattn.sum(1)
    gen = qids.clone(); cur = qattn.clone(); fin = torch.zeros(1, dtype=torch.bool, device=device); pkv = None; nxt = gen
    P = qids.shape[1]
    for step in range(max_new):
        mgr.set_valid(cur)
        mgr.set_query_rows(cur.bool() if step == 0 else torch.ones(1, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(1, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        t = out.logits[:, -1].argmax(-1); t = torch.where(fin, torch.full_like(t, pad), t); fin = fin | (t == eos)
        gen = torch.cat([gen, t.unsqueeze(1)], 1); cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break
    return decode_texts(tok, gen, P, eos, pad)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
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
    K = args.n_chunks

    def chunk(paras):
        """split paragraphs into chunks. para-level: 1 paragraph = 1 chunk (fine-grained)."""
        if args.para_level:
            return list(paras)
        out = [[] for _ in range(K)]
        for i, p in enumerate(paras):
            out[i % K].append(p)
        return ["\n\n".join(c) for c in out]

    print(f"{'nParas':>7}{'~tok':>8}{'Vanilla':>9}{'Ours':>7}{'gap':>7}")
    gen = build_examples_niah if args.niah_single else build_examples
    for n_paras in [int(x) for x in args.n_paras.split(",")]:
        exs = gen(n_paras, args.num_q, args.seed)
        sl = so = 0.0; ntot = 0; toklen = 0
        v_oom = 0
        for ex in exs:
            full = "\n\n".join(ex["paras"])
            chunks = chunk(ex["paras"])
            # baseline: vanilla single sequence (may OOM at very long length -> a feasibility point for us)
            try:
                ls = independent(model, tok, mgr,
                                 [{"context": full, "question": ex["question"], "references": [ex["answer"]],
                                   "has_supporting": True}], device, 16, args.max_prompt_length)
                a_ls, _ = extract_answer(ls[0], "hotpot"); sl += compute_em(a_ls, [ex["answer"]])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); v_oom += 1
            # ours: K parallel chunks, a SINGLE query reads the whole bank (real method, light memory)
            mq = ours_read(model, tok, mgr, chunks, ex["question"], device, 16, args.max_prompt_length, topk=args.topk)
            a_mq, _ = extract_answer(mq[0], "hotpot"); so += compute_em(a_mq, [ex["answer"]]); ntot += 1
            if ntot == 1:
                toklen = len(tok(full)["input_ids"])
        vn = ntot - v_oom
        L = (100.0 * sl / vn) if vn else float("nan"); O = 100.0 * so / ntot
        vstr = f"{L:>9.1f}" if vn else f"{'OOMx'+str(v_oom):>9}"
        print(f"{n_paras:>7}{toklen:>8}{vstr}{O:>7.1f}  vOOM={v_oom}", flush=True)


if __name__ == "__main__":
    main()
