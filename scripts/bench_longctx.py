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
from scripts.eval_multiquery import independent, multiquery
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
        for ex in exs:
            full = "\n\n".join(ex["paras"])
            # baseline: one vanilla (standard model, whole context in one sequence)
            ls = independent(model, tok, mgr,
                             [{"context": full, "question": ex["question"], "references": [ex["answer"]],
                               "has_supporting": True}], device, 16, args.max_prompt_length)
            # ours: K parallel chunks, single query reads the whole bank (duplicate the query K times,
            # take chunk-0's answer). topk optional.
            chunks = chunk(ex["paras"])
            items = [{"context": c, "question": ex["question"], "references": [ex["answer"]],
                      "has_supporting": True} for c in chunks]
            mq = multiquery(model, tok, mgr, items, device, 16, args.max_prompt_length, topk=args.topk)
            a_ls, _ = extract_answer(ls[0], "hotpot"); a_mq, _ = extract_answer(mq[0], "hotpot")
            sl += compute_em(a_ls, [ex["answer"]]); so += compute_em(a_mq, [ex["answer"]]); ntot += 1
            if ntot == 1:
                toklen = len(tok(full)["input_ids"])
        L = 100.0 * sl / ntot; O = 100.0 * so / ntot
        print(f"{n_paras:>7}{toklen:>8}{L:>9.1f}{O:>7.1f}{O - L:>+7.1f}", flush=True)


if __name__ == "__main__":
    main()
