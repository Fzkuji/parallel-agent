#!/usr/bin/env python3
"""Synthetic needle-in-a-haystack to test whether parallel encoding BEATS Concat at long context.

Hypothesis: Concat puts the whole haystack in ONE long sequence -> the needle suffers
lost-in-the-middle as length grows. Our parallel encoding splits the haystack into G
chunks, each encoded INDEPENDENTLY from position 0 -> every chunk's needle sits in a short,
"position-0" window and never gets buried. So as total length grows, Concat should degrade
while MultiQuery (ours) stays flat -> a crossover where ours > Oracle.

Setup: G queries share one haystack of `n_chunks` filler chunks. Each query i has its
needle planted in chunk i (a unique key->value fact). We sweep chunk length (=> total
length) and report EM for Independent / MultiQuery / Concat(Oracle).
"""
import argparse, os, sys, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.eval_multiquery import independent, multiquery, build_prompt
from src.evaluation.basic import compute_em
from src.inference import extract_answer

FILLER = ("The garden was quiet that afternoon and the wind moved slowly through the old trees. "
          "Nobody remembered exactly when the fence had been painted last, but it hardly mattered now. "
          "A distant train sounded twice and then faded into the steady hum of the summer air. ")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", default=None)
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--chunk-words", default="80,200,400,800",
                   help="filler words per chunk to sweep (=> total length grows)")
    p.add_argument("--needle-depth", type=float, default=0.5, help="fractional depth of needle in its chunk")
    p.add_argument("--num-groups", type=int, default=40)
    p.add_argument("--max-prompt-length", type=int, default=32768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="/tmp/niah")
    return p.parse_args()


def make_chunk(rng, words, key, val, depth):
    """A filler chunk with one needle 'The access code for KEY is VAL.' at fractional depth."""
    filler_words = (FILLER * 200).split()
    body = rng.sample(filler_words, min(words, len(filler_words)))
    needle = f"The access code for {key} is {val}."
    pos = int(len(body) * depth)
    body = body[:pos] + needle.split() + body[pos:]
    return " ".join(body)


def make_group(rng, G, words, depth):
    items = []
    for i in range(G):
        key = f"item-{rng.randint(1000, 9999)}-{i}"
        val = str(rng.randint(100000, 999999))
        ctx = make_chunk(rng, words, key, val, depth)
        items.append({"context": ctx,
                      "question": f"What is the access code for {key}? Answer with the number only.",
                      "references": [val], "answer_tokens": 12, "has_supporting": True})
    return {"items": items}


def union_context(items):
    return "\n\n".join(it["context"] for it in items)


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

    def em(items, txts):
        c = 0
        for it, t in zip(items, txts):
            a, _ = extract_answer(t, "hotpot")
            c += compute_em(a, it["references"]) or (it["references"][0] in t)
        return 100.0 * c / len(items)

    print(f"{'chunkW':>7}{'~totalTok':>10}{'Indep':>8}{'MultiQ':>8}{'Concat':>8}")
    for words in [int(x) for x in args.chunk_words.split(",")]:
        rng = random.Random(args.seed)
        groups = [make_group(rng, args.n_agents, words, args.needle_depth) for _ in range(args.num_groups)]
        si = sm = so = 0.0; ntot = 0
        for g in groups:
            items = g["items"]; uni = union_context(items)
            it = independent(model, tok, mgr, items, device, 16, args.max_prompt_length)
            mq = multiquery(model, tok, mgr, items, device, 16, args.max_prompt_length, topk=0)
            orc = independent(model, tok, mgr, [{**t, "context": uni} for t in items], device, 16, args.max_prompt_length)
            si += em(items, it); sm += em(items, mq); so += em(items, orc); ntot += 1
        tot = len(tok(union_context(groups[0]["items"]))["input_ids"])
        print(f"{words:>7}{tot:>10}{si/ntot:>8.1f}{sm/ntot:>8.1f}{so/ntot:>8.1f}", flush=True)


if __name__ == "__main__":
    main()
