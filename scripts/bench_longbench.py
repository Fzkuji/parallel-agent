#!/usr/bin/env python3
"""LongBench multi-hop QA main table: Concat vs APE vs CrossKV(ours) vs Oracle.

The standard long-context RAG benchmark used by APE (arXiv 2502.05431) and others.
Each LongBench sample is a single long `context` string (gold + distractor passages
concatenated) plus a short `input` question; the gold `answers` is a list. We split
the context into passages and feed the SAME content four ways:

  Concat  (baseline) : the whole context in ONE sequence (standard long-context).
  APE                : each passage encoded INDEPENDENTLY from position 0 into a shared
                       bank; the query reads the whole bank with APE realignment
                       (temperature + LSE scaling). Training-free.
  Ours (CrossKV)     : same independent encoding, LoRA-trained query reading the bank.
  Oracle             : concat, but only the passages that contain a gold answer (upper
                       bound on retrieval AND on cross-passage attention).
  APE-Oracle         : independent encoding, but only over the gold passages (isolates the
                       *encoding* ceiling of parallel encoding under perfect selection).
  Ours-Oracle        : same, with the LoRA-trained reader (our encoding ceiling).

  The two *-Oracle arms separate "selection ability" from "encoding ability": if APE/Ours
  on gold-only approach Oracle(concat), the encoding is sound and the main-table gap is
  purely selection; if they stay far below, the independent encoding itself is the ceiling.

Metric: official LongBench qa_f1_score (token F1 x100), max_new_tokens=32, official
prompt template. Run the full 200-sample test split per task for the main table.
"""
import argparse, os, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.eval_multiquery import (
    independent, build_prompt, context_mask_for, _maxsim_scores,
    adaptive_allowed, decode_texts,
)
from scripts.bench_distract import bank_read
from src.inference import extract_box_answer

# ---- official LongBench metric (THUDM/LongBench/metrics.py, verbatim) ------------
import string
from collections import Counter


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1(pred_tokens, gt_tokens):
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_tokens)
    r = num_same / len(gt_tokens)
    return 2 * p * r / (p + r)


def qa_f1_score(prediction, ground_truth):
    return _f1(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def best_f1(prediction, answers):
    return max((qa_f1_score(prediction, a) for a in answers), default=0.0)


# ---- LongBench official prompt for multi-hop QA --------------------------------
LB_PROMPT = ("Answer the question based on the given passages. Only give me the answer "
             "and do not output any other words.\n\nThe following are given passages.\n{context}\n\n"
             "Answer the question based on the given passages. Only give me the answer and do "
             "not output any other words.\n\nQuestion: {input}\nAnswer:")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", default=None)
    p.add_argument("--tasks", default="2wikimqa,hotpotqa,musique")
    p.add_argument("--data-dir", default="/mnt/data/zichuanfu/longbench_export",
                   help="dir of <task>.jsonl exported from THUDM/LongBench (offline)")
    p.add_argument("--arms", default="concat,ape,ours,oracle")
    p.add_argument("--num-q", type=int, default=200, help="LongBench has 200/task; 0=all")
    p.add_argument("--ape-temp", type=float, default=0.9)
    p.add_argument("--ape-scale", type=float, default=0.9)
    p.add_argument("--adaptive", action="store_true",
                   help="ours: parameter-free per-step selective read (default: read all)")
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--max-prompt-length", type=int, default=131072)
    p.add_argument("--think", action="store_true",
                   help="enable <think> for all arms (Qwen3); raise --max-new accordingly")
    p.add_argument("--no-think", dest="think", action="store_false")
    p.set_defaults(think=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="/tmp/longbench")
    return p.parse_args()


def split_passages(context):
    """Split a LongBench context blob into passages for independent encoding.
    Multi-hop QA contexts are 'Passage 1:\\n...\\nPassage 2:\\n...' style; fall back to
    blank-line blocks. Keep each passage non-empty."""
    parts = re.split(r"\n(?=Passage\s*\d+[:：])", context.strip())
    if len(parts) < 2:
        parts = [b for b in context.split("\n\n") if b.strip()]
    if len(parts) < 2:
        parts = [b for b in context.split("\n") if b.strip()]
    return [p.strip() for p in parts if p.strip()]


_STOP = set("a an the of in on at to for and or but who what where when which whose was were "
            "is are be been being did do does of by with from as that this his her its their "
            "born wife husband father mother son daughter died death place country city".split())


def _entities(text):
    """Capitalized multi-word entity spans (e.g. 'Francis I Rakoczi') for bridge matching."""
    ents = re.findall(r"[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+)*", text)
    out = set()
    for e in ents:
        ne = normalize_answer(e)
        if ne and ne not in _STOP and len(ne) > 2:
            out.add(ne)
    return out


def oracle_passages(passages, answers, question=""):
    """TRUE multi-hop oracle: keep every passage that contains a gold ANSWER string OR a
    QUESTION entity (the bridge passage — e.g. for 'where was the wife of Francis I Rakoczi
    born', the 'Francis I Rakoczi' passage names the wife and is required even though it does
    NOT contain the final answer). The old answer-only rule dropped bridge passages, making the
    'oracle' input incomplete and (wrongly) worse than reading all passages."""
    na = [normalize_answer(a) for a in answers if a.strip()]
    # yes/no and other non-entity answers ("no" is a substring of arbitrary prose) make the
    # answer-substring branch fire on near-random passages -> disable it for those; rely on the
    # question-entity bridge passages instead (hotpot yes/no fix).
    _nonentity = {"yes", "no", "true", "false"}
    use_ans = [a for a in na if a not in _nonentity and len(a) > 2]
    qents = _entities(question)
    keep = []
    for p in passages:
        npp = normalize_answer(p)
        title = ""
        for ln in p.split("\n"):
            ln = ln.strip()
            if ln and not re.match(r"passage\s*\d+", ln.lower()):
                title = ln; break
        tents = _entities(title)
        hit_ans = any(a and a in npp for a in use_ans)
        hit_q = bool(qents & tents)  # this passage is ABOUT a question entity -> bridge
        if hit_ans or hit_q:
            keep.append(p)
    return keep or passages[:2]  # never empty


def lb_prompt(context, question):
    return LB_PROMPT.format(context=context, input=question)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    from src.templates import set_think_tokens
    set_think_tokens(args.think)  # <think> on/off for ALL arms uniformly (fair comparison)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

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

    import json as _json

    def load_task(task):
        path = os.path.join(args.data_dir, f"{task}.jsonl")
        with open(path) as f:
            return [_json.loads(line) for line in f if line.strip()]

    table = {}
    print(f"{'task':>10}" + "".join(f"{a:>9}" for a in arms))
    for task in tasks:
        data = load_task(task)
        if args.num_q:
            data = data[: args.num_q]
        scores = {a: 0.0 for a in arms}; n = 0
        preds = {a: [] for a in arms}  # dump (q, gold, pred) per arm for offline LLM-judge
        for ex in data:
            passages = split_passages(ex["context"])
            answers = ex["answers"]
            q = ex["input"]
            for a in arms:
                if a == "concat":
                    full = "\n\n".join(passages)
                    out = independent(model, tok, mgr,
                                      [{"context": full, "question": q,
                                        "references": answers, "has_supporting": True}],
                                      device, args.max_new, args.max_prompt_length)[0]
                elif a == "oracle":
                    op = oracle_passages(passages, answers, q)
                    full = "\n\n".join(op)
                    out = independent(model, tok, mgr,
                                      [{"context": full, "question": q,
                                        "references": answers, "has_supporting": True}],
                                      device, args.max_new, args.max_prompt_length)[0]
                elif a == "ape":
                    out, _ = bank_read(model, tok, mgr, passages, q, device,
                                       args.max_new, args.max_prompt_length,
                                       temp=args.ape_temp, scale=args.ape_scale, adaptive=False)
                elif a == "ape_oracle":  # APE encoding ceiling under perfect selection
                    out, _ = bank_read(model, tok, mgr, oracle_passages(passages, answers, q), q,
                                       device, args.max_new, args.max_prompt_length,
                                       temp=args.ape_temp, scale=args.ape_scale, adaptive=False)
                elif a == "ours_oracle":  # our encoding ceiling under perfect selection
                    out, _ = bank_read(model, tok, mgr, oracle_passages(passages, answers, q), q,
                                       device, args.max_new, args.max_prompt_length,
                                       temp=1.0, scale=1.0, adaptive=False)
                else:  # ours
                    out, _ = bank_read(model, tok, mgr, passages, q, device,
                                       args.max_new, args.max_prompt_length,
                                       temp=1.0, scale=1.0, adaptive=args.adaptive)
                # the generation prompt asks for <answer>..</answer>; strip the tags before
                # scoring (best_f1 of "<answer>Ozalj</answer>" vs "Ozalj" is wrecked by the tags).
                ans, _ = extract_box_answer(out)
                scores[a] += best_f1(ans, answers)
                preds[a].append({"task": task, "question": q, "gold": answers, "pred": out})
            n += 1
            if n % 25 == 0:
                run = "  ".join(f"{a}={100.0*scores[a]/n:.1f}" for a in arms)
                print(f"  [{task} {n}/{len(data)}] {run}", flush=True)
        row = {a: round(100.0 * scores[a] / n, 2) for a in arms}
        table[task] = row
        print(f"{task:>10}" + "".join(f"{row[a]:>9.2f}" for a in arms), flush=True)
        import json as _j
        with open(os.path.join(args.output_dir, f"preds_{task}.jsonl"), "w") as pf:
            for a in arms:
                for r in preds[a]:
                    pf.write(_j.dumps({**r, "arm": a}) + "\n")

    # averages
    if table:
        avg = {a: round(sum(table[t][a] for t in table) / len(table), 2) for a in arms}
        print(f"{'AVG':>10}" + "".join(f"{avg[a]:>9.2f}" for a in arms))
        table["AVG"] = avg

    import json
    tag = "ours_adapt" if args.adaptive else "ours_all"
    out_path = os.path.join(args.output_dir, f"longbench_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "table": table}, f, indent=2)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
