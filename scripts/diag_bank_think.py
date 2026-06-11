#!/usr/bin/env python3
"""Does THINKING mitigate (or worsen) bank-read degeneration? — the deciding 2x2.

Crosses model x thinking on the SAME bank-read decode path:
  model    : Qwen2.5-7B (no native thinking, no QK-norm) | Qwen3-8B (native thinking, QK-norm)
  thinking : enable_thinking=False (direct answer) | True (Qwen3 <think>...</think>)

For each cell we report degeneration broken into the two segments of the output, because the
failure may live in only one of them:
  think_run = max identical-token run INSIDE <think>...</think>
  ans_run   = max identical-token run in the post-</think> answer (or whole gen if no think tag)

Decision rule the user set:
  - if thinking LOWERS the run (vs no-think) on Qwen3 -> thinking mitigates -> pursue think-distill.
  - if thinking does NOT help (same/worse) -> abandon the think route, go selective-read / Qwen2.5.

Qwen2.5 is the control: enable_thinking is a no-op there, so its two columns should match and stay
at run~1.0 — proving the decode path itself is sound and any Qwen3 movement is thinking-specific.
"""
import argparse, os, sys, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.bench_distract import build_examples, make_paras
from scripts.eval_multiquery import context_mask_for
from src.prompts import build_single_prompt
from src.templates import build_chat_prompt
from src.models import Question


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-path", default=None,
                   help="merge a LoRA before diagnosing: does training remove the degeneration?")
    p.add_argument("--tag", required=True)
    p.add_argument("--n-items", type=int, default=12)
    p.add_argument("--n-paras", type=int, default=4)
    p.add_argument("--seg-cap", type=int, default=768)
    p.add_argument("--max-new", type=int, default=320)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                   help="fp32 isolates bf16 numerical overflow as the degeneration cause")
    p.add_argument("--gpu", type=int, default=5)
    return p.parse_args()


def prompt(tok, context, question, think):
    q = Question(qid="q", text=question, priority=1.0, answer_tokens=12, type_hint=None, references=[])
    sp, up = build_single_prompt(context, q, dataset="hotpot")
    return build_chat_prompt(tok, up, system_prompt=sp, enable_thinking=think)


def max_run(ids):
    if not ids:
        return 0
    best = cur = 1
    for i in range(1, len(ids)):
        cur = cur + 1 if ids[i] == ids[i - 1] else 1
        best = max(best, cur)
    return best


@torch.no_grad()
def bank_gen(model, tok, mgr, chunks, question, device, max_new, seg_cap, think):
    cp = [prompt(tok, c, question, think) for c in chunks]
    enc = tok(cp, return_tensors="pt", padding=True, truncation=True, max_length=seg_cap)
    cids = enc["input_ids"].to(device); cattn = enc["attention_mask"].to(device)
    cm = context_mask_for(tok, cp, chunks, cids.shape[1], cattn, 1600).to(device)
    mgr.set_enabled(True); mgr.exclude_self = False; mgr.set_allowed(None); mgr.set_realign(1.0, 1.0)
    mgr.start_capture(cm)
    for s in range(0, len(chunks), 8):
        mgr.context_mask = cm[s:s + 8].bool(); mgr.set_valid(cattn[s:s + 8])
        model(input_ids=cids[s:s + 8], attention_mask=cattn[s:s + 8], use_cache=False)
    off = int(cattn.sum(1).max().item())
    qp = prompt(tok, "", question, think)
    enc = tok([qp], return_tensors="pt", add_special_tokens=False)
    qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    mgr.start_use()
    pos = (qattn.long().cumsum(1) - 1).clamp(min=0) + off
    nxt_pos = off + qattn.sum(1)
    cur = qattn.clone()
    fin = torch.zeros(1, dtype=torch.bool, device=device); pkv = None; nxt = qids.clone()
    out_ids = []
    for step in range(max_new):
        mgr.set_valid(cur)
        mgr.set_query_rows(cur.bool() if step == 0 else torch.ones(1, 1, dtype=torch.bool, device=device))
        pid = pos if step == 0 else nxt_pos.view(1, 1)
        out = model(input_ids=nxt, attention_mask=cur, position_ids=pid, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if step > 0:
            nxt_pos = nxt_pos + 1
        t = out.logits[:, -1].argmax(-1); t = torch.where(fin, torch.full_like(t, pad), t)
        fin = fin | (t == eos)
        out_ids.append(int(t))
        cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break
    return out_ids


def split_think(tok, ids):
    """split generated ids into (think_ids, answer_ids) on the </think> boundary."""
    try:
        end = tok.convert_tokens_to_ids("</think>")
    except Exception:
        end = None
    if end is not None and end in ids:
        i = ids.index(end)
        return ids[:i], ids[i + 1:]
    return [], ids  # no think tag -> all of it is "answer"


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu); device = f"cuda:{args.gpu}"
    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dt, attn_implementation="sdpa").to(device).eval()
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload().eval()
        print("loaded LoRA:", args.lora_path, flush=True)
    mgr = BatchCrossCache(list(range(len(model.model.layers)))); mgr.register(model)
    parsed, pool = build_examples(
        "/mnt/data/zichuanfu/.cache/huggingface/datasets/RUC-NLPIR___flash_rag_datasets",
        "2wikimultihopqa", args.n_items, 42)
    items = [{"paras": make_paras(ex, args.n_paras, pool, rng), "q": ex["question"]}
             for ex in parsed[:args.n_items]]

    for think in (False, True):
        tlabel = "think" if think else "nothink"
        t_runs, a_runs, had_think, samples = [], [], 0, None
        for it in items:
            ids = bank_gen(model, tok, mgr, it["paras"], it["q"], device, args.max_new, args.seg_cap, think)
            th, an = split_think(tok, ids)
            if th:
                had_think += 1
            t_runs.append(max_run(th)); a_runs.append(max_run(an))
            if samples is None:
                samples = tok.decode(ids)
            torch.cuda.empty_cache()
        n = len(items)
        a_degen = sum(1 for r in a_runs if r >= 5) / n
        print(f"[{args.tag:8s} {args.dtype:4s} {tlabel:7s}] think_run={sum(t_runs)/n:5.1f}  "
              f"ans_run={sum(a_runs)/n:5.1f}  ans_degen={a_degen:.2f}  "
              f"had_think={had_think}/{n}  sample={samples[:80]!r}", flush=True)


if __name__ == "__main__":
    main()
