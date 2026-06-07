#!/usr/bin/env python3
"""RAFT / best-of-N reward training for the bank reader — the one route whose fixed point
is NOT capped by a teacher (reward = answer correctness, so the gradient can point ABOVE
the gold-only oracle by exploiting independent encoding's distractor isolation).

Per training group: capture the bank (gold + distractors), SAMPLE N answers per query at
temperature T, score each by qa_f1 vs the gold answer, keep the BEST sample per query as a
new SFT target (gated: must beat the greedy answer's reward AND contain the gold substring,
with a length penalty to block F1-hacking), then SFT on the winners via use_loss. Inference
path is unchanged — this only shapes LoRA weights.

Go/no-go probe (probe_raft.py) confirmed best-of-16 (66.6) > oracle (64.4) on 2wiki, i.e.
the latent ability exists in sampling; RAFT fixates it into greedy.
"""
import argparse, os, sys, logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from src.datasets import load_multiquery_hotpot_groups
from scripts.train_multiquery_lora import (
    build_prompt, context_mask_for, left_pad, capture, use_loss,
)
from scripts.bench_longbench import best_f1
from scripts.bench_distract import compute_subem
from src.inference import extract_box_answer
from scripts.eval_multiquery import decode_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--warm-start", default=None, help="LoRA to continue (the trained reader seed)")
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--paragraphs-per-agent", type=int, default=6)
    p.add_argument("--num-groups", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-prompt-length", type=int, default=1600)
    p.add_argument("--mix-agents", default="2,4")
    p.add_argument("--distract-train", action="store_true", default=True)
    p.add_argument("--raft-n", type=int, default=16)
    p.add_argument("--raft-temp", type=float, default=0.8)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--beat-greedy", action="store_true", default=True,
                   help="only keep a winner if it strictly beats the greedy answer's reward")
    p.add_argument("--no-beat-greedy", dest="beat_greedy", action="store_false",
                   help="keep any correct (subem>=1) winner, even if greedy already matched it -> "
                        "higher winrate, reinforces correct outputs (fixes low-winrate RAFT)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def sample_answer(model, mgr, tok, question, device, off, max_plen, max_new, temp):
    """One bank-read decode (temp<=0 -> greedy, else multinomial). Bank already captured."""
    eos = tok.eos_token_id; pad = tok.pad_token_id or eos
    qp = build_prompt(tok, "", question)
    enc = tok([qp], return_tensors="pt", padding=True, truncation=True, max_length=max_plen)
    qids = enc["input_ids"].to(device); qattn = enc["attention_mask"].to(device)
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
        logits = out.logits[:, -1]
        t = logits.argmax(-1) if temp <= 0 else \
            torch.multinomial((logits / temp).softmax(-1), 1).squeeze(-1)
        t = torch.where(fin, torch.full_like(t, pad), t)
        fin = fin | (t == eos)
        gen = torch.cat([gen, t.unsqueeze(1)], 1)
        cur = torch.cat([cur, (~fin).long().unsqueeze(1)], 1); nxt = t.unsqueeze(1)
        if fin.all():
            break
    text = decode_texts(tok, gen, P, eos, pad)[0]
    del out, pkv                       # release the per-sample KV cache promptly
    return text


def reward(pred, refs):
    """qa_f1 with a length penalty (block F1-hacking by padding the answer)."""
    ans, _ = extract_box_answer(pred)
    f1 = best_f1(ans, refs)
    gl = max(1, len(refs[0])); pl = max(1, len(ans))
    return f1 * min(1.0, 1.5 * gl / pl)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"; tok.truncation_side = "left"
    device = "cuda"
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device)
    if args.warm_start:
        model = PeftModel.from_pretrained(base, args.warm_start, is_trainable=True)
        log.info("RAFT warm-start from %s", args.warm_start)
    else:
        lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(base, lora)
    model.train(); model.print_trainable_parameters()
    nl = model.config.num_hidden_layers
    mgr = BatchCrossCache(list(range(nl))); mgr.register(model)

    Gs = [int(x) for x in args.mix_agents.split(",")]
    per = max(1, args.num_groups // len(Gs))
    groups = []
    for gi, G in enumerate(Gs):
        groups += load_multiquery_hotpot_groups(
            split="train", n_agents=G, paragraphs_per_agent=args.paragraphs_per_agent,
            max_groups=per, seed=args.seed + gi, only_bridge=True,
            require_min_supporting=2, cross_question_distractor_pool=True)
    import random as _r
    _r.Random(args.seed).shuffle(groups)
    ctx_pool = [it["context"] for g in groups for it in g["items"]]
    rng = _r.Random(args.seed)
    log.info("RAFT groups=%d", len(groups))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    step = 0; run = 0.0; n_win = 0; n_q = 0
    for ep in range(args.epochs):
        for g in groups:
            items = g["items"]
            cap_items = items
            if args.distract_train:
                own = {it["context"] for it in items}
                k = rng.randint(0, len(items) * 2)
                pool = [c for c in rng.sample(ctx_pool, min(len(ctx_pool), k * 3 + 1)) if c not in own]
                cap_items = items + [{"context": c, "question": "", "references": [""],
                                      "has_supporting": False} for c in pool[:k]]
            # capture once under no_grad (bank is fixed evidence); sample under no_grad
            model.eval()
            with torch.no_grad():
                off, _, _ = capture(model, mgr, tok, cap_items, device, args.max_prompt_length, False)
                winners = []
                for it in items:
                    refs = it["references"]
                    greedy = sample_answer(model, mgr, tok, it["question"], device, off,
                                           args.max_prompt_length, args.max_new, 0.0)
                    gr = reward(greedy, refs)
                    best_s, best_r = greedy, gr
                    for _ in range(args.raft_n):
                        s = sample_answer(model, mgr, tok, it["question"], device, off,
                                          args.max_prompt_length, args.max_new, args.raft_temp)
                        r = reward(s, refs)
                        if r > best_r:
                            best_s, best_r = s, r
                    n_q += 1
                    ans, _ = extract_box_answer(best_s)
                    keep = compute_subem(ans, refs) >= 1
                    if args.beat_greedy:
                        keep = keep and best_r > gr + 1e-6
                    if keep:
                        winners.append({**it, "references": [ans]}); n_win += 1
            model.train()
            if not winners:
                continue
            # SFT on the winners over the SAME captured bank (recapture WITH grad on query path)
            off2, _, _ = capture(model, mgr, tok, cap_items, device, args.max_prompt_length, False)
            loss = use_loss(model, mgr, tok, winners, device, off2, args.max_prompt_length)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            run += loss.item(); step += 1
            del loss
            mgr.bank = {}                       # drop the captured bank KV between groups
            if step % 8 == 0:
                torch.cuda.empty_cache()
            if step % 25 == 0:
                log.info("ep%d step%d loss=%.4f winrate=%.2f", ep, step, run / 25,
                         n_win / max(1, n_q)); run = 0.0
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    log.info("saved RAFT LoRA to %s  (final winrate=%.3f)", args.output_dir, n_win / max(1, n_q))


if __name__ == "__main__":
    main()
