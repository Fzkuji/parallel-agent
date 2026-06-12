#!/usr/bin/env python3
"""Supervised per-passage relevance scorer over the bank — the selection head that survives
high density.

Why: generative <select> fails at 16 segments (cross-segment attention competition + index
binding); zero-shot isolated scoring fails (no calibration). This trains the isolated probe:
for each (passage, question) the query reads EXACTLY ONE segment (set_allowed = eye) and is
teacher-forced on " yes"/" no" given gold has_supporting labels. No competition, no index
binding, supervised — a cross-encoder whose passage side is the cacheable parallel bank.

At inference: score all C passages in one batched forward, take top-k / threshold for the
two-stage reread.
"""
import argparse, os, sys, json, random, logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cross_batch.batch_crosscache import BatchCrossCache
from scripts.bench_standard import bp
from scripts.bench_twostage import capture
from scripts.train_grpo_bank import left_pad

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PROBE = (" Does the passage shown to you contain information needed to answer this question? "
         "Reply yes or no.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--traj", required=True, help="rows with group_items + has_supporting labels")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-segs", type=int, default=16)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"; tok.truncation_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    model.train(); model.print_trainable_parameters()
    mgr = BatchCrossCache(list(range(model.config.num_hidden_layers))); mgr.register(model)

    rows = [json.loads(l) for l in open(args.traj) if l.strip()]
    rng.shuffle(rows)
    log.info("scorer rows=%d", len(rows))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    yes_ids = tok(" yes", add_special_tokens=False)["input_ids"]
    no_ids = tok(" no", add_special_tokens=False)["input_ids"]

    step = 0; run = 0.0; runa = 0.0
    for ep in range(args.epochs):
        for r in rows:
            gi = r["group_items"][: args.max_segs]
            chunks = [g["context"] for g in gi]
            labels01 = [1 if g.get("has_supporting", True) else 0 for g in gi]
            C = len(chunks)
            try:
                with torch.no_grad():
                    off = capture(model, tok, mgr, chunks, r["question"], device,
                                  768, args.max_prompt_length)
                probe = bp(tok, "", r["question"] + PROBE, False)
                qids = tok(probe, add_special_tokens=False)["input_ids"]
                ids_l, lab_l = [], []
                for y in labels01:
                    t = yes_ids if y else no_ids
                    ids_l.append(qids + t)
                    lab_l.append([-100] * len(qids) + t)
                pad = tok.pad_token_id or tok.eos_token_id
                ids, attn = left_pad(ids_l, pad, device)
                labels, _ = left_pad(lab_l, -100, device)
                labels = labels.masked_fill(attn == 0, -100)
                mgr.set_allowed(torch.eye(C, dtype=torch.bool, device=device))
                pos = (attn.long().cumsum(1) - 1).clamp(min=0) + off
                mgr.set_valid(attn); mgr.set_query_rows(attn.bool()); mgr.start_use()
                out = model(input_ids=ids, attention_mask=attn, position_ids=pos,
                            labels=labels, use_cache=False)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
                run += loss.item()
                # quick train-time recall proxy: yes-logit ranking puts gold on top?
                with torch.no_grad():
                    lp = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
                    lab = labels[:, 1:]; mask = lab != -100
                    # score every row as if its target were ' yes'
                runa += 0.0
            except torch.OutOfMemoryError:
                opt.zero_grad(set_to_none=True); mgr.bank = {}
                torch.cuda.empty_cache()
                log.warning("OOM at step%d (C=%d) skipped", step, C)
            mgr.set_allowed(None); mgr.bank = {}
            step += 1
            if step % 8 == 0:
                torch.cuda.empty_cache()
            if step % 25 == 0:
                log.info("ep%d step%d loss=%.4f", ep, step, run / 25); run = 0.0
            if args.save_every and step % args.save_every == 0:
                model.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    log.info("saved passage scorer to %s", args.output_dir)


if __name__ == "__main__":
    main()
