#!/usr/bin/env python3
"""Evaluate CSA-v2 vs Independent baseline on SQuAD validation, G=1..5.

Loads the same base model once. For each group size G:
  - Independent: forward each question alone (no CSA), greedy decode.
  - CSA-v2: same forward, but with the trained CrossSequenceAttentionV2 module
    inserted at the last layer's output during decode.

Reports EM and F1 per condition per G.
"""

import argparse
import json
import logging
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cross_batch.attention import CrossSequenceAttentionV2
from src.cross_batch.generator import CrossBatchGenerator
from src.datasets.squad import load_squad_groups
from src.evaluation.basic import compute_em, compute_f1
from src.inference import extract_answer
from src.models import Question
from src.prompts import build_single_prompt
from src.templates import build_chat_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/root/autodl-fs/models/Qwen2.5-7B-Instruct")
    p.add_argument("--checkpoint", required=True, help="Path to CSA-v2 best/final_model.pt")
    p.add_argument("--num-eval-contexts", type=int, default=80)
    p.add_argument("--group-sizes", type=str, default="1,2,3,4,5")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="/root/autodl-tmp/work/csa_v2_eval")
    return p.parse_args()


def build_prompts(tokenizer, ctx, qs):
    prompts = []
    for q in qs:
        sp, up = build_single_prompt(ctx, q, dataset="squad")
        prompts.append(build_chat_prompt(tokenizer, up, system_prompt=sp))
    return prompts


def make_questions(group, group_size):
    qs = []
    for q_idx, q in enumerate(group["questions"][:group_size]):
        qs.append(
            Question(
                qid=f"G{q_idx}",
                text=q["text"],
                priority=1.0,
                answer_tokens=q.get("answer_tokens", 12),
                type_hint=None,
                references=q.get("references", []),
            )
        )
    return qs


@torch.no_grad()
def run_condition(generator, tokenizer, eval_groups, group_size, enable_csa, max_new_tokens, device):
    em_total, f1_total, n = 0.0, 0.0, 0
    for group in eval_groups:
        if len(group["questions"]) < group_size:
            continue
        qs = make_questions(group, group_size)
        prompts = build_prompts(tokenizer, group["context"], qs)
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536)
        outputs = generator.generate(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            enable_cross_batch=enable_csa,
        )
        prompt_len = enc["input_ids"].shape[1]
        for i, q in enumerate(qs):
            tokens = []
            for t in outputs["sequences"][i][prompt_len:].tolist():
                if t in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                    break
                tokens.append(t)
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            ans, _ = extract_answer(text, "squad")
            em_total += compute_em(ans, q.references)
            f1_total += compute_f1(ans, q.references)
            n += 1
    if n == 0:
        return {"em": 0.0, "f1": 0.0, "n": 0}
    return {"em": em_total / n * 100, "f1": f1_total / n * 100, "n": n}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    group_sizes = [int(x) for x in args.group_sizes.split(",")]
    max_g = max(group_sizes)

    log.info("Loading model %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    hidden_size = model.config.hidden_size

    log.info("Loading checkpoint %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    csa = CrossSequenceAttentionV2(hidden_size=hidden_size, num_heads=8, use_gate=True, adaptive_top_k=True)
    csa.load_state_dict(ckpt["cross_batch_module"])
    if "lm_head" in ckpt:
        # Restore the trained lm_head too -- it co-evolved with CSA.
        model.lm_head.load_state_dict(ckpt["lm_head"])
        log.info("Restored lm_head from checkpoint")

    device = "cuda"
    csa.to(device).to(torch.bfloat16)
    csa.eval()

    generator = CrossBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=csa,
        mix_method="csa_v2",
        mix_layer=-1,
        device=device,
    )

    log.info("Loading SQuAD validation groups (need >=%d Qs/ctx, max %d ctxs)",
             max_g, args.num_eval_contexts)
    eval_groups = load_squad_groups(
        split="validation",
        min_questions=max_g,
        max_questions=max_g,
        max_contexts=args.num_eval_contexts,
        fixed_question_count=max_g,
        seed=args.seed,
    )
    log.info("Got %d eval groups", len(eval_groups))

    results = {}
    for G in group_sizes:
        for cond, enable in [("Independent", False), ("CSA-v2", True)]:
            r = run_condition(
                generator, tokenizer, eval_groups, G,
                enable_csa=enable,
                max_new_tokens=args.max_new_tokens, device=device,
            )
            key = f"G{G}_{cond}"
            results[key] = r
            log.info("  G=%d  %-12s  EM=%.2f  F1=%.2f  n=%d", G, cond, r["em"], r["f1"], r["n"])

    log.info("\n=== SUMMARY (EM%%) ===")
    log.info("%-13s | %s", "Strategy", "  ".join(f"G={g:>2}" for g in group_sizes))
    for cond in ["Independent", "CSA-v2"]:
        row = [f"{results[f'G{g}_{cond}']['em']:5.2f}" for g in group_sizes]
        log.info("%-13s | %s", cond, "  ".join(row))

    out = os.path.join(args.output_dir, "results.json")
    with open(out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()
