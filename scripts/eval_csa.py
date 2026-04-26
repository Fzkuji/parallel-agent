#!/usr/bin/env python3
"""Unified CSA-v2 evaluation script.

Supports two datasets:
  - squad      (validation, group_sizes=[1,2,3,4,5])
  - dhotpot    (distributed-context multi-hop, fixed n_agents)

Reports per-condition EM/F1, and for dhotpot also splits by has_supporting.

Single-GPU usage (no DDP needed for eval):
    python scripts/eval_csa.py --dataset dhotpot \\
        --model-path /path/to/Qwen2.5-7B-Instruct \\
        --checkpoint ./out/dhotpot_csa/best_model.pt \\
        --output-dir ./out/eval_dhotpot
"""

import argparse
import json
import logging
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cross_batch.attention import CrossSequenceAttentionV2
from src.cross_batch.generator import CrossBatchGenerator
from src.datasets import load_distributed_hotpot_groups, load_squad_groups
from src.evaluation.basic import compute_em, compute_f1
from src.inference import extract_answer
from src.models import Question
from src.prompts import build_single_prompt
from src.templates import build_chat_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--dataset", choices=["squad", "dhotpot"], required=True)
    p.add_argument("--num-eval-groups", type=int, default=200)
    p.add_argument("--group-sizes", type=str, default="1,2,3,4,5",
                   help="for squad only: comma-separated G values to test")
    p.add_argument("--n-agents", type=int, default=4,
                   help="for dhotpot only: G")
    p.add_argument("--paragraphs-per-agent", type=int, default=9,
                   help="for dhotpot only")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-prompt-length", type=int, default=1536)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    # LoRA must match training config to load adapter weights
    p.add_argument("--lora-rank", type=int, default=0)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", type=str, default="q_proj,k_proj,v_proj,o_proj")
    return p.parse_args()


def make_questions(items_or_questions, dataset):
    if dataset == "squad":
        # squad group format: {"context": ..., "questions": [{"text", "references"}]}
        return [
            Question(qid=f"Q{i}", text=q["text"], priority=1.0,
                     answer_tokens=q.get("answer_tokens", 12), type_hint=None,
                     references=q.get("references", []))
            for i, q in enumerate(items_or_questions)
        ]
    else:
        return [
            Question(qid=f"A{i}", text=it["question"], priority=1.0,
                     answer_tokens=it.get("answer_tokens", 12), type_hint=None,
                     references=it.get("references", []))
            for i, it in enumerate(items_or_questions)
        ]


def build_prompts(tokenizer, items, dataset, shared_context=None):
    prompts = []
    for it, q in zip(items, make_questions(items, dataset)):
        ctx = shared_context if dataset == "squad" else it["context"]
        sp, up = build_single_prompt(ctx, q, dataset=dataset if dataset != "dhotpot" else "hotpot")
        prompts.append(build_chat_prompt(tokenizer, up, system_prompt=sp))
    return prompts


@torch.no_grad()
def run_squad_condition(generator, tokenizer, eval_groups, group_size,
                        enable_csa, max_new_tokens, max_prompt_len, device):
    em_total, f1_total, n = 0.0, 0.0, 0
    for group in eval_groups:
        if len(group["questions"]) < group_size:
            continue
        items = group["questions"][:group_size]
        prompts = build_prompts(tokenizer, items, dataset="squad",
                                shared_context=group["context"])
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_prompt_len)
        out = generator.generate(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            max_new_tokens=max_new_tokens, do_sample=False,
            enable_cross_batch=enable_csa,
        )
        plen = enc["input_ids"].shape[1]
        for i, q in enumerate(make_questions(items, "squad")):
            tokens = []
            for t in out["sequences"][i][plen:].tolist():
                if t in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                    break
                tokens.append(t)
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            ans, _ = extract_answer(text, "squad")
            em_total += compute_em(ans, q.references)
            f1_total += compute_f1(ans, q.references)
            n += 1
    return {"em": 100 * em_total / max(n, 1), "f1": 100 * f1_total / max(n, 1), "n": n}


@torch.no_grad()
def run_dhotpot_condition(generator, tokenizer, eval_groups, enable_csa,
                          max_new_tokens, max_prompt_len, device):
    agg = {k: 0.0 for k in [
        "all_em", "all_f1", "supp_em", "supp_f1", "nosupp_em", "nosupp_f1"
    ]}
    agg.update({"n": 0, "supp_n": 0, "nosupp_n": 0})
    for group in eval_groups:
        items = group["items"]
        prompts = build_prompts(tokenizer, items, dataset="dhotpot")
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_prompt_len)
        out = generator.generate(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            max_new_tokens=max_new_tokens, do_sample=False,
            enable_cross_batch=enable_csa,
        )
        plen = enc["input_ids"].shape[1]
        for i, it in enumerate(items):
            tokens = []
            for t in out["sequences"][i][plen:].tolist():
                if t in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                    break
                tokens.append(t)
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            ans, _ = extract_answer(text, "hotpot")
            em = compute_em(ans, it["references"])
            f1 = compute_f1(ans, it["references"])
            agg["all_em"] += em; agg["all_f1"] += f1; agg["n"] += 1
            if it.get("has_supporting"):
                agg["supp_em"] += em; agg["supp_f1"] += f1; agg["supp_n"] += 1
            else:
                agg["nosupp_em"] += em; agg["nosupp_f1"] += f1; agg["nosupp_n"] += 1

    pct = lambda a, b: 100 * a / b if b else 0.0
    return {
        "em": pct(agg["all_em"], agg["n"]),
        "f1": pct(agg["all_f1"], agg["n"]),
        "n": agg["n"],
        "supp_em": pct(agg["supp_em"], agg["supp_n"]),
        "supp_f1": pct(agg["supp_f1"], agg["supp_n"]),
        "supp_n": agg["supp_n"],
        "nosupp_em": pct(agg["nosupp_em"], agg["nosupp_n"]),
        "nosupp_f1": pct(agg["nosupp_f1"], agg["nosupp_n"]),
        "nosupp_n": agg["nosupp_n"],
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Loading model %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    hidden_size = model.config.hidden_size

    if args.lora_rank > 0:
        targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        log.info("Wrapping model with LoRA r=%d", args.lora_rank)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            target_modules=targets, lora_dropout=args.lora_dropout,
            bias="none", task_type=TaskType.CAUSAL_LM,
        ))

    model.eval()

    log.info("Loading checkpoint %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    csa = CrossSequenceAttentionV2(hidden_size=hidden_size, num_heads=8,
                                   use_gate=True, adaptive_top_k=True)
    missing, unexpected = csa.load_state_dict(ckpt["cross_batch_module"], strict=False)
    if missing:
        log.warning("CSA missing keys: %s", missing)
    if unexpected:
        log.warning("CSA unexpected keys: %s", unexpected)
    if "lm_head" in ckpt:
        target_lm = (model.base_model.model.lm_head if args.lora_rank > 0
                     else model.lm_head)
        target_lm.load_state_dict(ckpt["lm_head"])
        log.info("Restored lm_head")
    if "lora" in ckpt and args.lora_rank > 0:
        cur = {n: p for n, p in model.named_parameters() if 'lora' in n.lower()}
        loaded = 0
        for n, v in ckpt["lora"].items():
            if n in cur:
                cur[n].data.copy_(v.to(cur[n].device))
                loaded += 1
        log.info("Restored LoRA: %d/%d", loaded, len(ckpt["lora"]))

    device = "cuda"
    csa.to(device).to(torch.bfloat16); csa.eval()

    generator = CrossBatchGenerator(
        model=model, tokenizer=tokenizer,
        cross_batch_module=csa, mix_method="csa_v2", mix_layer=-1, device=device,
    )

    results = {}
    if args.dataset == "squad":
        group_sizes = [int(x) for x in args.group_sizes.split(",")]
        max_g = max(group_sizes)
        log.info("Loading SQuAD val groups (need >=%d Qs/ctx)", max_g)
        eval_groups = load_squad_groups(
            split="validation",
            min_questions=max_g, max_questions=max_g,
            max_contexts=args.num_eval_groups,
            fixed_question_count=max_g, seed=args.seed,
        )
        log.info("Got %d eval groups", len(eval_groups))
        for G in group_sizes:
            for cond, enable in [("Independent", False), ("CSA-v2", True)]:
                r = run_squad_condition(generator, tokenizer, eval_groups, G,
                                        enable, args.max_new_tokens,
                                        args.max_prompt_length, device)
                results[f"G{G}_{cond}"] = r
                log.info("G=%d  %-12s  EM=%.2f  F1=%.2f  n=%d",
                         G, cond, r["em"], r["f1"], r["n"])

        log.info("\n=== SUMMARY EM (%%) ===")
        log.info("%-13s | %s", "Strategy", "  ".join(f"G={g}" for g in group_sizes))
        for cond in ["Independent", "CSA-v2"]:
            row = [f"{results[f'G{g}_{cond}']['em']:5.2f}" for g in group_sizes]
            log.info("%-13s | %s", cond, "  ".join(row))

    else:  # dhotpot
        log.info("Loading dhotpot val groups (n_agents=%d)", args.n_agents)
        eval_groups = load_distributed_hotpot_groups(
            split="validation", n_agents=args.n_agents,
            paragraphs_per_agent=args.paragraphs_per_agent,
            max_groups=args.num_eval_groups, seed=args.seed,
            only_bridge=True, require_min_supporting=2,
            cross_question_distractor_pool=True,
        )
        log.info("Got %d eval groups (%d total queries)",
                 len(eval_groups), len(eval_groups) * args.n_agents)
        for cond, enable in [("Independent", False), ("CSA-v2", True)]:
            r = run_dhotpot_condition(generator, tokenizer, eval_groups,
                                      enable, args.max_new_tokens,
                                      args.max_prompt_length, device)
            results[cond] = r
            log.info("%-12s overall EM=%.2f F1=%.2f  n=%d",
                     cond, r["em"], r["f1"], r["n"])
            log.info("              has_supp EM=%.2f F1=%.2f  n=%d",
                     r["supp_em"], r["supp_f1"], r["supp_n"])
            log.info("              no_supp  EM=%.2f F1=%.2f  n=%d",
                     r["nosupp_em"], r["nosupp_f1"], r["nosupp_n"])

        log.info("\n=== SUMMARY ===")
        log.info("%-12s | %-12s | %-12s | %-12s",
                 "Strategy", "overall EM", "has_supp EM", "no_supp EM")
        for cond in ["Independent", "CSA-v2"]:
            r = results[cond]
            log.info("%-12s | %-12.2f | %-12.2f | %-12.2f",
                     cond, r["em"], r["supp_em"], r["nosupp_em"])
        i, c = results["Independent"], results["CSA-v2"]
        log.info("Δ (CSA-Independent): overall=%+.2f  has_supp=%+.2f  no_supp=%+.2f",
                 c["em"] - i["em"], c["supp_em"] - i["supp_em"],
                 c["nosupp_em"] - i["nosupp_em"])

    out = os.path.join(args.output_dir, "results.json")
    with open(out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()
