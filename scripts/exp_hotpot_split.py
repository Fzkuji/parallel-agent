#!/usr/bin/env python3
"""
HotpotQA Document Split Experiment

Each question needs 2 supporting paragraphs.
Agent 1 sees paragraph 1 only, Agent 2 sees paragraph 2 only.
They must collaborate to answer — perfect multi-agent asymmetry.

Baselines:
  Oracle:      each agent sees ALL paragraphs (upper bound)
  Independent: each agent sees only its OWN paragraph (lower bound)
  All-in-One:  both paragraphs concatenated into one long prompt
  SSA (ours):  each agent sees its own paragraph, shares hidden state

Expected: Oracle > All-in-One >= SSA > Independent

Key claim: SSA enables collaboration without extra context length.

Usage:
    python scripts/exp_hotpot_split.py \
        --model /autodl-fs/data/models/Qwen2.5-7B-Instruct \
        --eval-samples 200
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/autodl-fs/data/models/Qwen2.5-7B-Instruct")
    p.add_argument("--eval-samples", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--ssa-alpha", type=float, default=0.9)
    p.add_argument("--output-dir", default="/root/autodl-tmp/hotpot_split_exp")
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def score_answer(pred: str, gold_list: list) -> float:
    """Simple token-level F1 between prediction and gold answers."""
    from src.evaluation.basic import compute_f1
    if not gold_list:
        return 0.0
    return max(compute_f1(pred, g) for g in gold_list)


def run_single_question(model, tokenizer, context: str, question: str,
                        references: list, max_new_tokens: int) -> float:
    """Run a single question with given context, return F1."""
    import torch
    from src.templates import build_chat_prompt
    from src.evaluation.basic import compute_f1

    # Build prompt directly
    system = "You are a helpful assistant. Answer the question based on the given context. Be concise."
    user = f"Context:\n{context[:2000]}\n\nQuestion: {question}\n\nAnswer:"
    prompt = build_chat_prompt(tokenizer, user, system_prompt=system)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(
        next(model.parameters()).device
    )
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        # Compute F1 against references
        return compute_f1(generated, references)
    except Exception as e:
        logger.warning(f"run_single_question error: {e}")
    return 0.0


def run_oracle(model, tokenizer, groups, max_new_tokens):
    """Oracle: full context."""
    import torch
    total, count = 0.0, 0
    for g in groups:
        f1 = run_single_question(
            model, tokenizer, g["context"], g["questions"][0]["text"],
            g["questions"][0]["references"], max_new_tokens,
        )
        total += f1
        count += 1
        torch.cuda.empty_cache()
    return total / max(count, 1)


def run_independent(model, tokenizer, groups, max_new_tokens):
    """Independent: each agent sees only its own paragraph, take best F1."""
    import torch
    total, count = 0.0, 0
    for g in groups:
        q = g["questions"][0]
        best_f1 = 0.0
        for i, split in enumerate(g["splits"]):
            f1 = run_single_question(
                model, tokenizer, split, q["text"],
                q["references"], max_new_tokens,
            )
            best_f1 = max(best_f1, f1)
            torch.cuda.empty_cache()
        total += best_f1
        count += 1
    return total / max(count, 1)


def run_all_in_one(model, tokenizer, groups, max_new_tokens):
    """All-in-One: concatenate both paragraphs."""
    import torch
    total, count = 0.0, 0
    for g in groups:
        q = g["questions"][0]
        combined = "\n\n".join(g["splits"])
        f1 = run_single_question(
            model, tokenizer, combined, q["text"],
            q["references"], max_new_tokens,
        )
        total += f1
        count += 1
        torch.cuda.empty_cache()
    return total / max(count, 1)


def run_ssa(model, tokenizer, groups, max_new_tokens, ssa_alpha=0.9, checkpoint=None):
    """SSA: 2 agents, each sees own paragraph, share hidden state z."""
    import torch
    from src.cross_batch import CrossBatchGenerator, SharedStateAttention
    from src.cross_batch.shared_state import SharedStateAttention as SSA
    from src.evaluation.basic import compute_f1
    from src.templates import build_chat_prompt

    device = str(next(model.lm_head.parameters()).device)
    hidden_size = model.config.hidden_size
    model_dtype = next(model.parameters()).dtype

    ssa = SSA(
        hidden_size=hidden_size, num_heads=8,
        alpha=ssa_alpha, use_gate=True,
    ).to(device=device, dtype=model_dtype)

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        ssa.load_state_dict(ckpt.get("cross_batch_module", ckpt))
        logger.info(f"Loaded checkpoint: {checkpoint}")
    ssa.eval()

    system = "You are a helpful assistant. Answer the question based on the given context. Be concise."

    total, count = 0.0, 0
    for g in groups:
        q_data = g["questions"][0]
        question = q_data["text"]
        references = q_data["references"]
        splits = g["splits"]

        # Build prompts for each agent (own chunk only)
        prompts = []
        for split in splits:
            user = f"Context:\n{split[:1500]}\n\nQuestion: {question}\n\nAnswer:"
            prompts.append(build_chat_prompt(tokenizer, user, system_prompt=system))

        try:
            # Tokenize batch (left pad)
            tokenizer.padding_side = "left"
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024)
            tokenizer.padding_side = "right"
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            prompt_len = input_ids.shape[1]

            with torch.no_grad():
                # Use CrossBatchGenerator for SSA inference
                gen = CrossBatchGenerator(
                    model=model, tokenizer=tokenizer,
                    cross_batch_module=ssa,
                    mix_method="shared_state", device=device,
                )
                out = gen.generate(
                    input_ids=input_ids, attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens, do_sample=False,
                    enable_cross_batch=True,
                )

            # Decode and score — use best F1 across agents
            best_f1 = 0.0
            for seq in out["sequences"]:
                generated = tokenizer.decode(
                    seq[prompt_len:], skip_special_tokens=True
                ).strip()
                f1 = compute_f1(generated, references)
                best_f1 = max(best_f1, f1)

            total += best_f1
            count += 1
        except Exception as e:
            logger.warning(f"SSA error: {e}")
        torch.cuda.empty_cache()

    return total / max(count, 1)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.datasets.hotpot_split import load_hotpot_split_groups, groups_to_cqa_format

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    logger.info(f"Loading HotpotQA split data (n={args.eval_samples}) ...")
    raw_groups = load_hotpot_split_groups(
        "validation", max_groups=args.eval_samples,
    )
    groups = groups_to_cqa_format(raw_groups)
    logger.info(f"Loaded {len(groups)} groups")

    # Type breakdown
    bridge = sum(1 for g in raw_groups if g.get("type") == "bridge")
    comparison = sum(1 for g in raw_groups if g.get("type") == "comparison")
    logger.info(f"Types: bridge={bridge}, comparison={comparison}")

    results = {}

    logger.info("\n=== Oracle (full context) ===")
    f1 = run_oracle(model, tokenizer, groups, args.max_new_tokens)
    results["oracle"] = f1
    logger.info(f"Oracle F1: {f1:.4f}")

    logger.info("\n=== Independent (own paragraph only) ===")
    f1 = run_independent(model, tokenizer, groups, args.max_new_tokens)
    results["independent"] = f1
    logger.info(f"Independent F1: {f1:.4f}")

    logger.info("\n=== All-in-One (paragraphs concatenated) ===")
    f1 = run_all_in_one(model, tokenizer, groups, args.max_new_tokens)
    results["all_in_one"] = f1
    logger.info(f"All-in-One F1: {f1:.4f}")

    logger.info("\n=== SSA (parallel, shared state, untrained) ===")
    f1 = run_ssa(model, tokenizer, groups, args.max_new_tokens,
                  ssa_alpha=args.ssa_alpha, checkpoint=args.checkpoint)
    results["ssa_untrained"] = f1
    logger.info(f"SSA (untrained) F1: {f1:.4f}")

    # Summary
    logger.info("\n" + "="*50)
    logger.info("SUMMARY (HotpotQA 2-agent split):")
    logger.info(f"{'Strategy':<25} {'F1':>8}")
    logger.info("-"*35)
    for name, f1 in results.items():
        logger.info(f"{name:<25} {f1:>8.4f}")

    gap = results.get("all_in_one", 0) - results.get("independent", 0)
    logger.info(f"\nCollaboration gap (All-in-One - Independent): {gap:+.4f}")
    logger.info("(Positive gap = collaboration helps, SSA should close this gap)")

    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "n_groups": len(groups),
            "bridge": bridge, "comparison": comparison,
            "results": results,
        }, f, indent=2)
    logger.info(f"\nSaved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
