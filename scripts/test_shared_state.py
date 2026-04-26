#!/usr/bin/env python3
"""
Quick sanity test for SharedStateAttention.

Compares:
1. Independent (baseline)
2. CSA-original (attention, untrained)
3. SSA (shared_state, untrained)

All untrained — just checking if SSA produces non-degenerate outputs
and if the z update logic works correctly.

Usage:
    python scripts/test_shared_state.py --model Qwen/Qwen2.5-7B-Instruct
    python scripts/test_shared_state.py --model Qwen/Qwen2.5-7B-Instruct --eval-samples 50
"""

import argparse
import os
import sys
import logging

# Use HF mirror for servers in China
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--eval-samples", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-questions", type=int, default=3)
    parser.add_argument("--max-questions", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="EMA decay for shared z (0=no memory, 1=no update)")
    parser.add_argument("--dataset", default="squad", choices=["squad", "cmb"])
    return parser.parse_args()


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.cross_batch import CrossBatchGenerator, SharedStateAttention, CrossBatchAttention
    from src.datasets.squad import load_squad_groups
    from src.datasets.cmb import load_cmb_groups
    from src.strategies.cross_batch import run_cross_batch_strategy
    from src.strategies.sequential_batch import run_sequential_strategy
    from src.evaluation import evaluate_predictions
    from src.models import Question

    args = parse_args()

    # ---- Load model ----
    logger.info(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = str(next(model.lm_head.parameters()).device)
    hidden_size = model.config.hidden_size
    model_dtype = next(model.parameters()).dtype

    # ---- Load data ----
    logger.info(f"Loading {args.dataset} ...")
    if args.dataset == "squad":
        groups = load_squad_groups(
            split="validation",
            max_contexts=args.eval_samples,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
        )
    else:
        groups = load_cmb_groups(
            split="test",
            max_groups=args.eval_samples,
        )

    logger.info(f"Loaded {len(groups)} groups")

    # ---- Build CSA modules ----
    # Original CSA (untrained)
    csa_module = CrossBatchAttention(
        hidden_size=hidden_size,
        num_heads=8,
        use_gate=True,
    ).to(device=device, dtype=model_dtype)

    # New SSA (untrained)
    ssa_module = SharedStateAttention(
        hidden_size=hidden_size,
        num_heads=8,
        alpha=args.alpha,
        use_gate=True,
    ).to(device=device, dtype=model_dtype)

    # ---- Run evaluations ----
    results = {}

    for strategy_name, module, mix_method in [
        ("independent",  None,        None),
        ("csa_original", csa_module,  "attention"),
        ("ssa",          ssa_module,  "shared_state"),
    ]:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {strategy_name}")

        all_preds = {}
        all_questions = {}
        total_em = 0
        total_f1 = 0
        count = 0

        for group in groups:
            background = group.get("background", group.get("context", ""))
            raw_questions = group["questions"]

            # Convert dicts to Question objects if needed
            questions = []
            for q in raw_questions:
                if isinstance(q, dict):
                    questions.append(Question(
                        qid=q["qid"],
                        text=q["text"],
                        priority=1.0,
                        answer_tokens=q.get("answer_tokens", 12),
                        type_hint=None,
                        references=q.get("references", []),
                    ))
                else:
                    questions.append(q)

            if module is None:
                # Independent: run each question separately (no history = independent)
                from src.strategies.sequential_batch import run_sequential_strategy
                result = run_sequential_strategy(
                    background=background,
                    questions=questions,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                    dataset=args.dataset,
                )
            else:
                gen = CrossBatchGenerator(
                    model=model,
                    tokenizer=tokenizer,
                    cross_batch_module=module,
                    mix_method=mix_method,
                    device=device,
                )
                result = run_cross_batch_strategy(
                    background=background,
                    questions=questions,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                    strategy_name=strategy_name,
                    dataset=args.dataset,
                    cross_batch_generator=gen,
                    mix_method=mix_method,
                    enable_cross_batch=True,
                )

            if result.metrics:
                em = result.metrics.get("em", result.metrics.get("accuracy", 0))
                f1 = result.metrics.get("f1", em)
                total_em += em * len(questions)
                total_f1 += f1 * len(questions)
                count += len(questions)

        avg_em = total_em / count if count > 0 else 0
        avg_f1 = total_f1 / count if count > 0 else 0
        results[strategy_name] = {"em": avg_em, "f1": avg_f1, "n": count}
        logger.info(f"  EM: {avg_em:.4f}  F1: {avg_f1:.4f}  (n={count})")

    # ---- Summary ----
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY (untrained — sanity check only):")
    logger.info(f"{'Strategy':<20} {'EM':>8} {'F1':>8}")
    logger.info("-" * 40)
    for name, r in results.items():
        logger.info(f"{name:<20} {r['em']:>8.4f} {r['f1']:>8.4f}")

    # Check SSA gate activation
    logger.info(f"\nSSA alpha={args.alpha}  (gate initialized to sigmoid(-3)≈0.047)")
    logger.info("Note: untrained gate should produce ~0 contribution; trained gate should open up.")


if __name__ == "__main__":
    main()
