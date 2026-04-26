#!/usr/bin/env python3
"""
Document Split QA Experiment

Multi-agent information asymmetry scenario:
- Document is split into N chunks, each agent sees only one chunk
- Question is the same for all agents
- Agents must collaborate (via SSA) to answer correctly

Baselines:
1. Oracle: each agent sees the FULL document (upper bound)
2. Independent: each agent sees only its OWN chunk (no collaboration, lower bound)
3. All-in-One: all chunks concatenated into one long context
4. Sequential: agents run one by one, each seeing previous agents' outputs
5. SSA (ours): agents run in parallel, sharing state via running z

Expected: Oracle >= SSA > Sequential > Independent
Key insight: SSA should outperform Independent by leveraging cross-chunk info,
             without the error propagation of Sequential.

Usage:
    python scripts/exp_doc_split.py \
        --model /autodl-fs/data/models/Qwen2.5-7B-Instruct \
        --n-agents 3 --eval-samples 100
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
    p.add_argument("--n-agents", type=int, default=3, help="Number of document splits")
    p.add_argument("--eval-samples", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--ssa-alpha", type=float, default=0.9)
    p.add_argument("--output-dir", default="/root/autodl-tmp/doc_split_exp")
    p.add_argument("--checkpoint", default=None, help="Path to trained SSA checkpoint")
    return p.parse_args()


def build_prompt(tokenizer, question: str, context: str) -> str:
    """Build a prompt for a single agent."""
    from src.prompts import build_single_prompt
    from src.templates import build_chat_prompt
    from src.models import Question

    q = Question(
        qid="Q1", text=question, priority=1.0,
        answer_tokens=20, type_hint=None, references=[],
    )
    system_prompt, user_prompt = build_single_prompt(context, q, dataset="squad")
    return build_chat_prompt(tokenizer, user_prompt, system_prompt=system_prompt)


def run_oracle(model, tokenizer, groups, max_new_tokens):
    """Oracle: each agent sees the full document."""
    from src.models import Question
    from src.strategies.sequential_batch import run_sequential_strategy

    total_f1, count = 0.0, 0
    for group in groups:
        q = group["questions"][0]
        questions = [Question(
            qid="Q1", text=q["text"], priority=1.0,
            answer_tokens=q.get("answer_tokens", 20), type_hint=None,
            references=q["references"],
        )]
        try:
            result = run_sequential_strategy(
                background=group["context"],
                questions=questions,
                tokenizer=tokenizer, model=model,
                max_new_tokens=max_new_tokens, dataset="squad",
            )
            if result.metrics:
                total_f1 += result.metrics.get("f1", 0)
                count += 1
        except Exception as e:
            logger.warning(f"Oracle error: {e}")
    return total_f1 / max(count, 1)


def run_independent(model, tokenizer, groups, max_new_tokens):
    """Independent: each agent sees only its own chunk."""
    import torch
    from src.models import Question
    from src.strategies.sequential_batch import run_sequential_strategy
    from src.evaluation.basic import compute_f1

    total_f1, count = 0.0, 0
    for group in groups:
        group_f1 = []
        for qi, q in enumerate(group["questions"]):
            local_ctx = group["splits"][qi]
            questions = [Question(
                qid="Q1", text=q["text"], priority=1.0,
                answer_tokens=q.get("answer_tokens", 20), type_hint=None,
                references=q["references"],
            )]
            try:
                result = run_sequential_strategy(
                    background=local_ctx,
                    questions=questions,
                    tokenizer=tokenizer, model=model,
                    max_new_tokens=max_new_tokens, dataset="squad",
                )
                if result.metrics:
                    group_f1.append(result.metrics.get("f1", 0))
            except Exception as e:
                logger.warning(f"Independent error: {e}")
                group_f1.append(0.0)
        if group_f1:
            # Take max F1 across agents (best agent gets credit)
            total_f1 += max(group_f1)
            count += 1
        torch.cuda.empty_cache()
    return total_f1 / max(count, 1)


def run_all_in_one(model, tokenizer, groups, max_new_tokens):
    """All-in-One: concatenate all chunks into one prompt."""
    from src.models import Question
    from src.strategies.sequential_batch import run_sequential_strategy

    total_f1, count = 0.0, 0
    for group in groups:
        q = group["questions"][0]
        # Concatenate all splits
        combined_context = "\n\n".join([
            f"[Document Part {i+1}]\n{s}"
            for i, s in enumerate(group["splits"])
        ])
        questions = [Question(
            qid="Q1", text=q["text"], priority=1.0,
            answer_tokens=q.get("answer_tokens", 20), type_hint=None,
            references=q["references"],
        )]
        try:
            result = run_sequential_strategy(
                background=combined_context,
                questions=questions,
                tokenizer=tokenizer, model=model,
                max_new_tokens=max_new_tokens, dataset="squad",
            )
            if result.metrics:
                total_f1 += result.metrics.get("f1", 0)
                count += 1
        except Exception as e:
            logger.warning(f"All-in-one error: {e}")
    return total_f1 / max(count, 1)


def run_ssa(model, tokenizer, groups, max_new_tokens, ssa_alpha=0.9, checkpoint=None):
    """SSA: agents run in parallel, each sees its own chunk, share state via z."""
    import torch
    from src.models import Question
    from src.cross_batch import CrossBatchGenerator, SharedStateAttention
    from src.strategies.cross_batch import run_cross_batch_strategy

    device = str(next(model.lm_head.parameters()).device)
    hidden_size = model.config.hidden_size
    model_dtype = next(model.parameters()).dtype

    ssa_module = SharedStateAttention(
        hidden_size=hidden_size,
        num_heads=8,
        alpha=ssa_alpha,
        use_gate=True,
    ).to(device=device, dtype=model_dtype)

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        ssa_module.load_state_dict(ckpt.get("cross_batch_module", ckpt))
        logger.info(f"Loaded SSA checkpoint from {checkpoint}")

    ssa_module.eval()

    total_f1, count = 0.0, 0
    for group in groups:
        # Each agent gets its own chunk as context
        questions = []
        for qi, q in enumerate(group["questions"]):
            questions.append(Question(
                qid=f"agent_{qi}", text=q["text"], priority=1.0,
                answer_tokens=q.get("answer_tokens", 20), type_hint=None,
                references=q["references"],
            ))

        # Build individual contexts (one chunk per agent)
        splits = group["splits"]

        # Run multi-agent inference: each agent uses its local chunk
        # We use the first split as "background" but override per-question context
        # by using the multi-context runner
        items = []
        for qi, (q, split) in enumerate(zip(questions, splits)):
            items.append({
                "qid": q.qid,
                "question": q.text,
                "context": split,  # local chunk only
                "references": q.references,
            })

        try:
            from src.strategies.cross_batch import run_cross_batch_multi_strategy
            gen = CrossBatchGenerator(
                model=model, tokenizer=tokenizer,
                cross_batch_module=ssa_module,
                mix_method="shared_state", device=device,
            )
            result = run_cross_batch_multi_strategy(
                items=items,
                tokenizer=tokenizer, model=model,
                max_new_tokens=max_new_tokens,
                strategy_name="ssa",
                dataset="squad",
                cross_batch_generator=gen,
                mix_method="shared_state",
                enable_cross_batch=True,
            )
            if result.metrics:
                total_f1 += result.metrics.get("f1", 0)
                count += 1
        except Exception as e:
            logger.warning(f"SSA error: {e}")
        torch.cuda.empty_cache()

    return total_f1 / max(count, 1)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.datasets.doc_split import load_doc_split_groups, groups_to_cqa_format

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

    logger.info(f"Loading doc-split data (n_agents={args.n_agents}, n={args.eval_samples}) ...")
    raw_groups = load_doc_split_groups(
        "validation",
        n_agents=args.n_agents,
        max_groups=args.eval_samples,
    )
    groups = groups_to_cqa_format(raw_groups)
    logger.info(f"Loaded {len(groups)} groups")

    # Stats: how many groups have answer in a single chunk vs spanning
    in_chunk = sum(1 for g in raw_groups if g["answer_chunk_idx"] >= 0)
    logger.info(f"Answer in single chunk: {in_chunk}/{len(raw_groups)} ({100*in_chunk//max(len(raw_groups),1)}%)")
    logger.info(f"Answer spans chunks: {len(raw_groups)-in_chunk}/{len(raw_groups)}")

    results = {}

    logger.info("\n=== Oracle (full document) ===")
    f1 = run_oracle(model, tokenizer, groups, args.max_new_tokens)
    results["oracle"] = f1
    logger.info(f"Oracle F1: {f1:.4f}")
    torch.cuda.empty_cache()

    logger.info("\n=== Independent (own chunk only) ===")
    f1 = run_independent(model, tokenizer, groups, args.max_new_tokens)
    results["independent"] = f1
    logger.info(f"Independent F1: {f1:.4f}")
    torch.cuda.empty_cache()

    logger.info("\n=== All-in-One (all chunks concatenated) ===")
    f1 = run_all_in_one(model, tokenizer, groups, args.max_new_tokens)
    results["all_in_one"] = f1
    logger.info(f"All-in-One F1: {f1:.4f}")
    torch.cuda.empty_cache()

    logger.info("\n=== SSA (parallel, shared state) ===")
    f1 = run_ssa(model, tokenizer, groups, args.max_new_tokens,
                  ssa_alpha=args.ssa_alpha, checkpoint=args.checkpoint)
    results["ssa"] = f1
    logger.info(f"SSA F1: {f1:.4f}")

    # Summary
    logger.info("\n" + "="*50)
    logger.info("SUMMARY:")
    logger.info(f"{'Strategy':<25} {'F1':>8}")
    logger.info("-"*35)
    for name, f1 in results.items():
        logger.info(f"{name:<25} {f1:>8.4f}")

    logger.info(f"\nSSA vs Independent: {'+'if results['ssa']>results['independent'] else ''}{results['ssa']-results['independent']:.4f}")
    logger.info(f"SSA vs All-in-One:  {'+'if results['ssa']>results['all_in_one'] else ''}{results['ssa']-results['all_in_one']:.4f}")

    with open(output_dir / "results.json", "w") as f:
        json.dump({"n_agents": args.n_agents, "n_groups": len(groups),
                   "in_chunk_pct": in_chunk/max(len(raw_groups),1),
                   "results": results}, f, indent=2)
    logger.info(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
