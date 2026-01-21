#!/usr/bin/env python3
"""Evaluate cross-batch module (trained on TriviaQA) against exp2a strategies on SQuAD.

This script compares:
1. Independent: Each question answered separately
2. All-in-One: All questions in single prompt
3. Seq (Shared, Rand): Sequential with shared context, random order
4. Seq (Shared, Rand, Full): Sequential with shared context, random order, full context every turn
5. Cross-Batch: Parallel processing with cross-batch attention module

Reports metrics: EM, F1, Lenient, PromptTok, GenTok, Latency(s)

Usage:
    # Train on TriviaQA first, then evaluate on SQuAD
    python scripts/preliminary/eval_cross_batch_strategies.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --checkpoint outputs/checkpoints/triviaqa/Qwen_Qwen2.5-0.5B-Instruct_crossbatch.pt \
        --n-groups 100

    # Multi-GPU parallel
    python scripts/preliminary/eval_cross_batch_strategies.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --checkpoint outputs/checkpoints/triviaqa/Qwen_Qwen2.5-7B-Instruct_crossbatch.pt \
        --use-vllm \
        --n-groups 500
"""

from __future__ import annotations

import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

import argparse
import json
import logging
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    ExperimentConfig,
    ExperimentResult,
    LLMClient,
    compute_exact_match,
    compute_f1,
    print_summary,
    save_results,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompts
SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the given passage.
Give a short, direct answer. Do not explain or elaborate."""

SYSTEM_PROMPT_MULTI = """You are a helpful assistant. Answer all questions based on the given passage.
Give short, direct answers. Format your response as:
Q1: [answer]
Q2: [answer]
..."""


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate cross-batch vs exp2a strategies')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='Model name')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to cross-batch checkpoint (trained on TriviaQA)')
    parser.add_argument('--n-groups', type=int, default=100,
                        help='Number of SQuAD groups to evaluate (-1 for all)')
    parser.add_argument('--n-questions', type=int, default=5,
                        help='Number of questions per group')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='outputs/preliminary',
                        help='Output directory')
    parser.add_argument('--use-local', action='store_true', default=True,
                        help='Use local model (default: True)')
    parser.add_argument('--use-vllm', action='store_true',
                        help='Use vLLM for inference')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='Tensor parallel size for vLLM')
    parser.add_argument('--strategies', type=str, nargs='+',
                        default=['independent', 'all_in_one', 'seq_shared_rand', 'seq_shared_rand_full', 'cross_batch'],
                        help='Strategies to evaluate')
    parser.add_argument('--summarize', action='store_true',
                        help='Only summarize existing results, do not run experiments')
    return parser.parse_args()


def load_squad_groups(
    n_groups: int = -1,
    n_questions: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load SQuAD dataset grouped by paragraph."""
    logger.info("Loading SQuAD dataset...")

    dataset = load_dataset("rajpurkar/squad", split="validation")

    # Group questions by context
    context_groups = defaultdict(list)
    for item in dataset:
        context = item["context"]
        context_groups[context].append({
            "question": item["question"],
            "answer": item["answers"]["text"][0] if item["answers"]["text"] else "",
            "id": item["id"],
        })

    # Filter groups with exactly n_questions questions
    valid_groups = []
    for context, questions in context_groups.items():
        if len(questions) == n_questions:
            valid_groups.append({
                "context": context,
                "questions": questions,
                "n_questions": len(questions),
            })

    # Shuffle and sample
    random.seed(seed)
    random.shuffle(valid_groups)
    if n_groups > 0:
        groups = valid_groups[:n_groups]
    else:
        groups = valid_groups

    logger.info(f"Loaded {len(groups)} groups with exactly {n_questions} questions each")
    return groups


def _extract_answer(response: str) -> str:
    """Extract answer from model response."""
    response = response.strip()
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    return response


def _parse_batch_answers(response: str, n_questions: int) -> Dict[int, str]:
    """Parse batch answers from response."""
    answers = {}

    # Method 1: Q1: answer format
    for i in range(n_questions):
        if i < n_questions - 1:
            pattern = rf"Q{i+1}[:\.\)]\s*(.+?)(?=Q{i+2}[:\.\)]|\n\n|$)"
        else:
            pattern = rf"Q{i+1}[:\.\)]\s*(.+?)(?:\n\n|$)"
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            tag_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
            if tag_match:
                answers[i] = tag_match.group(1).strip()
            else:
                first_line = content.split('\n')[0].strip()
                answers[i] = first_line.rstrip('.,;:')

    if len(answers) == n_questions:
        return answers

    # Method 2: Line-by-line
    answers = {}
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    for i, line in enumerate(lines[:n_questions]):
        clean = re.sub(r"^(?:Q\d+|[\d]+)[:\.\)]\s*", "", line, flags=re.IGNORECASE)
        tag_match = re.search(r'<answer>(.*?)</answer>', clean, re.DOTALL | re.IGNORECASE)
        if tag_match:
            answers[i] = tag_match.group(1).strip()
        else:
            answers[i] = clean.rstrip('.,;:')

    return answers


def compute_lenient_match(pred: str, gold: str) -> float:
    """Compute lenient match: 1.0 if gold is substring of pred or pred is substring of gold."""
    pred_lower = pred.lower().strip()
    gold_lower = gold.lower().strip()
    if not pred_lower or not gold_lower:
        return 0.0
    if gold_lower in pred_lower or pred_lower in gold_lower:
        return 1.0
    return 0.0


def run_independent(
    groups: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """Independent: Each question answered separately."""
    logger.info("Running Independent condition...")

    total_correct = 0
    total_f1 = 0.0
    total_lenient = 0.0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for group in tqdm(groups, desc="Independent"):
        context = group["context"]
        questions = group["questions"]

        for q_item in questions:
            question = q_item["question"]
            gold_answer = q_item["answer"]

            prompt = f"""Passage:
{context}

Question: {question}"""

            pred_raw, response = client.generate(prompt, max_tokens=128, system_prompt=SYSTEM_PROMPT)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            pred = _extract_answer(pred_raw)
            is_correct = compute_exact_match(pred, gold_answer) > 0
            f1_score = compute_f1(pred, gold_answer)
            lenient = compute_lenient_match(pred, gold_answer)

            total_correct += int(is_correct)
            total_f1 += f1_score
            total_lenient += lenient
            total_questions += 1

            details.append({
                "question": question,
                "gold_answer": gold_answer,
                "prediction": pred,
                "correct": is_correct,
                "f1": f1_score,
                "lenient": lenient,
            })

    em = total_correct / total_questions if total_questions > 0 else 0
    f1 = total_f1 / total_questions if total_questions > 0 else 0
    lenient = total_lenient / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="independent",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=em,
        metrics={"em": em, "f1": f1, "lenient": lenient},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        details=details,
    )


def run_all_in_one(
    groups: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """All-in-One: All questions in single prompt."""
    logger.info("Running All-in-One condition...")

    total_correct = 0
    total_f1 = 0.0
    total_lenient = 0.0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for group in tqdm(groups, desc="All-in-One"):
        context = group["context"]
        questions = group["questions"]

        q_list = "\n".join([f"Q{i+1}: {q['question']}" for i, q in enumerate(questions)])
        prompt = f"""Passage:
{context}

Questions:
{q_list}"""

        pred, response = client.generate(prompt, max_tokens=512, system_prompt=SYSTEM_PROMPT_MULTI)
        total_latency += response.latency
        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens

        answers = _parse_batch_answers(pred, len(questions))

        for i, q_item in enumerate(questions):
            gold_answer = q_item["answer"]
            pred_answer = answers.get(i, "")
            pred_answer = _extract_answer(pred_answer)

            is_correct = compute_exact_match(pred_answer, gold_answer) > 0
            f1_score = compute_f1(pred_answer, gold_answer)
            lenient = compute_lenient_match(pred_answer, gold_answer)

            total_correct += int(is_correct)
            total_f1 += f1_score
            total_lenient += lenient
            total_questions += 1

            details.append({
                "question": q_item["question"],
                "gold_answer": gold_answer,
                "prediction": pred_answer,
                "correct": is_correct,
                "f1": f1_score,
                "lenient": lenient,
            })

    em = total_correct / total_questions if total_questions > 0 else 0
    f1 = total_f1 / total_questions if total_questions > 0 else 0
    lenient = total_lenient / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="all_in_one",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=em,
        metrics={"em": em, "f1": f1, "lenient": lenient},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        details=details,
    )


def run_seq_shared_rand(
    groups: List[Dict[str, Any]],
    client: LLMClient,
    seed: int = 42,
) -> ExperimentResult:
    """Sequential with shared context, random order, context only in first turn."""
    logger.info("Running Seq. (Shared, Rand) condition...")

    random.seed(seed)

    total_correct = 0
    total_f1 = 0.0
    total_lenient = 0.0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_sequence_length = 0
    sum_completion_tokens = 0

    for group in tqdm(groups, desc="Seq.(Shared,Rand)"):
        context = group["context"]
        questions = group["questions"].copy()
        random.shuffle(questions)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        last_prompt_tokens = 0
        last_completion_tokens = 0

        for i, q_item in enumerate(questions):
            question = q_item["question"]
            gold_answer = q_item["answer"]

            if i == 0:
                user_content = f"""Passage:
{context}

Question: {question}"""
            else:
                user_content = f"Question: {question}"

            messages.append({"role": "user", "content": user_content})
            pred_raw, response = client.generate_with_messages(messages, max_tokens=128)

            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens
            sum_completion_tokens += response.completion_tokens
            last_prompt_tokens = response.prompt_tokens
            last_completion_tokens = response.completion_tokens

            pred = _extract_answer(pred_raw)
            messages.append({"role": "assistant", "content": pred_raw})

            is_correct = compute_exact_match(pred, gold_answer) > 0
            f1_score = compute_f1(pred, gold_answer)
            lenient = compute_lenient_match(pred, gold_answer)

            total_correct += int(is_correct)
            total_f1 += f1_score
            total_lenient += lenient
            total_questions += 1

            details.append({
                "question": question,
                "gold_answer": gold_answer,
                "prediction": pred,
                "correct": is_correct,
                "f1": f1_score,
                "lenient": lenient,
                "history_length": i,
            })

        total_sequence_length += last_prompt_tokens + last_completion_tokens

    em = total_correct / total_questions if total_questions > 0 else 0
    f1 = total_f1 / total_questions if total_questions > 0 else 0
    lenient = total_lenient / total_questions if total_questions > 0 else 0
    unique_prompt_tokens = total_sequence_length - sum_completion_tokens

    return ExperimentResult(
        condition="seq_shared_rand",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=em,
        metrics={"em": em, "f1": f1, "lenient": lenient},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=unique_prompt_tokens,
        total_completion_tokens=sum_completion_tokens,
        details=details,
    )


def run_seq_shared_rand_full(
    groups: List[Dict[str, Any]],
    client: LLMClient,
    seed: int = 42,
) -> ExperimentResult:
    """Sequential with shared context, random order, full context every turn."""
    logger.info("Running Seq. (Shared, Rand, Full) condition...")

    random.seed(seed)

    total_correct = 0
    total_f1 = 0.0
    total_lenient = 0.0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_sequence_length = 0
    sum_completion_tokens = 0

    for group in tqdm(groups, desc="Seq.(Shared,Rand,Full)"):
        context = group["context"]
        questions = group["questions"].copy()
        random.shuffle(questions)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        last_prompt_tokens = 0
        last_completion_tokens = 0

        for i, q_item in enumerate(questions):
            question = q_item["question"]
            gold_answer = q_item["answer"]

            # Full context every turn
            user_content = f"""Passage:
{context}

Question: {question}"""

            messages.append({"role": "user", "content": user_content})
            pred_raw, response = client.generate_with_messages(messages, max_tokens=128)

            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens
            sum_completion_tokens += response.completion_tokens
            last_prompt_tokens = response.prompt_tokens
            last_completion_tokens = response.completion_tokens

            pred = _extract_answer(pred_raw)
            messages.append({"role": "assistant", "content": pred_raw})

            is_correct = compute_exact_match(pred, gold_answer) > 0
            f1_score = compute_f1(pred, gold_answer)
            lenient = compute_lenient_match(pred, gold_answer)

            total_correct += int(is_correct)
            total_f1 += f1_score
            total_lenient += lenient
            total_questions += 1

            details.append({
                "question": question,
                "gold_answer": gold_answer,
                "prediction": pred,
                "correct": is_correct,
                "f1": f1_score,
                "lenient": lenient,
                "history_length": i,
            })

        total_sequence_length += last_prompt_tokens + last_completion_tokens

    em = total_correct / total_questions if total_questions > 0 else 0
    f1 = total_f1 / total_questions if total_questions > 0 else 0
    lenient = total_lenient / total_questions if total_questions > 0 else 0
    unique_prompt_tokens = total_sequence_length - sum_completion_tokens

    return ExperimentResult(
        condition="seq_shared_rand_full",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=em,
        metrics={"em": em, "f1": f1, "lenient": lenient},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=unique_prompt_tokens,
        total_completion_tokens=sum_completion_tokens,
        details=details,
    )


def run_cross_batch(
    groups: List[Dict[str, Any]],
    model_name: str,
    checkpoint_path: Optional[str],
    device: str = "cuda:0",
) -> ExperimentResult:
    """Cross-batch: Parallel processing with cross-batch attention module."""
    logger.info("Running Cross-Batch condition...")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.cross_batch.attention import CrossBatchAttention
    from src.cross_batch.generator import CrossBatchGenerator
    from src.strategies.cross_batch import run_cross_batch_multi_strategy

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Load cross-batch module
    hidden_size = model.config.hidden_size
    cross_batch_module = CrossBatchAttention(hidden_size=hidden_size)

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading cross-batch checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'cross_batch_module' in checkpoint:
            cross_batch_module.load_state_dict(checkpoint['cross_batch_module'])
        if 'lm_head' in checkpoint:
            model.lm_head.load_state_dict(checkpoint['lm_head'])
    else:
        logger.warning("No checkpoint provided or found, using untrained cross-batch module")

    cross_batch_module = cross_batch_module.to(device)

    # Create generator
    generator = CrossBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        device=device,
    )

    total_correct = 0
    total_f1 = 0.0
    total_lenient = 0.0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for group_idx, group in enumerate(tqdm(groups, desc="Cross-Batch")):
        context = group["context"]
        batch_items = []
        for q_idx, q in enumerate(group["questions"]):
            batch_items.append({
                "qid": f"G{group_idx}_Q{q_idx}",
                "question": q["question"],
                "context": context,
                "references": [q["answer"]],
            })

        result = run_cross_batch_multi_strategy(
            items=batch_items,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=32,
            strategy_name="cross_batch",
            dataset="squad",
            cross_batch_generator=generator,
            enable_cross_batch=True,
        )
        total_latency += result.latency

        # Collect metrics from result details
        if result.details and "questions" in result.details:
            for q_detail in result.details["questions"]:
                # The details use "gold_answers" (list) and "final_answer"
                gold_answers = q_detail.get("gold_answers", [])
                gold = gold_answers[0] if gold_answers else ""
                pred = q_detail.get("final_answer", "")

                is_correct = compute_exact_match(pred, gold) > 0
                f1_score = compute_f1(pred, gold)
                lenient = compute_lenient_match(pred, gold)

                total_correct += int(is_correct)
                total_f1 += f1_score
                total_lenient += lenient
                total_questions += 1

                details.append({
                    "question": q_detail.get("question", ""),
                    "gold_answer": gold,
                    "prediction": pred,
                    "correct": is_correct,
                    "f1": f1_score,
                    "lenient": lenient,
                })

        # Use result.prompt_tokens and result.generated_tokens (from StrategyResult)
        total_prompt_tokens += result.prompt_tokens
        total_completion_tokens += result.generated_tokens

    em = total_correct / total_questions if total_questions > 0 else 0
    f1 = total_f1 / total_questions if total_questions > 0 else 0
    lenient = total_lenient / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="cross_batch",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=em,
        metrics={"em": em, "f1": f1, "lenient": lenient},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        details=details,
    )


def print_results_table(results: List[ExperimentResult]):
    """Print results in a nice table format."""
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS: Cross-Batch vs exp2a Strategies on SQuAD")
    print("=" * 100)

    headers = ["Strategy", "EM", "F1", "Lenient", "PromptTok", "GenTok", "Latency(s)"]
    col_widths = [20, 8, 8, 8, 12, 10, 12]

    # Header row
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))

    for r in results:
        n = r.n_samples if r.n_samples > 0 else 1
        avg_prompt = r.prompt_tokens / n if r.prompt_tokens else 0
        avg_compl = r.completion_tokens / n if r.completion_tokens else 0
        avg_latency = r.latency / n if r.latency else 0

        row = [
            r.condition,
            f"{r.metrics.get('em', 0):.4f}",
            f"{r.metrics.get('f1', 0):.4f}",
            f"{r.metrics.get('lenient', 0):.4f}",
            f"{avg_prompt:.1f}",
            f"{avg_compl:.1f}",
            f"{avg_latency:.2f}",
        ]
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

    print("=" * 100 + "\n")


def main():
    args = parse_args()

    if args.summarize:
        # Just summarize existing results
        print("Summarizing existing results...")
        # TODO: Implement summarize_from_files for this script
        return

    # Load SQuAD groups
    groups = load_squad_groups(
        n_groups=args.n_groups,
        n_questions=args.n_questions,
        seed=args.seed,
    )

    results = []

    # Initialize LLM client for non-cross-batch strategies
    if any(s in args.strategies for s in ['independent', 'all_in_one', 'seq_shared_rand', 'seq_shared_rand_full']):
        client = LLMClient(
            model=args.model,
            use_local=args.use_local,
            use_vllm=args.use_vllm,
            tensor_parallel_size=args.tensor_parallel_size,
        )

        if 'independent' in args.strategies:
            results.append(run_independent(groups, client))

        if 'all_in_one' in args.strategies:
            results.append(run_all_in_one(groups, client))

        if 'seq_shared_rand' in args.strategies:
            results.append(run_seq_shared_rand(groups, client, seed=args.seed))

        if 'seq_shared_rand_full' in args.strategies:
            results.append(run_seq_shared_rand_full(groups, client, seed=args.seed))

    # Run cross-batch (uses different inference path)
    if 'cross_batch' in args.strategies:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        results.append(run_cross_batch(
            groups=groups,
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            device=device,
        ))

    # Print and save results
    print_results_table(results)

    # Save to file
    config = ExperimentConfig(
        exp_name="eval_cross_batch_strategies",
        dataset="squad",
        model=args.model,
        n_samples=args.n_groups,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    save_results(results, config)

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
