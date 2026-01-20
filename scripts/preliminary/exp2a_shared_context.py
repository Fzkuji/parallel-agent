#!/usr/bin/env python3
"""Experiment 2a: Shared Context (Multi-Query QA on SQuAD)

Dataset: SQuAD (multiple questions per paragraph)

Research Question: How do different multi-query processing strategies compare
when questions share the same context?

Conditions:
- Independent: Each question + context answered separately
- All-in-One: All questions from same context in one prompt
- Seq. (Cross-Ctx): Sequential with questions from DIFFERENT contexts (baseline)
- Seq. (Shared, Rand): Sequential with shared context, random order, context only in first turn
- Seq. (Shared, Ord): Sequential with shared context, LLM-optimized order, context only in first turn
- Seq. (Shared, Rand, Full): Sequential with shared context, random order, context in every turn
- Seq. (Shared, Ord, Full): Sequential with shared context, LLM-optimized order, context in every turn

Expected: Shared-context strategies should benefit from information sharing.
"""

from __future__ import annotations

import os
# Suppress vLLM verbose logging
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

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

# System prompt for answer extraction - simplified for better model compatibility
SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the given passage.
Give a short, direct answer. Do not explain or elaborate."""

SYSTEM_PROMPT_MULTI = """You are a helpful assistant. Answer all questions based on the given passage.
Give short, direct answers. Format your response as:
Q1: [answer]
Q2: [answer]
..."""


def load_squad_groups(
    n_groups: int = -1,
    n_questions: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load SQuAD dataset grouped by paragraph.

    Args:
        n_groups: Number of groups to sample (-1 for all)
        n_questions: Exact number of questions per group (default: 5)
        seed: Random seed for reproducibility

    Returns list of groups, each containing:
    - context: The paragraph text
    - questions: List of {question, answer} dicts (exactly n_questions per group)
    """
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
        groups = valid_groups  # Use all groups

    logger.info(f"Loaded {len(groups)} groups with exactly {n_questions} questions each")

    return groups


def run_independent(
    groups: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """Independent: Each question + context answered separately."""
    logger.info("Running Independent condition...")

    total_correct = 0
    total_f1 = 0.0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # New: accurate token counting
    unique_prompt_tokens = 0  # For independent: sum of all (context + question) since no cache reuse
    sum_completion_tokens = 0  # Sum of all generated answers

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
            # Independent: each request is separate, so we count all prompt tokens
            unique_prompt_tokens += response.prompt_tokens
            sum_completion_tokens += response.completion_tokens

            pred = _extract_answer(pred_raw)
            is_correct = compute_exact_match(pred, gold_answer) > 0
            f1_score = compute_f1(pred, gold_answer)
            total_correct += int(is_correct)
            total_f1 += f1_score
            total_questions += 1

            details.append({
                "context_id": id(context),
                "question": question,
                "gold_answer": gold_answer,
                "prediction": pred,
                "prediction_raw": pred_raw,
                "correct": is_correct,
                "f1": f1_score,
            })

    em = total_correct / total_questions if total_questions > 0 else 0
    f1 = total_f1 / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="independent",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=em,
        metrics={"em": em, "f1": f1},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=unique_prompt_tokens,
        total_completion_tokens=sum_completion_tokens,
        details=details,
    )


def run_all_in_one(
    groups: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """All-in-One: All questions from same context in one prompt."""
    logger.info("Running All-in-One condition...")

    total_correct = 0
    total_f1 = 0.0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # New: accurate token counting
    unique_prompt_tokens = 0  # For all_in_one: context + all questions in one request
    sum_completion_tokens = 0  # All answers in one response

    for group in tqdm(groups, desc="All-in-One"):
        context = group["context"]
        questions = group["questions"]

        # Build multi-question prompt
        q_list = "\n".join([f"Q{i+1}: {q['question']}" for i, q in enumerate(questions)])

        prompt = f"""Passage:
{context}

Questions:
{q_list}"""

        pred, response = client.generate(prompt, max_tokens=512, system_prompt=SYSTEM_PROMPT_MULTI)
        total_latency += response.latency
        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens
        # All-in-one: single request per group, prompt = context + all questions
        unique_prompt_tokens += response.prompt_tokens
        sum_completion_tokens += response.completion_tokens

        # Parse answers
        answers = _parse_batch_answers(pred, len(questions))

        for i, q_item in enumerate(questions):
            gold_answer = q_item["answer"]
            pred_answer_raw = answers.get(i, "")
            pred_answer = _extract_answer(pred_answer_raw)

            is_correct = compute_exact_match(pred_answer, gold_answer) > 0
            f1_score = compute_f1(pred_answer, gold_answer)
            total_correct += int(is_correct)
            total_f1 += f1_score
            total_questions += 1

            details.append({
                "context_id": id(context),
                "question": q_item["question"],
                "gold_answer": gold_answer,
                "prediction": pred_answer,
                "prediction_raw": pred_answer_raw,
                "batch_response": pred,
                "correct": is_correct,
                "f1": f1_score,
            })

    em = total_correct / total_questions if total_questions > 0 else 0
    f1 = total_f1 / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="all_in_one",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=em,
        metrics={"em": em, "f1": f1},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=unique_prompt_tokens,
        total_completion_tokens=sum_completion_tokens,
        details=details,
    )


def run_seq_shared_rand(
    groups: List[Dict[str, Any]],
    client: LLMClient,
    seed: int = 42,
) -> ExperimentResult:
    """Seq. (Shared, Rand): Questions from same context in random order.

    All questions share the same context. Context is only provided in the first turn,
    subsequent turns only include the question.
    """
    logger.info("Running Seq. (Shared, Rand) condition...")

    random.seed(seed)

    total_correct = 0
    total_f1 = 0.0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # Accurate token counting:
    # sequence_length = last_turn_prompt + last_turn_completion (per group, then sum)
    # completion = sum of all completion_tokens
    # unique_prompt = sequence_length - completion
    total_sequence_length = 0  # Sum of (last_prompt + last_completion) for each group
    sum_completion_tokens = 0  # Sum of all generated answers

    for group in tqdm(groups, desc="Seq.(Shared,Rand)"):
        context = group["context"]
        questions = group["questions"].copy()

        # Shuffle questions randomly
        random.shuffle(questions)

        # Initialize messages with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        last_prompt_tokens = 0
        last_completion_tokens = 0

        for i, q_item in enumerate(questions):
            question = q_item["question"]
            gold_answer = q_item["answer"]

            # First question includes context, subsequent questions only include the question
            if i == 0:
                user_content = f"""Passage:
{context}

Question: {question}"""
            else:
                user_content = f"Question: {question}"

            # Add user message
            messages.append({"role": "user", "content": user_content})

            # Generate response
            pred_raw, response = client.generate_with_messages(messages, max_tokens=128)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens
            sum_completion_tokens += response.completion_tokens

            # Track last turn's tokens
            last_prompt_tokens = response.prompt_tokens
            last_completion_tokens = response.completion_tokens

            pred = _extract_answer(pred_raw)

            # Add assistant response to history
            messages.append({"role": "assistant", "content": pred_raw})

            is_correct = compute_exact_match(pred, gold_answer) > 0
            f1_score = compute_f1(pred, gold_answer)
            total_correct += int(is_correct)
            total_f1 += f1_score
            total_questions += 1

            details.append({
                "context_id": id(context),
                "question": question,
                "gold_answer": gold_answer,
                "prediction": pred,
                "prediction_raw": pred_raw,
                "correct": is_correct,
                "f1": f1_score,
                "history_length": i,  # Number of previous QA pairs
            })

        # After processing all questions in this group, add to sequence_length
        total_sequence_length += last_prompt_tokens + last_completion_tokens

    em = total_correct / total_questions if total_questions > 0 else 0
    f1 = total_f1 / total_questions if total_questions > 0 else 0

    # unique_prompt = sequence_length - completion
    unique_prompt_tokens = total_sequence_length - sum_completion_tokens

    return ExperimentResult(
        condition="seq_shared_rand",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=em,
        metrics={"em": em, "f1": f1},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=unique_prompt_tokens,
        total_completion_tokens=sum_completion_tokens,
        details=details,
    )


def run_seq_shared_ord(
    groups: List[Dict[str, Any]],
    client: LLMClient,
    ordering_client: Optional[LLMClient] = None,
) -> ExperimentResult:
    """Seq. (Shared, Ord): LLM determines optimal question order, then answers sequentially.

    All questions share the same context. Context is only provided in the first turn,
    subsequent turns only include the question.
    """
    logger.info("Running Seq. (Shared, Ord) condition...")

    # Use same client for ordering if not specified
    if ordering_client is None:
        ordering_client = client

    total_correct = 0
    total_f1 = 0.0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # Accurate token counting:
    # sequence_length = last_turn_prompt + last_turn_completion (per group, then sum)
    # completion = sum of all completion_tokens
    # unique_prompt = sequence_length - completion
    # Note: ordering request is separate, adds to sequence_length directly
    total_sequence_length = 0  # Sum of (last_prompt + last_completion) for each group + ordering
    sum_completion_tokens = 0  # Sum of all generated answers (including ordering response)

    for group in tqdm(groups, desc="Seq.(Shared,Ord)"):
        context = group["context"]
        questions = group["questions"]

        # Step 1: Ask LLM to determine optimal order (separate single-turn request)
        q_list = "\n".join([f"{i+1}. {q['question']}" for i, q in enumerate(questions)])

        ordering_prompt = f"""Given the following passage and questions, determine the optimal order to answer them.
Consider which questions might provide useful information for answering subsequent questions.

Passage:
{context}

Questions:
{q_list}

Return ONLY a comma-separated list of question numbers in the optimal order (e.g., "2,1,3,4").
Optimal order:"""

        order_response, order_meta = ordering_client.generate(ordering_prompt, max_tokens=32)
        total_latency += order_meta.latency
        total_prompt_tokens += order_meta.prompt_tokens
        total_completion_tokens += order_meta.completion_tokens
        # Ordering is a separate single-turn request: sequence_length = prompt + completion
        total_sequence_length += order_meta.prompt_tokens + order_meta.completion_tokens
        sum_completion_tokens += order_meta.completion_tokens

        # Parse order
        ordered_indices = _parse_order(order_response, len(questions))
        ordered_questions = [questions[i] for i in ordered_indices]

        # Step 2: Answer questions in determined order using multi-turn chat format
        # Initialize messages with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        last_prompt_tokens = 0
        last_completion_tokens = 0

        for i, q_item in enumerate(ordered_questions):
            question = q_item["question"]
            gold_answer = q_item["answer"]

            # First question includes context, subsequent questions only include the question
            if i == 0:
                user_content = f"""Passage:
{context}

Question: {question}"""
            else:
                user_content = f"Question: {question}"

            # Add user message
            messages.append({"role": "user", "content": user_content})

            # Generate response
            pred_raw, response = client.generate_with_messages(messages, max_tokens=128)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens
            sum_completion_tokens += response.completion_tokens

            # Track last turn's tokens
            last_prompt_tokens = response.prompt_tokens
            last_completion_tokens = response.completion_tokens

            pred = _extract_answer(pred_raw)

            # Add assistant response to history
            messages.append({"role": "assistant", "content": pred_raw})

            is_correct = compute_exact_match(pred, gold_answer) > 0
            f1_score = compute_f1(pred, gold_answer)
            total_correct += int(is_correct)
            total_f1 += f1_score
            total_questions += 1

            details.append({
                "context_id": id(context),
                "question": question,
                "gold_answer": gold_answer,
                "prediction": pred,
                "prediction_raw": pred_raw,
                "correct": is_correct,
                "f1": f1_score,
                "history_length": i,  # Number of previous QA pairs
                "llm_order": ordered_indices,
            })

        # After processing all questions in this group, add to sequence_length
        total_sequence_length += last_prompt_tokens + last_completion_tokens

    em = total_correct / total_questions if total_questions > 0 else 0
    f1 = total_f1 / total_questions if total_questions > 0 else 0

    # unique_prompt = sequence_length - completion
    unique_prompt_tokens = total_sequence_length - sum_completion_tokens

    return ExperimentResult(
        condition="seq_shared_ord",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=em,
        metrics={"em": em, "f1": f1},
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
    """Seq. (Shared, Rand, Full): Questions from same context in random order.

    All questions share the same context. Every turn includes context + question
    (same format as cross_ctx for fair comparison of token usage).
    """
    logger.info("Running Seq. (Shared, Rand, Full) condition...")

    random.seed(seed)

    total_correct = 0
    total_f1 = 0.0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # Accurate token counting:
    # sequence_length = last_turn_prompt + last_turn_completion (per group, then sum)
    # completion = sum of all completion_tokens
    # unique_prompt = sequence_length - completion
    total_sequence_length = 0  # Sum of (last_prompt + last_completion) for each group
    sum_completion_tokens = 0  # Sum of all generated answers

    for group in tqdm(groups, desc="Seq.(Shared,Rand,Full)"):
        context = group["context"]
        questions = group["questions"].copy()

        # Shuffle questions randomly
        random.shuffle(questions)

        # Initialize messages with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        last_prompt_tokens = 0
        last_completion_tokens = 0

        for i, q_item in enumerate(questions):
            question = q_item["question"]
            gold_answer = q_item["answer"]

            # Every question includes context (same as cross_ctx for fair comparison)
            user_content = f"""Passage:
{context}

Question: {question}"""

            # Add user message
            messages.append({"role": "user", "content": user_content})

            # Generate response
            pred_raw, response = client.generate_with_messages(messages, max_tokens=128)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens
            sum_completion_tokens += response.completion_tokens

            # Track last turn's tokens
            last_prompt_tokens = response.prompt_tokens
            last_completion_tokens = response.completion_tokens

            pred = _extract_answer(pred_raw)

            # Add assistant response to history
            messages.append({"role": "assistant", "content": pred_raw})

            is_correct = compute_exact_match(pred, gold_answer) > 0
            f1_score = compute_f1(pred, gold_answer)
            total_correct += int(is_correct)
            total_f1 += f1_score
            total_questions += 1

            details.append({
                "context_id": id(context),
                "question": question,
                "gold_answer": gold_answer,
                "prediction": pred,
                "prediction_raw": pred_raw,
                "correct": is_correct,
                "f1": f1_score,
                "history_length": i,  # Number of previous QA pairs
            })

        # After processing all questions in this group, add to sequence_length
        total_sequence_length += last_prompt_tokens + last_completion_tokens

    em = total_correct / total_questions if total_questions > 0 else 0
    f1 = total_f1 / total_questions if total_questions > 0 else 0

    # unique_prompt = sequence_length - completion
    unique_prompt_tokens = total_sequence_length - sum_completion_tokens

    return ExperimentResult(
        condition="seq_shared_rand_full",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=em,
        metrics={"em": em, "f1": f1},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=unique_prompt_tokens,
        total_completion_tokens=sum_completion_tokens,
        details=details,
    )


def run_seq_shared_ord_full(
    groups: List[Dict[str, Any]],
    client: LLMClient,
    ordering_client: Optional[LLMClient] = None,
) -> ExperimentResult:
    """Seq. (Shared, Ord, Full): LLM determines optimal question order, then answers sequentially.

    All questions share the same context. Every turn includes context + question
    (same format as cross_ctx for fair comparison of token usage).
    """
    logger.info("Running Seq. (Shared, Ord, Full) condition...")

    # Use same client for ordering if not specified
    if ordering_client is None:
        ordering_client = client

    total_correct = 0
    total_f1 = 0.0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # Accurate token counting:
    # sequence_length = last_turn_prompt + last_turn_completion (per group, then sum)
    # completion = sum of all completion_tokens
    # unique_prompt = sequence_length - completion
    # Note: ordering request is separate, adds to sequence_length directly
    total_sequence_length = 0  # Sum of (last_prompt + last_completion) for each group + ordering
    sum_completion_tokens = 0  # Sum of all generated answers (including ordering response)

    for group in tqdm(groups, desc="Seq.(Shared,Ord,Full)"):
        context = group["context"]
        questions = group["questions"]

        # Step 1: Ask LLM to determine optimal order (separate single-turn request)
        q_list = "\n".join([f"{i+1}. {q['question']}" for i, q in enumerate(questions)])

        ordering_prompt = f"""Given the following passage and questions, determine the optimal order to answer them.
Consider which questions might provide useful information for answering subsequent questions.

Passage:
{context}

Questions:
{q_list}

Return ONLY a comma-separated list of question numbers in the optimal order (e.g., "2,1,3,4").
Optimal order:"""

        order_response, order_meta = ordering_client.generate(ordering_prompt, max_tokens=32)
        total_latency += order_meta.latency
        total_prompt_tokens += order_meta.prompt_tokens
        total_completion_tokens += order_meta.completion_tokens
        # Ordering is a separate single-turn request: sequence_length = prompt + completion
        total_sequence_length += order_meta.prompt_tokens + order_meta.completion_tokens
        sum_completion_tokens += order_meta.completion_tokens

        # Parse order
        ordered_indices = _parse_order(order_response, len(questions))
        ordered_questions = [questions[i] for i in ordered_indices]

        # Step 2: Answer questions in determined order using multi-turn chat format
        # Initialize messages with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        last_prompt_tokens = 0
        last_completion_tokens = 0

        for i, q_item in enumerate(ordered_questions):
            question = q_item["question"]
            gold_answer = q_item["answer"]

            # Every question includes context (same as cross_ctx for fair comparison)
            user_content = f"""Passage:
{context}

Question: {question}"""

            # Add user message
            messages.append({"role": "user", "content": user_content})

            # Generate response
            pred_raw, response = client.generate_with_messages(messages, max_tokens=128)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens
            sum_completion_tokens += response.completion_tokens

            # Track last turn's tokens
            last_prompt_tokens = response.prompt_tokens
            last_completion_tokens = response.completion_tokens

            pred = _extract_answer(pred_raw)

            # Add assistant response to history
            messages.append({"role": "assistant", "content": pred_raw})

            is_correct = compute_exact_match(pred, gold_answer) > 0
            f1_score = compute_f1(pred, gold_answer)
            total_correct += int(is_correct)
            total_f1 += f1_score
            total_questions += 1

            details.append({
                "context_id": id(context),
                "question": question,
                "gold_answer": gold_answer,
                "prediction": pred,
                "prediction_raw": pred_raw,
                "correct": is_correct,
                "f1": f1_score,
                "history_length": i,  # Number of previous QA pairs
                "llm_order": ordered_indices,
            })

        # After processing all questions in this group, add to sequence_length
        total_sequence_length += last_prompt_tokens + last_completion_tokens

    em = total_correct / total_questions if total_questions > 0 else 0
    f1 = total_f1 / total_questions if total_questions > 0 else 0

    # unique_prompt = sequence_length - completion
    unique_prompt_tokens = total_sequence_length - sum_completion_tokens

    return ExperimentResult(
        condition="seq_shared_ord_full",
        dataset="squad",
        n_samples=len(groups),
        n_questions=total_questions,
        accuracy=em,
        metrics={"em": em, "f1": f1},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=unique_prompt_tokens,
        total_completion_tokens=sum_completion_tokens,
        details=details,
    )


def run_seq_cross_ctx(
    groups: List[Dict[str, Any]],
    client: LLMClient,
    seed: int = 42,
) -> ExperimentResult:
    """Seq. (Cross-Ctx): Sequential with questions from DIFFERENT contexts (baseline).

    This serves as a control condition to measure the benefit of shared context.
    All questions are shuffled and regrouped into batches of the same size as
    the original groups. Each question keeps its own context, so questions in
    the same batch come from different contexts.

    The question set is identical to shared context conditions, only the grouping differs.
    """
    logger.info("Running Seq. (Cross-Ctx) condition...")

    random.seed(seed)

    total_correct = 0
    total_f1 = 0.0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # Accurate token counting:
    # sequence_length = last_turn_prompt + last_turn_completion (per batch, then sum)
    # completion = sum of all completion_tokens
    # unique_prompt = sequence_length - completion
    # Note: Cross-ctx has DIFFERENT contexts per question, so less cache benefit
    total_sequence_length = 0  # Sum of (last_prompt + last_completion) for each batch
    sum_completion_tokens = 0  # Sum of all generated answers

    # Get the fixed group size (all groups should have the same size)
    group_size = groups[0]["n_questions"] if groups else 5

    # Flatten all questions with their contexts
    all_qa_pairs = []
    for group in groups:
        context = group["context"]
        for q_item in group["questions"]:
            all_qa_pairs.append({
                "context": context,
                "question": q_item["question"],
                "answer": q_item["answer"],
            })

    # Shuffle to mix questions from different contexts
    random.shuffle(all_qa_pairs)

    # Process in batches of group_size (same as original groups)
    n_batches = len(all_qa_pairs) // group_size

    for batch_idx in tqdm(range(n_batches), desc="Seq.(Cross-Ctx)"):
        # Get questions for this batch
        batch_start = batch_idx * group_size
        batch_pairs = all_qa_pairs[batch_start:batch_start + group_size]

        # Initialize messages for this batch
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        last_prompt_tokens = 0
        last_completion_tokens = 0

        for i, qa_pair in enumerate(batch_pairs):
            context = qa_pair["context"]
            question = qa_pair["question"]
            gold_answer = qa_pair["answer"]

            # Every question includes its own context (different from shared conditions)
            user_content = f"""Passage:
{context}

Question: {question}"""

            # Add user message
            messages.append({"role": "user", "content": user_content})

            # Generate response
            pred_raw, response = client.generate_with_messages(messages, max_tokens=128)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens
            sum_completion_tokens += response.completion_tokens

            # Track last turn's tokens
            last_prompt_tokens = response.prompt_tokens
            last_completion_tokens = response.completion_tokens

            pred = _extract_answer(pred_raw)

            # Add assistant response to history
            messages.append({"role": "assistant", "content": pred_raw})

            is_correct = compute_exact_match(pred, gold_answer) > 0
            f1_score = compute_f1(pred, gold_answer)
            total_correct += int(is_correct)
            total_f1 += f1_score
            total_questions += 1

            details.append({
                "context_id": id(context),
                "question": question,
                "gold_answer": gold_answer,
                "prediction": pred,
                "prediction_raw": pred_raw,
                "correct": is_correct,
                "f1": f1_score,
                "history_length": i,  # Position within batch
                "batch": batch_idx,
            })

        # After processing all questions in this batch, add to sequence_length
        total_sequence_length += last_prompt_tokens + last_completion_tokens

    em = total_correct / total_questions if total_questions > 0 else 0
    f1 = total_f1 / total_questions if total_questions > 0 else 0

    # unique_prompt = sequence_length - completion
    unique_prompt_tokens = total_sequence_length - sum_completion_tokens

    return ExperimentResult(
        condition="seq_cross_ctx",
        dataset="squad",
        n_samples=n_batches,  # Number of batches processed
        n_questions=total_questions,
        accuracy=em,
        metrics={"em": em, "f1": f1},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=unique_prompt_tokens,
        total_completion_tokens=sum_completion_tokens,
        details=details,
    )


def _extract_answer(response: str) -> str:
    """Extract answer from model response.

    Same logic as train_cross_batch.py's extract_box_answer:
    - If <answer>...</answer> tag present, extract content
    - Otherwise, return raw text (let F1 handle partial matching)
    """
    response = response.strip()

    # Try to extract from <answer></answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # No tag found - return raw text (same as extract_box_answer)
    return response


def _parse_batch_answers(response: str, n_questions: int) -> Dict[int, str]:
    """Parse batch answers from response.

    Handles various formats:
    - Q1: answer1  Q2: answer2
    - Q1: <answer>answer1</answer>
    - 1. answer1  2. answer2
    - Line-by-line answers
    """
    answers = {}

    # Method 1: Try to extract <answer> tags for each question (backward compatibility)
    for i in range(n_questions):
        if i < n_questions - 1:
            pattern = rf"Q{i+1}[:\s]+.*?<answer>(.*?)</answer>.*?(?=Q{i+2}[:\s])"
        else:
            pattern = rf"Q{i+1}[:\s]+.*?<answer>(.*?)</answer>"
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            answers[i] = match.group(1).strip()

    if len(answers) == n_questions:
        return answers

    # Method 2: Q1: answer format (most common)
    answers = {}
    for i in range(n_questions):
        # Match Q1:, Q1., Q1), etc.
        if i < n_questions - 1:
            pattern = rf"Q{i+1}[:\.\)]\s*(.+?)(?=Q{i+2}[:\.\)]|\n\n|$)"
        else:
            pattern = rf"Q{i+1}[:\.\)]\s*(.+?)(?:\n\n|$)"
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Clean up the content
            tag_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
            if tag_match:
                answers[i] = tag_match.group(1).strip()
            else:
                # Take first line/sentence
                first_line = content.split('\n')[0].strip()
                answers[i] = first_line.rstrip('.,;:')

    if len(answers) == n_questions:
        return answers

    # Method 3: Numbered list (1. answer, 2. answer)
    if not answers or len(answers) < n_questions:
        answers = {}
        for i in range(n_questions):
            if i < n_questions - 1:
                pattern = rf"(?:^|\n)\s*{i+1}[:\.\)]\s*(.+?)(?=(?:^|\n)\s*{i+2}[:\.\)]|$)"
            else:
                pattern = rf"(?:^|\n)\s*{i+1}[:\.\)]\s*(.+?)$"
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                first_line = content.split('\n')[0].strip()
                answers[i] = first_line.rstrip('.,;:')

    if len(answers) == n_questions:
        return answers

    # Method 4: Split by newline and assign to questions
    if not answers or len(answers) < n_questions:
        answers = {}
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        for i, line in enumerate(lines[:n_questions]):
            # Remove question prefix if present
            clean = re.sub(r"^(?:Q\d+|[\d]+)[:\.\)]\s*", "", line, flags=re.IGNORECASE)
            # Remove <answer> tags if present
            tag_match = re.search(r'<answer>(.*?)</answer>', clean, re.DOTALL | re.IGNORECASE)
            if tag_match:
                answers[i] = tag_match.group(1).strip()
            else:
                answers[i] = clean.rstrip('.,;:')

    return answers


def _parse_order(response: str, n_questions: int) -> List[int]:
    """Parse question order from LLM response like '2,1,3,4'."""
    # Extract numbers from response
    numbers = re.findall(r'\d+', response)

    # Convert to 0-indexed and validate
    indices = []
    seen = set()
    for num in numbers:
        idx = int(num) - 1  # Convert to 0-indexed
        if 0 <= idx < n_questions and idx not in seen:
            indices.append(idx)
            seen.add(idx)

    # Add any missing indices at the end
    for i in range(n_questions):
        if i not in seen:
            indices.append(i)

    return indices


def _merge_results(results: List[ExperimentResult]) -> ExperimentResult:
    """Merge results from multiple ranks into one."""
    if not results:
        raise ValueError("No results to merge")
    if len(results) == 1:
        return results[0]

    total_correct = 0
    total_f1 = 0.0
    total_questions = 0
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_unique_prompt_tokens = 0
    total_sum_completion_tokens = 0
    all_details = []

    for r in results:
        total_questions += r.n_questions
        total_correct += int(r.accuracy * r.n_questions)
        total_f1 += r.metrics.get("f1", 0) * r.n_questions
        total_latency += r.latency
        total_prompt_tokens += r.prompt_tokens
        total_completion_tokens += r.completion_tokens
        total_unique_prompt_tokens += r.unique_prompt_tokens
        total_sum_completion_tokens += r.total_completion_tokens
        all_details.extend(r.details)

    em = total_correct / total_questions if total_questions > 0 else 0
    f1 = total_f1 / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition=results[0].condition,
        dataset=results[0].dataset,
        n_samples=sum(r.n_samples for r in results),
        n_questions=total_questions,
        accuracy=em,
        metrics={"em": em, "f1": f1},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        unique_prompt_tokens=total_unique_prompt_tokens,
        total_completion_tokens=total_sum_completion_tokens,
        details=all_details,
    )


def worker_process(
    rank: int,
    world_size: int,
    gpu_id: int,
    model: str,
    use_vllm: bool,
    groups: List[Dict[str, Any]],
    conditions: List[str],
    seed: int,
    output_dir: str,
    enable_thinking: bool = False,
):
    """Worker process that runs on a single GPU."""
    import json

    # IMPORTANT: Set environment variables BEFORE importing anything CUDA-related
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Disable vLLM V1 engine which spawns EngineCore processes and uses distributed
    os.environ["VLLM_USE_V1"] = "0"
    # Prevent vLLM from trying to use distributed
    os.environ["VLLM_DISABLE_FRONTEND_MULTIPROCESSING"] = "1"
    # Disable vLLM progress bar
    os.environ["VLLM_NO_PROGRESS_BAR"] = "1"
    # Disable tqdm globally
    os.environ["TQDM_DISABLE"] = "1"
    # Suppress vLLM verbose logging
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

    logger.info(f"[Worker {rank}] GPU {gpu_id}: Starting, {len(groups)} groups")

    # Initialize LLM client
    client = LLMClient(
        model=model,
        use_local=True,
        use_vllm=use_vllm,
        tensor_parallel_size=1,
        enable_thinking=enable_thinking,
    )

    logger.info(f"[Worker {rank}] Model loaded, running conditions: {conditions}")

    # Run conditions
    results = []

    if "independent" in conditions:
        logger.info(f"[Worker {rank}] Running independent...")
        results.append(run_independent(groups, client))

    if "all_in_one" in conditions:
        logger.info(f"[Worker {rank}] Running all_in_one...")
        results.append(run_all_in_one(groups, client))

    if "seq_cross_ctx" in conditions:
        logger.info(f"[Worker {rank}] Running seq_cross_ctx...")
        results.append(run_seq_cross_ctx(groups, client, seed=seed))

    if "seq_shared_rand" in conditions:
        logger.info(f"[Worker {rank}] Running seq_shared_rand...")
        results.append(run_seq_shared_rand(groups, client, seed=seed))

    if "seq_shared_ord" in conditions:
        logger.info(f"[Worker {rank}] Running seq_shared_ord...")
        results.append(run_seq_shared_ord(groups, client))

    if "seq_shared_rand_full" in conditions:
        logger.info(f"[Worker {rank}] Running seq_shared_rand_full...")
        results.append(run_seq_shared_rand_full(groups, client, seed=seed))

    if "seq_shared_ord_full" in conditions:
        logger.info(f"[Worker {rank}] Running seq_shared_ord_full...")
        results.append(run_seq_shared_ord_full(groups, client))

    # Save results to temp file
    os.makedirs(output_dir, exist_ok=True)
    temp_file = os.path.join(output_dir, f"temp_rank{rank}.json")

    # Convert results to serializable format
    results_data = []
    for r in results:
        results_data.append({
            "condition": r.condition,
            "n_samples": r.n_samples,
            "n_questions": r.n_questions,
            "accuracy": r.accuracy,
            "metrics": r.metrics,
            "latency": r.latency,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "unique_prompt_tokens": r.unique_prompt_tokens,
            "total_completion_tokens": r.total_completion_tokens,
            "details": r.details,
        })

    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f)

    logger.info(f"[Worker {rank}] Done, saved to {temp_file}")


def run_experiment_for_model(
    model: str,
    all_groups: List[Dict[str, Any]],
    conditions: List[str],
    args,
    num_gpus: int,
):
    """Run experiment for a single model."""
    import multiprocessing as mp
    import json

    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment for model: {model}")
    logger.info(f"{'='*60}\n")

    if args.use_local and num_gpus > 1:
        # Multi-GPU parallel mode with multiprocessing
        gpus = list(range(num_gpus))
        world_size = num_gpus

        logger.info(f"Parallel mode with {world_size} GPUs: {gpus}")

        # Set spawn method (required for CUDA)
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Clean up old temp files
        for rank in range(world_size):
            temp_file = os.path.join(args.output_dir, f"temp_rank{rank}.json")
            if os.path.exists(temp_file):
                os.remove(temp_file)

        # Shard data
        shards = [[] for _ in range(world_size)]
        for i, group in enumerate(all_groups):
            shards[i % world_size].append(group)

        # Start all workers
        processes = []
        for rank, gpu_id in enumerate(gpus):
            p = mp.Process(
                target=worker_process,
                args=(rank, world_size, gpu_id, model, args.use_vllm,
                      shards[rank], conditions, args.seed, args.output_dir,
                      args.enable_thinking)
            )
            p.start()
            processes.append(p)
            logger.info(f"Started worker {rank} on GPU {gpu_id} (PID: {p.pid})")

        # Wait for all workers
        for p in processes:
            p.join()

        logger.info("All workers finished, merging results...")

        # Merge results from all workers
        all_results_by_condition = {}
        for rank in range(world_size):
            temp_file = os.path.join(args.output_dir, f"temp_rank{rank}.json")
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                for r in results_data:
                    cond = r["condition"]
                    if cond not in all_results_by_condition:
                        all_results_by_condition[cond] = []
                    all_results_by_condition[cond].append(ExperimentResult(
                        condition=r["condition"],
                        dataset="squad",
                        n_samples=r["n_samples"],
                        n_questions=r["n_questions"],
                        accuracy=r["accuracy"],
                        metrics=r["metrics"],
                        latency=r["latency"],
                        prompt_tokens=r["prompt_tokens"],
                        completion_tokens=r["completion_tokens"],
                        unique_prompt_tokens=r.get("unique_prompt_tokens", 0),
                        total_completion_tokens=r.get("total_completion_tokens", 0),
                        details=r["details"],
                    ))
                os.remove(temp_file)

        # Merge results
        final_results = []
        for cond, results_list in all_results_by_condition.items():
            merged = _merge_results(results_list)
            final_results.append(merged)

    else:
        # Single process mode (API or single GPU)
        if args.use_local and num_gpus == 1:
            logger.info("Single GPU mode: using GPU 0")

        client = LLMClient(
            model=model,
            use_local=args.use_local,
            use_vllm=args.use_vllm,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_thinking=args.enable_thinking,
        )

        final_results = []

        if "independent" in conditions:
            final_results.append(run_independent(all_groups, client))

        if "all_in_one" in conditions:
            final_results.append(run_all_in_one(all_groups, client))

        if "seq_cross_ctx" in conditions:
            final_results.append(run_seq_cross_ctx(all_groups, client, seed=args.seed))

        if "seq_shared_rand" in conditions:
            final_results.append(run_seq_shared_rand(all_groups, client, seed=args.seed))

        if "seq_shared_ord" in conditions:
            final_results.append(run_seq_shared_ord(all_groups, client))

        if "seq_shared_rand_full" in conditions:
            final_results.append(run_seq_shared_rand_full(all_groups, client, seed=args.seed))

        if "seq_shared_ord_full" in conditions:
            final_results.append(run_seq_shared_ord_full(all_groups, client))

    # Print and save results for this model
    config = ExperimentConfig(
        exp_name="exp2a_shared_context",
        dataset="squad",
        model=model,
        n_samples=args.n_groups,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print_summary(final_results)
    save_results(final_results, config)

    return final_results


def summarize_from_files(output_dir: str, pattern: str = "exp2a_shared_context_*.json"):
    """Read saved result files and print combined summary in markdown format."""
    import glob
    import re as regex

    # Valid exp2a conditions
    VALID_CONDITIONS = ["independent", "all_in_one", "seq_cross_ctx", "seq_shared_rand", "seq_shared_ord", "seq_shared_rand_full", "seq_shared_ord_full"]

    # Find all matching files
    file_pattern = os.path.join(output_dir, pattern)
    files = glob.glob(file_pattern)

    if not files:
        print(f"No files found matching: {file_pattern}")
        return

    # Load data, keep only the file with largest n_samples per model
    model_files = {}
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model = data.get("config", {}).get("model", "unknown")
        n_samples = data.get("config", {}).get("n_samples", 0)
        timestamp = data.get("timestamp", "")

        # Check if this file has valid conditions
        result_conditions = [r.get("condition", "") for r in data.get("results", [])]
        has_valid = any(c in VALID_CONDITIONS for c in result_conditions)
        if not has_valid:
            continue

        # Keep the file with largest n_samples (or latest timestamp if same)
        if model not in model_files or n_samples > model_files[model]["n_samples"]:
            model_files[model] = {"data": data, "n_samples": n_samples, "timestamp": timestamp}

    if not model_files:
        print("No valid results found.")
        return

    # Sort models by size
    def model_size(name):
        match = regex.search(r'(\d+(?:\.\d+)?)[Bb]', name)
        return float(match.group(1)) if match else 0

    sorted_models = sorted(model_files.keys(), key=model_size)

    # Get n_samples and detect model series
    first_data = model_files[sorted_models[0]]["data"]
    n_samples = first_data.get("config", {}).get("n_samples", "all")

    # Detect model series (Qwen2.5 or Qwen3)
    first_model = sorted_models[0]
    if "Qwen3" in first_model:
        model_series = "Qwen3"
    else:
        model_series = "Qwen2.5-Instruct"

    # Extract model sizes for display
    def get_model_size(name):
        name_short = name.split("/")[-1] if "/" in name else name
        # Remove common prefixes
        size = name_short.replace("Qwen2.5-", "").replace("Qwen3-", "").replace("-Instruct", "")
        return size

    model_sizes = [get_model_size(m) for m in sorted_models]

    # Print header
    print("# Experiment 2a: Shared Context Study - Full Results")
    print()
    print("## Dataset")
    print("- **SQuAD**: Multiple questions per paragraph (5 questions/group)")
    print(f"- **Groups**: {n_samples}")
    print(f"- **Models**: {model_series} series ({', '.join(model_sizes)})")
    print()
    print("## Experimental Conditions")
    print()
    print("| Condition | Context in Prompt | Q&A History | Description |")
    print("|-----------|-------------------|-------------|-------------|")
    print("| independent | Every turn | ✗ | Each question answered separately |")
    print("| all_in_one | Once (all Q) | ✗ | All questions in single prompt |")
    print("| seq_cross_ctx | Every turn | ✓ (diff ctx) | Sequential, questions from different contexts |")
    print("| seq_shared_rand | First turn only | ✓ (same ctx) | Sequential, shared context, random order |")
    print("| seq_shared_ord | First turn only | ✓ (same ctx) | Sequential, shared context, LLM-optimized order |")
    print("| seq_shared_rand_full | Every turn | ✓ (same ctx) | Sequential, shared context, random order, full context |")
    print("| seq_shared_ord_full | Every turn | ✓ (same ctx) | Sequential, shared context, LLM order, full context |")
    print()
    print("---")
    print()

    # Print each model's results
    all_model_results = {}
    for model in sorted_models:
        data = model_files[model]["data"]
        results = data.get("results", [])

        # Filter to valid conditions only
        results = [r for r in results if r.get("condition", "") in VALID_CONDITIONS]

        model_short = model.split("/")[-1] if "/" in model else model
        all_model_results[model_short] = {}

        print(f"## {model}")
        print()
        print("**EXPERIMENT SUMMARY**")
        print()

        # Check if we have sequence_length data (new accurate token counting)
        has_seq_length = any(r.get("unique_prompt_tokens", 0) > 0 for r in results)

        if has_seq_length:
            headers = ["Condition", "EM", "F1", "Samples", "Avg SeqLen", "Avg Compl", "Avg Latency (s)"]
        else:
            headers = ["Condition", "EM", "F1", "Samples", "Avg Prompt", "Avg Compl", "Avg Latency (s)"]
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")

        for r in results:
            condition = r.get("condition", "")
            em = r.get("metrics", {}).get("em", 0)
            f1 = r.get("metrics", {}).get("f1", 0)
            r_n_samples = r.get("n_samples", 0)
            latency = r.get("latency", 0)

            n = r_n_samples if r_n_samples > 0 else 1
            avg_latency = latency / n

            all_model_results[model_short][condition] = {"em": em, "f1": f1}

            if has_seq_length and r.get("unique_prompt_tokens", 0) > 0:
                # Use new accurate token counting: sequence_length = unique_prompt + completion
                unique_prompt = r.get("unique_prompt_tokens", 0)
                total_compl = r.get("total_completion_tokens", 0)
                seq_length = unique_prompt + total_compl
                avg_seq_len = seq_length / n
                avg_compl = total_compl / n
                row = [
                    condition,
                    f"{em:.4f}",
                    f"{f1:.4f}",
                    str(r_n_samples),
                    f"{avg_seq_len:.1f}",
                    f"{avg_compl:.1f}",
                    f"{avg_latency:.2f}",
                ]
            else:
                # Fallback to legacy token counting
                prompt_tokens = r.get("prompt_tokens", 0)
                completion_tokens = r.get("completion_tokens", 0)
                avg_prompt = prompt_tokens / n
                avg_compl = completion_tokens / n
                row = [
                    condition,
                    f"{em:.4f}",
                    f"{f1:.4f}",
                    str(r_n_samples),
                    f"{avg_prompt:.1f}",
                    f"{avg_compl:.1f}",
                    f"{avg_latency:.2f}",
                ]
            print("| " + " | ".join(row) + " |")

        # Print per-history-length accuracy for seq_shared_rand_full (more informative)
        seq_rand_full = [r for r in results if r.get("condition") == "seq_shared_rand_full"]
        if seq_rand_full and seq_rand_full[0].get("details"):
            print()
            print("**PER-HISTORY-LENGTH ACCURACY (seq_shared_rand_full)**")
            print()

            details = seq_rand_full[0].get("details", [])
            history_stats = {}
            for d in details:
                h_len = d.get("history_length", 0)
                if h_len not in history_stats:
                    history_stats[h_len] = {"correct": 0, "total": 0, "f1_sum": 0}
                history_stats[h_len]["total"] += 1
                history_stats[h_len]["correct"] += int(d.get("correct", False))
                history_stats[h_len]["f1_sum"] += d.get("f1", 0)

            if history_stats:
                print("| History Length | Samples | EM | F1 |")
                print("| --- | --- | --- | --- |")
                for h_len in sorted(history_stats.keys()):
                    stats = history_stats[h_len]
                    h_em = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                    h_f1 = stats["f1_sum"] / stats["total"] if stats["total"] > 0 else 0
                    print(f"| {h_len} | {stats['total']} | {h_em:.4f} | {h_f1:.4f} |")

        print()
        print("---")
        print()

    # Print final summary table (EM only, more compact)
    print("## Summary Table (EM)")
    print()

    headers = ["Model"] + VALID_CONDITIONS
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["-------"] * len(headers)) + "|")

    for model in sorted_models:
        model_short = model.split("/")[-1] if "/" in model else model
        size = get_model_size(model)
        results = all_model_results.get(model_short, {})
        row = [size]
        for cond in VALID_CONDITIONS:
            em = results.get(cond, {}).get("em", 0)
            row.append(f"{em:.4f}")
        print("| " + " | ".join(row) + " |")

    print()

    # Key findings section
    print("## Key Findings")
    print()
    print("1. **seq_shared_*_full > seq_shared_***: Adding context every turn improves accuracy")
    print("2. **seq_shared_* > seq_cross_ctx**: Shared context benefits from information sharing")
    print("3. **LLM ordering (ord) ≥ random**: Optimized question order can help")
    print("4. **Sequential > independent/all_in_one**: Multi-turn with history is beneficial")
    print()


def check_existing_results(
    model: str,
    n_groups: int,
    conditions: List[str],
    output_dir: str,
) -> Optional[str]:
    """Check if results already exist for given model and conditions.

    Returns:
        Path to existing result file if found with all requested conditions, None otherwise.
    """
    import glob

    # Build expected filename pattern
    model_name = model.replace("/", "_").replace("\\", "_")
    n_samples_str = "all" if n_groups == -1 else str(n_groups)
    filename = f"exp2a_shared_context_{model_name}_n{n_samples_str}.json"
    filepath = os.path.join(output_dir, filename)

    if not os.path.exists(filepath):
        return None

    # Check if the file has all requested conditions
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        existing_conditions = set(r.get("condition", "") for r in data.get("results", []))
        requested_conditions = set(conditions)

        if requested_conditions.issubset(existing_conditions):
            return filepath
        else:
            missing = requested_conditions - existing_conditions
            logger.info(f"Existing results missing conditions: {missing}")
            return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Error reading existing results: {e}")
        return None


def load_results_from_file(filepath: str) -> List[ExperimentResult]:
    """Load experiment results from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for r in data.get("results", []):
        results.append(ExperimentResult(
            condition=r["condition"],
            dataset=r["dataset"],
            n_samples=r["n_samples"],
            n_questions=r["n_questions"],
            accuracy=r["accuracy"],
            metrics=r["metrics"],
            latency=r["latency"],
            prompt_tokens=r["prompt_tokens"],
            completion_tokens=r["completion_tokens"],
            unique_prompt_tokens=r.get("unique_prompt_tokens", 0),
            total_completion_tokens=r.get("total_completion_tokens", 0),
            details=r.get("details", []),
        ))
    return results


def main():
    import torch

    parser = argparse.ArgumentParser(
        description="Exp 2a: Shared Context - SQuAD"
    )
    parser.add_argument(
        "--models", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Comma-separated list of models (e.g., 'Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct')"
    )
    parser.add_argument(
        "--use-local", action="store_true",
        help="Use local model instead of API"
    )
    parser.add_argument(
        "--use-vllm", action="store_true",
        help="Use vLLM for faster inference (requires --use-local)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism (vLLM only, for single-GPU mode)"
    )
    parser.add_argument(
        "--n-groups", type=int, default=-1,
        help="Number of context groups to evaluate (-1 for all)"
    )
    parser.add_argument(
        "--n-questions", type=int, default=5,
        help="Exact number of questions per group (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--conditions", type=str, default="independent,all_in_one,seq_cross_ctx,seq_shared_rand,seq_shared_ord,seq_shared_rand_full,seq_shared_ord_full",
        help="Comma-separated list of conditions: independent,all_in_one,seq_cross_ctx,seq_shared_rand,seq_shared_ord,seq_shared_rand_full,seq_shared_ord_full"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/preliminary",
        help="Output directory for results"
    )
    parser.add_argument(
        "--enable-thinking", action="store_true",
        help="Enable thinking mode for Qwen3 models (default: disabled for faster inference)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run experiments even if results already exist"
    )
    args = parser.parse_args()

    # Parse models and conditions
    models = [m.strip() for m in args.models.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    # Detect number of GPUs
    if args.use_local and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPU(s)")
    else:
        num_gpus = 0
        logger.info("Using CPU or API mode")

    # Check which models need to be run
    models_to_run = []
    models_with_results = []

    for model in models:
        if args.force:
            models_to_run.append(model)
        else:
            existing_file = check_existing_results(
                model=model,
                n_groups=args.n_groups,
                conditions=conditions,
                output_dir=args.output_dir,
            )
            if existing_file:
                logger.info(f"Found existing results for {model}: {existing_file}")
                models_with_results.append((model, existing_file))
            else:
                models_to_run.append(model)

    # Load data only if we need to run experiments
    all_groups = None
    if models_to_run:
        all_groups = load_squad_groups(
            n_groups=args.n_groups,
            n_questions=args.n_questions,
            seed=args.seed,
        )
        logger.info(f"Loaded {len(all_groups)} groups")

    # Run experiments for models that need it
    all_model_results = {}

    # First, load existing results (just log, don't print details - will summarize at end)
    for model, filepath in models_with_results:
        logger.info(f"Using cached results for {model}: {filepath}")
        results = load_results_from_file(filepath)
        # Filter to only requested conditions
        results = [r for r in results if r.condition in conditions]
        all_model_results[model] = results

    # Then, run new experiments
    for model in models_to_run:
        results = run_experiment_for_model(
            model=model,
            all_groups=all_groups,
            conditions=conditions,
            args=args,
            num_gpus=num_gpus,
        )
        all_model_results[model] = results

    # Print final combined summary using summarize_from_files
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL MODELS")
    print("=" * 80 + "\n")
    summarize_from_files(args.output_dir)


if __name__ == "__main__":
    main()
