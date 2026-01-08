"""CMB (Chinese Medical Benchmark) dataset loader.

Dataset: FreedomIntelligence/CMB
Subsets:
- CMB-Clin: Clinical case analysis with multiple QA pairs per case
- CMB-Exam: Medical exam questions (single questions)

Also supports grouped variants from fzkuji/CMB-Exam-Grouped:
- subdomain: Questions grouped by shared medical terms
- context: Questions sharing the same background context
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Optional

from ..models import estimate_tokens

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def load_cmb_groups(
    split: str = "test",
    *,
    subset: str = "CMB-Clin",
    min_questions: int = 1,
    max_questions: Optional[int] = None,
    max_contexts: int = 3,
    seed: int = 13,
) -> List[dict]:
    """
    Load CMB clinical cases with multiple QA pairs sharing a description.

    Args:
        split: Dataset split (CMB-Clin only has "test" with 74 rows).
        subset: CMB subset (e.g., "CMB-Clin").
        min_questions: Minimum QA pairs per case to include.
        max_questions: Maximum QA pairs per case (truncate if exceeded).
        max_contexts: Maximum number of cases to return.
        seed: Random seed for shuffling.

    Returns:
        List of context dictionaries with questions.
        Format matches SQuAD: {context, title, questions: [{qid, text, answer_tokens, references}]}
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")

    raw_dataset = list(load_dataset("FreedomIntelligence/CMB", subset, split=split))
    if not raw_dataset:
        raise ValueError(f"Empty CMB split: {subset}/{split}")

    # Filter by minimum questions
    filtered = [
        row for row in raw_dataset
        if len(row.get("QA_pairs", [])) >= min_questions
    ]
    if not filtered:
        raise ValueError(f"No cases satisfy the minimum question requirement ({min_questions}).")

    rng = random.Random(seed)
    rng.shuffle(filtered)
    selected = filtered[:max_contexts] if max_contexts else filtered

    formatted: List[dict] = []
    for row in selected:
        # description contains the full case context
        context = row.get("description", "").strip()
        title = row.get("title", row.get("id", "CMB-Case"))
        qa_pairs = row.get("QA_pairs", [])

        if max_questions:
            qa_pairs = qa_pairs[:max_questions]

        questions = []
        for idx, qa in enumerate(qa_pairs):
            qid = f"Q{idx + 1}"
            question_text = qa.get("question", "").strip()
            answer_text = qa.get("answer", "").strip()
            answer_tokens = max(estimate_tokens(answer_text), 12)
            questions.append(
                {
                    "qid": qid,
                    "text": question_text,
                    "answer_tokens": answer_tokens,
                    "references": [answer_text] if answer_text else [],
                }
            )

        formatted.append(
            {
                "context": context,
                "title": title,
                "questions": questions,
            }
        )

    return formatted


def load_cmb_exam_random_groups(
    split: str = "train",
    *,
    questions_per_group: int = 5,
    max_contexts: int = 500,
    seed: int = 13,
) -> List[dict]:
    """
    Load CMB-Exam questions with random grouping (no shared context).

    This loads individual questions from original CMB-Exam and randomly groups them
    to create a baseline for comparison with context-based or subdomain groupings.

    Uses fzkuji/CMB-Exam-Grouped subdomain config as data source since original
    FreedomIntelligence/CMB has schema inconsistencies between splits.

    Args:
        split: Dataset split (train, val, test).
        questions_per_group: Number of questions per group.
        max_contexts: Maximum number of groups to return.
        seed: Random seed for shuffling.

    Returns:
        List of context groups in multi-context format:
        {title, items: [{qid, question, context, answer_tokens, references}]}
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")

    # Use fzkuji/CMB-Exam-Grouped as source since original CMB has schema issues
    raw_dataset = list(load_dataset("fzkuji/CMB-Exam-Grouped", "subdomain", split=split))
    if not raw_dataset:
        raise ValueError(f"Empty CMB-Exam-Grouped subdomain split: {split}")

    # Extract all individual questions from all groups
    all_questions = []
    for row in raw_dataset:
        questions_list = row.get("questions", [])
        for q in questions_list:
            all_questions.append(q)

    rng = random.Random(seed)
    rng.shuffle(all_questions)

    # Calculate how many samples we need
    total_needed = max_contexts * questions_per_group
    selected = all_questions[:total_needed]

    formatted: List[dict] = []
    for group_idx in range(max_contexts):
        start = group_idx * questions_per_group
        end = start + questions_per_group
        if end > len(selected):
            break

        items = []
        for i, q in enumerate(selected[start:end]):
            question_text = q.get("question", "").strip()
            answer_text = q.get("answer", "").strip()

            # Build context from question metadata
            exam_info = f"{q.get('exam_type', '')} - {q.get('exam_subject', '')}"

            # Format options if present
            options = q.get("option", {})
            if options:
                option_strs = []
                for key in ["A", "B", "C", "D", "E", "F"]:
                    val = options.get(key)
                    if val:
                        option_strs.append(f"{key}. {val}")
                options_text = "\n".join(option_strs)
                context = f"{exam_info}\n\n选项:\n{options_text}"
            else:
                context = exam_info

            items.append({
                "qid": f"Q{i + 1}",
                "question": question_text,
                "context": context,
                "answer_tokens": max(estimate_tokens(answer_text), 4),
                "references": [answer_text] if answer_text else [],
            })

        formatted.append({
            "title": f"random-{group_idx}",
            "items": items,
        })

    return formatted


def load_cmb_exam_subdomain_groups(
    split: str = "train",
    *,
    min_questions: int = 2,
    max_questions: Optional[int] = 5,
    max_contexts: int = 500,
    seed: int = 13,
) -> List[dict]:
    """
    Load CMB-Exam questions grouped by shared medical terms (subdomain).

    Uses fzkuji/CMB-Exam-Grouped "subdomain" config where each row is already a group
    with {medical_term, questions: [{question, answer, option, ...}], num_questions}.

    Args:
        split: Dataset split (train, val, test).
        min_questions: Minimum questions per group to include.
        max_questions: Maximum questions per group (truncate if exceeded).
        max_contexts: Maximum number of groups to return.
        seed: Random seed for shuffling.

    Returns:
        List of context groups in multi-context format:
        {title, items: [{qid, question, context, answer_tokens, references}]}
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")

    raw_dataset = list(load_dataset("fzkuji/CMB-Exam-Grouped", "subdomain", split=split))
    if not raw_dataset:
        raise ValueError(f"Empty CMB-Exam-Grouped subdomain split: {split}")

    # Filter by minimum questions
    valid_groups = [row for row in raw_dataset if row.get("num_questions", 0) >= min_questions]

    rng = random.Random(seed)
    rng.shuffle(valid_groups)
    selected_groups = valid_groups[:max_contexts] if max_contexts else valid_groups

    formatted: List[dict] = []
    for row in selected_groups:
        medical_term = row.get("medical_term", "unknown")
        raw_questions = row.get("questions", [])

        # Shuffle questions and limit per group
        rng.shuffle(raw_questions)
        if max_questions and len(raw_questions) > max_questions:
            raw_questions = raw_questions[:max_questions]

        # Build items for multi-context format
        items = []
        for i, q in enumerate(raw_questions):
            question_text = q.get("question", "").strip()
            answer_text = q.get("answer", "").strip()

            # Build context from question metadata
            exam_info = f"{q.get('exam_type', '')} - {q.get('exam_subject', '')}"

            # Format options if present
            options = q.get("option", {})
            if options:
                option_strs = []
                for key in ["A", "B", "C", "D", "E", "F"]:
                    val = options.get(key)
                    if val:
                        option_strs.append(f"{key}. {val}")
                options_text = "\n".join(option_strs)
                context = f"{exam_info}\n\n选项:\n{options_text}"
            else:
                context = exam_info

            items.append({
                "qid": f"Q{i + 1}",
                "question": question_text,
                "context": context,
                "answer_tokens": max(estimate_tokens(answer_text), 4),
                "references": [answer_text] if answer_text else [],
            })

        formatted.append({
            "title": f"subdomain-{medical_term}",
            "items": items,
        })

    return formatted


def load_cmb_exam_context_groups(
    split: str = "train",
    *,
    min_questions: int = 2,
    max_questions: Optional[int] = 5,
    max_contexts: int = 500,
    seed: int = 13,
) -> List[dict]:
    """
    Load CMB-Exam questions grouped by shared background context.

    Uses fzkuji/CMB-Exam-Grouped "context" config where each row is already a group
    with {background, questions: [{question, answer, option}], num_questions}.

    Args:
        split: Dataset split (train, val, test).
        min_questions: Minimum questions per group to include.
        max_questions: Maximum questions per group (truncate if exceeded).
        max_contexts: Maximum number of groups to return.
        seed: Random seed for shuffling.

    Returns:
        List of context groups:
        {context, title, questions: [{qid, text, answer_tokens, references}]}
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")

    raw_dataset = list(load_dataset("fzkuji/CMB-Exam-Grouped", "context", split=split))
    if not raw_dataset:
        raise ValueError(f"Empty CMB-Exam-Grouped context split: {split}")

    # Filter by minimum questions
    valid_groups = [row for row in raw_dataset if row.get("num_questions", 0) >= min_questions]

    rng = random.Random(seed)
    rng.shuffle(valid_groups)
    selected_groups = valid_groups[:max_contexts] if max_contexts else valid_groups

    formatted: List[dict] = []
    for idx, row in enumerate(selected_groups):
        background = row.get("background", "").strip()
        if not background:
            continue

        raw_questions = row.get("questions", [])

        # Shuffle questions and limit per group
        rng.shuffle(raw_questions)
        if max_questions and len(raw_questions) > max_questions:
            raw_questions = raw_questions[:max_questions]

        questions = []
        for q_idx, q in enumerate(raw_questions):
            question_text = q.get("question", "").strip()
            answer_text = q.get("answer", "").strip()

            # Format options if present
            options = q.get("option", {})
            if options:
                option_strs = []
                for key in ["A", "B", "C", "D", "E", "F"]:
                    val = options.get(key)
                    if val:
                        option_strs.append(f"{key}. {val}")
                if option_strs:
                    question_text = f"{question_text}\n{chr(10).join(option_strs)}"

            questions.append({
                "qid": f"Q{q_idx + 1}",
                "text": question_text,
                "answer_tokens": max(estimate_tokens(answer_text), 4),
                "references": [answer_text] if answer_text else [],
            })

        formatted.append({
            "context": background,
            "title": f"context-{idx}",
            "questions": questions,
        })

    return formatted
