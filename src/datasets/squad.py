"""SQuAD dataset loader."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional

from ..models import estimate_tokens

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def load_squad_groups(
    split: str,
    *,
    min_questions: int = 3,
    max_questions: Optional[int] = None,
    max_contexts: int = 3,
    seed: int = 13,
    fixed_question_count: Optional[int] = None,
) -> List[dict]:
    """
    Load SQuAD questions grouped by shared context.

    Args:
        split: Dataset split (e.g., "train", "validation").
        min_questions: Minimum questions per context to include.
        max_questions: Maximum questions per context (truncate if exceeded).
        max_contexts: Maximum number of contexts to return.
        seed: Random seed for shuffling.
        fixed_question_count: If set, take exactly this many questions from each context
                            (in order, not random). Useful for controlled experiments.

    Returns:
        List of context dictionaries with questions.
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")

    raw_dataset = load_dataset("squad", split=split)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in raw_dataset:
        grouped[row["context"]].append(row)

    contexts = [
        (ctx, rows) for ctx, rows in grouped.items() if len(rows) >= min_questions
    ]
    if not contexts:
        raise ValueError("No contexts satisfy the minimum question requirement.")

    rng = random.Random(seed)
    rng.shuffle(contexts)
    selected = contexts[:max_contexts] if max_contexts else contexts

    formatted: List[dict] = []
    for context_text, rows in selected:
        # Handle question sampling
        if fixed_question_count is not None:
            # Take exactly N questions in order (for controlled experiments)
            if len(rows) >= fixed_question_count:
                rows = rows[:fixed_question_count]
            else:
                # Skip contexts that don't have enough questions
                continue
        elif max_questions:
            # Determine actual range based on available questions
            actual_max = min(max_questions, len(rows))
            actual_min = min(min_questions, actual_max)

            # If min == max, take exactly that many questions in order (no shuffle)
            if actual_min == actual_max:
                num_questions = actual_max
                rows = rows[:num_questions]
            else:
                # Random sample count between min and max
                num_questions = rng.randint(actual_min, actual_max)
                # Shuffle and take first num_questions
                rng.shuffle(rows)
                rows = rows[:num_questions]
        questions = []
        for idx, row in enumerate(rows):
            qid = f"Q{idx + 1}"
            answers = row.get("answers", {}).get("text", [])
            answer_text = answers[0].strip() if answers else ""
            answer_tokens = max(estimate_tokens(answer_text), 12)
            questions.append(
                {
                    "qid": qid,
                    "text": row["question"].strip(),
                    "answer_tokens": answer_tokens,
                    "references": answers,
                }
            )
        formatted.append(
            {
                "context": context_text.strip(),
                "title": row.get("title", "SQuAD-Context"),
                "questions": questions,
            }
        )
    return formatted


def load_squad_random_questions(
    split: str,
    *,
    max_contexts: Optional[int] = None,
    seed: int = 13,
) -> List[dict]:
    """
    Sample individual SQuAD questions without grouping by shared context.
    Each question becomes its own context group.

    Args:
        split: Dataset split (e.g., "train", "validation").
        max_contexts: Maximum number of contexts to return. None = use all.
        seed: Random seed for shuffling.
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")

    raw_dataset = list(load_dataset("squad", split=split))
    rng = random.Random(seed)
    rng.shuffle(raw_dataset)
    selected = raw_dataset[:max_contexts] if max_contexts else raw_dataset

    groups: List[dict] = []
    for idx, row in enumerate(selected, start=1):
        answers = row.get("answers", {}).get("text", [])
        answer_text = answers[0].strip() if answers else ""
        answer_tokens = max(estimate_tokens(answer_text), 12)
        qid = "Q1"
        groups.append(
            {
                "context": row["context"].strip(),
                "title": row.get("title", f"SQuAD-{idx}"),
                "questions": [
                    {
                        "qid": qid,
                        "text": row["question"].strip(),
                        "answer_tokens": answer_tokens,
                        "references": answers,
                    }
                ],
            }
        )
    return groups
