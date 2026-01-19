"""DROP (Discrete Reasoning Over Paragraphs) dataset loader.

Dataset: ucinlp/drop
A reading comprehension benchmark requiring discrete reasoning (arithmetic, counting, sorting).

Each passage has:
- Multiple questions (~16 per passage on average)
- Questions requiring numerical reasoning, counting, sorting, etc.
- Multiple valid answer spans per question

Reference: https://arxiv.org/abs/1903.00161
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


def load_drop_groups(
    split: str = "validation",
    *,
    min_questions: int = 1,
    max_questions: Optional[int] = None,
    max_contexts: int = 3,
    seed: int = 13,
) -> List[dict]:
    """
    Load DROP passages with multiple questions per passage.

    Args:
        split: Dataset split ("train" or "validation").
        min_questions: Minimum questions per passage to include.
        max_questions: Maximum questions per passage (truncate if exceeded).
        max_contexts: Maximum number of passages to return.
        seed: Random seed for shuffling.

    Returns:
        List of context dictionaries with questions.
        Format matches other datasets: {context, title, questions: [{qid, text, answer_tokens, references}]}
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")

    raw_dataset = list(load_dataset("ucinlp/drop", split=split))
    if not raw_dataset:
        raise ValueError(f"Empty DROP split: {split}")

    # Group questions by passage
    passage_questions = defaultdict(list)
    passage_full_text = {}  # Store full passage text

    for sample in raw_dataset:
        passage_key = sample["section_id"]
        passage_full_text[passage_key] = sample["passage"]

        # Get all answer spans (multiple annotators may provide different answers)
        answer_spans = sample["answers_spans"]["spans"]
        # Deduplicate while preserving order
        seen = set()
        unique_answers = []
        for ans in answer_spans:
            if ans not in seen:
                seen.add(ans)
                unique_answers.append(ans)

        passage_questions[passage_key].append({
            "query_id": sample["query_id"],
            "question": sample["question"],
            "answers": unique_answers,
            "answer_types": sample["answers_spans"]["types"],
        })

    # Convert to list and filter by min_questions
    passages = []
    for passage_key, questions in passage_questions.items():
        if len(questions) >= min_questions:
            passages.append({
                "passage_key": passage_key,
                "passage": passage_full_text[passage_key],
                "questions": questions,
            })

    if not passages:
        raise ValueError(f"No passages satisfy the minimum question requirement ({min_questions}).")

    # Shuffle and select
    rng = random.Random(seed)
    rng.shuffle(passages)
    selected = passages[:max_contexts] if max_contexts else passages

    formatted: List[dict] = []
    for idx, passage_data in enumerate(selected):
        passage_text = passage_data["passage"]
        questions_data = passage_data["questions"]

        # Randomly sample number of questions between min_questions and max_questions
        if max_questions:
            actual_max = min(max_questions, len(questions_data))
            actual_min = min(min_questions, actual_max)
            num_questions = rng.randint(actual_min, actual_max)
            # Shuffle questions within passage
            rng.shuffle(questions_data)
            questions_data = questions_data[:num_questions]

        # Create title from first 50 chars of passage
        title = passage_text[:50].replace("\n", " ").strip() + "..."

        questions = []
        for q_idx, q in enumerate(questions_data):
            qid = f"Q{q_idx + 1}"
            question_text = q["question"]
            answers = q["answers"]

            # Estimate tokens based on longest answer
            max_answer_len = max(len(ans) for ans in answers) if answers else 10
            answer_tokens = max(estimate_tokens(answers[0]) if answers else 12, 12)

            questions.append({
                "qid": qid,
                "text": question_text,
                "answer_tokens": answer_tokens,
                "references": answers,  # Multiple valid answers
                "query_id": q["query_id"],  # Original query ID for reference
            })

        formatted.append({
            "context": passage_text,
            "title": title,
            "questions": questions,
        })

    return formatted
