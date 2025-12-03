"""QuALITY (Question Answering with Long Input Texts, Yes!) dataset loader.

Dataset: emozilla/quality
A challenging long-context reading comprehension dataset with multiple-choice questions.

Each article has:
- article: Long text (~5000 words on average)
- Multiple questions per article (~18 questions on average)
- 4 options per question
- Hard flag indicating difficult questions requiring full document understanding

Reference: https://github.com/nyu-mll/quality
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


def load_quality_groups(
    split: str = "validation",
    *,
    min_questions: int = 1,
    max_questions: Optional[int] = None,
    max_contexts: int = 3,
    seed: int = 13,
    hard_only: bool = False,
) -> List[dict]:
    """
    Load QuALITY articles with multiple questions per article.

    Args:
        split: Dataset split ("train" or "validation").
        min_questions: Minimum questions per article to include.
        max_questions: Maximum questions per article (truncate if exceeded).
        max_contexts: Maximum number of articles to return.
        seed: Random seed for shuffling.
        hard_only: If True, only include hard questions (requires full document understanding).

    Returns:
        List of context dictionaries with questions.
        Format matches other datasets: {context, title, questions: [{qid, text, answer_tokens, references, options}]}
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")

    raw_dataset = list(load_dataset("emozilla/quality", split=split))
    if not raw_dataset:
        raise ValueError(f"Empty QuALITY split: {split}")

    # Group questions by article
    article_questions = defaultdict(list)
    for sample in raw_dataset:
        article_key = sample["article"]
        article_questions[article_key].append({
            "question": sample["question"],
            "options": sample["options"],
            "answer": sample["answer"],  # 0-indexed
            "hard": sample["hard"],
        })

    # Convert to list and filter
    articles = []
    for article_text, questions in article_questions.items():
        # Filter hard questions if requested
        if hard_only:
            questions = [q for q in questions if q["hard"]]

        if len(questions) >= min_questions:
            articles.append({
                "article": article_text,
                "questions": questions,
            })

    if not articles:
        raise ValueError(f"No articles satisfy the minimum question requirement ({min_questions}).")

    # Shuffle and select
    rng = random.Random(seed)
    rng.shuffle(articles)
    selected = articles[:max_contexts] if max_contexts else articles

    formatted: List[dict] = []
    for idx, article_data in enumerate(selected):
        article_text = article_data["article"]
        questions_data = article_data["questions"]

        # Truncate questions if needed
        if max_questions:
            questions_data = questions_data[:max_questions]

        # Create title from first 50 chars of article
        title = article_text[:50].replace("\n", " ").strip() + "..."

        questions = []
        for q_idx, q in enumerate(questions_data):
            qid = f"Q{q_idx + 1}"
            question_text = q["question"]
            options = q["options"]
            correct_answer_idx = q["answer"]
            correct_answer = options[correct_answer_idx]

            # Format question with options for the model
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            full_question = f"{question_text}\n\nOptions:\n{options_text}"

            # Estimate tokens for the answer (letter + explanation)
            answer_tokens = max(estimate_tokens(correct_answer), 12)

            questions.append({
                "qid": qid,
                "text": full_question,
                "answer_tokens": answer_tokens,
                "references": [correct_answer, chr(65 + correct_answer_idx)],  # Accept full answer or just letter
                "options": options,
                "correct_index": correct_answer_idx,
                "hard": q["hard"],
            })

        formatted.append({
            "context": article_text,
            "title": title,
            "questions": questions,
        })

    return formatted
