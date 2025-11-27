"""QuAC (Question Answering in Context) dataset loader.

QuAC is a conversational QA dataset where questions are asked in context
of a Wikipedia article, with follow-up questions building on previous answers.
"""
from __future__ import annotations

import random
from typing import List, Optional

from ..models import estimate_tokens

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def load_quac_groups(
    split: str,
    *,
    min_questions: int = 3,
    max_questions: Optional[int] = None,
    max_contexts: int = 3,
    seed: int = 13,
) -> List[dict]:
    """
    Load QuAC dialogues as context groups.

    Each dialogue contains multiple questions about a Wikipedia section.
    Questions are conversational and may reference previous turns.

    Args:
        split: Dataset split ("train" or "validation").
        min_questions: Minimum questions per dialogue to include.
        max_questions: Maximum questions per dialogue (truncate if exceeded).
        max_contexts: Maximum number of dialogues to return.
        seed: Random seed for shuffling.

    Returns:
        List of context dictionaries with questions.
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")

    raw_dataset = list(load_dataset("quac", split=split))

    # Filter by minimum questions
    filtered = [
        row for row in raw_dataset
        if len(row.get("questions", [])) >= min_questions
    ]
    if not filtered:
        raise ValueError("No dialogues satisfy the minimum question requirement.")

    rng = random.Random(seed)
    rng.shuffle(filtered)
    selected = filtered[:max_contexts] if max_contexts else filtered

    formatted: List[dict] = []
    for row in selected:
        questions_raw = row.get("questions", [])
        answers_raw = row.get("answers", {})
        orig_answers = row.get("orig_answers", {})

        # Build context: background + main context
        background = row.get("background", "").strip()
        context = row.get("context", "").strip()
        section_title = row.get("section_title", "")

        if background and context:
            full_context = f"{background}\n\n{context}"
        else:
            full_context = context or background

        # Truncate questions if needed
        if max_questions:
            questions_raw = questions_raw[:max_questions]

        questions = []
        for idx, question_text in enumerate(questions_raw):
            qid = f"Q{idx + 1}"

            # Get all annotator answers for this question
            # answers_raw["texts"] is a list of lists (one list per question, each with multiple annotator answers)
            if idx < len(answers_raw.get("texts", [])):
                all_answers = answers_raw["texts"][idx]
                # Filter out CANNOTANSWER responses
                valid_answers = [a for a in all_answers if a != "CANNOTANSWER"]
            else:
                valid_answers = []

            # Fall back to orig_answers if no valid answers
            if not valid_answers and idx < len(orig_answers.get("texts", [])):
                orig_text = orig_answers["texts"][idx]
                if orig_text != "CANNOTANSWER":
                    valid_answers = [orig_text]

            # Skip questions with no valid answers
            if not valid_answers:
                continue

            # Estimate tokens from first answer
            answer_tokens = max(estimate_tokens(valid_answers[0]), 12)

            questions.append({
                "qid": qid,
                "text": question_text.strip(),
                "answer_tokens": answer_tokens,
                "references": valid_answers,
            })

        # Only include if we have enough questions after filtering
        if len(questions) >= min_questions:
            title = row.get("wikipedia_page_title", "QuAC-Dialogue")
            if section_title:
                title = f"{title}: {section_title}"

            formatted.append({
                "context": full_context,
                "title": title,
                "questions": questions,
            })

    return formatted
