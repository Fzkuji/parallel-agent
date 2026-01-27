"""CoQA (Conversational Question Answering) dataset loader.

Dataset: stanfordnlp/coqa
A conversational QA dataset with multi-turn dialogues.

Each story has:
- A source passage (story)
- Multiple question-answer turns (~15 per story on average)
- Questions that may depend on prior context

Reference: https://arxiv.org/abs/1808.07042
"""
from __future__ import annotations

import random
from typing import List, Optional

from ..models import estimate_tokens

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def load_coqa_groups(
    split: str = "validation",
    *,
    min_questions: int = 1,
    max_questions: Optional[int] = None,
    max_contexts: int = 3,
    seed: int = 13,
) -> List[dict]:
    """
    Load CoQA stories with multiple QA turns per story.

    Args:
        split: Dataset split ("train" or "validation").
        min_questions: Minimum questions per story to include.
        max_questions: Maximum questions per story (truncate if exceeded).
        max_contexts: Maximum number of stories to return.
        seed: Random seed for shuffling.

    Returns:
        List of context dictionaries with questions.
        Format matches other datasets: {context, title, questions: [{qid, text, answer_tokens, references}]}
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")

    raw_dataset = list(load_dataset("stanfordnlp/coqa", split=split, trust_remote_code=True))
    if not raw_dataset:
        raise ValueError(f"Empty CoQA split: {split}")

    # Filter by minimum questions
    filtered = [
        row for row in raw_dataset
        if len(row.get("questions", [])) >= min_questions
    ]
    if not filtered:
        raise ValueError(f"No stories satisfy the minimum question requirement ({min_questions}).")

    # Shuffle and select
    rng = random.Random(seed)
    rng.shuffle(filtered)
    selected = filtered[:max_contexts] if max_contexts else filtered

    formatted: List[dict] = []
    for idx, row in enumerate(selected):
        story = row.get("story", "").strip()
        source = row.get("source", "unknown")
        raw_questions = row.get("questions", [])
        raw_answers = row.get("answers", {})

        # Get answer texts
        answer_texts = raw_answers.get("input_text", [])

        # Randomly sample number of questions between min_questions and max_questions
        if max_questions:
            actual_max = min(max_questions, len(raw_questions))
            actual_min = min(min_questions, actual_max)
            num_questions = rng.randint(actual_min, actual_max)
            # For CoQA, we should keep the order since it's conversational
            # Just truncate to the first num_questions
            raw_questions = raw_questions[:num_questions]
            answer_texts = answer_texts[:num_questions]

        questions = []
        for q_idx, (question_text, answer_text) in enumerate(zip(raw_questions, answer_texts)):
            qid = f"Q{q_idx + 1}"

            # Estimate tokens based on answer
            answer_tokens = max(estimate_tokens(answer_text), 12)

            questions.append({
                "qid": qid,
                "text": question_text,
                "answer_tokens": answer_tokens,
                "references": [answer_text] if answer_text else [],
            })

        # Create title from source and first 30 chars of story
        title = f"{source}: {story[:30].replace(chr(10), ' ').strip()}..."

        formatted.append({
            "context": story,
            "title": title,
            "questions": questions,
        })

    return formatted
