"""Dataset loading utilities for SQuAD and HotpotQA."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional

from .models import Question, estimate_tokens
from .text_utils import detect_aggregate_question

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
) -> List[dict]:
    """
    Load SQuAD questions grouped by shared context.

    Args:
        split: Dataset split (e.g., "train", "validation").
        min_questions: Minimum questions per context to include.
        max_questions: Maximum questions per context (truncate if exceeded).
        max_contexts: Maximum number of contexts to return.
        seed: Random seed for shuffling.

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
        if max_questions:
            rows = rows[:max_questions]
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
    max_contexts: int = 3,
    seed: int = 13,
) -> List[dict]:
    """
    Sample individual SQuAD questions without grouping by shared context.
    Each question becomes its own context group.
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")

    raw_dataset = list(load_dataset("squad", split=split))
    rng = random.Random(seed)
    rng.shuffle(raw_dataset)
    selected = raw_dataset[:max_contexts]

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


def _format_hotpot_context(row: dict) -> str:
    """Format HotpotQA context from titles and sentences."""
    titles = row.get("context", {}).get("title", [])
    sentences = row.get("context", {}).get("sentences", [])
    pieces: List[str] = []
    for title, sent_list in zip(titles, sentences):
        sent_text = " ".join(s.strip() for s in sent_list)
        pieces.append(f"{title}: {sent_text}")
    return "\n".join(pieces)


def load_hotpot_groups(
    split: str,
    *,
    subset: str = "distractor",
    max_contexts: int = 3,
    min_questions: int = 1,
    max_questions: int = 3,
    group_size: int | None = None,
    seed: int = 13,
) -> List[dict]:
    """
    Load HotpotQA rows and bundle multiple independent contexts/questions into a single "step".

    Args:
        split: Dataset split (e.g., "train", "validation").
        subset: HotpotQA subset (e.g., "distractor", "fullwiki").
        max_contexts: Maximum number of context groups to return.
        min_questions: Minimum questions per group (for random sizing).
        max_questions: Maximum questions per group (for random sizing).
        group_size: Fixed group size (overrides min/max).
        seed: Random seed for shuffling.

    Returns:
        List of context group dictionaries.
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")

    raw_dataset = list(load_dataset("hotpotqa/hotpot_qa", subset, split=split))
    if not raw_dataset:
        raise ValueError("Empty HotpotQA split.")

    rng = random.Random(seed)
    rng.shuffle(raw_dataset)

    groups: List[dict] = []
    cursor = 0
    while len(groups) < max_contexts and cursor < len(raw_dataset):
        current_size = group_size or rng.randint(min_questions, max_questions)
        batch_rows = raw_dataset[cursor : cursor + current_size]
        if not batch_rows:
            break
        items: List[dict] = []
        for local_idx, row in enumerate(batch_rows):
            background = _format_hotpot_context(row)
            answer_text = row.get("answer", "").strip()
            answer_tokens = max(estimate_tokens(answer_text), 12)
            items.append(
                {
                    "qid": f"Q{local_idx + 1}",
                    "context": background,
                    "question": row.get("question", "").strip(),
                    "answer_tokens": answer_tokens,
                    "references": [answer_text] if answer_text else [],
                }
            )
        groups.append({"items": items, "title": batch_rows[0].get("id", f"Hotpot-{len(groups)+1}")})
        cursor += current_size

    if not groups:
        raise ValueError("No HotpotQA groups constructed; check subset/split parameters.")
    return groups


def build_questions_from_group(group: dict) -> List[Question]:
    """Convert a context group dictionary to a list of Question objects."""
    questions: List[Question] = []
    for payload in group["questions"]:
        text = payload["text"]
        type_hint = None
        question = Question(
            qid=payload["qid"],
            text=text,
            priority=1.0 + (0.2 if detect_aggregate_question(text, type_hint) else 0.0),
            answer_tokens=payload["answer_tokens"],
            type_hint=type_hint,
            references=list(payload.get("references", [])),
        )
        questions.append(question)
    return questions
