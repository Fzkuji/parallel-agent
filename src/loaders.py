"""Dataset loading utilities - re-exports from src/datasets/ for backward compatibility."""
from __future__ import annotations

from typing import List

from .datasets import (
    load_squad_groups,
    load_squad_random_questions,
    load_hotpot_groups,
    load_cmb_groups,
)
from .models import Question
from .text_utils import detect_aggregate_question

# Re-export all dataset loaders
__all__ = [
    "load_squad_groups",
    "load_squad_random_questions",
    "load_hotpot_groups",
    "load_cmb_groups",
    "build_questions_from_group",
]


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
