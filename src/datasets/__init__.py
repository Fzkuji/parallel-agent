"""Dataset loaders for QA benchmarks."""
from __future__ import annotations

from typing import List

from .squad import load_squad_groups, load_squad_random_questions
from .hotpot import load_hotpot_groups
from .cmb import load_cmb_groups, load_cmb_exam_random_groups, load_cmb_exam_subdomain_groups, load_cmb_exam_context_groups
from .quac import load_quac_groups
from .quality import load_quality_groups
from .drop import load_drop_groups
from .triviaqa import load_triviaqa, load_triviaqa_groups
from .mmlu import load_mmlu
from .gsm8k import load_gsm8k
from .similarity_grouped import (
    load_similarity_grouped_triviaqa,
    load_similarity_grouped_nq,
    load_similarity_grouped_generic,
)
from ..models import Question
from ..text_utils import detect_aggregate_question

__all__ = [
    "load_squad_groups",
    "load_squad_random_questions",
    "load_hotpot_groups",
    "load_cmb_groups",
    "load_cmb_exam_random_groups",
    "load_cmb_exam_subdomain_groups",
    "load_cmb_exam_context_groups",
    "load_quac_groups",
    "load_quality_groups",
    "load_drop_groups",
    "load_triviaqa",
    "load_triviaqa_groups",
    "load_mmlu",
    "load_gsm8k",
    "load_similarity_grouped_triviaqa",
    "load_similarity_grouped_nq",
    "load_similarity_grouped_generic",
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
