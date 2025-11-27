"""CMB (Chinese Medical Benchmark) dataset loader.

Dataset: FreedomIntelligence/CMB
Subset: CMB-Clin (Clinical case analysis with multiple QA pairs per case)

Each case has:
- title: Case title (e.g., "案例分析-腹外疝")
- id: Case identifier
- QA_pairs: List of {question, answer} dicts (3-4 questions per case)
- description: Shared context with patient info, exam results, etc.
"""
from __future__ import annotations

import random
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
