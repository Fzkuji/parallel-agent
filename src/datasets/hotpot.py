"""HotpotQA dataset loader."""
from __future__ import annotations

import random
from typing import List

from ..models import estimate_tokens

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


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
        current_size = rng.randint(min_questions, max_questions)
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
