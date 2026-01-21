"""TriviaQA dataset loader for open-domain QA evaluation."""

from typing import List, Dict, Optional
import random


def estimate_tokens(text: str) -> int:
    """Rough token count estimation."""
    return max(len(text.split()), len(text) // 4)


def load_triviaqa_groups(
    split: str = "train",
    max_groups: Optional[int] = None,
    min_questions: int = 1,
    max_questions: int = 10,
    seed: int = 42,
) -> List[Dict]:
    """
    Load TriviaQA dataset and group questions randomly.

    Since TriviaQA questions are independent (no shared context),
    we randomly group them together for cross-batch training.
    Each group has a random number of questions between min_questions and max_questions.

    Args:
        split: Dataset split to load ('train' or 'validation')
        max_groups: Maximum number of groups to return (None for all possible)
        min_questions: Minimum questions per group
        max_questions: Maximum questions per group
        seed: Random seed

    Returns:
        List of group dictionaries, each with:
        - context: Empty string (TriviaQA is open-domain)
        - title: Group identifier
        - questions: List of question dicts
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    # Load TriviaQA dataset
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split, trust_remote_code=True)

    rng = random.Random(seed)

    # Shuffle all questions
    all_items = list(dataset)
    rng.shuffle(all_items)

    # Group questions randomly
    groups = []
    idx = 0
    group_id = 0

    while idx < len(all_items):
        # Random group size between min and max
        group_size = rng.randint(min_questions, max_questions)

        # Get questions for this group
        group_items = all_items[idx:idx + group_size]
        if len(group_items) < min_questions:
            break  # Not enough questions left

        questions = []
        for q_idx, item in enumerate(group_items):
            question_text = item["question"].strip()

            # TriviaQA provides answer aliases
            answer_aliases = item.get("answer", {}).get("aliases", [])
            if not answer_aliases:
                answer_aliases = [item.get("answer", {}).get("value", "")]

            answer_text = answer_aliases[0] if answer_aliases else ""
            answer_tokens = max(estimate_tokens(answer_text), 12)

            questions.append({
                "qid": f"Q{q_idx + 1}",
                "text": question_text,
                "answer_tokens": answer_tokens,
                "references": answer_aliases,
            })

        groups.append({
            "context": "",  # TriviaQA is open-domain, no shared context
            "title": f"triviaqa-group-{group_id}",
            "questions": questions,
        })

        idx += group_size
        group_id += 1

        if max_groups and len(groups) >= max_groups:
            break

    return groups


def load_triviaqa(
    split: str = "validation",
    max_contexts: int = 100,
    min_questions: int = 1,
    max_questions: int = 1,
    seed: int = 42,
) -> List[Dict]:
    """
    Load TriviaQA dataset.

    TriviaQA is an open-domain QA dataset with multiple evidence documents.
    We treat each question-answer pair as a separate context.

    Args:
        split: Dataset split to load
        max_contexts: Maximum number of contexts to return
        min_questions: Minimum questions per context (always 1 for TriviaQA)
        max_questions: Maximum questions per context (always 1 for TriviaQA)
        seed: Random seed

    Returns:
        List of context dictionaries with questions
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    # Load TriviaQA dataset
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split, trust_remote_code=True)

    # Sample contexts
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    selected_indices = indices[:max_contexts] if max_contexts else indices

    formatted: List[Dict] = []

    for idx in selected_indices:
        item = dataset[int(idx)]

        question_text = item["question"].strip()

        # TriviaQA provides answer aliases
        answer_aliases = item.get("answer", {}).get("aliases", [])
        if not answer_aliases:
            answer_aliases = [item.get("answer", {}).get("value", "")]

        # For open-domain QA, we don't use context (or use minimal context)
        # This tests the model's world knowledge
        context = f"Question: {question_text}"

        # Estimate answer tokens from first alias
        answer_text = answer_aliases[0] if answer_aliases else ""
        answer_tokens = max(estimate_tokens(answer_text), 12)

        formatted.append({
            "context": context,
            "title": f"trivia-{idx}",
            "questions": [
                {
                    "qid": "Q1",
                    "question": question_text,
                    "answer_tokens": answer_tokens,
                    "references": answer_aliases,
                }
            ],
        })

    return formatted
