"""GSM8K dataset loader for math reasoning evaluation."""

from typing import List, Dict
import random
import re


def extract_number_from_answer(answer: str) -> str:
    """Extract the final numerical answer from GSM8K answer string."""
    # GSM8K answers are in format "#### 42"
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', answer)
    if match:
        # Remove commas from numbers
        return match.group(1).replace(',', '')
    return answer.strip()


def load_gsm8k(
    split: str = "train",
    max_contexts: int = 100,
    min_questions: int = 5,
    max_questions: int = 5,
    seed: int = 42,
    group_by_difficulty: bool = False,
) -> List[Dict]:
    """
    Load GSM8K (Grade School Math 8K) dataset.

    GSM8K consists of grade-school level math word problems.
    We group questions together as if they're from the same "context"
    to test collaborative solving.

    Args:
        split: Dataset split ('train' or 'test')
        max_contexts: Maximum number of question groups to return
        min_questions: Minimum questions per group
        max_questions: Maximum questions per group
        seed: Random seed
        group_by_difficulty: Whether to group by difficulty (default: random grouping)

    Returns:
        List of context dictionaries with questions
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main", split=split, trust_remote_code=True)

    # Convert to list for easier manipulation
    all_items = list(dataset)

    # Shuffle items
    rng = random.Random(seed)
    rng.shuffle(all_items)

    # Group items into contexts
    formatted: List[Dict] = []
    current_group = []
    group_idx = 0

    for item in all_items:
        current_group.append(item)

        # When group reaches desired size, create a context
        if len(current_group) >= max_questions:
            # Randomly decide actual group size
            actual_size = rng.randint(min_questions, max_questions)
            selected_items = current_group[:actual_size]

            # Build context (generic math problem set)
            context = f"Math Problem Set {group_idx + 1}"

            questions = []
            for i, q_item in enumerate(selected_items):
                question_text = q_item["question"].strip()
                answer_with_solution = q_item["answer"].strip()

                # Extract numerical answer
                numerical_answer = extract_number_from_answer(answer_with_solution)

                questions.append({
                    "qid": f"Q{i+1}",
                    "question": question_text,
                    "answer_tokens": 32,  # Math answers can be multi-step
                    "references": [numerical_answer],
                })

            if questions:
                formatted.append({
                    "context": context,
                    "title": f"gsm8k-set-{group_idx}",
                    "questions": questions,
                })
                group_idx += 1

            # Reset for next group
            current_group = []

            # Stop if we have enough contexts
            if max_contexts and group_idx >= max_contexts:
                break

    return formatted
