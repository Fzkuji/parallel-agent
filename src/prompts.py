import textwrap
from typing import Dict, List, Optional, Tuple

from src.models import Question

# Only squad dataset has "unknown" labels
EXTRACTIVE_DATASETS = {"squad"}


def build_dependency_prompt(
    background: str,
    question: Question,
    answers: Dict[str, str],
    dependencies: List[str],
    question_lookup: Dict[str, Question],
    dataset: Optional[str] = None,
) -> Tuple[str, str]:
    # Use question-specific context if available, otherwise use shared background
    effective_background = question.context if question.context else background

    # Only extractive QA datasets allow "unknown" responses
    if dataset in EXTRACTIVE_DATASETS:
        unknown_instruction = " If the answer is unknown, return <answer>unknown</answer>."
    else:
        unknown_instruction = ""

    system_prompt = (
        f"You are a helpful assistant that answers questions given background passages.\n"
        f"Provide the answer with format <answer>text</answer>.{unknown_instruction}\n\n"
        f"Background:\n{effective_background.strip()}"
    )

    user_lines: List[str] = []
    if dependencies:
        user_lines.append("Known previous answers:")
        for dep_id in dependencies:
            dep_question = question_lookup[dep_id]
            dep_answer = answers.get(dep_id, "").strip()
            user_lines.append(f"{dep_id} - {dep_question.text.strip()}")
            # Show previous answer with <answer> tags
            user_lines.append(f"Answer: <answer>{dep_answer}</answer>")
        user_lines.append("")

    user_lines.append(f"Question ({question.qid}): {question.text.strip()}")
    user_prompt = "\n".join(user_lines)
    return system_prompt, user_prompt


def build_single_prompt(
    background: str, question: Question, dataset: Optional[str] = None
) -> Tuple[str, str]:
    # Use question-specific context if available, otherwise use shared background
    effective_background = question.context if question.context else background

    # Only extractive QA datasets allow "unknown" responses
    if dataset in EXTRACTIVE_DATASETS:
        unknown_instruction = " If the answer is unknown, return <answer>unknown</answer>."
    else:
        unknown_instruction = ""

    system_prompt = (
        f"You are a helpful assistant that answers questions given background passages.\n"
        f"Provide the answer with format <answer>text</answer>.{unknown_instruction}\n\n"
        f"Background:\n{effective_background.strip()}"
    )
    user_prompt = f"Question ({question.qid}): {question.text.strip()}"
    return system_prompt, user_prompt
