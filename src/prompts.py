import textwrap
from typing import Dict, List, Optional, Tuple

from src.models import Question

# Only squad dataset has "unknown" labels
EXTRACTIVE_DATASETS = {"squad"}

# Datasets where answers should be extracted from context (not freely generated)
EXTRACTIVE_QA_DATASETS = {"squad", "quac", "hotpot"}

# Datasets that use direct answer format (no <answer> tags required)
DIRECT_ANSWER_DATASETS = {"cmb"}


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

    # CMB uses direct answer format without <answer> tags
    use_direct_format = dataset in DIRECT_ANSWER_DATASETS

    # Only extractive QA datasets allow "unknown" responses
    if dataset in EXTRACTIVE_DATASETS:
        unknown_instruction = " If the answer is unknown, return <answer>unknown</answer>."
    else:
        unknown_instruction = ""

    # Extractive QA datasets should extract answers from context
    if dataset in EXTRACTIVE_QA_DATASETS:
        extract_instruction = " Extract the answer directly from the background passage."
    else:
        extract_instruction = ""

    if use_direct_format:
        system_prompt = (
            f"You are a helpful medical assistant that answers questions given background passages.\n"
            f"Provide the answer directly without any special formatting.\n\n"
            f"Background:\n{effective_background.strip()}"
        )
    else:
        system_prompt = (
            f"You are a helpful assistant that answers questions given background passages.\n"
            f"Provide the answer with format <answer>text</answer>.{extract_instruction}{unknown_instruction}\n\n"
            f"Background:\n{effective_background.strip()}"
        )

    user_lines: List[str] = []
    if dependencies:
        user_lines.append("Known previous answers:")
        for dep_id in dependencies:
            dep_question = question_lookup[dep_id]
            dep_answer = answers.get(dep_id, "").strip()
            user_lines.append(f"{dep_id} - {dep_question.text.strip()}")
            # Show previous answer (without <answer> tags for CMB)
            if use_direct_format:
                user_lines.append(f"Answer: {dep_answer}")
            else:
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

    # CMB uses direct answer format without <answer> tags
    use_direct_format = dataset in DIRECT_ANSWER_DATASETS

    # Only extractive QA datasets allow "unknown" responses
    if dataset in EXTRACTIVE_DATASETS:
        unknown_instruction = " If the answer is unknown, return <answer>unknown</answer>."
    else:
        unknown_instruction = ""

    # Extractive QA datasets should extract answers from context
    if dataset in EXTRACTIVE_QA_DATASETS:
        extract_instruction = " Extract the answer directly from the background passage."
    else:
        extract_instruction = ""

    if use_direct_format:
        system_prompt = (
            f"You are a helpful medical assistant that answers questions given background passages.\n"
            f"Provide the answer directly without any special formatting.\n\n"
            f"Background:\n{effective_background.strip()}"
        )
    else:
        system_prompt = (
            f"You are a helpful assistant that answers questions given background passages.\n"
            f"Provide the answer with format <answer>text</answer>.{extract_instruction}{unknown_instruction}\n\n"
            f"Background:\n{effective_background.strip()}"
        )
    user_prompt = f"Question ({question.qid}): {question.text.strip()}"
    return system_prompt, user_prompt
