import textwrap
from typing import Dict, List, Optional, Tuple

from src.models import Question

# Only squad dataset has "unknown" labels
EXTRACTIVE_DATASETS = {"squad"}

# Datasets where answers should be extracted from context (not freely generated)
EXTRACTIVE_QA_DATASETS = {"squad", "quac", "hotpot"}

# Datasets that use direct answer format (no <answer> tags required)
DIRECT_ANSWER_DATASETS = {"cmb"}

# Multiple-choice datasets (respond with option letter only)
MULTIPLE_CHOICE_DATASETS = {"cmb_exam", "mmlu"}


def build_dependency_prompt(
    background: str,
    question: Question,
    answers: Dict[str, str],
    dependencies: List[str],
    question_lookup: Dict[str, Question],
    dataset: Optional[str] = None,
) -> Tuple[str, str]:
    """Build a prompt with dependency information.

    Uses <answer> tag format for answer extraction.
    """
    # Use question-specific context if available, otherwise use shared background
    effective_background = question.context if question.context else background

    # Multiple-choice questions: respond with option letter only
    use_mc_format = dataset in MULTIPLE_CHOICE_DATASETS

    if use_mc_format:
        system_prompt = (
            "You are a helpful assistant. Read the question and options, then reply with the single correct option letter (A, B, C, D, ...)."
        )
    else:
        system_prompt = (
            "You are a helpful assistant. Answer the question based on the given passage.\n"
            "You MUST wrap your answer in <answer></answer> tags."
        )

    user_lines: List[str] = []

    # Add context/passage
    if use_mc_format:
        user_lines.append(f"背景信息:\n{effective_background.strip()}")
    else:
        user_lines.append(f"Passage:\n{effective_background.strip()}")

    # Add dependency information if any
    if dependencies:
        user_lines.append("")
        user_lines.append("Known previous answers:")
        for dep_id in dependencies:
            dep_question = question_lookup[dep_id]
            dep_answer = answers.get(dep_id, "").strip()
            user_lines.append(f"- {dep_question.text.strip()}: {dep_answer}")

    # Add current question
    user_lines.append("")
    if use_mc_format:
        user_lines.append(f"问题: {question.text.strip()}")
    else:
        user_lines.append(f"Question: {question.text.strip()}")

    user_prompt = "\n".join(user_lines)
    return system_prompt, user_prompt


def build_single_prompt(
    background: str, question: Question, dataset: Optional[str] = None
) -> Tuple[str, str]:
    """Build a single-question prompt.

    Uses <answer> tag format for answer extraction.
    """
    # Use question-specific context if available, otherwise use shared background
    effective_background = question.context if question.context else background

    # Multiple-choice questions: respond with option letter only
    use_mc_format = dataset in MULTIPLE_CHOICE_DATASETS

    if use_mc_format:
        system_prompt = (
            "You are a helpful assistant. Read the question and options, then reply with the single correct option letter (A, B, C, D, ...)."
        )
        user_prompt = f"Passage:\n{effective_background.strip()}\n\nQuestion: {question.text.strip()}\n\nRespond with the option letter only."
    else:
        # Use same system prompt as other strategies (must match eval_utils.SYSTEM_PROMPT)
        system_prompt = (
            "You are a helpful assistant. Answer the question based on the given passage.\n"
            "You MUST wrap your answer in <answer></answer> tags. Be concise.\n\n"
            "Example:\n"
            "Question: What color is the sky?\n"
            "<answer>blue</answer>"
        )
        user_prompt = f"Passage:\n{effective_background.strip()}\n\nQuestion: {question.text.strip()}"

    return system_prompt, user_prompt
