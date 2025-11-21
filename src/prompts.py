import textwrap
from typing import Dict, List, Tuple

from python import Question


def build_dependency_prompt(
    background: str,
    question: Question,
    answers: Dict[str, str],
    dependencies: List[str],
    question_lookup: Dict[str, Question],
) -> Tuple[str, str]:
    system_prompt = textwrap.dedent(
        f"""You are a helpful assistant that answers questions given a background passage.
You may reason freely, but give the final answer in the format \\box{{...}}. Example: \\box{{42}}
If the answer is unknown, write \\box{{unknown}}.

Background:
{background.strip()}
"""
    ).strip()

    user_lines: List[str] = []
    if dependencies:
        user_lines.append("Known previous answers:")
        for dep_id in dependencies:
            dep_question = question_lookup[dep_id]
            dep_answer = answers.get(dep_id, "").strip()
            escaped = dep_answer.replace("}", "\\}")
            user_lines.append(f"{dep_id} - {dep_question.text.strip()}")
            user_lines.append(f"Answer: \\box{{{escaped}}}")
        user_lines.append("")

    user_lines.append(f"Question ({question.qid}): {question.text.strip()}")
    user_prompt = "\n".join(user_lines)
    return system_prompt, user_prompt


def build_single_prompt(background: str, question: Question) -> Tuple[str, str]:
    system_prompt = textwrap.dedent(
        f"""You are a helpful assistant that answers questions given a background passage.
Provide concise reasoning if helpful, but the final line of every response must be exactly \\box{{answer}}. If the answer is unknown, return \\box{{unknown}}.

Background:
{background.strip()}
"""
    ).strip()
    user_prompt = f"Question ({question.qid}): {question.text.strip()}"
    return system_prompt, user_prompt
