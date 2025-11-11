import re
import textwrap
from typing import Dict, List, Tuple

from python import Question


def build_answer_prompt(
    background: str,
    question: Question,
    answers: Dict[str, str],
    dependencies: List[str],
    question_lookup: Dict[str, Question],
) -> str:
    prompt_parts = [
        "You are a helpful assistant that answers questions given a background passage.",
        "You may reason freely, but give the final answer in the format \\box{...}. Example: \\box{42}",
        "If the answer is unknown, write \\box{unknown}.",
        "",
        "Background:",
        background.strip(),
        "",
    ]
    if dependencies:
        prompt_parts.append("Known previous answers:")
        for dep_id in dependencies:
            dep_question = question_lookup[dep_id]
            dep_answer = answers.get(dep_id, "").strip()
            prompt_parts.append(f"{dep_id} - {dep_question.text.strip()}")
            escaped = dep_answer.replace("}", "\\}")
            prompt_parts.append(f"Answer: \\box{{{escaped}}}")
        prompt_parts.append("")
    prompt_parts.append(f"Question ({question.qid}): {question.text.strip()}")
    return "\n".join(prompt_parts)


def build_batch_prompt(background: str, questions: List[Question]) -> str:
    question_lines = [f"Q{question.qid[1:]}: {question.text.strip()}" for question in questions]
    return textwrap.dedent(
        f"""
        You are a helpful assistant. Read the background and answer each question.
        You may reason for each question, but finish its answer on a separate line in the form:
        A1: \\box{{final answer}}
        Do not add any text after the \\box. If the answer is unknown, use \\box{{unknown}}.

        Example:
        A1 reasoning...
        A1: \\box{{final value}}

        Background:
        {background.strip()}

        Questions:
        {'; '.join(question_lines)}

        Provide the answers (each prefixed by A1/A2/etc and ending with \\box{{...}}):
        """
    ).strip()


def parse_batch_answers(text: str, questions: List[Question]) -> Dict[str, Tuple[str, bool]]:
    answers: Dict[str, Tuple[str, bool]] = {}
    matches = list(re.compile(r"\\box\{([^}]*)\}").finditer(text))
    for idx, question in enumerate(questions):
        if idx < len(matches):
            answers[question.qid] = (matches[idx].group(1).strip(), True)
        else:
            answers[question.qid] = ("", False)
    return answers
