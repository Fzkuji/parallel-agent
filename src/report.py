from typing import List

from src.models import Question

from .eval import normalize_answer
from .results import StrategyResult


def summarize_results(results: List[StrategyResult]) -> str:
    headers = [
        "Strategy",
        "EM",
        "F1",
        "Lenient ACC",
        "PromptTok",
        "GenTok",
        "Latency(s)",
        "Batches",
    ]
    rows = [
        [
            res.name,
            f"{res.metrics['strict_acc']:.3f}",
            f"{res.metrics['f1']:.3f}",
            f"{res.metrics['lenient_acc']:.3f}",
            res.prompt_tokens,
            res.generated_tokens,
            f"{res.latency:.2f}",
            res.batches,
        ]
        for res in results
    ]
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    header_line = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    row_lines = [" | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def print_answer_table(
    questions: List[Question],
    strategies: List[StrategyResult],
) -> None:
    headers = ["QID", "Question", "Gold"] + [res.name for res in strategies]
    rows = []
    max_answer_len = 40
    max_question_len = 60

    for question in questions:
        gold = "; ".join(question.references) if question.references else ""
        if len(gold) > max_answer_len:
            gold = gold[: max_answer_len - 3] + "..."

        question_text = question.text.strip()
        if len(question_text) > max_question_len:
            question_text = question_text[: max_question_len - 3] + "..."

        def mark_answer(ans: str) -> str:
            if not ans:
                return "∅"
            norm_ans = normalize_answer(ans)
            for ref in question.references:
                if normalize_answer(ref) == norm_ans:
                    return f"✓ {ans[:max_answer_len]}"
            return f"✗ {ans[:max_answer_len]}"

        row = [question.qid, question_text, gold]
        for res in strategies:
            row.append(mark_answer(res.answers.get(question.qid, "")))
        rows.append(row)

    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    header_line = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    row_lines = [" | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows]
    print("\nAnswer comparison (✓ = correct, ✗ = incorrect, ∅ = empty):")
    print("\n".join([header_line, separator, *row_lines]))
