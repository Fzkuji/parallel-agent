from typing import List, Optional

from src.models import Question

from .eval import normalize_answer, compute_rouge_l
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
    dataset: Optional[str] = None,
) -> None:
    """Print answer comparison table.

    For short-form QA (squad, hotpot): shows ✓/✗ based on exact match.
    For long-form QA (cmb): shows ROUGE-L score for each answer.
    """
    headers = ["QID", "Question", "Gold"] + [res.name for res in strategies]
    rows = []
    max_answer_len = 40
    max_question_len = 60
    use_rouge = dataset == "cmb"

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
            if use_rouge:
                # For CMB: show ROUGE-L score
                score = compute_rouge_l(ans, question.references)
                truncated = ans[:max_answer_len] if len(ans) > max_answer_len else ans
                return f"[{score:.2f}] {truncated}"
            else:
                # For SQuAD/HotpotQA: show ✓/✗ based on exact match
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

    if use_rouge:
        print("\nAnswer comparison ([score] = ROUGE-L, ∅ = empty):")
    else:
        print("\nAnswer comparison (✓ = correct, ✗ = incorrect, ∅ = empty):")
    print("\n".join([header_line, separator, *row_lines]))
