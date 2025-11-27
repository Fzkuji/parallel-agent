from typing import List, Optional

from src.models import Question

from .evaluation import normalize_answer, compute_rouge_l, get_metric_names
from .results import StrategyResult

# Display names for metrics (shorter names for table headers)
METRIC_DISPLAY_NAMES = {
    "strict_acc": "EM",
    "f1": "F1",
    "lenient_acc": "Lenient",
    "bleu4": "BLEU-4",
    "rouge1": "R-1",
    "rouge2": "R-2",
    "rougeL": "R-L",
}


def summarize_results(results: List[StrategyResult], dataset: Optional[str] = None) -> str:
    """Summarize strategy results in a table format.

    Metrics are automatically selected based on dataset configuration.
    """
    # Get metric names for this dataset (default to squad metrics if not specified)
    metric_names = get_metric_names(dataset) if dataset else ["strict_acc", "f1", "lenient_acc"]

    # Build headers: Strategy + metrics + common columns
    metric_headers = [METRIC_DISPLAY_NAMES.get(m, m) for m in metric_names]
    headers = ["Strategy"] + metric_headers + ["PromptTok", "GenTok", "Latency(s)", "Batches"]

    # Build rows
    rows = []
    for res in results:
        row = [res.name]
        # Add metric values
        for metric_name in metric_names:
            value = res.metrics.get(metric_name, 0)
            row.append(f"{value:.3f}")
        # Add common columns
        row.extend([
            res.prompt_tokens,
            res.generated_tokens,
            f"{res.latency:.2f}",
            res.batches,
        ])
        rows.append(row)

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
