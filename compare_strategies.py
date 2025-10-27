from __future__ import annotations

import argparse
import json
import logging
import os
import re
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from python import (
    EdgeCandidate,
    DependencyScheduler,
    HeuristicDependencyGenerator,
    Question,
    apply_dependencies,
    build_questions_from_group,
    load_squad_groups,
    select_dependency_edges,
)
from run_qwen_parallel import (
    LocalLLMDependencyGenerator,
    DEFAULT_SYSTEM_PROMPT,
    build_chat_prompt,
)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def compute_contains(prediction: str, references: List[str]) -> float:
    pred_norm = normalize_answer(prediction)
    for ref in references:
        ref_norm = normalize_answer(ref)
        if not ref_norm:
            continue
        if ref_norm in pred_norm or pred_norm in ref_norm:
            return 1.0
    return 0.0


def extract_json_answer(text: str) -> Optional[str]:
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                cleaned = part[4:].strip()
                break
            if part.startswith("{"):
                cleaned = part
                break
    try:
        payload = json.loads(cleaned)
        value = payload.get("answer")
        if isinstance(value, str):
            return value.strip()
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if match:
            snippet = match.group(0)
            try:
                payload = json.loads(snippet)
                value = payload.get("answer")
                if isinstance(value, str):
                    return value.strip()
            except json.JSONDecodeError:
                pass
    return None


def compute_f1(prediction: str, references: List[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best_f1 = 0.0
    for ref in references:
        ref_tokens = normalize_answer(ref).split()
        if not ref_tokens:
            continue
        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            continue
        common_count = sum(min(pred_tokens.count(tok), ref_tokens.count(tok)) for tok in common)
        precision = common_count / len(pred_tokens)
        recall = common_count / len(ref_tokens)
        if precision + recall == 0.0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def generate_answer(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: Optional[float] = None,
) -> Tuple[str, int, int, float, bool]:
    chat_prompt = build_chat_prompt(tokenizer, prompt, system_prompt=DEFAULT_SYSTEM_PROMPT)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature is not None and temperature > 0:
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
        )
    else:
        gen_kwargs.update(do_sample=False)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            return_dict_in_generate=True,
            output_scores=False,
            **gen_kwargs,
        )
    elapsed = time.perf_counter() - start
    sequences = generated.sequences
    prompt_tokens = inputs["input_ids"].shape[-1]
    generated_part = sequences[:, prompt_tokens:]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    trimmed_tokens: List[List[int]] = []
    for row in generated_part:
        tokens = row.tolist()
        actual: List[int] = []
        for token in tokens:
            if token in (eos_id, pad_id):
                break
            actual.append(token)
        trimmed_tokens.append(actual)

    decoded = [tokenizer.decode(tokens, skip_special_tokens=True).strip() for tokens in trimmed_tokens]
    raw_text = decoded[0] if decoded else ""
    extracted = extract_json_answer(raw_text)
    strict_valid = extracted is not None
    output_text = extracted if strict_valid else raw_text
    generated_tokens = len(trimmed_tokens[0]) if trimmed_tokens else 0
    return output_text, prompt_tokens, generated_tokens, elapsed, strict_valid


def build_answer_prompt(
    background: str,
    question: Question,
    answers: Dict[str, str],
    dependencies: List[str],
    question_lookup: Dict[str, Question],
) -> str:
    prompt_parts = [
        "You are a helpful assistant that answers questions given a background passage.",
        "Provide the shortest possible span from the passage that answers the question.",
        "Respond strictly as a JSON object: {\"answer\": \"<span-or-unknown>\"}.",
        "If uncertain, output {\"answer\": \"unknown\"}.",
        "Do not add any explanation or text outside the JSON object.",
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
            prompt_parts.append(f"Answer: {dep_answer}")
        prompt_parts.append("")
    prompt_parts.append(f"Question ({question.qid}): {question.text.strip()}")
    prompt_parts.append("Answer JSON:")
    return "\n".join(prompt_parts)


def build_sequential_prompt(history: str, question: Question) -> str:
    return f"{history}Question ({question.qid}): {question.text.strip()}\nAnswer JSON:"


def evaluate_predictions(predictions: Dict[str, Tuple[str, bool]], lookup: Dict[str, Question]) -> Dict[str, float]:
    total = len(predictions)
    if total == 0:
        return {"strict_acc": 0.0, "lenient_acc": 0.0}
    strict_sum = 0.0
    lenient_sum = 0.0
    for qid, (pred, strict_valid) in predictions.items():
        refs = lookup[qid].references
        strict_sum += compute_contains(pred, refs) if strict_valid else 0.0
        lenient_sum += compute_contains(pred, refs)
    return {"strict_acc": strict_sum / total, "lenient_acc": lenient_sum / total}


def build_batch_prompt(background: str, questions: List[Question]) -> str:
    question_lines = []
    for question in questions:
        question_lines.append(f"Q{question.qid[1:]}: {question.text.strip()}")
    prompt = textwrap.dedent(
        f"""
        You are a helpful assistant. Read the background and answer each question with the shortest possible span from the passage.
        If the answer is not explicitly stated, return 'unknown'.
        Return answers strictly in the format:
        A1: ...
        A2: ...
        etc.

        Background:
        {background.strip()}

        Questions:
        {'; '.join(question_lines)}

        Answers:
        """
    ).strip()
    return prompt


def parse_batch_answers(text: str, questions: List[Question]) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    pattern = re.compile(r"A(\d+)\s*[:：]\s*(.+)")
    for line in text.splitlines():
        line = line.strip()
        match = pattern.match(line)
        if not match:
            continue
        idx = match.group(1)
        value = match.group(2).strip()
        qid = f"Q{idx}"
        answers[qid] = value
    # ensure all have entries
    for question in questions:
        answers.setdefault(question.qid, "")
    return answers


@dataclass
class StrategyResult:
    name: str
    answers: Dict[str, str]
    prompt_tokens: int
    generated_tokens: int
    latency: float
    batches: int
    metrics: Dict[str, float]


def run_dependency_strategy(
    background: str,
    questions: List[Question],
    generator,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    cost_weight: float,
    min_confidence: float,
    max_dependencies: int,
    total_cost_budget: Optional[int],
    max_new_tokens: int,
) -> StrategyResult:
    question_lookup = {q.qid: q for q in questions}
    edges = generator.generate_edges(background, questions)
    selected = select_dependency_edges(
        question_lookup,
        edges,
        cost_weight=cost_weight,
        min_confidence=min_confidence,
        max_dependencies_per_target=max_dependencies,
        total_cost_budget=total_cost_budget,
        fmt_overhead=6,
    )
    apply_dependencies(question_lookup, selected)
    scheduler = DependencyScheduler(
        background,
        questions,
        max_batch_tokens=None,
        fmt_overhead_per_section=6,
        prefill_token_cost=0.8,
        generate_token_cost=1.2,
    )
    scheduler.build_dependencies(auto_infer=False)
    schedule = scheduler.schedule()

    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_latency = 0.0

    for batch in schedule.batches:
        batch_latencies: List[float] = []
        for qid in batch.question_ids:
            question = question_lookup[qid]
            deps = sorted(question.dependencies)
            prompt = build_answer_prompt(background, question, answers_text, deps, question_lookup)
            text, p_tokens, g_tokens, elapsed, strict_valid = generate_answer(
                tokenizer, model, prompt, max_new_tokens=max_new_tokens, temperature=None
            )
            answer_records[qid] = (text, strict_valid)
            answers_text[qid] = text
            total_prompt_tokens += p_tokens
            total_generated_tokens += g_tokens
            batch_latencies.append(elapsed)
        if batch_latencies:
            total_latency += max(batch_latencies)

    metrics = evaluate_predictions(answer_records, question_lookup)
    return StrategyResult(
        name="dependency_parallel",
        answers=answers_text,
        prompt_tokens=total_prompt_tokens,
        generated_tokens=total_generated_tokens,
        latency=total_latency,
        batches=len(schedule.batches),
        metrics=metrics,
    )


def run_sequential_strategy(
    background: str,
    questions: List[Question],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    max_new_tokens: int,
) -> StrategyResult:
    question_lookup = {q.qid: q for q in questions}
    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_latency = 0.0

    history = textwrap.dedent(
        f"""You are a helpful assistant that answers questions given a background passage.
Provide the shortest possible span from the passage that answers each question.
Respond strictly as JSON: {{\"answer\": \"<span-or-unknown>\"}}.
If the answer is not explicitly stated, return {{\"answer\": \"unknown\"}}.

Background:
{background.strip()}

"""
    )

    for question in questions:
        prompt = build_sequential_prompt(history, question)
        text, p_tokens, g_tokens, elapsed, strict_valid = generate_answer(
            tokenizer, model, prompt, max_new_tokens=max_new_tokens, temperature=None
        )
        answer_records[question.qid] = (text, strict_valid)
        answers_text[question.qid] = text
        total_prompt_tokens += p_tokens
        total_generated_tokens += g_tokens
        total_latency += elapsed
        safe_text = text.replace('"', '\\"')
        history = (
            f"{history}Question ({question.qid}): {question.text.strip()}\n"
            f"Answer JSON: {{\"answer\": \"{safe_text}\"}}\n\n"
        )

    metrics = evaluate_predictions(answer_records, question_lookup)
    return StrategyResult(
        name="sequential",
        answers=answers_text,
        prompt_tokens=total_prompt_tokens,
        generated_tokens=total_generated_tokens,
        latency=total_latency,
        batches=len(questions),
        metrics=metrics,
    )


def run_full_batch_strategy(
    background: str,
    questions: List[Question],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    max_new_tokens: int,
) -> StrategyResult:
    prompt = build_batch_prompt(background, questions)
    text, p_tokens, g_tokens, elapsed, strict_valid = generate_answer(
        tokenizer, model, prompt, max_new_tokens=max_new_tokens, temperature=None
    )
    answers = parse_batch_answers(text, questions)
    answer_records = {qid: (ans, bool(ans.strip())) for qid, ans in answers.items()}
    metrics = evaluate_predictions(answer_records, {q.qid: q for q in questions})
    return StrategyResult(
        name="full_batch",
        answers=answers,
        prompt_tokens=p_tokens,
        generated_tokens=g_tokens,
        latency=elapsed,
        batches=1,
        metrics=metrics,
    )


def summarize_results(results: List[StrategyResult]) -> str:
    headers = [
        "Strategy",
        "Strict ACC",
        "Lenient ACC",
        "PromptTok",
        "GenTok",
        "Latency(s)",
        "Batches",
    ]
    rows = []
    for res in results:
        rows.append(
            [
                res.name,
                f"{res.metrics['strict_acc']:.3f}",
                f"{res.metrics['lenient_acc']:.3f}",
                res.prompt_tokens,
                res.generated_tokens,
                f"{res.latency:.2f}",
                res.batches,
            ]
        )
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    header_line = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    row_lines = [" | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def print_answer_table(
    questions: List[Question],
    sequential: StrategyResult,
    full_batch: StrategyResult,
    dependency: StrategyResult,
) -> None:
    headers = ["QID", "Gold", "Sequential", "Full Batch", "Parallel"]
    rows = []
    for question in questions:
        gold = "; ".join(question.references) if question.references else ""
        rows.append(
            [
                question.qid,
                gold,
                sequential.answers.get(question.qid, ""),
                full_batch.answers.get(question.qid, ""),
                dependency.answers.get(question.qid, ""),
            ]
        )
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    header_line = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    row_lines = [" | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows]
    print("Answer comparison:")
    print("\n".join([header_line, separator, *row_lines]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sequential, full-batch, and dependency-aware QA strategies using a local Qwen model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B", help="Hugging Face model identifier or local path.")
    parser.add_argument("--split", default="train", help="SQuAD split to sample.")
    parser.add_argument("--context-count", type=int, default=3, help="Number of contexts to process.")
    parser.add_argument("--min-questions", type=int, default=3, help="Minimum questions per context.")
    parser.add_argument("--max-questions", type=int, default=5, help="Maximum questions per context.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for sampling contexts.")
    parser.add_argument("--cost-weight", type=float, default=0.01, help="Cost penalty weight for dependency selection.")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum edge confidence.")
    parser.add_argument("--max-dependencies", type=int, default=3, help="Max dependencies per question.")
    parser.add_argument("--total-cost-budget", type=int, default=None, help="Optional global dependency cost budget.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max new tokens for answer generation.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument("--no-llm-deps", action="store_true", help="Force heuristic dependency generator.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to dump metrics as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    logging.info("Loading tokenizer and model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    contexts = load_squad_groups(
        args.split,
        min_questions=args.min_questions,
        max_questions=args.max_questions,
        max_contexts=args.context_count,
        seed=args.seed,
    )

    if args.no_llm_deps:
        dep_generator = HeuristicDependencyGenerator()
        logging.info("Using heuristic dependency generator.")
    else:
        dep_generator = LocalLLMDependencyGenerator(tokenizer, model)
        logging.info("Using local LLM dependency generator.")

    overall_results: Dict[str, List[StrategyResult]] = {}
    serialized_contexts: List[dict] = []

    for idx, context_payload in enumerate(contexts, start=1):
        title = context_payload.get("title", f"context-{idx}")
        background = context_payload["context"]
        questions = build_questions_from_group(context_payload)
        logging.info("Processing context %d/%d: %s", idx, len(contexts), title)
        logging.info("Background preview: %s", background[:200].replace("\n", " "))
        for q in questions:
            logging.info("  %s: %s (gold: %s)", q.qid, q.text.strip(), q.references)

        seq_res = run_sequential_strategy(
            background,
            questions,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
        )

        batch_res = run_full_batch_strategy(
            background,
            questions,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens * len(questions),
        )

        dep_res = run_dependency_strategy(
            background,
            questions,
            dep_generator,
            tokenizer,
            model,
            cost_weight=args.cost_weight,
            min_confidence=args.min_confidence,
            max_dependencies=args.max_dependencies,
            total_cost_budget=args.total_cost_budget,
            max_new_tokens=args.max_new_tokens,
        )

        overall_results[title] = [seq_res, batch_res, dep_res]
        print(f"\n=== Context: {title} ===")
        print(summarize_results([seq_res, batch_res, dep_res]))
        print_answer_table(questions, seq_res, batch_res, dep_res)
        serialized_contexts.append(
            {
                "context": title,
                "strategies": [
                    {
                        "name": res.name,
                        "metrics": res.metrics,
                        "prompt_tokens": res.prompt_tokens,
                        "generated_tokens": res.generated_tokens,
                        "latency": res.latency,
                        "batches": res.batches,
                        "answers": res.answers,
                    }
                    for res in (seq_res, batch_res, dep_res)
                ],
            }
        )

    # aggregate
    strategy_totals: Dict[str, Dict[str, float]] = {}
    for results in overall_results.values():
        for res in results:
            stats = strategy_totals.setdefault(
                res.name,
                {
                    "strict": 0.0,
                    "lenient": 0.0,
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "latency": 0.0,
                    "count": 0,
                    "batches": 0,
                },
            )
            stats["strict"] += res.metrics["strict_acc"]
            stats["lenient"] += res.metrics["lenient_acc"]
            stats["prompt_tokens"] += res.prompt_tokens
            stats["generated_tokens"] += res.generated_tokens
            stats["latency"] += res.latency
            stats["batches"] += res.batches
            stats["count"] += 1

    summary_rows = []
    if strategy_totals:
        print("\n=== Overall Averages ===")
        headers = ["Strategy", "Strict ACC", "Lenient ACC", "PromptTok", "GenTok", "Latency(s)", "Batches"]
        rows = []
        for name, stats in strategy_totals.items():
            count = stats["count"]
            rows.append(
                [
                    name,
                    f"{stats['strict'] / count:.3f}",
                    f"{stats['lenient'] / count:.3f}",
                    int(stats["prompt_tokens"] / count),
                    int(stats["generated_tokens"] / count),
                    f"{stats['latency'] / count:.2f}",
                    f"{stats['batches'] / count:.2f}",
                ]
            )
            summary_rows.append(
                {
                    "name": name,
                    "strict_acc": stats["strict"] / count,
                    "lenient_acc": stats["lenient"] / count,
                    "prompt_tokens": stats["prompt_tokens"] / count,
                    "generated_tokens": stats["generated_tokens"] / count,
                    "latency": stats["latency"] / count,
                    "batches": stats["batches"] / count,
                }
            )
        widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
        header_line = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
        separator = "-+-".join("-" * width for width in widths)
        row_lines = [" | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows]
        print("\n".join([header_line, separator, *row_lines]))

    if args.json_out:
        payload = {
            "contexts": serialized_contexts,
            "averages": summary_rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logging.info("Wrote metrics JSON to %s", args.json_out)


if __name__ == "__main__":
    main()
