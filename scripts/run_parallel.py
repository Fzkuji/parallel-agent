"""Parallel QA inference on SQuAD using local LLM with dependency-based scheduling."""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import (
    DependencyGraphGenerator,
    DependencyScheduler,
    HeuristicDependencyGenerator,
    Question,
    apply_dependencies,
    build_questions_from_group,
    export_schedule_html,
    load_squad_groups,
    select_dependency_edges,
)
from src.inference import (
    DEFAULT_SYSTEM_PROMPT,
    LocalLLMDependencyGenerator,
    build_chat_prompt,
    extract_box_answer,
    set_think_tokens,
)


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punctuation(s: str) -> str:
        return re.sub(r"[^\w\s]", "", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    text = text.lower()
    text = remove_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)
    return text


def compute_em(prediction: str, references: List[str]) -> float:
    pred_norm = normalize_answer(prediction)
    for ref in references:
        if normalize_answer(ref) == pred_norm:
            return 1.0
    return 0.0


def compute_contains(prediction: str, references: List[str]) -> float:
    pred_norm = normalize_answer(prediction)
    for ref in references:
        ref_norm = normalize_answer(ref)
        if not ref_norm:
            continue
        if ref_norm in pred_norm or pred_norm in ref_norm:
            return 1.0
    return 0.0


def compute_f1(prediction: str, references: List[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best_f1 = 0.0
    for ref in references:
        ref_tokens = normalize_answer(ref).split()
        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            continue
        common_count = sum(min(pred_tokens.count(token), ref_tokens.count(token)) for token in common)
        precision = common_count / len(pred_tokens)
        recall = common_count / len(ref_tokens) if ref_tokens else 0.0
        if precision + recall == 0.0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def build_answer_prompt(
    background: str,
    question: Question,
    answers: Dict[str, str],
    dependencies: List[str],
    question_lookup: Dict[str, Question],
) -> str:
    prompt_parts = [
        "You are a helpful assistant that answers questions given background passages.",
        "You may reason step by step, but the final answer must appear exactly once as \\box{...}.",
        "If the answer is unknown, output \\box{unknown}. Do not omit the box.",
        "Place the \\box{...} on the last line by itself. Example: \\box{42}",
        "Omitting the final \\box leads to an incorrect answer.",
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
    prompt_parts.append("After reasoning, finish with a single line containing \\box{answer} and nothing after it.")
    return "\n".join(prompt_parts)


def answer_question(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    *,
    max_new_tokens: int = 96,
) -> Tuple[str, int, int, bool, float]:
    chat_prompt = build_chat_prompt(tokenizer, prompt, system_prompt=DEFAULT_SYSTEM_PROMPT)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[-1]
    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
    elapsed = time.perf_counter() - start
    sequences = generated.sequences
    generated_part = sequences[:, inputs["input_ids"].shape[-1] :]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    trimmed_tokens: List[int] = []
    for token in generated_part[0].tolist():
        if token in (eos_id, pad_id):
            break
        trimmed_tokens.append(token)
    raw_text = tokenizer.decode(trimmed_tokens, skip_special_tokens=True).strip()
    answer, strict_valid = extract_box_answer(raw_text)
    gen_tokens = len(trimmed_tokens)
    return answer, prompt_tokens, gen_tokens, strict_valid, elapsed


def evaluate_answers(
    predictions: Dict[str, Tuple[str, bool]], question_lookup: Dict[str, Question]
) -> Dict[str, float]:
    total = len(predictions)
    if total == 0:
        return {"strict_sum": 0.0, "lenient_sum": 0.0, "f1_sum": 0.0, "total": 0}
    strict_sum = 0.0
    lenient_sum = 0.0
    f1_sum = 0.0
    for qid, (pred, strict_valid) in predictions.items():
        references = question_lookup[qid].references
        if strict_valid:
            strict_sum += compute_em(pred, references)
        lenient_sum += compute_contains(pred, references)
        f1_sum += compute_f1(pred, references)
    return {
        "strict_sum": strict_sum,
        "lenient_sum": lenient_sum,
        "f1_sum": f1_sum,
        "total": total,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel QA inference on SQuAD using local LLM model with LLM-generated dependencies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B", help="Local Hugging Face model identifier or path.")
    parser.add_argument("--split", default="train", help="SQuAD split to sample.")
    parser.add_argument("--context-count", type=int, default=1, help="Number of contexts to process.")
    parser.add_argument("--min-questions", type=int, default=3, help="Minimum questions per context.")
    parser.add_argument("--max-questions", type=int, default=5, help="Maximum questions per context.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for context sampling.")
    parser.add_argument("--cost-weight", type=float, default=0.01, help="Cost penalty weight for dependency selection.")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum confidence to retain edges.")
    parser.add_argument("--max-dependencies", type=int, default=3, help="Maximum dependencies per question.")
    parser.add_argument("--total-cost-budget", type=int, default=None, help="Optional global dependency cost budget.")
    parser.add_argument("--html-dir", type=Path, default=None, help="Output directory for HTML visualisations.")
    parser.add_argument("--no-llm-deps", action="store_true", help="Disable LLM dependency generation (use heuristics).")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max new tokens for answer generation.")
    parser.add_argument(
        "--no-think-tokens",
        action="store_true",
        help="Disable <think></think> markers in prompts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    set_think_tokens(not args.no_think_tokens)
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
        generator: DependencyGraphGenerator = HeuristicDependencyGenerator()
        logging.info("Using heuristic dependency generator.")
    else:
        generator = LocalLLMDependencyGenerator(tokenizer, model)
        logging.info("Using local LLM dependency generator.")

    total_strict_sum = 0.0
    total_lenient_sum = 0.0
    total_f1_sum = 0.0
    total_questions = 0

    for idx, context_payload in enumerate(contexts, start=1):
        title = context_payload.get("title", f"context-{idx}")
        background = context_payload["context"]
        logging.info("Context %d/%d: %s", idx, len(contexts), title)

        questions_list = build_questions_from_group(context_payload)
        questions_dict = {q.qid: q for q in questions_list}
        for q in questions_list:
            logging.info(
                "  %s: %s (gold: %s)",
                q.qid,
                q.text.strip(),
                "; ".join(q.references) if q.references else "n/a",
            )

        edges = generator.generate_edges(background, questions_list, metadata=context_payload)
        selected = select_dependency_edges(
            questions_dict,
            edges,
            cost_weight=args.cost_weight,
            min_confidence=args.min_confidence,
            max_dependencies_per_target=args.max_dependencies,
            total_cost_budget=args.total_cost_budget,
            fmt_overhead=6,
        )
        apply_dependencies(questions_dict, selected)

        dep_metrics = getattr(generator, "last_metrics", None)
        dep_prompt_tokens = dep_metrics.get("prompt_tokens", 0.0) if isinstance(dep_metrics, dict) else 0.0
        dep_generated_tokens = dep_metrics.get("generated_tokens", 0.0) if isinstance(dep_metrics, dict) else 0.0
        dep_latency = dep_metrics.get("latency", 0.0) if isinstance(dep_metrics, dict) else 0.0

        scheduler = DependencyScheduler(
            background,
            questions_list,
            max_batch_tokens=None,
            fmt_overhead_per_section=6,
            prefill_token_cost=0.8,
            generate_token_cost=1.2,
        )
        scheduler.build_dependencies(auto_infer=False)
        schedule_result = scheduler.schedule()

        logging.info("Dependency edges selected:")
        for q in questions_list:
            logging.info("  %s -> %s", q.qid, sorted(q.dependencies))

        answers: Dict[str, Tuple[str, bool]] = {}
        answers_text: Dict[str, str] = {}
        total_latency = dep_latency
        total_prompt_tokens = dep_prompt_tokens
        total_generated_tokens = dep_generated_tokens
        for batch in schedule_result.batches:
            batch_latencies: List[float] = []
            for qid in batch.question_ids:
                question = questions_dict[qid]
                deps = sorted(question.dependencies)
                prompt = build_answer_prompt(background, question, answers_text, deps, questions_dict)
                ans_text, prompt_len, gen_len, strict_valid, elapsed = answer_question(
                    tokenizer,
                    model,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                answers[qid] = (ans_text, strict_valid)
                answers_text[qid] = ans_text
                logging.info("Answer %s: %s (len=%d tok)", qid, ans_text, gen_len)
                batch_latencies.append(elapsed)
                total_prompt_tokens += prompt_len
                total_generated_tokens += gen_len
            if batch_latencies:
                total_latency += max(batch_latencies)

        metrics = evaluate_answers(answers, questions_dict)
        total_strict_sum += metrics["strict_sum"]
        total_lenient_sum += metrics["lenient_sum"]
        total_f1_sum += metrics["f1_sum"]
        total_questions += metrics["total"]

        if metrics["total"] > 0:
            strict_acc = metrics["strict_sum"] / metrics["total"]
            f1 = metrics["f1_sum"] / metrics["total"]
            lenient_acc = metrics["lenient_sum"] / metrics["total"]
        else:
            strict_acc, f1, lenient_acc = 0.0, 0.0, 0.0

        logging.info(
            "Context %s metrics -> EM: %.3f | F1: %.3f | lenient ACC: %.3f",
            title,
            strict_acc,
            f1,
            lenient_acc,
        )
        logging.info("Estimated parallel latency (max per batch sum): %.2fs", total_latency)
        logging.info("Total tokens -> prompt %.0f, generated %.0f", total_prompt_tokens, total_generated_tokens)
        if dep_prompt_tokens or dep_generated_tokens:
            logging.info(
                "Dependency stage -> prompt tokens %.0f, generated tokens %.0f, latency %.2fs",
                dep_prompt_tokens,
                dep_generated_tokens,
                dep_latency,
            )

        scheduler.pretty_print_schedule(schedule_result)
        if args.html_dir:
            args.html_dir.mkdir(parents=True, exist_ok=True)
            html_path = args.html_dir / f"{title.replace(' ', '_')}_schedule.html"
            logging.info("Writing visualisation to %s", html_path)
            export_schedule_html(scheduler, schedule_result, html_path, title=f"{title} schedule")

    if total_questions > 0:
        logging.info(
            "Overall EM: %.3f | Overall F1: %.3f | Overall lenient ACC: %.3f",
            total_strict_sum / total_questions,
            total_f1_sum / total_questions,
            total_lenient_sum / total_questions,
        )


if __name__ == "__main__":
    main()
