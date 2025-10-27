from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that answers questions given a background passage."
PLANNER_SYSTEM_PROMPT = (
    "You are an expert planner. Analyse the questions and output only a JSON object describing dependencies."
)


def build_chat_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> str:
    system = (system_prompt or "").strip()
    user = user_prompt.strip()
    prompt_parts = []
    if system:
        prompt_parts.append(f"System: {system}")
    prompt_parts.append(f"User: {user}")
    prompt_parts.append("Assistant:")
    prompt_parts.append("<think></think>")
    return "\n\n".join(prompt_parts)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from python import (
    EdgeCandidate,
    DependencyGraphGenerator,
    DependencyScheduler,
    HeuristicDependencyGenerator,
    Question,
    apply_dependencies,
    build_questions_from_group,
    load_squad_groups,
    select_dependency_edges,
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


def extract_json_answer(text: str) -> Optional[str]:
    """Attempt to extract {"answer": "..."} from model text."""
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


def decode_generated(
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
) -> str:
    gen_tokens = generated_ids[0][input_ids.shape[-1] :]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return text.strip()


def extract_json_from_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                cleaned = part[4:].strip()
                break
    def try_parse(candidate: str) -> Optional[dict]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    parsed = try_parse(cleaned)
    if parsed is not None:
        return parsed

    # attempt to find first balanced JSON object within text
    start_positions = [idx for idx, ch in enumerate(cleaned) if ch == "{"]
    for start in start_positions:
        depth = 0
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : idx + 1]
                    parsed = try_parse(candidate)
                    if parsed is not None:
                        return parsed
                    break
    raise ValueError(f"Failed to parse JSON: {cleaned}")


def build_dependency_prompt(background: str, questions: List[Question]) -> str:
    question_lines = [f"{q.qid}: {q.text.strip()}" for q in questions]
    prompt = textwrap.dedent(
        f"""
        你将看到一段背景文本以及若干针对该背景的问题。请推断在回答这些问题时是否需要引用其他问题的答案。

        以 JSON 形式列出问题之间的依赖关系。只输出以下结构，不要额外解释：
        {{
          "edges": [
            {{"source": "Q1", "target": "Q3", "confidence": 0.72}},
            ...
          ]
        }}

        规则：
        - 只能使用给定问题的 ID 作为 source/target。
        - confidence 取值 0~1 的数字。
        - 无需依赖的题目可省略。
        - 禁止引用不存在的 ID，禁止循环依赖。

        背景：
        {background.strip()}

        问题：
        {os.linesep.join(question_lines)}
        """
    ).strip()
    return prompt


class LocalLLMDependencyGenerator(DependencyGraphGenerator):
    """Use the same local Qwen model to infer dependency edges."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate_edges(
        self,
        background: str,
        questions: List[Question],
        metadata: Optional[dict] = None,
    ) -> List[EdgeCandidate]:
        prompt = build_dependency_prompt(background, questions)
        chat_prompt = build_chat_prompt(self.tokenizer, prompt, system_prompt=PLANNER_SYSTEM_PROMPT)
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = decode_generated(self.tokenizer, inputs["input_ids"], generated)
        try:
            payload = extract_json_from_text(text)
        except ValueError as exc:
            snippet = str(exc)
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            logging.warning(
                "LLM dependency generation failed to produce valid JSON (%s); falling back to heuristics.",
                snippet,
            )
            heuristic = HeuristicDependencyGenerator()
            return heuristic.generate_edges(background, questions, metadata)
        edges_data = payload.get("edges", [])
        edges: List[EdgeCandidate] = []
        for item in edges_data:
            try:
                source = item["source"]
                target = item["target"]
            except KeyError:
                continue
            confidence = float(item.get("confidence", 0.7))
            edges.append(EdgeCandidate(source=source, target=target, confidence=confidence))
        return edges


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
        "If uncertain, return {\"answer\": \"unknown\"}. Do not add extra text.",
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
    prompt_parts.append("Answer in JSON:")
    return "\n".join(prompt_parts)


def answer_question(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    *,
    max_new_tokens: int = 96,
) -> Tuple[str, int, bool, float]:
    chat_prompt = build_chat_prompt(tokenizer, prompt, system_prompt=DEFAULT_SYSTEM_PROMPT)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
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
    extracted = extract_json_answer(raw_text)
    strict_valid = extracted is not None
    answer = extracted if strict_valid else raw_text
    gen_token_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    return answer, len(gen_token_ids), strict_valid, elapsed


def evaluate_answers(predictions: Dict[str, Tuple[str, bool]], question_lookup: Dict[str, Question]) -> Dict[str, float]:
    total = len(predictions)
    if total == 0:
        return {"strict_acc": 0.0, "lenient_acc": 0.0}
    strict_sum = 0.0
    lenient_sum = 0.0
    for qid, (pred, strict_valid) in predictions.items():
        references = question_lookup[qid].references
        strict_sum += compute_contains(pred, references) if strict_valid else 0.0
        lenient_sum += compute_contains(pred, references)
    return {"strict_acc": strict_sum / total, "lenient_acc": lenient_sum / total}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel QA inference on SQuAD using local Qwen model with LLM-generated dependencies.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
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

    aggregate_strict = []
    aggregate_lenient = []

    for idx, context_payload in enumerate(contexts, start=1):
        title = context_payload.get("title", f"context-{idx}")
        background = context_payload["context"]
        logging.info("Context %d/%d: %s", idx, len(contexts), title)

        questions_list = build_questions_from_group(context_payload)
        questions_dict = {q.qid: q for q in questions_list}

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
        total_latency = 0.0
        for batch in schedule_result.batches:
            batch_latencies: List[float] = []
            for qid in batch.question_ids:
                question = questions_dict[qid]
                deps = sorted(question.dependencies)
                prompt = build_answer_prompt(background, question, answers_text, deps, questions_dict)
                ans_text, gen_len, strict_valid, elapsed = answer_question(
                    tokenizer,
                    model,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                answers[qid] = (ans_text, strict_valid)
                answers_text[qid] = ans_text
                logging.info("Answer %s: %s (len=%d tok)", qid, ans_text, gen_len)
                batch_latencies.append(elapsed)
            if batch_latencies:
                total_latency += max(batch_latencies)

        metrics = evaluate_answers(answers, questions_dict)
        aggregate_strict.append(metrics["strict_acc"])
        aggregate_lenient.append(metrics["lenient_acc"])
        logging.info("Context %s metrics -> strict ACC: %.3f | lenient ACC: %.3f", title, metrics["strict_acc"], metrics["lenient_acc"])
        logging.info("Estimated parallel latency (max per batch sum): %.2fs", total_latency)

        scheduler.pretty_print_schedule(schedule_result)
        if args.html_dir:
            args.html_dir.mkdir(parents=True, exist_ok=True)
            html_path = args.html_dir / f"{title.replace(' ', '_')}_schedule.html"
            logging.info("Writing visualisation to %s", html_path)
            from python import export_schedule_html

            export_schedule_html(scheduler, schedule_result, html_path, title=f"{title} schedule")

    if aggregate_strict:
        logging.info(
            "Overall strict ACC: %.3f | Overall lenient ACC: %.3f",
            sum(aggregate_strict) / len(aggregate_strict),
            sum(aggregate_lenient) / len(aggregate_lenient),
        )


if __name__ == "__main__":
    main()
