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

BOX_PATTERN = re.compile(r"\\box\{([^}]*)\}")
USE_THINK_TOKENS = True


def set_think_tokens(enabled: bool) -> None:
    global USE_THINK_TOKENS
    USE_THINK_TOKENS = enabled


def build_chat_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Create a chat-style prompt with thinking disabled for Qwen3."""

    messages: List[Dict[str, str]] = []
    system = (system_prompt or "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_prompt.strip()})

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=USE_THINK_TOKENS,
        )
    else:
        # Fallback for tokenizers without chat template support
        parts = []
        if system:
            parts.append(f"System: {system}")
        parts.append(f"User: {user_prompt.strip()}")
        parts.append("Assistant:")
        prompt = "\n\n".join(parts)

        # Manually add thinking tokens for fallback case
        if USE_THINK_TOKENS:
            prompt = f"{prompt}<think></think>"

    return prompt

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


def extract_box_answer(text: str) -> Tuple[str, bool]:
    """Return the first \\box{...} content if present; otherwise fallback to raw text."""
    match = BOX_PATTERN.search(text)
    if match:
        return match.group(1).strip(), True
    return text.strip(), False


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

    # Remove thinking tokens first (handles both <think></think> and <think>...</think>)
    # This handles multi-line thinking blocks
    while True:
        think_start = cleaned.find("<think>")
        if think_start == -1:
            break
        think_end = cleaned.find("</think>", think_start)
        if think_end == -1:
            # Unclosed think tag, remove from start to end
            cleaned = cleaned[:think_start] + cleaned[think_start + 7:]
            break
        # Remove the entire <think>...</think> block
        cleaned = cleaned[:think_start] + cleaned[think_end + 8:]

    cleaned = cleaned.strip()

    # Handle markdown code blocks
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                cleaned = part[4:].strip()
                break
            elif part and not part.startswith("```"):
                # Try the first non-empty block that doesn't start with ```
                cleaned = part.strip()
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

        **重要：你的回答必须是markdown代码块格式，以```json开头，以```结尾。**

        输出格式示例：
        ```json
        {{
          "edges": [
            {{"source": "Q1", "target": "Q3", "confidence": 0.72}},
            {{"source": "Q2", "target": "Q4", "confidence": 0.85}}
          ]
        }}
        ```

        规则：
        - 只能使用给定问题的 ID 作为 source/target。
        - confidence 取值 0~1 的数字。
        - 无需依赖的题目可省略（返回空edges数组）。
        - 禁止引用不存在的 ID，禁止循环依赖。
        - 不需要额外解释或分析，只输出```json代码块。

        背景：
        {background.strip()}

        问题：
        {os.linesep.join(question_lines)}

        请直接输出```json代码块：
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
        self.last_metrics: Dict[str, float] = {"prompt_tokens": 0, "generated_tokens": 0, "latency": 0.0}

    def generate_edges(
        self,
        background: str,
        questions: List[Question],
        metadata: Optional[dict] = None,
    ) -> List[EdgeCandidate]:
        prompt = build_dependency_prompt(background, questions)
        chat_prompt = build_chat_prompt(self.tokenizer, prompt, system_prompt=PLANNER_SYSTEM_PROMPT)
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        start = time.perf_counter()
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
        elapsed = time.perf_counter() - start
        sequences = generated.sequences
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id
        tail_tokens: List[int] = []
        for token in sequences[0].tolist()[prompt_tokens:]:
            if token in (eos_id, pad_id):
                break
            tail_tokens.append(token)
        raw_text = self.tokenizer.decode(tail_tokens, skip_special_tokens=True).strip()
        text = raw_text
        gen_tokens = len(tail_tokens)
        self.last_metrics = {
            "prompt_tokens": float(prompt_tokens),
            "generated_tokens": float(gen_tokens),
            "latency": float(elapsed),
        }
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
            # retain the cost already incurred before fallback
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
            from python import export_schedule_html

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
