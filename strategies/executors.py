from __future__ import annotations

import logging
import os
import random
import time
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_qwen_parallel as rq
from python import (
    DependencyScheduler,
    Question,
    apply_dependencies,
    select_dependency_edges,
)

from .eval import evaluate_predictions
from .prompts import build_dependency_prompt, build_single_prompt
from .results import StrategyResult


DEFAULT_GENERATION_SEED = 13


def _reset_generation_seed(seed: int = DEFAULT_GENERATION_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def _strip_think_prefix(text: str) -> str:
    """Remove any <think>...</think> blocks (wherever they appear) from text.

    一些模型会在生成开头或中间回显 <think>...</think>，我们在拼接时统一剔除，
    避免对后续提示造成干扰，并保证不同实现路径（batch/ideal）的文本一致性。
    """
    s = text
    # 粗暴但稳妥：反复剔除所有成对的 <think>...</think> 块
    open_tag = "<think>"
    close_tag = "</think>"
    while True:
        start = s.find(open_tag)
        if start == -1:
            break
        end = s.find(close_tag, start)
        if end == -1:
            # 没有闭合，剔除起始标记以免影响后续解析
            s = s.replace(open_tag, "")
            break
        s = s[:start] + s[end + len(close_tag):]
    # 清理残余的开闭标签（例如只回显了闭合标签）
    s = s.replace(open_tag, "").replace(close_tag, "")
    return s.strip()


def _strip_assistant_prefix(text: str) -> str:
    """Remove spurious leading 'assistant' echoes from model outputs.

    Some batched paths may cause the model to echo an initial 'assistant' token
    as plain text. Trim a single leading line like 'assistant' or 'assistant:'
    (case-insensitive) plus following whitespace. Only affects the very start
    of the response to avoid removing valid content later.
    """
    s = text.lstrip()
    lower = s.lower()
    # Strip common prefixes: 'assistant', 'assistant:', possibly followed by newlines/spaces
    if lower.startswith("assistant:\n") or lower.startswith("assistant: "):
        s = s.split(":", 1)[1].lstrip()
    elif lower.startswith("assistant\n") or lower.startswith("assistant "):
        # remove the word and following whitespace/newline
        s = s[len("assistant"):].lstrip()
    return s


def run_dependency_ideal_strategy(
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
    strategy_name: str = "parallel_ideal",
) -> StrategyResult:
    question_lookup = {q.qid: q for q in questions}
    edges = generator.generate_edges(background, questions)
    dep_metrics = getattr(generator, "last_metrics", None)
    dep_prompt_tokens = dep_metrics.get("prompt_tokens", 0.0) if isinstance(dep_metrics, dict) else 0.0
    dep_generated_tokens = dep_metrics.get("generated_tokens", 0.0) if isinstance(dep_metrics, dict) else 0.0
    dep_latency = dep_metrics.get("latency", 0.0) if isinstance(dep_metrics, dict) else 0.0
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
    total_prompt_tokens = dep_prompt_tokens
    total_generated_tokens = dep_generated_tokens
    total_latency = dep_latency

    dependency_edges_detail = [
        {
            "source": edge.source,
            "target": target,
            "confidence": edge.confidence,
            "rationale": edge.rationale,
        }
        for target, edge_list in selected.items()
        for edge in edge_list
    ]
    batch_details: List[Dict[str, Any]] = []

    for batch in schedule.batches:
        # Freeze answers within this batch to avoid cross-sample leakage
        answers_snapshot = dict(answers_text)
        batch_prompts: List[str] = []
        batch_questions: List[Question] = []
        batch_deps_rendered: List[List[Dict[str, str]]] = []

        for qid in batch.question_ids:
            question = question_lookup[qid]
            deps = sorted(question.dependencies)
            system_prompt, user_prompt = build_dependency_prompt(
                background,
                question,
                answers_snapshot,
                deps,
                question_lookup,
            )
            chat_prompt = rq.build_chat_prompt(tokenizer, user_prompt, system_prompt=system_prompt)
            batch_prompts.append(chat_prompt)
            batch_questions.append(question)
            batch_deps_rendered.append(
                [
                    {"question_id": dep_id, "answer": answers_snapshot.get(dep_id, "")}
                    for dep_id in deps
                ]
            )

        batch_latencies: List[float] = []
        batch_questions_data: List[Dict[str, Any]] = []
        new_answers: Dict[str, str] = {}

        for prompt_text, question, deps_rendered in zip(batch_prompts, batch_questions, batch_deps_rendered):
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            prompt_tokens = inputs["input_ids"].shape[-1]

            start = time.perf_counter()
            _reset_generation_seed()
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
            elapsed = time.perf_counter() - start

            sequences = generated.sequences
            eos_id = tokenizer.eos_token_id
            pad_id = tokenizer.pad_token_id or eos_id
            tokens = []
            for token in sequences[0, prompt_tokens:].tolist():
                if token in (eos_id, pad_id):
                    break
                tokens.append(token)
            raw_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            raw_text = _strip_think_prefix(raw_text)
            raw_text = _strip_assistant_prefix(raw_text)
            final_answer, strict_valid = rq.extract_box_answer(raw_text)

            answer_records[question.qid] = (final_answer, strict_valid)
            new_answers[question.qid] = final_answer
            total_prompt_tokens += prompt_tokens
            total_generated_tokens += int(tokenizer(raw_text, return_tensors="pt").input_ids.shape[1])
            batch_latencies.append(elapsed)

            batch_questions_data.append(
                {
                    "question": question,
                    "dependencies": deps_rendered,
                    "chat_prompt": prompt_text,
                    "raw_text": raw_text,
                    "final_answer": final_answer,
                    "strict_valid": strict_valid,
                    "latency": elapsed,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": len(tokens),
                }
            )

        # Only after finishing the whole batch, publish answers
        answers_text.update(new_answers)

        batch_latency = sum(batch_latencies) if batch_latencies else 0.0
        batch_details.append(
            {
                "batch_id": batch.batch_id,
                "depth": batch.depth,
                "question_ids": batch.question_ids,
                "estimated_latency": batch.estimated_latency,
                "background_tokens": batch.background_tokens,
                "incremental_prefill_tokens": batch.incremental_prefill_tokens,
                "generation_tokens": batch.generation_tokens,
                "total_tokens": batch.total_tokens,
                "batch_latency": batch_latency,
                "questions": [
                    {
                        "question_id": data["question"].qid,
                        "question": data["question"].text.strip(),
                        "dependencies": data["dependencies"],
                        "prompt": data["chat_prompt"],
                        "raw_response": data["raw_text"],
                        "final_answer": data["final_answer"],
                        "strict_valid": data["strict_valid"],
                        "latency": data["latency"],
                        "prompt_tokens": data["prompt_tokens"],
                        "generated_tokens": data["generated_tokens"],
                    }
                    for data in batch_questions_data
                ],
            }
        )

        total_latency += batch_latency

    metrics = evaluate_predictions(answer_records, question_lookup)
    return StrategyResult(
        name=strategy_name,
        answers=answers_text,
        prompt_tokens=int(total_prompt_tokens),
        generated_tokens=int(total_generated_tokens),
        latency=total_latency,
        batches=len(schedule.batches),
        metrics=metrics,
        details={
            "dependency_stage": {
                "edges": dependency_edges_detail,
                "prompt_tokens": dep_prompt_tokens,
                "generated_tokens": dep_generated_tokens,
                "latency": dep_latency,
            },
            "batches": batch_details,
        },
    )


def run_dependency_batch_strategy(
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
    strategy_name: str = "parallel",
) -> StrategyResult:
    question_lookup = {q.qid: q for q in questions}
    edges = generator.generate_edges(background, questions)
    dep_metrics = getattr(generator, "last_metrics", None)
    dep_prompt_tokens = dep_metrics.get("prompt_tokens", 0.0) if isinstance(dep_metrics, dict) else 0.0
    dep_generated_tokens = dep_metrics.get("generated_tokens", 0.0) if isinstance(dep_metrics, dict) else 0.0
    dep_latency = dep_metrics.get("latency", 0.0) if isinstance(dep_metrics, dict) else 0.0
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
    total_prompt_tokens = dep_prompt_tokens
    total_generated_tokens = dep_generated_tokens
    total_latency = dep_latency

    dependency_edges_detail = [
        {
            "source": edge.source,
            "target": target,
            "confidence": edge.confidence,
            "rationale": edge.rationale,
        }
        for target, edge_list in selected.items()
        for edge in edge_list
    ]
    batch_details: List[Dict[str, Any]] = []

    for batch in schedule.batches:
        batch_text_prompts: List[str] = []
        batch_questions: List[Question] = []
        dep_answers_per_question: List[List[Dict[str, str]]] = []

        for qid in batch.question_ids:
            question = question_lookup[qid]
            deps = sorted(question.dependencies)
            system_prompt, user_prompt = build_dependency_prompt(
                background,
                question,
                answers_text,
                deps,
                question_lookup,
            )
            chat_prompt = rq.build_chat_prompt(tokenizer, user_prompt, system_prompt=system_prompt)
            batch_text_prompts.append(chat_prompt)
            batch_questions.append(question)
            dep_answers_per_question.append(
                [
                    {"question_id": dep_id, "answer": answers_text.get(dep_id, "")}
                    for dep_id in deps
                ]
            )

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(batch_text_prompts, return_tensors="pt", padding=True).to(model.device)
        attention = inputs["attention_mask"]
        input_lengths = attention.sum(dim=1).tolist()
        prompt_window = inputs["input_ids"].shape[-1]

        start = time.perf_counter()
        _reset_generation_seed()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
        tokenizer.padding_side = original_padding_side
        elapsed = time.perf_counter() - start

        sequences = generated.sequences
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id

        raw_texts = []
        for seq in sequences:
            tokens = []
            for token in seq[int(prompt_window):].tolist():
                if token in (eos_id, pad_id):
                    break
                tokens.append(token)
            rt = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            rt = _strip_think_prefix(rt)
            rt = _strip_assistant_prefix(rt)
            raw_texts.append(rt)
        boxes = list(map(rq.extract_box_answer, raw_texts))

        # Recount generated tokens by re-tokenizing the cleaned raw text
        gen_token_counts = [
            int(tokenizer(raw_texts[idx], return_tensors="pt").input_ids.shape[1])
            for idx in range(len(batch_questions))
        ]

        batch_info: Dict[str, Any] = {
            "batch_id": batch.batch_id,
            "depth": batch.depth,
            "question_ids": batch.question_ids,
            "estimated_latency": batch.estimated_latency,
            "background_tokens": batch.background_tokens,
            "incremental_prefill_tokens": batch.incremental_prefill_tokens,
            "generation_tokens": batch.generation_tokens,
            "total_tokens": batch.total_tokens,
            "batch_latency": elapsed,
            "questions": [],
        }

        for idx, question in enumerate(batch_questions):
            final_answer, strict_valid = boxes[idx]
            answer_records[question.qid] = (final_answer, strict_valid)
            answers_text[question.qid] = final_answer
            total_prompt_tokens += int(input_lengths[idx])
            total_generated_tokens += gen_token_counts[idx]

            batch_info["questions"].append(
                {
                    "question_id": question.qid,
                    "question": question.text.strip(),
                    "gold_answers": question.references,
                    "dependencies": dep_answers_per_question[idx],
                    "prompt": batch_text_prompts[idx],
                    "raw_response": raw_texts[idx],
                    "final_answer": final_answer,
                    "strict_valid": strict_valid,
                    "latency": elapsed,
                    "prompt_tokens": int(input_lengths[idx]),
                    "generated_tokens": gen_token_counts[idx],
                }
            )

        total_latency += elapsed
        batch_details.append(batch_info)

    metrics = evaluate_predictions(answer_records, question_lookup)
    return StrategyResult(
        name=strategy_name,
        answers=answers_text,
        prompt_tokens=int(total_prompt_tokens),
        generated_tokens=int(total_generated_tokens),
        latency=total_latency,
        batches=len(schedule.batches),
        metrics=metrics,
        details={
            "dependency_stage": {
                "edges": dependency_edges_detail,
                "prompt_tokens": dep_prompt_tokens,
                "generated_tokens": dep_generated_tokens,
                "latency": dep_latency,
            },
            "batches": batch_details,
        },
    )


def run_all_in_one_strategy(
    background: str,
    questions: List[Question],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    max_new_tokens: int,
) -> StrategyResult:
    question_lookup = {q.qid: q for q in questions}
    instructions = textwrap.dedent(
        """You are a helpful assistant that answers questions given a background passage.
You will receive multiple questions at once. Provide concise reasoning for each question, but you must answer them in the same order they are listed. For every question, end with a line of the form `Question (QID): \box{answer}` using the exact QID provided below. If a question is unanswerable, respond with \box{unknown} for that question. Do not skip any question and do not combine their answers."""
    ).strip()
    question_lines = [f"Question ({q.qid}): {q.text.strip()}" for q in questions]
    user_message = textwrap.dedent(
        f"""Background:
{background.strip()}

Questions:
{os.linesep.join(question_lines)}
"""
    ).strip()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_message},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=(not rq.USE_THINK_TOKENS),
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[-1]
    start = time.perf_counter()
    _reset_generation_seed()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
    elapsed = time.perf_counter() - start
    sequences = generated.sequences
    generated_part = sequences[:, prompt_tokens:]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    trimmed_tokens: List[int] = []
    for token in generated_part[0].tolist():
        if token in (eos_id, pad_id):
            break
        trimmed_tokens.append(token)
    raw_response = tokenizer.decode(trimmed_tokens, skip_special_tokens=True).strip()
    raw_response = _strip_think_prefix(raw_response)
    raw_response = _strip_assistant_prefix(raw_response)

    matches = list(rq.BOX_PATTERN.finditer(raw_response))
    segments: List[str] = []
    prev = 0
    for match in matches:
        segments.append(raw_response[prev : match.end()].strip())
        prev = match.end()
    if len(segments) < len(questions):
        segments.extend([raw_response] * (len(questions) - len(segments)))

    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    total_generated_tokens = int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1])
    detail_records: List[Dict[str, Any]] = []

    for idx, question in enumerate(questions):
        if idx < len(matches):
            answer_text = matches[idx].group(1).strip()
            strict_valid = True
        else:
            answer_text = segments[idx] if idx < len(segments) else raw_response
            strict_valid = False
        answer_records[question.qid] = (answer_text, strict_valid)
        answers_text[question.qid] = answer_text
        detail_records.append(
            {
                "question_id": question.qid,
                "question": question.text.strip(),
                "gold_answers": question.references,
                "prompt": chat_prompt,
                "raw_response": segments[idx] if idx < len(segments) else raw_response,
                "final_answer": answer_text,
                "strict_valid": strict_valid,
                "latency": elapsed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": total_generated_tokens,
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup)
    return StrategyResult(
        name="all_in_one",
        answers=answers_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=total_generated_tokens,
        latency=elapsed,
        batches=1,
        metrics=metrics,
        details={"turns": detail_records, "raw_combined_response": raw_response},
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

    system_message = textwrap.dedent(
        f"""You are a helpful assistant that answers questions given a background passage.
You will receive multiple questions one by one. Provide concise reasoning if helpful, but the final line of every response must be exactly \\box{{answer}}. If the answer is unknown, return \\box{{unknown}}.

Background:
{background.strip()}
"""
    ).strip()

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_message}]
    detail_records: List[Dict[str, Any]] = []

    for question in questions:
        user_message = f"Question ({question.qid}): {question.text.strip()}"
        messages.append({"role": "user", "content": user_message})

        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=(not rq.USE_THINK_TOKENS),
        )

        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        start = time.perf_counter()
        _reset_generation_seed()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
        elapsed = time.perf_counter() - start
        sequences = generated.sequences
        generated_part = sequences[:, prompt_tokens:]
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id
        trimmed_tokens: List[int] = []
        for token in generated_part[0].tolist():
            if token in (eos_id, pad_id):
                break
            trimmed_tokens.append(token)
        raw_response = tokenizer.decode(trimmed_tokens, skip_special_tokens=True).strip()
        raw_response = _strip_think_prefix(raw_response)
        raw_response = _strip_assistant_prefix(raw_response)
        final_answer, strict_valid = rq.extract_box_answer(raw_response)

        messages.append({"role": "assistant", "content": raw_response})

        answer_records[question.qid] = (final_answer, strict_valid)
        answers_text[question.qid] = final_answer
        total_prompt_tokens += prompt_tokens
        # 使用清洗后的文本重新统计生成 token 数，避免 <think> 影响统计
        total_generated_tokens += int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1])
        total_latency += elapsed

        detail_records.append(
            {
                "question_id": question.qid,
                "question": question.text.strip(),
                "gold_answers": question.references,
                "prompt": chat_prompt,
                "raw_response": raw_response,
                "final_answer": final_answer,
                "strict_valid": strict_valid,
                "latency": elapsed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1]),
            }
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
        details={"turns": detail_records},
    )


def run_independent_strategy(
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
    max_latency = 0.0
    detail_records: List[Dict[str, Any]] = []

    for question in questions:
        system_prompt, user_prompt = build_single_prompt(background, question)
        chat_prompt = rq.build_chat_prompt(
            tokenizer,
            user_prompt,
            system_prompt=system_prompt,
        )

        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        start = time.perf_counter()
        _reset_generation_seed()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
        elapsed = time.perf_counter() - start
        sequences = generated.sequences
        generated_part = sequences[:, prompt_tokens:]
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id
        trimmed_tokens: List[int] = []
        for token in generated_part[0].tolist():
            if token in (eos_id, pad_id):
                break
            trimmed_tokens.append(token)
        raw_response = tokenizer.decode(trimmed_tokens, skip_special_tokens=True).strip()
        raw_response = _strip_think_prefix(raw_response)
        raw_response = _strip_assistant_prefix(raw_response)
        final_answer, strict_valid = rq.extract_box_answer(raw_response)

        answer_records[question.qid] = (final_answer, strict_valid)
        answers_text[question.qid] = final_answer
        total_prompt_tokens += prompt_tokens
        total_generated_tokens += int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1])
        max_latency = max(max_latency, elapsed)

        detail_records.append(
            {
                "question_id": question.qid,
                "question": question.text.strip(),
                "gold_answers": question.references,
                "prompt": chat_prompt,
                "raw_response": raw_response,
                "final_answer": final_answer,
                "strict_valid": strict_valid,
                "latency": elapsed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1]),
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup)
    return StrategyResult(
        name="batch_ideal",
        answers=answers_text,
        prompt_tokens=total_prompt_tokens,
        generated_tokens=total_generated_tokens,
        latency=max_latency,
        batches=1,
        metrics=metrics,
        details={"turns": detail_records},
    )


def run_full_batch_strategy(
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
    per_question: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_latency = 0.0

    batch_chat_prompts: List[str] = []
    for question in questions:
        system_prompt, user_prompt = build_single_prompt(background, question)
        batch_chat_prompts.append(
            rq.build_chat_prompt(
                tokenizer,
                user_prompt,
                system_prompt=system_prompt,
            )
        )

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(batch_chat_prompts, return_tensors="pt", padding=True).to(model.device)
    attention = inputs["attention_mask"]
    input_lengths = attention.sum(dim=1).tolist()
    prompt_window = inputs["input_ids"].shape[-1]

    start = time.perf_counter()
    _reset_generation_seed()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
    tokenizer.padding_side = original_padding_side
    elapsed = time.perf_counter() - start
    sequences = generated.sequences
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    raw_texts = []
    boxes = []
    generated_token_counts = []
    for idx, seq in enumerate(sequences):
        tokens = []
        for token in seq[int(prompt_window):].tolist():
            if token in (eos_id, pad_id):
                break
            tokens.append(token)
        raw_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        raw_text = _strip_think_prefix(raw_text)
        raw_text = _strip_assistant_prefix(raw_text)
        raw_texts.append(raw_text)
        box = rq.extract_box_answer(raw_text)
        boxes.append(box)
        generated_token_counts.append(int(tokenizer(raw_text, return_tensors="pt").input_ids.shape[1]))

    total_prompt_tokens = sum(int(length) for length in input_lengths)
    total_generated_tokens = sum(generated_token_counts)
    total_latency = elapsed

    for idx, question in enumerate(questions):
        final_answer, strict_valid = boxes[idx]
        answer_records[question.qid] = (final_answer, strict_valid)
        answers_text[question.qid] = final_answer
        per_question.append(
            {
                "question_id": question.qid,
                "question": question.text.strip(),
                "gold_answers": question.references,
                "prompt": batch_chat_prompts[idx],
                "raw_response": raw_texts[idx],
                "final_answer": final_answer,
                "strict_valid": strict_valid,
                "latency": elapsed,
                "prompt_tokens": int(input_lengths[idx]),
                "generated_tokens": generated_token_counts[idx],
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup)
    return StrategyResult(
        name="batch",
        answers=answers_text,
        prompt_tokens=int(total_prompt_tokens),
        generated_tokens=int(total_generated_tokens),
        latency=total_latency,
        batches=1,
        metrics=metrics,
        details={"questions": per_question},
    )
