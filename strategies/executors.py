from __future__ import annotations

import logging
import time
import textwrap
from typing import Any, Dict, List, Optional, Tuple

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
from .prompts import build_answer_prompt
from .results import StrategyResult


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
        batch_latencies: List[float] = []
        batch_questions_data: List[Dict[str, Any]] = []

        for qid in batch.question_ids:
            question = question_lookup[qid]
            deps = sorted(question.dependencies)
            prompt = build_answer_prompt(background, question, answers_text, deps, question_lookup)
            chat_prompt = rq.build_chat_prompt(tokenizer, prompt)

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
            eos_id = tokenizer.eos_token_id
            pad_id = tokenizer.pad_token_id or eos_id

            tokens = []
            for token in sequences[0, prompt_tokens:].tolist():
                if token in (eos_id, pad_id):
                    break
                tokens.append(token)

            raw_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            final_answer, strict_valid = rq.extract_box_answer(raw_text)

            answer_records[question.qid] = (final_answer, strict_valid)
            answers_text[question.qid] = final_answer
            total_prompt_tokens += prompt_tokens
            total_generated_tokens += len(tokens)
            batch_latencies.append(elapsed)

            batch_questions_data.append(
                {
                    "question": question,
                    "dependencies": [
                        {"question_id": dep_id, "answer": answers_text.get(dep_id, "")}
                        for dep_id in deps
                    ],
                    "chat_prompt": chat_prompt,
                    "raw_text": raw_text,
                    "final_answer": final_answer,
                    "strict_valid": strict_valid,
                    "latency": elapsed,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": len(tokens),
                }
            )

        batch_latency = max(batch_latencies) if batch_latencies else 0.0
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
        batch_latency = 0.0
        batch_info: Dict[str, Any] = {
            "batch_id": batch.batch_id,
            "depth": batch.depth,
            "question_ids": batch.question_ids,
            "estimated_latency": batch.estimated_latency,
            "background_tokens": batch.background_tokens,
            "incremental_prefill_tokens": batch.incremental_prefill_tokens,
            "generation_tokens": batch.generation_tokens,
            "total_tokens": batch.total_tokens,
            "batch_latency": 0.0,
            "questions": [],
        }

        for qid in batch.question_ids:
            question = question_lookup[qid]
            deps = sorted(question.dependencies)
            prompt = build_answer_prompt(background, question, answers_text, deps, question_lookup)
            chat_prompt = rq.build_chat_prompt(tokenizer, prompt)

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
            eos_id = tokenizer.eos_token_id
            pad_id = tokenizer.pad_token_id or eos_id

            tokens = []
            for token in sequences[0, prompt_tokens:].tolist():
                if token in (eos_id, pad_id):
                    break
                tokens.append(token)
            raw_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            final_answer, strict_valid = rq.extract_box_answer(raw_text)

            answer_records[question.qid] = (final_answer, strict_valid)
            answers_text[question.qid] = final_answer
            total_prompt_tokens += prompt_tokens
            total_generated_tokens += len(tokens)
            batch_latency += elapsed

            batch_info["questions"].append(
                {
                    "question_id": question.qid,
                    "question": question.text.strip(),
                    "gold_answers": question.references,
                    "dependencies": [
                        {"question_id": dep_id, "answer": answers_text.get(dep_id, "")}
                        for dep_id in deps
                    ],
                    "prompt": chat_prompt,
                    "raw_response": raw_text,
                    "final_answer": final_answer,
                    "strict_valid": strict_valid,
                    "latency": elapsed,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": len(tokens),
                }
            )

        batch_info["batch_latency"] = batch_latency
        total_latency += batch_latency
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
        generated_part = sequences[:, prompt_tokens:]
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id
        trimmed_tokens: List[int] = []
        for token in generated_part[0].tolist():
            if token in (eos_id, pad_id):
                break
            trimmed_tokens.append(token)
        raw_response = tokenizer.decode(trimmed_tokens, skip_special_tokens=True).strip()
        final_answer, strict_valid = rq.extract_box_answer(raw_response)

        messages.append({"role": "assistant", "content": raw_response})

        answer_records[question.qid] = (final_answer, strict_valid)
        answers_text[question.qid] = final_answer
        total_prompt_tokens += prompt_tokens
        total_generated_tokens += len(trimmed_tokens)
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
                "generated_tokens": len(trimmed_tokens),
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

    system_message = textwrap.dedent(
        f"""You are a helpful assistant that answers questions given a background passage.
Provide concise reasoning if helpful, but the final line of every response must be exactly \\box{{answer}}. If the answer is unknown, return \\box{{unknown}}.

Background:
{background.strip()}
"""
    ).strip()

    for question in questions:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Question ({question.qid}): {question.text.strip()}"},
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
        generated_part = sequences[:, prompt_tokens:]
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id
        trimmed_tokens: List[int] = []
        for token in generated_part[0].tolist():
            if token in (eos_id, pad_id):
                break
            trimmed_tokens.append(token)
        raw_response = tokenizer.decode(trimmed_tokens, skip_special_tokens=True).strip()
        final_answer, strict_valid = rq.extract_box_answer(raw_response)

        answer_records[question.qid] = (final_answer, strict_valid)
        answers_text[question.qid] = final_answer
        total_prompt_tokens += prompt_tokens
        total_generated_tokens += len(trimmed_tokens)
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
                "generated_tokens": len(trimmed_tokens),
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup)
    return StrategyResult(
        name="batch_ideal",
        answers=answers_text,
        prompt_tokens=total_prompt_tokens,
        generated_tokens=total_generated_tokens,
        latency=max_latency,
        batches=len(questions),
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

    for question in questions:
        prompt = textwrap.dedent(
            f"""You are a helpful assistant that answers questions given a background passage.
Provide concise reasoning if helpful, but the final line of every response must be exactly \\box{{answer}}. If the answer is unknown, return \\box{{unknown}}.

Background:
{background.strip()}

Question ({question.qid}): {question.text.strip()}
"""
        ).strip()
        chat_prompt = rq.build_chat_prompt(tokenizer, prompt, system_prompt=None)
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
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id
        tokens = []
        for token in sequences[0, prompt_tokens:].tolist():
            if token in (eos_id, pad_id):
                break
            tokens.append(token)
        raw_response = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        final_answer, strict_valid = rq.extract_box_answer(raw_response)

        answer_records[question.qid] = (final_answer, strict_valid)
        answers_text[question.qid] = final_answer
        total_prompt_tokens += prompt_tokens
        total_generated_tokens += len(tokens)
        total_latency += elapsed

        per_question.append(
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
                "generated_tokens": len(tokens),
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup)
    return StrategyResult(
        name="batch",
        answers=answers_text,
        prompt_tokens=int(total_prompt_tokens),
        generated_tokens=int(total_generated_tokens),
        latency=total_latency,
        batches=len(questions),
        metrics=metrics,
        details={"questions": per_question},
    )
