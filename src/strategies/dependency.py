from __future__ import annotations

import logging
import time
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
from src.eval import evaluate_predictions
from src.prompts import build_dependency_prompt
from src.results import StrategyResult
from src.utils import DEFAULT_GENERATION_SEED, reset_generation_seed, strip_assistant_prefix, strip_think_prefix


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
            reset_generation_seed(DEFAULT_GENERATION_SEED)
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
            raw_text = strip_assistant_prefix(strip_think_prefix(raw_text))
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
        reset_generation_seed(DEFAULT_GENERATION_SEED)
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
            for token in seq[int(prompt_window) :].tolist():
                if token in (eos_id, pad_id):
                    break
                tokens.append(token)
            rt = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            rt = strip_assistant_prefix(strip_think_prefix(rt))
            raw_texts.append(rt)
        boxes = list(map(rq.extract_box_answer, raw_texts))

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
