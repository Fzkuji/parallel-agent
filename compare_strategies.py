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
from typing import Any, Dict, List, Optional, Tuple

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
import run_qwen_parallel as rq


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


def compute_em(prediction: str, references: List[str]) -> float:
    pred_norm = normalize_answer(prediction)
    for ref in references:
        if normalize_answer(ref) == pred_norm:
            return 1.0
    return 0.0


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
) -> Tuple[str, str, int, int, float, bool]:
    chat_prompt = rq.build_chat_prompt(tokenizer, prompt, system_prompt=rq.DEFAULT_SYSTEM_PROMPT)
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

    tokens = []
    for token in generated_part[0].tolist():
        if token in (eos_id, pad_id):
            break
        tokens.append(token)

    raw_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    output_text, strict_valid = rq.extract_box_answer(raw_text)
    generated_tokens = len(tokens)
    return output_text, raw_text, prompt_tokens, generated_tokens, elapsed, strict_valid


def build_answer_prompt(
    background: str,
    question: Question,
    answers: Dict[str, str],
    dependencies: List[str],
    question_lookup: Dict[str, Question],
) -> str:
    prompt_parts = [
        "You are a helpful assistant that answers questions given a background passage.",
        "You may reason freely, but give the final answer in the format \\box{...}. Example: \\box{42}",
        "If the answer is unknown, write \\box{unknown}.",
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
    return "\n".join(prompt_parts)


def evaluate_predictions(predictions: Dict[str, Tuple[str, bool]], lookup: Dict[str, Question]) -> Dict[str, float]:
    total = len(predictions)
    if total == 0:
        return {"strict_acc": 0.0, "lenient_acc": 0.0, "f1": 0.0}
    strict_sum = 0.0
    lenient_sum = 0.0
    f1_sum = 0.0
    for qid, (pred, strict_valid) in predictions.items():
        refs = lookup[qid].references
        if strict_valid:
            strict_sum += compute_em(pred, refs)
        lenient_sum += compute_contains(pred, refs)
        f1_sum += compute_f1(pred, refs)
    return {"strict_acc": strict_sum / total, "lenient_acc": lenient_sum / total, "f1": f1_sum / total}


def build_batch_prompt(background: str, questions: List[Question]) -> str:
    question_lines = []
    for question in questions:
        question_lines.append(f"Q{question.qid[1:]}: {question.text.strip()}")
    prompt = textwrap.dedent(
        f"""
        You are a helpful assistant. Read the background and answer each question.
        You may reason for each question, but finish its answer on a separate line in the form:
        A1: \\box{{final answer}}
        Do not add any text after the \\box. If the answer is unknown, use \\box{{unknown}}.

        Example:
        A1 reasoning...
        A1: \\box{{final value}}

        Background:
        {background.strip()}

        Questions:
        {'; '.join(question_lines)}

        Provide the answers (each prefixed by A1/A2/etc and ending with \\box{{...}}):
        """
    ).strip()
    return prompt


def parse_batch_answers(text: str, questions: List[Question]) -> Dict[str, Tuple[str, bool]]:
    answers: Dict[str, Tuple[str, bool]] = {}
    box_pattern = re.compile(r"\\box\{([^}]*)\}")
    matches = list(box_pattern.finditer(text))
    for idx, question in enumerate(questions):
        if idx < len(matches):
            answers[question.qid] = (matches[idx].group(1).strip(), True)
        else:
            answers[question.qid] = ("", False)
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
    details: Dict[str, Any]


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
        batch_text_prompts: List[str] = []
        batch_questions: List[Question] = []
        dep_answers_per_question: List[List[Dict[str, str]]] = []

        for qid in batch.question_ids:
            question = question_lookup[qid]
            deps = sorted(question.dependencies)
            prompt = build_answer_prompt(background, question, answers_text, deps, question_lookup)
            batch_text_prompts.append(prompt)
            batch_questions.append(question)
            dep_answers_per_question.append(
                [
                    {"question_id": dep_id, "answer": answers_text.get(dep_id, "")}
                    for dep_id in deps
                ]
            )

        batch_chat_prompts = [rq.build_chat_prompt(tokenizer, prompt) for prompt in batch_text_prompts]

        # Set left padding for batch generation
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        inputs = tokenizer(batch_chat_prompts, return_tensors="pt", padding=True).to(model.device)
        attention = inputs["attention_mask"]
        input_lengths = attention.sum(dim=1).tolist()

        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )

        # Restore original padding side
        tokenizer.padding_side = original_padding_side

        elapsed = time.perf_counter() - start

        sequences = generated.sequences
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id

        raw_texts = []
        for seq, input_len in zip(sequences, input_lengths):
            tokens = []
            for token in seq[int(input_len):].tolist():
                if token in (eos_id, pad_id):
                    break
                tokens.append(token)
            raw_texts.append(tokenizer.decode(tokens, skip_special_tokens=True).strip())
        boxes = list(map(rq.extract_box_answer, raw_texts))
        gen_token_counts = [
            sum(1 for token in sequences[idx, int(input_lengths[idx]):].tolist() if token not in (eos_id, pad_id))
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
            batch_latencies.append(elapsed)

            # Debug logging
            logging.debug(f"[DEPENDENCY] {question.qid}: {question.text.strip()}")
            logging.debug(f"  Gold: {question.references}")
            logging.debug(f"  Raw response: {raw_texts[idx][:200]}")
            logging.debug(f"  Final answer: {final_answer}")
            logging.debug(f"  Valid: {strict_valid}")

            batch_info["questions"].append(
                {
                    "question_id": question.qid,
                    "question": question.text.strip(),
                    "gold_answers": question.references,
                    "dependencies": dep_answers_per_question[idx],
                    "prompt": batch_chat_prompts[idx],
                    "raw_response": raw_texts[idx],
                    "final_answer": final_answer,
                    "strict_valid": strict_valid,
                    "latency": elapsed,
                    "prompt_tokens": int(input_lengths[idx]),
                    "generated_tokens": gen_token_counts[idx],
                }
            )

        if batch_latencies:
            total_latency += max(batch_latencies)
        batch_details.append(batch_info)

    metrics = evaluate_predictions(answer_records, question_lookup)
    return StrategyResult(
        name="dependency_parallel",
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

    # Initialize conversation history with system message
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
        # Add user question to conversation history
        user_message = f"Question ({question.qid}): {question.text.strip()}"
        messages.append({"role": "user", "content": user_message})

        # Generate prompt using chat template
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=rq.USE_THINK_TOKENS,
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

        # Update conversation history with assistant response
        messages.append({"role": "assistant", "content": raw_response})

        answer_records[question.qid] = (final_answer, strict_valid)
        answers_text[question.qid] = final_answer
        total_prompt_tokens += prompt_tokens
        total_generated_tokens += len(trimmed_tokens)
        total_latency += elapsed

        # Debug logging
        logging.debug(f"[SEQUENTIAL] {question.qid}: {question.text.strip()}")
        logging.debug(f"  Gold: {question.references}")
        logging.debug(f"  Raw response: {raw_response[:200]}")
        logging.debug(f"  Final answer: {final_answer}")
        logging.debug(f"  Valid: {strict_valid}")

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
    """
    Theoretical optimal parallel bound: Each question is inferred independently on separate machines.
    Assumes infinite parallelism (unlimited machines), so total latency = max individual latency.
    This establishes the best-case baseline for comparison:
    - vs Sequential: shows maximum parallelization benefit
    - vs Full_batch: shows overhead of single-GPU batch inference (padding, synchronization)
    - Should have same answer quality as full_batch (both are independent inference)
    """
    question_lookup = {q.qid: q for q in questions}
    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    total_prompt_tokens = 0
    total_generated_tokens = 0
    max_latency = 0.0  # Track the longest individual inference time
    detail_records: List[Dict[str, Any]] = []

    # System message template (same for each question)
    system_message = textwrap.dedent(
        f"""You are a helpful assistant that answers questions given a background passage.
Provide concise reasoning if helpful, but the final line of every response must be exactly \\box{{answer}}. If the answer is unknown, return \\box{{unknown}}.

Background:
{background.strip()}
"""
    ).strip()

    for question in questions:
        # Each question gets a fresh conversation with no shared history
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Question ({question.qid}): {question.text.strip()}"},
        ]

        # Generate prompt using chat template
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=rq.USE_THINK_TOKENS,
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
        max_latency = max(max_latency, elapsed)  # Track maximum latency (parallel upper bound)

        # Debug logging
        logging.debug(f"[INDEPENDENT] {question.qid}: {question.text.strip()}")
        logging.debug(f"  Gold: {question.references}")
        logging.debug(f"  Raw response: {raw_response[:200]}")
        logging.debug(f"  Final answer: {final_answer}")
        logging.debug(f"  Valid: {strict_valid}")

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
        name="independent",
        answers=answers_text,
        prompt_tokens=total_prompt_tokens,
        generated_tokens=total_generated_tokens,
        latency=max_latency,  # Use max latency to simulate perfect parallelization
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

    # Use the same system message structure as Independent strategy
    system_message = textwrap.dedent(
        f"""You are a helpful assistant that answers questions given a background passage.
Provide concise reasoning if helpful, but the final line of every response must be exactly \\box{{answer}}. If the answer is unknown, return \\box{{unknown}}.

Background:
{background.strip()}
"""
    ).strip()

    # Build chat prompts using the same structure as Independent
    chat_prompts = []
    for question in questions:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Question ({question.qid}): {question.text.strip()}"},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=rq.USE_THINK_TOKENS,
        )
        chat_prompts.append(chat_prompt)

    # Set left padding for batch generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(model.device)
    attention = inputs["attention_mask"]
    input_lengths = attention.sum(dim=1).tolist()

    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )

    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    elapsed = time.perf_counter() - start
    sequences = generated.sequences
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    per_question: List[Dict[str, Any]] = []
    total_prompt_tokens = sum(int(length) for length in input_lengths)
    total_generated_tokens = 0

    raw_texts = []
    boxes = []
    generated_token_counts = []
    for seq, input_len in zip(sequences, input_lengths):
        tokens = []
        for token in seq[int(input_len):].tolist():
            if token in (eos_id, pad_id):
                break
            tokens.append(token)
        
        raw_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        raw_texts.append(raw_text)
        
        box = rq.extract_box_answer(raw_text)
        boxes.append(box)
        
        generated_token_counts.append(len(tokens))

    total_generated_tokens = sum(generated_token_counts)

    answers_text = {
        question.qid: final_answer for question, (final_answer, _) in zip(questions, boxes)
    }
    answer_records = {
        question.qid: (final_answer, strict_valid)
        for question, (final_answer, strict_valid) in zip(questions, boxes)
    }

    # Debug logging
    for idx, question in enumerate(questions):
        logging.debug(f"[FULL_BATCH] {question.qid}: {question.text.strip()}")
        logging.debug(f"  Gold: {question.references}")
        logging.debug(f"  Raw response: {raw_texts[idx][:200]}")
        logging.debug(f"  Final answer: {boxes[idx][0]}")
        logging.debug(f"  Valid: {boxes[idx][1]}")

    per_question = [
        {
            "question_id": question.qid,
            "question": question.text.strip(),
            "gold_answers": question.references,
            "prompt": chat_prompts[idx],
            "raw_response": raw_texts[idx],
            "final_answer": boxes[idx][0],
            "strict_valid": boxes[idx][1],
            "prompt_tokens": int(input_lengths[idx]),
            "generated_tokens": generated_token_counts[idx],
        }
        for idx, question in enumerate(questions)
    ]

    metrics = evaluate_predictions(answer_records, question_lookup)
    return StrategyResult(
        name="full_batch",
        answers=answers_text,
        prompt_tokens=int(total_prompt_tokens),
        generated_tokens=int(total_generated_tokens),
        latency=elapsed,
        batches=1,
        metrics=metrics,
        details={"questions": per_question},
    )


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
    rows = []
    for res in results:
        rows.append(
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
        )
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    header_line = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    row_lines = [" | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def print_answer_table(
    questions: List[Question],
    sequential: StrategyResult,
    independent: StrategyResult,
    full_batch: StrategyResult,
    dependency: StrategyResult,
) -> None:
    headers = ["QID", "Question", "Gold", "Sequential", "Independent", "Full Batch", "Parallel"]
    rows = []
    max_answer_len = 40
    max_question_len = 60

    for question in questions:
        gold = "; ".join(question.references) if question.references else ""
        if len(gold) > max_answer_len:
            gold = gold[:max_answer_len-3] + "..."

        question_text = question.text.strip()
        if len(question_text) > max_question_len:
            question_text = question_text[:max_question_len-3] + "..."

        seq_ans = sequential.answers.get(question.qid, "")
        indep_ans = independent.answers.get(question.qid, "")
        batch_ans = full_batch.answers.get(question.qid, "")
        dep_ans = dependency.answers.get(question.qid, "")

        # Add markers for correct/incorrect
        def mark_answer(ans, gold_refs):
            if not ans:
                return "∅"
            norm_ans = normalize_answer(ans)
            for ref in gold_refs:
                if normalize_answer(ref) == norm_ans:
                    return f"✓ {ans[:max_answer_len]}"
            return f"✗ {ans[:max_answer_len]}"

        rows.append(
            [
                question.qid,
                question_text,
                gold,
                mark_answer(seq_ans, question.references),
                mark_answer(indep_ans, question.references),
                mark_answer(batch_ans, question.references),
                mark_answer(dep_ans, question.references),
            ]
        )

    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    header_line = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    row_lines = [" | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows]
    print("\nAnswer comparison (✓ = correct, ✗ = incorrect, ∅ = empty):")
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
    parser.add_argument(
        "--no-think-tokens",
        action="store_true",
        help="Disable <think></think> markers in prompts.",
    )
    parser.add_argument(
        "--verbose-debug",
        action="store_true",
        help="Print detailed prompts and responses for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rq.set_think_tokens(not args.no_think_tokens)
    log_level = logging.DEBUG if args.verbose_debug else getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")

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

    logging.info("Loaded %d contexts (requested: %d)", len(contexts), args.context_count)

    if args.no_llm_deps:
        dep_generator = HeuristicDependencyGenerator()
        logging.info("Using heuristic dependency generator.")
    else:
        dep_generator = rq.LocalLLMDependencyGenerator(tokenizer, model)
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

        indep_res = run_independent_strategy(
            background,
            questions,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,  # Each question gets the same per-question budget
        )

        batch_res = run_full_batch_strategy(
            background,
            questions,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,  # Each question gets the same budget in batch
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

        overall_results[title] = [seq_res, indep_res, batch_res, dep_res]
        print(f"\n=== Context: {title} ===")
        print(summarize_results([seq_res, indep_res, batch_res, dep_res]))
        print_answer_table(questions, seq_res, indep_res, batch_res, dep_res)
        # Add gold answers for comparison
        gold_answers = {q.qid: q.references for q in questions}

        serialized_contexts.append(
            {
                "context": title,
                "gold_answers": gold_answers,
                "questions_text": {q.qid: q.text.strip() for q in questions},
                "strategies": [
                    {
                        "name": res.name,
                        "metrics": res.metrics,
                        "prompt_tokens": res.prompt_tokens,
                        "generated_tokens": res.generated_tokens,
                        "latency": res.latency,
                        "batches": res.batches,
                        "answers": res.answers,
                        "details": res.details,
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
                    "f1": 0.0,
                    "lenient": 0.0,
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "latency": 0.0,
                    "count": 0,
                    "batches": 0,
                },
            )
            stats["strict"] += res.metrics["strict_acc"]
            stats["f1"] += res.metrics["f1"]
            stats["lenient"] += res.metrics["lenient_acc"]
            stats["prompt_tokens"] += res.prompt_tokens
            stats["generated_tokens"] += res.generated_tokens
            stats["latency"] += res.latency
            stats["batches"] += res.batches
            stats["count"] += 1

    summary_rows = []
    if strategy_totals:
        print("\n=== Overall Averages ===")
        headers = ["Strategy", "EM", "F1", "Lenient ACC", "PromptTok", "GenTok", "Latency(s)", "Batches"]
        rows = []
        for name, stats in strategy_totals.items():
            count = stats["count"]
            rows.append(
                [
                    name,
                    f"{stats['strict'] / count:.3f}",
                    f"{stats['f1'] / count:.3f}",
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
                    "f1": stats["f1"] / count,
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
        logging.info("Saving %d contexts to JSON", len(serialized_contexts))
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logging.info("Wrote metrics JSON to %s", args.json_out)


if __name__ == "__main__":
    main()
