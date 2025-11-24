from __future__ import annotations

import time
import textwrap
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_qwen_parallel as rq
from python import Question
from src.eval import evaluate_predictions
from src.prompts import build_single_prompt
from src.results import StrategyResult
from src.utils import (
    DEFAULT_GENERATION_SEED,
    reset_generation_seed,
    clean_model_text,
    strip_assistant_prefix,
    strip_think_prefix,
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
    prompt_token_lengths: List[int] = []
    total_generated_tokens = 0
    total_latency = 0.0

    # Align prompt with the single-question template used elsewhere
    system_message = (
        textwrap.dedent(
            r"""You are a helpful assistant that answers questions given background passages.
Provide the answer with format \\box{answer}. If the answer is unknown, return \\box{unknown}.

Background:
"""
        ).strip()
        + "\n"
        + background.strip()
    )

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
        generated_part = sequences[:, prompt_tokens:]
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id
        trimmed_tokens: List[int] = []
        for token in generated_part[0].tolist():
            if token in (eos_id, pad_id):
                break
            trimmed_tokens.append(token)
        raw_response = tokenizer.decode(trimmed_tokens, skip_special_tokens=True).strip()
        raw_response = clean_model_text(raw_response)
        final_answer, strict_valid = rq.extract_box_answer(raw_response)

        messages.append({"role": "assistant", "content": raw_response})

        answer_records[question.qid] = (final_answer, strict_valid)
        answers_text[question.qid] = final_answer
        prompt_token_lengths.append(prompt_tokens)
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
        # Count only the final prompt length (full history) once
        prompt_tokens=prompt_token_lengths[-1] if prompt_token_lengths else 0,
        generated_tokens=total_generated_tokens,
        latency=total_latency,
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
    boxes = []
    generated_token_counts = []
    for seq in sequences:
        tokens = []
        for token in seq[int(prompt_window) :].tolist():
            if token in (eos_id, pad_id):
                break
            tokens.append(token)
        raw_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        raw_text = strip_assistant_prefix(strip_think_prefix(raw_text))
        raw_texts.append(raw_text)
        box = rq.extract_box_answer(raw_text)
        boxes.append(box)
        generated_token_counts.append(int(tokenizer(raw_text, return_tensors="pt").input_ids.shape[1]))

    # Sum prompt lengths for all samples in the batch
    total_prompt_tokens = sum(int(length) for length in input_lengths) if input_lengths else 0
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


def run_batch_multi_strategy(
    items: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    max_new_tokens: int,
    strategy_name: str = "batch",
) -> StrategyResult:
    question_lookup = {
        item["qid"]: Question(
            qid=item["qid"],
            text=item["question"],
            priority=1.0,
            answer_tokens=item.get("answer_tokens", 12),
            type_hint=None,
            references=item.get("references", []),
        )
        for item in items
    }
    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    per_question: List[Dict[str, Any]] = []

    batch_chat_prompts: List[str] = []
    for item in items:
        q = question_lookup[item["qid"]]
        system_prompt, user_prompt = build_single_prompt(item["context"], q)
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
    boxes = []
    generated_token_counts = []
    for seq in sequences:
        tokens = []
        for token in seq[int(prompt_window) :].tolist():
            if token in (eos_id, pad_id):
                break
            tokens.append(token)
        raw_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        raw_text = clean_model_text(raw_text)
        raw_texts.append(raw_text)
        box = rq.extract_box_answer(raw_text)
        boxes.append(box)
        generated_token_counts.append(int(tokenizer(raw_text, return_tensors="pt").input_ids.shape[1]))

    total_prompt_tokens = sum(int(length) for length in input_lengths) if input_lengths else 0
    total_generated_tokens = sum(generated_token_counts)
    total_latency = elapsed

    for idx, item in enumerate(items):
        qid = item["qid"]
        final_answer, strict_valid = boxes[idx]
        answer_records[qid] = (final_answer, strict_valid)
        answers_text[qid] = final_answer
        per_question.append(
            {
                "question_id": qid,
                "question": item["question"],
                "gold_answers": item.get("references", []),
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
        name=strategy_name,
        answers=answers_text,
        prompt_tokens=int(total_prompt_tokens),
        generated_tokens=int(total_generated_tokens),
        latency=total_latency,
        batches=1,
        metrics=metrics,
        details={"questions": per_question},
    )


def run_sequential_multi_strategy(
    items: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    max_new_tokens: int,
    strategy_name: str = "sequential",
) -> StrategyResult:
    question_lookup = {
        item["qid"]: Question(
            qid=item["qid"],
            text=item["question"],
            priority=1.0,
            answer_tokens=item.get("answer_tokens", 12),
            type_hint=None,
            references=item.get("references", []),
        )
        for item in items
    }
    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    prompt_token_lengths: List[int] = []
    total_generated_tokens = 0
    total_latency = 0.0
    detail_records: List[Dict[str, Any]] = []

    system_message = textwrap.dedent(
        r"""You are a helpful assistant that answers questions given background passages.
Provide the answer with format \\box{answer}. If the answer is unknown, return \\box{unknown}."""
    ).strip()

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_message}]

    for item in items:
        user_message = f"Context ({item['qid']}):\n{item['context']}\nQuestion ({item['qid']}): {item['question']}"
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
        generated_part = sequences[:, prompt_tokens:]
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id
        trimmed_tokens: List[int] = []
        for token in generated_part[0].tolist():
            if token in (eos_id, pad_id):
                break
            trimmed_tokens.append(token)
        raw_response = tokenizer.decode(trimmed_tokens, skip_special_tokens=True).strip()
        raw_response = clean_model_text(raw_response)
        final_answer, strict_valid = rq.extract_box_answer(raw_response)

        messages.append({"role": "assistant", "content": raw_response})

        qid = item["qid"]
        answer_records[qid] = (final_answer, strict_valid)
        answers_text[qid] = final_answer
        prompt_token_lengths.append(prompt_tokens)
        gen_tokens = int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1])
        total_generated_tokens += gen_tokens
        total_latency += elapsed

        detail_records.append(
            {
                "question_id": qid,
                "question": item["question"],
                "gold_answers": item.get("references", []),
                "prompt": chat_prompt,
                "raw_response": raw_response,
                "final_answer": final_answer,
                "strict_valid": strict_valid,
                "latency": elapsed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": gen_tokens,
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup)
    return StrategyResult(
        name=strategy_name,
        answers=answers_text,
        # Count only the final prompt length (full history) once
        prompt_tokens=prompt_token_lengths[-1] if prompt_token_lengths else 0,
        generated_tokens=total_generated_tokens,
        latency=total_latency,
        batches=len(items),
        metrics=metrics,
        details={"turns": detail_records},
    )
