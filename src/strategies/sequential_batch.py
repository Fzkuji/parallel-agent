from __future__ import annotations

import logging
import time
import textwrap
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.models import Question, StrategyResult
from src.inference import USE_THINK_TOKENS, build_chat_prompt, extract_answer
from src.evaluation import evaluate_predictions
from src.prompts import (
    build_single_prompt,
    DIRECT_ANSWER_DATASETS,
    MULTIPLE_CHOICE_DATASETS,
)
from src.utils import (
    DEFAULT_GENERATION_SEED,
    reset_generation_seed,
    clean_model_text,
    strip_assistant_prefix,
    strip_think_prefix,
)

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.api_client import APIClient


def run_sequential_strategy(
    background: str,
    questions: List[Question],
    tokenizer: "AutoTokenizer",
    model: "AutoModelForCausalLM",
    *,
    max_new_tokens: int,
    dataset: str = None,
    api_client: Optional["APIClient"] = None,
) -> StrategyResult:
    question_lookup = {q.qid: q for q in questions}
    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    prompt_token_lengths: List[int] = []
    total_generated_tokens = 0
    total_latency = 0.0

    # CMB uses direct answer format without <answer> tags
    use_direct_format = dataset in DIRECT_ANSWER_DATASETS

    # Multiple-choice questions: respond with option letter only
    use_mc_format = dataset in MULTIPLE_CHOICE_DATASETS

    # Align prompt with the single-question template used elsewhere
    # Only squad dataset has "unknown" labels
    extractive_datasets = {"squad"}
    if dataset in extractive_datasets:
        unknown_instruction = " If the answer is unknown, return <answer>unknown</answer>."
    else:
        unknown_instruction = ""

    if use_mc_format:
        system_message = (
            f"你是一个医学考试助手。请根据题目和选项，直接回答正确选项的字母（如A、B、C、D、E）。\n\n"
            f"背景信息:\n{background.strip()}"
        )
    elif use_direct_format:
        system_message = (
            textwrap.dedent(
                f"""You are a helpful medical assistant that answers questions given background passages.
Provide the answer directly without any special formatting.

Background:
"""
            ).strip()
            + "\n"
            + background.strip()
        )
    else:
        system_message = (
            textwrap.dedent(
                f"""You are a helpful assistant that answers questions given background passages.
Provide the answer with format <answer>text</answer>.{unknown_instruction}

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

        # Use API or local model for generation
        if api_client is not None:
            max_retries = 3
            retry_delay = 2.0
            response = None
            elapsed = 0.0
            for attempt in range(max_retries):
                start = time.perf_counter()
                response = api_client.generate(messages, max_tokens=max_new_tokens)
                elapsed += time.perf_counter() - start
                if response.text and response.text.strip():
                    break
                elif attempt < max_retries - 1:
                    logging.warning(
                        "API returned empty response, retrying (%d/%d) after %.1fs delay...",
                        attempt + 1, max_retries, retry_delay
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    logging.warning("API returned empty response after %d retries", max_retries)
            raw_response = clean_model_text(response.text) if response else ""
            prompt_tokens = response.prompt_tokens if response else 0
            gen_tokens = response.completion_tokens if response else 0
            chat_prompt = str(messages)  # For logging
        else:
            import torch
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(not USE_THINK_TOKENS),
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
            gen_tokens = int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1])

        final_answer, strict_valid = extract_answer(raw_response, dataset)
        messages.append({"role": "assistant", "content": raw_response})

        answer_records[question.qid] = (final_answer, strict_valid)
        answers_text[question.qid] = final_answer
        prompt_token_lengths.append(prompt_tokens)
        total_generated_tokens += gen_tokens
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
                "generated_tokens": gen_tokens,
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
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
    tokenizer: "AutoTokenizer",
    model: "AutoModelForCausalLM",
    *,
    max_new_tokens: int,
    dataset: str = None,
    api_client: Optional["APIClient"] = None,
    strategy_name: str = "batch",
) -> StrategyResult:
    question_lookup = {q.qid: q for q in questions}
    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    per_question: List[Dict[str, Any]] = []

    # Build prompts for all questions
    batch_messages: List[List[Dict[str, str]]] = []
    batch_chat_prompts: List[str] = []
    for question in questions:
        system_prompt, user_prompt = build_single_prompt(background, question, dataset)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        batch_messages.append(messages)
        if tokenizer is not None:
            batch_chat_prompts.append(
                build_chat_prompt(
                    tokenizer,
                    user_prompt,
                    system_prompt=system_prompt,
                )
            )
        else:
            batch_chat_prompts.append(str(messages))

    # Use API or local model for generation
    if api_client is not None:
        # API mode: process each question sequentially with retry logic
        raw_texts = []
        boxes = []
        generated_token_counts = []
        input_lengths = []
        total_latency = 0.0
        max_retries = 3
        retry_delay = 2.0

        for messages in batch_messages:
            response = None
            current_delay = retry_delay
            for attempt in range(max_retries):
                start = time.perf_counter()
                response = api_client.generate(messages, max_tokens=max_new_tokens)
                elapsed = time.perf_counter() - start
                total_latency += elapsed
                if response.text and response.text.strip():
                    break
                elif attempt < max_retries - 1:
                    logging.warning(
                        "API returned empty response, retrying (%d/%d) after %.1fs delay...",
                        attempt + 1, max_retries, current_delay
                    )
                    time.sleep(current_delay)
                    current_delay *= 1.5
                else:
                    logging.warning("API returned empty response after %d retries", max_retries)

            raw_text = clean_model_text(response.text) if response else ""
            raw_texts.append(raw_text)
            boxes.append(extract_answer(raw_text, dataset))
            generated_token_counts.append(response.completion_tokens if response else 0)
            input_lengths.append(response.prompt_tokens if response else 0)

        total_prompt_tokens = sum(input_lengths)
        total_generated_tokens = sum(generated_token_counts)
    else:
        import torch
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
        total_latency = time.perf_counter() - start
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
            box = extract_answer(raw_text, dataset)
            boxes.append(box)
            generated_token_counts.append(int(tokenizer(raw_text, return_tensors="pt").input_ids.shape[1]))

        total_prompt_tokens = sum(int(length) for length in input_lengths) if input_lengths else 0
        total_generated_tokens = sum(generated_token_counts)

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
                "latency": total_latency / len(questions),
                "prompt_tokens": int(input_lengths[idx]),
                "generated_tokens": generated_token_counts[idx],
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
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


def run_batch_multi_strategy(
    items: List[Dict[str, Any]],
    tokenizer: "AutoTokenizer",
    model: "AutoModelForCausalLM",
    *,
    max_new_tokens: int,
    strategy_name: str = "batch",
    dataset: str = None,
    api_client: Optional["APIClient"] = None,
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

    # Build prompts for all items
    batch_messages: List[List[Dict[str, str]]] = []
    batch_chat_prompts: List[str] = []
    for item in items:
        q = question_lookup[item["qid"]]
        system_prompt, user_prompt = build_single_prompt(item["context"], q, dataset)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        batch_messages.append(messages)
        if tokenizer is not None:
            batch_chat_prompts.append(
                build_chat_prompt(
                    tokenizer,
                    user_prompt,
                    system_prompt=system_prompt,
                )
            )
        else:
            batch_chat_prompts.append(str(messages))

    # Use API or local model for generation
    if api_client is not None:
        # API mode: process each item sequentially with retry logic
        raw_texts = []
        boxes = []
        generated_token_counts = []
        input_lengths = []
        total_latency = 0.0
        max_retries = 3
        retry_delay = 2.0

        for messages in batch_messages:
            response = None
            current_delay = retry_delay
            for attempt in range(max_retries):
                start = time.perf_counter()
                response = api_client.generate(messages, max_tokens=max_new_tokens)
                elapsed = time.perf_counter() - start
                total_latency += elapsed
                if response.text and response.text.strip():
                    break
                elif attempt < max_retries - 1:
                    logging.warning(
                        "API returned empty response, retrying (%d/%d) after %.1fs delay...",
                        attempt + 1, max_retries, current_delay
                    )
                    time.sleep(current_delay)
                    current_delay *= 1.5
                else:
                    logging.warning("API returned empty response after %d retries", max_retries)

            raw_text = clean_model_text(response.text) if response else ""
            raw_texts.append(raw_text)
            boxes.append(extract_answer(raw_text, dataset))
            generated_token_counts.append(response.completion_tokens if response else 0)
            input_lengths.append(response.prompt_tokens if response else 0)

        total_prompt_tokens = sum(input_lengths)
        total_generated_tokens = sum(generated_token_counts)
    else:
        import torch
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
        total_latency = time.perf_counter() - start
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
            box = extract_answer(raw_text, dataset)
            boxes.append(box)
            generated_token_counts.append(int(tokenizer(raw_text, return_tensors="pt").input_ids.shape[1]))

        total_prompt_tokens = sum(int(length) for length in input_lengths) if input_lengths else 0
        total_generated_tokens = sum(generated_token_counts)

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
                "latency": total_latency / len(items),  # Average latency per question
                "prompt_tokens": int(input_lengths[idx]),
                "generated_tokens": generated_token_counts[idx],
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
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
    tokenizer: "AutoTokenizer",
    model: "AutoModelForCausalLM",
    *,
    max_new_tokens: int,
    strategy_name: str = "sequential",
    dataset: str = None,
    api_client: Optional["APIClient"] = None,
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

    # CMB uses direct answer format without <answer> tags
    use_direct_format = dataset in DIRECT_ANSWER_DATASETS

    # Multiple-choice questions: respond with option letter only
    use_mc_format = dataset in MULTIPLE_CHOICE_DATASETS

    # Only squad dataset has "unknown" labels
    extractive_datasets = {"squad"}
    if dataset in extractive_datasets:
        unknown_instruction = " If the answer is unknown, return <answer>unknown</answer>."
    else:
        unknown_instruction = ""

    if use_mc_format:
        system_message = (
            f"你是一个医学考试助手。请根据题目和选项，直接回答正确选项的字母（如A、B、C、D、E）。"
        )
    elif use_direct_format:
        system_message = (
            f"You are a helpful medical assistant that answers questions given background passages.\n"
            f"Provide the answer directly without any special formatting."
        )
    else:
        system_message = (
            f"You are a helpful assistant that answers questions given background passages.\n"
            f"Provide the answer with format <answer>text</answer>.{unknown_instruction}"
        )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_message}]

    for item in items:
        user_message = f"Context ({item['qid']}):\n{item['context']}\nQuestion ({item['qid']}): {item['question']}"
        messages.append({"role": "user", "content": user_message})

        # Use API or local model for generation
        if api_client is not None:
            max_retries = 3
            retry_delay = 2.0
            response = None
            elapsed = 0.0
            for attempt in range(max_retries):
                start = time.perf_counter()
                response = api_client.generate(messages, max_tokens=max_new_tokens)
                elapsed += time.perf_counter() - start
                if response.text and response.text.strip():
                    break
                elif attempt < max_retries - 1:
                    logging.warning(
                        "API returned empty response, retrying (%d/%d) after %.1fs delay...",
                        attempt + 1, max_retries, retry_delay
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    logging.warning("API returned empty response after %d retries", max_retries)
            raw_response = clean_model_text(response.text) if response else ""
            prompt_tokens = response.prompt_tokens if response else 0
            gen_tokens = response.completion_tokens if response else 0
            chat_prompt = str(messages)  # For logging
        else:
            import torch
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(not USE_THINK_TOKENS),
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
            gen_tokens = int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1])

        final_answer, strict_valid = extract_answer(raw_response, dataset)
        messages.append({"role": "assistant", "content": raw_response})

        qid = item["qid"]
        answer_records[qid] = (final_answer, strict_valid)
        answers_text[qid] = final_answer
        prompt_token_lengths.append(prompt_tokens)
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

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
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
