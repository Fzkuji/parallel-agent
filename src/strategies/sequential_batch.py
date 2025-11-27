from __future__ import annotations

import time
import textwrap
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import Question
from src.inference import USE_THINK_TOKENS, build_chat_prompt, extract_box_answer
from src.evaluation import evaluate_predictions
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
    dataset: str = None,
    use_kv_cache: bool = True,
) -> StrategyResult:
    """Run sequential QA strategy with optional KV cache optimization.

    Args:
        background: Context/background text
        questions: List of questions to answer
        tokenizer: Tokenizer instance
        model: Model instance
        max_new_tokens: Max tokens to generate per question
        dataset: Dataset name for evaluation metrics
        use_kv_cache: If True, reuse KV cache across questions (faster)
    """
    question_lookup = {q.qid: q for q in questions}
    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    prompt_token_lengths: List[int] = []
    total_generated_tokens = 0
    total_latency = 0.0

    # Only squad dataset has "unknown" labels
    extractive_datasets = {"squad"}
    if dataset in extractive_datasets:
        unknown_instruction = " If the answer is unknown, return <answer>unknown</answer>."
    else:
        unknown_instruction = ""

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

    # KV cache state
    past_key_values = None
    cached_seq_len = 0

    for question in questions:
        user_message = f"Question ({question.qid}): {question.text.strip()}"
        messages.append({"role": "user", "content": user_message})

        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=(not USE_THINK_TOKENS),
        )

        full_input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to(model.device)
        full_length = full_input_ids.shape[-1]

        start = time.perf_counter()
        reset_generation_seed(DEFAULT_GENERATION_SEED)

        # Determine if we can use KV cache
        use_cache_this_turn = (
            use_kv_cache
            and past_key_values is not None
            and cached_seq_len > 0
            and cached_seq_len < full_length
        )

        with torch.no_grad():
            if use_cache_this_turn:
                # Use KV cache: only process new tokens
                new_input_ids = full_input_ids[:, cached_seq_len:]
                # Build attention mask for full sequence (cached + new)
                attention_mask = torch.ones(
                    1, cached_seq_len + new_input_ids.shape[-1], device=model.device
                )
                generated = model.generate(
                    input_ids=new_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
                # generated.sequences = new_input_ids + generated_tokens
                prompt_tokens = full_length  # Report full prompt length for metrics
                gen_start_idx = new_input_ids.shape[-1]
            else:
                # Full generation (first turn or cache miss)
                generated = model.generate(
                    input_ids=full_input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
                prompt_tokens = full_length
                gen_start_idx = full_length

        elapsed = time.perf_counter() - start

        # Extract generated tokens
        if use_cache_this_turn:
            generated_part = generated.sequences[:, gen_start_idx:]
        else:
            generated_part = generated.sequences[:, gen_start_idx:]

        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id
        trimmed_tokens: List[int] = []
        for token in generated_part[0].tolist():
            if token in (eos_id, pad_id):
                break
            trimmed_tokens.append(token)

        raw_response = tokenizer.decode(trimmed_tokens, skip_special_tokens=True).strip()
        raw_response = clean_model_text(raw_response)
        final_answer, strict_valid = extract_box_answer(raw_response)

        messages.append({"role": "assistant", "content": raw_response})

        # Update KV cache state for next iteration
        if use_kv_cache and hasattr(generated, "past_key_values") and generated.past_key_values is not None:
            past_key_values = generated.past_key_values
            # Total cached length = previous cached + new input + generated (before EOS trim)
            if use_cache_this_turn:
                cached_seq_len = cached_seq_len + generated.sequences.shape[-1]
            else:
                cached_seq_len = generated.sequences.shape[-1]

            # Verify cache alignment with next prompt
            # Build next prompt (without generation prompt) to check alignment
            next_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=(not USE_THINK_TOKENS),
            )
            next_ids = tokenizer(next_prompt, return_tensors="pt").input_ids
            expected_len = next_ids.shape[-1]

            # If mismatch, reset cache (encoding divergence detected)
            if abs(cached_seq_len - expected_len) > 5:  # Allow small tolerance
                past_key_values = None
                cached_seq_len = 0

        answer_records[question.qid] = (final_answer, strict_valid)
        answers_text[question.qid] = final_answer
        prompt_token_lengths.append(prompt_tokens)
        gen_token_count = int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1])
        total_generated_tokens += gen_token_count
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
                "generated_tokens": gen_token_count,
                "used_kv_cache": use_cache_this_turn,
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
    return StrategyResult(
        name="sequential",
        answers=answers_text,
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
    dataset: str = None,
) -> StrategyResult:
    question_lookup = {q.qid: q for q in questions}
    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    per_question: List[Dict[str, Any]] = []

    batch_chat_prompts: List[str] = []
    for question in questions:
        system_prompt, user_prompt = build_single_prompt(background, question, dataset)
        batch_chat_prompts.append(
            build_chat_prompt(
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
        box = extract_box_answer(raw_text)
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

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
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
    dataset: str = None,
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
        system_prompt, user_prompt = build_single_prompt(item["context"], q, dataset)
        batch_chat_prompts.append(
            build_chat_prompt(
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
        box = extract_box_answer(raw_text)
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
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    max_new_tokens: int,
    strategy_name: str = "sequential",
    dataset: str = None,
    use_kv_cache: bool = True,
) -> StrategyResult:
    """Run sequential multi-context QA strategy with optional KV cache optimization.

    Args:
        items: List of items with qid, question, context, references
        tokenizer: Tokenizer instance
        model: Model instance
        max_new_tokens: Max tokens to generate per question
        strategy_name: Name for the strategy result
        dataset: Dataset name for evaluation metrics
        use_kv_cache: If True, reuse KV cache across questions (faster)
    """
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

    # Only squad dataset has "unknown" labels
    extractive_datasets = {"squad"}
    if dataset in extractive_datasets:
        unknown_instruction = " If the answer is unknown, return <answer>unknown</answer>."
    else:
        unknown_instruction = ""

    system_message = (
        f"You are a helpful assistant that answers questions given background passages.\n"
        f"Provide the answer with format <answer>text</answer>.{unknown_instruction}"
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_message}]

    # KV cache state
    past_key_values = None
    cached_seq_len = 0

    for item in items:
        user_message = f"Context ({item['qid']}):\n{item['context']}\nQuestion ({item['qid']}): {item['question']}"
        messages.append({"role": "user", "content": user_message})

        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=(not USE_THINK_TOKENS),
        )

        full_input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to(model.device)
        full_length = full_input_ids.shape[-1]

        start = time.perf_counter()
        reset_generation_seed(DEFAULT_GENERATION_SEED)

        # Determine if we can use KV cache
        use_cache_this_turn = (
            use_kv_cache
            and past_key_values is not None
            and cached_seq_len > 0
            and cached_seq_len < full_length
        )

        with torch.no_grad():
            if use_cache_this_turn:
                # Use KV cache: only process new tokens
                new_input_ids = full_input_ids[:, cached_seq_len:]
                attention_mask = torch.ones(
                    1, cached_seq_len + new_input_ids.shape[-1], device=model.device
                )
                generated = model.generate(
                    input_ids=new_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
                prompt_tokens = full_length
                gen_start_idx = new_input_ids.shape[-1]
            else:
                # Full generation (first turn or cache miss)
                generated = model.generate(
                    input_ids=full_input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
                prompt_tokens = full_length
                gen_start_idx = full_length

        elapsed = time.perf_counter() - start

        # Extract generated tokens
        generated_part = generated.sequences[:, gen_start_idx:]
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id
        trimmed_tokens: List[int] = []
        for token in generated_part[0].tolist():
            if token in (eos_id, pad_id):
                break
            trimmed_tokens.append(token)

        raw_response = tokenizer.decode(trimmed_tokens, skip_special_tokens=True).strip()
        raw_response = clean_model_text(raw_response)
        final_answer, strict_valid = extract_box_answer(raw_response)

        messages.append({"role": "assistant", "content": raw_response})

        # Update KV cache state for next iteration
        if use_kv_cache and hasattr(generated, "past_key_values") and generated.past_key_values is not None:
            past_key_values = generated.past_key_values
            if use_cache_this_turn:
                cached_seq_len = cached_seq_len + generated.sequences.shape[-1]
            else:
                cached_seq_len = generated.sequences.shape[-1]

            # Verify cache alignment
            next_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=(not USE_THINK_TOKENS),
            )
            next_ids = tokenizer(next_prompt, return_tensors="pt").input_ids
            expected_len = next_ids.shape[-1]

            if abs(cached_seq_len - expected_len) > 5:
                past_key_values = None
                cached_seq_len = 0

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
                "used_kv_cache": use_cache_this_turn,
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
    return StrategyResult(
        name=strategy_name,
        answers=answers_text,
        prompt_tokens=prompt_token_lengths[-1] if prompt_token_lengths else 0,
        generated_tokens=total_generated_tokens,
        latency=total_latency,
        batches=len(items),
        metrics=metrics,
        details={"turns": detail_records},
    )
