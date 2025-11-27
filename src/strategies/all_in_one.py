from __future__ import annotations

import os
import re
import time
import textwrap
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import Question
from src.inference import USE_THINK_TOKENS
from src.evaluation import evaluate_predictions
from src.results import StrategyResult
from src.utils import (
    DEFAULT_GENERATION_SEED,
    reset_generation_seed,
    strip_assistant_prefix,
    strip_think_prefix,
    clean_model_text,
)


# Only squad dataset has "unknown" labels
EXTRACTIVE_DATASETS = {"squad"}

# Datasets where answers should be extracted from context (not freely generated)
EXTRACTIVE_QA_DATASETS = {"squad", "quac", "hotpot"}


def run_all_in_one_strategy(
    background: str,
    questions,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    max_new_tokens: int,
    dataset: str = None,
) -> StrategyResult:
    question_lookup = {q.qid: q for q in questions}

    # Only extractive QA datasets allow "unknown" responses
    if dataset in EXTRACTIVE_DATASETS:
        unknown_rule = "- If unknown, use <answer>unknown</answer>\n"
    else:
        unknown_rule = ""

    # Extractive QA datasets should extract answers from context
    if dataset in EXTRACTIVE_QA_DATASETS:
        extract_rule = "- Extract answers directly from the background passage\n"
    else:
        extract_rule = ""

    instructions = (
        f"You are a helpful assistant that answers multiple questions from a single background.\n"
        f"Answer each question using exactly this format: QID: <answer>text</answer>\n\n"
        f"Example:\n"
        f"Q1: <answer>Paris</answer>\n"
        f"Q2: <answer>42</answer>\n\n"
        f"Rules:\n"
        f"- Use the exact question ID (e.g., Q1, Q2)\n"
        f"- Put answer inside <answer></answer> tags\n"
        f"{extract_rule}"
        f"{unknown_rule}"
        f"- One answer per line, no extra text"
    )
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
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    gen_tokens = sequences[:, prompt_tokens:]
    trimmed: List[int] = []
    for token in gen_tokens[0].tolist():
        if token in (eos_id, pad_id):
            break
        trimmed.append(token)
    raw_response = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
    raw_response = clean_model_text(raw_response)

    # Match format: Q1: <answer>text</answer>
    pattern = re.compile(r"(Q\d+)\s*:\s*<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(raw_response)
    found = {}
    for qid, ans in matches:
        if qid not in found:
            found[qid] = ans.strip()

    answers_text: Dict[str, str] = {}
    answer_records: Dict[str, Tuple[str, bool]] = {}
    detail_records: List[Dict[str, Any]] = []

    for question in questions:
        qid = question.qid
        if qid in found:
            ans_text = found[qid]
            strict_valid = True
        else:
            # If not found, mark as unknown instead of using fallback
            ans_text = "unknown"
            strict_valid = False
        answers_text[qid] = ans_text
        answer_records[qid] = (ans_text, strict_valid)
        detail_records.append(
            {
                "question_id": qid,
                "question": question.text.strip(),
                "gold_answers": question.references,
                "prompt": chat_prompt,
                "raw_response": raw_response,
                "final_answer": ans_text,
                "strict_valid": strict_valid,
                "latency": elapsed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1]),
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
    return StrategyResult(
        name="all_in_one",
        answers=answers_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1]),
        latency=elapsed,
        batches=1,
        metrics=metrics,
        details={"turns": detail_records, "raw_combined_response": raw_response},
    )


def run_all_in_one_multi_strategy(
    items,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    max_new_tokens: int,
    strategy_name: str = "all_in_one",
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
    answers_text: Dict[str, str] = {}
    answer_records: Dict[str, Tuple[str, bool]] = {}
    detail_records: List[Dict[str, Any]] = []

    # Only extractive QA datasets allow "unknown" responses
    if dataset in EXTRACTIVE_DATASETS:
        unknown_rule = "- If unknown, use <answer>unknown</answer>\n"
    else:
        unknown_rule = ""

    # Extractive QA datasets should extract answers from context
    if dataset in EXTRACTIVE_QA_DATASETS:
        extract_rule = "- Extract answers directly from the context\n"
    else:
        extract_rule = ""

    instructions = (
        f"You are a helpful assistant that answers multiple questions.\n"
        f"Answer each question using exactly this format: QID: <answer>text</answer>\n\n"
        f"Example:\n"
        f"Q1: <answer>Paris</answer>\n"
        f"Q2: <answer>42</answer>\n\n"
        f"Rules:\n"
        f"- Use the exact question ID (e.g., Q1, Q2)\n"
        f"- Put answer inside <answer></answer> tags\n"
        f"{extract_rule}"
        f"{unknown_rule}"
        f"- One answer per line, no extra text"
    )

    blocks = []
    for item in items:
        blocks.append(f"Context ({item['qid']}):\n{item['context']}\nQuestion ({item['qid']}): {item['question']}")
    user_message = "\n\n".join(blocks)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_message},
    ]
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
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    gen_tokens = sequences[:, prompt_tokens:]
    trimmed: List[int] = []
    for token in gen_tokens[0].tolist():
        if token in (eos_id, pad_id):
            break
        trimmed.append(token)
    raw_response = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
    raw_response = clean_model_text(raw_response)

    # Match format: Q1: <answer>text</answer>
    pattern = re.compile(r"(Q\d+)\s*:\s*<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(raw_response)

    for item in items:
        qid = item["qid"]
        ans = ""
        strict_valid = False
        for m_qid, m_ans in matches:
            if m_qid == qid:
                ans = m_ans.strip()
                strict_valid = True
                break
        if not strict_valid:
            # If not found, mark as unknown instead of using fallback
            ans = "unknown"
        answers_text[qid] = ans
        answer_records[qid] = (ans, strict_valid)
        detail_records.append(
            {
                "question_id": qid,
                "question": item["question"],
                "gold_answers": item.get("references", []),
                "prompt": chat_prompt,
                "raw_response": raw_response,
                "final_answer": ans,
                "strict_valid": strict_valid,
                "latency": elapsed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1]),
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
    return StrategyResult(
        name=strategy_name,
        answers=answers_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1]),
        latency=elapsed,
        batches=1,
        metrics=metrics,
        details={"turns": detail_records, "raw_combined_response": raw_response},
    )
