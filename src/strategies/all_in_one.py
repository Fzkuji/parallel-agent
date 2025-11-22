from __future__ import annotations

import os
import re
import time
import textwrap
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_qwen_parallel as rq
from src.eval import evaluate_predictions
from src.results import StrategyResult
from src.utils import DEFAULT_GENERATION_SEED, reset_generation_seed, strip_assistant_prefix, strip_think_prefix


def run_all_in_one_strategy(
    background: str,
    questions,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    max_new_tokens: int,
) -> StrategyResult:
    question_lookup = {q.qid: q for q in questions}
    instructions = textwrap.dedent(
        """You are a helpful assistant that answers multiple questions from a single background.
- Answer the questions in the exact order given.
- For each question, output exactly: Question (QID): {answer}
- Use braces { } around the answer. If unknown, put {unknown}.
- One line per question; no extra text before or after these lines."""
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
    trimmed = []
    for token in gen_tokens[0].tolist():
        if token in (eos_id, pad_id):
            break
        trimmed.append(token)
    raw_response = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
    raw_response = strip_think_prefix(strip_assistant_prefix(raw_response))

    pattern = re.compile(r"Question\\s*\\((Q\\d+)\\):\\s*\\{([^}]*)\\}", re.IGNORECASE)
    matches = list(pattern.finditer(raw_response))
    first_hits: Dict[str, Tuple[str, str]] = {}
    for m in matches:
        qid = m.group(1)
        ans = m.group(2).strip()
        span = m.group(0).strip()
        if qid in first_hits:
            continue
        first_hits[qid] = (ans, span)
    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    detail_records: List[Dict[str, Any]] = []

    for question in questions:
        qid = question.qid
        if qid in first_hits:
            answer_text, span_text = first_hits[qid]
            strict_valid = True
        else:
            answer_text = "unknown"
            span_text = ""
            strict_valid = False
        answer_records[question.qid] = (answer_text, strict_valid)
        answers_text[question.qid] = answer_text
        detail_records.append(
            {
                "question_id": question.qid,
                "question": question.text.strip(),
                "gold_answers": question.references,
                "prompt": chat_prompt,
                "raw_response": span_text or raw_response,
                "final_answer": answer_text,
                "strict_valid": strict_valid,
                "latency": elapsed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(tokenizer(raw_response, return_tensors="pt").input_ids.shape[1]),
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup)
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
