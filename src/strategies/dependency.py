from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.models import Question, StrategyResult
from src.templates import build_chat_prompt
from src.inference import extract_answer
from src.scheduler import DependencyScheduler
from src.selection import apply_dependencies, select_dependency_edges
from src.evaluation import evaluate_predictions
from src.prompts import build_dependency_prompt
from src.utils import DEFAULT_GENERATION_SEED, reset_generation_seed, clean_model_text

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.api_client import APIClient


def run_dependency_batch_strategy(
    background: str,
    questions: List[Question],
    generator,
    tokenizer: "AutoTokenizer",
    model: "AutoModelForCausalLM",
    *,
    cost_weight: float,
    min_confidence: float,
    max_dependencies: int,
    total_cost_budget: Optional[int],
    max_new_tokens: int,
    strategy_name: str = "parallel",
    dataset: str = None,
    api_client: Optional["APIClient"] = None,
) -> StrategyResult:
    question_lookup = {q.qid: q for q in questions}
    # Per requirement: build dependency edges using only the questions (ignore background passages)
    edges = generator.generate_edges("", questions)
    dep_metrics = getattr(generator, "last_metrics", None)
    dep_prompt_tokens = dep_metrics.get("prompt_tokens", 0.0) if isinstance(dep_metrics, dict) else 0.0
    dep_generated_tokens = dep_metrics.get("generated_tokens", 0.0) if isinstance(dep_metrics, dict) else 0.0
    dep_latency = dep_metrics.get("latency", 0.0) if isinstance(dep_metrics, dict) else 0.0
    # DAG inference details (prompt and raw response from LLM)
    dag_prompt = dep_metrics.get("dag_prompt", "") if isinstance(dep_metrics, dict) else ""
    dag_raw_response = dep_metrics.get("dag_raw_response", "") if isinstance(dep_metrics, dict) else ""
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
    # Prompt tokens will count one max prompt per batch to avoid double-counting shared background
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
        batch_messages: List[List[Dict[str, str]]] = []
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
                dataset,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            batch_messages.append(messages)
            if tokenizer is not None:
                chat_prompt = build_chat_prompt(tokenizer, user_prompt, system_prompt=system_prompt)
            else:
                chat_prompt = str(messages)
            batch_text_prompts.append(chat_prompt)
            batch_questions.append(question)
            dep_answers_per_question.append(
                [
                    {"question_id": dep_id, "answer": answers_text.get(dep_id, "")}
                    for dep_id in deps
                ]
            )

        # Use API or local model for generation
        if api_client is not None:
            # API mode: process each question sequentially with retry logic
            raw_texts = []
            gen_token_counts = []
            input_lengths = []
            elapsed = 0.0
            max_retries = 3
            retry_delay = 2.0  # seconds between retries

            for messages in batch_messages:
                response = None
                current_delay = retry_delay  # Reset delay for each question
                for attempt in range(max_retries):
                    start = time.perf_counter()
                    response = api_client.generate(messages, max_tokens=max_new_tokens)
                    elapsed += time.perf_counter() - start

                    # Check if response is valid (non-empty)
                    if response.text and response.text.strip():
                        break
                    elif attempt < max_retries - 1:
                        logging.warning(
                            "API returned empty response, retrying (%d/%d) after %.1fs delay...",
                            attempt + 1, max_retries, current_delay
                        )
                        time.sleep(current_delay)
                        current_delay *= 1.5  # Exponential backoff
                    else:
                        logging.warning(
                            "API returned empty response after %d retries", max_retries
                        )

                raw_text = clean_model_text(response.text) if response else ""
                raw_texts.append(raw_text)
                gen_token_counts.append(response.completion_tokens if response else 0)
                input_lengths.append(response.prompt_tokens if response else 0)

            boxes = [extract_answer(text, dataset) for text in raw_texts]
        else:
            import torch
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
                rt = clean_model_text(rt)
                raw_texts.append(rt)
            boxes = [extract_answer(text, dataset) for text in raw_texts]

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
            # keep prompt tokens per batch as the max across questions
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

        if input_lengths:
            total_prompt_tokens += sum(int(length) for length in input_lengths)
        total_latency += elapsed
        batch_details.append(batch_info)

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
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
                "dag_prompt": dag_prompt,
                "dag_raw_response": dag_raw_response,
            },
            "batches": batch_details,
        },
    )
