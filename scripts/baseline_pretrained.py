#!/usr/bin/env python3
"""
Baseline evaluation script for comparing different QA strategies.

Tests the following strategies (no training required):
- all_in_one: All questions in a single prompt
- sequential: Generate answers one by one
- batch: Parallel batch generation
- collab_llm: LLM-based dependency ordering

Usage:
    python scripts/eval_baselines.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset squad \
        --eval-samples 100 \
        --strategies all_in_one,sequential,batch,collab_llm \
        --num-gpus 8
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Available strategies that don't require training
AVAILABLE_STRATEGIES = ["all_in_one", "sequential", "batch", "collab_llm"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline QA strategies")

    # Model and dataset
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="squad",
                       choices=["squad", "hotpot", "quac", "drop", "triviaqa", "quality", "cmb"])
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--min-questions", type=int, default=3)
    parser.add_argument("--max-questions", type=int, default=5)
    parser.add_argument("--fixed-question-count", type=int, default=None,
                       help="Take exactly this many questions (in order) from each context")

    # Strategies
    parser.add_argument("--strategies", type=str, default="all_in_one,sequential,batch,collab_llm",
                       help=f"Comma-separated list of strategies. Available: {', '.join(AVAILABLE_STRATEGIES)}")

    # Inference
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for batch strategy. If None, batch entire context together.")

    # Hardware
    parser.add_argument("--num-gpus", type=int, default=None, help="Override auto GPU detection")
    parser.add_argument("--min-free-mem-gb", type=float, default=10.0)

    # Paths
    parser.add_argument("--output-dir", type=str, default="outputs/eval_baselines")
    parser.add_argument("--cache", action="store_true", help="Cache results for each strategy")
    parser.add_argument("--force", action="store_true", help="Force re-run even if cached")

    # vLLM
    parser.add_argument("--enable-thinking", action="store_true")

    # Dependency detection (for collab_llm)
    parser.add_argument("--cost-weight", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--max-dependencies", type=int, default=3)
    parser.add_argument("--total-cost-budget", type=float, default=float("inf"))

    # Other
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _get_available_gpus(min_free_memory_gb: float = 10.0) -> List[int]:
    """Get list of GPUs with sufficient free memory."""
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            available = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split(',')
                gpu_idx = int(parts[0].strip())
                free_mb = float(parts[1].strip())
                free_gb = free_mb / 1024
                if free_gb >= min_free_memory_gb:
                    available.append(gpu_idx)
            if available:
                return available
    except Exception:
        pass

    try:
        import torch
        count = torch.cuda.device_count()
        return list(range(count))
    except Exception:
        return list(range(8))


def _context_to_items(context_payload: dict) -> List[dict]:
    """Convert context format to items format."""
    if "items" in context_payload:
        return context_payload["items"]

    context = context_payload["context"]
    items = []
    for q in context_payload["questions"]:
        items.append({
            "qid": q["qid"],
            "question": q["text"],
            "context": context,
            "references": q["references"],
            "answer_tokens": q.get("answer_tokens", 12),
        })
    return items


# System prompt for answer extraction
SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the given passage.
You MUST wrap your answer in <answer></answer> tags. Be concise.

Example:
Question: What color is the sky?
<answer>blue</answer>"""


def _eval_worker(
    rank: int,
    world_size: int,
    gpu_id: int,
    model_name: str,
    eval_contexts: List[Dict],
    strategies: List[str],
    output_dir: str,
    max_new_tokens: int,
    dataset: str,
    enable_thinking: bool,
    dep_args: Dict[str, Any],
    batch_size: Optional[int] = None,
):
    """Worker process for evaluation on a single GPU."""
    # Set environment variables BEFORE any CUDA imports
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[Worker {rank}] GPU {gpu_id}: Starting, {len(eval_contexts)} contexts, strategies: {strategies}", flush=True)

    # Now import CUDA-related modules
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.models import Question
    from src.evaluation import evaluate_predictions
    from src.inference import extract_answer
    from src import LocalLLMDependencyGenerator

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set left padding for decoder-only models
    tokenizer.padding_side = "left"

    # Load model using Transformers
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[Worker {rank}] Model loaded", flush=True)

    # For collab_llm, reuse the same model for dependency generation
    dep_generator = None
    if "collab_llm" in strategies:
        dep_generator = LocalLLMDependencyGenerator(tokenizer, model)
        print(f"[Worker {rank}] Dependency generator initialized", flush=True)

    # Results per strategy
    shard_results = {strategy: {"contexts": []} for strategy in strategies}

    for idx, context_payload in enumerate(eval_contexts, start=1):
        items = _context_to_items(context_payload)
        title = context_payload.get("title", f"context-{idx}")

        if idx % 10 == 0:
            print(f"[Worker {rank}] Processing {idx}/{len(eval_contexts)}", flush=True)

        # Build question lookup
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

        context = items[0]["context"]
        questions = [
            {"qid": item["qid"], "text": item["question"]}
            for item in items
        ]

        # Run each strategy
        for strategy in strategies:
            if strategy == "all_in_one":
                result = _run_all_in_one(
                    model, tokenizer, context, questions, items,
                    question_lookup, max_new_tokens, dataset, enable_thinking
                )
            elif strategy == "sequential":
                result = _run_sequential(
                    model, tokenizer, items, question_lookup,
                    max_new_tokens, dataset, enable_thinking
                )
            elif strategy == "batch":
                result = _run_batch(
                    model, tokenizer, items, question_lookup,
                    max_new_tokens, dataset, enable_thinking, batch_size
                )
            elif strategy == "collab_llm":
                result = _run_collab_llm(
                    model, tokenizer, items, question_lookup,
                    max_new_tokens, dataset, enable_thinking,
                    dep_generator, dep_args
                )
            else:
                continue

            shard_results[strategy]["contexts"].append({
                "title": title,
                "metrics": result["metrics"],
                "latency": result["latency"],
                "prompt_tokens": result["prompt_tokens"],
                "generated_tokens": result["generated_tokens"],
                "prompt_tokens_api": result.get("prompt_tokens_api", result["prompt_tokens"]),
                "generated_tokens_api": result.get("generated_tokens_api", result["generated_tokens"]),
                "num_questions": len(items),
                # Dependency generation cost (for collab_llm)
                "dep_prompt_tokens": result.get("dep_prompt_tokens", 0),
                "dep_generated_tokens": result.get("dep_generated_tokens", 0),
                "dep_latency": result.get("dep_latency", 0),
            })

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    temp_file = os.path.join(output_dir, f"eval_shard_{rank}.json")
    with open(temp_file, 'w') as f:
        json.dump(shard_results, f)

    print(f"[Worker {rank}] Done, saved to {temp_file}", flush=True)

    # Clean up models to free GPU memory
    print(f"[Worker {rank}] Cleaning up models...", flush=True)
    del model
    if dep_generator is not None:
        del dep_generator

    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        # Force synchronize to ensure all CUDA operations complete
        torch.cuda.synchronize()
    except Exception:
        pass

    print(f"[Worker {rank}] Cleanup complete, exiting", flush=True)


def _run_all_in_one(model, tokenizer, context, questions, items, question_lookup,
                    max_new_tokens, dataset, enable_thinking):
    """Run all-in-one strategy: all questions in one prompt."""
    import torch
    import re
    from src.inference import extract_answer
    from src.evaluation import evaluate_predictions

    # Build all-in-one prompt (same logic as src/strategies/all_in_one.py)
    system_prompt = (
        "You are a helpful assistant that answers multiple questions from a single background.\n"
        "Answer each question using exactly this format: QID: <answer>text</answer>\n\n"
        "Example:\n"
        "Q1: <answer>Paris</answer>\n"
        "Q2: <answer>42</answer>\n\n"
        "Rules:\n"
        "- Use the exact question ID (e.g., Q1, Q2)\n"
        "- Put answer inside <answer></answer> tags\n"
        "- Extract answers directly from the background passage\n"
        "- One answer per line, no extra text"
    )

    question_lines = [f"Question ({q['qid']}): {q['text'].strip()}" for q in questions]
    user_prompt = f"Background:\n{context.strip()}\n\nQuestions:\n" + "\n".join(question_lines)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.perf_counter() - start_time

    # Extract generated text
    prompt_tokens = inputs["input_ids"].shape[1]
    generated = outputs[0][prompt_tokens:]
    generated_tokens = generated.shape[0]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True)

    # Parse all-in-one response: Q1: <answer>text</answer>
    pattern = re.compile(r"(Q\d+)\s*:\s*<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(raw_text)
    found = {}
    for qid, ans in matches:
        if qid not in found:
            found[qid] = ans.strip()

    answer_records = {}
    for item in items:
        qid = item["qid"]
        if qid in found:
            answer = found[qid]
            strict_valid = True
        else:
            answer = ""
            strict_valid = False
        answer_records[qid] = (answer, strict_valid)

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)

    return {
        "metrics": metrics,
        "latency": latency,
        "prompt_tokens": prompt_tokens,  # Original input tokens
        "generated_tokens": generated_tokens,  # Original generated tokens
        "prompt_tokens_api": prompt_tokens,  # API tokens (same for all_in_one)
        "generated_tokens_api": generated_tokens,  # API tokens (same)
    }


def _run_sequential(model, tokenizer, items, question_lookup,
                    max_new_tokens, dataset, enable_thinking):
    """Run sequential strategy: multi-turn conversation, context only in first turn.

    Uses multi-turn chat format:
    - Turn 1: [system, user(context + Q1)] -> assistant(A1)
    - Turn 2: [system, user(context + Q1), assistant(A1), user(Q2)] -> assistant(A2)
    - ...

    This avoids repeating context in each turn's user message.
    """
    import torch
    from src.inference import extract_answer
    from src.evaluation import evaluate_predictions

    answer_records = {}
    total_latency = 0
    total_prompt_tokens_api = 0
    total_generated_tokens = 0
    deduplicated_prompt_tokens = 0

    # Build conversation history as multi-turn messages
    context = items[0]["context"] if items else ""

    # Start with system message
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Track the previous turn's prompt tokens to calculate incremental cost
    prev_prompt_tokens = 0

    for i, item in enumerate(items):
        if i == 0:
            # First turn: include context
            user_content = f"Passage:\n{context}\n\nQuestion: {item['question']}"
        else:
            # Subsequent turns: only question (context already in history)
            user_content = f"Question: {item['question']}"

        messages.append({"role": "user", "content": user_content})

        try:
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        total_latency += time.perf_counter() - start_time

        # Extract generated text
        current_prompt_tokens = inputs["input_ids"].shape[1]
        generated = outputs[0][current_prompt_tokens:]
        generated_tokens = generated.shape[0]
        raw_text = tokenizer.decode(generated, skip_special_tokens=True)

        final_answer, strict_valid = extract_answer(raw_text, dataset)
        answer_records[item["qid"]] = (final_answer, strict_valid)

        # API tokens (actual tokens sent for this turn)
        total_prompt_tokens_api += current_prompt_tokens
        total_generated_tokens += generated_tokens

        # Calculate deduplicated prompt tokens:
        # First turn: full prompt tokens
        # Subsequent turns: incremental tokens (current - previous - previous_generated)
        # This gives us: new question + template overhead (without counting repeated history)
        if i == 0:
            deduplicated_prompt_tokens += current_prompt_tokens
        else:
            # Incremental cost = current prompt - (previous prompt + previous generated)
            # This is the cost of adding the new question to the conversation
            incremental_tokens = current_prompt_tokens - prev_prompt_tokens - prev_generated_tokens
            deduplicated_prompt_tokens += incremental_tokens

        # Save for next iteration
        prev_prompt_tokens = current_prompt_tokens
        prev_generated_tokens = generated_tokens

        # Add assistant response to history for next turn
        messages.append({"role": "assistant", "content": raw_text})

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)

    return {
        "metrics": metrics,
        "latency": total_latency,
        "prompt_tokens": deduplicated_prompt_tokens,  # Deduplicated: context (once) + all questions
        "generated_tokens": total_generated_tokens,
        "prompt_tokens_api": total_prompt_tokens_api,  # API tokens (grows with conversation history)
        "generated_tokens_api": total_generated_tokens,
    }


def _run_batch(model, tokenizer, items, question_lookup,
               max_new_tokens, dataset, enable_thinking, batch_size=None):
    """Run batch strategy: questions in parallel with configurable batch size."""
    import torch
    from src.inference import extract_answer
    from src.evaluation import evaluate_predictions

    context = items[0]["context"] if items else ""

    # Build prompts for all questions
    prompts = []
    for item in items:
        prompt = f"Passage:\n{item['context']}\n\nQuestion: {item['question']}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(full_prompt)

    # Determine batch size
    if batch_size is None:
        batch_size = len(prompts)  # Process all at once (original behavior)

    # Process in batches
    answer_records = {}
    total_prompt_tokens_api = 0
    total_generated_tokens = 0
    deduplicated_prompt_tokens = 0
    total_latency = 0
    context_tokens = len(tokenizer.encode(context))

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_items = items[batch_start:batch_end]

        # Tokenize batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        total_latency += time.perf_counter() - start_time

        # Extract answers
        for i, item in enumerate(batch_items):
            # Count non-padding tokens for API cost
            prompt_tokens = inputs["attention_mask"][i].sum().item()
            total_prompt_tokens_api += prompt_tokens

            # Extract generated tokens
            generated = outputs[i][inputs["input_ids"][i].shape[0]:]
            total_generated_tokens += generated.shape[0]

            # Decode
            raw_text = tokenizer.decode(generated, skip_special_tokens=True)
            final_answer, strict_valid = extract_answer(raw_text, dataset)
            answer_records[item["qid"]] = (final_answer, strict_valid)

            # Calculate deduplicated prompt tokens
            global_idx = batch_start + i
            if global_idx == 0:
                deduplicated_prompt_tokens += prompt_tokens
            else:
                deduplicated_prompt_tokens += prompt_tokens - context_tokens

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)

    return {
        "metrics": metrics,
        "latency": total_latency,
        "prompt_tokens": deduplicated_prompt_tokens,  # Deduplicated: context (once) + all questions
        "generated_tokens": total_generated_tokens,
        "prompt_tokens_api": total_prompt_tokens_api,  # API tokens (context repeated N times)
        "generated_tokens_api": total_generated_tokens,
    }


def _run_collab_llm(model, tokenizer, items, question_lookup,
                    max_new_tokens, dataset, enable_thinking,
                    dep_generator, dep_args):
    """Run collab_llm strategy: LLM decides execution order based on dependencies."""
    import torch
    from src.inference import extract_answer
    from src.evaluation import evaluate_predictions
    from src.models import Question
    from src.scheduler import DependencyScheduler
    from src.selection import apply_dependencies, select_dependency_edges

    # Build questions with context - use these for everything to ensure consistency
    dep_questions = [
        Question(
            qid=item["qid"],
            text=item["question"],
            priority=1.0,
            answer_tokens=item.get("answer_tokens", 12),
            type_hint=None,
            references=item.get("references", []),
            context=item["context"],
        )
        for item in items
    ]
    dep_question_lookup = {q.qid: q for q in dep_questions}

    # Generate dependency edges using LLM (and track cost)
    edges = dep_generator.generate_edges("", dep_questions)
    dep_metrics = dep_generator.last_metrics
    dep_prompt_tokens = int(dep_metrics.get("prompt_tokens", 0))
    dep_generated_tokens = int(dep_metrics.get("generated_tokens", 0))
    dep_latency = dep_metrics.get("latency", 0.0)

    # Debug: print LLM-determined order (edges represent sequential dependencies)
    if edges:
        # Reconstruct order from edges: first source, then follow the chain
        order = [edges[0].source] + [e.target for e in edges]
        print(f"  [collab_llm] LLM order: {order}", flush=True)
    else:
        print(f"  [collab_llm] LLM returned no ordering, using original order", flush=True)

    # Select and apply dependencies (preserves LLM order, only filters cycles/limits)
    selected = select_dependency_edges(
        dep_question_lookup,
        edges,
        max_dependencies_per_target=dep_args["max_dependencies"],
        total_cost_budget=dep_args["total_cost_budget"],
        fmt_overhead=6,
    )
    apply_dependencies(dep_question_lookup, selected)

    # Build scheduler and get schedule
    scheduler = DependencyScheduler(
        "",  # Empty background, each question has its own context
        dep_questions,
        max_batch_tokens=None,
        fmt_overhead_per_section=6,
        prefill_token_cost=0.8,
        generate_token_cost=1.2,
    )
    scheduler.build_dependencies(auto_infer=False)
    schedule = scheduler.schedule()

    # Debug: print schedule
    print(f"  [collab_llm] Schedule: {len(schedule.batches)} batches", flush=True)

    context = items[0]["context"] if items else ""

    # Generate answers in scheduled order using multi-turn conversation
    # Each question in the schedule order becomes a turn in the conversation
    answer_records = {}
    answers_text = {}
    total_latency = 0
    total_prompt_tokens_api = 0
    total_generated_tokens = 0
    deduplicated_prompt_tokens = 0

    # Build conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    is_first_question = True

    # Track the previous turn's tokens to calculate incremental cost
    prev_prompt_tokens = 0
    prev_generated_tokens = 0

    # Flatten batches to get execution order
    execution_order = []
    for batch in schedule.batches:
        execution_order.extend(batch.question_ids)

    for qid in execution_order:
        item = next(i for i in items if i["qid"] == qid)
        question = dep_question_lookup[qid]

        if is_first_question:
            # First turn: include context
            user_content = f"Passage:\n{context}\n\nQuestion: {item['question']}"
            is_first_question = False
        else:
            # Subsequent turns: only question + dependency answers if any
            deps = sorted(question.dependencies)
            if deps and answers_text:
                dep_answers = []
                for dep_qid in deps:
                    if dep_qid in answers_text:
                        dep_item = next((i for i in items if i["qid"] == dep_qid), None)
                        if dep_item:
                            dep_answers.append(f"Q: {dep_item['question']}\nA: {answers_text[dep_qid]}")
                if dep_answers:
                    user_content = "Related Q&A:\n" + "\n\n".join(dep_answers) + f"\n\nQuestion: {item['question']}"
                else:
                    user_content = f"Question: {item['question']}"
            else:
                user_content = f"Question: {item['question']}"

        messages.append({"role": "user", "content": user_content})

        try:
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        total_latency += time.perf_counter() - start_time

        # Extract generated text
        current_prompt_tokens = inputs["input_ids"].shape[1]
        generated = outputs[0][current_prompt_tokens:]
        generated_tokens = generated.shape[0]
        raw_text = tokenizer.decode(generated, skip_special_tokens=True)

        final_answer, strict_valid = extract_answer(raw_text, dataset)
        answer_records[item["qid"]] = (final_answer, strict_valid)
        answers_text[item["qid"]] = final_answer

        # API tokens (actual tokens sent)
        total_prompt_tokens_api += current_prompt_tokens
        total_generated_tokens += generated_tokens

        # Calculate deduplicated prompt tokens:
        # First turn: full prompt tokens
        # Subsequent turns: incremental tokens (current - previous - previous_generated)
        if prev_prompt_tokens == 0:
            # First turn
            deduplicated_prompt_tokens += current_prompt_tokens
        else:
            # Incremental cost = current prompt - (previous prompt + previous generated)
            incremental_tokens = current_prompt_tokens - prev_prompt_tokens - prev_generated_tokens
            deduplicated_prompt_tokens += incremental_tokens

        # Save for next iteration
        prev_prompt_tokens = current_prompt_tokens
        prev_generated_tokens = generated_tokens

        # Add assistant response to history
        messages.append({"role": "assistant", "content": raw_text})

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)

    return {
        "metrics": metrics,
        "latency": total_latency + dep_latency,  # Include dependency generation latency
        "prompt_tokens": deduplicated_prompt_tokens,  # Deduplicated: context (once) + all questions
        "generated_tokens": total_generated_tokens,
        "prompt_tokens_api": total_prompt_tokens_api + dep_prompt_tokens,  # Include dep generation cost
        "generated_tokens_api": total_generated_tokens + dep_generated_tokens,  # Include dep generation cost
        # Separate tracking for dependency generation cost
        "dep_prompt_tokens": dep_prompt_tokens,
        "dep_generated_tokens": dep_generated_tokens,
        "dep_latency": dep_latency,
    }


def load_dataset(dataset: str, split: str, max_contexts: int, min_questions: int,
                 max_questions: int, seed: int, fixed_question_count: Optional[int] = None) -> List[Dict]:
    """Load dataset based on name."""
    if dataset == "hotpot":
        from src.datasets.hotpot import load_hotpot_groups
        return load_hotpot_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "squad":
        from src.datasets.squad import load_squad_groups
        return load_squad_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed,
            fixed_question_count=fixed_question_count
        )
    elif dataset == "quac":
        from src.datasets.quac import load_quac_groups
        return load_quac_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "drop":
        from src.datasets.drop import load_drop_groups
        return load_drop_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "triviaqa":
        from src.datasets.triviaqa import load_triviaqa_groups
        return load_triviaqa_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "quality":
        from src.datasets.quality import load_quality_groups
        return load_quality_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "cmb":
        from src.datasets.cmb import load_cmb_groups
        return load_cmb_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def aggregate_results(output_dir: Path, num_gpus: int, strategies: List[str]) -> Dict[str, Any]:
    """Aggregate results from all GPU shards."""
    results = {strategy: {"contexts": []} for strategy in strategies}

    for rank in range(num_gpus):
        shard_file = output_dir / f"eval_shard_{rank}.json"
        if shard_file.exists():
            with open(shard_file, 'r') as f:
                shard_data = json.load(f)
                for strategy in strategies:
                    if strategy in shard_data:
                        results[strategy]["contexts"].extend(shard_data[strategy]["contexts"])
            shard_file.unlink()
        else:
            logger.warning(f"Missing shard file: {shard_file}")

    # Compute aggregate metrics for each strategy
    aggregated = {}
    for strategy in strategies:
        contexts = results[strategy]["contexts"]
        if not contexts:
            continue

        total_questions = sum(ctx["num_questions"] for ctx in contexts)
        total_contexts = len(contexts)
        total_latency = sum(ctx["latency"] for ctx in contexts)
        avg_questions_per_context = total_questions / total_contexts if total_contexts else 0

        # Original tokens (unique input only, no repeated previous answers)
        total_prompt_tokens = sum(ctx["prompt_tokens"] for ctx in contexts)
        total_generated_tokens = sum(ctx["generated_tokens"] for ctx in contexts)

        # API tokens (actual tokens sent/received, includes repeated content)
        total_prompt_tokens_api = sum(ctx.get("prompt_tokens_api", ctx["prompt_tokens"]) for ctx in contexts)
        total_generated_tokens_api = sum(ctx.get("generated_tokens_api", ctx["generated_tokens"]) for ctx in contexts)

        # Dependency generation cost (for collab_llm)
        total_dep_prompt_tokens = sum(ctx.get("dep_prompt_tokens", 0) for ctx in contexts)
        total_dep_generated_tokens = sum(ctx.get("dep_generated_tokens", 0) for ctx in contexts)
        total_dep_latency = sum(ctx.get("dep_latency", 0) for ctx in contexts)

        # Weighted averages for metrics
        total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in contexts)
        total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in contexts)
        total_lenient = sum(ctx["metrics"].get("lenient_acc", 0) * ctx["num_questions"] for ctx in contexts)

        aggregated[strategy] = {
            "aggregate_metrics": {
                "strict_acc": total_em / total_questions if total_questions > 0 else 0,
                "f1": total_f1 / total_questions if total_questions > 0 else 0,
                "lenient_acc": total_lenient / total_questions if total_questions > 0 else 0,
                "avg_latency": total_latency / total_contexts if total_contexts else 0,
                "total_latency": total_latency,
                # Original tokens (unique input)
                "total_prompt_tokens": total_prompt_tokens,
                "total_generated_tokens": total_generated_tokens,
                "avg_prompt_tokens": total_prompt_tokens / total_contexts if total_contexts else 0,
                "avg_generated_tokens": total_generated_tokens / total_contexts if total_contexts else 0,
                # API tokens (actual cost)
                "total_prompt_tokens_api": total_prompt_tokens_api,
                "total_generated_tokens_api": total_generated_tokens_api,
                "avg_prompt_tokens_api": total_prompt_tokens_api / total_contexts if total_contexts else 0,
                "avg_generated_tokens_api": total_generated_tokens_api / total_contexts if total_contexts else 0,
                "num_contexts": total_contexts,
                "num_questions": total_questions,
                "avg_questions_per_context": avg_questions_per_context,
                # Dependency generation cost (for collab_llm)
                "total_dep_prompt_tokens": total_dep_prompt_tokens,
                "total_dep_generated_tokens": total_dep_generated_tokens,
                "total_dep_latency": total_dep_latency,
                "avg_dep_prompt_tokens": total_dep_prompt_tokens / total_contexts if total_contexts else 0,
                "avg_dep_generated_tokens": total_dep_generated_tokens / total_contexts if total_contexts else 0,
                "avg_dep_latency": total_dep_latency / total_contexts if total_contexts else 0,
            },
            "contexts": contexts,
        }

    return aggregated


def _get_cache_key(args) -> str:
    """Generate a cache key based on evaluation parameters."""
    # Include parameters that affect results
    key_parts = [
        args.model.replace("/", "_"),
        args.dataset,
        f"n{args.eval_samples}",
        f"q{args.min_questions}-{args.max_questions}",
        f"tok{args.max_new_tokens}",
    ]
    return "_".join(key_parts)


def _load_cached_results(cache_dir: Path, cache_key: str, strategies: List[str]) -> Dict[str, Any]:
    """Load cached results for strategies that have been evaluated before.

    Returns:
        Dictionary mapping strategy name to cached results (or empty dict if not cached)
    """
    cached = {}
    for strategy in strategies:
        cache_file = cache_dir / f"cache_{cache_key}_{strategy}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                cached[strategy] = data
                logger.info(f"  Loaded cached results for {strategy}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"  Failed to load cache for {strategy}: {e}")
    return cached


def _save_cached_results(cache_dir: Path, cache_key: str, results: Dict[str, Any], args) -> None:
    """Save results to cache files (one per strategy)."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    for strategy, data in results.items():
        cache_file = cache_dir / f"cache_{cache_key}_{strategy}.json"
        # Add metadata to cached results
        data_with_meta = {
            "config": {
                "model": args.model,
                "dataset": args.dataset,
                "eval_samples": args.eval_samples,
                "min_questions": args.min_questions,
                "max_questions": args.max_questions,
                "max_new_tokens": args.max_new_tokens,
            },
            **data,
        }
        with open(cache_file, 'w') as f:
            json.dump(data_with_meta, f, indent=2)
        logger.info(f"  Saved cache for {strategy} to {cache_file}")


def main():
    args = parse_args()

    # Parse strategies
    strategies = [s.strip() for s in args.strategies.split(',')]
    invalid_strategies = [s for s in strategies if s not in AVAILABLE_STRATEGIES]
    if invalid_strategies:
        logger.error(f"Invalid strategies: {invalid_strategies}. Available: {AVAILABLE_STRATEGIES}")
        sys.exit(1)

    logger.info(f"Strategies to evaluate: {strategies}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate cache key and check for cached results
    cache_key = _get_cache_key(args)
    logger.info(f"Cache key: {cache_key}")

    # Load cached results (unless --force)
    cached_results = {}
    strategies_to_run = strategies.copy()

    if not args.force:
        logger.info("Checking for cached results...")
        cached_results = _load_cached_results(output_dir, cache_key, strategies)
        strategies_to_run = [s for s in strategies if s not in cached_results]

        if cached_results:
            logger.info(f"Found cached results for: {list(cached_results.keys())}")
        if strategies_to_run:
            logger.info(f"Will evaluate: {strategies_to_run}")
        else:
            logger.info("All strategies already cached, using cached results")
            _print_summary(cached_results, strategies, dataset=args.dataset)
            return
    else:
        logger.info("--force specified, ignoring cache")

    # Load evaluation data
    logger.info(f"Loading evaluation data: {args.eval_samples} samples from {args.dataset}")
    if args.fixed_question_count:
        logger.info(f"Using fixed question count: {args.fixed_question_count} questions per context")
    eval_contexts = load_dataset(
        args.dataset,
        split="validation",
        max_contexts=args.eval_samples,
        min_questions=args.min_questions,
        max_questions=args.max_questions,
        seed=args.seed,
        fixed_question_count=args.fixed_question_count,
    )
    logger.info(f"Loaded {len(eval_contexts)} evaluation contexts")

    # Auto-detect GPUs
    available_gpus = _get_available_gpus(min_free_memory_gb=args.min_free_mem_gb)
    logger.info(f"Available GPUs with sufficient memory: {available_gpus}")

    if args.num_gpus is None:
        num_gpus = len(available_gpus) if available_gpus else 1
        gpu_ids = available_gpus[:num_gpus] if available_gpus else list(range(num_gpus))
    else:
        num_gpus = args.num_gpus
        gpu_ids = list(range(num_gpus))

    logger.info(f"Using {num_gpus} GPU(s): {gpu_ids}")

    # Shard data across GPUs
    shards = [[] for _ in range(num_gpus)]
    for i, ctx in enumerate(eval_contexts):
        shards[i % num_gpus].append(ctx)

    # Set spawn method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Dependency args for collab_llm
    dep_args = {
        "cost_weight": args.cost_weight,
        "min_confidence": args.min_confidence,
        "max_dependencies": args.max_dependencies,
        "total_cost_budget": args.total_cost_budget,
    }

    # Start workers (only for strategies that need to be run)
    logger.info(f"Starting {num_gpus} workers for strategies: {strategies_to_run}...")
    processes = []
    for rank in range(num_gpus):
        gpu_id = gpu_ids[rank]
        p = mp.Process(
            target=_eval_worker,
            args=(
                rank, num_gpus, gpu_id, args.model,
                shards[rank], strategies_to_run, str(output_dir),
                args.max_new_tokens, args.dataset, args.enable_thinking,
                dep_args, args.batch_size,
            )
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker {rank} on GPU {gpu_id} (PID: {p.pid})")

    # Wait for workers
    timeout = 3600  # 1 hour
    for p in processes:
        p.join(timeout=timeout)
        if p.is_alive():
            logger.warning(f"Worker {p.pid} timed out, terminating...")
            p.terminate()
            p.join(timeout=10)
            if p.is_alive():
                p.kill()

    logger.info("All workers finished, gathering results...")

    # Aggregate results for newly evaluated strategies
    new_results = aggregate_results(output_dir, num_gpus, strategies_to_run)

    # Save new results to cache
    if new_results:
        logger.info("Saving results to cache...")
        _save_cached_results(output_dir, cache_key, new_results, args)

    # Merge cached and new results
    all_results = {**cached_results, **new_results}

    if not all_results:
        raise RuntimeError("No results collected. Check worker logs.")

    # Print summary
    _print_summary(all_results, strategies, dataset=args.dataset)

    # Save summary results to JSON for easy access by other scripts
    results_file = output_dir / f"results_{args.dataset}.json"
    summary_data = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "eval_samples": args.eval_samples,
            "min_questions": args.min_questions,
            "max_questions": args.max_questions,
            "seed": args.seed,
        },
        "strategies": all_results,
    }
    with open(results_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")


def _print_summary(all_results: Dict, strategies: List[str], dataset: str = "squad"):
    """Print summary of results."""
    logger.info("\n" + "=" * 110)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 110)

    # Get sample counts from first strategy
    first_strategy = next((s for s in strategies if s in all_results), None)
    if first_strategy:
        first_metrics = all_results[first_strategy]["aggregate_metrics"]
        avg_q = first_metrics.get('avg_questions_per_context', 0)
        logger.info(f"Contexts: {first_metrics.get('num_contexts', 0)}, Questions: {first_metrics.get('num_questions', 0)}, Avg Q/Context: {avg_q:.2f}")

    # Detailed per-strategy summary
    for strategy in strategies:
        if strategy not in all_results:
            continue
        metrics = all_results[strategy]["aggregate_metrics"]
        logger.info(f"\n{strategy}:")
        logger.info(f"  EM:             {metrics['strict_acc']:.4f}")
        logger.info(f"  F1:             {metrics['f1']:.4f}")
        logger.info(f"  Lenient:        {metrics.get('lenient_acc', 0):.4f}")
        logger.info(f"  Prompt Tok:     {metrics['total_prompt_tokens']:,} (avg: {metrics.get('avg_prompt_tokens', 0):.1f})")
        logger.info(f"  Gen Tok:        {metrics['total_generated_tokens']:,} (avg: {metrics.get('avg_generated_tokens', 0):.1f})")
        logger.info(f"  PromptTok_API:  {metrics.get('total_prompt_tokens_api', 0):,} (avg: {metrics.get('avg_prompt_tokens_api', 0):.1f})")
        logger.info(f"  GenTok_API:     {metrics.get('total_generated_tokens_api', 0):,} (avg: {metrics.get('avg_generated_tokens_api', 0):.1f})")
        logger.info(f"  Latency:        {metrics.get('total_latency', 0):.2f}s (avg: {metrics['avg_latency']:.2f}s)")
        # Show dependency generation cost for collab_llm
        if metrics.get('total_dep_prompt_tokens', 0) > 0:
            logger.info(f"  Dep PromptTok:  {metrics.get('total_dep_prompt_tokens', 0):,} (avg: {metrics.get('avg_dep_prompt_tokens', 0):.1f})")
            logger.info(f"  Dep GenTok:     {metrics.get('total_dep_generated_tokens', 0):,} (avg: {metrics.get('avg_dep_generated_tokens', 0):.1f})")
            logger.info(f"  Dep Latency:    {metrics.get('total_dep_latency', 0):.2f}s (avg: {metrics.get('avg_dep_latency', 0):.2f}s)")

    # Combined comparison table
    logger.info("\n" + "=" * 160)
    logger.info("=== Results Summary ===")
    header = f"{'Strategy':<15} | {'EM':>6} | {'F1':>6} | {'Lenient':>7} | {'Q/Ctx':>5} | {'PromptTok':>10} | {'GenTok':>8} | {'PromptTok_API':>13} | {'GenTok_API':>10} | {'DepTok':>8} | {'Latency':>8}"
    separator = "-" * len(header)
    logger.info(header)
    logger.info(separator)

    for strategy in strategies:
        if strategy not in all_results:
            continue
        metrics = all_results[strategy]["aggregate_metrics"]
        # DepTok = dependency generation tokens (prompt + generated)
        dep_tok = metrics.get('avg_dep_prompt_tokens', 0) + metrics.get('avg_dep_generated_tokens', 0)
        logger.info(
            f"{strategy:<15} | "
            f"{metrics['strict_acc']:>6.3f} | "
            f"{metrics['f1']:>6.3f} | "
            f"{metrics.get('lenient_acc', 0):>7.3f} | "
            f"{metrics.get('avg_questions_per_context', 0):>5.1f} | "
            f"{metrics.get('avg_prompt_tokens', 0):>10.1f} | "
            f"{metrics.get('avg_generated_tokens', 0):>8.1f} | "
            f"{metrics.get('avg_prompt_tokens_api', 0):>13.1f} | "
            f"{metrics.get('avg_generated_tokens_api', 0):>10.1f} | "
            f"{dep_tok:>8.1f} | "
            f"{metrics['avg_latency']:>6.2f}s"
        )

    logger.info("=" * 160)


if __name__ == "__main__":
    main()
