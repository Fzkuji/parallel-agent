"""
vLLM-based parallel baseline inference.

This module provides multi-GPU parallel inference using vLLM with multiprocessing.
Each GPU runs a separate worker process with its own vLLM instance.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Simple prompt format (same as exp2a_shared_context.py)
SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the given passage.
Give a short, direct answer. Do not explain or elaborate."""


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


def _baseline_worker(
    rank: int,
    world_size: int,
    gpu_id: int,
    model_name: str,
    eval_contexts: List[Dict],
    output_dir: str,
    max_new_tokens: int,
    dataset: str,
    enable_thinking: bool,
):
    """Worker process for vLLM baseline inference on a single GPU.

    IMPORTANT: This runs in a separate process with isolated CUDA context.
    Environment variables must be set BEFORE any CUDA imports.
    """
    # IMPORTANT: Set environment variables BEFORE importing anything CUDA-related
    # This is exactly the same as exp1_answer_dependency.py
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Disable vLLM V1 engine which spawns EngineCore processes
    os.environ["VLLM_USE_V1"] = "0"
    # Prevent vLLM from trying to use distributed
    os.environ["VLLM_DISABLE_FRONTEND_MULTIPROCESSING"] = "1"
    # Disable vLLM progress bar
    os.environ["VLLM_NO_PROGRESS_BAR"] = "1"
    # Disable tqdm globally
    os.environ["TQDM_DISABLE"] = "1"
    # Suppress vLLM verbose logging
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

    print(f"[Worker {rank}] GPU {gpu_id}: Starting, {len(eval_contexts)} contexts", flush=True)

    # Now import CUDA-related modules
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from src.models import Question
    from src.evaluation import evaluate_predictions
    from src.inference import extract_answer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load vLLM model (same as utils.py LLMClient)
    vllm_model = LLM(
        model=model_name,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="half",
        gpu_memory_utilization=0.9,
        disable_log_stats=True,
    )
    print(f"[Worker {rank}] Model loaded", flush=True)

    # Run inference
    shard_results = {"contexts": []}
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)

    for idx, context_payload in enumerate(eval_contexts, start=1):
        items = _context_to_items(context_payload)
        title = context_payload.get("title", f"context-{idx}")

        if idx % 10 == 0:
            print(f"[Worker {rank}] Processing {idx}/{len(eval_contexts)}", flush=True)

        # Build prompts
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

        # Generate
        start_time = time.perf_counter()
        outputs = vllm_model.generate(prompts, sampling_params, use_tqdm=False)
        latency = time.perf_counter() - start_time

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

        # Process results
        answer_records = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for i, item in enumerate(items):
            output = outputs[i]
            raw_text = output.outputs[0].text
            final_answer, strict_valid = extract_answer(raw_text, dataset)
            answer_records[item["qid"]] = (final_answer, strict_valid)
            total_prompt_tokens += len(output.prompt_token_ids)
            total_completion_tokens += len(output.outputs[0].token_ids)

        metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)

        shard_results["contexts"].append({
            "title": title,
            "metrics": metrics,
            "latency": latency,
            "prompt_tokens": total_prompt_tokens,
            "generated_tokens": total_completion_tokens,
            "num_questions": len(items),
        })

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    temp_file = os.path.join(output_dir, f"baseline_shard_{rank}.json")
    with open(temp_file, 'w') as f:
        json.dump(shard_results, f)

    print(f"[Worker {rank}] Done, saved to {temp_file}", flush=True)


def run_vllm_baseline(
    model_name: str,
    eval_contexts: List[Dict],
    output_dir: Path,
    max_new_tokens: int = 96,
    dataset: str = "squad",
    enable_thinking: bool = False,
    cache_baseline: bool = True,
    force: bool = False,
    num_gpus: Optional[int] = None,
) -> Dict[str, Any]:
    """Run vLLM baseline inference in parallel across multiple GPUs.

    Args:
        model_name: HuggingFace model name
        eval_contexts: List of evaluation contexts
        output_dir: Directory to save results
        max_new_tokens: Maximum tokens to generate
        dataset: Dataset name for evaluation
        enable_thinking: Enable thinking mode for Qwen3
        cache_baseline: Whether to cache results
        force: Force re-run even if cached
        num_gpus: Number of GPUs to use (auto-detect if None)

    Returns:
        Dictionary with aggregate metrics and per-context results
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "baseline_results.json"

    # Auto-detect GPUs
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_gpus = max(1, num_gpus)

    # Check cache
    if cache_path.exists() and cache_baseline and not force:
        logger.info(f"Loading cached baseline from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    # Delete old cache if force
    if force and cache_path.exists():
        logger.info("--force specified, removing cached baseline")
        cache_path.unlink()

    # Clean up old shard files
    for shard_file in output_dir.glob("baseline_shard_*.json"):
        shard_file.unlink()

    logger.info(f"Running vLLM baseline on {num_gpus} GPU(s) with {len(eval_contexts)} contexts...")

    # Shard data across GPUs
    shards = [[] for _ in range(num_gpus)]
    for i, ctx in enumerate(eval_contexts):
        shards[i % num_gpus].append(ctx)

    # Set spawn method for multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Start all workers (same as exp1_answer_dependency.py)
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=_baseline_worker,
            args=(
                rank, num_gpus, rank, model_name,
                shards[rank], str(output_dir),
                max_new_tokens, dataset, enable_thinking,
            )
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker {rank} on GPU {rank} (PID: {p.pid})")

    # Wait for all workers
    timeout = 3600  # 1 hour max
    for p in processes:
        p.join(timeout=timeout)
        if p.is_alive():
            logger.warning(f"Worker {p.pid} timed out, terminating...")
            p.terminate()
            p.join(timeout=10)
            if p.is_alive():
                p.kill()

    logger.info("All workers finished, gathering results...")

    # Gather results from all shards
    all_contexts = []
    for rank in range(num_gpus):
        shard_file = output_dir / f"baseline_shard_{rank}.json"
        if shard_file.exists():
            with open(shard_file, 'r') as f:
                shard_data = json.load(f)
                all_contexts.extend(shard_data["contexts"])
            shard_file.unlink()
        else:
            logger.warning(f"Missing shard file: {shard_file}")

    if not all_contexts:
        raise RuntimeError("No baseline results collected. Check worker logs.")

    # Aggregate metrics
    total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in all_contexts)
    total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in all_contexts)
    total_questions = sum(ctx["num_questions"] for ctx in all_contexts)
    total_latency = sum(ctx["latency"] for ctx in all_contexts)

    baseline_results = {
        "aggregate_metrics": {
            "strict_acc": total_em / total_questions if total_questions > 0 else 0,
            "f1": total_f1 / total_questions if total_questions > 0 else 0,
            "avg_latency": total_latency / len(all_contexts) if all_contexts else 0,
            "total_prompt_tokens": sum(ctx["prompt_tokens"] for ctx in all_contexts),
            "total_generated_tokens": sum(ctx["generated_tokens"] for ctx in all_contexts),
        },
        "contexts": all_contexts,
    }

    # Save cache
    if cache_baseline:
        with open(cache_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        logger.info(f"Saved baseline to {cache_path}")

    return baseline_results
