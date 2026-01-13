#!/usr/bin/env python3
"""Experiment 1: Answer Dependency (Semantic - Strong)

Dataset: MoreHopQA (3-5 hop reasoning with gold sub-questions and sub-answers)

Research Question: Does decomposing questions and passing prior answers help?

3 Conditions:
- gold: Only ask the last sub-question (with gold answers pre-embedded in question text)
- sequential: Answer sub-questions in order, replacing embedded gold answers with predictions
- main_question: Ask the main question directly (no decomposition)

Key insight: MoreHopQA sub-questions have gold answers pre-embedded in question text.
E.g., Q1: "When was X born?" -> A1: "June 23, 1992"
      Q2: "What is the sum of digits in June 23, 1992?" <- A1 is already embedded
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm

from utils import (
    ExperimentConfig,
    ExperimentResult,
    LLMClient,
    compute_exact_match,
    compute_f1,
    normalize_answer,
    print_summary,
    save_results,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_morehopqa(n_samples: int = -1, seed: int = 42) -> List[Dict[str, Any]]:
    """Load MoreHopQA dataset.

    Each sample contains:
    - question: The main question
    - answer: The final answer
    - decomposition: List of {question, answer, paragraph_support_title} for sub-questions
    - context: List of [title, [sentences...]] for supporting paragraphs
    """
    logger.info("Loading MoreHopQA dataset...")

    valid_samples = []

    # Download raw JSON directly from HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download
        import json as json_module

        logger.info("Downloading MoreHopQA from HuggingFace Hub...")
        file_path = hf_hub_download(
            repo_id="alabnii/morehopqa",
            filename="data/with_human_verification.json",
            repo_type="dataset",
        )
        with open(file_path, "r", encoding="utf-8") as f:
            data = json_module.load(f)

        for item in data:
            decomp = item.get("question_decomposition", [])
            context = item.get("context", [])
            if decomp and len(decomp) >= 2:
                valid_samples.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "decomposition": decomp,
                    "context": context,
                    "n_hops": len(decomp),
                })

        logger.info(f"Loaded {len(valid_samples)} samples from MoreHopQA")

    except Exception as e:
        logger.warning(f"Failed to load MoreHopQA: {e}")
        logger.info("Falling back to HotpotQA...")

        # Fallback to HotpotQA
        dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)

        for item in dataset:
            # Use supporting facts to create decomposition
            sf = item.get("supporting_facts", {})
            titles = sf.get("title", [])
            sent_ids = sf.get("sent_id", [])

            if len(set(titles)) >= 2:  # At least 2 different sources
                context_dict = {t: s for t, s in zip(item["context"]["title"], item["context"]["sentences"])}
                decomp = []
                seen_titles = set()
                for title, sid in zip(titles, sent_ids):
                    if title not in seen_titles and title in context_dict:
                        sents = context_dict[title]
                        if sid < len(sents):
                            decomp.append({
                                "question": f"What information about {title} is relevant?",
                                "answer": sents[sid],
                                "paragraph_support_title": title,
                            })
                            seen_titles.add(title)

                if len(decomp) >= 2:
                    # Build context in MoreHopQA format
                    context = [[t, s] for t, s in zip(item["context"]["title"], item["context"]["sentences"])]
                    valid_samples.append({
                        "question": item["question"],
                        "answer": item["answer"],
                        "decomposition": decomp,
                        "context": context,
                        "n_hops": len(decomp),
                    })

        logger.info(f"Loaded {len(valid_samples)} samples from HotpotQA (fallback)")

    # Shuffle and sample
    random.seed(seed)
    random.shuffle(valid_samples)
    if n_samples > 0:
        samples = valid_samples[:n_samples]
    else:
        samples = valid_samples  # Use all samples

    logger.info(f"Loaded {len(samples)} samples with valid decomposition")
    logger.info(f"Hop distribution: {_count_hops(samples)}")

    return samples


def _count_hops(samples: List[Dict]) -> Dict[int, int]:
    """Count samples by number of hops."""
    counts = {}
    for s in samples:
        n = s["n_hops"]
        counts[n] = counts.get(n, 0) + 1
    return dict(sorted(counts.items()))


def _format_context(context: List[List]) -> str:
    """Format context paragraphs into a string."""
    if not context:
        return ""

    parts = []
    for item in context:
        if len(item) >= 2:
            title = item[0]
            sentences = item[1]
            if isinstance(sentences, list):
                text = " ".join(sentences)
            else:
                text = str(sentences)
            parts.append(f"[{title}]: {text}")

    return "\n".join(parts)


def _extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} format."""
    # Match \boxed{...} with nested braces support
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()  # Return last match
    # Fallback: return original text
    return text.strip()


def _evaluate_answer(pred: str, gold: str) -> Tuple[float, float]:
    """Evaluate prediction against gold answer.

    Returns:
        Tuple of (exact_match, f1_score)
    """
    # Extract answer from \boxed{} if present
    pred = _extract_boxed_answer(pred)

    em = compute_exact_match(pred, gold)
    f1 = compute_f1(pred, gold)

    return em, f1


def _replace_embedded_answer(question: str, gold_answer: str, pred_answer: str) -> str:
    """Replace embedded gold answer in question text with predicted answer.

    Args:
        question: The question text (may contain gold_answer embedded)
        gold_answer: The gold answer to replace
        pred_answer: The predicted answer to substitute

    Returns:
        Question text with gold_answer replaced by pred_answer
    """
    if not gold_answer or not pred_answer:
        return question

    # Try case-insensitive replacement
    pattern = re.compile(re.escape(gold_answer), re.IGNORECASE)
    new_question = pattern.sub(pred_answer, question)

    return new_question


def run_gold(
    samples: List[Dict[str, Any]],
    client: LLMClient,
    use_context: bool = True,
    max_tokens: int = 2048,
    batch_size: int = 16,
) -> ExperimentResult:
    """Gold condition: Only ask the last sub-question (with gold answers pre-embedded).

    This is the upper bound - the last question already contains all prior gold answers.

    Args:
        samples: List of samples
        client: LLM client
        use_context: If True, include reference context paragraphs
        max_tokens: Max tokens to generate
        batch_size: Batch size for inference

    Returns:
        ExperimentResult
    """
    condition = "gold"
    logger.info(f"Running {condition} condition...")

    total_em = 0.0
    total_f1 = 0.0
    total_samples = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Collect all prompts for batch processing
    all_prompts = []
    all_metadata = []  # (sample_idx, last_q, gold_answer)

    for sample_idx, sample in enumerate(samples):
        decomp = sample["decomposition"]
        context_str = _format_context(sample.get("context", [])) if use_context else ""

        # Only the last sub-question
        last_step = decomp[-1]
        last_q = last_step["question"]
        gold_answer = last_step["answer"]

        # Build prompt
        prompt_parts = []
        if context_str:
            prompt_parts.append(f"Reference Information:\n{context_str}")
        prompt_parts.append(f"Answer this question. Put your final answer in \\boxed{{}}.\nQ: {last_q}")
        prompt = "\n\n".join(prompt_parts)

        all_prompts.append(prompt)
        all_metadata.append((sample_idx, last_q, gold_answer, prompt))

    # Process in batches
    all_results = []
    for batch_start in tqdm(range(0, len(all_prompts), batch_size), desc=condition):
        batch_prompts = all_prompts[batch_start:batch_start + batch_size]
        batch_results = client.generate_batch(batch_prompts, max_tokens=max_tokens)
        all_results.extend(batch_results)

    # Evaluate results
    for idx, (pred, response) in enumerate(all_results):
        sample_idx, last_q, gold_answer, prompt_used = all_metadata[idx]
        sample = samples[sample_idx]

        total_latency += response.latency
        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens

        em, f1 = _evaluate_answer(pred, gold_answer)
        total_em += em
        total_f1 += f1
        total_samples += 1

        details.append({
            "main_question": sample["question"],
            "last_sub_question": last_q,
            "gold_answer": gold_answer,
            "prompt": prompt_used,
            "prediction": pred.strip(),
            "extracted_answer": _extract_boxed_answer(pred),
            "em": em,
            "f1": f1,
            "n_hops": sample["n_hops"],
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        })

    avg_em = total_em / total_samples if total_samples > 0 else 0
    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0

    return ExperimentResult(
        condition=condition,
        dataset="morehopqa",
        n_samples=len(samples),
        n_questions=total_samples,
        accuracy=avg_em,
        metrics={
            "em": avg_em,
            "f1": avg_f1,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_sequential(
    samples: List[Dict[str, Any]],
    client: LLMClient,
    use_context: bool = True,
    max_tokens: int = 2048,
) -> ExperimentResult:
    """Sequential condition: Answer sub-questions in order, replacing embedded answers.

    Key difference from old implementation:
    - Replace gold answers embedded in subsequent questions with model predictions
    - This tests error propagation

    Args:
        samples: List of samples
        client: LLM client
        use_context: If True, include reference context paragraphs
        max_tokens: Max tokens to generate

    Returns:
        ExperimentResult
    """
    condition = "sequential"
    logger.info(f"Running {condition} condition...")

    total_em = 0.0
    total_f1 = 0.0
    total_samples = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for sample in tqdm(samples, desc=condition):
        decomp = sample["decomposition"]
        context_str = _format_context(sample.get("context", [])) if use_context else ""

        prior_qa = []
        sample_details = []

        # Track gold -> pred answer mappings for replacement
        answer_replacements = {}  # gold_answer -> pred_answer

        for i, step in enumerate(decomp):
            original_q = step["question"]
            gold_answer = step["answer"]

            # Replace any embedded gold answers with predicted answers
            current_q = original_q
            for gold_ans, pred_ans in answer_replacements.items():
                current_q = _replace_embedded_answer(current_q, gold_ans, pred_ans)

            # Build prompt
            prompt_parts = []

            if context_str:
                prompt_parts.append(f"Reference Information:\n{context_str}")

            if prior_qa:
                qa_context = "Previous Q&A:\n" + "\n".join(
                    [f"Q: {q}\nA: {a}" for q, a in prior_qa]
                )
                prompt_parts.append(qa_context)

            prompt_parts.append(f"Now answer this question. Put your final answer in \\boxed{{}}.\nQ: {current_q}")
            prompt = "\n\n".join(prompt_parts)

            pred, response = client.generate(prompt, max_tokens=max_tokens)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            pred_answer = _extract_boxed_answer(pred)
            em, f1 = _evaluate_answer(pred, gold_answer)

            sample_details.append({
                "sub_id": i,
                "original_question": original_q,
                "modified_question": current_q,
                "gold_answer": gold_answer,
                "prompt": prompt,
                "prediction": pred.strip(),
                "extracted_answer": pred_answer,
                "em": em,
                "f1": f1,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            })

            # Update mappings for next iteration
            prior_qa.append((current_q, pred_answer))
            answer_replacements[gold_answer] = pred_answer

        # Only count the LAST step for metrics
        if sample_details:
            last_step = sample_details[-1]
            total_em += last_step["em"]
            total_f1 += last_step["f1"]
            total_samples += 1

        details.append({
            "main_question": sample["question"],
            "gold_answer": sample["answer"],
            "n_hops": sample["n_hops"],
            "sub_questions": sample_details,
            "final_em": sample_details[-1]["em"] if sample_details else 0,
            "final_f1": sample_details[-1]["f1"] if sample_details else 0,
        })

    avg_em = total_em / total_samples if total_samples > 0 else 0
    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0

    return ExperimentResult(
        condition=condition,
        dataset="morehopqa",
        n_samples=len(samples),
        n_questions=total_samples,
        accuracy=avg_em,
        metrics={
            "em": avg_em,
            "f1": avg_f1,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_main_question(
    samples: List[Dict[str, Any]],
    client: LLMClient,
    use_context: bool = True,
    max_tokens: int = 2048,
    batch_size: int = 16,
) -> ExperimentResult:
    """Main question condition: Ask the main question directly (no decomposition).

    This tests whether the model can do multi-hop reasoning without decomposition.

    Args:
        samples: List of samples
        client: LLM client
        use_context: If True, include reference context paragraphs
        max_tokens: Max tokens to generate
        batch_size: Batch size for inference

    Returns:
        ExperimentResult
    """
    condition = "main_question"
    logger.info(f"Running {condition} condition...")

    total_em = 0.0
    total_f1 = 0.0
    total_samples = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Collect all prompts for batch processing
    all_prompts = []
    all_metadata = []  # (sample_idx, main_q, gold_answer)

    for sample_idx, sample in enumerate(samples):
        main_q = sample["question"]
        gold_answer = sample["answer"]
        context_str = _format_context(sample.get("context", [])) if use_context else ""

        # Build prompt
        prompt_parts = []
        if context_str:
            prompt_parts.append(f"Reference Information:\n{context_str}")
        prompt_parts.append(f"Answer this question. Put your final answer in \\boxed{{}}.\nQ: {main_q}")
        prompt = "\n\n".join(prompt_parts)

        all_prompts.append(prompt)
        all_metadata.append((sample_idx, main_q, gold_answer, prompt))

    # Process in batches
    all_results = []
    for batch_start in tqdm(range(0, len(all_prompts), batch_size), desc=condition):
        batch_prompts = all_prompts[batch_start:batch_start + batch_size]
        batch_results = client.generate_batch(batch_prompts, max_tokens=max_tokens)
        all_results.extend(batch_results)

    # Evaluate results
    for idx, (pred, response) in enumerate(all_results):
        sample_idx, main_q, gold_answer, prompt_used = all_metadata[idx]
        sample = samples[sample_idx]

        total_latency += response.latency
        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens

        em, f1 = _evaluate_answer(pred, gold_answer)
        total_em += em
        total_f1 += f1
        total_samples += 1

        details.append({
            "main_question": main_q,
            "gold_answer": gold_answer,
            "prompt": prompt_used,
            "prediction": pred.strip(),
            "extracted_answer": _extract_boxed_answer(pred),
            "em": em,
            "f1": f1,
            "n_hops": sample["n_hops"],
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        })

    avg_em = total_em / total_samples if total_samples > 0 else 0
    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0

    return ExperimentResult(
        condition=condition,
        dataset="morehopqa",
        n_samples=len(samples),
        n_questions=total_samples,
        accuracy=avg_em,
        metrics={
            "em": avg_em,
            "f1": avg_f1,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def worker_process(
    rank: int,
    world_size: int,
    gpu_id: int,
    model: str,
    use_vllm: bool,
    samples: List[Dict[str, Any]],
    conditions: List[str],
    seed: int,
    output_dir: str,
    max_tokens: int = 2048,
):
    """Worker process that runs on a single GPU."""
    import os
    import json

    # IMPORTANT: Set environment variables BEFORE importing anything CUDA-related
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Disable vLLM V1 engine which spawns EngineCore processes and uses distributed
    os.environ["VLLM_USE_V1"] = "0"
    # Prevent vLLM from trying to use distributed
    os.environ["VLLM_DISABLE_FRONTEND_MULTIPROCESSING"] = "1"
    # Disable vLLM progress bar
    os.environ["VLLM_NO_PROGRESS_BAR"] = "1"
    # Disable tqdm globally
    os.environ["TQDM_DISABLE"] = "1"

    logger.info(f"[Worker {rank}] GPU {gpu_id}: Starting, {len(samples)} samples")

    # Initialize LLM client
    client = LLMClient(
        model=model,
        use_local=True,
        use_vllm=use_vllm,
        tensor_parallel_size=1,
    )

    logger.info(f"[Worker {rank}] Model loaded, running conditions: {conditions}")

    # Run conditions
    results = []

    for cond in conditions:
        logger.info(f"[Worker {rank}] Running {cond}...")

        if cond == "gold":
            results.append(run_gold(samples, client, use_context=True, max_tokens=max_tokens))
        elif cond == "sequential":
            results.append(run_sequential(samples, client, use_context=True, max_tokens=max_tokens))
        elif cond == "main_question":
            results.append(run_main_question(samples, client, use_context=True, max_tokens=max_tokens))

    # Save results to temp file
    os.makedirs(output_dir, exist_ok=True)
    temp_file = os.path.join(output_dir, f"temp_rank{rank}.json")

    # Convert results to serializable format
    results_data = []
    for r in results:
        results_data.append({
            "condition": r.condition,
            "n_samples": r.n_samples,
            "n_questions": r.n_questions,
            "accuracy": r.accuracy,
            "metrics": r.metrics,
            "latency": r.latency,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "details": r.details,
        })

    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f)

    logger.info(f"[Worker {rank}] Done, saved to {temp_file}")


def run_experiment_for_model(
    model: str,
    all_samples: List[Dict[str, Any]],
    conditions: List[str],
    args,
    num_gpus: int,
):
    """Run experiment for a single model."""
    import multiprocessing as mp
    import json
    import os

    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment for model: {model}")
    logger.info(f"{'='*60}\n")

    if args.use_local and num_gpus > 1:
        # Multi-GPU parallel mode with multiprocessing
        gpus = list(range(num_gpus))
        world_size = num_gpus

        logger.info(f"Parallel mode with {world_size} GPUs: {gpus}")

        # Set spawn method (required for CUDA)
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Clean up old temp files
        for rank in range(world_size):
            temp_file = os.path.join(args.output_dir, f"temp_rank{rank}.json")
            if os.path.exists(temp_file):
                os.remove(temp_file)

        # Shard data
        shards = [[] for _ in range(world_size)]
        for i, sample in enumerate(all_samples):
            shards[i % world_size].append(sample)

        # Start all workers
        processes = []
        for rank, gpu_id in enumerate(gpus):
            p = mp.Process(
                target=worker_process,
                args=(rank, world_size, gpu_id, model, args.use_vllm,
                      shards[rank], conditions, args.seed, args.output_dir, args.max_tokens)
            )
            p.start()
            processes.append(p)
            logger.info(f"Started worker {rank} on GPU {gpu_id} (PID: {p.pid})")

        # Wait for all workers
        for p in processes:
            p.join()

        logger.info("All workers finished, merging results...")

        # Merge results from all workers
        all_results_by_condition = {}
        for rank in range(world_size):
            temp_file = os.path.join(args.output_dir, f"temp_rank{rank}.json")
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                for r in results_data:
                    cond = r["condition"]
                    if cond not in all_results_by_condition:
                        all_results_by_condition[cond] = []
                    all_results_by_condition[cond].append(ExperimentResult(
                        condition=r["condition"],
                        dataset="morehopqa",
                        n_samples=r["n_samples"],
                        n_questions=r["n_questions"],
                        accuracy=r["accuracy"],
                        metrics=r["metrics"],
                        latency=r["latency"],
                        prompt_tokens=r["prompt_tokens"],
                        completion_tokens=r["completion_tokens"],
                        details=r["details"],
                    ))
                os.remove(temp_file)

        # Merge results
        final_results = []
        for cond, results_list in all_results_by_condition.items():
            merged = _merge_results(results_list)
            final_results.append(merged)

    else:
        # Single process mode (API or single GPU)
        if args.use_local and num_gpus == 1:
            logger.info("Single GPU mode: using GPU 0")

        client = LLMClient(
            model=model,
            use_local=args.use_local,
            use_vllm=args.use_vllm,
            tensor_parallel_size=1,
        )

        final_results = []

        for cond in conditions:
            if cond == "gold":
                final_results.append(run_gold(all_samples, client, use_context=True, max_tokens=args.max_tokens))
            elif cond == "sequential":
                final_results.append(run_sequential(all_samples, client, use_context=True, max_tokens=args.max_tokens))
            elif cond == "main_question":
                final_results.append(run_main_question(all_samples, client, use_context=True, max_tokens=args.max_tokens))

    # Print and save results for this model
    config = ExperimentConfig(
        exp_name="exp1_answer_dependency",
        dataset="morehopqa",
        model=model,
        n_samples=args.n_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*60}")
    print(f"Results for: {model}")
    print(f"{'='*60}")
    print_summary(final_results)
    save_results(final_results, config)

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Exp 1: Answer Dependency - MoreHopQA"
    )
    parser.add_argument(
        "--models", type=str, default="gpt-4o-mini",
        help="Comma-separated list of models (e.g., 'Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct')"
    )
    parser.add_argument(
        "--use-local", action="store_true",
        help="Use local model instead of API"
    )
    parser.add_argument(
        "--use-vllm", action="store_true",
        help="Use vLLM for faster inference (requires --use-local)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=-1,
        help="Number of samples to evaluate (-1 for all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--conditions", type=str,
        default="gold,sequential,main_question",
        help="Comma-separated list of conditions to run"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/preliminary",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Maximum number of tokens to generate"
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    logger.info(f"Models to evaluate: {models}")
    logger.info(f"Conditions: {conditions}")

    # Load data once (shared across all models)
    all_samples = load_morehopqa(n_samples=args.n_samples, seed=args.seed)

    # Auto-detect available GPUs
    num_gpus = 0
    if args.use_local:
        try:
            import torch
            num_gpus = torch.cuda.device_count()
            logger.info(f"Detected {num_gpus} available GPU(s)")
        except ImportError:
            logger.warning("PyTorch not available, using single process mode")

    # Run experiment for each model
    all_model_results = {}
    for model in models:
        results = run_experiment_for_model(model, all_samples, conditions, args, num_gpus)
        all_model_results[model] = results

    # Print final summary if multiple models
    if len(models) > 1:
        print(f"\n{'='*80}")
        print("FINAL SUMMARY - ALL MODELS")
        print(f"{'='*80}\n")
        for model, results in all_model_results.items():
            print(f"\n## {model}")
            print_summary(results)


def _merge_results(results: List[ExperimentResult]) -> ExperimentResult:
    """Merge results from multiple ranks into one."""
    if not results:
        raise ValueError("No results to merge")
    if len(results) == 1:
        return results[0]

    # Sum up metrics
    total_em = 0.0
    total_f1 = 0.0
    total_questions = 0
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    all_details = []

    for r in results:
        total_questions += r.n_questions
        # EM is stored in accuracy field and metrics["em"]
        total_em += r.metrics.get("em", r.accuracy) * r.n_questions
        total_f1 += r.metrics.get("f1", 0) * r.n_questions
        total_latency += r.latency
        total_prompt_tokens += r.prompt_tokens
        total_completion_tokens += r.completion_tokens
        all_details.extend(r.details)

    avg_em = total_em / total_questions if total_questions > 0 else 0
    avg_f1 = total_f1 / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition=results[0].condition,
        dataset=results[0].dataset,
        n_samples=sum(r.n_samples for r in results),
        n_questions=total_questions,
        accuracy=avg_em,
        metrics={
            "em": avg_em,
            "f1": avg_f1,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=all_details,
    )


if __name__ == "__main__":
    main()
