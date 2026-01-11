#!/usr/bin/env python3
"""Experiment 1: Answer Dependency (Semantic - Strong)

Dataset: MoreHopQA (3-5 hop reasoning with gold sub-questions and sub-answers)

Research Question: Does answering questions in correct order with prior context
improve multi-step reasoning performance?

Conditions:
- Oracle (Sequential + Pred Context): Follow decomposition order, pass prior Q&A with predicted answers
- Oracle-Gold (Sequential + Gold Context): Follow decomposition order, pass prior Q&A with gold answers
- No-Context (Sequential + Prior Q&A only): Follow decomposition order, pass prior Q&A but NO reference context
- Independent (No Context): Answer each sub-question independently, no prior info
- Shuffled (Random Order + Context): Random order but still pass prior Q&A

Expected: Oracle-Gold > Oracle > Shuffled > Independent
(No-Context tests if model relies on reference context or prior Q&A chain)
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm

from utils import (
    ExperimentConfig,
    ExperimentResult,
    LLMClient,
    compute_contains,
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


def _evaluate_answer(pred: str, gold: str) -> bool:
    """Evaluate if prediction matches gold answer."""
    pred = pred.strip().lower()
    gold = gold.strip().lower()

    # Exact match
    if pred == gold or gold in pred:
        return True

    # Use contains metric
    return compute_contains(pred, gold) > 0


def run_oracle(
    samples: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """Oracle condition: Sequential order with prior Q&A context.

    - Answer sub-questions in correct order
    - Pass prior Q&A as context
    - Include supporting paragraphs
    """
    logger.info("Running Oracle condition (Sequential + Context)...")

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_prompt_chars = 0
    total_response_chars = 0

    for sample in tqdm(samples, desc="Oracle"):
        decomp = sample["decomposition"]
        context_str = _format_context(sample.get("context", []))

        # Build context with prior Q&A
        prior_qa = []
        sample_correct = 0
        sample_details = []

        for i, step in enumerate(decomp):
            sub_q = step["question"]
            gold_answer = step["answer"]

            # Build prompt with context and prior Q&A
            prompt_parts = []

            # Add supporting context
            if context_str:
                prompt_parts.append(f"Reference Information:\n{context_str}")

            # Add prior Q&A
            if prior_qa:
                qa_context = "Previous Q&A:\n" + "\n".join(
                    [f"Q: {q}\nA: {a}" for q, a in prior_qa]
                )
                prompt_parts.append(qa_context)

            prompt_parts.append(f"Now answer this question:\nQ: {sub_q}\nA:")
            prompt = "\n\n".join(prompt_parts)

            pred, response = client.generate(prompt, max_tokens=256)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens
            total_prompt_chars += len(prompt)
            total_response_chars += len(pred)

            # Evaluate this sub-question
            is_correct = _evaluate_answer(pred, gold_answer)
            if is_correct:
                total_correct += 1
                sample_correct += 1
            total_questions += 1

            sample_details.append({
                "sub_id": i,
                "question": sub_q,
                "gold_answer": gold_answer,
                "prediction": pred.strip(),
                "correct": is_correct,
                "prompt_len": len(prompt),
                "response_len": len(pred),
            })

            # Pass predicted answer (not gold) to next step
            prior_qa.append((sub_q, pred.strip()))

        details.append({
            "main_question": sample["question"],
            "n_hops": sample["n_hops"],
            "sub_questions": sample_details,
            "accuracy": sample_correct / len(decomp) if decomp else 0,
        })

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    avg_prompt_len = total_prompt_chars / total_questions if total_questions > 0 else 0
    avg_response_len = total_response_chars / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="oracle",
        dataset="morehopqa",
        n_samples=len(samples),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "sub_question_accuracy": accuracy,
            "avg_prompt_len": avg_prompt_len,
            "avg_response_len": avg_response_len,
            "total_prompt_chars": total_prompt_chars,
            "total_response_chars": total_response_chars,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_oracle_gold(
    samples: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """Oracle-Gold condition: Sequential order with gold answers as context.

    - Answer sub-questions in correct order
    - Pass gold answers (not predicted) as context
    - Include supporting paragraphs
    - This establishes upper bound to test error propagation hypothesis
    """
    logger.info("Running Oracle-Gold condition (Sequential + Gold Context)...")

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_prompt_chars = 0
    total_response_chars = 0

    for sample in tqdm(samples, desc="Oracle-Gold"):
        decomp = sample["decomposition"]
        context_str = _format_context(sample.get("context", []))

        # Build context with prior Q&A (using gold answers)
        prior_qa = []
        sample_correct = 0
        sample_details = []

        for i, step in enumerate(decomp):
            sub_q = step["question"]
            gold_answer = step["answer"]

            # Build prompt with context and prior Q&A
            prompt_parts = []

            # Add supporting context
            if context_str:
                prompt_parts.append(f"Reference Information:\n{context_str}")

            # Add prior Q&A (with gold answers)
            if prior_qa:
                qa_context = "Previous Q&A:\n" + "\n".join(
                    [f"Q: {q}\nA: {a}" for q, a in prior_qa]
                )
                prompt_parts.append(qa_context)

            prompt_parts.append(f"Now answer this question:\nQ: {sub_q}\nA:")
            prompt = "\n\n".join(prompt_parts)

            pred, response = client.generate(prompt, max_tokens=256)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens
            total_prompt_chars += len(prompt)
            total_response_chars += len(pred)

            # Evaluate this sub-question
            is_correct = _evaluate_answer(pred, gold_answer)
            if is_correct:
                total_correct += 1
                sample_correct += 1
            total_questions += 1

            sample_details.append({
                "sub_id": i,
                "question": sub_q,
                "gold_answer": gold_answer,
                "prediction": pred.strip(),
                "correct": is_correct,
                "prompt_len": len(prompt),
                "response_len": len(pred),
            })

            # Pass gold answer (not predicted) to next step
            prior_qa.append((sub_q, gold_answer))

        details.append({
            "main_question": sample["question"],
            "n_hops": sample["n_hops"],
            "sub_questions": sample_details,
            "accuracy": sample_correct / len(decomp) if decomp else 0,
        })

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    avg_prompt_len = total_prompt_chars / total_questions if total_questions > 0 else 0
    avg_response_len = total_response_chars / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="oracle_gold",
        dataset="morehopqa",
        n_samples=len(samples),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "sub_question_accuracy": accuracy,
            "avg_prompt_len": avg_prompt_len,
            "avg_response_len": avg_response_len,
            "total_prompt_chars": total_prompt_chars,
            "total_response_chars": total_response_chars,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_no_context(
    samples: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """No-Context condition: Sequential order with prior Q&A but NO reference context.

    - Answer sub-questions in correct order
    - Pass prior Q&A as context (with predicted answers)
    - NO supporting paragraphs (to test if model relies on context vs prior Q&A)
    """
    logger.info("Running No-Context condition (Sequential + Prior Q&A, No Reference)...")

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_prompt_chars = 0
    total_response_chars = 0

    for sample in tqdm(samples, desc="No-Context"):
        decomp = sample["decomposition"]
        # NOTE: No context_str here - that's the key difference

        # Build context with prior Q&A only
        prior_qa = []
        sample_correct = 0
        sample_details = []

        for i, step in enumerate(decomp):
            sub_q = step["question"]
            gold_answer = step["answer"]

            # Build prompt with prior Q&A only (NO reference context)
            prompt_parts = []

            # Add prior Q&A
            if prior_qa:
                qa_context = "Previous Q&A:\n" + "\n".join(
                    [f"Q: {q}\nA: {a}" for q, a in prior_qa]
                )
                prompt_parts.append(qa_context)
                prompt_parts.append(f"Now answer this question:\nQ: {sub_q}\nA:")
            else:
                prompt_parts.append(f"Q: {sub_q}\nA:")

            prompt = "\n\n".join(prompt_parts)

            pred, response = client.generate(prompt, max_tokens=256)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens
            total_prompt_chars += len(prompt)
            total_response_chars += len(pred)

            # Evaluate this sub-question
            is_correct = _evaluate_answer(pred, gold_answer)
            if is_correct:
                total_correct += 1
                sample_correct += 1
            total_questions += 1

            sample_details.append({
                "sub_id": i,
                "question": sub_q,
                "gold_answer": gold_answer,
                "prediction": pred.strip(),
                "correct": is_correct,
                "prompt_len": len(prompt),
                "response_len": len(pred),
            })

            # Pass predicted answer to next step
            prior_qa.append((sub_q, pred.strip()))

        details.append({
            "main_question": sample["question"],
            "n_hops": sample["n_hops"],
            "sub_questions": sample_details,
            "accuracy": sample_correct / len(decomp) if decomp else 0,
        })

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    avg_prompt_len = total_prompt_chars / total_questions if total_questions > 0 else 0
    avg_response_len = total_response_chars / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="no_context",
        dataset="morehopqa",
        n_samples=len(samples),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "sub_question_accuracy": accuracy,
            "avg_prompt_len": avg_prompt_len,
            "avg_response_len": avg_response_len,
            "total_prompt_chars": total_prompt_chars,
            "total_response_chars": total_response_chars,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_independent(
    samples: List[Dict[str, Any]],
    client: LLMClient,
    batch_size: int = 16,
) -> ExperimentResult:
    """Independent condition: Each sub-question answered independently.

    - No prior Q&A context
    - Still include supporting paragraphs
    - Uses batch inference for efficiency
    """
    logger.info(f"Running Independent condition (No Context, batch_size={batch_size})...")

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_prompt_chars = 0
    total_response_chars = 0

    # Collect all prompts and metadata for batch processing
    all_prompts = []
    all_metadata = []  # (sample_idx, sub_idx, sub_q, gold_answer, context_str)

    for sample_idx, sample in enumerate(samples):
        decomp = sample["decomposition"]
        context_str = _format_context(sample.get("context", []))

        for i, step in enumerate(decomp):
            sub_q = step["question"]
            gold_answer = step["answer"]

            # Build prompt with context only (no prior Q&A)
            prompt_parts = []
            if context_str:
                prompt_parts.append(f"Reference Information:\n{context_str}")
            prompt_parts.append(f"Answer this question:\nQ: {sub_q}\nA:")
            prompt = "\n\n".join(prompt_parts)

            all_prompts.append(prompt)
            all_metadata.append((sample_idx, i, sub_q, gold_answer))

    # Process in batches
    all_results = []
    for batch_start in tqdm(range(0, len(all_prompts), batch_size), desc="Independent"):
        batch_prompts = all_prompts[batch_start:batch_start + batch_size]
        batch_results = client.generate_batch(batch_prompts, max_tokens=256)
        all_results.extend(batch_results)

    # Organize results by sample
    sample_results = {}  # sample_idx -> list of (sub_idx, sub_q, gold_answer, pred, response)
    for idx, (pred, response) in enumerate(all_results):
        sample_idx, sub_idx, sub_q, gold_answer = all_metadata[idx]
        if sample_idx not in sample_results:
            sample_results[sample_idx] = []
        sample_results[sample_idx].append((sub_idx, sub_q, gold_answer, pred, response, all_prompts[idx]))

    # Evaluate and build details
    for sample_idx, sample in enumerate(samples):
        decomp = sample["decomposition"]
        sample_correct = 0
        sample_details = []

        results = sample_results.get(sample_idx, [])
        results.sort(key=lambda x: x[0])  # Sort by sub_idx

        for sub_idx, sub_q, gold_answer, pred, response, prompt in results:
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens
            total_prompt_chars += len(prompt)
            total_response_chars += len(pred)

            is_correct = _evaluate_answer(pred, gold_answer)
            if is_correct:
                total_correct += 1
                sample_correct += 1
            total_questions += 1

            sample_details.append({
                "sub_id": sub_idx,
                "question": sub_q,
                "gold_answer": gold_answer,
                "prediction": pred.strip(),
                "correct": is_correct,
                "prompt_len": len(prompt),
                "response_len": len(pred),
            })

        details.append({
            "main_question": sample["question"],
            "n_hops": sample["n_hops"],
            "sub_questions": sample_details,
            "accuracy": sample_correct / len(decomp) if decomp else 0,
        })

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    avg_prompt_len = total_prompt_chars / total_questions if total_questions > 0 else 0
    avg_response_len = total_response_chars / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="independent",
        dataset="morehopqa",
        n_samples=len(samples),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "sub_question_accuracy": accuracy,
            "avg_prompt_len": avg_prompt_len,
            "avg_response_len": avg_response_len,
            "total_prompt_chars": total_prompt_chars,
            "total_response_chars": total_response_chars,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_shuffled(
    samples: List[Dict[str, Any]],
    client: LLMClient,
    seed: int = 42,
) -> ExperimentResult:
    """Shuffled condition: Random order but with prior Q&A context.

    - Answer sub-questions in random order
    - Pass prior Q&A as context (but in wrong order)
    - Include supporting paragraphs
    """
    logger.info("Running Shuffled condition (Random Order + Context)...")

    random.seed(seed)

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_prompt_chars = 0
    total_response_chars = 0

    for sample in tqdm(samples, desc="Shuffled"):
        decomp = sample["decomposition"]
        context_str = _format_context(sample.get("context", []))

        # Shuffle order
        indices = list(range(len(decomp)))
        random.shuffle(indices)

        # Build context with prior Q&A (in shuffled order)
        prior_qa = []
        predictions = {}
        prompt_lens = {}
        response_lens = {}
        sample_correct = 0
        sample_details = []

        for idx in indices:
            step = decomp[idx]
            sub_q = step["question"]
            gold_answer = step["answer"]

            # Build prompt with context and prior Q&A
            prompt_parts = []

            if context_str:
                prompt_parts.append(f"Reference Information:\n{context_str}")

            if prior_qa:
                qa_context = "Previous Q&A:\n" + "\n".join(
                    [f"Q: {q}\nA: {a}" for q, a in prior_qa]
                )
                prompt_parts.append(qa_context)

            prompt_parts.append(f"Now answer this question:\nQ: {sub_q}\nA:")
            prompt = "\n\n".join(prompt_parts)

            pred, response = client.generate(prompt, max_tokens=256)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens
            total_prompt_chars += len(prompt)
            total_response_chars += len(pred)

            predictions[idx] = pred.strip()
            prompt_lens[idx] = len(prompt)
            response_lens[idx] = len(pred)

            # Pass predicted answer to next step
            prior_qa.append((sub_q, pred.strip()))

        # Evaluate all sub-questions (in original order)
        for i, step in enumerate(decomp):
            gold_answer = step["answer"]
            pred = predictions.get(i, "")

            is_correct = _evaluate_answer(pred, gold_answer)
            if is_correct:
                total_correct += 1
                sample_correct += 1
            total_questions += 1

            sample_details.append({
                "sub_id": i,
                "question": step["question"],
                "gold_answer": gold_answer,
                "prediction": pred,
                "correct": is_correct,
                "prompt_len": prompt_lens.get(i, 0),
                "response_len": response_lens.get(i, 0),
            })

        details.append({
            "main_question": sample["question"],
            "n_hops": sample["n_hops"],
            "shuffled_order": indices,
            "sub_questions": sample_details,
            "accuracy": sample_correct / len(decomp) if decomp else 0,
        })

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    avg_prompt_len = total_prompt_chars / total_questions if total_questions > 0 else 0
    avg_response_len = total_response_chars / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="shuffled",
        dataset="morehopqa",
        n_samples=len(samples),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "sub_question_accuracy": accuracy,
            "avg_prompt_len": avg_prompt_len,
            "avg_response_len": avg_response_len,
            "total_prompt_chars": total_prompt_chars,
            "total_response_chars": total_response_chars,
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

    if "oracle" in conditions:
        logger.info(f"[Worker {rank}] Running oracle...")
        results.append(run_oracle(samples, client))

    if "oracle_gold" in conditions:
        logger.info(f"[Worker {rank}] Running oracle_gold...")
        results.append(run_oracle_gold(samples, client))

    if "no_context" in conditions:
        logger.info(f"[Worker {rank}] Running no_context...")
        results.append(run_no_context(samples, client))

    if "independent" in conditions:
        logger.info(f"[Worker {rank}] Running independent...")
        results.append(run_independent(samples, client))

    if "shuffled" in conditions:
        logger.info(f"[Worker {rank}] Running shuffled...")
        results.append(run_shuffled(samples, client, seed=seed))

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


def main():
    import multiprocessing as mp
    import json

    parser = argparse.ArgumentParser(
        description="Exp 1: Answer Dependency - MoreHopQA"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="Model to use for inference (e.g., gpt-4o-mini, Qwen/Qwen2.5-7B-Instruct)"
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
        "--conditions", type=str, default="oracle,oracle_gold,no_context,independent,shuffled",
        help="Comma-separated list of conditions to run (oracle, oracle_gold, no_context, independent, shuffled)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/preliminary",
        help="Output directory for results"
    )
    # Parallel mode
    parser.add_argument(
        "--parallel", action="store_true",
        help="Enable parallel inference with multiple GPUs"
    )
    parser.add_argument(
        "--gpus", type=str, default="0,1,2,3,4,5,6,7",
        help="Comma-separated list of GPUs for parallel mode"
    )
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]

    # Load data
    all_samples = load_morehopqa(n_samples=args.n_samples, seed=args.seed)

    if args.parallel and args.use_local:
        # Parallel mode with multiprocessing
        gpus = [int(g.strip()) for g in args.gpus.split(",")]
        world_size = len(gpus)

        logger.info(f"Parallel mode with {world_size} GPUs: {gpus}")

        # Set spawn method (required for CUDA)
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Clean up old temp files
        import os
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
                args=(rank, world_size, gpu_id, args.model, args.use_vllm,
                      shards[rank], conditions, args.seed, args.output_dir)
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

        # Merge and print results
        final_results = []
        for cond, results_list in all_results_by_condition.items():
            merged = _merge_results(results_list)
            final_results.append(merged)

        config = ExperimentConfig(
            exp_name="exp1_answer_dependency",
            dataset="morehopqa",
            model=args.model,
            n_samples=args.n_samples,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print_summary(final_results)
        save_results(final_results, config)

    else:
        # Single process mode (API or single GPU)
        config = ExperimentConfig(
            exp_name="exp1_answer_dependency",
            dataset="morehopqa",
            model=args.model,
            n_samples=args.n_samples,
            seed=args.seed,
            output_dir=args.output_dir,
        )

        client = LLMClient(
            model=args.model,
            use_local=args.use_local,
            use_vllm=args.use_vllm,
            tensor_parallel_size=1,
        )

        results = []

        if "oracle" in conditions:
            results.append(run_oracle(all_samples, client))
        if "oracle_gold" in conditions:
            results.append(run_oracle_gold(all_samples, client))
        if "no_context" in conditions:
            results.append(run_no_context(all_samples, client))
        if "independent" in conditions:
            results.append(run_independent(all_samples, client))
        if "shuffled" in conditions:
            results.append(run_shuffled(all_samples, client, seed=args.seed))

        print_summary(results)
        save_results(results, config)


def _merge_results(results: List[ExperimentResult]) -> ExperimentResult:
    """Merge results from multiple ranks into one."""
    if not results:
        raise ValueError("No results to merge")
    if len(results) == 1:
        return results[0]

    # Sum up metrics
    total_correct = 0
    total_questions = 0
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_prompt_chars = 0
    total_response_chars = 0
    all_details = []

    for r in results:
        total_questions += r.n_questions
        total_correct += int(r.accuracy * r.n_questions)
        total_latency += r.latency
        total_prompt_tokens += r.prompt_tokens
        total_completion_tokens += r.completion_tokens
        total_prompt_chars += r.metrics.get("total_prompt_chars", 0)
        total_response_chars += r.metrics.get("total_response_chars", 0)
        all_details.extend(r.details)

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    avg_prompt_len = total_prompt_chars / total_questions if total_questions > 0 else 0
    avg_response_len = total_response_chars / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition=results[0].condition,
        dataset=results[0].dataset,
        n_samples=sum(r.n_samples for r in results),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={
            "sub_question_accuracy": accuracy,
            "avg_prompt_len": avg_prompt_len,
            "avg_response_len": avg_response_len,
            "total_prompt_chars": total_prompt_chars,
            "total_response_chars": total_response_chars,
        },
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=all_details,
    )


if __name__ == "__main__":
    main()
