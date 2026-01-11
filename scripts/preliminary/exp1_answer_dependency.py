#!/usr/bin/env python3
"""Experiment 1: Answer Dependency (Semantic - Strong)

Dataset: MoreHopQA (3-5 hop reasoning with gold sub-questions and sub-answers)

Research Question: Does answering questions in correct order with prior context
improve multi-step reasoning performance?

Conditions:
- Oracle (Sequential + Context): Follow decomposition order, pass prior Q&A with context
- Independent (No Context): Answer each sub-question independently, no prior info
- Shuffled (Random Order + Context): Random order but still pass prior Q&A

Expected: Oracle > Shuffled > Independent
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
    setup_distributed,
    cleanup_distributed,
    shard_data,
    gather_results,
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

            pred, response = client.generate(prompt, max_tokens=128)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

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

    return ExperimentResult(
        condition="oracle",
        dataset="morehopqa",
        n_samples=len(samples),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"sub_question_accuracy": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def run_independent(
    samples: List[Dict[str, Any]],
    client: LLMClient,
) -> ExperimentResult:
    """Independent condition: Each sub-question answered independently.

    - No prior Q&A context
    - Still include supporting paragraphs
    """
    logger.info("Running Independent condition (No Context)...")

    total_correct = 0
    total_questions = 0
    details = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for sample in tqdm(samples, desc="Independent"):
        decomp = sample["decomposition"]
        context_str = _format_context(sample.get("context", []))

        sample_correct = 0
        sample_details = []

        for i, step in enumerate(decomp):
            sub_q = step["question"]
            gold_answer = step["answer"]

            # Build prompt with context only (no prior Q&A)
            prompt_parts = []

            if context_str:
                prompt_parts.append(f"Reference Information:\n{context_str}")

            prompt_parts.append(f"Answer this question:\nQ: {sub_q}\nA:")
            prompt = "\n\n".join(prompt_parts)

            pred, response = client.generate(prompt, max_tokens=128)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

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
            })

        details.append({
            "main_question": sample["question"],
            "n_hops": sample["n_hops"],
            "sub_questions": sample_details,
            "accuracy": sample_correct / len(decomp) if decomp else 0,
        })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="independent",
        dataset="morehopqa",
        n_samples=len(samples),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"sub_question_accuracy": accuracy},
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

    for sample in tqdm(samples, desc="Shuffled"):
        decomp = sample["decomposition"]
        context_str = _format_context(sample.get("context", []))

        # Shuffle order
        indices = list(range(len(decomp)))
        random.shuffle(indices)

        # Build context with prior Q&A (in shuffled order)
        prior_qa = []
        predictions = {}
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

            pred, response = client.generate(prompt, max_tokens=128)
            total_latency += response.latency
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            predictions[idx] = pred.strip()

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
            })

        details.append({
            "main_question": sample["question"],
            "n_hops": sample["n_hops"],
            "shuffled_order": indices,
            "sub_questions": sample_details,
            "accuracy": sample_correct / len(decomp) if decomp else 0,
        })

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition="shuffled",
        dataset="morehopqa",
        n_samples=len(samples),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"sub_question_accuracy": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=details,
    )


def main():
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
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism (vLLM only)"
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
        "--conditions", type=str, default="oracle,independent,shuffled",
        help="Comma-separated list of conditions to run"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/preliminary",
        help="Output directory for results"
    )
    args = parser.parse_args()

    # Setup distributed (each GPU loads one model)
    rank, world_size = setup_distributed()

    # Configuration
    config = ExperimentConfig(
        exp_name="exp1_answer_dependency",
        dataset="morehopqa",
        model=args.model,
        n_samples=args.n_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Load data (all ranks load the same data, then shard)
    all_samples = load_morehopqa(n_samples=args.n_samples, seed=args.seed)

    # Shard data across GPUs
    samples = shard_data(all_samples, rank, world_size)
    logger.info(f"Rank {rank}/{world_size}: processing {len(samples)}/{len(all_samples)} samples")

    # Initialize LLM client (will use current GPU)
    client = LLMClient(
        model=args.model,
        use_local=args.use_local,
        use_vllm=args.use_vllm,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Run conditions on local shard
    conditions = [c.strip() for c in args.conditions.split(",")]
    local_results = []

    if "oracle" in conditions:
        local_results.append(run_oracle(samples, client))

    if "independent" in conditions:
        local_results.append(run_independent(samples, client))

    if "shuffled" in conditions:
        local_results.append(run_shuffled(samples, client, seed=args.seed))

    # Gather results from all GPUs
    all_results_by_condition = {}
    for result in local_results:
        gathered = gather_results([result], world_size)
        if rank == 0:
            # Merge results from all ranks for this condition
            merged = _merge_results(gathered)
            all_results_by_condition[result.condition] = merged

    # Print and save results (only rank 0)
    if rank == 0:
        results = list(all_results_by_condition.values())
        print_summary(results)
        save_results(results, config)

    cleanup_distributed()


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
    all_details = []

    for r in results:
        total_questions += r.n_questions
        total_correct += int(r.accuracy * r.n_questions)
        total_latency += r.latency
        total_prompt_tokens += r.prompt_tokens
        total_completion_tokens += r.completion_tokens
        all_details.extend(r.details)

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    return ExperimentResult(
        condition=results[0].condition,
        dataset=results[0].dataset,
        n_samples=sum(r.n_samples for r in results),
        n_questions=total_questions,
        accuracy=accuracy,
        metrics={"sub_question_accuracy": accuracy},
        latency=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        details=all_details,
    )


if __name__ == "__main__":
    main()
