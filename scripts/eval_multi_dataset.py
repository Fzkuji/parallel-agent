#!/usr/bin/env python3
"""
Evaluate multiple QA datasets with shared context structure.

Tests SQuAD, DROP, and CoQA datasets on the same model with different strategies.
All datasets have multiple questions per context/passage.

Dataset Statistics:
- SQuAD 1.1: 87,599 train / 10,570 val, ~5 questions per passage
- DROP: 77,400 train / 9,535 val, ~16 questions per passage
- CoQA: 7,199 train / 500 val, ~15 questions per story

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_multi_dataset.py \
        --model "Qwen/Qwen2.5-7B-Instruct" \
        --datasets "squad,drop,coqa" \
        --group-sizes "1,4,8" \
        --output-dir outputs/multi_dataset_eval
"""

import argparse
import json
import logging
import os
import sys
import random
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_available_gpus():
    """Get list of available GPU IDs from CUDA_VISIBLE_DEVICES or detect all GPUs."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible:
        gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
        return gpu_ids
    else:
        try:
            import torch
            return list(range(torch.cuda.device_count()))
        except:
            return [0]


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-dataset QA evaluation")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--datasets", type=str, default="squad,drop,coqa",
                       help="Datasets to test (comma-separated)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-sizes", type=str, default="1,4,8",
                       help="Group sizes to test (questions per group)")
    parser.add_argument("--output-dir", type=str, default="outputs/multi_dataset_eval")
    parser.add_argument("--max-contexts", type=int, default=50,
                       help="Maximum number of contexts per dataset")
    parser.add_argument("--strategies", type=str, default="batch,all_in_one,sequential",
                       help="Strategies to test")

    return parser.parse_args()


def load_dataset_questions(dataset: str, seed: int, min_questions: int, max_contexts: int) -> Tuple[List[dict], List[dict]]:
    """
    Load questions from the specified dataset.

    Returns:
        Tuple of (all_questions, contexts)
        - all_questions: flat list of {qid, question, context, references, answer_tokens}
        - contexts: original context groups
    """
    if dataset == "squad":
        from src.datasets.squad import load_squad_groups
        contexts = load_squad_groups(
            split="validation",
            min_questions=min_questions,
            max_questions=min_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    elif dataset == "drop":
        from src.datasets.drop import load_drop_groups
        contexts = load_drop_groups(
            split="validation",
            min_questions=min_questions,
            max_questions=min_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    elif dataset == "coqa":
        from src.datasets.coqa import load_coqa_groups
        contexts = load_coqa_groups(
            split="validation",
            min_questions=min_questions,
            max_questions=min_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Flatten to list of questions
    all_questions = []
    for ctx in contexts:
        context_text = ctx["context"]
        for q in ctx["questions"]:
            all_questions.append({
                "qid": f"{ctx['title']}_{q['qid']}",
                "question": q["text"],
                "context": context_text,
                "references": q.get("references", []),
                "answer_tokens": q.get("answer_tokens", 32),
            })

    return all_questions, contexts


def group_questions(all_questions: List[dict], group_size: int) -> List[List[dict]]:
    """Group questions into chunks of group_size."""
    groups = []
    for i in range(0, len(all_questions), group_size):
        groups.append(all_questions[i:i+group_size])
    return groups


def gpu_worker(
    worker_id: int,
    physical_gpu_id: int,
    groups: List[List[dict]],
    group_indices: List[int],
    args_dict: dict,
    strategies_to_run: List[str],
    result_queue: mp.Queue,
):
    """GPU worker that loads model and processes assigned groups."""
    import time
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    device = "cuda:0"

    # Load model
    print(f"[GPU {physical_gpu_id}] Loading model {args_dict['model']}...")
    tokenizer = AutoTokenizer.from_pretrained(args_dict["model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args_dict["model"],
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[GPU {physical_gpu_id}] Model loaded, processing {len(groups)} groups")

    SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the given passage.
You MUST wrap your answer in <answer></answer> tags. Be concise.

Example:
Question: What color is the sky?
<answer>blue</answer>"""

    # Import extract_answer
    from src.inference import extract_answer

    # Initialize results
    results = {s: {"predictions": {}, "latency": 0.0} for s in strategies_to_run}

    # Process each group
    for local_idx, (group_idx, group) in enumerate(zip(group_indices, groups)):
        if local_idx % 10 == 0:
            print(f"[GPU {physical_gpu_id}] Processing group {local_idx+1}/{len(groups)}")

        context = group[0]["context"]

        # Strategy: all_in_one
        if "all_in_one" in results:
            questions_text = "\n".join([f"{i+1}. {q['question']}" for i, q in enumerate(group)])
            prompt = f"Passage:\n{context}\n\nQuestions:\n{questions_text}\n\nAnswer each question with numbered responses. Wrap each answer in <answer></answer> tags.\nExample format:\n1. <answer>answer1</answer>\n2. <answer>answer2</answer>"

            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)

            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50 * len(group),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            results["all_in_one"]["latency"] += time.perf_counter() - start

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Try multiple parsing strategies
            answers = []

            # Strategy 1: Look for numbered <answer> tags
            numbered_answers = re.findall(r'\d+\.\s*<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
            if len(numbered_answers) >= len(group):
                answers = numbered_answers

            # Strategy 2: Look for any <answer> tags
            if not answers:
                answers = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)

            # Strategy 3: Look for numbered answers without tags
            if len(answers) < len(group):
                lines = response.strip().split('\n')
                numbered_lines = []
                for line in lines:
                    match = re.match(r'^(\d+)[\.\)]\s*(.+)', line.strip())
                    if match:
                        numbered_lines.append((int(match.group(1)), match.group(2).strip()))
                if len(numbered_lines) >= len(group):
                    numbered_lines.sort(key=lambda x: x[0])
                    answers = [a[1] for a in numbered_lines[:len(group)]]

            for i, q in enumerate(group):
                ans = answers[i].strip() if i < len(answers) else ""
                ans = re.sub(r'</?answer>', '', ans).strip()
                results["all_in_one"]["predictions"][q["qid"]] = (ans, len(ans) > 0)

        # Strategy: sequential
        if "sequential" in results:
            conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
            conversation.append({"role": "user", "content": f"Passage:\n{context}\n\nI will ask you questions about this passage."})
            conversation.append({"role": "assistant", "content": "I've read the passage. Please ask your questions."})

            for q in group:
                conversation.append({"role": "user", "content": f"Question: {q['question']}"})
                text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(device)

                start = time.perf_counter()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                results["sequential"]["latency"] += time.perf_counter() - start

                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                answer, valid = extract_answer(response, args_dict.get("dataset"))
                conversation.append({"role": "assistant", "content": response})
                results["sequential"]["predictions"][q["qid"]] = (answer, valid)

        # Strategy: batch (independent)
        if "batch" in results:
            for q in group:
                prompt = f"Passage:\n{context}\n\nQuestion: {q['question']}"
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(device)

                start = time.perf_counter()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                results["batch"]["latency"] += time.perf_counter() - start

                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                answer, valid = extract_answer(response, args_dict.get("dataset"))
                results["batch"]["predictions"][q["qid"]] = (answer, valid)

    print(f"[GPU {physical_gpu_id}] Done processing {len(groups)} groups")
    result_queue.put((worker_id, results))


def run_evaluation(args, dataset: str, group_size: int, all_questions: List[dict], output_dir: Path, gpu_ids: List[int]) -> dict:
    """Run evaluation for a specific dataset and group size."""
    num_gpus = len(gpu_ids)
    logger.info(f"\n{'='*80}")
    logger.info(f"Dataset: {dataset.upper()}, Group Size: {group_size}")
    logger.info(f"Total questions: {len(all_questions)}, Groups: {len(all_questions)//group_size}")
    logger.info(f"{'='*80}\n")

    strategies_to_run = [s.strip() for s in args.strategies.split(',')]
    groups = group_questions(all_questions, group_size)
    num_groups = len(groups)
    groups_per_gpu = (num_groups + num_gpus - 1) // num_gpus

    args_dict = {"model": args.model, "dataset": dataset}

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    start_time = time.perf_counter()

    for worker_id, physical_gpu_id in enumerate(gpu_ids):
        start_idx = worker_id * groups_per_gpu
        end_idx = min(start_idx + groups_per_gpu, num_groups)
        if start_idx >= num_groups:
            break

        p = ctx.Process(
            target=gpu_worker,
            args=(
                worker_id, physical_gpu_id,
                groups[start_idx:end_idx], list(range(start_idx, end_idx)),
                args_dict, strategies_to_run, result_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Collect results
    all_results = {s: {"predictions": {}, "latency": 0.0} for s in strategies_to_run}
    for _ in range(len(processes)):
        worker_id, worker_results = result_queue.get()
        for strategy, data in worker_results.items():
            all_results[strategy]["predictions"].update(data["predictions"])
            all_results[strategy]["latency"] += data["latency"]

    for p in processes:
        p.join()

    wall_time = time.perf_counter() - start_time

    # Compute metrics
    from src.evaluation import evaluate_predictions
    from src.models import Question

    summary = {"wall_time": wall_time}
    for strategy in strategies_to_run:
        preds = all_results[strategy]["predictions"]
        lookup = {}  # Dict[str, Question]
        predictions_for_eval = {}  # Dict[str, Tuple[str, bool]]

        for q in all_questions:
            qid = q["qid"]
            lookup[qid] = Question(
                qid=qid, text=q["question"], priority=1.0, answer_tokens=32,
                references=q.get("references", []),
            )
            if qid in preds:
                pred_ans, valid = preds[qid]
                predictions_for_eval[qid] = (pred_ans, valid)  # 2-tuple: (prediction, strict_valid)

        metrics = evaluate_predictions(predictions_for_eval, lookup, dataset=dataset)
        summary[strategy] = {
            "metrics": {
                "strict_acc": metrics["strict_acc"],
                "f1": metrics["f1"],
            },
            "latency": all_results[strategy]["latency"],
            "num_questions": len(all_questions),
        }
        logger.info(f"{strategy}: EM={metrics['strict_acc']:.3f}, F1={metrics['f1']:.3f}")

    return summary


STRATEGY_DISPLAY = {
    "batch": "Independent",
    "all_in_one": "All-in-One",
    "sequential": "Sequential",
}


def main():
    args = parse_args()
    gpu_ids = get_available_gpus()
    num_gpus = len(gpu_ids)

    datasets = [d.strip() for d in args.datasets.split(',')]
    group_sizes = [int(x.strip()) for x in args.group_sizes.split(',')]
    max_group_size = max(group_sizes)

    logger.info(f"Testing datasets: {datasets}")
    logger.info(f"Group sizes: {group_sizes}")
    logger.info(f"Using {num_gpus} GPUs: {gpu_ids}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset in datasets:
        logger.info(f"\n{'#'*80}")
        logger.info(f"DATASET: {dataset.upper()}")
        logger.info(f"{'#'*80}\n")

        try:
            all_questions, contexts = load_dataset_questions(
                dataset, args.seed, max_group_size, args.max_contexts
            )
            logger.info(f"Loaded {len(all_questions)} questions from {len(contexts)} contexts")
        except Exception as e:
            logger.error(f"Failed to load {dataset}: {e}")
            continue

        dataset_results = {}
        for group_size in group_sizes:
            if len(all_questions) % group_size != 0:
                # Trim questions to fit group size
                trim_to = (len(all_questions) // group_size) * group_size
                questions_subset = all_questions[:trim_to]
            else:
                questions_subset = all_questions

            if len(questions_subset) == 0:
                logger.warning(f"No questions for group size {group_size}")
                continue

            summary = run_evaluation(args, dataset, group_size, questions_subset, output_dir, gpu_ids)
            dataset_results[group_size] = summary

        all_results[dataset] = dataset_results

        # Save per-dataset results
        with open(output_dir / f"{dataset}_results.json", 'w') as f:
            json.dump(dataset_results, f, indent=2)

    # Generate summary table
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("# Multi-Dataset QA Evaluation\n")
        f.write(f"# Model: {args.model}\n")
        f.write(f"# Contexts per dataset: {args.max_contexts}\n\n")

        strategies = [s.strip() for s in args.strategies.split(',')]

        for dataset in datasets:
            if dataset not in all_results:
                continue

            f.write(f"\n## {dataset.upper()}\n\n")
            header = f"{'GroupSize':<10}"
            for s in strategies:
                header += f" | {STRATEGY_DISPLAY.get(s, s):>12}"
            f.write(header + "\n")
            f.write("-" * (12 + 15 * len(strategies)) + "\n")

            for gs in sorted(all_results[dataset].keys()):
                row = f"{gs:<10}"
                for s in strategies:
                    if s in all_results[dataset][gs]:
                        acc = all_results[dataset][gs][s]["metrics"]["strict_acc"]
                        row += f" | {acc*100:>11.1f}%"
                    else:
                        row += f" | {'--':>12}"
                f.write(row + "\n")

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    with open(summary_file, 'r') as f:
        print(f.read())

    # Save all results
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
