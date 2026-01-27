#!/usr/bin/env python3
"""
Experiment 3: N-Shot Effect Analysis

Tests how the number of consecutive questions (N-shot) affects model performance
across different question types.

Research Question:
  Does answering multiple questions sequentially improve performance through
  context accumulation and in-context learning?

Setup:
  For each dataset, test N = 0, 1, 2, 3, 4, 5 shot configurations:
    - 0-shot: Independent answering (baseline, batch mode)
    - N-shot: Answer N questions sequentially with context accumulation

Datasets and their characteristics:
  - SQuAD: Extractive QA, short phrase answers, shared context
  - DROP: Numeric reasoning, number answers, shared context
  - TriviaQA: Open domain QA, entity answers, no shared context
  - MMLU: Multiple choice, letter answers (A/B/C/D), grouped by subject
  - GSM8K: Math reasoning, numeric answers, independent problems

Metrics:
  - Overall Accuracy by N-shot
  - Per-position accuracy within N-shot sessions
  - N-shot benefit = N-shot_acc - 0-shot_acc

Usage:
  # Auto-detect GPUs and run in parallel:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python exp3_nshot_effect.py --model Qwen/Qwen2.5-7B-Instruct --datasets squad drop

  # Single GPU:
  CUDA_VISIBLE_DEVICES=0 python exp3_nshot_effect.py --model Qwen/Qwen2.5-7B-Instruct --datasets squad
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import argparse
import multiprocessing as mp
from typing import Dict, List, Any, Tuple


def get_available_gpus() -> List[int]:
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


def load_dataset_for_nshot(
    dataset_name: str,
    split: str,
    max_contexts: int,
    n_shot: int,
    seed: int = 42,
) -> Tuple[List[Dict], str]:
    """Load dataset with exactly n_shot questions per context."""
    from src.datasets.squad import load_squad_groups
    from src.datasets.drop import load_drop_groups
    from src.datasets.triviaqa import load_triviaqa_groups
    from src.datasets.mmlu import load_mmlu
    from src.datasets.gsm8k import load_gsm8k

    effective_n = max(1, n_shot)

    if dataset_name == "squad":
        data = load_squad_groups(
            split=split, max_contexts=max_contexts,
            min_questions=effective_n, max_questions=effective_n, seed=seed
        )
        return data, "squad"
    elif dataset_name == "drop":
        data = load_drop_groups(
            split=split, max_contexts=max_contexts,
            min_questions=effective_n, max_questions=effective_n, seed=seed
        )
        return data, "drop"
    elif dataset_name == "triviaqa":
        data = load_triviaqa_groups(
            split=split, max_groups=max_contexts,
            min_questions=effective_n, max_questions=effective_n, seed=seed
        )
        return data, "triviaqa"
    elif dataset_name == "mmlu":
        data = load_mmlu(
            split=split, max_contexts=max_contexts,
            min_questions=effective_n, max_questions=effective_n, seed=seed
        )
        return data, "mmlu"
    elif dataset_name == "gsm8k":
        gsm_split = "test" if split == "validation" else split
        data = load_gsm8k(
            split=gsm_split, max_contexts=max_contexts,
            min_questions=effective_n, max_questions=effective_n, seed=seed
        )
        return data, "gsm8k"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def gpu_worker(
    worker_id: int,
    gpu_id: int,
    data_chunk: List[Dict],
    n_shot: int,
    args_dict: Dict,
    result_queue: mp.Queue,
):
    """Worker process that runs on a specific GPU."""
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.models import Question
    from src.strategies import run_sequential_strategy
    from src.prompts import build_single_prompt, MULTIPLE_CHOICE_DATASETS
    from src.inference import extract_answer
    from src.evaluation import evaluate_predictions
    from src.evaluation.basic import compute_em, compute_choice_accuracy

    # Set device
    device = f"cuda:{gpu_id}"
    print(f"[Worker {worker_id}] Loading model on GPU {gpu_id}...")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args_dict["model"])
    model = AutoModelForCausalLM.from_pretrained(
        args_dict["model"],
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = args_dict["dataset"]
    max_new_tokens = args_dict["max_new_tokens"]
    batch_size = args_dict["batch_size"]

    print(f"[Worker {worker_id}] Processing {len(data_chunk)} contexts...")

    # Run inference
    if n_shot == 0:
        # 0-shot: batched inference
        result = run_batched_0shot_worker(
            data_chunk, tokenizer, model, max_new_tokens, dataset, batch_size, worker_id
        )
    else:
        # N-shot: sequential inference
        result = run_nshot_worker(
            data_chunk, n_shot, tokenizer, model, max_new_tokens, dataset, worker_id
        )

    print(f"[Worker {worker_id}] Done. Correct: {result['total_correct']}/{result['total_questions']}")
    result_queue.put((worker_id, result))


def run_batched_0shot_worker(
    data: List[Dict],
    tokenizer,
    model,
    max_new_tokens: int,
    dataset: str,
    batch_size: int,
    worker_id: int,
) -> Dict[str, Any]:
    """Run 0-shot batched inference on a worker."""
    import torch
    from tqdm import tqdm
    from src.models import Question
    from src.prompts import build_single_prompt, MULTIPLE_CHOICE_DATASETS
    from src.inference import extract_answer
    from src.evaluation import evaluate_predictions
    from src.evaluation.basic import compute_em, compute_choice_accuracy

    all_items = []
    question_lookup = {}

    for ctx_idx, context_data in enumerate(data):
        background = context_data["context"]
        for q_data in context_data["questions"]:
            qid = f"w{worker_id}_ctx{ctx_idx}_{q_data['qid']}"
            question = Question(
                qid=qid,
                text=q_data.get("text", q_data.get("question", "")),
                priority=1.0,
                answer_tokens=q_data.get("answer_tokens", 32),
                type_hint=None,
                references=q_data.get("references", [])
            )
            question_lookup[qid] = question
            all_items.append({
                "qid": qid,
                "question": question,
                "background": background,
            })

    if not all_items:
        return {
            "total_correct": 0, "total_questions": 0,
            "position_metrics": {}, "details": [],
        }

    # Build prompts
    all_prompts = []
    for item in all_items:
        system_prompt, user_prompt = build_single_prompt(
            item["background"], item["question"], dataset
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_prompts.append(prompt)

    # Batch inference
    answer_records = {}
    detail_records = []

    for batch_start in tqdm(range(0, len(all_prompts), batch_size),
                            desc=f"[W{worker_id}] 0-shot", disable=(worker_id != 0)):
        batch_end = min(batch_start + batch_size, len(all_prompts))
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_items = all_items[batch_start:batch_end]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )

        for i, item in enumerate(batch_items):
            input_len = inputs["input_ids"][i].shape[0]
            generated = outputs[i][input_len:]
            response = tokenizer.decode(generated, skip_special_tokens=True)

            answer, valid = extract_answer(response, dataset)
            answer_records[item["qid"]] = (answer, valid)

            refs = item["question"].references
            if dataset in MULTIPLE_CHOICE_DATASETS:
                correct = compute_choice_accuracy(answer, refs) > 0
            else:
                correct = compute_em(answer, refs) > 0

            detail_records.append({
                "qid": item["qid"],
                "question": item["question"].text,
                "references": refs,
                "response": response,
                "extracted_answer": answer,
                "valid": valid,
                "correct": correct,
                "position": 1,
            })

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
    total_questions = len(answer_records)
    total_correct = int(round(metrics.get("strict_acc", 0) * total_questions))

    return {
        "total_correct": total_correct,
        "total_questions": total_questions,
        "position_metrics": {},
        "details": detail_records,
    }


def run_nshot_worker(
    data: List[Dict],
    n_shot: int,
    tokenizer,
    model,
    max_new_tokens: int,
    dataset: str,
    worker_id: int,
) -> Dict[str, Any]:
    """Run N-shot sequential inference on a worker."""
    from tqdm import tqdm
    from src.models import Question
    from src.strategies import run_sequential_strategy

    position_correct = {}
    position_total = {}
    all_correct = 0
    all_total = 0
    detail_records = []

    for ctx_idx, context_data in enumerate(tqdm(data, desc=f"[W{worker_id}] {n_shot}-shot",
                                                  disable=(worker_id != 0))):
        background = context_data["context"]
        questions = [
            Question(
                qid=q["qid"],
                text=q.get("text", q.get("question", "")),
                priority=1.0,
                answer_tokens=q.get("answer_tokens", 32),
                type_hint=None,
                references=q.get("references", [])
            )
            for q in context_data["questions"]
        ]

        result = run_sequential_strategy(
            background=background,
            questions=questions,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
            dataset=dataset,
        )

        n_questions = len(questions)
        n_correct = int(round(result.metrics.get("strict_acc", 0) * n_questions))
        all_correct += n_correct
        all_total += n_questions

        if "turns" in result.details:
            for i, turn in enumerate(result.details["turns"]):
                pos = i + 1
                if pos not in position_correct:
                    position_correct[pos] = 0
                    position_total[pos] = 0
                position_total[pos] += 1
                is_correct = turn.get("strict_valid", False)
                if is_correct:
                    position_correct[pos] += 1

                detail_records.append({
                    "qid": turn.get("question_id", f"w{worker_id}_ctx{ctx_idx}_q{i}"),
                    "question": turn.get("question", questions[i].text if i < len(questions) else ""),
                    "references": turn.get("gold_answers", questions[i].references if i < len(questions) else []),
                    "response": turn.get("raw_response", ""),
                    "extracted_answer": turn.get("final_answer", ""),
                    "valid": turn.get("strict_valid", False),
                    "correct": is_correct,
                    "position": pos,
                })

    position_metrics = {}
    for pos in sorted(position_correct.keys()):
        if position_total[pos] > 0:
            position_metrics[pos] = {
                "accuracy": position_correct[pos] / position_total[pos],
                "correct": position_correct[pos],
                "total": position_total[pos],
            }

    return {
        "total_correct": all_correct,
        "total_questions": all_total,
        "position_metrics": position_metrics,
        "details": detail_records,
    }


def run_nshot_parallel(
    data: List[Dict],
    n_shot: int,
    args_dict: Dict,
    gpu_ids: List[int],
) -> Dict[str, Any]:
    """Run N-shot experiment in parallel across GPUs."""
    num_gpus = len(gpu_ids)
    num_contexts = len(data)
    contexts_per_gpu = (num_contexts + num_gpus - 1) // num_gpus

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for worker_id, gpu_id in enumerate(gpu_ids):
        start_idx = worker_id * contexts_per_gpu
        end_idx = min(start_idx + contexts_per_gpu, num_contexts)
        if start_idx >= num_contexts:
            break

        data_chunk = data[start_idx:end_idx]
        p = ctx.Process(
            target=gpu_worker,
            args=(worker_id, gpu_id, data_chunk, n_shot, args_dict, result_queue),
        )
        p.start()
        processes.append(p)

    # Collect results
    all_results = []
    for _ in range(len(processes)):
        worker_id, result = result_queue.get()
        all_results.append(result)

    for p in processes:
        p.join()

    # Aggregate results
    total_correct = sum(r["total_correct"] for r in all_results)
    total_questions = sum(r["total_questions"] for r in all_results)
    accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    # Aggregate position metrics
    combined_position = {}
    for r in all_results:
        for pos, pm in r["position_metrics"].items():
            if pos not in combined_position:
                combined_position[pos] = {"correct": 0, "total": 0}
            combined_position[pos]["correct"] += pm["correct"]
            combined_position[pos]["total"] += pm["total"]

    for pos in combined_position:
        combined_position[pos]["accuracy"] = (
            combined_position[pos]["correct"] / combined_position[pos]["total"]
            if combined_position[pos]["total"] > 0 else 0.0
        )

    # Collect all details
    all_details = []
    for r in all_results:
        all_details.extend(r.get("details", []))

    return {
        "n_shot": n_shot,
        "accuracy": accuracy,
        "total_questions": total_questions,
        "total_correct": total_correct,
        "position_metrics": combined_position,
        "details": all_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: N-Shot Effect Analysis")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--datasets", type=str, nargs="+",
                       default=["squad", "drop", "triviaqa", "mmlu", "gsm8k"])
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--max-contexts", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs/preliminary/exp3")
    args = parser.parse_args()

    # Get available GPUs
    gpu_ids = get_available_gpus()
    print(f"Using GPUs: {gpu_ids}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shots = sorted(args.shots)
    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}\n")

        dataset_results = {}

        for n_shot in shots:
            print(f"\n--- Testing {n_shot}-shot ---")

            data, dataset_key = load_dataset_for_nshot(
                dataset_name, split="validation",
                max_contexts=args.max_contexts, n_shot=n_shot, seed=42
            )
            print(f"Loaded {len(data)} contexts")

            args_dict = {
                "model": args.model,
                "dataset": dataset_key,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
            }

            result = run_nshot_parallel(data, n_shot, args_dict, gpu_ids)

            dataset_results[str(n_shot)] = {
                "n_shot": n_shot,
                "accuracy": result["accuracy"],
                "total_questions": result["total_questions"],
                "total_correct": result["total_correct"],
                "position_metrics": result["position_metrics"],
            }

            # Save detailed results
            if result["details"]:
                details_file = output_dir / f"details_{dataset_name}_{n_shot}shot.jsonl"
                with open(details_file, 'w') as f:
                    for record in result["details"]:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(f"  Details saved to: {details_file}")

            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Correct: {result['total_correct']}/{result['total_questions']}")
            if result["position_metrics"]:
                print(f"  Position accuracy:")
                for pos in sorted(result["position_metrics"].keys()):
                    pm = result["position_metrics"][pos]
                    print(f"    Position {pos}: {pm['accuracy']:.4f} ({pm['correct']}/{pm['total']})")

        # Compute benefits
        baseline_acc = dataset_results["0"]["accuracy"]
        for n_shot_str, res in dataset_results.items():
            res["benefit"] = res["accuracy"] - baseline_acc

        all_results[dataset_name] = dataset_results

        print(f"\n{'='*40}")
        print(f"Summary for {dataset_name}:")
        print(f"{'='*40}")
        print(f"{'N-shot':<10} {'Accuracy':>10} {'Benefit':>10}")
        print(f"{'-'*40}")
        for n_shot in shots:
            r = dataset_results[str(n_shot)]
            print(f"{n_shot:<10} {r['accuracy']:>9.2%} {r['benefit']:>+9.2%}")
        print(f"{'='*40}\n")

    # Save results
    output_file = output_dir / "exp3_nshot_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model": args.model,
            "datasets": args.datasets,
            "shots": shots,
            "max_contexts": args.max_contexts,
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*70}")
    header = f"{'Dataset':<12}"
    for n_shot in shots:
        header += f" {n_shot}-shot".rjust(10)
    print(header)
    print(f"{'-'*70}")
    for dataset_name in args.datasets:
        row = f"{dataset_name:<12}"
        for n_shot in shots:
            acc = all_results[dataset_name][str(n_shot)]["accuracy"]
            row += f" {acc:>9.2%}"
        print(row)
    print(f"{'='*70}")

    print(f"\nN-shot Benefit (vs 0-shot):")
    print(f"{'-'*70}")
    header = f"{'Dataset':<12}"
    for n_shot in shots[1:]:
        header += f" {n_shot}-shot".rjust(10)
    print(header)
    print(f"{'-'*70}")
    for dataset_name in args.datasets:
        row = f"{dataset_name:<12}"
        for n_shot in shots[1:]:
            benefit = all_results[dataset_name][str(n_shot)]["benefit"]
            row += f" {benefit:>+9.2%}"
        print(row)
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
