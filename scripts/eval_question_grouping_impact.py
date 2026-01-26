#!/usr/bin/env python3
"""
Question grouping impact experiment.

Tests how grouping size affects strategy performance on the same 180 questions.

Setup:
- Load 9 contexts with 20 questions each (180 total questions)
- Test with different grouping sizes: 1, 5, 10, 20 questions per group

For each grouping size:
- all_in_one: Processes N questions per prompt
- sequential: Processes N questions sequentially in one conversation
- batch: Processes N questions in parallel batch
- collab_llm: Processes N questions with dependency ordering

All configurations test the exact same 180 questions.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Question grouping impact experiment")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="squad")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-sizes", type=str, default="1,5,10,20",
                       help="Group sizes to test (questions per group)")
    parser.add_argument("--output-dir", type=str, default="outputs/grouping_study")

    return parser.parse_args()


def load_all_questions(seed=42):
    """Load all 180 questions from the 9 contexts with 20 questions each."""
    from src.datasets.squad import load_squad_groups

    contexts = load_squad_groups(
        split="validation",
        max_contexts=1000,
        min_questions=20,
        max_questions=20,
        seed=seed,
        fixed_question_count=20  # Load all 20 questions
    )

    logger.info(f"Loaded {len(contexts)} contexts")

    # Flatten all questions into a single list
    all_questions = []
    for ctx_idx, ctx in enumerate(contexts):
        for q in ctx["questions"]:
            all_questions.append({
                "qid": f"ctx{ctx_idx}_{q['qid']}",
                "question": q["text"],
                "context": ctx["context"],
                "references": q["references"],
                "answer_tokens": q.get("answer_tokens", 12),
                "context_idx": ctx_idx,
                "title": ctx.get("title", "Unknown"),
            })

    logger.info(f"Total questions: {len(all_questions)}")
    return all_questions, contexts


def group_questions(all_questions, group_size):
    """Group questions into chunks of group_size."""
    groups = []
    for i in range(0, len(all_questions), group_size):
        groups.append(all_questions[i:i+group_size])
    return groups


def run_evaluation(args, group_size, all_questions, output_dir):
    """Run evaluation with specific group size."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.models import Question
    from src.inference import extract_answer
    from src.evaluation import evaluate_predictions

    logger.info(f"\n{'='*80}")
    logger.info(f"GROUP SIZE: {group_size} questions per group")
    logger.info(f"Total questions: {len(all_questions)}, Groups: {len(all_questions)//group_size}")
    logger.info(f"{'='*80}\n")

    # Group questions
    groups = group_questions(all_questions, group_size)

    # Load model
    logger.info("Loading model...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded\n")

    SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the given passage.
You MUST wrap your answer in <answer></answer> tags. Be concise.

Example:
Question: What color is the sky?
<answer>blue</answer>"""

    results = {
        "all_in_one": {"predictions": {}, "latency": 0},
        "sequential": {"predictions": {}, "latency": 0},
        "batch": {"predictions": {}, "latency": 0},
    }

    # Process each group
    for group_idx, group in enumerate(groups):
        if group_idx % 10 == 0:
            logger.info(f"Processing group {group_idx+1}/{len(groups)}")

        # Strategy 1: all_in_one - all questions in one prompt
        import time
        context = group[0]["context"]
        questions_text = "\n".join([f"{i+1}. {q['question']}" for i, q in enumerate(group)])
        prompt = f"Passage:\n{context}\n\nQuestions:\n{questions_text}\n\nPlease answer each question."

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda:0")
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=96*len(group), do_sample=False, pad_token_id=tokenizer.pad_token_id)
        results["all_in_one"]["latency"] += time.perf_counter() - start

        raw_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Parse answers (simplified - just extract each <answer>...</answer>)
        import re
        answers = re.findall(r'<answer>(.*?)</answer>', raw_text, re.DOTALL)
        for i, q in enumerate(group):
            answer = answers[i].strip() if i < len(answers) else ""
            results["all_in_one"]["predictions"][q["qid"]] = (answer, True)

        # Strategy 2: sequential - one by one in conversation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for i, q in enumerate(group):
            prompt = f"Passage:\n{q['context']}\n\nQuestion: {q['question']}" if i == 0 else f"Question: {q['question']}"
            messages.append({"role": "user", "content": prompt})

            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda:0")

            start = time.perf_counter()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            results["sequential"]["latency"] += time.perf_counter() - start

            raw_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            answer, valid = extract_answer(raw_text, args.dataset)
            results["sequential"]["predictions"][q["qid"]] = (answer, valid)

            messages.append({"role": "assistant", "content": raw_text})

        # Strategy 3: batch - all in parallel
        prompts = []
        for q in group:
            prompt = f"Passage:\n{q['context']}\n\nQuestion: {q['question']}"
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(full_prompt)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda:0")

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        results["batch"]["latency"] += time.perf_counter() - start

        for i, q in enumerate(group):
            generated = outputs[i][inputs["input_ids"][i].shape[0]:]
            raw_text = tokenizer.decode(generated, skip_special_tokens=True)
            answer, valid = extract_answer(raw_text, args.dataset)
            results["batch"]["predictions"][q["qid"]] = (answer, valid)

    # Evaluate all strategies
    logger.info("\nEvaluating predictions...")

    question_lookup = {
        q["qid"]: Question(
            qid=q["qid"],
            text=q["question"],
            priority=1.0,
            answer_tokens=q["answer_tokens"],
            type_hint=None,
            references=q["references"],
        )
        for q in all_questions
    }

    summary = {}
    for strategy_name, result_data in results.items():
        metrics = evaluate_predictions(result_data["predictions"], question_lookup, dataset=args.dataset)
        summary[strategy_name] = {
            "metrics": metrics,
            "latency": result_data["latency"],
        }
        logger.info(f"{strategy_name}: EM={metrics['strict_acc']:.3f}, Latency={result_data['latency']:.2f}s")

    return summary


def main():
    args = parse_args()

    # Parse group sizes
    group_sizes = [int(x.strip()) for x in args.group_sizes.split(',')]
    logger.info(f"Testing group sizes: {group_sizes}")
    logger.info(f"All tests will use the same 180 questions\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all 180 questions once
    all_questions, contexts = load_all_questions(args.seed)

    # Verify we have 180 questions
    if len(all_questions) != 180:
        logger.error(f"Expected 180 questions, got {len(all_questions)}")
        return

    # Run evaluation for each group size
    results_by_group_size = {}

    for group_size in group_sizes:
        if 180 % group_size != 0:
            logger.warning(f"Skipping group size {group_size} - doesn't divide 180 evenly")
            continue

        summary = run_evaluation(args, group_size, all_questions, output_dir)
        results_by_group_size[group_size] = summary

        # Save results
        output_file = output_dir / f"group_size_{group_size}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

    # Generate final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*80}\n")

    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("# Question Grouping Impact Experiment\n")
        f.write("# Dataset: 180 questions from 9 contexts (20 questions each)\n")
        f.write("# All tests use the same 180 questions, just grouped differently\n\n")

        f.write("## Results - EM (Exact Match)\n\n")
        f.write(f"{'GroupSize':<10} | {'all_in_one':>11} | {'sequential':>11} | {'batch':>11} | {'NumGroups':>10}\n")
        f.write("-" * 70 + "\n")

        for gs in sorted(results_by_group_size.keys()):
            result = results_by_group_size[gs]
            num_groups = 180 // gs
            f.write(f"{gs:<10} | ")
            f.write(f"{result['all_in_one']['metrics']['strict_acc']:>11.3f} | ")
            f.write(f"{result['sequential']['metrics']['strict_acc']:>11.3f} | ")
            f.write(f"{result['batch']['metrics']['strict_acc']:>11.3f} | ")
            f.write(f"{num_groups:>10}\n")

        f.write("\n## Latency (seconds)\n\n")
        f.write(f"{'GroupSize':<10} | {'all_in_one':>11} | {'sequential':>11} | {'batch':>11}\n")
        f.write("-" * 60 + "\n")

        for gs in sorted(results_by_group_size.keys()):
            result = results_by_group_size[gs]
            f.write(f"{gs:<10} | ")
            f.write(f"{result['all_in_one']['latency']:>11.2f} | ")
            f.write(f"{result['sequential']['latency']:>11.2f} | ")
            f.write(f"{result['batch']['latency']:>11.2f}\n")

    logger.info(f"Summary saved to {summary_file}\n")
    with open(summary_file, 'r') as f:
        print(f.read())

    # Save all results
    all_results_file = output_dir / "all_results.json"
    with open(all_results_file, 'w') as f:
        json.dump(results_by_group_size, f, indent=2)
    logger.info(f"\nAll results saved to {all_results_file}")


if __name__ == "__main__":
    main()
