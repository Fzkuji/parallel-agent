#!/usr/bin/env python3
"""
Question grouping impact experiment.

Tests how grouping size affects strategy performance on the same set of questions.

Setup:
- Load contexts with at least N questions each (N = max group size)
- Test with different grouping sizes: 1, 4, 8, 12, 16 questions per group

For each grouping size:
- all_in_one: Processes N questions per prompt
- sequential: Processes N questions sequentially in one conversation
- batch: Processes N questions in parallel batch

All configurations test the exact same questions.
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
    parser.add_argument("--dataset", type=str, default="squad", choices=["squad", "cmb"],
                       help="Dataset to use (squad or cmb)")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-sizes", type=str, default="1,4,8,12,16",
                       help="Group sizes to test (questions per group)")
    parser.add_argument("--output-dir", type=str, default="outputs/grouping_study")
    parser.add_argument("--max-contexts", type=int, default=100,
                       help="Maximum number of contexts to use (default: 100)")
    parser.add_argument("--cross-batch-checkpoint", type=str, default=None,
                       help="Path to Cross-Batch checkpoint (if not provided, skip cross_batch strategy)")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Embedding model for Memory strategy")
    parser.add_argument("--strategies", type=str, default="batch,all_in_one,sequential,memory,cross_batch",
                       help="Comma-separated strategies to evaluate")

    return parser.parse_args()


def load_all_questions(dataset="squad", seed=42, min_questions=16, max_contexts=100):
    """Load questions from contexts with at least min_questions questions each."""
    if dataset == "squad":
        from src.datasets.squad import load_squad_groups
        contexts = load_squad_groups(
            split="validation",
            max_contexts=max_contexts,
            min_questions=min_questions,
            max_questions=1000,  # No upper limit
            seed=seed,
            fixed_question_count=min_questions  # Load exactly min_questions questions per context
        )
    elif dataset == "cmb":
        from src.datasets.cmb import load_cmb_exam_context_groups
        contexts = load_cmb_exam_context_groups(
            split="test",
            min_questions=min_questions,
            max_questions=min_questions,  # fixed count
            max_contexts=max_contexts,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    logger.info(f"Loaded {len(contexts)} contexts (each with {min_questions} questions)")

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


def load_memory_bank(dataset="squad", seed=42, max_samples=1000):
    """Load training set QA pairs as memory bank for few-shot retrieval."""
    if dataset == "squad":
        from src.datasets.squad import load_squad_groups
        contexts = load_squad_groups(
            split="train",
            max_contexts=max_samples,
            min_questions=1,
            max_questions=5,
            seed=seed,
        )
    elif dataset == "cmb":
        from src.datasets.cmb import load_cmb_exam_context_groups
        contexts = load_cmb_exam_context_groups(
            split="train",
            min_questions=1,
            max_questions=5,
            max_contexts=max_samples,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Flatten to QA pairs
    memory_bank = []
    for ctx in contexts:
        for q in ctx["questions"]:
            if q["references"]:  # Only include if has answer
                memory_bank.append({
                    "question": q["text"],
                    "answer": q["references"][0],
                    "context": ctx["context"],
                })
    logger.info(f"Loaded {len(memory_bank)} QA pairs for memory bank")
    return memory_bank


def build_memory_embeddings(memory_bank, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """Compute embeddings for memory bank questions."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")

    logger.info(f"Computing embeddings with {embedding_model}...")
    model = SentenceTransformer(embedding_model)
    questions = [item["question"] for item in memory_bank]
    embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    return embeddings, model


def retrieve_similar_examples(query_embedding, memory_embeddings, memory_bank, top_k=3):
    """Retrieve top-k most similar examples from memory bank."""
    import numpy as np
    # Normalize for cosine similarity
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    memory_norms = memory_embeddings / (np.linalg.norm(memory_embeddings, axis=1, keepdims=True) + 1e-9)
    similarities = np.dot(memory_norms, query_norm)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [memory_bank[i] for i in top_indices]


def group_questions(all_questions, group_size):
    """Group questions into chunks of group_size."""
    groups = []
    for i in range(0, len(all_questions), group_size):
        groups.append(all_questions[i:i+group_size])
    return groups


def run_evaluation(args, group_size, all_questions, output_dir, memory_bank=None, memory_embeddings=None, embedding_model=None):
    """Run evaluation with specific group size."""
    import time
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.models import Question
    from src.inference import extract_answer
    from src.evaluation import evaluate_predictions

    logger.info(f"\n{'='*80}")
    logger.info(f"GROUP SIZE: {group_size} questions per group")
    logger.info(f"Total questions: {len(all_questions)}, Groups: {len(all_questions)//group_size}")
    logger.info(f"{'='*80}\n")

    # Parse strategies to run
    strategies_to_run = [s.strip() for s in args.strategies.split(',')]

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

    # Initialize results for strategies
    results = {}
    for strategy in strategies_to_run:
        if strategy == "cross_batch" and not args.cross_batch_checkpoint:
            logger.info("Skipping cross_batch strategy (no checkpoint provided)")
            continue
        results[strategy] = {"predictions": {}, "latency": 0}

    # Initialize Cross-Batch generator if needed
    cross_batch_generator = None
    if "cross_batch" in results:
        try:
            from src.strategies.cross_batch import run_cross_batch_strategy
            from src.cross_batch import CrossBatchGenerator, SimpleCrossBatchGate
            # Load checkpoint
            checkpoint = torch.load(args.cross_batch_checkpoint, map_location="cuda:0")
            config = checkpoint.get("config", {})
            mix_method = config.get("module_type", "simple")
            hidden_size = model.config.hidden_size

            if mix_method == "simple":
                cross_batch_module = SimpleCrossBatchGate(hidden_size=hidden_size)
            else:
                from src.cross_batch import CrossBatchAttention
                cross_batch_module = CrossBatchAttention(hidden_size=hidden_size)

            cross_batch_module.load_state_dict(checkpoint["cross_batch_module"])
            cross_batch_module.to("cuda:0")

            cross_batch_generator = CrossBatchGenerator(
                model=model,
                tokenizer=tokenizer,
                cross_batch_module=cross_batch_module,
                mix_method=mix_method,
                mix_layer=config.get("mix_layer", -1),
                device="cuda:0",
            )
            logger.info("Cross-Batch generator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Cross-Batch: {e}")
            del results["cross_batch"]

    # Process each group
    for group_idx, group in enumerate(groups):
        if group_idx % 10 == 0:
            logger.info(f"Processing group {group_idx+1}/{len(groups)}")

        context = group[0]["context"]

        # Strategy: all_in_one - all questions in one prompt
        if "all_in_one" in results:
            questions_text = "\n".join([f"{i+1}. {q['question']}" for i, q in enumerate(group)])
            # Improved prompt to encourage numbered answers with tags
            prompt = f"Passage:\n{context}\n\nQuestions:\n{questions_text}\n\nAnswer each question with numbered responses. Wrap each answer in <answer></answer> tags.\nExample format:\n1. <answer>answer1</answer>\n2. <answer>answer2</answer>"

            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda:0")
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=96*len(group), do_sample=False, pad_token_id=tokenizer.pad_token_id)
            results["all_in_one"]["latency"] += time.perf_counter() - start

            raw_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Try multiple parsing strategies
            answers = []

            # Strategy 1: Look for numbered <answer> tags like "1. <answer>xxx</answer>"
            numbered_answers = re.findall(r'\d+\.\s*<answer>(.*?)</answer>', raw_text, re.DOTALL | re.IGNORECASE)
            if len(numbered_answers) >= len(group):
                answers = numbered_answers

            # Strategy 2: Look for any <answer> tags
            if not answers:
                answers = re.findall(r'<answer>(.*?)</answer>', raw_text, re.DOTALL | re.IGNORECASE)

            # Strategy 3: Look for numbered answers without tags like "1. answer1\n2. answer2"
            if len(answers) < len(group):
                # Try to extract by line numbers
                lines = raw_text.strip().split('\n')
                numbered_lines = []
                for line in lines:
                    match = re.match(r'^(\d+)[\.\)]\s*(.+)', line.strip())
                    if match:
                        numbered_lines.append((int(match.group(1)), match.group(2).strip()))
                if len(numbered_lines) >= len(group):
                    # Sort by number and extract answers
                    numbered_lines.sort(key=lambda x: x[0])
                    answers = [a[1] for a in numbered_lines[:len(group)]]

            for i, q in enumerate(group):
                answer = answers[i].strip() if i < len(answers) else ""
                # Clean up any remaining tags in the answer
                answer = re.sub(r'</?answer>', '', answer).strip()
                results["all_in_one"]["predictions"][q["qid"]] = (answer, len(answer) > 0)

        # Strategy: sequential - one by one in conversation
        if "sequential" in results:
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

        # Strategy: batch - all in parallel (Independent)
        if "batch" in results:
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

        # Strategy: memory - 3-shot with embedding retrieval
        if "memory" in results and memory_bank and memory_embeddings is not None and embedding_model:
            prompts = []
            for q in group:
                # Compute query embedding
                query_emb = embedding_model.encode([q["question"]], convert_to_numpy=True)[0]
                similar_examples = retrieve_similar_examples(query_emb, memory_embeddings, memory_bank, top_k=3)

                # Build few-shot prompt
                examples_text = ""
                for idx, ex in enumerate(similar_examples, 1):
                    examples_text += f"Example {idx}:\nPassage: {ex['context'][:500]}...\nQuestion: {ex['question']}\n<answer>{ex['answer']}</answer>\n\n"

                prompt = f"{examples_text}Now answer:\nPassage:\n{q['context']}\n\nQuestion: {q['question']}"
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(full_prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to("cuda:0")

            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            results["memory"]["latency"] += time.perf_counter() - start

            for i, q in enumerate(group):
                generated = outputs[i][inputs["input_ids"][i].shape[0]:]
                raw_text = tokenizer.decode(generated, skip_special_tokens=True)
                answer, valid = extract_answer(raw_text, args.dataset)
                results["memory"]["predictions"][q["qid"]] = (answer, valid)

        # Strategy: cross_batch - Cross-Batch with trained checkpoint
        if "cross_batch" in results and cross_batch_generator:
            from src.models import Question as Q
            questions_obj = [
                Q(qid=q["qid"], text=q["question"], priority=1.0,
                  answer_tokens=q["answer_tokens"], type_hint=None, references=q["references"])
                for q in group
            ]

            start = time.perf_counter()
            try:
                from src.strategies.cross_batch import run_cross_batch_strategy
                result = run_cross_batch_strategy(
                    background=context,
                    questions=questions_obj,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=96,
                    dataset=args.dataset,
                    cross_batch_generator=cross_batch_generator,
                    enable_cross_batch=True,
                )
                results["cross_batch"]["latency"] += time.perf_counter() - start

                for q in group:
                    answer = result.answers.get(q["qid"], "")
                    # Find validity from details
                    valid = True
                    for detail in result.details.get("questions", []):
                        if detail["question_id"] == q["qid"]:
                            valid = detail.get("strict_valid", True)
                            break
                    results["cross_batch"]["predictions"][q["qid"]] = (answer, valid)
            except Exception as e:
                logger.warning(f"Cross-Batch failed for group {group_idx}: {e}")
                results["cross_batch"]["latency"] += time.perf_counter() - start
                for q in group:
                    results["cross_batch"]["predictions"][q["qid"]] = ("", False)

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


STRATEGY_ORDER = ["batch", "all_in_one", "sequential", "memory", "cross_batch"]
STRATEGY_DISPLAY = {
    "batch": "Independent",
    "all_in_one": "All-in-One",
    "sequential": "Sequential",
    "memory": "Memory",
    "cross_batch": "Cross-Batch",
}


def main():
    args = parse_args()

    # Parse group sizes
    group_sizes = [int(x.strip()) for x in args.group_sizes.split(',')]
    max_group_size = max(group_sizes)
    logger.info(f"Testing group sizes: {group_sizes}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Will load contexts with at least {max_group_size} questions each\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions - use max group size as minimum questions per context
    all_questions, contexts = load_all_questions(
        dataset=args.dataset,
        seed=args.seed,
        min_questions=max_group_size,
        max_contexts=args.max_contexts,
    )

    # Load memory bank if memory strategy is enabled
    strategies_to_run = [s.strip() for s in args.strategies.split(',')]
    memory_bank = None
    memory_embeddings = None
    embedding_model = None
    if "memory" in strategies_to_run:
        logger.info("\nLoading memory bank for Memory strategy...")
        memory_bank = load_memory_bank(args.dataset, args.seed)
        memory_embeddings, embedding_model = build_memory_embeddings(memory_bank, args.embedding_model)

    # Find LCM of all group sizes to ensure divisibility
    from math import gcd
    from functools import reduce
    def lcm(a, b):
        return a * b // gcd(a, b)
    lcm_value = reduce(lcm, group_sizes)

    # Trim to largest multiple of LCM
    total_questions = len(all_questions)
    usable_questions = (total_questions // lcm_value) * lcm_value

    if usable_questions < total_questions:
        logger.info(f"Trimming from {total_questions} to {usable_questions} questions (divisible by all group sizes)")
        all_questions = all_questions[:usable_questions]
        total_questions = usable_questions

    logger.info(f"Using {len(contexts)} contexts with {total_questions} total questions")
    logger.info(f"All tests will use the same {total_questions} questions\n")

    # Run evaluation for each group size
    results_by_group_size = {}

    for group_size in group_sizes:
        if total_questions % group_size != 0:
            logger.warning(f"Skipping group size {group_size} - doesn't divide {total_questions} evenly")
            continue

        summary = run_evaluation(
            args, group_size, all_questions, output_dir,
            memory_bank=memory_bank,
            memory_embeddings=memory_embeddings,
            embedding_model=embedding_model,
        )
        results_by_group_size[group_size] = summary

        # Save results
        output_file = output_dir / f"group_size_{group_size}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

    # Generate final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*80}\n")

    # Determine which strategies were actually run
    if results_by_group_size:
        first_result = next(iter(results_by_group_size.values()))
        actual_strategies = [s for s in STRATEGY_ORDER if s in first_result]
    else:
        actual_strategies = []

    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("# Question Grouping Impact Experiment\n")
        f.write(f"# Model: {args.model}\n")
        f.write(f"# Dataset: {args.dataset} - {total_questions} questions from {len(contexts)} contexts ({max_group_size} questions each)\n")
        f.write(f"# All tests use the same {total_questions} questions, just grouped differently\n\n")

        f.write("## Results - EM (Exact Match)\n\n")
        # Header
        header = f"{'GroupSize':<10}"
        for strategy in actual_strategies:
            header += f" | {STRATEGY_DISPLAY[strategy]:>12}"
        header += f" | {'NumGroups':>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for gs in sorted(results_by_group_size.keys()):
            result = results_by_group_size[gs]
            num_groups = total_questions // gs
            line = f"{gs:<10}"
            for strategy in actual_strategies:
                if strategy in result:
                    acc = result[strategy]['metrics']['strict_acc']
                    line += f" | {acc:>12.3f}"
                else:
                    line += f" | {'--':>12}"
            line += f" | {num_groups:>10}"
            f.write(line + "\n")

        f.write("\n## Latency (seconds)\n\n")
        header = f"{'GroupSize':<10}"
        for strategy in actual_strategies:
            header += f" | {STRATEGY_DISPLAY[strategy]:>12}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for gs in sorted(results_by_group_size.keys()):
            result = results_by_group_size[gs]
            line = f"{gs:<10}"
            for strategy in actual_strategies:
                if strategy in result:
                    lat = result[strategy]['latency']
                    line += f" | {lat:>12.2f}"
                else:
                    line += f" | {'--':>12}"
            f.write(line + "\n")

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
