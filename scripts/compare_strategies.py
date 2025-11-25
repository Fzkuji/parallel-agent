from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval import compute_em

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import (
    BertAttentionDependencyGenerator,
    HeuristicDependencyGenerator,
    LocalLLMDependencyGenerator,
    load_hotpot_groups,
    load_squad_random_questions,
    build_questions_from_group,
    load_squad_groups,
    Question,
    set_think_tokens,
)
from src.strategies import (
    StrategyResult,
    run_all_in_one_strategy,
    run_all_in_one_multi_strategy,
    print_answer_table,
    run_dependency_batch_strategy,
    run_full_batch_strategy,
    run_sequential_strategy,
    run_batch_multi_strategy,
    run_sequential_multi_strategy,
    summarize_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sequential, batch, and dependency-aware QA strategies with optional BERT dependencies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["squad", "hotpot"], default="squad", help="Dataset to evaluate.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B", help="Hugging Face model identifier or local path.")
    parser.add_argument("--split", default="train", help="SQuAD split to sample.")
    parser.add_argument("--context-count", type=int, default=3, help="Number of contexts to process.")
    parser.add_argument("--min-questions", type=int, default=3, help="Minimum questions per context.")
    parser.add_argument("--max-questions", type=int, default=5, help="Maximum questions per context.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for sampling contexts.")
    parser.add_argument(
        "--squad-random-questions",
        action="store_true",
        help="For SQuAD, sample individual questions randomly instead of grouping by shared context.",
    )
    parser.add_argument("--hotpot-subset", default="distractor", help="HotpotQA subset (e.g., distractor).")

    parser.add_argument("--cost-weight", type=float, default=0.01, help="Cost penalty weight for dependency selection.")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum edge confidence.")
    parser.add_argument("--max-dependencies", type=int, default=3, help="Max dependencies per question.")
    parser.add_argument("--total-cost-budget", type=int, default=None, help="Optional global dependency cost budget.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max new tokens for answer generation.")

    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated list of strategies to test (e.g., 'all_in_one,sequential,batch'). "
             "Available: all_in_one, sequential, batch, parallel, parallel_bert. "
             "If not specified, all strategies will be tested.",
    )
    parser.add_argument("--no-llm-deps", action="store_true", help="Force heuristic dependency generator.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to dump metrics as JSON.")
    parser.add_argument("--no-think-tokens", action="store_true", help="Disable <think></think> markers.")
    parser.add_argument("--use-think-tokens", action="store_true", help="Enable <think></think> markers.")
    parser.add_argument("--verbose-debug", action="store_true", help="Print detailed prompts and responses.")
    parser.add_argument("--deterministic", action="store_true", help="Enable strong determinism (slower but stable).")

    parser.add_argument("--bert-model-name", default="bert-base-uncased", help="Encoder-only model for attention DAGs.")
    parser.add_argument(
        "--bert-attention-threshold",
        type=float,
        default=0.02,
        help="Minimum attention mass to create a candidate edge.",
    )
    parser.add_argument(
        "--bert-dependency-threshold",
        type=float,
        default=None,
        help="Minimum confidence for BERT dependencies (defaults to attention threshold).",
    )
    parser.add_argument(
        "--bert-max-question-tokens",
        type=int,
        default=64,
        help="Max wordpiece tokens per question for the attention encoder.",
    )
    parser.add_argument(
        "--bert-max-seq-length",
        type=int,
        default=512,
        help="Max packed token length for the attention encoder.",
    )
    parser.add_argument(
        "--bert-cost-weight",
        type=float,
        default=0.0,
        help="Cost penalty weight for BERT-based dependency selection.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total shards/world size for data-parallel eval (defaults to WORLD_SIZE or 1).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Shard index/rank (defaults to LOCAL_RANK or 0).",
    )
    return parser.parse_args()


def resolve_dependency_generators(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> tuple:
    if args.no_llm_deps:
        dep_generator = HeuristicDependencyGenerator()
        logging.info("Using heuristic dependency generator.")
    else:
        dep_generator = LocalLLMDependencyGenerator(tokenizer, model)
        logging.info("Using local LLM dependency generator.")

    bert_conf_threshold = args.bert_dependency_threshold or args.bert_attention_threshold
    logging.info(
        "Initializing BERT attention dependency generator (%s, threshold=%.4f)",
        args.bert_model_name,
        args.bert_attention_threshold,
    )
    bert_dep_generator = BertAttentionDependencyGenerator(
        model_name=args.bert_model_name,
        attention_threshold=args.bert_attention_threshold,
        max_question_tokens=args.bert_max_question_tokens,
        max_total_tokens=args.bert_max_seq_length,
    )
    return dep_generator, bert_dep_generator, bert_conf_threshold


ALL_STRATEGIES = ["all_in_one", "sequential", "batch", "parallel", "parallel_bert"]


def run_all_strategies(
    background: str,
    questions,
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    dep_generator,
    bert_dep_generator,
    args: argparse.Namespace,
    bert_conf_threshold: float,
    selected_strategies: Optional[List[str]] = None,
) -> List[StrategyResult]:
    # Default to all strategies if none specified
    if selected_strategies is None:
        selected_strategies = ALL_STRATEGIES

    results: List[StrategyResult] = []

    # Multi-context mode: if questions is None and items provided, use items-based strategies
    if background is None and isinstance(questions, list) and questions and isinstance(questions[0], dict) and "context" in questions[0]:
        items = questions
        if "all_in_one" in selected_strategies:
            results.append(run_all_in_one_multi_strategy(
                items,
                tokenizer,
                model,
                max_new_tokens=args.max_new_tokens,
                strategy_name="all_in_one",
            ))
        if "sequential" in selected_strategies:
            results.append(run_sequential_multi_strategy(
                items,
                tokenizer,
                model,
                max_new_tokens=args.max_new_tokens,
                strategy_name="sequential",
            ))
        if "batch" in selected_strategies:
            results.append(run_batch_multi_strategy(
                items,
                tokenizer,
                model,
                max_new_tokens=args.max_new_tokens,
                strategy_name="batch",
            ))
        # For dependency strategies, avoid duplicating all contexts per question.
        # Embed each question's own context inline, and leave shared background empty.
        if "parallel" in selected_strategies or "parallel_bert" in selected_strategies:
            merged_background = ""
            dep_questions = [
                Question(
                    qid=item["qid"],
                    text=f"Context:\n{item['context']}\nQuestion: {item['question']}",
                    priority=1.0,
                    answer_tokens=item.get("answer_tokens", 12),
                    type_hint=None,
                    references=item.get("references", []),
                )
                for item in items
            ]
            if "parallel" in selected_strategies:
                results.append(run_dependency_batch_strategy(
                    merged_background,
                    dep_questions,
                    generator=dep_generator,
                    tokenizer=tokenizer,
                    model=model,
                    cost_weight=args.cost_weight,
                    min_confidence=args.min_confidence,
                    max_dependencies=args.max_dependencies,
                    total_cost_budget=args.total_cost_budget,
                    max_new_tokens=args.max_new_tokens,
                    strategy_name="parallel",
                ))
            if "parallel_bert" in selected_strategies:
                results.append(run_dependency_batch_strategy(
                    merged_background,
                    dep_questions,
                    generator=bert_dep_generator,
                    tokenizer=tokenizer,
                    model=model,
                    cost_weight=args.cost_weight,
                    min_confidence=args.min_confidence,
                    max_dependencies=args.max_dependencies,
                    total_cost_budget=args.total_cost_budget,
                    max_new_tokens=args.max_new_tokens,
                    strategy_name="parallel_bert",
                ))
        return results

    if "all_in_one" in selected_strategies:
        results.append(run_all_in_one_strategy(
            background,
            questions,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
        ))
    if "sequential" in selected_strategies:
        results.append(run_sequential_strategy(
            background,
            questions,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
        ))
    if "batch" in selected_strategies:
        results.append(run_full_batch_strategy(
            background,
            questions,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
        ))
    if "parallel" in selected_strategies:
        results.append(run_dependency_batch_strategy(
            background,
            questions,
            dep_generator,
            tokenizer,
            model,
            cost_weight=args.cost_weight,
            min_confidence=args.min_confidence,
            max_dependencies=args.max_dependencies,
            total_cost_budget=args.total_cost_budget,
            max_new_tokens=args.max_new_tokens,
            strategy_name="parallel",
        ))
    if "parallel_bert" in selected_strategies:
        results.append(run_dependency_batch_strategy(
            background,
            questions,
            bert_dep_generator,
            tokenizer,
            model,
            cost_weight=args.bert_cost_weight,
            min_confidence=bert_conf_threshold,
            max_dependencies=args.max_dependencies,
            total_cost_budget=args.total_cost_budget,
            max_new_tokens=args.max_new_tokens,
            strategy_name="parallel_bert",
        ))
    return results


def compute_aggregate_metrics(serialized_contexts: List[dict]) -> str:
    """Compute and format aggregate metrics across all contexts and strategies."""
    preferred_order = [
        "all_in_one",
        "sequential",
        "batch",
        "parallel",
        "parallel_bert",
    ]
    strategy_totals: Dict[str, Dict[str, float]] = {}

    for ctx in serialized_contexts:
        for strategy in ctx.get("strategies", []):
            name = strategy.get("name", "unknown")
            metrics = strategy.get("metrics", {})
            stats = strategy_totals.setdefault(
                name,
                {
                    "strict": 0.0,
                    "f1": 0.0,
                    "lenient": 0.0,
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "latency": 0.0,
                    "count": 0,
                    "batches": 0,
                },
            )
            stats["strict"] += metrics.get("strict_acc", 0.0)
            stats["f1"] += metrics.get("f1", 0.0)
            stats["lenient"] += metrics.get("lenient_acc", 0.0)
            stats["prompt_tokens"] += strategy.get("prompt_tokens", 0)
            stats["generated_tokens"] += strategy.get("generated_tokens", 0)
            stats["latency"] += strategy.get("latency", 0.0)
            stats["batches"] += strategy.get("batches", 0)
            stats["count"] += 1

    summary_lines = ["\n=== Aggregate Metrics ==="]
    header = "Strategy | EM | F1 | Lenient | PromptTok | GenTok | Latency(s) | Batches"
    separator = "-" * len(header)
    summary_lines.extend([header, separator])

    ordered_names = preferred_order + [
        name for name in strategy_totals.keys() if name not in preferred_order
    ]
    for name in ordered_names:
        stats = strategy_totals.get(name)
        if not stats:
            continue
        count = stats["count"] or 1
        avg_prompt = stats["prompt_tokens"] / count
        avg_gen = stats["generated_tokens"] / count
        avg_latency = stats["latency"] / count
        avg_batches = stats["batches"] / count
        summary_lines.append(
            f"{name:<13} | "
            f"{stats['strict']/count:.3f} | "
            f"{stats['f1']/count:.3f} | "
            f"{stats['lenient']/count:.3f} | "
            f"{avg_prompt:.2f} | "
            f"{avg_gen:.2f} | "
            f"{avg_latency:.2f} | "
            f"{avg_batches:.2f}"
        )
    return "\n".join(summary_lines)


def extract_error_cases(serialized_contexts: List[dict]) -> List[dict]:
    """Extract contexts containing questions where at least one strategy answered incorrectly."""
    error_contexts = []
    for ctx in serialized_contexts:
        gold_answers = ctx.get("gold_answers", {})
        strategies = ctx.get("strategies", [])

        # Check each question
        error_questions = {}
        for qid, refs in gold_answers.items():
            # Check if all strategies got this question correct
            all_correct = True
            for strategy in strategies:
                answer = strategy.get("answers", {}).get(qid, "")
                if compute_em(answer, refs) < 1.0:
                    all_correct = False
                    break

            if not all_correct:
                error_questions[qid] = refs

        # If any question has errors, include this context (filtered)
        if error_questions:
            filtered_ctx = {
                "context": ctx.get("context"),
                "questions_text": {qid: ctx.get("questions_text", {}).get(qid, "") for qid in error_questions},
                "gold_answers": error_questions,
                "strategies": [
                    {
                        "name": s.get("name"),
                        "answers": {qid: s.get("answers", {}).get(qid, "") for qid in error_questions},
                        "metrics": s.get("metrics"),
                    }
                    for s in strategies
                ],
            }
            error_contexts.append(filtered_ctx)

    return error_contexts


def generate_output_folder_name(args: argparse.Namespace, timestamp: str) -> str:
    """Generate descriptive output folder name with timestamp and experiment parameters."""
    model_short = args.model_name.split("/")[-1]
    # Build dataset identifier: squad_train or hotpot_distractor_train
    if args.dataset == "hotpot":
        dataset_id = f"hotpot_{args.hotpot_subset}_{args.split}"
    else:
        dataset_id = f"squad_{args.split}"
    # Format: timestamp_dataset_model_n{samples}_q{questions}
    return f"{timestamp}_{dataset_id}_{model_short}_n{args.context_count}_q{args.min_questions}-{args.max_questions}"


def save_experiment_results(
    output_path: Optional[Path],
    serialized_contexts: List[dict],
    args: argparse.Namespace,
    output_folder_name: str,
) -> None:
    """Save experiment results to JSON files and config.txt in the output folder."""
    if not output_path:
        return

    # Create output folder with pre-determined name
    base_path = Path(output_path).parent if Path(output_path).suffix == ".json" else Path(output_path)
    output_dir = base_path / output_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config/parameters
    config = {
        "model": args.model_name,
        "dataset": args.dataset,
        "split": args.split,
        "context_count": args.context_count,
        "min_questions": args.min_questions,
        "max_questions": args.max_questions,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "strategies": args.strategies,
        "timestamp": datetime.now().isoformat(),
    }
    if args.dataset == "hotpot":
        config["hotpot_subset"] = args.hotpot_subset

    # Save config to readable txt file
    config_txt_path = output_dir / "config.txt"
    with open(config_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Experiment Configuration\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.dataset}\n")
        if args.dataset == "hotpot":
            f.write(f"  - HuggingFace: hotpotqa/hotpot_qa ({args.hotpot_subset})\n")
        else:
            f.write(f"  - HuggingFace: rajpurkar/squad\n")
        f.write(f"  - Split: {args.split}\n")
        f.write(f"\nSamples: {args.context_count}\n")
        f.write(f"Questions per sample: {args.min_questions}-{args.max_questions}\n")
        f.write(f"Max new tokens: {args.max_new_tokens}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Strategies: {args.strategies or 'all'}\n")
        f.write(f"\nTimestamp: {config['timestamp']}\n")
    logging.info("Wrote config to %s", config_txt_path)

    # Save full results
    full_payload = {
        "config": config,
        "contexts": serialized_contexts,
    }
    full_path = output_dir / "full_results.json"
    with open(full_path, "w", encoding="utf-8") as fout:
        json.dump(full_payload, fout, indent=2, ensure_ascii=False)
    logging.info("Wrote full results to %s", full_path)

    # Save error-only results (questions where at least one strategy got wrong)
    error_contexts = extract_error_cases(serialized_contexts)
    if error_contexts:
        error_payload = {
            "config": config,
            "description": "Questions where at least one strategy produced incorrect answer",
            "total_error_contexts": len(error_contexts),
            "contexts": error_contexts,
        }
        error_path = output_dir / "errors.json"
        with open(error_path, "w", encoding="utf-8") as fout:
            json.dump(error_payload, fout, indent=2, ensure_ascii=False)
        logging.info("Wrote error analysis to %s (%d contexts with errors)", error_path, len(error_contexts))
    else:
        logging.info("No errors found - all strategies answered all questions correctly")


def main() -> None:
    args = parse_args()
    # Default: disable think tokens for batch compatibility; allow opt-in via --use-think-tokens
    use_think = args.use_think_tokens and not args.no_think_tokens
    set_think_tokens(use_think)
    log_level = logging.DEBUG if args.verbose_debug else getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")
    world_size = args.num_shards or int(os.environ.get("WORLD_SIZE", 1))
    rank = args.shard_index if args.shard_index is not None else int(os.environ.get("LOCAL_RANK", 0))

    # Generate output folder name once at the start (same for all ranks)
    # Use a fixed timestamp based on seed to ensure all ranks use the same folder
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder_name = generate_output_folder_name(args, run_timestamp) if args.json_out else ""

    logging.info("Loading tokenizer and model: %s", args.model_name)
    if args.deterministic:
        # Strong determinism: disable TF32, set deterministic algorithms
        try:
            torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
        except Exception:
            pass
        torch.use_deterministic_algorithms(True)

    # Bind each rank to a single GPU to avoid piling all processes on cuda:0
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if num_devices == 0:
            raise RuntimeError("CUDA_VISIBLE_DEVICES is set but no GPUs detected")
        device_id = rank % num_devices
        torch.cuda.set_device(device_id)
        logging.info("Rank %d using cuda:%d (visible devices: %d)", rank, device_id, num_devices)

    # Initialize distributed backend for multi-GPU (required for gathering results)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        logging.info("Initialized distributed backend: %s (rank %d/%d)", backend, rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Prefer float32 under deterministic mode to minimize divergence
    load_dtype = torch.float32 if args.deterministic or not torch.cuda.is_available() else torch.float16
    device_map = "auto"
    if world_size > 1 and torch.cuda.is_available():
        device_map = {"": f"cuda:{torch.cuda.current_device()}"}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device_map,
        torch_dtype=load_dtype,
    )
    # Force eager attention if configurable (post-load best effort)
    if args.deterministic and hasattr(model, "config") and hasattr(model.config, "attn_implementation"):
        try:
            model.config.attn_implementation = "eager"  # type: ignore[attr-defined]
        except Exception:
            pass
    model.eval()

    if args.dataset == "hotpot":
        contexts = load_hotpot_groups(
            args.split,
            subset=args.hotpot_subset,
            max_contexts=args.context_count,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            seed=args.seed,
        )
    else:
        if args.squad_random_questions:
            contexts = load_squad_random_questions(
                args.split,
                max_contexts=args.context_count,
                seed=args.seed,
            )
        else:
            contexts = load_squad_groups(
                args.split,
                min_questions=args.min_questions,
                max_questions=args.max_questions,
                max_contexts=args.context_count,
                seed=args.seed,
            )
    if world_size > 1:
        contexts = contexts[rank::world_size]
        logging.info("Sharding contexts: rank %d/%d -> %d contexts", rank, world_size, len(contexts))
    logging.info("Loaded %d contexts (requested: %d)", len(contexts), args.context_count)

    dep_generator, bert_dep_generator, bert_conf_threshold = resolve_dependency_generators(args, tokenizer, model)

    # Parse selected strategies
    selected_strategies = None
    if args.strategies:
        selected_strategies = [s.strip() for s in args.strategies.split(",")]
        invalid = [s for s in selected_strategies if s not in ALL_STRATEGIES]
        if invalid:
            raise ValueError(f"Invalid strategies: {invalid}. Available: {ALL_STRATEGIES}")
        logging.info("Running selected strategies: %s", selected_strategies)
    else:
        logging.info("Running all strategies: %s", ALL_STRATEGIES)

    overall_results: Dict[str, List[StrategyResult]] = {}
    serialized_contexts: List[dict] = []

    for idx, context_payload in enumerate(contexts, start=1):
        title = context_payload.get("title", f"context-{idx}")
        if "items" in context_payload:
            items = context_payload["items"]
            logging.info("Processing multi-context group %d/%d: %s (items=%d)", idx, len(contexts), title, len(items))
            strategy_list = run_all_strategies(
                None,
                items,
                tokenizer=tokenizer,
                model=model,
                dep_generator=dep_generator,
                bert_dep_generator=bert_dep_generator,
                args=args,
                bert_conf_threshold=bert_conf_threshold,
                selected_strategies=selected_strategies,
            )
            questions = [
                Question(
                    qid=item["qid"],
                    text=item["question"],
                    priority=1.0,
                    answer_tokens=item.get("answer_tokens", 12),
                    type_hint=None,
                    references=item.get("references", []),
                )
                for item in items
            ]
            background = ""
        else:
            background = context_payload["context"]
            questions = build_questions_from_group(context_payload)
            logging.info("Processing context %d/%d: %s", idx, len(contexts), title)
            logging.info("Background preview: %s", background[:200].replace("\n", " "))

            strategy_list = run_all_strategies(
                background,
                questions,
                tokenizer=tokenizer,
                model=model,
                dep_generator=dep_generator,
                bert_dep_generator=bert_dep_generator,
                args=args,
                bert_conf_threshold=bert_conf_threshold,
                selected_strategies=selected_strategies,
            )
        overall_results[title] = strategy_list

        print(f"\n=== Context: {title} ===")
        print(summarize_results(strategy_list))
        print_answer_table(questions, strategy_list)

        serialized_contexts.append(
            {
                "context": title,
                "questions_text": {q.qid: q.text.strip() for q in questions},
                "gold_answers": {q.qid: q.references for q in questions},
                "strategies": [
                    {
                        "name": res.name,
                        "metrics": {k: round(v, 2) for k, v in res.metrics.items()},
                        "prompt_tokens": res.prompt_tokens,
                        "generated_tokens": res.generated_tokens,
                        "latency": round(res.latency, 2),
                        "batches": res.batches,
                        "answers": res.answers,
                        "details": res.details,
                    }
                    for res in strategy_list
                ],
            }
        )

    # Distributed output handling: gather to rank0 and write a single file
    if world_size > 1 and dist.is_initialized():
        # Use torch.distributed to gather results from all ranks
        gather_list: List[List[dict]] = [None for _ in range(world_size)]  # type: ignore
        dist.all_gather_object(gather_list, serialized_contexts)
        if rank == 0:
            merged = []
            for ctx_list in gather_list:
                if ctx_list:
                    merged.extend(ctx_list)
            # Print aggregate metrics from all ranks (only on rank 0)
            print(compute_aggregate_metrics(merged))
            save_experiment_results(args.json_out, merged, args, output_folder_name)
    else:
        # Single process mode: print and save local results
        print(compute_aggregate_metrics(serialized_contexts))
        save_experiment_results(args.json_out, serialized_contexts, args, output_folder_name)

    # Cleanup distributed backend
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
