from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_qwen_parallel as rq
from python import (
    BertAttentionDependencyGenerator,
    HeuristicDependencyGenerator,
    load_hotpot_groups,
    load_squad_random_questions,
    build_questions_from_group,
    load_squad_groups,
    Question,
)
from strategies import (
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
from strategies import StrategyResult


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
    parser.add_argument(
        "--hotpot-group-size",
        type=int,
        default=None,
        help="Number of HotpotQA samples to bundle into one multi-question context (defaults to min-questions).",
    )

    parser.add_argument("--cost-weight", type=float, default=0.01, help="Cost penalty weight for dependency selection.")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum edge confidence.")
    parser.add_argument("--max-dependencies", type=int, default=3, help="Max dependencies per question.")
    parser.add_argument("--total-cost-budget", type=int, default=None, help="Optional global dependency cost budget.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max new tokens for answer generation.")

    parser.add_argument("--log-level", default="INFO", help="Logging level.")
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
        dep_generator = rq.LocalLLMDependencyGenerator(tokenizer, model)
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
) -> List[StrategyResult]:
    # Multi-context mode: if questions is None and items provided, use items-based strategies
    if background is None and isinstance(questions, list) and questions and isinstance(questions[0], dict) and "context" in questions[0]:
        items = questions
        all_in_one = run_all_in_one_multi_strategy(
            items,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            strategy_name="all_in_one",
        )
        sequential = run_sequential_multi_strategy(
            items,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            strategy_name="sequential",
        )
        batch = run_batch_multi_strategy(
            items,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            strategy_name="batch",
        )
        # For dependency strategies, synthesize a merged background so the model can decide relations.
        merged_background_parts = [f"Context ({item['qid']}):\n{item['context']}" for item in items]
        merged_background = "\n\n".join(merged_background_parts)
        dep_questions = [
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
        parallel = run_dependency_batch_strategy(
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
        )
        parallel_bert = run_dependency_batch_strategy(
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
        )
        return [all_in_one, sequential, batch, parallel, parallel_bert]

    all_in_one = run_all_in_one_strategy(
        background,
        questions,
        tokenizer,
        model,
        max_new_tokens=args.max_new_tokens,
    )
    sequential = run_sequential_strategy(
        background,
        questions,
        tokenizer,
        model,
        max_new_tokens=args.max_new_tokens,
    )
    batch = run_full_batch_strategy(
        background,
        questions,
        tokenizer,
        model,
        max_new_tokens=args.max_new_tokens,
    )
    parallel = run_dependency_batch_strategy(
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
    )
    parallel_bert = run_dependency_batch_strategy(
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
    )
    return [
        all_in_one,
        sequential,
        batch,
        parallel,
        parallel_bert,
    ]


def aggregate_overall(overall_results: Dict[str, List[StrategyResult]]) -> str:
    preferred_order = [
        "all_in_one",
        "sequential",
        "batch",
        "parallel",
        "parallel_bert",
    ]
    strategy_totals: Dict[str, Dict[str, float]] = {}
    for results in overall_results.values():
        for res in results:
            stats = strategy_totals.setdefault(
                res.name,
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
            stats["strict"] += res.metrics["strict_acc"]
            stats["f1"] += res.metrics["f1"]
            stats["lenient"] += res.metrics["lenient_acc"]
            stats["prompt_tokens"] += res.prompt_tokens
            stats["generated_tokens"] += res.generated_tokens
            stats["latency"] += res.latency
            stats["batches"] += res.batches
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
        avg_prompt = int(round(stats["prompt_tokens"] / count))
        avg_gen = int(round(stats["generated_tokens"] / count))
        avg_latency = stats["latency"] / count
        avg_batches = int(round(stats["batches"] / count))
        summary_lines.append(
            f"{name:<13} | "
            f"{stats['strict']/count:.3f} | "
            f"{stats['f1']/count:.3f} | "
            f"{stats['lenient']/count:.3f} | "
            f"{avg_prompt} | "
            f"{avg_gen} | "
            f"{avg_latency:.2f} | "
            f"{avg_batches}"
        )
    return "\n".join(summary_lines)


def maybe_dump_json(
    path: Optional[Path],
    serialized_contexts: List[dict],
    args: argparse.Namespace,
) -> None:
    if not path:
        return
    payload = {
        "model": args.model_name,
        "contexts": serialized_contexts,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)
    logging.info("Wrote JSON summary to %s", path)


def main() -> None:
    args = parse_args()
    # Default: disable think tokens for batch compatibility; allow opt-in via --use-think-tokens
    use_think = args.use_think_tokens and not args.no_think_tokens
    rq.set_think_tokens(use_think)
    log_level = logging.DEBUG if args.verbose_debug else getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")
    world_size = args.num_shards or int(os.environ.get("WORLD_SIZE", 1))
    rank = args.shard_index if args.shard_index is not None else int(os.environ.get("LOCAL_RANK", 0))

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
            group_size=args.hotpot_group_size,
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
                        "metrics": res.metrics,
                        "prompt_tokens": res.prompt_tokens,
                        "generated_tokens": res.generated_tokens,
                        "latency": res.latency,
                        "batches": res.batches,
                        "answers": res.answers,
                        "details": res.details,
                    }
                    for res in strategy_list
                ],
            }
        )

    print(aggregate_overall(overall_results))
    # Sharded output handling
    if world_size > 1 and args.json_out:
        shard_path = Path(f"{args.json_out}.shard{rank}of{world_size}")
        maybe_dump_json(shard_path, serialized_contexts, args)
        if rank == 0:
            # Wait for other shards then merge
            for r in range(world_size):
                path = Path(f"{args.json_out}.shard{r}of{world_size}")
                while not path.exists():
                    time.sleep(1)
            merged = []
            for r in range(world_size):
                path = Path(f"{args.json_out}.shard{r}of{world_size}")
                with open(path, "r", encoding="utf-8") as fin:
                    payload = json.load(fin)
                    merged.extend(payload.get("contexts", []))
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as fout:
                json.dump({"model": args.model_name, "contexts": merged}, fout, indent=2, ensure_ascii=False)
            logging.info("Merged shards into %s", args.json_out)
    else:
        maybe_dump_json(args.json_out, serialized_contexts, args)


if __name__ == "__main__":
    main()
