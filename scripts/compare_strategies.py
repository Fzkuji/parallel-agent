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

from src.evaluation import (
    compute_em,
    get_dataset_metrics,
    get_metric_names,
)

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import (
    BertAttentionDependencyGenerator,
    HeuristicDependencyGenerator,
    LocalLLMDependencyGenerator,
    APILLMDependencyGenerator,
    load_cmb_groups,
    load_drop_groups,
    load_hotpot_groups,
    load_quac_groups,
    load_quality_groups,
    load_squad_random_questions,
    build_questions_from_group,
    load_squad_groups,
    Question,
    set_think_tokens,
    APIClient,
)
from src.datasets import load_cmb_exam_random_groups, load_cmb_exam_subdomain_groups, load_cmb_exam_context_groups
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
from src.strategies.cross_batch import (
    run_cross_batch_strategy,
    run_cross_batch_multi_strategy,
)


def get_checkpoint_path(base_dir: str, dataset: str, model_name: str, mode: str) -> str:
    """
    Generate checkpoint path in format: {base_dir}/{dataset}/{model_name}_{mode}.pt

    Args:
        base_dir: Base checkpoint directory (e.g., outputs/checkpoints)
        dataset: Dataset name (e.g., squad, hotpot)
        model_name: Model name (e.g., Qwen/Qwen2.5-14B-Instruct)
        mode: Checkpoint mode (e.g., baseline, crossbatch, lora_only, lora_lmhead)

    Returns:
        Full checkpoint path
    """
    safe_model_name = model_name.replace('/', '_')
    return os.path.join(base_dir, dataset, f'{safe_model_name}_{mode}.pt')


def auto_find_checkpoints(args) -> dict:
    """
    Auto-find checkpoint paths based on model name and dataset.

    Returns dict with keys: baseline, crossbatch, lora, lora_lmhead
    Each value is either the found path or None if not found.
    """
    base_dir = getattr(args, 'checkpoint_dir', 'outputs/checkpoints')
    dataset = args.dataset
    model_name = args.model_name

    checkpoints = {}
    modes = {
        'baseline': 'baseline',
        'crossbatch': 'crossbatch',
        'lora_lmhead': 'lora_lmhead',
        'lora_crossbatch': 'lora_crossbatch',
    }

    for key, mode in modes.items():
        path = get_checkpoint_path(base_dir, dataset, model_name, mode)
        if os.path.exists(path):
            checkpoints[key] = path
            logging.info(f"Found {key} checkpoint: {path}")
        else:
            checkpoints[key] = None
            logging.debug(f"No {key} checkpoint at: {path}")

    return checkpoints


def create_vllm_model(model_name: str, tensor_parallel_size: int = 1):
    """Create a vLLM model for fast inference."""
    try:
        from vllm import LLM, SamplingParams
        model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        return model
    except ImportError:
        logging.warning("vLLM not installed. Install with: pip install vllm")
        return None


def run_vllm_batch_strategy(
    items: List[Dict],
    vllm_model,
    tokenizer: "AutoTokenizer",
    *,
    max_new_tokens: int,
    strategy_name: str = "batch_vllm",
    dataset: str = None,
) -> "StrategyResult":
    """Run batch strategy using vLLM for fast inference."""
    from vllm import SamplingParams
    from src.prompts import build_single_prompt
    from src.inference import build_chat_prompt, extract_answer
    from src.evaluation import evaluate_predictions
    from src.models import Question, StrategyResult

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

    # Build prompts for all items
    batch_chat_prompts = []
    for item in items:
        q = question_lookup[item["qid"]]
        system_prompt, user_prompt = build_single_prompt(item["context"], q, dataset)
        batch_chat_prompts.append(
            build_chat_prompt(tokenizer, user_prompt, system_prompt=system_prompt)
        )

    # Use vLLM for inference
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    start = time.perf_counter()
    outputs = vllm_model.generate(batch_chat_prompts, sampling_params)
    total_latency = time.perf_counter() - start

    # Process outputs
    answer_records = {}
    answers_text = {}
    per_question = []
    total_prompt_tokens = 0
    total_generated_tokens = 0

    for idx, (item, output) in enumerate(zip(items, outputs)):
        qid = item["qid"]
        raw_text = output.outputs[0].text.strip()
        final_answer, strict_valid = extract_answer(raw_text, dataset)
        answer_records[qid] = (final_answer, strict_valid)
        answers_text[qid] = final_answer

        prompt_tokens = len(output.prompt_token_ids)
        gen_tokens = len(output.outputs[0].token_ids)
        total_prompt_tokens += prompt_tokens
        total_generated_tokens += gen_tokens

        per_question.append({
            "question_id": qid,
            "question": item["question"],
            "gold_answers": item.get("references", []),
            "prompt": batch_chat_prompts[idx],
            "raw_response": raw_text,
            "final_answer": final_answer,
            "strict_valid": strict_valid,
            "latency": total_latency / len(items),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_tokens,
        })

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
    return StrategyResult(
        name=strategy_name,
        answers=answers_text,
        prompt_tokens=total_prompt_tokens,
        generated_tokens=total_generated_tokens,
        latency=total_latency,
        batches=1,
        metrics=metrics,
        details={"questions": per_question},
    )


def run_vllm_sequential_strategy(
    items: List[Dict],
    vllm_model,
    tokenizer: "AutoTokenizer",
    *,
    max_new_tokens: int,
    strategy_name: str = "sequential_vllm",
    dataset: str = None,
) -> "StrategyResult":
    """Run sequential strategy using vLLM (one prompt at a time for fair comparison)."""
    from vllm import SamplingParams
    from src.prompts import build_single_prompt
    from src.inference import build_chat_prompt, extract_answer
    from src.evaluation import evaluate_predictions
    from src.models import Question, StrategyResult

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

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    answer_records = {}
    answers_text = {}
    per_question = []
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_latency = 0.0

    for item in items:
        qid = item["qid"]
        q = question_lookup[qid]
        system_prompt, user_prompt = build_single_prompt(item["context"], q, dataset)
        chat_prompt = build_chat_prompt(tokenizer, user_prompt, system_prompt=system_prompt)

        start = time.perf_counter()
        outputs = vllm_model.generate([chat_prompt], sampling_params)
        elapsed = time.perf_counter() - start
        total_latency += elapsed

        output = outputs[0]
        raw_text = output.outputs[0].text.strip()
        final_answer, strict_valid = extract_answer(raw_text, dataset)
        answer_records[qid] = (final_answer, strict_valid)
        answers_text[qid] = final_answer

        prompt_tokens = len(output.prompt_token_ids)
        gen_tokens = len(output.outputs[0].token_ids)
        total_prompt_tokens += prompt_tokens
        total_generated_tokens += gen_tokens

        per_question.append({
            "question_id": qid,
            "question": item["question"],
            "gold_answers": item.get("references", []),
            "prompt": chat_prompt,
            "raw_response": raw_text,
            "final_answer": final_answer,
            "strict_valid": strict_valid,
            "latency": elapsed,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_tokens,
        })

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
    return StrategyResult(
        name=strategy_name,
        answers=answers_text,
        prompt_tokens=total_prompt_tokens,
        generated_tokens=total_generated_tokens,
        latency=total_latency,
        batches=len(items),
        metrics=metrics,
        details={"turns": per_question},
    )


def run_vllm_all_in_one_strategy(
    items: List[Dict],
    vllm_model,
    tokenizer: "AutoTokenizer",
    *,
    max_new_tokens: int,
    strategy_name: str = "all_in_one_vllm",
    dataset: str = None,
) -> "StrategyResult":
    """Run all-in-one strategy using vLLM (all questions in one prompt)."""
    from vllm import SamplingParams
    from src.prompts import build_all_in_one_prompt
    from src.inference import build_chat_prompt, extract_answer
    from src.evaluation import evaluate_predictions
    from src.models import Question, StrategyResult

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

    # Get context (all items should have same context for all_in_one)
    context = items[0]["context"] if items else ""
    questions = list(question_lookup.values())

    # Build all-in-one prompt
    system_prompt, user_prompt = build_all_in_one_prompt(context, questions, dataset)
    chat_prompt = build_chat_prompt(tokenizer, user_prompt, system_prompt=system_prompt)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens * len(items),  # More tokens for multiple answers
        temperature=0.0,
        top_p=1.0,
    )

    start = time.perf_counter()
    outputs = vllm_model.generate([chat_prompt], sampling_params)
    total_latency = time.perf_counter() - start

    output = outputs[0]
    raw_text = output.outputs[0].text.strip()
    prompt_tokens = len(output.prompt_token_ids)
    gen_tokens = len(output.outputs[0].token_ids)

    # Parse multi-answer response
    answer_records = {}
    answers_text = {}
    per_question = []

    # Try to extract individual answers from the combined response
    lines = raw_text.split('\n')
    for q in questions:
        # Look for answer pattern like "Q1: answer" or "1. answer"
        found = False
        for line in lines:
            if q.qid in line or f"Q{q.qid.replace('Q', '')}" in line:
                final_answer, strict_valid = extract_answer(line, dataset)
                answer_records[q.qid] = (final_answer, strict_valid)
                answers_text[q.qid] = final_answer
                found = True
                break
        if not found:
            # Fallback: use empty answer
            answer_records[q.qid] = ("", False)
            answers_text[q.qid] = ""

        per_question.append({
            "question_id": q.qid,
            "question": q.text,
            "gold_answers": q.references,
            "final_answer": answers_text.get(q.qid, ""),
            "latency": total_latency / len(items),
        })

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
    return StrategyResult(
        name=strategy_name,
        answers=answers_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=gen_tokens,
        latency=total_latency,
        batches=1,
        metrics=metrics,
        details={"questions": per_question, "raw_response": raw_text},
    )


def run_vllm_strategies(
    items: List[Dict],
    vllm_model,
    tokenizer: "AutoTokenizer",
    dep_generator,
    bert_dep_generator,
    *,
    args,
    bert_conf_threshold: float,
    selected_strategies: List[str],
    eval_dataset: str = None,
) -> List["StrategyResult"]:
    """Run vLLM-compatible strategies."""
    results = []
    effective_dataset = eval_dataset or args.dataset

    if "all_in_one" in selected_strategies:
        results.append(run_vllm_all_in_one_strategy(
            items,
            vllm_model,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            strategy_name="all_in_one_vllm",
            dataset=effective_dataset,
        ))

    if "sequential" in selected_strategies:
        results.append(run_vllm_sequential_strategy(
            items,
            vllm_model,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            strategy_name="sequential_vllm",
            dataset=effective_dataset,
        ))

    if "batch" in selected_strategies:
        results.append(run_vllm_batch_strategy(
            items,
            vllm_model,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            strategy_name="batch_vllm",
            dataset=effective_dataset,
        ))

    # For collab_llm and collab_bert, we use the vLLM model for the main generation
    # but the dependency generation still uses the regular model or API
    # This is a simplified version - the full implementation would require
    # modifying the dependency batch strategy to use vLLM

    return results


def get_eval_dataset(args: argparse.Namespace) -> str:
    """Get the dataset identifier for evaluation based on args.

    CMB-Exam subsets (random, subdomain, context) use 'cmb_exam' metrics (accuracy),
    while CMB-Clin uses 'cmb' metrics (BLEU/ROUGE).
    """
    if args.dataset == "cmb":
        if args.cmb_subset in ("random", "subdomain", "context"):
            return "cmb_exam"
        return "cmb"  # CMB-Clin uses BLEU/ROUGE
    # Direct cmb_exam_* datasets
    if args.dataset in ("cmb_exam_context", "cmb_exam_subdomain", "cmb_exam_random"):
        return "cmb_exam"
    return args.dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sequential, batch, and dependency-aware QA strategies with optional BERT dependencies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["squad", "hotpot", "quac", "cmb", "quality", "drop", "cmb_exam_context", "cmb_exam_subdomain", "cmb_exam_random"], default="squad", help="Dataset to evaluate.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B", help="Hugging Face model identifier or local path.")
    parser.add_argument("--split", default="train", help="Dataset split to sample.")
    parser.add_argument("--context-count", type=int, default=3, help="Number of contexts to process.")
    parser.add_argument("--min-questions", type=int, default=3, help="Minimum questions per context.")
    parser.add_argument("--max-questions", type=int, default=5, help="Maximum questions per context.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for sampling contexts.")
    parser.add_argument("--shuffle-questions", action="store_true", default=True,
                        help="Shuffle question order before inference (default: True). Important for sequential strategy to avoid order bias.")
    parser.add_argument("--no-shuffle-questions", dest="shuffle_questions", action="store_false",
                        help="Disable question shuffling (use original order from dataset).")
    
    # Set default for shuffle_questions (argparse doesn't support default=True with store_true)
    parser.set_defaults(shuffle_questions=True)
    parser.add_argument(
        "--squad-random-questions",
        action="store_true",
        help="For SQuAD, sample individual questions randomly instead of grouping by shared context.",
    )
    parser.add_argument("--hotpot-subset", default="distractor", help="HotpotQA subset (e.g., distractor).")
    parser.add_argument("--cmb-subset", default="CMB-Clin",
                        help="CMB subset: 'CMB-Clin' (clinical cases), 'random' (random grouping baseline), 'subdomain' (grouped by medical terms), 'context' (shared background).")
    parser.add_argument("--quality-hard-only", action="store_true", help="For QuALITY, only use hard questions.")

    parser.add_argument("--cost-weight", type=float, default=0.0, help="Cost penalty weight for dependency selection (set to 0 to let model decide).")
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
             "Available: all_in_one, sequential, batch, collab_llm, collab_bert, collab_hidden. "
             "If not specified, all strategies will be tested.",
    )
    parser.add_argument("--no-llm-deps", action="store_true", help="Force heuristic dependency generator.")
    parser.add_argument("--json-out", type=Path, default=Path("outputs_json"), help="Path to save experiment results (default: outputs_json).")
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
    # LLM-based evaluation via OpenRouter API
    parser.add_argument(
        "--eval-model",
        type=str,
        default=None,
        help="OpenRouter model ID for LLM-based evaluation (e.g., 'openai/gpt-4o'). "
             "Requires OPENROUTER_API_KEY env var.",
    )
    # API-based inference (instead of local model)
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use API-based inference instead of local model. "
             "Requires setting appropriate API key environment variable.",
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default=None,
        help="Model identifier for API inference (e.g., 'deepseek-chat', 'qwen/qwen3-30b-a3b'). "
             "If not specified, uses --model-name.",
    )
    parser.add_argument(
        "--api-provider",
        type=str,
        default=None,
        help="API provider (e.g., 'openai', 'openrouter', 'together', 'deepseek'). "
             "Auto-detected from model name if not specified.",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="Custom API base URL (overrides provider default).",
    )
    # Collaborative hidden (collab_hidden) strategy arguments
    parser.add_argument(
        "--collab-hidden-checkpoint",
        type=str,
        default=None,
        help="Path to trained collab_hidden module checkpoint.",
    )
    # Checkpoint directory for auto-discovery
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/checkpoints",
        help="Base directory for checkpoint auto-discovery (default: outputs/checkpoints). "
             "Checkpoints are expected at: {checkpoint-dir}/{dataset}/{model_name}_{mode}.pt",
    )
    parser.add_argument(
        "--auto-checkpoints",
        action="store_true",
        help="Auto-discover checkpoints based on --checkpoint-dir, --dataset, and --model-name.",
    )
    # LoRA checkpoint arguments (explicit paths override auto-discovery)
    parser.add_argument(
        "--lora-lmhead-checkpoint",
        type=str,
        default=None,
        help="Path to LoRA+lm_head checkpoint for lora_lmhead strategy (overrides auto-discovery).",
    )
    parser.add_argument(
        "--lora-crossbatch-checkpoint",
        type=str,
        default=None,
        help="Path to LoRA+lm_head+cross-batch checkpoint for lora_crossbatch strategy (overrides auto-discovery).",
    )
    # vLLM inference
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Also run strategies with vLLM for comparison. Only for non-finetuned strategies.",
    )
    parser.add_argument(
        "--collab-hidden-mix-method",
        type=str,
        default="attention",
        choices=["attention", "mixer"],
        help="Collab hidden mixing method (attention or mixer).",
    )
    parser.add_argument(
        "--collab-hidden-mix-layer",
        type=int,
        default=-1,
        help="Which layer's hidden state to mix (-1 for last layer).",
    )
    # Batch finetuned (baseline) checkpoint
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default=None,
        help="Path to trained baseline (lm_head only) checkpoint for batch_finetuned strategy.",
    )
    return parser.parse_args()


def resolve_dependency_generators(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    api_client: Optional[APIClient] = None,
) -> tuple:
    if args.no_llm_deps:
        dep_generator = HeuristicDependencyGenerator()
        logging.info("Using heuristic dependency generator.")
    elif api_client is not None:
        # Use API for LLM-based dependency generation
        dep_generator = APILLMDependencyGenerator(api_client)
        logging.info("Using API-based LLM dependency generator.")
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


ALL_STRATEGIES = ["all_in_one", "sequential", "batch", "finetuned", "collab_llm", "collab_bert", "collab_hidden", "lora_lmhead", "lora_crossbatch"]

# Strategies that can be run with vLLM (no finetuning required)
VLLM_STRATEGIES = ["all_in_one", "sequential", "batch", "collab_llm", "collab_bert"]


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
    api_client: Optional[APIClient] = None,
    eval_dataset: Optional[str] = None,
) -> List[StrategyResult]:
    # Default to all strategies if none specified
    if selected_strategies is None:
        selected_strategies = ALL_STRATEGIES

    results: List[StrategyResult] = []
    # Use eval_dataset if provided, otherwise fall back to args.dataset
    effective_dataset = eval_dataset or args.dataset

    # Shuffle questions if requested (important for sequential strategy)
    shuffle_questions = getattr(args, 'shuffle_questions', True)
    if shuffle_questions:
        import random
        rng = random.Random(args.seed)
        if isinstance(questions, list) and questions:
            if isinstance(questions[0], Question):
                # Single-context mode: List[Question]
                questions = questions.copy()
                rng.shuffle(questions)
            elif isinstance(questions[0], dict):
                # Multi-context mode: List[Dict]
                questions = questions.copy()
                rng.shuffle(questions)

    # Multi-context mode: if questions is None and items provided, use items-based strategies
    if background is None and isinstance(questions, list) and questions and isinstance(questions[0], dict) and "context" in questions[0]:
        items = questions  # Already shuffled above if shuffle_questions=True
        if "all_in_one" in selected_strategies:
            results.append(run_all_in_one_multi_strategy(
                items,
                tokenizer,
                model,
                max_new_tokens=args.max_new_tokens,
                strategy_name="all_in_one",
                dataset=effective_dataset,
                api_client=api_client,
            ))
        if "sequential" in selected_strategies:
            results.append(run_sequential_multi_strategy(
                items,
                tokenizer,
                model,
                max_new_tokens=args.max_new_tokens,
                strategy_name="sequential",
                dataset=effective_dataset,
                api_client=api_client,
            ))
        if "batch" in selected_strategies:
            results.append(run_batch_multi_strategy(
                items,
                tokenizer,
                model,
                max_new_tokens=args.max_new_tokens,
                strategy_name="batch",
                dataset=effective_dataset,
                api_client=api_client,
            ))
        if "finetuned" in selected_strategies:
            # Load baseline checkpoint and temporarily replace lm_head
            import torch
            if args.baseline_checkpoint and hasattr(model, 'lm_head'):
                checkpoint = torch.load(args.baseline_checkpoint, map_location=model.device)
                original_lm_head_state = {k: v.clone() for k, v in model.lm_head.state_dict().items()}
                model.lm_head.load_state_dict(checkpoint["lm_head"])
                results.append(run_batch_multi_strategy(
                    items,
                    tokenizer,
                    model,
                    max_new_tokens=args.max_new_tokens,
                    strategy_name="finetuned",
                    dataset=effective_dataset,
                    api_client=api_client,
                ))
                # Restore original lm_head
                model.lm_head.load_state_dict(original_lm_head_state)
            else:
                logging.warning("Skipping finetuned strategy: --baseline-checkpoint not provided or model has no lm_head")
        # For dependency strategies in multi-context mode, store context per question
        # This ensures consistent prompt format with batch strategy (context in system message)
        if "collab_llm" in selected_strategies or "collab_bert" in selected_strategies:
            dep_questions = [
                Question(
                    qid=item["qid"],
                    text=item["question"],
                    priority=1.0,
                    answer_tokens=item.get("answer_tokens", 12),
                    type_hint=None,
                    references=item.get("references", []),
                    context=item["context"],  # Store context per question
                )
                for item in items
            ]
            if "collab_llm" in selected_strategies:
                results.append(run_dependency_batch_strategy(
                    "",  # Empty shared background; each question has its own context
                    dep_questions,
                    generator=dep_generator,
                    tokenizer=tokenizer,
                    model=model,
                    cost_weight=args.cost_weight,
                    min_confidence=args.min_confidence,
                    max_dependencies=args.max_dependencies,
                    total_cost_budget=args.total_cost_budget,
                    max_new_tokens=args.max_new_tokens,
                    strategy_name="collab_llm",
                    dataset=effective_dataset,
                    api_client=api_client,
                ))
            if "collab_bert" in selected_strategies:
                results.append(run_dependency_batch_strategy(
                    "",  # Empty shared background; each question has its own context
                    dep_questions,
                    generator=bert_dep_generator,
                    tokenizer=tokenizer,
                    model=model,
                    cost_weight=args.cost_weight,
                    min_confidence=args.min_confidence,
                    max_dependencies=args.max_dependencies,
                    total_cost_budget=args.total_cost_budget,
                    max_new_tokens=args.max_new_tokens,
                    strategy_name="collab_bert",
                    dataset=effective_dataset,
                    api_client=api_client,
                ))
        if "collab_hidden" in selected_strategies:
            results.append(run_cross_batch_multi_strategy(
                items,
                tokenizer,
                model,
                max_new_tokens=args.max_new_tokens,
                strategy_name="collab_hidden",
                dataset=effective_dataset,
                mix_method=args.collab_hidden_mix_method,
                mix_layer=args.collab_hidden_mix_layer,
                checkpoint_path=args.collab_hidden_checkpoint,
                enable_cross_batch=True,
            ))
        if "lora_lmhead" in selected_strategies:
            import torch
            if args.lora_lmhead_checkpoint and hasattr(model, 'lm_head'):
                try:
                    from peft import LoraConfig, get_peft_model, TaskType
                    checkpoint = torch.load(args.lora_lmhead_checkpoint, map_location=model.device)
                    original_lm_head_state = {k: v.clone() for k, v in model.lm_head.state_dict().items()}
                    if 'lm_head' in checkpoint:
                        model.lm_head.load_state_dict(checkpoint['lm_head'])
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=16,
                        lora_alpha=32,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        bias="none",
                    )
                    lora_model = get_peft_model(model, lora_config)
                    if 'lora' in checkpoint:
                        current_state = lora_model.state_dict()
                        current_state.update(checkpoint['lora'])
                        lora_model.load_state_dict(current_state)
                    results.append(run_batch_multi_strategy(
                        items,
                        tokenizer,
                        lora_model,
                        max_new_tokens=args.max_new_tokens,
                        strategy_name="lora_lmhead",
                        dataset=effective_dataset,
                        api_client=api_client,
                    ))
                    del lora_model
                    model.lm_head.load_state_dict(original_lm_head_state)
                except Exception as e:
                    logging.warning(f"Skipping lora_lmhead strategy: {e}")
            else:
                logging.warning("Skipping lora_lmhead strategy: --lora-lmhead-checkpoint not provided or model has no lm_head")
        if "lora_crossbatch" in selected_strategies:
            import torch
            if args.lora_crossbatch_checkpoint and hasattr(model, 'lm_head'):
                try:
                    from peft import LoraConfig, get_peft_model, TaskType
                    from src.cross_batch.attention import CrossBatchAttention
                    checkpoint = torch.load(args.lora_crossbatch_checkpoint, map_location=model.device)
                    original_lm_head_state = {k: v.clone() for k, v in model.lm_head.state_dict().items()}
                    if 'lm_head' in checkpoint:
                        model.lm_head.load_state_dict(checkpoint['lm_head'])
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=16,
                        lora_alpha=32,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        bias="none",
                    )
                    lora_model = get_peft_model(model, lora_config)
                    if 'lora' in checkpoint:
                        current_state = lora_model.state_dict()
                        current_state.update(checkpoint['lora'])
                        lora_model.load_state_dict(current_state)
                    # Load cross-batch module
                    cross_batch_module = CrossBatchAttention(hidden_size=model.config.hidden_size)
                    if 'cross_batch_module' in checkpoint:
                        cross_batch_module.load_state_dict(checkpoint['cross_batch_module'])
                    cross_batch_module = cross_batch_module.to(model.device)
                    results.append(run_cross_batch_multi_strategy(
                        items,
                        tokenizer,
                        lora_model.base_model,
                        max_new_tokens=args.max_new_tokens,
                        strategy_name="lora_crossbatch",
                        dataset=effective_dataset,
                        mix_method="attention",
                        mix_layer=-1,
                        checkpoint_path=None,  # Already loaded
                        enable_cross_batch=True,
                        cross_batch_module=cross_batch_module,
                    ))
                    del lora_model, cross_batch_module
                    model.lm_head.load_state_dict(original_lm_head_state)
                except Exception as e:
                    logging.warning(f"Skipping lora_crossbatch strategy: {e}")
            else:
                logging.warning("Skipping lora_crossbatch strategy: --lora-crossbatch-checkpoint not provided or model has no lm_head")
        return results

    if "all_in_one" in selected_strategies:
        results.append(run_all_in_one_strategy(
            background,
            questions,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            dataset=effective_dataset,
            api_client=api_client,
        ))
    if "sequential" in selected_strategies:
        results.append(run_sequential_strategy(
            background,
            questions,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            dataset=effective_dataset,
            api_client=api_client,
        ))
    if "batch" in selected_strategies:
        results.append(run_full_batch_strategy(
            background,
            questions,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            dataset=effective_dataset,
            api_client=api_client,
        ))
    if "finetuned" in selected_strategies:
        # Load baseline checkpoint and temporarily replace lm_head
        import torch
        if args.baseline_checkpoint and hasattr(model, 'lm_head'):
            checkpoint = torch.load(args.baseline_checkpoint, map_location=model.device)
            original_lm_head_state = {k: v.clone() for k, v in model.lm_head.state_dict().items()}
            model.lm_head.load_state_dict(checkpoint["lm_head"])
            results.append(run_full_batch_strategy(
                background,
                questions,
                tokenizer,
                model,
                max_new_tokens=args.max_new_tokens,
                dataset=effective_dataset,
                api_client=api_client,
                strategy_name="finetuned",
            ))
            # Restore original lm_head
            model.lm_head.load_state_dict(original_lm_head_state)
        else:
            logging.warning("Skipping finetuned strategy: --baseline-checkpoint not provided or model has no lm_head")
    if "collab_llm" in selected_strategies:
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
            strategy_name="collab_llm",
            dataset=effective_dataset,
            api_client=api_client,
        ))
    if "collab_bert" in selected_strategies:
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
            strategy_name="collab_bert",
            dataset=effective_dataset,
            api_client=api_client,
        ))
    if "collab_hidden" in selected_strategies:
        results.append(run_cross_batch_strategy(
            background,
            questions,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            strategy_name="collab_hidden",
            dataset=effective_dataset,
            mix_method=args.collab_hidden_mix_method,
            mix_layer=args.collab_hidden_mix_layer,
            checkpoint_path=args.collab_hidden_checkpoint,
            enable_cross_batch=True,
        ))
    if "lora_lmhead" in selected_strategies:
        # Load LoRA + lm_head checkpoint and run batch strategy
        import torch
        if args.lora_lmhead_checkpoint and hasattr(model, 'lm_head'):
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                checkpoint = torch.load(args.lora_lmhead_checkpoint, map_location=model.device)
                # Save original lm_head state
                original_lm_head_state = {k: v.clone() for k, v in model.lm_head.state_dict().items()}
                # Load lm_head weights
                if 'lm_head' in checkpoint:
                    model.lm_head.load_state_dict(checkpoint['lm_head'])
                # Apply LoRA configuration
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                )
                lora_model = get_peft_model(model, lora_config)
                # Load LoRA weights
                if 'lora' in checkpoint:
                    current_state = lora_model.state_dict()
                    current_state.update(checkpoint['lora'])
                    lora_model.load_state_dict(current_state)
                results.append(run_full_batch_strategy(
                    background,
                    questions,
                    tokenizer,
                    lora_model,
                    max_new_tokens=args.max_new_tokens,
                    dataset=effective_dataset,
                    api_client=api_client,
                    strategy_name="lora_lmhead",
                ))
                # Clean up and restore
                del lora_model
                model.lm_head.load_state_dict(original_lm_head_state)
            except Exception as e:
                logging.warning(f"Skipping lora_lmhead strategy: {e}")
        else:
            logging.warning("Skipping lora_lmhead strategy: --lora-lmhead-checkpoint not provided or model has no lm_head")
    if "lora_crossbatch" in selected_strategies:
        # Load LoRA + lm_head + cross-batch checkpoint
        import torch
        if args.lora_crossbatch_checkpoint and hasattr(model, 'lm_head'):
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                from src.cross_batch.attention import CrossBatchAttention
                checkpoint = torch.load(args.lora_crossbatch_checkpoint, map_location=model.device)
                original_lm_head_state = {k: v.clone() for k, v in model.lm_head.state_dict().items()}
                if 'lm_head' in checkpoint:
                    model.lm_head.load_state_dict(checkpoint['lm_head'])
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                )
                lora_model = get_peft_model(model, lora_config)
                if 'lora' in checkpoint:
                    current_state = lora_model.state_dict()
                    current_state.update(checkpoint['lora'])
                    lora_model.load_state_dict(current_state)
                # Load cross-batch module
                cross_batch_module = CrossBatchAttention(hidden_size=model.config.hidden_size)
                if 'cross_batch_module' in checkpoint:
                    cross_batch_module.load_state_dict(checkpoint['cross_batch_module'])
                cross_batch_module = cross_batch_module.to(model.device)
                results.append(run_cross_batch_strategy(
                    background,
                    questions,
                    tokenizer,
                    lora_model.base_model,
                    max_new_tokens=args.max_new_tokens,
                    strategy_name="lora_crossbatch",
                    dataset=effective_dataset,
                    mix_method="attention",
                    mix_layer=-1,
                    checkpoint_path=None,
                    enable_cross_batch=True,
                    cross_batch_module=cross_batch_module,
                ))
                del lora_model, cross_batch_module
                model.lm_head.load_state_dict(original_lm_head_state)
            except Exception as e:
                logging.warning(f"Skipping lora_crossbatch strategy: {e}")
        else:
            logging.warning("Skipping lora_crossbatch strategy: --lora-crossbatch-checkpoint not provided or model has no lm_head")
    return results


def compute_aggregate_metrics(serialized_contexts: List[dict], dataset: str = "squad") -> str:
    """Compute and format aggregate metrics across all contexts and strategies.

    Metrics are weighted by the number of questions in each context group,
    so groups with more questions contribute proportionally more to the average.

    Args:
        serialized_contexts: List of serialized context results
        dataset: Dataset name to determine which metrics to display

    Returns:
        Formatted string with aggregate metrics
    """
    preferred_order = [
        "all_in_one",
        "sequential",
        "batch",
        "finetuned",
        "collab_llm",
        "collab_bert",
        "collab_hidden",
        "lora_lmhead",
        "lora_crossbatch",
        # vLLM strategies
        "all_in_one_vllm",
        "sequential_vllm",
        "batch_vllm",
        "collab_llm_vllm",
        "collab_bert_vllm",
    ]

    # Get the metric names for this dataset
    metric_names = get_metric_names(dataset)
    llm_metric_names = ["llm_fluency", "llm_relevance", "llm_completeness", "llm_proficiency", "llm_average"]

    strategy_totals: Dict[str, Dict[str, float]] = {}

    for ctx in serialized_contexts:
        strategies = ctx.get("strategies", {})
        # Get number of questions in this context
        # Try different ways to get question count
        questions_text = ctx.get("questions_text", [])
        n_questions = len(questions_text) if isinstance(questions_text, list) else 1

        # Handle both dict format (new) and list format (legacy)
        if isinstance(strategies, list):
            strategy_items = [(s.get("name", "unknown"), s) for s in strategies]
        else:
            strategy_items = list(strategies.items())

        for name, strategy in strategy_items:
            metrics = strategy.get("metrics", {})
            # Fallback: try to get question count from answers dict
            if n_questions <= 1:
                answers = strategy.get("answers", {})
                if answers:
                    n_questions = len(answers)

            if name not in strategy_totals:
                strategy_totals[name] = {
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "latency": 0.0,
                    "context_count": 0,  # Number of context groups
                    "question_count": 0,  # Total number of questions (for weighted avg)
                    "batches": 0,
                }
                # Initialize dataset-specific metrics (weighted sums)
                for m in metric_names:
                    strategy_totals[name][m] = 0.0
                # Initialize LLM metrics (weighted sums)
                for m in llm_metric_names:
                    strategy_totals[name][m] = 0.0

            stats = strategy_totals[name]
            # Accumulate dataset-specific metrics weighted by question count
            # metrics[m] is the average for this context, multiply by n_questions to get sum
            for m in metric_names:
                stats[m] += metrics.get(m, 0.0) * n_questions
            # Accumulate LLM metrics weighted by question count
            for m in llm_metric_names:
                stats[m] += metrics.get(m, 0.0) * n_questions
            stats["prompt_tokens"] += strategy.get("prompt_tokens", 0)
            stats["generated_tokens"] += strategy.get("generated_tokens", 0)
            stats["latency"] += strategy.get("latency", 0.0)
            stats["batches"] += strategy.get("batches", 0)
            stats["context_count"] += 1
            stats["question_count"] += n_questions

    summary_lines = ["\n=== Aggregate Metrics ==="]

    # Check if we have LLM metrics
    has_llm_metrics = any(
        strategy_totals.get(n, {}).get("llm_average", 0) > 0
        for n in strategy_totals
    )

    # Build header based on dataset metrics
    if dataset == "cmb":
        if has_llm_metrics:
            header = "Strategy | BLEU-4 | R-1 | R-2 | R-L | LLM-Avg | Latency(s) | Batches"
        else:
            header = "Strategy | BLEU-4 | R-1 | R-2 | R-L | PromptTok | GenTok | Latency(s) | Batches"
    elif dataset == "cmb_exam":
        header = "Strategy | Acc | PromptTok | GenTok | Latency(s) | Batches"
    else:
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
        # Use question_count for metrics (weighted average), context_count for other stats
        q_count = stats["question_count"] or 1
        ctx_count = stats["context_count"] or 1
        avg_prompt = stats["prompt_tokens"] / ctx_count
        avg_gen = stats["generated_tokens"] / ctx_count
        avg_latency = stats["latency"] / ctx_count
        avg_batches = stats["batches"] / ctx_count

        if dataset == "cmb":
            if has_llm_metrics:
                summary_lines.append(
                    f"{name:<13} | "
                    f"{stats['bleu4']/q_count:.3f} | "
                    f"{stats['rouge1']/q_count:.3f} | "
                    f"{stats['rouge2']/q_count:.3f} | "
                    f"{stats['rougeL']/q_count:.3f} | "
                    f"{stats['llm_average']/q_count:.2f} | "
                    f"{avg_latency:.2f} | "
                    f"{avg_batches:.2f}"
                )
            else:
                summary_lines.append(
                    f"{name:<13} | "
                    f"{stats['bleu4']/q_count:.3f} | "
                    f"{stats['rouge1']/q_count:.3f} | "
                    f"{stats['rouge2']/q_count:.3f} | "
                    f"{stats['rougeL']/q_count:.3f} | "
                    f"{avg_prompt:.2f} | "
                    f"{avg_gen:.2f} | "
                    f"{avg_latency:.2f} | "
                    f"{avg_batches:.2f}"
                )
        elif dataset == "cmb_exam":
            summary_lines.append(
                f"{name:<13} | "
                f"{stats.get('acc', 0)/q_count:.3f} | "
                f"{avg_prompt:.2f} | "
                f"{avg_gen:.2f} | "
                f"{avg_latency:.2f} | "
                f"{avg_batches:.2f}"
            )
        else:
            summary_lines.append(
                f"{name:<13} | "
                f"{stats['strict_acc']/q_count:.3f} | "
                f"{stats['f1']/q_count:.3f} | "
                f"{stats['lenient_acc']/q_count:.3f} | "
                f"{avg_prompt:.2f} | "
                f"{avg_gen:.2f} | "
                f"{avg_latency:.2f} | "
                f"{avg_batches:.2f}"
            )

    # If LLM metrics present, add detailed breakdown
    if has_llm_metrics:
        summary_lines.append("\n--- LLM Evaluation Details ---")
        summary_lines.append("Strategy | Fluency | Relevance | Completeness | Proficiency | Average")
        summary_lines.append("-" * 70)
        for name in ordered_names:
            stats = strategy_totals.get(name)
            if not stats:
                continue
            q_count = stats["question_count"] or 1
            summary_lines.append(
                f"{name:<13} | "
                f"{stats['llm_fluency']/q_count:.2f} | "
                f"{stats['llm_relevance']/q_count:.2f} | "
                f"{stats['llm_completeness']/q_count:.2f} | "
                f"{stats['llm_proficiency']/q_count:.2f} | "
                f"{stats['llm_average']/q_count:.2f}"
            )

    return "\n".join(summary_lines)


def _parse_qid_from_string(s: str) -> tuple:
    """Parse 'Qx: content' format and return (qid, content)."""
    if ": " in s:
        qid, content = s.split(": ", 1)
        return qid.strip(), content
    return "", s


def extract_error_cases(serialized_contexts: List[dict]) -> List[dict]:
    """Extract contexts containing questions where at least one strategy answered incorrectly."""
    error_contexts = []
    for ctx in serialized_contexts:
        # Handle list format (new) or dict format (legacy)
        gold_answers_raw = ctx.get("gold_answers", [])
        if isinstance(gold_answers_raw, list):
            # New list format: ["Q1: answer1", "Q2: answer2"]
            gold_answers = {}
            for item in gold_answers_raw:
                qid, content = _parse_qid_from_string(item)
                if qid:
                    gold_answers[qid] = [content] if content else []
        else:
            # Legacy dict format
            gold_answers = gold_answers_raw

        strategies = ctx.get("strategies", {})
        # Handle both dict format (new) and list format (legacy)
        if isinstance(strategies, list):
            strategy_items = [(s.get("name", "unknown"), s) for s in strategies]
        else:
            strategy_items = list(strategies.items())

        # Check each question
        error_qids = set()
        for qid, refs in gold_answers.items():
            # Check if all strategies got this question correct
            all_correct = True
            for name, strategy in strategy_items:
                answer = strategy.get("answers", {}).get(qid, "")
                if compute_em(answer, refs) < 1.0:
                    all_correct = False
                    break

            if not all_correct:
                error_qids.add(qid)

        # If any question has errors, include this context (filtered)
        if error_qids:
            # Filter questions_text and gold_answers to only error cases
            questions_text_raw = ctx.get("questions_text", [])
            if isinstance(questions_text_raw, list):
                error_questions_text = [q for q in questions_text_raw if _parse_qid_from_string(q)[0] in error_qids]
                error_gold_answers = [g for g in gold_answers_raw if _parse_qid_from_string(g)[0] in error_qids]
            else:
                error_questions_text = [f"{qid}: {questions_text_raw.get(qid, '')}" for qid in error_qids]
                error_gold_answers = [f"{qid}: {gold_answers.get(qid, [''])[0]}" for qid in error_qids]

            # Build strategies data for error cases
            error_strategies = {}
            for name, s in strategy_items:
                error_strategies[name] = {
                    "answers": {qid: s.get("answers", {}).get(qid, "") for qid in error_qids},
                    "metrics": s.get("metrics"),
                }

            filtered_ctx = {
                "context": ctx.get("context"),
                "questions_text": error_questions_text,
                "gold_answers": error_gold_answers,
                "strategies": error_strategies,
            }
            error_contexts.append(filtered_ctx)

    return error_contexts


def generate_output_folder_name(args: argparse.Namespace, timestamp: str) -> str:
    """Generate descriptive output folder name with timestamp and experiment parameters."""
    # Use api_model if specified, otherwise use model_name
    model_name = args.api_model if args.use_api and args.api_model else args.model_name
    model_short = model_name.split("/")[-1]
    # Build dataset identifier: squad_train, hotpot_distractor_train, cmb_CMB-Clin_test, quac_train
    if args.dataset == "hotpot":
        dataset_id = f"hotpot_{args.hotpot_subset}_{args.split}"
    elif args.dataset == "cmb":
        dataset_id = f"cmb_{args.cmb_subset}_{args.split}"
    elif args.dataset == "quac":
        dataset_id = f"quac_{args.split}"
    elif args.dataset == "quality":
        hard_suffix = "_hard" if args.quality_hard_only else ""
        dataset_id = f"quality{hard_suffix}_{args.split}"
    elif args.dataset == "drop":
        dataset_id = f"drop_{args.split}"
    else:
        dataset_id = f"squad_{args.split}"
    # Format: timestamp_dataset_model_ctx{context_count}_q{questions}
    return f"{timestamp}_{dataset_id}_{model_short}_ctx{args.context_count}_q{args.min_questions}-{args.max_questions}"


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
    elif args.dataset == "cmb":
        config["cmb_subset"] = args.cmb_subset
    elif args.dataset == "quality":
        config["quality_hard_only"] = args.quality_hard_only

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
        elif args.dataset == "cmb":
            f.write(f"  - HuggingFace: FreedomIntelligence/CMB ({args.cmb_subset})\n")
        elif args.dataset == "quality":
            hard_str = " (hard only)" if args.quality_hard_only else ""
            f.write(f"  - HuggingFace: emozilla/quality{hard_str}\n")
        elif args.dataset == "drop":
            f.write(f"  - HuggingFace: ucinlp/drop\n")
        elif args.dataset == "quac":
            f.write(f"  - HuggingFace: quac\n")
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
    # Determine evaluation dataset (CMB-Exam subsets use accuracy, CMB-Clin uses BLEU/ROUGE)
    eval_dataset = get_eval_dataset(args)
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

    # API mode: use API for inference instead of local model
    api_client = None
    tokenizer = None
    model = None

    if args.use_api:
        api_model = args.api_model or args.model_name
        # Convert api_provider to provider_order list for OpenRouter routing
        provider_order = [args.api_provider] if args.api_provider else None
        logging.info("Using API mode with model: %s (provider_order: %s)", api_model, provider_order or "auto")
        api_client = APIClient(
            model=api_model,
            base_url=args.api_base_url,
            provider_order=provider_order,
        )
        # For API mode, we still need a tokenizer for prompt formatting (use a small one)
        # Try to load a lightweight tokenizer for chat template
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            logging.info("Loaded lightweight tokenizer for prompt formatting")
        except Exception:
            tokenizer = None
            logging.info("No tokenizer loaded - will use API directly")
    else:
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
    elif args.dataset == "cmb":
        if args.cmb_subset == "random":
            # Original CMB-Exam with random grouping (no shared context)
            contexts = load_cmb_exam_random_groups(
                args.split,
                questions_per_group=args.max_questions or 5,
                max_contexts=args.context_count,
                seed=args.seed,
            )
        elif args.cmb_subset == "subdomain":
            contexts = load_cmb_exam_subdomain_groups(
                args.split,
                min_questions=args.min_questions,
                max_questions=args.max_questions,
                max_contexts=args.context_count,
                seed=args.seed,
            )
        elif args.cmb_subset == "context":
            contexts = load_cmb_exam_context_groups(
                args.split,
                min_questions=args.min_questions,
                max_questions=args.max_questions,
                max_contexts=args.context_count,
                seed=args.seed,
            )
        else:
            # Original CMB-Clin or other FreedomIntelligence/CMB subsets
            contexts = load_cmb_groups(
                args.split,
                subset=args.cmb_subset,
                min_questions=args.min_questions,
                max_questions=args.max_questions,
                max_contexts=args.context_count,
                seed=args.seed,
            )
    elif args.dataset == "quac":
        contexts = load_quac_groups(
            args.split,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            max_contexts=args.context_count,
            seed=args.seed,
        )
    elif args.dataset == "quality":
        contexts = load_quality_groups(
            args.split,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            max_contexts=args.context_count,
            seed=args.seed,
            hard_only=args.quality_hard_only,
        )
    elif args.dataset == "drop":
        contexts = load_drop_groups(
            args.split,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            max_contexts=args.context_count,
            seed=args.seed,
        )
    elif args.dataset == "cmb_exam_context":
        # CMB-Exam: support slice syntax (e.g., "train[50:]", "train[:100]")
        # Only convert exact matches, preserve slice syntax
        split = args.split
        if split == "train":  # Exact match only, not "train[50:]"
            split = "test"  # Legacy: convert train to test
        elif split == "validation":
            split = "val"
        contexts = load_cmb_exam_context_groups(
            split,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            max_contexts=args.context_count,
            seed=args.seed,
        )
    elif args.dataset == "cmb_exam_subdomain":
        # Support slice syntax
        split = args.split
        if split == "train":
            split = "test"
        elif split == "validation":
            split = "val"
        contexts = load_cmb_exam_subdomain_groups(
            split,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            max_contexts=args.context_count,
            seed=args.seed,
        )
    elif args.dataset == "cmb_exam_random":
        # Support slice syntax
        split = args.split
        if split == "train":
            split = "test"
        elif split == "validation":
            split = "val"
        contexts = load_cmb_exam_random_groups(
            split,
            questions_per_group=args.max_questions or 5,
            max_contexts=args.context_count,
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

    dep_generator, bert_dep_generator, bert_conf_threshold = resolve_dependency_generators(args, tokenizer, model, api_client)

    # Initialize LLM evaluator if requested
    llm_evaluator = None
    if args.eval_model:
        try:
            from src.evaluation.llm import OpenRouterEvaluator
            llm_evaluator = OpenRouterEvaluator(model=args.eval_model)
            logging.info("Initialized LLM evaluator with model: %s", args.eval_model)
        except Exception as e:
            logging.warning("Failed to initialize LLM evaluator: %s", e)

    # Parse selected strategies
    selected_strategies = None
    if args.strategies:
        selected_strategies = [s.strip() for s in args.strategies.split(",")]
        invalid = [s for s in selected_strategies if s not in ALL_STRATEGIES]
        if invalid:
            raise ValueError(f"Invalid strategies: {invalid}. Available: {ALL_STRATEGIES}")
    else:
        selected_strategies = ALL_STRATEGIES.copy()

    # Auto-discover checkpoints if requested
    if args.auto_checkpoints:
        logging.info("Auto-discovering checkpoints from: %s/%s/", args.checkpoint_dir, args.dataset)
        discovered = auto_find_checkpoints(args)
        # Use discovered paths if explicit paths not provided
        if not args.baseline_checkpoint and discovered.get('baseline'):
            args.baseline_checkpoint = discovered['baseline']
        if not args.collab_hidden_checkpoint and discovered.get('crossbatch'):
            args.collab_hidden_checkpoint = discovered['crossbatch']
        if not args.lora_lmhead_checkpoint and discovered.get('lora_lmhead'):
            args.lora_lmhead_checkpoint = discovered['lora_lmhead']
        if not args.lora_crossbatch_checkpoint and discovered.get('lora_crossbatch'):
            args.lora_crossbatch_checkpoint = discovered['lora_crossbatch']

    # Skip collab_hidden if no checkpoint provided
    if "collab_hidden" in selected_strategies and not args.collab_hidden_checkpoint:
        logging.warning("Skipping collab_hidden strategy: --collab-hidden-checkpoint not provided")
        selected_strategies.remove("collab_hidden")

    # Skip lora_lmhead if no checkpoint provided
    if "lora_lmhead" in selected_strategies and not args.lora_lmhead_checkpoint:
        logging.warning("Skipping lora_lmhead strategy: --lora-lmhead-checkpoint not provided")
        selected_strategies.remove("lora_lmhead")

    # Skip lora_crossbatch if no checkpoint provided
    if "lora_crossbatch" in selected_strategies and not args.lora_crossbatch_checkpoint:
        logging.warning("Skipping lora_crossbatch strategy: --lora-crossbatch-checkpoint not provided")
        selected_strategies.remove("lora_crossbatch")

    logging.info("Running strategies: %s", selected_strategies)

    # Create vLLM model if requested
    vllm_model = None
    if args.use_vllm and not args.use_api:
        logging.info("Initializing vLLM model for comparison...")
        vllm_model = create_vllm_model(args.model_name)
        if vllm_model:
            logging.info("vLLM model initialized successfully")
        else:
            logging.warning("Failed to create vLLM model, skipping vLLM strategies")

    overall_results: Dict[str, List[StrategyResult]] = {}
    serialized_contexts: List[dict] = []
    vllm_serialized_contexts: List[dict] = []  # Separate storage for vLLM results

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
                api_client=api_client,
                eval_dataset=eval_dataset,
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
                api_client=api_client,
                eval_dataset=eval_dataset,
            )
        overall_results[title] = strategy_list

        print(f"\n=== Context: {title} ===")
        print(summarize_results(strategy_list, dataset=eval_dataset))
        print_answer_table(questions, strategy_list, dataset=eval_dataset)

        # Build serialization structure (simple list format)
        questions_text = [f"{q.qid}: {q.text.strip()}" for q in questions]
        gold_answers = [f"{q.qid}: {q.references[0] if q.references else ''}" for q in questions]

        # Get dataset-specific metrics
        dataset_metrics = get_dataset_metrics(eval_dataset)

        strategies_data = {}
        for res in strategy_list:
            # Start with base metrics from strategy result
            metrics_dict = {k: round(v, 4) for k, v in res.metrics.items()}

            # Compute dataset-specific metrics
            metric_sums = {name: 0.0 for name in dataset_metrics}
            for q in questions:
                pred = res.answers.get(q.qid, "")
                refs = q.references
                for metric_name, metric_func in dataset_metrics.items():
                    metric_sums[metric_name] += metric_func(pred, refs)

            n_questions = len(questions) or 1
            for metric_name, total in metric_sums.items():
                metrics_dict[metric_name] = round(total / n_questions, 4)

            # Compute LLM evaluation metrics if evaluator is available
            if llm_evaluator:
                try:
                    from src.evaluation.llm import compute_llm_metrics
                    eval_items = []
                    for q in questions:
                        pred = res.answers.get(q.qid, "")
                        ref = q.references[0] if q.references else ""
                        eval_items.append((background, q.text, ref, pred))
                    llm_results = llm_evaluator.evaluate_batch(eval_items, show_progress=False)
                    llm_metrics = compute_llm_metrics(llm_results)
                    metrics_dict.update({k: round(v, 4) for k, v in llm_metrics.items()})
                except Exception as e:
                    logging.warning("LLM evaluation failed for strategy %s: %s", res.name, e)

            strategy_entry = {
                "answers": res.answers,
                "metrics": metrics_dict,
                "prompt_tokens": res.prompt_tokens,
                "generated_tokens": res.generated_tokens,
                "latency": round(res.latency, 2),
                "batches": res.batches,
                "details": res.details,  # Raw prompts, responses, per-turn/per-question data
            }
            # For collaborative strategies (collab_llm, collab_bert), promote DAG info to top level
            if res.details and "dependency_stage" in res.details:
                dep_stage = res.details["dependency_stage"]
                strategy_entry["dag_edges"] = dep_stage.get("edges", [])
                strategy_entry["dag_generation"] = {
                    "prompt_tokens": dep_stage.get("prompt_tokens", 0),
                    "generated_tokens": dep_stage.get("generated_tokens", 0),
                    "latency": dep_stage.get("latency", 0),
                }
            # Keep batch execution details if present
            if res.details and "batches" in res.details:
                strategy_entry["batch_details"] = res.details["batches"]
            strategies_data[res.name] = strategy_entry

        serialized_contexts.append({
            "context": title,
            "questions_text": questions_text,
            "gold_answers": gold_answers,
            "strategies": strategies_data,
        })

        # Run vLLM strategies if model is available
        if vllm_model is not None:
            # Determine which vLLM strategies to run
            vllm_strategies_to_run = [s for s in VLLM_STRATEGIES if s in selected_strategies]
            if vllm_strategies_to_run:
                # Build items for vLLM (need context per item)
                if "items" in context_payload:
                    vllm_items = items
                else:
                    vllm_items = [
                        {
                            "qid": q.qid,
                            "question": q.text,
                            "context": background,
                            "references": q.references,
                        }
                        for q in questions
                    ]

                vllm_strategy_list = run_vllm_strategies(
                    vllm_items,
                    vllm_model,
                    tokenizer,
                    dep_generator,
                    bert_dep_generator,
                    args=args,
                    bert_conf_threshold=bert_conf_threshold,
                    selected_strategies=vllm_strategies_to_run,
                    eval_dataset=eval_dataset,
                )

                # Build vLLM serialization structure
                vllm_strategies_data = {}
                for res in vllm_strategy_list:
                    metrics_dict = {k: round(v, 4) for k, v in res.metrics.items()}
                    metric_sums = {name: 0.0 for name in dataset_metrics}
                    for q in questions:
                        pred = res.answers.get(q.qid, "")
                        refs = q.references
                        for metric_name, metric_func in dataset_metrics.items():
                            metric_sums[metric_name] += metric_func(pred, refs)
                    n_questions = len(questions) or 1
                    for metric_name, total in metric_sums.items():
                        metrics_dict[metric_name] = round(total / n_questions, 4)

                    vllm_strategies_data[res.name] = {
                        "answers": res.answers,
                        "metrics": metrics_dict,
                        "prompt_tokens": res.prompt_tokens,
                        "generated_tokens": res.generated_tokens,
                        "latency": round(res.latency, 2),
                        "batches": res.batches,
                        "details": res.details,
                    }

                vllm_serialized_contexts.append({
                    "context": title,
                    "questions_text": questions_text,
                    "gold_answers": gold_answers,
                    "strategies": vllm_strategies_data,
                })

    # Distributed output handling: gather to rank0 and write a single file
    if world_size > 1 and dist.is_initialized():
        # Use torch.distributed to gather results from all ranks
        gather_list: List[List[dict]] = [None for _ in range(world_size)]  # type: ignore
        dist.all_gather_object(gather_list, serialized_contexts)
        vllm_gather_list: List[List[dict]] = [None for _ in range(world_size)]  # type: ignore
        if vllm_serialized_contexts:
            dist.all_gather_object(vllm_gather_list, vllm_serialized_contexts)
        if rank == 0:
            merged = []
            for ctx_list in gather_list:
                if ctx_list:
                    merged.extend(ctx_list)
            # Print aggregate metrics from all ranks (only on rank 0)
            print("\n" + "=" * 60)
            print("Regular Inference Results")
            print("=" * 60)
            print(compute_aggregate_metrics(merged, dataset=eval_dataset))
            save_experiment_results(args.json_out, merged, args, output_folder_name)

            # Print vLLM results if available
            if any(vllm_gather_list):
                vllm_merged = []
                for ctx_list in vllm_gather_list:
                    if ctx_list:
                        vllm_merged.extend(ctx_list)
                if vllm_merged:
                    print("\n" + "=" * 60)
                    print("vLLM Inference Results")
                    print("=" * 60)
                    print(compute_aggregate_metrics(vllm_merged, dataset=eval_dataset))
    else:
        # Single process mode: print and save local results
        print("\n" + "=" * 60)
        print("Regular Inference Results")
        print("=" * 60)
        print(compute_aggregate_metrics(serialized_contexts, dataset=eval_dataset))
        save_experiment_results(args.json_out, serialized_contexts, args, output_folder_name)

        # Print vLLM results if available
        if vllm_serialized_contexts:
            print("\n" + "=" * 60)
            print("vLLM Inference Results")
            print("=" * 60)
            print(compute_aggregate_metrics(vllm_serialized_contexts, dataset=eval_dataset))

    # Cleanup distributed backend
    if dist.is_initialized():
        dist.destroy_process_group()

    # Cleanup vLLM model
    if vllm_model is not None:
        del vllm_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
