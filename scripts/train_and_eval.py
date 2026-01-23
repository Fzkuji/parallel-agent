"""
Complete training and evaluation pipeline for cross-batch module.

This script:
1. Runs baseline evaluation using vLLM (fast, single-sample inference)
2. Trains cross-batch module using transformers
3. Evaluates cross-batch model using transformers

Baseline uses vLLM for fast inference (same as exp2a_shared_context.py).
Cross-batch uses transformers for compatibility with the training code.

Usage:
  Multi-GPU (8 GPUs):
    torchrun --nproc_per_node=8 scripts/train_and_eval.py \
        --model Qwen/Qwen2.5-7B-Instruct --epochs 3 --cache-baseline

  Single GPU:
    python scripts/train_and_eval.py \
        --model Qwen/Qwen2.5-7B-Instruct --epochs 3 --cache-baseline
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.squad import load_squad_groups
from src.datasets.hotpot import load_hotpot_groups
from src.models import Question
from src.strategies.cross_batch import run_cross_batch_multi_strategy
from src.cross_batch import (
    CrossBatchTrainer,
    SQuADGroupedDataset,
    SimpleCrossBatchGate,
    MultiLayerCrossBatch,
    MultiLayerCrossBatchAttention,
    CrossBatchAttention,
    CrossBatchEmbeddingMixer,
)
from src.evaluation import evaluate_predictions
from src.inference import extract_answer

# Simple prompt format (same as exp2a_shared_context.py)
SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the given passage.
Give a short, direct answer. Do not explain or elaborate."""


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate cross-batch module")

    # Model and dataset
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--dataset", type=str, default="squad", help="Dataset name")
    parser.add_argument("--train-samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--eval-samples", type=int, default=100, help="Number of evaluation samples")
    parser.add_argument("--min-questions", type=int, default=3, help="Min questions per context")
    parser.add_argument("--max-questions", type=int, default=5, help="Max questions per context")

    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (contexts per batch)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--module-type", type=str, default="multi_layer", choices=["simple", "multi_layer", "multi_layer_attention", "attention", "mixer"], help="Cross-batch module type")
    parser.add_argument("--mix-layers", type=str, default=None, help="Comma-separated layer indices for multi_layer mode (None = all layers)")

    # Inference
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max new tokens for generation")

    # Paths
    parser.add_argument("--output-dir", type=str, default="outputs/train_and_eval", help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints", help="Checkpoint directory")
    parser.add_argument("--cache-baseline", action="store_true", help="Cache baseline results to avoid recomputation")

    # LoRA parameters
    parser.add_argument("--use-lora", action="store_true", default=False, help="Use LoRA for model training")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="LoRA target modules (comma-separated)")

    # vLLM options
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode for Qwen3 models")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    return parser.parse_args()


def context_to_items(context_payload: dict) -> List[dict]:
    """Convert context format to items format for batch strategy.

    Supports both SQuAD and HotpotQA formats.
    """
    # HotpotQA format: already has items
    if "items" in context_payload:
        return context_payload["items"]

    # SQuAD format: convert from shared context
    context = context_payload["context"]
    items = []
    for q in context_payload["questions"]:
        items.append({
            "qid": q["qid"],
            "question": q["text"],
            "context": context,
            "references": q["references"],
            "answer_tokens": q.get("answer_tokens", 12),
        })
    return items


class VLLMClient:
    """vLLM client for fast baseline inference."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        enable_thinking: bool = False,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Please install vllm: pip install vllm")

        self.enable_thinking = enable_thinking

        # Suppress vLLM verbose logging
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

        logging.info(f"Loading vLLM model: {model} (tensor_parallel_size={tensor_parallel_size})")

        # Load tokenizer for chat template support
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._vllm_model = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="half",
            gpu_memory_utilization=0.9,
            disable_log_stats=True,
        )
        logging.info(f"vLLM model loaded, enable_thinking={enable_thinking}")

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        system_prompt: Optional[str] = None,
    ) -> List[Tuple[str, int, int, float]]:
        """Generate responses for multiple prompts in batch.

        Returns:
            List of (response_text, prompt_tokens, completion_tokens, latency) tuples
        """
        from vllm import SamplingParams

        # Build full prompts using chat template
        full_prompts = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            if hasattr(self._tokenizer, "apply_chat_template"):
                try:
                    full_prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.enable_thinking,
                    )
                except TypeError:
                    full_prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            else:
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                else:
                    full_prompt = prompt
            full_prompts.append(full_prompt)

        # Greedy decoding for reproducibility
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
        )

        start_time = time.perf_counter()
        outputs = self._vllm_model.generate(full_prompts, sampling_params, use_tqdm=False)
        total_latency = time.perf_counter() - start_time

        results = []
        for output in outputs:
            text = output.outputs[0].text
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
            results.append((text, prompt_tokens, completion_tokens, total_latency / len(prompts)))

        return results


def run_baseline_vllm(
    items: List[Dict],
    vllm_client: VLLMClient,
    max_new_tokens: int,
    dataset: str,
) -> Dict[str, Any]:
    """Run baseline evaluation using vLLM (single-sample inference).

    Uses the same simple prompt format as exp2a_shared_context.py.
    """
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

    # Build prompts for all items (simple format, context in user message)
    prompts = []
    for item in items:
        prompt = f"Passage:\n{item['context']}\n\nQuestion: {item['question']}"
        prompts.append(prompt)

    # Generate all responses in batch
    results = vllm_client.generate_batch(
        prompts,
        max_tokens=max_new_tokens,
        system_prompt=SYSTEM_PROMPT,
    )

    # Extract answers and compute metrics
    answer_records = {}
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, item in enumerate(items):
        raw_text, prompt_tokens, completion_tokens, latency = results[idx]
        final_answer, strict_valid = extract_answer(raw_text, dataset)

        answer_records[item["qid"]] = (final_answer, strict_valid)
        total_latency += latency
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)

    return {
        "metrics": metrics,
        "latency": total_latency,
        "prompt_tokens": total_prompt_tokens,
        "generated_tokens": total_completion_tokens,
    }


def run_baseline_on_shard_vllm(
    args,
    eval_contexts: List[Dict],
    vllm_client: VLLMClient,
) -> Dict[str, Any]:
    """Run baseline strategy on this rank's shard using vLLM."""
    shard_results = {"contexts": []}

    for idx, context_payload in enumerate(eval_contexts, start=1):
        items = context_to_items(context_payload)
        title = context_payload.get("title", f"context-{idx}")

        logging.info(f"Baseline: Processing {idx}/{len(eval_contexts)}: {title}")

        result = run_baseline_vllm(
            items,
            vllm_client,
            max_new_tokens=args.max_new_tokens,
            dataset=args.dataset,
        )

        context_result = {
            "title": title,
            "metrics": result["metrics"],
            "latency": result["latency"],
            "prompt_tokens": result["prompt_tokens"],
            "generated_tokens": result["generated_tokens"],
            "num_questions": len(items),
        }
        shard_results["contexts"].append(context_result)

    return shard_results


def load_or_run_baseline(
    args,
    eval_contexts: List[Dict],
    cache_path: Path,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    """Load cached baseline or run on all ranks and gather."""
    # Rank 0 checks cache
    if rank == 0 and cache_path.exists() and args.cache_baseline:
        logging.info(f"Loading cached baseline from {cache_path}")
        with open(cache_path, 'r') as f:
            baseline_results = json.load(f)
        return baseline_results

    # Initialize vLLM client for this rank
    # For multi-GPU, each rank uses its own GPU
    if world_size > 1 and torch.cuda.is_available():
        # Set CUDA_VISIBLE_DEVICES for this rank
        gpu_id = rank % torch.cuda.device_count()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Disable vLLM V1 engine for compatibility
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["VLLM_DISABLE_FRONTEND_MULTIPROCESSING"] = "1"

    vllm_client = VLLMClient(
        model=args.model,
        tensor_parallel_size=1,
        enable_thinking=args.enable_thinking,
    )

    # All ranks run baseline on their shard
    logging.info("Running baseline on shard...")
    shard_results = run_baseline_on_shard_vllm(args, eval_contexts, vllm_client)
    logging.info(f"Baseline shard complete: {len(shard_results['contexts'])} contexts")

    # Free vLLM model to save memory for training
    del vllm_client
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Gather results
    if world_size > 1 and dist.is_initialized():
        logging.info("Waiting for all ranks to finish baseline...")
        dist.barrier()
        logging.info("All ranks ready, gathering results...")
        all_shards = [None for _ in range(world_size)]
        dist.all_gather_object(all_shards, shard_results)

        if rank == 0:
            # Merge
            all_contexts = []
            for shard in all_shards:
                if shard:
                    all_contexts.extend(shard["contexts"])

            # Aggregate
            total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in all_contexts)
            total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in all_contexts)
            total_questions = sum(ctx["num_questions"] for ctx in all_contexts)
            total_latency = sum(ctx["latency"] for ctx in all_contexts)
            total_prompt_tokens = sum(ctx["prompt_tokens"] for ctx in all_contexts)
            total_generated_tokens = sum(ctx["generated_tokens"] for ctx in all_contexts)

            baseline_results = {
                "aggregate_metrics": {
                    "strict_acc": total_em / total_questions if total_questions > 0 else 0,
                    "f1": total_f1 / total_questions if total_questions > 0 else 0,
                    "avg_latency": total_latency / len(all_contexts),
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_generated_tokens": total_generated_tokens,
                },
                "contexts": all_contexts,
            }

            # Save cache
            if args.cache_baseline:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(baseline_results, f, indent=2)
                logging.info(f"Saved baseline to {cache_path}")

            return baseline_results
        else:
            return {}  # Non-zero ranks don't need baseline
    else:
        # Single GPU
        total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in shard_results["contexts"])
        total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in shard_results["contexts"])
        total_questions = sum(ctx["num_questions"] for ctx in shard_results["contexts"])
        total_latency = sum(ctx["latency"] for ctx in shard_results["contexts"])

        baseline_results = {
            "aggregate_metrics": {
                "strict_acc": total_em / total_questions if total_questions > 0 else 0,
                "f1": total_f1 / total_questions if total_questions > 0 else 0,
                "avg_latency": total_latency / len(shard_results["contexts"]) if shard_results["contexts"] else 0,
                "total_prompt_tokens": sum(ctx["prompt_tokens"] for ctx in shard_results["contexts"]),
                "total_generated_tokens": sum(ctx["generated_tokens"] for ctx in shard_results["contexts"]),
            },
            "contexts": shard_results["contexts"],
        }

        if args.cache_baseline:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(baseline_results, f, indent=2)

        return baseline_results


def evaluate_checkpoint_on_shard(
    args,
    checkpoint_path: str,
    eval_contexts: List[Dict],
    tokenizer,
    model,
    epoch: int,
    lora_model=None,
) -> Dict[str, Any]:
    """Evaluate checkpoint on this rank's shard using transformers."""
    mix_layers = None
    if args.mix_layers:
        mix_layers = [int(x.strip()) for x in args.mix_layers.split(',')]

    # Load LoRA weights from checkpoint if using LoRA
    if args.use_lora and lora_model is not None:
        checkpoint = torch.load(checkpoint_path, map_location=model.device if hasattr(model, 'device') else 'cuda')
        if 'lora' in checkpoint:
            current_state = model.state_dict()
            current_state.update(checkpoint['lora'])
            model.load_state_dict(current_state)
            logging.info(f"Loaded {len(checkpoint['lora'])} LoRA tensors from checkpoint")

    shard_results = {"contexts": []}

    for idx, context_payload in enumerate(eval_contexts, start=1):
        items = context_to_items(context_payload)
        title = context_payload.get("title", f"context-{idx}")

        logging.info(f"Epoch {epoch}: Evaluating {idx}/{len(eval_contexts)}: {title}")

        result = run_cross_batch_multi_strategy(
            items,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            strategy_name=f"collab_hidden_epoch{epoch}",
            dataset=args.dataset,
            mix_method=args.module_type,
            mix_layer=-1,
            mix_layers=mix_layers,
            checkpoint_path=checkpoint_path,
            enable_cross_batch=True,
        )

        context_result = {
            "title": title,
            "metrics": result.metrics,
            "latency": result.latency,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "num_questions": len(items),
        }
        shard_results["contexts"].append(context_result)

    return shard_results


def evaluate_checkpoint(
    args,
    checkpoint_path: str,
    eval_contexts: List[Dict],
    tokenizer,
    model,
    epoch: int,
    rank: int,
    world_size: int,
    lora_model=None,
) -> Dict[str, Any]:
    """Evaluate checkpoint across all ranks and gather."""
    # All ranks evaluate their shard
    shard_results = evaluate_checkpoint_on_shard(
        args,
        checkpoint_path,
        eval_contexts,
        tokenizer,
        model,
        epoch,
        lora_model=lora_model,
    )
    logging.info(f"Epoch {epoch} evaluation shard complete: {len(shard_results['contexts'])} contexts")

    # Gather results
    if world_size > 1 and dist.is_initialized():
        logging.info("Waiting for all ranks to finish evaluation...")
        dist.barrier()
        logging.info("All ranks ready, gathering results...")
        all_shards = [None for _ in range(world_size)]
        dist.all_gather_object(all_shards, shard_results)

        if rank == 0:
            # Merge
            all_contexts = []
            for shard in all_shards:
                if shard:
                    all_contexts.extend(shard["contexts"])

            # Aggregate
            total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in all_contexts)
            total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in all_contexts)
            total_questions = sum(ctx["num_questions"] for ctx in all_contexts)
            total_latency = sum(ctx["latency"] for ctx in all_contexts)

            return {
                "epoch": epoch,
                "aggregate_metrics": {
                    "strict_acc": total_em / total_questions if total_questions > 0 else 0,
                    "f1": total_f1 / total_questions if total_questions > 0 else 0,
                    "avg_latency": total_latency / len(all_contexts),
                    "total_prompt_tokens": sum(ctx["prompt_tokens"] for ctx in all_contexts),
                    "total_generated_tokens": sum(ctx["generated_tokens"] for ctx in all_contexts),
                },
            }
        else:
            return {}  # Non-zero ranks
    else:
        # Single GPU
        total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in shard_results["contexts"])
        total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in shard_results["contexts"])
        total_questions = sum(ctx["num_questions"] for ctx in shard_results["contexts"])
        total_latency = sum(ctx["latency"] for ctx in shard_results["contexts"])

        return {
            "epoch": epoch,
            "aggregate_metrics": {
                "strict_acc": total_em / total_questions if total_questions > 0 else 0,
                "f1": total_f1 / total_questions if total_questions > 0 else 0,
                "avg_latency": total_latency / len(shard_results["contexts"]) if shard_results["contexts"] else 0,
                "total_prompt_tokens": sum(ctx["prompt_tokens"] for ctx in shard_results["contexts"]),
                "total_generated_tokens": sum(ctx["generated_tokens"] for ctx in shard_results["contexts"]),
            },
        }


def main():
    args = parse_args()

    # Detect distributed setup
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))

    # Setup logging
    log_prefix = f"[Rank {rank}] " if world_size > 1 else ""
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format=f"{log_prefix}%(message)s",
    )

    # Create output directory (only on rank 0)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize distributed backend
    if world_size > 1:
        if torch.cuda.is_available():
            device_id = rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            if rank == 0:
                logging.info(f"Using {world_size} GPUs for evaluation")
            logging.info(f"Rank {rank} using cuda:{device_id}")

        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            timeout = torch.distributed.timedelta(minutes=30)
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timeout)

    # Load evaluation data (all ranks load full dataset, then shard)
    if rank == 0:
        logging.info(f"Loading evaluation data: {args.eval_samples} samples")

    if args.dataset == "hotpot":
        eval_contexts = load_hotpot_groups(
            split="validation",
            max_contexts=args.eval_samples,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            seed=args.seed + 1000,
        )
    else:  # squad
        eval_contexts = load_squad_groups(
            split="validation",
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            max_contexts=args.eval_samples,
            seed=args.seed + 1000,
        )

    # Shard evaluation data across GPUs
    if world_size > 1:
        eval_contexts = eval_contexts[rank::world_size]
        logging.info(f"Evaluating {len(eval_contexts)} contexts on this shard")

    # Get or compute baseline results using vLLM
    baseline_cache_path = output_dir / "baseline_results.json"
    if rank == 0:
        logging.info("=" * 60)
        logging.info("STEP 1: Baseline (vLLM) Strategy")
        logging.info("=" * 60)

    baseline_results = load_or_run_baseline(
        args,
        eval_contexts,
        baseline_cache_path,
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        logging.info("\nBaseline Results:")
        logging.info(f"  EM:  {baseline_results['aggregate_metrics']['strict_acc']:.4f}")
        logging.info(f"  F1:  {baseline_results['aggregate_metrics']['f1']:.4f}")
        logging.info(f"  Latency: {baseline_results['aggregate_metrics']['avg_latency']:.2f}s")

    # Synchronize before loading transformers model
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    # Load model and tokenizer for training and cross-batch evaluation
    if rank == 0:
        logging.info(f"\nLoading transformers model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # For multi-GPU, each rank loads model on its own GPU
    if world_size > 1 and torch.cuda.is_available():
        device_map = {"": f"cuda:{torch.cuda.current_device()}"}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.eval()

    # Training (all ranks participate)
    safe_model_name = args.model.replace('/', '_')
    checkpoint_base = Path(args.checkpoint_dir) / args.dataset / safe_model_name
    if rank == 0:
        checkpoint_base.mkdir(parents=True, exist_ok=True)
        logging.info("\n" + "=" * 60)
        logging.info("STEP 2: Training Cross-Batch Module")
        logging.info("=" * 60)

    # Apply LoRA if enabled (all ranks)
    lora_model = None
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            lora_model = get_peft_model(model, lora_config)
            if rank == 0:
                lora_model.print_trainable_parameters()
                logging.info(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, target={target_modules}")
            model = lora_model
        except ImportError:
            if rank == 0:
                logging.warning("peft not installed. Cannot use LoRA. Install with: pip install peft")
            args.use_lora = False

    # All ranks load training data
    if rank == 0:
        logging.info(f"Loading training data: {args.train_samples} samples")

    if args.dataset == "hotpot":
        train_groups = load_hotpot_groups(
            split="train",
            max_contexts=args.train_samples,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            seed=args.seed,
        )
    else:  # squad
        train_groups = load_squad_groups(
            split="train",
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            max_contexts=args.train_samples,
            seed=args.seed,
        )

    # Shard training data across GPUs (for DDP)
    if world_size > 1:
        train_groups = train_groups[rank::world_size]
        logging.info(f"Training on {len(train_groups)} contexts (shard {rank}/{world_size})")

    # All ranks create dataset
    train_dataset = SQuADGroupedDataset(
        tokenizer=tokenizer,
        groups=train_groups,
        dataset_name=args.dataset,
    )
    total_questions = sum(len(g.get("questions", g.get("items", []))) for g in train_groups)
    if rank == 0:
        logging.info(f"Training dataset: {len(train_dataset)} contexts, {total_questions} questions (per rank)")

    # Parse mix_layers
    mix_layers = None
    if args.mix_layers:
        mix_layers = [int(x.strip()) for x in args.mix_layers.split(',')]

    # All ranks create cross-batch module
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    if args.module_type == "multi_layer":
        layer_indices = mix_layers if mix_layers else list(range(num_layers))
        cross_batch_module = MultiLayerCrossBatch(
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_indices=layer_indices,
            temperature=1.0,
        )
        if rank == 0:
            logging.info(f"Using MultiLayerCrossBatch with {len(layer_indices)} layers: {layer_indices[:5]}...")
    elif args.module_type == "multi_layer_attention":
        layer_indices = mix_layers if mix_layers else list(range(num_layers))
        cross_batch_module = MultiLayerCrossBatchAttention(
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_indices=layer_indices,
            num_heads=8,
            temperature=1.0,
            use_gate=True,
        )
        if rank == 0:
            logging.info(f"Using MultiLayerCrossBatchAttention with {len(layer_indices)} layers: {layer_indices[:5]}...")
    elif args.module_type == "simple":
        cross_batch_module = SimpleCrossBatchGate(hidden_size=hidden_size, temperature=1.0)
    elif args.module_type == "attention":
        cross_batch_module = CrossBatchAttention(hidden_size=hidden_size, num_heads=8, temperature=1.0)
    else:  # mixer
        cross_batch_module = CrossBatchEmbeddingMixer(hidden_size=hidden_size, temperature=1.0)

    if rank == 0:
        num_params = sum(p.numel() for p in cross_batch_module.parameters())
        logging.info(f"Cross-batch module parameters: {num_params:,}")

    # Create trainer (all ranks)
    device = str(next(model.parameters()).device)
    local_rank = rank if world_size > 1 else -1
    trainer = CrossBatchTrainer(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        device=device,
        learning_rate=args.lr,
        train_lm_head=False,
        train_lora=args.use_lora,
        local_rank=local_rank,
        mix_layer=-1,
    )

    # Results tracking
    all_results = {
        "config": vars(args),
        "baseline": baseline_results.get("aggregate_metrics", {}) if rank == 0 else {},
        "epochs": [],
    }

    # Train and evaluate for each epoch
    for epoch in range(1, args.epochs + 1):
        if rank == 0:
            logging.info(f"\n{'=' * 60}")
            logging.info(f"Epoch {epoch}/{args.epochs}")
            logging.info("=" * 60)

        logging.info("Training...")

        # All ranks train for one epoch (DDP)
        history = trainer.train(
            train_dataset=train_dataset,
            num_epochs=1,
            batch_size=args.batch_size,
            save_dir=None,
            distributed=(world_size > 1),
            rank=rank,
            world_size=world_size,
            grouped=True,
        )

        if rank == 0:
            logging.info(f"Training complete - Loss: {history['train_loss'][-1]:.4f}, Improvement: {history['improvement'][-1]:.4f}")

            # Save checkpoint (only rank 0)
            checkpoint_path = str(checkpoint_base / f"{args.module_type}_frozen_epoch{epoch}.pt")
            if hasattr(trainer, 'cross_batch_module_unwrapped'):
                module_state = trainer.cross_batch_module_unwrapped.state_dict()
            else:
                module_state = cross_batch_module.state_dict()

            checkpoint = {
                'cross_batch_module': module_state,
                'config': {
                    'model': args.model,
                    'dataset': args.dataset,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'train_samples': args.train_samples,
                    'epochs': epoch,
                    'module_type': args.module_type,
                    'mix_layer': -1,
                    'mix_layers': mix_layers,
                    'use_lora': args.use_lora,
                    'lora_r': args.lora_r if args.use_lora else None,
                    'lora_alpha': args.lora_alpha if args.use_lora else None,
                    'lora_target_modules': args.lora_target_modules if args.use_lora else None,
                },
            }
            if args.use_lora and lora_model is not None:
                lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()}
                checkpoint['lora'] = lora_state_dict
                logging.info(f"Saved {len(lora_state_dict)} LoRA tensors")
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

        # Sync all ranks before evaluation
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
            checkpoint_path_list = [str(checkpoint_base / f"{args.module_type}_frozen_epoch{epoch}.pt")]
            dist.broadcast_object_list(checkpoint_path_list, src=0)
            checkpoint_path = checkpoint_path_list[0]
        else:
            checkpoint_path = str(checkpoint_base / f"{args.module_type}_frozen_epoch{epoch}.pt")

        # All ranks evaluate
        if rank == 0:
            logging.info(f"\nEvaluating epoch {epoch}...")

        epoch_results = evaluate_checkpoint(
            args,
            checkpoint_path,
            eval_contexts,
            tokenizer,
            model,
            epoch,
            rank=rank,
            world_size=world_size,
            lora_model=lora_model if args.use_lora else None,
        )

        # Display and save results (only rank 0)
        if rank == 0:
            logging.info(f"\nEpoch {epoch} Results:")
            logging.info(f"  EM:  {epoch_results['aggregate_metrics']['strict_acc']:.4f} "
                        f"(Δ {epoch_results['aggregate_metrics']['strict_acc'] - baseline_results['aggregate_metrics']['strict_acc']:+.4f})")
            logging.info(f"  F1:  {epoch_results['aggregate_metrics']['f1']:.4f} "
                        f"(Δ {epoch_results['aggregate_metrics']['f1'] - baseline_results['aggregate_metrics']['f1']:+.4f})")
            logging.info(f"  Latency: {epoch_results['aggregate_metrics']['avg_latency']:.2f}s")

            all_results["epochs"].append(epoch_results["aggregate_metrics"])

            results_path = output_dir / f"results_epoch{epoch}.json"
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            logging.info(f"Saved results to {results_path}")

    # Final summary (only rank 0)
    if rank == 0:
        logging.info("\n" + "=" * 60)
        logging.info("FINAL SUMMARY")
        logging.info("=" * 60)

        logging.info("\nBaseline (vLLM):")
        logging.info(f"  EM: {baseline_results['aggregate_metrics']['strict_acc']:.4f}")
        logging.info(f"  F1: {baseline_results['aggregate_metrics']['f1']:.4f}")

        logging.info("\nCross-Batch Results by Epoch:")
        for ep, epoch_metrics in enumerate(all_results["epochs"], start=1):
            logging.info(f"\nEpoch {ep}:")
            logging.info(f"  EM: {epoch_metrics['strict_acc']:.4f} "
                        f"(Δ {epoch_metrics['strict_acc'] - baseline_results['aggregate_metrics']['strict_acc']:+.4f})")
            logging.info(f"  F1: {epoch_metrics['f1']:.4f} "
                        f"(Δ {epoch_metrics['f1'] - baseline_results['aggregate_metrics']['f1']:+.4f})")

        # Find best epoch
        if all_results["epochs"]:
            best_epoch = max(range(len(all_results["epochs"])),
                             key=lambda i: all_results["epochs"][i]["f1"]) + 1
            logging.info(f"\nBest Epoch: {best_epoch} (F1: {all_results['epochs'][best_epoch-1]['f1']:.4f})")

        # Save final results
        final_results_path = output_dir / "final_results.json"
        with open(final_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"\nSaved final results to {final_results_path}")

        logging.info("\n" + "=" * 60)
        logging.info("DONE!")
        logging.info("=" * 60)

    # Cleanup
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
