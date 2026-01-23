#!/usr/bin/env python3
"""
SFT-LoRA training and evaluation script.

This script trains a LoRA adapter on the QA task and evaluates the trained model.

Usage:
    python scripts/train_and_eval_sft.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset squad \
        --epochs 3 \
        --train-samples 1000 \
        --eval-samples 100 \
        --num-gpus 8
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate SFT-LoRA model")

    # Model and dataset
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="squad",
                       choices=["squad", "hotpot", "quac", "drop", "triviaqa", "quality", "cmb"])
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--min-questions", type=int, default=3)
    parser.add_argument("--max-questions", type=int, default=5)

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-seq-length", type=int, default=2048)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

    # Inference
    parser.add_argument("--max-new-tokens", type=int, default=96)

    # Hardware
    parser.add_argument("--num-gpus", type=int, default=None, help="Override auto GPU detection")
    parser.add_argument("--min-free-mem-gb", type=float, default=10.0)

    # Paths
    parser.add_argument("--output-dir", type=str, default="outputs/sft_lora")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints/sft_lora")

    # Evaluation
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation (requires trained checkpoint)")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to trained LoRA checkpoint")
    parser.add_argument("--compare-baseline", action="store_true", help="Also evaluate baseline (no LoRA)")

    # vLLM
    parser.add_argument("--enable-thinking", action="store_true")

    # Other
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _get_available_gpus(min_free_memory_gb: float = 10.0) -> List[int]:
    """Get list of GPUs with sufficient free memory."""
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            available = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split(',')
                gpu_idx = int(parts[0].strip())
                free_mb = float(parts[1].strip())
                free_gb = free_mb / 1024
                if free_gb >= min_free_memory_gb:
                    available.append(gpu_idx)
            if available:
                return available
    except Exception:
        pass

    try:
        import torch
        count = torch.cuda.device_count()
        return list(range(count))
    except Exception:
        return list(range(8))


def _context_to_items(context_payload: dict) -> List[dict]:
    """Convert context format to items format."""
    if "items" in context_payload:
        return context_payload["items"]

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


def load_dataset(dataset: str, split: str, max_contexts: int, min_questions: int,
                 max_questions: int, seed: int) -> List[Dict]:
    """Load dataset based on name."""
    if dataset == "hotpot":
        from src.datasets.hotpot import load_hotpot_groups
        return load_hotpot_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "squad":
        from src.datasets.squad import load_squad_groups
        return load_squad_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "quac":
        from src.datasets.quac import load_quac_groups
        return load_quac_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "drop":
        from src.datasets.drop import load_drop_groups
        return load_drop_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "triviaqa":
        from src.datasets.triviaqa import load_triviaqa_groups
        return load_triviaqa_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "quality":
        from src.datasets.quality import load_quality_groups
        return load_quality_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    elif dataset == "cmb":
        from src.datasets.cmb import load_cmb_groups
        return load_cmb_groups(
            split=split, max_contexts=max_contexts,
            min_questions=min_questions, max_questions=max_questions, seed=seed
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# System prompt for answer extraction
SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the given passage.
You MUST wrap your answer in <answer></answer> tags. Be concise.

Example:
Question: What color is the sky?
<answer>blue</answer>"""


def prepare_training_data(train_groups: List[Dict], tokenizer, max_seq_length: int) -> List[Dict]:
    """Prepare training data in instruction format for SFT."""
    training_examples = []

    for group in train_groups:
        items = _context_to_items(group)

        for item in items:
            context = item["context"]
            question = item["question"]
            # Use first reference answer as target
            references = item.get("references", [])
            if not references:
                continue
            answer = references[0]

            # Format as chat messages
            user_content = f"Passage:\n{context}\n\nQuestion: {question}"
            assistant_content = f"<answer>{answer}</answer>"

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]

            # Format using chat template
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                # Fallback format
                text = f"System: {SYSTEM_PROMPT}\n\nUser: {user_content}\n\nAssistant: {assistant_content}"

            training_examples.append({
                "text": text,
                "qid": item["qid"],
            })

    return training_examples


def train_lora(
    model_name: str,
    train_groups: List[Dict],
    output_dir: str,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str],
    seed: int,
) -> str:
    """Train LoRA adapter using SFT."""
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Configure LoRA
    logger.info(f"Configuring LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare training data
    logger.info("Preparing training data...")
    training_examples = prepare_training_data(train_groups, tokenizer, max_seq_length)
    logger.info(f"Prepared {len(training_examples)} training examples")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    dataset = Dataset.from_list(training_examples)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "qid"],
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=seed,
        remove_unused_columns=False,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    final_checkpoint_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)
    logger.info(f"Saved final checkpoint to {final_checkpoint_dir}")

    return final_checkpoint_dir


def _eval_worker(
    rank: int,
    world_size: int,
    gpu_id: int,
    model_name: str,
    eval_contexts: List[Dict],
    output_dir: str,
    max_new_tokens: int,
    dataset: str,
    enable_thinking: bool,
    lora_checkpoint_path: Optional[str],
    strategy_name: str,
):
    """Worker process for evaluation on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_DISABLE_FRONTEND_MULTIPROCESSING"] = "1"
    os.environ["VLLM_NO_PROGRESS_BAR"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

    print(f"[Worker {rank}] GPU {gpu_id}: Starting, {len(eval_contexts)} contexts, strategy: {strategy_name}", flush=True)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.models import Question
    from src.evaluation import evaluate_predictions
    from src.inference import extract_answer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (with LoRA if specified)
    if lora_checkpoint_path:
        from peft import PeftModel
        print(f"[Worker {rank}] Loading model with LoRA from {lora_checkpoint_path}", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        print(f"[Worker {rank}] Loading base model (no LoRA)", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
        )

    model.eval()
    print(f"[Worker {rank}] Model loaded", flush=True)

    shard_results = {"contexts": []}

    for idx, context_payload in enumerate(eval_contexts, start=1):
        items = _context_to_items(context_payload)
        title = context_payload.get("title", f"context-{idx}")

        if idx % 10 == 0:
            print(f"[Worker {rank}] Processing {idx}/{len(eval_contexts)}", flush=True)

        # Build question lookup
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

        # Batch inference
        prompts = []
        for item in items:
            prompt = f"Passage:\n{item['context']}\n\nQuestion: {item['question']}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            try:
                full_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                full_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            prompts.append(full_prompt)

        # Tokenize
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        latency = time.perf_counter() - start_time

        # Decode and extract answers
        answer_records = {}
        total_prompt_tokens = inputs["input_ids"].numel()
        total_generated_tokens = 0

        for i, item in enumerate(items):
            input_len = inputs["input_ids"][i].shape[0]
            generated = outputs[i][input_len:]
            total_generated_tokens += generated.shape[0]

            raw_text = tokenizer.decode(generated, skip_special_tokens=True)
            final_answer, strict_valid = extract_answer(raw_text, dataset)
            answer_records[item["qid"]] = (final_answer, strict_valid)

        metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)

        shard_results["contexts"].append({
            "title": title,
            "metrics": metrics,
            "latency": latency,
            "prompt_tokens": total_prompt_tokens,
            "generated_tokens": total_generated_tokens,
            "num_questions": len(items),
        })

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    temp_file = os.path.join(output_dir, f"eval_{strategy_name}_shard_{rank}.json")
    with open(temp_file, 'w') as f:
        json.dump(shard_results, f)

    print(f"[Worker {rank}] Done, saved to {temp_file}", flush=True)


def run_parallel_eval(
    model_name: str,
    eval_contexts: List[Dict],
    output_dir: str,
    max_new_tokens: int,
    dataset: str,
    enable_thinking: bool,
    lora_checkpoint_path: Optional[str],
    strategy_name: str,
    num_gpus: int,
    gpu_ids: List[int],
) -> Dict[str, Any]:
    """Run parallel evaluation across multiple GPUs."""
    # Shard data across GPUs
    shards = [[] for _ in range(num_gpus)]
    for i, ctx in enumerate(eval_contexts):
        shards[i % num_gpus].append(ctx)

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    logger.info(f"Starting {num_gpus} eval workers on GPUs: {gpu_ids}")

    processes = []
    for rank in range(num_gpus):
        gpu_id = gpu_ids[rank]
        p = mp.Process(
            target=_eval_worker,
            args=(
                rank, num_gpus, gpu_id, model_name,
                shards[rank], output_dir, max_new_tokens, dataset,
                enable_thinking, lora_checkpoint_path, strategy_name,
            ),
        )
        p.start()
        processes.append(p)
        logger.info(f"Started eval worker {rank} on GPU {gpu_id} (PID: {p.pid})")

    for p in processes:
        p.join()

    logger.info("All eval workers finished, gathering results...")

    # Gather results
    all_contexts = []
    for rank in range(num_gpus):
        shard_file = os.path.join(output_dir, f"eval_{strategy_name}_shard_{rank}.json")
        if os.path.exists(shard_file):
            with open(shard_file, 'r') as f:
                shard_data = json.load(f)
                all_contexts.extend(shard_data["contexts"])
            os.unlink(shard_file)
        else:
            logger.warning(f"Missing shard file: {shard_file}")

    if not all_contexts:
        return {}

    # Aggregate metrics
    total_questions = sum(ctx["num_questions"] for ctx in all_contexts)
    total_contexts = len(all_contexts)
    total_latency = sum(ctx["latency"] for ctx in all_contexts)
    total_prompt_tokens = sum(ctx["prompt_tokens"] for ctx in all_contexts)
    total_generated_tokens = sum(ctx["generated_tokens"] for ctx in all_contexts)

    # Weighted averages for metrics
    total_em = sum(ctx["metrics"].get("strict_acc", 0) * ctx["num_questions"] for ctx in all_contexts)
    total_f1 = sum(ctx["metrics"].get("f1", 0) * ctx["num_questions"] for ctx in all_contexts)
    total_lenient = sum(ctx["metrics"].get("lenient_acc", 0) * ctx["num_questions"] for ctx in all_contexts)

    return {
        "aggregate_metrics": {
            "strict_acc": total_em / total_questions if total_questions > 0 else 0,
            "f1": total_f1 / total_questions if total_questions > 0 else 0,
            "lenient_acc": total_lenient / total_questions if total_questions > 0 else 0,
            "avg_latency": total_latency / total_contexts if total_contexts else 0,
            "total_latency": total_latency,
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "avg_prompt_tokens": total_prompt_tokens / total_contexts if total_contexts else 0,
            "avg_generated_tokens": total_generated_tokens / total_contexts if total_contexts else 0,
            "num_contexts": total_contexts,
            "num_questions": total_questions,
        },
        "contexts": all_contexts,
    }


def main():
    import torch

    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect GPUs
    available_gpus = _get_available_gpus(min_free_memory_gb=args.min_free_mem_gb)
    logger.info(f"Available GPUs with sufficient memory: {available_gpus}")

    if args.num_gpus is None:
        num_gpus = len(available_gpus) if available_gpus else 1
        gpu_ids = available_gpus[:num_gpus] if available_gpus else list(range(num_gpus))
    else:
        num_gpus = args.num_gpus
        gpu_ids = list(range(num_gpus))

    logger.info(f"Using {num_gpus} GPU(s): {gpu_ids}")

    # Load evaluation data
    logger.info(f"Loading evaluation data: {args.eval_samples} samples from {args.dataset}")
    eval_contexts = load_dataset(
        args.dataset,
        split="validation",
        max_contexts=args.eval_samples,
        min_questions=args.min_questions,
        max_questions=args.max_questions,
        seed=args.seed + 1000,
    )
    logger.info(f"Loaded {len(eval_contexts)} evaluation contexts")

    # Results storage
    all_results = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
        },
    }

    # Step 1: Evaluate baseline if requested
    if args.compare_baseline:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Evaluating Baseline (no LoRA)")
        logger.info("=" * 60)

        baseline_results = run_parallel_eval(
            model_name=args.model,
            eval_contexts=eval_contexts,
            output_dir=str(output_dir),
            max_new_tokens=args.max_new_tokens,
            dataset=args.dataset,
            enable_thinking=args.enable_thinking,
            lora_checkpoint_path=None,
            strategy_name="baseline",
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
        )

        if baseline_results:
            all_results["baseline"] = baseline_results["aggregate_metrics"]
            m = baseline_results['aggregate_metrics']
            logger.info(f"\nBaseline Results:")
            logger.info(f"  EM:           {m['strict_acc']:.4f}")
            logger.info(f"  F1:           {m['f1']:.4f}")
            logger.info(f"  Lenient:      {m.get('lenient_acc', 0):.4f}")
            logger.info(f"  Prompt Tok:   {m['total_prompt_tokens']:,} (avg: {m.get('avg_prompt_tokens', 0):.1f})")
            logger.info(f"  Gen Tok:      {m['total_generated_tokens']:,} (avg: {m.get('avg_generated_tokens', 0):.1f})")
            logger.info(f"  Latency:      {m.get('total_latency', 0):.2f}s (avg: {m['avg_latency']:.2f}s)")

    # Step 2: Training (unless eval-only)
    lora_checkpoint_path = args.checkpoint_path
    if not args.eval_only:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Training SFT-LoRA")
        logger.info("=" * 60)

        # Load training data
        logger.info(f"Loading training data: {args.train_samples} samples")
        train_groups = load_dataset(
            args.dataset,
            split="train",
            max_contexts=args.train_samples,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            seed=args.seed,
        )
        logger.info(f"Loaded {len(train_groups)} training contexts")

        # Parse target modules
        lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',')]

        # Train
        lora_checkpoint_path = train_lora(
            model_name=args.model,
            train_groups=train_groups,
            output_dir=str(checkpoint_dir / args.dataset / args.model.replace('/', '_')),
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
            max_seq_length=args.max_seq_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=lora_target_modules,
            seed=args.seed,
        )

    if not lora_checkpoint_path:
        logger.error("No LoRA checkpoint available. Either train a model or provide --checkpoint-path")
        sys.exit(1)

    # Step 3: Evaluate SFT-LoRA
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Evaluating SFT-LoRA")
    logger.info("=" * 60)

    sft_results = run_parallel_eval(
        model_name=args.model,
        eval_contexts=eval_contexts,
        output_dir=str(output_dir),
        max_new_tokens=args.max_new_tokens,
        dataset=args.dataset,
        enable_thinking=args.enable_thinking,
        lora_checkpoint_path=lora_checkpoint_path,
        strategy_name="sft_lora",
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
    )

    if sft_results:
        all_results["sft_lora"] = sft_results["aggregate_metrics"]
        m = sft_results['aggregate_metrics']
        logger.info(f"\nSFT-LoRA Results:")
        logger.info(f"  EM:           {m['strict_acc']:.4f}")
        logger.info(f"  F1:           {m['f1']:.4f}")
        logger.info(f"  Lenient:      {m.get('lenient_acc', 0):.4f}")
        logger.info(f"  Prompt Tok:   {m['total_prompt_tokens']:,} (avg: {m.get('avg_prompt_tokens', 0):.1f})")
        logger.info(f"  Gen Tok:      {m['total_generated_tokens']:,} (avg: {m.get('avg_generated_tokens', 0):.1f})")
        logger.info(f"  Latency:      {m.get('total_latency', 0):.2f}s (avg: {m['avg_latency']:.2f}s)")

    # Final summary
    logger.info("\n" + "=" * 90)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 90)

    # Get sample counts
    for key in ["baseline", "sft_lora"]:
        if key in all_results:
            m = all_results[key]
            logger.info(f"Contexts: {m.get('num_contexts', 0)}, Questions: {m.get('num_questions', 0)}")
            break

    header = f"{'Strategy':<15} | {'EM':>6} | {'F1':>6} | {'Lenient':>7} | {'PromptTok':>10} | {'GenTok':>8} | {'Latency':>10}"
    separator = "-" * len(header)
    logger.info(header)
    logger.info(separator)

    if "baseline" in all_results:
        m = all_results["baseline"]
        logger.info(
            f"{'baseline':<15} | "
            f"{m['strict_acc']:>6.3f} | "
            f"{m['f1']:>6.3f} | "
            f"{m.get('lenient_acc', 0):>7.3f} | "
            f"{m.get('avg_prompt_tokens', 0):>10.1f} | "
            f"{m.get('avg_generated_tokens', 0):>8.1f} | "
            f"{m['avg_latency']:>8.2f}s"
        )

    if "sft_lora" in all_results:
        m = all_results["sft_lora"]
        logger.info(
            f"{'sft_lora':<15} | "
            f"{m['strict_acc']:>6.3f} | "
            f"{m['f1']:>6.3f} | "
            f"{m.get('lenient_acc', 0):>7.3f} | "
            f"{m.get('avg_prompt_tokens', 0):>10.1f} | "
            f"{m.get('avg_generated_tokens', 0):>8.1f} | "
            f"{m['avg_latency']:>8.2f}s"
        )

        if "baseline" in all_results:
            baseline_m = all_results["baseline"]
            logger.info(f"\nImprovement over baseline:")
            logger.info(f"  EM:      {m['strict_acc'] - baseline_m['strict_acc']:+.4f}")
            logger.info(f"  F1:      {m['f1'] - baseline_m['f1']:+.4f}")
            logger.info(f"  Lenient: {m.get('lenient_acc', 0) - baseline_m.get('lenient_acc', 0):+.4f}")

    # Save results
    results_path = output_dir / f"sft_lora_results_{args.dataset}.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved results to {results_path}")

    logger.info("\n" + "=" * 60)
    logger.info("DONE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
