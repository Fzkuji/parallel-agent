#!/usr/bin/env python3
"""
SFT-LoRA baseline training and evaluation script.

This script trains LoRA adapters on the QA task and evaluates the trained models.
By default, it trains and evaluates BOTH formats (batch and sequential) and
outputs a comparison table.

Usage:
    # Evaluate both formats (default)
    python scripts/baseline_sft.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset squad \
        --eval-samples 100 \
        --min-questions 5 \
        --max-questions 10

    # Train and evaluate a specific format only
    python scripts/baseline_sft.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset squad \
        --train-format batch \
        --epochs 3 \
        --train-samples 1000
"""

# Disable tokenizers parallelism to avoid fork deadlock warnings in multiprocessing
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    # Training format
    parser.add_argument("--train-format", type=str, default="all",
                       choices=["batch", "sequential", "all"],
                       help="Training format: 'batch' (each question independent), "
                            "'sequential' (multi-turn conversation, train answer only), "
                            "or 'all' (evaluate both formats, default)")

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
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
    parser.add_argument("--compare-baseline", action="store_true", help="(Deprecated: baseline is always evaluated now)")

    # vLLM
    parser.add_argument("--enable-thinking", action="store_true")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if cached results exist")

    # Internal: DDP training worker mode (used by torchrun)
    parser.add_argument("--train-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--train-data-path", type=str, default=None, help=argparse.SUPPRESS)

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


def prepare_batch_training_data(
    train_groups: List[Dict],
    tokenizer,
    max_seq_length: int,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:
    """Prepare training data in batch format: each question is independent.

    Each training example is a single-turn conversation:
    - System: SYSTEM_PROMPT
    - User: Passage + Question
    - Assistant: <answer>...</answer>

    Args:
        train_groups: List of training context groups
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length
        max_samples: Maximum number of samples to return (random sampling if exceeded)
        seed: Random seed for sampling
    """
    import random

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

    # Random sampling if max_samples is specified and we have more examples
    if max_samples is not None and len(training_examples) > max_samples:
        rng = random.Random(seed)
        training_examples = rng.sample(training_examples, max_samples)

    return training_examples


def prepare_sequential_training_data(train_groups: List[Dict], tokenizer, max_seq_length: int) -> List[Dict]:
    """Prepare training data in sequential (multi-turn) format.

    Each training example is a multi-turn conversation where:
    - Turn 1: User provides context + Q1, Assistant answers A1
    - Turn 2: User provides Q2 (no context), Assistant answers A2
    - ...

    Only the assistant's answer portions are trained (labels masked for prompts).
    This mirrors the sequential evaluation strategy.
    """
    training_examples = []

    for group in train_groups:
        items = _context_to_items(group)
        if not items:
            continue

        # Get context from first item (all items in a group share the same context)
        context = items[0]["context"]

        # Build multi-turn conversation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for i, item in enumerate(items):
            question = item["question"]
            references = item.get("references", [])
            if not references:
                continue
            answer = references[0]

            if i == 0:
                # First turn: include context
                user_content = f"Passage:\n{context}\n\nQuestion: {question}"
            else:
                # Subsequent turns: only question (context already in history)
                user_content = f"Question: {question}"

            assistant_content = f"<answer>{answer}</answer>"

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})

        # Only add if we have at least one QA pair
        if len(messages) > 1:
            # Format using chat template
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                # Fallback format
                parts = []
                for msg in messages:
                    role = msg["role"].capitalize()
                    content = msg["content"]
                    parts.append(f"{role}: {content}")
                text = "\n\n".join(parts)

            # Check sequence length
            token_count = len(tokenizer.encode(text))
            if token_count <= max_seq_length:
                training_examples.append({
                    "text": text,
                    "qids": [item["qid"] for item in items if item.get("references")],
                    "num_turns": len([m for m in messages if m["role"] == "assistant"]),
                })
            else:
                # If too long, fall back to individual questions
                for item in items:
                    question = item["question"]
                    references = item.get("references", [])
                    if not references:
                        continue
                    answer = references[0]

                    user_content = f"Passage:\n{context}\n\nQuestion: {question}"
                    assistant_content = f"<answer>{answer}</answer>"

                    single_messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]

                    try:
                        single_text = tokenizer.apply_chat_template(
                            single_messages, tokenize=False, add_generation_prompt=False
                        )
                    except Exception:
                        single_text = f"System: {SYSTEM_PROMPT}\n\nUser: {user_content}\n\nAssistant: {assistant_content}"

                    training_examples.append({
                        "text": single_text,
                        "qids": [item["qid"]],
                        "num_turns": 1,
                    })

    return training_examples


def prepare_training_data(
    train_groups: List[Dict],
    tokenizer,
    max_seq_length: int,
    train_format: str = "batch",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:
    """Prepare training data based on the specified format.

    Args:
        train_groups: List of training context groups
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length
        train_format: "batch" or "sequential"
        max_samples: For batch format, maximum number of samples (random sampling)
        seed: Random seed for sampling

    Returns:
        List of training examples with "text" field
    """
    if train_format == "sequential":
        return prepare_sequential_training_data(train_groups, tokenizer, max_seq_length)
    else:
        return prepare_batch_training_data(
            train_groups, tokenizer, max_seq_length,
            max_samples=max_samples, seed=seed
        )


class DataCollatorForCausalLMWithMasking:
    """Mask everything except assistant responses for causal LM training.

    This collator pads dynamically and finds assistant spans by matching token
    patterns instead of decoding full strings, which is both faster and more
    robust. Only tokens between the assistant start marker and the following
    end marker contribute to the loss.
    """

    def __init__(
        self,
        tokenizer,
        assistant_token_pattern: str = "<|im_start|>assistant",
        end_token_pattern: str = "<|im_end|>",
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        # Pre-compute token ID patterns so we can scan input_ids directly
        self.assistant_start_ids = tokenizer.encode(
            assistant_token_pattern, add_special_tokens=False
        )
        self.end_ids = tokenizer.encode(end_token_pattern, add_special_tokens=False)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        self.newline_id = newline_ids[0] if len(newline_ids) == 1 else None

        if not self.assistant_start_ids or not self.end_ids:
            raise ValueError("Failed to build assistant/end token patterns for masking")

    def _find_assistant_spans(self, token_ids: List[int]) -> List[tuple]:
        """Return (start, end) token indices for each assistant response."""
        spans = []
        i = 0
        n = len(token_ids)
        start_len = len(self.assistant_start_ids)
        end_len = len(self.end_ids)

        while i <= n - start_len:
            if token_ids[i:i + start_len] == self.assistant_start_ids:
                start = i + start_len
                # Skip optional newline immediately after the assistant marker
                if self.newline_id is not None and start < n and token_ids[start] == self.newline_id:
                    start += 1

                end = start
                while end <= n - end_len and token_ids[end:end + end_len] != self.end_ids:
                    end += 1

                spans.append((start, end))
                i = end + end_len
            else:
                i += 1

        return spans

    def __call__(self, features: List[Dict]) -> Dict:
        import torch

        # Manual dynamic padding to avoid fast-tokenizer pad warning
        batch_size = len(features)
        to_list = lambda x: x.tolist() if hasattr(x, "tolist") else x
        input_lists = [to_list(f["input_ids"]) for f in features]

        max_len = max(len(ids) for ids in input_lists)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, ids in enumerate(input_lists):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1

        labels = torch.full_like(input_ids, -100)

        for row_idx, token_row in enumerate(input_ids.tolist()):
            for start, end in self._find_assistant_spans(token_row):
                labels[row_idx, start:end] = input_ids[row_idx, start:end]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _train_lora_worker(
    model_name: str,
    train_data_path: str,
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
    train_format: str = "batch",
) -> str:
    """Internal training function executed by each DDP process.

    This function is called by torchrun for DDP training.
    """
    import torch
    import torch.distributed as dist
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    # Check if running in distributed mode
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = local_rank == 0

    if is_main_process:
        logger.info(f"Loading tokenizer and model (world_size={world_size})...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For DDP training, don't use device_map="auto"
    # Let the Trainer handle device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
    )

    # Configure LoRA
    if is_main_process:
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
    if is_main_process:
        model.print_trainable_parameters()

    # Load pre-prepared training data
    if is_main_process:
        logger.info(f"Loading training data from {train_data_path}...")
    with open(train_data_path, 'r') as f:
        training_examples = json.load(f)
    if is_main_process:
        logger.info(f"Loaded {len(training_examples)} training examples")

    # Determine columns to remove
    columns_to_remove = ["text"]
    if "qid" in training_examples[0]:
        columns_to_remove.append("qid")
    if "qids" in training_examples[0]:
        columns_to_remove.append("qids")
    if "num_turns" in training_examples[0]:
        columns_to_remove.append("num_turns")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    dataset = Dataset.from_list(training_examples)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
    )

    # Data collator - always use custom masking to train only on assistant answer tokens
    if is_main_process:
        logger.info("Using custom data collator with assistant-only masking")
    data_collator = DataCollatorForCausalLMWithMasking(tokenizer)

    # Training arguments with DDP support
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
        # DDP settings
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    if is_main_process:
        logger.info(f"Starting DDP training with {world_size} GPU(s)...")
    trainer.train()

    # Save the final model (only on main process)
    final_checkpoint_dir = os.path.join(output_dir, "final")
    if is_main_process:
        model.save_pretrained(final_checkpoint_dir)
        tokenizer.save_pretrained(final_checkpoint_dir)
        logger.info(f"Saved final checkpoint to {final_checkpoint_dir}")

    return final_checkpoint_dir


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
    train_format: str = "batch",
    num_gpus: int = 1,
    gpu_ids: Optional[List[int]] = None,
    max_train_samples: Optional[int] = None,
) -> str:
    """Train LoRA adapter using SFT with DDP multi-GPU support.

    This function prepares the training data and launches torchrun for DDP training.

    Args:
        train_format: "batch" (each question independent) or "sequential" (multi-turn)
        num_gpus: Number of GPUs for DDP training
        gpu_ids: List of GPU IDs to use
        max_train_samples: For batch format, maximum number of question samples
    """
    import subprocess
    from transformers import AutoTokenizer

    logger.info("Preparing training data...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare training data
    # For batch format, max_train_samples controls random sampling of individual questions
    training_examples = prepare_training_data(
        train_groups, tokenizer, max_seq_length, train_format,
        max_samples=max_train_samples, seed=seed
    )
    logger.info(f"Prepared {len(training_examples)} training examples")

    # Save training data to temp file for DDP workers to load
    os.makedirs(output_dir, exist_ok=True)
    train_data_path = os.path.join(output_dir, "train_data.json")
    with open(train_data_path, 'w') as f:
        json.dump(training_examples, f)
    logger.info(f"Saved training data to {train_data_path}")

    # Prepare environment for torchrun
    env = os.environ.copy()
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        logger.info(f"Using GPUs: {gpu_ids}")

    # Build torchrun command
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(num_gpus),
        "--master_port", str(29500 + seed % 1000),  # Avoid port conflicts
        script_path,
        "--train-worker",
        "--model", model_name,
        "--train-data-path", train_data_path,
        "--output-dir", output_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--lr", str(learning_rate),
        "--warmup-ratio", str(warmup_ratio),
        "--max-seq-length", str(max_seq_length),
        "--lora-r", str(lora_r),
        "--lora-alpha", str(lora_alpha),
        "--lora-dropout", str(lora_dropout),
        "--lora-target-modules", ",".join(lora_target_modules),
        "--seed", str(seed),
        "--train-format", train_format,
    ]

    logger.info(f"Launching DDP training with {num_gpus} GPUs...")
    logger.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    # Clean up temp file
    if os.path.exists(train_data_path):
        os.unlink(train_data_path)

    final_checkpoint_dir = os.path.join(output_dir, "final")
    logger.info(f"Training complete, checkpoint at {final_checkpoint_dir}")

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
    eval_format: str = "batch",
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
    # Set left padding for decoder-only models
    tokenizer.padding_side = "left"

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

        context = items[0]["context"]

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

        if eval_format == "sequential":
            # Sequential evaluation: multi-turn conversation
            answer_records = {}
            total_latency = 0
            total_prompt_tokens_api = 0
            total_generated_tokens = 0
            deduplicated_prompt_tokens = 0

            # Build conversation history
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Track previous turn's tokens for incremental calculation
            prev_prompt_tokens = 0
            prev_generated_tokens = 0

            for i, item in enumerate(items):
                if i == 0:
                    user_content = f"Passage:\n{context}\n\nQuestion: {item['question']}"
                else:
                    user_content = f"Question: {item['question']}"

                messages.append({"role": "user", "content": user_content})

                try:
                    full_prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                except TypeError:
                    full_prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )

                inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                start_time = time.perf_counter()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                total_latency += time.perf_counter() - start_time

                # Get actual tokens from model output
                current_prompt_tokens = inputs["input_ids"].shape[1]
                generated = outputs[0][current_prompt_tokens:]
                generated_tokens = generated.shape[0]

                total_generated_tokens += generated_tokens
                total_prompt_tokens_api += current_prompt_tokens

                # Calculate deduplicated prompt tokens:
                # First turn: full prompt tokens
                # Subsequent turns: incremental tokens (current - previous - previous_generated)
                if i == 0:
                    deduplicated_prompt_tokens += current_prompt_tokens
                else:
                    incremental_tokens = current_prompt_tokens - prev_prompt_tokens - prev_generated_tokens
                    deduplicated_prompt_tokens += incremental_tokens

                # Save for next iteration
                prev_prompt_tokens = current_prompt_tokens
                prev_generated_tokens = generated_tokens

                raw_text = tokenizer.decode(generated, skip_special_tokens=True)
                final_answer, strict_valid = extract_answer(raw_text, dataset)
                answer_records[item["qid"]] = (final_answer, strict_valid)

                # Add assistant response to history
                messages.append({"role": "assistant", "content": raw_text})

            metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)

            shard_results["contexts"].append({
                "title": title,
                "metrics": metrics,
                "latency": total_latency,
                "prompt_tokens": deduplicated_prompt_tokens,
                "generated_tokens": total_generated_tokens,
                "prompt_tokens_api": total_prompt_tokens_api,
                "generated_tokens_api": total_generated_tokens,
                "num_questions": len(items),
            })
        else:
            # Batch inference (default)
            # Build prompts for all questions
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

            # Calculate context tokens for deduplication
            context_tokens = len(tokenizer.encode(context))

            # Decode and extract answers
            answer_records = {}
            total_prompt_tokens_api = 0
            total_generated_tokens = 0
            deduplicated_prompt_tokens = 0

            for i, item in enumerate(items):
                # Count non-padding tokens for API cost
                attention_mask = inputs["attention_mask"][i]
                prompt_tokens = attention_mask.sum().item()
                total_prompt_tokens_api += prompt_tokens

                generated = outputs[i][inputs["input_ids"][i].shape[0]:]
                total_generated_tokens += generated.shape[0]

                # Calculate deduplicated prompt tokens:
                # First question: full prompt tokens
                # Subsequent questions: prompt tokens - context tokens (context counted only once)
                if i == 0:
                    deduplicated_prompt_tokens += prompt_tokens
                else:
                    deduplicated_prompt_tokens += prompt_tokens - context_tokens

                raw_text = tokenizer.decode(generated, skip_special_tokens=True)
                final_answer, strict_valid = extract_answer(raw_text, dataset)
                answer_records[item["qid"]] = (final_answer, strict_valid)

            metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)

            shard_results["contexts"].append({
                "title": title,
                "metrics": metrics,
                "latency": latency,
                "prompt_tokens": deduplicated_prompt_tokens,  # Deduplicated: context once + questions
                "generated_tokens": total_generated_tokens,
                "prompt_tokens_api": total_prompt_tokens_api,  # API tokens (context repeated N times)
                "generated_tokens_api": total_generated_tokens,
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
    eval_format: str = "batch",
) -> Dict[str, Any]:
    """Run parallel evaluation across multiple GPUs.

    Args:
        eval_format: "batch" (each question independent) or "sequential" (multi-turn)
    """
    # Shard data across GPUs
    shards = [[] for _ in range(num_gpus)]
    for i, ctx in enumerate(eval_contexts):
        shards[i % num_gpus].append(ctx)

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    logger.info(f"Starting {num_gpus} eval workers on GPUs: {gpu_ids} (eval_format: {eval_format})")

    processes = []
    for rank in range(num_gpus):
        gpu_id = gpu_ids[rank]
        p = mp.Process(
            target=_eval_worker,
            args=(
                rank, num_gpus, gpu_id, model_name,
                shards[rank], output_dir, max_new_tokens, dataset,
                enable_thinking, lora_checkpoint_path, strategy_name,
                eval_format,
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

    # Original tokens (unique input only)
    total_prompt_tokens = sum(ctx["prompt_tokens"] for ctx in all_contexts)
    total_generated_tokens = sum(ctx["generated_tokens"] for ctx in all_contexts)

    # API tokens (actual tokens sent/received)
    total_prompt_tokens_api = sum(ctx.get("prompt_tokens_api", ctx["prompt_tokens"]) for ctx in all_contexts)
    total_generated_tokens_api = sum(ctx.get("generated_tokens_api", ctx["generated_tokens"]) for ctx in all_contexts)

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
            # Original tokens (unique input)
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "avg_prompt_tokens": total_prompt_tokens / total_contexts if total_contexts else 0,
            "avg_generated_tokens": total_generated_tokens / total_contexts if total_contexts else 0,
            # API tokens (actual cost)
            "total_prompt_tokens_api": total_prompt_tokens_api,
            "total_generated_tokens_api": total_generated_tokens_api,
            "avg_prompt_tokens_api": total_prompt_tokens_api / total_contexts if total_contexts else 0,
            "avg_generated_tokens_api": total_generated_tokens_api / total_contexts if total_contexts else 0,
            "num_contexts": total_contexts,
            "num_questions": total_questions,
        },
        "contexts": all_contexts,
    }


def _get_eval_cache_key(args, strategy_name: str, eval_format: str, lora_checkpoint_path: Optional[str] = None) -> str:
    """Generate a cache key for evaluation results."""
    key_parts = [
        args.model.replace("/", "_"),
        args.dataset,
        f"n{args.eval_samples}",
        f"q{args.min_questions}-{args.max_questions}",
        f"tok{args.max_new_tokens}",
        f"fmt_{eval_format}",
        strategy_name,
    ]
    # Include checkpoint path hash for SFT-LoRA
    if lora_checkpoint_path:
        import hashlib
        path_hash = hashlib.md5(lora_checkpoint_path.encode()).hexdigest()[:8]
        key_parts.append(f"ckpt_{path_hash}")
    return "_".join(key_parts)


def _load_eval_cache(cache_dir: Path, cache_key: str) -> Optional[Dict[str, Any]]:
    """Load cached evaluation results if available."""
    cache_file = cache_dir / f"eval_cache_{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded cached results from {cache_file}")
            return data
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cache: {e}")
    return None


def _save_eval_cache(cache_dir: Path, cache_key: str, results: Dict[str, Any], args, eval_format: str) -> None:
    """Save evaluation results to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"eval_cache_{cache_key}.json"

    data_with_meta = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "eval_samples": args.eval_samples,
            "min_questions": args.min_questions,
            "max_questions": args.max_questions,
            "max_new_tokens": args.max_new_tokens,
            "train_format": eval_format,
        },
        **results,
    }
    with open(cache_file, 'w') as f:
        json.dump(data_with_meta, f, indent=2)
    logger.info(f"Saved results to {cache_file}")


def _train_and_eval_single_format(
    args,
    train_format: str,
    eval_contexts: List[Dict],
    train_groups: Optional[List[Dict]],
    output_dir: Path,
    checkpoint_dir: Path,
    num_gpus: int,
    gpu_ids: List[int],
) -> Dict[str, Any]:
    """Train and evaluate a single format (batch or sequential).

    Returns dict with 'sft_lora' and optionally 'baseline' keys.
    """
    import torch

    results = {}

    logger.info(f"\n{'='*60}")
    logger.info(f"FORMAT: {train_format.upper()}")
    logger.info(f"{'='*60}")

    # Step 1: Always evaluate baseline (pretrained model) for comparison
    logger.info(f"\n[{train_format}] Evaluating Baseline (pretrained, no LoRA)")

    baseline_cache_key = _get_eval_cache_key(args, "baseline", train_format, None)
    baseline_results = None

    if not args.force:
        baseline_results = _load_eval_cache(output_dir, baseline_cache_key)

    if baseline_results is None:
        baseline_results = run_parallel_eval(
            model_name=args.model,
            eval_contexts=eval_contexts,
            output_dir=str(output_dir),
            max_new_tokens=args.max_new_tokens,
            dataset=args.dataset,
            enable_thinking=args.enable_thinking,
            lora_checkpoint_path=None,
            strategy_name=f"baseline_{train_format}",
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            eval_format=train_format,
        )
        if baseline_results:
            _save_eval_cache(output_dir, baseline_cache_key, baseline_results, args, train_format)

    if baseline_results:
        results["baseline"] = baseline_results.get("aggregate_metrics", baseline_results)

    # Step 2: Training (unless eval-only)
    lora_checkpoint_path = args.checkpoint_path
    if not args.eval_only and train_groups is not None:
        logger.info(f"\n[{train_format}] Training SFT-LoRA")

        lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',')]

        lora_checkpoint_path = train_lora(
            model_name=args.model,
            train_groups=train_groups,
            output_dir=str(checkpoint_dir / args.dataset / args.model.replace('/', '_') / train_format),
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
            train_format=train_format,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            # Don't limit batch samples - use all questions from loaded contexts
            # This ensures batch and sequential train on the same amount of data
            max_train_samples=None,
        )

    if not lora_checkpoint_path:
        logger.warning(f"[{train_format}] No LoRA checkpoint available, skipping SFT-LoRA evaluation")
        return results

    # Step 3: Evaluate SFT-LoRA
    logger.info(f"\n[{train_format}] Evaluating SFT-LoRA")

    sft_cache_key = _get_eval_cache_key(args, "sft_lora", train_format, lora_checkpoint_path)
    sft_results = None

    if not args.force:
        sft_results = _load_eval_cache(output_dir, sft_cache_key)

    if sft_results is None:
        sft_results = run_parallel_eval(
            model_name=args.model,
            eval_contexts=eval_contexts,
            output_dir=str(output_dir),
            max_new_tokens=args.max_new_tokens,
            dataset=args.dataset,
            enable_thinking=args.enable_thinking,
            lora_checkpoint_path=lora_checkpoint_path,
            strategy_name=f"sft_lora_{train_format}",
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            eval_format=train_format,
        )
        if sft_results:
            _save_eval_cache(output_dir, sft_cache_key, sft_results, args, train_format)

    if sft_results:
        results["sft_lora"] = sft_results.get("aggregate_metrics", sft_results)

    return results


def _print_comparison_table(all_format_results: Dict[str, Dict], formats: List[str]) -> None:
    """Print a comparison table for all formats."""
    logger.info("\n" + "=" * 160)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 160)

    # Get sample counts from first available result
    for fmt in formats:
        if fmt in all_format_results:
            for key in ["sft_lora", "baseline"]:
                if key in all_format_results[fmt]:
                    m = all_format_results[fmt][key]
                    logger.info(f"Contexts: {m.get('num_contexts', 0)}, Questions: {m.get('num_questions', 0)}")
                    break
            break

    # Combined results table
    header = f"{'Strategy':<20} | {'EM':>6} | {'F1':>6} | {'Lenient':>7} | {'Q/Ctx':>5} | {'PromptTok':>10} | {'GenTok':>8} | {'PromptTok_API':>13} | {'GenTok_API':>10} | {'DepTok':>8} | {'Latency':>8}"
    separator = "-" * len(header)
    logger.info("\n" + header)
    logger.info(separator)

    for fmt in formats:
        if fmt not in all_format_results:
            continue

        results = all_format_results[fmt]

        # Print baseline for this format
        if "baseline" in results:
            m = results["baseline"]
            strategy_name = f"baseline_{fmt}"
            avg_q_per_ctx = m.get('num_questions', 0) / max(m.get('num_contexts', 1), 1)
            logger.info(
                f"{strategy_name:<20} | "
                f"{m['strict_acc']:>6.3f} | "
                f"{m['f1']:>6.3f} | "
                f"{m.get('lenient_acc', 0):>7.3f} | "
                f"{avg_q_per_ctx:>5.1f} | "
                f"{m.get('avg_prompt_tokens', 0):>10.1f} | "
                f"{m.get('avg_generated_tokens', 0):>8.1f} | "
                f"{m.get('avg_prompt_tokens_api', 0):>13.1f} | "
                f"{m.get('avg_generated_tokens_api', 0):>10.1f} | "
                f"{m.get('avg_dependency_tokens', 0):>8.1f} | "
                f"{m['avg_latency']:>6.2f}s"
            )

        # Print SFT-LoRA for this format
        if "sft_lora" in results:
            m = results["sft_lora"]
            strategy_name = f"sft_lora_{fmt}"
            avg_q_per_ctx = m.get('num_questions', 0) / max(m.get('num_contexts', 1), 1)
            logger.info(
                f"{strategy_name:<20} | "
                f"{m['strict_acc']:>6.3f} | "
                f"{m['f1']:>6.3f} | "
                f"{m.get('lenient_acc', 0):>7.3f} | "
                f"{avg_q_per_ctx:>5.1f} | "
                f"{m.get('avg_prompt_tokens', 0):>10.1f} | "
                f"{m.get('avg_generated_tokens', 0):>8.1f} | "
                f"{m.get('avg_prompt_tokens_api', 0):>13.1f} | "
                f"{m.get('avg_generated_tokens_api', 0):>10.1f} | "
                f"{m.get('avg_dependency_tokens', 0):>8.1f} | "
                f"{m['avg_latency']:>6.2f}s"
            )

    logger.info("=" * 160)


def _run_train_worker():
    """Entry point for DDP training worker (called by torchrun)."""
    args = parse_args()

    lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',')]

    _train_lora_worker(
        model_name=args.model,
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
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
        train_format=args.train_format,
    )


def main():
    import torch

    args = parse_args()

    # Check if running as DDP training worker
    if args.train_worker:
        _run_train_worker()
        return

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

    # Load training data (unless eval-only)
    train_groups = None
    if not args.eval_only:
        # train_samples represents the number of QA pairs to train on
        # For fair comparison, both formats should train on the same number of QA pairs
        avg_questions_per_context = (args.min_questions + args.max_questions) / 2
        # Load enough contexts to have at least train_samples questions
        max_contexts = int(args.train_samples / avg_questions_per_context) + 10

        logger.info(f"Loading training data: target {args.train_samples} QA pairs, loading {max_contexts} contexts")
        train_groups = load_dataset(
            args.dataset,
            split="train",
            max_contexts=max_contexts,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            seed=args.seed,
        )
        # Count actual questions loaded
        total_questions = sum(
            len(_context_to_items(g)) for g in train_groups
        )
        logger.info(f"Loaded {len(train_groups)} training contexts ({total_questions} QA pairs)")

    # Determine which formats to evaluate
    if args.train_format == "all":
        formats_to_eval = ["batch", "sequential"]
    else:
        formats_to_eval = [args.train_format]

    logger.info(f"Formats to evaluate: {formats_to_eval}")

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
            "train_format": args.train_format,
        },
    }

    # Train and evaluate each format
    all_format_results = {}
    for train_format in formats_to_eval:
        format_results = _train_and_eval_single_format(
            args=args,
            train_format=train_format,
            eval_contexts=eval_contexts,
            train_groups=train_groups,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
        )
        all_format_results[train_format] = format_results
        all_results[train_format] = format_results

    # Print comparison table
    _print_comparison_table(all_format_results, formats_to_eval)

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
