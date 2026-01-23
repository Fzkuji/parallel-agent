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

    # Training format
    parser.add_argument("--train-format", type=str, default="batch",
                       choices=["batch", "sequential"],
                       help="Training format: 'batch' (each question independent) or "
                            "'sequential' (multi-turn conversation, train answer only)")

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


def prepare_batch_training_data(train_groups: List[Dict], tokenizer, max_seq_length: int) -> List[Dict]:
    """Prepare training data in batch format: each question is independent.

    Each training example is a single-turn conversation:
    - System: SYSTEM_PROMPT
    - User: Passage + Question
    - Assistant: <answer>...</answer>
    """
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
    train_format: str = "batch"
) -> List[Dict]:
    """Prepare training data based on the specified format.

    Args:
        train_groups: List of training context groups
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length
        train_format: "batch" or "sequential"

    Returns:
        List of training examples with "text" field
    """
    if train_format == "sequential":
        return prepare_sequential_training_data(train_groups, tokenizer, max_seq_length)
    else:
        return prepare_batch_training_data(train_groups, tokenizer, max_seq_length)


class DataCollatorForCausalLMWithMasking:
    """Data collator that masks non-answer tokens in the labels.

    For SFT training, we want to only compute loss on the assistant's response tokens,
    not on the system/user prompt tokens. This collator:
    1. Creates labels as a copy of input_ids
    2. Masks (sets to -100) all tokens before each assistant response
    """

    def __init__(self, tokenizer, assistant_token_pattern: str = "<|im_start|>assistant"):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        # Find the token pattern that indicates assistant turn start
        self.assistant_token_pattern = assistant_token_pattern

    def __call__(self, features: List[Dict]) -> Dict:
        import torch

        # Stack input_ids and attention_mask
        input_ids = torch.stack([torch.tensor(f["input_ids"]) for f in features])
        attention_mask = torch.stack([torch.tensor(f["attention_mask"]) for f in features])

        # Create labels - mask everything except assistant responses
        labels = input_ids.clone()

        # Find assistant start tokens and mask everything before them
        for i in range(labels.shape[0]):
            # Decode to find assistant positions (more robust than token matching)
            text = self.tokenizer.decode(input_ids[i])

            # Find all assistant turn positions
            # For Qwen format: <|im_start|>assistant\n....<|im_end|>
            # We want to keep only the content after <|im_start|>assistant\n
            assistant_marker = "<|im_start|>assistant"
            end_marker = "<|im_end|>"

            # Start by masking everything
            labels[i] = torch.full_like(labels[i], -100)

            # Find positions to unmask (assistant content only)
            current_pos = 0
            text_so_far = ""
            char_to_token = []

            # Build character to token mapping
            for token_idx in range(len(input_ids[i])):
                token_text = self.tokenizer.decode([input_ids[i][token_idx]])
                for _ in token_text:
                    char_to_token.append(token_idx)
                text_so_far += token_text

            # Find assistant responses and unmask them
            search_pos = 0
            while True:
                assistant_start = text_so_far.find(assistant_marker, search_pos)
                if assistant_start == -1:
                    break

                # Find the newline after assistant marker
                content_start = text_so_far.find("\n", assistant_start)
                if content_start == -1:
                    break
                content_start += 1  # Skip the newline

                # Find the end marker
                content_end = text_so_far.find(end_marker, content_start)
                if content_end == -1:
                    content_end = len(text_so_far)

                # Map character positions to token positions
                if content_start < len(char_to_token) and content_end <= len(char_to_token):
                    token_start = char_to_token[content_start] if content_start < len(char_to_token) else len(input_ids[i])
                    token_end = char_to_token[content_end - 1] + 1 if content_end <= len(char_to_token) and content_end > 0 else len(input_ids[i])

                    # Unmask assistant content tokens
                    labels[i, token_start:token_end] = input_ids[i, token_start:token_end]

                search_pos = content_end + 1

            # Also mask padding tokens
            labels[i][input_ids[i] == self.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


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
) -> str:
    """Train LoRA adapter using SFT.

    Args:
        train_format: "batch" (each question independent) or "sequential" (multi-turn)
    """
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
    logger.info(f"Preparing training data (format: {train_format})...")
    training_examples = prepare_training_data(train_groups, tokenizer, max_seq_length, train_format)
    logger.info(f"Prepared {len(training_examples)} training examples")

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
            padding="max_length",
        )

    dataset = Dataset.from_list(training_examples)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
    )

    # Data collator - use custom masking for sequential format
    if train_format == "sequential":
        logger.info("Using custom data collator with assistant-only masking")
        data_collator = DataCollatorForCausalLMWithMasking(tokenizer)
    else:
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

            # Build conversation history
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Calculate deduplicated prompt tokens (context once + all questions with template overhead)
            first_user = f"Passage:\n{context}\n\nQuestion: {items[0]['question']}"
            first_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": first_user},
            ]
            try:
                first_full = tokenizer.apply_chat_template(
                    first_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                first_full = tokenizer.apply_chat_template(
                    first_messages, tokenize=False, add_generation_prompt=True,
                )
            first_turn_tokens = len(tokenizer.encode(first_full))

            # For subsequent turns, measure incremental token cost with template overhead
            additional_turn_tokens = 0
            for item in items[1:]:
                question_content = f"Question: {item['question']}"
                # Measure incremental cost by comparing two-turn vs one-turn template
                two_turn_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "dummy"},
                    {"role": "assistant", "content": "dummy"},
                    {"role": "user", "content": question_content},
                ]
                one_turn_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "dummy"},
                    {"role": "assistant", "content": "dummy"},
                ]
                try:
                    two_turn_full = tokenizer.apply_chat_template(
                        two_turn_messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                    one_turn_full = tokenizer.apply_chat_template(
                        one_turn_messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                except TypeError:
                    two_turn_full = tokenizer.apply_chat_template(
                        two_turn_messages, tokenize=False, add_generation_prompt=True,
                    )
                    one_turn_full = tokenizer.apply_chat_template(
                        one_turn_messages, tokenize=False, add_generation_prompt=True,
                    )
                additional_turn_tokens += len(tokenizer.encode(two_turn_full)) - len(tokenizer.encode(one_turn_full))

            total_prompt_tokens = first_turn_tokens + additional_turn_tokens

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

                input_len = inputs["input_ids"].shape[1]
                generated = outputs[0][input_len:]
                total_generated_tokens += generated.shape[0]
                total_prompt_tokens_api += input_len

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
                "prompt_tokens": total_prompt_tokens,
                "generated_tokens": total_generated_tokens,
                "prompt_tokens_api": total_prompt_tokens_api,
                "generated_tokens_api": total_generated_tokens,
                "num_questions": len(items),
            })
        else:
            # Batch inference (default)
            # Calculate deduplicated prompt tokens (context once + all questions with template overhead)
            first_prompt = f"Passage:\n{context}\n\nQuestion: {items[0]['question']}"
            first_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": first_prompt},
            ]
            try:
                first_full = tokenizer.apply_chat_template(
                    first_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                first_full = tokenizer.apply_chat_template(
                    first_messages, tokenize=False, add_generation_prompt=True,
                )
            first_question_tokens = len(tokenizer.encode(first_full))

            # For additional questions, measure incremental cost with template overhead
            additional_question_tokens = 0
            for item in items[1:]:
                question_content = f"Question: {item['question']}"
                two_turn_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "dummy"},
                    {"role": "assistant", "content": "dummy"},
                    {"role": "user", "content": question_content},
                ]
                one_turn_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "dummy"},
                    {"role": "assistant", "content": "dummy"},
                ]
                try:
                    two_turn_full = tokenizer.apply_chat_template(
                        two_turn_messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                    one_turn_full = tokenizer.apply_chat_template(
                        one_turn_messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                except TypeError:
                    two_turn_full = tokenizer.apply_chat_template(
                        two_turn_messages, tokenize=False, add_generation_prompt=True,
                    )
                    one_turn_full = tokenizer.apply_chat_template(
                        one_turn_messages, tokenize=False, add_generation_prompt=True,
                    )
                additional_question_tokens += len(tokenizer.encode(two_turn_full)) - len(tokenizer.encode(one_turn_full))

            deduplicated_prompt_tokens = first_question_tokens + additional_question_tokens

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
            total_prompt_tokens_api = 0
            total_generated_tokens = 0

            for i, item in enumerate(items):
                # Count non-padding tokens for API cost
                attention_mask = inputs["attention_mask"][i]
                input_len = attention_mask.sum().item()
                total_prompt_tokens_api += input_len

                generated = outputs[i][inputs["input_ids"][i].shape[0]:]
                total_generated_tokens += generated.shape[0]

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
            "train_format": args.train_format,
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
            eval_format=args.train_format,  # Use same format as training
        )

        if baseline_results:
            all_results["baseline"] = baseline_results["aggregate_metrics"]
            m = baseline_results['aggregate_metrics']
            logger.info(f"\nBaseline Results:")
            logger.info(f"  EM:             {m['strict_acc']:.4f}")
            logger.info(f"  F1:             {m['f1']:.4f}")
            logger.info(f"  Lenient:        {m.get('lenient_acc', 0):.4f}")
            logger.info(f"  Prompt Tok:     {m['total_prompt_tokens']:,} (avg: {m.get('avg_prompt_tokens', 0):.1f})")
            logger.info(f"  Gen Tok:        {m['total_generated_tokens']:,} (avg: {m.get('avg_generated_tokens', 0):.1f})")
            logger.info(f"  PromptTok_API:  {m.get('total_prompt_tokens_api', 0):,} (avg: {m.get('avg_prompt_tokens_api', 0):.1f})")
            logger.info(f"  GenTok_API:     {m.get('total_generated_tokens_api', 0):,} (avg: {m.get('avg_generated_tokens_api', 0):.1f})")
            logger.info(f"  Latency:        {m.get('total_latency', 0):.2f}s (avg: {m['avg_latency']:.2f}s)")

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
        logger.info(f"Training format: {args.train_format}")
        lora_checkpoint_path = train_lora(
            model_name=args.model,
            train_groups=train_groups,
            output_dir=str(checkpoint_dir / args.dataset / args.model.replace('/', '_') / args.train_format),
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
        eval_format=args.train_format,  # Use same format as training
    )

    if sft_results:
        all_results["sft_lora"] = sft_results["aggregate_metrics"]
        m = sft_results['aggregate_metrics']
        logger.info(f"\nSFT-LoRA Results:")
        logger.info(f"  EM:             {m['strict_acc']:.4f}")
        logger.info(f"  F1:             {m['f1']:.4f}")
        logger.info(f"  Lenient:        {m.get('lenient_acc', 0):.4f}")
        logger.info(f"  Prompt Tok:     {m['total_prompt_tokens']:,} (avg: {m.get('avg_prompt_tokens', 0):.1f})")
        logger.info(f"  Gen Tok:        {m['total_generated_tokens']:,} (avg: {m.get('avg_generated_tokens', 0):.1f})")
        logger.info(f"  PromptTok_API:  {m.get('total_prompt_tokens_api', 0):,} (avg: {m.get('avg_prompt_tokens_api', 0):.1f})")
        logger.info(f"  GenTok_API:     {m.get('total_generated_tokens_api', 0):,} (avg: {m.get('avg_generated_tokens_api', 0):.1f})")
        logger.info(f"  Latency:        {m.get('total_latency', 0):.2f}s (avg: {m['avg_latency']:.2f}s)")

    # Final summary
    logger.info("\n" + "=" * 110)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 110)

    # Get sample counts
    for key in ["baseline", "sft_lora"]:
        if key in all_results:
            m = all_results[key]
            logger.info(f"Contexts: {m.get('num_contexts', 0)}, Questions: {m.get('num_questions', 0)}")
            break

    # Combined results table
    logger.info("\n=== Results Summary ===")
    header = f"{'Strategy':<15} | {'EM':>6} | {'F1':>6} | {'Lenient':>7} | {'PromptTok':>10} | {'GenTok':>8} | {'PromptTok_API':>13} | {'GenTok_API':>10} | {'Latency':>8}"
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
            f"{m.get('avg_prompt_tokens_api', 0):>13.1f} | "
            f"{m.get('avg_generated_tokens_api', 0):>10.1f} | "
            f"{m['avg_latency']:>6.2f}s"
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
            f"{m.get('avg_prompt_tokens_api', 0):>13.1f} | "
            f"{m.get('avg_generated_tokens_api', 0):>10.1f} | "
            f"{m['avg_latency']:>6.2f}s"
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
