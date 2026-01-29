#!/usr/bin/env python3
"""
Question grouping impact experiment with multi-GPU support.

Tests how grouping size affects strategy performance on the same set of questions.

Setup:
- Load contexts with at least N questions each (N = max group size)
- Test with different grouping sizes: 1, 4, 8, 12, 16 questions per group

For each grouping size:
- all_in_one: Processes N questions per prompt
- sequential: Processes N questions sequentially in one conversation
- batch: Processes N questions in parallel batch

All configurations test the exact same questions.
Multi-GPU: Each GPU loads one model and processes a subset of groups in parallel.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval_utils import (
    get_available_gpus,
    context_to_items,
    aggregate_context_results,
    save_evaluation_results,
    print_results_summary,
    get_cache_key,
    load_cached_results,
    save_cached_results,
    SYSTEM_PROMPT,
    ALL_IN_ONE_SYSTEM_PROMPT,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Question grouping impact experiment")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="squad",
                       choices=["squad", "hotpot", "quac", "quality", "drop", "cmb", "triviaqa", "coqa"],
                       help="Dataset to use")
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
    parser.add_argument("--lora-checkpoint", type=str, default=None,
                       help="Path to LoRA checkpoint for SFT evaluation (applies to batch/sequential strategies)")
    parser.add_argument("--sft-format", type=str, default=None, choices=["batch", "sequential"],
                       help="SFT training format (batch or sequential), used with --lora-checkpoint")

    return parser.parse_args()


def load_all_questions(dataset="squad", seed=42, min_questions=16, max_contexts=100):
    """Load questions from contexts with at least min_questions questions each."""
    from src.eval_utils import load_dataset_groups

    # Determine appropriate split for each dataset
    split_map = {
        "squad": "validation",
        "hotpot": "validation",
        "quac": "validation",
        "quality": "dev",
        "drop": "validation",
        "cmb": "train",
        "triviaqa": "validation",
        "coqa": "validation",
    }
    split = split_map.get(dataset, "validation")

    contexts = load_dataset_groups(
        dataset=dataset,
        split=split,
        max_contexts=max_contexts,
        min_questions=min_questions,
        max_questions=1000,  # No upper limit
        seed=seed,
        fixed_question_count=min_questions if dataset in ["squad", "cmb"] else None
    )

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
    from src.eval_utils import load_dataset_groups

    # Determine appropriate train split for each dataset
    train_split_map = {
        "squad": "train",
        "hotpot": "train",
        "quac": "train",
        "quality": "train",
        "drop": "train",
        "cmb": "train",
        "triviaqa": "train",
        "coqa": "train",
    }
    split = train_split_map.get(dataset, "train")

    contexts = load_dataset_groups(
        dataset=dataset,
        split=split,
        max_contexts=max_samples,
        min_questions=1,
        max_questions=5,
        seed=seed,
    )

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


def retrieve_similar_examples(query_embedding, memory_embeddings, memory_bank, top_k=3, exclude_context=None):
    """Retrieve top-k most similar examples from memory bank, excluding examples from the same context.

    Args:
        exclude_context: Context text to exclude (prevent data leakage)
    """
    import numpy as np
    # Normalize for cosine similarity
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    memory_norms = memory_embeddings / (np.linalg.norm(memory_embeddings, axis=1, keepdims=True) + 1e-9)
    similarities = np.dot(memory_norms, query_norm)

    # Sort all indices by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]

    # Filter out examples with same context
    selected = []
    for idx in sorted_indices:
        if exclude_context and memory_bank[idx]["context"] == exclude_context:
            continue  # Skip examples from same context
        selected.append(memory_bank[idx])
        if len(selected) >= top_k:
            break

    return selected


def group_questions(all_questions, group_size):
    """Group questions into chunks of group_size."""
    groups = []
    for i in range(0, len(all_questions), group_size):
        groups.append(all_questions[i:i+group_size])
    return groups


def gpu_worker(
    worker_id: int,
    physical_gpu_id: int,
    groups: List[List[dict]],
    group_indices: List[int],
    args_dict: dict,
    strategies_to_run: List[str],
    memory_bank: Optional[List[dict]],
    memory_embeddings_list: Optional[List],  # numpy array as list for pickling
    result_queue: mp.Queue,
):
    """
    Worker function that runs on a single GPU.
    Loads the model and processes assigned groups.
    """
    import os
    import time
    import re
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # Re-import for multiprocessing (spawn method requires this)
    from src.eval_utils import SYSTEM_PROMPT, ALL_IN_ONE_SYSTEM_PROMPT

    # Set GPU - use the physical GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    device = "cuda:0"

    # Reconstruct memory embeddings from list
    memory_embeddings = np.array(memory_embeddings_list) if memory_embeddings_list is not None else None

    # Load embedding model for memory strategy if needed
    embedding_model = None
    if "memory" in strategies_to_run and memory_bank:
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(args_dict["embedding_model"])
        except ImportError:
            pass

    # Load LLM model
    print(f"[GPU {physical_gpu_id}] Loading model {args_dict['model']}...")
    tokenizer = AutoTokenizer.from_pretrained(args_dict["model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args_dict["model"],
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    # Load LoRA checkpoint if provided
    if args_dict.get("lora_checkpoint"):
        try:
            from peft import PeftModel
            print(f"[GPU {physical_gpu_id}] Loading LoRA checkpoint: {args_dict['lora_checkpoint']}")
            model = PeftModel.from_pretrained(model, args_dict["lora_checkpoint"])
            model = model.merge_and_unload()  # Merge LoRA weights into base model
            print(f"[GPU {physical_gpu_id}] LoRA weights loaded and merged")
        except ImportError:
            print(f"[GPU {physical_gpu_id}] Warning: peft not installed, skipping LoRA loading")
        except Exception as e:
            print(f"[GPU {physical_gpu_id}] Error loading LoRA: {e}")

    model.eval()
    print(f"[GPU {physical_gpu_id}] Model loaded, processing {len(groups)} groups")

    # Initialize results for strategies with detailed token tracking
    results = {}
    for strategy in strategies_to_run:
        # Cross-Batch now supports random init (no checkpoint required)
        results[strategy] = {
            "predictions": {},
            "latency": 0,
            "prompt_tokens": 0,           # Deduplicated (context counted once per group)
            "generated_tokens": 0,
            "prompt_tokens_api": 0,       # Actual API tokens sent
            "generated_tokens_api": 0,    # Actual tokens generated
            "contexts": [],               # Per-group detailed results
        }

    # Initialize Cross-Batch generator if needed
    cross_batch_generator = None
    if "cross_batch" in results:
        try:
            from src.cross_batch import CrossBatchGenerator, CrossBatchAttention

            hidden_size = model.config.hidden_size

            # Load checkpoint first to get config (if available)
            checkpoint = None
            config = {}
            checkpoint_path = args_dict.get("cross_batch_checkpoint")

            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"[GPU {physical_gpu_id}] Loading Cross-Batch checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                config = checkpoint.get("config", {})
                print(f"[GPU {physical_gpu_id}] Checkpoint config: use_gate={config.get('use_gate', False)}")

            # Create CSA module with correct config
            use_gate = config.get("use_gate", False)
            cross_batch_module = CrossBatchAttention(hidden_size=hidden_size, use_gate=use_gate)

            # Load checkpoint weights
            if checkpoint is not None:
                cross_batch_module.load_state_dict(checkpoint["cross_batch_module"])
                print(f"[GPU {physical_gpu_id}] Loaded trained CSA weights")
            else:
                print(f"[GPU {physical_gpu_id}] Using random init CSA (no checkpoint)")

            cross_batch_module.to(device)

            # ============ DEBUG: Print CSA module weights ============
            print(f"\n[GPU {physical_gpu_id}] ===== CSA Module Debug Info =====")
            print(f"[GPU {physical_gpu_id}] use_gate: {cross_batch_module.use_gate}")

            # Check out_proj weights (should be 0 for LoRA-style init)
            out_proj_norm = cross_batch_module.out_proj.weight.norm().item()
            out_proj_max = cross_batch_module.out_proj.weight.abs().max().item()
            print(f"[GPU {physical_gpu_id}] out_proj.weight: norm={out_proj_norm:.6f}, max={out_proj_max:.6f}")

            # Check Q/K/V weights
            q_norm = cross_batch_module.q_proj.weight.norm().item()
            k_norm = cross_batch_module.k_proj.weight.norm().item()
            v_norm = cross_batch_module.v_proj.weight.norm().item()
            print(f"[GPU {physical_gpu_id}] Q/K/V norms: Q={q_norm:.4f}, K={k_norm:.4f}, V={v_norm:.4f}")

            # Check gate weights if use_gate=True
            if cross_batch_module.use_gate:
                gate_final = cross_batch_module.gate_net[-2]  # Linear before Sigmoid
                gate_bias = gate_final.bias.item() if gate_final.bias.numel() == 1 else gate_final.bias[0].item()
                gate_weight_norm = gate_final.weight.norm().item()
                print(f"[GPU {physical_gpu_id}] gate: bias[0]={gate_bias:.4f}, weight_norm={gate_weight_norm:.6f}")

            # Expected values for LoRA-style init:
            # - out_proj should be ~0 (norm=0, max=0)
            # - Q/K/V should be ~normal (norm around hidden_size * 0.02)
            expected_qkv_norm = (hidden_size ** 0.5) * 0.02
            print(f"[GPU {physical_gpu_id}] Expected Q/K/V norm (if init properly): ~{expected_qkv_norm:.4f}")

            if out_proj_norm < 0.001:
                print(f"[GPU {physical_gpu_id}] ✓ out_proj is ~0 (LoRA-style init correct)")
            else:
                print(f"[GPU {physical_gpu_id}] ✗ out_proj is NOT 0! CSA will affect output!")
            print(f"[GPU {physical_gpu_id}] =====================================\n")
            # ============ END DEBUG ============

            # Load LoRA weights if present in Cross-Batch checkpoint
            if checkpoint and "lora" in checkpoint and config.get("use_lora"):
                try:
                    from peft import PeftModel, LoraConfig, get_peft_model
                    print(f"[GPU {physical_gpu_id}] Loading LoRA from Cross-Batch checkpoint ({len(checkpoint['lora'])} tensors)")

                    # Get LoRA config from checkpoint
                    target_modules = config.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj")
                    if isinstance(target_modules, str):
                        target_modules = target_modules.split(",")

                    lora_config = LoraConfig(
                        r=config.get("lora_r", 16),
                        lora_alpha=config.get("lora_alpha", 32),
                        target_modules=target_modules,
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM",
                    )

                    # Apply LoRA structure to model
                    model = get_peft_model(model, lora_config)

                    # Load LoRA weights from checkpoint
                    # The checkpoint contains full state_dict with 'base_model.model...' prefixes
                    incompatible = model.load_state_dict(checkpoint["lora"], strict=False)
                    print(f"[GPU {physical_gpu_id}] Loaded LoRA, missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}")

                    # Merge LoRA into base model for inference
                    model = model.merge_and_unload()
                    model.eval()
                    print(f"[GPU {physical_gpu_id}] LoRA merged into base model")
                except Exception as e:
                    print(f"[GPU {physical_gpu_id}] ERROR loading LoRA: {e}")
                    import traceback
                    traceback.print_exc()

            mix_method = config.get("module_type", "attention")
            mix_layer = config.get("mix_layer", -1)

            cross_batch_generator = CrossBatchGenerator(
                model=model,
                tokenizer=tokenizer,
                cross_batch_module=cross_batch_module,
                mix_method=mix_method,
                mix_layer=mix_layer,
                device=device,
            )
            print(f"[GPU {physical_gpu_id}] Cross-Batch generator initialized")
        except Exception as e:
            print(f"[GPU {physical_gpu_id}] Failed to initialize Cross-Batch: {e}")
            import traceback
            traceback.print_exc()
            if "cross_batch" in results:
                del results["cross_batch"]

    # Import extract_answer
    from src.inference import extract_answer

    # Process each group
    for local_idx, (group_idx, group) in enumerate(zip(group_indices, groups)):
        if local_idx % 10 == 0:
            print(f"[GPU {physical_gpu_id}] Processing group {local_idx+1}/{len(groups)} (global idx: {group_idx})")

        context = group[0]["context"]

        # Strategy: all_in_one - all questions in one prompt
        if "all_in_one" in results:
            questions_text = "\n".join([f"{i+1}. {q['question']}" for i, q in enumerate(group)])
            prompt = f"Passage:\n{context}\n\nQuestions:\n{questions_text}\n\nAnswer each question with numbered responses. Wrap each answer in <answer></answer> tags.\nExample format:\n1. <answer>answer1</answer>\n2. <answer>answer2</answer>"

            messages = [{"role": "system", "content": ALL_IN_ONE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer([full_prompt], return_tensors="pt").to(device)
            prompt_tokens = inputs["input_ids"].shape[1]

            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=96*len(group), do_sample=False, pad_token_id=tokenizer.pad_token_id)
            latency = time.perf_counter() - start
            results["all_in_one"]["latency"] += latency

            generated_tokens = outputs[0].shape[0] - prompt_tokens
            results["all_in_one"]["prompt_tokens"] += prompt_tokens
            results["all_in_one"]["generated_tokens"] += generated_tokens
            results["all_in_one"]["prompt_tokens_api"] += prompt_tokens
            results["all_in_one"]["generated_tokens_api"] += generated_tokens

            raw_text = tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)

            # Try multiple parsing strategies
            answers = []

            # Strategy 1: Look for numbered <answer> tags
            numbered_answers = re.findall(r'\d+\.\s*<answer>(.*?)</answer>', raw_text, re.DOTALL | re.IGNORECASE)
            if len(numbered_answers) >= len(group):
                answers = numbered_answers

            # Strategy 2: Look for any <answer> tags
            if not answers:
                answers = re.findall(r'<answer>(.*?)</answer>', raw_text, re.DOTALL | re.IGNORECASE)

            # Strategy 3: Look for numbered answers without tags
            if len(answers) < len(group):
                lines = raw_text.strip().split('\n')
                numbered_lines = []
                for line in lines:
                    match = re.match(r'^(\d+)[\.\)]\s*(.+)', line.strip())
                    if match:
                        numbered_lines.append((int(match.group(1)), match.group(2).strip()))
                if len(numbered_lines) >= len(group):
                    numbered_lines.sort(key=lambda x: x[0])
                    answers = [a[1] for a in numbered_lines[:len(group)]]

            for i, q in enumerate(group):
                answer = answers[i].strip() if i < len(answers) else ""
                answer = re.sub(r'</?answer>', '', answer).strip()
                results["all_in_one"]["predictions"][q["qid"]] = (answer, len(answer) > 0)

        # Strategy: sequential - one by one in conversation
        if "sequential" in results:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            seq_prompt_tokens_api = 0
            seq_generated_tokens = 0
            seq_deduplicated_tokens = 0
            prev_prompt_tokens = 0
            prev_gen_tokens = 0

            for i, q in enumerate(group):
                prompt = f"Passage:\n{q['context']}\n\nQuestion: {q['question']}" if i == 0 else f"Question: {q['question']}"
                messages.append({"role": "user", "content": prompt})

                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([full_prompt], return_tensors="pt").to(device)
                current_prompt_tokens = inputs["input_ids"].shape[1]

                start = time.perf_counter()
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                results["sequential"]["latency"] += time.perf_counter() - start

                gen_tokens = output[0].shape[0] - current_prompt_tokens
                seq_prompt_tokens_api += current_prompt_tokens
                seq_generated_tokens += gen_tokens

                # Deduplicated: first turn full, subsequent turns incremental
                if i == 0:
                    seq_deduplicated_tokens += current_prompt_tokens
                else:
                    incremental = current_prompt_tokens - prev_prompt_tokens - prev_gen_tokens
                    seq_deduplicated_tokens += incremental

                prev_prompt_tokens = current_prompt_tokens
                prev_gen_tokens = gen_tokens

                raw_text = tokenizer.decode(output[0][current_prompt_tokens:], skip_special_tokens=True)
                answer, valid = extract_answer(raw_text, args_dict["dataset"])
                results["sequential"]["predictions"][q["qid"]] = (answer, valid)
                messages.append({"role": "assistant", "content": raw_text})

            results["sequential"]["prompt_tokens"] += seq_deduplicated_tokens
            results["sequential"]["generated_tokens"] += seq_generated_tokens
            results["sequential"]["prompt_tokens_api"] += seq_prompt_tokens_api
            results["sequential"]["generated_tokens_api"] += seq_generated_tokens

        # Strategy: batch - all in parallel (Independent)
        if "batch" in results:
            prompts = []
            for q in group:
                prompt = f"Passage:\n{q['context']}\n\nQuestion: {q['question']}"
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(full_prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
            context_tokens = len(tokenizer.encode(context))

            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            results["batch"]["latency"] += time.perf_counter() - start

            batch_prompt_tokens_api = 0
            batch_generated_tokens = 0
            batch_deduplicated_tokens = 0

            for i, q in enumerate(group):
                # Count non-padding tokens
                prompt_tokens = inputs["attention_mask"][i].sum().item()
                batch_prompt_tokens_api += prompt_tokens

                generated = outputs[i][inputs["input_ids"][i].shape[0]:]
                gen_tokens = generated.shape[0]
                batch_generated_tokens += gen_tokens

                # Deduplicated: first question full, subsequent questions minus context
                if i == 0:
                    batch_deduplicated_tokens += prompt_tokens
                else:
                    batch_deduplicated_tokens += prompt_tokens - context_tokens

                raw_text = tokenizer.decode(generated, skip_special_tokens=True)
                answer, valid = extract_answer(raw_text, args_dict["dataset"])
                results["batch"]["predictions"][q["qid"]] = (answer, valid)

            results["batch"]["prompt_tokens"] += batch_deduplicated_tokens
            results["batch"]["generated_tokens"] += batch_generated_tokens
            results["batch"]["prompt_tokens_api"] += batch_prompt_tokens_api
            results["batch"]["generated_tokens_api"] += batch_generated_tokens

        # Strategy: memory - 3-shot sequential (like sequential but with 3 examples in first turn)
        if "memory" in results and memory_bank and memory_embeddings is not None and embedding_model:
            # Retrieve 3 examples based on first question in group (exclude same context to prevent data leakage)
            first_q = group[0]
            query_emb = embedding_model.encode([first_q["question"]], convert_to_numpy=True)[0]
            similar_examples = retrieve_similar_examples(query_emb, memory_embeddings, memory_bank, top_k=3, exclude_context=context)

            # Build examples text
            examples_text = ""
            for idx, ex in enumerate(similar_examples, 1):
                ex_context = ex['context'][:300] if len(ex['context']) > 300 else ex['context']
                examples_text += f"Example {idx}:\nPassage: {ex_context}\nQuestion: {ex['question']}\n<answer>{ex['answer']}</answer>\n\n"

            # Sequential conversation with 3-shot examples in first turn
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            mem_prompt_tokens_api = 0
            mem_generated_tokens = 0
            mem_deduplicated_tokens = 0
            prev_prompt_tokens = 0
            prev_gen_tokens = 0

            for i, q in enumerate(group):
                if i == 0:
                    # First turn: include 3 examples + passage + question
                    prompt = f"{examples_text}Now answer:\nPassage:\n{q['context']}\n\nQuestion: {q['question']}"
                else:
                    # Subsequent turns: just the question
                    prompt = f"Question: {q['question']}"

                messages.append({"role": "user", "content": prompt})
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([full_prompt], return_tensors="pt").to(device)
                current_prompt_tokens = inputs["input_ids"].shape[1]

                start = time.perf_counter()
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                results["memory"]["latency"] += time.perf_counter() - start

                gen_tokens = output[0].shape[0] - current_prompt_tokens
                mem_prompt_tokens_api += current_prompt_tokens
                mem_generated_tokens += gen_tokens

                # Deduplicated: first turn full, subsequent turns incremental
                if i == 0:
                    mem_deduplicated_tokens += current_prompt_tokens
                else:
                    incremental = current_prompt_tokens - prev_prompt_tokens - prev_gen_tokens
                    mem_deduplicated_tokens += incremental

                prev_prompt_tokens = current_prompt_tokens
                prev_gen_tokens = gen_tokens

                raw_text = tokenizer.decode(output[0][current_prompt_tokens:], skip_special_tokens=True)
                answer, valid = extract_answer(raw_text, args_dict["dataset"])
                results["memory"]["predictions"][q["qid"]] = (answer, valid)
                messages.append({"role": "assistant", "content": raw_text})

            results["memory"]["prompt_tokens"] += mem_deduplicated_tokens
            results["memory"]["generated_tokens"] += mem_generated_tokens
            results["memory"]["prompt_tokens_api"] += mem_prompt_tokens_api
            results["memory"]["generated_tokens_api"] += mem_generated_tokens

        # Strategy: cross_batch - Cross-Batch with trained checkpoint
        if "cross_batch" in results and cross_batch_generator:
            # DEBUG: Print batch size for first few groups
            if local_idx < 3:
                print(f"[GPU {physical_gpu_id}] Cross-Batch group {local_idx}: batch_size={len(group)}")

            start = time.perf_counter()
            try:
                # Use multi-context strategy since each question has its own context
                from src.strategies.cross_batch import run_cross_batch_multi_strategy
                result = run_cross_batch_multi_strategy(
                    items=group,  # Each item has qid, question, context, references
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=96,
                    dataset=args_dict["dataset"],
                    cross_batch_generator=cross_batch_generator,
                    enable_cross_batch=True,
                )
                results["cross_batch"]["latency"] += time.perf_counter() - start

                # Extract token statistics and predictions
                context_tokens = len(tokenizer.encode(context))
                cb_prompt_tokens_api = 0
                cb_generated_tokens = 0
                cb_deduplicated_tokens = 0

                for i, q in enumerate(group):
                    answer = result.answers.get(q["qid"], "")
                    valid = True
                    prompt_tokens = 0
                    gen_tokens = 0

                    for detail in result.details.get("questions", []):
                        if detail["question_id"] == q["qid"]:
                            valid = detail.get("strict_valid", True)
                            prompt_tokens = detail.get("prompt_tokens", 0)
                            gen_tokens = detail.get("generated_tokens", 0)
                            break

                    results["cross_batch"]["predictions"][q["qid"]] = (answer, valid)

                    # Track tokens
                    cb_prompt_tokens_api += prompt_tokens
                    cb_generated_tokens += gen_tokens

                    # Deduplicated: first question full, subsequent questions minus context
                    if i == 0:
                        cb_deduplicated_tokens += prompt_tokens
                    else:
                        cb_deduplicated_tokens += prompt_tokens - context_tokens

                results["cross_batch"]["prompt_tokens"] += cb_deduplicated_tokens
                results["cross_batch"]["generated_tokens"] += cb_generated_tokens
                results["cross_batch"]["prompt_tokens_api"] += cb_prompt_tokens_api
                results["cross_batch"]["generated_tokens_api"] += cb_generated_tokens
            except Exception as e:
                print(f"[GPU {physical_gpu_id}] Cross-Batch failed for group {group_idx}: {e}")
                results["cross_batch"]["latency"] += time.perf_counter() - start
                for q in group:
                    results["cross_batch"]["predictions"][q["qid"]] = ("", False)

    print(f"[GPU {physical_gpu_id}] Done processing {len(groups)} groups")
    result_queue.put((worker_id, results))


def run_evaluation_multi_gpu(args, group_size, all_questions, output_dir, gpu_ids, memory_bank=None, memory_embeddings=None, embedding_model=None):
    """Run evaluation with specific group size using multiple GPUs."""
    import time
    from src.models import Question
    from src.evaluation import evaluate_predictions

    num_gpus = len(gpu_ids)
    logger.info(f"\n{'='*80}")
    logger.info(f"GROUP SIZE: {group_size} questions per group")
    logger.info(f"Total questions: {len(all_questions)}, Groups: {len(all_questions)//group_size}")
    logger.info(f"Using {num_gpus} GPUs: {gpu_ids}")
    logger.info(f"{'='*80}\n")

    # Parse strategies to run
    strategies_to_run = [s.strip() for s in args.strategies.split(',')]

    # Group questions
    groups = group_questions(all_questions, group_size)
    num_groups = len(groups)

    # Distribute groups across GPUs
    groups_per_gpu = (num_groups + num_gpus - 1) // num_gpus

    # Convert args to dict for pickling
    args_dict = {
        "model": args.model,
        "dataset": args.dataset,
        "cross_batch_checkpoint": args.cross_batch_checkpoint,
        "embedding_model": args.embedding_model,
        "lora_checkpoint": args.lora_checkpoint,
        "sft_format": args.sft_format,
    }

    # Convert memory embeddings to list for pickling
    memory_embeddings_list = memory_embeddings.tolist() if memory_embeddings is not None else None

    # Create result queue
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    # Start workers
    processes = []
    start_time = time.perf_counter()

    for worker_id, physical_gpu_id in enumerate(gpu_ids):
        start_idx = worker_id * groups_per_gpu
        end_idx = min(start_idx + groups_per_gpu, num_groups)

        if start_idx >= num_groups:
            break

        gpu_groups = groups[start_idx:end_idx]
        gpu_indices = list(range(start_idx, end_idx))

        p = ctx.Process(
            target=gpu_worker,
            args=(
                worker_id,
                physical_gpu_id,
                gpu_groups,
                gpu_indices,
                args_dict,
                strategies_to_run,
                memory_bank,
                memory_embeddings_list,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    logger.info(f"Started {len(processes)} GPU workers")

    # Collect results from all workers
    all_results = {}
    for _ in range(len(processes)):
        gpu_id, results = result_queue.get()
        logger.info(f"Received results from GPU {gpu_id}")
        for strategy, data in results.items():
            if strategy not in all_results:
                all_results[strategy] = {
                    "predictions": {},
                    "latency": 0,
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "prompt_tokens_api": 0,
                    "generated_tokens_api": 0,
                }
            all_results[strategy]["predictions"].update(data["predictions"])
            all_results[strategy]["latency"] += data["latency"]
            all_results[strategy]["prompt_tokens"] += data.get("prompt_tokens", 0)
            all_results[strategy]["generated_tokens"] += data.get("generated_tokens", 0)
            all_results[strategy]["prompt_tokens_api"] += data.get("prompt_tokens_api", 0)
            all_results[strategy]["generated_tokens_api"] += data.get("generated_tokens_api", 0)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    total_time = time.perf_counter() - start_time
    logger.info(f"All GPUs finished in {total_time:.2f}s")

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
    num_groups = len(groups)
    # Use mapped dataset name for evaluation (e.g., "cmb" -> "cmb_exam_context" for accuracy metrics)
    eval_dataset = EVAL_DATASET_MAP.get(args.dataset, args.dataset)

    # Print a few examples before full evaluation
    logger.info("\n" + "="*80)
    logger.info("SAMPLE PREDICTIONS (first 3 questions)")
    logger.info("="*80)
    sample_questions = list(question_lookup.values())[:3]
    for q in sample_questions:
        logger.info(f"\nQuestion: {q.text}")
        logger.info(f"References: {q.references}")
        for strategy_name, result_data in all_results.items():
            pred, valid = result_data["predictions"].get(q.qid, ("", False))
            logger.info(f"  {strategy_name}: {pred} {'✓' if valid else '✗'}")
    logger.info("="*80 + "\n")

    for strategy_name, result_data in all_results.items():
        metrics = evaluate_predictions(result_data["predictions"], question_lookup, dataset=eval_dataset)
        summary[strategy_name] = {
            "metrics": metrics,
            "latency": result_data["latency"],
            "wall_time": total_time,
            # Token statistics
            "total_prompt_tokens": result_data.get("prompt_tokens", 0),
            "total_generated_tokens": result_data.get("generated_tokens", 0),
            "total_prompt_tokens_api": result_data.get("prompt_tokens_api", 0),
            "total_generated_tokens_api": result_data.get("generated_tokens_api", 0),
            # Averages per group
            "avg_prompt_tokens": result_data.get("prompt_tokens", 0) / num_groups if num_groups > 0 else 0,
            "avg_generated_tokens": result_data.get("generated_tokens", 0) / num_groups if num_groups > 0 else 0,
            "avg_prompt_tokens_api": result_data.get("prompt_tokens_api", 0) / num_groups if num_groups > 0 else 0,
            "avg_generated_tokens_api": result_data.get("generated_tokens_api", 0) / num_groups if num_groups > 0 else 0,
            "num_groups": num_groups,
            "num_questions": len(all_questions),
        }
        # Get primary metric for logging (varies by dataset)
        primary_metric = next(iter(metrics.keys())) if metrics else "unknown"
        primary_value = metrics.get(primary_metric, 0.0)
        logger.info(f"{strategy_name}: {primary_metric}={primary_value:.3f}, "
                   f"PromptTok={result_data.get('prompt_tokens', 0):,}, "
                   f"GenTok={result_data.get('generated_tokens', 0):,}, "
                   f"GPU time={result_data['latency']:.2f}s")

    return summary


STRATEGY_ORDER = ["batch", "all_in_one", "sequential", "memory", "cross_batch"]
STRATEGY_DISPLAY = {
    "batch": "Independent",
    "all_in_one": "All-in-One",
    "sequential": "Sequential",
    "memory": "Memory",
    "cross_batch": "Cross-Batch",
}

# Mapping from loader dataset name to evaluation dataset name
# CMB loader uses "cmb" but loads CMB-Exam (multiple choice), so evaluation should use "cmb_exam_context"
EVAL_DATASET_MAP = {
    "cmb": "cmb_exam_context",  # CMB-Exam context groups -> accuracy metric
}


def main():
    args = parse_args()

    # Get available GPUs from CUDA_VISIBLE_DEVICES
    gpu_ids = get_available_gpus()
    num_gpus = len(gpu_ids)

    # Parse group sizes
    group_sizes = [int(x.strip()) for x in args.group_sizes.split(',')]
    max_group_size = max(group_sizes)
    logger.info(f"Testing group sizes: {group_sizes}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Using {num_gpus} GPUs: {gpu_ids}")
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

        summary = run_evaluation_multi_gpu(
            args, group_size, all_questions, output_dir, gpu_ids,
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
        f.write(f"# GPUs: {gpu_ids}\n")
        f.write(f"# All tests use the same {total_questions} questions, just grouped differently\n\n")

        # Detect available metrics from first result
        available_metrics = []
        if results_by_group_size and actual_strategies:
            first_gs = next(iter(results_by_group_size.keys()))
            first_strategy = next(iter(actual_strategies))
            if first_strategy in results_by_group_size[first_gs]:
                available_metrics = list(results_by_group_size[first_gs][first_strategy]['metrics'].keys())

        # Define metric display names
        metric_display = {
            "strict_acc": "EM (Exact Match)",
            "f1": "F1 Score",
            "lenient_acc": "Lenient Accuracy",
            "bleu4": "BLEU-4",
            "rouge1": "ROUGE-1",
            "rouge2": "ROUGE-2",
            "rougeL": "ROUGE-L",
            "acc": "Accuracy",
        }

        # Write a table for each available metric
        for metric_name in available_metrics:
            display_name = metric_display.get(metric_name, metric_name)
            f.write(f"## Results - {display_name}\n\n")
            header = f"{'GroupSize':<10}"
            for strategy in actual_strategies:
                header += f" | {STRATEGY_DISPLAY[strategy]:>12}"
            if metric_name == available_metrics[0]:
                header += f" | {'NumGroups':>10}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            for gs in sorted(results_by_group_size.keys()):
                result = results_by_group_size[gs]
                num_groups = total_questions // gs
                line = f"{gs:<10}"
                for strategy in actual_strategies:
                    if strategy in result:
                        value = result[strategy]['metrics'].get(metric_name, 0)
                        line += f" | {value:>12.3f}"
                    else:
                        line += f" | {'--':>12}"
                if metric_name == available_metrics[0]:
                    line += f" | {num_groups:>10}"
                f.write(line + "\n")
            f.write("\n")

        f.write("\n## Token Statistics (Total)\n\n")
        header = f"{'GroupSize':<10}"
        for strategy in actual_strategies:
            header += f" | {STRATEGY_DISPLAY[strategy]:>12}"
        f.write(header + " (Prompt Tokens - Deduplicated)\n")
        f.write("-" * len(header) + "\n")

        for gs in sorted(results_by_group_size.keys()):
            result = results_by_group_size[gs]
            line = f"{gs:<10}"
            for strategy in actual_strategies:
                if strategy in result:
                    tokens = result[strategy].get('total_prompt_tokens', 0)
                    line += f" | {tokens:>12,}"
                else:
                    line += f" | {'--':>12}"
            f.write(line + "\n")

        f.write("\n## Token Statistics (API)\n\n")
        header = f"{'GroupSize':<10}"
        for strategy in actual_strategies:
            header += f" | {STRATEGY_DISPLAY[strategy]:>12}"
        f.write(header + " (Prompt Tokens - API)\n")
        f.write("-" * len(header) + "\n")

        for gs in sorted(results_by_group_size.keys()):
            result = results_by_group_size[gs]
            line = f"{gs:<10}"
            for strategy in actual_strategies:
                if strategy in result:
                    tokens = result[strategy].get('total_prompt_tokens_api', 0)
                    line += f" | {tokens:>12,}"
                else:
                    line += f" | {'--':>12}"
            f.write(line + "\n")

        f.write("\n## Generated Tokens (Total)\n\n")
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
                    tokens = result[strategy].get('total_generated_tokens', 0)
                    line += f" | {tokens:>12,}"
                else:
                    line += f" | {'--':>12}"
            f.write(line + "\n")

        f.write("\n## GPU Latency (seconds)\n\n")
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
                    latency = result[strategy].get('latency', 0)
                    line += f" | {latency:>12.2f}"
                else:
                    line += f" | {'--':>12}"
            f.write(line + "\n")

        f.write("\n## Wall Time (seconds)\n\n")
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
                    wall_time = result[strategy].get('wall_time', result[strategy]['latency'])
                    line += f" | {wall_time:>12.2f}"
                else:
                    line += f" | {'--':>12}"
            f.write(line + "\n")

        # Per-question average tables
        f.write("\n## Avg Prompt Tokens per Question (Deduplicated)\n\n")
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
                    tokens = result[strategy].get('total_prompt_tokens', 0)
                    num_q = result[strategy].get('num_questions', 1)
                    avg = tokens / num_q if num_q > 0 else 0
                    line += f" | {avg:>12.1f}"
                else:
                    line += f" | {'--':>12}"
            f.write(line + "\n")

        f.write("\n## Avg Prompt Tokens per Question (API)\n\n")
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
                    tokens = result[strategy].get('total_prompt_tokens_api', 0)
                    num_q = result[strategy].get('num_questions', 1)
                    avg = tokens / num_q if num_q > 0 else 0
                    line += f" | {avg:>12.1f}"
                else:
                    line += f" | {'--':>12}"
            f.write(line + "\n")

        f.write("\n## Avg Generated Tokens per Question\n\n")
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
                    tokens = result[strategy].get('total_generated_tokens', 0)
                    num_q = result[strategy].get('num_questions', 1)
                    avg = tokens / num_q if num_q > 0 else 0
                    line += f" | {avg:>12.1f}"
                else:
                    line += f" | {'--':>12}"
            f.write(line + "\n")

        f.write("\n## Avg Latency per Question (seconds)\n\n")
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
                    latency = result[strategy].get('latency', 0)
                    num_q = result[strategy].get('num_questions', 1)
                    avg = latency / num_q if num_q > 0 else 0
                    line += f" | {avg:>12.4f}"
                else:
                    line += f" | {'--':>12}"
            f.write(line + "\n")

    logger.info(f"Summary saved to {summary_file}\n")
    with open(summary_file, 'r') as f:
        print(f.read())

    # Save all results with config
    all_results_data = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "seed": args.seed,
            "group_sizes": group_sizes,
            "max_contexts": args.max_contexts,
            "strategies": strategies_to_run,
            "num_contexts": len(contexts),
            "total_questions": total_questions,
        },
        "results_by_group_size": results_by_group_size,
    }
    all_results_file = output_dir / "all_results.json"
    with open(all_results_file, 'w') as f:
        json.dump(all_results_data, f, indent=2)
    logger.info(f"\nAll results saved to {all_results_file}")


if __name__ == "__main__":
    main()
