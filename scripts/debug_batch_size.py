#!/usr/bin/env python3
"""
Debug script to compare batch_size=1 vs batch_size=2 behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.cross_batch import CrossBatchGenerator, CrossBatchAttention
from src.eval_utils import load_dataset_groups, SYSTEM_PROMPT
from src.models import Question
from src.templates import build_chat_prompt

def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda:0"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # Load 2 questions from same context
    contexts = load_dataset_groups(
        dataset="squad",
        split="validation",
        max_contexts=1,
        min_questions=2,
        max_questions=1000,
        seed=42,
        fixed_question_count=2,
    )

    ctx = contexts[0]
    q1, q2 = ctx["questions"][:2]

    print("\n" + "="*80)
    print("CONTEXT:", ctx["context"][:200] + "...")
    print("Q1:", q1["text"])
    print("Q2:", q2["text"])
    print("="*80)

    # Create CSA generator
    hidden_size = model.config.hidden_size
    csa_module = CrossBatchAttention(hidden_size=hidden_size)
    csa_module.to(device)
    csa_module.eval()

    generator = CrossBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=csa_module,
        mix_method="attention",
        mix_layer=-1,
        device=device,
    )

    # Prepare prompts
    prompts = []
    for q in [q1, q2]:
        prompt = f"Passage:\n{ctx['context']}\n\nQuestion: {q['text']}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(full_prompt)

    print("\n" + "="*80)
    print("TEST 1: Q1 alone (batch_size=1)")
    print("="*80)

    encoded_1 = tokenizer([prompts[0]], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs_1 = generator.generate(
            input_ids=encoded_1["input_ids"],
            attention_mask=encoded_1["attention_mask"],
            max_new_tokens=96,
            do_sample=False,
            enable_cross_batch=True,  # Should have no effect when batch=1
        )

    prompt_len_1 = encoded_1["input_ids"].shape[1]
    tokens_1 = []
    for token in outputs_1["sequences"][0][prompt_len_1:].tolist():
        if token in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            break
        tokens_1.append(token)
    answer_1_solo = tokenizer.decode(tokens_1, skip_special_tokens=True).strip()
    print(f"Q1 (solo): {answer_1_solo}")

    print("\n" + "="*80)
    print("TEST 2: Q2 alone (batch_size=1)")
    print("="*80)

    encoded_2 = tokenizer([prompts[1]], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs_2 = generator.generate(
            input_ids=encoded_2["input_ids"],
            attention_mask=encoded_2["attention_mask"],
            max_new_tokens=96,
            do_sample=False,
            enable_cross_batch=True,
        )

    prompt_len_2 = encoded_2["input_ids"].shape[1]
    tokens_2 = []
    for token in outputs_2["sequences"][0][prompt_len_2:].tolist():
        if token in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            break
        tokens_2.append(token)
    answer_2_solo = tokenizer.decode(tokens_2, skip_special_tokens=True).strip()
    print(f"Q2 (solo): {answer_2_solo}")

    print("\n" + "="*80)
    print("TEST 3: Q1 + Q2 together (batch_size=2)")
    print("="*80)

    encoded_batch = tokenizer(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs_batch = generator.generate(
            input_ids=encoded_batch["input_ids"],
            attention_mask=encoded_batch["attention_mask"],
            max_new_tokens=96,
            do_sample=False,
            enable_cross_batch=True,
        )

    prompt_len_batch = encoded_batch["input_ids"].shape[1]

    # Q1 in batch
    tokens_1_batch = []
    for token in outputs_batch["sequences"][0][prompt_len_batch:].tolist():
        if token in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            break
        tokens_1_batch.append(token)
    answer_1_batch = tokenizer.decode(tokens_1_batch, skip_special_tokens=True).strip()

    # Q2 in batch
    tokens_2_batch = []
    for token in outputs_batch["sequences"][1][prompt_len_batch:].tolist():
        if token in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            break
        tokens_2_batch.append(token)
    answer_2_batch = tokenizer.decode(tokens_2_batch, skip_special_tokens=True).strip()

    print(f"Q1 (in batch): {answer_1_batch}")
    print(f"Q2 (in batch): {answer_2_batch}")

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    print(f"\nQ1 solo:     {answer_1_solo}")
    print(f"Q1 in batch: {answer_1_batch}")
    print(f"Match: {answer_1_solo == answer_1_batch}")

    print(f"\nQ2 solo:     {answer_2_solo}")
    print(f"Q2 in batch: {answer_2_batch}")
    print(f"Match: {answer_2_solo == answer_2_batch}")

    print(f"\nReferences Q1: {q1['references']}")
    print(f"References Q2: {q2['references']}")

if __name__ == "__main__":
    main()
