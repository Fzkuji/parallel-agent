#!/usr/bin/env python3
"""
Test if batch inference is truly independent.

Compares:
1. Batch inference (all questions at once)
2. Sequential single inference (one question at a time)

If batch is truly independent, results should be identical.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.datasets.squad import load_squad_groups

# Load model
model_name = "Qwen/Qwen2.5-7B-Instruct"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    trust_remote_code=True,
)
model.eval()
print("Model loaded")

# Load one 20-question context
contexts = load_squad_groups(
    split="validation",
    max_contexts=1,
    min_questions=20,
    max_questions=20,
    seed=42,
    fixed_question_count=20
)

if len(contexts) == 0:
    print("No 20-question contexts found!")
    sys.exit(1)

context_data = contexts[0]
items = []
for q in context_data["questions"]:
    items.append({
        "qid": q["qid"],
        "question": q["text"],
        "context": context_data["context"],
        "references": q["references"],
    })

print(f"\nTesting with {len(items)} questions from context: {context_data.get('title', 'Unknown')}")

SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the given passage.
You MUST wrap your answer in <answer></answer> tags. Be concise.

Example:
Question: What color is the sky?
<answer>blue</answer>"""

# Method 1: Batch inference
print("\n=== Method 1: Batch Inference ===")
prompts = []
for item in items:
    prompt = f"Passage:\n{item['context']}\n\nQuestion: {item['question']}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompts.append(full_prompt)

inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=96,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

batch_answers = []
for i, item in enumerate(items):
    generated = outputs[i][inputs["input_ids"][i].shape[0]:]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True)
    batch_answers.append(raw_text)
    print(f"Q{i+1} batch: {raw_text[:50]}...")

# Method 2: Individual inference
print("\n=== Method 2: Individual Inference ===")
individual_answers = []
for i, item in enumerate(items):
    prompt = f"Passage:\n{item['context']}\n\nQuestion: {item['question']}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs_single = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs_single = {k: v.to(model.device) for k, v in inputs_single.items()}

    with torch.no_grad():
        output_single = model.generate(
            **inputs_single,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs_single["input_ids"].shape[1]
    generated = output_single[0][prompt_len:]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True)
    individual_answers.append(raw_text)
    print(f"Q{i+1} indiv: {raw_text[:50]}...")

# Compare
print("\n=== Comparison ===")
differences = 0
for i in range(len(items)):
    if batch_answers[i] != individual_answers[i]:
        differences += 1
        print(f"\nQ{i+1} DIFFERS:")
        print(f"  Batch:      {batch_answers[i][:100]}")
        print(f"  Individual: {individual_answers[i][:100]}")

if differences == 0:
    print("✓ ALL ANSWERS IDENTICAL - Batch is truly independent!")
else:
    print(f"\n✗ {differences}/{len(items)} answers differ - Batch has interference!")
