#!/usr/bin/env python3
"""
Debug why context 0 gets 80% EM in batch 5Q evaluation.

Reproduces the exact batch evaluation for context 0 with 5 questions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.datasets.squad import load_squad_groups
from src.inference import extract_answer
from src.evaluation import evaluate_predictions
from src.models import Question

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
print("Model loaded\n")

# Load contexts
contexts = load_squad_groups(
    split="validation",
    max_contexts=1000,
    min_questions=20,
    max_questions=20,
    seed=42,
    fixed_question_count=5
)

# Get context 0 (the one with degradation)
context_data = contexts[0]

# Build items
items = []
for q in context_data["questions"]:
    items.append({
        "qid": q["qid"],
        "question": q["text"],
        "context": context_data["context"],
        "references": q["references"],
        "answer_tokens": q.get("answer_tokens", 12),
    })

print(f"Context 0: {context_data['title']}")
print(f"Questions: {len(items)}\n")

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

# System prompt (same as baseline_pretrained.py)
SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the given passage.
You MUST wrap your answer in <answer></answer> tags. Be concise.

Example:
Question: What color is the sky?
<answer>blue</answer>"""

# Build prompts
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

# Tokenize batch
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate
print("Generating answers...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=96,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

# Extract answers
answer_records = {}
for i, item in enumerate(items):
    generated = outputs[i][inputs["input_ids"][i].shape[0]:]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True)
    final_answer, strict_valid = extract_answer(raw_text, "squad")
    answer_records[item["qid"]] = (final_answer, strict_valid)

    print(f"\nQ{i+1} ({item['qid']}): {item['question']}")
    print(f"  Raw: {raw_text}")
    print(f"  Extracted: {final_answer}")
    print(f"  References: {item['references']}")

# Evaluate
metrics = evaluate_predictions(answer_records, question_lookup, dataset="squad")

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"EM: {metrics['strict_acc']:.1%}")
print(f"F1: {metrics['f1']:.3f}")

if metrics['strict_acc'] < 1.0:
    print("\n⚠️ Not 100% accurate!")
    print("This reproduces the 80% EM seen in the evaluation.")
else:
    print("\n✓ 100% accurate!")
    print("Cannot reproduce the degradation. Something else is different.")
