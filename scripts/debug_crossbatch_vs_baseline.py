#!/usr/bin/env python3
"""
Debug script to compare Cross-Batch generator vs standard model.generate()

Tests whether the accuracy drop is due to:
1. Prompt format difference
2. Custom generation loop bugs
3. CSA module issues
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.eval_utils import load_dataset_groups
from src.prompts import build_single_prompt
from src.templates import build_chat_prompt
from src.models import Question
from src.inference import extract_answer
from src.evaluation import evaluate_predictions

def test_generation_methods(model_name="Qwen/Qwen2.5-7B-Instruct", num_contexts=10):
    """Compare standard generate() vs CrossBatchGenerator."""

    device = "cuda:0"

    # Load model
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

    # Load test data (same as eval_question_grouping_impact.py)
    contexts = load_dataset_groups(
        dataset="squad",
        split="validation",
        max_contexts=num_contexts,
        min_questions=5,
        max_questions=1000,
        seed=42,
        fixed_question_count=5,
    )

    print(f"Loaded {len(contexts)} contexts")

    # Test on first context (5 questions)
    ctx = contexts[0]
    questions = [
        Question(
            qid=q["qid"],
            text=q["text"],
            priority=1.0,
            answer_tokens=12,
            type_hint=None,
            references=q["references"],
        )
        for q in ctx["questions"]
    ]

    print(f"\nTesting on context with {len(questions)} questions...")
    print(f"Context: {ctx['context'][:100]}...")

    # Method 1: Standard model.generate() (like Independent strategy)
    print("\n" + "="*60)
    print("Method 1: Standard model.generate() (Independent)")
    print("="*60)

    prompts_standard = []
    for q in questions:
        from src.eval_utils import SYSTEM_PROMPT
        prompt = f"Passage:\n{ctx['context']}\n\nQuestion: {q.text}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts_standard.append(full_prompt)

    inputs = tokenizer(prompts_standard, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    answers_standard = []
    for i, q in enumerate(questions):
        generated = outputs[i][inputs["input_ids"][i].shape[0]:]
        raw_text = tokenizer.decode(generated, skip_special_tokens=True)
        answer, valid = extract_answer(raw_text, "squad")
        answers_standard.append((answer, valid))
        print(f"Q{i+1}: {q.text[:50]}... → {answer[:30]}...")

    # Method 2: CrossBatchGenerator with enable_cross_batch=False
    print("\n" + "="*60)
    print("Method 2: CrossBatchGenerator (enable_cross_batch=False)")
    print("="*60)

    from src.cross_batch import CrossBatchGenerator, CrossBatchAttention

    # Create CSA module (with zero-init)
    hidden_size = model.config.hidden_size
    cross_batch_module = CrossBatchAttention(hidden_size=hidden_size)
    cross_batch_module.to(device)
    cross_batch_module.eval()

    generator = CrossBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        mix_method="attention",
        mix_layer=-1,
        device=device,
    )

    # Build prompts (same format as strategies/cross_batch.py)
    prompts_cb = []
    for q in questions:
        system_prompt, user_prompt = build_single_prompt(ctx['context'], q, dataset="squad")
        full_prompt = build_chat_prompt(tokenizer, user_prompt, system_prompt=system_prompt)
        prompts_cb.append(full_prompt)

    tokenizer.padding_side = "left"
    encoded = tokenizer(prompts_cb, return_tensors="pt", padding=True)
    tokenizer.padding_side = "right"

    outputs_cb = generator.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=96,
        do_sample=False,
        enable_cross_batch=False,  # Disable CSA
    )

    sequences = outputs_cb["sequences"]
    prompt_window = encoded["input_ids"].shape[-1]

    answers_cb = []
    for i, q in enumerate(questions):
        tokens = []
        for token in sequences[i][prompt_window:].tolist():
            if token in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                break
            tokens.append(token)
        raw_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        answer, valid = extract_answer(raw_text, "squad")
        answers_cb.append((answer, valid))
        print(f"Q{i+1}: {q.text[:50]}... → {answer[:30]}...")

    # Compare
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)

    question_lookup = {q.qid: q for q in questions}

    preds_standard = {q.qid: answers_standard[i] for i, q in enumerate(questions)}
    preds_cb = {q.qid: answers_cb[i] for i, q in enumerate(questions)}

    metrics_standard = evaluate_predictions(preds_standard, question_lookup, dataset="squad")
    metrics_cb = evaluate_predictions(preds_cb, question_lookup, dataset="squad")

    print(f"Standard generate():     EM={metrics_standard.get('strict_acc', 0):.3f}, F1={metrics_standard.get('f1', 0):.3f}")
    print(f"CrossBatch (CSA off):    EM={metrics_cb.get('strict_acc', 0):.3f}, F1={metrics_cb.get('f1', 0):.3f}")

    print("\nPrompt format comparison:")
    print(f"  Standard == CrossBatch: {prompts_standard[0] == prompts_cb[0]}")
    if prompts_standard[0] != prompts_cb[0]:
        print("\n  Standard prompt:")
        print(f"  {prompts_standard[0][:200]}...")
        print("\n  CrossBatch prompt:")
        print(f"  {prompts_cb[0][:200]}...")

if __name__ == "__main__":
    test_generation_methods()
