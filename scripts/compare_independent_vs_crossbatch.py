#!/usr/bin/env python3
"""
Compare Independent (batch) vs Cross-Batch generator to find why results differ.

Tests:
1. Independent: Standard model.generate() with batch parallel
2. Cross-Batch (CSA off): CrossBatchGenerator with enable_cross_batch=False
3. Cross-Batch (CSA on): CrossBatchGenerator with enable_cross_batch=True (random init)

If Cross-Batch (CSA off) = Independent = 80.4%, generator logic is correct.
If Cross-Batch (CSA on, random) = Sequential ≈ 73%, there's a hidden sequential behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.eval_utils import load_dataset_groups, SYSTEM_PROMPT
from src.models import Question
from src.inference import extract_answer
from src.evaluation import evaluate_predictions
from src.cross_batch import CrossBatchGenerator, CrossBatchAttention
from src.prompts import build_single_prompt
from src.templates import build_chat_prompt

def run_comparison(model_name="Qwen/Qwen2.5-7B-Instruct", num_contexts=20):
    """Compare three methods on the same data."""

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

    # Load test data (G=2, 同一个 context 的 2 个问题)
    contexts = load_dataset_groups(
        dataset="squad",
        split="validation",
        max_contexts=num_contexts,
        min_questions=2,
        max_questions=1000,
        seed=42,
        fixed_question_count=2,
    )

    print(f"\nTesting on {len(contexts)} contexts (2 questions each = {len(contexts)*2} total)")

    all_predictions = {
        "independent": {},
        "crossbatch_off": {},
        "crossbatch_on": {},
    }

    question_lookup = {}

    for ctx_idx, ctx in enumerate(contexts):
        questions = [
            Question(
                qid=f"ctx{ctx_idx}_{q['qid']}",
                text=q["text"],
                priority=1.0,
                answer_tokens=12,
                type_hint=None,
                references=q["references"],
            )
            for q in ctx["questions"][:2]  # Take first 2
        ]

        for q in questions:
            question_lookup[q.qid] = q

        # Method 1: Independent (standard batch generate)
        prompts_ind = []
        for q in questions:
            prompt = f"Passage:\n{ctx['context']}\n\nQuestion: {q.text}"
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts_ind.append(full_prompt)

        inputs = tokenizer(prompts_ind, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.pad_token_id)

        for i, q in enumerate(questions):
            generated = outputs[i][inputs["input_ids"][i].shape[0]:]
            raw_text = tokenizer.decode(generated, skip_special_tokens=True)
            answer, valid = extract_answer(raw_text, "squad")
            all_predictions["independent"][q.qid] = (answer, valid)

        # Method 2: Cross-Batch (CSA off)
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

        prompts_cb = []
        for q in questions:
            system_prompt, user_prompt = build_single_prompt(ctx['context'], q, dataset="squad")
            full_prompt = build_chat_prompt(tokenizer, user_prompt, system_prompt=system_prompt)
            prompts_cb.append(full_prompt)

        tokenizer.padding_side = "left"
        encoded = tokenizer(prompts_cb, return_tensors="pt", padding=True)
        tokenizer.padding_side = "right"

        # CSA off
        outputs_cb_off = generator.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=96,
            do_sample=False,
            enable_cross_batch=False,
        )

        prompt_window = encoded["input_ids"].shape[-1]
        for i, q in enumerate(questions):
            tokens = []
            for token in outputs_cb_off["sequences"][i][prompt_window:].tolist():
                if token in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                    break
                tokens.append(token)
            raw_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            answer, valid = extract_answer(raw_text, "squad")
            all_predictions["crossbatch_off"][q.qid] = (answer, valid)

        # CSA on (random init)
        outputs_cb_on = generator.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=96,
            do_sample=False,
            enable_cross_batch=True,
        )

        for i, q in enumerate(questions):
            tokens = []
            for token in outputs_cb_on["sequences"][i][prompt_window:].tolist():
                if token in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                    break
                tokens.append(token)
            raw_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            answer, valid = extract_answer(raw_text, "squad")
            all_predictions["crossbatch_on"][q.qid] = (answer, valid)

        if (ctx_idx + 1) % 5 == 0:
            print(f"Processed {ctx_idx + 1}/{len(contexts)} contexts...")

    # Evaluate all
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)

    metrics_ind = evaluate_predictions(all_predictions["independent"], question_lookup, dataset="squad")
    metrics_cb_off = evaluate_predictions(all_predictions["crossbatch_off"], question_lookup, dataset="squad")
    metrics_cb_on = evaluate_predictions(all_predictions["crossbatch_on"], question_lookup, dataset="squad")

    print(f"Independent (batch):        EM={metrics_ind.get('strict_acc', 0):.3f}, F1={metrics_ind.get('f1', 0):.3f}")
    print(f"Cross-Batch (CSA off):      EM={metrics_cb_off.get('strict_acc', 0):.3f}, F1={metrics_cb_off.get('f1', 0):.3f}")
    print(f"Cross-Batch (CSA on, rand): EM={metrics_cb_on.get('strict_acc', 0):.3f}, F1={metrics_cb_on.get('f1', 0):.3f}")

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    if abs(metrics_ind.get('strict_acc', 0) - metrics_cb_off.get('strict_acc', 0)) < 0.01:
        print("✓ Cross-Batch generator (CSA off) = Independent (generator logic correct)")
    else:
        print("✗ Cross-Batch generator (CSA off) ≠ Independent (generator has bugs)")

    if abs(metrics_cb_off.get('strict_acc', 0) - metrics_cb_on.get('strict_acc', 0)) < 0.01:
        print("✓ Random CSA has no effect (initialization working)")
    else:
        print(f"✗ Random CSA changes results by {abs(metrics_cb_off.get('strict_acc', 0) - metrics_cb_on.get('strict_acc', 0)):.1%}")
        print("  This explains why trained CSA (with Improvement=0) still differs from baseline")

if __name__ == "__main__":
    run_comparison()
