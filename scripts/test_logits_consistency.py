#!/usr/bin/env python3
"""
Test if manually computing lm_head(hidden_states[-1]) matches outputs.logits
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_logits_consistency(model_name="Qwen/Qwen2.5-7B-Instruct"):
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()

    # Test input
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    print("\n" + "="*60)
    print("TEST: outputs.logits vs lm_head(hidden_states[-1])")
    print("="*60)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

        # Method 1: Model's native logits
        native_logits = outputs.logits[:, -1, :]

        # Method 2: Manual lm_head on hidden_states[-1]
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        manual_logits = model.lm_head(last_hidden)

        # Compare
        diff = (native_logits - manual_logits).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        native_argmax = native_logits.argmax().item()
        manual_argmax = manual_logits.argmax().item()
        match = (native_argmax == manual_argmax)

        print(f"\nMax difference:  {max_diff:.10f}")
        print(f"Mean difference: {mean_diff:.10f}")
        print(f"\nNative argmax: {native_argmax}")
        print(f"Manual argmax: {manual_argmax}")
        print(f"Match: {match}")

        if max_diff < 1e-5:
            print("\n✓ PASS: Logits are numerically identical")
        elif match:
            print("\n⚠ WARNING: Logits differ but argmax matches")
        else:
            print("\n✗ FAIL: Logits differ significantly!")

        # Test with batch
        print("\n" + "="*60)
        print("TEST: Batch of 2 sequences")
        print("="*60)

        texts = ["Hello, how are you?", "What is your name?"]
        batch_inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda:0")

        outputs_batch = model(**batch_inputs, output_hidden_states=True)

        native_logits_batch = outputs_batch.logits[:, -1, :]
        last_hidden_batch = outputs_batch.hidden_states[-1][:, -1, :]
        manual_logits_batch = model.lm_head(last_hidden_batch)

        diff_batch = (native_logits_batch - manual_logits_batch).abs()
        max_diff_batch = diff_batch.max().item()
        mean_diff_batch = diff_batch.mean().item()

        native_argmax_batch = native_logits_batch.argmax(dim=-1).tolist()
        manual_argmax_batch = manual_logits_batch.argmax(dim=-1).tolist()
        match_batch = (native_argmax_batch == manual_argmax_batch)

        print(f"\nMax difference:  {max_diff_batch:.10f}")
        print(f"Mean difference: {mean_diff_batch:.10f}")
        print(f"\nNative argmax: {native_argmax_batch}")
        print(f"Manual argmax: {manual_argmax_batch}")
        print(f"Match: {match_batch}")

        if max_diff_batch < 1e-5:
            print("\n✓ PASS: Batch logits are numerically identical")
        elif match_batch:
            print("\n⚠ WARNING: Batch logits differ but argmax matches")
        else:
            print("\n✗ FAIL: Batch logits differ significantly!")

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    if max_diff < 1e-5 and max_diff_batch < 1e-5:
        print("✓ lm_head(hidden_states[-1]) == outputs.logits")
        print("  Problem is NOT in logits computation path")
    else:
        print("✗ lm_head(hidden_states[-1]) ≠ outputs.logits")
        print("  This explains why CSA results differ!")

if __name__ == "__main__":
    test_logits_consistency()
