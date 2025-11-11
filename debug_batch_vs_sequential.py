#!/usr/bin/env python3
"""
Utility script to compare per-sample and batched generation outputs.

Usage:
    python debug_batch_vs_sequential.py \
        --model-name sshleifer/tiny-gpt2 \
        --max-new-tokens 32 \
        --prompt "Question 1?" \
        --prompt "Another question?"
"""

from __future__ import annotations

import argparse
import random
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def trim_generation(
    tokenizer: AutoTokenizer,
    sequences: torch.Tensor,
    prompt_lengths: List[int],
) -> List[str]:
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    outputs: List[str] = []
    for idx, seq in enumerate(sequences):
        tokens = []
        for token in seq[int(prompt_lengths[idx]) :].tolist():
            if token in (eos_id, pad_id):
                break
            tokens.append(token)
        outputs.append(tokenizer.decode(tokens, skip_special_tokens=True).strip())
    return outputs


def generate_per_sample(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompts: List[str],
    *,
    max_new_tokens: int,
) -> List[str]:
    responses: List[str] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            sequences = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            ).sequences
        responses.extend(trim_generation(tokenizer, sequences, [prompt_len]))
    return responses


def generate_batched(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompts: List[str],
    *,
    max_new_tokens: int,
) -> List[str]:
    original_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    with torch.no_grad():
        sequences = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        ).sequences
    tokenizer.padding_side = original_padding
    return trim_generation(tokenizer, sequences, prompt_lengths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare per-sample vs batch generation outputs.")
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Custom prompt(s). Provide multiple times to add more. Defaults to built-in samples.",
    )
    args = parser.parse_args()

    prompts = args.prompts or [
        "Q1: Where was the treaty signed?\nBackground: The treaty was signed in Paris.",
        "Q2: Who signed the treaty?\nBackground: Representatives from both nations signed the treaty.",
        "Q3: When was the treaty signed?\nBackground: It happened in 1895.",
    ]

    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).eval()

    single_outputs = generate_per_sample(tokenizer, model, prompts, max_new_tokens=args.max_new_tokens)
    batched_outputs = generate_batched(tokenizer, model, prompts, max_new_tokens=args.max_new_tokens)

    all_match = True
    for idx, (single, batched) in enumerate(zip(single_outputs, batched_outputs), 1):
        match = single == batched
        all_match = all_match and match
        status = "✓ MATCH" if match else "✗ DIFF"
        print(f"\nPrompt #{idx}: {status}")
        print(f"Per-sample: {single or '<empty>'}")
        print(f"Batched   : {batched or '<empty>'}")

    if all_match:
        print("\nAll outputs are identical.")
    else:
        print("\nOutputs differ. Adjust prompts/model/seed or inspect above diffs.")


if __name__ == "__main__":
    main()
