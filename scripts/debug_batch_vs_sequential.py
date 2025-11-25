#!/usr/bin/env python3
"""
Utility script to compare sequential (per-sample) vs batched generation outputs
for the exact same set of prompts, and verify they are identical.

支持两种提示模式：
- raw：直接把传入的字符串作为提示（与普通 `tokenizer(prompt)` 一致）
- chat：使用 src.inference.build_chat_prompt 封装 system/user 结构，和主流程一致

示例：
  python debug_batch_vs_sequential.py \
    --model-name sshleifer/tiny-gpt2 \
    --mode chat \
    --background "..." \
    --question "Q1: ...?" --question "Q2: ...?" \
    --max-new-tokens 64 --seed 13

或直接传 raw 提示：
  python debug_batch_vs_sequential.py \
    --model-name sshleifer/tiny-gpt2 \
    --mode raw \
    --prompt "Q1 ..." --prompt "Q2 ..."
"""

from __future__ import annotations

import argparse
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference import build_chat_prompt, set_think_tokens


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
                num_beams=1,
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
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        ).sequences
    tokenizer.padding_side = original_padding
    return trim_generation(tokenizer, sequences, prompt_lengths)


def build_prompts(
    mode: str,
    tokenizer: AutoTokenizer,
    *,
    prompts: Optional[List[str]] = None,
    background: str = "",
    questions: Optional[List[str]] = None,
    system: Optional[str] = None,
    use_think_tokens: bool = False,
) -> List[str]:
    """Return final prompts ready for tokenization.

    - raw: returns `prompts` as-is
    - chat: builds chat-style prompts consistent with pipeline
    """
    set_think_tokens(use_think_tokens)
    system_msg = (system or "You are a helpful assistant that answers questions given background passages.").strip()

    if mode == "raw":
        return prompts or []

    if mode == "chat":
        out: List[str] = []
        qs = questions or []
        for q in qs:
            user_prompt = f"Background:\n{background.strip()}\n\n{q.strip()}"
            out.append(build_chat_prompt(tokenizer, user_prompt, system_prompt=system_msg))
        if prompts:  # also support direct prompts in chat mode
            for p in prompts:
                out.append(build_chat_prompt(tokenizer, p, system_prompt=system_msg))
        return out

    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare per-sample vs batch generation outputs.")
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--mode", choices=["raw", "chat"], default="chat")
    parser.add_argument("--system", type=str, default=None, help="System message for chat mode.")
    parser.add_argument("--background", type=str, default="", help="Background text for chat mode.")
    parser.add_argument("--question", action="append", dest="questions", help="Question(s) for chat mode.")
    parser.add_argument("--prompt", action="append", dest="prompts", help="Raw prompt(s) or extra prompts in chat mode.")
    parser.add_argument("--use-think-tokens", action="store_true")
    parser.add_argument("--strict-exit", action="store_true", help="Exit 1 if any mismatch.")
    args = parser.parse_args()

    # Defaults if nothing provided
    default_questions = [
        "Q1: Where was the treaty signed?",
        "Q2: Who signed the treaty?",
        "Q3: When was the treaty signed?",
    ]
    default_background = "The treaty was signed in Paris in 1895 by representatives from both nations."

    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).eval()

    # Build final prompts list
    prompts_list = build_prompts(
        args.mode,
        tokenizer,
        prompts=args.prompts,
        background=args.background or default_background,
        questions=args.questions or default_questions,
        system=args.system,
        use_think_tokens=args.use_think_tokens,
    )

    single_outputs = generate_per_sample(tokenizer, model, prompts_list, max_new_tokens=args.max_new_tokens)
    batched_outputs = generate_batched(tokenizer, model, prompts_list, max_new_tokens=args.max_new_tokens)

    all_match = True
    for idx, (single, batched, ptext) in enumerate(zip(single_outputs, batched_outputs, prompts_list), 1):
        match = single == batched
        all_match = all_match and match
        status = "✓ MATCH" if match else "✗ DIFF"
        print(f"\nPrompt #{idx}: {status}")
        print(f"Prompt text: {ptext[:160]}{'...' if len(ptext)>160 else ''}")
        print(f"Per-sample: {single or '<empty>'}")
        print(f"Batched   : {batched or '<empty>'}")

    if all_match:
        print("\nAll outputs are identical.")
    else:
        print("\nOutputs differ. Adjust prompts/model/seed or inspect above diffs.")
        if args.strict_exit:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
