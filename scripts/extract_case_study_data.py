#!/usr/bin/env python3
"""
Extract case study data from trained CSA model and save to JSON.
Run this on the server with GPU, then download the JSON for local plotting.

Usage:
    python scripts/extract_case_study_data.py \
        --checkpoint outputs/checkpoints/squad/Qwen_Qwen2.5-7B-Instruct_attention_frozen_lora.pt \
        --output case_study_data.json \
        --num-questions 5 \
        --context-index 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Extract case study data from CSA model")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name or path")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to CSA checkpoint")
    parser.add_argument("--output", type=str, default="case_study_data.json",
                       help="Output JSON file path")
    parser.add_argument("--num-questions", type=int, default=5,
                       help="Number of questions per context")
    parser.add_argument("--context-index", type=int, default=0,
                       help="Which context to use (0 = first)")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                       help="Maximum tokens to generate per answer")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for dataset loading")

    return parser.parse_args()


def load_model_and_csa(model_name: str, checkpoint_path: str, device: str):
    """Load model, tokenizer, and CSA module from checkpoint."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.cross_batch import CrossBatchGenerator, CrossBatchAttention

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    # Create CSA module
    hidden_size = model.config.hidden_size
    use_gate = config.get("use_gate", True)

    csa_module = CrossBatchAttention(
        hidden_size=hidden_size,
        use_gate=use_gate,
        num_heads=8,
    )
    csa_module.load_state_dict(checkpoint["cross_batch_module"])
    csa_module.to(device)
    csa_module.eval()

    # Create generator
    generator = CrossBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=csa_module,
        mix_method=config.get("module_type", "attention"),
        mix_layer=config.get("mix_layer", -1),
        device=device,
    )

    print(f"CSA config: use_gate={use_gate}")

    return model, tokenizer, csa_module, generator


def load_dataset_sample(num_questions: int, context_index: int, seed: int):
    """Load a sample from SQuAD dataset."""
    from src.datasets.squad import load_squad_groups

    print(f"Loading SQuAD dataset (seed={seed})...")
    groups = load_squad_groups(
        split="validation",
        min_questions=num_questions,
        max_questions=num_questions,
        max_contexts=context_index + 10,  # Load a few extra in case some don't have enough questions
        seed=seed,
        fixed_question_count=num_questions,
    )

    if context_index >= len(groups):
        print(f"Warning: context_index {context_index} >= available contexts {len(groups)}, using 0")
        context_index = 0

    sample = groups[context_index]
    return sample


def extract_attention_weights(csa_module, hidden_states: torch.Tensor) -> np.ndarray:
    """Extract attention weights from CSA module given hidden states."""
    batch_size = hidden_states.size(0)

    if batch_size <= 1:
        return np.zeros((batch_size, batch_size))

    with torch.no_grad():
        # Compute Q, K
        q = csa_module.q_proj(hidden_states).view(batch_size, csa_module.num_heads, csa_module.head_dim)
        k = csa_module.k_proj(hidden_states).view(batch_size, csa_module.num_heads, csa_module.head_dim)

        q = q.permute(1, 0, 2)  # [num_heads, batch, head_dim]
        k = k.permute(1, 0, 2)

        # Attention weights: [num_heads, batch, batch]
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (csa_module.head_dim ** 0.5 * csa_module.temperature)

        # Mask out self (diagonal)
        eye_mask = torch.eye(batch_size, device=hidden_states.device, dtype=torch.bool)
        attn_weights = attn_weights.masked_fill(eye_mask.unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Average across heads
        avg_attn = attn_weights.mean(dim=0).cpu().numpy()

    return avg_attn


def generate_and_capture_attention(
    model,
    tokenizer,
    csa_module,
    generator,
    context: str,
    questions: list,
    max_new_tokens: int,
    device: str,
):
    """Generate answers and capture attention weights."""
    from src.strategies.batch import build_batch_prompt

    # Storage for attention weights
    captured_attention = []

    def attention_hook(module, input, output):
        """Capture attention during generation."""
        hidden_states = input[0]
        batch_size = hidden_states.size(0)
        if batch_size > 1:
            attn = extract_attention_weights(module, hidden_states)
            captured_attention.append(attn)

    # Register hook
    hook_handle = csa_module.register_forward_hook(attention_hook)

    try:
        # Build prompts
        prompts = []
        for q in questions:
            prompt = build_batch_prompt(context, q["text"])
            prompts.append(prompt)

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate with CSA
        output_ids = generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode answers
        answers = []
        for i, (in_ids, out_ids) in enumerate(zip(input_ids, output_ids)):
            # Get generated part only
            generated_ids = out_ids[len(in_ids):]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
            answers.append(answer.strip())

    finally:
        hook_handle.remove()

    # Average attention across all generation steps
    if captured_attention:
        avg_attention = np.mean(captured_attention, axis=0)
    else:
        n = len(questions)
        avg_attention = np.zeros((n, n))

    return answers, avg_attention


def main():
    args = parse_args()

    # Load model and CSA
    model, tokenizer, csa_module, generator = load_model_and_csa(
        args.model, args.checkpoint, args.device
    )

    # Load dataset sample
    sample = load_dataset_sample(
        args.num_questions, args.context_index, args.seed
    )

    context = sample["context"]
    questions = sample["questions"]

    print(f"\nContext ({len(context)} chars): {context[:200]}...")
    print(f"\nQuestions ({len(questions)}):")
    for i, q in enumerate(questions):
        print(f"  {i+1}. {q['text']}")
        print(f"      Ref: {q['references']}")

    # Generate and capture attention
    print("\nGenerating answers...")
    answers, attention_matrix = generate_and_capture_attention(
        model, tokenizer, csa_module, generator,
        context, questions,
        args.max_new_tokens, args.device,
    )

    print(f"\nGenerated answers:")
    for i, a in enumerate(answers):
        print(f"  {i+1}. {a}")

    # Prepare output data
    output_data = {
        "context": context,
        "questions": [
            {
                "question": q["text"],
                "references": q.get("references", []),
            }
            for q in questions
        ],
        "answers": answers,
        "attention_matrix": attention_matrix.tolist(),
        "metadata": {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "num_questions": args.num_questions,
            "context_index": args.context_index,
        }
    }

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")
    print("Download this file and run: python scripts/plot_case_study.py")


if __name__ == "__main__":
    main()
