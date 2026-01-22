"""Cross-batch generation strategy for multi-question answering."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.models import Question, StrategyResult
from src.inference import extract_answer, build_chat_prompt
from src.evaluation import evaluate_predictions
from src.prompts import build_single_prompt
from src.utils import clean_model_text

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.cross_batch import CrossBatchGenerator


def run_cross_batch_strategy(
    background: str,
    questions: List[Question],
    tokenizer: "AutoTokenizer",
    model: "AutoModelForCausalLM",
    *,
    max_new_tokens: int,
    strategy_name: str = "collab_hidden",
    dataset: str = None,
    cross_batch_generator: Optional["CrossBatchGenerator"] = None,
    mix_method: str = "attention",
    mix_layer: int = -1,
    mix_layers: Optional[List[int]] = None,
    checkpoint_path: Optional[str] = None,
    enable_cross_batch: bool = True,
    cross_batch_module: Optional["torch.nn.Module"] = None,
) -> StrategyResult:
    """
    Run cross-batch generation strategy.

    This strategy generates answers for all questions in parallel, with
    cross-batch interaction enabled during token generation. The hidden
    states from different samples influence each other, potentially
    improving coherence for related questions.

    Args:
        background: Shared background/context text
        questions: List of Question objects
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        max_new_tokens: Maximum tokens to generate per question
        dataset: Dataset name for formatting
        cross_batch_generator: Pre-initialized CrossBatchGenerator (optional)
        mix_method: Cross-batch mixing method ("attention", "mixer", "simple", "multi_layer")
        mix_layer: Which layer's hidden state to mix (-1 for last)
        mix_layers: List of layer indices for multi_layer mode
        checkpoint_path: Path to trained cross-batch module checkpoint
        enable_cross_batch: Whether to enable cross-batch interaction
        cross_batch_module: Pre-initialized cross-batch module (optional, overrides checkpoint)

    Returns:
        StrategyResult with answers and metrics
    """
    import torch
    from src.cross_batch import (
        CrossBatchGenerator, CrossBatchAttention, CrossBatchEmbeddingMixer,
        SimpleCrossBatchGate, MultiLayerCrossBatch,
    )

    question_lookup = {q.qid: q for q in questions}
    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    per_question: List[Dict[str, Any]] = []

    # Initialize cross-batch generator if not provided
    if cross_batch_generator is None:
        # Get device - handle distributed models
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            # Distributed model - use lm_head device
            if hasattr(model, 'lm_head'):
                device = str(next(model.lm_head.parameters()).device)
            else:
                device = "cuda"
        else:
            device = str(model.device)
        hidden_size = model.config.hidden_size

        # Use provided cross_batch_module or create new one
        if cross_batch_module is None:
            num_layers = model.config.num_hidden_layers

            # Load checkpoint if provided (to get config)
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                config = checkpoint.get("config", {})
                # Use checkpoint config if available
                if "module_type" in config:
                    mix_method = config["module_type"]
                if "mix_layer" in config:
                    mix_layer = config["mix_layer"]
                if "mix_layers" in config and config["mix_layers"]:
                    mix_layers = config["mix_layers"]

            # Create cross-batch module based on method
            if mix_method == "multi_layer":
                layer_indices = mix_layers if mix_layers else list(range(num_layers // 2, num_layers))
                cross_batch_module = MultiLayerCrossBatch(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    layer_indices=layer_indices,
                )
            elif mix_method == "simple":
                cross_batch_module = SimpleCrossBatchGate(hidden_size=hidden_size)
            elif mix_method == "attention":
                cross_batch_module = CrossBatchAttention(hidden_size=hidden_size)
            else:  # mixer
                cross_batch_module = CrossBatchEmbeddingMixer(hidden_size=hidden_size)

            # Load checkpoint weights
            if checkpoint_path:
                cross_batch_module.load_state_dict(checkpoint["cross_batch_module"])
                if "lm_head" in checkpoint and hasattr(model, 'lm_head'):
                    model.lm_head.load_state_dict(checkpoint["lm_head"])

        # Determine effective mix_layer for generator
        effective_mix_layer = mix_layers if mix_layers else mix_layer

        cross_batch_generator = CrossBatchGenerator(
            model=model,
            tokenizer=tokenizer,
            cross_batch_module=cross_batch_module,
            mix_method=mix_method,
            mix_layer=effective_mix_layer,
            device=device,
        )

    # Build prompts for all questions (using chat template like other strategies)
    batch_prompts: List[str] = []
    for question in questions:
        system_prompt, user_prompt = build_single_prompt(background, question, dataset)
        full_prompt = build_chat_prompt(tokenizer, user_prompt, system_prompt=system_prompt)
        batch_prompts.append(full_prompt)

    # Tokenize all prompts with left padding
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    tokenizer.padding_side = original_padding_side

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    input_lengths = attention_mask.sum(dim=1).tolist()
    prompt_window = input_ids.shape[-1]

    # Generate with cross-batch interaction
    start = time.perf_counter()
    outputs = cross_batch_generator.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        enable_cross_batch=enable_cross_batch,
    )
    total_latency = time.perf_counter() - start

    # Process outputs
    sequences = outputs["sequences"]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    raw_texts = []
    boxes = []
    generated_token_counts = []

    for seq in sequences:
        tokens = []
        for token in seq[int(prompt_window):].tolist():
            if token in (eos_id, pad_id):
                break
            tokens.append(token)
        raw_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        raw_text = clean_model_text(raw_text)
        raw_texts.append(raw_text)
        box = extract_answer(raw_text, dataset)
        boxes.append(box)
        generated_token_counts.append(len(tokens))

    # Build results
    total_prompt_tokens = sum(int(length) for length in input_lengths)
    total_generated_tokens = sum(generated_token_counts)

    for idx, question in enumerate(questions):
        final_answer, strict_valid = boxes[idx]
        answer_records[question.qid] = (final_answer, strict_valid)
        answers_text[question.qid] = final_answer
        per_question.append(
            {
                "question_id": question.qid,
                "question": question.text.strip(),
                "gold_answers": question.references,
                "prompt": batch_prompts[idx],
                "raw_response": raw_texts[idx],
                "final_answer": final_answer,
                "strict_valid": strict_valid,
                "latency": total_latency / len(questions),
                "prompt_tokens": int(input_lengths[idx]),
                "generated_tokens": generated_token_counts[idx],
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
    return StrategyResult(
        name=strategy_name,
        answers=answers_text,
        prompt_tokens=int(total_prompt_tokens),
        generated_tokens=int(total_generated_tokens),
        latency=total_latency,
        batches=1,
        metrics=metrics,
        details={
            "questions": per_question,
            "cross_batch_enabled": enable_cross_batch,
            "mix_method": mix_method,
            "mix_layer": mix_layer,
        },
    )


def run_cross_batch_multi_strategy(
    items: List[Dict[str, Any]],
    tokenizer: "AutoTokenizer",
    model: "AutoModelForCausalLM",
    *,
    max_new_tokens: int,
    strategy_name: str = "collab_hidden",
    dataset: str = None,
    cross_batch_generator: Optional["CrossBatchGenerator"] = None,
    mix_method: str = "attention",
    mix_layer: int = -1,
    mix_layers: Optional[List[int]] = None,
    checkpoint_path: Optional[str] = None,
    enable_cross_batch: bool = True,
    cross_batch_module: Optional["torch.nn.Module"] = None,
) -> StrategyResult:
    """
    Run cross-batch generation strategy for multi-context items.

    Each item has its own context, but cross-batch interaction allows
    information sharing between samples during generation.

    Args:
        items: List of dicts with 'qid', 'question', 'context', 'references'
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        max_new_tokens: Maximum tokens to generate per question
        strategy_name: Name for the strategy result
        dataset: Dataset name for formatting
        cross_batch_generator: Pre-initialized CrossBatchGenerator (optional)
        mix_method: Cross-batch mixing method ("attention", "mixer", "simple", "multi_layer")
        mix_layer: Which layer's hidden state to mix (-1 for last)
        mix_layers: List of layer indices for multi_layer mode
        checkpoint_path: Path to trained cross-batch module checkpoint
        enable_cross_batch: Whether to enable cross-batch interaction
        cross_batch_module: Pre-initialized cross-batch module (optional, overrides checkpoint)

    Returns:
        StrategyResult with answers and metrics
    """
    import torch
    from src.cross_batch import (
        CrossBatchGenerator, CrossBatchAttention, CrossBatchEmbeddingMixer,
        SimpleCrossBatchGate, MultiLayerCrossBatch,
    )

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
    answer_records: Dict[str, Tuple[str, bool]] = {}
    answers_text: Dict[str, str] = {}
    per_question: List[Dict[str, Any]] = []

    # Initialize cross-batch generator if not provided
    if cross_batch_generator is None:
        # Get device - handle distributed models
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            # Distributed model - use lm_head device
            if hasattr(model, 'lm_head'):
                device = str(next(model.lm_head.parameters()).device)
            else:
                device = "cuda"
        else:
            device = str(model.device)
        hidden_size = model.config.hidden_size

        # Use provided cross_batch_module or create new one
        if cross_batch_module is None:
            num_layers = model.config.num_hidden_layers

            # Load checkpoint if provided (to get config)
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                config = checkpoint.get("config", {})
                # Use checkpoint config if available
                if "module_type" in config:
                    mix_method = config["module_type"]
                if "mix_layer" in config:
                    mix_layer = config["mix_layer"]
                if "mix_layers" in config and config["mix_layers"]:
                    mix_layers = config["mix_layers"]

            # Create cross-batch module based on method
            if mix_method == "multi_layer":
                layer_indices = mix_layers if mix_layers else list(range(num_layers // 2, num_layers))
                cross_batch_module = MultiLayerCrossBatch(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    layer_indices=layer_indices,
                )
            elif mix_method == "simple":
                cross_batch_module = SimpleCrossBatchGate(hidden_size=hidden_size)
            elif mix_method == "attention":
                cross_batch_module = CrossBatchAttention(hidden_size=hidden_size)
            else:  # mixer
                cross_batch_module = CrossBatchEmbeddingMixer(hidden_size=hidden_size)

            # Load checkpoint weights
            if checkpoint_path:
                cross_batch_module.load_state_dict(checkpoint["cross_batch_module"])
                if "lm_head" in checkpoint and hasattr(model, 'lm_head'):
                    model.lm_head.load_state_dict(checkpoint["lm_head"])

        # Determine effective mix_layer for generator
        effective_mix_layer = mix_layers if mix_layers else mix_layer

        cross_batch_generator = CrossBatchGenerator(
            model=model,
            tokenizer=tokenizer,
            cross_batch_module=cross_batch_module,
            mix_method=mix_method,
            mix_layer=effective_mix_layer,
            device=device,
        )

    # Build prompts for all items (using chat template like other strategies)
    batch_prompts: List[str] = []
    for item in items:
        q = question_lookup[item["qid"]]
        system_prompt, user_prompt = build_single_prompt(item["context"], q, dataset)
        full_prompt = build_chat_prompt(tokenizer, user_prompt, system_prompt=system_prompt)
        batch_prompts.append(full_prompt)

    # Tokenize all prompts with left padding
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    tokenizer.padding_side = original_padding_side

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    input_lengths = attention_mask.sum(dim=1).tolist()
    prompt_window = input_ids.shape[-1]

    # Generate with cross-batch interaction
    start = time.perf_counter()
    outputs = cross_batch_generator.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        enable_cross_batch=enable_cross_batch,
    )
    total_latency = time.perf_counter() - start

    # Process outputs
    sequences = outputs["sequences"]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    raw_texts = []
    boxes = []
    generated_token_counts = []

    for seq in sequences:
        tokens = []
        for token in seq[int(prompt_window):].tolist():
            if token in (eos_id, pad_id):
                break
            tokens.append(token)
        raw_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        raw_text = clean_model_text(raw_text)
        raw_texts.append(raw_text)
        box = extract_answer(raw_text, dataset)
        boxes.append(box)
        generated_token_counts.append(len(tokens))

    # Build results
    total_prompt_tokens = sum(int(length) for length in input_lengths)
    total_generated_tokens = sum(generated_token_counts)

    for idx, item in enumerate(items):
        qid = item["qid"]
        final_answer, strict_valid = boxes[idx]
        answer_records[qid] = (final_answer, strict_valid)
        answers_text[qid] = final_answer
        per_question.append(
            {
                "question_id": qid,
                "question": item["question"],
                "gold_answers": item.get("references", []),
                "prompt": batch_prompts[idx],
                "raw_response": raw_texts[idx],
                "final_answer": final_answer,
                "strict_valid": strict_valid,
                "latency": total_latency / len(items),
                "prompt_tokens": int(input_lengths[idx]),
                "generated_tokens": generated_token_counts[idx],
            }
        )

    metrics = evaluate_predictions(answer_records, question_lookup, dataset=dataset)
    return StrategyResult(
        name=strategy_name,
        answers=answers_text,
        prompt_tokens=int(total_prompt_tokens),
        generated_tokens=int(total_generated_tokens),
        latency=total_latency,
        batches=1,
        metrics=metrics,
        details={
            "questions": per_question,
            "cross_batch_enabled": enable_cross_batch,
            "mix_method": mix_method,
            "mix_layer": mix_layer,
        },
    )
