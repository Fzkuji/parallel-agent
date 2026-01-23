"""Shared inference utilities for LLM-based QA systems."""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from .generators import DependencyGraphGenerator, HeuristicDependencyGenerator
from .models import EdgeCandidate, Question
from .templates import (
    USE_THINK_TOKENS,
    set_think_tokens,
    build_chat_prompt,
    build_chat_prompt_from_messages,
    DEFAULT_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .api_client import APIClient


BOX_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def extract_box_answer(text: str) -> Tuple[str, bool]:
    """Return the first <answer>...</answer> content if present; otherwise fallback to raw text.

    Returns:
        Tuple of (answer, valid) where valid=True if <answer> tags were found.
    """
    match = BOX_PATTERN.search(text)
    if match:
        return match.group(1).strip(), True
    return text.strip(), False


def extract_option_letter(text: str) -> Tuple[str, bool]:
    """Extract first option letter (A-F) from text for multiple-choice questions.

    Strategy:
    1. Find the first uppercase A-F letter in the text
    2. This handles formats like "A", "A.", "A、", "答案是A", "<answer>D</answer>"

    Returns:
        Tuple of (letter, found) where:
        - letter: The first A-F uppercase letter found, or empty string if none
        - found: True if a letter was found, False otherwise
    """
    # Find the first uppercase A-F letter
    for char in text:
        if char in "ABCDEF":
            return char, True

    # Fallback: check if there's a lowercase a-f (model might output lowercase)
    for char in text:
        if char in "abcdef":
            return char.upper(), True

    return "", False


def extract_answer(text: str, dataset: str = None) -> Tuple[str, bool]:
    """Extract answer from text based on dataset type.

    For multiple-choice datasets, extracts the first option letter.
    For other datasets, looks for <answer>...</answer> tags.
    """
    if dataset in {"cmb_exam", "mmlu"}:
        return extract_option_letter(text)
    return extract_box_answer(text)


def extract_json_from_text(text: str) -> dict:
    """Extract JSON from LLM output, handling markdown code blocks and thinking tokens."""
    cleaned = text.strip()

    # Remove thinking tokens first (handles both <think></think> and <think>...</think>)
    # This handles multi-line thinking blocks
    while True:
        think_start = cleaned.find("<think>")
        if think_start == -1:
            break
        think_end = cleaned.find("</think>", think_start)
        if think_end == -1:
            # Unclosed think tag, remove from start to end
            cleaned = cleaned[:think_start] + cleaned[think_start + 7 :]
            break
        # Remove the entire <think>...</think> block
        cleaned = cleaned[:think_start] + cleaned[think_end + 8 :]

    cleaned = cleaned.strip()

    # Handle markdown code blocks
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                cleaned = part[4:].strip()
                break
            elif part and not part.startswith("```"):
                # Try the first non-empty block that doesn't start with ```
                cleaned = part.strip()
                break

    def try_parse(candidate: str) -> Optional[dict]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    parsed = try_parse(cleaned)
    if parsed is not None:
        return parsed

    # attempt to find first balanced JSON object within text
    start_positions = [idx for idx, ch in enumerate(cleaned) if ch == "{"]
    for start in start_positions:
        depth = 0
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : idx + 1]
                    parsed = try_parse(candidate)
                    if parsed is not None:
                        return parsed
                    break
    raise ValueError(f"Failed to parse JSON: {cleaned}")


def build_dependency_prompt(background: str, questions: List[Question]) -> str:
    """Build prompt for LLM-based question ordering.

    The prompt asks the model to output the optimal order to answer questions,
    considering dependencies between them.
    """
    question_lines = [f"{q.qid}: {q.text.strip()}" for q in questions]
    prompt = textwrap.dedent(
        f"""
        给定背景文本和若干问题，请确定最佳的回答顺序。某些问题的答案可能对回答其他问题有帮助。

        **重要：直接输出JSON，不要解释。**

        输出格式：
        ```json
        {{"order": ["Q1", "Q3", "Q2", "Q4"]}}
        ```

        规则：
        - 输出所有问题ID，按推荐的回答顺序排列
        - 如果问题A的答案对问题B有帮助，则A应排在B之前
        - 如果问题之间无依赖，保持原顺序

        背景：
        {background.strip()}

        问题：
        {os.linesep.join(question_lines)}
        """
    ).strip()
    return prompt


class LocalLLMDependencyGenerator(DependencyGraphGenerator):
    """Use a local LLM model to infer dependency edges."""

    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        model: "AutoModelForCausalLM",
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.last_metrics: Dict[str, float] = {"prompt_tokens": 0, "generated_tokens": 0, "latency": 0.0}

    def generate_edges(
        self,
        background: str,
        questions: List[Question],
        metadata: Optional[dict] = None,
    ) -> List[EdgeCandidate]:
        import torch

        prompt = build_dependency_prompt(background, questions)
        # Disable thinking mode for dependency generation - we need direct JSON output
        chat_prompt = build_chat_prompt(self.tokenizer, prompt, system_prompt=PLANNER_SYSTEM_PROMPT, enable_thinking=False)
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        start = time.perf_counter()
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
        elapsed = time.perf_counter() - start
        sequences = generated.sequences
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id
        tail_tokens: List[int] = []
        for token in sequences[0].tolist()[prompt_tokens:]:
            if token in (eos_id, pad_id):
                break
            tail_tokens.append(token)
        raw_text = self.tokenizer.decode(tail_tokens, skip_special_tokens=True).strip()
        text = raw_text
        gen_tokens = len(tail_tokens)
        self.last_metrics = {
            "prompt_tokens": float(prompt_tokens),
            "generated_tokens": float(gen_tokens),
            "latency": float(elapsed),
            "dag_prompt": chat_prompt,
            "dag_raw_response": raw_text,
        }
        try:
            payload = extract_json_from_text(text)
        except ValueError as exc:
            snippet = str(exc)
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            logging.warning(
                "LLM ordering failed to produce valid JSON (%s); using original order.",
                snippet,
            )
            # Return empty edges - will use original order
            return []

        # Parse order list and convert to edges
        order = payload.get("order", [])
        if not order or not isinstance(order, list):
            # No ordering provided, use original order
            return []

        # Convert order to edges: each question depends on the one before it
        # e.g., order=["Q1", "Q3", "Q2"] -> edges: Q3 depends on Q1, Q2 depends on Q3
        edges: List[EdgeCandidate] = []
        for i in range(1, len(order)):
            source = order[i - 1]  # Previous question in order
            target = order[i]      # Current question depends on previous
            if isinstance(source, str) and isinstance(target, str):
                edges.append(EdgeCandidate(source=source, target=target, confidence=1.0))
        return edges


# =============================================================================
# Unified Generation Interface (supports both local models and API)
# =============================================================================

@dataclass
class GenerationResult:
    """Result from a generation call."""
    text: str
    prompt_tokens: int
    generated_tokens: int
    latency: float


def generate_with_local_model(
    prompt: str,
    tokenizer: "AutoTokenizer",
    model: "AutoModelForCausalLM",
    *,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 0.0,
) -> GenerationResult:
    """Generate text using a local HuggingFace model.

    Args:
        prompt: The full prompt string (already formatted)
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling
        temperature: Sampling temperature

    Returns:
        GenerationResult with generated text and metrics
    """
    import torch
    from .utils import reset_generation_seed, DEFAULT_GENERATION_SEED, clean_model_text

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[-1]

    start = time.perf_counter()
    reset_generation_seed(DEFAULT_GENERATION_SEED)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )

    elapsed = time.perf_counter() - start
    sequences = generated.sequences

    # Extract generated tokens (excluding prompt)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    tail_tokens: List[int] = []
    for token in sequences[0].tolist()[prompt_tokens:]:
        if token in (eos_id, pad_id):
            break
        tail_tokens.append(token)

    raw_text = tokenizer.decode(tail_tokens, skip_special_tokens=True).strip()
    raw_text = clean_model_text(raw_text)

    return GenerationResult(
        text=raw_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=len(tail_tokens),
        latency=elapsed,
    )


def generate_with_api(
    messages: List[Dict[str, str]],
    api_client: "APIClient",
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> GenerationResult:
    """Generate text using an API client.

    Args:
        messages: List of message dicts with 'role' and 'content'
        api_client: APIClient instance
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        GenerationResult with generated text and metrics
    """
    response = api_client.generate(
        messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    return GenerationResult(
        text=response.text,
        prompt_tokens=response.prompt_tokens,
        generated_tokens=response.completion_tokens,
        latency=response.latency,
    )


def generate_completion(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    model: Any,
    *,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 0.0,
    api_client: Optional["APIClient"] = None,
) -> GenerationResult:
    """Unified generation interface supporting both local models and API.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: HuggingFace tokenizer (ignored if api_client is provided)
        model: HuggingFace model (ignored if api_client is provided)
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling (local model only)
        temperature: Sampling temperature
        api_client: Optional APIClient for API-based inference

    Returns:
        GenerationResult with generated text and metrics
    """
    if api_client is not None:
        return generate_with_api(
            messages,
            api_client,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    else:
        # Build prompt from messages using tokenizer's chat template
        prompt = build_chat_prompt_from_messages(tokenizer, messages)
        return generate_with_local_model(
            prompt,
            tokenizer,
            model,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )


class APILLMDependencyGenerator(DependencyGraphGenerator):
    """Generate dependencies using an API-based LLM."""

    def __init__(
        self,
        api_client: "APIClient",
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.api_client = api_client
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.last_metrics: Dict[str, float] = {
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "latency": 0.0,
        }

    def generate_edges(
        self,
        background: str,
        questions: List[Question],
        metadata: Optional[dict] = None,
    ) -> List[EdgeCandidate]:
        prompt = build_dependency_prompt(background, questions)

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        result = generate_with_api(
            messages,
            self.api_client,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        self.last_metrics = {
            "prompt_tokens": float(result.prompt_tokens),
            "generated_tokens": float(result.generated_tokens),
            "latency": float(result.latency),
            "dag_prompt": prompt,
            "dag_raw_response": result.text,
        }

        try:
            payload = extract_json_from_text(result.text)
        except ValueError as exc:
            snippet = str(exc)
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            logging.warning(
                "API ordering failed to parse JSON (%s); using original order.",
                snippet,
            )
            # Return empty edges - will use original order
            return []

        # Parse order list and convert to edges
        order = payload.get("order", [])
        if not order or not isinstance(order, list):
            # No ordering provided, use original order
            return []

        # Convert order to edges: each question depends on the one before it
        # e.g., order=["Q1", "Q3", "Q2"] -> edges: Q3 depends on Q1, Q2 depends on Q3
        edges: List[EdgeCandidate] = []
        for i in range(1, len(order)):
            source = order[i - 1]  # Previous question in order
            target = order[i]      # Current question depends on previous
            if isinstance(source, str) and isinstance(target, str):
                edges.append(EdgeCandidate(source=source, target=target, confidence=1.0))
        return edges
