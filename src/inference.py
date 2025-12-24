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

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .api_client import APIClient


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that answers questions given background passages."
PLANNER_SYSTEM_PROMPT = (
    "You are an expert planner. Analyse the questions and output only a JSON object describing dependencies."
)

BOX_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

# Control whether to add <think></think> tags in prompts
# True: Add empty <think>\n\n</think>\n\n tags (prevents actual thinking, just provides structure)
# False: No thinking tags at all
# Note: In batch inference with left padding, Qwen3 may regenerate chat template markers
# when thinking tokens are present. Setting to False for batch compatibility.
USE_THINK_TOKENS = False


def set_think_tokens(enabled: bool) -> None:
    """Enable/disable adding <think></think> tags to prompts."""
    global USE_THINK_TOKENS
    USE_THINK_TOKENS = enabled


def build_chat_prompt(
    tokenizer: "AutoTokenizer",
    user_prompt: str,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Create a chat-style prompt using the tokenizer's chat template.

    Unifies assistant generation start handling across all strategies by
    relying on add_generation_prompt=True. Thinking tags are controlled via
    USE_THINK_TOKENS and the tokenizer's enable_thinking flag.
    """

    messages: List[Dict[str, str]] = []
    system = (system_prompt or "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_prompt.strip()})

    if hasattr(tokenizer, "apply_chat_template"):
        # Qwen3 quirk: enable_thinking parameter is inverted
        # - enable_thinking=False → adds <think>\n\n</think>\n\n to prompt
        # - enable_thinking=True  → no thinking tags
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=(not USE_THINK_TOKENS),
        )
    else:
        # Fallback for tokenizers without chat template support
        parts = []
        if system:
            parts.append(f"System: {system}")
        parts.append(f"User: {user_prompt.strip()}")
        parts.append("Assistant:")
        prompt = "\n\n".join(parts)
        if USE_THINK_TOKENS:
            prompt = f"{prompt}\n<think>\n\n</think>\n\n"

    return prompt


def extract_box_answer(text: str) -> Tuple[str, bool]:
    """Return the first <answer>...</answer> content if present; otherwise fallback to raw text."""
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

    For multiple-choice datasets (cmb_exam), extracts the first option letter.
    For other datasets, looks for <answer>...</answer> tags.

    Args:
        text: Raw model output
        dataset: Dataset name to determine extraction method

    Returns:
        Tuple of (answer, valid) where valid indicates clean extraction
    """
    if dataset == "cmb_exam":
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
    """Build prompt for LLM-based dependency inference.

    The prompt asks the model to output edges ordered by importance (most important first).
    The position in the list will be used to assign confidence scores.
    Uses compact format: {"source": "target"} instead of {"source": "Q1", "target": "Q2"}.
    """
    question_lines = [f"{q.qid}: {q.text.strip()}" for q in questions]
    prompt = textwrap.dedent(
        f"""
        你将看到一段背景文本以及若干针对该背景的问题。请推断在回答这些问题时是否需要引用其他问题的答案。

        **重要：你的回答必须是markdown代码块格式，以```json开头，以```结尾。**

        输出格式示例（source依赖于target，即回答source时需要先知道target的答案）：
        ```json
        {{"edges": [{{"Q3": "Q1"}}, {{"Q4": "Q2"}}]}}
        ```

        规则：
        - 格式为 {{"被依赖问题ID": "依赖问题ID"}}，表示回答前者需要后者的答案。
        - **按依赖关系的重要性从高到低排序输出**。
        - 无依赖返回空数组：{{"edges": []}}
        - 禁止循环依赖。

        背景：
        {background.strip()}

        问题：
        {os.linesep.join(question_lines)}

        请直接输出```json代码块：
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
        chat_prompt = build_chat_prompt(self.tokenizer, prompt, system_prompt=PLANNER_SYSTEM_PROMPT)
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
                "LLM dependency generation failed to produce valid JSON (%s); falling back to heuristics.",
                snippet,
            )
            heuristic = HeuristicDependencyGenerator()
            # retain the cost already incurred before fallback
            return heuristic.generate_edges(background, questions, metadata)
        edges_data = payload.get("edges", [])
        edges: List[EdgeCandidate] = []
        total_edges = len(edges_data)
        for idx, item in enumerate(edges_data):
            # Compact format: {"Q3": "Q1"} means Q3 depends on Q1 (source=Q1, target=Q3)
            if len(item) == 1:
                target, source = next(iter(item.items()))
            else:
                # Fallback to old format for compatibility
                try:
                    source = item["source"]
                    target = item["target"]
                except KeyError:
                    continue
            # Position-based confidence: first edge = 1.0, last edge = 0.5
            confidence = 1.0 - (idx / max(total_edges, 1)) * 0.5
            # Handle case where source is a list (e.g., {"Q3": ["Q1", "Q2"]})
            if isinstance(source, list):
                for src in source:
                    if isinstance(src, str) and isinstance(target, str):
                        edges.append(EdgeCandidate(source=src, target=target, confidence=confidence))
            elif isinstance(source, str) and isinstance(target, str):
                edges.append(EdgeCandidate(source=source, target=target, confidence=confidence))
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


def build_chat_prompt_from_messages(
    tokenizer: "AutoTokenizer",
    messages: List[Dict[str, str]],
) -> str:
    """Build a chat prompt string from messages using tokenizer's template.

    Args:
        tokenizer: HuggingFace tokenizer with chat template
        messages: List of message dicts

    Returns:
        Formatted prompt string
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=(not USE_THINK_TOKENS),
        )
    else:
        # Fallback for tokenizers without chat template
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)


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
                "API dependency generation failed to parse JSON (%s); falling back to heuristics.",
                snippet,
            )
            heuristic = HeuristicDependencyGenerator()
            return heuristic.generate_edges(background, questions, metadata)

        edges_data = payload.get("edges", [])
        edges: List[EdgeCandidate] = []
        total_edges = len(edges_data)
        for idx, item in enumerate(edges_data):
            # Compact format: {"Q3": "Q1"} means Q3 depends on Q1 (source=Q1, target=Q3)
            if len(item) == 1:
                target, source = next(iter(item.items()))
            else:
                # Fallback to old format for compatibility
                try:
                    source = item["source"]
                    target = item["target"]
                except KeyError:
                    continue
            # Position-based confidence: first edge = 1.0, last edge = 0.5
            confidence = 1.0 - (idx / max(total_edges, 1)) * 0.5
            # Handle case where source is a list (e.g., {"Q3": ["Q1", "Q2"]})
            if isinstance(source, list):
                for src in source:
                    if isinstance(src, str) and isinstance(target, str):
                        edges.append(EdgeCandidate(source=src, target=target, confidence=confidence))
            elif isinstance(source, str) and isinstance(target, str):
                edges.append(EdgeCandidate(source=source, target=target, confidence=confidence))

        return edges
