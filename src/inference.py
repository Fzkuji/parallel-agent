"""Shared inference utilities for LLM-based QA systems."""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .generators import DependencyGraphGenerator, HeuristicDependencyGenerator
from .models import EdgeCandidate, Question

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


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
    """Build prompt for LLM-based dependency inference."""
    question_lines = [f"{q.qid}: {q.text.strip()}" for q in questions]
    prompt = textwrap.dedent(
        f"""
        你将看到一段背景文本以及若干针对该背景的问题。请推断在回答这些问题时是否需要引用其他问题的答案。

        **重要：你的回答必须是markdown代码块格式，以```json开头，以```结尾。**

        输出格式示例：
        ```json
        {{
          "edges": [
            {{"source": "Q1", "target": "Q3", "confidence": 0.72}},
            {{"source": "Q2", "target": "Q4", "confidence": 0.85}}
          ]
        }}
        ```

        规则：
        - 只能使用给定问题的 ID 作为 source/target。
        - confidence 取值 0~1 的数字。
        - 无需依赖的题目可省略（返回空edges数组）。
        - 禁止引用不存在的 ID，禁止循环依赖。
        - 不需要额外解释或分析，只输出```json代码块。

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
        for item in edges_data:
            try:
                source = item["source"]
                target = item["target"]
            except KeyError:
                continue
            val = item.get("confidence", 0.7)
            try:
                confidence = float(0.7 if val is None else val)
            except (TypeError, ValueError):
                confidence = 0.7
            edges.append(EdgeCandidate(source=source, target=target, confidence=confidence))
        return edges
