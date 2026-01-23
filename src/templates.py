"""Model template utilities for chat prompt construction.

This module handles model-specific template settings like thinking mode control.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from transformers import AutoTokenizer


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


def get_think_tokens() -> bool:
    """Get current think tokens setting."""
    return USE_THINK_TOKENS


# Default system prompts
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that answers questions given background passages."

PLANNER_SYSTEM_PROMPT = (
    "You are an expert planner. Analyse the questions and output only a JSON object describing dependencies. "
    "Do NOT explain your reasoning. Output ONLY valid JSON, nothing else."
)


def build_chat_prompt(
    tokenizer: "AutoTokenizer",
    user_prompt: str,
    system_prompt: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
) -> str:
    """Create a chat-style prompt using the tokenizer's chat template.

    Unifies assistant generation start handling across all strategies by
    relying on add_generation_prompt=True. Thinking tags are controlled via
    USE_THINK_TOKENS and the tokenizer's enable_thinking flag.

    Args:
        tokenizer: HuggingFace tokenizer
        user_prompt: User message content
        system_prompt: System message content (optional, defaults to DEFAULT_SYSTEM_PROMPT)
        enable_thinking: Override for enable_thinking parameter.
            - None: Use global USE_THINK_TOKENS setting
            - True: Enable thinking mode (Qwen3 will output <think>...</think>)
            - False: Disable thinking mode (direct output, no thinking)
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    messages: List[Dict[str, str]] = []
    system = (system_prompt or "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_prompt.strip()})

    # Determine enable_thinking value
    if enable_thinking is None:
        # Use global setting (inverted for legacy compatibility)
        thinking_flag = not USE_THINK_TOKENS
    else:
        # Use explicit override
        thinking_flag = enable_thinking

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking_flag,
            )
        except TypeError:
            # Tokenizer doesn't support enable_thinking parameter
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
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


def build_chat_prompt_from_messages(
    tokenizer: "AutoTokenizer",
    messages: List[Dict[str, str]],
    enable_thinking: Optional[bool] = None,
) -> str:
    """Build a chat prompt string from messages using tokenizer's template.

    Args:
        tokenizer: HuggingFace tokenizer with chat template
        messages: List of message dicts with 'role' and 'content'
        enable_thinking: Override for enable_thinking parameter.

    Returns:
        Formatted prompt string
    """
    # Determine enable_thinking value
    if enable_thinking is None:
        thinking_flag = not USE_THINK_TOKENS
    else:
        thinking_flag = enable_thinking

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking_flag,
            )
        except TypeError:
            # Tokenizer doesn't support enable_thinking parameter
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
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


def apply_chat_template(
    tokenizer: "AutoTokenizer",
    messages: List[Dict[str, str]],
    enable_thinking: Optional[bool] = None,
) -> str:
    """Wrapper for tokenizer.apply_chat_template with thinking mode handling.

    This is a convenience function that handles the enable_thinking parameter
    for tokenizers that support it (like Qwen3) and falls back gracefully
    for tokenizers that don't.

    Args:
        tokenizer: HuggingFace tokenizer
        messages: List of message dicts
        enable_thinking: Override for enable_thinking parameter.
            - None: Use global USE_THINK_TOKENS setting
            - True: Enable thinking mode
            - False: Disable thinking mode

    Returns:
        Formatted prompt string
    """
    if enable_thinking is None:
        thinking_flag = not USE_THINK_TOKENS
    else:
        thinking_flag = enable_thinking

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking_flag,
        )
    except TypeError:
        # Tokenizer doesn't support enable_thinking parameter
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
