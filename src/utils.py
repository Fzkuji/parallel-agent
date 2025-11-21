from __future__ import annotations

import random
from typing import List

import numpy as np
import torch


DEFAULT_GENERATION_SEED = 13


def reset_generation_seed(seed: int = DEFAULT_GENERATION_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def strip_think_prefix(text: str) -> str:
    """Remove any <think>...</think> blocks wherever they appear."""
    s = text
    open_tag = "<think>"
    close_tag = "</think>"
    while True:
        start = s.find(open_tag)
        if start == -1:
            break
        end = s.find(close_tag, start)
        if end == -1:
            s = s.replace(open_tag, "")
            break
        s = s[:start] + s[end + len(close_tag) :]
    s = s.replace(open_tag, "").replace(close_tag, "")
    return s.strip()


def strip_assistant_prefix(text: str) -> str:
    """Remove spurious leading 'assistant' echoes from model outputs."""
    s = text.lstrip()
    lower = s.lower()
    if lower.startswith("assistant:\n") or lower.startswith("assistant: "):
        s = s.split(":", 1)[1].lstrip()
    elif lower.startswith("assistant\n") or lower.startswith("assistant "):
        s = s[len("assistant") :].lstrip()
    return s


def trim_after_prompt(tokens: List[int], eos_id: int, pad_id: int) -> List[int]:
    trimmed: List[int] = []
    for token in tokens:
        if token in (eos_id, pad_id):
            break
        trimmed.append(token)
    return trimmed
