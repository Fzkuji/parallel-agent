"""
Document Split QA Dataset

Multi-agent information asymmetry scenario:
- Each "agent" (sequence) only sees a SPLIT of the original document
- Questions require information from across splits to answer correctly
- This directly mirrors multi-agent collaboration with information asymmetry

Setup:
  - Take a SQuAD context (e.g., 500 tokens)
  - Split into N chunks (e.g., 2-3 chunks of ~150 tokens each)
  - Each sequence gets one chunk as its "local context"
  - The question is the same for all sequences
  - Only the sequence whose chunk contains the answer can answer directly
  - Other sequences NEED the shared state z to get context clues

This creates genuine information asymmetry, just like multi-agent papers.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def split_text_into_chunks(text: str, n_chunks: int) -> List[str]:
    """Split text into N roughly equal chunks by sentences."""
    # Split by sentence boundaries
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ".!?" and len(current) > 20:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    if not sentences:
        # Fallback: split by characters
        chunk_size = max(1, len(text) // n_chunks)
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)][:n_chunks]

    # Distribute sentences into chunks
    chunk_size = max(1, len(sentences) // n_chunks)
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else len(sentences)
        chunks.append(" ".join(sentences[start:end]))

    # Remove empty chunks
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        chunks = [text]

    return chunks


def find_answer_chunk(chunks: List[str], answer: str) -> int:
    """Find which chunk contains the answer (for analysis/oracle purposes)."""
    answer_lower = answer.lower()
    for i, chunk in enumerate(chunks):
        if answer_lower in chunk.lower():
            return i
    return -1  # answer not found in any chunk


def load_doc_split_groups(
    split: str,
    *,
    n_agents: int = 3,          # number of parallel agents (splits)
    max_groups: int = 200,
    min_context_words: int = 100,  # minimum words in context to be worth splitting
    seed: int = 42,
) -> List[dict]:
    """
    Load SQuAD data as Document Split QA groups.

    Each group:
    - "splits": list of N text chunks (one per agent)
    - "question": the shared question
    - "references": gold answers
    - "answer_chunk_idx": which chunk contains the answer (-1 if none)
    - "full_context": original full context (for oracle/sequential baselines)

    Args:
        split: Dataset split ("train" or "validation")
        n_agents: Number of document splits (= number of parallel sequences)
        max_groups: Maximum number of groups to return
        min_context_words: Skip contexts shorter than this
        seed: Random seed
    """
    if load_dataset is None:
        raise RuntimeError("Install datasets: pip install datasets")

    raw = load_dataset("squad", split=split)
    rng = random.Random(seed)

    # Collect all samples
    samples = list(raw)
    rng.shuffle(samples)

    groups = []
    for row in samples:
        if len(groups) >= max_groups:
            break

        context = row["context"].strip()
        question = row["question"].strip()
        answers = row.get("answers", {}).get("text", [])
        if not answers:
            continue

        # Skip short contexts
        if len(context.split()) < min_context_words:
            continue

        # Split document into N chunks
        chunks = split_text_into_chunks(context, n_agents)
        if len(chunks) < 2:
            continue

        # Pad to n_agents if fewer chunks
        while len(chunks) < n_agents:
            chunks.append(chunks[-1])  # repeat last chunk
        chunks = chunks[:n_agents]

        # Find which chunk has the answer
        answer_chunk_idx = find_answer_chunk(chunks, answers[0])

        groups.append({
            "splits": chunks,               # N text chunks, one per agent
            "question": question,
            "references": answers,
            "answer_chunk_idx": answer_chunk_idx,  # -1 if answer spans chunks
            "full_context": context,
            "title": row.get("title", ""),
            "n_agents": n_agents,
        })

    return groups


def groups_to_cqa_format(doc_split_groups: List[dict]) -> List[dict]:
    """
    Convert doc_split groups to the standard CQA format used by other strategies.

    Each agent gets its local chunk as context, but the question is the same.
    This tests whether agents can collaborate to answer a question no single
    agent has complete information for.
    """
    from ..models import estimate_tokens, Question

    cqa_groups = []
    for g in doc_split_groups:
        n = g["n_agents"]
        questions = []
        for i in range(n):
            answer_text = g["references"][0] if g["references"] else ""
            questions.append({
                "qid": f"agent_{i}",
                "text": g["question"],
                "answer_tokens": max(estimate_tokens(answer_text), 12),
                "references": g["references"],
                # Each agent only sees its own chunk
                "local_context": g["splits"][i],
            })
        cqa_groups.append({
            "context": g["full_context"],    # full context for oracle/sequential
            "splits": g["splits"],            # individual splits for multi-agent
            "title": g.get("title", ""),
            "questions": questions,
            "answer_chunk_idx": g["answer_chunk_idx"],
        })
    return cqa_groups
