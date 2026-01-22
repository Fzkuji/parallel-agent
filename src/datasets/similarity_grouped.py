"""Similarity-based grouped dataset loader for cross-batch training.

Groups questions by semantic similarity using sentence embeddings.
This provides meaningful cross-batch interactions during training,
as semantically related questions can benefit from each other's context.
"""

from typing import List, Dict, Optional, Tuple
import random
import numpy as np


def compute_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute sentence embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        model_name: Sentence transformer model name
        batch_size: Batch size for embedding computation
        device: Device to use for computation

    Returns:
        numpy array of shape [len(texts), embedding_dim]
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "Please install sentence-transformers: pip install sentence-transformers"
        )

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 1000,
        convert_to_numpy=True,
    )
    return embeddings


def group_by_similarity(
    items: List[Dict],
    embeddings: np.ndarray,
    min_group_size: int = 2,
    max_group_size: int = 8,
    similarity_threshold: float = 0.5,
    seed: int = 42,
) -> List[List[int]]:
    """
    Group items by semantic similarity using greedy clustering.

    Args:
        items: List of item dictionaries
        embeddings: Precomputed embeddings [n_items, embed_dim]
        min_group_size: Minimum items per group
        max_group_size: Maximum items per group
        similarity_threshold: Minimum cosine similarity to include in group
        seed: Random seed

    Returns:
        List of groups, each group is a list of item indices
    """
    rng = np.random.RandomState(seed)
    n_items = len(items)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    # Compute similarity matrix
    similarity_matrix = np.dot(normalized, normalized.T)

    # Track which items have been assigned
    assigned = set()
    groups = []

    # Shuffle indices for random starting points
    indices = list(range(n_items))
    rng.shuffle(indices)

    for seed_idx in indices:
        if seed_idx in assigned:
            continue

        # Start a new group with this seed
        group = [seed_idx]
        assigned.add(seed_idx)

        # Find similar items
        similarities = similarity_matrix[seed_idx]

        # Get candidate indices sorted by similarity (descending)
        candidates = [
            (i, similarities[i])
            for i in range(n_items)
            if i not in assigned and similarities[i] >= similarity_threshold
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Add candidates until max_group_size
        for cand_idx, _ in candidates:
            if len(group) >= max_group_size:
                break
            group.append(cand_idx)
            assigned.add(cand_idx)

        # Only keep groups that meet minimum size
        if len(group) >= min_group_size:
            groups.append(group)
        else:
            # Put items back for potential grouping
            for idx in group:
                assigned.discard(idx)

    # Handle remaining ungrouped items
    remaining = [i for i in range(n_items) if i not in assigned]
    if remaining:
        # Group remaining items randomly
        rng.shuffle(remaining)
        for i in range(0, len(remaining), max_group_size):
            group = remaining[i:i + max_group_size]
            if len(group) >= min_group_size:
                groups.append(group)

    return groups


def estimate_tokens(text: str) -> int:
    """Rough token count estimation."""
    return max(len(text.split()), len(text) // 4)


def load_similarity_grouped_triviaqa(
    split: str = "train",
    max_groups: Optional[int] = None,
    min_questions: int = 2,
    max_questions: int = 8,
    similarity_threshold: float = 0.5,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    seed: int = 42,
    device: str = "cpu",
) -> List[Dict]:
    """
    Load TriviaQA dataset and group questions by semantic similarity.

    Unlike random grouping, this creates groups of semantically related questions,
    which provides more meaningful cross-batch training signal.

    Args:
        split: Dataset split to load ('train' or 'validation')
        max_groups: Maximum number of groups to return
        min_questions: Minimum questions per group
        max_questions: Maximum questions per group
        similarity_threshold: Minimum cosine similarity for grouping
        embedding_model: Sentence transformer model for similarity
        seed: Random seed
        device: Device for embedding computation

    Returns:
        List of group dictionaries, each with:
        - context: Empty string (TriviaQA is open-domain)
        - title: Group identifier
        - questions: List of question dicts
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print(f"Loading TriviaQA {split} split...")
    dataset = load_dataset(
        "mandarjoshi/trivia_qa", "rc.nocontext", split=split, trust_remote_code=True
    )

    # Convert to list of items
    all_items = []
    for idx, item in enumerate(dataset):
        question_text = item["question"].strip()
        answer_aliases = item.get("answer", {}).get("aliases", [])
        if not answer_aliases:
            answer_aliases = [item.get("answer", {}).get("value", "")]

        answer_text = answer_aliases[0] if answer_aliases else ""
        answer_tokens = max(estimate_tokens(answer_text), 12)

        all_items.append({
            "original_idx": idx,
            "question": question_text,
            "answer_aliases": answer_aliases,
            "answer_tokens": answer_tokens,
        })

    print(f"Computing embeddings for {len(all_items)} questions...")
    questions_text = [item["question"] for item in all_items]
    embeddings = compute_embeddings(
        questions_text, model_name=embedding_model, device=device
    )

    print(f"Grouping by similarity (threshold={similarity_threshold})...")
    group_indices = group_by_similarity(
        all_items,
        embeddings,
        min_group_size=min_questions,
        max_group_size=max_questions,
        similarity_threshold=similarity_threshold,
        seed=seed,
    )

    # Limit groups if needed
    if max_groups and len(group_indices) > max_groups:
        rng = random.Random(seed)
        rng.shuffle(group_indices)
        group_indices = group_indices[:max_groups]

    # Build output format
    groups = []
    for group_id, indices in enumerate(group_indices):
        questions = []
        for q_idx, item_idx in enumerate(indices):
            item = all_items[item_idx]
            questions.append({
                "qid": f"Q{q_idx + 1}",
                "text": item["question"],
                "answer_tokens": item["answer_tokens"],
                "references": item["answer_aliases"],
            })

        groups.append({
            "context": "",  # TriviaQA is open-domain
            "title": f"triviaqa-sim-group-{group_id}",
            "questions": questions,
        })

    print(f"Created {len(groups)} similarity-based groups")

    # Print statistics
    group_sizes = [len(g["questions"]) for g in groups]
    print(f"Group sizes: min={min(group_sizes)}, max={max(group_sizes)}, "
          f"mean={sum(group_sizes)/len(group_sizes):.1f}")

    return groups


def load_similarity_grouped_nq(
    split: str = "train",
    max_groups: Optional[int] = None,
    min_questions: int = 2,
    max_questions: int = 8,
    similarity_threshold: float = 0.5,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    seed: int = 42,
    device: str = "cpu",
) -> List[Dict]:
    """
    Load Natural Questions dataset and group by semantic similarity.

    Args:
        split: Dataset split ('train' or 'validation')
        max_groups: Maximum number of groups
        min_questions: Minimum questions per group
        max_questions: Maximum questions per group
        similarity_threshold: Minimum cosine similarity for grouping
        embedding_model: Sentence transformer model
        seed: Random seed
        device: Device for embedding computation

    Returns:
        List of group dictionaries
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print(f"Loading Natural Questions {split} split...")
    # Use the simplified version without documents
    dataset = load_dataset("google-research-datasets/natural_questions", "default", split=split)

    # Extract questions with short answers
    all_items = []
    for idx, item in enumerate(dataset):
        question_text = item["question"]["text"].strip()

        # Get short answers
        annotations = item.get("annotations", {})
        short_answers = []
        if annotations:
            for ann in annotations.get("short_answers", []):
                if ann.get("text"):
                    short_answers.append(ann["text"])

        if not short_answers:
            continue  # Skip questions without short answers

        answer_text = short_answers[0]
        answer_tokens = max(estimate_tokens(answer_text), 12)

        all_items.append({
            "original_idx": idx,
            "question": question_text,
            "answer_aliases": short_answers,
            "answer_tokens": answer_tokens,
        })

        # Limit to manageable size for memory
        if len(all_items) >= 100000:
            break

    if not all_items:
        raise ValueError("No valid questions found in Natural Questions dataset")

    print(f"Computing embeddings for {len(all_items)} questions...")
    questions_text = [item["question"] for item in all_items]
    embeddings = compute_embeddings(
        questions_text, model_name=embedding_model, device=device
    )

    print(f"Grouping by similarity (threshold={similarity_threshold})...")
    group_indices = group_by_similarity(
        all_items,
        embeddings,
        min_group_size=min_questions,
        max_group_size=max_questions,
        similarity_threshold=similarity_threshold,
        seed=seed,
    )

    if max_groups and len(group_indices) > max_groups:
        rng = random.Random(seed)
        rng.shuffle(group_indices)
        group_indices = group_indices[:max_groups]

    groups = []
    for group_id, indices in enumerate(group_indices):
        questions = []
        for q_idx, item_idx in enumerate(indices):
            item = all_items[item_idx]
            questions.append({
                "qid": f"Q{q_idx + 1}",
                "text": item["question"],
                "answer_tokens": item["answer_tokens"],
                "references": item["answer_aliases"],
            })

        groups.append({
            "context": "",
            "title": f"nq-sim-group-{group_id}",
            "questions": questions,
        })

    print(f"Created {len(groups)} similarity-based groups from Natural Questions")
    return groups


def load_similarity_grouped_generic(
    questions: List[str],
    answers: List[List[str]],
    max_groups: Optional[int] = None,
    min_questions: int = 2,
    max_questions: int = 8,
    similarity_threshold: float = 0.5,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    seed: int = 42,
    device: str = "cpu",
) -> List[Dict]:
    """
    Create similarity-based groups from arbitrary question-answer pairs.

    Args:
        questions: List of question strings
        answers: List of answer lists (each question can have multiple valid answers)
        max_groups: Maximum number of groups
        min_questions: Minimum questions per group
        max_questions: Maximum questions per group
        similarity_threshold: Minimum cosine similarity for grouping
        embedding_model: Sentence transformer model
        seed: Random seed
        device: Device for embedding computation

    Returns:
        List of group dictionaries
    """
    assert len(questions) == len(answers), "Questions and answers must have same length"

    all_items = []
    for idx, (q, a_list) in enumerate(zip(questions, answers)):
        answer_text = a_list[0] if a_list else ""
        all_items.append({
            "original_idx": idx,
            "question": q.strip(),
            "answer_aliases": a_list,
            "answer_tokens": max(estimate_tokens(answer_text), 12),
        })

    print(f"Computing embeddings for {len(all_items)} questions...")
    embeddings = compute_embeddings(
        [item["question"] for item in all_items],
        model_name=embedding_model,
        device=device,
    )

    print(f"Grouping by similarity...")
    group_indices = group_by_similarity(
        all_items,
        embeddings,
        min_group_size=min_questions,
        max_group_size=max_questions,
        similarity_threshold=similarity_threshold,
        seed=seed,
    )

    if max_groups and len(group_indices) > max_groups:
        rng = random.Random(seed)
        rng.shuffle(group_indices)
        group_indices = group_indices[:max_groups]

    groups = []
    for group_id, indices in enumerate(group_indices):
        questions_list = []
        for q_idx, item_idx in enumerate(indices):
            item = all_items[item_idx]
            questions_list.append({
                "qid": f"Q{q_idx + 1}",
                "text": item["question"],
                "answer_tokens": item["answer_tokens"],
                "references": item["answer_aliases"],
            })

        groups.append({
            "context": "",
            "title": f"sim-group-{group_id}",
            "questions": questions_list,
        })

    return groups
