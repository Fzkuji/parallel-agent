"""
HotpotQA Document Split Dataset

HotpotQA is perfect for multi-agent information asymmetry:
- Each question requires reasoning over 2 supporting paragraphs
- Agent 1 gets paragraph 1, Agent 2 gets paragraph 2
- Neither agent alone has enough info to answer — they MUST collaborate

This directly mirrors multi-agent papers where agents have complementary info.

Each group:
  - "splits": [paragraph_1, paragraph_2] — one per agent
  - "question": the shared multi-hop question
  - "references": gold answer
  - "full_context": all paragraphs concatenated (for oracle baseline)
"""

from __future__ import annotations

import random
from typing import List, Optional

from ..models import estimate_tokens

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def load_hotpot_split_groups(
    split: str,
    *,
    subset: str = "distractor",
    max_groups: int = 200,
    seed: int = 42,
    use_supporting_only: bool = True,
) -> List[dict]:
    """
    Load HotpotQA as 2-agent document split groups.

    Each question needs 2 supporting paragraphs. We give each agent one paragraph.
    Agent 1 sees supporting_para_1, Agent 2 sees supporting_para_2.
    The question is shared. Neither agent alone can answer — they need to collaborate.

    Args:
        split: "train" or "validation"
        subset: "distractor" (default) or "fullwiki"
        max_groups: max number of questions to use
        seed: random seed
        use_supporting_only: if True, each agent only sees its supporting paragraph
                             if False, each agent also sees distractor paragraphs
    """
    if load_dataset is None:
        raise RuntimeError("Install datasets: pip install datasets")

    raw = load_dataset("hotpotqa/hotpot_qa", subset, split=split)
    rng = random.Random(seed)

    samples = list(raw)
    rng.shuffle(samples)

    groups = []
    for row in samples:
        if len(groups) >= max_groups:
            break

        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        if not question or not answer:
            continue

        # Get supporting facts info
        supporting_facts = row.get("supporting_facts", {})
        sup_titles = supporting_facts.get("title", [])
        sup_titles_set = set(sup_titles)

        # Get all context paragraphs
        ctx = row.get("context", {})
        all_titles = ctx.get("title", [])
        all_sentences = ctx.get("sentences", [])

        # Build paragraph map
        para_map = {}
        for title, sents in zip(all_titles, all_sentences):
            para_map[title] = " ".join(s.strip() for s in sents)

        if use_supporting_only:
            # Get the 2 supporting paragraphs only
            sup_paras = []
            for title in sup_titles_set:
                if title in para_map:
                    sup_paras.append((title, para_map[title]))

            if len(sup_paras) < 2:
                continue  # Need exactly 2 supporting paragraphs

            # Take first 2 supporting paragraphs
            para1_title, para1_text = sup_paras[0]
            para2_title, para2_text = sup_paras[1]
        else:
            # Each agent gets supporting + distractor paragraphs
            # Build supporting paragraphs
            sup_paras = [(t, para_map[t]) for t in sup_titles_set if t in para_map]
            if len(sup_paras) < 2:
                continue

            # Distractors = non-supporting
            distractor_titles = [t for t in all_titles if t not in sup_titles_set]

            # Agent 1: first supporting + half distractors
            # Agent 2: second supporting + other half distractors
            mid = len(distractor_titles) // 2
            d1 = [para_map[t] for t in distractor_titles[:mid] if t in para_map]
            d2 = [para_map[t] for t in distractor_titles[mid:] if t in para_map]

            para1_title, para1_text = sup_paras[0]
            para2_title, para2_text = sup_paras[1]

            para1_text = para1_text + "\n\n" + "\n\n".join(d1) if d1 else para1_text
            para2_text = para2_text + "\n\n" + "\n\n".join(d2) if d2 else para2_text

        # Full context for oracle
        full_context = "\n\n".join([
            f"{t}: {para_map[t]}" for t in all_titles if t in para_map
        ])

        answer_tokens = max(estimate_tokens(answer), 12)

        groups.append({
            "splits": [
                f"[Source: {para1_title}]\n{para1_text}",
                f"[Source: {para2_title}]\n{para2_text}",
            ],
            "question": question,
            "references": [answer],
            "answer_tokens": answer_tokens,
            "full_context": full_context,
            "n_agents": 2,
            "qid": row.get("id", f"hotpot_{len(groups)}"),
            "type": row.get("type", ""),  # "bridge" or "comparison"
        })

    return groups


def groups_to_cqa_format(hotpot_groups: List[dict]) -> List[dict]:
    """Convert to standard CQA format used by strategies."""
    cqa_groups = []
    for g in hotpot_groups:
        n = g["n_agents"]
        questions = []
        for i in range(n):
            questions.append({
                "qid": f"agent_{i}",
                "text": g["question"],
                "answer_tokens": g.get("answer_tokens", 20),
                "references": g["references"],
                "local_context": g["splits"][i],
            })
        cqa_groups.append({
            "context": g["full_context"],
            "splits": g["splits"],
            "title": g.get("qid", ""),
            "questions": questions,
            "type": g.get("type", ""),
        })
    return cqa_groups
