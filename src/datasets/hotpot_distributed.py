"""Distributed-Context HotpotQA: distractor-rich multi-hop CQA.

Each batch of G queries asks the SAME multi-hop question. Supporting paragraphs
are split across queries, and every query's context is padded with K distractor
paragraphs. When G exceeds the number of supporting paragraphs, the surplus
queries get only distractors -- they must rely on CSA to recover information
from peers that hold supporting evidence.

Setup example (G=4, K=8, supporting=2):
    query_0 ctx: 1 supp_A + 8 distractors  (shuffled)
    query_1 ctx: 1 supp_B + 8 distractors  (shuffled)
    query_2 ctx: 9 distractors only        (no supporting evidence!)
    query_3 ctx: 9 distractors only        (no supporting evidence!)
    All four queries: same multi-hop question.

Independent baseline: queries 2/3 cannot answer (no evidence); queries 0/1 see
only one hop and may be misled by distractors.
CSA: queries cross-attend hidden states; queries with evidence can leak info to
queries without it. This is where the protocol pays off.
"""

from __future__ import annotations

import random
from typing import List, Optional

from ..models import estimate_tokens

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def _format_paragraph(title: str, text: str) -> str:
    return f"[{title}] {text}"


def load_distributed_hotpot_groups(
    split: str,
    *,
    subset: str = "distractor",
    max_groups: int = 500,
    n_agents: int = 4,
    paragraphs_per_agent: int = 9,
    seed: int = 42,
    only_bridge: bool = True,
    require_min_supporting: int = 2,
    cross_question_distractor_pool: bool = True,
) -> List[dict]:
    """Build distributed-context multi-hop CQA groups.

    Args:
        split: "train" / "validation"
        subset: "distractor" (10 paragraphs per question, 2 supporting + 8 distractors)
        max_groups: how many multi-hop questions to use as groups
        n_agents: G, queries per group. 2 supporting are spread; the rest get
            no supporting (only distractors).
        paragraphs_per_agent: how many paragraphs to pack into each query's
            context. Includes the supporting one (if assigned). Excess slots
            are filled with distractors.
        only_bridge: keep only bridge-type questions (skip comparison-type which
            often have answers in both paragraphs and don't need real cross-hop
            reasoning).
        require_min_supporting: drop questions where fewer than this many
            supporting paragraphs are present in the context.
        cross_question_distractor_pool: if True, build a global pool of
            distractor paragraphs and sample from it (more diverse than each
            question's own 8 distractors). This makes the noise harder.

    Returns:
        List of groups in the "items" format expected by SQuADGroupedDataset.
        Each group: {"items": [{"context", "question", "references"}, ...], "title"}.
    """
    if load_dataset is None:
        raise RuntimeError("Install datasets: pip install datasets")

    raw = list(load_dataset("hotpotqa/hotpot_qa", subset, split=split))
    rng = random.Random(seed)
    rng.shuffle(raw)

    # Phase 1: parse rows, build per-question (supporting, distractor) pools.
    parsed = []
    global_distractor_pool: List[str] = []
    for row in raw:
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        if not question or not answer:
            continue
        if only_bridge and row.get("type", "") != "bridge":
            continue

        sup_titles = set(row.get("supporting_facts", {}).get("title", []))
        ctx = row.get("context", {})
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])

        para_map = {}
        for t, sents in zip(titles, sentences):
            para_map[t] = " ".join(s.strip() for s in sents)

        sup_paras = [(t, para_map[t]) for t in sup_titles if t in para_map]
        if len(sup_paras) < require_min_supporting:
            continue

        distractor_paras = [(t, para_map[t]) for t in titles if t not in sup_titles]

        parsed.append({
            "qid": row.get("id", f"hp_{len(parsed)}"),
            "question": question,
            "answer": answer,
            "supporting": sup_paras,
            "distractors": distractor_paras,
        })

        # Add distractor texts to the global pool (only for first N rows; cap to
        # avoid blowing memory for huge splits).
        if len(global_distractor_pool) < 5000:
            for t, p in distractor_paras:
                global_distractor_pool.append(_format_paragraph(t, p))

    if not parsed:
        raise ValueError("No HotpotQA rows passed filters; relax constraints.")

    # Phase 2: for each parsed question, build n_agents queries.
    groups = []
    for i, q in enumerate(parsed):
        if len(groups) >= max_groups:
            break

        sup_paras = list(q["supporting"])
        rng.shuffle(sup_paras)
        # Cap supporting spread at min(n_agents, len(sup_paras)) — surplus
        # agents get distractors only.
        n_supp_assigned = min(n_agents, len(sup_paras))

        # Per-question distractor pool: union of own distractors and (optionally)
        # cross-question pool.
        own_distractors = [_format_paragraph(t, p) for t, p in q["distractors"]]
        if cross_question_distractor_pool and global_distractor_pool:
            distractor_pool = list(global_distractor_pool)
        else:
            distractor_pool = list(own_distractors)

        items = []
        for agent_idx in range(n_agents):
            slots = []
            if agent_idx < n_supp_assigned:
                t, p = sup_paras[agent_idx]
                slots.append(_format_paragraph(t, p))

            # Fill remaining slots with distractors (sampled without replacement).
            need = paragraphs_per_agent - len(slots)
            sampled = rng.sample(distractor_pool, k=min(need, len(distractor_pool)))
            slots.extend(sampled)

            # Shuffle so the supporting paragraph isn't always at position 0.
            rng.shuffle(slots)
            context = "\n\n".join(slots)

            items.append({
                "context": context,
                "question": q["question"],
                "references": [q["answer"]],
                "answer_tokens": max(estimate_tokens(q["answer"]), 12),
                "has_supporting": agent_idx < n_supp_assigned,
            })

        groups.append({
            "items": items,
            "title": q["qid"],
            "n_supporting_assigned": n_supp_assigned,
        })

    return groups
