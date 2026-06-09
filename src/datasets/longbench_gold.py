"""TRUE gold-passage oracle for LongBench multi-hop QA.

LongBench export jsonl has no supporting_facts. We recover the real gold passages by
matching each LongBench question (by normalized text) to the flashrag dataset, which carries
gold titles in its metadata:
  - 2wiki / hotpot : metadata["supporting_facts"]["title"]  (list of gold passage titles)
  - musique        : metadata["question_decomposition"][*]["support_paragraph"]["title"]

Verified: normalized-question match covers 200/200 LongBench samples for all three tasks.
This replaces the entity-bridge approximation in bench_longbench.oracle_passages, which is
unreliable for hotpot yes/no questions (the answer-substring branch fires on arbitrary prose).
"""
import glob
import re

import pyarrow as pa

FLASHRAG = "/mnt/data/zichuanfu/.cache/huggingface/datasets/RUC-NLPIR___flash_rag_datasets"

# LongBench task name -> flashrag dataset dir
_TASK2FR = {
    "2wikimqa": "2wikimultihopqa",
    "hotpotqa": "hotpotqa",
    "musique": "musique",
}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _norm_title(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _load_arrow(name, split):
    fs = sorted(glob.glob(f"{FLASHRAG}/{name}/0.0.0/*/flash_rag_datasets-{split}*.arrow"))
    tables = []
    for f in fs:
        with pa.memory_map(f) as src:
            try:
                t = pa.ipc.open_stream(src).read_all()
            except Exception:
                t = pa.ipc.open_file(src).read_all()
        tables.append(t)
    return pa.concat_tables(tables) if tables else None


def _gold_titles_from_meta(name, meta):
    """Return the set of normalized gold passage titles for one flashrag row."""
    titles = set()
    if name == "musique":
        for hop in (meta.get("question_decomposition") or []):
            sp = hop.get("support_paragraph") or {}
            t = sp.get("title")
            if t:
                titles.add(_norm_title(t))
    else:  # 2wiki / hotpot: supporting_facts.title is a list
        sf = meta.get("supporting_facts") or {}
        for t in (sf.get("title") or []):
            if t:
                titles.add(_norm_title(t))
    return titles


_CACHE = {}


def gold_title_index(task):
    """norm(question) -> set(norm gold title) over flashrag dev+train+test for `task`. Cached."""
    if task in _CACHE:
        return _CACHE[task]
    name = _TASK2FR.get(task)
    if name is None:
        _CACHE[task] = {}
        return {}
    idx = {}
    for split in ("dev", "train", "test"):
        tb = _load_arrow(name, split)
        if tb is None:
            continue
        qs = tb.column("question").to_pylist()
        ms = tb.column("metadata").to_pylist()
        for q, m in zip(qs, ms):
            if not isinstance(m, dict):
                continue
            ts = _gold_titles_from_meta(name, m)
            if ts:
                idx[_norm(q)] = ts
    _CACHE[task] = idx
    return idx


def _passage_title(p):
    for ln in p.split("\n"):
        ln = ln.strip()
        if ln and not re.match(r"passage\s*\d+", ln.lower()):
            return ln
    return ""


def gold_passages_flashrag(passages, question, task):
    """Select LongBench passages whose title matches a flashrag gold title for this question.
    Returns the gold subset, or None if the question is not found in flashrag (caller falls
    back to the entity-bridge approximation)."""
    idx = gold_title_index(task)
    gold_titles = idx.get(_norm(question))
    if not gold_titles:
        return None
    keep = [p for p in passages if _norm_title(_passage_title(p)) in gold_titles]
    return keep or None
