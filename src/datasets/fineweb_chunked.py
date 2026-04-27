"""FineWeb-Edu chunked loader for CSA distillation pretraining.

Each "group" is a single long document split into G non-overlapping chunks.
The student CSA training will:
  - have G "queries" each see one chunk
  - oracle (teacher) sees the full document
  - CSA must learn to recover the oracle's cross-chunk information through
    hidden-state communication

Chunking strategy:
  - Filter documents with token_count >= G * chunk_size + buffer
  - Tokenize, take consecutive non-overlapping windows of `chunk_size` tokens
  - Decode chunks back to text so they look like normal prompts
"""

from __future__ import annotations

import logging
import os
import random
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset, load_from_disk
except ImportError:
    load_dataset = None
    load_from_disk = None


def stream_fineweb_edu_documents(
    *,
    cache_dir: str = None,
    dataset_path: str = None,
    split: str = "train",
    name: str = "sample-10BT",
    min_token_count: int = 4500,
    seed: int = 42,
) -> Iterator[dict]:
    """Stream FineWeb-Edu docs that pass the min length filter.

    If `dataset_path` points at an on-disk arrow dataset directory we use
    `load_from_disk`. Otherwise we go through `load_dataset(name=name, ...)`,
    which transparently uses the HF cache at `cache_dir` if set.
    """
    if load_dataset is None:
        raise RuntimeError("Install `datasets`")

    if dataset_path and os.path.isdir(dataset_path):
        ds = load_from_disk(dataset_path)
        if hasattr(ds, "keys") and split in ds:
            ds = ds[split]
    else:
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", name=name, split=split, **kwargs
        )

    # Shuffle seed-stable so different ranks/runs see comparable data.
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)

    for idx in indices:
        row = ds[idx]
        if row.get("token_count", 0) < min_token_count:
            continue
        text = row.get("text", "")
        if not text:
            continue
        yield {
            "text": text,
            "doc_id": row.get("id", str(idx)),
            "token_count": row["token_count"],
        }


def chunk_document_by_tokens(
    text: str,
    tokenizer,
    *,
    n_chunks: int = 4,
    chunk_tokens: int = 1024,
    skip_head_tokens: int = 0,
) -> Optional[List[List[int]]]:
    """Tokenize, take G non-overlapping consecutive windows of token ids.

    Returns None if the doc cannot fill all G chunks. We return token ids
    directly (not decoded text) so the trainer doesn't need to re-encode —
    decode/re-encode round-trip can change length and break chunk boundaries.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if skip_head_tokens:
        ids = ids[skip_head_tokens:]
    needed = n_chunks * chunk_tokens
    if len(ids) < needed:
        return None
    chunks = []
    for c in range(n_chunks):
        start = c * chunk_tokens
        end = start + chunk_tokens
        chunks.append(ids[start:end])
    return chunks


class FineWebChunkedIterable:
    """Yields {"chunks": [...], "doc_id": ...} groups one at a time.

    Used with `IterableDataset` semantics so we don't have to materialize all
    chunked docs upfront.
    """

    def __init__(
        self,
        tokenizer,
        *,
        cache_dir: str = None,
        dataset_path: str = None,
        split: str = "train",
        name: str = "sample-10BT",
        n_chunks: int = 4,
        chunk_tokens: int = 1024,
        max_groups: int = 5000,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.dataset_path = dataset_path
        self.split = split
        self.name = name
        self.n_chunks = n_chunks
        self.chunk_tokens = chunk_tokens
        self.max_groups = max_groups
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self._min_tokens = max(n_chunks * chunk_tokens + 64, 4500)

    def __iter__(self):
        produced = 0
        seen = 0
        for doc in stream_fineweb_edu_documents(
            cache_dir=self.cache_dir,
            dataset_path=self.dataset_path,
            split=self.split,
            name=self.name,
            min_token_count=self._min_tokens,
            seed=self.seed,
        ):
            if produced >= self.max_groups:
                break
            seen += 1
            if seen % self.world_size != self.rank:
                continue  # round-robin shard across ranks
            chunks = chunk_document_by_tokens(
                doc["text"],
                self.tokenizer,
                n_chunks=self.n_chunks,
                chunk_tokens=self.chunk_tokens,
            )
            if chunks is None:
                continue
            yield {
                "chunks": chunks,
                "doc_id": doc["doc_id"],
            }
            produced += 1

    def __len__(self):
        # IterableDataset doesn't strictly need __len__, but trainer logging
        # calls it. Use the upper bound.
        return self.max_groups // max(self.world_size, 1)
