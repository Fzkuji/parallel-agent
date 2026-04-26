#!/usr/bin/env python3
"""Pretrain CSA-v2 by distilling from an oracle that sees the full document.

Idea:
  - Take a long FineWeb-Edu document, split into G non-overlapping chunks.
  - Oracle: forward the concatenated full doc, take last hidden at each
    chunk boundary -> [G, d] target.
  - Student: G chunks as a batch of size G, base hidden at each chunk's end,
    apply CSA cross-batch attention -> mixed [G, d].
  - Loss: MSE(mixed, oracle).
  - Only CSA gets gradients; base model + lm_head are frozen.

DDP usage (8x H20, each card processes a different shard of documents):
    torchrun --standalone --nproc_per_node=8 scripts/pretrain_csa_distill.py \\
        --model-path /YOUR_PATH/Qwen2.5-7B-Instruct \\
        --fineweb-path /mnt/data/zichuanfu/.cache/huggingface/datasets/HuggingFaceFW___fineweb-edu/sample-10BT \\
        --max-groups 5000 --epochs 1 --n-chunks 4 --chunk-tokens 1024 \\
        --output-dir ./out/csa_distill
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cross_batch.attention import CrossSequenceAttentionV2
from src.cross_batch.distill_trainer import CSADistillTrainer
from src.datasets import FineWebChunkedIterable

logging.basicConfig(level=logging.INFO, format="%(asctime)s [r%(name)s] %(message)s")
log = logging.getLogger("0")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)

    # Dataset
    p.add_argument("--fineweb-path", default=None,
                   help="Local path to FineWeb-Edu arrow dataset directory")
    p.add_argument("--fineweb-cache-dir", default=None,
                   help="HF cache directory for FineWeb-Edu (alternative to path)")
    p.add_argument("--fineweb-name", default="sample-10BT")
    p.add_argument("--fineweb-split", default="train")
    p.add_argument("--max-groups", type=int, default=5000)

    # Chunking
    p.add_argument("--n-chunks", type=int, default=4)
    p.add_argument("--chunk-tokens", type=int, default=1024)

    # CSA
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--csa-init-from", default=None)

    # Training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def setup_distributed():
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, -1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    global log
    log = logging.getLogger(f"{rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        log.info("DDP: rank=%d world_size=%d local_rank=%d", rank, world_size, local_rank)
        log.info("Loading %s", args.model_path)

    torch.manual_seed(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if local_rank >= 0:
        device = f"cuda:{local_rank}"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
    else:
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
    hidden_size = model.config.hidden_size

    csa = CrossSequenceAttentionV2(
        hidden_size=hidden_size, num_heads=args.num_heads,
        use_gate=True, adaptive_top_k=True,
    )
    if args.csa_init_from:
        if is_main:
            log.info("Init CSA from %s", args.csa_init_from)
        ckpt = torch.load(args.csa_init_from, map_location="cpu")
        missing, unexpected = csa.load_state_dict(
            ckpt["cross_batch_module"], strict=False
        )
        if is_main and (missing or unexpected):
            log.warning("CSA init missing=%s unexpected=%s", missing, unexpected)

    trainer = CSADistillTrainer(
        model=model, tokenizer=tokenizer, cross_batch_module=csa,
        device=device, learning_rate=args.lr,
        local_rank=local_rank,
        n_chunks=args.n_chunks, chunk_tokens=args.chunk_tokens,
    )

    if is_main:
        log.info("Streaming FineWeb-Edu chunked groups (max=%d / rank, n_chunks=%d, chunk_tokens=%d)",
                 args.max_groups, args.n_chunks, args.chunk_tokens)
    groups = FineWebChunkedIterable(
        tokenizer=tokenizer,
        cache_dir=args.fineweb_cache_dir,
        dataset_path=args.fineweb_path,
        split=args.fineweb_split,
        name=args.fineweb_name,
        n_chunks=args.n_chunks,
        chunk_tokens=args.chunk_tokens,
        max_groups=args.max_groups,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )

    history = trainer.train(
        groups=groups,
        num_epochs=args.epochs,
        save_dir=args.output_dir if is_main else None,
        rank=rank,
        log_every=args.log_every,
        save_every=args.save_every,
        total_groups=args.max_groups,
    )

    if is_main:
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        log.info("Done. Saved to %s", args.output_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
