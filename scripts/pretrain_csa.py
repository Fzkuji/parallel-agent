#!/usr/bin/env python3
"""Step 1: Continual pretraining of base model + CSA on FineWeb-Edu chunked docs.

Standard next-token CE loss. Both base model and CSA are trainable. CSA is
inserted at multiple layer boundaries (every 4 layers by default) via forward
hooks, with a single shared CSA module.

DDP usage (8x H20):
    torchrun --standalone --nproc_per_node=8 scripts/pretrain_csa.py \\
        --model-path /YOUR_PATH/Qwen2.5-7B-Instruct \\
        --fineweb-path /mnt/data/.../HuggingFaceFW___fineweb-edu/sample-10BT \\
        --max-groups 5000 --epochs 1 \\
        --n-chunks 4 --chunk-tokens 1024 \\
        --output-dir ./out/csa_pretrain
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
from src.cross_batch.multi_layer_hook import MultiLayerCSAModule, default_layer_indices
from src.cross_batch.pretrain_trainer import CSAPretrainer
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
    p.add_argument("--csa-every", type=int, default=4,
                   help="Insert CSA every N layers (default: 4)")
    p.add_argument("--csa-init-from", default=None)
    p.add_argument("--resume-from", default=None,
                   help="Resume full checkpoint (model + CSA) from path")

    # Optim
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--base-lr", type=float, default=1e-5)
    p.add_argument("--csa-lr", type=float, default=5e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--no-8bit-adam", action="store_true",
                   help="Disable 8-bit AdamW (use fp32 AdamW)")
    p.add_argument("--no-grad-checkpoint", action="store_true",
                   help="Disable gradient checkpointing")

    # Logging / saving
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

    if not args.no_grad_checkpoint:
        # use_reentrant=False is required when using forward hooks together
        # with gradient checkpointing — reentrant mode replays forward and
        # double-fires hooks.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # HF requires this when grad ckpt is on with embeddings frozen, but
        # for full FT we have all params trainable so embeddings are too.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if is_main:
            log.info("Gradient checkpointing enabled (use_reentrant=False)")

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    layer_indices = default_layer_indices(num_layers, every=args.csa_every)
    if is_main:
        log.info(
            "CSA layer indices (every %d, num_layers=%d): %s",
            args.csa_every, num_layers, layer_indices,
        )

    csa = CrossSequenceAttentionV2(
        hidden_size=hidden_size, num_heads=args.num_heads,
        use_gate=True, adaptive_top_k=True,
    )
    if args.csa_init_from:
        if is_main:
            log.info("Init CSA from %s", args.csa_init_from)
        ckpt = torch.load(args.csa_init_from, map_location="cpu")
        csa_state = ckpt.get("cross_batch_module", ckpt.get("csa", ckpt))
        # If loaded ckpt is a MultiLayerCSAModule state, it has csa.* prefix.
        # Strip it so we load into the inner CSA only.
        stripped = {}
        for k, v in csa_state.items():
            if k.startswith("csa."):
                stripped[k[len("csa."):]] = v
            else:
                stripped[k] = v
        missing, unexpected = csa.load_state_dict(stripped, strict=False)
        if is_main and (missing or unexpected):
            log.warning("CSA init missing=%s unexpected=%s", missing, unexpected)

    csa_module = MultiLayerCSAModule(csa=csa, layer_indices=layer_indices)
    csa_module.register(model)

    trainer = CSAPretrainer(
        model=model, tokenizer=tokenizer, csa_module=csa_module,
        device=device,
        base_lr=args.base_lr, csa_lr=args.csa_lr,
        local_rank=local_rank,
        n_chunks=args.n_chunks, chunk_tokens=args.chunk_tokens,
        warmup_steps=args.warmup_steps,
        use_8bit_adam=not args.no_8bit_adam,
    )

    if args.resume_from:
        if is_main:
            log.info("Resuming from %s", args.resume_from)
        trainer.load_checkpoint(args.resume_from)

    if is_main:
        log.info(
            "Streaming FineWeb-Edu (max=%d / rank, n_chunks=%d, chunk_tokens=%d)",
            args.max_groups, args.n_chunks, args.chunk_tokens,
        )
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
        log.info("Done. Saved to %s", args.output_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
