#!/usr/bin/env python3
"""Step 2: Full-FT dHotpot finetuning of base model + multi-layer CSA.

Init from Step 1 checkpoint (base model + CSA already pretrained on
FineWeb-Edu chunked next-token prediction). Continues with the same
multi-layer CSA hook architecture, full FT, but on dHotpot Q&A loss
(CE on answer tokens only).

DDP usage:
    torchrun --standalone --nproc_per_node=8 scripts/finetune_csa.py \\
        --model-path /YOUR_PATH/Qwen2.5-7B-Instruct \\
        --resume-from ./out/csa_pretrain/best_model.pt \\
        --num-train-groups 4000 --epochs 3 \\
        --n-agents 4 --paragraphs-per-agent 9 \\
        --output-dir ./out/dhotpot_csa_pretrained
"""

import argparse
import json
import logging
import os
import random
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cross_batch.attention import CrossSequenceAttentionV2
from src.cross_batch.multi_layer_hook import MultiLayerCSAModule, default_layer_indices
from src.cross_batch.finetune_trainer import CSAFineTuner
from src.cross_batch.trainer import SQuADGroupedDataset
from src.datasets import load_distributed_hotpot_groups, load_squad_groups

logging.basicConfig(level=logging.INFO, format="%(asctime)s [r%(name)s] %(message)s")
log = logging.getLogger("0")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--resume-from", default=None,
                   help="Step 1 checkpoint to init base model + CSA from")

    # Dataset
    p.add_argument("--dataset", choices=["squad", "dhotpot"], default="dhotpot")
    p.add_argument("--num-train-groups", type=int, default=4000)
    p.add_argument("--questions-per-group", type=int, default=4)
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--paragraphs-per-agent", type=int, default=9)
    p.add_argument("--max-length", type=int, default=2048)

    # CSA
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--csa-every", type=int, default=4)

    # Optim
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--base-lr", type=float, default=5e-6)
    p.add_argument("--csa-lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--no-8bit-adam", action="store_true")
    p.add_argument("--no-grad-checkpoint", action="store_true")

    # Logging
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


def load_groups(args):
    if args.dataset == "squad":
        return load_squad_groups(
            split="train",
            min_questions=args.questions_per_group,
            max_questions=args.questions_per_group,
            max_contexts=args.num_train_groups,
            fixed_question_count=args.questions_per_group,
            seed=args.seed,
        )
    return load_distributed_hotpot_groups(
        split="train",
        n_agents=args.n_agents,
        paragraphs_per_agent=args.paragraphs_per_agent,
        max_groups=args.num_train_groups,
        seed=args.seed,
        only_bridge=True,
        require_min_supporting=2,
        cross_question_distractor_pool=True,
    )


class _GroupIter:
    """Adapter: iterate a SQuADGroupedDataset under a DistributedSampler.

    Yields lists of {"prompt", "answer", "full_text"} dicts (one group per step).
    """
    def __init__(self, dataset, sampler):
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        for idx in self.sampler:
            yield self.dataset[idx]

    def __len__(self):
        return len(self.sampler)


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

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

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
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if is_main:
            log.info("Gradient checkpointing enabled")

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    layer_indices = default_layer_indices(num_layers, every=args.csa_every)
    if is_main:
        log.info("CSA layer indices: %s", layer_indices)

    csa = CrossSequenceAttentionV2(
        hidden_size=hidden_size, num_heads=args.num_heads,
        use_gate=True, adaptive_top_k=True,
    )
    csa_module = MultiLayerCSAModule(csa=csa, layer_indices=layer_indices)
    csa_module.register(model)

    trainer = CSAFineTuner(
        model=model, tokenizer=tokenizer, csa_module=csa_module,
        device=device,
        base_lr=args.base_lr, csa_lr=args.csa_lr,
        local_rank=local_rank,
        warmup_steps=args.warmup_steps,
        use_8bit_adam=not args.no_8bit_adam,
        max_length=args.max_length,
    )

    if args.resume_from:
        if is_main:
            log.info("Loading Step 1 checkpoint: %s", args.resume_from)
        trainer.load_checkpoint(args.resume_from)

    if is_main:
        log.info("Loading %s training groups", args.dataset)
    groups = load_groups(args)
    if is_main:
        log.info("Got %d groups", len(groups))

    train_dataset = SQuADGroupedDataset(
        tokenizer=tokenizer, groups=groups,
        max_length=args.max_length,
        dataset_name="hotpot" if args.dataset == "dhotpot" else "squad",
        evidence_dropout_prob=0.0,
    )

    sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if local_rank >= 0 else None
    )

    iterable = (
        _GroupIter(train_dataset, sampler) if sampler is not None
        else (train_dataset[i] for i in range(len(train_dataset)))
    )

    history = trainer.train(
        groups=iterable,
        num_epochs=args.epochs,
        save_dir=args.output_dir if is_main else None,
        rank=rank,
        log_every=args.log_every,
        save_every=args.save_every,
        total_groups=len(train_dataset),
        sampler=sampler,
    )

    if is_main:
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        log.info("Done. Saved to %s", args.output_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
