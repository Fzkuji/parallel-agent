#!/usr/bin/env python3
"""Unified CSA-v2 training script with DDP + warm-up + LoRA support.

Two datasets supported via --dataset:
  - squad      (warm-up phase: same context, multiple questions, no distractors)
  - dhotpot    (target phase: distributed-context multi-hop with distractors)

DDP usage (8x H20 GPUs, one model copy per GPU):
    torchrun --standalone --nproc_per_node=8 scripts/train_csa.py \\
        --dataset dhotpot --num-train-groups 4000 --epochs 3 \\
        --contexts-per-batch 1 --output-dir ./out/dhotpot_csa

Single-GPU usage (no torchrun):
    python scripts/train_csa.py --dataset squad ...

Warm-up -> finetune chain:
    # 1. warm-up CSA on SQuAD (lm_head frozen, no LoRA)
    torchrun ... train_csa.py --dataset squad --no-train-lm-head \\
        --epochs 2 --output-dir ./out/csa_warmup
    # 2. finetune on dHotpot, init from warm-up
    torchrun ... train_csa.py --dataset dhotpot \\
        --csa-init-from ./out/csa_warmup/best_model.pt \\
        --epochs 3 --output-dir ./out/dhotpot_csa_warm
"""

import argparse
import json
import logging
import os
import sys
import random

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cross_batch.attention import CrossSequenceAttentionV2
from src.cross_batch.trainer import CrossBatchTrainer, SQuADGroupedDataset
from src.datasets import load_distributed_hotpot_groups, load_squad_groups

logging.basicConfig(level=logging.INFO, format="%(asctime)s [r%(name)s] %(message)s")
log = logging.getLogger("0")


def parse_args():
    p = argparse.ArgumentParser()

    # Model + tokenizer
    p.add_argument("--model-path", required=True, help="local model directory or HF id")

    # Dataset
    p.add_argument("--dataset", choices=["squad", "dhotpot"], required=True)
    p.add_argument("--num-train-groups", type=int, default=2000)
    p.add_argument("--questions-per-group", type=int, default=4,
                   help="for squad: questions sharing the context")
    p.add_argument("--n-agents", type=int, default=4,
                   help="for dhotpot: G queries per group, all asking the same question")
    p.add_argument("--paragraphs-per-agent", type=int, default=9,
                   help="for dhotpot: paragraphs in each query's context")
    p.add_argument("--evidence-dropout-prob", type=float, default=0.0)
    p.add_argument("--max-length", type=int, default=2048)

    # Training
    p.add_argument("--contexts-per-batch", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)

    # CSA
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--csa-init-from", type=str, default=None,
                   help="path to a checkpoint (best_model.pt) to initialize CSA weights from (warm-up)")
    p.add_argument("--aux-contrast-lambda", type=float, default=0.1)
    p.add_argument("--aux-contrast-temp", type=float, default=0.1)

    # What to train
    p.add_argument("--mode", choices=["csa", "finetuned"], default="csa")
    p.add_argument("--no-train-lm-head", action="store_true")
    p.add_argument("--lora-rank", type=int, default=0)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", type=str, default="q_proj,k_proj,v_proj,o_proj")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def setup_distributed():
    """Initialize torch.distributed. Returns (rank, world_size, local_rank)."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, -1  # single GPU, no DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_dataset_groups(args, world_size: int, rank: int):
    if args.dataset == "squad":
        # Warm-up: same context, multiple questions, no distractors. Simple
        # collaborative pattern lets CSA learn basic routing without noise.
        groups = load_squad_groups(
            split="train",
            min_questions=args.questions_per_group,
            max_questions=args.questions_per_group,
            max_contexts=args.num_train_groups,
            fixed_question_count=args.questions_per_group,
            seed=args.seed,
        )
    elif args.dataset == "dhotpot":
        groups = load_distributed_hotpot_groups(
            split="train",
            n_agents=args.n_agents,
            paragraphs_per_agent=args.paragraphs_per_agent,
            max_groups=args.num_train_groups,
            seed=args.seed,
            only_bridge=True,
            require_min_supporting=2,
            cross_question_distractor_pool=True,
        )
    else:
        raise ValueError(args.dataset)
    return groups


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    global log
    log = logging.getLogger(f"{rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    if is_main:
        log.info("DDP: rank=%d world_size=%d local_rank=%d", rank, world_size, local_rank)
        log.info("Loading %s", args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # When using DDP we want one full model copy per GPU. Load to the local
    # device explicitly (no device_map="auto", which would try to shard).
    if local_rank >= 0:
        device = f"cuda:{local_rank}"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
    else:
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
    hidden_size = model.config.hidden_size

    # LoRA wrap (optional)
    use_lora = args.lora_rank > 0
    if use_lora:
        targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        if is_main:
            log.info("LoRA: rank=%d alpha=%d targets=%s", args.lora_rank, args.lora_alpha, targets)
        lora_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            target_modules=targets, lora_dropout=args.lora_dropout,
            bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        if is_main:
            model.print_trainable_parameters()

    # CSA module
    csa = CrossSequenceAttentionV2(
        hidden_size=hidden_size, num_heads=args.num_heads,
        use_gate=True, adaptive_top_k=True,
    )
    if args.csa_init_from:
        if is_main:
            log.info("Init CSA from %s", args.csa_init_from)
        ckpt = torch.load(args.csa_init_from, map_location="cpu")
        missing, unexpected = csa.load_state_dict(ckpt["cross_batch_module"], strict=False)
        if is_main and (missing or unexpected):
            log.warning("CSA init missing=%s unexpected=%s", missing, unexpected)

    if args.mode == "finetuned":
        for p_ in csa.parameters():
            p_.requires_grad = False
        if is_main:
            log.info("Finetuned mode: CSA frozen, only lm_head%s trains",
                     " + LoRA" if use_lora else "")

    # Dataset
    if is_main:
        log.info("Loading %s training groups (max=%d)", args.dataset, args.num_train_groups)
    groups = load_dataset_groups(args, world_size, rank)
    if is_main:
        log.info("Got %d groups", len(groups))

    train_dataset = SQuADGroupedDataset(
        tokenizer=tokenizer, groups=groups,
        max_length=args.max_length,
        dataset_name="hotpot" if args.dataset == "dhotpot" else "squad",
        evidence_dropout_prob=args.evidence_dropout_prob,
    )

    # Trainer
    trainer = CrossBatchTrainer(
        model=model, tokenizer=tokenizer, cross_batch_module=csa,
        device=device, learning_rate=args.lr, weight_decay=0.01,
        train_lm_head=not args.no_train_lm_head,
        train_lora=use_lora,
        local_rank=local_rank,
        mix_layer=-1,
        aux_contrast_lambda=args.aux_contrast_lambda if args.mode == "csa" else 0.0,
        aux_contrast_temp=args.aux_contrast_temp,
    )

    trainer.train(
        train_dataset=train_dataset,
        num_epochs=args.epochs,
        batch_size=args.contexts_per_batch,
        max_length=args.max_length,
        save_dir=args.output_dir if is_main else None,
        grouped=True,
        distributed=(local_rank >= 0),
        rank=rank,
        world_size=world_size,
    )

    if is_main:
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        log.info("Done. Saved to %s", args.output_dir)

    cleanup_distributed()


if __name__ == "__main__":
    main()
