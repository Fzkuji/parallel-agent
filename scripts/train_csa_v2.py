#!/usr/bin/env python3
"""Train CSA-v2 (question-aware routing + adaptive top-k + contrastive aux loss).

Differences vs original CSA:
- Routing (Q, K) is computed from a fixed question embedding (mean-pool of the
  prompt's last-layer hidden), not the current decode-step hidden.
- Adaptive top-k: keeps max(1, ceil((G-1)/2)) cross-sequence connections.
- Gate bias = 0 (not -3); avoids the early-training stall.
- Optional InfoNCE contrastive loss on the routing projections (q_proj, k_proj),
  driven by context_ids, gives the routing path immediate gradient even with
  out_proj=0 init.
- Optional evidence dropout: with `--evidence-dropout-prob` probability, each
  sample's context is randomly truncated to `--evidence-dropout-keep` ratio,
  forcing the model to use peer questions through CSA.
"""

import argparse
import json
import logging
import os
import sys
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cross_batch.attention import CrossSequenceAttentionV2
from src.cross_batch.trainer import (
    CrossBatchTrainer,
    SQuADGroupedDataset,
    multi_context_collate_fn,
)
from src.datasets.squad import load_squad_groups

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/root/autodl-fs/models/Qwen2.5-7B-Instruct")
    p.add_argument("--dataset", default="squad", choices=["squad"])
    p.add_argument("--num-train-contexts", type=int, default=2000)
    p.add_argument("--questions-per-context", type=int, default=4)
    p.add_argument("--contexts-per-batch", type=int, default=2,
                   help="how many context groups per training step (multi-context batching)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--evidence-dropout-prob", type=float, default=0.3)
    p.add_argument("--evidence-dropout-keep", type=float, default=0.5)
    p.add_argument("--aux-contrast-lambda", type=float, default=0.1)
    p.add_argument("--aux-contrast-temp", type=float, default=0.1)
    p.add_argument("--no-train-lm-head", action="store_true",
                   help="Freeze lm_head (default: train it together with CSA)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="/root/autodl-tmp/work/csa_v2")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    log.info("Loading tokenizer + model: %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    hidden_size = model.config.hidden_size

    log.info("Building CSA-v2 (hidden=%d, heads=%d, adaptive_top_k=True)",
             hidden_size, args.num_heads)
    csa = CrossSequenceAttentionV2(
        hidden_size=hidden_size,
        num_heads=args.num_heads,
        use_gate=True,
        adaptive_top_k=True,
    )
    csa_params = sum(p.numel() for p in csa.parameters())
    log.info("CSA-v2 params: %s", f"{csa_params:,}")

    log.info("Loading %s training groups...", args.dataset)
    train_groups = load_squad_groups(
        split="train",
        min_questions=args.questions_per_context,
        max_questions=args.questions_per_context,
        max_contexts=args.num_train_contexts,
        fixed_question_count=args.questions_per_context,
        seed=args.seed,
    )
    log.info("Got %d training context groups (Q/group=%d)",
             len(train_groups), args.questions_per_context)

    train_dataset = SQuADGroupedDataset(
        tokenizer=tokenizer,
        groups=train_groups,
        max_length=args.max_length,
        dataset_name=args.dataset,
        evidence_dropout_prob=args.evidence_dropout_prob,
        evidence_dropout_keep=args.evidence_dropout_keep,
    )

    trainer = CrossBatchTrainer(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=csa,
        device="cuda",
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        train_lm_head=not args.no_train_lm_head,
        train_lora=False,
        local_rank=-1,
        mix_layer=-1,
        aux_contrast_lambda=args.aux_contrast_lambda,
        aux_contrast_temp=args.aux_contrast_temp,
    )

    log.info("Starting training: epochs=%d, contexts_per_batch=%d, total_contexts=%d",
             args.epochs, args.contexts_per_batch, len(train_groups))
    history = trainer.train(
        train_dataset=train_dataset,
        num_epochs=args.epochs,
        batch_size=args.contexts_per_batch,
        max_length=args.max_length,
        save_dir=args.output_dir,
        grouped=True,
    )

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items()}, f, indent=2)
    log.info("Done. Saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
