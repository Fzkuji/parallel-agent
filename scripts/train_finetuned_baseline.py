#!/usr/bin/env python3
"""Finetuned baseline: same training setup as CSA-v2 but with the CSA module
disabled (use_gate=True but adaptive_top_k=False AND we never call its forward
in the generator path -- only lm_head moves).

This isolates the contribution of CSA from the lm_head fine-tuning. Use the
same data, hyperparameters, and number of training steps as train_csa_v2.py.

Implementation: pass a tiny CSA module that doesn't actually do cross-batch
mixing -- we re-use SimpleCrossBatchGate with bias=-30 (sigmoid ~ 0) which
keeps the cross-batch contribution at machine zero, so only lm_head trains.
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cross_batch.attention import CrossSequenceAttentionV2
from src.cross_batch.trainer import CrossBatchTrainer, SQuADGroupedDataset
from src.datasets.squad import load_squad_groups

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/root/autodl-fs/models/Qwen2.5-7B-Instruct")
    p.add_argument("--num-train-contexts", type=int, default=1000)
    p.add_argument("--questions-per-context", type=int, default=4)
    p.add_argument("--contexts-per-batch", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="/root/autodl-tmp/work/finetuned_baseline")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    log.info("Loading %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    hidden_size = model.config.hidden_size

    # CSA module exists for trainer plumbing but is suppressed: we keep
    # out_proj at zero (default init) AND freeze its parameters so only
    # lm_head receives gradients. The trainer still calls it each step,
    # but its contribution is exactly zero.
    csa_dummy = CrossSequenceAttentionV2(hidden_size=hidden_size, num_heads=8, use_gate=True)
    for p in csa_dummy.parameters():
        p.requires_grad = False

    log.info("Loading SQuAD train groups...")
    groups = load_squad_groups(
        split="train",
        min_questions=args.questions_per_context,
        max_questions=args.questions_per_context,
        max_contexts=args.num_train_contexts,
        fixed_question_count=args.questions_per_context,
        seed=args.seed,
    )
    log.info("Got %d groups", len(groups))

    train_dataset = SQuADGroupedDataset(
        tokenizer=tokenizer, groups=groups, max_length=args.max_length, dataset_name="squad",
        evidence_dropout_prob=0.0,
    )

    trainer = CrossBatchTrainer(
        model=model, tokenizer=tokenizer, cross_batch_module=csa_dummy,
        device="cuda", learning_rate=args.lr, weight_decay=0.01,
        train_lm_head=True, train_lora=False, local_rank=-1, mix_layer=-1,
        aux_contrast_lambda=0.0,
    )
    trainer.train(
        train_dataset=train_dataset, num_epochs=args.epochs,
        batch_size=args.contexts_per_batch, max_length=args.max_length,
        save_dir=args.output_dir, grouped=True,
    )
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log.info("Done. Saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
