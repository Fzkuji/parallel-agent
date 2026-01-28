#!/usr/bin/env python3
"""Inspect Cross-Batch checkpoint contents."""

import sys
import torch

if len(sys.argv) < 2:
    print("Usage: python inspect_checkpoint.py <checkpoint_path>")
    sys.exit(1)

ckpt_path = sys.argv[1]
print(f"Loading checkpoint: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location='cpu')

print("\n=== Checkpoint Keys ===")
print(list(ckpt.keys()))

print("\n=== Config ===")
config = ckpt.get('config', {})
for key, value in sorted(config.items()):
    print(f"  {key}: {value}")

print("\n=== Module Sizes ===")
if 'cross_batch_module' in ckpt:
    csa_params = sum(p.numel() for p in ckpt['cross_batch_module'].values())
    print(f"  CSA module: {csa_params:,} parameters")

if 'lm_head' in ckpt:
    lm_params = sum(p.numel() for p in ckpt['lm_head'].values())
    print(f"  LM head: {lm_params:,} parameters")

if 'lora' in ckpt:
    lora_params = sum(p.numel() for p in ckpt['lora'].values())
    print(f"  LoRA: {lora_params:,} parameters")
    print(f"  LoRA keys: {len(ckpt['lora'])} tensors")
    print(f"  Sample LoRA keys: {list(ckpt['lora'].keys())[:5]}")

print("\n=== Summary ===")
print(f"Has CSA: {'cross_batch_module' in ckpt}")
print(f"Has LM head: {'lm_head' in ckpt}")
print(f"Has LoRA: {'lora' in ckpt}")
print(f"train_lm_head: {config.get('train_lm_head')}")
print(f"use_lora: {config.get('use_lora')}")
