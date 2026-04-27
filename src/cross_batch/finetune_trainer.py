"""Full-FT finetuning of base model + CSA on Q&A grouped data (e.g. dHotpot).

Mirror of CSAPretrainer (next-token CE + multi-layer CSA hooks + full FT)
but adapted for Q&A:
- Each batch is a "group" of G samples sharing context (or distributed
  context, like dHotpot). All G are forwarded together as batch=G so the
  CSA hooks have peers to attend to.
- Loss is computed only on answer tokens; prompt positions are masked
  with -100 so they don't contribute to CE.
- All G items in a group are padded to the same length (right-pad), so we
  can stack into [G, L].
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Iterable, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .multi_layer_hook import MultiLayerCSAModule

logger = logging.getLogger(__name__)


def _cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def collate_qa_group(
    group_examples: List[dict],
    tokenizer,
    max_length: int = 2048,
    device: Optional[str] = None,
) -> Optional[dict]:
    """Build [G, L] tensors for one group of Q&A samples.

    group_examples: list of {"prompt": str, "answer": str}, length G.

    Returns dict with input_ids, attention_mask, labels tensors of shape
    [G, L], where L = max prompt+answer length in the group (capped at
    max_length). Returns None if any sample is empty after tokenization.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    rows = []
    for ex in group_examples:
        p_ids = tokenizer.encode(ex["prompt"], add_special_tokens=False)
        a_ids = tokenizer.encode(ex["answer"], add_special_tokens=False)
        if not p_ids or not a_ids:
            return None
        # Truncate prompt from the LEFT so the question end (most relevant) survives.
        budget = max_length - len(a_ids)
        if budget <= 0:
            # Answer alone exceeds budget; skip group.
            return None
        if len(p_ids) > budget:
            p_ids = p_ids[-budget:]
        rows.append((p_ids, a_ids))

    L = max(len(p) + len(a) for p, a in rows)
    G = len(rows)

    input_ids = torch.full((G, L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((G, L), dtype=torch.long)
    labels = torch.full((G, L), -100, dtype=torch.long)

    for i, (p, a) in enumerate(rows):
        n = len(p) + len(a)
        input_ids[i, :n] = torch.tensor(p + a, dtype=torch.long)
        attention_mask[i, :n] = 1
        # Labels: -100 for prompt, actual ids for answer, -100 for pad.
        labels[i, len(p):n] = torch.tensor(a, dtype=torch.long)

    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class CSAFineTuner:
    """Full-FT Q&A finetuning with multi-layer CSA hooks.

    Caller must register CSA hooks on the model BEFORE constructing this
    trainer (so that the hooks survive into DDP wrap).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        csa_module: MultiLayerCSAModule,
        device: str,
        base_lr: float = 5e-6,
        csa_lr: float = 2e-4,
        weight_decay: float = 0.01,
        local_rank: int = -1,
        warmup_steps: int = 50,
        use_8bit_adam: bool = True,
        max_length: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.csa_module = csa_module
        self.device = device
        self.local_rank = local_rank
        self.is_distributed = local_rank >= 0
        self.warmup_steps = warmup_steps
        self.max_length = max_length

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train()

        self.dtype = next(self.model.parameters()).dtype
        self.csa_module.to(device=device, dtype=self.dtype)
        self.csa_module.train()

        if self.is_distributed:
            self.model = DDP(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
            self.csa_module = DDP(
                self.csa_module, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False,
            )

        base_params = [p for p in self._unwrapped_model().parameters() if p.requires_grad]
        csa_params = [p for p in self._unwrapped_csa().parameters() if p.requires_grad]
        param_groups = [
            {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": csa_params, "lr": csa_lr, "weight_decay": weight_decay},
        ]

        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(param_groups)
                logger.info("Using 8-bit AdamW (bitsandbytes)")
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to fp32 AdamW")
                self.optimizer = AdamW(param_groups)
        else:
            self.optimizer = AdamW(param_groups)

        self.scheduler = None

    def _unwrapped_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _unwrapped_csa(self) -> MultiLayerCSAModule:
        return self.csa_module.module if isinstance(self.csa_module, DDP) else self.csa_module

    def training_step(self, group_examples: List[dict]) -> dict:
        """One step on one group of Q&A samples (length G)."""
        if len(group_examples) < 2:
            # CSA needs >= 2 peers to attend across; skip degenerate groups.
            return {"loss": None}

        batch = collate_qa_group(
            group_examples, self.tokenizer,
            max_length=self.max_length, device=self.device,
        )
        if batch is None:
            return {"loss": None}

        # Provide attention mask context so hook ignores pad positions
        # when computing per-chunk summary.
        self._unwrapped_csa().set_context(
            question_emb=None,
            attention_mask=batch["attention_mask"],
        )

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        self._unwrapped_csa().clear_context()
        return {"loss": outputs.loss}

    def train(
        self,
        groups: Iterable[List[dict]],
        num_epochs: int = 1,
        save_dir: Optional[str] = None,
        rank: int = 0,
        log_every: int = 20,
        save_every: int = 500,
        total_groups: Optional[int] = None,
        grad_clip: float = 1.0,
        sampler=None,
    ):
        is_main = rank == 0
        if is_main and save_dir:
            os.makedirs(save_dir, exist_ok=True)

        groups_list = list(groups) if not hasattr(groups, "__len__") else groups
        per_epoch = len(groups_list) if hasattr(groups_list, "__len__") else (total_groups or 1000)
        total_steps = per_epoch * num_epochs
        self.scheduler = _cosine_with_warmup(
            self.optimizer, total_steps=max(total_steps, 1), warmup_steps=self.warmup_steps,
        )

        history = {"loss": [], "lr_base": [], "lr_csa": []}
        best_loss = float("inf")
        running = {"loss": 0.0, "n": 0}
        global_step = 0

        for epoch in range(1, num_epochs + 1):
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            if is_main:
                logger.info("=" * 50)
                logger.info("Epoch %d/%d", epoch, num_epochs)
                logger.info("=" * 50)

            for step, group in enumerate(groups_list):
                # `group` may be a list of examples directly, or a dict from
                # the SQuADGroupedDataset that wraps a list under a key.
                if isinstance(group, dict) and "examples" in group:
                    examples = group["examples"]
                elif isinstance(group, list):
                    examples = group
                else:
                    # Unknown shape — try to iterate
                    examples = list(group)

                metrics = self.training_step(examples)
                if metrics["loss"] is None:
                    continue
                loss = metrics["loss"]

                self.optimizer.zero_grad()
                loss.backward()

                params_to_clip = [
                    p for p in self._unwrapped_model().parameters() if p.requires_grad
                ] + [
                    p for p in self._unwrapped_csa().parameters() if p.requires_grad
                ]
                torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                running["loss"] += loss.item()
                running["n"] += 1
                global_step += 1

                if is_main and global_step % log_every == 0:
                    n = running["n"]
                    lr_base = self.optimizer.param_groups[0]["lr"]
                    lr_csa = self.optimizer.param_groups[1]["lr"]
                    alphas = self._unwrapped_csa().alphas_summary()
                    logger.info(
                        "step %d  loss=%.4f  lr_base=%.2e  lr_csa=%.2e  alphas=[%s]",
                        global_step,
                        running["loss"] / n,
                        lr_base, lr_csa, alphas,
                    )

                if (
                    is_main
                    and save_dir
                    and global_step % save_every == 0
                    and running["n"] > 0
                ):
                    avg_loss = running["loss"] / running["n"]
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        self.save_checkpoint(os.path.join(save_dir, "best_model.pt"))
                        logger.info(
                            "Saved best model (loss=%.4f at step %d)",
                            best_loss, global_step,
                        )

            if running["n"] > 0:
                history["loss"].append(running["loss"] / running["n"])
                history["lr_base"].append(self.optimizer.param_groups[0]["lr"])
                history["lr_csa"].append(self.optimizer.param_groups[1]["lr"])
                if is_main:
                    logger.info(
                        "Epoch %d done — avg_loss=%.4f", epoch, history["loss"][-1],
                    )

        if is_main and save_dir:
            self.save_checkpoint(os.path.join(save_dir, "final_model.pt"))
            with open(os.path.join(save_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

        return history

    def save_checkpoint(self, path: str):
        state = {
            "model": self._unwrapped_model().state_dict(),
            "csa_module": self._unwrapped_csa().state_dict(),
        }
        torch.save(state, path)
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str, strict: bool = False):
        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt:
            self._unwrapped_model().load_state_dict(ckpt["model"], strict=strict)
        if "csa_module" in ckpt:
            self._unwrapped_csa().load_state_dict(ckpt["csa_module"], strict=strict)
        logger.info("Checkpoint loaded: %s", path)
