"""Full-FT pretraining of CSA + base model on chunked documents.

Setup:
- A long document is split into G non-overlapping chunks of L tokens each.
- The G chunks are forwarded as a batch of size G through the base model.
- A shared CSA module fires at multiple layer boundaries (every 4 layers
  by default) via forward hooks, allowing each chunk's hidden to be
  influenced by the others.
- Loss is standard next-token CE on every position. Both the base model
  parameters AND the CSA module are trainable.

Why this is "regular pretraining":
- Same loss as base LLM pretraining (next-token CE).
- No oracle / no MSE / no two-pass distillation.
- The only architectural difference is that CSA is in the residual stream.
"""

from __future__ import annotations

import json
import logging
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
        # cosine from 1.0 -> 0.0
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


class CSAPretrainer:
    """Full-FT pretraining: base model + CSA both trainable.

    The trainer expects a `MultiLayerCSAModule` already constructed and
    registered against the model's decoder layers. It does not register
    hooks itself — caller controls hook lifetime so eval / Step 2 reuse
    the same registration code path.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        csa_module: MultiLayerCSAModule,
        device: str,
        base_lr: float = 1e-5,
        csa_lr: float = 5e-4,
        weight_decay: float = 0.01,
        local_rank: int = -1,
        n_chunks: int = 4,
        chunk_tokens: int = 1024,
        warmup_steps: int = 100,
        use_8bit_adam: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.csa_module = csa_module
        self.device = device
        self.local_rank = local_rank
        self.is_distributed = local_rank >= 0
        self.n_chunks = n_chunks
        self.chunk_tokens = chunk_tokens
        self.warmup_steps = warmup_steps

        # All base model params trainable
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train()

        # Match CSA dtype to base model
        self.dtype = next(self.model.parameters()).dtype
        self.csa_module.to(device=device, dtype=self.dtype)
        self.csa_module.train()

        # DDP wrap. Wrap each separately so optimizer can see distinct param
        # groups; alternative would be a single nn.Module owning both.
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

        # Optimizer with two LR groups. base lower (1e-5), CSA higher (5e-4).
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

        self.scheduler = None  # built in train()

    def _unwrapped_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _unwrapped_csa(self) -> MultiLayerCSAModule:
        return self.csa_module.module if isinstance(self.csa_module, DDP) else self.csa_module

    def training_step(self, group: dict) -> dict:
        """One step on one document group.

        group: {"chunks": [List[int] * G], "doc_id": ...}
            chunks: G token-id lists, each of length self.chunk_tokens.
        """
        chunks = group["chunks"]
        if len(chunks) != self.n_chunks:
            return {"loss": None}

        # Sanity: same length
        lens = {len(c) for c in chunks}
        if len(lens) != 1:
            raise ValueError(f"chunks have inconsistent lengths: {sorted(lens)}")

        ids = torch.tensor(chunks, dtype=torch.long, device=self.device)  # [G, L]
        attention_mask = torch.ones_like(ids)

        # Provide context to CSA hooks. No explicit question_emb during
        # pretraining — let hooks use mean-pool of current hidden as q_emb.
        self._unwrapped_csa().set_context(
            question_emb=None,
            attention_mask=attention_mask,
        )

        # HF auto-shifts labels: predicts token i from positions < i. Pad
        # positions don't exist here (chunks are exact length), so labels=ids.
        outputs = self.model(
            input_ids=ids,
            attention_mask=attention_mask,
            labels=ids,
        )

        self._unwrapped_csa().clear_context()

        return {"loss": outputs.loss}

    def train(
        self,
        groups: Iterable[dict],
        num_epochs: int = 1,
        save_dir: Optional[str] = None,
        rank: int = 0,
        log_every: int = 20,
        save_every: int = 500,
        total_groups: Optional[int] = None,
        grad_clip: float = 1.0,
    ):
        is_main = rank == 0
        if is_main and save_dir:
            os.makedirs(save_dir, exist_ok=True)

        total_steps = (total_groups or 1000) * num_epochs
        self.scheduler = _cosine_with_warmup(
            self.optimizer, total_steps=max(total_steps, 1), warmup_steps=self.warmup_steps,
        )

        history = {"loss": [], "lr_base": [], "lr_csa": []}
        best_loss = float("inf")
        running = {"loss": 0.0, "n": 0}
        global_step = 0

        for epoch in range(1, num_epochs + 1):
            if is_main:
                logger.info("=" * 50)
                logger.info("Epoch %d/%d", epoch, num_epochs)
                logger.info("=" * 50)

            for step, group in enumerate(groups):
                metrics = self.training_step(group)
                if metrics["loss"] is None:
                    continue
                loss = metrics["loss"]

                self.optimizer.zero_grad()
                loss.backward()

                # Clip across both groups together (simpler than per-group).
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
                        lr_base,
                        lr_csa,
                        alphas,
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
        """Save full model + CSA state. Note: this is ~14GB for 7B bf16."""
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
