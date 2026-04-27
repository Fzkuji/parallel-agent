"""CSA distillation trainer.

Each step:
  1. Oracle pass (no_grad): forward the full document (concat of all G chunks)
     once; extract last-layer hidden states at G "boundary" positions —
     specifically the last token of each chunk in the concatenated sequence.
     These are the "what each query should look like if it had all the info"
     targets.
  2. Student pass (with grad on CSA only): forward the G chunks as a batch of
     size G, take the last-token hidden of each chunk, run CSA cross-batch
     attention, get mixed_hidden [G, d].
  3. Loss = MSE(mixed_hidden, oracle_targets).

The base model and lm_head are completely frozen. Only CSA gets gradients.
"""

import logging
import os
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .attention import CrossSequenceAttentionV2

logger = logging.getLogger(__name__)


def _mean_pool_with_mask(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B, L, d], [B, L] -> [B, d]"""
    m = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


class CSADistillTrainer:
    """Trains CSA-v2 to match an oracle's cross-chunk hidden via MSE."""

    def __init__(
        self,
        model,
        tokenizer,
        cross_batch_module: CrossSequenceAttentionV2,
        device: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        local_rank: int = -1,
        n_chunks: int = 4,
        chunk_tokens: int = 1024,
        n_target_positions: int = 32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.local_rank = local_rank
        self.is_distributed = local_rank >= 0
        self.n_chunks = n_chunks
        self.chunk_tokens = chunk_tokens
        self.n_target_positions = n_target_positions

        # Freeze the entire model.
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.dtype = next(self.model.parameters()).dtype

        # Move CSA to device with matching dtype.
        self.csa = cross_batch_module.to(device=device, dtype=self.dtype)
        self.csa.train()

        # Wrap with DDP if distributed.
        if self.is_distributed:
            self.csa = DDP(
                self.csa, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False,
            )

        params = list(self.csa.parameters())
        self.optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = None

    @property
    def csa_module(self) -> CrossSequenceAttentionV2:
        return self.csa.module if self.is_distributed else self.csa

    def _build_oracle_inputs(self, per_chunk_ids: List[List[int]]):
        """Concat chunks into one long prompt; return input_ids and the
        per-chunk "end" token positions in the concatenated sequence.

        We accept token ids directly (not strings) — re-encoding decoded text
        is not length-stable, so the dataset produces ids and we use them as-is.
        """
        # Defensive: ensure all chunks share the same length (required to
        # batch them into a tensor below). If a stray short chunk slipped
        # through, raising here makes the bug obvious in the log.
        lens = {len(ids) for ids in per_chunk_ids}
        if len(lens) != 1:
            raise ValueError(f"chunks have inconsistent lengths: {sorted(lens)}")
        concat_ids = []
        chunk_end_positions = []  # 0-indexed token end of each chunk in concat
        for ids in per_chunk_ids:
            concat_ids.extend(ids)
            chunk_end_positions.append(len(concat_ids) - 1)
        return concat_ids, chunk_end_positions, per_chunk_ids

    def _build_student_inputs(self, per_chunk_ids: List[List[int]]):
        """Build a batched input of size G from per-chunk token id lists.

        Use right-padding here (not left) so the chunk's actual content sits at
        the start, and the last meaningful token position is len(ids)-1 per row
        (we don't pad chunks because they are equal length by construction).
        """
        # All chunks should be exactly chunk_tokens long. Stack directly.
        return torch.tensor(per_chunk_ids, dtype=torch.long)

    @torch.no_grad()
    def _compute_oracle_hiddens(
        self,
        concat_ids: List[int],
        chunk_end_positions: List[int],
    ) -> torch.Tensor:
        """Returns oracle hidden states at chunk-end positions, [G, d]."""
        ids = torch.tensor(concat_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        mask = torch.ones_like(ids)
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][0]  # [L, d]
        positions = torch.tensor(chunk_end_positions, dtype=torch.long, device=self.device)
        return last_hidden[positions]  # [G, d]

    def _compute_student_mixed(self, per_chunk_ids: List[List[int]]) -> torch.Tensor:
        """G chunks -> base_hidden -> CSA -> mixed_hidden, [G, d]."""
        ids = torch.tensor(per_chunk_ids, dtype=torch.long, device=self.device)  # [G, L]
        mask = torch.ones_like(ids)
        with torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]  # [G, L, d]
            # Each chunk's last token = last position (no padding).
            last_token_hidden = last_hidden[:, -1, :]  # [G, d]
            # Question embedding for routing: mean pool over the chunk.
            q_emb = _mean_pool_with_mask(last_hidden, mask)  # [G, d]
        last_token_hidden = last_token_hidden.detach().requires_grad_(False)
        q_emb = q_emb.detach().requires_grad_(False)
        mixed = self.csa(last_token_hidden, question_emb=q_emb)  # [G, d]
        return mixed

    def training_step(self, group: dict) -> dict:
        """One distillation step on one document group.

        group: {"chunks": [List[int] * G], "doc_id": ...} — chunks are token id
        lists, each of length self.chunk_tokens.
        """
        chunks = group["chunks"]
        if len(chunks) != self.n_chunks:
            return {"loss": None}

        concat_ids, chunk_end_positions, per_chunk_ids = self._build_oracle_inputs(chunks)
        oracle = self._compute_oracle_hiddens(concat_ids, chunk_end_positions)  # [G, d]
        mixed = self._compute_student_mixed(per_chunk_ids)  # [G, d]

        # Also record student baseline (no CSA) to monitor drift.
        with torch.no_grad():
            ids = torch.tensor(per_chunk_ids, dtype=torch.long, device=self.device)
            mask = torch.ones_like(ids)
            out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            base_hidden_last = out.hidden_states[-1][:, -1, :]  # [G, d]
            baseline_mse = F.mse_loss(base_hidden_last.float(), oracle.float()).item()

        loss = F.mse_loss(mixed.float(), oracle.float())
        return {
            "loss": loss,
            "loss_value": loss.item(),
            "baseline_mse": baseline_mse,
            "improvement": baseline_mse - loss.item(),
        }

    def train(
        self,
        groups: Iterable[dict],
        num_epochs: int = 1,
        save_dir: Optional[str] = None,
        rank: int = 0,
        log_every: int = 20,
        save_every: int = 500,
        total_groups: Optional[int] = None,
    ):
        """Iterate over groups, do one step per group."""
        is_main = rank == 0
        if is_main and save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Build LR scheduler with a rough total-step estimate.
        total_steps = (total_groups or 1000) * num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-6
        )

        best_improvement = float("-inf")
        history = {"loss": [], "baseline_mse": [], "improvement": []}

        for epoch in range(1, num_epochs + 1):
            if is_main:
                logger.info("=" * 50)
                logger.info("Epoch %d/%d", epoch, num_epochs)
                logger.info("=" * 50)

            running = {"loss": 0.0, "baseline_mse": 0.0, "improvement": 0.0, "n": 0}

            for step, group in enumerate(groups):
                metrics = self.training_step(group)
                if metrics["loss"] is None:
                    continue
                loss = metrics["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                params = (
                    list(self.csa_module.parameters())
                    if self.is_distributed
                    else list(self.csa.parameters())
                )
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                running["loss"] += metrics["loss_value"]
                running["baseline_mse"] += metrics["baseline_mse"]
                running["improvement"] += metrics["improvement"]
                running["n"] += 1

                if is_main and (step + 1) % log_every == 0:
                    n = running["n"]
                    logger.info(
                        "  step %d/?  loss=%.4f  baseline=%.4f  improve=%.4f",
                        step + 1,
                        running["loss"] / n,
                        running["baseline_mse"] / n,
                        running["improvement"] / n,
                    )

                if (
                    is_main
                    and save_dir
                    and (step + 1) % save_every == 0
                    and running["n"] > 0
                ):
                    avg_imp = running["improvement"] / running["n"]
                    if avg_imp > best_improvement:
                        best_improvement = avg_imp
                        self.save_checkpoint(os.path.join(save_dir, "best_model.pt"))
                        logger.info(
                            "  Saved best model (improvement=%.4f at step %d)",
                            best_improvement, step + 1,
                        )

            if running["n"] > 0:
                history["loss"].append(running["loss"] / running["n"])
                history["baseline_mse"].append(running["baseline_mse"] / running["n"])
                history["improvement"].append(running["improvement"] / running["n"])
                if is_main:
                    logger.info(
                        "Epoch %d - loss=%.4f  baseline=%.4f  improve=%.4f",
                        epoch,
                        history["loss"][-1],
                        history["baseline_mse"][-1],
                        history["improvement"][-1],
                    )

        if is_main and save_dir:
            self.save_checkpoint(os.path.join(save_dir, "final_model.pt"))

        return history

    def save_checkpoint(self, path: str):
        save_dict = {"cross_batch_module": self.csa_module.state_dict()}
        torch.save(save_dict, path)
        logger.info("Checkpoint saved: %s", path)
