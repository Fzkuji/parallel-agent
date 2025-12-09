"""
Fine-tuning script for cross-batch interaction module.

The cross-batch module is trained to improve generation quality by learning
to share relevant information between samples in the same batch.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm

from .generator import CrossBatchGenerator
from .attention import CrossBatchAttention, CrossBatchEmbeddingMixer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_instruct_model(model_name_or_tokenizer) -> bool:
    """Check if a model is an instruct/chat model based on its name or tokenizer."""
    if hasattr(model_name_or_tokenizer, 'name_or_path'):
        name = model_name_or_tokenizer.name_or_path.lower()
    else:
        name = str(model_name_or_tokenizer).lower()

    instruct_keywords = ['instruct', 'chat', 'it', 'rlhf', 'dpo', 'sft']
    return any(kw in name for kw in instruct_keywords)


class SQuADDataset(Dataset):
    """SQuAD dataset for training cross-batch module."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_samples: Optional[int] = None,
        max_length: int = 512,
        use_chat_template: bool = None,  # None = auto-detect
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Auto-detect whether to use chat template
        if use_chat_template is None:
            self.use_chat_template = is_instruct_model(tokenizer) and hasattr(tokenizer, 'apply_chat_template')
        else:
            self.use_chat_template = use_chat_template

        # Load dataset
        dataset = load_dataset("squad", split=split)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.examples = []
        for item in dataset:
            prompt = self._format_prompt(item["context"], item["question"])
            answer = item["answers"]["text"][0] if item["answers"]["text"] else ""
            self.examples.append({
                "prompt": prompt,
                "answer": answer,
                "full_text": prompt + answer,  # No space for chat template
            })

    def _format_prompt(self, context: str, question: str) -> str:
        if self.use_chat_template:
            # Use chat template for instruct models
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer the question based only on the given context. Give a short, direct answer without explanation."
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer with only the answer, nothing else."
                }
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        else:
            # Simple completion format for base models
            return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch: List[Dict], tokenizer: PreTrainedTokenizer, max_length: int = 512):
    """Collate function for DataLoader."""
    prompts = [item["prompt"] for item in batch]
    answers = [item["answer"] for item in batch]
    full_texts = [item["full_text"] for item in batch]

    # Tokenize prompts (for input)
    prompt_encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # Tokenize full texts (for labels)
    full_encodings = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    return {
        "input_ids": prompt_encodings["input_ids"],
        "attention_mask": prompt_encodings["attention_mask"],
        "labels": full_encodings["input_ids"],
        "labels_attention_mask": full_encodings["attention_mask"],
        "answers": answers,
    }


def _resolve_base_model(model: PreTrainedModel) -> nn.Module:
    """Return the transformer block so we can run it without the lm_head."""
    candidate_attrs = ("model", "transformer", "decoder", "base_model")
    for attr in candidate_attrs:
        module = getattr(model, attr, None)
        if module is not None:
            return module
    raise ValueError("Could not find base transformer module on the model")


def _forward_hidden_states(
    base_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Optional[tuple] = None,
    use_cache: bool = False,
) -> tuple:
    """Run the frozen base model and return its hidden states and optionally KV cache.

    Returns:
        Tuple of (hidden_states, past_key_values) where:
        - hidden_states: List of hidden states from each layer
        - past_key_values: KV cache for next step (None if use_cache=False)
    """
    try:
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=use_cache,
            return_dict=True,
        )
    except TypeError:
        # Some base models might not accept use_cache or past_key_values
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else None
    if hidden_states is None and isinstance(outputs, tuple) and len(outputs) >= 3:
        hidden_states = outputs[2]
    if hidden_states is None:
        raise ValueError("Base model did not return hidden states")

    new_past_key_values = outputs.past_key_values if hasattr(outputs, "past_key_values") else None
    return hidden_states, new_past_key_values


def _clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Detach module weights to CPU so we can restore best states later."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


class LMHeadOnlyTrainer:
    """
    Trainer for fine-tuning ONLY the lm_head (baseline, no cross-batch).

    This is used as a fair baseline comparison to verify whether
    improvements come from cross-batch interaction or just lm_head finetuning.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model_dtype = next(model.parameters()).dtype
        self.base_model = _resolve_base_model(self.model)

        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Unfreeze and train only lm_head
        if hasattr(self.model, 'lm_head'):
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            trainable_params = list(self.model.lm_head.parameters())
            logger.info("Training lm_head only (no cross-batch)")
        else:
            raise ValueError("Model does not have lm_head")

        self.optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = None
        self.best_lm_head_state = None

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        labels_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss WITHOUT cross-batch interaction."""
        batch_size = input_ids.size(0)
        prompt_lengths = attention_mask.sum(dim=1)
        answer_lengths = labels_attention_mask.sum(dim=1) - prompt_lengths
        max_answer_len = answer_lengths.max().item()

        total_loss = 0.0
        num_tokens = 0

        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()
        max_gen_steps = min(max_answer_len, 32)

        # First step: process the full prompt and get KV cache
        past_key_values = None
        with torch.no_grad():
            hidden_states, past_key_values = _forward_hidden_states(
                self.base_model,
                current_ids,
                current_mask,
                past_key_values=None,
                use_cache=True,
            )
            seq_lengths = current_mask.sum(dim=1) - 1
            last_hidden = hidden_states[-1][torch.arange(batch_size, device=self.device), seq_lengths]

        for step in range(max_gen_steps):
            # NO cross-batch mixing - just use lm_head directly
            logits = self.model.lm_head(last_hidden)

            target_positions = prompt_lengths + step
            valid_mask = target_positions < labels.size(1)
            target_positions_clamped = target_positions.clamp(max=labels.size(1) - 1)
            target_tokens = labels[torch.arange(batch_size, device=self.device), target_positions_clamped]

            if valid_mask.any():
                valid_logits = logits[valid_mask]
                valid_targets = target_tokens[valid_mask]
                non_pad_mask = valid_targets != self.tokenizer.pad_token_id
                if non_pad_mask.any():
                    step_loss = F.cross_entropy(
                        valid_logits[non_pad_mask],
                        valid_targets[non_pad_mask],
                    )
                    total_loss = total_loss + step_loss
                    num_tokens += non_pad_mask.sum().item()

            # Early stopping
            all_finished = (~valid_mask).all() or (valid_mask.any() and non_pad_mask.sum() == 0)
            if all_finished:
                break

            # Prepare next step with KV cache
            next_tokens = target_tokens.unsqueeze(1)
            next_mask = valid_mask.long().unsqueeze(1)
            current_mask = torch.cat([current_mask, next_mask], dim=1)

            with torch.no_grad():
                hidden_states, past_key_values = _forward_hidden_states(
                    self.base_model,
                    next_tokens,
                    current_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                last_hidden = hidden_states[-1][:, -1, :]

        if num_tokens > 0:
            avg_loss = total_loss / num_tokens
        else:
            avg_loss = torch.tensor(0.0, device=self.device)

        return {"loss": avg_loss, "num_tokens": num_tokens}

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            labels_attention_mask = batch["labels_attention_mask"].to(self.device)

            self.optimizer.zero_grad()
            loss_dict = self.compute_loss(input_ids, attention_mask, labels, labels_attention_mask)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.lm_head.parameters(), 1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {"loss": total_loss / max(num_batches, 1)}

    def train(
        self,
        train_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 8,
        max_length: int = 512,
        save_dir: Optional[str] = "checkpoints_lmhead_only",
    ) -> Dict[str, Any]:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.tokenizer, max_length),
            drop_last=True,
        )

        total_steps = len(train_loader) * num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)

        history = {"train_loss": []}
        best_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{num_epochs} (LM Head Only)")
            logger.info(f"{'='*50}")

            metrics = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(metrics["loss"])
            logger.info(f"Epoch {epoch} - Loss: {metrics['loss']:.4f}")

            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                self.best_lm_head_state = _clone_state_dict(self.model.lm_head)
                if save_dir:
                    self.save_checkpoint(os.path.join(save_dir, "best_model.pt"))

        # Restore best state for in-memory evaluation
        if self.best_lm_head_state is not None:
            self.model.lm_head.load_state_dict(self.best_lm_head_state)

        if save_dir:
            self.save_checkpoint(os.path.join(save_dir, "final_model.pt"))
            with open(os.path.join(save_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=2)

        return history

    def save_checkpoint(self, path: str):
        """Save model checkpoint (only lm_head)."""
        save_dict = {"lm_head": self.model.lm_head.state_dict()}
        torch.save(save_dict, path)
        logger.info(f"Checkpoint saved to {path} (lm_head)")

    def load_checkpoint(self, path: str):
        """Load model checkpoint (only lm_head)."""
        checkpoint = torch.load(path, map_location=self.device)
        if "lm_head" in checkpoint and hasattr(self.model, 'lm_head'):
            self.model.lm_head.load_state_dict(checkpoint["lm_head"])
            logger.info(f"Checkpoint loaded from {path} (lm_head)")
        else:
            raise ValueError(f"Checkpoint {path} does not contain lm_head weights")


class CrossBatchTrainer:
    """
    Trainer for fine-tuning the cross-batch interaction module.

    Training objective: The cross-batch module should help each sample
    generate better answers by leveraging information from other samples.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cross_batch_module: nn.Module,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        train_lm_head: bool = True,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.train_lm_head = train_lm_head
        self.base_model = _resolve_base_model(self.model)

        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Get model dtype
        self.model_dtype = next(model.parameters()).dtype

        # Setup cross-batch module - use same dtype as model
        self.cross_batch_module = cross_batch_module.to(device=device, dtype=self.model_dtype)
        self.cross_batch_module.train()

        # Collect trainable parameters
        trainable_params = list(self.cross_batch_module.parameters())

        # Optionally train lm_head together
        if train_lm_head and hasattr(self.model, 'lm_head'):
            # Unfreeze lm_head (keep same dtype as model)
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            trainable_params.extend(self.model.lm_head.parameters())
            logger.info("Training cross-batch module + lm_head")
        else:
            logger.info("Training cross-batch module only")

        # Optimizer for trainable parameters
        self.optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = None
        self.best_cross_batch_state = None
        self.best_lm_head_state = None

    def _hidden_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Convert hidden states to logits."""
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head(hidden_states)
        elif hasattr(self.model, 'embed_out'):
            return self.model.embed_out(hidden_states)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            return F.linear(hidden_states, self.model.transformer.wte.weight)
        else:
            raise ValueError("Could not find output projection layer")

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        labels_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss over the FULL answer sequence with KV cache optimization.

        For each token position in the answer, we:
        1. Get hidden states from the model (using KV cache for efficiency)
        2. Apply cross-batch mixing
        3. Predict next token
        4. Accumulate loss

        Uses dynamic answer length based on actual labels, with early stopping
        when all samples have finished.
        """
        batch_size = input_ids.size(0)
        prompt_lengths = attention_mask.sum(dim=1)  # [batch]
        total_label_lengths = labels_attention_mask.sum(dim=1)  # [batch]
        max_answer_len = (total_label_lengths - prompt_lengths).max().item()

        # Get lm_head for computing logits
        if self.train_lm_head and hasattr(self.model, 'lm_head'):
            use_lm_head_module = True
        else:
            use_lm_head_module = False
            with torch.no_grad():
                if hasattr(self.model, 'lm_head'):
                    lm_weight = self.model.lm_head.weight
                    lm_bias = self.model.lm_head.bias if self.model.lm_head.bias is not None else None
                elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                    lm_weight = self.model.transformer.wte.weight
                    lm_bias = None
                else:
                    raise ValueError("Could not find output projection layer")

        total_loss = 0.0
        total_baseline_loss = 0.0
        num_tokens = 0

        # First step: process the full prompt and get KV cache
        past_key_values = None
        current_mask = attention_mask.clone()

        with torch.no_grad():
            hidden_states, past_key_values = _forward_hidden_states(
                self.base_model,
                input_ids,
                current_mask,
                past_key_values=None,
                use_cache=True,
            )
            # Get last token's hidden state from prompt
            seq_lengths = current_mask.sum(dim=1) - 1
            last_hidden = hidden_states[-1][torch.arange(batch_size, device=self.device), seq_lengths]

        # Generate answer tokens one by one using KV cache
        for step in range(max_answer_len):
            # Apply cross-batch interaction
            mixed_hidden = self.cross_batch_module(last_hidden)

            # Compute logits
            if use_lm_head_module:
                logits = self.model.lm_head(mixed_hidden)
            else:
                logits = F.linear(mixed_hidden, lm_weight, lm_bias)

            # Get target tokens from labels
            target_positions = prompt_lengths + step
            valid_mask = target_positions < labels.size(1)
            target_positions_clamped = target_positions.clamp(max=labels.size(1) - 1)
            target_tokens = labels[torch.arange(batch_size, device=self.device), target_positions_clamped]

            # Only compute loss for valid positions
            if valid_mask.any():
                valid_logits = logits[valid_mask]
                valid_targets = target_tokens[valid_mask]

                # Skip padding tokens
                non_pad_mask = valid_targets != self.tokenizer.pad_token_id
                if non_pad_mask.any():
                    step_loss = F.cross_entropy(
                        valid_logits[non_pad_mask],
                        valid_targets[non_pad_mask],
                    )
                    total_loss = total_loss + step_loss

                    # Baseline loss (without cross-batch mixing)
                    with torch.no_grad():
                        if use_lm_head_module:
                            baseline_logits = self.model.lm_head(last_hidden)
                        else:
                            baseline_logits = F.linear(last_hidden, lm_weight, lm_bias)
                        valid_baseline = baseline_logits[valid_mask]
                        baseline_step_loss = F.cross_entropy(
                            valid_baseline[non_pad_mask],
                            valid_targets[non_pad_mask],
                        )
                        total_baseline_loss = total_baseline_loss + baseline_step_loss.item()

                    num_tokens += non_pad_mask.sum().item()

            # Early stopping: all samples have finished
            all_finished = (~valid_mask).all() or (valid_mask.any() and non_pad_mask.sum() == 0)
            if all_finished:
                break

            # Prepare next step: use teacher forcing (ground truth token)
            next_token_ids = target_tokens.unsqueeze(1)  # [batch, 1]

            # Update attention mask for next step
            next_mask = valid_mask.long().unsqueeze(1)  # [batch, 1]
            # Full attention mask for KV cache (need to include all previous positions)
            current_mask = torch.cat([current_mask, next_mask], dim=1)

            # Forward pass with KV cache: only process the new token
            with torch.no_grad():
                hidden_states, past_key_values = _forward_hidden_states(
                    self.base_model,
                    next_token_ids,
                    current_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                # Hidden state for the new token is at position 0 (since we only passed 1 token)
                last_hidden = hidden_states[-1][:, -1, :]  # [batch, hidden]

        # Average loss over all tokens
        if num_tokens > 0:
            avg_loss = total_loss / num_tokens
            avg_baseline_loss = total_baseline_loss / num_tokens
        else:
            avg_loss = torch.tensor(0.0, device=self.device)
            avg_baseline_loss = 0.0

        return {
            "loss": avg_loss,
            "baseline_loss": torch.tensor(avg_baseline_loss, device=self.device),
            "improvement": torch.tensor(avg_baseline_loss, device=self.device) - avg_loss,
            "num_tokens": num_tokens,
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.cross_batch_module.train()

        total_loss = 0.0
        total_baseline_loss = 0.0
        total_improvement = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            labels_attention_mask = batch["labels_attention_mask"].to(self.device)

            # Skip batches with only 1 sample (no cross-batch possible)
            if input_ids.size(0) < 2:
                continue

            self.optimizer.zero_grad()

            loss_dict = self.compute_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                labels_attention_mask=labels_attention_mask,
            )

            loss = loss_dict["loss"]
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.cross_batch_module.parameters(), 1.0)

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            total_baseline_loss += loss_dict["baseline_loss"].item()
            total_improvement += loss_dict["improvement"].item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "baseline": f"{loss_dict['baseline_loss'].item():.4f}",
                "improve": f"{loss_dict['improvement'].item():.4f}",
            })

        return {
            "loss": total_loss / max(num_batches, 1),
            "baseline_loss": total_baseline_loss / max(num_batches, 1),
            "improvement": total_improvement / max(num_batches, 1),
        }

    def train(
        self,
        train_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 8,
        max_length: int = 512,
        save_dir: Optional[str] = "checkpoints",
        eval_dataset: Optional[Dataset] = None,
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            train_dataset: Training dataset
            num_epochs: Number of training epochs
            batch_size: Batch size
            max_length: Maximum sequence length
            save_dir: Directory to save checkpoints
            eval_dataset: Optional evaluation dataset

        Returns:
            Training history
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.tokenizer, max_length),
            drop_last=True,  # Ensure we always have full batches for cross-batch
        )

        # Setup scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )

        history = {
            "train_loss": [],
            "baseline_loss": [],
            "improvement": [],
        }

        best_improvement = float('-inf')

        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*50}")

            metrics = self.train_epoch(train_loader, epoch)

            history["train_loss"].append(metrics["loss"])
            history["baseline_loss"].append(metrics["baseline_loss"])
            history["improvement"].append(metrics["improvement"])

            logger.info(f"Epoch {epoch} - Loss: {metrics['loss']:.4f}, "
                       f"Baseline: {metrics['baseline_loss']:.4f}, "
                       f"Improvement: {metrics['improvement']:.4f}")

            # Save best model
            if metrics["improvement"] > best_improvement:
                best_improvement = metrics["improvement"]
                self.best_cross_batch_state = _clone_state_dict(self.cross_batch_module)
                if self.train_lm_head and hasattr(self.model, 'lm_head'):
                    self.best_lm_head_state = _clone_state_dict(self.model.lm_head)
                if save_dir:
                    self.save_checkpoint(os.path.join(save_dir, "best_model.pt"))
                    logger.info(f"Saved best model with improvement: {best_improvement:.4f}")
                else:
                    logger.info(f"New best improvement: {best_improvement:.4f}")

            # Save periodic checkpoint
            if save_dir and epoch % 5 == 0:
                self.save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch{epoch}.pt"))

        # Restore best states for subsequent evaluation
        if self.best_cross_batch_state is not None:
            self.cross_batch_module.load_state_dict(self.best_cross_batch_state)
        if self.best_lm_head_state is not None and hasattr(self.model, 'lm_head'):
            self.model.lm_head.load_state_dict(self.best_lm_head_state)

        if save_dir:
            # Save final model
            self.save_checkpoint(os.path.join(save_dir, "final_model.pt"))

            # Save training history
            with open(os.path.join(save_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=2)

        return history

    def save_checkpoint(self, path: str):
        """Save model checkpoint (only trained modules, not full model)."""
        save_dict = {
            "cross_batch_module": self.cross_batch_module.state_dict(),
        }
        # Also save lm_head if it was trained
        if self.train_lm_head and hasattr(self.model, 'lm_head'):
            save_dict["lm_head"] = self.model.lm_head.state_dict()
        torch.save(save_dict, path)
        logger.info(f"Checkpoint saved to {path} (cross_batch_module"
                   + (", lm_head" if self.train_lm_head else "") + ")")

    def load_checkpoint(self, path: str):
        """Load model checkpoint (only trained modules)."""
        checkpoint = torch.load(path, map_location=self.device)
        self.cross_batch_module.load_state_dict(checkpoint["cross_batch_module"])
        loaded_modules = ["cross_batch_module"]
        if "lm_head" in checkpoint and hasattr(self.model, 'lm_head'):
            self.model.lm_head.load_state_dict(checkpoint["lm_head"])
            loaded_modules.append("lm_head")
        logger.info(f"Checkpoint loaded from {path} ({', '.join(loaded_modules)})")


def train_cross_batch_module(
    model_name: str = "gpt2",
    mix_method: str = "attention",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_samples: int = 5000,
    save_dir: Optional[str] = "checkpoints",
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Convenience function to train cross-batch module.

    Args:
        model_name: HuggingFace model name
        mix_method: "attention" or "mixer"
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_samples: Maximum training samples
        save_dir: Directory to save checkpoints
        device: Device to train on

    Returns:
        Training history
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use float32 for training to avoid dtype mixing issues
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_size = model.config.hidden_size

    # Create cross-batch module
    if mix_method == "attention":
        cross_batch_module = CrossBatchAttention(
            hidden_size=hidden_size,
            num_heads=8,
            temperature=1.0,
        )
    else:
        cross_batch_module = CrossBatchEmbeddingMixer(
            hidden_size=hidden_size,
            temperature=1.0,
        )

    # Create trainer
    trainer = CrossBatchTrainer(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        device=device,
        learning_rate=learning_rate,
    )

    # Create dataset
    logger.info("Loading SQuAD dataset...")
    train_dataset = SQuADDataset(
        tokenizer=tokenizer,
        split="train",
        max_samples=max_samples,
    )

    # Train
    logger.info("Starting training...")
    history = trainer.train(
        train_dataset=train_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_dir=save_dir,
    )

    return history
