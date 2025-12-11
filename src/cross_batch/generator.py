"""
Modified generation loop with cross-batch interaction.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from .attention import CrossBatchAttention, CrossBatchEmbeddingMixer


class CrossBatchGenerator:
    """
    Generator that enables cross-batch interaction during token generation.
    Each new token's hidden state is influenced by hidden states from other samples.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cross_batch_module: Optional[nn.Module] = None,
        mix_method: str = "attention",  # "attention" or "mixer"
        mix_layer: int = -1,  # which layer's hidden state to mix (-1 for last)
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.mix_layer = mix_layer

        # Get hidden size and dtype from model config
        hidden_size = model.config.hidden_size
        model_dtype = next(model.parameters()).dtype

        # Initialize cross-batch module with matching dtype
        if cross_batch_module is not None:
            self.cross_batch_module = cross_batch_module.to(device=device, dtype=model_dtype)
        elif mix_method == "attention":
            self.cross_batch_module = CrossBatchAttention(
                hidden_size=hidden_size,
                num_heads=8,
                temperature=1.0,
            ).to(device=device, dtype=model_dtype)
        else:
            self.cross_batch_module = CrossBatchEmbeddingMixer(
                hidden_size=hidden_size,
                temperature=1.0,
                mix_ratio=0.1,
            ).to(device=device, dtype=model_dtype)

        self.model.eval()
        self.cross_batch_module.eval()

    def _hidden_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Convert hidden states to logits using model's output projection."""
        # Handle different model architectures
        if hasattr(self.model, 'lm_head'):
            # Models like GPT-2, GPT-Neo, LLaMA, etc.
            return self.model.lm_head(hidden_states)
        elif hasattr(self.model, 'embed_out'):
            # Some models use embed_out
            return self.model.embed_out(hidden_states)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            # GPT-2 style with tied embeddings
            # hidden_states @ wte.weight.T
            return torch.nn.functional.linear(hidden_states, self.model.transformer.wte.weight)
        else:
            raise ValueError("Could not find output projection layer in model")

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        min_new_tokens: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        enable_cross_batch: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate tokens with cross-batch interaction.

        Args:
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            min_new_tokens: Minimum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token id
            eos_token_id: End of sequence token id
            enable_cross_batch: Whether to enable cross-batch interaction

        Returns:
            Dictionary containing generated sequences and metadata
        """
        batch_size = input_ids.size(0)
        input_ids = input_ids.to(self.device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.device)

        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Store cross-batch mixing weights for analysis
        cross_batch_weights_history = []

        # Generation loop
        generated_ids = input_ids.clone()
        past_key_values = None

        for step in range(max_new_tokens):
            # Forward pass
            if past_key_values is not None:
                # Use only the last token for efficiency
                model_input_ids = generated_ids[:, -1:]
                model_attention_mask = attention_mask
            else:
                model_input_ids = generated_ids
                model_attention_mask = attention_mask

            outputs = self.model(
                input_ids=model_input_ids,
                attention_mask=model_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )

            # Get hidden states from specified layer
            hidden_states = outputs.hidden_states[self.mix_layer]
            # Shape: [batch_size, seq_len, hidden_size]

            # Get the last token's hidden state
            last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]

            # Apply cross-batch interaction
            if enable_cross_batch and batch_size > 1:
                valid_mask = ~finished
                mixed_hidden = self.cross_batch_module(
                    last_hidden,
                    attention_mask=valid_mask,
                )
                # Project back to logits using the model's output projection
                next_token_logits = self._hidden_to_logits(mixed_hidden)
            else:
                next_token_logits = outputs.logits[:, -1, :]

            # Apply temperature, top-k, top-p only when sampling
            if do_sample:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

            # Sample or greedy
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Replace tokens for finished sequences with pad
            next_tokens = next_tokens.masked_fill(finished, pad_token_id)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                (~finished).long().unsqueeze(-1)
            ], dim=-1)

            # Update past_key_values
            past_key_values = outputs.past_key_values

            # Check for EOS
            if step >= min_new_tokens - 1:
                finished = finished | (next_tokens == eos_token_id)

            if finished.all():
                break

        return {
            "sequences": generated_ids,
            "generated_tokens": generated_ids[:, input_ids.size(1):],
            "cross_batch_weights": cross_batch_weights_history,
        }

    def generate_text(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        **kwargs,
    ) -> List[str]:
        """
        Generate text from string prompts.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum new tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            List of generated text strings
        """
        # Tokenize
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Generate
        outputs = self.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        # Decode
        generated_texts = self.tokenizer.batch_decode(
            outputs["sequences"],
            skip_special_tokens=True,
        )

        return generated_texts
