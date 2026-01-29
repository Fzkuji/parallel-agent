"""
Modified generation loop with cross-batch interaction.

Supports three modes:
1. Single layer: Apply cross-batch at one layer (default: last layer)
2. Multi-layer: Apply SimpleCrossBatchGate at multiple layers
3. Legacy: Original CrossBatchAttention/Mixer for backward compatibility
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
from .attention import (
    CrossBatchAttention,
    CrossBatchEmbeddingMixer,
    SimpleCrossBatchGate,
    MultiLayerCrossBatch,
    MultiLayerCrossBatchAttention,
)


class CrossBatchGenerator:
    """
    Generator that enables cross-batch interaction during token generation.
    Each new token's hidden state is influenced by hidden states from other samples.

    Supports:
    - Single layer mixing with CrossBatchAttention/EmbeddingMixer/SimpleCrossBatchGate
    - Multi-layer mixing with MultiLayerCrossBatch
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cross_batch_module: Optional[nn.Module] = None,
        mix_method: str = "attention",  # "attention", "mixer", "simple", or "multi_layer"
        mix_layer: Union[int, List[int]] = -1,  # which layer(s) to mix
        device: str = "cuda",
    ):
        # Check if model uses device_map (distributed across GPUs)
        is_distributed = hasattr(model, 'hf_device_map') and model.hf_device_map

        if is_distributed:
            # Don't move distributed model, use its existing device placement
            self.model = model
            # Get the device where lm_head is located (for cross_batch_module)
            if hasattr(model, 'lm_head'):
                self.device = str(next(model.lm_head.parameters()).device)
            else:
                self.device = device
            # Get input device (where embeddings are)
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                self.input_device = str(next(model.model.embed_tokens.parameters()).device)
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                self.input_device = str(next(model.transformer.wte.parameters()).device)
            else:
                self.input_device = self.device
        else:
            self.model = model.to(device)
            self.device = device
            self.input_device = device

        self.tokenizer = tokenizer
        self.mix_method = mix_method

        # Handle mix_layer - can be int or list
        if isinstance(mix_layer, int):
            self.mix_layers = [mix_layer]
            self.is_multi_layer = False
        else:
            self.mix_layers = list(mix_layer)
            self.is_multi_layer = len(self.mix_layers) > 1

        # For backward compatibility
        self.mix_layer = self.mix_layers[-1] if self.mix_layers else -1

        # Get hidden size and dtype from model config
        hidden_size = model.config.hidden_size
        num_layers = model.config.num_hidden_layers
        model_dtype = next(model.parameters()).dtype

        # Initialize cross-batch module with matching dtype
        # Place on same device as lm_head (self.device)
        if cross_batch_module is not None:
            self.cross_batch_module = cross_batch_module.to(device=self.device, dtype=model_dtype)
        elif mix_method == "multi_layer":
            # Multi-layer with SimpleCrossBatchGate at each layer
            self.cross_batch_module = MultiLayerCrossBatch(
                hidden_size=hidden_size,
                num_layers=num_layers,
                layer_indices=self.mix_layers if self.mix_layers[0] != -1 else None,
                temperature=1.0,
            ).to(device=self.device, dtype=model_dtype)
            self.is_multi_layer = True
        elif mix_method == "multi_layer_attention":
            # Multi-layer with full CrossBatchAttention at each layer
            self.cross_batch_module = MultiLayerCrossBatchAttention(
                hidden_size=hidden_size,
                num_layers=num_layers,
                layer_indices=self.mix_layers if self.mix_layers[0] != -1 else None,
                num_heads=8,
                temperature=1.0,
                use_gate=True,  # Default to using gate
            ).to(device=self.device, dtype=model_dtype)
            self.is_multi_layer = True
        elif mix_method == "simple":
            # Single layer with SimpleCrossBatchGate
            self.cross_batch_module = SimpleCrossBatchGate(
                hidden_size=hidden_size,
                temperature=1.0,
            ).to(device=self.device, dtype=model_dtype)
        elif mix_method == "attention":
            self.cross_batch_module = CrossBatchAttention(
                hidden_size=hidden_size,
                num_heads=8,
                temperature=1.0,
            ).to(device=self.device, dtype=model_dtype)
        else:  # "mixer"
            self.cross_batch_module = CrossBatchEmbeddingMixer(
                hidden_size=hidden_size,
                temperature=1.0,
                mix_ratio=0.1,
            ).to(device=self.device, dtype=model_dtype)

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
        input_ids = input_ids.to(self.input_device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.input_device)

        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        # Build set of all stop token IDs (handle multiple stop tokens for chat models)
        stop_token_ids = {eos_token_id, pad_token_id}
        # Add additional stop tokens from model config
        if hasattr(self.model.config, 'eos_token_id'):
            config_eos = self.model.config.eos_token_id
            if isinstance(config_eos, list):
                stop_token_ids.update(config_eos)
            elif config_eos is not None:
                stop_token_ids.add(config_eos)
        # Add common chat model stop tokens by name
        for stop_name in ['<|im_end|>', '<|endoftext|>', '<|eot_id|>', '</s>']:
            stop_id = self.tokenizer.convert_tokens_to_ids(stop_name)
            if stop_id != self.tokenizer.unk_token_id:
                stop_token_ids.add(stop_id)
        stop_token_ids.discard(None)

        # Track which sequences have finished (on input_device to match attention_mask operations)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.input_device)

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

            # Apply cross-batch interaction
            # Key insight: Only apply CSA to ACTIVE (unfinished) sequences
            # - Finished sequences should use model's native logits (will be masked to pad anyway)
            # - Mixing finished sequences' hidden states into CSA would pollute cross-batch info
            active_mask = ~finished  # [batch_size] - True for active sequences
            num_active = active_mask.sum().item()

            if enable_cross_batch and batch_size > 1 and num_active > 1:
                # Get model's native logits for all sequences first
                model_logits = outputs.logits[:, -1, :].to(self.device)  # [batch, vocab]

                # Extract hidden states only for active sequences
                active_indices = active_mask.nonzero(as_tuple=True)[0]

                if self.is_multi_layer and isinstance(self.cross_batch_module, (MultiLayerCrossBatch, MultiLayerCrossBatchAttention)):
                    # Multi-layer mode: apply cross-batch module at each selected layer
                    # Only use active sequences for cross-batch
                    accumulated_delta = None

                    for layer_idx in self.cross_batch_module.layer_indices:
                        # Handle negative indices
                        actual_idx = layer_idx if layer_idx >= 0 else len(outputs.hidden_states) + layer_idx
                        # Only take active sequences' hidden states
                        layer_hidden_all = outputs.hidden_states[actual_idx][:, -1, :].to(self.device)
                        layer_hidden = layer_hidden_all[active_indices]  # [num_active, hidden]

                        # Apply the layer-specific module (only to active sequences)
                        mixed = self.cross_batch_module(layer_idx, layer_hidden)
                        delta = mixed - layer_hidden

                        if accumulated_delta is None:
                            accumulated_delta = delta
                        else:
                            accumulated_delta = accumulated_delta + delta

                    # Average the deltas and add to final hidden state
                    num_layers = len(self.cross_batch_module.layer_indices)
                    final_hidden_all = outputs.hidden_states[-1][:, -1, :].to(self.device)
                    final_hidden = final_hidden_all[active_indices]
                    mixed_hidden = final_hidden + accumulated_delta / num_layers
                else:
                    # Single layer mode: use hidden_states[-1] which is ALREADY normalized
                    last_hidden_all = outputs.hidden_states[-1][:, -1, :].to(self.device)
                    last_hidden = last_hidden_all[active_indices]  # [num_active, hidden]
                    mixed_hidden = self.cross_batch_module(last_hidden)

                # Project active sequences' hidden states to logits
                active_logits = self._hidden_to_logits(mixed_hidden)  # [num_active, vocab]

                # Combine: use CSA logits for active, model logits for finished
                next_token_logits = model_logits.clone()
                next_token_logits[active_indices] = active_logits
            else:
                # batch_size == 1, CSA disabled, or only 1 active sequence
                # Use model's native logits directly
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

            # Move next_tokens to input_device for operations with finished and generated_ids
            next_tokens = next_tokens.to(self.input_device)

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

            # Check for any stop token
            if step >= min_new_tokens - 1:
                for stop_id in stop_token_ids:
                    finished = finished | (next_tokens == stop_id)

            if finished.all():
                break

        return {
            "sequences": generated_ids,
            "generated_tokens": generated_ids[:, input_ids.size(1):],
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
