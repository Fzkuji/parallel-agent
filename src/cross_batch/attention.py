"""
Cross-batch attention mechanism for sharing information between samples during generation.

Simplified design: H_out = H + W @ cross_batch_info
- Original hidden state H is preserved unchanged
- Only learns what information to extract from other samples and add
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossBatchAttention(nn.Module):
    """
    Simplified cross-batch attention: H_out = H + scale * W @ attention(H_others)

    The original hidden state is preserved, we only ADD information from other samples.
    This makes learning easier - only need to learn "what to add", not "how to reconstruct".
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.temperature = temperature

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Q, K for computing attention weights
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # V and out_proj for extracting information to add
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        # Learnable scale for the additive term (start small)
        self.scale = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ≈ 0.047

        self._init_weights()

    def _init_weights(self):
        """Initialize weights - start with near-zero output."""
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        # Initialize out_proj to zero so initial output is just H
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, hidden_size]
            attention_mask: [batch_size] bool mask for valid samples

        Returns:
            output = hidden_states + scale * cross_batch_info
        """
        batch_size = hidden_states.size(0)

        if batch_size == 1:
            return hidden_states

        # Compute Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, self.num_heads, self.head_dim)

        # [num_heads, batch, head_dim]
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        # Attention weights: [num_heads, batch, batch]
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5 * self.temperature)

        # Mask out self-attention (diagonal)
        eye_mask = torch.eye(batch_size, device=hidden_states.device, dtype=torch.bool)
        attn_weights = attn_weights.masked_fill(eye_mask.unsqueeze(0), float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(0).unsqueeze(1)
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Gather information from other samples
        cross_batch_output = torch.bmm(attn_weights, v)  # [num_heads, batch, head_dim]
        cross_batch_output = cross_batch_output.permute(1, 0, 2)  # [batch, num_heads, head_dim]
        cross_batch_output = cross_batch_output.reshape(batch_size, self.hidden_size)
        cross_batch_output = self.out_proj(cross_batch_output)

        # ADDITIVE: H_out = H + scale * cross_batch_info
        scale = torch.sigmoid(self.scale)
        output = hidden_states + scale * cross_batch_output

        return output


class CrossBatchEmbeddingMixer(nn.Module):
    """
    Simplified cross-batch mixer: H_out = H + scale * W @ weighted_sum(H_others)

    Uses similarity-based attention to gather info from other samples,
    then ADDS it to the original (not replaces).
    """

    def __init__(
        self,
        hidden_size: int,
        temperature: float = 1.0,
        mix_ratio: float = 0.1,  # ignored, kept for API compatibility
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature

        # Project for computing similarity
        self.similarity_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Project for what to add from other samples
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Learnable scale - start very small (sigmoid(-5) ≈ 0.007)
        self.scale = nn.Parameter(torch.tensor(-5.0))

        self._init_weights()

    def _init_weights(self):
        """Initialize to near-zero output for stable training."""
        nn.init.normal_(self.similarity_proj.weight, mean=0.0, std=0.02)
        # Value projection starts at zero -> no contribution initially
        nn.init.zeros_(self.value_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, hidden_size]
            attention_mask: [batch_size] bool mask for valid samples

        Returns:
            output = hidden_states + scale * cross_batch_info
        """
        batch_size = hidden_states.size(0)

        if batch_size == 1:
            return hidden_states

        # Compute similarity-based attention weights
        projected = self.similarity_proj(hidden_states)
        normalized = F.normalize(projected, p=2, dim=-1)
        similarity = torch.mm(normalized, normalized.t()) / self.temperature

        # Mask out self
        eye_mask = torch.eye(batch_size, device=hidden_states.device, dtype=torch.bool)
        similarity = similarity.masked_fill(eye_mask, float('-inf'))

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(0)
            similarity = similarity.masked_fill(~mask, float('-inf'))

        weights = F.softmax(similarity, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        # Gather and project information from other samples
        values = self.value_proj(hidden_states)
        cross_batch_info = torch.mm(weights, values)

        # ADDITIVE: H_out = H + scale * cross_batch_info
        scale = torch.sigmoid(self.scale)
        output = hidden_states + scale * cross_batch_info

        return output
