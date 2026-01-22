"""
Cross-batch attention mechanism for sharing information between samples during generation.

Two modes:
1. Scale mode (default): H_out = H + scale * cross_batch_info
   - Simple learnable scalar controls mixing
2. Gate mode: H_out = H + gate(H, cross_batch_info) * cross_batch_info
   - Question-aware gating: model decides when to use cross-batch info
   - Gate is computed based on both original hidden state and cross-batch info
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossBatchAttention(nn.Module):
    """
    Cross-batch attention with optional question-aware gating.

    Scale mode: H_out = H + scale * W @ attention(H_others)
    Gate mode:  H_out = H + gate(H, info) * W @ attention(H_others)

    The original hidden state is preserved, we only ADD information from other samples.
    Gate mode allows the model to dynamically decide when cross-batch info is useful.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        temperature: float = 1.0,
        self_only: bool = False,
        use_gate: bool = False,
        gate_hidden_size: int = None,
    ):
        """
        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout probability
            temperature: Softmax temperature
            self_only: If True, attention only attends to self (diagonal only).
                       This disables cross-batch interaction for ablation study.
                       The module still has the same parameters, but no cross-sample info flows.
            use_gate: If True, use question-aware gating instead of scalar scale.
            gate_hidden_size: Hidden size for gate MLP (default: hidden_size // 4)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.temperature = temperature
        self.self_only = self_only
        self.use_gate = use_gate

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Q, K for computing attention weights
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # V and out_proj for extracting information to add
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        if use_gate:
            # Question-aware gating network
            # Input: concat(H, cross_batch_info) -> gate value per dimension
            gate_hidden = gate_hidden_size or hidden_size // 4
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_size * 2, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, hidden_size),
                nn.Sigmoid(),
            )
            # Initialize gate to output small values initially
            self._init_gate_weights()
        else:
            # Learnable scale for the additive term (start small)
            self.scale = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ≈ 0.047

        self._init_weights()

    def _init_gate_weights(self):
        """Initialize gate network to output small values initially."""
        # Initialize the final linear layer to output values near 0
        # This ensures H_out ≈ H at the start of training
        final_layer = self.gate_net[-2]  # Linear before Sigmoid
        nn.init.zeros_(final_layer.weight)
        # Bias initialized to -3 so sigmoid(-3) ≈ 0.047
        nn.init.constant_(final_layer.bias, -3.0)

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
            # Still pass through parameters to maintain DDP gradient sync
            # The output equals input, but parameters participate in the computation graph
            dummy = self.out_proj(self.v_proj(hidden_states)).sum() * 0.0
            if self.use_gate:
                gate_input = torch.cat([hidden_states, hidden_states], dim=-1)
                dummy = dummy + self.gate_net(gate_input).sum() * 0.0
            return hidden_states + dummy

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

        eye_mask = torch.eye(batch_size, device=hidden_states.device, dtype=torch.bool)
        
        if self.self_only:
            # Ablation: only attend to self (diagonal only, no cross-batch interaction)
            # Mask out everything except diagonal
            attn_weights = attn_weights.masked_fill(~eye_mask.unsqueeze(0), float('-inf'))
        else:
            # Normal: mask out self-attention (diagonal), attend to other samples
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

        if self.use_gate:
            # Question-aware gating: gate(H, cross_batch_info) per dimension
            # Gate decides how much cross-batch info to use based on both H and the info
            gate_input = torch.cat([hidden_states, cross_batch_output], dim=-1)
            gate = self.gate_net(gate_input)  # [batch, hidden_size]
            output = hidden_states + gate * cross_batch_output
        else:
            # ADDITIVE: H_out = H + scale * cross_batch_info
            scale = torch.sigmoid(self.scale)
            output = hidden_states + scale * cross_batch_output

        return output


class CrossBatchEmbeddingMixer(nn.Module):
    """
    Cross-batch mixer with optional question-aware gating.

    Scale mode: H_out = H + scale * W @ weighted_sum(H_others)
    Gate mode:  H_out = H + gate(H, info) * W @ weighted_sum(H_others)

    Uses similarity-based attention to gather info from other samples,
    then ADDS it to the original (not replaces).
    """

    def __init__(
        self,
        hidden_size: int,
        temperature: float = 1.0,
        mix_ratio: float = 0.1,  # ignored, kept for API compatibility
        use_gate: bool = False,
        gate_hidden_size: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.use_gate = use_gate

        # Project for computing similarity
        self.similarity_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Project for what to add from other samples
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        if use_gate:
            # Question-aware gating network
            gate_hidden = gate_hidden_size or hidden_size // 4
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_size * 2, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, hidden_size),
                nn.Sigmoid(),
            )
            self._init_gate_weights()
        else:
            # Learnable scale - start very small (sigmoid(-5) ≈ 0.007)
            self.scale = nn.Parameter(torch.tensor(-5.0))

        self._init_weights()

    def _init_gate_weights(self):
        """Initialize gate network to output small values initially."""
        final_layer = self.gate_net[-2]
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, -3.0)

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
            # Still pass through parameters to maintain DDP gradient sync
            dummy = self.value_proj(self.similarity_proj(hidden_states)).sum() * 0.0
            if self.use_gate:
                gate_input = torch.cat([hidden_states, hidden_states], dim=-1)
                dummy = dummy + self.gate_net(gate_input).sum() * 0.0
            return hidden_states + dummy

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

        if self.use_gate:
            # Question-aware gating
            gate_input = torch.cat([hidden_states, cross_batch_info], dim=-1)
            gate = self.gate_net(gate_input)
            output = hidden_states + gate * cross_batch_info
        else:
            # ADDITIVE: H_out = H + scale * cross_batch_info
            scale = torch.sigmoid(self.scale)
            output = hidden_states + scale * cross_batch_info

        return output
