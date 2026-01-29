"""
Cross-batch attention mechanism for sharing information between samples during generation.

Three variants:
1. CrossBatchAttention: Full attention with learnable Q/K/V projections
2. CrossBatchEmbeddingMixer: Similarity-based mixing with value projection
3. SimpleCrossBatchGate: Minimal version - only learns a gate, uses raw cosine similarity
   - Parameter efficient: only ~2*d parameters per layer
   - Can be applied at multiple layers efficiently
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class CrossBatchAttention(nn.Module):
    """
    Cross-batch attention with optional question-aware gating.

    Scale mode: H_out = H + scale * A
    Gate mode:  H_out = H + gate(H, A) ⊙ A

    where A is the cross-sequence attention output.

    The gate follows the paper design:
        g_i = σ(MLP([LN(h_i); LN(a_i); LN(h_i) ⊙ LN(a_i)]))
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
        top_k: int = None,
    ):
        """
        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout probability
            temperature: Softmax temperature
            self_only: If True, attention only attends to self (diagonal only).
                       This disables cross-batch interaction for ablation study.
            use_gate: If True, use question-aware gating instead of scalar scale.
            gate_hidden_size: Hidden size for gate MLP (default: hidden_size // 4)
            top_k: If set, only keep top-k attention connections per query to reduce noise.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.temperature = temperature
        self.self_only = self_only
        self.use_gate = use_gate
        self.top_k = top_k

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Q, K for computing attention weights
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # V and out_proj for extracting information to add
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        if use_gate:
            # Question-aware gating network following paper design:
            # Input: [LN(h); LN(a); LN(h) ⊙ LN(a)] -> MLP -> sigmoid
            # Input dimension: 3 * hidden_size
            self.ln_h = nn.LayerNorm(hidden_size)
            self.ln_a = nn.LayerNorm(hidden_size)

            gate_hidden = gate_hidden_size or hidden_size // 4
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_size * 3, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, hidden_size),
                nn.Sigmoid(),
            )
            # Initialize gate to output small values initially
            self._init_gate_weights()

        self._init_weights()

    def _init_gate_weights(self):
        """Initialize gate network to output very small values initially."""
        # Initialize the final linear layer to output very small values
        # Ensures H_out ≈ H initially (untrained model = baseline)
        final_layer = self.gate_net[-2]  # Linear before Sigmoid
        nn.init.zeros_(final_layer.weight)
        # Bias initialized to -10 so sigmoid(-10) ≈ 0.00005 (very small, safe)
        nn.init.constant_(final_layer.bias, -10.0)

    def _init_weights(self):
        """LoRA-style initialization: out_proj=0, others normal.

        Q, K, V use normal init (std=0.02) for gradient flow.
        out_proj uses zero init (like LoRA B matrix) to ensure initial output = 0.
        This guarantees untrained model = baseline performance.
        """
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        # Zero init for out_proj (like LoRA B) - ensures cross_batch_output = 0 initially
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
            output = hidden_states + gate ⊙ cross_batch_info (or scale * cross_batch_info)
        """
        batch_size = hidden_states.size(0)

        if batch_size == 1:
            # Still pass through ALL parameters to maintain DDP gradient sync
            # The output equals input, but parameters participate in the computation graph
            # Must include q_proj, k_proj, v_proj, out_proj and gate components
            dummy = (
                self.q_proj(hidden_states).sum() * 0.0 +
                self.k_proj(hidden_states).sum() * 0.0 +
                self.out_proj(self.v_proj(hidden_states)).sum() * 0.0
            )
            if self.use_gate:
                ln_h = self.ln_h(hidden_states)
                ln_a = self.ln_a(hidden_states)
                gate_input = torch.cat([ln_h, ln_a, ln_h * ln_a], dim=-1)
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

        # Self-exclusion mask: additive -inf on diagonal
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

        # Top-k sparsification: only keep top-k connections per query
        if self.top_k is not None and self.top_k < batch_size - 1:
            # For each query, keep only top-k attention scores
            # attn_weights: [num_heads, batch, batch]
            top_k_values, _ = torch.topk(attn_weights, k=self.top_k, dim=-1)
            threshold = top_k_values[:, :, -1:].expand_as(attn_weights)
            # Mask out values below the k-th largest
            attn_weights = attn_weights.masked_fill(attn_weights < threshold, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Gather information from other samples
        cross_batch_output = torch.bmm(attn_weights, v)  # [num_heads, batch, head_dim]
        cross_batch_output = cross_batch_output.permute(1, 0, 2)  # [batch, num_heads, head_dim]
        cross_batch_output = cross_batch_output.reshape(batch_size, self.hidden_size)
        cross_batch_output = self.out_proj(cross_batch_output)

        # DEBUG: Log cross_batch_output norm (should be ~0 if out_proj=0)
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        self._debug_count += 1
        if self._debug_count <= 3:
            cb_norm = cross_batch_output.norm().item()
            cb_max = cross_batch_output.abs().max().item()
            h_norm = hidden_states.norm().item()
            print(f"[CSA #{self._debug_count}] batch={batch_size}, cross_batch_output: norm={cb_norm:.6f}, max={cb_max:.6f}, hidden_norm={h_norm:.4f}")
            if cb_norm < 0.001:
                print(f"[CSA #{self._debug_count}] ✓ cross_batch_output ≈ 0 (as expected)")
            else:
                print(f"[CSA #{self._debug_count}] ✗ cross_batch_output ≠ 0! This will change output!")

        if self.use_gate:
            # Question-aware gating following paper design:
            # g_i = σ(MLP([LN(h_i); LN(a_i); LN(h_i) ⊙ LN(a_i)]))
            ln_h = self.ln_h(hidden_states)
            ln_a = self.ln_a(cross_batch_output)
            gate_input = torch.cat([ln_h, ln_a, ln_h * ln_a], dim=-1)
            gate = self.gate_net(gate_input)  # [batch, hidden_size]

            # DEBUG: Log gate values
            if self._debug_count <= 3:
                gate_mean = gate.mean().item()
                gate_max = gate.max().item()
                contribution = (gate * cross_batch_output).norm().item()
                print(f"[CSA #{self._debug_count}] gate: mean={gate_mean:.6f}, max={gate_max:.6f}, contribution_norm={contribution:.6f}")

            output = hidden_states + gate * cross_batch_output
        else:
            # ADDITIVE: H_out = H + cross_batch_info
            # out_proj initialized to 0, so initially cross_batch_output = 0
            output = hidden_states + cross_batch_output

        return output


class CrossBatchEmbeddingMixer(nn.Module):
    """
    Cross-batch mixer with optional question-aware gating.

    Scale mode: H_out = H + scale * W @ weighted_sum(H_others)
    Gate mode:  H_out = H + gate(H, info) ⊙ W @ weighted_sum(H_others)

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
        top_k: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.use_gate = use_gate
        self.top_k = top_k

        # Project for computing similarity
        self.similarity_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Project for what to add from other samples
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        if use_gate:
            # Question-aware gating network following paper design
            self.ln_h = nn.LayerNorm(hidden_size)
            self.ln_a = nn.LayerNorm(hidden_size)

            gate_hidden = gate_hidden_size or hidden_size // 4
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_size * 3, gate_hidden),
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
            output = hidden_states + gate ⊙ cross_batch_info (or scale * cross_batch_info)
        """
        batch_size = hidden_states.size(0)

        if batch_size == 1:
            # Still pass through ALL parameters to maintain DDP gradient sync
            dummy = self.value_proj(self.similarity_proj(hidden_states)).sum() * 0.0
            if self.use_gate:
                ln_h = self.ln_h(hidden_states)
                ln_a = self.ln_a(hidden_states)
                gate_input = torch.cat([ln_h, ln_a, ln_h * ln_a], dim=-1)
                dummy = dummy + self.gate_net(gate_input).sum() * 0.0
            else:
                # Include scale parameter
                dummy = dummy + self.scale * 0.0
            return hidden_states + dummy

        # Compute similarity-based attention weights
        projected = self.similarity_proj(hidden_states)
        normalized = F.normalize(projected, p=2, dim=-1)
        similarity = torch.mm(normalized, normalized.t()) / self.temperature

        # Mask out self (additive -inf on diagonal)
        eye_mask = torch.eye(batch_size, device=hidden_states.device, dtype=torch.bool)
        similarity = similarity.masked_fill(eye_mask, float('-inf'))

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(0)
            similarity = similarity.masked_fill(~mask, float('-inf'))

        # Top-k sparsification
        if self.top_k is not None and self.top_k < batch_size - 1:
            top_k_values, _ = torch.topk(similarity, k=self.top_k, dim=-1)
            threshold = top_k_values[:, -1:].expand_as(similarity)
            similarity = similarity.masked_fill(similarity < threshold, float('-inf'))

        weights = F.softmax(similarity, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        # Gather and project information from other samples
        values = self.value_proj(hidden_states)
        cross_batch_info = torch.mm(weights, values)

        if self.use_gate:
            # Question-aware gating following paper design
            ln_h = self.ln_h(hidden_states)
            ln_a = self.ln_a(cross_batch_info)
            gate_input = torch.cat([ln_h, ln_a, ln_h * ln_a], dim=-1)
            gate = self.gate_net(gate_input)
            output = hidden_states + gate * cross_batch_info
        else:
            # ADDITIVE: H_out = H + scale * cross_batch_info
            scale = torch.sigmoid(self.scale)
            output = hidden_states + scale * cross_batch_info

        return output


class SimpleCrossBatchGate(nn.Module):
    """
    Minimal cross-batch module - only learns a gate, no Q/K/V projections.

    Uses raw cosine similarity as attention weights (not learned).
    Only learns a simple gate: g = σ(W_g @ [h; cross_info] + b_g)

    Parameter count: 2 * hidden_size + 1 (for the gate linear layer)

    Formula: H_out = H + gate * weighted_avg(H_others)
    where weights = softmax(cosine_similarity(H, H_others))
    """

    def __init__(
        self,
        hidden_size: int,
        temperature: float = 1.0,
        top_k: int = None,
    ):
        """
        Args:
            hidden_size: Hidden dimension size
            temperature: Softmax temperature for attention
            top_k: If set, only keep top-k attention connections per query
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.top_k = top_k

        # Only learn a simple gate: [h; cross_info] -> scalar gate
        # This is extremely parameter efficient
        self.gate = nn.Linear(hidden_size * 2, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize gate to very small values (conservative start)."""
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -10.0)  # sigmoid(-10) ≈ 0.00005 (nearly 0%)

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
            output = hidden_states + gate * cross_batch_info
        """
        batch_size = hidden_states.size(0)

        if batch_size == 1:
            # DDP gradient sync: ensure gate participates in computation
            dummy = self.gate(torch.cat([hidden_states, hidden_states], dim=-1)).sum() * 0.0
            return hidden_states + dummy

        # Compute cosine similarity as attention weights (no learning)
        normalized = F.normalize(hidden_states, p=2, dim=-1)
        similarity = torch.mm(normalized, normalized.t()) / self.temperature

        # Mask out self (diagonal)
        eye_mask = torch.eye(batch_size, device=hidden_states.device, dtype=torch.bool)
        similarity = similarity.masked_fill(eye_mask, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(0)
            similarity = similarity.masked_fill(~mask, float('-inf'))

        # Top-k sparsification
        if self.top_k is not None and self.top_k < batch_size - 1:
            top_k_values, _ = torch.topk(similarity, k=self.top_k, dim=-1)
            threshold = top_k_values[:, -1:].expand_as(similarity)
            similarity = similarity.masked_fill(similarity < threshold, float('-inf'))

        weights = F.softmax(similarity, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        # Weighted average of other samples (no projection)
        cross_batch_info = torch.mm(weights, hidden_states)

        # Simple gate: [h; cross_info] -> scalar
        gate_input = torch.cat([hidden_states, cross_batch_info], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))  # [batch, 1]

        output = hidden_states + gate * cross_batch_info
        return output


class MultiLayerCrossBatch(nn.Module):
    """
    Applies SimpleCrossBatchGate at multiple layers.

    Each layer has its own gate parameters, allowing the model to learn
    different mixing strategies at different depths.

    Total parameters: num_layers * (2 * hidden_size + 1)
    For a 32-layer model with hidden_size=4096: 32 * 8193 ≈ 262K parameters
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        layer_indices: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_k: int = None,
    ):
        """
        Args:
            hidden_size: Hidden dimension size
            num_layers: Total number of layers in the model
            layer_indices: Which layers to apply cross-batch (None = all layers)
            temperature: Softmax temperature for attention
            top_k: If set, only keep top-k attention connections per query
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # If no specific layers specified, use all layers
        if layer_indices is None:
            self.layer_indices = list(range(num_layers))
        else:
            self.layer_indices = layer_indices

        # Create a gate module for each selected layer
        self.gates = nn.ModuleDict({
            str(i): SimpleCrossBatchGate(
                hidden_size=hidden_size,
                temperature=temperature,
                top_k=top_k,
            )
            for i in self.layer_indices
        })

    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-batch mixing for a specific layer.

        Args:
            layer_idx: Which layer's hidden states
            hidden_states: [batch_size, hidden_size]
            attention_mask: [batch_size] bool mask for valid samples

        Returns:
            Mixed hidden states if layer_idx in layer_indices, else unchanged
        """
        if layer_idx in self.layer_indices:
            return self.gates[str(layer_idx)](hidden_states, attention_mask)
        return hidden_states

    def get_layer_gate(self, layer_idx: int) -> Optional[SimpleCrossBatchGate]:
        """Get the gate module for a specific layer."""
        if layer_idx in self.layer_indices:
            return self.gates[str(layer_idx)]
        return None


class MultiLayerCrossBatchAttention(nn.Module):
    """
    Applies CrossBatchAttention at multiple layers (with learnable Q/K/V).

    Each layer has its own attention module, allowing the model to learn
    different cross-batch interaction patterns at different depths.

    Total parameters per layer: 4*d^2 (Q/K/V/out projections) + scale or gate params
    For a 7B model (hidden_size=3584) with 14 layers: ~14 * 4 * 3584^2 ≈ 720M parameters
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        layer_indices: Optional[List[int]] = None,
        num_heads: int = 8,
        temperature: float = 1.0,
        use_gate: bool = True,
        top_k: int = None,
    ):
        """
        Args:
            hidden_size: Hidden dimension size
            num_layers: Total number of layers in the model
            layer_indices: Which layers to apply cross-batch (None = all layers)
            num_heads: Number of attention heads
            temperature: Softmax temperature for attention
            use_gate: Whether to use question-aware gating
            top_k: If set, only keep top-k attention connections per query
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # If no specific layers specified, use all layers
        if layer_indices is None:
            self.layer_indices = list(range(num_layers))
        else:
            self.layer_indices = layer_indices

        # Create a full attention module for each selected layer
        self.attentions = nn.ModuleDict({
            str(i): CrossBatchAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                temperature=temperature,
                use_gate=use_gate,
                top_k=top_k,
            )
            for i in self.layer_indices
        })

    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-batch attention for a specific layer.

        Args:
            layer_idx: Which layer's hidden states
            hidden_states: [batch_size, hidden_size]
            attention_mask: [batch_size] bool mask for valid samples

        Returns:
            Mixed hidden states if layer_idx in layer_indices, else unchanged
        """
        if layer_idx in self.layer_indices:
            return self.attentions[str(layer_idx)](hidden_states, attention_mask)
        return hidden_states

    def get_layer_attention(self, layer_idx: int) -> Optional[CrossBatchAttention]:
        """Get the attention module for a specific layer."""
        if layer_idx in self.layer_indices:
            return self.attentions[str(layer_idx)]
        return None
