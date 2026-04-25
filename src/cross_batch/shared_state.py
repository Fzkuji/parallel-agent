"""
Shared State Attention (SSA) - Cross-batch communication via a running shared vector.

Instead of each sequence attending to all other sequences (O(N^2)),
we maintain a shared state vector z that aggregates information from all sequences.
Each sequence reads from z and writes back to it every decode step.

z^t = alpha * mean(h_1^t, ..., h_N^t) + (1-alpha) * z^(t-1)
h_i' = h_i + gate_i * CrossAttn(h_i, z^t)

Benefits vs original CSA:
- O(N) instead of O(N^2)
- z accumulates sequence history, not just current-step snapshots
- Better signal: z encodes "what all sequences are generating so far"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SharedStateAttention(nn.Module):
    """
    Cross-batch communication via a shared running state vector z.

    z is maintained externally (in the generator) and passed in each forward call.
    This module only handles the read step: h_i' = h_i + gate_i * f(h_i, z)

    Args:
        hidden_size: Hidden dimension
        num_heads: Attention heads for cross-attn (h_i queries z)
        alpha: EMA decay for z update (0 = no memory, 1 = no update)
        use_gate: Use learned gate (recommended) vs scalar scale
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        alpha: float = 0.9,
        use_gate: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.alpha = alpha
        self.use_gate = use_gate

        assert hidden_size % num_heads == 0

        # Cross-attn: each h_i queries the shared state z
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        if use_gate:
            self.ln_h = nn.LayerNorm(hidden_size)
            self.ln_a = nn.LayerNorm(hidden_size)
            gate_hidden = hidden_size // 4
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_size * 3, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, hidden_size),
                nn.Sigmoid(),
            )
            # Init gate to small values
            nn.init.constant_(self.gate_net[-2].bias, -3.0)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        # LoRA-style: out_proj=0 ensures untrained model = baseline
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [N, d] - current step hidden states
        shared_z: torch.Tensor,        # [1, d] or [K, d] - shared state
    ) -> torch.Tensor:
        """
        Read from shared_z and update hidden_states.

        Args:
            hidden_states: [N, d] batch of current hidden states
            shared_z: [M, d] shared state (M=1 for single vector, M=K for multi-slot)

        Returns:
            updated hidden_states [N, d]
        """
        N = hidden_states.size(0)
        M = shared_z.size(0)  # number of slots in shared state

        # Q from each sequence, K/V from shared state
        q = self.q_proj(hidden_states).view(N, self.num_heads, self.head_dim)  # [N, H, d_h]
        k = self.k_proj(shared_z).view(M, self.num_heads, self.head_dim)       # [M, H, d_h]
        v = self.v_proj(shared_z).view(M, self.num_heads, self.head_dim)       # [M, H, d_h]

        # [H, N, d_h] x [H, d_h, M] -> [H, N, M]
        q = q.permute(1, 0, 2)   # [H, N, d_h]
        k = k.permute(1, 0, 2)   # [H, M, d_h]
        v = v.permute(1, 0, 2)   # [H, M, d_h]

        attn = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)  # [H, N, M]
        attn = F.softmax(attn, dim=-1)

        # [H, N, M] x [H, M, d_h] -> [H, N, d_h]
        out = torch.bmm(attn, v)               # [H, N, d_h]
        out = out.permute(1, 0, 2).reshape(N, self.hidden_size)  # [N, d]
        cross_info = self.out_proj(out)         # [N, d]

        if self.use_gate:
            ln_h = self.ln_h(hidden_states)
            ln_a = self.ln_a(cross_info)
            gate = self.gate_net(torch.cat([ln_h, ln_a, ln_h * ln_a], dim=-1))  # [N, d]
            return hidden_states + gate * cross_info
        else:
            return hidden_states + cross_info

    @staticmethod
    def update_shared_z(
        shared_z: Optional[torch.Tensor],
        hidden_states: torch.Tensor,   # [N, d]
        alpha: float = 0.9,
        num_slots: int = 1,
    ) -> torch.Tensor:
        """
        Update the shared state z using EMA of current hidden states.

        For num_slots=1: z = alpha * z_prev + (1-alpha) * mean(H)
        For num_slots>1: each slot tracks top-k sequences by similarity (future extension)

        Args:
            shared_z: previous z [num_slots, d], or None (first step)
            hidden_states: [N, d] current step hidden states
            alpha: EMA decay
            num_slots: number of slots (currently only 1 supported cleanly)

        Returns:
            updated z [num_slots, d]
        """
        # mean over batch dim -> [1, d]
        new_z = hidden_states.mean(dim=0, keepdim=True)  # [1, d]

        if shared_z is None or alpha == 0.0:
            return new_z

        # EMA update
        return alpha * shared_z + (1 - alpha) * new_z
