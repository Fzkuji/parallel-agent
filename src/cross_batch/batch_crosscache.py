"""True batched cross-cache attention for parallel decoding.

G sequences live in the batch dimension, each with its OWN context cache,
prefilled independently. At EVERY layer, each sequence's query attends its own
cache PLUS the sibling sequences' CONTEXT KV (a shared bank gathered across the
batch, self excluded). Standard batched decode -> G answers in parallel; the only
change is that attention reaches across the batch into the context bank.

Implemented with F.scaled_dot_product_attention over extended K/V (own cache ++
sibling context bank) so it is numerically clean (the earlier manual-softmax
all-layer version corrupted). Sibling context KV keeps its prefill RoPE (natural
per-sequence positions -- validated to not break cross-attention).
"""
from __future__ import annotations
import math
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


def _layers(model):
    base = model
    for _ in range(4):
        if hasattr(base, "layers"):
            return base.layers
        base = getattr(base, "model", None)
        if base is None:
            break
    raise ValueError("no layers")


class BatchCrossCache:
    def __init__(self, layer_indices: List[int]):
        self.idx = set(layer_indices)
        self.enabled = True
        self.mode = "use"                                  # "capture" | "use"
        self.exclude_self = False                          # whether a query skips its own batch row
        self.context_mask: Optional[torch.Tensor] = None   # [B,T] bool valid context tokens (capture)
        self.valid: Optional[torch.Tensor] = None          # [B,Tc] bool non-pad cache positions
        self.query_rows: Optional[torch.Tensor] = None     # [B,T] rows allowed to read the bank
        self.valid_all = True                              # no padding in own cache -> flash-eligible
        self.qrows_all = True                              # every query row reads the bank
        self.record_attn = False                           # analysis: record query->bank attention mass
        self.attn_record: List = []                        # [B,B] per-context mass, appended per layer/step
        self.allowed: Optional[torch.Tensor] = None        # [Bq,Bctx] bool: query b may read context j
        self.temperature = 1.0                             # APE: context-segment attention temperature (T<1 sharpens)
        self.scale_s = 1.0                                 # APE: context-segment LSE scaling (S<1 down-weights bank)
        self.bank: Dict[int, Tuple] = {}
        self._orig: Dict[int, callable] = {}

    def set_enabled(self, e):
        self.enabled = bool(e)

    def set_realign(self, temperature=1.0, scale=1.0):
        """APE attention realignment: temperature sharpens the bank segment, scale (S)
        re-weights it in log space. T=S=1 -> exact union softmax (current behaviour)."""
        self.temperature = float(temperature)
        self.scale_s = float(scale)

    def set_valid(self, valid):
        self.valid = valid.bool()
        self.valid_all = bool(self.valid.all())          # cheap flag: no padding -> flash-eligible

    def set_query_rows(self, qr):
        self.query_rows = qr.bool()
        self.qrows_all = bool(self.query_rows.all())      # all rows read the bank -> no row block

    def start_capture(self, context_mask):
        """Phase 1: prefill the contexts; capture their KV into a shared bank."""
        self.mode = "capture"
        self.context_mask = context_mask.bool()
        self.bank = {}

    def start_use(self):
        """Phase 2: queries read the captured bank."""
        self.mode = "use"

    def set_allowed(self, allowed):
        """allowed[b,j]=True -> query b may read context j's bank tokens (top-k selectivity)."""
        self.allowed = allowed.bool() if allowed is not None else None

    def reset(self, context_mask):
        self.mode = "use"
        self.context_mask = context_mask.bool()
        self.bank = {}

    def register(self, model):
        layers = _layers(model)
        for i in range(len(layers)):
            if i in self.idx:
                attn = layers[i].self_attn
                self._orig[i] = attn.forward
                attn.forward = types.MethodType(self._make(), attn)

    def _make(self):
        mgr = self

        def own_mask(B, T, Tc, dtype, device):
            neg = torch.finfo(dtype).min
            valid = mgr.valid if (mgr.valid is not None and mgr.valid.shape[1] == Tc) \
                else torch.ones(B, Tc, dtype=torch.bool, device=device)
            ao = valid.view(B, 1, 1, Tc).expand(B, 1, T, Tc).clone()
            if T > 1:
                ao = ao & torch.tril(torch.ones(T, Tc, device=device, dtype=torch.bool)).view(1, 1, T, Tc)
            return torch.zeros(B, 1, T, Tc, dtype=dtype, device=device).masked_fill(~ao, neg)

        def fwd(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kw):
            B, T = hidden_states.shape[0], hidden_states.shape[1]
            if not mgr.enabled:
                return mgr._orig[self.layer_idx](
                    hidden_states, position_embeddings, attention_mask,
                    past_key_values, cache_position, **kw)
            shp = (B, T, -1, self.head_dim)
            q = self.q_proj(hidden_states).view(shp).transpose(1, 2)
            k = self.k_proj(hidden_states).view(shp).transpose(1, 2)
            v = self.v_proj(hidden_states).view(shp).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if past_key_values is not None:
                k, v = past_key_values.update(
                    k, v, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})
            nkv = k.shape[1]; nh = q.shape[1]; rep = nh // nkv; Tc = k.shape[2]
            neg = torch.finfo(q.dtype).min

            if mgr.mode == "capture":
                # store each context's KV into the shared bank, then standard attention
                cm = mgr.context_mask
                if T > 1 and cm is not None and cm.shape == (B, T):
                    ii = cm.nonzero(as_tuple=False)
                    prev = mgr.bank.get(self.layer_idx)
                    nk = k[ii[:, 0], :, ii[:, 1], :]; nv = v[ii[:, 0], :, ii[:, 1], :]; ns = ii[:, 0]
                    if prev is not None and prev[0] is not None:
                        nk = torch.cat([prev[0], nk]); nv = torch.cat([prev[1], nv]); ns = torch.cat([prev[2], ns])
                    mgr.bank[self.layer_idx] = (nk, nv, ns)
                o = F.scaled_dot_product_attention(q, repeat_kv(k, rep), repeat_kv(v, rep),
                                                   attn_mask=own_mask(B, T, Tc, q.dtype, q.device))
                return self.o_proj(o.transpose(1, 2).reshape(B, T, -1)), None

            # mode == "use": query attends own cache + the shared context bank
            bank = mgr.bank.get(self.layer_idx)
            if bank is None or bank[0] is None or bank[0].shape[0] == 0:
                o = F.scaled_dot_product_attention(q, repeat_kv(k, rep), repeat_kv(v, rep),
                                                   attn_mask=own_mask(B, T, Tc, q.dtype, q.device))
                return self.o_proj(o.transpose(1, 2).reshape(B, T, -1)), None
            bK, bV, bseq = bank
            M = bK.shape[0]
            bKe = repeat_kv(bK.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1), rep)
            bVe = repeat_kv(bV.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1), rep)
            if mgr.record_attn:
                # ANALYSIS: record how much query->bank attention mass lands on each source context.
                d = q.shape[-1]; sc = 1.0 / math.sqrt(d)
                own_a = own_mask(B, T, Tc, q.dtype, q.device)
                s_own = torch.matmul(q, repeat_kv(k, rep).transpose(-1, -2)) * sc + own_a
                s_bank = torch.matmul(q, bKe.transpose(-1, -2)) * sc            # [B,H,T,M]
                full = torch.cat([s_own, s_bank], dim=-1).softmax(dim=-1)
                w_bank = full[..., Tc:].mean(dim=1).mean(dim=1).float()         # [B,M] avg over heads & query tokens
                per_ctx = torch.zeros(B, B, device=q.device, dtype=w_bank.dtype)
                per_ctx.scatter_add_(1, bseq.view(1, M).expand(B, M).long(), w_bank)   # mass per source context
                mgr.attn_record.append(per_ctx.detach().float().cpu())
            # FAST decode path: single new token, full read (no top-k / exclude / padding) and no
            # realignment -> the additive mask is all-zero, so drop it and let SDPA use FlashAttention.
            # Numerically identical to the masked path (adding a zero mask is a no-op).
            if (T == 1 and mgr.temperature == 1.0 and mgr.scale_s == 1.0
                    and mgr.allowed is None and not mgr.exclude_self
                    and mgr.qrows_all and mgr.valid_all
                    and mgr.valid is not None and mgr.valid.shape[1] == Tc):
                K = torch.cat([repeat_kv(k, rep), bKe], dim=2)
                V = torch.cat([repeat_kv(v, rep), bVe], dim=2)
                out = F.scaled_dot_product_attention(q, K, V)        # no mask -> flash kernel
                return self.o_proj(out.transpose(1, 2).reshape(B, T, -1)), None
            own = own_mask(B, T, Tc, q.dtype, q.device)
            row_block = (~mgr.query_rows).view(B, 1, T, 1) if (mgr.query_rows is not None and
                         mgr.query_rows.shape[1] == T) else torch.zeros(B, 1, T, 1, dtype=torch.bool, device=q.device)
            block = row_block.expand(B, 1, T, M).clone()
            if mgr.exclude_self:
                excl = (bseq.view(1, M) == torch.arange(B, device=q.device).view(B, 1)).view(B, 1, 1, M)
                block = block | excl
            if mgr.allowed is not None and mgr.allowed.shape[0] == B:
                # query b only reads contexts in its top-k: block bank token m if not allowed[b, bseq[m]]
                sel = mgr.allowed[:, bseq]                       # [B, M] : allowed[b, source-context-of-m]
                block = block | (~sel).view(B, 1, 1, M)
            bank_m = torch.zeros(B, 1, T, M, dtype=q.dtype, device=q.device).masked_fill(block, neg)
            if mgr.temperature == 1.0 and mgr.scale_s == 1.0:
                K = torch.cat([repeat_kv(k, rep), bKe], dim=2)
                V = torch.cat([repeat_kv(v, rep), bVe], dim=2)
                out = F.scaled_dot_product_attention(q, K, V, attn_mask=torch.cat([own, bank_m], dim=-1))
                return self.o_proj(out.transpose(1, 2).reshape(B, T, -1)), None
            # APE realignment: own (local) and bank (context) segments computed separately,
            # bank sharpened by temperature T and re-weighted by S in log space, then merged.
            d = q.shape[-1]; sc = 1.0 / math.sqrt(d)
            Ko = repeat_kv(k, rep); Vo = repeat_kv(v, rep)
            s_own = torch.matmul(q, Ko.transpose(-1, -2)) * sc + own
            s_bank = torch.matmul(q, bKe.transpose(-1, -2)) * (sc / mgr.temperature) + bank_m
            lse_own = torch.logsumexp(s_own, dim=-1, keepdim=True)
            lse_bank = torch.logsumexp(s_bank, dim=-1, keepdim=True) * (mgr.scale_s * mgr.temperature)
            mmax = torch.maximum(lse_own, lse_bank)
            wo = torch.exp(lse_own - mmax); wb = torch.exp(lse_bank - mmax)
            o_own = torch.matmul(torch.softmax(s_own, dim=-1), Vo)
            o_bank = torch.matmul(torch.softmax(s_bank, dim=-1), bVe)
            out = (wo * o_own + wb * o_bank) / (wo + wb).clamp(min=1e-9)
            return self.o_proj(out.transpose(1, 2).reshape(B, T, -1)), None

        return fwd
