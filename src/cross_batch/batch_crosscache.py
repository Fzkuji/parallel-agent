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
import os
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
        self.rec_ctx_w = 0                                 # >0: record per-source-context bank mass (read-fraction)
        self.ctx_mass = None                               # [B, rec_ctx_w] accumulated bank mass per source context
        # PROGRESSIVE PRUNING: every decode step, accumulate each query's attention to each source
        # context; contexts the model keeps ignoring are dropped (monotonically) so later steps read
        # a shrinking, focused bank. Driven by the model's OWN past-token attention; no fixed k/threshold.
        self.prune = False                                 # enable progressive convergence pruning
        self.attn_accum: Optional[torch.Tensor] = None     # [B, Cctx] accumulated attention per context
        self.prune_steps = 0                               # decode steps seen (warmup before pruning)
        self._prune_updated_step = -1                      # guard: update survivor set once per step
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
        self.attn_accum = None
        self.prune_steps = 0
        self._prune_updated_step = -1

    def set_prune(self, on):
        self.prune = bool(on)

    def _update_survivors(self, w_ctx, step):
        """Accumulate per-context attention (w_ctx: [B,C]) and monotonically drop contexts whose
        accumulated attention stays far below the surviving ones. Updates self.allowed in place.
        Parameter-free: the drop test uses the survivors' own attention distribution, not a constant.
        Conservative: starts after a short warmup and only drops contexts already near zero."""
        B, C = w_ctx.shape
        if self.attn_accum is None or self.attn_accum.shape != w_ctx.shape:
            self.attn_accum = torch.zeros_like(w_ctx)
            self.allowed = torch.ones(B, C, dtype=torch.bool, device=w_ctx.device)
        self.attn_accum = self.attn_accum + w_ctx
        alive = self.allowed
        nalive = alive.float().sum(dim=1, keepdim=True).clamp(min=1)
        # drop a context iff, among ALIVE contexts, it is BOTH cumulatively below the average AND
        # below the average in THIS step (still being ignored now). Double condition -> conservative
        # (a briefly-attended context is spared); both thresholds are the alive MEAN, a pure data
        # statistic -- NO fixed constant. Gradual: only the consistently-ignored set drops, and it
        # converges as generation proceeds. Keep the per-query top context and >=1 alive.
        cum_mean = (self.attn_accum * alive).sum(dim=1, keepdim=True) / nalive
        now_mean = (w_ctx * alive).sum(dim=1, keepdim=True) / nalive
        acc_a = self.attn_accum.masked_fill(~alive, -1.0)
        keep_top = acc_a == acc_a.max(dim=1, keepdim=True).values
        drop = alive & (self.attn_accum < cum_mean) & (w_ctx < now_mean) & ~keep_top
        if bool(drop.any()):
            self.allowed = alive & ~drop

    def set_allowed(self, allowed):
        """allowed[b,j]=True -> query b may read context j's bank tokens (top-k selectivity)."""
        self.allowed = allowed.bool() if allowed is not None else None

    def start_relevance(self, c):
        """Record per-source-context query->bank attention mass (for read-fraction top-k)."""
        self.rec_ctx_w = int(c); self.ctx_mass = None

    def relevance(self):
        r = self.ctx_mass; self.rec_ctx_w = 0; self.ctx_mass = None
        return r            # [B, c] summed over steps & layers, or None

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
            # Qwen3 applies per-head RMSNorm to q/k (on head_dim) BEFORE RoPE; the original
            # qwen2-style hook skipped it, corrupting every hooked attention on Qwen3. No-op on
            # Qwen2.5 (no q_norm attribute).
            if getattr(self, "q_norm", None) is not None:
                q = self.q_norm(q)
            if getattr(self, "k_norm", None) is not None:
                k = self.k_norm(k)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if past_key_values is not None:
                k, v = past_key_values.update(
                    k, v, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})
            nkv = k.shape[1]; nh = q.shape[1]; rep = nh // nkv; Tc = k.shape[2]
            neg = torch.finfo(q.dtype).min
            if os.environ.get("CROSSDBG") and self.layer_idx == min(mgr.idx):
                _bk = mgr.bank.get(self.layer_idx)
                _M = (_bk[0].shape[0] if (_bk and _bk[0] is not None) else 0)
                print(f"[CROSSDBG] mode={mgr.mode} B={B} T={T} Tc={Tc} nh={nh} rep={rep} bankM={_M} "
                      f"mem={torch.cuda.memory_allocated()/1e9:.1f}G", flush=True)

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
                # capture = per-segment causal self-attention. The float own_mask is a [B,1,T,T]
                # tensor (B=10, T=1761 -> ~2GB/layer) that forces SDPA to build the full score matrix
                # -> OOM. Padding is on the LEFT (padding_side="left"); the output rows at padded
                # positions are discarded (only context-mask tokens enter the bank), so we only need
                # the REAL token rows to attend correctly. is_causal=True (FlashAttention, no score
                # matrix) is exact for the unpadded case; with left padding a real token would also
                # see the padded prefix, so we pass a compact per-key BOOL mask (key padding only,
                # broadcast over query rows) which SDPA handles without materializing [B,H,T,T].
                _valid = mgr.valid if (mgr.valid is not None and mgr.valid.shape[1] == Tc) else None
                if _valid is None or bool(_valid.all()) or T == 1:
                    # no padding -> is_causal lets SDPA use FlashAttention (no [B,H,T,T] score matrix).
                    o = F.scaled_dot_product_attention(q, repeat_kv(k, rep), repeat_kv(v, rep),
                                                       is_causal=(T > 1))
                else:
                    # Left-padded batch. Any explicit attn_mask makes SDPA materialize the full
                    # [B,H,T,T] score matrix (~2GB/layer at B=10,T=1761) -> OOM. Instead process each
                    # segment over only its REAL tokens with is_causal=True (FlashAttention, no score
                    # matrix); padded query rows are discarded anyway (only context tokens enter bank).
                    kr = repeat_kv(k, rep); vr = repeat_kv(v, rep)
                    o = torch.zeros_like(q)
                    for b in range(B):
                        sel = _valid[b].nonzero(as_tuple=True)[0]
                        if sel.numel() == 0:
                            continue
                        s0 = int(sel[0].item())  # left padding -> real tokens are a contiguous suffix
                        ob = F.scaled_dot_product_attention(
                            q[b:b+1, :, s0:, :], kr[b:b+1, :, s0:, :], vr[b:b+1, :, s0:, :],
                            is_causal=True)
                        o[b:b+1, :, s0:, :] = ob
                return self.o_proj(o.transpose(1, 2).reshape(B, T, -1)), None

            # mode == "use": query attends own cache + the shared context bank
            bank = mgr.bank.get(self.layer_idx)
            if bank is None or bank[0] is None or bank[0].shape[0] == 0:
                o = F.scaled_dot_product_attention(q, repeat_kv(k, rep), repeat_kv(v, rep),
                                                   attn_mask=own_mask(B, T, Tc, q.dtype, q.device))
                return self.o_proj(o.transpose(1, 2).reshape(B, T, -1)), None
            bK, bV, bseq = bank
            M = bK.shape[0]
            if mgr.allowed is not None and mgr.allowed.shape[0] == B and not mgr.record_attn:
                # SELECTIVE GATHER: keep only the bank tokens some query is allowed to read (their
                # union), so QK^T is computed over the SELECTED subset, not the full bank -> real FLOP
                # saving (maximal at B=1 serving). Numerically identical to masking the full bank.
                keep = mgr.allowed[:, bseq].any(dim=0)              # [M] tokens any query may read
                if not bool(keep.all()):
                    idx = keep.nonzero(as_tuple=True)[0]
                    bK = bK[idx]; bV = bV[idx]; bseq = bseq[idx]; M = idx.numel()
            # MEMORY: keep the bank K/V at n_kv_heads (NO per-step/per-layer GQA repeat -> no
            # [B,nh,M,hd] materialization, which was the decode-step OOM). These are cheap expand
            # views: [B, n_kv, M, head_dim]. SDPA uses enable_gqa=True; the manual-matmul paths
            # broadcast over the rep groups (see _gqa_scores / _gqa_ctx below). Math-identical.
            bKn = bK.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)
            bVn = bV.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)

            def _gqa_scores(qx, kkv):
                # qx:[B,nh,T,hd], kkv:[B,nkv,Mk,hd] -> [B,nh,T,Mk] without repeating kkv.
                Tq = qx.shape[2]; Mk = kkv.shape[2]
                qg = qx.reshape(B, nkv, rep, Tq, qx.shape[-1])
                return torch.matmul(qg, kkv.unsqueeze(2).transpose(-1, -2)).reshape(B, nh, Tq, Mk)

            def _gqa_ctx(wx, vkv):
                # wx:[B,nh,T,Mk] softmax weights, vkv:[B,nkv,Mk,hd] -> [B,nh,T,hd] without repeating vkv.
                Tq = wx.shape[2]; Mk = wx.shape[3]
                wg = wx.reshape(B, nkv, rep, Tq, Mk)
                return torch.matmul(wg, vkv.unsqueeze(2)).reshape(B, nh, Tq, vkv.shape[-1])
            if mgr.prune and T == 1 and self.layer_idx == min(mgr.idx) and Tc != mgr._prune_updated_step:
                # PROGRESSIVE PRUNING: aggregate THIS step's query->bank attention per source context,
                # accumulate, and (monotonically) drop the consistently-ignored contexts for NEXT steps.
                # Uses the model's own past-token attention; updates mgr.allowed -> later steps gather a
                # shrinking survivor set (faster) and stop attending the noise (cleaner). One layer/step.
                mgr._prune_updated_step = Tc
                d2 = q.shape[-1]; sc2 = 1.0 / math.sqrt(d2)
                own2 = own_mask(B, T, Tc, q.dtype, q.device)
                s_own2 = _gqa_scores(q, k) * sc2 + own2
                s_bank2 = _gqa_scores(q, bKn) * sc2
                full2 = torch.cat([s_own2, s_bank2], dim=-1).softmax(dim=-1)
                w_bank2 = full2[..., Tc:].mean(dim=1).mean(dim=1).float()       # [B,M] avg heads & query tok
                Cn = mgr.allowed.shape[1] if mgr.allowed is not None else int(bseq.max().item()) + 1
                w_ctx = torch.zeros(B, Cn, device=q.device, dtype=w_bank2.dtype)
                w_ctx.scatter_add_(1, bseq.view(1, M).expand(B, M).long(), w_bank2)
                mgr._update_survivors(w_ctx, Tc)
            if mgr.rec_ctx_w > 0:
                # READ-FRACTION telemetry: per-source-context bank mass, for top-k selective read.
                d = q.shape[-1]; sc = 1.0 / math.sqrt(d)
                own_a = own_mask(B, T, Tc, q.dtype, q.device)
                s_own = _gqa_scores(q, k) * sc + own_a
                s_bank = _gqa_scores(q, bKn) * sc
                full = torch.cat([s_own, s_bank], dim=-1).softmax(dim=-1)
                w_bank = full[..., Tc:].mean(dim=1).mean(dim=1).float()         # [B,M]
                pc = torch.zeros(B, mgr.rec_ctx_w, device=q.device, dtype=w_bank.dtype)
                pc.scatter_add_(1, bseq.view(1, M).expand(B, M).long(), w_bank)
                mgr.ctx_mass = pc.detach().float().cpu() if mgr.ctx_mass is None else mgr.ctx_mass + pc.detach().float().cpu()
            if mgr.record_attn:
                # ANALYSIS: record how much query->bank attention mass lands on each source context.
                d = q.shape[-1]; sc = 1.0 / math.sqrt(d)
                own_a = own_mask(B, T, Tc, q.dtype, q.device)
                s_own = _gqa_scores(q, k) * sc + own_a
                s_bank = _gqa_scores(q, bKn) * sc                               # [B,H,T,M]
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
                K = torch.cat([k, bKn], dim=2)                       # [B,n_kv,Tc+M,hd], un-repeated
                V = torch.cat([v, bVn], dim=2)
                out = F.scaled_dot_product_attention(q, K, V, enable_gqa=True)  # no mask -> flash kernel
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
                K = torch.cat([k, bKn], dim=2)                       # un-repeated -> enable_gqa
                V = torch.cat([v, bVn], dim=2)
                out = F.scaled_dot_product_attention(
                    q, K, V, attn_mask=torch.cat([own, bank_m], dim=-1), enable_gqa=True)
                return self.o_proj(out.transpose(1, 2).reshape(B, T, -1)), None
            # APE realignment: own (local) and bank (context) segments computed separately,
            # bank sharpened by temperature T and re-weighted by S in log space, then merged.
            d = q.shape[-1]; sc = 1.0 / math.sqrt(d)
            Ko = repeat_kv(k, rep); Vo = repeat_kv(v, rep)          # own is local (Tc tokens) -> cheap
            s_own = torch.matmul(q, Ko.transpose(-1, -2)) * sc + own
            s_bank = _gqa_scores(q, bKn) * (sc / mgr.temperature) + bank_m   # bank: no [B,nh,M,hd] repeat
            lse_own = torch.logsumexp(s_own, dim=-1, keepdim=True)
            lse_bank = torch.logsumexp(s_bank, dim=-1, keepdim=True) * (mgr.scale_s * mgr.temperature)
            mmax = torch.maximum(lse_own, lse_bank)
            wo = torch.exp(lse_own - mmax); wb = torch.exp(lse_bank - mmax)
            o_own = torch.matmul(torch.softmax(s_own, dim=-1), Vo)
            o_bank = _gqa_ctx(torch.softmax(s_bank, dim=-1), bVn)   # bank ctx: no repeat
            out = (wo * o_own + wb * o_bank) / (wo + wb).clamp(min=1e-9)
            return self.o_proj(out.transpose(1, 2).reshape(B, T, -1)), None

        return fwd
