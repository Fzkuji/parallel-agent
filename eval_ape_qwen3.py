"""Faithful port of the OFFICIAL APE three-phase parallel encoding to Qwen3-8B.

Mirrors /mnt/data/zichuanfu/APE/ape/ape_llama.py + demo_ape.py semantics EXACTLY, only
swapping flash_attn for SDPA (transformers 4.56 'ensemble' env has Qwen3 support but no
flash_attn) and adding Qwen3 QK-norm. Three official mechanisms preserved:

  (1) SHARED PREFIX: prefix (task instruction) encoded once, broadcast (.repeat(bsz,...))
      to every context segment so each segment attends the shared prefix. After phase 2 the
      prefix KV is kept ONCE and context KV flattened by mask.
  (2) TEMPERATURE T=0.9: in the query phase, the context segment's attention uses
      softmax_scale = 1/(sqrt(head_dim)*T); prefix/query segment uses the standard scale.
  (3) SCALE S=0.9: when merging the (prefix+query) LSE with the context LSE, the context
      LSE is multiplied by (S*T) before the two-way softmax.

Qwen3 specifics handled: q_norm/k_norm are per-head RMSNorm over head_dim, applied BEFORE
RoPE (see Qwen3Attention.forward). RoPE is computed at model level from position_ids, so we
drive position_ids per phase manually (mirroring the official position bookkeeping):
  prefix:  [0, len_prefix)
  context: [len_prefix, len_prefix+seg_len)   (every segment shares this range -> parallel)
  query:   re-fed prompt tokens start at max(cache positions)+1 = len_prefix + max_seg_len

This file does NOT touch /mnt/data/zichuanfu/parallel-agent/src/** ; it only borrowed the
QK-norm + online-LSE-merge pattern as reference.
"""
import argparse
import json
import math
import random
import re
import string
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


# ----------------------------------------------------------------------------- utils
def seed_everything(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


# ---- LongBench official qa_f1 (identical to eval_ape_longbench.py) ----
def normalize_answer(s):
    def rm_art(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def ws(t): return " ".join(t.split())
    def rp(t): return "".join(c for c in t if c not in set(string.punctuation))
    return ws(rm_art(rp(s.lower())))


def qa_f1(pred, gt):
    pt = normalize_answer(pred).split(); gt_ = normalize_answer(gt).split()
    com = Counter(pt) & Counter(gt_); ns = sum(com.values())
    if ns == 0:
        return 0.0
    p = ns / len(pt); r = ns / len(gt_)
    return 2 * p * r / (p + r)


def best_f1(pred, answers):
    return max((qa_f1(pred, a) for a in answers), default=0.0)


def split_passages(context):
    parts = re.split(r"\n(?=Passage\s*\d+[:：])", context.strip())
    if len(parts) < 2:
        parts = [b for b in context.split("\n\n") if b.strip()]
    return [p.strip() for p in parts if p.strip()]


# LongBench 2wikimqa official prompt format (same strings as eval_ape_longbench.py).
PREFIX_INSTR = ("Answer the question based on the given passages. Only give me the answer and do not "
                "output any other words.\n\nThe following are given passages.\n")
SUFFIX_TMPL = ("\n\nAnswer the question based on the given passages. Only give me the answer and do not "
               "output any other words.\n\nQuestion: {q}\nAnswer:")


# ----------------------------------------------------------------------------- APE manager
class APEManager:
    """Swaps every Qwen3Attention.forward with a phase-aware custom forward. Phases:
    'prefix' / 'context' / 'query'. Per-layer KV banks are held here as plain tensors,
    exactly like the official tuple-style past_key_value (k, v, position)."""

    def __init__(self, model, temperature=0.9, scale=0.9):
        self.model = model
        self.T = float(temperature)
        self.S = float(scale)
        self.phase = "prefix"
        self.len_prefix = 0
        self.len_context = 0
        # per-layer cache: dict[layer_idx] -> (k, v) with k/v shape [B, n_kv, seq, hd] (RoPE applied)
        self.bank = {}
        self._orig = {}
        self._register()

    def _layers(self):
        return self.model.model.layers

    def _register(self):
        for i, layer in enumerate(self._layers()):
            attn = layer.self_attn
            attn.layer_idx = i
            self._orig[i] = attn.forward
            mgr = self

            def make(attn_mod, idx):
                def fwd(self_attn, hidden_states, position_embeddings, attention_mask=None,
                        past_key_values=None, cache_position=None, **kw):
                    return mgr._attn(self_attn, idx, hidden_states, position_embeddings)
                return fwd

            import types
            attn.forward = types.MethodType(make(attn, i), attn)

    # -- the three phases share projection + QK-norm + RoPE, then branch on self.phase --
    def _project(self, attn, hidden_states, position_embeddings):
        input_shape = hidden_states.shape[:-1]                    # (B, T) -- robust to extra squeeze
        B, T = input_shape[0], input_shape[1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        # Qwen3: q_norm/k_norm (RMSNorm over head_dim) applied BEFORE RoPE.
        q = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v, B, T

    def _attn(self, attn, idx, hidden_states, position_embeddings):
        q, k, v, B, T = self._project(attn, hidden_states, position_embeddings)
        rep = attn.config.num_attention_heads // attn.config.num_key_value_heads
        hd = attn.head_dim

        if self.phase == "prefix":
            # Standard causal self-attention. Store prefix KV (one copy per batch row; bsz=1 here).
            self.bank[idx] = (k, v)
            self.len_prefix = T
            out = F.scaled_dot_product_attention(q, repeat_kv(k, rep), repeat_kv(v, rep),
                                                 is_causal=(T > 1))
            return attn.o_proj(out.transpose(1, 2).reshape(B, T, -1)), None

        if self.phase == "context":
            # Each context segment (batch dim) attends [broadcast prefix KV] ++ [own context KV].
            past_k, past_v = self.bank[idx]                       # [B, n_kv, len_prefix, hd] (broadcast)
            full_k = torch.cat([past_k, k], dim=2)
            full_v = torch.cat([past_v, v], dim=2)
            self.bank[idx] = (full_k, full_v)
            out = F.scaled_dot_product_attention(q, repeat_kv(full_k, rep), repeat_kv(full_v, rep),
                                                 is_causal=(T > 1))
            return attn.o_proj(out.transpose(1, 2).reshape(B, T, -1)), None

        # phase == "query": re-fed prompt tokens (B=1) attend the stitched cache:
        #   cache = [prefix (len_prefix)] ++ [flattened context (len_context)]
        # split the cache into the CONTEXT segment (T=0.9 scale) and the OTHER segment
        # (prefix + already-generated query tokens, standard scale), then LSE-merge.
        past_k, past_v = self.bank[idx]                          # [1, n_kv, len_prefix+len_context, hd]
        full_k = torch.cat([past_k, k], dim=2)
        full_v = torch.cat([past_v, v], dim=2)
        self.bank[idx] = (full_k, full_v)

        Kr = repeat_kv(full_k, rep); Vr = repeat_kv(full_v, rep)  # [1, nh, S, hd]
        lp, lc = self.len_prefix, self.len_context
        K_ctx = Kr[:, :, lp:lp + lc]; V_ctx = Vr[:, :, lp:lp + lc]
        K_oth = torch.cat([Kr[:, :, :lp], Kr[:, :, lp + lc:]], dim=2)
        V_oth = torch.cat([Vr[:, :, :lp], Vr[:, :, lp + lc:]], dim=2)

        # OTHER segment: prefix + ALL query tokens (cached + current), standard scale.
        # prefix block: always visible. query block: causal -- the current T rows (the LAST T
        # query columns) may see all earlier query columns but not later ones. Works for both the
        # prefill step (Lqc=0, T=Lin) and decode steps (Lqc>0, T=1).
        sc = 1.0 / math.sqrt(hd)
        Sq = K_oth.shape[2]                                      # = lp + Lq_total
        Lqt = Sq - lp                                            # total query cols (cached + current)
        neg = torch.finfo(q.dtype).min
        oth_mask = torch.zeros(1, 1, T, Sq, dtype=q.dtype, device=q.device)
        # the T current rows correspond to the LAST T query columns: row r -> query col (Lqt-T+r).
        row_qcol = torch.arange(Lqt - T, Lqt, device=q.device).view(T, 1)
        key_qcol = torch.arange(Lqt, device=q.device).view(1, Lqt)
        causal = (key_qcol > row_qcol)                           # [T, Lqt] : future query cols masked
        oth_mask[:, :, :, lp:] = torch.where(causal, neg, 0.0).view(1, 1, T, Lqt)
        s_oth = torch.matmul(q, K_oth.transpose(-1, -2)) * sc + oth_mask
        lse_oth = torch.logsumexp(s_oth, dim=-1, keepdim=True)
        o_oth = torch.matmul(torch.softmax(s_oth, dim=-1), V_oth)

        # CONTEXT segment: temperature T, non-causal (the query sees all context tokens).
        s_ctx = torch.matmul(q, K_ctx.transpose(-1, -2)) * (1.0 / (math.sqrt(hd) * self.T))
        lse_ctx = torch.logsumexp(s_ctx, dim=-1, keepdim=True)
        o_ctx = torch.matmul(torch.softmax(s_ctx, dim=-1), V_ctx)

        # merge: context LSE scaled by (S*T), two-way softmax over the two segments' LSEs.
        lse_ctx = lse_ctx * (self.S * self.T)
        w = torch.softmax(torch.cat([lse_ctx, lse_oth], dim=-1), dim=-1)  # [1, nh, T, 2]
        out = w[..., 0:1] * o_ctx + w[..., 1:2] * o_oth
        return attn.o_proj(out.transpose(1, 2).reshape(B, T, -1)), None

    # ------------------------------------------------------------------ phase drivers
    def _rotary(self, hidden, position_ids):
        return self.model.model.rotary_emb(hidden, position_ids)

    def _run_layers(self, hidden, position_ids):
        pe = self._rotary(hidden, position_ids)
        for layer in self._layers():
            # Qwen3DecoderLayer.forward returns a bare Tensor in transformers 4.56 (NOT a tuple).
            hidden = layer(hidden, position_embeddings=pe, attention_mask=None)
        return hidden

    @torch.no_grad()
    def generate(self, tok, prefix_ids, ctx_ids, ctx_mask, query_ids, max_new, device):
        """prefix_ids [1,Lp], ctx_ids [B,Lseg] (left/right padded), ctx_mask [B*Lseg] valid mask,
        query_ids [1,Lq]. Returns decoded answer string."""
        emb = self.model.model.embed_tokens
        # ---- phase 1: prefix ----
        self.phase = "prefix"
        self.bank = {}
        lp = prefix_ids.shape[1]
        pos = torch.arange(lp, device=device).unsqueeze(0)
        h = emb(prefix_ids.to(device))
        self._run_layers(h, pos)                                 # fills bank with prefix KV
        # broadcast prefix KV to bsz context segments
        bsz = ctx_ids.shape[0]
        for i in self.bank:
            k, v = self.bank[i]
            self.bank[i] = (k.repeat(bsz, 1, 1, 1), v.repeat(bsz, 1, 1, 1))
        # ---- phase 2: context ----
        self.phase = "context"
        seg = ctx_ids.shape[1]
        pos = torch.arange(lp, lp + seg, device=device).unsqueeze(0).expand(bsz, seg)
        h = emb(ctx_ids.to(device))
        self._run_layers(h, pos)                                 # appends context KV per segment
        # ---- stitch: keep prefix once, flatten context by mask ----
        mask = ctx_mask.to(device).bool()                        # [B*seg]
        for i in self.bank:
            k, v = self.bank[i]                                  # [B, n_kv, lp+seg, hd]
            nkv, hd = k.shape[1], k.shape[3]
            pk = k[:1, :, :lp, :]                                # prefix (one copy)
            pv = v[:1, :, :lp, :]
            ck = k[:, :, lp:, :].permute(0, 2, 1, 3).reshape(bsz * seg, nkv, hd)[mask]
            cv = v[:, :, lp:, :].permute(0, 2, 1, 3).reshape(bsz * seg, nkv, hd)[mask]
            ck = ck.permute(1, 0, 2).unsqueeze(0)                # [1, n_kv, len_context, hd]
            cv = cv.permute(1, 0, 2).unsqueeze(0)
            self.bank[i] = (torch.cat([pk, ck], dim=2), torch.cat([pv, cv], dim=2))
        self.len_prefix = lp
        self.len_context = int(mask.sum().item())
        # ---- phase 3: query ----  re-feed [prefix ++ context_flat ++ query] as query tokens.
        self.phase = "query"
        ctx_flat = ctx_ids.reshape(-1)[mask.cpu()].unsqueeze(0).to(device)
        input_ids = torch.cat([prefix_ids.to(device), ctx_flat, query_ids.to(device)], dim=-1)
        # query positions start at max(cache positions)+1 = lp + seg  (parallel segments share range)
        start = lp + seg
        Lin = input_ids.shape[1]
        out_ids = []
        cur_ids = input_ids
        cur_pos = torch.arange(start, start + Lin, device=device).unsqueeze(0)
        eos = tok.eos_token_id
        # each query-phase forward APPENDS the new tokens' KV to self.bank (see _attn query branch),
        # so the cache grows by one token per decode step -- standard incremental decoding.
        for step in range(max_new):
            h = emb(cur_ids)
            h = self._run_layers(h, cur_pos)                     # pre-final-norm hidden states
            h = self.model.model.norm(h)                         # Qwen3 final RMSNorm
            logits = self.model.lm_head(h)
            nxt = logits[:, -1, :].argmax(-1)
            tid = int(nxt.item())
            if tid == eos:
                break
            out_ids.append(tid)
            cur_ids = nxt.view(1, 1)
            cur_pos = cur_pos[:, -1:] + 1
        return tok.decode(out_ids, skip_special_tokens=True)

    def restore(self):
        for i, layer in enumerate(self._layers()):
            layer.self_attn.forward = self._orig[i]


# ----------------------------------------------------------------------------- prompt build
# Qwen3 chat template (enable_thinking=False) is:
#   <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
# We split it so the prefix = user header + instruction, the context = passages, the suffix =
# question + assistant header. Each part is tokenized with add_special_tokens=False so the markers
# come only from these strings (no extra BOS -- Qwen3 has no BOS in its chat template).
PREFIX_HEAD = "<|im_start|>user\n"
ASSIST_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_prefix_str():
    return PREFIX_HEAD + PREFIX_INSTR


def build_suffix_str(question):
    return SUFFIX_TMPL.format(q=question) + ASSIST_TAIL


@torch.no_grad()
def concat_generate(tok, model, passages, question, device, max_new, seg_cap):
    """Same prompt content as APE, but a single standard causal pass (no APE). Sanity baseline."""
    # truncate each passage to seg_cap tokens (match APE's per-segment cap) then join.
    ctx_pieces = []
    for p in passages:
        ids = tok(p, add_special_tokens=False, truncation=True, max_length=seg_cap).input_ids
        ctx_pieces.append(tok.decode(ids, skip_special_tokens=False))
    context = "".join(ctx_pieces)
    full = build_prefix_str() + context + build_suffix_str(question)
    ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, num_beams=1,
                         pad_token_id=tok.pad_token_id)[0]
    return tok.decode(out[ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def ape_generate(mgr, tok, passages, question, device, max_new, seg_cap):
    prefix_ids = tok(build_prefix_str(), return_tensors="pt", add_special_tokens=False).input_ids
    query_ids = tok(build_suffix_str(question), return_tensors="pt", add_special_tokens=False).input_ids
    enc = tok(passages, return_tensors="pt", add_special_tokens=False, truncation=True,
              max_length=seg_cap, padding=True)
    ctx_ids = enc.input_ids                                       # [B, seg] right-padded
    ctx_mask = (ctx_ids != tok.pad_token_id).reshape(-1)
    return mgr.generate(tok, prefix_ids, ctx_ids, ctx_mask, query_ids, max_new, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/data/zichuanfu/models/Qwen3-8B")
    ap.add_argument("--task", default="2wikimqa")
    ap.add_argument("--data-dir", default="/mnt/data/zichuanfu/longbench_export")
    ap.add_argument("--num-q", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--scale", type=float, default=0.9)
    ap.add_argument("--seg-cap", type=int, default=4500)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--mode", choices=["ape", "concat", "both"], default="both")
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    seed_everything(42)
    device = torch.device(f"cuda:{args.gpu}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    data = [json.loads(l) for l in open(f"{args.data_dir}/{args.task}.jsonl") if l.strip()][:args.num_q]

    # CONCAT baseline (uses the unmodified model) -- run FIRST while attention is unpatched.
    if args.mode in ("concat", "both"):
        tot = 0.0; n = 0
        for ex in data:
            ps = split_passages(ex["context"])
            try:
                pred = concat_generate(tok, model, ps, ex["input"], device, args.max_new, args.seg_cap)
            except Exception as e:
                print(f"[concat skip {n}] {type(e).__name__}: {str(e)[:80]}", flush=True); pred = ""
            tot += best_f1(pred, ex["answers"]); n += 1
            if args.debug:
                print(f"[concat] GOLD={ex['answers']} | PRED={pred[:120]!r}", flush=True)
            if n % 25 == 0:
                print(f"  [concat {n}/{len(data)}] qa_f1={100*tot/n:.2f}", flush=True)
        print(f"{args.task} Qwen3 CONCAT qa_f1={100*tot/n:.2f} (n={n})", flush=True)

    # OFFICIAL-APE (patched attention).
    if args.mode in ("ape", "both"):
        mgr = APEManager(model, temperature=args.temperature, scale=args.scale)
        tot = 0.0; n = 0
        for ex in data:
            ps = split_passages(ex["context"])
            try:
                pred = ape_generate(mgr, tok, ps, ex["input"], device, args.max_new, args.seg_cap)
            except Exception as e:
                print(f"[ape skip {n}] {type(e).__name__}: {str(e)[:80]}", flush=True); pred = ""
            tot += best_f1(pred, ex["answers"]); n += 1
            if args.debug:
                print(f"[ape] GOLD={ex['answers']} | PRED={pred[:120]!r}", flush=True)
            if n % 25 == 0:
                print(f"  [ape {n}/{len(data)}] qa_f1={100*tot/n:.2f}", flush=True)
        mgr.restore()
        print(f"{args.task} Qwen3 OFFICIAL-APE (T={args.temperature} S={args.scale}, passage) "
              f"qa_f1={100*tot/n:.2f} (n={n})", flush=True)


if __name__ == "__main__":
    main()
