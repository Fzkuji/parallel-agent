#!/usr/bin/env python3
"""
Cross-Sequence Communication Methods - Comparative Experiment

Tests 5 methods for cross-sequence information sharing in batch-parallel multi-agent inference:
1. Context Slot Exchange (CSE): Share prefill-end hidden state, cross-attn during decode
2. Cross-Sequence KV Injection (KVI): Inject other sequences' last-K KV pairs into attention
3. Shared Memory Bank (SMB): External memory matrix read/written by all sequences per layer
4. Gated Residual Mixing (GRM): Average hidden states at specific positions, gate + inject
5. Hidden State Averaging (HSA): Simple position-wise averaging with gate (baseline)

All methods compared against:
- Independent: No cross-sequence interaction
- Oracle: Full context concatenation

Dataset: HotpotQA 2-agent split (each agent sees one supporting paragraph)
Model: Qwen2.5-7B-Instruct
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ============================================================
# Method 1: Context Slot Exchange (CSE)
# ============================================================
class ContextSlotExchange(nn.Module):
    """
    After prefill, extract each sequence's last hidden state as "context slot".
    During decode, each sequence attends to other sequences' context slots.
    Slots are FIXED after prefill - no updates during generation.
    """
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ≈ 0.047
        self.ln = nn.LayerNorm(hidden_size)
        
        nn.init.zeros_(self.out_proj.weight)
    
    def set_context_slots(self, slots: torch.Tensor):
        """Set context slots from prefill. slots: [N, d]"""
        self.register_buffer('_slots', slots, persistent=False)
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [N, d] current decode step. Returns modified [N, d]."""
        if not hasattr(self, '_slots') or self._slots is None:
            return hidden
        
        N = hidden.size(0)
        slots = self._slots  # [N, d]
        
        q = self.q_proj(hidden).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(slots).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(slots).view(N, self.num_heads, self.head_dim)
        
        # Each sequence attends to ALL slots (including own)
        q = q.permute(1, 0, 2)  # [H, N, d_h]
        k = k.permute(1, 0, 2)  # [H, N, d_h]
        v = v.permute(1, 0, 2)  # [H, N, d_h]
        
        attn = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(1, 0, 2).reshape(N, self.hidden_size)
        
        cross_info = self.out_proj(self.ln(out))
        g = torch.sigmoid(self.gate)
        return hidden + g * cross_info


# ============================================================
# Method 2: Cross-Sequence KV Injection (KVI)
# ============================================================
class KVInjection(nn.Module):
    """
    During generation, inject other sequences' context KV (last K positions)
    into each sequence's attention computation.
    
    Implementation: After normal self-attn, do an additional cross-attn to
    other sequences' stored KV pairs.
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, num_kv_slots: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_slots = num_kv_slots
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(-3.0))
        self.ln = nn.LayerNorm(hidden_size)
        
        nn.init.zeros_(self.out_proj.weight)
    
    def set_context_kvs(self, context_hiddens: torch.Tensor, attention_mask: torch.Tensor):
        """
        Extract last K hidden states from each sequence's context as KV pairs.
        context_hiddens: [N, L, d] - full prefill hidden states
        attention_mask: [N, L] - to find real token positions
        """
        N, L, d = context_hiddens.shape
        K = self.num_kv_slots
        
        kvs = []
        for i in range(N):
            real_len = attention_mask[i].sum().item()
            start = max(0, int(real_len) - K)
            end = int(real_len)
            kv = context_hiddens[i, start:end, :]  # [<=K, d]
            # Pad if needed
            if kv.size(0) < K:
                pad = torch.zeros(K - kv.size(0), d, device=kv.device, dtype=kv.dtype)
                kv = torch.cat([pad, kv], dim=0)
            kvs.append(kv)
        
        self.register_buffer('_kvs', torch.stack(kvs, dim=0), persistent=False)  # [N, K, d]
    
    def forward(self, hidden: torch.Tensor, seq_idx: Optional[int] = None) -> torch.Tensor:
        """hidden: [N, d]. Each sequence attends to OTHER sequences' KV slots."""
        if not hasattr(self, '_kvs') or self._kvs is None:
            return hidden
        
        N = hidden.size(0)
        K = self._kvs.size(1)
        
        # Gather all other sequences' KVs for each sequence
        # For simplicity: attend to ALL sequences' KVs (including own)
        all_kvs = self._kvs.reshape(N * K, self.hidden_size)  # [N*K, d]
        
        q = self.q_proj(hidden).view(N, self.num_heads, self.head_dim)  # [N, H, d_h]
        k = self.k_proj(all_kvs).view(N * K, self.num_heads, self.head_dim)  # [N*K, H, d_h]
        v = self.v_proj(all_kvs).view(N * K, self.num_heads, self.head_dim)
        
        q = q.permute(1, 0, 2)  # [H, N, d_h]
        k = k.permute(1, 0, 2)  # [H, N*K, d_h]
        v = v.permute(1, 0, 2)
        
        attn = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)  # [H, N, N*K]
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(1, 0, 2).reshape(N, self.hidden_size)
        
        cross_info = self.out_proj(self.ln(out))
        g = torch.sigmoid(self.gate)
        return hidden + g * cross_info


# ============================================================
# Method 3: Shared Memory Bank (SMB)
# ============================================================
class SharedMemoryBank(nn.Module):
    """
    Fixed-size memory bank M ∈ R^{K×d} shared by all sequences.
    Each sequence reads from M and writes to M per step.
    """
    def __init__(self, hidden_size: int, num_slots: int = 8, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Memory slots (learnable init)
        self.memory_init = nn.Parameter(torch.randn(num_slots, hidden_size) * 0.02)
        
        # Read: cross-attn from hidden to memory
        self.read_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.read_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.read_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.read_out = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Write: cross-attn from memory to hidden (update memory)
        self.write_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.write_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.write_v = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.gate = nn.Parameter(torch.tensor(-3.0))
        self.ln = nn.LayerNorm(hidden_size)
        self.write_gate = nn.Parameter(torch.tensor(-2.0))
        
        nn.init.zeros_(self.read_out.weight)
    
    def reset_memory(self):
        self.register_buffer('_memory', self.memory_init.clone(), persistent=False)
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [N, d]. Read from memory, write back."""
        if not hasattr(self, '_memory'):
            self.reset_memory()
        
        N = hidden.size(0)
        M = self._memory  # [K, d]
        K = M.size(0)
        
        # Read: each sequence queries memory
        q = self.read_q(hidden).view(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        k = self.read_k(M).view(K, self.num_heads, self.head_dim).permute(1, 0, 2)
        v = self.read_v(M).view(K, self.num_heads, self.head_dim).permute(1, 0, 2)
        
        attn = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        read_out = torch.bmm(attn, v).permute(1, 0, 2).reshape(N, self.hidden_size)
        
        cross_info = self.read_out(self.ln(read_out))
        g = torch.sigmoid(self.gate)
        result = hidden + g * cross_info
        
        # Write: update memory with info from all sequences
        wg = torch.sigmoid(self.write_gate)
        write_q = self.write_q(M)  # [K, d]
        write_k = self.write_k(hidden)  # [N, d]
        write_v = self.write_v(hidden)  # [N, d]
        
        write_attn = torch.mm(write_q, write_k.t()) / (self.hidden_size ** 0.5)  # [K, N]
        write_attn = F.softmax(write_attn, dim=-1)
        new_info = torch.mm(write_attn, write_v)  # [K, d]
        
        self._memory = (1 - wg) * self._memory + wg * new_info
        
        return result


# ============================================================
# Method 4: Gated Residual Mixing (GRM)
# ============================================================
class GatedResidualMixing(nn.Module):
    """
    Extract summary from each sequence, aggregate, gate and inject back.
    Simplest meaningful cross-sequence method.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.summary_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mix_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Parameter(torch.tensor(-3.0))
        self.ln = nn.LayerNorm(hidden_size)
        
        nn.init.zeros_(self.mix_mlp[-1].weight)
        nn.init.zeros_(self.mix_mlp[-1].bias)
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [N, d]"""
        N = hidden.size(0)
        
        summaries = self.summary_proj(hidden)  # [N, d]
        global_ctx = summaries.mean(dim=0, keepdim=True).expand(N, -1)  # [N, d]
        
        mixed = self.mix_mlp(torch.cat([hidden, global_ctx], dim=-1))  # [N, d]
        mixed = self.ln(mixed)
        g = torch.sigmoid(self.gate)
        return hidden + g * mixed


# ============================================================
# Method 5: Hidden State Averaging (HSA) - simplest baseline
# ============================================================
class HiddenStateAveraging(nn.Module):
    """
    Average all sequences' hidden states, gate and add back.
    Minimal parameters, minimal complexity.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(-3.0))
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [N, d]"""
        h_mean = hidden.mean(dim=0, keepdim=True).expand_as(hidden)
        diff = self.ln(h_mean - hidden)
        g = torch.sigmoid(self.gate)
        return hidden + g * diff


# ============================================================
# Unified Generator with pluggable cross-seq method
# ============================================================
class CrossSeqGenerator:
    """Generate with any cross-sequence method."""
    
    def __init__(self, model, tokenizer, method: nn.Module, device='cuda'):
        # Don't move model if device_map='auto' was used
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            self.model = model
        else:
            self.model = model.to(device)
        self.tokenizer = tokenizer
        self.method = method.to(device=device, dtype=next(model.parameters()).dtype)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 32,
        do_sample: bool = False,
    ) -> torch.Tensor:
        batch_size = input_ids.size(0)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id
        
        # Stop tokens
        stop_ids = {eos_id, pad_id}
        if hasattr(self.model.config, 'eos_token_id'):
            cfg_eos = self.model.config.eos_token_id
            if isinstance(cfg_eos, list):
                stop_ids.update(cfg_eos)
            elif cfg_eos is not None:
                stop_ids.add(cfg_eos)
        for name in ['<|im_end|>', '<|endoftext|>']:
            tid = self.tokenizer.convert_tokens_to_ids(name)
            if tid != self.tokenizer.unk_token_id:
                stop_ids.add(tid)
        stop_ids.discard(None)
        
        # Phase 1: Prefill
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        past_kv = outputs.past_key_values
        
        # Extract context info for methods that need it
        last_hidden_all = outputs.hidden_states[-1]  # [N, L, d]
        
        if isinstance(self.method, ContextSlotExchange):
            # Get last real token position for each sequence
            real_lens = attention_mask.sum(dim=1).long()
            slots = torch.stack([
                last_hidden_all[i, real_lens[i]-1, :] for i in range(batch_size)
            ])
            self.method.set_context_slots(slots)
        
        elif isinstance(self.method, KVInjection):
            self.method.set_context_kvs(last_hidden_all, attention_mask)
        
        elif isinstance(self.method, SharedMemoryBank):
            self.method.reset_memory()
            # Initialize memory from prefill context
            real_lens = attention_mask.sum(dim=1).long()
            context_summaries = torch.stack([
                last_hidden_all[i, real_lens[i]-1, :] for i in range(batch_size)
            ])
            # Write initial context to memory
            self.method(context_summaries)
        
        # Phase 2: Decode with cross-seq interaction
        generated_ids = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for step in range(max_new_tokens):
            model_input = generated_ids[:, -1:]
            
            outputs = self.model(
                input_ids=model_input,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
            )
            past_kv = outputs.past_key_values
            
            # Get last hidden state and apply cross-seq method
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # [N, d]
            
            if batch_size > 1:
                mixed = self.method(last_hidden)
            else:
                mixed = last_hidden
            
            # Project to logits
            # Must apply final layer norm before lm_head!
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
                mixed = self.model.model.norm(mixed)
            logits = self.model.lm_head(mixed)
            
            if do_sample:
                next_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
            else:
                next_tokens = logits.argmax(dim=-1)
            
            next_tokens = next_tokens.masked_fill(finished, pad_id)
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([
                attention_mask, (~finished).long().unsqueeze(-1)
            ], dim=-1)
            
            for sid in stop_ids:
                finished = finished | (next_tokens == sid)
            if finished.all():
                break
        
        return generated_ids


# ============================================================
# Evaluation Functions
# ============================================================
def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_answer(text: str) -> str:
    """Extract answer from generated text."""
    # Try to find answer after common patterns
    for prefix in ['Answer:', 'answer:', 'The answer is', 'the answer is']:
        if prefix in text:
            ans = text.split(prefix)[-1].strip()
            # Take first line
            ans = ans.split('\n')[0].strip().rstrip('.')
            return ans
    # Fall back to last line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()


# ============================================================
# Main Experiment
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/autodl-fs/data/models/Qwen2.5-7B-Instruct')
    parser.add_argument('--num_groups', type=int, default=200)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--output_dir', default='/root/autodl-tmp/cross_seq_exp')
    parser.add_argument('--methods', nargs='+', 
                        default=['independent', 'cse', 'kvi', 'smb', 'grm', 'hsa'],
                        help='Methods to test')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    log.info(f"Loading model {args.model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation='eager',
    )
    model.eval()
    hidden_size = model.config.hidden_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load HotpotQA
    log.info("Loading HotpotQA...")
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    
    # Prepare 2-agent split groups
    groups = []
    for item in ds:
        sp_facts = item['supporting_facts']
        titles = list(set(sp_facts['title']))
        if len(titles) >= 2:
            # Find the two supporting paragraphs
            ctx_dict = {}
            for t, ss in zip(item['context']['title'], item['context']['sentences']):
                ctx_dict[t] = ''.join(ss)
            if titles[0] in ctx_dict and titles[1] in ctx_dict:
                groups.append({
                    'question': item['question'],
                    'answer': item['answer'],
                    'context_1': f"{titles[0]}: {ctx_dict[titles[0]]}",
                    'context_2': f"{titles[1]}: {ctx_dict[titles[1]]}",
                    'full_context': f"{titles[0]}: {ctx_dict[titles[0]]}\n{titles[1]}: {ctx_dict[titles[1]]}",
                })
        if len(groups) >= args.num_groups:
            break
    
    log.info(f"Prepared {len(groups)} groups")
    
    # Build prompts using chat template
    def make_prompt(context: str, question: str) -> str:
        messages = [
            {"role": "system", "content": "Answer the question concisely based on the given context. Give only the answer, no explanation."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Method registry
    method_classes = {
        'cse': lambda: ContextSlotExchange(hidden_size, num_heads=8),
        'kvi': lambda: KVInjection(hidden_size, num_heads=8, num_kv_slots=8),
        'smb': lambda: SharedMemoryBank(hidden_size, num_slots=8, num_heads=8),
        'grm': lambda: GatedResidualMixing(hidden_size),
        'hsa': lambda: HiddenStateAveraging(hidden_size),
    }
    
    results = {}
    
    # Run each method
    for method_name in args.methods:
        log.info(f"\n{'='*60}")
        log.info(f"Testing method: {method_name}")
        log.info(f"{'='*60}")
        
        f1_scores = []
        
        for gi, group in enumerate(groups):
            if method_name == 'independent':
                # Each agent generates independently
                prompts = [
                    make_prompt(group['context_1'], group['question']),
                    make_prompt(group['context_2'], group['question']),
                ]
            elif method_name == 'oracle':
                # Both agents see full context
                prompts = [
                    make_prompt(group['full_context'], group['question']),
                ]
            else:
                # Cross-seq methods: 2 agents in same batch
                prompts = [
                    make_prompt(group['context_1'], group['question']),
                    make_prompt(group['context_2'], group['question']),
                ]
            
            # Tokenize
            encoded = tokenizer(
                prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048
            )
            
            if method_name == 'independent' or method_name == 'oracle':
                # Standard generation
                with torch.no_grad():
                    out = model.generate(
                        input_ids=encoded['input_ids'].to(device),
                        attention_mask=encoded['attention_mask'].to(device),
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                gen_ids = out[:, encoded['input_ids'].size(1):]
                texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                
                # For independent, take best F1 among agents
                best_f1 = 0.0
                for t in texts:
                    ans = extract_answer(t)
                    f1 = compute_f1(ans, group['answer'])
                    best_f1 = max(best_f1, f1)
                f1_scores.append(best_f1)
            else:
                # Cross-seq method
                method_module = method_classes[method_name]()
                gen = CrossSeqGenerator(model, tokenizer, method_module, device=device)
                
                out = gen.generate(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
                gen_ids = out[:, encoded['input_ids'].size(1):]
                texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                
                best_f1 = 0.0
                for t in texts:
                    ans = extract_answer(t)
                    f1 = compute_f1(ans, group['answer'])
                    best_f1 = max(best_f1, f1)
                f1_scores.append(best_f1)
            
            if (gi + 1) % 20 == 0:
                avg_f1 = sum(f1_scores) / len(f1_scores)
                log.info(f"  [{method_name}] {gi+1}/{len(groups)}: avg F1 = {avg_f1:.4f}")
        
        avg_f1 = sum(f1_scores) / len(f1_scores)
        results[method_name] = {
            'avg_f1': avg_f1,
            'scores': f1_scores,
        }
        log.info(f"  {method_name} FINAL avg F1 = {avg_f1:.4f}")
    
    # Summary
    log.info(f"\n{'='*60}")
    log.info("RESULTS SUMMARY")
    log.info(f"{'='*60}")
    for name, r in sorted(results.items(), key=lambda x: x[1]['avg_f1'], reverse=True):
        log.info(f"  {name:15s}: F1 = {r['avg_f1']:.4f}")
    
    # Save
    save_results = {k: {'avg_f1': v['avg_f1']} for k, v in results.items()}
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)
    log.info(f"Saved to {args.output_dir}/results.json")


if __name__ == '__main__':
    main()
