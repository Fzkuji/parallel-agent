#!/usr/bin/env python3
"""
Cross-Sequence Communication Methods - V2 with forced information asymmetry

Key fix from V1: Use MuSiQue dataset where each question REQUIRES multiple paragraphs.
Split the required paragraphs across agents so neither can answer alone.

Also fix: untrained cross-seq methods should NOT affect generation (gate too high).
Test with gate=0 first to confirm methods don't break generation, then with trained gate.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================
# Cross-Sequence Methods (same as v1 but with gate fix)
# ============================================================

class ContextSlotExchange(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(-3.0))
        self.ln = nn.LayerNorm(hidden_size)
        nn.init.zeros_(self.out_proj.weight)
    
    def set_context_slots(self, slots: torch.Tensor):
        self.register_buffer('_slots', slots, persistent=False)
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, '_slots') or self._slots is None:
            return hidden
        N = hidden.size(0)
        slots = self._slots
        q = self.q_proj(hidden).view(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        k = self.k_proj(slots).view(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        v = self.v_proj(slots).view(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        attn = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(1, 0, 2).reshape(N, self.hidden_size)
        cross_info = self.out_proj(self.ln(out))
        g = torch.sigmoid(self.gate)
        return hidden + g * cross_info


class GatedResidualMixing(nn.Module):
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
        N = hidden.size(0)
        summaries = self.summary_proj(hidden)
        global_ctx = summaries.mean(dim=0, keepdim=True).expand(N, -1)
        mixed = self.mix_mlp(torch.cat([hidden, global_ctx], dim=-1))
        mixed = self.ln(mixed)
        g = torch.sigmoid(self.gate)
        return hidden + g * mixed


class HiddenStateAveraging(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(-3.0))
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h_mean = hidden.mean(dim=0, keepdim=True).expand_as(hidden)
        diff = self.ln(h_mean - hidden)
        g = torch.sigmoid(self.gate)
        return hidden + g * diff


# ============================================================
# Generator
# ============================================================
class CrossSeqGenerator:
    def __init__(self, model, tokenizer, method: nn.Module, device='cuda'):
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            self.model = model
        else:
            self.model = model.to(device)
        self.tokenizer = tokenizer
        self.method = method.to(device=device, dtype=next(model.parameters()).dtype)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens=32, do_sample=False):
        batch_size = input_ids.size(0)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id
        stop_ids = {eos_id, pad_id}
        if hasattr(self.model.config, 'eos_token_id'):
            cfg_eos = self.model.config.eos_token_id
            if isinstance(cfg_eos, list): stop_ids.update(cfg_eos)
            elif cfg_eos is not None: stop_ids.add(cfg_eos)
        for name in ['<|im_end|>', '<|endoftext|>']:
            tid = self.tokenizer.convert_tokens_to_ids(name)
            if tid != self.tokenizer.unk_token_id: stop_ids.add(tid)
        stop_ids.discard(None)
        
        # Prefill
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=True, output_hidden_states=True,
        )
        past_kv = outputs.past_key_values
        last_hidden_all = outputs.hidden_states[-1]
        
        if isinstance(self.method, ContextSlotExchange):
            real_lens = attention_mask.sum(dim=1).long()
            slots = torch.stack([last_hidden_all[i, real_lens[i]-1, :] for i in range(batch_size)])
            self.method.set_context_slots(slots)
        
        # First token from prefill logits
        last_hidden = last_hidden_all[:, -1, :]  # [N, d]
        if batch_size > 1:
            mixed = self.method(last_hidden)
        else:
            mixed = last_hidden
        logits = self.model.lm_head(mixed)
        next_tokens = logits.argmax(dim=-1)
        
        generated_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=self.device, dtype=attention_mask.dtype)], dim=-1)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        for sid in stop_ids:
            finished = finished | (next_tokens == sid)
        
        # Continue decode
        for step in range(max_new_tokens - 1):
            if finished.all(): break
            
            model_input = generated_ids[:, -1:]
            outputs = self.model(
                input_ids=model_input, attention_mask=attention_mask,
                past_key_values=past_kv, use_cache=True, output_hidden_states=True,
            )
            past_kv = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            
            if batch_size > 1:
                mixed = self.method(last_hidden)
            else:
                mixed = last_hidden
            
            logits = self.model.lm_head(mixed)
            next_tokens = logits.argmax(dim=-1)
            next_tokens = next_tokens.masked_fill(finished, pad_id)
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, (~finished).long().unsqueeze(-1)], dim=-1)
            
            for sid in stop_ids:
                finished = finished | (next_tokens == sid)
        
        return generated_ids


# ============================================================
# Eval utilities
# ============================================================
def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens: return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common: return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_answer(text: str) -> str:
    text = text.strip()
    for prefix in ['Answer:', 'answer:', 'The answer is', 'the answer is']:
        if prefix in text:
            ans = text.split(prefix)[-1].strip()
            return ans.split('\n')[0].strip().rstrip('.')
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[-1] if lines else text


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/autodl-fs/data/models/Qwen2.5-7B-Instruct')
    parser.add_argument('--num_groups', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--output_dir', default='/root/autodl-tmp/cross_seq_v2')
    parser.add_argument('--methods', nargs='+', 
                        default=['independent', 'oracle', 'cse', 'grm', 'hsa'])
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
        args.model_path, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True, attn_implementation='eager',
    )
    model.eval()
    hidden_size = model.config.hidden_size
    device = 'cuda'
    
    # Load HotpotQA but with STRICT 2-hop filtering
    log.info("Loading HotpotQA (strict 2-hop)...")
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    
    groups = []
    skipped_single_para = 0
    
    for item in ds:
        if item['type'] != 'bridge':  # Only bridge questions (require chaining)
            continue
        
        sp_facts = item['supporting_facts']
        titles = list(set(sp_facts['title']))
        if len(titles) < 2:
            continue
        
        ctx_dict = {}
        for t, ss in zip(item['context']['title'], item['context']['sentences']):
            ctx_dict[t] = ''.join(ss)
        
        if titles[0] not in ctx_dict or titles[1] not in ctx_dict:
            continue
        
        c1 = f"{titles[0]}: {ctx_dict[titles[0]]}"
        c2 = f"{titles[1]}: {ctx_dict[titles[1]]}"
        
        groups.append({
            'question': item['question'],
            'answer': item['answer'],
            'context_1': c1,
            'context_2': c2,
            'full_context': f"{c1}\n\n{c2}",
        })
        
        if len(groups) >= args.num_groups:
            break
    
    log.info(f"Prepared {len(groups)} bridge-type groups")
    
    def make_prompt(context: str, question: str) -> str:
        messages = [
            {"role": "system", "content": "Answer the question concisely based on the given context. Give only the answer, no explanation."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    method_classes = {
        'cse': lambda: ContextSlotExchange(hidden_size, num_heads=8),
        'grm': lambda: GatedResidualMixing(hidden_size),
        'hsa': lambda: HiddenStateAveraging(hidden_size),
    }
    
    results = {}
    
    # Also track per-agent scores to understand information distribution
    for method_name in args.methods:
        log.info(f"\n{'='*60}")
        log.info(f"Testing method: {method_name}")
        log.info(f"{'='*60}")
        
        f1_scores = []
        agent1_scores = []
        agent2_scores = []
        
        for gi, group in enumerate(groups):
            if method_name == 'oracle':
                prompts = [make_prompt(group['full_context'], group['question'])]
                encoded = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048)
                with torch.no_grad():
                    out = model.generate(
                        input_ids=encoded['input_ids'].to(device),
                        attention_mask=encoded['attention_mask'].to(device),
                        max_new_tokens=args.max_new_tokens, do_sample=False,
                    )
                gen_ids = out[:, encoded['input_ids'].size(1):]
                texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                ans = extract_answer(texts[0])
                f1 = compute_f1(ans, group['answer'])
                f1_scores.append(f1)
                agent1_scores.append(f1)
                agent2_scores.append(f1)
                
            elif method_name == 'independent':
                prompts = [
                    make_prompt(group['context_1'], group['question']),
                    make_prompt(group['context_2'], group['question']),
                ]
                encoded = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048)
                with torch.no_grad():
                    out = model.generate(
                        input_ids=encoded['input_ids'].to(device),
                        attention_mask=encoded['attention_mask'].to(device),
                        max_new_tokens=args.max_new_tokens, do_sample=False,
                    )
                gen_ids = out[:, encoded['input_ids'].size(1):]
                texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                
                f1_1 = compute_f1(extract_answer(texts[0]), group['answer'])
                f1_2 = compute_f1(extract_answer(texts[1]), group['answer'])
                best_f1 = max(f1_1, f1_2)
                f1_scores.append(best_f1)
                agent1_scores.append(f1_1)
                agent2_scores.append(f1_2)
                
            else:
                prompts = [
                    make_prompt(group['context_1'], group['question']),
                    make_prompt(group['context_2'], group['question']),
                ]
                encoded = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048)
                
                method_module = method_classes[method_name]()
                gen = CrossSeqGenerator(model, tokenizer, method_module, device=device)
                out = gen.generate(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                )
                gen_ids = out[:, encoded['input_ids'].size(1):]
                texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                
                f1_1 = compute_f1(extract_answer(texts[0]), group['answer'])
                f1_2 = compute_f1(extract_answer(texts[1]), group['answer'])
                best_f1 = max(f1_1, f1_2)
                f1_scores.append(best_f1)
                agent1_scores.append(f1_1)
                agent2_scores.append(f1_2)
            
            if (gi + 1) % 20 == 0:
                avg = sum(f1_scores) / len(f1_scores)
                a1 = sum(agent1_scores) / len(agent1_scores)
                a2 = sum(agent2_scores) / len(agent2_scores)
                log.info(f"  [{method_name}] {gi+1}/{len(groups)}: avg={avg:.4f} a1={a1:.4f} a2={a2:.4f}")
        
        avg_f1 = sum(f1_scores) / len(f1_scores)
        avg_a1 = sum(agent1_scores) / len(agent1_scores)
        avg_a2 = sum(agent2_scores) / len(agent2_scores)
        results[method_name] = {
            'avg_f1': avg_f1,
            'agent1_f1': avg_a1,
            'agent2_f1': avg_a2,
        }
        log.info(f"  {method_name} FINAL: avg={avg_f1:.4f} a1={avg_a1:.4f} a2={avg_a2:.4f}")
    
    # Summary
    log.info(f"\n{'='*60}")
    log.info("RESULTS SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  {'Method':15s} {'Best F1':>10s} {'Agent1':>10s} {'Agent2':>10s}")
    log.info(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    for name, r in sorted(results.items(), key=lambda x: x[1]['avg_f1'], reverse=True):
        log.info(f"  {name:15s} {r['avg_f1']:10.4f} {r['agent1_f1']:10.4f} {r['agent2_f1']:10.4f}")
    
    # Compute collaboration gap
    if 'oracle' in results and 'independent' in results:
        gap = results['oracle']['avg_f1'] - results['independent']['avg_f1']
        log.info(f"\n  Collaboration gap (Oracle - Independent) = {gap:+.4f}")
        if gap <= 0:
            log.info("  ⚠️  No collaboration gap! This dataset/split doesn't need cross-sequence info.")
        else:
            log.info(f"  ✅ Collaboration gap exists! Cross-sequence methods can potentially gain up to {gap:.4f}")
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved to {args.output_dir}/results.json")


if __name__ == '__main__':
    main()
