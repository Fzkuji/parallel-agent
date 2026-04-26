#!/usr/bin/env python3
"""
Cross-Sequence Communication - V3: Multi-question batch with shuffled contexts

Setup:
- Take K questions from HotpotQA, each with ~10 paragraphs (2 supporting + 8 distractors)
- Create a batch of N sequences, each: [1 paragraph] + [1 question]
- Paragraphs are shuffled across sequences
- Each sequence has only 1 paragraph, needs cross-seq info to answer

Baselines:
- Independent: each sequence answers with just its own paragraph
- Oracle: each sequence gets ALL paragraphs for its question concatenated
- Cross-seq methods: each sequence can access other sequences' hidden states

This creates STRONG information asymmetry where cross-seq communication is essential.
"""

import os
import sys
import json
import time
import random
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
# Cross-Sequence Methods
# ============================================================

class ContextSlotExchange(nn.Module):
    """Share prefill-end hidden states as context slots."""
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
        k = self.k_proj(slots).view(slots.size(0), self.num_heads, self.head_dim).permute(1, 0, 2)
        v = self.v_proj(slots).view(slots.size(0), self.num_heads, self.head_dim).permute(1, 0, 2)
        attn = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(1, 0, 2).reshape(N, self.hidden_size)
        cross_info = self.out_proj(self.ln(out))
        g = torch.sigmoid(self.gate)
        return hidden + g * cross_info


class GatedResidualMixing(nn.Module):
    """Average + MLP mixing."""
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
    """Simple averaging baseline."""
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
# Generator (fixed: no double-forward of last token)
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
        
        # Set up context slots from prefill
        if isinstance(self.method, ContextSlotExchange):
            real_lens = attention_mask.sum(dim=1).long()
            slots = torch.stack([last_hidden_all[i, real_lens[i]-1, :] for i in range(batch_size)])
            self.method.set_context_slots(slots)
        
        # First token from prefill
        last_hidden = last_hidden_all[:, -1, :]
        if batch_size > 1:
            mixed = self.method(last_hidden)
        else:
            mixed = last_hidden
        logits = self.model.lm_head(mixed)
        next_tokens = logits.argmax(dim=-1)
        
        generated_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones(batch_size, 1, device=self.device, dtype=attention_mask.dtype)
        ], dim=-1)
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
# Evaluation
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
            return text.split(prefix)[-1].strip().split('\n')[0].strip().rstrip('.')
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[-1] if lines else text


# ============================================================
# Data preparation: multi-question batch with shuffled contexts
# ============================================================
def prepare_batches(dataset, tokenizer, num_batches=50, questions_per_batch=4, max_paragraphs_per_q=5):
    """
    Create batches where each batch has K questions, each with up to M paragraphs.
    Each sequence = [1 paragraph] + [1 question].
    
    Returns list of batches, each batch is a dict with:
    - prompts: list of prompt strings
    - questions: list of question strings (for each sequence)
    - answers: list of answer strings (for each sequence)
    - has_supporting: list of bool (whether this sequence's paragraph is supporting)
    - question_ids: list of int (which question this sequence belongs to)
    """
    # Filter bridge-type questions with 2+ supporting paragraphs
    valid_items = []
    for item in dataset:
        if item['type'] != 'bridge':
            continue
        sp_titles = list(set(item['supporting_facts']['title']))
        if len(sp_titles) < 2:
            continue
        
        ctx_dict = {}
        for t, ss in zip(item['context']['title'], item['context']['sentences']):
            ctx_dict[t] = ''.join(ss)
        
        # Check supporting paragraphs exist
        if all(t in ctx_dict for t in sp_titles):
            valid_items.append({
                'question': item['question'],
                'answer': item['answer'],
                'supporting_titles': sp_titles,
                'all_paragraphs': [(t, ctx_dict[t]) for t in ctx_dict],
                'supporting_paragraphs': [(t, ctx_dict[t]) for t in sp_titles],
            })
    
    random.seed(42)
    random.shuffle(valid_items)
    
    batches = []
    idx = 0
    
    for bi in range(num_batches):
        if idx + questions_per_batch > len(valid_items):
            break
        
        batch_questions = valid_items[idx:idx+questions_per_batch]
        idx += questions_per_batch
        
        prompts = []
        questions = []
        answers = []
        has_supporting = []
        question_ids = []
        paragraph_titles = []
        
        for qi, q_item in enumerate(batch_questions):
            # Take up to max_paragraphs_per_q paragraphs per question
            # Always include the 2 supporting ones
            paras = list(q_item['supporting_paragraphs'])
            other_paras = [p for p in q_item['all_paragraphs'] if p[0] not in q_item['supporting_titles']]
            random.shuffle(other_paras)
            remaining = max_paragraphs_per_q - len(paras)
            if remaining > 0:
                paras.extend(other_paras[:remaining])
            
            for title, text in paras:
                is_supporting = title in q_item['supporting_titles']
                
                messages = [
                    {"role": "system", "content": "Answer the question concisely based on the given context. Give only the answer, no explanation. If the context doesn't contain enough information, still try your best."},
                    {"role": "user", "content": f"Context: {title}: {text}\n\nQuestion: {q_item['question']}"},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                prompts.append(prompt)
                questions.append(q_item['question'])
                answers.append(q_item['answer'])
                has_supporting.append(is_supporting)
                question_ids.append(qi)
                paragraph_titles.append(title)
        
        batches.append({
            'prompts': prompts,
            'questions': questions,
            'answers': answers,
            'has_supporting': has_supporting,
            'question_ids': question_ids,
            'paragraph_titles': paragraph_titles,
            'num_questions': questions_per_batch,
        })
    
    return batches


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/autodl-fs/data/models/Qwen2.5-7B-Instruct')
    parser.add_argument('--num_batches', type=int, default=50)
    parser.add_argument('--questions_per_batch', type=int, default=4)
    parser.add_argument('--max_paragraphs_per_q', type=int, default=5)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--output_dir', default='/root/autodl-tmp/cross_seq_v3')
    parser.add_argument('--methods', nargs='+',
                        default=['independent', 'oracle', 'cse', 'grm', 'hsa'])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    log.info("Loading HotpotQA...")
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    
    log.info("Preparing batches...")
    batches = prepare_batches(
        ds, tokenizer,
        num_batches=args.num_batches,
        questions_per_batch=args.questions_per_batch,
        max_paragraphs_per_q=args.max_paragraphs_per_q,
    )
    log.info(f"Prepared {len(batches)} batches, ~{len(batches[0]['prompts'])} sequences each")
    
    # Method registry
    method_classes = {
        'cse': lambda: ContextSlotExchange(hidden_size, num_heads=8),
        'grm': lambda: GatedResidualMixing(hidden_size),
        'hsa': lambda: HiddenStateAveraging(hidden_size),
    }
    
    results = {}
    
    for method_name in args.methods:
        log.info(f"\n{'='*60}")
        log.info(f"Testing method: {method_name}")
        log.info(f"{'='*60}")
        
        all_f1 = []
        supporting_f1 = []  # F1 for sequences with supporting paragraphs
        distractor_f1 = []  # F1 for sequences with distractors
        per_question_best_f1 = []  # Best F1 per question
        
        for bi, batch in enumerate(batches):
            N = len(batch['prompts'])
            
            if method_name == 'oracle':
                # Each sequence gets ALL paragraphs for its question
                # Group by question
                q_groups = {}
                for i, qi in enumerate(batch['question_ids']):
                    if qi not in q_groups:
                        q_groups[qi] = {'titles': [], 'paras': [], 'question': batch['questions'][i], 'answer': batch['answers'][i]}
                    q_groups[qi]['titles'].append(batch['paragraph_titles'][i])
                    # Extract paragraph text from prompt (hacky but works)
                    q_groups[qi]['paras'].append(f"{batch['paragraph_titles'][i]}")
                
                # For oracle, create one prompt per question with all context
                oracle_f1s = []
                for qi, qg in q_groups.items():
                    # Collect all paragraphs for this question from all sequences
                    full_ctx_parts = []
                    for i, qid in enumerate(batch['question_ids']):
                        if qid == qi:
                            # Extract context from prompt
                            prompt_text = batch['prompts'][i]
                            # Parse context from the prompt
                            if 'Context: ' in prompt_text:
                                ctx_part = prompt_text.split('Context: ')[1].split('\n\nQuestion:')[0]
                                full_ctx_parts.append(ctx_part)
                    
                    full_context = '\n\n'.join(full_ctx_parts)
                    messages = [
                        {"role": "system", "content": "Answer the question concisely based on the given context. Give only the answer, no explanation."},
                        {"role": "user", "content": f"Context: {full_context}\n\nQuestion: {qg['question']}"},
                    ]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    encoded = tokenizer([prompt], return_tensors='pt', padding=True, truncation=True, max_length=4096)
                    
                    with torch.no_grad():
                        out = model.generate(
                            input_ids=encoded['input_ids'].to(device),
                            attention_mask=encoded['attention_mask'].to(device),
                            max_new_tokens=args.max_new_tokens, do_sample=False,
                        )
                    gen = out[:, encoded['input_ids'].size(1):]
                    text = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
                    f1 = compute_f1(extract_answer(text), qg['answer'])
                    oracle_f1s.append(f1)
                
                avg_f1 = sum(oracle_f1s) / len(oracle_f1s)
                per_question_best_f1.extend(oracle_f1s)
                all_f1.extend(oracle_f1s)
                
            elif method_name == 'independent':
                # Each sequence independently
                encoded = tokenizer(
                    batch['prompts'], return_tensors='pt', padding=True,
                    truncation=True, max_length=2048,
                )
                with torch.no_grad():
                    out = model.generate(
                        input_ids=encoded['input_ids'].to(device),
                        attention_mask=encoded['attention_mask'].to(device),
                        max_new_tokens=args.max_new_tokens, do_sample=False,
                    )
                gen_ids = out[:, encoded['input_ids'].size(1):]
                texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                
                for i, text in enumerate(texts):
                    f1 = compute_f1(extract_answer(text), batch['answers'][i])
                    all_f1.append(f1)
                    if batch['has_supporting'][i]:
                        supporting_f1.append(f1)
                    else:
                        distractor_f1.append(f1)
                
                # Per-question best
                q_f1s = {}
                for i, qi in enumerate(batch['question_ids']):
                    f1 = compute_f1(extract_answer(texts[i]), batch['answers'][i])
                    if qi not in q_f1s:
                        q_f1s[qi] = []
                    q_f1s[qi].append(f1)
                for qi, f1s in q_f1s.items():
                    per_question_best_f1.append(max(f1s))
                    
            else:
                # Cross-seq method
                encoded = tokenizer(
                    batch['prompts'], return_tensors='pt', padding=True,
                    truncation=True, max_length=2048,
                )
                method_module = method_classes[method_name]()
                gen = CrossSeqGenerator(model, tokenizer, method_module, device=device)
                out = gen.generate(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                )
                gen_ids = out[:, encoded['input_ids'].size(1):]
                texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                
                for i, text in enumerate(texts):
                    f1 = compute_f1(extract_answer(text), batch['answers'][i])
                    all_f1.append(f1)
                    if batch['has_supporting'][i]:
                        supporting_f1.append(f1)
                    else:
                        distractor_f1.append(f1)
                
                q_f1s = {}
                for i, qi in enumerate(batch['question_ids']):
                    f1 = compute_f1(extract_answer(texts[i]), batch['answers'][i])
                    if qi not in q_f1s: q_f1s[qi] = []
                    q_f1s[qi].append(f1)
                for qi, f1s in q_f1s.items():
                    per_question_best_f1.append(max(f1s))
            
            if (bi + 1) % 10 == 0:
                avg = sum(all_f1) / len(all_f1)
                log.info(f"  [{method_name}] batch {bi+1}/{len(batches)}: overall_avg={avg:.4f}")
        
        r = {
            'overall_avg_f1': sum(all_f1) / len(all_f1) if all_f1 else 0,
            'per_question_best_f1': sum(per_question_best_f1) / len(per_question_best_f1) if per_question_best_f1 else 0,
        }
        if supporting_f1:
            r['supporting_f1'] = sum(supporting_f1) / len(supporting_f1)
        if distractor_f1:
            r['distractor_f1'] = sum(distractor_f1) / len(distractor_f1)
        
        results[method_name] = r
        log.info(f"  {method_name} FINAL: overall={r['overall_avg_f1']:.4f} best_per_q={r['per_question_best_f1']:.4f}")
        if 'supporting_f1' in r:
            log.info(f"    supporting={r['supporting_f1']:.4f} distractor={r.get('distractor_f1', 0):.4f}")
    
    # Summary
    log.info(f"\n{'='*60}")
    log.info("RESULTS SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  {'Method':15s} {'Overall':>10s} {'BestPerQ':>10s} {'Support':>10s} {'Distract':>10s}")
    log.info(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name in args.methods:
        r = results.get(name, {})
        log.info(f"  {name:15s} {r.get('overall_avg_f1',0):10.4f} {r.get('per_question_best_f1',0):10.4f} {r.get('supporting_f1',0):10.4f} {r.get('distractor_f1',0):10.4f}")
    
    if 'oracle' in results and 'independent' in results:
        gap = results['oracle']['overall_avg_f1'] - results['independent']['overall_avg_f1']
        log.info(f"\n  Oracle vs Independent (overall): {gap:+.4f}")
        gap2 = results['oracle']['overall_avg_f1'] - results['independent'].get('supporting_f1', 0)
        log.info(f"  Oracle vs Independent (supporting only): {gap2:+.4f}")
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved to {args.output_dir}/results.json")


if __name__ == '__main__':
    main()
