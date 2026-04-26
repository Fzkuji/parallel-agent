#!/usr/bin/env python3
"""
Pure LoRA baseline (no CSE) for comparison.
Same training setup as train_cse_v3.py but without any cross-sequence module.
"""

import os
import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def prepare_training_data(dataset, tokenizer, num_batches=200, questions_per_batch=4, max_paragraphs_per_q=5):
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
        
        sequences = []
        for qi, q_item in enumerate(batch_questions):
            paras = list(q_item['supporting_paragraphs'])
            other_paras = [p for p in q_item['all_paragraphs'] if p[0] not in q_item['supporting_titles']]
            random.shuffle(other_paras)
            remaining = max_paragraphs_per_q - len(paras)
            if remaining > 0:
                paras.extend(other_paras[:remaining])
            
            for title, text in paras:
                context = f"{title}: {text}"
                messages = [
                    {"role": "system", "content": "Answer the question concisely based on the given context. Give only the answer, no explanation. If the context doesn't contain enough information, still try your best."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {q_item['question']}"},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                full_text = prompt + q_item['answer']
                
                sequences.append({
                    'prompt': prompt,
                    'full_text': full_text,
                    'answer': q_item['answer'],
                    'question_id': qi,
                    'is_supporting': title in q_item['supporting_titles'],
                })
        batches.append(sequences)
    return batches


def collate_batch(sequences, tokenizer, max_length=512):
    prompts = [s['prompt'] for s in sequences]
    full_texts = [s['full_text'] for s in sequences]
    
    encoded = tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    prompt_encoded = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    labels = input_ids.clone()
    
    for i in range(input_ids.size(0)):
        prompt_len = prompt_encoded['attention_mask'][i].sum().item()
        labels[i, :prompt_len] = -100
        labels[i, attention_mask[i] == 0] = -100
    
    return input_ids, attention_mask, labels


def compute_f1(prediction, ground_truth):
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens: return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common: return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_answer(text):
    text = text.strip()
    for prefix in ['Answer:', 'answer:', 'The answer is', 'the answer is']:
        if prefix in text:
            return text.split(prefix)[-1].strip().split('\n')[0].strip().rstrip('.')
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[-1] if lines else text


def evaluate(model, tokenizer, eval_batches, device='cuda', max_new_tokens=32):
    model.eval()
    all_f1 = []
    supporting_f1 = []
    
    for batch in eval_batches:
        prompts = [s['prompt'] for s in batch]
        encoded = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048)
        
        with torch.no_grad():
            out = model.generate(
                input_ids=encoded['input_ids'].to(device),
                attention_mask=encoded['attention_mask'].to(device),
                max_new_tokens=max_new_tokens, do_sample=False,
            )
        gen_ids = out[:, encoded['input_ids'].size(1):]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        
        for i, text in enumerate(texts):
            ans = extract_answer(text)
            f1 = compute_f1(ans, batch[i]['answer'])
            all_f1.append(f1)
            if batch[i]['is_supporting']:
                supporting_f1.append(f1)
    
    return {
        'overall': sum(all_f1) / len(all_f1) if all_f1 else 0,
        'supporting': sum(supporting_f1) / len(supporting_f1) if supporting_f1 else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/autodl-fs/data/models/Qwen2.5-7B-Instruct')
    parser.add_argument('--num_train_batches', type=int, default=500)
    parser.add_argument('--num_eval_batches', type=int, default=10)
    parser.add_argument('--questions_per_batch', type=int, default=2)
    parser.add_argument('--max_paragraphs_per_q', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--output_dir', default='/root/autodl-tmp/lora_baseline')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    log.info(f"Loading model {args.model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
    )
    
    model.gradient_checkpointing_enable()
    
    # Add LoRA (same config as CSE training)
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    device = 'cuda'
    
    # Load data (same seed, same split as CSE)
    log.info("Loading HotpotQA...")
    from datasets import load_dataset
    ds_train = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    ds_val = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    
    train_batches = prepare_training_data(
        ds_train, tokenizer, num_batches=args.num_train_batches,
        questions_per_batch=args.questions_per_batch, max_paragraphs_per_q=args.max_paragraphs_per_q,
    )
    eval_batches = prepare_training_data(
        ds_val, tokenizer, num_batches=args.num_eval_batches,
        questions_per_batch=args.questions_per_batch, max_paragraphs_per_q=args.max_paragraphs_per_q,
    )
    log.info(f"Train: {len(train_batches)} batches, Eval: {len(eval_batches)} batches")
    
    # Optimizer (only LoRA params)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    
    best_f1 = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        random.shuffle(train_batches)
        optimizer.zero_grad()
        
        for bi, batch_seqs in enumerate(train_batches):
            input_ids, attention_mask, labels = collate_batch(batch_seqs, tokenizer)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()
            
            if (bi + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_losses.append(outputs.loss.item())
            
            if (bi + 1) % 20 == 0:
                avg_loss = sum(epoch_losses[-20:]) / 20
                log.info(f"  Epoch {epoch+1} [{bi+1}/{len(train_batches)}] loss={avg_loss:.4f}")
        
        if len(train_batches) % args.gradient_accumulation != 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        log.info(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        
        # Eval
        log.info("Running evaluation...")
        eval_results = evaluate(model, tokenizer, eval_batches[:5], device=device)
        log.info(f"  Eval: overall={eval_results['overall']:.4f} supporting={eval_results['supporting']:.4f}")
        
        if eval_results['overall'] > best_f1:
            best_f1 = eval_results['overall']
            model.save_pretrained(os.path.join(args.output_dir, 'best'))
            log.info(f"  New best! F1={best_f1:.4f}")
    
    log.info(f"\nTraining done. Best F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()
