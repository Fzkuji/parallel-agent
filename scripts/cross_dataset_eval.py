#!/usr/bin/env python3
"""
跨数据集泛化评估脚本

实验设计：
1. 在 SQuAD 上训练 Cross-Batch Attention 模块
2. 在 SQuAD 和 CMB-Exam 上评估模型的泛化能力

支持的策略：
- original: 原始模型（无微调）
- baseline: 只训练 lm_head
- crossbatch: lm_head + Cross-Batch Attention
- lora_lmhead: LoRA + lm_head
- lora_crossbatch: LoRA + lm_head + Cross-Batch Attention

用法：
    # 评估在 SQuAD 上训练的模型（自动在 SQuAD 和 CMB-Exam 上评估）
    python scripts/cross_dataset_eval.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --train-dataset squad \\
        --eval-datasets squad,cmb_exam_context \\
        --eval-samples 100

    # 多卡并行评估
    torchrun --nproc_per_node=8 scripts/cross_dataset_eval.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --train-dataset squad \\
        --eval-datasets squad,cmb_exam_context
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from src.cross_batch.attention import CrossBatchAttention
from src.cross_batch.generator import CrossBatchGenerator
from src.strategies.cross_batch import run_cross_batch_multi_strategy
from src import (
    load_squad_groups,
    load_cmb_exam_context_groups,
    load_cmb_exam_subdomain_groups,
    load_cmb_exam_random_groups,
)


def parse_args():
    parser = argparse.ArgumentParser(description='跨数据集泛化评估脚本')

    # 模型参数
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='模型名称')

    # 数据集参数
    parser.add_argument('--train-dataset', type=str, default='squad',
                        help='训练数据集（用于查找 checkpoint）')
    parser.add_argument('--eval-datasets', type=str, default='squad,cmb_exam_context',
                        help='评估数据集，逗号分隔 (squad, cmb_exam_context, cmb_exam_subdomain, cmb_exam_random)')
    parser.add_argument('--eval-samples', type=int, default=None,
                        help='每个数据集的评估样本数/context数 (default: None, 使用全部)')
    parser.add_argument('--min-questions', type=int, default=3,
                        help='每个 context 最少问题数')
    parser.add_argument('--max-questions', type=int, default=5,
                        help='每个 context 最多问题数')
    parser.add_argument('--split', type=str, default='validation',
                        help='评估用的数据集 split')

    # Checkpoint 参数
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/checkpoints',
                        help='Checkpoint 基础目录')
    parser.add_argument('--strategies', type=str, default='original,baseline,crossbatch,lora_lmhead,lora_crossbatch',
                        help='要评估的策略，逗号分隔')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--max-new-tokens', type=int, default=32,
                        help='生成的最大 token 数')
    parser.add_argument('--self-only', action='store_true',
                        help='Ablation: CrossBatch 只关注自己（禁用跨样本交互）')

    return parser.parse_args()


def setup_ddp():
    """初始化 DDP 环境"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_ddp():
    """清理 DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def print_rank0(msg, rank):
    if is_main_process(rank):
        print(msg)


def get_checkpoint_path(base_dir: str, dataset: str, model_name: str, mode: str) -> str:
    """生成 checkpoint 路径"""
    safe_model_name = model_name.replace('/', '_')
    return os.path.join(base_dir, dataset, f'{safe_model_name}_{mode}.pt')


def load_eval_data(dataset: str, args) -> List[dict]:
    """加载评估数据集"""
    split = args.split
    max_contexts = args.eval_samples
    min_questions = args.min_questions
    max_questions = args.max_questions
    seed = args.seed

    if dataset == "squad":
        groups = load_squad_groups(
            split=split,
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    elif dataset == "cmb_exam_context":
        groups = load_cmb_exam_context_groups(
            split=split if split != "validation" else "val",
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    elif dataset == "cmb_exam_subdomain":
        raw_groups = load_cmb_exam_subdomain_groups(
            split=split if split != "validation" else "val",
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
        # 转换为 single-context 格式
        groups = []
        for g in raw_groups:
            questions = []
            for item in g["items"]:
                questions.append({
                    "qid": item["qid"],
                    "text": item["question"],
                    "answer_tokens": item.get("answer_tokens", 4),
                    "references": item.get("references", []),
                })
            context = g["items"][0]["context"] if g["items"] else ""
            groups.append({
                "context": context,
                "title": g["title"],
                "questions": questions,
            })
    elif dataset == "cmb_exam_random":
        raw_groups = load_cmb_exam_random_groups(
            split=split if split != "validation" else "val",
            questions_per_group=max_questions or 5,
            max_contexts=max_contexts,
            seed=seed,
        )
        # 转换为 single-context 格式
        groups = []
        for g in raw_groups:
            questions = []
            for item in g["items"]:
                questions.append({
                    "qid": item["qid"],
                    "text": item["question"],
                    "answer_tokens": item.get("answer_tokens", 4),
                    "references": item.get("references", []),
                })
            context = g["items"][0]["context"] if g["items"] else ""
            groups.append({
                "context": context,
                "title": g["title"],
                "questions": questions,
            })
    else:
        raise ValueError(f"不支持的数据集: {dataset}")

    return groups


def evaluate_on_dataset(
    dataset: str,
    model,
    tokenizer,
    cross_batch_module,
    device,
    args,
    enable_cross_batch: bool,
    strategy_name: str,
    rank: int,
    world_size: int,
) -> dict:
    """在指定数据集上评估"""
    # 加载数据
    groups = load_eval_data(dataset, args)

    # 分布式评估
    groups_per_rank = len(groups) // world_size if world_size > 1 else len(groups)
    start_idx = rank * groups_per_rank
    end_idx = start_idx + groups_per_rank if rank < world_size - 1 else len(groups)
    local_groups = groups[start_idx:end_idx]

    # 创建 generator
    generator = CrossBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        device=device,
    )

    # 检测数据集类型
    is_cmb_exam = dataset.startswith("cmb_exam")

    # 评估
    local_acc_sum = 0.0
    local_f1_sum = 0.0
    local_count = 0

    for group_idx, group in enumerate(local_groups):
        context = group["context"]
        batch_items = []
        for q_idx, q in enumerate(group["questions"]):
            unique_qid = f"G{start_idx + group_idx}_Q{q_idx}"
            batch_items.append({
                "qid": unique_qid,
                "question": q["text"],
                "context": context,
                "references": q["references"],
            })

        result = run_cross_batch_multi_strategy(
            items=batch_items,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.max_new_tokens,
            strategy_name=strategy_name,
            dataset=dataset,
            cross_batch_generator=generator,
            enable_cross_batch=enable_cross_batch,
        )

        if is_cmb_exam:
            local_acc_sum += result.metrics.get("acc", 0.0) * len(batch_items)
        else:
            local_acc_sum += result.metrics.get("strict_acc", 0.0) * len(batch_items)
            local_f1_sum += result.metrics.get("f1", 0.0) * len(batch_items)
        local_count += len(batch_items)

    # 汇总结果
    if world_size > 1:
        local_tensor = torch.tensor([local_acc_sum, local_f1_sum, local_count],
                                    dtype=torch.float64, device=device)
        dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
        total_acc = local_tensor[0].item()
        total_f1 = local_tensor[1].item()
        total_count = local_tensor[2].item()
    else:
        total_acc = local_acc_sum
        total_f1 = local_f1_sum
        total_count = local_count

    if is_cmb_exam:
        return {
            "acc": total_acc / total_count * 100 if total_count > 0 else 0.0,
            "n_samples": int(total_count),
        }
    else:
        return {
            "exact_match": total_acc / total_count * 100 if total_count > 0 else 0.0,
            "f1": total_f1 / total_count * 100 if total_count > 0 else 0.0,
            "n_samples": int(total_count),
        }


def format_metrics(metrics: dict, dataset: str) -> str:
    """格式化指标"""
    if dataset.startswith("cmb_exam"):
        return f'Acc: {metrics.get("acc", 0.0):.2f}%'
    else:
        return f'EM: {metrics.get("exact_match", 0.0):.2f}%, F1: {metrics.get("f1", 0.0):.2f}%'


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    device = f'cuda:{local_rank}'

    gc.collect()
    torch.cuda.empty_cache()

    eval_datasets = [d.strip() for d in args.eval_datasets.split(',')]
    strategies = [s.strip() for s in args.strategies.split(',')]

    print_rank0('=' * 70, rank)
    print_rank0(f'跨数据集泛化评估', rank)
    print_rank0(f'模型: {args.model}', rank)
    print_rank0(f'训练数据集: {args.train_dataset}', rank)
    print_rank0(f'评估数据集: {eval_datasets}', rank)
    print_rank0(f'评估策略: {strategies}', rank)
    print_rank0('=' * 70, rank)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 存储所有结果
    all_results = {}

    for strategy in strategies:
        print_rank0(f'\n{"="*50}', rank)
        print_rank0(f'评估策略: {strategy}', rank)
        print_rank0(f'{"="*50}', rank)

        # 加载模型和 checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16
        ).to(device)

        cross_batch_module = CrossBatchAttention(
            hidden_size=model.config.hidden_size,
            self_only=args.self_only
        )
        enable_cross_batch = False

        if strategy == 'original':
            # 原始模型，不加载任何 checkpoint
            pass

        elif strategy == 'baseline':
            checkpoint_path = get_checkpoint_path(
                args.checkpoint_dir, args.train_dataset, args.model, 'baseline'
            )
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.lm_head.load_state_dict(checkpoint['lm_head'])
                print_rank0(f'已加载 baseline checkpoint: {checkpoint_path}', rank)
            else:
                print_rank0(f'警告: 未找到 baseline checkpoint: {checkpoint_path}', rank)

        elif strategy == 'crossbatch':
            checkpoint_path = get_checkpoint_path(
                args.checkpoint_dir, args.train_dataset, args.model, 'crossbatch'
            )
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                cross_batch_module.load_state_dict(checkpoint['cross_batch_module'])
                if 'lm_head' in checkpoint:
                    model.lm_head.load_state_dict(checkpoint['lm_head'])
                enable_cross_batch = True
                print_rank0(f'已加载 crossbatch checkpoint: {checkpoint_path}', rank)
            else:
                print_rank0(f'警告: 未找到 crossbatch checkpoint: {checkpoint_path}', rank)

        elif strategy == 'lora_lmhead':
            checkpoint_path = get_checkpoint_path(
                args.checkpoint_dir, args.train_dataset, args.model, 'lora_lmhead'
            )
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if 'lm_head' in checkpoint:
                    model.lm_head.load_state_dict(checkpoint['lm_head'])
                # 应用 LoRA
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                )
                model = get_peft_model(model, lora_config)
                if 'lora' in checkpoint:
                    current_state = model.state_dict()
                    current_state.update(checkpoint['lora'])
                    model.load_state_dict(current_state)
                model = model.base_model  # 评估时使用 base model
                print_rank0(f'已加载 lora_lmhead checkpoint: {checkpoint_path}', rank)
            else:
                print_rank0(f'警告: 未找到 lora_lmhead checkpoint: {checkpoint_path}', rank)

        elif strategy == 'lora_crossbatch':
            checkpoint_path = get_checkpoint_path(
                args.checkpoint_dir, args.train_dataset, args.model, 'lora_crossbatch'
            )
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if 'lm_head' in checkpoint:
                    model.lm_head.load_state_dict(checkpoint['lm_head'])
                if 'cross_batch_module' in checkpoint:
                    cross_batch_module.load_state_dict(checkpoint['cross_batch_module'])
                # 应用 LoRA
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                )
                model = get_peft_model(model, lora_config)
                if 'lora' in checkpoint:
                    current_state = model.state_dict()
                    current_state.update(checkpoint['lora'])
                    model.load_state_dict(current_state)
                model = model.base_model  # 评估时使用 base model
                enable_cross_batch = True
                print_rank0(f'已加载 lora_crossbatch checkpoint: {checkpoint_path}', rank)
            else:
                print_rank0(f'警告: 未找到 lora_crossbatch checkpoint: {checkpoint_path}', rank)

        cross_batch_module = cross_batch_module.to(device)

        # 在每个评估数据集上评估
        strategy_results = {}
        for eval_dataset in eval_datasets:
            print_rank0(f'\n评估 {eval_dataset}...', rank)

            metrics = evaluate_on_dataset(
                dataset=eval_dataset,
                model=model,
                tokenizer=tokenizer,
                cross_batch_module=cross_batch_module,
                device=device,
                args=args,
                enable_cross_batch=enable_cross_batch,
                strategy_name=strategy,
                rank=rank,
                world_size=world_size,
            )

            strategy_results[eval_dataset] = metrics
            print_rank0(f'  {eval_dataset}: {format_metrics(metrics, eval_dataset)}', rank)

        all_results[strategy] = strategy_results

        # 清理模型
        del model, cross_batch_module
        gc.collect()
        torch.cuda.empty_cache()

        if world_size > 1:
            dist.barrier()

    # 打印汇总表格
    if is_main_process(rank):
        print('\n' + '=' * 70)
        print('跨数据集泛化评估结果汇总')
        print('=' * 70)

        # 构建表格
        header = ['策略'] + eval_datasets
        rows = []

        for strategy in strategies:
            row = [strategy]
            for dataset in eval_datasets:
                if strategy in all_results and dataset in all_results[strategy]:
                    metrics = all_results[strategy][dataset]
                    if dataset.startswith('cmb_exam'):
                        row.append(f'{metrics.get("acc", 0.0):.2f}%')
                    else:
                        row.append(f'{metrics.get("exact_match", 0.0):.2f}%')
                else:
                    row.append('-')
            rows.append(row)

        # 打印表格
        col_widths = [max(len(str(row[i])) for row in [header] + rows) + 2
                      for i in range(len(header))]

        header_line = '|'.join(str(h).center(w) for h, w in zip(header, col_widths))
        print(header_line)
        print('-' * len(header_line))

        for row in rows:
            print('|'.join(str(cell).center(w) for cell, w in zip(row, col_widths)))

        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'outputs/cross_dataset_eval_{timestamp}.json'
        os.makedirs('outputs', exist_ok=True)

        summary = {
            'config': {
                'model': args.model,
                'train_dataset': args.train_dataset,
                'eval_datasets': eval_datasets,
                'strategies': strategies,
                'eval_samples': args.eval_samples,
                'min_questions': args.min_questions,
                'max_questions': args.max_questions,
            },
            'results': all_results,
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f'\n结果已保存到: {output_file}')
        print('=' * 70)

    cleanup_ddp()


if __name__ == '__main__':
    main()
