"""
Cross-batch 模块训练脚本 (支持 DDP 多卡并行)
包含训练和三方对比评估，支持多数据集

参数:
  --model           模型名称 (default: Qwen/Qwen2.5-0.5B-Instruct)
  --dataset         数据集 (default: squad, 可选: squad, hotpot, quac, drop)
  --max-samples     训练样本数 (default: None, 使用全部训练集)
  --eval-samples    评估样本数/context数 (default: None, 使用全部验证集)
  --min-questions   每个 context 最少问题数 (default: 3)
  --max-questions   每个 context 最多问题数 (default: 5)
  --epochs          训练轮数 (default: 1)
  --batch-size      每卡 batch size (default: 8)
  --lr              学习率 (default: 1e-4)
  --save-dir        保存 checkpoint 的目录 (default: outputs/checkpoints)
  --force           强制重新训练，即使 checkpoint 已存在

训练用法:
  # 使用全部数据训练 (默认)
  torchrun --nproc_per_node=8 scripts/train_cross_batch.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --batch-size 4

  # 快速测试 (少量数据)
  python scripts/train_cross_batch.py \\
      --max-samples 1000 \\
      --eval-samples 100

  # 指定每个 context 的问题数量
  python scripts/train_cross_batch.py \\
      --min-questions 5 \\
      --max-questions 5 \\
      --eval-samples 100

推理用法 (加载训练好的 checkpoint):
  python scripts/compare_strategies.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --strategies collab_hidden \\
      --collab-hidden-checkpoint outputs/checkpoints/Qwen_Qwen2.5-7B-Instruct_crossbatch.pt \\
      --dataset squad \\
      --max-contexts 100
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import gc
import json
import os
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from src.cross_batch.trainer import (
    LMHeadOnlyTrainer,
    CrossBatchTrainer,
    LoRATrainer,
    LoRACrossBatchTrainer,
    SQuADDataset,
    SQuADGroupedDataset,
)
from src.cross_batch.attention import CrossBatchAttention
from src.cross_batch.generator import CrossBatchGenerator
from src.strategies.cross_batch import run_cross_batch_multi_strategy
from src import (
    load_squad_random_questions,
    load_squad_groups,
    load_hotpot_groups,
    load_quac_groups,
    load_drop_groups,
)


def get_checkpoint_path(base_dir: str, dataset: str, model_name: str, mode: str) -> str:
    """
    Generate checkpoint path in format: {base_dir}/{dataset}/{model_name}_{mode}.pt

    Args:
        base_dir: Base checkpoint directory (e.g., outputs/checkpoints)
        dataset: Dataset name (e.g., squad, hotpot)
        model_name: Model name (e.g., Qwen/Qwen2.5-14B-Instruct)
        mode: Checkpoint mode (e.g., baseline, crossbatch, lora_only, lora_lmhead)

    Returns:
        Full checkpoint path
    """
    # Sanitize model name for filesystem
    safe_model_name = model_name.replace('/', '_')
    checkpoint_dir = os.path.join(base_dir, dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f'{safe_model_name}_{mode}.pt')


def should_skip_training(checkpoint_path: str, force: bool, rank: int = 0) -> bool:
    """
    Check if training should be skipped for this checkpoint.

    Returns True if checkpoint exists and force=False.
    """
    if force:
        return False
    exists = os.path.exists(checkpoint_path)
    if exists and rank == 0:
        print(f'Checkpoint 已存在: {checkpoint_path}，跳过训练 (使用 --force 强制重新训练)')
    return exists


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-batch 模块训练脚本')
    # 模型参数
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='模型名称 (default: Qwen/Qwen2.5-0.5B-Instruct)')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='squad',
                        choices=['squad', 'hotpot', 'quac', 'drop'],
                        help='数据集 (default: squad)')
    parser.add_argument('--split', type=str, default='validation',
                        help='评估用的数据集 split (default: validation)')
    parser.add_argument('--min-questions', type=int, default=3,
                        help='每个 context 最少问题数 (default: 3)')
    parser.add_argument('--max-questions', type=int, default=5,
                        help='每个 context 最多问题数 (default: 5)')
    parser.add_argument('--squad-random-questions', action='store_true',
                        help='SQuAD: 随机采样问题而非按 context 分组')

    # 训练参数
    parser.add_argument('--max-samples', type=int, default=None,
                        help='训练样本数 (default: None, 使用全部数据)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='训练轮数 (default: 1)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='每卡 batch size (default: 8)')
    parser.add_argument('--eval-samples', type=int, default=None,
                        help='评估 context 数 (default: None, 使用全部验证集)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (default: 1e-4)')
    parser.add_argument('--save-dir', type=str, default='outputs/checkpoints',
                        help='保存 checkpoint 的基础目录 (default: outputs/checkpoints)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (default: 42)')
    parser.add_argument('--force', action='store_true',
                        help='强制重新训练，即使 checkpoint 已存在')
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


def load_eval_data(args):
    """根据数据集参数加载评估数据，返回统一格式的 groups"""
    dataset = args.dataset
    split = args.split
    max_contexts = args.eval_samples
    min_questions = args.min_questions
    max_questions = args.max_questions
    seed = args.seed

    if dataset == "squad":
        if args.squad_random_questions:
            # 随机采样问题，每个问题有自己的 context
            groups = load_squad_random_questions(
                split=split,
                max_contexts=max_contexts,
                seed=seed,
            )
        else:
            # 按 context 分组
            groups = load_squad_groups(
                split=split,
                min_questions=min_questions,
                max_questions=max_questions,
                max_contexts=max_contexts,
                seed=seed,
            )
    elif dataset == "hotpot":
        groups = load_hotpot_groups(
            split=split,
            subset="fullwiki",  # 默认使用 fullwiki
            max_contexts=max_contexts,
            min_questions=min_questions,
            max_questions=max_questions,
            seed=seed,
        )
    elif dataset == "quac":
        groups = load_quac_groups(
            split=split,
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    elif dataset == "drop":
        groups = load_drop_groups(
            split=split,
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return groups


def load_train_data(args):
    """加载训练数据，返回统一格式的 groups（用于分组训练）"""
    dataset = args.dataset
    max_contexts = args.max_samples  # 训练时用 max_samples 控制数量
    min_questions = args.min_questions
    max_questions = args.max_questions
    seed = args.seed

    if dataset == "squad":
        groups = load_squad_groups(
            split="train",
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    elif dataset == "hotpot":
        groups = load_hotpot_groups(
            split="train",
            subset="fullwiki",
            max_contexts=max_contexts,
            min_questions=min_questions,
            max_questions=max_questions,
            seed=seed,
        )
    elif dataset == "quac":
        groups = load_quac_groups(
            split="train",
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    elif dataset == "drop":
        groups = load_drop_groups(
            split="train",
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return groups


def evaluate_with_strategy(model, tokenizer, cross_batch_module, device, args,
                           enable_cross_batch=True, strategy_name="eval",
                           rank=0, world_size=1):
    """使用和 compare_strategies.py 相同的评估逻辑，支持分布式评估，返回详细推理结果

    每个 context 作为一个独立的 batch 处理，确保同一 context 的问题能够互相共享信息
    """
    # 加载评估数据
    groups = load_eval_data(args)

    # 分布式评估：每个 rank 处理一部分 context
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

    # 按 context 评估，每个 context 的问题作为一个 batch
    local_strict_acc_sum = 0.0
    local_f1_sum = 0.0
    local_count = 0
    all_details = []  # 保存详细推理结果

    for group_idx, group in enumerate(local_groups):
        # 将该 context 的所有问题转换为 items
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
            max_new_tokens=32,
            strategy_name=strategy_name,
            dataset=args.dataset,
            cross_batch_generator=generator,
            enable_cross_batch=enable_cross_batch,
        )
        local_strict_acc_sum += result.metrics.get("strict_acc", 0.0) * len(batch_items)
        local_f1_sum += result.metrics.get("f1", 0.0) * len(batch_items)
        local_count += len(batch_items)
        # 保存详细结果
        if result.details and "questions" in result.details:
            all_details.extend(result.details["questions"])

    # 汇总所有 rank 的结果
    if world_size > 1:
        local_tensor = torch.tensor([local_strict_acc_sum, local_f1_sum, local_count],
                                     dtype=torch.float64, device=device)
        dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
        total_strict_acc = local_tensor[0].item()
        total_f1 = local_tensor[1].item()
        total_count = local_tensor[2].item()
    else:
        total_strict_acc = local_strict_acc_sum
        total_f1 = local_f1_sum
        total_count = local_count

    return {
        "exact_match": total_strict_acc / total_count * 100 if total_count > 0 else 0.0,
        "f1": total_f1 / total_count * 100 if total_count > 0 else 0.0,
        "details": all_details,  # 包含每个问题的详细推理结果
    }


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    device = f'cuda:{local_rank}'

    gc.collect()
    torch.cuda.empty_cache()

    print_rank0('=' * 60, rank)
    print_rank0(f'训练配置: {args.model}', rank)
    print_rank0(f'数据集: {args.dataset}, 问题数: {args.min_questions}-{args.max_questions}', rank)
    print_rank0(f'样本数: {args.max_samples}, Epochs: {args.epochs}, Batch: {args.batch_size}', rank)
    print_rank0(f'World size: {world_size}, 总 batch size: {args.batch_size * world_size}', rank)
    print_rank0('=' * 60, rank)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集 (按 context 分组)
    train_groups = load_train_data(args)
    train_dataset = SQuADGroupedDataset(
        tokenizer=tokenizer,
        groups=train_groups,
        dataset_name=args.dataset,
    )
    total_questions = sum(len(g["questions"]) for g in train_groups)
    print_rank0(f'训练数据集: {len(train_dataset)} 个 context, {total_questions} 个问题', rank)

    if is_main_process(rank):
        os.makedirs('outputs/inference_results', exist_ok=True)
    all_results = {}

    # 1. 评估原始模型 (所有卡并行评估)
    print_rank0('\n[1/6] 评估原始模型 (未微调)', rank)
    print_rank0('-' * 40, rank)
    model_original = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    cross_batch_original = CrossBatchAttention(hidden_size=model_original.config.hidden_size)

    metrics_original = evaluate_with_strategy(
        model_original, tokenizer, cross_batch_original, device, args,
        enable_cross_batch=False, strategy_name="original",
        rank=rank, world_size=world_size
    )
    all_results['original'] = metrics_original
    print_rank0(f'原始模型 - EM: {metrics_original["exact_match"]:.2f}, F1: {metrics_original["f1"]:.2f}', rank)

    del model_original, cross_batch_original
    gc.collect()
    torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()

    # 2. Baseline 训练 + 测试
    print_rank0('\n[2/6] Baseline：训练后立刻评估 (只训练 lm_head)', rank)
    print_rank0('-' * 40, rank)

    baseline_checkpoint_path = get_checkpoint_path(args.save_dir, args.dataset, args.model, 'baseline')
    if should_skip_training(baseline_checkpoint_path, args.force, rank):
        # 加载已有 checkpoint 进行评估
        model_baseline = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
        checkpoint = torch.load(baseline_checkpoint_path, map_location=device)
        model_baseline.lm_head.load_state_dict(checkpoint['lm_head'])
        history_baseline = {'train_loss': [0.0]}  # placeholder
        print_rank0('已加载 Baseline checkpoint', rank)
    else:
        model_baseline = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
        trainer_baseline = LMHeadOnlyTrainer(
            model=model_baseline,
            tokenizer=tokenizer,
            device=device,
            learning_rate=args.lr,
        )

        # DDP 训练 (按 context 分组，每个 context 一个 batch)
        history_baseline = trainer_baseline.train(
            train_dataset=train_dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=None,
            distributed=(world_size > 1),
            rank=rank,
            world_size=world_size,
            grouped=True,
        )
        print_rank0(f'Baseline 最终 Loss: {history_baseline["train_loss"][-1]:.4f}', rank)
        del trainer_baseline

        # 保存 baseline lm_head checkpoint
        if is_main_process(rank):
            baseline_checkpoint = {
                'lm_head': model_baseline.lm_head.state_dict(),
                'config': {
                    'model': args.model,
                    'dataset': args.dataset,
                    'hidden_size': model_baseline.config.hidden_size,
                    'train_samples': args.max_samples,
                    'epochs': args.epochs,
                },
            }
            torch.save(baseline_checkpoint, baseline_checkpoint_path)
            print(f'\nBaseline checkpoint 已保存到: {baseline_checkpoint_path}')

    # 所有卡并行评估
    cross_batch_baseline = CrossBatchAttention(hidden_size=model_baseline.config.hidden_size)
    metrics_baseline = evaluate_with_strategy(
        model_baseline, tokenizer, cross_batch_baseline, device, args,
        enable_cross_batch=False, strategy_name="baseline",
        rank=rank, world_size=world_size
    )
    all_results['baseline'] = metrics_baseline
    print_rank0(f'Baseline - EM: {metrics_baseline["exact_match"]:.2f}, F1: {metrics_baseline["f1"]:.2f}', rank)
    del cross_batch_baseline

    del model_baseline
    gc.collect()
    torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()

    # 3. Cross-Batch 训练 + 测试
    print_rank0('\n[3/6] Cross-Batch：训练后立刻评估 (lm_head + cross-batch)', rank)
    print_rank0('-' * 40, rank)

    crossbatch_checkpoint_path = get_checkpoint_path(args.save_dir, args.dataset, args.model, 'crossbatch')
    if should_skip_training(crossbatch_checkpoint_path, args.force, rank):
        # 加载已有 checkpoint 进行评估
        model_crossbatch = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
        cross_batch_module = CrossBatchAttention(hidden_size=model_crossbatch.config.hidden_size)
        checkpoint = torch.load(crossbatch_checkpoint_path, map_location=device)
        cross_batch_module.load_state_dict(checkpoint['cross_batch_module'])
        if 'lm_head' in checkpoint:
            model_crossbatch.lm_head.load_state_dict(checkpoint['lm_head'])
        history_crossbatch = {'train_loss': [0.0]}  # placeholder
        print_rank0('已加载 Cross-Batch checkpoint', rank)
    else:
        model_crossbatch = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
        cross_batch_module = CrossBatchAttention(hidden_size=model_crossbatch.config.hidden_size)
        trainer_crossbatch = CrossBatchTrainer(
            model=model_crossbatch,
            tokenizer=tokenizer,
            cross_batch_module=cross_batch_module,
            device=device,
            learning_rate=args.lr,
            train_lm_head=True,
        )

        history_crossbatch = trainer_crossbatch.train(
            train_dataset=train_dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=None,
            distributed=(world_size > 1),
            rank=rank,
            world_size=world_size,
            grouped=True,
        )
        print_rank0(f'Cross-Batch 最终 Loss: {history_crossbatch["train_loss"][-1]:.4f}', rank)
        del trainer_crossbatch

        # 只在 rank 0 上保存 checkpoint
        if is_main_process(rank):
            checkpoint = {
                'cross_batch_module': cross_batch_module.state_dict(),
                'lm_head': model_crossbatch.lm_head.state_dict(),
                'config': {
                    'model': args.model,
                    'dataset': args.dataset,
                    'hidden_size': model_crossbatch.config.hidden_size,
                    'train_samples': args.max_samples,
                    'epochs': args.epochs,
                },
            }
            torch.save(checkpoint, crossbatch_checkpoint_path)
            print(f'\nCross-Batch checkpoint 已保存到: {crossbatch_checkpoint_path}')

    # 所有卡并行评估
    metrics_crossbatch = evaluate_with_strategy(
        model_crossbatch, tokenizer, cross_batch_module, device, args,
        enable_cross_batch=True, strategy_name="crossbatch",
        rank=rank, world_size=world_size
    )
    all_results['crossbatch'] = metrics_crossbatch
    print_rank0(f'Cross-Batch - EM: {metrics_crossbatch["exact_match"]:.2f}, F1: {metrics_crossbatch["f1"]:.2f}', rank)

    del model_crossbatch, cross_batch_module
    gc.collect()
    torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()

    # 4. LoRA + lm_head 训练 + 测试
    print_rank0('\n[4/5] LoRA + lm_head：训练后立刻评估 (LoRA + lm_head)', rank)
    print_rank0('-' * 40, rank)

    lora_lmhead_checkpoint_path = get_checkpoint_path(args.save_dir, args.dataset, args.model, 'lora_lmhead')
    if should_skip_training(lora_lmhead_checkpoint_path, args.force, rank):
        # 加载已有 checkpoint 进行评估
        model_lora_lmhead = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
        checkpoint = torch.load(lora_lmhead_checkpoint_path, map_location=device)
        if 'lm_head' in checkpoint:
            model_lora_lmhead.lm_head.load_state_dict(checkpoint['lm_head'])
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model_lora_lmhead = get_peft_model(model_lora_lmhead, lora_config)
        if 'lora' in checkpoint:
            current_state = model_lora_lmhead.state_dict()
            current_state.update(checkpoint['lora'])
            model_lora_lmhead.load_state_dict(current_state)
        history_lora_lmhead = {'train_loss': [0.0]}  # placeholder
        print_rank0('已加载 LoRA + lm_head checkpoint', rank)
        eval_model_lmhead = model_lora_lmhead.base_model
    else:
        model_lora_lmhead = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
        trainer_lora_lmhead = LoRATrainer(
            model=model_lora_lmhead,
            tokenizer=tokenizer,
            device=device,
            learning_rate=args.lr,
            train_lm_head=True,  # 训练 LoRA + lm_head
        )

        history_lora_lmhead = trainer_lora_lmhead.train(
            train_dataset=train_dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=None,
            distributed=(world_size > 1),
            rank=rank,
            world_size=world_size,
            grouped=True,
        )
        print_rank0(f'LoRA + lm_head 最终 Loss: {history_lora_lmhead["train_loss"][-1]:.4f}', rank)

        # 保存 LoRA + lm_head checkpoint
        if is_main_process(rank):
            lora_state = {k: v for k, v in trainer_lora_lmhead.model.state_dict().items() if 'lora' in k.lower()}
            torch.save({
                'lora': lora_state,
                'lm_head': trainer_lora_lmhead.lm_head.state_dict(),
                'config': {
                    'model': args.model,
                    'dataset': args.dataset,
                    'hidden_size': model_lora_lmhead.config.hidden_size,
                    'train_samples': args.max_samples,
                    'epochs': args.epochs,
                },
            }, lora_lmhead_checkpoint_path)
            print(f'\nLoRA + lm_head checkpoint 已保存到: {lora_lmhead_checkpoint_path}')

        eval_model_lmhead = trainer_lora_lmhead.model.base_model
        del trainer_lora_lmhead

    # 评估
    cross_batch_lora_lmhead = CrossBatchAttention(hidden_size=model_lora_lmhead.config.hidden_size)
    metrics_lora_lmhead = evaluate_with_strategy(
        eval_model_lmhead, tokenizer, cross_batch_lora_lmhead, device, args,
        enable_cross_batch=False, strategy_name="lora_lmhead",
        rank=rank, world_size=world_size
    )
    all_results['lora_lmhead'] = metrics_lora_lmhead
    print_rank0(f'LoRA + lm_head - EM: {metrics_lora_lmhead["exact_match"]:.2f}, F1: {metrics_lora_lmhead["f1"]:.2f}', rank)

    del model_lora_lmhead, cross_batch_lora_lmhead
    gc.collect()
    torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()

    # 5. LoRA + lm_head + cross-batch 训练 + 测试
    print_rank0('\n[5/5] LoRA + lm_head + cross-batch：训练后立刻评估', rank)
    print_rank0('-' * 40, rank)

    lora_crossbatch_checkpoint_path = get_checkpoint_path(args.save_dir, args.dataset, args.model, 'lora_crossbatch')
    if should_skip_training(lora_crossbatch_checkpoint_path, args.force, rank):
        # 加载已有 checkpoint 进行评估
        model_lora_crossbatch = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
        cross_batch_lora_cb = CrossBatchAttention(hidden_size=model_lora_crossbatch.config.hidden_size)
        checkpoint = torch.load(lora_crossbatch_checkpoint_path, map_location=device)
        if 'lm_head' in checkpoint:
            model_lora_crossbatch.lm_head.load_state_dict(checkpoint['lm_head'])
        if 'cross_batch_module' in checkpoint:
            cross_batch_lora_cb.load_state_dict(checkpoint['cross_batch_module'])
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model_lora_crossbatch = get_peft_model(model_lora_crossbatch, lora_config)
        if 'lora' in checkpoint:
            current_state = model_lora_crossbatch.state_dict()
            current_state.update(checkpoint['lora'])
            model_lora_crossbatch.load_state_dict(current_state)
        history_lora_crossbatch = {'train_loss': [0.0]}  # placeholder
        print_rank0('已加载 LoRA + lm_head + cross-batch checkpoint', rank)
        eval_model_lora_cb = model_lora_crossbatch.base_model
    else:
        model_lora_crossbatch = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
        cross_batch_lora_cb = CrossBatchAttention(hidden_size=model_lora_crossbatch.config.hidden_size)
        trainer_lora_crossbatch = LoRACrossBatchTrainer(
            model=model_lora_crossbatch,
            tokenizer=tokenizer,
            cross_batch_module=cross_batch_lora_cb,
            device=device,
            learning_rate=args.lr,
        )

        history_lora_crossbatch = trainer_lora_crossbatch.train(
            train_dataset=train_dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            distributed=(world_size > 1),
            rank=rank,
            world_size=world_size,
            grouped=True,
        )
        print_rank0(f'LoRA + lm_head + cross-batch 最终 Loss: {history_lora_crossbatch["train_loss"][-1]:.4f}', rank)

        # 保存 checkpoint
        if is_main_process(rank):
            lora_state = {k: v for k, v in trainer_lora_crossbatch.model.state_dict().items() if 'lora' in k.lower()}
            torch.save({
                'lora': lora_state,
                'lm_head': trainer_lora_crossbatch.lm_head.state_dict(),
                'cross_batch_module': cross_batch_lora_cb.state_dict(),
                'config': {
                    'model': args.model,
                    'dataset': args.dataset,
                    'hidden_size': model_lora_crossbatch.config.hidden_size,
                    'train_samples': args.max_samples,
                    'epochs': args.epochs,
                },
            }, lora_crossbatch_checkpoint_path)
            print(f'\nLoRA + lm_head + cross-batch checkpoint 已保存到: {lora_crossbatch_checkpoint_path}')

        eval_model_lora_cb = trainer_lora_crossbatch.model.base_model
        del trainer_lora_crossbatch

    # 评估 (启用 cross-batch)
    metrics_lora_crossbatch = evaluate_with_strategy(
        eval_model_lora_cb, tokenizer, cross_batch_lora_cb, device, args,
        enable_cross_batch=True, strategy_name="lora_crossbatch",
        rank=rank, world_size=world_size
    )
    all_results['lora_crossbatch'] = metrics_lora_crossbatch
    print_rank0(f'LoRA + lm_head + cross-batch - EM: {metrics_lora_crossbatch["exact_match"]:.2f}, F1: {metrics_lora_crossbatch["f1"]:.2f}', rank)

    del model_lora_crossbatch, cross_batch_lora_cb
    gc.collect()
    torch.cuda.empty_cache()

    # 保存结果 (只在 rank 0)
    if is_main_process(rank):
        # 提取详细结果用于单独保存
        inference_details = {
            'original': all_results['original'].get('details', []),
            'baseline': all_results['baseline'].get('details', []),
            'crossbatch': all_results['crossbatch'].get('details', []),
            'lora_lmhead': all_results['lora_lmhead'].get('details', []),
            'lora_crossbatch': all_results['lora_crossbatch'].get('details', []),
        }

        # metrics 中不包含 details（太大）
        metrics_only = {
            k: {key: val for key, val in v.items() if key != 'details'}
            for k, v in all_results.items()
        }

        summary = {
            'config': {
                'model': args.model,
                'dataset': args.dataset,
                'min_questions': args.min_questions,
                'max_questions': args.max_questions,
                'train_samples': args.max_samples,
                'eval_samples': args.eval_samples,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'world_size': world_size,
            },
            'training_history': {
                'baseline': history_baseline['train_loss'],
                'crossbatch': history_crossbatch['train_loss'],
                'lora_lmhead': history_lora_lmhead['train_loss'],
                'lora_crossbatch': history_lora_crossbatch['train_loss'],
            },
            'metrics': metrics_only,
        }

        os.makedirs('outputs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'outputs/training_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # 保存详细推理结果到单独文件
        details_file = f'outputs/inference_details_{timestamp}.json'
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(inference_details, f, indent=2, ensure_ascii=False)
        print(f'\n详细推理结果已保存到: {details_file}')

        # 打印总结 - 表1: 训练 Loss
        print('\n' + '=' * 60)
        print('表1: 训练 Loss')
        print('=' * 60)
        print(f'| 方法 | Final Loss |')
        print(f'|------|------------|')
        print(f'| Baseline (lm_head) | {history_baseline["train_loss"][-1]:.4f} |')
        print(f'| lm_head + Cross-Batch | {history_crossbatch["train_loss"][-1]:.4f} |')
        print(f'| LoRA + lm_head | {history_lora_lmhead["train_loss"][-1]:.4f} |')
        print(f'| LoRA + lm_head + Cross-Batch | {history_lora_crossbatch["train_loss"][-1]:.4f} |')

        # 打印总结 - 表2: 评估指标
        print('\n' + '=' * 60)
        print('表2: 评估指标 (EM / F1)')
        print('=' * 60)
        print(f'| 方法 | EM | F1 |')
        print(f'|------|-----|-----|')
        print(f'| 原始模型 | {all_results["original"]["exact_match"]:.2f} | {all_results["original"]["f1"]:.2f} |')
        print(f'| Baseline (lm_head) | {all_results["baseline"]["exact_match"]:.2f} | {all_results["baseline"]["f1"]:.2f} |')
        print(f'| lm_head + Cross-Batch | {all_results["crossbatch"]["exact_match"]:.2f} | {all_results["crossbatch"]["f1"]:.2f} |')
        print(f'| LoRA + lm_head | {all_results["lora_lmhead"]["exact_match"]:.2f} | {all_results["lora_lmhead"]["f1"]:.2f} |')
        print(f'| LoRA + lm_head + Cross-Batch | {all_results["lora_crossbatch"]["exact_match"]:.2f} | {all_results["lora_crossbatch"]["f1"]:.2f} |')

        print('\n改进分析:')
        orig_f1 = all_results['original']['f1']
        base_f1 = all_results['baseline']['f1']
        cross_f1 = all_results['crossbatch']['f1']
        lora_lmhead_f1 = all_results['lora_lmhead']['f1']
        lora_crossbatch_f1 = all_results['lora_crossbatch']['f1']

        print(f'  Baseline vs 原始: F1 {base_f1 - orig_f1:+.2f}')
        print(f'  lm_head + Cross-Batch vs 原始: F1 {cross_f1 - orig_f1:+.2f}')
        print(f'  lm_head + Cross-Batch vs Baseline: F1 {cross_f1 - base_f1:+.2f}')
        print(f'  LoRA + lm_head vs 原始: F1 {lora_lmhead_f1 - orig_f1:+.2f}')
        print(f'  LoRA + lm_head + Cross-Batch vs 原始: F1 {lora_crossbatch_f1 - orig_f1:+.2f}')
        print(f'  LoRA + lm_head + Cross-Batch vs LoRA + lm_head: F1 {lora_crossbatch_f1 - lora_lmhead_f1:+.2f}')

        # 找出最佳方法
        methods = {
            'Baseline': base_f1,
            'lm_head + Cross-Batch': cross_f1,
            'LoRA + lm_head': lora_lmhead_f1,
            'LoRA + lm_head + Cross-Batch': lora_crossbatch_f1,
        }
        best_method = max(methods, key=methods.get)
        print(f'\n=> 最佳方法: {best_method} (F1: {methods[best_method]:.2f})')

        print(f'\n结果已保存到: {output_file}')
        print('=' * 60)

    cleanup_ddp()


if __name__ == '__main__':
    main()
