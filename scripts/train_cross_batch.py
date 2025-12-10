"""
Cross-batch 模块训练脚本 (支持 DDP 多卡并行)
包含训练和三方对比评估

参数:
  --model         模型名称 (default: Qwen/Qwen2.5-0.5B-Instruct)
  --max-samples   训练样本数 (default: 2000)
  --epochs        训练轮数 (default: 1)
  --batch-size    每卡 batch size (default: 8)
  --eval-samples  评估样本数 (default: 100)
  --lr            学习率 (default: 1e-4)
  --save-dir      保存 checkpoint 的目录 (default: outputs/checkpoints)

训练用法:
  # 0.5B 模型 (单卡)
  python scripts/train_cross_batch.py

  # 0.5B 模型 (8卡并行)
  torchrun --nproc_per_node=8 scripts/train_cross_batch.py

  # 7B 模型 (8卡并行)
  torchrun --nproc_per_node=8 scripts/train_cross_batch.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --batch-size 4 \\
      --max-samples 50000 \\
      --epochs 1 \\
      --eval-samples 1000

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
from src.cross_batch.trainer import LMHeadOnlyTrainer, CrossBatchTrainer, SQuADDataset
from src.cross_batch.attention import CrossBatchAttention
from src.cross_batch.generator import CrossBatchGenerator
from src.strategies.cross_batch import run_cross_batch_multi_strategy
from src import load_squad_random_questions


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-batch 模块训练脚本')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='模型名称 (default: Qwen/Qwen2.5-0.5B-Instruct)')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='训练样本数 (default: 2000)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='训练轮数 (default: 1)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='每卡 batch size (default: 8)')
    parser.add_argument('--eval-samples', type=int, default=100,
                        help='评估样本数 (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (default: 1e-4)')
    parser.add_argument('--save-dir', type=str, default='outputs/checkpoints',
                        help='保存 checkpoint 的目录 (default: outputs/checkpoints)')
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


def evaluate_with_strategy(model, tokenizer, cross_batch_module, device, eval_samples,
                           enable_cross_batch=True, strategy_name="eval", eval_batch_size=8,
                           rank=0, world_size=1):
    """使用和 compare_strategies.py 相同的评估逻辑，支持分布式评估"""
    # 加载 SQuAD 验证集 (随机采样问题，每个问题有自己的 context)
    groups = load_squad_random_questions(
        split="validation",
        max_contexts=eval_samples,
        seed=42,
    )

    # 转换为 items 格式 (和 run_cross_batch_multi_strategy 兼容)
    all_items = []
    for group_idx, group in enumerate(groups):
        context = group["context"]
        for q_idx, q in enumerate(group["questions"]):
            # 生成唯一 qid，避免所有问题都叫 Q1
            unique_qid = f"G{group_idx}_Q{q_idx}"
            all_items.append({
                "qid": unique_qid,
                "question": q["text"],
                "context": context,
                "references": q["references"],
            })

    # 分布式评估：每个 rank 处理一部分数据
    items_per_rank = len(all_items) // world_size
    start_idx = rank * items_per_rank
    end_idx = start_idx + items_per_rank if rank < world_size - 1 else len(all_items)
    items = all_items[start_idx:end_idx]

    # 创建 generator
    generator = CrossBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        device=device,
    )

    # 分 batch 评估，避免 OOM
    local_strict_acc_sum = 0.0
    local_f1_sum = 0.0
    local_count = 0

    for i in range(0, len(items), eval_batch_size):
        batch_items = items[i:i + eval_batch_size]
        result = run_cross_batch_multi_strategy(
            items=batch_items,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=32,
            strategy_name=strategy_name,
            dataset="squad",
            cross_batch_generator=generator,
            enable_cross_batch=enable_cross_batch,
        )
        local_strict_acc_sum += result.metrics.get("strict_acc", 0.0) * len(batch_items)
        local_f1_sum += result.metrics.get("f1", 0.0) * len(batch_items)
        local_count += len(batch_items)

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
    }


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    device = f'cuda:{local_rank}'

    gc.collect()
    torch.cuda.empty_cache()

    print_rank0('=' * 60, rank)
    print_rank0(f'训练配置: {args.model}', rank)
    print_rank0(f'样本数: {args.max_samples}, Epochs: {args.epochs}, Batch: {args.batch_size}', rank)
    print_rank0(f'World size: {world_size}, 总 batch size: {args.batch_size * world_size}', rank)
    print_rank0('=' * 60, rank)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    train_dataset = SQuADDataset(tokenizer=tokenizer, split='train', max_samples=args.max_samples)
    print_rank0(f'训练数据集大小: {len(train_dataset)}', rank)

    if is_main_process(rank):
        os.makedirs('outputs/inference_results', exist_ok=True)
    all_results = {}

    # 1. 评估原始模型 (所有卡并行评估)
    print_rank0('\n[1/3] 评估原始模型 (未微调)', rank)
    print_rank0('-' * 40, rank)
    model_original = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    cross_batch_original = CrossBatchAttention(hidden_size=model_original.config.hidden_size)

    metrics_original = evaluate_with_strategy(
        model_original, tokenizer, cross_batch_original, device,
        args.eval_samples, enable_cross_batch=False, strategy_name="original",
        eval_batch_size=args.batch_size, rank=rank, world_size=world_size
    )
    all_results['original'] = metrics_original
    print_rank0(f'原始模型 - EM: {metrics_original["exact_match"]:.2f}, F1: {metrics_original["f1"]:.2f}', rank)

    del model_original, cross_batch_original
    gc.collect()
    torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()

    # 2. Baseline 训练 + 测试
    print_rank0('\n[2/3] Baseline：训练后立刻评估 (只训练 lm_head)', rank)
    print_rank0('-' * 40, rank)

    model_baseline = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    trainer_baseline = LMHeadOnlyTrainer(
        model=model_baseline,
        tokenizer=tokenizer,
        device=device,
        learning_rate=args.lr,
    )

    # DDP 训练
    history_baseline = trainer_baseline.train(
        train_dataset=train_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=None,
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size,
    )
    print_rank0(f'Baseline 最终 Loss: {history_baseline["train_loss"][-1]:.4f}', rank)

    # 所有卡并行评估
    cross_batch_baseline = CrossBatchAttention(hidden_size=model_baseline.config.hidden_size)
    metrics_baseline = evaluate_with_strategy(
        model_baseline, tokenizer, cross_batch_baseline, device,
        args.eval_samples, enable_cross_batch=False, strategy_name="baseline",
        eval_batch_size=args.batch_size, rank=rank, world_size=world_size
    )
    all_results['baseline'] = metrics_baseline
    print_rank0(f'Baseline - EM: {metrics_baseline["exact_match"]:.2f}, F1: {metrics_baseline["f1"]:.2f}', rank)
    del cross_batch_baseline

    del trainer_baseline, model_baseline
    gc.collect()
    torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()

    # 3. Cross-Batch 训练 + 测试
    print_rank0('\n[3/3] Cross-Batch：训练后立刻评估 (lm_head + cross-batch)', rank)
    print_rank0('-' * 40, rank)

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
    )
    print_rank0(f'Cross-Batch 最终 Loss: {history_crossbatch["train_loss"][-1]:.4f}', rank)

    # 所有卡并行评估
    metrics_crossbatch = evaluate_with_strategy(
        model_crossbatch, tokenizer, trainer_crossbatch.cross_batch_module, device,
        args.eval_samples, enable_cross_batch=True, strategy_name="crossbatch",
        eval_batch_size=args.batch_size, rank=rank, world_size=world_size
    )
    all_results['crossbatch'] = metrics_crossbatch
    print_rank0(f'Cross-Batch - EM: {metrics_crossbatch["exact_match"]:.2f}, F1: {metrics_crossbatch["f1"]:.2f}', rank)

    # 只在 rank 0 上保存 checkpoint
    if is_main_process(rank):
        # 保存 checkpoint
        os.makedirs(args.save_dir, exist_ok=True)
        model_name = args.model.replace('/', '_')
        checkpoint_path = os.path.join(args.save_dir, f'{model_name}_crossbatch.pt')
        checkpoint = {
            'cross_batch_module': trainer_crossbatch.cross_batch_module.state_dict(),
            'lm_head': model_crossbatch.lm_head.state_dict(),
            'config': {
                'model': args.model,
                'hidden_size': model_crossbatch.config.hidden_size,
                'train_samples': args.max_samples,
                'epochs': args.epochs,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        print(f'\nCheckpoint 已保存到: {checkpoint_path}')

    del trainer_crossbatch, model_crossbatch, cross_batch_module
    gc.collect()
    torch.cuda.empty_cache()

    # 保存结果 (只在 rank 0)
    if is_main_process(rank):
        summary = {
            'config': {
                'model': args.model,
                'train_samples': args.max_samples,
                'eval_samples': args.eval_samples,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'world_size': world_size,
            },
            'training_history': {
                'baseline': history_baseline['train_loss'],
                'crossbatch': history_crossbatch['train_loss'],
            },
            'metrics': all_results,
        }

        os.makedirs('outputs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'outputs/training_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # 打印总结
        print('\n' + '=' * 60)
        print('三方对比总结')
        print('=' * 60)
        print(f'| 方法 | EM | F1 |')
        print(f'|------|-----|-----|')
        print(f'| 原始模型 | {all_results["original"]["exact_match"]:.2f} | {all_results["original"]["f1"]:.2f} |')
        print(f'| Baseline (lm_head) | {all_results["baseline"]["exact_match"]:.2f} | {all_results["baseline"]["f1"]:.2f} |')
        print(f'| Cross-Batch | {all_results["crossbatch"]["exact_match"]:.2f} | {all_results["crossbatch"]["f1"]:.2f} |')

        print('\n改进分析:')
        orig_f1 = all_results['original']['f1']
        base_f1 = all_results['baseline']['f1']
        cross_f1 = all_results['crossbatch']['f1']

        print(f'  Baseline vs 原始: F1 {base_f1 - orig_f1:+.2f}')
        print(f'  Cross-Batch vs 原始: F1 {cross_f1 - orig_f1:+.2f}')
        print(f'  Cross-Batch vs Baseline: F1 {cross_f1 - base_f1:+.2f}')

        if cross_f1 - base_f1 > 1.0:
            print('\n=> Cross-Batch 确实带来了改进！')
        elif cross_f1 - base_f1 < -1.0:
            print('\n=> Cross-Batch 反而损害了性能')
        else:
            print('\n=> 差异较小，需要更多实验验证')

        print(f'\n结果已保存到: {output_file}')
        print('=' * 60)

    cleanup_ddp()


if __name__ == '__main__':
    main()
