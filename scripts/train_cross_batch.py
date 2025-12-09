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

用法:
  # 0.5B 模型 (单卡)
  python scripts/train_cross_batch.py

  # 0.5B 模型 (8卡并行)
  torchrun --nproc_per_node=8 scripts/train_cross_batch.py

  # 7B 模型 (8卡并行)
  torchrun --nproc_per_node=8 scripts/train_cross_batch.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --batch-size 2 \
      --max-samples 10000 \
      --epochs 1 \
      --eval-samples 100
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
                           enable_cross_batch=True, strategy_name="eval"):
    """使用和 compare_strategies.py 相同的评估逻辑"""
    # 加载 SQuAD 验证集 (随机采样问题，每个问题有自己的 context)
    groups = load_squad_random_questions(
        split="validation",
        max_contexts=eval_samples,
        seed=42,
    )

    # 转换为 items 格式 (和 run_cross_batch_multi_strategy 兼容)
    items = []
    for group in groups:
        context = group["context"]
        for q in group["questions"]:
            items.append({
                "qid": q["qid"],
                "question": q["text"],
                "context": context,
                "references": q["references"],
            })

    # 创建 generator
    generator = CrossBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        device=device,
    )

    # 运行评估
    result = run_cross_batch_multi_strategy(
        items=items,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=32,
        strategy_name=strategy_name,
        dataset="squad",
        cross_batch_generator=generator,
        enable_cross_batch=enable_cross_batch,
    )

    return {
        "exact_match": result.metrics.get("em", 0.0) * 100,
        "f1": result.metrics.get("f1", 0.0) * 100,
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

    # 1. 评估原始模型 (只在 rank 0 上做)
    if is_main_process(rank):
        print('\n[1/3] 评估原始模型 (未微调)')
        print('-' * 40)
        model_original = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
        cross_batch_original = CrossBatchAttention(hidden_size=model_original.config.hidden_size)

        metrics_original = evaluate_with_strategy(
            model_original, tokenizer, cross_batch_original, device,
            args.eval_samples, enable_cross_batch=False, strategy_name="original"
        )
        all_results['original'] = metrics_original
        print(f'原始模型 - EM: {metrics_original["exact_match"]:.2f}, F1: {metrics_original["f1"]:.2f}')

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

    # 只在 rank 0 上评估
    if is_main_process(rank):
        cross_batch_baseline = CrossBatchAttention(hidden_size=model_baseline.config.hidden_size)
        metrics_baseline = evaluate_with_strategy(
            model_baseline, tokenizer, cross_batch_baseline, device,
            args.eval_samples, enable_cross_batch=False, strategy_name="baseline"
        )
        all_results['baseline'] = metrics_baseline
        print(f'Baseline - EM: {metrics_baseline["exact_match"]:.2f}, F1: {metrics_baseline["f1"]:.2f}')
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

    # 只在 rank 0 上评估
    if is_main_process(rank):
        metrics_crossbatch = evaluate_with_strategy(
            model_crossbatch, tokenizer, trainer_crossbatch.cross_batch_module, device,
            args.eval_samples, enable_cross_batch=True, strategy_name="crossbatch"
        )
        all_results['crossbatch'] = metrics_crossbatch
        print(f'Cross-Batch - EM: {metrics_crossbatch["exact_match"]:.2f}, F1: {metrics_crossbatch["f1"]:.2f}')

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
