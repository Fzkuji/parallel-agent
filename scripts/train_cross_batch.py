"""
Cross-batch 模块训练脚本 (支持 DDP 多卡并行)

参数:
  --model           模型名称 (default: Qwen/Qwen2.5-0.5B-Instruct)
  --dataset         数据集 (default: squad)
  --max-samples     训练样本数 (default: None, 使用全部训练集)
  --eval-samples    评估样本数/context数 (default: None, 使用全部验证集)
  --min-questions   每个 context 最少问题数 (default: 1)
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
import gc
import json
import os
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.cross_batch.trainer import (
    CrossBatchTrainer,
    SQuADGroupedDataset,
)
from src.cross_batch.attention import (
    CrossBatchAttention,
    SimpleCrossBatchGate,
    MultiLayerCrossBatch,
    MultiLayerCrossBatchAttention,
)
from src.cross_batch.generator import CrossBatchGenerator
from src.strategies.cross_batch import run_cross_batch_multi_strategy
from src import (
    load_squad_random_questions,
    load_squad_groups,
    load_hotpot_groups,
    load_quac_groups,
    load_drop_groups,
    load_cmb_groups,
    load_cmb_exam_random_groups,
    load_cmb_exam_subdomain_groups,
    load_cmb_exam_context_groups,
    load_triviaqa_groups,
    load_similarity_grouped_triviaqa,
)


def get_checkpoint_path(base_dir: str, dataset: str, model_name: str, mode: str) -> str:
    """Generate checkpoint path."""
    safe_model_name = model_name.replace('/', '_')
    checkpoint_dir = os.path.join(base_dir, dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f'{safe_model_name}_{mode}.pt')


def should_skip_training(checkpoint_path: str, force: bool, rank: int = 0) -> bool:
    """Check if training should be skipped."""
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
                        choices=['squad', 'hotpot', 'quac', 'drop', 'triviaqa', 'triviaqa_sim',
                                 'cmb_clin', 'cmb_exam_context', 'cmb_exam_subdomain', 'cmb_exam_random'],
                        help='数据集 (default: squad)')
    parser.add_argument('--split', type=str, default='validation',
                        help='评估用的数据集 split (default: validation)')
    parser.add_argument('--train-split', type=str, default='train',
                        help='训练用的数据集 split (default: train)')
    parser.add_argument('--min-questions', type=int, default=1,
                        help='每个 context 最少问题数 (default: 1)')
    parser.add_argument('--max-questions', type=int, default=5,
                        help='每个 context 最多问题数 (default: 5)')
    parser.add_argument('--squad-random-questions', action='store_true',
                        help='SQuAD: 随机采样问题而非按 context 分组')

    # 相似度分组参数
    parser.add_argument('--similarity-threshold', type=float, default=0.5,
                        help='相似度分组的阈值 (default: 0.5)')
    parser.add_argument('--embedding-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='相似度分组使用的 embedding 模型')

    # 训练参数
    parser.add_argument('--max-samples', type=int, default=None,
                        help='训练样本数 (default: None, 使用全部数据)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='训练轮数 (default: 1)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='每卡 batch size (default: 16)')
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

    # Cross-batch 模块参数
    parser.add_argument('--module-type', type=str, default='attention',
                        choices=['attention', 'multi_layer_attention'],
                        help='Cross-batch 模块类型: attention (单层完整QKV), multi_layer_attention (多层完整QKV)')
    parser.add_argument('--mix-layer', type=int, default=-1,
                        help='单层模式使用的层 (-1=最后层, 正数=中间层, 如 16 表示第16层)')
    parser.add_argument('--mix-layers', type=str, default=None,
                        help='多层模式使用的层列表 (逗号分隔, 如 "8,12,16,20,24,28")')
    parser.add_argument('--self-only', action='store_true',
                        help='Ablation: CrossBatch 模块只关注自己（对角线 attention），禁用跨样本交互')
    parser.add_argument('--use-gate', action='store_true',
                        help='使用 Question-Aware Gating 而非固定 scale (仅 attention 模式)')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Top-k sparsification: 每个 query 只保留 top-k 个最相关的连接 (default: None, 不限制)')
    parser.add_argument('--train-lm-head', action='store_true', default=False,
                        help='同时训练 lm_head (default: False)')
    parser.add_argument('--no-train-lm-head', action='store_false', dest='train_lm_head',
                        help='不训练 lm_head，只训练 cross-batch 模块')

    # LoRA 参数
    parser.add_argument('--use-lora', action='store_true', default=False,
                        help='使用 LoRA 微调模型')
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank (default: 16)')
    parser.add_argument('--lora-alpha', type=int, default=32,
                        help='LoRA alpha (default: 32)')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                        help='LoRA dropout (default: 0.05)')
    parser.add_argument('--lora-target-modules', type=str, default='q_proj,k_proj,v_proj,o_proj',
                        help='LoRA target modules (逗号分隔, default: q_proj,k_proj,v_proj,o_proj)')
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
    """根据数据集参数加载评估数据"""
    dataset = args.dataset
    split = args.split
    max_contexts = args.eval_samples
    min_questions = args.min_questions
    max_questions = args.max_questions
    seed = args.seed

    if dataset == "squad":
        if args.squad_random_questions:
            groups = load_squad_random_questions(
                split=split,
                max_contexts=max_contexts,
                seed=seed,
            )
        else:
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
            subset="fullwiki",
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
    elif dataset == "cmb_clin":
        groups = load_cmb_groups(
            split="test",
            subset="CMB-Clin",
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
    elif dataset == "triviaqa":
        groups = load_triviaqa_groups(
            split=split,
            max_groups=max_contexts,
            min_questions=min_questions,
            max_questions=max_questions,
            seed=seed,
        )
    elif dataset == "triviaqa_sim":
        groups = load_similarity_grouped_triviaqa(
            split=split,
            max_groups=max_contexts,
            min_questions=min_questions,
            max_questions=max_questions,
            similarity_threshold=getattr(args, 'similarity_threshold', 0.5),
            embedding_model=getattr(args, 'embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            seed=seed,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return groups


def load_train_data(args, rank=0):
    """加载训练数据"""
    dataset = args.dataset
    max_contexts = args.max_samples
    min_questions = args.min_questions
    max_questions = args.max_questions
    seed = args.seed
    train_split = getattr(args, 'train_split', 'train')

    if rank == 0 and dataset == "triviaqa_sim":
        print(f"使用相似度分组 TriviaQA，阈值: {getattr(args, 'similarity_threshold', 0.5)}")

    if dataset == "squad":
        groups = load_squad_groups(
            split=train_split,
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    elif dataset == "hotpot":
        groups = load_hotpot_groups(
            split=train_split,
            subset="fullwiki",
            max_contexts=max_contexts,
            min_questions=min_questions,
            max_questions=max_questions,
            seed=seed,
        )
    elif dataset == "quac":
        groups = load_quac_groups(
            split=train_split,
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    elif dataset == "drop":
        groups = load_drop_groups(
            split=train_split,
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    elif dataset == "cmb_clin":
        groups = load_cmb_groups(
            split="test",
            subset="CMB-Clin",
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    elif dataset == "cmb_exam_context":
        groups = load_cmb_exam_context_groups(
            split=train_split,
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    elif dataset == "cmb_exam_subdomain":
        raw_groups = load_cmb_exam_subdomain_groups(
            split=train_split,
            min_questions=min_questions,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
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
            split=train_split,
            questions_per_group=max_questions or 5,
            max_contexts=max_contexts,
            seed=seed,
        )
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
    elif dataset == "triviaqa":
        groups = load_triviaqa_groups(
            split=train_split,
            max_groups=max_contexts,
            min_questions=min_questions,
            max_questions=max_questions,
            seed=seed,
        )
    elif dataset == "triviaqa_sim":
        groups = load_similarity_grouped_triviaqa(
            split=train_split,
            max_groups=max_contexts,
            min_questions=min_questions,
            max_questions=max_questions,
            similarity_threshold=getattr(args, 'similarity_threshold', 0.5),
            embedding_model=getattr(args, 'embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            seed=seed,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return groups


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
    if args.use_gate:
        print_rank0(f'Cross-batch 模块: Question-Aware Gating', rank)
    if args.top_k is not None:
        print_rank0(f'Top-k sparsification: k={args.top_k}', rank)
    if not args.train_lm_head:
        print_rank0(f'只训练 cross-batch 模块，不训练 lm_head', rank)
    print_rank0('=' * 60, rank)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集 (按 context 分组)
    train_groups = load_train_data(args, rank=rank)

    # 平衡采样：确保不同问题数的 context 数量均衡
    if args.min_questions < args.max_questions:
        import random
        from collections import defaultdict

        # 按问题数分组
        groups_by_qcount = defaultdict(list)
        for g in train_groups:
            num_q = len(g.get("questions", g.get("items", [])))
            groups_by_qcount[num_q].append(g)

        # 统计每组数量
        counts = {q: len(groups) for q, groups in groups_by_qcount.items()}
        print_rank0(f'原始分布: {dict(sorted(counts.items()))}', rank)

        # 找到最小的组（限制因素）
        min_count = min(counts.values()) if counts else 0
        target_per_group = min(min_count, args.max_samples // len(groups_by_qcount)) if args.max_samples else min_count

        # 每组采样相同数量
        rng = random.Random(args.seed)
        balanced_groups = []
        for q_count in sorted(groups_by_qcount.keys()):
            group_contexts = groups_by_qcount[q_count]
            sampled = rng.sample(group_contexts, min(target_per_group, len(group_contexts)))
            balanced_groups.extend(sampled)

        train_groups = balanced_groups
        rng.shuffle(train_groups)  # 打乱顺序

        new_counts = defaultdict(int)
        for g in train_groups:
            num_q = len(g.get("questions", g.get("items", [])))
            new_counts[num_q] += 1
        print_rank0(f'平衡后分布: {dict(sorted(new_counts.items()))}', rank)

    train_dataset = SQuADGroupedDataset(
        tokenizer=tokenizer,
        groups=train_groups,
        dataset_name=args.dataset,
    )
    # Handle both SQuAD format (questions) and HotpotQA format (items)
    total_questions = sum(len(g.get("questions", g.get("items", []))) for g in train_groups)
    print_rank0(f'训练数据集: {len(train_dataset)} 个 context, {total_questions} 个问题', rank)

    # 解析多层配置
    mix_layers = None
    if args.mix_layers:
        mix_layers = [int(x.strip()) for x in args.mix_layers.split(',')]

    # 构建 checkpoint 路径
    mode_suffix = args.module_type
    if args.module_type == 'multi_layer' and mix_layers:
        mode_suffix += f'_L{len(mix_layers)}'
    elif args.mix_layer != -1:
        mode_suffix += f'_L{args.mix_layer}'
    if args.use_gate and args.module_type == 'attention':
        mode_suffix += '_gate'
    if not args.train_lm_head:
        mode_suffix += '_frozen'
    if args.use_lora:
        mode_suffix += '_lora'

    checkpoint_path = get_checkpoint_path(args.save_dir, args.dataset, args.model, mode_suffix)

    if should_skip_training(checkpoint_path, args.force, rank):
        print_rank0('训练已跳过', rank)
        cleanup_ddp()
        return

    # 加载模型
    print_rank0('\n开始训练 Cross-Batch 模块...', rank)
    print_rank0('-' * 40, rank)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    # 应用 LoRA (如果启用)
    lora_model = None
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            lora_model = get_peft_model(model, lora_config)
            lora_model.print_trainable_parameters()
            print_rank0(f'LoRA 配置: r={args.lora_r}, alpha={args.lora_alpha}, target_modules={target_modules}', rank)
            # 使用 LoRA 包装的模型进行训练
            model = lora_model
        except ImportError:
            print_rank0('警告: peft 未安装，无法使用 LoRA。安装: pip install peft', rank)
            args.use_lora = False

    # 创建 cross-batch 模块
    if args.module_type == 'multi_layer':
        # 多层模式: 每层一个 SimpleCrossBatchGate
        layer_indices = mix_layers if mix_layers else list(range(num_layers // 2, num_layers))  # 默认后半层
        cross_batch_module = MultiLayerCrossBatch(
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_indices=layer_indices,
            top_k=args.top_k,
        )
        print_rank0(f'模块类型: MultiLayerCrossBatch, 层数: {len(layer_indices)}, 层: {layer_indices}', rank)
    elif args.module_type == 'multi_layer_attention':
        # 多层 attention 模式: 每层一个完整的 CrossBatchAttention
        layer_indices = mix_layers if mix_layers else list(range(num_layers // 2, num_layers))  # 默认后半层
        cross_batch_module = MultiLayerCrossBatchAttention(
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_indices=layer_indices,
            use_gate=args.use_gate,
            top_k=args.top_k,
        )
        print_rank0(f'模块类型: MultiLayerCrossBatchAttention, 层数: {len(layer_indices)}, 层: {layer_indices}, gate={args.use_gate}', rank)
    else:  # 'attention' (default)
        # Cross-attention with learnable Q/K/V projections
        cross_batch_module = CrossBatchAttention(
            hidden_size=hidden_size,
            self_only=args.self_only,
            use_gate=args.use_gate,
            top_k=args.top_k,
        )
        print_rank0(f'模块类型: CrossBatchAttention, gate={args.use_gate}, 使用层: {args.mix_layer}', rank)

    # 计算参数量
    num_params = sum(p.numel() for p in cross_batch_module.parameters())
    print_rank0(f'Cross-batch 模块参数量: {num_params:,}', rank)

    trainer = CrossBatchTrainer(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        device=device,
        learning_rate=args.lr,
        train_lm_head=args.train_lm_head,
        train_lora=args.use_lora,
        local_rank=local_rank if world_size > 1 else -1,
        mix_layer=args.mix_layer,
    )

    history = trainer.train(
        train_dataset=train_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=None,
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size,
        grouped=True,
    )

    print_rank0(f'最终 Loss: {history["train_loss"][-1]:.4f}', rank)
    print_rank0(f'最终 Improvement: {history["improvement"][-1]:.4f}', rank)

    # 保存 checkpoint
    if is_main_process(rank):
        # 获取 base model 的 config (LoRA model 可能没有直接的 config)
        base_model = model.base_model if hasattr(model, 'base_model') else model
        checkpoint = {
            'cross_batch_module': trainer.cross_batch_module_unwrapped.state_dict(),
            'config': {
                'model': args.model,
                'dataset': args.dataset,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'train_samples': args.max_samples,
                'epochs': args.epochs,
                'module_type': args.module_type,
                'mix_layer': args.mix_layer,
                'mix_layers': mix_layers,
                'use_gate': args.use_gate,
                'top_k': args.top_k,
                'train_lm_head': args.train_lm_head,
                'use_lora': args.use_lora,
                'lora_r': args.lora_r if args.use_lora else None,
                'lora_alpha': args.lora_alpha if args.use_lora else None,
                'lora_target_modules': args.lora_target_modules if args.use_lora else None,
            },
        }
        if args.train_lm_head:
            # 获取 lm_head (可能在 base_model 里)
            lm_head = base_model.lm_head if hasattr(base_model, 'lm_head') else model.lm_head
            checkpoint['lm_head'] = lm_head.state_dict()
        # 保存 LoRA 权重
        if args.use_lora and lora_model is not None:
            # 只保存 LoRA adapter 参数
            lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()}
            checkpoint['lora'] = lora_state_dict
            print_rank0(f'保存 LoRA 参数: {len(lora_state_dict)} 个张量', rank)
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint 已保存到: {checkpoint_path}')

        # 保存训练历史
        os.makedirs('outputs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'outputs/training_{args.dataset}_{timestamp}.json'
        summary = {
            'config': {
                'model': args.model,
                'dataset': args.dataset,
                'min_questions': args.min_questions,
                'max_questions': args.max_questions,
                'train_samples': args.max_samples,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'world_size': world_size,
                'module_type': args.module_type,
                'mix_layer': args.mix_layer,
                'mix_layers': mix_layers,
                'use_gate': args.use_gate,
                'top_k': args.top_k,
                'train_lm_head': args.train_lm_head,
                'num_params': num_params,
            },
            'training_history': history,
            'checkpoint': checkpoint_path,
        }
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print('\n' + '=' * 60)
        print('训练完成!')
        print('=' * 60)
        print(f'\nCheckpoint 保存位置: {checkpoint_path}')
        print(f'训练历史已保存到: {output_file}')
        print('=' * 60)

    del model, cross_batch_module, trainer
    gc.collect()
    torch.cuda.empty_cache()

    cleanup_ddp()


if __name__ == '__main__':
    main()
