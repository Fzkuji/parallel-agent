"""
Cross-batch 模块训练脚本
包含训练和三方对比评估
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import gc
import json
import os
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.cross_batch.trainer import LMHeadOnlyTrainer, CrossBatchTrainer, SQuADDataset
from src.cross_batch.attention import CrossBatchAttention
from src.cross_batch.generator import CrossBatchGenerator
from src.cross_batch.eval import SquadEvaluator

# 配置
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
DEVICE = 'cuda'
MAX_SAMPLES = 2000
NUM_EPOCHS = 1
BATCH_SIZE = 8
EVAL_SAMPLES = 100

def main():
    gc.collect()
    torch.cuda.empty_cache()

    print('=' * 60)
    print(f'训练配置: {MODEL_NAME}')
    print(f'样本数: {MAX_SAMPLES}, Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}')
    print('=' * 60)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    train_dataset = SQuADDataset(tokenizer=tokenizer, split='train', max_samples=MAX_SAMPLES)
    print(f'训练数据集大小: {len(train_dataset)}')

    os.makedirs('outputs/inference_results', exist_ok=True)
    all_results = {}

    # 1. 评估原始模型
    print('\n[1/3] 评估原始模型 (未微调)')
    print('-' * 40)
    model_original = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    generator_original = CrossBatchGenerator(
        model=model_original,
        tokenizer=tokenizer,
        cross_batch_module=CrossBatchAttention(hidden_size=model_original.config.hidden_size),
        device=DEVICE,
    )
    evaluator_original = SquadEvaluator(generator_original, tokenizer, split='validation', max_samples=EVAL_SAMPLES)
    results_original = evaluator_original.evaluate(batch_size=BATCH_SIZE, max_new_tokens=32, enable_cross_batch=False)
    all_results['original'] = results_original['metrics']
    print(f'原始模型 - EM: {results_original["metrics"]["exact_match"]:.2f}, F1: {results_original["metrics"]["f1"]:.2f}')

    del model_original, generator_original, evaluator_original
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Baseline 训练 + 测试
    print('\n[2/3] Baseline：训练后立刻评估 (只训练 lm_head)')
    print('-' * 40)
    model_baseline = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    trainer_baseline = LMHeadOnlyTrainer(
        model=model_baseline,
        tokenizer=tokenizer,
        device=DEVICE,
        learning_rate=1e-4,
    )
    history_baseline = trainer_baseline.train(
        train_dataset=train_dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir=None,
    )
    print(f'Baseline 最终 Loss: {history_baseline["train_loss"][-1]:.4f}')

    generator_baseline = CrossBatchGenerator(
        model=model_baseline,
        tokenizer=tokenizer,
        device=DEVICE,
    )
    evaluator_baseline = SquadEvaluator(generator_baseline, tokenizer, split='validation', max_samples=EVAL_SAMPLES)
    results_baseline = evaluator_baseline.evaluate(batch_size=BATCH_SIZE, max_new_tokens=32, enable_cross_batch=False)
    all_results['baseline'] = results_baseline['metrics']
    print(f'Baseline - EM: {results_baseline["metrics"]["exact_match"]:.2f}, F1: {results_baseline["metrics"]["f1"]:.2f}')

    del generator_baseline, evaluator_baseline, trainer_baseline, model_baseline
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Cross-Batch 训练 + 测试
    print('\n[3/3] Cross-Batch：训练后立刻评估 (lm_head + cross-batch)')
    print('-' * 40)
    model_crossbatch = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    cross_batch_module = CrossBatchAttention(hidden_size=model_crossbatch.config.hidden_size)
    trainer_crossbatch = CrossBatchTrainer(
        model=model_crossbatch,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        device=DEVICE,
        learning_rate=1e-4,
        train_lm_head=True,
    )
    history_crossbatch = trainer_crossbatch.train(
        train_dataset=train_dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir=None,
    )
    print(f'Cross-Batch 最终 Loss: {history_crossbatch["train_loss"][-1]:.4f}')

    generator_crossbatch = CrossBatchGenerator(
        model=model_crossbatch,
        tokenizer=tokenizer,
        cross_batch_module=trainer_crossbatch.cross_batch_module,
        device=DEVICE,
    )
    evaluator_crossbatch = SquadEvaluator(generator_crossbatch, tokenizer, split='validation', max_samples=EVAL_SAMPLES)
    results_crossbatch = evaluator_crossbatch.evaluate(batch_size=BATCH_SIZE, max_new_tokens=32, enable_cross_batch=True)
    all_results['crossbatch'] = results_crossbatch['metrics']
    print(f'Cross-Batch - EM: {results_crossbatch["metrics"]["exact_match"]:.2f}, F1: {results_crossbatch["metrics"]["f1"]:.2f}')

    del generator_crossbatch, evaluator_crossbatch, trainer_crossbatch, model_crossbatch, cross_batch_module
    gc.collect()
    torch.cuda.empty_cache()

    # 保存结果
    summary = {
        'config': {
            'model': MODEL_NAME,
            'train_samples': MAX_SAMPLES,
            'eval_samples': EVAL_SAMPLES,
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
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

if __name__ == '__main__':
    main()
