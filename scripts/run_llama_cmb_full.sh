#!/bin/bash
# Llama3-8B + CMB - 完整训练和推理流程

set -e

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
TRAIN_DATASET="cmb_exam_random"  # 训练用随机分组
TEST_DATASET="cmb"               # 推理用 context 分组

echo "=========================================="
echo "Step 1: Training CSA on CMB (random grouping)"
echo "=========================================="

torchrun --nproc_per_node=8 scripts/train_cross_batch.py \
    --model "$MODEL" \
    --dataset "$TRAIN_DATASET" \
    --max-samples 10000 \
    --min-questions 1 \
    --max-questions 5 \
    --train-lm-head \
    --lr 1e-5 \
    --batch-size 16 \
    --seed 42 \
    --force

echo ""
echo "=========================================="
echo "Step 2: Inference on CMB test set (context grouping)"
echo "=========================================="

python scripts/eval_question_grouping_impact.py \
    --model "$MODEL" \
    --dataset "$TEST_DATASET" \
    --group-sizes "1,2,3,4,5" \
    --strategies "cross_batch" \
    --cross-batch-checkpoint "outputs/checkpoints/cmb_exam_random/meta-llama_Meta-Llama-3-8B-Instruct_attention_gate.pt" \
    --max-contexts 100 \
    --seed 42 \
    --output-dir "outputs/crossbatch_llama_cmb_final"

echo ""
echo "=========================================="
echo "DONE: Llama3-8B + CMB"
echo "=========================================="
echo "Results saved to: outputs/crossbatch_llama_cmb_final"
