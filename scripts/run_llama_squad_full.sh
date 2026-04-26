#!/bin/bash
# Llama3-8B + SQuAD - 完整训练和推理流程

set -e

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
DATASET="squad"

echo "=========================================="
echo "Step 1: Training CSA on SQuAD"
echo "=========================================="

torchrun --nproc_per_node=8 scripts/train_cross_batch.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
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
echo "Step 2: Inference on SQuAD test set"
echo "=========================================="

python scripts/eval_question_grouping_impact.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --group-sizes "1,2,3,4,5" \
    --strategies "cross_batch" \
    --cross-batch-checkpoint "outputs/checkpoints/squad/meta-llama_Meta-Llama-3-8B-Instruct_attention_gate.pt" \
    --max-contexts 100 \
    --seed 42 \
    --output-dir "outputs/crossbatch_llama_squad_final"

echo ""
echo "=========================================="
echo "DONE: Llama3-8B + SQuAD"
echo "=========================================="
echo "Results saved to: outputs/crossbatch_llama_squad_final"
