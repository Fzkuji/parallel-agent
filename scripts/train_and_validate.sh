#!/bin/bash
# Train CSA and immediately validate on test set

set -e  # Exit on error

MODEL=$1
DATASET=$2

if [ -z "$MODEL" ] || [ -z "$DATASET" ]; then
    echo "Usage: $0 <model> <dataset>"
    echo "Example: $0 'Qwen/Qwen2.5-7B-Instruct' squad"
    exit 1
fi

# Convert model name to filename-safe format
MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')

echo "=========================================="
echo "Training CSA: $MODEL on $DATASET"
echo "=========================================="

# Train with conservative settings
torchrun --nproc_per_node=8 scripts/train_cross_batch.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --max-samples 1000 \
    --min-questions 1 \
    --max-questions 5 \
    --epochs 1 \
    --batch-size 4 \
    --lr 1e-5 \
    --seed 42 \
    --force

echo ""
echo "=========================================="
echo "Validating CSA on test set"
echo "=========================================="

# Validate
python scripts/eval_question_grouping_impact.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --group-sizes "1,2,3,4,5" \
    --strategies "cross_batch" \
    --cross-batch-checkpoint "outputs/checkpoints/${DATASET}/${MODEL_SAFE}_attention_frozen.pt" \
    --max-contexts 100 \
    --seed 42 \
    --output-dir "outputs/crossbatch_${MODEL_SAFE}_${DATASET}"

echo ""
echo "=========================================="
echo "DONE: Results saved to outputs/crossbatch_${MODEL_SAFE}_${DATASET}"
echo "=========================================="
