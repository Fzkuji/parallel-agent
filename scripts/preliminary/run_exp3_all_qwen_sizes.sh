#!/bin/bash

# Run Experiment 3: N-Shot Effect Analysis across all Qwen2.5 Instruct model sizes
# Tests: 0.5B, 1.5B, 3B, 7B, 14B, 32B

cd "$(dirname "$0")/../.."

MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
)

DATASETS="squad drop triviaqa mmlu gsm8k"
SHOTS="0 1 2 3 4 5"
MAX_CONTEXTS=100
MAX_NEW_TOKENS=512
BATCH_SIZE=16

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "Testing model: $MODEL"
    echo "========================================"

    # Extract model size for output directory
    MODEL_SIZE=$(echo "$MODEL" | grep -oE '[0-9]+\.?[0-9]*B')
    OUTPUT_DIR="outputs/preliminary/exp3/qwen2.5-${MODEL_SIZE}"

    python scripts/preliminary/exp3_nshot_effect.py \
        --model "$MODEL" \
        --datasets $DATASETS \
        --shots $SHOTS \
        --max-contexts $MAX_CONTEXTS \
        --max-new-tokens $MAX_NEW_TOKENS \
        --batch-size $BATCH_SIZE \
        --output-dir "$OUTPUT_DIR"

    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
done

echo "========================================"
echo "All models tested!"
echo "========================================"
