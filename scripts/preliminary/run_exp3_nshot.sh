#!/bin/bash

# Run Experiment 3: N-Shot Effect Analysis
# Tests how the number of consecutive questions (0-5 shot) affects performance

cd "$(dirname "$0")/../.."

python scripts/preliminary/exp3_nshot_effect.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --datasets squad drop triviaqa mmlu gsm8k \
    --shots 0 1 2 3 4 5 \
    --max-contexts 100 \
    --max-new-tokens 64 \
    --output-dir outputs/preliminary/exp3
