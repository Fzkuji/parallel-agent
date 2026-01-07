#!/bin/bash
# 一次性测试指定策略在所有数据集上的效果

set -e

MODEL="Qwen/Qwen2.5-7B-Instruct"
STRATEGIES="all_in_one,sequential,batch,collab_llm"
NPROC=8
OUTPUT_DIR="outputs_json"
MAX_TOKENS=1024

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Testing strategies: $STRATEGIES"
echo "Model: $MODEL"
echo "GPUs: $NPROC"
echo "=========================================="

# SQuAD
echo -e "\n[1/8] Running SQuAD..."
torchrun --nproc_per_node=$NPROC scripts/compare_strategies.py \
    --dataset squad \
    --model-name "$MODEL" \
    --context-count 100 \
    --min-questions 5 --max-questions 8 \
    --max-new-tokens $MAX_TOKENS \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_squad.json"

# HotpotQA
echo -e "\n[2/8] Running HotpotQA..."
torchrun --nproc_per_node=$NPROC scripts/compare_strategies.py \
    --dataset hotpot \
    --hotpot-subset distractor \
    --model-name "$MODEL" \
    --context-count 100 \
    --min-questions 4 --max-questions 4 \
    --max-new-tokens $MAX_TOKENS \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_hotpot.json"

# QuAC
echo -e "\n[3/8] Running QuAC..."
torchrun --nproc_per_node=$NPROC scripts/compare_strategies.py \
    --dataset quac \
    --split train \
    --model-name "$MODEL" \
    --context-count 100 \
    --min-questions 3 --max-questions 6 \
    --max-new-tokens 256 \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_quac.json"

# QuALITY
echo -e "\n[4/8] Running QuALITY..."
torchrun --nproc_per_node=$NPROC scripts/compare_strategies.py \
    --dataset quality \
    --split validation \
    --model-name "$MODEL" \
    --context-count 20 \
    --min-questions 5 --max-questions 10 \
    --max-new-tokens 256 \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_quality.json"

# DROP
echo -e "\n[5/8] Running DROP..."
torchrun --nproc_per_node=$NPROC scripts/compare_strategies.py \
    --dataset drop \
    --split validation \
    --model-name "$MODEL" \
    --context-count 50 \
    --min-questions 5 --max-questions 10 \
    --max-new-tokens 128 \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_drop.json"

# CMB-Clin (Chinese Medical Clinical)
echo -e "\n[6/8] Running CMB-Clin..."
torchrun --nproc_per_node=$NPROC scripts/compare_strategies.py \
    --dataset cmb \
    --cmb-subset CMB-Clin \
    --split test \
    --model-name "$MODEL" \
    --context-count 74 \
    --min-questions 2 --max-questions 4 \
    --max-new-tokens $MAX_TOKENS \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_cmb_clin.json"

# CMB-Exam Context
echo -e "\n[7/8] Running CMB-Exam (context grouping)..."
torchrun --nproc_per_node=$NPROC scripts/compare_strategies.py \
    --dataset cmb \
    --cmb-subset context \
    --split test \
    --model-name "$MODEL" \
    --context-count 100 \
    --min-questions 2 --max-questions 6 \
    --max-new-tokens 64 \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_cmb_exam_context.json"

# CMB-Exam Random (baseline)
echo -e "\n[8/8] Running CMB-Exam (random grouping)..."
torchrun --nproc_per_node=$NPROC scripts/compare_strategies.py \
    --dataset cmb \
    --cmb-subset random \
    --split test \
    --model-name "$MODEL" \
    --context-count 100 \
    --min-questions 5 --max-questions 5 \
    --max-new-tokens 64 \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_cmb_exam_random.json"

echo -e "\n=========================================="
echo "All datasets completed!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=========================================="
