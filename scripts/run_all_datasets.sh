#!/bin/bash
# 一次性测试指定策略在所有数据集上的效果

set -e

MODEL="Qwen/Qwen2.5-7B-Instruct"
STRATEGIES="all_in_one,sequential,batch,collab_llm"
NPROC=8
OUTPUT_DIR="outputs_json"
MAX_TOKENS=1024

mkdir -p "$OUTPUT_DIR"

# 汇总文件
SUMMARY_FILE="$OUTPUT_DIR/summary_$(date +%Y%m%d_%H%M%S).txt"

echo "=========================================="
echo "Testing strategies: $STRATEGIES"
echo "Model: $MODEL"
echo "GPUs: $NPROC"
echo "=========================================="

# 运行单个数据集并捕获结果
run_dataset() {
    local name="$1"
    local log_file="$OUTPUT_DIR/${name}_log.txt"
    shift

    echo -e "\n>>> Running $name..."

    # 运行并保存输出
    torchrun --nproc_per_node=$NPROC scripts/compare_strategies.py "$@" 2>&1 | tee "$log_file"

    # 提取 Aggregate Metrics 部分并保存到汇总文件
    echo -e "\n=== $name ===" >> "$SUMMARY_FILE"
    sed -n '/=== Aggregate Metrics ===/,/^INFO\|^$/p' "$log_file" | head -20 >> "$SUMMARY_FILE"
}

# SQuAD
run_dataset "SQuAD" \
    --dataset squad \
    --model-name "$MODEL" \
    --context-count 100 \
    --min-questions 5 --max-questions 8 \
    --max-new-tokens $MAX_TOKENS \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_squad.json"

# HotpotQA
run_dataset "HotpotQA" \
    --dataset hotpot \
    --hotpot-subset distractor \
    --model-name "$MODEL" \
    --context-count 100 \
    --min-questions 4 --max-questions 4 \
    --max-new-tokens $MAX_TOKENS \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_hotpot.json"

# QuAC
run_dataset "QuAC" \
    --dataset quac \
    --split train \
    --model-name "$MODEL" \
    --context-count 100 \
    --min-questions 3 --max-questions 6 \
    --max-new-tokens 256 \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_quac.json"

# QuALITY
run_dataset "QuALITY" \
    --dataset quality \
    --split validation \
    --model-name "$MODEL" \
    --context-count 20 \
    --min-questions 5 --max-questions 10 \
    --max-new-tokens 256 \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_quality.json"

# DROP
run_dataset "DROP" \
    --dataset drop \
    --split validation \
    --model-name "$MODEL" \
    --context-count 50 \
    --min-questions 5 --max-questions 10 \
    --max-new-tokens 128 \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_drop.json"

# CMB-Clin
run_dataset "CMB-Clin" \
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
run_dataset "CMB-Exam-Context" \
    --dataset cmb \
    --cmb-subset context \
    --split test \
    --model-name "$MODEL" \
    --context-count 100 \
    --min-questions 2 --max-questions 6 \
    --max-new-tokens 64 \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_cmb_exam_context.json"

# CMB-Exam Random
run_dataset "CMB-Exam-Random" \
    --dataset cmb \
    --cmb-subset random \
    --split test \
    --model-name "$MODEL" \
    --context-count 100 \
    --min-questions 5 --max-questions 5 \
    --max-new-tokens 64 \
    --strategies "$STRATEGIES" \
    --json-out "$OUTPUT_DIR/results_cmb_exam_random.json"

# 打印汇总结果
echo -e "\n"
echo "############################################################"
echo "#                    ALL RESULTS SUMMARY                   #"
echo "############################################################"
cat "$SUMMARY_FILE"
echo -e "\n=========================================="
echo "All datasets completed!"
echo "Summary saved to: $SUMMARY_FILE"
echo "=========================================="
