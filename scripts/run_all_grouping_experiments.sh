#!/bin/bash
# Run all grouping impact experiments for the paper table
# Usage: bash scripts/run_all_grouping_experiments.sh
#
# Strategies tested:
# - Independent (batch): Each question processed separately
# - All-in-One: All G questions in one prompt
# - Sequential: Multi-turn conversation
# - Memory: 3-shot with embedding-based retrieval from training set
# - Cross-Batch: Cross-question dependency modeling (requires checkpoint)

set -e

OUTPUT_BASE="outputs/grouping_study_full"
CHECKPOINT_BASE="outputs/cross_batch_checkpoints"
mkdir -p $OUTPUT_BASE
mkdir -p $CHECKPOINT_BASE

echo "=============================================="
echo "Running all grouping impact experiments"
echo "=============================================="

# Models
MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

# Datasets and their group sizes
# SQuAD: 1,4,8,12,16
# CMB: 1,2,4,6,8

# Check if cross-batch training is needed
TRAIN_CROSS_BATCH=${TRAIN_CROSS_BATCH:-false}

if [ "$TRAIN_CROSS_BATCH" = true ]; then
    echo ""
    echo "=== Training Cross-Batch Checkpoints ==="
    echo ""

    for MODEL in "${MODELS[@]}"; do
        MODEL_SHORT=$(echo $MODEL | sed 's/.*\///' | sed 's/-Instruct//')

        for DATASET in "squad" "cmb"; do
            CKPT_DIR="${CHECKPOINT_BASE}/${DATASET}_${MODEL_SHORT}"
            if [ -f "${CKPT_DIR}/best.pt" ]; then
                echo "Checkpoint exists: ${CKPT_DIR}/best.pt, skipping..."
            else
                echo "Training Cross-Batch for ${DATASET} + ${MODEL_SHORT}..."
                python scripts/train_cross_batch.py \
                    --model "$MODEL" \
                    --dataset "$DATASET" \
                    --module-type simple \
                    --epochs 3 \
                    --batch-size 4 \
                    --save-dir "$CKPT_DIR"
                echo "Done training: ${DATASET} + ${MODEL_SHORT}"
            fi
        done
    done
fi

echo ""
echo "=== SQuAD Experiments ==="
echo ""

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo $MODEL | sed 's/.*\///' | sed 's/-Instruct//')
    CKPT_PATH="${CHECKPOINT_BASE}/squad_${MODEL_SHORT}/best.pt"

    echo "Running SQuAD with $MODEL_SHORT..."

    # Build command with optional cross-batch checkpoint
    CMD="python scripts/eval_question_grouping_impact.py \
        --model \"$MODEL\" \
        --dataset squad \
        --group-sizes \"1,4,8,12,16\" \
        --output-dir \"$OUTPUT_BASE/squad_${MODEL_SHORT}\" \
        --seed 42"

    if [ -f "$CKPT_PATH" ]; then
        CMD="$CMD --cross-batch-checkpoint \"$CKPT_PATH\""
        echo "  Using Cross-Batch checkpoint: $CKPT_PATH"
    else
        echo "  No Cross-Batch checkpoint found, skipping cross_batch strategy"
        CMD="$CMD --strategies \"batch,all_in_one,sequential,memory\""
    fi

    eval $CMD

    echo "Done: SQuAD + $MODEL_SHORT"
    echo ""
done

echo ""
echo "=== CMB Experiments ==="
echo ""

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo $MODEL | sed 's/.*\///' | sed 's/-Instruct//')
    CKPT_PATH="${CHECKPOINT_BASE}/cmb_${MODEL_SHORT}/best.pt"

    echo "Running CMB with $MODEL_SHORT..."

    # Build command with optional cross-batch checkpoint
    CMD="python scripts/eval_question_grouping_impact.py \
        --model \"$MODEL\" \
        --dataset cmb \
        --group-sizes \"1,2,4,6,8\" \
        --output-dir \"$OUTPUT_BASE/cmb_${MODEL_SHORT}\" \
        --seed 42"

    if [ -f "$CKPT_PATH" ]; then
        CMD="$CMD --cross-batch-checkpoint \"$CKPT_PATH\""
        echo "  Using Cross-Batch checkpoint: $CKPT_PATH"
    else
        echo "  No Cross-Batch checkpoint found, skipping cross_batch strategy"
        CMD="$CMD --strategies \"batch,all_in_one,sequential,memory\""
    fi

    eval $CMD

    echo "Done: CMB + $MODEL_SHORT"
    echo ""
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_BASE"
echo "=============================================="

# Generate combined summary
echo ""
echo "Generating combined summary..."

python << 'PYTHON_SCRIPT'
import json
from pathlib import Path

output_base = Path("outputs/grouping_study_full")
models = ["Meta-Llama-3-8B", "Qwen2.5-7B"]
datasets = {
    "squad": [1, 4, 8, 12, 16],
    "cmb": [1, 2, 4, 6, 8]
}
strategy_order = ["batch", "all_in_one", "sequential", "memory", "cross_batch"]
strategy_display = {
    "batch": "Independent",
    "all_in_one": "All-in-One",
    "sequential": "Sequential",
    "memory": "Memory",
    "cross_batch": "Cross-Batch",
}

print("\n" + "="*120)
print("COMBINED RESULTS SUMMARY")
print("="*120)

for dataset, group_sizes in datasets.items():
    print(f"\n--- {dataset.upper()} ---")
    print(f"{'Model':<20} {'Strategy':<15}", end="")
    for g in group_sizes:
        print(f" G={g:>2}", end="")
    print()
    print("-" * (35 + 6 * len(group_sizes)))

    for model in models:
        result_dir = output_base / f"{dataset}_{model}"
        all_results_file = result_dir / "all_results.json"

        if not all_results_file.exists():
            print(f"{model:<20} Results not found")
            continue

        with open(all_results_file) as f:
            results = json.load(f)

        for strategy in strategy_order:
            display_name = strategy_display[strategy]
            print(f"{model:<20} {display_name:<15}", end="")
            for g in group_sizes:
                g_str = str(g)
                if g_str in results and strategy in results[g_str]:
                    acc = results[g_str][strategy]["metrics"]["strict_acc"] * 100
                    print(f" {acc:>5.1f}", end="")
                else:
                    print(f"   -- ", end="")
            print()
        print()

print("="*120)
PYTHON_SCRIPT

echo ""
echo "Done!"
