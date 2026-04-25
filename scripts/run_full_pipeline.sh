#!/usr/bin/env bash
# Sequential pipeline: train Finetuned baseline -> train CSA-v2 -> eval both.
# All four phases share the same data + steps so the EM differences directly
# attribute the gain to (lm_head SFT) vs (lm_head SFT + CSA).
#
# Usage: bash scripts/run_full_pipeline.sh

set -euo pipefail

WORK=/root/autodl-tmp/work
MODEL=/root/autodl-fs/models/Qwen2.5-7B-Instruct
LOG=$WORK/pipeline.log

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$WORK/cache/huggingface
export PYTHONPATH=$WORK/parallel-agent

cd "$PYTHONPATH"

NCTX=1000
QPC=4
CPB=2
EPOCHS=3
EVAL_NCTX=80

stamp() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p "$WORK"
: > "$LOG"

stamp "PHASE 1: train Finetuned baseline"
python scripts/train_finetuned_baseline.py \
    --model-path "$MODEL" \
    --num-train-contexts $NCTX \
    --questions-per-context $QPC \
    --contexts-per-batch $CPB \
    --epochs $EPOCHS \
    --output-dir "$WORK/finetuned_run2" \
    >> "$LOG" 2>&1

stamp "PHASE 2: train CSA-v2"
python scripts/train_csa_v2.py \
    --model-path "$MODEL" \
    --num-train-contexts $NCTX \
    --questions-per-context $QPC \
    --contexts-per-batch $CPB \
    --epochs $EPOCHS \
    --output-dir "$WORK/csa_v2_run2" \
    >> "$LOG" 2>&1

stamp "PHASE 3: eval Finetuned baseline"
python scripts/eval_csa_v2.py \
    --model-path "$MODEL" \
    --checkpoint "$WORK/finetuned_run2/best_model.pt" \
    --num-eval-contexts $EVAL_NCTX \
    --group-sizes 1,2,3,4,5 \
    --output-dir "$WORK/eval_baseline2" \
    >> "$LOG" 2>&1

stamp "PHASE 4: eval CSA-v2"
python scripts/eval_csa_v2.py \
    --model-path "$MODEL" \
    --checkpoint "$WORK/csa_v2_run2/best_model.pt" \
    --num-eval-contexts $EVAL_NCTX \
    --group-sizes 1,2,3,4,5 \
    --output-dir "$WORK/eval_csa_v2_2" \
    >> "$LOG" 2>&1

stamp "ALL DONE"
echo "===== SUMMARY =====" >> "$LOG"
grep -A 6 "SUMMARY" "$WORK/eval_baseline2.log" >> "$LOG" 2>/dev/null || true
grep -A 6 "SUMMARY" "$WORK/eval_csa_v2_2.log" >> "$LOG" 2>/dev/null || true
