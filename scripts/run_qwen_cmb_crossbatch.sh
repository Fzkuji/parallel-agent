#!/bin/bash
# Qwen2.5-7B + CMB Cross-Batch 推理脚本

python scripts/eval_question_grouping_impact.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --dataset cmb \
    --group-sizes "1,2,3,4,5" \
    --strategies "cross_batch" \
    --cross-batch-checkpoint "outputs/checkpoints/cmb_exam_random/Qwen_Qwen2.5-7B-Instruct_attention_gate.pt" \
    --max-contexts 100 \
    --seed 42 \
    --output-dir "outputs/crossbatch_qwen_cmb_trained"
