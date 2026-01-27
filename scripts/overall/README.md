# Overall Experiment Results

This directory contains the complete experimental results for question grouping impact studies.

## Experiments

All experiments test group sizes G=1,2,3,4,5 with three baseline strategies:
- **Independent (batch)**: Each question processed separately in parallel
- **All-in-One**: All G questions in one prompt
- **Sequential**: Multi-turn conversation

### 1. SQuAD Dataset

**Llama3-8B** ([llama3-8b_squad_g1-5.txt](llama3-8b_squad_g1-5.txt))
- Model: meta-llama/Meta-Llama-3-8B-Instruct
- 480 questions total from 100 contexts
- Metrics: EM, F1, Lenient Accuracy

**Qwen2.5-7B** ([qwen2.5-7b_squad_g1-5.txt](qwen2.5-7b_squad_g1-5.txt))
- Model: Qwen/Qwen2.5-7B-Instruct
- 480 questions total from 100 contexts
- Metrics: EM, F1, Lenient Accuracy

### 2. CMB Dataset (Chinese Medical Benchmark)

**Llama3-8B** ([llama3-8b_cmb_g1-5.txt](llama3-8b_cmb_g1-5.txt))
- Model: meta-llama/Meta-Llama-3-8B-Instruct
- 420 questions total from 84-420 contexts (varies by G)
- Metrics: Accuracy (multiple choice)

**Qwen2.5-7B** ([qwen2.5-7b_cmb_g1-5.txt](qwen2.5-7b_cmb_g1-5.txt))
- Model: Qwen/Qwen2.5-7B-Instruct
- Questions vary by group size
- Metrics: Accuracy (multiple choice)
- Note: Only accuracy metrics available (no latency/token data)

## Key Findings

### SQuAD
- **Independent** maintains consistent accuracy (~78-80%) across all G
- **All-in-One** degrades significantly at G=4 (~50-54%), recovers at G=5 (~73-79%)
- **Sequential** generally outperforms All-in-One at larger G
- Token efficiency improves with larger G (deduplicated tokens decrease)

### CMB
- **Independent** maintains consistent accuracy (~54-60%) across all G
- **All-in-One** shows smaller variations compared to SQuAD
- **Sequential** slightly outperforms Independent at some group sizes
- Qwen2.5-7B significantly outperforms Llama3-8B (60% vs 54%)

## Data Format

Each file contains:
- Accuracy/EM metrics per group size
- Token statistics (deduplicated and API tokens)
- Latency measurements (GPU time and wall time)
- Per-question averages

## Commands Used

**SQuAD experiments:**
```bash
python scripts/eval_question_grouping_impact.py \
    --model "{MODEL}" \
    --dataset squad \
    --group-sizes "1,2,3,4,5" \
    --strategies "batch,all_in_one,sequential" \
    --max-contexts 100 \
    --output-dir "outputs/grouping_impact"
```

**CMB experiments:**
```bash
python scripts/eval_question_grouping_impact.py \
    --model "{MODEL}" \
    --dataset cmb \
    --group-sizes "1,2,3,4,5" \
    --strategies "batch,all_in_one,sequential" \
    --max-contexts 100 \
    --output-dir "outputs/grouping_impact"
```

## Next Steps

Missing data for full table:
- Memory strategy (3-shot with embedding retrieval)
- Cross-Batch strategy (trained checkpoint with cross-question dependency modeling)
