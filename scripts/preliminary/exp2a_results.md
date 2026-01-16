# Experiment 2a: Shared Context Study - Full Results

## Dataset
- **SQuAD**: Multiple questions per paragraph (3-6 questions/group)
- **Groups**: 500
- **Models**: Qwen2.5-7B-Instruct

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| independent | Each question answered separately with full context |
| all_in_one | All questions answered in single prompt |
| seq_cross_ctx | Sequential with Q&A history from DIFFERENT contexts |
| seq_shared_rand | Sequential with shared context, random order |
| seq_shared_ord | Sequential with shared context, LLM-optimized order |

---

## Qwen/Qwen2.5-7B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Groups | Questions | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| independent | 0.2383 | 0.4829 | 500 | 2283 | 925.2 | 225.7 | 1.32 |
| all_in_one | 0.3841 | 0.5934 | 500 | 2283 | 276.3 | 148.0 | 0.87 |
| seq_cross_ctx | 0.1235 | 0.3341 | 500 | 2283 | 1293.9 | 265.7 | 1.58 |
| seq_shared_rand | 0.1288 | 0.3451 | 500 | 2283 | 1365.5 | 259.3 | 1.51 |
| seq_shared_ord | 0.1253 | 0.3463 | 500 | 2283 | 1651.0 | 290.7 | 1.72 |

**PER-HISTORY-LENGTH ACCURACY (seq_shared_rand)**

| History Length | Samples | EM | F1 |
| --- | --- | --- | --- |
| 0 | 500 | 0.2620 | 0.4963 |
| 1 | 500 | 0.1460 | 0.3730 |
| 2 | 500 | 0.0840 | 0.3069 |
| 3 | 451 | 0.0710 | 0.2664 |
| 4 | 331 | 0.0514 | 0.2403 |
| 5 | 1 | 0.0000 | 0.0800 |

---

## Summary Table (EM)

| Model | all_in_one | independent | seq_cross_ctx | seq_shared_ord | seq_shared_rand |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 0.3841 | 0.2383 | 0.1235 | 0.1253 | 0.1288 |

## Summary Table (F1)

| Model | all_in_one | independent | seq_cross_ctx | seq_shared_ord | seq_shared_rand |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 0.5934 | 0.4829 | 0.3341 | 0.3463 | 0.3451 |

