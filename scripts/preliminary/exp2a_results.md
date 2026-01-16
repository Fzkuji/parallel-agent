# Experiment 2a: Shared Context Study - Full Results

## Dataset
- **SQuAD**: Multiple questions per paragraph (3-6 questions/group)
- **Groups**: 1000
- **Models**: Qwen2.5-Instruct series (0.5B, 3B, 7B, 14B, 32B)

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| independent | Each question answered separately with full context |
| all_in_one | All questions answered in single prompt |
| seq_cross_ctx | Sequential with Q&A history from DIFFERENT contexts |
| seq_shared_rand | Sequential with shared context, random order |
| seq_shared_ord | Sequential with shared context, LLM-optimized order |

---

## Qwen/Qwen2.5-0.5B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Groups | Questions | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| independent | 0.0026 | 0.1142 | 1000 | 4566 | 927.5 | 279.5 | 0.47 |
| all_in_one | 0.0699 | 0.2066 | 1000 | 4566 | 277.7 | 240.1 | 0.40 |
| seq_cross_ctx | 0.0007 | 0.1089 | 1000 | 4566 | 1528.9 | 285.9 | 0.49 |
| seq_shared_rand | 0.0013 | 0.1083 | 1000 | 4566 | 1652.5 | 285.7 | 0.49 |
| seq_shared_ord | 0.0009 | 0.1093 | 1000 | 4566 | 1949.7 | 317.0 | 0.55 |

**PER-HISTORY-LENGTH ACCURACY (seq_shared_rand)**

| History Length | Samples | EM | F1 |
| --- | --- | --- | --- |
| 0 | 1000 | 0.0040 | 0.1144 |
| 1 | 1000 | 0.0000 | 0.1041 |
| 2 | 1000 | 0.0020 | 0.1153 |
| 3 | 900 | 0.0000 | 0.1048 |
| 4 | 663 | 0.0000 | 0.1000 |
| 5 | 3 | 0.0000 | 0.1170 |

---

## Qwen/Qwen2.5-3B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Groups | Questions | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| independent | 0.0002 | 0.1282 | 1000 | 4566 | 927.5 | 265.4 | 1.01 |
| all_in_one | 0.2806 | 0.4436 | 1000 | 4566 | 277.7 | 73.4 | 0.29 |
| seq_cross_ctx | 0.0015 | 0.1285 | 1000 | 4566 | 1503.2 | 270.1 | 1.05 |
| seq_shared_rand | 0.0026 | 0.1297 | 1000 | 4566 | 1615.1 | 268.7 | 1.03 |
| seq_shared_ord | 0.0015 | 0.1280 | 1000 | 4566 | 1919.3 | 302.9 | 1.17 |

**PER-HISTORY-LENGTH ACCURACY (seq_shared_rand)**

| History Length | Samples | EM | F1 |
| --- | --- | --- | --- |
| 0 | 1000 | 0.0000 | 0.1281 |
| 1 | 1000 | 0.0080 | 0.1378 |
| 2 | 1000 | 0.0020 | 0.1338 |
| 3 | 900 | 0.0011 | 0.1254 |
| 4 | 663 | 0.0015 | 0.1195 |
| 5 | 3 | 0.0000 | 0.1780 |

---

## Qwen/Qwen2.5-7B-Instruct

**EXPERIMENT SUMMARY** (Updated 2026-01-17 with corrected n500 data)

| Condition | EM | F1 | Groups | Questions |
| --- | --- | --- | --- | --- |
| independent | 0.6237 | 0.7869 | 500 | 2283 |
| all_in_one | 0.5278 | 0.6741 | 500 | 2283 |
| seq_cross_ctx | 0.6198 | 0.7831 | 500 | 2283 |
| seq_shared_rand | 0.6040 | 0.7754 | 500 | 2283 |
| seq_shared_ord | 0.6154 | 0.7860 | 500 | 2283 |

*Note: Previous n1000 data showed anomalously low EM (~0.001) for independent/sequential conditions. Corrected with n500 rerun.*

---

## Qwen/Qwen2.5-14B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Groups | Questions | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| independent | 0.4376 | 0.6685 | 500 | 2283 | 952.6 | 551.2 | 5.98 |
| all_in_one | 0.5304 | 0.6593 | 500 | 2283 | 283.3 | 228.0 | 2.51 |
| seq_cross_ctx | 0.5226 | 0.7294 | 500 | 2283 | 1176.1 | 495.7 | 5.41 |
| seq_shared_rand | 0.5313 | 0.7389 | 500 | 2283 | 1212.9 | 466.4 | 5.04 |
| seq_shared_ord | 0.5300 | 0.7452 | 500 | 2283 | 1509.4 | 514.5 | 5.62 |

**PER-HISTORY-LENGTH ACCURACY (seq_shared_rand)**

| History Length | Samples | EM | F1 |
| --- | --- | --- | --- |
| 0 | 500 | 0.4340 | 0.6607 |
| 1 | 500 | 0.5440 | 0.7534 |
| 2 | 500 | 0.5820 | 0.7709 |
| 3 | 451 | 0.5432 | 0.7561 |
| 4 | 331 | 0.5680 | 0.7646 |
| 5 | 1 | 0.0000 | 0.3333 |

---

## Qwen/Qwen2.5-32B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Groups | Questions | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| independent | 0.5738 | 0.7578 | 500 | 2283 | 952.6 | 294.9 | 6.46 |
| all_in_one | 0.5804 | 0.7237 | 500 | 2283 | 283.3 | 73.2 | 1.70 |
| seq_cross_ctx | 0.6224 | 0.7943 | 500 | 2283 | 1174.0 | 291.8 | 6.47 |
| seq_shared_rand | 0.6338 | 0.8028 | 500 | 2283 | 1209.8 | 264.7 | 5.79 |
| seq_shared_ord | 0.6325 | 0.8038 | 500 | 2283 | 1509.3 | 297.6 | 6.59 |

**PER-HISTORY-LENGTH ACCURACY (seq_shared_rand)**

| History Length | Samples | EM | F1 |
| --- | --- | --- | --- |
| 0 | 500 | 0.5960 | 0.7589 |
| 1 | 500 | 0.6320 | 0.8060 |
| 2 | 500 | 0.6620 | 0.8260 |
| 3 | 451 | 0.6253 | 0.8139 |
| 4 | 331 | 0.6647 | 0.8152 |
| 5 | 1 | 0.0000 | 0.5000 |

---

## Summary Table (EM)

| Model | independent | all_in_one | seq_cross_ctx | seq_shared_rand | seq_shared_ord |
|-------|-------|-------|-------|-------|-------|
| 0.5B | 0.0026 | 0.0699 | 0.0007 | 0.0013 | 0.0009 |
| 3B | 0.0002 | 0.2806 | 0.0015 | 0.0026 | 0.0015 |
| 7B | 0.6237 | 0.5278 | 0.6198 | 0.6040 | 0.6154 |
| 14B | 0.4376 | 0.5304 | 0.5226 | 0.5313 | 0.5300 |
| 32B | 0.5738 | 0.5804 | 0.6224 | 0.6338 | 0.6325 |

## Summary Table (F1)

| Model | independent | all_in_one | seq_cross_ctx | seq_shared_rand | seq_shared_ord |
|-------|-------|-------|-------|-------|-------|
| 0.5B | 0.1142 | 0.2066 | 0.1089 | 0.1083 | 0.1093 |
| 3B | 0.1282 | 0.4436 | 0.1285 | 0.1297 | 0.1280 |
| 7B | 0.7869 | 0.6741 | 0.7831 | 0.7754 | 0.7860 |
| 14B | 0.6685 | 0.6593 | 0.7294 | 0.7389 | 0.7452 |
| 32B | 0.7578 | 0.7237 | 0.7943 | 0.8028 | 0.8038 |
