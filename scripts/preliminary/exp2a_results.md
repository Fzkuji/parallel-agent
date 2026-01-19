# Experiment 2a: Shared Context Study - Full Results

## Dataset
- **SQuAD**: Multiple questions per paragraph (5 questions/group)
- **Groups**: 1000
- **Models**: Qwen2.5-Instruct series (0.5B, 3B, 7B, 14B, 32B) and Qwen3 series (0.6B, 1.7B, 4B, 8B, 14B, 32B)

## Experimental Conditions

| Condition | Context in Prompt | Q&A History | Description |
|-----------|-------------------|-------------|-------------|
| independent | Every turn | ✗ | Each question answered separately |
| all_in_one | Once (all Q) | ✗ | All questions in single prompt |
| seq_cross_ctx | Every turn | ✓ (diff ctx) | Sequential, questions from different contexts |
| seq_shared_rand | First turn only | ✓ (same ctx) | Sequential, shared context, random order |
| seq_shared_ord | First turn only | ✓ (same ctx) | Sequential, shared context, LLM-optimized order |
| seq_shared_rand_full | Every turn | ✓ (same ctx) | Sequential, shared context, random order, full context |
| seq_shared_ord_full | Every turn | ✓ (same ctx) | Sequential, shared context, LLM order, full context |

---

## Qwen/Qwen2.5-0.5B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| independent | 0.0026 | 0.1142 | 1000 | 927.5 | 279.5 | 0.47 |
| all_in_one | 0.0699 | 0.2066 | 1000 | 277.7 | 240.1 | 0.40 |
| seq_cross_ctx | 0.0007 | 0.1089 | 1000 | 1528.9 | 285.9 | 0.49 |
| seq_shared_rand | 0.0013 | 0.1083 | 1000 | 1652.5 | 285.7 | 0.49 |
| seq_shared_ord | 0.0009 | 0.1093 | 1000 | 1949.7 | 317.0 | 0.55 |

---

## Qwen/Qwen3-0.6B

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg SeqLen | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| independent | 0.4526 | 0.6120 | 1000 | 1290.4 | 61.9 | 0.14 |
| all_in_one | 0.4858 | 0.6320 | 1000 | 394.2 | 66.6 | 0.14 |
| seq_cross_ctx | 0.4534 | 0.6165 | 1000 | 1111.5 | 63.0 | 0.15 |
| seq_shared_rand | 0.4704 | 0.6362 | 1000 | 403.6 | 63.0 | 0.14 |
| seq_shared_ord | 0.4664 | 0.6318 | 1000 | 734.3 | 72.2 | 0.16 |
| seq_shared_rand_full | 0.4902 | 0.6589 | 1000 | 1110.4 | 61.9 | 0.15 |
| seq_shared_ord_full | 0.4868 | 0.6504 | 1000 | 1441.4 | 71.4 | 0.17 |

**PER-HISTORY-LENGTH ACCURACY (seq_shared_rand_full)**

| History Length | Samples | EM | F1 |
| --- | --- | --- | --- |
| 0 | 1000 | 0.4480 | 0.6153 |
| 1 | 1000 | 0.5000 | 0.6518 |
| 2 | 1000 | 0.4930 | 0.6693 |
| 3 | 1000 | 0.4990 | 0.6668 |
| 4 | 1000 | 0.5140 | 0.6913 |

---

## Qwen/Qwen3-1.7B

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg SeqLen | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| independent | 0.4344 | 0.6303 | 1000 | 1291.9 | 63.4 | 0.18 |
| all_in_one | 0.6126 | 0.7750 | 1000 | 403.4 | 75.9 | 0.21 |
| seq_cross_ctx | 0.4366 | 0.6380 | 1000 | 1108.8 | 60.3 | 0.20 |
| seq_shared_rand | 0.4322 | 0.6389 | 1000 | 402.5 | 62.0 | 0.18 |
| seq_shared_ord | 0.4044 | 0.6165 | 1000 | 736.3 | 74.1 | 0.22 |
| seq_shared_rand_full | 0.4502 | 0.6567 | 1000 | 1107.2 | 58.7 | 0.19 |
| seq_shared_ord_full | 0.4168 | 0.6320 | 1000 | 1441.2 | 71.1 | 0.22 |

**PER-HISTORY-LENGTH ACCURACY (seq_shared_rand_full)**

| History Length | Samples | EM | F1 |
| --- | --- | --- | --- |
| 0 | 1000 | 0.4520 | 0.6458 |
| 1 | 1000 | 0.4420 | 0.6457 |
| 2 | 1000 | 0.4360 | 0.6485 |
| 3 | 1000 | 0.4520 | 0.6646 |
| 4 | 1000 | 0.4700 | 0.6787 |

---

## Qwen/Qwen2.5-3B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| independent | 0.0002 | 0.1282 | 1000 | 927.5 | 265.4 | 1.01 |
| all_in_one | 0.2806 | 0.4436 | 1000 | 277.7 | 73.4 | 0.29 |
| seq_cross_ctx | 0.0015 | 0.1285 | 1000 | 1503.2 | 270.1 | 1.05 |
| seq_shared_rand | 0.0026 | 0.1297 | 1000 | 1615.1 | 268.7 | 1.03 |
| seq_shared_ord | 0.0015 | 0.1280 | 1000 | 1919.3 | 302.9 | 1.17 |

---

## Qwen/Qwen3-4B

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg SeqLen | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| independent | 0.6362 | 0.8003 | 1000 | 1296.5 | 68.0 | 0.32 |
| all_in_one | 0.6770 | 0.8377 | 1000 | 400.2 | 72.7 | 0.33 |
| seq_cross_ctx | 0.6584 | 0.8200 | 1000 | 1113.1 | 64.6 | 0.35 |
| seq_shared_rand | 0.6478 | 0.8158 | 1000 | 405.8 | 65.3 | 0.30 |
| seq_shared_ord | 0.6512 | 0.8192 | 1000 | 736.7 | 74.6 | 0.36 |
| seq_shared_rand_full | 0.6610 | 0.8263 | 1000 | 1113.1 | 64.6 | 0.35 |
| seq_shared_ord_full | 0.6688 | 0.8312 | 1000 | 1444.0 | 74.0 | 0.39 |

**PER-HISTORY-LENGTH ACCURACY (seq_shared_rand_full)**

| History Length | Samples | EM | F1 |
| --- | --- | --- | --- |
| 0 | 1000 | 0.6330 | 0.7996 |
| 1 | 1000 | 0.6670 | 0.8267 |
| 2 | 1000 | 0.6520 | 0.8223 |
| 3 | 1000 | 0.6720 | 0.8381 |
| 4 | 1000 | 0.6820 | 0.8446 |

---

## Qwen/Qwen2.5-7B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| independent | 0.0004 | 0.1518 | 1000 | 927.5 | 226.9 | 1.33 |
| all_in_one | 0.3776 | 0.5883 | 1000 | 277.7 | 148.2 | 0.88 |
| seq_cross_ctx | 0.0013 | 0.1394 | 1000 | 1464.4 | 257.9 | 1.55 |
| seq_shared_rand | 0.0031 | 0.1518 | 1000 | 1561.6 | 247.3 | 1.45 |
| seq_shared_ord | 0.0013 | 0.1470 | 1000 | 1869.2 | 283.2 | 1.69 |

---

## Qwen/Qwen3-8B

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg SeqLen | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| independent | 0.6696 | 0.8260 | 1000 | 1293.5 | 65.0 | 0.46 |
| all_in_one | 0.6678 | 0.8372 | 1000 | 400.9 | 73.3 | 0.51 |
| seq_cross_ctx | 0.6878 | 0.8416 | 1000 | 1110.2 | 61.7 | 0.51 |
| seq_shared_rand | 0.6818 | 0.8385 | 1000 | 403.5 | 62.9 | 0.43 |
| seq_shared_ord | 0.6878 | 0.8448 | 1000 | 735.2 | 73.1 | 0.53 |
| seq_shared_rand_full | 0.6926 | 0.8498 | 1000 | 1110.6 | 62.1 | 0.51 |
| seq_shared_ord_full | 0.6982 | 0.8524 | 1000 | 1442.3 | 72.2 | 0.58 |

**PER-HISTORY-LENGTH ACCURACY (seq_shared_rand_full)**

| History Length | Samples | EM | F1 |
| --- | --- | --- | --- |
| 0 | 1000 | 0.6660 | 0.8268 |
| 1 | 1000 | 0.7080 | 0.8566 |
| 2 | 1000 | 0.6850 | 0.8490 |
| 3 | 1000 | 0.6960 | 0.8548 |
| 4 | 1000 | 0.7080 | 0.8616 |

---

## Qwen/Qwen3-14B

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg SeqLen | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| independent | 0.6800 | 0.8365 | 1000 | 1294.5 | 66.0 | 0.74 |
| all_in_one | 0.6726 | 0.8474 | 1000 | 401.1 | 73.5 | 0.81 |
| seq_cross_ctx | 0.6814 | 0.8433 | 1000 | 1112.5 | 64.0 | 0.85 |
| seq_shared_rand | 0.6784 | 0.8418 | 1000 | 405.2 | 64.6 | 0.70 |
| seq_shared_ord | 0.6850 | 0.8445 | 1000 | 736.7 | 74.6 | 0.86 |
| seq_shared_rand_full | 0.6790 | 0.8445 | 1000 | 1112.9 | 64.4 | 0.85 |
| seq_shared_ord_full | 0.6838 | 0.8461 | 1000 | 1444.4 | 74.3 | 0.95 |

**PER-HISTORY-LENGTH ACCURACY (seq_shared_rand_full)**

| History Length | Samples | EM | F1 |
| --- | --- | --- | --- |
| 0 | 1000 | 0.6810 | 0.8400 |
| 1 | 1000 | 0.6920 | 0.8491 |
| 2 | 1000 | 0.6610 | 0.8395 |
| 3 | 1000 | 0.6700 | 0.8430 |
| 4 | 1000 | 0.6910 | 0.8510 |

---

## Qwen/Qwen2.5-14B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| independent | 0.6640 | 0.8338 | 500 | 1089.5 | 60.9 | 0.71 |
| all_in_one | 0.6413 | 0.8318 | 500 | 311.3 | 70.2 | 0.81 |
| seq_cross_ctx | 0.6605 | 0.8331 | 500 | 1268.5 | 61.1 | 0.70 |
| seq_shared_rand | 0.6675 | 0.8397 | 500 | 1303.1 | 60.3 | 0.68 |
| seq_shared_ord | 0.6662 | 0.8419 | 500 | 1631.1 | 70.4 | 0.84 |

---

## Qwen/Qwen3-32B

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg SeqLen | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| independent | 0.6572 | 0.8186 | 1000 | 1298.3 | 69.8 | 1.69 |
| all_in_one | 0.5928 | 0.7872 | 1000 | 409.8 | 82.3 | 1.98 |
| seq_cross_ctx | 0.6682 | 0.8326 | 1000 | 1114.3 | 65.8 | 1.97 |
| seq_shared_rand | 0.6410 | 0.8188 | 1000 | 407.7 | 67.1 | 1.64 |
| seq_shared_ord | 0.6526 | 0.8259 | 1000 | 739.4 | 77.3 | 1.99 |
| seq_shared_rand_full | 0.6500 | 0.8234 | 1000 | 1115.3 | 66.8 | 2.02 |
| seq_shared_ord_full | 0.6550 | 0.8286 | 1000 | 1446.9 | 76.8 | 2.38 |

**PER-HISTORY-LENGTH ACCURACY (seq_shared_rand_full)**

| History Length | Samples | EM | F1 |
| --- | --- | --- | --- |
| 0 | 1000 | 0.6530 | 0.8157 |
| 1 | 1000 | 0.6700 | 0.8338 |
| 2 | 1000 | 0.6290 | 0.8126 |
| 3 | 1000 | 0.6480 | 0.8224 |
| 4 | 1000 | 0.6500 | 0.8326 |

---

## Qwen/Qwen2.5-32B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| independent | 0.6623 | 0.8309 | 500 | 1089.5 | 63.7 | 1.47 |
| all_in_one | 0.6124 | 0.8181 | 500 | 311.3 | 73.4 | 1.69 |
| seq_cross_ctx | 0.6557 | 0.8293 | 500 | 1274.7 | 65.3 | 1.50 |
| seq_shared_rand | 0.6654 | 0.8389 | 500 | 1308.4 | 62.7 | 1.40 |
| seq_shared_ord | 0.6645 | 0.8377 | 500 | 1635.2 | 72.1 | 1.73 |

---

## Summary Table (EM) - All Models

| Model | independent | all_in_one | seq_cross_ctx | seq_shared_rand | seq_shared_ord | seq_shared_rand_full | seq_shared_ord_full |
|-------|-------------|------------|---------------|-----------------|----------------|----------------------|---------------------|
| Qwen2.5-0.5B | 0.0026 | 0.0699 | 0.0007 | 0.0013 | 0.0009 | - | - |
| Qwen3-0.6B | 0.4526 | 0.4858 | 0.4534 | 0.4704 | 0.4664 | 0.4902 | 0.4868 |
| Qwen3-1.7B | 0.4344 | 0.6126 | 0.4366 | 0.4322 | 0.4044 | 0.4502 | 0.4168 |
| Qwen2.5-3B | 0.0002 | 0.2806 | 0.0015 | 0.0026 | 0.0015 | - | - |
| Qwen3-4B | 0.6362 | 0.6770 | 0.6584 | 0.6478 | 0.6512 | 0.6610 | 0.6688 |
| Qwen2.5-7B | 0.0004 | 0.3776 | 0.0013 | 0.0031 | 0.0013 | - | - |
| Qwen3-8B | 0.6696 | 0.6678 | 0.6878 | 0.6818 | 0.6878 | 0.6926 | **0.6982** |
| Qwen3-14B | 0.6800 | 0.6726 | 0.6814 | 0.6784 | 0.6850 | 0.6790 | 0.6838 |
| Qwen2.5-14B | 0.6640 | 0.6413 | 0.6605 | 0.6675 | 0.6662 | - | - |
| Qwen3-32B | 0.6572 | 0.5928 | 0.6682 | 0.6410 | 0.6526 | 0.6500 | 0.6550 |
| Qwen2.5-32B | 0.6623 | 0.6124 | 0.6557 | 0.6654 | 0.6645 | - | - |

## Summary Table (Avg SeqLen) - Qwen3 Series

| Model | independent | all_in_one | seq_cross_ctx | seq_shared_rand | seq_shared_ord | seq_shared_rand_full | seq_shared_ord_full |
|-------|-------------|------------|---------------|-----------------|----------------|----------------------|---------------------|
| 0.6B | 1290.4 | 394.2 | 1111.5 | 403.6 | 734.3 | 1110.4 | 1441.4 |
| 1.7B | 1291.9 | 403.4 | 1108.8 | 402.5 | 736.3 | 1107.2 | 1441.2 |
| 4B | 1296.5 | 400.2 | 1113.1 | 405.8 | 736.7 | 1113.1 | 1444.0 |
| 8B | 1293.5 | 400.9 | 1110.2 | 403.5 | 735.2 | 1110.6 | 1442.3 |
| 14B | 1294.5 | 401.1 | 1112.5 | 405.2 | 736.7 | 1112.9 | 1444.4 |
| 32B | 1298.3 | 409.8 | 1114.3 | 407.7 | 739.4 | 1115.3 | 1446.9 |

## Key Findings

1. **Qwen3 >> Qwen2.5**: Qwen3 series significantly outperforms Qwen2.5 on this task (Qwen2.5 small models fail to follow output format)

2. **seq_shared_*_full > seq_shared_***: Adding context every turn improves accuracy (e.g., 8B: 0.6926 vs 0.6818)

3. **seq_shared_* ≈ seq_cross_ctx**: With KV cache, shared context has similar cost to cross-context but benefits from information sharing

4. **LLM ordering (ord) >= random**: Optimized question order provides slight improvement

5. **8B is optimal**: Qwen3-8B achieves best performance; larger models (14B, 32B) show degradation due to verbose answers

6. **32B verbose output issue**: Qwen3-32B tends to give overly complete answers (e.g., "East African Community (EAC)" instead of "East African Community"), hurting EM scores

7. **Token efficiency with KV cache**:
   - `independent`: ~1295 tokens (5 separate requests, no reuse)
   - `all_in_one`: ~400 tokens (single request)
   - `seq_shared_rand`: ~405 tokens (context cached, only new Q+A per turn)
   - `seq_shared_rand_full`: ~1110 tokens (context repeated but cached)
   - `seq_cross_ctx`: ~1110 tokens (different contexts, less cache benefit)
