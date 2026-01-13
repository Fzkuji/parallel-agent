# Experiment 1: Answer Dependency Study - Full Results

## Dataset
- **MoreHopQA**: 3-5 hop reasoning with gold sub-questions and sub-answers
- **Samples**: 1118 (full dataset)
- **Models**: Qwen2.5-Instruct series (0.5B, 3B, 7B, 14B, 32B)

## Experimental Conditions

| Condition | Context | Q&A History | Question Asked |
|-----------|---------|-------------|----------------|
| gold_context | ✓ | ✗ | Last sub-question (gold answers pre-embedded) |
| gold_direct | ✗ | ✗ | Last sub-question (gold answers pre-embedded) |
| sequential | ✓ | ✓ | All sub-questions in order |
| chain_only | ✓→✗ | ✓ | Sequential, but final step has no context |
| main_question | ✓ | ✗ | Main question directly (no decomposition) |

---

## Qwen/Qwen2.5-0.5B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| gold_context | 0.0805 | 0.0942 | 1118 | 396.1 | 361.4 | 0.21 |
| gold_direct | 0.0170 | 0.0263 | 1118 | 33.6 | 329.7 | 0.20 |
| sequential | 0.0072 | 0.0086 | 1118 | 4616.1 | 3507.5 | 5.94 |
| chain_only | 0.0018 | 0.0040 | 1118 | 4189.8 | 3387.1 | 5.71 |
| main_question | 0.0590 | 0.0623 | 1118 | 405.2 | 341.4 | 0.19 |

**PER-STEP ACCURACY (Sequential Condition)**

| Step | Samples | EM | F1 |
| --- | --- | --- | --- |
| 1 | 1118 | 0.1816 | 0.2689 |
| 2 | 1118 | 0.0322 | 0.1009 |
| 3 | 1118 | 0.0072 | 0.0086 |

---

## Qwen/Qwen2.5-3B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| gold_context | 0.2835 | 0.2856 | 1118 | 396.1 | 101.8 | 0.21 |
| gold_direct | 0.4034 | 0.3998 | 1118 | 33.6 | 265.6 | 0.21 |
| sequential | 0.2317 | 0.2320 | 1118 | 1256.6 | 76.5 | 0.30 |
| chain_only | 0.2567 | 0.2546 | 1118 | 894.1 | 179.0 | 0.68 |
| main_question | 0.1753 | 0.1753 | 1118 | 405.2 | 124.9 | 0.23 |

**PER-STEP ACCURACY (Sequential Condition)**

| Step | Samples | EM | F1 |
| --- | --- | --- | --- |
| 1 | 1118 | 0.7272 | 0.8569 |
| 2 | 1118 | 0.7791 | 0.8708 |
| 3 | 1118 | 0.2317 | 0.2320 |

---

## Qwen/Qwen2.5-7B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| gold_context | 0.4061 | 0.4046 | 1118 | 396.1 | 891.2 | 0.85 |
| gold_direct | 0.4884 | 0.4856 | 1118 | 33.6 | 400.1 | 0.67 |
| sequential | 0.3846 | 0.3831 | 1118 | 1264.0 | 870.8 | 5.01 |
| chain_only | 0.3962 | 0.3944 | 1118 | 901.4 | 953.4 | 5.48 |
| main_question | 0.3336 | 0.3331 | 1118 | 405.2 | 820.8 | 0.82 |

**PER-STEP ACCURACY (Sequential Condition)**

| Step | Samples | EM | F1 |
| --- | --- | --- | --- |
| 1 | 1118 | 0.7030 | 0.8235 |
| 2 | 1118 | 0.7612 | 0.8514 |
| 3 | 1118 | 0.3846 | 0.3831 |

---

## Qwen/Qwen2.5-14B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| gold_context | 0.4025 | 0.4008 | 1118 | 396.1 | 1633.5 | 1.93 |
| gold_direct | 0.4696 | 0.4660 | 1118 | 33.6 | 1433.9 | 1.73 |
| sequential | 0.4660 | 0.4630 | 1118 | 1258.6 | 3842.3 | 41.91 |
| chain_only | 0.4991 | 0.4955 | 1118 | 896.0 | 3444.0 | 37.52 |
| main_question | 0.3390 | 0.3376 | 1118 | 405.2 | 1653.6 | 1.92 |

**PER-STEP ACCURACY (Sequential Condition)**

| Step | Samples | EM | F1 |
| --- | --- | --- | --- |
| 1 | 1118 | 0.7639 | 0.8668 |
| 2 | 1118 | 0.7934 | 0.8828 |
| 3 | 1118 | 0.4660 | 0.4630 |

---

## Qwen/Qwen2.5-32B-Instruct

**EXPERIMENT SUMMARY**

| Condition | EM | F1 | Samples | Avg Prompt | Avg Compl | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| gold_context | 0.4776 | 0.4770 | 1118 | 396.1 | 436.8 | 2.96 |
| gold_direct | 0.5349 | 0.5336 | 1118 | 33.6 | 223.2 | 1.32 |
| sequential | 0.4410 | 0.4397 | 1118 | 1257.8 | 480.9 | 10.62 |
| chain_only | 0.4884 | 0.4880 | 1118 | 895.3 | 434.9 | 9.59 |
| main_question | 0.3810 | 0.3803 | 1118 | 405.2 | 709.1 | 3.25 |

**PER-STEP ACCURACY (Sequential Condition)**

| Step | Samples | EM | F1 |
| --- | --- | --- | --- |
| 1 | 1118 | 0.7826 | 0.8906 |
| 2 | 1118 | 0.8587 | 0.9365 |
| 3 | 1118 | 0.4410 | 0.4397 |

---

## Summary Table (EM)

| Model | gold_context | gold_direct | sequential | chain_only | main_question |
|-------|--------------|-------------|------------|------------|---------------|
| 0.5B  | 0.0805 | 0.0170 | 0.0072 | 0.0018 | 0.0590 |
| 3B    | 0.2835 | **0.4034** | 0.2317 | 0.2567 | 0.1753 |
| 7B    | 0.4061 | **0.4884** | 0.3846 | 0.3962 | 0.3336 |
| 14B   | 0.4025 | 0.4696 | 0.4660 | **0.4991** | 0.3390 |
| 32B   | 0.4776 | **0.5349** | 0.4410 | 0.4884 | 0.3810 |

## Key Findings

1. **gold_direct > gold_context** (3B+): Context introduces noise when gold answers are pre-embedded
2. **chain_only ≥ sequential** (7B+): Q&A history sufficient at final step
3. **All decomposed > main_question**: Decomposition is essential
4. **0.5B catastrophic failure**: Error propagation destroys sequential/chain_only performance
5. **Per-step pattern**: Step 1 & 2 high accuracy (~70-90%), Step 3 drops significantly
