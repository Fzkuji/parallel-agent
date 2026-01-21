# Experiment 2b: The Role of Semantic Similarity

## Overview

This experiment investigates whether **semantic similarity alone** (without explicit context overlap) affects cross-query interaction in sequential answering.

## Research Question

> Do questions from the same domain benefit from sequential answering more than questions from different domains?

## Dataset

- **Dataset**: MATH (mathematical problems)
- **Domains**: 7 (algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus)
- **Questions per domain**: 98
- **Total samples**: 686 per condition
- **Models**: Qwen3 series (0.6B, 1.7B, 4B, 8B, 14B, 32B)

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| **Batch** (independent) | Each question answered separately |
| **Seq. (Same Subject)** | 7 questions all from same domain, answered sequentially with history |
| **Seq. (Different Subjects)** | 7 questions from 7 different domains, answered sequentially with history |

## Results

### Accuracy by Model Size

| Model | Batch | Seq. (Same Subject) | Seq. (Different Subjects) |
|-------|-------|---------------------|---------------------------|
| 0.6B  | 39.65% | 39.07% | 37.76% |
| 1.7B  | 47.81% | 48.69% | 45.04% |
| 4B    | 51.60% | 51.60% | 53.50% |
| 8B    | 49.42% | 48.98% | 50.00% |
| 14B   | 52.77% | 56.27% | 54.96% |
| 32B   | 57.00% | 58.16% | 56.41% |

### ICL Benefit Analysis

| Model | Same Subject - Batch | Different Subjects - Batch | Same - Different |
|-------|---------------------|---------------------------|------------------|
| 0.6B  | -0.58% | -1.89% | +1.31% |
| 1.7B  | +0.88% | -2.77% | +3.65% |
| 4B    | ±0.00% | +1.90% | -1.90% |
| 8B    | -0.44% | +0.58% | -1.02% |
| 14B   | +3.50% | +2.19% | +1.31% |
| 32B   | +1.16% | -0.59% | +1.75% |

## Key Findings

### 1. Model Capacity Threshold

**Small models (≤4B)** cannot effectively exploit semantic similarity:
- 0.6B: Sequential answering hurts performance regardless of domain similarity
- 1.7B: Same-domain helps slightly (+0.88%), but cross-domain hurts significantly (-2.77%)
- 4B/8B: Mixed results, no clear pattern

**Large models (14B-32B)** benefit from sequential answering:
- 14B: Same-domain provides +3.50% improvement
- 32B: Same-domain provides +1.16% improvement
- Both show "Same > Different" pattern

### 2. Semantic Similarity Effect

For capable models (14B+):
- **Same Subject > Different Subjects**: Confirms that semantic similarity helps
- The gap between same-domain and cross-domain ranges from +1.31% to +3.65%

### 3. The "Capacity Threshold" Pattern

There appears to be a threshold around **14B parameters**:
- Below threshold: Models may be distracted by additional context
- Above threshold: Models can leverage semantic relationships

## Implications

1. **Cross-Batch Attention Design**: Should account for model capacity
   - For smaller models: May need to be more selective about what information to share
   - For larger models: Can benefit from broader information sharing

2. **Semantic Grouping**: Batching semantically similar questions together may improve performance for sufficiently capable models

3. **Practical Applications**:
   - Production systems using large models (14B+) should consider semantic clustering
   - Systems using smaller models may benefit more from independent processing

## Average Sequence Length

| Model | Batch | Seq. (Same) | Seq. (Different) |
|-------|-------|-------------|------------------|
| 0.6B  | 3545  | 5583        | 5597             |
| 1.7B  | 3860  | 5991        | 5988             |
| 4B    | 3877  | 5987        | 5991             |
| 8B    | 3931  | 6058        | 6062             |
| 14B   | 3843  | 5911        | 5910             |
| 32B   | 3766  | 5843        | 5826             |

Note: Sequential conditions have ~60% longer context due to including previous Q&A pairs.

## Latency (seconds per question)

| Model | Batch | Seq. (Same) | Seq. (Different) |
|-------|-------|-------------|------------------|
| 0.6B  | 0.73  | 0.71        | 0.72             |
| 1.7B  | 1.06  | 1.07        | 1.07             |
| 4B    | 1.77  | 1.77        | 1.78             |
| 8B    | 2.73  | 2.74        | 2.74             |
| 14B   | 4.21  | 4.04        | 4.05             |
| 32B   | 8.79  | 8.69        | 8.65             |

Latency is similar across conditions, suggesting the performance differences are due to information utilization rather than computational overhead.
