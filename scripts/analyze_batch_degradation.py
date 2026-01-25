#!/usr/bin/env python3
"""
Analyze why batch strategy performance degrades from 1Q to 5Q.

If batch is truly independent, then:
- Questions that were correct in 1Q should remain correct in 5Q
- Performance drop should only come from Q2-Q5 being harder
"""

import json
import sys
from pathlib import Path

results_dir = Path("outputs/controlled_q_count")

# Load results
with open(results_dir / "pretrained_fixed_q1.json", 'r') as f:
    data_1q = json.load(f)

with open(results_dir / "pretrained_fixed_q5.json", 'r') as f:
    data_5q = json.load(f)

batch_1q = data_1q['strategies']['batch']
batch_5q = data_5q['strategies']['batch']

print("="*80)
print("BATCH STRATEGY DEGRADATION ANALYSIS")
print("="*80)

print(f"\nOverall EM:")
print(f"  1Q: {batch_1q['aggregate_metrics']['strict_acc']:.1%} (9 questions)")
print(f"  5Q: {batch_5q['aggregate_metrics']['strict_acc']:.1%} (45 questions)")

# Analyze per-context
print(f"\n{'Context':<10} | {'1Q EM':<8} | {'5Q EM':<8} | {'Status':<20}")
print("-"*60)

contexts_1q = batch_1q['contexts']
contexts_5q = batch_5q['contexts']

degraded_contexts = []

for i in range(len(contexts_1q)):
    em_1q = contexts_1q[i]['metrics']['strict_acc']
    em_5q = contexts_5q[i]['metrics']['strict_acc']

    if em_1q > em_5q:
        status = "⚠️ DEGRADED"
        degraded_contexts.append(i)
    elif em_1q < em_5q:
        status = "✓ Improved"
    else:
        status = "= Same"

    print(f"{i:<10} | {em_1q:<8.1%} | {em_5q:<8.1%} | {status}")

print(f"\nDegraded contexts: {len(degraded_contexts)} out of {len(contexts_1q)}")

if degraded_contexts:
    print("\n" + "="*80)
    print("CRITICAL ISSUE DETECTED!")
    print("="*80)
    print("\nSome contexts that were 100% correct in 1Q have lower accuracy in 5Q.")
    print("This suggests batch inference is NOT truly independent when processing")
    print("multiple questions from the same context together.")
    print("\nPossible causes:")
    print("1. Padding/attention interference when batching multiple questions")
    print("2. Context length affecting generation quality")
    print("3. Implementation bug in batch processing")

    # Calculate what EM should be if Q1 remained correct
    total_correct_5q = sum(ctx['metrics']['strict_acc'] * 5 for ctx in contexts_5q)
    total_q_5q = 45

    # If all Q1s remained correct (9 questions)
    # and current total correct is total_correct_5q
    # then Q2-Q5 correct = total_correct_5q - 9 (if Q1s were correct)
    # But some Q1s might have become incorrect

    print(f"\nRecommendation:")
    print("Re-run test_batch_independence.py with 5 questions from same context")
    print("to verify if batch processing multiple questions causes interference.")
else:
    print("\n" + "="*80)
    print("NO DEGRADATION DETECTED")
    print("="*80)
    print("\nAll contexts maintained or improved accuracy from 1Q to 5Q.")
    print("Performance drop is purely due to Q2-Q5 being harder questions.")
    print("Batch independence is verified.")

# Calculate theoretical performance
print("\n" + "="*80)
print("THEORETICAL ANALYSIS")
print("="*80)

# If batch is independent and Q1s remain 100% correct
# Total correct in 5Q = 9 (from Q1s) + X (from Q2-Q5)
# EM_5q = (9 + X) / 45
# X = 45 * EM_5q - 9

total_correct_5q = 45 * batch_5q['aggregate_metrics']['strict_acc']
expected_q1_correct = 9  # Should be 9 if independent
implied_q2_to_q5_correct = total_correct_5q - expected_q1_correct

print(f"\nIf batch is independent:")
print(f"  Q1s correct (expected): 9 / 9 = 100%")
print(f"  Q2-Q5 correct (implied): {implied_q2_to_q5_correct:.1f} / 36 = {implied_q2_to_q5_correct/36:.1%}")

print(f"\nActual performance:")
print(f"  Total correct: {total_correct_5q:.1f} / 45 = {batch_5q['aggregate_metrics']['strict_acc']:.1%}")

# Check if degradation exists
if degraded_contexts:
    print(f"\n❌ But {len(degraded_contexts)} contexts show degradation!")
    print(f"   This contradicts the independence assumption.")
