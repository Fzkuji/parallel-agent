#!/usr/bin/env python3
"""Analyze question count distribution in SQuAD dataset."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict, Counter
from datasets import load_dataset

# Load SQuAD validation set
print("Loading SQuAD validation set...")
raw_dataset = load_dataset("squad", split="validation")

# Group by context
grouped = defaultdict(list)
for row in raw_dataset:
    grouped[row["context"]].append(row)

# Count questions per context
question_counts = [len(rows) for rows in grouped.values()]

# Distribution
distribution = Counter(question_counts)
print(f"\nTotal contexts: {len(grouped)}")
print(f"Total questions: {sum(question_counts)}")
print(f"\nQuestion count distribution:")
print(f"{'Questions':<12} {'Contexts':<12} {'Cumulative':<12}")
print("-" * 36)

cumulative = 0
for count in sorted(distribution.keys(), reverse=True):
    num_contexts = distribution[count]
    cumulative += num_contexts
    print(f"{count:<12} {num_contexts:<12} {cumulative:<12}")

# Show how many contexts have at least N questions
print(f"\n{'Min Questions':<15} {'Contexts Available':<20} {'Total Questions':<15}")
print("-" * 50)
for min_q in [1, 5, 10, 12, 14, 16, 18, 20, 25, 30]:
    contexts_with_min = sum(1 for c in question_counts if c >= min_q)
    total_q = sum(min(c, min_q) for c in question_counts if c >= min_q) if min_q else 0
    print(f"{min_q:<15} {contexts_with_min:<20} {total_q:<15}")
