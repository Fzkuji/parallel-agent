#!/usr/bin/env python3
"""
Analyze SQuAD dataset question distribution.

Shows statistics about how many questions each context has.
"""

import sys
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.squad import load_squad_groups


def analyze_distribution(split="validation", max_contexts=None):
    """Analyze question count distribution in SQuAD dataset."""
    print(f"\nAnalyzing SQuAD {split} split...")

    # Load dataset without filtering
    groups = load_squad_groups(
        split=split,
        max_contexts=max_contexts,
        min_questions=1,  # Accept all
        max_questions=None,  # No upper limit
        seed=42
    )

    # Count questions per context
    question_counts = []
    for group in groups:
        # Extract questions from group
        if isinstance(group, dict):
            if 'questions' in group:
                num_q = len(group['questions'])
            elif 'items' in group:
                num_q = len(group['items'])
            else:
                # Try to count from context structure
                num_q = sum(1 for k in group.keys() if k.startswith('q'))
                if num_q == 0:
                    num_q = 1  # At least one question
        else:
            num_q = 1

        question_counts.append(num_q)

    # Statistics
    print(f"\n=== Statistics ===")
    print(f"Total contexts: {len(question_counts)}")
    print(f"Total questions: {sum(question_counts)}")
    print(f"Average questions/context: {np.mean(question_counts):.2f}")
    print(f"Median questions/context: {np.median(question_counts):.0f}")
    print(f"Std dev: {np.std(question_counts):.2f}")
    print(f"Min questions/context: {min(question_counts)}")
    print(f"Max questions/context: {max(question_counts)}")

    # Distribution
    count_dist = Counter(question_counts)
    print(f"\n=== Distribution ===")
    print(f"{'Questions':<12} | {'Contexts':<10} | {'Percentage':<12} | {'Cumulative %'}")
    print("-" * 60)

    cumulative = 0
    for q_count in sorted(count_dist.keys()):
        num_contexts = count_dist[q_count]
        percentage = num_contexts / len(question_counts) * 100
        cumulative += percentage
        print(f"{q_count:<12} | {num_contexts:<10} | {percentage:>10.2f}% | {cumulative:>10.2f}%")

    # Contexts available for different min_questions settings
    print(f"\n=== Contexts Available for Different Question Counts ===")
    for threshold in [1, 4, 8, 12, 16, 20]:
        available = sum(1 for c in question_counts if c >= threshold)
        percentage = available / len(question_counts) * 100
        print(f">= {threshold:2d} questions: {available:4d} contexts ({percentage:5.1f}%)")

    # Plot distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram
    ax1.hist(question_counts, bins=range(1, max(question_counts)+2), alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Questions per Context', fontsize=12)
    ax1.set_ylabel('Number of Contexts', fontsize=12)
    ax1.set_title(f'SQuAD {split.capitalize()} - Question Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Cumulative
    sorted_counts = sorted(question_counts)
    cumulative_pct = np.arange(1, len(sorted_counts)+1) / len(sorted_counts) * 100
    ax2.plot(sorted_counts, cumulative_pct, linewidth=2)
    ax2.set_xlabel('Questions per Context', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_title(f'Cumulative Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add threshold lines
    for threshold in [1, 4, 8, 12, 16, 20]:
        available_pct = sum(1 for c in question_counts if c >= threshold) / len(question_counts) * 100
        if threshold <= max(question_counts):
            ax2.axvline(x=threshold, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax2.text(threshold, 5, f'{threshold}Q', rotation=90, fontsize=9, color='red')

    plt.tight_layout()
    plot_file = f"results/squad_{split}_distribution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {plot_file}")


if __name__ == "__main__":
    # Analyze validation split
    analyze_distribution("validation", max_contexts=None)

    # Also analyze train split
    print("\n" + "="*80 + "\n")
    analyze_distribution("train", max_contexts=1000)
