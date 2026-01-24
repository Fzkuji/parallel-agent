#!/usr/bin/env python3
"""
Generate visualizations for question count impact study.

Creates plots showing:
1. Pretrained strategy performance vs question count
2. SFT training impact vs question count
3. Token efficiency comparison
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read results
results_file = Path("outputs/question_count_study/all_results.json")
with open(results_file, 'r') as f:
    data = json.load(f)

# Extract data for plotting
q_counts = sorted([int(k) for k in data.keys()])

# Pretrained EM scores
pretrained_em = {
    'all_in_one': [],
    'sequential': [],
    'batch': [],
    'collab_llm': []
}

# SFT EM scores
sft_batch_baseline = []
sft_batch_trained = []
sft_seq_baseline = []
sft_seq_trained = []

# Token efficiency
seq_tokens = []
batch_tokens = []
collab_tokens = []

for q in q_counts:
    q_str = str(q)

    # Pretrained
    if "pretrained" in data[q_str]:
        strategies = data[q_str]["pretrained"].get("strategies", {})
        for strategy in pretrained_em.keys():
            metrics = strategies.get(strategy, {}).get("aggregate_metrics", {})
            pretrained_em[strategy].append(metrics.get("strict_acc", 0))

        # Tokens
        seq_metrics = strategies.get("sequential", {}).get("aggregate_metrics", {})
        batch_metrics = strategies.get("batch", {}).get("aggregate_metrics", {})
        collab_metrics = strategies.get("collab_llm", {}).get("aggregate_metrics", {})
        seq_tokens.append(seq_metrics.get("avg_prompt_tokens", 0))
        batch_tokens.append(batch_metrics.get("avg_prompt_tokens", 0))
        collab_tokens.append(collab_metrics.get("avg_prompt_tokens", 0))

    # SFT
    if "sft" in data[q_str]:
        batch_data = data[q_str]["sft"].get("batch", {})
        seq_data = data[q_str]["sft"].get("sequential", {})

        sft_batch_baseline.append(batch_data.get("baseline", {}).get("strict_acc", 0))
        sft_batch_trained.append(batch_data.get("sft_lora", {}).get("strict_acc", 0))
        sft_seq_baseline.append(seq_data.get("baseline", {}).get("strict_acc", 0))
        sft_seq_trained.append(seq_data.get("sft_lora", {}).get("strict_acc", 0))

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Question Count Impact Study - SQuAD Dataset', fontsize=16, fontweight='bold')

# Plot 1: Pretrained Strategy Performance
ax1 = axes[0, 0]
ax1.plot(q_counts, pretrained_em['all_in_one'], 'o-', label='all_in_one', linewidth=2, markersize=8)
ax1.plot(q_counts, pretrained_em['sequential'], 's-', label='sequential', linewidth=2, markersize=8)
ax1.plot(q_counts, pretrained_em['batch'], '^-', label='batch', linewidth=2, markersize=8)
ax1.plot(q_counts, pretrained_em['collab_llm'], 'd-', label='collab_llm', linewidth=2, markersize=8)
ax1.set_xlabel('Questions per Context', fontsize=12)
ax1.set_ylabel('EM (Exact Match)', fontsize=12)
ax1.set_title('Pretrained Strategy Performance', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(q_counts)

# Plot 2: SFT Training Impact
ax2 = axes[0, 1]
width = 0.35
x = np.arange(len(q_counts))
batch_delta = [t - b for t, b in zip(sft_batch_trained, sft_batch_baseline)]
seq_delta = [t - b for t, b in zip(sft_seq_trained, sft_seq_baseline)]

bars1 = ax2.bar(x - width/2, batch_delta, width, label='Batch SFT Δ', alpha=0.8)
bars2 = ax2.bar(x + width/2, seq_delta, width, label='Sequential SFT Δ', alpha=0.8)

ax2.set_xlabel('Questions per Context', fontsize=12)
ax2.set_ylabel('EM Improvement (Δ)', fontsize=12)
ax2.set_title('SFT Training Impact', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(q_counts)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 0.005:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9)

# Plot 3: Token Efficiency
ax3 = axes[1, 0]
ax3.plot(q_counts, seq_tokens, 's-', label='Sequential', linewidth=2, markersize=8)
ax3.plot(q_counts, batch_tokens, '^-', label='Batch', linewidth=2, markersize=8)
ax3.plot(q_counts, collab_tokens, 'd-', label='Collab_LLM', linewidth=2, markersize=8)
ax3.set_xlabel('Questions per Context', fontsize=12)
ax3.set_ylabel('Avg PromptTok (Deduplicated)', fontsize=12)
ax3.set_title('Token Efficiency Comparison', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(q_counts)

# Plot 4: SFT Before vs After
ax4 = axes[1, 1]
ax4.plot(q_counts, sft_batch_baseline, 'o--', label='Batch Baseline', linewidth=2, markersize=8, alpha=0.6)
ax4.plot(q_counts, sft_batch_trained, 'o-', label='Batch SFT', linewidth=2, markersize=8)
ax4.plot(q_counts, sft_seq_baseline, 's--', label='Sequential Baseline', linewidth=2, markersize=8, alpha=0.6)
ax4.plot(q_counts, sft_seq_trained, 's-', label='Sequential SFT', linewidth=2, markersize=8)
ax4.set_xlabel('Questions per Context', fontsize=12)
ax4.set_ylabel('EM (Exact Match)', fontsize=12)
ax4.set_title('SFT Training: Before vs After', fontsize=14, fontweight='bold')
ax4.legend(loc='best', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(q_counts)

plt.tight_layout()
plt.savefig('results/question_count_study_plots.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to results/question_count_study_plots.png")

# Also create a high-level summary plot
fig2, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot all strategies on one chart
ax.plot(q_counts, pretrained_em['all_in_one'], 'o-', label='all_in_one (pretrained)', linewidth=2.5, markersize=10)
ax.plot(q_counts, pretrained_em['sequential'], 's-', label='sequential (pretrained)', linewidth=2.5, markersize=10)
ax.plot(q_counts, pretrained_em['batch'], '^-', label='batch (pretrained)', linewidth=2.5, markersize=10)
ax.plot(q_counts, pretrained_em['collab_llm'], 'd-', label='collab_llm (pretrained)', linewidth=2.5, markersize=10)
ax.plot(q_counts, sft_batch_trained, '^--', label='batch (SFT)', linewidth=2.5, markersize=10, alpha=0.7)
ax.plot(q_counts, sft_seq_trained, 's--', label='sequential (SFT)', linewidth=2.5, markersize=10, alpha=0.7)

ax.set_xlabel('Questions per Context', fontsize=14, fontweight='bold')
ax.set_ylabel('EM (Exact Match)', fontsize=14, fontweight='bold')
ax.set_title('Strategy Performance vs Question Count\nSQuAD Dataset, 100 Contexts, Qwen2.5-7B',
             fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=11, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xticks(q_counts)
ax.set_ylim([0, 1.0])

# Add annotation for all_in_one failure
if len(q_counts) >= 3 and pretrained_em['all_in_one'][2] < 0.6:
    ax.annotate('all_in_one\nCOLLAPSE!',
                xy=(q_counts[2], pretrained_em['all_in_one'][2]),
                xytext=(q_counts[2]-2, 0.4),
                fontsize=11, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.savefig('results/question_count_study_summary.png', dpi=300, bbox_inches='tight')
print(f"Summary plot saved to results/question_count_study_summary.png")

print("\nDone! Generated 2 visualization files:")
print("  - results/question_count_study_plots.png (4-panel detailed view)")
print("  - results/question_count_study_summary.png (single overview chart)")
