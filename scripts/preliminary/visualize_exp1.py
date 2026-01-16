#!/usr/bin/env python3
"""
Visualization script for Experiment 1: Answer Dependency Study on MoreHopQA

This script generates figures showing how different conditions perform across model sizes.
"""

import matplotlib.pyplot as plt
import numpy as np

# Results from Exp 1
models = ["0.5B", "3B", "7B", "14B", "32B"]
model_sizes = [0.5, 3, 7, 14, 32]  # For x-axis scaling

# Accuracy (EM %) for each condition - Updated 2026-01-16
results = {
    "gold_context": [8.05, 28.44, 40.61, 40.25, 47.76],
    "gold_direct": [1.70, 40.43, 48.75, 46.87, 53.58],
    "sequential": [0.72, 23.17, 38.46, 46.60, 44.10],
    "chain_only": [0.18, 25.67, 39.62, 49.91, 48.84],
    "main_question": [5.90, 17.53, 33.36, 33.90, 38.10],
}

# Color scheme
colors = {
    "gold_context": "#2ecc71",  # Green
    "gold_direct": "#27ae60",   # Dark green
    "sequential": "#3498db",    # Blue
    "chain_only": "#2980b9",    # Dark blue
    "main_question": "#e74c3c", # Red
}

# Markers
markers = {
    "gold_context": "o",
    "gold_direct": "s",
    "sequential": "^",
    "chain_only": "v",
    "main_question": "D",
}

# Labels for legend
labels = {
    "gold_context": "Gold + Context",
    "gold_direct": "Gold Direct (No Context)",
    "sequential": "Sequential",
    "chain_only": "Chain Only (No Context at Final)",
    "main_question": "Main Question Only",
}


def plot_line_chart():
    """Create line chart showing accuracy vs model size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for condition, accuracies in results.items():
        ax.plot(
            models, accuracies,
            marker=markers[condition],
            color=colors[condition],
            label=labels[condition],
            linewidth=2,
            markersize=8
        )

    ax.set_xlabel("Model Size", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Exp 1: Answer Dependency Study on MoreHopQA", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 60)

    plt.tight_layout()
    return fig


def plot_grouped_bar():
    """Create grouped bar chart comparing conditions across models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.15

    conditions = ["gold_context", "gold_direct", "sequential", "chain_only", "main_question"]

    for i, condition in enumerate(conditions):
        offset = (i - 2) * width
        bars = ax.bar(
            x + offset, results[condition],
            width,
            label=labels[condition],
            color=colors[condition]
        )

    ax.set_xlabel("Model Size", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Exp 1: Answer Dependency Study on MoreHopQA", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 60)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_context_comparison():
    """
    Create a focused comparison showing the impact of context.
    Compares: gold_context vs gold_direct, sequential vs chain_only
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(models))
    width = 0.35

    # Plot 1: gold_context vs gold_direct
    ax1.bar(x - width/2, results["gold_context"], width, label="With Context", color="#2ecc71")
    ax1.bar(x + width/2, results["gold_direct"], width, label="Without Context", color="#27ae60", hatch="//")

    ax1.set_xlabel("Model Size", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Gold Condition: Upper Bound Reference", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_ylim(0, 60)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add improvement annotations
    for i, (with_ctx, without_ctx) in enumerate(zip(results["gold_context"], results["gold_direct"])):
        diff = without_ctx - with_ctx
        ax1.annotate(f"+{diff:.1f}%", xy=(x[i], without_ctx + 2), ha='center', fontsize=9, color='green')

    # Plot 2: sequential vs chain_only
    ax2.bar(x - width/2, results["sequential"], width, label="Sequential (Full Context)", color="#3498db")
    ax2.bar(x + width/2, results["chain_only"], width, label="Chain Only (No Final Context)", color="#2980b9", hatch="//")

    ax2.set_xlabel("Model Size", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Sequential: Q&A History Sufficiency", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.set_ylim(0, 60)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add improvement annotations
    for i, (seq, chain) in enumerate(zip(results["sequential"], results["chain_only"])):
        diff = chain - seq
        color = 'green' if diff > 0 else 'red'
        sign = '+' if diff > 0 else ''
        ax2.annotate(f"{sign}{diff:.1f}%", xy=(x[i], max(seq, chain) + 2), ha='center', fontsize=9, color=color)

    plt.tight_layout()
    return fig


def plot_error_propagation():
    """Show error propagation effect in small models vs large models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compare gold_direct (upper bound) vs sequential (with error propagation)
    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, results["gold_direct"], width, label="Gold Direct (Upper Bound)", color="#27ae60")
    ax.bar(x, results["sequential"], width, label="Sequential", color="#3498db")
    ax.bar(x + width, results["main_question"], width, label="Main Question Only", color="#e74c3c")

    ax.set_xlabel("Model Size", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Error Propagation: Small Models Suffer Most", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, 60)
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight 0.5B catastrophic failure
    ax.annotate(
        "Catastrophic\nFailure (<1%)",
        xy=(0, 0.72), xytext=(0.5, 20),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10, color="red", ha='center'
    )

    plt.tight_layout()
    return fig


def save_all_figures(output_dir: str = None):
    """Generate and save all figures."""
    import os

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(output_dir, "figures")

    os.makedirs(output_dir, exist_ok=True)

    # Generate and save figures
    figures = {
        "exp1_line_chart.pdf": plot_line_chart(),
        "exp1_grouped_bar.pdf": plot_grouped_bar(),
        "exp1_context_comparison.pdf": plot_context_comparison(),
        "exp1_error_propagation.pdf": plot_error_propagation(),
    }

    for filename, fig in figures.items():
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        # Also save PNG for preview
        fig.savefig(filepath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')

    plt.close('all')
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    save_all_figures()
