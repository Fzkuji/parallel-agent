#!/usr/bin/env python3
"""
Generate publication-ready figures for the CSA paper.

This script produces:
1. Main attention heatmap figure (for Section 4/5)
2. Performance comparison chart
3. Ablation study visualization
4. Case study illustration

Usage:
    python scripts/generate_paper_figures.py \
        --results-dir "outputs/grouping_study" \
        --output-dir "figures" \
        --checkpoint "outputs/checkpoints/squad/model.pt"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Configure matplotlib for publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'text.usetex': False,  # Set to True if LaTeX is available
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
})

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper figures")

    parser.add_argument("--results-dir", type=str, default="outputs/grouping_study",
                       help="Directory with evaluation results")
    parser.add_argument("--output-dir", type=str, default="figures",
                       help="Output directory for figures")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="CSA checkpoint for attention visualization")
    parser.add_argument("--format", type=str, default="pdf",
                       choices=["pdf", "png", "eps"],
                       help="Output format")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for raster formats")

    return parser.parse_args()


# Color palette for consistency
COLORS = {
    'independent': '#1f77b4',   # Blue
    'all_in_one': '#ff7f0e',    # Orange
    'sequential': '#2ca02c',    # Green
    'memory': '#9467bd',        # Purple
    'csa': '#d62728',           # Red
    'csa_trained': '#d62728',   # Red
}

STRATEGY_LABELS = {
    'batch': 'Independent',
    'all_in_one': 'All-in-One',
    'sequential': 'Sequential',
    'memory': 'Memory-Aug.',
    'cross_batch': 'CSA (Ours)',
}

MARKERS = {
    'batch': 'o',
    'all_in_one': 's',
    'sequential': '^',
    'memory': 'd',
    'cross_batch': '*',
}


def load_results(results_dir: Path) -> Dict:
    """Load evaluation results from JSON files."""
    results = {}

    # Try loading all_results.json first
    all_results_path = results_dir / "all_results.json"
    if all_results_path.exists():
        with open(all_results_path, 'r') as f:
            data = json.load(f)
            return data.get("results_by_group_size", {})

    # Otherwise, load individual group files
    for f in results_dir.glob("group_size_*.json"):
        try:
            group_size = int(f.stem.split('_')[-1])
            with open(f, 'r') as fp:
                results[group_size] = json.load(fp)
        except (ValueError, json.JSONDecodeError):
            continue

    return results


def plot_performance_vs_group_size(
    results: Dict,
    metric: str = "strict_acc",
    output_path: Optional[Path] = None,
    title: str = "Performance vs. Group Size",
    ylabel: str = "Exact Match (%)",
    figsize: Tuple[float, float] = (6, 4.5),
):
    """Plot strategy performance across different group sizes.

    Args:
        results: Dict mapping group_size -> strategy_name -> metrics
        metric: Metric to plot (e.g., "strict_acc", "f1", "acc")
        output_path: Path to save figure
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    group_sizes = sorted([int(k) for k in results.keys()])

    # Collect data for each strategy
    strategies_data = {}
    for gs in group_sizes:
        gs_data = results[str(gs)] if str(gs) in results else results.get(gs, {})
        for strategy, data in gs_data.items():
            if strategy not in strategies_data:
                strategies_data[strategy] = {'sizes': [], 'values': []}
            value = data.get('metrics', {}).get(metric, 0)
            # Convert to percentage
            if value <= 1:
                value *= 100
            strategies_data[strategy]['sizes'].append(gs)
            strategies_data[strategy]['values'].append(value)

    # Define plot order
    plot_order = ['batch', 'sequential', 'all_in_one', 'memory', 'cross_batch']

    for strategy in plot_order:
        if strategy not in strategies_data:
            continue

        data = strategies_data[strategy]
        color = COLORS.get(strategy, COLORS.get('csa'))
        label = STRATEGY_LABELS.get(strategy, strategy)
        marker = MARKERS.get(strategy, 'o')

        # Special styling for CSA
        linewidth = 2.5 if strategy == 'cross_batch' else 1.8
        markersize = 12 if strategy == 'cross_batch' else 8
        alpha = 1.0 if strategy == 'cross_batch' else 0.85

        ax.plot(data['sizes'], data['values'],
               marker=marker, color=color, label=label,
               linewidth=linewidth, markersize=markersize, alpha=alpha)

    ax.set_xlabel('Questions per Group (G)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    ax.set_xticks(group_sizes)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)

    # Set y-axis limits
    all_values = [v for d in strategies_data.values() for v in d['values']]
    if all_values:
        ymin = max(0, min(all_values) - 5)
        ymax = min(100, max(all_values) + 5)
        ax.set_ylim([ymin, ymax])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_latency_comparison(
    results: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (6, 4.5),
):
    """Plot latency comparison across strategies.

    Args:
        results: Dict mapping group_size -> strategy_name -> metrics
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    group_sizes = sorted([int(k) for k in results.keys()])

    # Collect latency data
    strategies_data = {}
    for gs in group_sizes:
        gs_data = results[str(gs)] if str(gs) in results else results.get(gs, {})
        for strategy, data in gs_data.items():
            if strategy not in strategies_data:
                strategies_data[strategy] = {'sizes': [], 'values': []}
            # Per-question latency
            latency = data.get('latency', 0)
            num_q = data.get('num_questions', 1)
            per_q_latency = latency / num_q if num_q > 0 else 0
            strategies_data[strategy]['sizes'].append(gs)
            strategies_data[strategy]['values'].append(per_q_latency)

    plot_order = ['batch', 'sequential', 'all_in_one', 'memory', 'cross_batch']

    for strategy in plot_order:
        if strategy not in strategies_data:
            continue

        data = strategies_data[strategy]
        color = COLORS.get(strategy, COLORS.get('csa'))
        label = STRATEGY_LABELS.get(strategy, strategy)
        marker = MARKERS.get(strategy, 'o')

        linewidth = 2.5 if strategy == 'cross_batch' else 1.8
        markersize = 12 if strategy == 'cross_batch' else 8

        ax.plot(data['sizes'], data['values'],
               marker=marker, color=color, label=label,
               linewidth=linewidth, markersize=markersize)

    ax.set_xlabel('Questions per Group (G)', fontsize=12)
    ax.set_ylabel('Latency per Question (s)', fontsize=12)
    ax.set_title('Inference Latency Comparison', fontsize=13, fontweight='bold')

    ax.set_xticks(group_sizes)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_token_efficiency(
    results: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (6, 4.5),
):
    """Plot token efficiency comparison.

    Args:
        results: Dict mapping group_size -> strategy_name -> metrics
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    group_sizes = sorted([int(k) for k in results.keys()])

    # Collect token data
    strategies_data = {}
    for gs in group_sizes:
        gs_data = results[str(gs)] if str(gs) in results else results.get(gs, {})
        for strategy, data in gs_data.items():
            if strategy not in strategies_data:
                strategies_data[strategy] = {'sizes': [], 'values': []}
            # Per-question tokens (deduplicated)
            tokens = data.get('total_prompt_tokens', 0)
            num_q = data.get('num_questions', 1)
            per_q_tokens = tokens / num_q if num_q > 0 else 0
            strategies_data[strategy]['sizes'].append(gs)
            strategies_data[strategy]['values'].append(per_q_tokens)

    plot_order = ['batch', 'sequential', 'all_in_one', 'memory', 'cross_batch']

    for strategy in plot_order:
        if strategy not in strategies_data:
            continue

        data = strategies_data[strategy]
        color = COLORS.get(strategy, COLORS.get('csa'))
        label = STRATEGY_LABELS.get(strategy, strategy)
        marker = MARKERS.get(strategy, 'o')

        linewidth = 2.5 if strategy == 'cross_batch' else 1.8
        markersize = 12 if strategy == 'cross_batch' else 8

        ax.plot(data['sizes'], data['values'],
               marker=marker, color=color, label=label,
               linewidth=linewidth, markersize=markersize)

    ax.set_xlabel('Questions per Group (G)', fontsize=12)
    ax.set_ylabel('Tokens per Question', fontsize=12)
    ax.set_title('Token Efficiency (Deduplicated)', fontsize=13, fontweight='bold')

    ax.set_xticks(group_sizes)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_combined_performance_latency(
    results: Dict,
    metric: str = "strict_acc",
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 4),
):
    """Plot performance and latency side by side.

    Args:
        results: Dict mapping group_size -> strategy_name -> metrics
        metric: Metric to plot
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    group_sizes = sorted([int(k) for k in results.keys()])

    # Collect data
    perf_data = {}
    latency_data = {}

    for gs in group_sizes:
        gs_data = results[str(gs)] if str(gs) in results else results.get(gs, {})
        for strategy, data in gs_data.items():
            if strategy not in perf_data:
                perf_data[strategy] = {'sizes': [], 'values': []}
                latency_data[strategy] = {'sizes': [], 'values': []}

            # Performance
            value = data.get('metrics', {}).get(metric, 0)
            if value <= 1:
                value *= 100
            perf_data[strategy]['sizes'].append(gs)
            perf_data[strategy]['values'].append(value)

            # Latency
            latency = data.get('latency', 0)
            num_q = data.get('num_questions', 1)
            per_q_latency = latency / num_q if num_q > 0 else 0
            latency_data[strategy]['sizes'].append(gs)
            latency_data[strategy]['values'].append(per_q_latency)

    plot_order = ['batch', 'sequential', 'all_in_one', 'memory', 'cross_batch']

    # Plot performance
    for strategy in plot_order:
        if strategy not in perf_data:
            continue
        data = perf_data[strategy]
        color = COLORS.get(strategy, COLORS.get('csa'))
        label = STRATEGY_LABELS.get(strategy, strategy)
        marker = MARKERS.get(strategy, 'o')
        linewidth = 2.5 if strategy == 'cross_batch' else 1.8
        markersize = 12 if strategy == 'cross_batch' else 8

        ax1.plot(data['sizes'], data['values'],
                marker=marker, color=color, label=label,
                linewidth=linewidth, markersize=markersize)

    ax1.set_xlabel('Questions per Group (G)', fontsize=12)
    ax1.set_ylabel('Exact Match (%)', fontsize=12)
    ax1.set_title('(a) Answer Quality', fontsize=13, fontweight='bold')
    ax1.set_xticks(group_sizes)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', framealpha=0.9, fontsize=9)

    # Plot latency
    for strategy in plot_order:
        if strategy not in latency_data:
            continue
        data = latency_data[strategy]
        color = COLORS.get(strategy, COLORS.get('csa'))
        label = STRATEGY_LABELS.get(strategy, strategy)
        marker = MARKERS.get(strategy, 'o')
        linewidth = 2.5 if strategy == 'cross_batch' else 1.8
        markersize = 12 if strategy == 'cross_batch' else 8

        ax2.plot(data['sizes'], data['values'],
                marker=marker, color=color, label=label,
                linewidth=linewidth, markersize=markersize)

    ax2.set_xlabel('Questions per Group (G)', fontsize=12)
    ax2.set_ylabel('Latency per Question (s)', fontsize=12)
    ax2.set_title('(b) Inference Efficiency', fontsize=13, fontweight='bold')
    ax2.set_xticks(group_sizes)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', framealpha=0.9, fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_attention_example(
    attention_matrix: np.ndarray,
    question_texts: List[str],
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (5.5, 5),
):
    """Plot a clean attention heatmap for the paper.

    Args:
        attention_matrix: [n, n] attention weights
        question_texts: Short question labels
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    n = len(attention_matrix)

    # Create colormap
    colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
    cmap = LinearSegmentedColormap.from_list("blues", colors, N=256)

    # Mask diagonal
    mask = np.eye(n, dtype=bool)
    masked_attn = np.ma.array(attention_matrix, mask=mask)

    im = ax.imshow(masked_attn, cmap=cmap, aspect='auto', vmin=0, vmax=max(0.5, attention_matrix.max()))

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Attention Weight', fontsize=11)

    # Labels
    labels = [f"$q_{{{i+1}}}$" for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    # Value annotations
    for i in range(n):
        for j in range(n):
            if i != j:
                value = attention_matrix[i, j]
                color = 'white' if value > 0.35 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color=color)

    # Mark diagonal
    for i in range(n):
        ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                    fill=True, facecolor='#f0f0f0',
                    edgecolor='#cccccc', linewidth=0.5))

    ax.set_xlabel('Source Question (Key)', fontsize=12)
    ax.set_ylabel('Target Question (Query)', fontsize=12)
    ax.set_title('Cross-Sequence Attention Weights', fontsize=13, fontweight='bold')

    # Add question text annotations on the right
    if question_texts:
        for i, text in enumerate(question_texts[:n]):
            # Truncate long questions
            short_text = text[:25] + "..." if len(text) > 25 else text
            ax.annotate(
                f"$q_{{{i+1}}}$: {short_text}",
                xy=(1.02, 1 - (i + 0.5) / n),
                xycoords='axes fraction',
                fontsize=8,
                va='center',
                ha='left',
            )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_bar_comparison(
    results: Dict,
    group_size: int = 5,
    metric: str = "strict_acc",
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (7, 4),
):
    """Plot bar chart comparing strategies at a fixed group size.

    Args:
        results: Dict mapping group_size -> strategy_name -> metrics
        group_size: Which group size to show
        metric: Metric to plot
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    gs_data = results.get(str(group_size), results.get(group_size, {}))

    strategies = ['batch', 'all_in_one', 'sequential', 'memory', 'cross_batch']
    values = []
    colors_list = []
    labels = []

    for strategy in strategies:
        if strategy in gs_data:
            value = gs_data[strategy].get('metrics', {}).get(metric, 0)
            if value <= 1:
                value *= 100
            values.append(value)
            colors_list.append(COLORS.get(strategy, '#888888'))
            labels.append(STRATEGY_LABELS.get(strategy, strategy))

    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors_list, alpha=0.85, edgecolor='white', linewidth=1.5)

    # Highlight CSA bar
    if 'cross_batch' in gs_data:
        idx = strategies.index('cross_batch')
        bars[idx].set_edgecolor(COLORS['csa'])
        bars[idx].set_linewidth(2.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Exact Match (%)', fontsize=12)
    ax.set_title(f'Strategy Comparison (G={group_size})', fontsize=13, fontweight='bold')

    ax.set_ylim([0, max(values) * 1.1])
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def generate_sample_attention_matrix(n: int = 5, seed: int = 42) -> np.ndarray:
    """Generate a sample attention matrix for demonstration.

    This creates a realistic-looking attention pattern where:
    - Similar questions (adjacent indices) have higher attention
    - Diagonal is masked (self-attention excluded)

    Args:
        n: Number of questions
        seed: Random seed

    Returns:
        [n, n] attention matrix
    """
    np.random.seed(seed)

    # Create base attention with some structure
    attn = np.random.rand(n, n) * 0.3

    # Add stronger connections between adjacent questions
    for i in range(n):
        for j in range(n):
            if abs(i - j) == 1:
                attn[i, j] += 0.3 + np.random.rand() * 0.2
            elif abs(i - j) == 2:
                attn[i, j] += 0.1 + np.random.rand() * 0.15

    # Zero out diagonal
    np.fill_diagonal(attn, 0)

    # Normalize rows (softmax-like, excluding diagonal)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        row_sum = attn[i, mask].sum()
        if row_sum > 0:
            attn[i, mask] /= row_sum

    return attn


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Output format: {args.format}")

    # Load results if available
    results_dir = Path(args.results_dir)
    results = {}

    if results_dir.exists():
        results = load_results(results_dir)
        print(f"Loaded results from {results_dir}")
        print(f"Available group sizes: {sorted(results.keys())}")
    else:
        print(f"No results directory found at {results_dir}")
        print("Will generate demonstration figures with sample data")

    # Generate figures
    fmt = args.format

    # 1. Performance vs Group Size
    if results:
        print("\n1. Generating performance comparison...")
        plot_performance_vs_group_size(
            results,
            metric="strict_acc",
            output_path=output_dir / f"performance_vs_groupsize.{fmt}",
            title="Answer Quality vs. Group Size (SQuAD)",
            ylabel="Exact Match (%)",
        )

        # 2. Latency comparison
        print("2. Generating latency comparison...")
        plot_latency_comparison(
            results,
            output_path=output_dir / f"latency_comparison.{fmt}",
        )

        # 3. Token efficiency
        print("3. Generating token efficiency...")
        plot_token_efficiency(
            results,
            output_path=output_dir / f"token_efficiency.{fmt}",
        )

        # 4. Combined figure
        print("4. Generating combined figure...")
        plot_combined_performance_latency(
            results,
            metric="strict_acc",
            output_path=output_dir / f"combined_perf_latency.{fmt}",
        )

        # 5. Bar chart at G=5
        if '5' in results or 5 in results:
            print("5. Generating bar comparison (G=5)...")
            plot_bar_comparison(
                results,
                group_size=5,
                metric="strict_acc",
                output_path=output_dir / f"bar_comparison_g5.{fmt}",
            )

    # 6. Sample attention heatmap (demonstration)
    print("6. Generating attention heatmap example...")
    sample_attn = generate_sample_attention_matrix(5, seed=42)
    sample_questions = [
        "What is the capital?",
        "When was it founded?",
        "Who is the leader?",
        "What is the population?",
        "What language is spoken?",
    ]
    plot_attention_example(
        sample_attn,
        sample_questions,
        output_path=output_dir / f"attention_example.{fmt}",
    )

    print(f"\nDone! Generated figures in {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob(f"*.{fmt}")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
