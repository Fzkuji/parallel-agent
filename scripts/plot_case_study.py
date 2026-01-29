#!/usr/bin/env python3
"""
Plot case study figure from JSON data.
Run locally after downloading case_study_data.json from server.

Usage:
    python plot_case_study.py [input.json] [output.png]
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ============ CONFIGURATION ============
# Modify these parameters to adjust the figure

# Figure size (width, height)
FIGSIZE = (14, 4)

# Width ratio [left Q&A, right heatmap]
WIDTH_RATIO = [1.6, 1]

# Space between subplots
WSPACE = 0.08

# Q&A text vertical position (0=bottom, 1=top)
TEXT_Y = 0.45

# Subplot label positions
LABEL_A_Y = -0.02
LABEL_B_Y = -0.22

# Font sizes
FONT_QA = 9
FONT_HEATMAP_LABEL = 12
FONT_HEATMAP_VALUE = 10
FONT_AXIS = 12
FONT_TITLE = 11

# Text truncation
MAX_Q_LEN = 70
MAX_A_LEN = 50
MAX_REF_LEN = 50

# ============ END CONFIGURATION ============


def plot_case_study(data, output_path="case_study.png"):
    """Plot case study figure."""

    questions = data['questions']
    answers = data['answers']
    attention_matrix = np.array(data['attention_matrix'])
    n = len(questions)

    # Create figure
    fig, (ax_qa, ax_attn) = plt.subplots(
        1, 2, figsize=FIGSIZE, facecolor='white',
        gridspec_kw={'width_ratios': WIDTH_RATIO, 'wspace': WSPACE}
    )

    # ===== (a) Q&A Text =====
    ax_qa.axis('off')

    text_lines = []
    for i, (q, a) in enumerate(zip(questions, answers)):
        refs = q.get('references', [])
        ref = refs[0] if refs else "N/A"

        q_text = q['question'][:MAX_Q_LEN] + "..." if len(q['question']) > MAX_Q_LEN else q['question']
        a_text = a.strip()[:MAX_A_LEN] + "..." if len(a.strip()) > MAX_A_LEN else a.strip()
        ref_text = ref[:MAX_REF_LEN] + "..." if len(ref) > MAX_REF_LEN else ref

        text_lines.append(f"$q_{{{i+1}}}$: {q_text}")
        text_lines.append(f"      Gen: {a_text}")
        text_lines.append(f"      Ref: {ref_text}")
        if i < n - 1:
            text_lines.append("")

    ax_qa.text(0.0, TEXT_Y, "\n".join(text_lines), transform=ax_qa.transAxes,
               fontsize=FONT_QA, va='center', fontfamily='monospace', linespacing=1.5)

    ax_qa.text(0.5, LABEL_A_Y, '(a) Questions and Answers', transform=ax_qa.transAxes,
               fontsize=FONT_TITLE, fontweight='bold', va='top', ha='center')

    # ===== (b) Attention Heatmap =====
    labels = [f"$q_{{{i+1}}}$" for i in range(n)]

    # Mask diagonal
    mask = np.eye(n, dtype=bool)
    masked_attn = np.ma.array(attention_matrix, mask=mask)

    # Colormap
    colors = ["#FFFFFF", "#E3F2FD", "#90CAF9", "#42A5F5", "#1976D2", "#0D47A1"]
    cmap = LinearSegmentedColormap.from_list("blue", colors, N=256)

    im = ax_attn.imshow(masked_attn, cmap=cmap, aspect='equal',
                        vmin=0, vmax=max(0.5, attention_matrix.max()))

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_attn, shrink=0.8, pad=0.02)
    cbar.set_label('Attention', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Ticks
    ax_attn.set_xticks(range(n))
    ax_attn.set_yticks(range(n))
    ax_attn.set_xticklabels(labels, fontsize=FONT_HEATMAP_LABEL)
    ax_attn.set_yticklabels(labels, fontsize=FONT_HEATMAP_LABEL)

    # Values
    for i in range(n):
        for j in range(n):
            if i != j:
                val = attention_matrix[i, j]
                color = 'white' if val > 0.35 else '#333333'
                ax_attn.text(j, i, f'{val:.2f}', ha='center', va='center',
                             fontsize=FONT_HEATMAP_VALUE, fontweight='bold', color=color)

    # Diagonal
    for i in range(n):
        ax_attn.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=True, facecolor='#F5F5F5', edgecolor='#BDBDBD', linewidth=0.5
        ))
        ax_attn.text(i, i, '—', ha='center', va='center', fontsize=11, color='#9E9E9E')

    ax_attn.set_xlabel('Source (Key)', fontsize=FONT_AXIS)
    ax_attn.set_ylabel('Target (Query)', fontsize=FONT_AXIS)

    ax_attn.text(0.5, LABEL_B_Y, '(b) CSA Attention Matrix', transform=ax_attn.transAxes,
                 fontsize=FONT_TITLE, fontweight='bold', va='top', ha='center')

    # Grid
    ax_attn.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_attn.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_attn.grid(which='minor', color='white', linewidth=1)
    ax_attn.tick_params(which='minor', size=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.show()


def main():
    # Parse args
    input_json = sys.argv[1] if len(sys.argv) > 1 else "case_study_data.json"
    output_png = sys.argv[2] if len(sys.argv) > 2 else "case_study.png"

    print(f"Input: {input_json}")
    print(f"Output: {output_png}")

    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    plot_case_study(data, output_png)


if __name__ == "__main__":
    main()
