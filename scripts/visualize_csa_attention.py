#!/usr/bin/env python3
"""
Visualize Cross-Sequence Attention (CSA) for trained models.

This script generates visualizations for:
1. Cross-batch attention heatmaps - showing information flow between questions
2. Gate weight distribution - showing learned gate values
3. Hidden state similarity matrix - showing semantic relationships
4. Case study examples - showing input/output with attention patterns

Usage:
    python scripts/visualize_csa_attention.py \
        --model "Qwen/Qwen2.5-7B-Instruct" \
        --checkpoint "outputs/checkpoints/squad/Qwen_Qwen2.5-7B-Instruct_attention_frozen_lora.pt" \
        --dataset squad \
        --num-examples 5 \
        --output-dir "visualizations"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CSA attention patterns")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name or path")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to CSA checkpoint")
    parser.add_argument("--dataset", type=str, default="squad",
                       choices=["squad", "hotpot", "cmb", "triviaqa"],
                       help="Dataset to use for examples")
    parser.add_argument("--num-examples", type=int, default=5,
                       help="Number of example groups to visualize")
    parser.add_argument("--output-dir", type=str, default="visualizations",
                       help="Output directory for figures")
    parser.add_argument("--group-size", type=int, default=5,
                       help="Number of questions per group")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--save-format", type=str, default="pdf",
                       choices=["pdf", "png", "svg"],
                       help="Output format for figures")

    return parser.parse_args()


class CSAVisualizer:
    """Visualizer for Cross-Sequence Attention patterns."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        device: str = "cuda:0",
    ):
        self.device = device
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path

        # Load model and tokenizer
        self._load_model()

        # Storage for captured attention weights
        self.attention_weights = []
        self.gate_values = []
        self.hidden_states = []

    def _load_model(self):
        """Load model, tokenizer, and CSA checkpoint."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from src.cross_batch import CrossBatchGenerator, CrossBatchAttention

        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Load checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", {})

        # Create CSA module
        hidden_size = self.model.config.hidden_size
        use_gate = config.get("use_gate", True)

        self.csa_module = CrossBatchAttention(
            hidden_size=hidden_size,
            use_gate=use_gate,
            num_heads=8,
        )
        self.csa_module.load_state_dict(checkpoint["cross_batch_module"])
        self.csa_module.to(self.device)
        self.csa_module.eval()

        # Create generator
        self.generator = CrossBatchGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            cross_batch_module=self.csa_module,
            mix_method=config.get("module_type", "attention"),
            mix_layer=config.get("mix_layer", -1),
            device=self.device,
        )

        print(f"CSA config: use_gate={use_gate}")

    def _register_hooks(self):
        """Register forward hooks to capture attention weights and gate values."""
        self.attention_weights = []
        self.gate_values = []
        self.hidden_states = []

        def attention_hook(module, input, output):
            """Capture attention weights from CSA module."""
            hidden_states = input[0]
            batch_size = hidden_states.size(0)

            if batch_size <= 1:
                return

            # Compute attention weights (similar to forward pass)
            q = module.q_proj(hidden_states).view(batch_size, module.num_heads, module.head_dim)
            k = module.k_proj(hidden_states).view(batch_size, module.num_heads, module.head_dim)

            q = q.permute(1, 0, 2)  # [num_heads, batch, head_dim]
            k = k.permute(1, 0, 2)

            # Attention weights: [num_heads, batch, batch]
            attn_weights = torch.bmm(q, k.transpose(1, 2)) / (module.head_dim ** 0.5 * module.temperature)

            # Mask out self (diagonal)
            eye_mask = torch.eye(batch_size, device=hidden_states.device, dtype=torch.bool)
            attn_weights = attn_weights.masked_fill(eye_mask.unsqueeze(0), float('-inf'))

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

            # Average across heads
            avg_attn = attn_weights.mean(dim=0).detach().cpu().numpy()
            self.attention_weights.append(avg_attn)

            # Capture gate values if using gate
            if module.use_gate:
                v = module.v_proj(hidden_states).view(batch_size, module.num_heads, module.head_dim)
                v = v.permute(1, 0, 2)

                cross_batch_output = torch.bmm(attn_weights.detach(), v)
                cross_batch_output = cross_batch_output.permute(1, 0, 2)
                cross_batch_output = cross_batch_output.reshape(batch_size, module.hidden_size)
                cross_batch_output = module.out_proj(cross_batch_output)

                ln_h = module.ln_h(hidden_states)
                ln_a = module.ln_a(cross_batch_output)
                gate_input = torch.cat([ln_h, ln_a, ln_h * ln_a], dim=-1)
                gate = module.gate_net(gate_input)

                self.gate_values.append(gate.detach().cpu().numpy())

            # Store hidden states for similarity analysis
            self.hidden_states.append(hidden_states.detach().cpu().numpy())

        # Register hook on CSA module
        self.hook_handle = self.csa_module.register_forward_hook(attention_hook)

    def _remove_hooks(self):
        """Remove registered hooks."""
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()

    def generate_with_attention(
        self,
        questions: List[Dict],
        max_new_tokens: int = 96,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Generate answers and capture attention patterns.

        Args:
            questions: List of question dicts with 'question', 'context', 'qid', 'references'
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (answers, attention_data)
        """
        from src.eval_utils import SYSTEM_PROMPT

        self._register_hooks()

        try:
            # Build prompts
            prompts = []
            for q in questions:
                prompt = f"Passage:\n{q['context']}\n\nQuestion: {q['question']}"
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
                full_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(full_prompt)

            # Tokenize
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            # Generate with CSA
            outputs = self.generator.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                enable_cross_batch=True,
            )

            # Decode answers
            answers = []
            for i, seq in enumerate(outputs["generated_tokens"]):
                text = self.tokenizer.decode(seq, skip_special_tokens=True)
                answers.append(text)

            # Compile attention data
            attention_data = {
                "attention_weights": self.attention_weights.copy(),
                "gate_values": self.gate_values.copy(),
                "hidden_states": self.hidden_states.copy(),
            }

            return answers, attention_data

        finally:
            self._remove_hooks()

    def plot_attention_heatmap(
        self,
        attention_matrix: np.ndarray,
        question_labels: List[str],
        title: str = "Cross-Sequence Attention",
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ):
        """Plot attention heatmap showing information flow between questions.

        Args:
            attention_matrix: [batch, batch] attention weights
            question_labels: Labels for each question (short versions)
            title: Plot title
            output_path: Path to save figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create custom colormap (white to blue)
        colors = ["#ffffff", "#e6f3ff", "#99d6ff", "#4db8ff", "#0099ff", "#0066cc"]
        cmap = LinearSegmentedColormap.from_list("csa_blue", colors, N=256)

        # Plot heatmap
        mask = np.eye(len(attention_matrix), dtype=bool)  # Mask diagonal
        masked_attn = np.ma.array(attention_matrix, mask=mask)

        im = ax.imshow(masked_attn, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attention Weight', fontsize=12)

        # Set ticks
        ax.set_xticks(range(len(question_labels)))
        ax.set_yticks(range(len(question_labels)))
        ax.set_xticklabels(question_labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(question_labels, fontsize=10)

        # Add value annotations
        for i in range(len(attention_matrix)):
            for j in range(len(attention_matrix)):
                if i != j:
                    value = attention_matrix[i, j]
                    color = 'white' if value > 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           fontsize=9, color=color)

        ax.set_xlabel('Key (Source Question)', fontsize=12)
        ax.set_ylabel('Query (Target Question)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Mark diagonal
        for i in range(len(attention_matrix)):
            ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                        fill=True, facecolor='lightgray',
                        edgecolor='gray', linewidth=1))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention heatmap to {output_path}")

        return fig

    def plot_gate_distribution(
        self,
        gate_values: List[np.ndarray],
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5),
    ):
        """Plot distribution of gate values.

        Args:
            gate_values: List of gate value arrays from multiple generation steps
            output_path: Path to save figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Flatten all gate values
        all_gates = np.concatenate([g.flatten() for g in gate_values])

        # Left: Histogram
        ax1 = axes[0]
        ax1.hist(all_gates, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax1.axvline(x=all_gates.mean(), color='red', linestyle='--',
                   label=f'Mean: {all_gates.mean():.4f}')
        ax1.axvline(x=np.median(all_gates), color='orange', linestyle='--',
                   label=f'Median: {np.median(all_gates):.4f}')
        ax1.set_xlabel('Gate Value', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Gate Value Distribution', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.set_xlim([0, 1])

        # Right: Per-dimension statistics (sample first few dimensions)
        ax2 = axes[1]
        if len(gate_values) > 0:
            sample_gates = gate_values[-1]  # Use last batch
            dim_means = sample_gates.mean(axis=0)[:100]  # First 100 dimensions
            ax2.bar(range(len(dim_means)), dim_means, color='steelblue', alpha=0.8)
            ax2.axhline(y=dim_means.mean(), color='red', linestyle='--',
                       label=f'Overall Mean: {dim_means.mean():.4f}')
            ax2.set_xlabel('Hidden Dimension (first 100)', fontsize=12)
            ax2.set_ylabel('Mean Gate Value', fontsize=12)
            ax2.set_title('Gate Values by Dimension', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved gate distribution to {output_path}")

        return fig

    def plot_similarity_matrix(
        self,
        hidden_states: np.ndarray,
        question_labels: List[str],
        title: str = "Question Similarity (Hidden States)",
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 7),
    ):
        """Plot cosine similarity matrix of hidden states.

        Args:
            hidden_states: [batch, hidden_dim] hidden state vectors
            question_labels: Labels for each question
            title: Plot title
            output_path: Path to save figure
            figsize: Figure size
        """
        # Compute cosine similarity
        norms = np.linalg.norm(hidden_states, axis=1, keepdims=True)
        normalized = hidden_states / (norms + 1e-9)
        similarity = normalized @ normalized.T

        fig, ax = plt.subplots(figsize=figsize)

        # Plot with diverging colormap
        im = ax.imshow(similarity, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Cosine Similarity', fontsize=12)

        # Set ticks
        ax.set_xticks(range(len(question_labels)))
        ax.set_yticks(range(len(question_labels)))
        ax.set_xticklabels(question_labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(question_labels, fontsize=10)

        # Add value annotations
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                value = similarity[i, j]
                color = 'white' if abs(value) > 0.7 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       fontsize=9, color=color)

        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved similarity matrix to {output_path}")

        return fig

    def plot_case_study(
        self,
        questions: List[Dict],
        answers: List[str],
        attention_matrix: np.ndarray,
        output_path: Optional[str] = None,
        figsize: Tuple[float, float] = (18, 6),
    ):
        """Plot case study with Q&A text and attention heatmap side by side.

        Layout: [Q&A Text] | [Attention Heatmap]
        Aspect ratio approximately 3:1 (width:height)

        Args:
            questions: List of question dicts
            answers: Generated answers
            attention_matrix: Attention weights
            output_path: Path to save figure
            figsize: Figure size (default 18x6 for ~3:1 ratio)
        """
        n = len(questions)

        # Create figure with custom layout
        fig = plt.figure(figsize=figsize, facecolor='white')

        # Grid: Q&A section (wider) | Attention heatmap
        gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1], wspace=0.02)

        # ===== Left: Q&A Text =====
        ax_qa = fig.add_subplot(gs[0])
        ax_qa.axis('off')

        # Build text content
        text_lines = []

        for i, (q, a) in enumerate(zip(questions, answers)):
            ref = q['references'][0] if q['references'] else "N/A"

            # Truncate if needed
            q_text = q['question'][:80] + "..." if len(q['question']) > 80 else q['question']
            a_text = a.strip()[:60] + "..." if len(a.strip()) > 60 else a.strip()
            ref_text = ref[:60] + "..." if len(ref) > 60 else ref

            text_lines.append(f"$q_{{{i+1}}}$: {q_text}")
            text_lines.append(f"      Gen: {a_text}")
            text_lines.append(f"      Ref: {ref_text}")
            text_lines.append("")

        # Join and display
        text_content = "\n".join(text_lines)
        ax_qa.text(0.02, 0.95, text_content, transform=ax_qa.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  linespacing=1.4)
        ax_qa.set_title('Questions & Answers', fontsize=13, fontweight='bold',
                       loc='left', pad=10)

        # ===== Right: Attention Heatmap =====
        ax_attn = fig.add_subplot(gs[1])

        # Labels
        labels = [f"$q_{{{i+1}}}$" for i in range(n)]

        # Mask diagonal
        mask = np.eye(n, dtype=bool)
        masked_attn = np.ma.array(attention_matrix, mask=mask)

        # Custom colormap
        colors = ["#FFFFFF", "#E3F2FD", "#90CAF9", "#42A5F5", "#1976D2", "#0D47A1"]
        cmap = LinearSegmentedColormap.from_list("csa_blue", colors, N=256)

        im = ax_attn.imshow(masked_attn, cmap=cmap, aspect='equal',
                           vmin=0, vmax=max(0.5, attention_matrix.max()))

        # Colorbar
        cbar = plt.colorbar(im, ax=ax_attn, shrink=0.8, pad=0.02)
        cbar.set_label('Attention', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        # Ticks
        ax_attn.set_xticks(range(n))
        ax_attn.set_yticks(range(n))
        ax_attn.set_xticklabels(labels, fontsize=11)
        ax_attn.set_yticklabels(labels, fontsize=11)

        # Value annotations
        for i in range(n):
            for j in range(n):
                if i != j:
                    value = attention_matrix[i, j]
                    color = 'white' if value > 0.35 else '#333333'
                    ax_attn.text(j, i, f'{value:.2f}', ha='center', va='center',
                               fontsize=9, fontweight='bold', color=color)

        # Diagonal markers
        for i in range(n):
            ax_attn.add_patch(plt.Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                fill=True, facecolor='#F5F5F5',
                edgecolor='#BDBDBD', linewidth=0.5
            ))
            ax_attn.text(i, i, '—', ha='center', va='center',
                        fontsize=10, color='#9E9E9E')

        ax_attn.set_xlabel('Source (Key)', fontsize=11)
        ax_attn.set_ylabel('Target (Query)', fontsize=11)
        ax_attn.set_title('Cross-Sequence Attention', fontsize=13, fontweight='bold', pad=10)

        # Minor grid for heatmap
        ax_attn.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax_attn.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax_attn.grid(which='minor', color='white', linewidth=1)
        ax_attn.tick_params(which='minor', size=0)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved case study to {output_path}")

        return fig

    def plot_attention_evolution(
        self,
        attention_history: List[np.ndarray],
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 4),
    ):
        """Plot how attention patterns evolve during generation.

        Args:
            attention_history: List of attention matrices from each generation step
            output_path: Path to save figure
            figsize: Figure size
        """
        n_steps = min(len(attention_history), 5)  # Show at most 5 steps

        fig, axes = plt.subplots(1, n_steps, figsize=figsize)
        if n_steps == 1:
            axes = [axes]

        colors = ["#ffffff", "#e6f3ff", "#99d6ff", "#4db8ff", "#0099ff", "#0066cc"]
        cmap = LinearSegmentedColormap.from_list("csa_blue", colors, N=256)

        step_indices = np.linspace(0, len(attention_history)-1, n_steps, dtype=int)

        for ax_idx, step_idx in enumerate(step_indices):
            attn = attention_history[step_idx]
            n = attn.shape[0]

            mask = np.eye(n, dtype=bool)
            masked_attn = np.ma.array(attn, mask=mask)

            im = axes[ax_idx].imshow(masked_attn, cmap=cmap, aspect='auto', vmin=0, vmax=1)
            axes[ax_idx].set_title(f'Step {step_idx + 1}', fontsize=12)

            # Minimal ticks
            axes[ax_idx].set_xticks([])
            axes[ax_idx].set_yticks([])

            if ax_idx == 0:
                axes[ax_idx].set_ylabel('Query')
            if ax_idx == n_steps // 2:
                axes[ax_idx].set_xlabel('Key')

        # Shared colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, location='right')
        cbar.set_label('Attention Weight', fontsize=10)

        plt.suptitle('Attention Evolution During Generation', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention evolution to {output_path}")

        return fig


def load_examples(dataset: str, num_examples: int, group_size: int, seed: int = 42):
    """Load example question groups from dataset.

    Args:
        dataset: Dataset name
        num_examples: Number of example groups
        group_size: Questions per group
        seed: Random seed

    Returns:
        List of question groups
    """
    from src.eval_utils import load_dataset_groups

    split_map = {
        "squad": "validation",
        "hotpot": "validation",
        "cmb": "train",
        "triviaqa": "validation",
    }
    split = split_map.get(dataset, "validation")

    contexts = load_dataset_groups(
        dataset=dataset,
        split=split,
        max_contexts=num_examples,
        min_questions=group_size,
        max_questions=group_size + 5,
        seed=seed,
        fixed_question_count=group_size,
    )

    # Convert to flat question list format
    groups = []
    for ctx in contexts:
        questions = []
        for q in ctx["questions"][:group_size]:
            questions.append({
                "qid": q["qid"],
                "question": q["text"],
                "context": ctx["context"],
                "references": q["references"],
            })
        groups.append(questions)

    return groups


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer
    print("Initializing CSA Visualizer...")
    visualizer = CSAVisualizer(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Load examples
    print(f"\nLoading {args.num_examples} example groups from {args.dataset}...")
    groups = load_examples(
        dataset=args.dataset,
        num_examples=args.num_examples,
        group_size=args.group_size,
        seed=args.seed,
    )
    print(f"Loaded {len(groups)} groups with {args.group_size} questions each")

    # Process each group and visualize
    all_attention_weights = []
    all_gate_values = []
    all_hidden_states = []

    for group_idx, questions in enumerate(groups):
        print(f"\n{'='*60}")
        print(f"Processing Group {group_idx + 1}/{len(groups)}")
        print(f"{'='*60}")

        # Generate and capture attention
        answers, attention_data = visualizer.generate_with_attention(
            questions=questions,
            max_new_tokens=96,
        )

        # Print Q&A
        for i, (q, a) in enumerate(zip(questions, answers)):
            print(f"\nQ{i+1}: {q['question'][:80]}...")
            print(f"A{i+1}: {a[:80]}...")
            ref = q['references'][0] if q['references'] else 'N/A'
            print(f"Ref: {ref[:80]}...")

        # Store data
        all_attention_weights.extend(attention_data["attention_weights"])
        all_gate_values.extend(attention_data["gate_values"])
        all_hidden_states.extend(attention_data["hidden_states"])

        # Create question labels
        labels = [f"Q{i+1}" for i in range(len(questions))]

        # 1. Plot attention heatmap for this group
        if attention_data["attention_weights"]:
            # Use last attention weights (from final generation step)
            final_attn = attention_data["attention_weights"][-1]
            visualizer.plot_attention_heatmap(
                attention_matrix=final_attn,
                question_labels=labels,
                title=f"Cross-Sequence Attention (Group {group_idx + 1})",
                output_path=output_dir / f"attention_heatmap_group{group_idx + 1}.{args.save_format}",
            )

        # 2. Plot case study
        if attention_data["attention_weights"]:
            visualizer.plot_case_study(
                questions=questions,
                answers=answers,
                attention_matrix=attention_data["attention_weights"][-1],
                output_path=output_dir / f"case_study_group{group_idx + 1}.{args.save_format}",
            )

        # 3. Plot similarity matrix
        if attention_data["hidden_states"]:
            visualizer.plot_similarity_matrix(
                hidden_states=attention_data["hidden_states"][-1],
                question_labels=labels,
                title=f"Question Similarity (Group {group_idx + 1})",
                output_path=output_dir / f"similarity_matrix_group{group_idx + 1}.{args.save_format}",
            )

    # 4. Plot overall gate distribution (aggregated across all groups)
    if all_gate_values:
        visualizer.plot_gate_distribution(
            gate_values=all_gate_values,
            output_path=output_dir / f"gate_distribution_all.{args.save_format}",
        )

    # 5. Plot attention evolution (using first group)
    if all_attention_weights:
        visualizer.plot_attention_evolution(
            attention_history=all_attention_weights[:10],  # First 10 steps
            output_path=output_dir / f"attention_evolution.{args.save_format}",
        )

    # Save metadata
    metadata = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "num_examples": args.num_examples,
        "group_size": args.group_size,
        "seed": args.seed,
        "num_attention_samples": len(all_attention_weights),
        "num_gate_samples": len(all_gate_values),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Visualization complete! Outputs saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob(f"*.{args.save_format}")):
        print(f"  - {f.name}")
    print(f"  - metadata.json")


if __name__ == "__main__":
    main()
