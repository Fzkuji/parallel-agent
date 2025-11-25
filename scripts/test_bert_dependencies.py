#!/usr/bin/env python3
"""
Attention-based dependency tester.

This script packs every question from a SQuAD context into a single BERT encoder
pass, aggregates token-to-token self-attention weights, and treats the
question-level attention strength as dependency confidence (higher attention →
stronger dependency). The final graph is pruned by the usual cost-aware
selection logic to keep it acyclic.

HOW IT WORKS:
-------------
1. Load question-answer groups from SQuAD.
2. Concatenate all questions (with [CLS]/[SEP]) and run a bidirectional BERT encoder.
3. Aggregate the attention paid from each question's tokens to every other question.
4. Create candidate edges where the averaged attention ≥ attention_threshold.
5. Apply edge selection (confidence threshold, max dependencies, DAG constraint).
6. Visualize and optionally export the resulting attention-driven dependency graph.

KEY CONCEPTS:
-------------
- **Attention Weight**: Mean BERT self-attention from question_i tokens to question_j tokens.
- **Candidate Edge**: Generated when attention weight ≥ attention_threshold.
- **Dependency Threshold**: Minimum confidence retained after selection (defaults to attention threshold).
- **DAG Output**: Cycle prevention keeps the final dependency graph single-directional.

USAGE EXAMPLES:
---------------
# 标准运行命令（BERT encoder，注意力生成依赖）
python test_bert_dependencies.py --model-name bert-base-uncased --context-count 4 --attention-threshold 0.02 --dependency-threshold 0.02 --max-dependencies 3 --show-attention-summary

# 调试：打印注意力矩阵并放宽阈值
python test_bert_dependencies.py --model-name bert-base-uncased --context-count 1 --attention-threshold 0.05 --dependency-threshold 0.05 --show-attention-matrix

OUTPUT:
-------
- Console: Attention matrix (optional), candidate edges, selected DAG, dependency summary
- JSON file: Full dump of questions + attention-based edges for the first context
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (
    BertAttentionDependencyGenerator,
    EdgeCandidate,
    Question,
    load_squad_groups,
    select_dependency_edges,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def visualize_dependency_graph(
    questions: List[Question],
    edges: List[EdgeCandidate],
    selected_edges: Dict[str, List[EdgeCandidate]],
) -> None:
    print("\n" + "=" * 80)
    print("ATTENTION-BASED DEPENDENCY GRAPH")
    print("=" * 80)

    selected_pairs = {
        (edge.source, edge.target)
        for target_edges in selected_edges.values()
        for edge in target_edges
    }
    adjacency: Dict[str, List[tuple]] = {q.qid: [] for q in questions}
    for edge in edges:
        adjacency[edge.source].append((edge.target, edge.confidence, (edge.source, edge.target) in selected_pairs))

    print("\nQUESTIONS:")
    print("-" * 80)
    for question in questions:
        print(f"{question.qid}: {question.text}")

    print("\n\nALL CANDIDATE EDGES (from attention weights):")
    print("-" * 80)
    print(f"{'Source':<8} {'Target':<8} {'Weight':<12} {'Selected':<10} {'Rationale'}")
    print("-" * 80)
    for edge in sorted(edges, key=lambda e: -e.confidence):
        mark = "✓" if (edge.source, edge.target) in selected_pairs else "✗"
        print(f"{edge.source:<8} {edge.target:<8} {edge.confidence:<12.3f} {mark:<10} {edge.rationale or ''}")

    print("\n\nSELECTED DEPENDENCIES:")
    print("-" * 80)
    for question in questions:
        chosen = [(target, conf) for target, conf, sel in adjacency[question.qid] if sel]
        if chosen:
            seq = ", ".join(f"{target} ({conf:.3f})" for target, conf in chosen)
            print(f"{question.qid} → {seq}")

    print("\n\nDEPENDENCY SUMMARY (targets and their prerequisites):")
    print("-" * 80)
    reverse: Dict[str, List[tuple]] = {q.qid: [] for q in questions}
    for target, edges_list in selected_edges.items():
        for edge in edges_list:
            reverse[target].append((edge.source, edge.confidence))
    for question in questions:
        deps = reverse[question.qid]
        if deps:
            deps_str = ", ".join(f"{src} ({conf:.3f})" for src, conf in deps)
            print(f"{question.qid}: depends on {deps_str}")
        else:
            print(f"{question.qid}: independent")

    print("\n" + "=" * 80)


def print_attention_matrix(
    questions: List[Question],
    attention_matrix,
) -> None:
    print("\n" + "=" * 80)
    print("QUESTION-LEVEL ATTENTION MATRIX")
    print("=" * 80)
    print(f"\n{'':8}", end="")
    for question in questions:
        print(f"{question.qid:>8}", end="")
    print()
    for i, question in enumerate(questions):
        print(f"{question.qid:8}", end="")
        for j in range(len(questions)):
            if i == j:
                print(f"{'---':>8}", end="")
            else:
                print(f"{attention_matrix[i, j]:>8.3f}", end="")
        print()
    print("\n" + "=" * 80)


def print_attention_summary(
    questions: List[Question],
    attention_matrix,
) -> None:
    print("\n" + "=" * 80)
    print("QUESTION-LEVEL ATTENTION SUMMARY")
    print("=" * 80)
    for i, src in enumerate(questions):
        items = []
        for j, tgt in enumerate(questions):
            if i == j:
                continue
            items.append(f"{tgt.qid}:{attention_matrix[i, j]:.3f}")
        row = " | ".join(items) if items else "n/a"
        print(f"{src.qid} -> {row}")
    print("\n" + "=" * 80)


def export_to_json(
    background: str,
    questions: List[Question],
    edges: List[EdgeCandidate],
    selected_edges: Dict[str, List[EdgeCandidate]],
    output_file: str,
) -> None:
    selected_pairs = {
        (edge.source, edge.target)
        for target_edges in selected_edges.values()
        for edge in target_edges
    }
    dependencies = {
        target: [edge.source for edge in edges_list]
        for target, edges_list in selected_edges.items()
    }
    payload = {
        "background": background,
        "questions": [
            {
                "qid": question.qid,
                "text": question.text,
                "gold_answers": question.references,
                "dependencies": dependencies.get(question.qid, []),
            }
            for question in questions
        ],
        "all_edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "confidence": edge.confidence,
                "rationale": edge.rationale,
                "selected": (edge.source, edge.target) in selected_pairs,
            }
            for edge in edges
        ],
    }
    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)
    logging.info("Results exported to %s", output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention-based BERT dependency tester")
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Encoder-only Hugging Face model to pull attentions from (default: bert-base-uncased)",
    )
    parser.add_argument(
        "--attention-threshold",
        type=float,
        default=0.08,
        help="Minimum attention weight required to form a candidate edge (default: 0.08)",
    )
    parser.add_argument(
        "--max-question-tokens",
        type=int,
        default=64,
        help="Maximum wordpiece tokens retained per question before packing (default: 64)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum packed sequence length fed into the encoder (default: 512)",
    )
    parser.add_argument(
        "--dependency-threshold",
        type=float,
        default=None,
        help="Minimum confidence required after selection (defaults to attention-threshold)",
    )
    parser.add_argument(
        "--max-dependencies",
        type=int,
        default=3,
        help="Maximum dependencies allowed per question (default: 3)",
    )
    parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.0,
        help="Penalty weight applied to dependency cost during selection (default: 0.0)",
    )
    parser.add_argument(
        "--min-questions",
        type=int,
        default=5,
        help="Minimum questions per context (default: 5)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=5,
        help="Maximum questions per context (default: 5)",
    )
    parser.add_argument(
        "--context-count",
        type=int,
        default=2,
        help="Number of SQuAD contexts to load (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bert_attention_dependencies.json",
        help="Output JSON path for first context dump (default: bert_attention_dependencies.json)",
    )
    parser.add_argument(
        "--show-attention-matrix",
        action="store_true",
        help="Display the aggregated question-level attention matrix",
    )
    parser.add_argument(
        "--show-attention-summary",
        action="store_true",
        help="Print a compact table of attention weights per question pair",
    )

    args = parser.parse_args()
    dependency_threshold = args.dependency_threshold or args.attention_threshold

    logging.info("Loading SQuAD contexts...")
    try:
        raw_groups = load_squad_groups(
            split="train",
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            max_contexts=args.context_count,
        )
    except Exception as exc:
        logging.error("Failed to load SQuAD contexts: %s", exc)
        sys.exit(1)

    contexts = []
    for group in raw_groups:
        questions = [
            Question(
                qid=item["qid"],
                text=item["text"],
                references=item["references"],
                answer_tokens=item["answer_tokens"],
                type_hint=None,
                explicit_dependencies=[],
            )
            for item in group["questions"]
        ]
        contexts.append((group["title"], group["context"], questions))
    logging.info("Loaded %d contexts", len(contexts))

    generator = BertAttentionDependencyGenerator(
        model_name=args.model_name,
        attention_threshold=args.attention_threshold,
        max_question_tokens=args.max_question_tokens,
        max_total_tokens=args.max_seq_length,
    )

    for idx, (title, background, questions) in enumerate(contexts, 1):
        print(f"\n{'=' * 80}")
        print(f"CONTEXT {idx}/{len(contexts)}: {title}")
        print(f"{'=' * 80}")
        print(f"\nBackground: {background[:200]}...")

        logging.info("Computing attention weights for context: %s", title)
        attention_matrix = generator.compute_question_attention_matrix(questions)
        edges = generator.build_edges_from_scores(questions, attention_matrix)
        logging.info("Generated %d candidate edges via attention", len(edges))

        question_map = {question.qid: question for question in questions}
        selected_edges = select_dependency_edges(
            questions=question_map,
            edge_candidates=edges,
            min_confidence=dependency_threshold,
            max_dependencies_per_target=args.max_dependencies,
            cost_weight=args.cost_weight,
            prevent_cycles=True,
        )
        total_selected = sum(len(edge_list) for edge_list in selected_edges.values())
        logging.info("Selected %d edges after filtering", total_selected)

        if args.show_attention_matrix:
            print_attention_matrix(questions, attention_matrix)
        if args.show_attention_summary or not edges:
            print_attention_summary(questions, attention_matrix)

        visualize_dependency_graph(questions, edges, selected_edges)

        if idx == 1:
            export_to_json(background, questions, edges, selected_edges, args.output)


if __name__ == "__main__":
    main()
