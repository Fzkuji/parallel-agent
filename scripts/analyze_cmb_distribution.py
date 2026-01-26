#!/usr/bin/env python3
"""Analyze question count distribution in CMB dataset."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from datasets import load_dataset

def analyze_cmb_clin():
    """Analyze CMB-Clin dataset (clinical cases with multiple QA pairs)."""
    print("\n" + "="*60)
    print("CMB-Clin Analysis")
    print("="*60)
    
    raw = load_dataset("FreedomIntelligence/CMB", "CMB-Clin", split="test")
    
    qa_counts = [len(row.get("QA_pairs", [])) for row in raw]
    distribution = Counter(qa_counts)
    
    print(f"\nTotal cases: {len(raw)}")
    print(f"Total QA pairs: {sum(qa_counts)}")
    
    print(f"\nQA pairs per case distribution:")
    print(f"{'QA Count':<12} {'Cases':<12} {'Cumulative':<12}")
    print("-" * 36)
    
    cumulative = 0
    for count in sorted(distribution.keys(), reverse=True):
        num_cases = distribution[count]
        cumulative += num_cases
        print(f"{count:<12} {num_cases:<12} {cumulative:<12}")
    
    print(f"\n{'Min QA Pairs':<15} {'Cases Available':<20} {'Total QA Pairs':<15}")
    print("-" * 50)
    for min_q in [1, 2, 3, 4, 5, 8, 10, 12, 16]:
        cases = sum(1 for c in qa_counts if c >= min_q)
        total_q = sum(min(c, min_q) for c in qa_counts if c >= min_q)
        print(f"{min_q:<15} {cases:<20} {total_q:<15}")


def analyze_cmb_exam_context():
    """Analyze CMB-Exam-Grouped context config."""
    print("\n" + "="*60)
    print("CMB-Exam-Grouped (context) Analysis")
    print("="*60)
    
    for split in ["train", "val", "test"]:
        try:
            raw = load_dataset("fzkuji/CMB-Exam-Grouped", "context", split=split)
            qa_counts = [row.get("num_questions", 0) for row in raw]
            distribution = Counter(qa_counts)
            
            print(f"\n--- Split: {split} ---")
            print(f"Total groups: {len(raw)}")
            print(f"Total questions: {sum(qa_counts)}")
            
            print(f"\nQuestions per group distribution:")
            print(f"{'Q Count':<12} {'Groups':<12} {'Cumulative':<12}")
            print("-" * 36)
            
            cumulative = 0
            for count in sorted(distribution.keys(), reverse=True)[:15]:
                num_groups = distribution[count]
                cumulative += num_groups
                print(f"{count:<12} {num_groups:<12} {cumulative:<12}")
            
            print(f"\n{'Min Questions':<15} {'Groups Available':<20} {'Total Questions':<15}")
            print("-" * 50)
            for min_q in [1, 2, 4, 8, 12, 16, 20]:
                groups = sum(1 for c in qa_counts if c >= min_q)
                total_q = sum(min(c, min_q) for c in qa_counts if c >= min_q)
                print(f"{min_q:<15} {groups:<20} {total_q:<15}")
        except Exception as e:
            print(f"\nError loading {split}: {e}")


def analyze_cmb_exam_subdomain():
    """Analyze CMB-Exam-Grouped subdomain config."""
    print("\n" + "="*60)
    print("CMB-Exam-Grouped (subdomain) Analysis")
    print("="*60)
    
    for split in ["train", "val", "test"]:
        try:
            raw = load_dataset("fzkuji/CMB-Exam-Grouped", "subdomain", split=split)
            qa_counts = [row.get("num_questions", 0) for row in raw]
            distribution = Counter(qa_counts)
            
            print(f"\n--- Split: {split} ---")
            print(f"Total groups: {len(raw)}")
            print(f"Total questions: {sum(qa_counts)}")
            
            print(f"\nQuestions per group distribution:")
            print(f"{'Q Count':<12} {'Groups':<12} {'Cumulative':<12}")
            print("-" * 36)
            
            cumulative = 0
            for count in sorted(distribution.keys(), reverse=True)[:15]:
                num_groups = distribution[count]
                cumulative += num_groups
                print(f"{count:<12} {num_groups:<12} {cumulative:<12}")
            
            print(f"\n{'Min Questions':<15} {'Groups Available':<20} {'Total Questions':<15}")
            print("-" * 50)
            for min_q in [1, 2, 4, 8, 12, 16, 20]:
                groups = sum(1 for c in qa_counts if c >= min_q)
                total_q = sum(min(c, min_q) for c in qa_counts if c >= min_q)
                print(f"{min_q:<15} {groups:<20} {total_q:<15}")
        except Exception as e:
            print(f"\nError loading {split}: {e}")


if __name__ == "__main__":
    print("Loading and analyzing CMB datasets...")
    
    analyze_cmb_clin()
    analyze_cmb_exam_context()
    analyze_cmb_exam_subdomain()
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
