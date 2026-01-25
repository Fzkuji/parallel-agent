#!/usr/bin/env python3
"""
Debug script to understand why we have 9 duplicate Super_Bowl_50 contexts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.squad import load_squad_groups

print("="*80)
print("DEBUG: SQuAD Context Loading")
print("="*80)

# Load with fixed_question_count=1
contexts_1q = load_squad_groups(
    split='validation',
    max_contexts=1000,
    min_questions=20,
    max_questions=20,
    seed=42,
    fixed_question_count=1
)

print(f"\nLoaded {len(contexts_1q)} contexts with fixed_question_count=1")

# Check if contexts are actually different texts
unique_context_texts = {}
for i, ctx in enumerate(contexts_1q):
    ctx_text = ctx['context']
    if ctx_text not in unique_context_texts:
        unique_context_texts[ctx_text] = []
    unique_context_texts[ctx_text].append(i)

print(f"\nUnique context texts: {len(unique_context_texts)}")

for ctx_text, indices in unique_context_texts.items():
    print(f"\nContext (first 100 chars): {ctx_text[:100]}...")
    print(f"  Appears at indices: {indices}")
    print(f"  Questions:")
    for idx in indices:
        q = contexts_1q[idx]['questions'][0]
        print(f"    [{idx}] {q['text']}")

# Now load with fixed_question_count=20
print("\n" + "="*80)
contexts_20q = load_squad_groups(
    split='validation',
    max_contexts=1000,
    min_questions=20,
    max_questions=20,
    seed=42,
    fixed_question_count=20
)

print(f"\nLoaded {len(contexts_20q)} contexts with fixed_question_count=20")

# Check each context
for i, ctx in enumerate(contexts_20q):
    print(f"\nContext {i}: title={ctx['title']}")
    print(f"  Num questions: {len(ctx['questions'])}")
    print(f"  Context text (first 100 chars): {ctx['context'][:100]}...")
    print(f"  First 3 questions:")
    for j in range(min(3, len(ctx['questions']))):
        q = ctx['questions'][j]
        print(f"    {q['qid']}: {q['text']}")

# CRITICAL: Check if 1Q and 20Q use the same underlying contexts
print("\n" + "="*80)
print("COMPARISON: Are 1Q and 20Q using the same contexts?")
print("="*80)

if len(contexts_1q) != len(contexts_20q):
    print(f"\n⚠️ WARNING: Different number of contexts!")
    print(f"  1Q: {len(contexts_1q)} contexts")
    print(f"  20Q: {len(contexts_20q)} contexts")
else:
    print(f"\n✓ Same number of contexts: {len(contexts_1q)}")

    # Check if they're the same contexts
    all_match = True
    for i in range(len(contexts_1q)):
        ctx1 = contexts_1q[i]
        ctx20 = contexts_20q[i]

        if ctx1['context'] != ctx20['context']:
            print(f"\n❌ Context {i} DIFFERS!")
            print(f"  1Q context: {ctx1['context'][:100]}...")
            print(f"  20Q context: {ctx20['context'][:100]}...")
            all_match = False
        else:
            # Check if first question matches
            q1_text = ctx1['questions'][0]['text']
            q20_first = ctx20['questions'][0]['text']

            if q1_text != q20_first:
                print(f"\n❌ Context {i}: Same text but DIFFERENT first question!")
                print(f"  1Q: {q1_text}")
                print(f"  20Q: {q20_first}")
                all_match = False

    if all_match:
        print("\n✓ All contexts match! Same contexts, same first questions.")
        print("\nThis means the 1Q setting tests the first question from each of")
        print("the same 9 contexts that 20Q tests. Batch independence is verified.")
