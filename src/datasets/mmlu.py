"""MMLU dataset loader for multiple-choice evaluation."""

from typing import List, Dict
import random


def load_mmlu(
    split: str = "validation",
    max_contexts: int = 100,
    min_questions: int = 5,
    max_questions: int = 5,
    seed: int = 42,
    subjects: List[str] = None,
) -> List[Dict]:
    """
    Load MMLU (Massive Multitask Language Understanding) dataset.

    MMLU consists of multiple-choice questions across 57 subjects.
    We group questions by subject to create contexts with multiple related questions.

    Args:
        split: Dataset split ('validation', 'test', 'dev')
        max_contexts: Maximum number of subject groups to return
        min_questions: Minimum questions per subject
        max_questions: Maximum questions per subject
        seed: Random seed
        subjects: List of specific subjects to include (None = all subjects)

    Returns:
        List of context dictionaries with questions
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    # Map split names
    if split == "validation":
        split = "dev"  # MMLU uses 'dev' for validation

    # Load MMLU dataset (all subjects)
    if subjects is None:
        # All 57 MMLU subjects
        subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology", "high_school_statistics",
            "high_school_us_history", "high_school_world_history", "human_aging",
            "human_sexuality", "international_law", "jurisprudence",
            "logical_fallacies", "machine_learning", "management", "marketing",
            "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
            "nutrition", "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology", "us_foreign_policy",
            "virology", "world_religions"
        ]

    # Group questions by subject
    subject_groups = {}

    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split=split, trust_remote_code=True)
            subject_groups[subject] = list(dataset)
        except Exception as e:
            print(f"Warning: Could not load subject '{subject}': {e}")
            continue

    # Sample subjects
    rng = random.Random(seed)
    available_subjects = list(subject_groups.keys())
    rng.shuffle(available_subjects)
    selected_subjects = available_subjects[:max_contexts] if max_contexts else available_subjects

    formatted: List[Dict] = []

    for subject in selected_subjects:
        items = subject_groups[subject]

        # Sample questions for this subject
        rng.shuffle(items)
        num_questions = rng.randint(min_questions, min(max_questions, len(items)))
        selected_items = items[:num_questions]

        # Build context (just subject name)
        context = f"Subject: {subject.replace('_', ' ').title()}"

        questions = []
        for i, item in enumerate(selected_items):
            question_text = item["question"].strip()
            choices = item["choices"]
            correct_idx = item["answer"]  # 0-3 for A-D

            # Format as multiple choice
            choice_letters = ["A", "B", "C", "D"]
            formatted_choices = "\n".join(
                f"{letter}. {choice}"
                for letter, choice in zip(choice_letters, choices)
            )
            full_question = f"{question_text}\n{formatted_choices}"

            # Answer is the letter (A, B, C, or D)
            correct_answer = choice_letters[correct_idx]

            questions.append({
                "qid": f"Q{i+1}",
                "question": full_question,
                "answer_tokens": 5,  # MCQ answers are typically short
                "references": [correct_answer, correct_answer.lower()],
            })

        if questions:
            formatted.append({
                "context": context,
                "title": subject,
                "questions": questions,
            })

    return formatted
