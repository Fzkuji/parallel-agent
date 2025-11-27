"""Core library for dependency-aware parallel question answering."""

# Data models
from .models import (
    BatchAssignment,
    EdgeCandidate,
    Question,
    ScheduleResult,
    estimate_tokens,
)

# Evaluation metrics
from .eval import (
    DATASET_METRICS,
    compute_bleu4,
    compute_contains,
    compute_em,
    compute_f1,
    compute_rouge1,
    compute_rouge2,
    compute_rouge_l,
    evaluate_for_dataset,
    evaluate_predictions,
    get_dataset_metrics,
    get_metric_names,
)

# Text utilities
from .text_utils import (
    AGGREGATE_KEYWORDS,
    REFERENCE_KEYWORDS,
    STOPWORDS,
    detect_aggregate_question,
    detect_reference_question,
    extract_keywords,
)

# Scheduler
from .scheduler import (
    DependencyScheduler,
    export_schedule_html,
)

# Dependency generators
from .generators import (
    BertAttentionDependencyGenerator,
    DependencyGraphGenerator,
    HeuristicDependencyGenerator,
    LLMDependencyGenerator,
)

# Edge selection
from .selection import (
    apply_dependencies,
    compute_dependency_cost,
    select_dependency_edges,
)

# Data loaders
from .loaders import (
    build_questions_from_group,
    load_cmb_groups,
    load_hotpot_groups,
    load_squad_groups,
    load_squad_random_questions,
)

# Inference utilities
from .inference import (
    BOX_PATTERN,
    DEFAULT_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    USE_THINK_TOKENS,
    LocalLLMDependencyGenerator,
    build_chat_prompt,
    build_dependency_prompt,
    extract_box_answer,
    extract_json_from_text,
    set_think_tokens,
)

__all__ = [
    # Models
    "Question",
    "EdgeCandidate",
    "BatchAssignment",
    "ScheduleResult",
    "estimate_tokens",
    # Evaluation metrics
    "compute_em",
    "compute_f1",
    "compute_contains",
    "compute_bleu4",
    "compute_rouge1",
    "compute_rouge2",
    "compute_rouge_l",
    "evaluate_predictions",
    "evaluate_for_dataset",
    "get_dataset_metrics",
    "get_metric_names",
    "DATASET_METRICS",
    # Text utils
    "STOPWORDS",
    "REFERENCE_KEYWORDS",
    "AGGREGATE_KEYWORDS",
    "extract_keywords",
    "detect_reference_question",
    "detect_aggregate_question",
    # Scheduler
    "DependencyScheduler",
    "export_schedule_html",
    # Generators
    "DependencyGraphGenerator",
    "HeuristicDependencyGenerator",
    "LLMDependencyGenerator",
    "BertAttentionDependencyGenerator",
    # Selection
    "compute_dependency_cost",
    "select_dependency_edges",
    "apply_dependencies",
    # Loaders
    "load_squad_groups",
    "load_squad_random_questions",
    "load_hotpot_groups",
    "load_cmb_groups",
    "build_questions_from_group",
    # Inference utilities
    "DEFAULT_SYSTEM_PROMPT",
    "PLANNER_SYSTEM_PROMPT",
    "BOX_PATTERN",
    "USE_THINK_TOKENS",
    "set_think_tokens",
    "build_chat_prompt",
    "extract_box_answer",
    "extract_json_from_text",
    "build_dependency_prompt",
    "LocalLLMDependencyGenerator",
]
