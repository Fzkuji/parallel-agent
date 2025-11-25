# Battlenet Project Guide (For Claude)

This document provides a quick reference for understanding the project structure and key concepts.

## Project Overview

**Purpose**: Compare different question-answering strategies on QA datasets (SQuAD, HotpotQA) using local LLMs. The core research question is: **Can dependency-aware parallel processing improve multi-question answering efficiency without sacrificing accuracy?**

**Key Idea**: When answering multiple questions about the same context, some questions may depend on others (e.g., "Where was the treaty signed?" → "Who signed the treaty there?"). This project explores whether detecting these dependencies and scheduling questions accordingly can improve performance.

## Directory Structure

```
battlenet/
├── scripts/                    # Executable scripts
│   ├── compare_strategies.py   # Main entry point - compares all strategies
│   ├── run_parallel.py         # Simplified parallel-only runner
│   ├── test_bert_dependencies.py  # Test BERT attention-based dependency detection
│   └── debug_batch_vs_sequential.py  # Debug tool for batch generation
├── src/                        # Core library
│   ├── __init__.py            # Public API exports
│   ├── models.py              # Data classes: Question, EdgeCandidate, BatchAssignment
│   ├── loaders.py             # Dataset loaders: load_squad_groups, load_hotpot_groups
│   ├── eval.py                # Evaluation metrics: compute_em, compute_f1, compute_contains
│   ├── inference.py           # LLM inference utilities: build_chat_prompt, extract_json
│   ├── generators.py          # Dependency generators: Heuristic, LLM, BERT-attention
│   ├── scheduler.py           # DependencyScheduler: schedules questions into batches
│   ├── selection.py           # Edge selection: select_dependency_edges
│   ├── strategies/            # Strategy implementations
│   │   ├── executors.py       # Strategy runners
│   │   ├── all_in_one.py      # All-in-one strategy
│   │   ├── sequential_batch.py # Sequential and batch strategies
│   │   └── dependency.py      # Dependency-aware parallel strategy
│   ├── report.py              # Result formatting and printing
│   ├── results.py             # StrategyResult dataclass
│   └── prompts.py             # Prompt templates
└── README.md                  # User-facing documentation
```

## Five QA Strategies

| Strategy | Description | Batches | Use Case |
|----------|-------------|---------|----------|
| `all_in_one` | All questions in one prompt | 1 | Baseline, simple |
| `sequential` | One question at a time | N | No parallelism |
| `batch` | All questions in parallel, one batch | 1 | Max parallelism, no deps |
| `parallel` | Dependency-scheduled batches (LLM deps) | Variable | Smart parallelism |
| `parallel_bert` | Dependency-scheduled batches (BERT deps) | Variable | Fast dep detection |

## Key Data Structures

### Question (src/models.py)
```python
@dataclass
class Question:
    qid: str                    # e.g., "Q1", "Q2"
    text: str                   # Question text
    references: List[str]       # Gold answers for evaluation
    dependencies: Set[str]      # QIDs this question depends on
    answer_tokens: int          # Estimated answer length
```

### EdgeCandidate (src/models.py)
```python
@dataclass
class EdgeCandidate:
    source: str      # Source question QID (provides info)
    target: str      # Target question QID (needs info from source)
    confidence: float  # 0.0-1.0 confidence score
    rationale: str   # Why this dependency exists
```

### StrategyResult (src/results.py)
```python
@dataclass
class StrategyResult:
    name: str                    # Strategy name
    answers: Dict[str, str]      # {qid: answer}
    metrics: Dict[str, float]    # {strict_acc, lenient_acc, f1}
    prompt_tokens: int           # Total input tokens
    generated_tokens: int        # Total output tokens
    latency: float              # Total time in seconds
    batches: int                # Number of LLM calls
```

## Evaluation Metrics (src/eval.py)

| Metric | Function | Description |
|--------|----------|-------------|
| **EM (Strict)** | `compute_em()` | Exact match after normalization |
| **F1** | `compute_f1()` | Token-level F1 score |
| **Lenient** | `compute_contains()` | Bidirectional substring containment |

**Normalization**: lowercase, remove punctuation, remove articles (a/an/the)

## Dependency Detection Methods

### 1. Heuristic (HeuristicDependencyGenerator)
- Keyword matching for reference words ("it", "they", "the answer above")
- Fast but low accuracy

### 2. LLM-based (LocalLLMDependencyGenerator)
- Uses the same LLM to analyze question pairs
- High accuracy but slow (O(n²) LLM calls)

### 3. BERT Attention (BertAttentionDependencyGenerator)
- Packs all questions into BERT, extracts attention weights
- Question-level attention → dependency confidence
- Fast and reasonably accurate

## Main Script: compare_strategies.py

### Key Arguments
```bash
--dataset {squad,hotpot}     # Dataset choice
--model-name <hf_model>      # HuggingFace model ID
--context-count N            # Number of context groups
--min-questions M            # Min questions per group
--max-questions K            # Max questions per group
--strategies "s1,s2,..."     # Comma-separated strategy list
--json-out <path>            # Output directory for results
```

### Multi-GPU Support
```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py ...
```
- Uses `torch.distributed` for gathering results
- Only rank 0 saves final results
- All ranks' data is merged before saving

### Output Structure
```
outputs_json/
└── 20251125_120000_squad_train_Qwen2.5-7B-Instruct_n10_q3-5/
    ├── config.txt          # Human-readable config
    ├── full_results.json   # Complete results
    └── errors.json         # Error cases only
```

## Dataset Formats

### SQuAD (rajpurkar/squad)
- Multiple questions share one context/background
- Questions may have natural dependencies

### HotpotQA (hotpotqa/hotpot_qa)
- Each question has its own context (multi-hop reasoning)
- Questions are bundled into groups for testing
- Subset options: `distractor`, `fullwiki`

## Common Code Patterns

### Loading Data
```python
from src import load_squad_groups, load_hotpot_groups

contexts = load_squad_groups("train", min_questions=3, max_questions=8, max_contexts=10)
# Returns: List[dict] with keys: title, context, questions
```

### Running Strategies
```python
from src.strategies import run_all_in_one_strategy, run_dependency_batch_strategy

result = run_all_in_one_strategy(background, questions, tokenizer, model, max_new_tokens=96)
# Returns: StrategyResult
```

### Evaluating Answers
```python
from src.eval import compute_em, compute_f1, evaluate_predictions

em = compute_em(prediction, references)  # 0.0 or 1.0
f1 = compute_f1(prediction, references)  # 0.0 to 1.0
```

## Important Implementation Details

1. **JSON Answer Extraction**: Answers are expected in `{"answer": "..."}` format for strict evaluation
2. **Batch Padding**: Left-padding for decoder-only models in batch generation
3. **Distributed Saving**: Only rank 0 gathers and saves results in multi-GPU mode
4. **Folder Naming**: `{timestamp}_{dataset}_{split}_{model}_n{samples}_q{questions}`

## Typical Workflow

1. Load dataset → `load_squad_groups()` / `load_hotpot_groups()`
2. For each context group:
   - Build Question objects
   - Run each strategy → get StrategyResult
   - Evaluate and collect metrics
3. Aggregate metrics across all contexts
4. Save results to JSON

## Key Files to Read First

1. `scripts/compare_strategies.py` - Main entry point, orchestrates everything
2. `src/models.py` - Core data structures
3. `src/strategies/executors.py` - Strategy implementations
4. `src/eval.py` - Evaluation metrics
5. `src/inference.py` - LLM inference utilities
