# Parallel Decoding Experiments

This project explores dependency-aware question answering and collaborative generation strategies on SQuAD, HotpotQA, CMB, and other datasets. It contains core library code in `src/` and runnable scripts in `scripts/`.

## Project Structure

```
parallel-agent/
├── src/                       # Core library
│   ├── models.py              # Question, StrategyResult, etc.
│   ├── generators.py          # Dependency generators (Heuristic, LLM, BERT)
│   ├── prompts.py             # Prompt building utilities
│   ├── selection.py           # Cost-aware edge selection
│   ├── utils.py               # Common utilities
│   ├── report.py              # Results reporting
│   ├── datasets/              # Dataset loaders
│   │   ├── squad.py           # SQuAD dataset
│   │   ├── hotpot.py          # HotpotQA dataset
│   │   ├── cmb.py             # CMB (Chinese Medical Benchmark)
│   │   ├── quac.py            # QuAC conversational QA
│   │   ├── quality.py         # QuALITY long-context
│   │   └── drop.py            # DROP discrete reasoning
│   ├── evaluation/            # Evaluation metrics
│   │   ├── basic.py           # EM, F1, lenient metrics
│   │   ├── generation.py      # BLEU, ROUGE metrics
│   │   └── llm.py             # LLM-based evaluation
│   ├── strategies/            # Strategy implementations
│   │   ├── all_in_one.py      # Single prompt strategy
│   │   ├── sequential_batch.py # Sequential & batch strategies
│   │   ├── dependency.py      # Dependency-aware strategy
│   │   └── cross_batch.py     # Cross-batch generation strategy
│   └── cross_batch/           # Cross-batch generation module
│       ├── attention.py       # CrossBatchAttention, CrossBatchEmbeddingMixer
│       ├── generator.py       # CrossBatchGenerator
│       ├── trainer.py         # Training utilities
│       └── eval.py            # Evaluation for cross-batch
├── scripts/                   # Runnable scripts
│   ├── compare_strategies.py  # Multi-strategy comparison
│   ├── train_cross_batch.py   # Train cross-batch module
│   ├── run_parallel.py        # Single-context dependency pipeline
│   └── test_bert_dependencies.py # BERT-based dependency experiments
└── README.md
```

## Strategies Overview

The main comparison script (`scripts/compare_strategies.py`) supports the following strategies:

| Strategy | Description |
|----------|-------------|
| `all_in_one` | Single prompt with all questions |
| `sequential` | One question per turn (history grows) |
| `batch` | All questions answered in one parallel batch |
| `collab_llm` | Dependency DAG via LLM analysis, answers in dependency order |
| `collab_bert` | Dependency DAG via BERT attention, answers in dependency order |
| `collab_hidden` | Cross-batch hidden state mixing during generation (requires trained checkpoint) |

**Note:** The `collab_hidden` strategy requires a trained checkpoint (`--collab-hidden-checkpoint`). It will be automatically skipped if no checkpoint is provided. This strategy is only available with local models, not with API-based inference.

## Quick Start

### Basic Usage

```bash
python scripts/compare_strategies.py \
  --dataset squad \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --context-count 10 \
  --min-questions 5 \
  --max-questions 5 \
  --max-new-tokens 128
```

### Multi-GPU Usage

```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset squad \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --context-count 100 \
  --min-questions 8 \
  --max-questions 8 \
  --max-new-tokens 1024 \
  --json-out outputs_json/results_squad.json
```

### API-Based Inference

Use `--use-api` for API-based inference instead of local models. Supports OpenRouter, DeepSeek, and other providers.

```bash
export OPENROUTER_API_KEY=your_key_here

python scripts/compare_strategies.py \
  --dataset squad \
  --use-api \
  --api-model "qwen/qwen3-30b-a3b" \
  --context-count 10 \
  --strategies "all_in_one,sequential,batch,collab_llm,collab_bert"
```

**Note:** `collab_hidden` is not supported with API inference as it requires access to model hidden states.

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | Dataset: `squad`, `hotpot`, `quac`, `cmb`, `quality`, `drop` |
| `--model-name` | HuggingFace model ID or local path |
| `--context-count` | Number of sampled groups |
| `--min-questions` / `--max-questions` | Questions per group |
| `--max-new-tokens` | Generation token limit |
| `--strategies` | Comma-separated list (e.g., `all_in_one,batch,collab_llm`) |
| `--json-out` | Output directory for results |
| `--use-api` | Use API-based inference |
| `--api-model` | Model ID for API inference |

### Collaborative Strategy Arguments

| Argument | Description |
|----------|-------------|
| `--no-llm-deps` | Use heuristic instead of LLM for dependency generation |
| `--max-dependencies` | Max edges per question |
| `--min-confidence` | Minimum edge confidence threshold |
| `--bert-model-name` | BERT model for `collab_bert` (default: `bert-base-uncased`) |
| `--bert-attention-threshold` | Attention threshold for BERT edges |
| `--collab-hidden-checkpoint` | Path to trained collab_hidden module checkpoint |
| `--collab-hidden-mix-method` | Mixing method: `attention` or `mixer` |
| `--collab-hidden-mix-layer` | Which layer to apply mixing (-1 for last) |

## Dataset Examples

### SQuAD

```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset squad \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --context-count 100 \
  --min-questions 8 \
  --max-questions 8 \
  --max-new-tokens 1024 \
  --json-out outputs_json/results_squad.json
```

### HotpotQA (Multi-Context)

Each question has its own context. Strategies automatically switch to multi-context mode.

```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset hotpot \
  --hotpot-subset distractor \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --context-count 1000 \
  --min-questions 4 \
  --max-questions 4 \
  --max-new-tokens 1024 \
  --json-out outputs_json/results_hotpot.json
```

### CMB (Chinese Medical Benchmark)

Uses BLEU-4 and ROUGE metrics. Optionally add `--eval-model` for LLM-based evaluation.

```bash
export OPENROUTER_API_KEY=your_key_here  # Optional

torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset cmb \
  --cmb-subset CMB-Clin \
  --split test \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --context-count 74 \
  --min-questions 2 \
  --max-questions 4 \
  --max-new-tokens 1024 \
  --json-out outputs_json/results_cmb.json \
  --eval-model openai/gpt-4o  # Optional
```

### QuAC (Conversational QA)

Questions build on each other in a conversation.

```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset quac \
  --split train \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --context-count 100 \
  --min-questions 3 \
  --max-questions 6 \
  --max-new-tokens 256 \
  --json-out outputs_json/results_quac.json
```

### QuALITY (Long-Context)

Long-context reading comprehension (~5000 words per article).

```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset quality \
  --split validation \
  --quality-hard-only \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --context-count 20 \
  --min-questions 5 \
  --max-questions 10 \
  --max-new-tokens 256 \
  --json-out outputs_json/results_quality.json
```

### DROP (Discrete Reasoning)

Requires arithmetic, counting, and sorting.

```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset drop \
  --split validation \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --context-count 50 \
  --min-questions 5 \
  --max-questions 10 \
  --max-new-tokens 128 \
  --json-out outputs_json/results_drop.json
```

## Cross-Batch Module (collab_hidden)

The `collab_hidden` strategy uses cross-batch attention mechanisms that enable information sharing between samples during parallel generation. This requires training a cross-batch module first.

### Training

```bash
python scripts/train_cross_batch.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --mix-method attention \
  --num-epochs 3 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --save-dir ./checkpoints
```

### Using with compare_strategies.py

```bash
python scripts/compare_strategies.py \
  --dataset squad \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --strategies "batch,collab_hidden" \
  --collab-hidden-checkpoint ./checkpoints/best_model.pt \
  --context-count 10 \
  --max-new-tokens 128
```

## Other Scripts

### run_parallel.py

Single-context dependency pipeline for debugging.

```bash
python scripts/run_parallel.py \
  --model-name Qwen/Qwen3-4B \
  --context-count 1 \
  --min-questions 3 \
  --max-questions 5
```

### test_bert_dependencies.py

Offline BERT-based dependency experiments.

```bash
python scripts/test_bert_dependencies.py \
  --model-name bert-base-uncased \
  --context-count 4 \
  --attention-threshold 0.02 \
  --dependency-threshold 0.02 \
  --max-dependencies 3 \
  --show-attention-summary
```

## Evaluation Metrics

Metrics are automatically selected based on dataset:

### Short-form QA (SQuAD, HotpotQA, QuAC, QuALITY, DROP)

| Metric | Description |
|--------|-------------|
| **EM (Strict)** | Exact match after normalization |
| **F1** | Token-level F1 score |
| **Lenient** | Bidirectional substring containment |

### Long-form QA (CMB)

| Metric | Description |
|--------|-------------|
| **BLEU-4** | 4-gram precision with brevity penalty |
| **ROUGE-1/2/L** | Unigram/Bigram/LCS overlap F1 |

### LLM Evaluation (Optional, via `--eval-model`)

| Dimension | Description |
|-----------|-------------|
| **Fluency** | Language quality (1-5) |
| **Relevance** | Answer relevance (1-5) |
| **Completeness** | Information coverage (1-5) |
| **Proficiency** | Domain accuracy (1-5) |

## Requirements

- `transformers`, `datasets`, `torch`
- Optional: `nltk` (BLEU), `rouge-chinese` + `jieba` (Chinese ROUGE), `httpx` (API calls)
- GPU with sufficient RAM (7B model needs ~16GB, 14B needs ~32GB)
