# Parallel Decoding Experiments

This project explores dependency-aware question answering and collaborative generation strategies on SQuAD, HotpotQA, CMB, and other datasets. It contains core library code in `src/` and runnable scripts in `scripts/`.

## Project Structure

```
parallel-agent/
├── src/                       # Core library
│   ├── models.py              # Question, StrategyResult, etc.
│   ├── generators.py          # Dependency generators (Heuristic, LLM)
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
│   ├── eval_baselines.py      # Evaluate baseline strategies (vLLM, multi-GPU)
│   ├── train_and_eval_sft.py  # Train and evaluate SFT-LoRA
│   ├── compare_strategies.py  # Multi-strategy comparison (all strategies)
│   ├── train_cross_batch.py   # Train cross-batch module (DDP)
│   ├── cross_dataset_eval.py  # Cross-dataset generalization evaluation
│   └── run_parallel.py        # Single-context dependency pipeline
└── README.md
```

## Scripts Overview

The project provides multiple scripts for different use cases:

### Evaluation Scripts

| Script | Purpose | Features |
|--------|---------|----------|
| `eval_baselines.py` | Evaluate baseline strategies (no training required) | vLLM inference, multi-GPU parallel, result caching |
| `compare_strategies.py` | Compare all strategies including trained models | HuggingFace inference, supports all strategies, requires checkpoints for trained strategies |
| `cross_dataset_eval.py` | Cross-dataset generalization evaluation | Evaluate on multiple datasets, supports LoRA and Cross-Batch |

### Training Scripts

| Script | Purpose | Features |
|--------|---------|----------|
| `train_and_eval_sft.py` | Train and evaluate SFT-LoRA model | LoRA fine-tuning, multi-GPU evaluation |
| `train_cross_batch.py` | Train Cross-Batch module | DDP multi-GPU training, multiple module types |

### Which Script to Use?

```
Need to evaluate baseline strategies (no training)?
  → Use eval_baselines.py (fastest, vLLM-based)

Need to train a model?
  → SFT-LoRA: Use train_and_eval_sft.py
  → Cross-Batch: Use train_cross_batch.py

Need to compare trained models with baselines?
  → Use compare_strategies.py (supports all strategies)

Need to test generalization across datasets?
  → Use cross_dataset_eval.py
```

## Strategies Overview

The main comparison script (`scripts/compare_strategies.py`) supports the following strategies:

| Strategy | Description |
|----------|-------------|
| `all_in_one` | Single prompt with all questions |
| `sequential` | One question per turn (history grows) |
| `batch` | All questions answered in one parallel batch |
| `collab_llm` | Dependency DAG via LLM analysis, answers in dependency order |
| `collab_hidden` | Cross-batch hidden state mixing during generation (requires trained checkpoint) |

**Note:** The `collab_hidden` strategy requires a trained checkpoint (`--collab-hidden-checkpoint`). It will be automatically skipped if no checkpoint is provided. This strategy is only available with local models, not with API-based inference.

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

GPU with sufficient RAM: 7B model needs ~16GB, 14B needs ~32GB.

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
  --strategies "all_in_one,sequential,batch,collab_llm"
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
| `--strategies` | Comma-separated list (see table below) |
| `--json-out` | Output directory for results |
| `--use-api` | Use API-based inference |
| `--api-model` | Model ID for API inference |
| `--use-vllm` | Also run vLLM inference for comparison |
| `--auto-checkpoints` | Auto-discover trained checkpoints |

### Available Strategies

| Strategy | HF | vLLM | Requires Checkpoint |
|----------|:--:|:----:|:-------------------:|
| `all_in_one` | Yes | Yes | No |
| `sequential` | Yes | Yes | No |
| `batch` | Yes | Yes | No |
| `collab_llm` | Yes | Yes | No |
| `finetuned` | Yes | No | `--baseline-checkpoint` |
| `collab_hidden` | Yes | No | `--collab-hidden-checkpoint` |
| `lora_lmhead` | Yes | No | `--lora-lmhead-checkpoint` |
| `lora_crossbatch` | Yes | No | `--lora-crossbatch-checkpoint` |

### Collaborative Strategy Arguments

| Argument | Description |
|----------|-------------|
| `--no-llm-deps` | Use heuristic instead of LLM for dependency generation |
| `--max-dependencies` | Max edges per question |
| `--min-confidence` | Minimum edge confidence threshold |
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

### CMB-Clin (Chinese Medical Clinical Cases)

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

### CMB-Exam (Chinese Medical Exam)

Multiple-choice questions with accuracy metric. Three grouping modes:

| Mode | Description |
|------|-------------|
| `random` | Random grouping baseline (no shared context) |
| `subdomain` | Grouped by medical specialty/terms |
| `context` | Questions with shared background/context |

```bash
# Random grouping (baseline)
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset cmb \
  --cmb-subset random \
  --split test \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --context-count 100 \
  --min-questions 5 \
  --max-questions 5 \
  --max-new-tokens 64 \
  --json-out outputs_json/results_cmb_exam_random.json

# Subdomain grouping
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset cmb \
  --cmb-subset subdomain \
  --split test \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --context-count 100 \
  --min-questions 3 \
  --max-questions 8 \
  --max-new-tokens 64 \
  --json-out outputs_json/results_cmb_exam_subdomain.json

# Context grouping (shared background)
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset cmb \
  --cmb-subset context \
  --split test \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --context-count 100 \
  --min-questions 2 \
  --max-questions 6 \
  --max-new-tokens 64 \
  --json-out outputs_json/results_cmb_exam_context.json
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

## Cross-Batch Module Training & Evaluation

The `collab_hidden` strategy uses cross-batch attention mechanisms that enable information sharing between samples during parallel generation. The training script trains and evaluates multiple fine-tuning approaches:

| Method | Description |
|--------|-------------|
| `baseline` | Only train lm_head |
| `crossbatch` | Train lm_head + cross-batch attention module |
| `lora_lmhead` | LoRA adapters + lm_head |
| `lora_crossbatch` | LoRA + lm_head + cross-batch attention |

### Training (DDP Multi-GPU)

The training script supports DDP (DistributedDataParallel) for multi-GPU training. Use `torchrun` to launch:

```bash
# Full training on SQuAD (8 GPUs)
torchrun --nproc_per_node=8 scripts/train_cross_batch.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset squad \
    --batch-size 4 \
    --epochs 1

# Training with Question-Aware Gating (recommended for better cross-batch selection)
torchrun --nproc_per_node=8 scripts/train_cross_batch.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset squad \
    --batch-size 4 \
    --use-gate

# Train only cross-batch module, freeze base model
torchrun --nproc_per_node=8 scripts/train_cross_batch.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset squad \
    --batch-size 4 \
    --use-gate \
    --freeze-base-model

# Train on TriviaQA with similarity-based grouping (semantic clustering)
torchrun --nproc_per_node=8 scripts/train_cross_batch.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset triviaqa_sim \
    --batch-size 4 \
    --similarity-threshold 0.5

# Quick test with limited data (single GPU)
python scripts/train_cross_batch.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --max-samples 1000 \
    --eval-samples 100

# Force re-training (overwrite existing checkpoints)
torchrun --nproc_per_node=8 scripts/train_cross_batch.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset squad \
    --batch-size 4 \
    --force
```

Checkpoints are saved to `outputs/checkpoints/{dataset}/`:
- `Qwen_Qwen2.5-7B-Instruct_baseline.pt`
- `Qwen_Qwen2.5-7B-Instruct_crossbatch.pt`
- `Qwen_Qwen2.5-7B-Instruct_lora_lmhead.pt`
- `Qwen_Qwen2.5-7B-Instruct_lora_crossbatch.pt`

### Training Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Model name (default: `Qwen/Qwen2.5-0.5B-Instruct`) |
| `--dataset` | Dataset: `squad`, `hotpot`, `quac`, `drop`, `triviaqa`, `triviaqa_sim`, `cmb_clin`, `cmb_exam_context`, `cmb_exam_subdomain`, `cmb_exam_random` |
| `--max-samples` | Training samples (default: all) |
| `--eval-samples` | Evaluation contexts (default: all) |
| `--min-questions` / `--max-questions` | Questions per context (default: 1-5) |
| `--epochs` | Training epochs (default: 1) |
| `--batch-size` | Per-GPU batch size (default: 8) |
| `--lr` | Learning rate (default: 1e-4) |
| `--save-dir` | Checkpoint directory (default: `outputs/checkpoints`) |
| `--force` | Force re-training even if checkpoint exists |
| `--use-gate` | Use Question-Aware Gating instead of scalar scale |
| `--freeze-base-model` | Freeze base model, only train cross-batch module |
| `--similarity-threshold` | Threshold for similarity-based grouping (default: 0.5) |
| `--embedding-model` | Sentence transformer for similarity grouping |

### Cross-Batch Module Options

| Mode | Description |
|------|-------------|
| **Scale mode** (default) | `H_out = H + scale * cross_batch_info` - Simple learnable scalar |
| **Gate mode** (`--use-gate`) | `H_out = H + gate(H, info) * cross_batch_info` - Question-aware gating decides when to use cross-batch info |

### Dataset Grouping Options

| Dataset | Description |
|---------|-------------|
| `triviaqa` | TriviaQA with random grouping |
| `triviaqa_sim` | TriviaQA with semantic similarity-based grouping (recommended for cross-batch training) |

### CMB Training Examples

The CMB dataset has two subsets with different training use cases:

| Dataset | Description |
|---------|-------------|
| `cmb_clin` | CMB-Clin clinical cases (only 74 test samples, not recommended for training) |
| `cmb_exam_context` | CMB-Exam with shared background grouping (recommended, from fzkuji/CMB-Exam-Grouped) |
| `cmb_exam_subdomain` | CMB-Exam grouped by medical specialty |
| `cmb_exam_random` | CMB-Exam random grouping (baseline) |

```bash
# Train on CMB-Exam with context grouping (recommended)
torchrun --nproc_per_node=8 scripts/train_cross_batch.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset cmb_exam_context \
    --batch-size 4 \
    --epochs 1

# Train on CMB-Exam with subdomain grouping
torchrun --nproc_per_node=8 scripts/train_cross_batch.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset cmb_exam_subdomain \
    --batch-size 4

# Train on CMB-Exam with random grouping (baseline)
torchrun --nproc_per_node=8 scripts/train_cross_batch.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset cmb_exam_random \
    --batch-size 4
```

### Evaluation with Trained Checkpoints

```bash
# Auto-discover checkpoints and evaluate all strategies
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
    --dataset squad \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --context-count 100 \
    --min-questions 3 \
    --max-questions 5 \
    --auto-checkpoints

# With vLLM comparison (non-finetuned strategies only)
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
    --dataset squad \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --context-count 100 \
    --auto-checkpoints \
    --use-vllm

# Manual checkpoint paths
python scripts/compare_strategies.py \
    --dataset squad \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --strategies "batch,collab_hidden" \
    --collab-hidden-checkpoint outputs/checkpoints/squad/Qwen_Qwen2.5-7B-Instruct_crossbatch.pt \
    --context-count 10
```

### Checkpoint Arguments for compare_strategies.py

| Argument | Description |
|----------|-------------|
| `--auto-checkpoints` | Auto-discover checkpoints based on model/dataset |
| `--checkpoint-dir` | Base directory for checkpoint discovery (default: `outputs/checkpoints`) |
| `--baseline-checkpoint` | Path to baseline (lm_head only) checkpoint |
| `--collab-hidden-checkpoint` | Path to cross-batch module checkpoint |
| `--lora-lmhead-checkpoint` | Path to LoRA + lm_head checkpoint |
| `--lora-crossbatch-checkpoint` | Path to LoRA + cross-batch checkpoint |

## Baseline Evaluation (eval_baselines.py)

Fast evaluation of baseline strategies using vLLM with multi-GPU parallelism. No training required.

### Supported Strategies

| Strategy | Description | Token Tracking |
|----------|-------------|----------------|
| `all_in_one` | All questions in one prompt | PromptTok = PromptTok_API |
| `sequential` | One question at a time, with previous answers concatenated | PromptTok_API > PromptTok (includes history) |
| `batch` | All questions answered in parallel (no history) | PromptTok = PromptTok_API |
| `collab_llm` | LLM-based dependency ordering with relevant answers | PromptTok_API includes dependency cost |

### Usage

```bash
# Evaluate all baseline strategies
python scripts/eval_baselines.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset squad \
    --eval-samples 100 \
    --strategies all_in_one,sequential,batch,collab_llm \
    --num-gpus 8

# Evaluate specific strategies with more questions per context
python scripts/eval_baselines.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset squad \
    --eval-samples 100 \
    --strategies sequential,collab_llm \
    --min-questions 5 \
    --max-questions 10
```

### Output Metrics

The script outputs a detailed summary table:

```
=== Results Summary ===
Strategy        |     EM |     F1 | Lenient | Q/Ctx | PromptTok |   GenTok | PromptTok_API | GenTok_API |   DepTok |  Latency
-----------------------------------------------------------------------------------------------------------------------------
all_in_one      |  0.783 |  0.897 |   0.947 |   3.7 |     346.8 |     54.5 |         346.8 |       54.5 |      0.0 |   0.46s
sequential      |  0.818 |  0.918 |   0.957 |   3.7 |     889.6 |     43.8 |        1020.7 |       43.8 |      0.0 |   0.38s
batch           |  0.826 |  0.920 |   0.957 |   3.7 |     889.6 |     43.6 |         889.6 |       43.6 |      0.0 |   0.14s
collab_llm      |  0.826 |  0.920 |   0.957 |   3.7 |     889.6 |     43.6 |        1050.2 |       49.3 |    160.6 |   0.16s
```

| Column | Description |
|--------|-------------|
| Q/Ctx | Average questions per context |
| PromptTok | Original input tokens (context + question only) |
| GenTok | Generated tokens |
| PromptTok_API | Actual API tokens (includes concatenated history for sequential/collab_llm) |
| GenTok_API | Actual generated tokens (includes dependency generation for collab_llm) |
| DepTok | Dependency generation tokens (collab_llm only) |

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--model` | HuggingFace model ID (default: `Qwen/Qwen2.5-7B-Instruct`) |
| `--dataset` | Dataset: `squad`, `hotpot`, `quac`, `drop`, `triviaqa`, `quality`, `cmb` |
| `--eval-samples` | Number of contexts to evaluate |
| `--strategies` | Comma-separated list of strategies |
| `--num-gpus` | Number of GPUs (auto-detected if not specified) |
| `--min-questions` / `--max-questions` | Questions per context (default: 3-5) |
| `--min-free-mem-gb` | Minimum free GPU memory in GB (default: 10) |
| `--cache` | Enable result caching |

## SFT-LoRA Training (train_and_eval_sft.py)

Train a LoRA adapter on the QA task and evaluate the trained model.

### Usage

```bash
# Train and evaluate
python scripts/train_and_eval_sft.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset squad \
    --epochs 3 \
    --train-samples 1000 \
    --eval-samples 100 \
    --num-gpus 8

# Evaluate only (requires trained checkpoint)
python scripts/train_and_eval_sft.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset squad \
    --eval-only \
    --checkpoint-path outputs/checkpoints/sft_lora/xxx

# Compare with baseline
python scripts/train_and_eval_sft.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset squad \
    --epochs 3 \
    --compare-baseline
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--model` | HuggingFace model ID |
| `--dataset` | Dataset for training and evaluation |
| `--train-samples` | Number of training samples |
| `--eval-samples` | Number of evaluation contexts |
| `--epochs` | Training epochs (default: 3) |
| `--batch-size` | Training batch size (default: 1) |
| `--lr` | Learning rate (default: 2e-4) |
| `--lora-r` | LoRA rank (default: 16) |
| `--lora-alpha` | LoRA alpha (default: 32) |
| `--eval-only` | Skip training, only evaluate |
| `--checkpoint-path` | Path to trained checkpoint |
| `--compare-baseline` | Also evaluate baseline (no LoRA) |

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