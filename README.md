# Parallel Decoding Experiments

This project explores dependency-aware question answering and cross-batch generation on SQuAD, HotpotQA, CMB, and other datasets using local LLM models. It contains core library code in `src/` and runnable scripts in `scripts/`.

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
│       └── eval.py            # SQuAD evaluation for cross-batch
├── scripts/                   # Runnable scripts
│   ├── compare_strategies.py  # Multi-strategy comparison
│   ├── run_parallel.py        # Single-context dependency pipeline
│   ├── test_bert_dependencies.py # BERT-based dependency experiments
│   └── debug_batch_vs_sequential.py
└── README.md
```

## scripts/compare_strategies.py

Runs multiple strategies side-by-side on sampled contexts:

1. **all_in_one** – single prompt with all questions (paired with their contexts).
2. **sequential** – one question per turn (history grows).
3. **batch** – all questions answered in one batch.
4. **parallel** – dependency DAG (LLM edges).
5. **parallel_bert** – dependency DAG (BERT attention edges).
6. **cross_batch** – parallel generation with cross-batch attention for information sharing.

For SQuAD, questions share a background; for HotpotQA, each question has its own context and strategies switch to multi-context mode automatically. Metrics reported per strategy: strict/lenient accuracy, prompt/gen tokens, latency, and batch count; averages are shown at the end.

Use the `--bert-*` flags (model name, thresholds, token caps, and cost weight) to tune attention-based dependencies; `--no-llm-deps` disables LLM edge generation.

### Key arguments

- `--dataset {squad,hotpot,quac,cmb,quality,drop}`: choose dataset. Hotpot enables multi-context mode (each question has its own context). CMB is Chinese Medical Benchmark with clinical case analysis. QuALITY is long-context reading comprehension (~5000 words per article). DROP requires discrete reasoning (arithmetic, counting, sorting).
- `--model-name`: HF model id or local path.
- `--context-count`: number of sampled groups/steps.
- `--min-questions / --max-questions`: number of questions per group.
- `--hotpot-subset`: HotpotQA subset (e.g., `distractor`).
- `--cmb-subset`: CMB subset (default: `CMB-Clin`).
- `--max-new-tokens`: generation cap.
- `--json-out`: write summary JSON; on multi-GPU a single merged file is produced.
- `--log-level`: logging verbosity.
- `--strategies`: comma-separated list of strategies to test (e.g., `--strategies "all_in_one,sequential,batch"`). Available: `all_in_one`, `sequential`, `batch`, `parallel`, `parallel_bert`. If not specified, all strategies will be tested.
- `--no-llm-deps`: disable LLM edge generation (parallel uses heuristics/attention only).
- `--max-dependencies`: cap edges per target.
- `--min-confidence`: minimum edge confidence.
- `--cost-weight` / `--total-cost-budget`: tune dependency selection cost.
- `--eval-model`: OpenRouter model ID for LLM-based evaluation (e.g., `openai/gpt-4o`). Requires `OPENROUTER_API_KEY` env var.

### SQuAD (multi-GPU)

```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset squad \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --context-count 10 \
  --min-questions 8 \
  --max-questions 8 \
  --max-new-tokens 1024 \
  --json-out outputs_json/results_squad.json \
  --log-level INFO
```

### HotpotQA (multi-GPU, multi-context)

```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset hotpot \
  --hotpot-subset distractor \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --context-count 1000 \
  --min-questions 4 \
  --max-questions 4 \
  --max-new-tokens 1024 \
  --json-out outputs_json/results_hotpot.json \
  --log-level INFO
```

### CMB (Chinese Medical Benchmark)

CMB uses BLEU-4 and ROUGE metrics automatically. Optionally add `--eval-model` for LLM-based evaluation (requires `OPENROUTER_API_KEY`).

```bash
export OPENROUTER_API_KEY=your_key_here  # Optional: for LLM evaluation

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
  --log-level INFO \
  --eval-model openai/gpt-4o  # Optional: requires OPENROUTER_API_KEY
```

### QuAC (Conversational QA)

QuAC is a conversational QA dataset where questions build on each other. Uses EM, F1, and Lenient metrics.

```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset quac \
  --split train \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --context-count 100 \
  --min-questions 3 \
  --max-questions 6 \
  --max-new-tokens 256 \
  --json-out outputs_json/results_quac.json \
  --log-level INFO
```

### QuALITY (Long-Context Reading Comprehension)

QuALITY is a challenging long-context multiple-choice reading comprehension dataset. Each article is ~5000 words with ~18 questions per article. Use `--quality-hard-only` to focus on difficult questions that require full document understanding.

```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset quality \
  --split validation \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --context-count 20 \
  --min-questions 5 \
  --max-questions 10 \
  --max-new-tokens 256 \
  --json-out outputs_json/results_quality.json \
  --log-level INFO
```

For hard questions only (recommended for testing complex reasoning):

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
  --json-out outputs_json/results_quality_hard.json \
  --log-level INFO
```

### DROP (Discrete Reasoning Over Paragraphs)

DROP is a reading comprehension benchmark requiring discrete reasoning (arithmetic, counting, sorting). Each passage has ~16 questions on average.

```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset drop \
  --split validation \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --context-count 50 \
  --min-questions 5 \
  --max-questions 10 \
  --max-new-tokens 128 \
  --json-out outputs_json/results_drop.json \
  --log-level INFO
```

## scripts/run_parallel.py

Single-context dependency pipeline with heuristic/LLM edges, cost-aware scheduling, and optional HTML output. Useful for debugging or exploring dependency-based scheduling on individual contexts.

```bash
python scripts/run_parallel.py \
  --model-name Qwen/Qwen3-4B \
  --context-count 1 \
  --min-questions 3 \
  --max-questions 5
```

## scripts/test_bert_dependencies.py

Offline BERT-based dependency experiments. Packs every question into a single BERT encoder pass, aggregates token-to-token attentions, and converts those weights into dependency confidences.

```bash
python scripts/test_bert_dependencies.py \
  --model-name bert-base-uncased \
  --context-count 4 \
  --attention-threshold 0.02 \
  --dependency-threshold 0.02 \
  --max-dependencies 3 \
  --show-attention-summary \
  --show-attention-matrix
```

## Notes

- Requires `transformers`, `datasets`, `torch`.
- Optional dependencies:
  - `nltk` for BLEU-4 metric (fallback implementation available)
  - `rouge-chinese` + `jieba` for Chinese ROUGE metrics (fallback implementation available)
  - `httpx` or `requests` for LLM-based evaluation via OpenRouter API
- Ensure you have enough GPU RAM (Qwen3-4B needs ~10 GB).
- Fallback heuristics trigger automatically if the LLM fails to return valid JSON.
- Generation length now trims at the first EOS, so `GenTok` reflects actual answer length rather than `max_new_tokens`.
- Answers are returned as JSON (`{"answer": ...}`) so strict accuracy can be computed reliably; lenient accuracy still checks substring containment.

## Evaluation Metrics

Metrics are automatically selected based on dataset:

### SQuAD / HotpotQA / QuAC / QuALITY / DROP (Short-form QA)
| Metric | Description |
|--------|-------------|
| **EM (Strict)** | Exact match after normalization |
| **F1** | Token-level F1 score |
| **Lenient** | Bidirectional substring containment |

### CMB (Long-form Medical QA)
| Metric | Description |
|--------|-------------|
| **BLEU-4** | 4-gram precision with brevity penalty |
| **ROUGE-1** | Unigram overlap F1 |
| **ROUGE-2** | Bigram overlap F1 |
| **ROUGE-L** | Longest common subsequence F1 |

### LLM Evaluation (Optional, via `--eval-model`)
| Dimension | Description |
|-----------|-------------|
| **Fluency** | Language quality and readability (1-5) |
| **Relevance** | How well the answer addresses the question (1-5) |
| **Completeness** | Coverage of key information (1-5) |
| **Proficiency** | Medical accuracy and terminology (1-5) |

## Cross-Batch Generation Module

The `src/cross_batch/` module implements cross-batch attention mechanisms that enable information sharing between samples during parallel generation. This can improve answer quality for related questions by allowing the model to leverage context from other samples in the batch.

### Architecture

The cross-batch module uses an additive design: `H_out = H + scale * cross_batch_info`, where the original hidden state is preserved and only additional information from other samples is added.

**Key Components:**

- **CrossBatchAttention**: Multi-head attention mechanism for cross-sample information sharing
- **CrossBatchEmbeddingMixer**: Similarity-based mixer using cosine similarity for attention weights
- **CrossBatchGenerator**: Wrapper that hooks into the model's forward pass to apply cross-batch mixing

### Usage

#### Basic Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.cross_batch import CrossBatchGenerator, CrossBatchAttention

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create cross-batch module
cross_batch_module = CrossBatchAttention(hidden_size=model.config.hidden_size)

# Create generator
generator = CrossBatchGenerator(
    model=model,
    tokenizer=tokenizer,
    cross_batch_module=cross_batch_module,
    mix_layer=-1,  # Apply at last layer
)

# Generate with cross-batch interaction
prompts = ["Question 1: ...", "Question 2: ...", "Question 3: ..."]
encoded = tokenizer(prompts, return_tensors="pt", padding=True)

outputs = generator.generate(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"],
    max_new_tokens=50,
    enable_cross_batch=True,
)
```

#### Training the Cross-Batch Module

```python
from src.cross_batch import CrossBatchTrainer, train_cross_batch_module

# Quick training with convenience function
history = train_cross_batch_module(
    model_name="gpt2",
    mix_method="attention",
    num_epochs=3,
    batch_size=8,
    learning_rate=1e-4,
    save_dir="./checkpoints",
)

# Or use the trainer directly for more control
trainer = CrossBatchTrainer(
    model=model,
    tokenizer=tokenizer,
    cross_batch_module=cross_batch_module,
    train_lm_head=True,  # Also fine-tune lm_head
)

history = trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=3,
    batch_size=8,
    save_dir="./checkpoints",
)
```

#### Evaluation on SQuAD

```python
from src.cross_batch import SquadEvaluator, run_comparison_eval

# Run comparison evaluation
results = run_comparison_eval(
    generator=generator,
    tokenizer=tokenizer,
    batch_size=4,
    max_samples=100,
    max_new_tokens=32,
)

print(f"Cross-batch EM: {results['cross_batch']['metrics']['exact_match']:.2f}")
print(f"Standard EM: {results['standard']['metrics']['exact_match']:.2f}")
print(f"Improvement: {results['difference']['exact_match']:+.2f}")
```

### Cross-Batch Strategy in compare_strategies.py

The cross-batch strategy can be used in the main comparison script:

```bash
python scripts/compare_strategies.py \
  --dataset squad \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --strategies "batch,cross_batch" \
  --context-count 10 \
  --max-new-tokens 128
```

### Module API Reference

| Class | Description |
|-------|-------------|
| `CrossBatchAttention` | Multi-head attention for cross-batch mixing |
| `CrossBatchEmbeddingMixer` | Similarity-based cross-batch mixer |
| `CrossBatchGenerator` | Generation wrapper with cross-batch hooks |
| `CrossBatchTrainer` | Training for cross-batch module |
| `LMHeadOnlyTrainer` | Baseline trainer (no cross-batch) |
| `SQuADDataset` | Dataset class for training |
| `SquadEvaluator` | Evaluation on SQuAD dataset |
| `train_cross_batch_module` | Convenience training function |
| `run_comparison_eval` | Compare cross-batch vs standard generation |
