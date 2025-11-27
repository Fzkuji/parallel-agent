# Parallel Decoding Experiments

This project explores dependency-aware question answering on SQuAD, HotpotQA, and CMB using local LLM models. It contains core library code in `src/` and runnable scripts in `scripts/`.

## Project Structure

```
battlenet/
├── src/                    # Core library
│   ├── models.py          # Question, EdgeCandidate, etc.
│   ├── scheduler.py       # DependencyScheduler, HTML visualiser
│   ├── generators.py      # Dependency generators (Heuristic, LLM, BERT)
│   ├── inference.py       # Chat prompt building, answer extraction
│   ├── loaders.py         # SQuAD/HotpotQA data loading
│   ├── selection.py       # Cost-aware edge selection
│   ├── strategies/        # Strategy implementations
│   └── ...
├── scripts/               # Runnable scripts
│   ├── compare_strategies.py
│   ├── run_parallel.py
│   ├── test_bert_dependencies.py
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

For SQuAD, questions share a background; for HotpotQA, each question has its own context and strategies switch to multi-context mode automatically. Metrics reported per strategy: strict/lenient accuracy, prompt/gen tokens, latency, and batch count; averages are shown at the end.

Use the `--bert-*` flags (model name, thresholds, token caps, and cost weight) to tune attention-based dependencies; `--no-llm-deps` disables LLM edge generation.

### Key arguments

- `--dataset {squad,hotpot,cmb}`: choose dataset. Hotpot enables multi-context mode (each question has its own context). CMB is Chinese Medical Benchmark with clinical case analysis.
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

### SQuAD / HotpotQA (Short-form QA)
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
