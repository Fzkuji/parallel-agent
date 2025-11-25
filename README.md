# Parallel Decoding Experiments

This project explores dependency-aware question answering on SQuAD and HotpotQA using local LLM models. It contains core library code in `src/` and runnable scripts in `scripts/`.

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

- `--dataset {squad,hotpot}`: choose dataset. Hotpot enables multi-context mode (each question has its own context).
- `--model-name`: HF model id or local path.
- `--context-count`: number of sampled groups/steps.
- `--min-questions / --max-questions`: number of questions per group (Hotpot uses this for bundle size when `--hotpot-group-size` is not set).
- `--hotpot-subset`: HotpotQA split (e.g., `distractor`).
- `--hotpot-group-size`: fixed number of Hotpot items per group (overrides min/max random size).
- `--max-new-tokens`: generation cap.
- `--json-out`: write summary JSON; on multi-GPU a single merged file is produced.
- `--log-level`: logging verbosity.
- `--strategies`: comma-separated list of strategies to test (e.g., `--strategies "all_in_one,sequential,batch"`). Available: `all_in_one`, `sequential`, `batch`, `parallel`, `parallel_bert`. If not specified, all strategies will be tested.
- `--no-llm-deps`: disable LLM edge generation (parallel uses heuristics/attention only).
- `--max-dependencies`: cap edges per target.
- `--min-confidence`: minimum edge confidence.
- `--cost-weight` / `--total-cost-budget`: tune dependency selection cost.

### SQuAD (multi-GPU)

```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset squad \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --context-count 10 \
  --min-questions 3 \
  --max-questions 20 \
  --max-new-tokens 1024 \
  --json-out outputs_json/results_squad.json \
  --log-level INFO
```

### HotpotQA (multi-GPU, multi-context)

```bash
torchrun --nproc_per_node=8 scripts/compare_strategies.py \
  --dataset hotpot \
  --hotpot-subset distractor \
  --hotpot-group-size 10 \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --context-count 10 \
  --min-questions 3 \
  --max-questions 20 \
  --max-new-tokens 1024 \
  --json-out outputs_json/results_hotpot.json \
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

- Requires `transformers`, `datasets`, `torch`, and optionally `openai` if using external APIs.
- Ensure you have enough GPU RAM (Qwen3-4B needs ~10 GB).
- Fallback heuristics trigger automatically if the LLM fails to return valid JSON.
- Generation length now trims at the first EOS, so `GenTok` reflects actual answer length rather than `max_new_tokens`.
- Answers are returned as JSON (`{"answer": ...}`) so strict accuracy can be computed reliably; lenient accuracy still checks substring containment.
