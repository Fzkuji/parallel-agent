# Parallel Decoding Experiments with Qwen3

This project explores dependency‑aware question answering on SQuAD using a local Qwen/Qwen3‑4B model. It contains two main Python entrypoints:

## python.py

Core library with:

- `Question`, `DependencyScheduler`, HTML visualiser.
- Heuristic + LLM dependency generators, cost‑aware edge selection.
- SQuAD context grouping utilities.

Use it as a toolkit; scripts import from here.

## run_qwen_parallel.py

Dependency-aware pipeline:

1. Load SQuAD contexts (group by `context`).
2. Generate dependency graph (local Qwen by default; `--no-llm-deps` falls back to heuristics).
3. Run cost-aware edge selection.
4. Schedule batches per DAG layer; prompt Qwen per batch.
5. Produce EM/F1 metrics and optional HTML visualisations.

### Example

```bash
# Heuristic dependencies
python run_qwen_parallel.py \
  --model-name Qwen/Qwen3-4B \
  --context-count 1 \
  --min-questions 3 \
  --max-questions 3 \
  --no-llm-deps \
  --max-new-tokens 128  # allow longer generations

# LLM dependencies + HTML output
python run_qwen_parallel.py \
  --model-name Qwen/Qwen3-4B \
  --context-count 5 \
  --min-questions 3 \
  --max-questions 5 \
  --max-dependencies 2 \
  --cost-weight 0.015 \
  --max-new-tokens 128 \
  --html-dir outputs_html
```

## compare_strategies.py

Runs three strategies side-by-side on sampled contexts:

1. **sequential** – questions answered one-by-one (no reuse).
2. **full_batch** – model answers all questions in a single prompt.
3. **dependency_parallel** – DAG-driven batches (same as `run_qwen_parallel.py`).
4. **parallel_ideal** – dependency DAG with theoretical infinite-parallel batches.
5. **parallel_bert** – uses BERT self-attention weights to derive dependencies, then runs the same batch executor.
6. **parallel_bert_ideal** – theoretical upper bound for the attention-derived DAG.

For each context it prints per-strategy strict accuracy (`answer` JSON matches gold exactly), lenient accuracy (answer text contains the gold span), prompt/generation token usage, latency, and batch count; averages are reported at the end.

Use the `--bert-*` flags (model name, attention/dependency thresholds, token caps, and cost weight) to tune the attention-based strategies without affecting the LLM-driven dependency runs.

### Example

```bash
python compare_strategies.py \
  --model-name Qwen/Qwen3-4B \
  --context-count 3 \
  --min-questions 3 \
  --max-questions 5 \
  --max-new-tokens 128 \
  --json-out outputs_json/results.json
```

Additional flags mirror `run_qwen_parallel.py` (`--no-llm-deps`, `--max-dependencies`, `--cost-weight`, etc.).

### SQuAD (multi-GPU)

```bash
torchrun --nproc_per_node=8 compare_strategies.py \
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
torchrun --nproc_per_node=8 compare_strategies.py \
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

## Offline dependency experiments

`test_bert_dependencies.py` now packs every question into a single BERT encoder pass, aggregates token-to-token attentions, and converts those weights into dependency confidences. This reproduces the “question A attends strongly to question B” signal without running the full Qwen pipeline.

Example:

```bash
python test_bert_dependencies.py \
  --model-name bert-base-uncased \
  --context-count 4 \
  --attention-threshold 0.02 \
  --dependency-threshold 0.02 \
  --max-dependencies 3 \
  --show-attention-summary \
  --show-attention-matrix \
  --show-attention-summary
```

## Notes

- Requires `transformers`, `datasets`, `torch`, and optionally `openai` if using external APIs.
- Ensure you have enough GPU RAM (Qwen3‑4B needs ~10 GB).
- Fallback heuristics trigger automatically if the LLM fails to return valid JSON.
- Generation length now trims at the first EOS, so `GenTok` reflects actual answer length rather than `max_new_tokens`.
- Answers are returned as JSON (`{"answer": ...}`) so strict accuracy can be computed reliably; lenient accuracy still checks substring containment.
