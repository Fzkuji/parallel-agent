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

For each context it prints per-strategy strict accuracy (`answer` JSON matches gold exactly), lenient accuracy (answer text contains the gold span), prompt/generation token usage, latency, and batch count; averages are reported at the end.

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

## Notes

- Requires `transformers`, `datasets`, `torch`, and optionally `openai` if using external APIs.
- Ensure you have enough GPU RAM (Qwen3‑4B needs ~10 GB).
- Fallback heuristics trigger automatically if the LLM fails to return valid JSON.
- Generation length now trims at the first EOS, so `GenTok` reflects actual answer length rather than `max_new_tokens`.
- Answers are returned as JSON (`{"answer": ...}`) so strict accuracy can be computed reliably; lenient accuracy still checks substring containment.
