# CrossKV — Thinking-Augmented Parallel Context Encoding

Independent per-passage KV encoding (cheap, parallel-prefill inference) made to match
full-attention quality on multi-hop QA via a 3-stage training pipeline on Qwen3-8B.

## Method: 3 stages

### A. Context construction (independent encoding)
Each passage is encoded **independently from RoPE position 0** into a shared KV bank
(`capture` phase). A query reads the whole bank (`use` phase). Inference stays at
parallel-encoding cost (no long concatenated sequence). Inherited from PCW/APE.
- **Our tweak (CASA, optional, gated):** replace APE's two fixed scalars (temperature,
  scale) with a per-passage capture-time **key-norm-derived additive bank-logit bias** that
  lifts passages whose key-norms collapsed (the independent-encoding failure mode). APE is the
  special case bias=0. Additive+clipped → can only help. `--casa-c 0` reproduces APE exactly.

### B. Distill (performance recovery) — cross-encoding-mode self-distillation
- **Teacher** = the SAME Qwen3-8B but **full-attention over gold passages + thinking**;
  generates `<think>…</think><answer>…</answer>` trajectories (84% correct, kept only the correct).
- **Student** = the bank-read (independent encoding) over the SAME gold passages, teacher-forced
  to reproduce the full think+answer while reading the **independent** bank. LoRA r16 q/k/v/o.
- The full-attention "self" teaches the independent-encoding "self" to re-derive the dropped
  cross-passage bridge via generation-time thinking.
- **Our tweak (Bridge-Attention KD, optional, gated):** small auxiliary loss aligning the
  student's per-passage bank-attention to the teacher's, gated to bridge tokens, λ annealed→0.

### C. GRPO / RAFT (reinforce the deployed read ability)
After distill recovers gold-only performance, reinforce reading **all passages with
distractors** (the deployed setting). Reward = answer correctness; can exceed the teacher.
- **Our tweak (read-fraction Pareto reward, the headline original point):** among already-correct
  rollouts, prefer the one that read the **smallest fraction of the bank** (via the per-query
  `allowed` top-k mask). Strictly lexicographic (correctness first) → never below vanilla RAFT.
  This reward is **only definable** because we own an explicit per-context KV bank — concat
  cannot express "read a sub-bank". Uncontested lane: Block-Attention/KVLink/KV-Fusion read the
  full bank, APE/PCW have no learned selection.

## Scripts

| file | role |
|---|---|
| `src/cross_batch/batch_crosscache.py` | the bank: `capture`/`use`, `set_realign` (APE), `set_allowed` (selective), `record_attn`. Qwen3 QK-norm applied. |
| `scripts/bench_longbench.py` | main eval. arms concat/oracle/ape/ape_oracle/ours/ours_oracle. `--think`, qa_f1 + preds dump. `oracle_passages` = gold answer OR question-entity bridge (yes/no-safe). |
| `scripts/eval_multiquery.py` | `independent()` (concat/oracle), `build_prompt`, decode utils. |
| `scripts/gen_think_traces.py` | (B) vLLM: teacher full-attention+think trajectories over gold, keep correct. |
| `scripts/train_think_distill.py` | (B) self-distill: bank-read student reproduces think+answer over independent bank. |
| `scripts/train_multiquery_lora.py` | base LoRA reader training; `use_loss` (think_answer target supported), gold-only distill. |
| `scripts/raft_multiquery.py` | (C) RAFT/GRPO: sample N think rollouts over the bank, reward correctness, SFT winners. `--think`. |
| `scripts/llm_judge.py` + `run_judge.py` | Qwen2.5-32B SimpleQA judge (semantic correctness; strips `<think>`). Report alongside qa_f1. |
| `scripts/vllm_oracle.py` | fast vLLM generation for full-attention arms / teacher traces only (NOT the bank hook). |
| `scripts/bench_distract.py` | distractor-accumulation curve (SubEM), `bank_read`. |
| `scripts/probe_*.py`, `diag_qwen3.py`, `test_qwen3_hook.py` | one-off diagnostics (Qwen3 hook consistency, forced-think, teacher ceiling). |

## Current results (Qwen3-8B, LongBench 2wiki 100q, qa_f1, --think, fixed oracle)

| arm | encoding | selection | read | 2wiki |
|---|---|---|---|---|
| oracle | full-attn | gold | — | **59.5** (true upper bound) |
| ours_oracle | independent | gold | distilled | **59.5** (ties oracle) |
| ape_oracle | independent | gold | no-train | 57.2 |
| concat | full-attn | all+distractors | — | 38.6 |
| **ours** | independent | all+distractors | distilled | **35.0** (deployed; the gap C attacks) |
| ape | independent | all+distractors | no-train | 28.7 |

Key reads: distill ties the true oracle on gold-only (ours_oracle 59.5 = oracle 59.5);
distill also helps read-all (ape 28.7 → ours 35.0, +6.3); the remaining gap is reading **all
passages under distractors** (35.0 vs 59.5), which stage C targets. (hotpot oracle still needs a
supporting-facts-based fix; 2wiki is clean.)
