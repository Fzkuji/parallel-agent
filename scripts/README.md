# scripts/ 索引

> 本目录是实验脚本的累积，含**当前主线**与大量**历史探索**。脚本被 import、被 `.sh`
> 和服务器 `git pull` 引用，**不要随意移动/重命名**（会断引用、断服务器同步）。本文件用于
> 区分哪些在用、哪些是历史。

## 当前主线（在用）

| 脚本 | 作用 |
|---|---|
| `bench_distract.py` | 主评测：bank_read（ours/ape）vs concat，distractor 累积曲线；含 SHARED_PREFIX/PURE_PASSAGE 开关 |
| `bench_longbench.py` | LongBench 多跳 QA 主表评测（Concat/APE/Ours/Oracle），含 flashrag gold 对齐 |
| `gen_think_traces.py` | 生成 think 自蒸馏轨迹（teacher 完整输入 → think+answer，F1≥0.5 过滤）|
| `train_think_distill.py` | think 自蒸馏 LoRA 训练 |
| `probe_readfrac.py` | read-fraction top-k 多 frac 扫描 |
| `bench_niah.py` | 单跳 NIAH 长上下文 |
| `bench_efficiency.py` / `bench_serving.py` / `bench_async_exit.py` | 效率/吞吐/提前退出基准 |
| `llm_judge.py` / `run_judge.py` | LLM-as-judge 答案正确性评测 |
| `vllm_oracle.py` | oracle（喂 gold）参考分 |

> APE 官方复现脚本在仓库根目录 `eval_ape_qwen3.py`（Qwen3+SDPA），另有 `../APE/`（服务器，官方 Llama flash 版）。

## 历史探索（保留，非当前主线）

**CSA 系列**（早期 pooled cross-segment attention，已被 bank-read 取代）：
`eval_csa.py` `eval_csa_hook.py` `eval_csda.py` `eval_cspf.py` `finetune_csa.py`
`finetune_csda.py` `finetune_cspf.py` `pretrain_csa.py` `pretrain_csa_distill.py`
`train_csa.py` `visualize_csa_attention.py`

**KV-share 系列**（早期共享 KV 尝试）：
`eval_kvshare.py` `eval_shared_kv.py` `finetune_kvshare.py` `finetune_kvshare_distill.py`
`probe_kvshare.py`

**cross-seq / split 早期版本**：
`exp_cross_seq_methods.py` `exp_cross_seq_v2.py` `exp_cross_seq_v3.py`
`exp_doc_split.py` `exp_hotpot_split.py` `compare_independent_vs_crossbatch.py`

**RAFT / multiquery 训练探索**：
`probe_raft.py` `raft_multiquery.py` `train_multiquery_lora.py`
`train_multiquery_lora_distill32b.patch.py` `cache_teacher32b_logits.py`

## 诊断 / probe（一次性，按需）

`analyze_attention.py` `analyze_batch_degradation.py` `analyze_cmb_distribution.py`
`analyze_squad_distribution.py` `diag_qwen3.py` `inspect_checkpoint.py`
`probe_cache.py` `probe_contrast.py` `probe_entity.py` `probe_forced_think.py`
`probe_gen.py` `probe_teacher.py`

## 一次性 eval（历史，按数据集/设定切分）

`baseline_pretrained.py` `baseline_sft.py` `cross_dataset_eval.py` `eval_20q_contexts.py`
`eval_batch_crosscache.py` `eval_batch_size_impact.py` `eval_controlled_question_count.py`
`eval_crosscache.py` `eval_headroom.py` `eval_multi_dataset.py` `eval_multiquery.py`
`eval_question_count_impact.py` `eval_question_grouping_impact.py`
`eval_sft_grouping_impact.py` `train_and_eval.py` `train_cross_batch.py` `train_lora_baseline.py`

## 测试 / debug

`test_crossbatch_baseline.py`（在仓库根目录）`test_logits_consistency.py` `test_qwen3_hook.py`
`test_shared_state.py` `debug_batch_size.py` `debug_crossbatch_vs_baseline.py`

## 图表 / 汇总

`generate_paper_figures.py` `plot_case_study.py` `plot_question_count_results.py`
`extract_case_study_data.py` `summarize_results.py`
（论文图主目录在仓库外 `../scripts/`，不上传）

## shell 驱动

`run_all_datasets.sh` `run_all_grouping_experiments.sh` `run_llama_cmb_full.sh`
`run_llama_squad_full.sh` `run_qwen_cmb_crossbatch.sh` `eval_when_ready.sh`
`train_and_validate.sh` `train_eval_chain.sh`

## 子目录

`overall/` `preliminary/` — 早期分组实验产物。
