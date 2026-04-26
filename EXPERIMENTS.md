# CSA-v2 实验运行手册（H20 8 卡服务器）

## 环境

```bash
pip install torch transformers datasets peft accelerate
```

## 一句话总览

跑这条命令链，3-4 小时拿到完整 paper-ready 数据：

```bash
# 1. Warm-up: CSA 在简单 SQuAD grouped 上学基础协作 (~30 min on 8 GPUs)
torchrun --standalone --nproc_per_node=8 scripts/train_csa.py \
    --dataset squad --model-path /path/to/Qwen2.5-7B-Instruct \
    --num-train-groups 4000 --questions-per-group 4 \
    --epochs 2 --no-train-lm-head \
    --output-dir ./out/csa_warmup

# 2. Finetune: 用 warm-up 权重 init，迁移到 distributed-context HotpotQA (~60 min on 8 GPUs)
torchrun --standalone --nproc_per_node=8 scripts/train_csa.py \
    --dataset dhotpot --model-path /path/to/Qwen2.5-7B-Instruct \
    --num-train-groups 4000 --n-agents 4 --paragraphs-per-agent 9 \
    --epochs 3 \
    --csa-init-from ./out/csa_warmup/best_model.pt \
    --output-dir ./out/dhotpot_csa_warm

# 3. Eval (~15 min, 单卡跑足够)
python scripts/eval_csa.py \
    --dataset dhotpot --model-path /path/to/Qwen2.5-7B-Instruct \
    --checkpoint ./out/dhotpot_csa_warm/best_model.pt \
    --num-eval-groups 200 --n-agents 4 --paragraphs-per-agent 9 \
    --output-dir ./out/eval_dhotpot_warm
```

## 实验设计逻辑

之前在单卡跑下来的关键发现：

| Setup | overall EM | has_supp | no_supp | Δ (CSA on vs off) |
|---|---|---|---|---|
| Finetuned baseline (lm_head only) | 20.50 | 36.75 | 4.25 | — |
| no-LoRA + lm_head + CSA | 21.25 | 37.25 | 5.25 | **+1.88** |
| LoRA + lm_head + CSA | 19.00 | 33.50 | 4.50 | -0.88 |
| CSA-only (no LoRA, no lm_head) | 15.62 | 29.50 | 1.75 | **-5.38** ← 反直觉 |

**核心问题**：CSA 直接训会让 train loss 大降但 EM 反而恶化。CSA 学到的是概率分布上的 hack，不是真 evidence transfer。

**修复 (本次代码改动)**：
1. **CSA 输出加 LayerNorm (`cross_norm`)**: 让 CSA 注入的 hidden 跟 base last-layer 分布对齐，lm_head 不再迷路。
2. **Warm-up**: 先在 SQuAD grouped 这种简单"同 context 多 query"任务上让 CSA 学会基础 attention pattern (1-2 epoch, lm_head 冻结)，再 finetune 到带 distractor 的 dHotpot。
3. **DDP**: 8 卡数据并行，effective batch size 8x，训练快 8 倍。

## 主要 ablation

如果上面 1+2+3 都开有效（Δ ≥ +5 EM），可以补这些 ablation 拆分贡献：

```bash
# Ablation A: 不 warm-up (跳过 step 1，直接 dhotpot finetune)
torchrun --standalone --nproc_per_node=8 scripts/train_csa.py \
    --dataset dhotpot --model-path /path/to/Qwen2.5-7B-Instruct \
    --num-train-groups 4000 --epochs 3 \
    --output-dir ./out/dhotpot_csa_no_warm

# Ablation B: 关闭 cross_norm (需要单独改 attention.py 把 cross_norm 注释掉，然后跑)
# - 注释 attention.py:633 处 cross_info = self.cross_norm(cross_info)
# 跑 1+2 流程

# Ablation C: 训 lm_head（默认现在是冻结）
torchrun --standalone --nproc_per_node=8 scripts/train_csa.py \
    --dataset dhotpot --csa-init-from ./out/csa_warmup/best_model.pt \
    # 不加 --no-train-lm-head
    --output-dir ./out/dhotpot_csa_with_lmhead

# Ablation D: 不同 G (n_agents=2/4/6/8)
for G in 2 4 6 8; do
    torchrun ... --n-agents $G --output-dir ./out/dhotpot_g${G}
done
```

## 显存预估 (H20, 96GB/卡)

- 7B base bf16 + grads (only LoRA/CSA/lm_head trainable): ~25 GB / 卡
- KV cache + activations (4 sequences × 1500 tokens × 28 layers): ~10 GB / 卡
- 总 ~35 GB / 卡，留 ~60 GB margin。8 卡同时训没压力。

## 单卡 fallback

如果只能用 1 卡，把 `torchrun --standalone --nproc_per_node=8` 换成 `python`：
```bash
python scripts/train_csa.py --dataset dhotpot ...
```
训练时间 ~8 倍（约 5-6 小时）。

## 检查点结构

`best_model.pt` 是个 dict，包含：
- `cross_batch_module`: CSA-v2 weights (~64M)
- `lm_head`: 如果训了 lm_head（默认）
- `lora`: 如果加了 `--lora-rank > 0`

evals 时按需加载（eval_csa.py 会自动检测）。

## 文件清单

核心代码（不要动）：
- `src/cross_batch/attention.py` — CSA-v2 with cross_norm
- `src/cross_batch/generator.py` — 推理 generator
- `src/cross_batch/trainer.py` — 训练 loop（支持 DDP/LoRA/lm_head 选项）
- `src/datasets/hotpot_distributed.py` — distributed-context HotpotQA loader

入口脚本：
- `scripts/train_csa.py` — 唯一训练入口（参数化 dataset/LoRA/init）
- `scripts/eval_csa.py` — 唯一评估入口

## 常见问题

**Q: torchrun 起不来？**
A: 确认 `LOCAL_RANK` env 在 process 里能读到。`--standalone` 会自动 set 这个。

**Q: HuggingFace 网络慢？**
A: 用 mirror：`export HF_ENDPOINT=https://hf-mirror.com`

**Q: 第一次跑下载 HotpotQA 慢？**
A: 第一次会下载 ~700MB，之后缓存。也可以用 `HF_HOME` 指定 cache 路径。

**Q: best_model.pt 怎么选的？**
A: 按 train loss `improvement` 选（CSA on loss vs CSA off loss 的差）。但注意之前发现这个 metric 跟 EM 不一致——必要时直接用 `final_model.pt`。
