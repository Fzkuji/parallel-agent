# CSA-v2 实验运行手册（H20 8 卡服务器）

## 环境

```bash
pip install torch transformers datasets peft accelerate bitsandbytes
```

## ★ 推荐流程：FineWeb-Edu 全量预训练 → dHotpot 全量微调 → eval

CSA 不再是单层 post-hoc 的小模块。本流程把它做成 transformer 的常规组件：
- **每隔 4 层**（28 层模型 → 第 3,7,11,15,19,23,27 层）插入一次 CSA
- **共享同一个 CSA 模块** + 每层一个可学 scalar gate（α 初始为 0）
- **base model + CSA 全部参数都训**（continual pretraining + full FT）
- **训练目标 = next-token CE**（跟 base LLM 预训练一样）

参数总量 ≈ 7B + 60M（CSA shared）+ 7（per-layer α），CSA 引入的额外参数 < 1%。

```bash
# Step 1: 在 FineWeb-Edu 上做 continual pretraining (~30-60 min on 8 GPUs)
# 全量微调 base model + 共享 CSA，next-token CE loss
torchrun --standalone --nproc_per_node=8 scripts/pretrain_csa.py \
    --model-path /YOUR_PATH/Qwen2.5-7B-Instruct \
    --fineweb-path /mnt/data/zichuanfu/.cache/huggingface/datasets/HuggingFaceFW___fineweb-edu/sample-10BT \
    --max-groups 5000 --epochs 1 \
    --n-chunks 4 --chunk-tokens 1024 \
    --csa-every 4 \
    --base-lr 1e-5 --csa-lr 5e-4 \
    --output-dir ./out/csa_pretrain

# Step 2: dHotpot 全量微调（接 Step 1 的 ckpt 继续训）
torchrun --standalone --nproc_per_node=8 scripts/finetune_csa.py \
    --model-path /YOUR_PATH/Qwen2.5-7B-Instruct \
    --resume-from ./out/csa_pretrain/best_model.pt \
    --num-train-groups 4000 --n-agents 4 --paragraphs-per-agent 9 \
    --epochs 3 \
    --base-lr 5e-6 --csa-lr 2e-4 \
    --output-dir ./out/dhotpot_csa_pretrained

# Step 3: Eval (DDP 分片，~3 min)
torchrun --standalone --nproc_per_node=8 scripts/eval_csa_hook.py \
    --dataset dhotpot --model-path /YOUR_PATH/Qwen2.5-7B-Instruct \
    --checkpoint ./out/dhotpot_csa_pretrained/best_model.pt \
    --num-eval-groups 200 --n-agents 4 --paragraphs-per-agent 9 \
    --csa-every 4 \
    --output-dir ./out/eval_dhotpot_pretrained
```

为什么这条比之前的 single-layer post-hoc CSA 更合理：
- CSA 在 forward 过程中实际改写 hidden（不是事后加在最后一层）
- 多层注入 → cross-batch 信息有机会被后续层进一步处理
- 训练目标和 base LLM 一致（next-token CE）→ 没有 train/test mismatch
- 全量微调 → 不依赖"frozen lm_head 能消化 CSA 输出"这种脆弱假设

## 关键超参选择

| 参数 | Step 1 值 | Step 2 值 | 说明 |
|---|---|---|---|
| base_lr | 1e-5 | 5e-6 | 全量微调用小 LR 防破坏预训练 |
| csa_lr | 5e-4 | 2e-4 | CSA 是新模块，可以大点 |
| csa_every | 4 | 4 | 每 4 层插一次 CSA |
| 8-bit AdamW | 默认开 | 默认开 | 省 ~40GB optimizer state |
| Gradient ckpt | 默认开 | 默认开 | 省激活值显存，慢 ~30% |
| Warmup steps | 100 | 50 | 让 α 先慢慢 ramp up |

## 显存估算（H20 96GB/卡）

7B 全量微调 + bf16 weights + 8-bit AdamW + grad checkpointing：

| 项 | GB/卡 |
|---|---|
| Model bf16 | 14 |
| Gradients bf16 | 14 |
| 8-bit AdamW state | 14 |
| Activations (grad ckpt) | ~5 |
| CSA + buffers | ~1 |
| 合计 | ~48 |

预算 96GB，留约 50GB margin。

## 单卡 fallback

去掉 `torchrun --standalone --nproc_per_node=8`，换成 `python`：
```bash
python scripts/pretrain_csa.py ...
python scripts/finetune_csa.py ...
python scripts/eval_csa_hook.py ...
```
训练时间约 ×8。

## 检查点结构（新流程）

`best_model.pt` 是个 dict：
- `model`: full base model state_dict（~14GB bf16）
- `csa_module`: MultiLayerCSAModule state_dict（共享 CSA + per-layer α，~60MB）

## 文件清单

核心代码：
- `src/cross_batch/attention.py` — CSA-v2 模块（不动）
- `src/cross_batch/multi_layer_hook.py` — 多层 hook + per-layer α gate
- `src/cross_batch/pretrain_trainer.py` — Step 1 trainer（FineWeb 预训练）
- `src/cross_batch/finetune_trainer.py` — Step 2 trainer（dHotpot 微调）
- `src/datasets/fineweb_chunked.py` — Step 1 数据流
- `src/datasets/hotpot_distributed.py` — Step 2/3 dHotpot 加载

入口脚本：
- `scripts/pretrain_csa.py` — Step 1
- `scripts/finetune_csa.py` — Step 2
- `scripts/eval_csa_hook.py` — Step 3

## 旧流程（已废弃，保留作 ablation）

旧流程用 single-layer post-hoc CSA + frozen base model + warm-up→finetune。
对应代码：
- `src/cross_batch/trainer.py` — 旧训练器
- `src/cross_batch/distill_trainer.py` — MSE 蒸馏（已废弃）
- `scripts/train_csa.py` / `scripts/eval_csa.py` / `scripts/pretrain_csa_distill.py`

不再用作主线。如果要复现旧实验做 ablation，直接调旧脚本。

## 常见问题

**Q: bitsandbytes 装不上？**
A: 加 `--no-8bit-adam`，回退到 fp32 AdamW。显存会更紧（多 ~30GB）。

**Q: gradient checkpointing 报错 "use_reentrant"？**
A: 我们已经显式设 `use_reentrant=False`。如果还报错，加 `--no-grad-checkpoint`，但显存可能装不下。

**Q: torchrun 起不来？**
A: `--standalone` 自动 set `LOCAL_RANK`。如果环境没读到，手动 export。

**Q: HuggingFace 网络慢？**
A: `export HF_ENDPOINT=https://hf-mirror.com`

**Q: 第一次下 FineWeb 卡？**
A: sample-10BT 大约 30GB。本地路径用 `--fineweb-path`，避免每次重新下。

**Q: best_model.pt 怎么选的？**
A: 按 running average loss 选最低的。如果想用最后的状态，加载 `final_model.pt`。
