# 实验结果总表（2026-06-11）

协议：flashrag dev，每题 gold 段落 + 干扰段补足 n_paras，SubEM，n=50/seed。
no-think `max_new=32`，think `max_new=256~512`。
编码方式：concat（全注意力基线）/ ape（并行编码+temp0.9/scale0.9）/ ours（bank-read 并行编码）。
Qwen2.5 无原生 thinking（think 列与 nothink 实测一致，故只列 nothink）。
LoRA 均为合并后评测（concat/ape/ours 同权重，公平）。

---

## 表 1 Qwen2.5-7B-Instruct × 2wiki（4段，5 seeds）

| 微调 | 编码 | s0 | s1 | s2 | s3 | s4 | 均值 |
|---|---|---|---|---|---|---|---|
| 无 | concat | 68 | 58 | 62 | 60 | 60 | **61.6** |
| 无 | ape | 42 | 34 | 34 | 36 | 32 | 35.6 |
| 无 | ours | 52 | 40 | 40 | 42 | 52 | **45.2** |
| mix LoRA | concat | 70 | 58 | 74 | 70 | 68 | **68.0** |
| mix LoRA | ape | 52 | 46 | 58 | 48 | 40 | 48.8 |
| mix LoRA | ours | 60 | 44 | 56 | 54 | 50 | **52.8** |
| 纯CE LoRA | concat | 64 | – | – | – | – | |
| 纯CE LoRA | ape | 38 | – | – | – | – | |
| 纯CE LoRA | ours | 44 | – | – | – | – | |
| KL蒸馏 LoRA | concat | 58 | – | – | – | – | |
| KL蒸馏 LoRA | ape | 38 | – | – | – | – | |
| KL蒸馏 LoRA | ours | 46 | – | – | – | – | |

要点：mix 训练 ours +7.6（逐 seed +8/+4/+16/+12/−2 配对），concat 也 +6.4（LoRA 通用提升）。
纯CE 在本设置回退（52→44），KL 全面回退——两个配方弃用。

## 表 2 Qwen2.5 × 2wiki 干扰密度扫描（seed0）

| 微调 | 编码 | 4段 | 8段 | 16段 | 32段 |
|---|---|---|---|---|---|
| 无 | concat | 68 | 48 | 48 | 48 |
| 无 | ape | 42 | 34 | 28 | 26 |
| 无 | ours | 52 | 28 | 26 | 22 |
| mix LoRA | concat | 70 | 62 | 48 | 46 |
| mix LoRA | ape | 52 | 50 | 34 | 34 |
| mix LoRA | ours | 60 | 52 | 40 | 30 |

np8 另有 3 seeds：base ours 均值 27.3，mix ours 均值 44.0（Δ+16.7，最大增益密度点）。
要点：训练增益全密度有效（+8~+24），但 concat 抗稀释强于预期，32 段无交叉（gap 16）。

## 表 3 Qwen2.5 × 2wiki selective read（mix LoRA，ours，按注意力质量选段重解码）

| 段数 (seed) | 读全部 | 读1/2 | 读1/4 | 读1/8 |
|---|---|---|---|---|
| 8 (s0) | 52 | **54** | 50 | – |
| 16 (s0) | 40 | 40 | 36 | – |
| 32 (s0) | 30 | 30 | 30 | **34** |
| 32 (s1) | 32 | – | 32 | **34** |

要点：读取量砍 2~8 倍精度零损或小赚；np32 读 1/8 连续两 seed 反超。

## 表 4 Qwen2.5 × hotpot（4段，3 seeds）

| 微调 | 编码 | s0 | s1 | s2 | 均值 |
|---|---|---|---|---|---|
| 无 | concat | 40 | 50 | 46 | 45.3 |
| 无 | ape | 22 | 12 | 18 | 17.3 |
| 无 | ours | 20 | 12 | 18 | **16.7** |
| mix LoRA | concat | 44 | 48 | 50 | 47.3 |
| mix LoRA | ape | 22 | 22 | 26 | 23.3 |
| mix LoRA | ours | 32 | 22 | 32 | **28.7** |
| 纯CE LoRA | concat / ape / ours | 46 / 30 / 32 | – | – | |
| KL LoRA | concat / ape / ours | 48 / 24 / 30 | – | – | |

要点：训练增益 +12 稳定（+12/+10/+14），但 hotpot gap 仍 ~19。

## 表 5 Qwen2.5 × hotpot selective（mix LoRA，ours）

| 段数 (seed) | 读全部 | 读1/2 | 读1/4 |
|---|---|---|---|
| 8 (s0) | 26 | 24 | **28** |
| 16 (s0) | 12 | 12 | **20** |
| 16 (s1) | 8 | 8 | **12** |
| 16 (s2) | 18 | 18 | **30** |

要点：hotpot 干扰更脏，selective 增益更大且种子稳健（np16 三 seeds +8/+4/+12，均值 +8）。
2wiki np32 读1/8 三 seeds +4/+2/+4。规律：**selective 增益随干扰密度上升**。

## 表 6 Qwen3-8B × 2wiki（4段，seed0）——think × 微调 全交叉

| 微调 | 编码 | think | nothink |
|---|---|---|---|
| 无 | concat | 88 | 74 |
| 无 | ape | 8 | 32 |
| 无 | ours | 18 | 22 |
| think-distill LoRA | concat | **92** | 64 |
| think-distill LoRA | ape | 50 | 56 |
| think-distill LoRA | ours | **60** | **58** |

要点：
1. 训练救活 think 解码：ours think 18→60（+42），ape 8→50。
2. nothink 下 trained ours 58 vs concat 64：gap 仅 6，当前最接近 concat 的配置。

## 表 6b Qwen3 trained（think-distill）迁移 + 种子 + 密度（think / nothink）

| 设置 | concat | ape | ours | ours 的 think 增益 |
|---|---|---|---|---|
| 2wiki 4段 s0 | 92 / 64 | 50 / 56 | 60 / 58 | +2 |
| 2wiki 4段 s1 | 88 / 50 | 48 / 40 | 52 / 46 | +6 |
| 2wiki 8段 s0 | 90 / 60 | 52 / 52 | **56 / 40** | **+16** |
| hotpot 4段 s0 | 54 / 44 | 28 / 24 | **30 / 20** | **+10** |

要点（改写了"思考不兑现"的旧判断）：
1. **think 对 ours 的增益随难度/密度上升**：np4 +2~+6 → np8 +16、hotpot +10。简单设置掩盖了思考的桥接作用，难场景下思考确实在补桥。
2. ours think 在全部设置 ≥ ape think。
3. concat think 的增益更大（+26~+40），gap 在 think 模式仍然大（np8: 90 vs 56）。

## 表 7 Qwen3 退化机制诊断（bank-read 解码，max_run / 退化样本率，n=12）

| 模型 | 微调 | 模式 | max_run | 退化率 |
|---|---|---|---|---|
| Qwen2.5-7B | 无 | nothink / think | 1.0 / 1.0 | 0% / 0% |
| Qwen3-8B | 无 | nothink | 54.2 | 17% |
| Qwen3-8B | 无 | think | 80.3 | 25% |
| Qwen3-8B (fp32) | 无 | nothink / think | 54.2 / 80.4 | 17% / 25% |
| Qwen3-8B | think-distill | nothink | **1.0** | **0%** |
| Qwen3-8B | think-distill | think | 19.2 | 8% |

要点：退化 Qwen3 特有、精度无关（fp32=bf16）、训练在 nothink 下完全消除 / think 下大部分消除。

## 表 8 思考净贡献 A/B（Qwen2.5，同 2463 条 teacher-correct 轨迹、同脚本/lr/capture，唯一变量=target）

| 数据集 | 微调 | concat | ape | ours |
|---|---|---|---|---|
| 2wiki 4段 s0 | CoT臂（推理+答案）| 68 | 44 | 46 |
| 2wiki 4段 s0 | 答案臂（纯答案）| 70 | 26 | **60** |
| hotpot 4段 s0 | CoT臂 | 48 | 26 | **28** |
| hotpot 4段 s0 | 答案臂 | 48 | 8 | 24 |

种子复测（2wiki，ours，CoT vs 答案臂）：s0 46/60、s1 38/46、s2 50/52 →
**均值 44.7 vs 52.7，Δ=−8.0，三种子全负**；hotpot s1 20/20（域内 Δ≈0）。
要点（思考净贡献 = CoT − 答案臂）：**域外 −8 稳健，域内 ≈0**。
Qwen2.5（无原生 thinking）硬塞 CoT 蒸馏无净收益、域外有害；对照 Qwen3 think-distill
（think 增益 +2~+16 随难度涨）→ **思考的桥接收益依赖原生 thinking 能力**。
答案臂 3 种子 52.7 ≈ mix 配方 53.3（两者本质都是答案 CE，互相印证）。
评测协议：两臂统一 max_new=320（CoT 臂需走完推理，避免截断 artifact）。

## 表 9 Qwen3 训练增益配对（nothink，2wiki）

| 段数 | base ours | trained ours | Δ |
|---|---|---|---|
| 4 (s0) | 22 | 58 | +36 |
| 8 (s0) | 20 | 40 | +20 |

---

## 进行中（结果待补）

- Qwen2.5 A/B 种子复测（2wiki s1/s2 双臂、hotpot s1 双臂）
- **Qwen3 对称 A/B**：q3 think 轨迹剥答案训答案臂（3755 条，ckpt/q3_ansonly_distill，过夜）——Qwen3 上思考净贡献的干净对照
- musique：flashrag 缓存无 context 字段，待换数据源
