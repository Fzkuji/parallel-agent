# 多轮并行编码设计（Multi-turn Parallel Encoding）

> 状态：设计稿，未实现。本文件记录推演结论，作为后续实现的依据。

## 1. 动机

我们的 bank-read（CrossKV）单轮方法：每段 passage 各自从位置 0 独立编码进共享 KV bank，
query 读 bank。优点是 prefill 便宜（N 个 O(L²) 并行，而非 Concat 的 O((N·L)²)）；
代价是跨段桥接在编码时被切断，靠解码时的 thinking 补回。

APE（ICLR 2025）同属并行编码，但它是**纯单轮、一次性**结构：prefix + 全部 context 并行编码
+ query 全读 → 出一次答案即结束。**APE 不处理多轮对话**，而且它"故意隔绝 context"的设计
与"多轮历史必须连通"是冲突的。

本设计把单轮 bank-read 扩展为**会话级累积**的多轮结构，这是 APE 明确没做、且与其设计冲突的
增量创新点。

## 2. 核心规则：谁"睁眼"、谁"蒙眼"

判据只有一个——**算这段 KV 时，它的注意力能覆盖到谁**。

| 内容 | 编码方式 | 算它时能注意到 |
|---|---|---|
| 每轮的文章（passage） | 蒙眼（独立并行编码） | 只有自己 + 自己的 sink |
| 每轮的问题（question） | 睁眼（因果） | 此刻 bank 里已累积的全部：本轮文章 + 之前各轮的文章/问题/回答 |
| 每轮的回答（answer） | 睁眼（因果） | 全部 + 它自己这轮的问题 |

**只有"文章"是蒙眼的；问题和回答都睁眼、都能看见前面累积的一切。**
跨轮的信息汇总，全部发生在"问题/回答"这少量睁眼 token 的注意力里，**不在文章编码里**。
文章 token 数远多于问题/回答，所以成本主要还是省的。

## 3. 位置布局

每段文章仍各自从位置 0 独立编码（并行、可复用的前提）。位置区分只在 query/回答**读取**时给。
轮内文章共用同一段位置区间（重叠），轮间往后错开 —— 轮次靠位置先后隐式编码。

以 2 轮为例：

```
pos 0        : 第1轮 sink (<|im_start|>)
pos 1 ~ X    : 第1轮 N 篇文章   （并行，共用 [1, X]，X = max(段长)，短段右对齐补齐）
pos X+1 ~ A  : 第1轮 问题       （因果，读 sink + 第1轮文章）
pos A+1 ~ Y  : 第1轮 回答       （因果，读第1轮文章 + 问题）
pos Y+1      : 第2轮 sink       （N 篇文章共用同一位置 → 等效 1 个 sink）
pos Y+2 ~ Z  : 第2轮 N 篇文章   （并行，共用 [Y+2, Z]）
pos Z+1 ~ B  : 第2轮 问题       （因果，读【全部】两轮文章 + 第1轮问答）
pos B+1 ~    : 第2轮 回答       （因果，读全部 + 第2轮问题）
```

要点：
- 第 t 轮文章**蒙眼**，不因"接在前面后面"就去看前面 —— 仍只看自己。
- 跨轮连接由第 t 轮的**问题/回答**（睁眼）完成。
- 每轮 N 篇文章共用同一位置的 sink → 每轮等效 1 个 sink，不是 N 个（顺手规避了
  parallel encoding 的"N 个复制 sink"问题，与 APE shared-prefix 殊途同归）。
- sink 行为：每轮文章编码时前面有 sink 垫着，比"裸文章从位置 0"更接近训练分布。

### sink 的实现细节
每段文章 = `[sink token] + [文章内容]`，并行编码；同轮 N 段的 sink 位置重叠（都在该轮起点）。

## 4. 实现可行性（基于现有代码）

核心注意力代码 `src/cross_batch/batch_crosscache.py` 一行不用改。要新增的都是**外层编排**：

1. **bank 累积**：capture 阶段已是 `torch.cat([prev, new])` 追加语义（`bseq` 一并 cat）。
   每轮调一次 capture，新文章 KV 自动并入 bank。**已支持。**
2. **位置分段**：位置由外层喂入（`bench_distract.py` 的 `pid = (qattn.cumsum-1).clamp + off`）。
   多轮 = 让 `off` 按轮递推（第 t 轮 off = 前面全部内容长度）。**改外层 off 递推，不碰核心。**
3. **回答入 bank**：每轮回答生成完，其 KV 并入 bank 并打"回答段"标记，供下轮读。**需新增（外层加法）。**
4. **蒙眼/睁眼**：文章在 capture 各自因果（per-segment）；问题/回答在 use 读全部 bank。**已是现成两套规则。**

**结论：能实现，无结构性障碍。**

## 5. Selective read：用"拼接"，不用 mask

需要只读部分段（selective）时，**不要**用 mask（把全部段放进去再屏蔽 → SDPA 退回非 flash）。
改用**拼接式**（与 APE 的 stitch 同款）：

> 先挑出要读的段 → `torch.cat` 成一条连续 K/V → 喂模型做无 mask 注意力 → 走 flash。

我们的 bank 按段分散存（每段带 `bseq`），天然支持按段挑选拼接，比 APE「已 stitch 成整块」
更灵活。现有代码的 `allowed`/`block` 是旧的 mask 路径；selective 应改为拼接路径。

代码里 `bK = bK[idx]` 的 SELECTIVE GATHER 注释已是这个雏形（物理删段、QK 只在选中子集上算、真省 FLOP）。

## 6. Flash attention 支持

我们用 **SDPA**（非 `flash_attn` 库，因 Qwen3 QK-norm 与官方 flash patch 不兼容）。
SDPA 在**无显式 mask** 时自动走其内置 FlashAttention 后端。

| 场景 | 能否 flash |
|---|---|
| 文章独立编码（capture, is_causal） | ✅ |
| query 读全部 bank、无 selective/无 APE 温度 | ✅（快路，丢 mask） |
| 多轮"读全部 + 位置区分轮次" | ✅ |
| selective 用**拼接**实现（挑段→cat→无 mask） | ✅ |
| selective 用 **mask** 实现（旧路径） | ❌ |
| batch 内多样本**各读各的** + padding | ❌（要 mask） |

**与 APE 同口径**：APE 读全部、无 selective，也靠"无 mask 才 flash"。读全部时我们不比它差；
mask 只在我们想多做 selective/多轮时出现，是增量不是欠债。

## 7. 吞吐：多样本各读各的怎么办

不强求塞进一个定长 batch tensor。"各读各的"的标准解法：

- **batch=1（真实服务）**：挑段→拼接→flash，无任何额外代价。主场景。
- **多样本读相同段**：普通 batch，本来就 flash。
- **多样本各读各的**：
  - 朴素：一个一个（或几个一组）分开算，每个都是无 mask flash；代价是吞吐略低。
  - 工业级：**continuous batching + PagedAttention + varlen flash kernel**
    —— 不 padding、请求随到随走、每请求一张页表（= 我们的 bseq 索引映射）、
    varlen kernel 一次处理多个不等长序列且各读各的页、全程 flash。
    这正是 vLLM 的核心机制，思想与我们 bank 分段存储 + 索引映射完全一致。

**落地建议**：若最终走 **vLLM 部署**，continuous batching / paged / prefix-caching 等吞吐机制
自动到手（vLLM 原生支持 Qwen3，已处理 QK-norm）。真正的工程是把我们的并行编码 bank 逻辑
接进 vLLM，而非手写 varlen kernel。

## 8. 待落地清单

- [ ] 外层 `off` 多轮递推 + 回答 KV 入 bank（打回答段标记）
- [ ] selective 从 mask 路径改为拼接路径（对齐 APE stitch）
- [ ] 多轮训练数据（现蒸馏数据全是单轮 hotpot）：构造多轮、跨轮指代样本
- [ ] 多轮评测集：验证"第 t 轮确实用上了前面轮的文档/问答"
- [ ] 速度实测：ours(SDPA) vs APE，同模型同输入，prefill + decode 分别计时（尚无数据，需测）
- [ ] （可选）vLLM 部署路径：bank 逻辑接入，继承吞吐机制
