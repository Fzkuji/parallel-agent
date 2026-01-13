# Preliminary Experiments: Query Relationship Analysis

验证问题关系建模的有效性。采用统一实验框架：**Oracle → Method (LLM-based) → Random**

## 实验框架

| Condition | Description | Purpose |
|-----------|-------------|---------|
| **Oracle** | 按数据集标注的最优顺序/分组 | 性能上界 |
| **Method (LLM-based)** | LLM 自动检测关系并优化顺序/分组 | 验证方法有效性 |
| **Random** | 完全打乱顺序/随机分组 | 性能下界 |

**Hypothesis**: Oracle ≥ Method > Random，且 Random 性能随问题数量 n 增加而下降

## 实验列表

### Exp 1: Answer Dependency (语义-强)

- **Dataset**: MoreHopQA (3-5 hop reasoning, 含 gold sub-questions/sub-answers 和 context)
- **RQ**: 问题分解和传递prior答案是否提升多跳推理性能？
- **Key Insight**: MoreHopQA的子问题已包含前序答案（如 Q2: "June 23, 1992 的数字之和？"）
- **Setup** (3个条件):

  | 条件 | 描述 | 示例 |
  |------|------|------|
  | gold | 只问最后一个子问题（答案已嵌入） | "June 23, 1992 的数字之和？" |
  | sequential | 逐步回答，用模型预测替换嵌入的答案 | "[模型预测的日期] 的数字之和？" |
  | main_question | 直接问主问题（不分解） | "导演X电影的人死于哪天的数字之和？" |

- **评估**: EM/F1
- **预期**: gold > sequential > main_question

### Exp 2a: Shared Context (语义-中)

- **Dataset**: SQuAD (同一段落多个问题)
- **RQ**: 共享同一上下文的问题放在同一 batch 是否互相促进？
- **Setup**:
  - Oracle: 同一段落的问题放一组
  - Random: 不同段落的问题混在一起
- **预期**: Oracle > Random（共享上下文带来信息复用）

### Exp 2b: Related Domain (语义-弱)

- **Dataset**: MATH (7 个数学领域)
- **RQ**: 相关领域的问题放在同一 batch 是否互相促进？
- **Setup**:
  - Oracle: 按数学领域分组
  - Method: Encoder embedding 相似度分组
  - Random: 跨领域随机混合
- **预期**: Oracle ≥ Method > Random

### Exp 3: Format Similarity (结构相关)

- **Dataset**: ARC-Challenge (科学选择题)
- **RQ**: 同类型题目放一起是否提升格式一致性？
- **Setup**:
  - Oracle: 选择题统一格式推理
  - Method: Rule-based 识别题型并分组
  - Random: 与其他题型混合
- **Metrics**: Accuracy, Format Consistency, Answer Validity
- **预期**: Format Consistency: Oracle ≈ Method >> Random

## 使用方法

### 使用本地模型（推荐，免费）

```bash
# 快速测试 - 使用 Qwen2.5-7B
python scripts/preliminary/run_all.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --use-local \
    --quick

# 使用其他本地模型
python scripts/preliminary/exp2a_shared_context.py \
    --model Qwen/Qwen3-8B \
    --use-local \
    --n-groups 50
```

### 使用 API

```bash
# 快速测试（少量样本）
python scripts/preliminary/run_all.py --model gpt-4o-mini --quick

# 标准运行
python scripts/preliminary/run_all.py --model gpt-4o-mini

# 完整运行（更多样本）
python scripts/preliminary/run_all.py --model gpt-4o --full
```

### 运行单个实验

```bash
# Exp 1: Answer Dependency (本地模型)
python scripts/preliminary/exp1_answer_dependency.py \
    --models Qwen/Qwen2.5-7B-Instruct \
    --use-local \
    --use-vllm \
    --n-samples 50 \
    --conditions gold,sequential,main_question

# Exp 2a: Shared Context
python scripts/preliminary/exp2a_shared_context.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --use-local \
    --n-groups 50 \
    --conditions oracle,random

# Exp 2b: Related Domain
python scripts/preliminary/exp2b_related_domain.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --use-local \
    --n-per-domain 20 \
    --conditions oracle,method,random

# Exp 3: Format Similarity
python scripts/preliminary/exp3_format_similarity.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --use-local \
    --n-samples 100 \
    --conditions oracle,method,random
```

## 依赖

```bash
# 基础依赖
pip install datasets sentence-transformers scikit-learn tqdm

# 本地模型需要
pip install torch transformers accelerate
```

API 模式需要设置 key：

```bash
export OPENAI_API_KEY=your-key
# 或
export OPENROUTER_API_KEY=your-key
```

## 输出

结果保存到 `outputs/preliminary/` 目录，格式为 JSON：

```json
{
  "config": {...},
  "timestamp": "20250111_120000",
  "results": [
    {
      "condition": "oracle",
      "accuracy": 0.85,
      "metrics": {...},
      "details": [...]
    },
    ...
  ],
  "summary": {
    "oracle": {"accuracy": 0.85},
    "method": {"accuracy": 0.80},
    "random": {"accuracy": 0.45}
  }
}
```

## 文件结构

```
scripts/preliminary/
├── __init__.py              # 模块初始化
├── utils.py                 # 通用工具函数
├── exp1_answer_dependency.py    # Exp 1: MoreHopQA
├── exp2a_shared_context.py      # Exp 2a: SQuAD
├── exp2b_related_domain.py      # Exp 2b: MATH
├── exp3_format_similarity.py    # Exp 3: ARC-Challenge
├── run_all.py               # 统一运行脚本
└── README.md                # 本文件
```
