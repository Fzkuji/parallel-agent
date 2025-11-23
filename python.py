from __future__ import annotations

import argparse
import json
import logging
import os
import random
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from datasets import Dataset, load_dataset
except ImportError:  # pragma: no cover - optional dependency
    Dataset = None  # type: ignore
    load_dataset = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "in",
    "on",
    "for",
    "with",
    "that",
    "which",
    "what",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "is",
    "are",
    "does",
    "do",
    "did",
    "was",
    "were",
    "how",
    "why",
    "请",
    "以及",
    "哪些",
    "什么",
}

REFERENCE_KEYWORDS = {
    "上述",
    "前文",
    "前面",
    "前一个",
    "这",
    "那",
    "它",
    "他们",
    "他",
    "她",
    "其",
    "这些",
    "those",
    "them",
    "it",
    "that",
    "previous",
}

AGGREGATE_KEYWORDS = {
    "总共",
    "总计",
    "一共",
    "合计",
    "总数",
    "平均",
    "列表",
    "列出",
    "罗列",
    "排序",
    "排名",
    "比较",
    "对比",
    "整体",
    "全部",
    "汇总",
    "aggregate",
    "list",
    "compare",
    "total",
    "average",
}


def estimate_tokens(text: str) -> int:
    ascii_tokens = text.strip().split()
    chinese_chars = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    approx = len(ascii_tokens) * 1.4 + len(chinese_chars) * 0.8
    if any(ch.isdigit() for ch in text):
        approx += 0.2 * sum(ch.isdigit() for ch in text)
    return max(1, int(approx))


def extract_keywords(text: str) -> Set[str]:
    normalized = []
    for ch in text.lower():
        if ch.isalnum():
            normalized.append(ch)
        else:
            normalized.append(" ")
    tokens = {
        tok for tok in "".join(normalized).split() if tok and tok not in STOPWORDS
    }
    tokens.update({ch for ch in text if "\u4e00" <= ch <= "\u9fff"})
    return tokens


def detect_reference_question(text: str) -> bool:
    return any(keyword in text for keyword in REFERENCE_KEYWORDS)


def detect_aggregate_question(text: str, type_hint: Optional[str]) -> bool:
    if type_hint and type_hint.lower() in {"aggregate", "list", "compare"}:
        return True
    return any(keyword in text for keyword in AGGREGATE_KEYWORDS)


@dataclass
class Question:
    qid: str
    text: str
    priority: float = 1.0
    answer_tokens: int = 32
    explicit_dependencies: Iterable[str] = field(default_factory=list)
    type_hint: Optional[str] = None
    tokens: int = field(init=False)
    dependencies: Set[str] = field(default_factory=set, init=False)
    references: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tokens = estimate_tokens(self.text)
        self.dependencies = set(self.explicit_dependencies)


@dataclass
class EdgeCandidate:
    source: str
    target: str
    confidence: float = 1.0
    rationale: Optional[str] = None


@dataclass
class BatchAssignment:
    batch_id: int
    question_ids: List[str]
    depth: int
    priority_sum: float
    value_score: float
    background_tokens: int
    incremental_prefill_tokens: int
    generation_tokens: int
    total_tokens: int
    estimated_latency: float


@dataclass
class ScheduleResult:
    batches: List[BatchAssignment]
    question_depths: Dict[str, int]
    dependency_graph: Dict[str, Set[str]]
    total_background_tokens: int
    total_incremental_prefill_tokens: int
    total_generation_tokens: int
    total_compute_tokens: int
    total_priority: float
    value_score: float
    total_estimated_latency: float


class DependencyScheduler:
    def __init__(
        self,
        background: str,
        questions: Iterable[Question],
        *,
        max_batch_tokens: Optional[int] = None,
        fmt_overhead_per_section: int = 6,
        prefill_token_cost: float = 1.0,
        generate_token_cost: float = 1.0,
    ) -> None:
        self.background = background
        self.background_tokens = estimate_tokens(background)
        self.questions: Dict[str, Question] = {q.qid: q for q in questions}
        self.max_batch_tokens = max_batch_tokens
        self.format_overhead = fmt_overhead_per_section
        self.prefill_token_cost = prefill_token_cost
        self.generate_token_cost = generate_token_cost
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.in_degree: Dict[str, int] = {qid: 0 for qid in self.questions}

    def build_dependencies(self, auto_infer: bool = True) -> None:
        ordered_ids = list(self.questions.keys())
        if auto_infer:
            for idx, qid in enumerate(ordered_ids):
                question = self.questions[qid]
                if detect_reference_question(question.text):
                    if idx > 0:
                        question.dependencies.add(ordered_ids[idx - 1])
                if detect_aggregate_question(question.text, question.type_hint):
                    prior_ids = ordered_ids[:idx]
                    current_kw = extract_keywords(question.text)
                    for prev in prior_ids:
                        prev_kw = extract_keywords(self.questions[prev].text)
                        if current_kw & prev_kw:
                            question.dependencies.add(prev)
        for qid, question in self.questions.items():
            question.dependencies = {
                dep for dep in question.dependencies if dep in self.questions and dep != qid
            }
        self._rebuild_graph()

    def _rebuild_graph(self) -> None:
        self.graph = defaultdict(set)
        self.in_degree = {qid: 0 for qid in self.questions}
        for qid, question in self.questions.items():
            for dep in question.dependencies:
                self.graph[dep].add(qid)
                self.in_degree[qid] += 1

    def describe_dependency_layers(self) -> List[List[str]]:
        indegree = dict(self.in_degree)
        ready = [qid for qid, deg in indegree.items() if deg == 0]
        layers: List[List[str]] = []
        visited = 0
        while ready:
            ready.sort()
            layer = ready
            layers.append(layer)
            next_ready: List[str] = []
            for node in layer:
                visited += 1
                for neighbor in self.graph.get(node, ()):
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        next_ready.append(neighbor)
            ready = next_ready
        if visited != len(self.questions):
            raise ValueError("Dependency graph contains cycles; cannot layer graph.")
        return layers

    def _cost_for_question(self, qid: str) -> Tuple[int, int]:
        question = self.questions[qid]
        dep_answers = sum(
            self.questions[dep].answer_tokens for dep in question.dependencies
        )
        dep_format = len(question.dependencies) * self.format_overhead
        incremental_prefill = question.tokens + dep_answers + dep_format
        return incremental_prefill, question.answer_tokens

    def schedule(self) -> ScheduleResult:
        ready: Set[str] = {qid for qid, deg in self.in_degree.items() if deg == 0}
        in_degree = dict(self.in_degree)
        question_depths: Dict[str, int] = {}
        batches: List[BatchAssignment] = []
        scheduled = 0
        batch_id = 0

        while ready:
            batch_ids = sorted(ready)
            incremental_prefill = 0
            generation_tokens = 0
            depth = 0
            priority_sum = 0.0
            value_score = 0.0
            max_generation = 0

            for qid in batch_ids:
                question = self.questions[qid]
                if question.dependencies:
                    dep_depths = [question_depths[dep] for dep in question.dependencies]
                    current_depth = max(dep_depths) + 1
                else:
                    current_depth = 0
                question_depths[qid] = current_depth
                depth = max(depth, current_depth)
                priority_sum += question.priority
                value_score += question.priority / (1.0 + current_depth)
                inc, gen = self._cost_for_question(qid)
                incremental_prefill += inc
                generation_tokens += gen
                max_generation = max(max_generation, gen)

            total_prefill = self.background_tokens + incremental_prefill
            total_tokens = total_prefill + generation_tokens
            estimated_latency = (
                self.prefill_token_cost * total_prefill
                + self.generate_token_cost * max_generation
            )

            batches.append(
                BatchAssignment(
                    batch_id=batch_id,
                    question_ids=batch_ids,
                    depth=depth,
                    priority_sum=priority_sum,
                    value_score=value_score,
                    background_tokens=self.background_tokens,
                    incremental_prefill_tokens=incremental_prefill,
                    generation_tokens=generation_tokens,
                    total_tokens=total_tokens,
                    estimated_latency=estimated_latency,
                )
            )
            batch_id += 1

            for qid in batch_ids:
                ready.remove(qid)
                scheduled += 1
                for neighbor in self.graph.get(qid, ()):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        ready.add(neighbor)

        if scheduled != len(self.questions):
            missing = set(self.questions) - set(question_depths)
            raise ValueError(f"Cyclic dependencies detected; unresolved: {missing}")

        total_background_tokens = len(batches) * self.background_tokens
        total_incremental = sum(batch.incremental_prefill_tokens for batch in batches)
        total_generation = sum(batch.generation_tokens for batch in batches)
        total_latency = sum(batch.estimated_latency for batch in batches)

        return ScheduleResult(
            batches=batches,
            question_depths=question_depths,
            dependency_graph={k: set(v) for k, v in self.graph.items()},
            total_background_tokens=total_background_tokens,
            total_incremental_prefill_tokens=total_incremental,
            total_generation_tokens=total_generation,
            total_compute_tokens=total_background_tokens + total_incremental + total_generation,
            total_priority=sum(q.priority for q in self.questions.values()),
            value_score=sum(batch.value_score for batch in batches),
            total_estimated_latency=total_latency,
        )

    def pretty_print_schedule(self, result: ScheduleResult) -> None:
        print("Dependency-aware parallel schedule (greedy)")
        print("=" * 60)
        for batch in result.batches:
            print(
                f"Batch {batch.batch_id} | depth {batch.depth} | questions {batch.question_ids}"
            )
            print(
                f"  tokens -> background {batch.background_tokens}, incremental {batch.incremental_prefill_tokens}, "
                f"generation {batch.generation_tokens}, total {batch.total_tokens}"
            )
            print(
                f"  priority sum {batch.priority_sum:.2f}, value score {batch.value_score:.2f}, est latency {batch.estimated_latency:.1f}"
            )
        print("-" * 60)
        print(
            f"Totals: background {result.total_background_tokens}, incremental {result.total_incremental_prefill_tokens}, "
            f"generation {result.total_generation_tokens}"
        )
        print(f"Overall compute tokens: {result.total_compute_tokens}")
        print(f"Total estimated latency: {result.total_estimated_latency:.1f}")
        print(f"Value score (higher better): {result.value_score:.2f}")
        print("=" * 60)


def export_schedule_html(
    scheduler: DependencyScheduler,
    result: ScheduleResult,
    path: Path,
    *,
    title: str = "Greedy 调度可视化",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        layers = scheduler.describe_dependency_layers()
    except ValueError:
        layers = [list(scheduler.questions.keys())]

    width = 900
    min_height = 420
    per_node_height = 110
    node_count = len(scheduler.questions)
    height = max(min_height, int(node_count * per_node_height * 0.75))
    x_margin = 110
    y_margin = 70

    layer_count = max(len(layers), 1)
    layer_spacing = (width - 2 * x_margin) / max(layer_count - 1, 1)

    positions: Dict[str, Dict[str, float]] = {}
    for layer_idx, layer in enumerate(layers):
        if layer_count == 1:
            x = width / 2
        else:
            x = x_margin + layer_idx * layer_spacing

        if len(layer) == 1:
            y_positions = [height / 2]
        else:
            span = max(height - 2 * y_margin, 1)
            step = span / (len(layer) - 1)
            y_positions = [y_margin + i * step for i in range(len(layer))]

        for idx, qid in enumerate(layer):
            positions[qid] = {
                "x": round(x, 2),
                "y": round(y_positions[idx], 2),
                "layer": layer_idx,
            }

    nodes_data = []
    for qid, question in scheduler.questions.items():
        pos = positions.get(qid, {"x": width / 2, "y": height / 2, "layer": 0})
        short_label = question.text.strip().replace("\n", " ")
        if len(short_label) > 16:
            short_label = f"{short_label[:14]}…"
        nodes_data.append(
            {
                "id": qid,
                "label": question.text.strip(),
                "short_label": short_label,
                "priority": question.priority,
                "answer_tokens": question.answer_tokens,
                "prompt_tokens": question.tokens,
                "dependencies": sorted(question.dependencies),
                "layer": pos["layer"],
                "x": pos["x"],
                "y": pos["y"],
                "depth": result.question_depths.get(qid, 0),
            }
        )

    edges_data = []
    for src, targets in result.dependency_graph.items():
        for tgt in targets:
            edges_data.append({"source": src, "target": tgt})

    batches_data = []
    for batch in result.batches:
        batches_data.append(
            {
                "id": batch.batch_id,
                "nodes": batch.question_ids,
                "depth": batch.depth,
                "value": round(batch.value_score, 3),
                "priority_sum": round(batch.priority_sum, 3),
                "tokens": {
                    "background": batch.background_tokens,
                    "incremental": batch.incremental_prefill_tokens,
                    "generation": batch.generation_tokens,
                    "total": batch.total_tokens,
                },
                "estimated_latency": round(batch.estimated_latency, 3),
            }
        )

    graph_data = {
        "title": title,
        "nodes": nodes_data,
        "edges": edges_data,
        "batches": batches_data,
        "metrics": {
            "total_background_tokens": result.total_background_tokens,
            "total_incremental_tokens": result.total_incremental_prefill_tokens,
            "total_generation_tokens": result.total_generation_tokens,
            "total_compute_tokens": result.total_compute_tokens,
            "total_estimated_latency": round(result.total_estimated_latency, 3),
            "value_score": round(result.value_score, 3),
        },
        "config": {"width": width, "height": height},
    }

    data_json = json.dumps(graph_data, ensure_ascii=False).replace("</", "<\\/")

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      margin: 20px;
      color: #1f2933;
      background-color: #f8fafc;
    }}
    h1 {{
      margin-bottom: 0.25rem;
    }}
    .metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
      font-size: 0.95rem;
    }}
    .metrics span {{
      background: #e2e8f0;
      padding: 6px 12px;
      border-radius: 999px;
    }}
    .slider {{
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .slider input[type=range] {{
      width: 320px;
    }}
    svg {{
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12);
    }}
    .edge {{
      stroke: #cbd5e1;
      stroke-width: 2px;
      opacity: 0.7;
    }}
    .edge.done {{
      stroke: #0ea5e9;
      opacity: 0.9;
    }}
    .edge.current {{
      stroke: #f97316;
      opacity: 0.95;
    }}
    .node {{
      stroke: #1e293b;
      stroke-width: 2px;
    }}
    .node.pending {{
      fill: #f1f5f9;
    }}
    .node.done {{
      fill: #0ea5e9;
    }}
    .node.current {{
      fill: #f97316;
    }}
    text {{
      font-size: 12px;
      text-anchor: middle;
    }}
    text.id {{
      font-weight: 600;
      font-size: 14px;
      fill: #0f172a;
    }}
    text.label {{
      fill: #1f2937;
    }}
    .info {{
      margin-top: 16px;
      padding: 12px;
      background: #e0f2fe;
      border-left: 4px solid #0284c7;
      border-radius: 4px;
      font-size: 0.95rem;
      max-width: 900px;
      white-space: pre-line;
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="metrics" id="metrics"></div>
  <div class="slider">
    <label for="batchSlider">批次选择：</label>
    <input type="range" min="0" id="batchSlider" />
    <span id="batchLabel"></span>
  </div>
  <svg id="graph"></svg>
  <div class="info" id="batchInfo"></div>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script>
    const data = {data_json};
    const svg = d3.select("#graph")
      .attr("width", data.config.width)
      .attr("height", data.config.height);

    const nodeById = Object.fromEntries(data.nodes.map(d => [d.id, d]));

    const edges = svg.selectAll("line")
      .data(data.edges)
      .enter()
      .append("line")
      .attr("class", "edge")
      .attr("x1", d => nodeById[d.source].x)
      .attr("y1", d => nodeById[d.source].y)
      .attr("x2", d => nodeById[d.target].x)
      .attr("y2", d => nodeById[d.target].y);

    const nodeGroups = svg.selectAll("g.node-group")
      .data(data.nodes)
      .enter()
      .append("g")
      .attr("class", "node-group")
      .attr("transform", d => `translate(${{d.x}}, ${{d.y}})`);

    nodeGroups.append("circle")
      .attr("r", 28)
      .attr("class", "node pending");

    nodeGroups.append("text")
      .attr("class", "id")
      .attr("y", -2)
      .text(d => d.id);

    nodeGroups.append("text")
      .attr("class", "label")
      .attr("y", 16)
      .text(d => d.short_label);

    nodeGroups.append("title")
      .text(d => `${{d.id}}\\n优先级: ${{d.priority}} | 回答 tokens: ${{d.answer_tokens}}`);

    const slider = document.getElementById("batchSlider");
    slider.setAttribute("max", data.batches.length);
    slider.value = 0;

    const batchLabel = document.getElementById("batchLabel");
    const batchInfo = document.getElementById("batchInfo");
    const metricsContainer = document.getElementById("metrics");

    const metricsEntries = [
      ["Value score", data.metrics.value_score],
      ["Compute tokens", data.metrics.total_compute_tokens],
      ["Background tokens", data.metrics.total_background_tokens],
      ["Incremental tokens", data.metrics.total_incremental_tokens],
      ["Generation tokens", data.metrics.total_generation_tokens],
      ["Estimated latency", data.metrics.total_estimated_latency]
    ];
    metricsEntries.forEach(([label, value]) => {{
      const span = document.createElement("span");
      span.textContent = `${{label}}: ${{value}}`;
      metricsContainer.appendChild(span);
    }});

    function update(step) {{
      const executed = new Set();
      for (let i = 0; i < step; i += 1) {{
        data.batches[i].nodes.forEach(n => executed.add(n));
      }}
      const current = new Set(step > 0 ? data.batches[step - 1].nodes : []);

      nodeGroups.selectAll("circle")
        .attr("class", d => {{
          if (current.has(d.id)) return "node current";
          if (executed.has(d.id)) return "node done";
          return "node pending";
        }});

      nodeGroups.selectAll("text.id")
        .attr("fill", d => current.has(d.id) ? "#ffffff" : executed.has(d.id) ? "#0b3d3d" : "#0f172a");

      nodeGroups.selectAll("text.label")
        .attr("fill", d => current.has(d.id) ? "#fff7ed" : executed.has(d.id) ? "#022c22" : "#1f2937");

      edges.attr("class", d => {{
        if (current.has(d.target)) return "edge current";
        if (executed.has(d.target)) return "edge done";
        return "edge";
      }});

      if (step === 0) {{
        batchLabel.textContent = "步骤 0：未执行任何批次";
        batchInfo.textContent = "拖动滑块以查看每个批次的节点、token 成本和估计延迟。";
      }} else {{
        const batch = data.batches[step - 1];
        const lines = [
          `批次 ${{batch.id}} ，节点：${{batch.nodes.join(", ")}}`,
          `深度：${{batch.depth}} | Value score：${{batch.value.toFixed(3)}} | 优先级和：${{batch.priority_sum.toFixed(3)}}`,
          `Token：background={{batch.tokens.background}}, incremental={{batch.tokens.incremental}}, generation={{batch.tokens.generation}}, total={{batch.tokens.total}}`,
          `估计延迟：${{batch.estimated_latency.toFixed(1)}}`
        ];
        batchLabel.textContent = `步骤 ${{step}}：批次 ${{batch.id}}`;
        batchInfo.textContent = lines.join("\\n");
      }}
    }}

    slider.addEventListener("input", (event) => update(Number(event.target.value)));
    update(0);
  </script>
</body>
</html>
"""

    path.write_text(html, encoding="utf-8")


class DependencyGraphGenerator:
    """Interface for dependency graph generation strategies."""

    def generate_edges(
        self,
        background: str,
        questions: Sequence[Question],
        metadata: Optional[dict] = None,
    ) -> List[EdgeCandidate]:
        raise NotImplementedError


class HeuristicDependencyGenerator(DependencyGraphGenerator):
    """Fallback generator based on simple lexical rules."""

    def generate_edges(
        self,
        background: str,
        questions: Sequence[Question],
        metadata: Optional[dict] = None,
    ) -> List[EdgeCandidate]:
        edges: List[EdgeCandidate] = []
        for idx, question in enumerate(questions):
            kw = extract_keywords(question.text)
            for prev in questions[:idx]:
                prev_kw = extract_keywords(prev.text)
                if kw and prev_kw and kw & prev_kw:
                    edges.append(
                        EdgeCandidate(
                            source=prev.qid,
                            target=question.qid,
                            confidence=0.55,
                            rationale="keyword-overlap",
                        )
                    )
            if detect_reference_question(question.text) and idx > 0:
                edges.append(
                    EdgeCandidate(
                        source=questions[idx - 1].qid,
                        target=question.qid,
                        confidence=0.85,
                        rationale="reference-token",
                    )
                )
            if detect_aggregate_question(question.text, question.type_hint):
                for prev in questions[:idx]:
                    edges.append(
                        EdgeCandidate(
                            source=prev.qid,
                            target=question.qid,
                            confidence=0.65,
                            rationale="aggregate-question",
                        )
                    )
        return edges


class LLMDependencyGenerator(DependencyGraphGenerator):
    """Generate dependencies using an LLM via OpenAI's API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
        client: Optional["OpenAI"] = None,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError(
                "openai package not installed. Install `pip install openai` to use LLMDependencyGenerator."
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for LLMDependencyGenerator.")
        self.client = client or OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _build_prompt(background: str, questions: Sequence[Question]) -> str:
        question_lines = [f"{q.qid}: {q.text.strip()}" for q in questions]
        prompt = textwrap.dedent(
            f"""
            你将看到一段背景文本以及针对该背景的若干个问题。目标是推断回答这些问题时是否需要引用其他问题的答案。

            输出要求：
            - 仅使用提供的问题 ID（例如 "Q1"）。
            - 如果某问题可独立回答，可以不给它添加依赖。
            - 返回必须是 JSON，格式如下：
              {{
                "edges": [
                  {{"source": "Q1", "target": "Q3", "confidence": 0.72, "rationale": "Q3 需要 Q1 的事实"}},
                  ...
                ]
              }}
            - confidence 范围 0~1，表示依赖可信度；rationale 可简要说明原因。
            - 不能出现循环依赖；若不确定可保持稀疏。

            背景：
            {background.strip()}

            问题列表：
            {os.linesep.join(question_lines)}

            请直接输出 JSON，不要添加多余解释。
            """
        ).strip()
        return prompt

    @staticmethod
    def _extract_text(response: object) -> str:
        if hasattr(response, "output"):
            parts: List[str] = []
            for item in getattr(response, "output"):
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) == "text":
                        parts.append(getattr(content, "text", ""))
            return "".join(parts).strip()
        if hasattr(response, "choices"):
            texts: List[str] = []
            for choice in getattr(response, "choices"):
                message = getattr(choice, "message", None)
                if isinstance(message, dict):
                    texts.append(message.get("content", ""))
                else:
                    texts.append(getattr(choice, "text", ""))
            return "".join(texts).strip()
        raise ValueError("Unsupported OpenAI response format.")

    @staticmethod
    def _extract_json_payload(text: str) -> dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned)

    def generate_edges(
        self,
        background: str,
        questions: Sequence[Question],
        metadata: Optional[dict] = None,
    ) -> List[EdgeCandidate]:
        prompt = self._build_prompt(background, questions)
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": "You are an expert planner that reason about dependencies between questions."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"LLM dependency generation failed: {exc}") from exc
        text = self._extract_text(response)
        try:
            payload = self._extract_json_payload(text)
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to parse LLM JSON output: {text}") from exc
        edges: List[EdgeCandidate] = []
        for item in payload.get("edges", []):
            try:
                source = item["source"]
                target = item["target"]
            except KeyError:
                continue
            confidence = float(item.get("confidence", 0.7))
            rationale = item.get("rationale")
            edges.append(EdgeCandidate(source=source, target=target, confidence=confidence, rationale=rationale))
        return edges




class BertAttentionDependencyGenerator(DependencyGraphGenerator):
    """
    Generate dependencies using token-to-token attention weights from a BERT encoder.

    This generator concatenates all questions into a single sequence (with [CLS]/[SEP]
    boundaries), runs a bidirectional BERT encoder, and aggregates the self-attention
    weights from tokens of question_i to tokens of question_j. The aggregated weight
    becomes the confidence score for the edge Q_i -> Q_j.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        *,
        attention_threshold: float = 0.08,
        max_question_tokens: int = 64,
        max_total_tokens: int = 512,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: Any encoder-only Hugging Face checkpoint (e.g., bert-base-uncased).
            attention_threshold: Minimum aggregated attention weight to emit an edge.
            max_question_tokens: Truncate each question to this many wordpiece tokens.
            max_total_tokens: Truncate the packed sequence to this many tokens.
            device: Optional Torch device override (default: auto cuda/cpu selection).
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers package not installed. "
                "Install with: pip install transformers torch"
            ) from exc

        self.attention_threshold = attention_threshold
        self.max_question_tokens = max_question_tokens
        self.max_total_tokens = max_total_tokens

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.last_metrics: Dict[str, float] = {}
        self._last_packed_token_count = 0

    def _pack_questions(self, questions: Sequence[Question]) -> Tuple[List[int], List[int]]:
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
        if cls_id is None or sep_id is None:
            raise RuntimeError("Tokenizer must provide CLS and SEP (or EOS) token ids.")

        token_ids: List[int] = [cls_id]
        owners: List[int] = [-1]  # Track which question owns each token

        for idx, question in enumerate(questions):
            pieces = self.tokenizer.encode(
                question.text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_question_tokens,
            )
            token_ids.extend(pieces)
            owners.extend([idx] * len(pieces))
            token_ids.append(sep_id)
            owners.append(-1)

        if len(token_ids) > self.max_total_tokens:
            token_ids = token_ids[: self.max_total_tokens]
            owners = owners[: self.max_total_tokens]

        self._last_packed_token_count = len(token_ids)
        return token_ids, owners

    def compute_question_attention_matrix(
        self,
        questions: Sequence[Question],
    ):
        if len(questions) <= 1:
            import numpy as np

            return np.zeros((len(questions), len(questions)))

        token_ids, owners = self._pack_questions(questions)
        input_ids = self._torch.tensor([token_ids], dtype=self._torch.long, device=self.device)
        attention_mask = self._torch.ones_like(input_ids, dtype=self._torch.long)

        with self._torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            raise RuntimeError("Model did not return attentions; ensure it supports output_attentions=True.")

        attn_tensor = self._torch.stack(attentions)  # (layers, batch, heads, seq, seq)
        attn_tensor = attn_tensor[:, 0]  # drop batch dim
        attn_matrix = attn_tensor.mean(dim=1).mean(dim=0)  # average over heads then layers
        attn_matrix = attn_matrix.detach().cpu().numpy()

        import numpy as np

        question_token_indices: Dict[int, List[int]] = defaultdict(list)
        for pos, owner in enumerate(owners):
            if owner >= 0:
                question_token_indices[owner].append(pos)

        num_questions = len(questions)
        question_attention = np.zeros((num_questions, num_questions), dtype=float)
        for i in range(num_questions):
            src_tokens = question_token_indices.get(i)
            if not src_tokens:
                continue
            for j in range(num_questions):
                if i == j:
                    continue
                tgt_tokens = question_token_indices.get(j)
                if not tgt_tokens:
                    continue
                submatrix = attn_matrix[np.ix_(src_tokens, tgt_tokens)]
                if submatrix.size == 0:
                    continue
                # Sum attention mass to target tokens per source token,
                # then average over all source tokens to keep values in [0, 1].
                attention_share = submatrix.sum(axis=1).mean()
                question_attention[i, j] = float(attention_share)

        return question_attention

    def generate_edges(
        self,
        background: str,
        questions: Sequence[Question],
        metadata: Optional[dict] = None,
    ) -> List[EdgeCandidate]:
        start = time.perf_counter()
        scores = self.compute_question_attention_matrix(questions)
        edges = self.build_edges_from_scores(questions, scores)
        elapsed = time.perf_counter() - start
        self.last_metrics = {
            "latency": elapsed,
            "prompt_tokens": float(self._last_packed_token_count),
            "generated_tokens": 0.0,
        }
        return edges

    def build_edges_from_scores(
        self,
        questions: Sequence[Question],
        scores,
    ) -> List[EdgeCandidate]:
        edges: List[EdgeCandidate] = []
        for i, source in enumerate(questions):
            for j, target in enumerate(questions):
                if i == j:
                    continue
                weight = scores[i, j]
                if weight < self.attention_threshold:
                    continue
                edges.append(
                    EdgeCandidate(
                        source=source.qid,
                        target=target.qid,
                        confidence=float(weight),
                        rationale=f"attention-weight={weight:.3f}",
                    )
                )
        return edges


def compute_dependency_cost(
    questions: Dict[str, Question],
    source: str,
    *,
    fmt_overhead: int = 6,
) -> int:
    if source not in questions:
        return 0
    question = questions[source]
    return question.tokens + question.answer_tokens + fmt_overhead


def _creates_cycle(adjacency: Dict[str, Set[str]], source: str, target: str) -> bool:
    stack = [target]
    visited: Set[str] = set()
    while stack:
        node = stack.pop()
        if node == source:
            return True
        for nxt in adjacency.get(node, ()):
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return False


def select_dependency_edges(
    questions: Dict[str, Question],
    edge_candidates: Sequence[EdgeCandidate],
    *,
    cost_weight: float = 0.01,
    min_confidence: float = 0.35,
    max_dependencies_per_target: int = 3,
    total_cost_budget: Optional[int] = None,
    fmt_overhead: int = 6,
    prevent_cycles: bool = True,
) -> Dict[str, List[EdgeCandidate]]:
    scored_edges: List[Tuple[float, EdgeCandidate, int]] = []
    for edge in edge_candidates:
        if edge.source not in questions or edge.target not in questions:
            continue
        if edge.source == edge.target:
            continue
        confidence = max(0.0, min(1.0, edge.confidence))
        if confidence < min_confidence:
            continue
        cost = compute_dependency_cost(questions, edge.source, fmt_overhead=fmt_overhead)
        score = confidence - cost_weight * cost
        scored_edges.append((score, edge, cost))

    scored_edges.sort(key=lambda item: item[0], reverse=True)
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    selected: Dict[str, List[EdgeCandidate]] = defaultdict(list)
    accumulated_cost = 0

    for score, edge, cost in scored_edges:
        if score <= 0:
            continue
        target_edges = selected[edge.target]
        if len(target_edges) >= max_dependencies_per_target:
            continue
        if prevent_cycles and _creates_cycle(adjacency, edge.source, edge.target):
            continue
        if total_cost_budget is not None and accumulated_cost + cost > total_cost_budget:
            continue
        adjacency[edge.source].add(edge.target)
        target_edges.append(edge)
        accumulated_cost += cost

    return selected


def apply_dependencies(
    questions: Dict[str, Question],
    selected_edges: Dict[str, List[EdgeCandidate]],
) -> None:
    for question in questions.values():
        deps = {edge.source for edge in selected_edges.get(question.qid, [])}
        question.dependencies = deps


def load_squad_groups(
    split: str,
    *,
    min_questions: int = 3,
    max_questions: Optional[int] = None,
    max_contexts: int = 3,
    seed: int = 13,
) -> List[dict]:
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")  # pragma: no cover

    raw_dataset = load_dataset("squad", split=split)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in raw_dataset:
        grouped[row["context"]].append(row)

    contexts = [
        (ctx, rows) for ctx, rows in grouped.items() if len(rows) >= min_questions
    ]
    if not contexts:
        raise ValueError("No contexts satisfy the minimum question requirement.")

    rng = random.Random(seed)
    rng.shuffle(contexts)
    selected = contexts[:max_contexts] if max_contexts else contexts

    formatted: List[dict] = []
    for context_text, rows in selected:
        if max_questions:
            rows = rows[:max_questions]
        questions = []
        for idx, row in enumerate(rows):
            qid = f"Q{idx + 1}"
            answers = row.get("answers", {}).get("text", [])
            answer_text = answers[0].strip() if answers else ""
            answer_tokens = max(estimate_tokens(answer_text), 12)
            questions.append(
                {
                    "qid": qid,
                    "text": row["question"].strip(),
                    "answer_tokens": answer_tokens,
                    "references": answers,
                }
            )
        formatted.append(
            {
                "context": context_text.strip(),
                "title": row.get("title", "SQuAD-Context"),
                "questions": questions,
            }
        )
    return formatted


def load_squad_random_questions(
    split: str,
    *,
    max_contexts: int = 3,
    seed: int = 13,
) -> List[dict]:
    """
    Sample individual SQuAD questions without grouping by shared context.
    Each question becomes its own context group.
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")  # pragma: no cover

    raw_dataset = list(load_dataset("squad", split=split))
    rng = random.Random(seed)
    rng.shuffle(raw_dataset)
    selected = raw_dataset[:max_contexts]

    groups: List[dict] = []
    for idx, row in enumerate(selected, start=1):
        answers = row.get("answers", {}).get("text", [])
        answer_text = answers[0].strip() if answers else ""
        answer_tokens = max(estimate_tokens(answer_text), 12)
        qid = "Q1"
        groups.append(
            {
                "context": row["context"].strip(),
                "title": row.get("title", f"SQuAD-{idx}"),
                "questions": [
                    {
                        "qid": qid,
                        "text": row["question"].strip(),
                        "answer_tokens": answer_tokens,
                        "references": answers,
                    }
                ],
            }
        )
    return groups


def _format_hotpot_context(row: dict) -> str:
    titles = row.get("context", {}).get("title", [])
    sentences = row.get("context", {}).get("sentences", [])
    pieces: List[str] = []
    for title, sent_list in zip(titles, sentences):
        sent_text = " ".join(s.strip() for s in sent_list)
        pieces.append(f"{title}: {sent_text}")
    return "\n".join(pieces)


def load_hotpot_groups(
    split: str,
    *,
    subset: str = "distractor",
    max_contexts: int = 3,
    seed: int = 13,
) -> List[dict]:
    """
    Load HotpotQA rows as independent contexts (one question per context).
    """
    if load_dataset is None:
        raise RuntimeError("datasets package not available; install with `pip install datasets`.")  # pragma: no cover

    raw_dataset = list(load_dataset("hotpotqa/hotpot_qa", subset, split=split))
    if not raw_dataset:
        raise ValueError("Empty HotpotQA split.")

    rng = random.Random(seed)
    rng.shuffle(raw_dataset)

    groups: List[dict] = []
    for idx, row in enumerate(raw_dataset):
        if len(groups) >= max_contexts:
            break
        background = _format_hotpot_context(row)
        answer_text = row.get("answer", "").strip()
        answer_tokens = max(estimate_tokens(answer_text), 12)
        groups.append(
            {
                "context": background,
                "title": row.get("id", f"Hotpot-{idx+1}"),
                "questions": [
                    {
                        "qid": "Q1",
                        "text": row.get("question", "").strip(),
                        "answer_tokens": answer_tokens,
                        "references": [answer_text] if answer_text else [],
                    }
                ],
            }
        )

    if not groups:
        raise ValueError("No HotpotQA groups constructed; check subset/split parameters.")
    return groups


def build_questions_from_group(group: dict) -> List[Question]:
    questions: List[Question] = []
    for payload in group["questions"]:
        text = payload["text"]
        type_hint = None
        question = Question(
            qid=payload["qid"],
            text=text,
            priority=1.0 + (0.2 if detect_aggregate_question(text, type_hint) else 0.0),
            answer_tokens=payload["answer_tokens"],
            type_hint=type_hint,
            references=list(payload.get("references", [])),
        )
        questions.append(question)
    return questions


def run_pipeline_for_context(
    context_payload: dict,
    generator: DependencyGraphGenerator,
    *,
    cost_weight: float = 0.01,
    min_confidence: float = 0.35,
    max_dependencies_per_target: int = 3,
    total_cost_budget: Optional[int] = None,
    fmt_overhead: int = 6,
    html_dir: Optional[Path] = None,
) -> ScheduleResult:
    background = context_payload["context"]
    questions_list = build_questions_from_group(context_payload)
    edges = generator.generate_edges(background, questions_list, metadata=context_payload)
    questions_dict = {q.qid: q for q in questions_list}

    selected = select_dependency_edges(
        questions_dict,
        edges,
        cost_weight=cost_weight,
        min_confidence=min_confidence,
        max_dependencies_per_target=max_dependencies_per_target,
        total_cost_budget=total_cost_budget,
        fmt_overhead=fmt_overhead,
    )
    apply_dependencies(questions_dict, selected)

    scheduler = DependencyScheduler(
        background,
        questions_list,
        max_batch_tokens=None,
        fmt_overhead_per_section=fmt_overhead,
        prefill_token_cost=0.8,
        generate_token_cost=1.2,
    )
    scheduler.build_dependencies(auto_infer=False)
    result = scheduler.schedule()

    logging.info("Context: %s | batches=%d", context_payload.get("title", "Untitled"), len(result.batches))
    for q in questions_list:
        logging.info("  %s -> deps %s", q.qid, sorted(q.dependencies))

    scheduler.pretty_print_schedule(result)

    if html_dir:
        html_dir.mkdir(parents=True, exist_ok=True)
        safe_title = context_payload.get("title", "context").replace(" ", "_")
        html_path = html_dir / f"{safe_title}_schedule.html"
        export_schedule_html(scheduler, result, html_path)
        logging.info("Wrote interactive visualisation to %s", html_path)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM-guided dependency graphs for SQuAD contexts and run greedy scheduling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", default="train", help="SQuAD split to sample.")
    parser.add_argument("--context-count", type=int, default=1, help="Number of contexts to process.")
    parser.add_argument("--min-questions", type=int, default=3, help="Minimum questions per context.")
    parser.add_argument("--max-questions", type=int, default=6, help="Maximum questions per context.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for sampling contexts.")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM to generate dependencies (requires OpenAI API key).")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model for dependency inference.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature.")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Minimum confidence to keep a dependency edge.")
    parser.add_argument("--cost-weight", type=float, default=0.01, help="Cost penalty weight when selecting edges.")
    parser.add_argument("--max-dependencies", type=int, default=3, help="Max dependencies per question.")
    parser.add_argument("--total-cost-budget", type=int, default=None, help="Optional global dependency cost budget.")
    parser.add_argument("--html-dir", type=Path, default=None, help="Directory to output HTML visualisations.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    contexts = load_squad_groups(
        args.split,
        min_questions=args.min_questions,
        max_questions=args.max_questions,
        max_contexts=args.context_count,
        seed=args.seed,
    )

    if args.use_llm:
        generator: DependencyGraphGenerator = LLMDependencyGenerator(
            model=args.openai_model,
            temperature=args.temperature,
        )
    else:
        logging.info("Using heuristic dependency generator (no LLM).")
        generator = HeuristicDependencyGenerator()

    for idx, context_payload in enumerate(contexts, start=1):
        logging.info(
            "Processing context %d/%d: %s",
            idx,
            len(contexts),
            context_payload.get("title", f"context-{idx}"),
        )
        run_pipeline_for_context(
            context_payload,
            generator,
            cost_weight=args.cost_weight,
            min_confidence=args.min_confidence,
            max_dependencies_per_target=args.max_dependencies,
            total_cost_budget=args.total_cost_budget,
            fmt_overhead=6,
            html_dir=args.html_dir,
        )


if __name__ == "__main__":
    main()
