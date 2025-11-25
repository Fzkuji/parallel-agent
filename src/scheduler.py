"""Dependency-aware batch scheduler for parallel question answering."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .models import BatchAssignment, Question, ScheduleResult, estimate_tokens
from .text_utils import (
    detect_aggregate_question,
    detect_reference_question,
    extract_keywords,
)


class DependencyScheduler:
    """Schedules questions into batches based on their dependencies."""

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
        """Build dependency graph, optionally inferring from text."""
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
        """Return questions grouped by dependency layer (topological order)."""
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
        """Schedule questions into batches based on dependencies."""
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
        """Print a human-readable schedule summary."""
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
    """Export an interactive HTML visualization of the schedule."""
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
          `Token：background=${{batch.tokens.background}}, incremental=${{batch.tokens.incremental}}, generation=${{batch.tokens.generation}}, total=${{batch.tokens.total}}`,
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
