"""Dependency graph generators for question scheduling."""
from __future__ import annotations

import json
import os
import textwrap
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from .models import EdgeCandidate, Question
from .text_utils import (
    detect_aggregate_question,
    detect_reference_question,
    extract_keywords,
)

if TYPE_CHECKING:
    from openai import OpenAI


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
        try:
            from openai import OpenAI as OpenAIClient
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed. Install `pip install openai` to use LLMDependencyGenerator."
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for LLMDependencyGenerator.")
        self.client = client or OpenAIClient(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _build_prompt(background: str, questions: Sequence[Question]) -> str:
        question_lines = [f"{q.qid}: {q.text.strip()}" for q in questions]
        prompt = textwrap.dedent(
            f"""
            你将看到一段背景文本以及若干针对该背景的问题。请分析这些问题，判断回答它们的最佳顺序。

            如果先回答某个问题会对回答另一个问题有帮助（例如：提供相关信息、建立推理基础、或者逻辑上应该先处理），请指出这种顺序关系。

            输出格式示例（表示回答Q3之前最好先回答Q1，回答Q4之前最好先回答Q2）：
            {{"edges": [{{"Q3": "Q1"}}, {{"Q4": "Q2"}}]}}

            规则：
            - 格式为 {{"后回答的问题ID": "先回答的问题ID"}}。
            - **按顺序关系的重要性从高到低排序输出**。
            - 如果问题之间没有明显的顺序关系（可以并行回答），返回空数组：{{"edges": []}}
            - 不要形成循环（如A在B前，B又在A前）。

            背景：
            {background.strip()}

            问题：
            {os.linesep.join(question_lines)}

            请直接输出JSON：
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
        except Exception as exc:
            raise RuntimeError(f"LLM dependency generation failed: {exc}") from exc
        text = self._extract_text(response)
        try:
            payload = self._extract_json_payload(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse LLM JSON output: {text}") from exc
        edges_data = payload.get("edges", [])
        edges: List[EdgeCandidate] = []
        total_edges = len(edges_data)
        for idx, item in enumerate(edges_data):
            # Compact format: {"Q3": "Q1"} means Q3 depends on Q1 (source=Q1, target=Q3)
            if len(item) == 1:
                target, source = next(iter(item.items()))
            else:
                # Fallback to old format for compatibility
                try:
                    source = item["source"]
                    target = item["target"]
                except KeyError:
                    continue
            # Handle case where source is a list (e.g., {"Q3": ["Q1", "Q2"]})
            if isinstance(source, list):
                sources = source
            else:
                sources = [source]
            # Position-based confidence: first edge = 1.0, last edge = 0.5
            confidence = 1.0 - (idx / max(total_edges, 1)) * 0.5
            for src in sources:
                if isinstance(src, str) and isinstance(target, str):
                    edges.append(EdgeCandidate(source=src, target=target, confidence=confidence))
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
        except ImportError as exc:
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
        owners: List[int] = [-1]

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

        attn_tensor = self._torch.stack(attentions)
        attn_tensor = attn_tensor[:, 0]
        attn_matrix = attn_tensor.mean(dim=1).mean(dim=0)
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
