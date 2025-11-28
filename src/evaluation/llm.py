"""LLM-based evaluation using OpenRouter API.

This module provides GPT-4 style evaluation for long-form generation tasks,
particularly useful for CMB (Chinese Medical Benchmark) where traditional
metrics like EM/F1 are not suitable.

Evaluation dimensions (for CMB-Clin):
1. Fluency (流畅性): Language quality and readability
2. Relevance (相关性): How well the answer addresses the question
3. Completeness (完整性): Coverage of key information
4. Proficiency (医学专业性): Medical accuracy and terminology
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import httpx for async requests, fall back to requests
try:
    import httpx
    HTTPX_AVAILABLE = True
    # Suppress httpx INFO logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class LLMEvalResult:
    """Result from LLM-based evaluation."""
    fluency: float  # 1-5
    relevance: float  # 1-5
    completeness: float  # 1-5
    proficiency: float  # 1-5
    average: float  # Average of all dimensions
    raw_response: str  # Raw LLM response for debugging


# CMB-Clin evaluation prompt template (Chinese)
CMB_EVAL_PROMPT_TEMPLATE = """你是一名专业的临床医学评估专家。请根据以下评估标准，对模型生成的答案进行评分。

## 病例背景
{context}

## 问题
{question}

## 参考答案
{reference}

## 模型答案
{prediction}

## 评估维度和评分标准

### 1. 流畅性 (Fluency) [1-5分]
评估答案的语言表达是否流畅、通顺、易于理解。
- 5分：语言表达非常流畅自然，逻辑清晰，专业术语使用恰当
- 4分：语言表达较为流畅，偶有小瑕疵但不影响理解
- 3分：语言基本通顺，但有一些表达不够清晰的地方
- 2分：语言表达存在较多问题，影响理解
- 1分：语言混乱，难以理解

### 2. 相关性 (Relevance) [1-5分]
评估答案是否切题，是否针对问题进行回答。
- 5分：完全切题，直接回答了问题的核心内容
- 4分：基本切题，回答了问题的主要方面
- 3分：部分切题，但有一些偏离或冗余内容
- 2分：切题度较低，大部分内容与问题关联不大
- 1分：完全跑题，答非所问

### 3. 完整性 (Completeness) [1-5分]
评估答案是否涵盖了问题所需的所有关键信息。
- 5分：非常完整，涵盖了所有关键点和必要细节
- 4分：较为完整，涵盖了主要关键点
- 3分：基本完整，但遗漏了一些重要信息
- 2分：不够完整，遗漏了较多关键信息
- 1分：非常不完整，仅涉及极少部分内容

### 4. 医学专业性 (Proficiency) [1-5分]
评估答案的医学准确性和专业水平。
- 5分：医学内容完全准确，展现出很高的专业水平
- 4分：医学内容基本准确，专业性较好
- 3分：医学内容大体正确，但有一些不够精确的地方
- 2分：存在明显的医学错误或不专业的表述
- 1分：医学内容严重错误，可能导致误导

## 输出格式
请以JSON格式输出评分结果，格式如下：
```json
{{
    "fluency": <1-5的整数>,
    "relevance": <1-5的整数>,
    "completeness": <1-5的整数>,
    "proficiency": <1-5的整数>,
    "reasoning": "<简要说明评分理由>"
}}
```

请严格按照上述格式输出，只输出JSON，不要有其他内容。"""


class OpenRouterEvaluator:
    """LLM-based evaluator using OpenRouter API."""

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 5,
        retry_delay: float = 2.0,
        timeout: float = 60.0,
        request_delay: float = 1.0,
    ):
        """
        Initialize OpenRouter evaluator.

        Args:
            model: Model ID on OpenRouter (e.g., "openai/gpt-4o", "anthropic/claude-3-opus")
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            base_url: OpenRouter API base URL
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Base delay between retries (seconds), uses exponential backoff
            timeout: Request timeout (seconds)
            request_delay: Delay between batch requests to avoid rate limiting (seconds)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter."
            )
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.request_delay = request_delay

        if not HTTPX_AVAILABLE and not REQUESTS_AVAILABLE:
            raise RuntimeError(
                "Either httpx or requests package is required. "
                "Install with: pip install httpx or pip install requests"
            )

    def _make_request(self, messages: List[Dict[str, str]]) -> str:
        """Make API request to OpenRouter."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/battlenet-qa",
            "X-Title": "Battlenet QA Evaluation",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,  # Deterministic for evaluation
            "max_tokens": 512,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                if HTTPX_AVAILABLE:
                    with httpx.Client(timeout=self.timeout) as client:
                        response = client.post(url, headers=headers, json=payload)
                        response.raise_for_status()
                        data = response.json()
                else:
                    response = requests.post(
                        url, headers=headers, json=payload, timeout=self.timeout
                    )
                    response.raise_for_status()
                    data = response.json()

                return data["choices"][0]["message"]["content"]

            except Exception as e:
                last_error = e
                # Check for rate limiting (429)
                is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
                if is_rate_limit:
                    # Exponential backoff for rate limiting: 5s, 10s, 20s, 40s, 80s
                    wait_time = self.retry_delay * (2 ** (attempt + 1))
                    logger.warning(f"Rate limited (attempt {attempt + 1}), waiting {wait_time:.1f}s...")
                else:
                    # Linear backoff for other errors
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.warning(f"API request failed (attempt {attempt + 1}): {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)

        raise RuntimeError(f"API request failed after {self.max_retries} attempts: {last_error}")

    def _parse_scores(self, response: str) -> Dict[str, Any]:
        """Parse scores from LLM response."""
        # Handle empty or whitespace-only responses
        if not response or not response.strip():
            logger.warning("Received empty response from LLM")
            return {
                "fluency": 3,
                "relevance": 3,
                "completeness": 3,
                "proficiency": 3,
                "reasoning": "Empty response from LLM",
            }

        # Try to extract JSON from markdown code blocks first
        code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to extract JSON object (allowing nested content)
        json_match = re.search(r"\{[\s\S]*?\}", response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse the entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Last resort: extract numbers manually with flexible patterns
        scores = {}
        # Chinese key mappings
        key_aliases = {
            "fluency": ["fluency", "流畅性", "流畅"],
            "relevance": ["relevance", "相关性", "相关"],
            "completeness": ["completeness", "完整性", "完整"],
            "proficiency": ["proficiency", "专业性", "医学专业性", "专业"],
        }

        for key, aliases in key_aliases.items():
            for alias in aliases:
                # Try various patterns: "key": 5, key: 5, key=5, key：5 (Chinese colon)
                patterns = [
                    rf'"{alias}"[\s:：]+(\d+)',  # "key": 5
                    rf"'{alias}'[\s:：]+(\d+)",  # 'key': 5
                    rf'{alias}[\s:：]+(\d+)',    # key: 5 or key：5
                    rf'{alias}\s*[=]\s*(\d+)',   # key=5
                ]
                found = False
                for pattern in patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        scores[key] = int(match.group(1))
                        found = True
                        break
                if found:
                    break

        if len(scores) < 4:
            # Log more context for debugging
            logger.warning(f"Failed to parse scores (found {len(scores)}/4) from response: {response[:300]}...")
            # Return default scores
            return {
                "fluency": 3,
                "relevance": 3,
                "completeness": 3,
                "proficiency": 3,
                "reasoning": "Failed to parse LLM response",
            }

        return scores

    def evaluate_single(
        self,
        context: str,
        question: str,
        reference: str,
        prediction: str,
    ) -> LLMEvalResult:
        """
        Evaluate a single prediction.

        Args:
            context: Case/context background
            question: The question being answered
            reference: Gold/reference answer
            prediction: Model's predicted answer

        Returns:
            LLMEvalResult with scores for each dimension
        """
        prompt = CMB_EVAL_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            reference=reference,
            prediction=prediction,
        )

        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(messages)

        scores = self._parse_scores(response)
        fluency = float(scores.get("fluency", 3))
        relevance = float(scores.get("relevance", 3))
        completeness = float(scores.get("completeness", 3))
        proficiency = float(scores.get("proficiency", 3))

        # Clamp to valid range
        fluency = max(1.0, min(5.0, fluency))
        relevance = max(1.0, min(5.0, relevance))
        completeness = max(1.0, min(5.0, completeness))
        proficiency = max(1.0, min(5.0, proficiency))

        average = (fluency + relevance + completeness + proficiency) / 4.0

        return LLMEvalResult(
            fluency=fluency,
            relevance=relevance,
            completeness=completeness,
            proficiency=proficiency,
            average=average,
            raw_response=response,
        )

    def evaluate_batch(
        self,
        items: List[Tuple[str, str, str, str]],
        show_progress: bool = True,
    ) -> List[LLMEvalResult]:
        """
        Evaluate a batch of predictions.

        Args:
            items: List of (context, question, reference, prediction) tuples
            show_progress: Whether to show progress information

        Returns:
            List of LLMEvalResult for each item
        """
        results = []
        total = len(items)

        for i, (context, question, reference, prediction) in enumerate(items):
            if show_progress:
                logger.info(f"LLM evaluation progress: {i + 1}/{total}")

            try:
                result = self.evaluate_single(context, question, reference, prediction)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate item {i + 1}: {e}")
                # Append default result on failure
                results.append(LLMEvalResult(
                    fluency=0.0,
                    relevance=0.0,
                    completeness=0.0,
                    proficiency=0.0,
                    average=0.0,
                    raw_response=f"Error: {e}",
                ))

            # Rate limiting: delay between requests
            if i < total - 1 and self.request_delay > 0:
                time.sleep(self.request_delay)

        return results


def compute_llm_metrics(
    results: List[LLMEvalResult],
) -> Dict[str, float]:
    """
    Compute aggregate metrics from LLM evaluation results.

    Args:
        results: List of LLMEvalResult from evaluation

    Returns:
        Dict with averaged metrics (fluency, relevance, completeness,
        proficiency, llm_average)
    """
    if not results:
        return {
            "llm_fluency": 0.0,
            "llm_relevance": 0.0,
            "llm_completeness": 0.0,
            "llm_proficiency": 0.0,
            "llm_average": 0.0,
        }

    n = len(results)
    return {
        "llm_fluency": sum(r.fluency for r in results) / n,
        "llm_relevance": sum(r.relevance for r in results) / n,
        "llm_completeness": sum(r.completeness for r in results) / n,
        "llm_proficiency": sum(r.proficiency for r in results) / n,
        "llm_average": sum(r.average for r in results) / n,
    }
