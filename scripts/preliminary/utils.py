"""Utility functions for preliminary experiments.

Provides:
- LLM API client wrapper (supports API and local models)
- Evaluation metrics
- Result logging and saving
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SimpleResponse:
    """Simple response object for local models."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    condition: str  # "oracle", "method", "random"
    dataset: str
    n_samples: int
    n_questions: int
    accuracy: float
    metrics: Dict[str, float] = field(default_factory=dict)
    latency: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    exp_name: str
    dataset: str
    model: str = "gpt-4o-mini"
    n_samples: int = 100
    seed: int = 42
    output_dir: str = "outputs/preliminary"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMClient:
    """Wrapper for LLM calls with retry and logging.

    Supports both API-based models and local models (via transformers/vLLM).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
        use_local: bool = False,
        device: str = "auto",
        use_vllm: bool = False,
        tensor_parallel_size: int = 1,
    ):
        self.model = model
        self.temperature = temperature
        self.use_local = use_local
        self.use_vllm = use_vllm
        self._tokenizer = None
        self._model = None
        self._vllm_model = None

        if use_local:
            if use_vllm:
                self._init_vllm_model(model, tensor_parallel_size)
            else:
                self._init_local_model(model, device)
        else:
            self._init_api_client(model, temperature, max_retries)

    def _init_api_client(self, model: str, temperature: float, max_retries: int):
        """Initialize API client."""
        from src.api_client import APIClient
        self.client = APIClient(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
        )
        logger.info(f"Initialized API client with model: {model}")

    def _init_vllm_model(self, model: str, tensor_parallel_size: int):
        """Initialize model with vLLM for fast multi-GPU inference."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Please install vllm: pip install vllm")

        logger.info(f"Loading model with vLLM: {model} (tensor_parallel_size={tensor_parallel_size})")

        self._vllm_model = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="half",
        )
        logger.info(f"vLLM model loaded with {tensor_parallel_size} GPU(s)")

    def _init_local_model(self, model: str, device: str):
        """Initialize local model with transformers."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")

        logger.info(f"Loading local model: {model}")

        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            self._model = self._model.to(device)

        logger.info(f"Model loaded on {device}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, Union[SimpleResponse, Any]]:
        """Generate response from LLM.

        Returns:
            Tuple of (response_text, response_object)
        """
        if self.use_local:
            if self.use_vllm:
                return self._generate_vllm(prompt, max_tokens, system_prompt)
            else:
                return self._generate_local(prompt, max_tokens, system_prompt)
        else:
            return self._generate_api(prompt, max_tokens, system_prompt)

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> List[Tuple[str, SimpleResponse]]:
        """Generate responses for multiple prompts in batch (vLLM only).

        Returns:
            List of (response_text, response_object) tuples
        """
        if self.use_vllm and self._vllm_model is not None:
            return self._generate_vllm_batch(prompts, max_tokens, system_prompt)
        else:
            # Fallback to sequential generation
            return [self.generate(p, max_tokens, system_prompt) for p in prompts]

    def _generate_vllm(
        self,
        prompt: str,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> Tuple[str, SimpleResponse]:
        """Generate using vLLM."""
        results = self._generate_vllm_batch([prompt], max_tokens, system_prompt)
        return results[0]

    def _generate_vllm_batch(
        self,
        prompts: List[str],
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> List[Tuple[str, SimpleResponse]]:
        """Generate batch responses using vLLM."""
        from vllm import SamplingParams
        import time

        # Build full prompts with system prompt if provided
        full_prompts = []
        for prompt in prompts:
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            full_prompts.append(full_prompt)

        sampling_params = SamplingParams(
            temperature=self.temperature if self.temperature > 0 else 0,
            max_tokens=max_tokens,
        )

        start_time = time.perf_counter()
        outputs = self._vllm_model.generate(full_prompts, sampling_params)
        total_latency = time.perf_counter() - start_time

        results = []
        for output in outputs:
            text = output.outputs[0].text
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)

            response = SimpleResponse(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency=total_latency / len(prompts),  # Average latency per sample
            )
            results.append((text, response))

        return results

    def _generate_api(
        self,
        prompt: str,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> Tuple[str, Any]:
        """Generate using API client."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.generate(messages=messages, max_tokens=max_tokens)
        return response.text, response

    def _generate_local(
        self,
        prompt: str,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> Tuple[str, SimpleResponse]:
        """Generate using local model."""
        import torch

        # Build chat template if model supports it
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            full_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        # Tokenize
        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        # Generate
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        latency = time.perf_counter() - start_time

        # Decode (only new tokens)
        new_tokens = outputs[0][prompt_tokens:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion_tokens = len(new_tokens)

        response = SimpleResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency=latency,
        )
        return text, response

    def detect_dependencies(
        self,
        questions: List[str],
        context: Optional[str] = None,
    ) -> List[Tuple[int, int]]:
        """Use LLM to detect dependencies between questions.

        Returns:
            List of (source_idx, target_idx) tuples indicating Q_source -> Q_target dependency
        """
        # Build prompt
        q_list = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(questions)])

        prompt = f"""Analyze the following questions and identify dependencies between them.
A dependency exists when answering one question requires the answer from another question.

{f"Context: {context}" if context else ""}

Questions:
{q_list}

For each dependency found, output in the format: Qi -> Qj (meaning Qj depends on Qi's answer)
If no dependencies exist, output: NONE

Output only the dependencies, one per line:"""

        response, _ = self.generate(prompt, max_tokens=256)

        # Parse dependencies
        dependencies = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if "NONE" in line.upper():
                break
            if "->" in line:
                try:
                    parts = line.split("->")
                    src = int(parts[0].strip().replace("Q", "")) - 1
                    tgt = int(parts[1].strip().split()[0].replace("Q", "")) - 1
                    if 0 <= src < len(questions) and 0 <= tgt < len(questions):
                        dependencies.append((src, tgt))
                except (ValueError, IndexError):
                    continue

        return dependencies


def compute_exact_match(prediction: str, reference: str) -> float:
    """Compute exact match score."""
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    return 1.0 if pred == ref else 0.0


def compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_contains(prediction: str, reference: str) -> float:
    """Check if prediction contains the reference answer."""
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    return 1.0 if ref in pred else 0.0


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    import re
    import string

    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def save_results(
    results: List[ExperimentResult],
    config: ExperimentConfig,
    output_dir: Optional[str] = None,
) -> str:
    """Save experiment results to JSON file.

    Returns:
        Path to saved file
    """
    output_dir = output_dir or config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config.exp_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    data = {
        "config": config.to_dict(),
        "timestamp": timestamp,
        "results": [asdict(r) for r in results],
        "summary": {
            condition: {
                "accuracy": next(
                    (r.accuracy for r in results if r.condition == condition),
                    None
                )
            }
            for condition in ["oracle", "method", "random"]
        }
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {filepath}")
    return filepath


def print_summary(results: List[ExperimentResult]) -> None:
    """Print summary of experiment results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for result in results:
        print(f"\n{result.condition.upper()}:")
        print(f"  Accuracy: {result.accuracy:.4f}")
        print(f"  Samples: {result.n_samples}")
        print(f"  Questions: {result.n_questions}")
        if result.latency > 0:
            print(f"  Latency: {result.latency:.2f}s")
        for metric, value in result.metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 60)

    # Print comparison
    oracle = next((r for r in results if r.condition == "oracle"), None)
    method = next((r for r in results if r.condition == "method"), None)
    random = next((r for r in results if r.condition == "random"), None)

    if oracle and random:
        gap = oracle.accuracy - random.accuracy
        print(f"\nOracle - Random Gap: {gap:.4f}")
    if method and random:
        gap = method.accuracy - random.accuracy
        print(f"Method - Random Gap: {gap:.4f}")
    if oracle and method:
        recovery = (method.accuracy - random.accuracy) / (oracle.accuracy - random.accuracy) if (oracle.accuracy - random.accuracy) > 0 else 0
        print(f"Method Recovery Rate: {recovery:.2%}")


def topological_sort(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    """Topological sort of n nodes with given directed edges.

    Args:
        n: Number of nodes (0 to n-1)
        edges: List of (source, target) edges

    Returns:
        Sorted list of node indices, or original order if cycle detected
    """
    from collections import defaultdict, deque

    # Build adjacency list and in-degree count
    adj = defaultdict(list)
    in_degree = [0] * n

    for src, tgt in edges:
        adj[src].append(tgt)
        in_degree[tgt] += 1

    # Kahn's algorithm
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If not all nodes are sorted, there's a cycle - return original order
    if len(result) != n:
        logger.warning("Cycle detected in dependency graph, using original order")
        return list(range(n))

    return result
