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

import torch
import torch.distributed as dist

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
        enable_thinking: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.use_local = use_local
        self.use_vllm = use_vllm
        self.enable_thinking = enable_thinking
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
        """Initialize model with vLLM for fast inference.

        NOTE: For multi-GPU data parallelism, CUDA_VISIBLE_DEVICES should be set
        by the caller (e.g., in worker_process) BEFORE calling this function.
        """
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("Please install vllm and transformers")

        logger.info(f"Loading model with vLLM: {model} (tensor_parallel_size={tensor_parallel_size})")

        # Load tokenizer for chat template support (needed for Qwen3, etc.)
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._vllm_model = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="half",
            gpu_memory_utilization=0.9,
            disable_log_stats=True,
        )
        logger.info(f"vLLM model loaded with {tensor_parallel_size} GPU(s), enable_thinking={self.enable_thinking}")

    def _init_local_model(self, model: str, device: str):
        """Initialize local model with transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")

        # Determine device - use current CUDA device if set by distributed setup
        if device == "auto":
            if torch.cuda.is_available():
                # Use current device (set by torch.cuda.set_device in distributed setup)
                device = f"cuda:{torch.cuda.current_device()}"
            else:
                device = "cpu"

        logger.info(f"Loading local model: {model} -> {device}")

        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        is_cuda = device.startswith("cuda")

        # Load model - for multi-GPU, load to CPU first then move to avoid OOM
        # device_map can cause all processes to allocate on GPU 0 during loading
        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16 if is_cuda else torch.float32,
            device_map=None,  # Don't use device_map to avoid multi-process conflicts
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self._model = self._model.to(device)
        self._model.eval()

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

        # Build full prompts using chat template if tokenizer supports it
        full_prompts = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            if hasattr(self._tokenizer, "apply_chat_template"):
                # Use chat template (supports Qwen3 enable_thinking, etc.)
                try:
                    full_prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.enable_thinking,
                    )
                except TypeError:
                    # Fallback for models that don't support enable_thinking
                    full_prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            else:
                # Fallback for models without chat template
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                else:
                    full_prompt = prompt
            full_prompts.append(full_prompt)

        # Sampling params: Qwen3 recommends temp=0.7, top_p=0.8 for non-thinking mode
        if self.temperature > 0:
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=0.8 if not self.enable_thinking else 0.95,
                top_k=20,
                max_tokens=max_tokens,
            )
        else:
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=max_tokens,
            )

        start_time = time.perf_counter()
        outputs = self._vllm_model.generate(full_prompts, sampling_params, use_tqdm=False)
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
            try:
                # Support Qwen3 enable_thinking parameter
                full_prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                # Fallback for models that don't support enable_thinking
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


# Use official SQuAD evaluation from Hugging Face evaluate library
# This ensures consistent, standard evaluation metrics
try:
    import evaluate
    _squad_metric = evaluate.load("squad")
except Exception:
    _squad_metric = None


def compute_squad_metrics(prediction: str, reference: str) -> dict:
    """Compute EM and F1 using official SQuAD evaluation.

    Returns:
        dict with 'exact_match' and 'f1' keys (values 0-100 scale)
    """
    # Edge case: if both normalize to empty (e.g., answer is "a" which gets removed as article)
    # Both official SQuAD metric and fallback may return F1=0 in this case, but it should be 1.0
    pred_norm = _normalize_answer(prediction)
    ref_norm = _normalize_answer(reference)
    if not pred_norm and not ref_norm:
        # Both empty after normalization - treat as exact match
        return {"exact_match": 100.0, "f1": 100.0}

    if _squad_metric is None:
        # Fallback to manual implementation if evaluate not available
        em = _compute_exact_match_manual(prediction, reference)
        f1 = _compute_f1_manual(prediction, reference)
        return {"exact_match": em * 100, "f1": f1 * 100}

    # Format for SQuAD metric: predictions and references need specific format
    predictions = [{"id": "0", "prediction_text": prediction}]
    references = [{"id": "0", "answers": {"text": [reference], "answer_start": [0]}}]

    result = _squad_metric.compute(predictions=predictions, references=references)

    # Ensure F1 >= EM (fix edge cases in official metric)
    if result["f1"] < result["exact_match"]:
        result["f1"] = result["exact_match"]

    return result


def compute_exact_match(prediction: str, reference: str) -> float:
    """Compute exact match score using official SQuAD evaluation.

    Returns: 1.0 if exact match, 0.0 otherwise
    """
    result = compute_squad_metrics(prediction, reference)
    return result["exact_match"] / 100.0


def compute_f1(prediction: str, reference: str) -> float:
    """Compute F1 score using official SQuAD evaluation.

    Returns: F1 score in range [0, 1]
    """
    result = compute_squad_metrics(prediction, reference)
    return result["f1"] / 100.0


def _normalize_answer(text: str) -> str:
    """Normalize answer for comparison (SQuAD style)."""
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


def _compute_exact_match_manual(prediction: str, reference: str) -> float:
    """Fallback: Compute exact match score manually."""
    pred = _normalize_answer(prediction)
    ref = _normalize_answer(reference)
    return 1.0 if pred == ref else 0.0


def _compute_f1_manual(prediction: str, reference: str) -> float:
    """Fallback: Compute token-level F1 score manually (SQuAD style)."""
    from collections import Counter

    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens = _normalize_answer(reference).split()

    # Edge case: if both normalize to empty (e.g., answer is "a" which gets removed as article)
    # Treat as exact match if both are empty
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Use Counter for proper frequency-based overlap
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = pred_counter & ref_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_contains(prediction: str, reference: str) -> float:
    """Check if prediction contains the reference answer (lenient metric).

    WARNING: This is a lenient metric that may give inflated scores.
    Use compute_exact_match for standard evaluation.
    """
    pred = _normalize_answer(prediction)
    ref = _normalize_answer(reference)
    return 1.0 if ref in pred else 0.0


# Public alias for backward compatibility
normalize_answer = _normalize_answer


def save_results(
    results: List[ExperimentResult],
    config: ExperimentConfig,
    output_dir: Optional[str] = None,
) -> str:
    """Save experiment results to JSON file.

    Filename is based on config settings (exp_name, model, n_samples).
    Same settings will overwrite previous results.

    Returns:
        Path to saved file
    """
    output_dir = output_dir or config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create filename based on config (sanitize model name)
    model_name = config.model.replace("/", "_").replace("\\", "_")
    n_samples_str = "all" if config.n_samples == -1 else str(config.n_samples)
    filename = f"{config.exp_name}_{model_name}_n{n_samples_str}.json"
    filepath = os.path.join(output_dir, filename)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "config": config.to_dict(),
        "timestamp": timestamp,
        "results": [asdict(r) for r in results],
        "summary": {
            r.condition: {
                "em": r.metrics.get("em", r.accuracy),
                "f1": r.metrics.get("f1", 0),
            }
            for r in results
        }
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {filepath}")
    return filepath


def print_summary(results: List[ExperimentResult]) -> None:
    """Print summary of experiment results as markdown table."""
    print("\n**EXPERIMENT SUMMARY**\n")

    if not results:
        print("No results to display.")
        return

    # Build header (show per-sample averages for tokens and latency)
    headers = ["Condition", "EM", "F1", "Samples", "Avg Prompt", "Avg Compl", "Avg Latency (s)"]

    # Print markdown table header
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    # Print each result row
    for result in results:
        em = result.metrics.get("em", 0)
        f1 = result.metrics.get("f1", 0)
        n = result.n_samples if result.n_samples > 0 else 1
        avg_prompt = result.prompt_tokens / n
        avg_compl = result.completion_tokens / n
        avg_latency = result.latency / n
        row = [
            result.condition,
            f"{em:.4f}",
            f"{f1:.4f}",
            str(result.n_samples),
            f"{avg_prompt:.1f}",
            f"{avg_compl:.1f}",
            f"{avg_latency:.2f}" if result.latency > 0 else "-",
        ]
        print("| " + " | ".join(row) + " |")

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


def setup_distributed(timeout_hours: float = 2.0) -> Tuple[int, int]:
    """Setup distributed training environment.

    Args:
        timeout_hours: Timeout for distributed operations in hours (default: 2 hours)

    Returns:
        Tuple of (rank, world_size)
    """
    from datetime import timedelta

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            device_id = rank % num_devices
            torch.cuda.set_device(device_id)
            logger.info(f"Rank {rank} using cuda:{device_id} (visible devices: {num_devices})")

        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            timeout = timedelta(hours=timeout_hours)
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timeout)
            logger.info(f"Initialized distributed backend: {backend} (rank {rank}/{world_size}, timeout={timeout_hours}h)")

    return rank, world_size


def cleanup_distributed():
    """Cleanup distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def shard_data(data: List[Any], rank: int, world_size: int) -> List[Any]:
    """Shard data across processes.

    Args:
        data: List of data items
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        Subset of data for this rank
    """
    return data[rank::world_size]


def gather_results(local_results: List[Any], world_size: int) -> List[Any]:
    """Gather results from all processes.

    Args:
        local_results: Results from this process
        world_size: Total number of processes

    Returns:
        Combined results from all processes (on rank 0), or local_results (on other ranks)
    """
    if world_size <= 1 or not dist.is_initialized():
        return local_results

    # Gather from all ranks
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, local_results)

    # Merge results
    merged = []
    for results in gather_list:
        if results:
            merged.extend(results)

    return merged
