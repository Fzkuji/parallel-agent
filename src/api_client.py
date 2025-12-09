"""API client for OpenAI-compatible inference endpoints.

Supports various API providers:
- OpenAI (api.openai.com)
- OpenRouter (openrouter.ai)
- Together (api.together.xyz)
- Deepseek (api.deepseek.com)
- Any OpenAI-compatible endpoint
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import httpx
except ImportError:
    httpx = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class APIResponse:
    """Response from API generation."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float


class APIClient:
    """Client for OpenAI-compatible API endpoints.

    Supports multiple providers through different base URLs:
    - OpenAI: https://api.openai.com/v1
    - OpenRouter: https://openrouter.ai/api/v1
    - Together: https://api.together.xyz/v1
    - Deepseek: https://api.deepseek.com/v1

    Environment variables:
    - OPENAI_API_KEY: For OpenAI
    - OPENROUTER_API_KEY: For OpenRouter
    - TOGETHER_API_KEY: For Together
    - DEEPSEEK_API_KEY: For Deepseek
    - API_BASE_URL: Custom base URL (optional)
    - API_KEY: Generic API key (fallback)
    """

    # Known providers and their configurations
    PROVIDERS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "env_key": "OPENAI_API_KEY",
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "env_key": "OPENROUTER_API_KEY",
        },
        "together": {
            "base_url": "https://api.together.xyz/v1",
            "env_key": "TOGETHER_API_KEY",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "env_key": "DEEPSEEK_API_KEY",
        },
        "google-vertex": {
            "base_url": "https://openrouter.ai/api/v1",  # OpenRouter proxy for Vertex AI
            "env_key": "OPENROUTER_API_KEY",
        },
    }

    def __init__(
        self,
        model: str,
        *,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: float = 120.0,
        provider_order: Optional[List[str]] = None,
    ) -> None:
        """Initialize API client.

        Args:
            model: Model identifier (e.g., 'gpt-4o', 'deepseek-chat', 'Qwen/Qwen2.5-72B-Instruct')
            provider: Provider name ('openai', 'openrouter', 'together', 'deepseek') or None for auto-detect
            base_url: Custom base URL (overrides provider default)
            api_key: API key (overrides environment variable)
            temperature: Sampling temperature (0.0 for deterministic)
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            provider_order: OpenRouter provider routing order (e.g., ['google-vertex', 'together'])
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.disable_thinking = self._is_thinking_model(model)
        self.provider_order = provider_order  # For OpenRouter provider routing

        # Auto-detect provider from model name if not specified
        if provider is None:
            provider = self._detect_provider(model)

        self.provider = provider

        # Resolve base URL
        if base_url:
            self.base_url = base_url
        elif provider and provider in self.PROVIDERS:
            self.base_url = self.PROVIDERS[provider]["base_url"]
        else:
            # Default to OpenRouter for provider routing support
            self.base_url = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")

        # Resolve API key
        if api_key:
            self.api_key = api_key
        elif provider and provider in self.PROVIDERS:
            env_key = self.PROVIDERS[provider]["env_key"]
            self.api_key = os.environ.get(env_key) or os.environ.get("API_KEY")
        else:
            self.api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                f"No API key found. Set {self.PROVIDERS.get(provider, {}).get('env_key', 'API_KEY')} "
                "environment variable or pass api_key parameter."
            )

        # Initialize client
        self._init_client()

        logging.info(
            "Initialized API client: provider=%s, model=%s, base_url=%s, provider_order=%s, disable_thinking=%s",
            self.provider, self.model, self.base_url, self.provider_order, self.disable_thinking
        )

    def _is_thinking_model(self, model: str) -> str:
        """Check if model is a thinking/reasoning model and return the disable method.

        Returns:
            - "max_tokens": Use reasoning.max_tokens=0 (Qwen3, DeepSeek)
            - "mandatory": Reasoning cannot be disabled, will output in reasoning_content field
            - "": Not a thinking model, no special handling needed
        """
        model_lower = model.lower()
        # Qwen3 models have thinking mode by default - use max_tokens method
        if "qwen3" in model_lower or "qwen-3" in model_lower:
            return "max_tokens"
        # DeepSeek R1 and similar reasoning models - use max_tokens method
        if "deepseek-r1" in model_lower or "deepseek-reasoner" in model_lower:
            return "max_tokens"
        # gpt-oss-120b has mandatory reasoning - cannot be disabled
        # The actual answer appears in message.content, reasoning in reasoning_content
        if "gpt-oss" in model_lower:
            return "mandatory"
        # o1, o3 reasoning models from OpenAI - reasoning is mandatory
        if model_lower.startswith(("o1", "o3")):
            return "mandatory"
        return ""

    def _detect_provider(self, model: str) -> str:
        """Auto-detect provider from model name."""
        model_lower = model.lower()

        if model_lower.startswith("deepseek"):
            return "deepseek"
        elif "/" in model:
            # Models with org/name format are typically on OpenRouter or Together
            # Check environment variables to determine
            if os.environ.get("OPENROUTER_API_KEY"):
                return "openrouter"
            elif os.environ.get("TOGETHER_API_KEY"):
                return "together"
        elif model_lower.startswith(("gpt-", "o1-", "chatgpt")):
            return "openai"

        # Default to openrouter as it supports most models
        return "openrouter"

    def _init_client(self) -> None:
        """Initialize the OpenAI client."""
        if OpenAI is None:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        disable_thinking: bool = False,
    ) -> APIResponse:
        """Generate a completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature
            stop: Stop sequences
            disable_thinking: If True, disable thinking mode for Qwen3/reasoning models

        Returns:
            APIResponse with generated text and token counts
        """
        temp = temperature if temperature is not None else self.temperature

        start_time = time.perf_counter()

        # Build extra parameters for the API call
        extra_params: Dict[str, Any] = {}

        # Disable thinking mode for reasoning models (where possible)
        # Different models use different parameters:
        # - Qwen3, DeepSeek: reasoning.max_tokens = 0
        # - gpt-oss-120b, OpenAI o1/o3: reasoning is mandatory, cannot be disabled
        thinking_method = self.disable_thinking
        if thinking_method == "max_tokens":
            extra_params["reasoning"] = {"max_tokens": 0}
        # Note: "mandatory" models cannot have reasoning disabled, so we don't add any params

        # OpenRouter provider routing: specify which providers to use
        # See: https://openrouter.ai/docs/provider-routing
        if self.provider_order:
            extra_params["provider"] = {
                "order": self.provider_order,
                "allow_fallbacks": True,
            }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp,
            stop=stop,
            extra_body=extra_params if extra_params else None,
        )

        latency = time.perf_counter() - start_time

        # Extract response with error handling
        if response is None or not hasattr(response, 'choices') or not response.choices:
            logging.warning("API returned empty response, returning empty result")
            return APIResponse(
                text="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency=latency,
            )
        choice = response.choices[0]
        message = choice.message
        text = message.content or ""

        # Handle thinking models that may output reasoning in a separate field
        # Some APIs (like DeepSeek, Qwen) use reasoning_content field
        reasoning = getattr(message, 'reasoning_content', None) or ""

        # If text is empty but we have reasoning, the model might have put
        # everything in thinking tags. Log a warning.
        if not text and reasoning:
            logging.warning(
                "Model returned empty content but has reasoning_content. "
                "This model may require thinking mode handling."
            )
            # Optionally include reasoning in text for debugging
            # text = reasoning  # Uncomment if you want to use reasoning as fallback

        # Extract usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        return APIResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency=latency,
        )

    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        *,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> List[APIResponse]:
        """Generate completions for a batch of message lists.

        Note: This processes requests sequentially. For true batching,
        use async methods or provider-specific batch APIs.

        Args:
            batch_messages: List of message lists
            max_tokens: Maximum tokens to generate per request
            temperature: Override default temperature
            stop: Stop sequences

        Returns:
            List of APIResponse objects
        """
        results = []
        for messages in batch_messages:
            result = self.generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            results.append(result)
        return results


class APIModelWrapper:
    """Wrapper to make APIClient compatible with local model interface.

    This allows using API-based models with existing strategy code
    that expects a transformers-style model.
    """

    def __init__(self, api_client: APIClient) -> None:
        self.api_client = api_client
        self.device = "api"  # Dummy device
        self._config = type("Config", (), {"model_type": "api"})()

    @property
    def config(self):
        return self._config


class APITokenizerWrapper:
    """Wrapper to make APIClient compatible with tokenizer interface.

    This provides tokenizer-like methods for API-based inference.
    Note: Token counting is approximate for API mode.
    """

    def __init__(
        self,
        api_client: APIClient,
        *,
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
    ) -> None:
        self.api_client = api_client
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.eos_token_id = 0  # Dummy ID
        self.pad_token_id = 0  # Dummy ID
        self.padding_side = "left"

        # Try to load tiktoken for token counting
        self._encoder = None
        try:
            import tiktoken
            # Use cl100k_base as default (GPT-4 encoding)
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> Union[str, List[int]]:
        """Format messages into a chat template string.

        For API mode, we just return the messages as-is since
        the API handles formatting internally.
        """
        # Return a serialized version for logging/debugging
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"[{role}]: {content}")
        return "\n".join(parts)

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: str = "pt",
        padding: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Tokenize text (approximate for API mode)."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Estimate token counts
        token_counts = []
        for t in texts:
            if self._encoder:
                count = len(self._encoder.encode(t))
            else:
                # Rough estimate: ~4 chars per token
                count = len(t) // 4
            token_counts.append(count)

        # Create dummy tensors for compatibility
        import torch
        max_len = max(token_counts) if token_counts else 0
        batch_size = len(texts)

        # Create dummy input_ids and attention_mask
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, max_len), dtype=torch.long)

        # Store actual token counts in the tensor shape
        # This is a hack to pass token counts through the existing interface
        class DummyTensor:
            def __init__(self, shape, counts):
                self._shape = shape
                self._counts = counts

            @property
            def shape(self):
                return self._shape

            def to(self, device):
                return self

            def sum(self, dim=None):
                if dim == 1:
                    return type("Tensor", (), {"tolist": lambda: self._counts})()
                return sum(self._counts)

        return {
            "input_ids": DummyTensor((batch_size, max_len), token_counts),
            "attention_mask": DummyTensor((batch_size, max_len), token_counts),
            "_token_counts": token_counts,
            "_texts": texts,
        }

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        """Decode tokens (no-op for API mode)."""
        # In API mode, we don't have actual tokens to decode
        # This is called on response text, which is already decoded
        if isinstance(token_ids, str):
            return token_ids
        return ""

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> List[int]:
        """Encode text to token IDs (approximate for API mode)."""
        if self._encoder:
            return self._encoder.encode(text)
        # Return dummy token IDs
        return list(range(len(text) // 4))


def create_api_inference(
    model: str,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    provider_order: Optional[List[str]] = None,
) -> Tuple[APITokenizerWrapper, APIModelWrapper, APIClient]:
    """Create API-based inference components.

    Returns tokenizer wrapper, model wrapper, and API client
    that can be used with existing strategy code.

    Args:
        model: Model identifier
        provider: Provider name or None for auto-detect
        base_url: Custom base URL
        api_key: API key
        temperature: Sampling temperature
        provider_order: OpenRouter provider routing order (e.g., ['google-vertex', 'together'])

    Returns:
        Tuple of (tokenizer, model, api_client)
    """
    client = APIClient(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        provider_order=provider_order,
    )

    tokenizer = APITokenizerWrapper(client)
    model_wrapper = APIModelWrapper(client)

    return tokenizer, model_wrapper, client
