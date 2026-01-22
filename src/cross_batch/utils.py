"""Shared utility functions for cross-batch module."""

from transformers import PreTrainedTokenizer


def is_instruct_model(model_name_or_tokenizer) -> bool:
    """Check if a model is an instruct/chat model based on its name or tokenizer.

    Args:
        model_name_or_tokenizer: Either a model name string or a tokenizer object

    Returns:
        True if the model appears to be an instruct/chat model
    """
    if hasattr(model_name_or_tokenizer, 'name_or_path'):
        name = model_name_or_tokenizer.name_or_path.lower()
    else:
        name = str(model_name_or_tokenizer).lower()

    instruct_keywords = ['instruct', 'chat', 'it', 'rlhf', 'dpo', 'sft']
    return any(kw in name for kw in instruct_keywords)


def get_eos_token(tokenizer: PreTrainedTokenizer) -> str:
    """Get the appropriate EOS token for a tokenizer.

    For chat models like Qwen, use <|im_end|> instead of the default EOS token.

    Args:
        tokenizer: The tokenizer to get EOS token for

    Returns:
        The EOS token string
    """
    eos_token = tokenizer.eos_token or ""
    # For Qwen models, use <|im_end|> as the stop token
    if hasattr(tokenizer, 'im_end_id') or '<|im_end|>' in tokenizer.get_vocab():
        eos_token = "<|im_end|>"
    return eos_token
