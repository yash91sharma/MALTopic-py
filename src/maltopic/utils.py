import pandas as pd

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False


def validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Validates a Pandas DataFrame for required columns and non-emptiness.

    Args:
        df: The Pandas DataFrame to validate.
        required_columns: A list of column names that must be present in the DataFrame.

    Raises:
        ValueError: If the DataFrame is empty or if any required columns are missing.
    """
    if df.empty:
        raise ValueError("DataFrame is empty. It must contain at least one row.")

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"DataFrame is missing required columns: {', '.join(missing_columns)}"
        )

    return None


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string for a given model.

    Args:
        text: The text to count tokens for
        model_name: The model name to use for token counting (defaults to gpt-4)

    Returns:
        Number of tokens in the text

    Raises:
        ImportError: If tiktoken is not available
    """
    if not TIKTOKEN_AVAILABLE or tiktoken is None:
        raise ImportError(
            "tiktoken is required for token counting. Install with: pip install tiktoken"
        )

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to cl100k_base encoding for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def split_text_into_batches(
    labeled_responses: list[str],
    max_tokens_per_batch: int = 100000,
    model_name: str = "gpt-4",
) -> list[list[str]]:
    """
    Split a list of labeled responses into batches based on token limits.

    Args:
        labeled_responses: List of labeled text responses
        max_tokens_per_batch: Maximum tokens per batch (defaults to 100k)
        model_name: Model name for token counting

    Returns:
        List of batches, where each batch is a list of responses
    """
    if not labeled_responses:
        return []

    batches = []
    current_batch = []
    current_batch_tokens = 0

    # Reserve some tokens for instructions and formatting
    instruction_buffer = 2000
    effective_max_tokens = max_tokens_per_batch - instruction_buffer

    for response in labeled_responses:
        response_tokens = count_tokens(response, model_name)

        # If single response exceeds limit, include it alone (will be handled by error handling)
        if response_tokens > effective_max_tokens:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_tokens = 0
            batches.append([response])
            continue

        # If adding this response would exceed the limit, start a new batch
        if (
            current_batch
            and current_batch_tokens + response_tokens > effective_max_tokens
        ):
            batches.append(current_batch)
            current_batch = [response]
            current_batch_tokens = response_tokens
        else:
            current_batch.append(response)
            current_batch_tokens += response_tokens

    # Add the last batch if it has content
    if current_batch:
        batches.append(current_batch)

    return batches


def is_token_limit_error(error: Exception) -> bool:
    """
    Check if an exception is related to token limits.

    Args:
        error: The exception to check

    Returns:
        True if the error is related to token limits
    """
    error_str = str(error).lower()
    token_error_indicators = [
        "maximum context length",
        "token limit",
        "context_length_exceeded",
        "too many tokens",
        "context window",
        "input is too long",
    ]

    return any(indicator in error_str for indicator in token_error_indicators)
