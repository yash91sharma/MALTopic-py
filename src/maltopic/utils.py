import json

import pandas as pd
from tqdm import tqdm

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


def validate_topic_structure(topics: list[dict[str, str]]) -> None:
    """
    Validate that topics have the expected structure.

    Args:
        topics: List of topic dictionaries to validate

    Raises:
        ValueError: If topics don't have the expected structure
    """
    required_fields = {"name", "description", "relevance", "representative_words"}

    for i, topic in enumerate(topics):
        if not isinstance(topic, dict):
            raise ValueError(f"Topic {i} is not a dictionary")

        missing_fields = required_fields - set(topic.keys())
        if missing_fields:
            raise ValueError(f"Topic {i} missing required fields: {missing_fields}")

        # Ensure all values are strings except representative_words which can be a list
        for field, value in topic.items():
            if field == "representative_words":
                if not isinstance(value, (list, str)):
                    raise ValueError(
                        f"Topic {i} field '{field}' must be a list or string"
                    )
            else:
                if not isinstance(value, str):
                    raise ValueError(f"Topic {i} field '{field}' must be a string")


def generate_topics_from_text(
    llm_client, instructions: str, input_text: str
) -> list[dict[str, str]]:
    """
    Generate topics from a single text input.

    Args:
        llm_client: The LLM client instance
        instructions: The instruction prompt for the LLM
        input_text: The input text containing all responses

    Returns:
        List of topic dictionaries
    """
    raw_response = llm_client.generate(instructions=instructions, input=input_text)

    return parse_topics_response(raw_response)


def generate_topics_with_batching(
    llm_client,
    instructions: str,
    labeled_columns: list[str],
    topic_mining_context: str,
    default_model_name: str,
) -> list[dict[str, str]]:
    """
    Generate topics using batching when token limits are exceeded.

    Args:
        llm_client: The LLM client instance
        instructions: The instruction prompt for the LLM
        labeled_columns: List of labeled response strings
        topic_mining_context: Context for topic mining
        default_model_name: Default model name for token counting

    Returns:
        Consolidated list of topic dictionaries
    """
    try:
        batches = split_text_into_batches(
            labeled_columns,
            max_tokens_per_batch=100000,
            model_name=default_model_name,
        )
    except ImportError:
        # Fallback to simple batching if tiktoken is not available
        batch_size = max(1, len(labeled_columns) // 4)  # Split into ~4 batches
        batches = [
            labeled_columns[i : i + batch_size]
            for i in range(0, len(labeled_columns), batch_size)
        ]

    print(f"Processing {len(batches)} batches...")

    all_topics = []

    for i, batch in enumerate(tqdm(batches, desc="Processing batches")):
        batch_input = "\n\n".join(batch)

        try:
            batch_topics = generate_topics_from_text(
                llm_client, instructions, batch_input
            )
            all_topics.extend(batch_topics)
            print(f"Batch {i+1}/{len(batches)}: Generated {len(batch_topics)} topics")
        except Exception as e:
            print(f"Error processing batch {i+1}: {str(e)}")
            continue

    return consolidate_topics(all_topics)


def parse_topics_response(raw_response: str) -> list[dict[str, str]]:
    """
    Parse the LLM response into topic dictionaries.

    Args:
        raw_response: Raw JSON response from LLM

    Returns:
        List of topic dictionaries
    """
    topics = []

    try:
        parsed_topics = json.loads(raw_response)
        for topic in parsed_topics:
            for key in topic:
                if key != "representative_words" and not isinstance(topic[key], str):
                    topic[key] = str(topic[key])
            topics.append(topic)
    except json.JSONDecodeError:
        raise ValueError(
            f"Failed to parse LLM response as JSON: {raw_response[:100]}..."
        )
    except Exception as e:
        raise ValueError(f"Error processing topics: {str(e)}")

    return topics


def consolidate_topics(
    all_topics: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Consolidate topics from multiple batches, removing duplicates and merging similar ones.

    This is a dumb(er) method. Use the dedup agent for a smarter consolidation.

    Args:
        all_topics: List of all topics from different batches

    Returns:
        Consolidated list of unique topics
    """
    if not all_topics:
        return []

    # Simple deduplication based on topic names
    seen_names = set()
    unique_topics = []

    for topic in all_topics:
        topic_name = topic.get("name", "").lower().strip()
        if topic_name and topic_name not in seen_names:
            seen_names.add(topic_name)
            unique_topics.append(topic)

    print(
        f"Consolidated {len(all_topics)} topics into {len(unique_topics)} unique topics"
    )
    return unique_topics
