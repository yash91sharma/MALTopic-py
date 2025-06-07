# MALTopic: Multi-Agent LLM Topic Modeling Library

MALTopic is a powerful library designed for topic modeling using a multi-agent approach. It leverages the capabilities of large language models (LLMs) to enhance the analysis of survey responses by integrating structured and unstructured data.

MALTopic as a research paper was published in 2025 World AI IoT Congress. Links here.

## Features

- **Multi-Agent Framework**: Decomposes topic modeling into specialized tasks executed by individual LLM agents.
- **Data Enrichment**: Enhances textual responses using structured and categorical survey data.
- **Latent Theme Extraction**: Extracts meaningful topics from enriched responses.
- **Topic Deduplication**: Intelligently refines and consolidates identified topics using LLM-powered semantic analysis for better interpretability.
- **Automatic Batching**: Handles large datasets by automatically splitting data into manageable batches when token limits are exceeded.
- **Intelligent Error Handling**: Detects token limit errors and seamlessly switches to batching mode without user intervention.
- **Comprehensive Statistics Tracking**: Automatically tracks LLM usage, token consumption, API performance, and costs with detailed metrics and reporting.

## Installation

To install the MALTopic library, you can use pip:

```bash
pip install maltopic
```

## Usage

To use the MALTopic library, you need to initialize the main class with your API key and model name. You can choose between different LLMs such as OpenAI, Google Gemini (not supported yet), or Llama (not supported yet).

```python
from maltopic import MALTopic

# Initialize the MALTopic class
client = MALTopic(
    api_key="your_api_key",
    default_model_name="gpt-4.1-nano",
    llm_type="openai",
)

enriched_df = client.enrich_free_text_with_structured_data(
        survey_context="context about survey, why, how of it...",
        free_text_column="column_1",
        structured_data_columns=["columns_2", "column_3"],
        df=df,
        examples=["free text response, category 1 -> free text response with additional context", "..."], # optional
    )

topics = client.generate_topics(
        topic_mining_context="context about what kind of topics you want to mine",
        df=enriched_df,
        enriched_column="column_1" + "_enriched", # MALTopic adds _enriched as the suffix.
    )

# Optionally deduplicate and merge similar topics for cleaner results
deduplicated_topics = client.deduplicate_topics(
        topics=topics,
        survey_context="context about survey, why, how of it...",
    )

print(deduplicated_topics)

# Access comprehensive statistics anytime
stats = client.get_stats()
print(f"Total tokens used: {stats['overview']['total_tokens_used']:,}")
print(f"API calls made: {stats['overview']['total_calls_made']}")
print(f"Success rate: {stats['overview']['success_rate_percent']}%")

# Print detailed formatted statistics
client.print_stats()
```

## Automatic Batching for Large Datasets

MALTopic v1.1.0 introduces intelligent automatic batching to handle large datasets that may exceed LLM token limits. This feature works seamlessly in the background:

### How It Works

1. **Automatic Detection**: When `generate_topics` encounters a token limit error, it automatically detects this and switches to batching mode.

2. **Smart Splitting**: The library uses `tiktoken` (OpenAI's token counting library) to intelligently split your data into optimally-sized batches based on actual token counts.

3. **Batch Processing**: Each batch is processed independently, with progress tracking to keep you informed.

4. **Topic Consolidation**: Topics from all batches are automatically merged and deduplicated to provide a clean, comprehensive result.

### Key Benefits

- **No Code Changes Required**: Existing code works without modification - batching happens automatically when needed.
- **Optimal Performance**: Uses actual token counting for precise batch sizing, maximizing efficiency.
- **Robust Fallback**: Even works without `tiktoken` by falling back to simple batch splitting.
- **Progress Visibility**: Shows batch processing progress so you know what's happening.
- **Quality Preservation**: Maintains topic quality through intelligent consolidation and deduplication.

### Example Output

When batching is triggered, you'll see output like:
```
Token limit exceeded, splitting into batches...
Processing 3 batches...
Processing batches: 100%|██████████| 3/3 [00:45<00:00, 15.2s/it]
Batch 1/3: Generated 12 topics
Batch 2/3: Generated 8 topics  
Batch 3/3: Generated 10 topics
Consolidated 30 topics into 25 unique topics
```

This feature makes MALTopic suitable for processing large-scale survey datasets without worrying about token limitations.

## Intelligent Topic Deduplication

MALTopic v1.2.0 introduces intelligent topic deduplication that goes beyond simple string matching to provide semantic analysis and consolidation of similar topics.

### How It Works

1. **Semantic Analysis**: Uses LLM to analyze topic meanings, descriptions, and context rather than just comparing names.

2. **Smart Merging**: Identifies topics with significant semantic overlap (>80% similarity) and intelligently merges them while preserving unique perspectives.

3. **Structure Preservation**: Maintains the original topic structure and combines information from merged topics:
   - **Names**: Chooses the most descriptive and comprehensive name
   - **Descriptions**: Combines descriptions to capture all relevant aspects  
   - **Relevance**: Merges relevance information from all source topics
   - **Representative Words**: Combines word lists, removing duplicates

4. **Quality Preservation**: Preserves genuinely unique topics that represent distinct concepts with no significant overlap.

### Key Benefits

- **Higher Quality Results**: Eliminates redundant or highly similar topics for cleaner analysis
- **Semantic Understanding**: Goes beyond keyword matching to understand topic meanings
- **Flexible Control**: Can be used optionally - existing workflows continue to work unchanged
- **Robust Fallback**: Returns original topics unchanged if deduplication fails
- **Context-Aware**: Uses survey context to make better merging decisions

### Usage Example

```python
# Generate topics as usual
topics = client.generate_topics(
    topic_mining_context="Extract themes from customer feedback",
    df=enriched_df,
    enriched_column="feedback_enriched"
)

# Apply intelligent deduplication
deduplicated_topics = client.deduplicate_topics(
    topics=topics,
    survey_context="Customer satisfaction survey for mobile app"
)

print(f"Original topics: {len(topics)}")
print(f"After deduplication: {len(deduplicated_topics)}")
```

### Example Output

When deduplication is applied, you'll see output like:
```
Deduplicated 15 topics into 12 unique topics
```

This feature is particularly useful when:
- Working with large datasets that produce many overlapping topics
- You need cleaner, more consolidated results for reporting
- Multiple batches have generated similar topics that need consolidation

## Comprehensive Statistics Tracking

MALTopic includes built-in statistics tracking that automatically monitors your LLM usage, providing valuable insights into token consumption, API performance, and costs.

### Key Metrics Tracked

- **Token Usage**: Input, output, and total tokens from all API calls
- **API Performance**: Call counts, success/failure rates, and response times  
- **Model Breakdown**: Statistics separated by each model used
- **Cost Monitoring**: Data needed to calculate estimated API costs
- **Real-time Updates**: Statistics update automatically as you use the library

### Accessing Statistics

MALTopic provides three simple methods to access your usage statistics:

```python
# Get comprehensive statistics as a dictionary
stats = client.get_stats()
print(f"Total tokens used: {stats['overview']['total_tokens_used']:,}")
print(f"Average response time: {stats['averages']['avg_response_time_seconds']:.2f}s")

# Print a formatted summary to console
client.print_stats()

# Reset statistics to start fresh
client.reset_stats()
```

### Example Statistics Output

When you call `client.print_stats()`, you'll see output like:

```
============================================================
MALTopic Library Usage Statistics
============================================================

📊 Overview:
  Total Tokens Used: 2,450
  - Input Tokens: 1,800
  - Output Tokens: 650
  Total API Calls: 8
  - Successful: 8
  - Failed: 0
  Success Rate: 100.0%
  Uptime: 125.3 seconds

📈 Averages:
  Avg Tokens per Call: 306.3
  - Avg Input Tokens: 225.0
  - Avg Output Tokens: 81.3
  Avg Response Time: 2.15s

🤖 Model Breakdown:
  gpt-4:
    Calls: 8 (Success: 8, Failed: 0)
    Tokens: 2,450 (Avg: 306.3)
    Success Rate: 100.0%
============================================================
```

### Cost Estimation Example

Use the statistics to estimate your API costs:

```python
stats = client.get_stats()

# Example with GPT-4 pricing (as of 2024)
input_cost = (stats['overview']['total_input_tokens'] / 1000) * 0.03  # $0.03 per 1K input tokens
output_cost = (stats['overview']['total_output_tokens'] / 1000) * 0.06  # $0.06 per 1K output tokens
total_estimated_cost = input_cost + output_cost

print(f"Estimated API cost: ${total_estimated_cost:.4f}")
```

### Benefits

- **Cost Control**: Monitor token usage to manage API expenses
- **Performance Optimization**: Identify bottlenecks and optimize prompts
- **Error Monitoring**: Track success rates to catch issues early
- **Usage Insights**: Understand patterns across different models and operations

Statistics tracking is **automatic** and **privacy-focused** - no data leaves your environment, and statistics are stored only in memory during your session.

## Method Reference

### Core Methods

#### `enrich_free_text_with_structured_data()`
Enhances free-text survey responses with structured data context.

**Parameters:**
- `survey_context` (str): Context about the survey purpose and methodology
- `free_text_column` (str): Name of the column containing free-text responses
- `structured_data_columns` (list[str]): List of column names with structured data to use for enrichment
- `df` (pandas.DataFrame): DataFrame containing the survey data
- `examples` (list[str], optional): Examples of enrichment format

**Returns:** DataFrame with enriched text in a new column with "_enriched" suffix

#### `generate_topics()`
Extracts latent themes and topics from enriched survey responses.

**Parameters:**
- `topic_mining_context` (str): Context about what kind of topics to extract
- `df` (pandas.DataFrame): DataFrame containing enriched data
- `enriched_column` (str): Name of the column containing enriched text

**Returns:** List of topic dictionaries with structure:
```python
{
    "name": "Topic Name",
    "description": "Detailed description of the topic",
    "relevance": "Who this topic is relevant to",
    "representative_words": ["word1", "word2", "word3"]
}
```

#### `deduplicate_topics()`
Intelligently consolidates similar topics using semantic analysis.

**Parameters:**
- `topics` (list[dict]): List of topic dictionaries to deduplicate
- `survey_context` (str): Context about the survey to help with merging decisions

**Returns:** List of deduplicated topic dictionaries with the same structure as input

#### `get_stats()`
Returns comprehensive statistics about LLM usage and performance.

**Returns:** Dictionary containing:
- `overview`: Total tokens, calls, success rates, and uptime
- `averages`: Average tokens per call, response times, etc.
- `model_breakdown`: Statistics separated by model
- `recent_calls`: Details of the most recent API calls

#### `print_stats()`
Prints a formatted summary of statistics to the console.

**Returns:** None (prints to console)

#### `reset_stats()`
Resets all statistics to zero and starts tracking fresh.

**Returns:** None

## Agents

- **Enrichment Agent**: Enhances free-text responses using structured data.
- **Topic Modeling Agent**: Extracts latent themes from enriched responses.
- **Deduplication Agent**: Intelligently refines and consolidates the extracted topics using LLM-powered semantic analysis.

## Changelog

For detailed release notes and version history, see [CHANGELOG.md](https://github.com/yash91sharma/MALTopic-py/blob/master/CHANGELOG.md).

### v1.3.0 (June 2025)
- **NEW**: Comprehensive statistics tracking with automatic LLM usage monitoring
- **NEW**: `get_stats()`, `print_stats()`, and `reset_stats()` methods for statistics access
- **NEW**: Real-time token usage, API performance, and cost monitoring
- **NEW**: Model-specific statistics breakdown and detailed metrics
- **IMPROVED**: Enhanced visibility into LLM usage patterns and costs

### v1.2.0 (June 2025)
- **NEW**: Intelligent topic deduplication using LLM-powered semantic analysis
- **NEW**: `deduplicate_topics()` method for consolidating similar topics
- **NEW**: Advanced topic merging that preserves meaningful distinctions
- **NEW**: Context-aware deduplication that considers survey background
- **NEW**: Robust error handling with fallback to original topics
- **IMPROVED**: Enhanced topic quality through semantic consolidation
- **IMPROVED**: Better user control over topic refinement process

### v1.1.0 (June 2025)
- **NEW**: Automatic batching for large datasets that exceed LLM token limits
- **NEW**: Intelligent token counting using tiktoken for optimal batch sizing
- **NEW**: Automatic error detection and seamless fallback to batching mode
- **NEW**: Topic consolidation and deduplication across batches
- **NEW**: Progress tracking for batch processing operations
- **IMPROVED**: Enhanced error handling and user feedback
- **IMPROVED**: Graceful degradation when tiktoken is not available

### v1.0.0 (May 2025)
- Multi-agent framework for topic modeling
- Data enrichment capabilities  
- Basic topic extraction functionality

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Citation

If you use MALTopic in your research, please cite:

```bibtex
@software{Sharma2025maltopic,
  author = {Sharma, Yash},
  title = {MALTopic: A library for topic modeling},
  year = {2025},
  url = {https://github.com/yash91sharma/MALTopic-py}
}
