# MALTopic: Multi-Agent LLM Topic Modeling Library

MALTopic is a powerful library designed for topic modeling using a multi-agent approach. It leverages the capabilities of large language models (LLMs) to enhance the analysis of survey responses by integrating structured and unstructured data.

MALTopic as a research paper was published in 2025 World AI IoT Congress. Links here.

## Features

- **Multi-Agent Framework**: Decomposes topic modeling into specialized tasks executed by individual LLM agents.
- **Data Enrichment**: Enhances textual responses using structured and categorical survey data.
- **Latent Theme Extraction**: Extracts meaningful topics from enriched responses.
- **Topic Deduplication**: Refines and consolidates identified topics for better interpretability.
- **Automatic Batching**: Handles large datasets by automatically splitting data into manageable batches when token limits are exceeded.
- **Intelligent Error Handling**: Detects token limit errors and seamlessly switches to batching mode without user intervention.

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

print(topics)
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

## Agents

- **Enrichment Agent**: Enhances free-text responses using structured data.
- **Topic Modeling Agent**: Extracts latent themes from enriched responses.
- **Deduplication Agent**: Refines and consolidates the extracted topics. (not supported yet)

## Changelog

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
