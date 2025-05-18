# MALTopic: Multi-Agent LLM Topic Modeling Library

MALTopic is a powerful library designed for topic modeling using a multi-agent approach. It leverages the capabilities of large language models (LLMs) to enhance the analysis of survey responses by integrating structured and unstructured data.

## Features

- **Multi-Agent Framework**: Decomposes topic modeling into specialized tasks executed by individual LLM agents.
- **Data Enrichment**: Enhances textual responses using structured and categorical survey data.
- **Latent Theme Extraction**: Extracts meaningful topics from enriched responses.
- **Topic Deduplication**: Refines and consolidates identified topics for better interpretability.

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
mal_topic = MALTopic(api_key='your_api_key', model_name='gpt-4o-mini', llm_type='openai')

# Call individual agent functions
# TODO: Add examples of use
```

## Agents

- **Enrichment Agent**: Enhances free-text responses using structured data.
- **Topic Modeling Agent**: Extracts latent themes from enriched responses.
- **Deduplication Agent**: Refines and consolidates the extracted topics.

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
