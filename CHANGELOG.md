# Changelog

All notable changes to MALTopic will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-06-06

### Added
- **Intelligent Topic Deduplication**: New `deduplicate_topics()` method that uses LLM-powered semantic analysis to consolidate similar topics
- **Advanced Topic Merging**: Smart consolidation that preserves meaningful distinctions while merging topics with >80% semantic overlap
- **Context-Aware Processing**: Deduplication considers survey context to make better merging decisions
- **Robust Error Handling**: Graceful fallback to original topics if deduplication fails
- **Structure Preservation**: Maintains original topic structure while intelligently combining:
  - Names: Chooses most descriptive and comprehensive names
  - Descriptions: Combines descriptions to capture all relevant aspects
  - Relevance: Merges relevance information from source topics
  - Representative Words: Combines word lists, removing duplicates

### Improved
- Enhanced topic quality through semantic consolidation
- Better user control over topic refinement process
- Updated documentation with comprehensive usage examples
- Enhanced method reference documentation

### Changed
- Updated project description to highlight intelligent deduplication capabilities
- Deduplication Agent is now fully supported (removed "not supported yet" notation)

## [1.1.0] - 2025-06-01

### Added
- **Automatic Batching**: Intelligent handling of large datasets that exceed LLM token limits
- **Smart Token Counting**: Uses `tiktoken` library for precise batch sizing based on actual token counts
- **Automatic Error Detection**: Seamless fallback to batching mode when token limits are exceeded
- **Topic Consolidation**: Automatic merging and deduplication of topics across batches
- **Progress Tracking**: Visual progress indicators for batch processing operations
- **Robust Fallback**: Graceful degradation when `tiktoken` is not available

### Improved
- Enhanced error handling and user feedback
- Better performance for large-scale survey datasets
- No code changes required - existing workflows continue to work

## [1.0.0] - 2025-05-01

### Added
- **Multi-Agent Framework**: Core architecture for topic modeling using specialized LLM agents
- **Data Enrichment**: Enhancement of free-text responses using structured and categorical survey data
- **Topic Extraction**: Basic functionality for extracting latent themes from enriched responses
- **Enrichment Agent**: Specialized agent for enhancing textual responses with structured data
- **Topic Modeling Agent**: Dedicated agent for extracting meaningful topics from enriched data
- Initial API design with `enrich_free_text_with_structured_data()` and `generate_topics()` methods
- Support for OpenAI GPT models
- Comprehensive test suite
- MIT License
- Basic documentation and usage examples

### Infrastructure
- Poetry-based project structure
- Python 3.12+ support
- Core dependencies: openai, pandas, tqdm, tiktoken
- Automated testing with pytest
- Package publishing to PyPI

---

## Legend

- **Added**: New features
- **Changed**: Changes in existing functionality  
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements
- **Improved**: Enhancements to existing features
