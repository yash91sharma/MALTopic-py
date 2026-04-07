# Changelog

All notable changes to MALTopic will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.1] - 06 April, 2026

### Improved
- Improved documentation

## [1.5.0] - 06 April, 2026

### Added
- **GUI**: New GUI mode, where users can use Maltopic without any code. Just enter `maltopic-gui` in the terminal after installing and get started.

## [1.4.1] - 11 March, 2026

### Fixed
- Minor code/maintenance improvements.

## [1.4.0] - 16 December, 2025

### Added
- **Model Parameter Control**: New `override_model_params` parameter in `MALTopic` initialization for fine-grained control over OpenAI API parameters
- **Flexible Configuration**: Three modes of operation:
  - `None` (default): Automatic intelligent parameter handling based on model type
  - `dict`: Custom parameters that completely replace defaults
  - `{}`: Minimal parameters mode with only base required parameters

### Fixed
- Resolved 400 errors when using GPT-5 and reasoning models with unsupported parameters

## [1.3.2] - 29 August, 2025
- Added a link to the paper.

## [1.3.1] - 07 June, 2025
- Update changelog and readme files.

## [1.3.0] - 07 June, 2025

### Added
- **Comprehensive Statistics Tracking**: Automatic monitoring of LLM usage, token consumption, API performance, and costs
- **New Methods**: `get_stats()`, `print_stats()`, and `reset_stats()` for accessing and managing statistics
- **Model-Specific Statistics**: Detailed breakdown of usage and performance by model
- **Real-Time Updates**: Statistics update automatically as the library is used
- **Cost Monitoring**: Track data needed for API cost estimation
- **Recent Calls**: Access details of the most recent API calls

### Improved
- Enhanced visibility into LLM usage patterns and costs
- Improved documentation and usage examples for statistics features
- More robust and user-friendly statistics reporting

### Changed
- Statistics tracking is now enabled by default and privacy-focused (in-memory only)
- Updated documentation to reflect new statistics features

## [1.2.0] - 06 June, 2025

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

## [1.1.0] - 01 June, 2025

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

## [1.0.0] - 01 June, 2025

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

---
