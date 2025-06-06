from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.maltopic import utils
from src.maltopic.core import MALTopic


class TestTokenHandling:
    """Test token counting and batching functionality."""

    @patch("src.maltopic.utils.tiktoken")
    def test_count_tokens_with_tiktoken(self, mock_tiktoken):
        """Test token counting when tiktoken is available."""
        # Mock tiktoken behavior
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        # Test with utils.TIKTOKEN_AVAILABLE = True
        with patch("src.maltopic.utils.TIKTOKEN_AVAILABLE", True):
            result = utils.count_tokens("test text", "gpt-4")
            assert result == 5

    def test_count_tokens_without_tiktoken(self):
        """Test token counting when tiktoken is not available."""
        with patch("src.maltopic.utils.TIKTOKEN_AVAILABLE", False):
            with patch("src.maltopic.utils.tiktoken", None):
                with pytest.raises(ImportError) as excinfo:
                    utils.count_tokens("test text")
                assert "tiktoken is required" in str(excinfo.value)

    @patch("src.maltopic.utils.tiktoken")
    def test_count_tokens_unknown_model(self, mock_tiktoken):
        """Test token counting with unknown model falls back to cl100k_base."""
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        # First call raises KeyError, second returns encoding
        mock_tiktoken.encoding_for_model.side_effect = KeyError("unknown model")
        mock_tiktoken.get_encoding.return_value = mock_encoding

        with patch("src.maltopic.utils.TIKTOKEN_AVAILABLE", True):
            result = utils.count_tokens("test", "unknown-model")
            assert result == 3
            mock_tiktoken.get_encoding.assert_called_with("cl100k_base")

    @patch("src.maltopic.utils.count_tokens")
    def test_split_text_into_batches(self, mock_count_tokens):
        """Test splitting text into batches based on token limits."""

        # Mock token counts: each response has different token counts
        def token_count_side_effect(text, model):
            token_map = {
                "resp1": 100,
                "resp2": 200,
                "resp3": 1500,
                "resp4": 300,
                "resp5": 400,
            }
            return token_map.get(text, 0)

        mock_count_tokens.side_effect = token_count_side_effect

        responses = ["resp1", "resp2", "resp3", "resp4", "resp5"]
        batches = utils.split_text_into_batches(responses, max_tokens_per_batch=2500)

        # effective_max_tokens = 2500 - 2000 = 500
        # resp1 (100) -> batch 1 (100 < 500)
        # resp2 (200) -> batch 1 (100 + 200 = 300 < 500)
        # resp3 (1500) -> exceeds effective limit, goes alone in batch 2
        # resp4 (300) -> batch 3 (300 < 500)
        # resp5 (400) -> batch 3 (300 + 400 = 700 > 500), so new batch 4

        assert len(batches) == 4
        assert batches[0] == ["resp1", "resp2"]
        assert batches[1] == ["resp3"]  # Exceeds limit, goes alone
        assert batches[2] == ["resp4"]
        assert batches[3] == ["resp5"]

    def test_split_text_empty_input(self):
        """Test splitting empty input returns empty list."""
        result = utils.split_text_into_batches([])
        assert result == []

    def test_is_token_limit_error(self):
        """Test detection of token limit errors."""
        # Test various token limit error messages
        token_errors = [
            Exception("maximum context length exceeded"),
            Exception("Token limit reached"),
            Exception("context_length_exceeded"),
            Exception("Input is too long for this model"),
            Exception("too many tokens in request"),
        ]

        for error in token_errors:
            assert utils.is_token_limit_error(error) is True

        # Test non-token errors
        other_errors = [
            Exception("Network error"),
            Exception("Invalid API key"),
            Exception("Model not found"),
        ]

        for error in other_errors:
            assert utils.is_token_limit_error(error) is False


class TestGenerateTopicsWithBatching:
    """Test the updated generate_topics method with batching functionality."""

    @pytest.fixture
    def sample_enriched_df(self):
        return pd.DataFrame(
            {
                "response_enriched": [
                    "Response 1 about topic A",
                    "Response 2 about topic B",
                    "Response 3 about topic A",
                    "Response 4 about topic C",
                    "Response 5 about topic B",
                ]
            }
        )

    @patch("src.maltopic.core.utils.validate_dataframe")
    def test_generate_topics_success_no_batching(
        self, mock_validate, sample_enriched_df
    ):
        """Test successful topic generation without batching."""
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.default_model_name = "gpt-4"

        # Mock successful response
        mock_response = '[{"name": "Topic A", "description": "About A", "relevance": "High", "representative_words": ["word1", "word2"]}]'
        m.llm_client.generate.return_value = mock_response

        topics = m.generate_topics(
            topic_mining_context="context",
            df=sample_enriched_df,
            enriched_column="response_enriched",
        )

        assert len(topics) == 1
        assert topics[0]["name"] == "Topic A"
        mock_validate.assert_called_once()

    @patch("src.maltopic.core.utils.validate_dataframe")
    @patch("src.maltopic.core.utils.is_token_limit_error")
    @patch("src.maltopic.core.utils.split_text_into_batches")
    def test_generate_topics_with_batching(
        self, mock_split, mock_is_token_error, mock_validate, sample_enriched_df
    ):
        """Test topic generation with batching when token limit is exceeded."""
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.default_model_name = "gpt-4"

        # First call raises token limit error, subsequent calls succeed
        token_error = Exception("maximum context length exceeded")
        batch1_response = '[{"name": "Topic A", "description": "About A", "relevance": "High", "representative_words": ["word1"]}]'
        batch2_response = '[{"name": "Topic B", "description": "About B", "relevance": "Medium", "representative_words": ["word2"]}]'

        m.llm_client.generate.side_effect = [
            token_error,
            batch1_response,
            batch2_response,
        ]

        # Mock token error detection
        mock_is_token_error.return_value = True

        # Mock batching
        mock_split.return_value = [
            ["1: Response 1", "2: Response 2"],
            ["3: Response 3", "4: Response 4"],
        ]

        topics = m.generate_topics(
            topic_mining_context="context",
            df=sample_enriched_df,
            enriched_column="response_enriched",
        )

        # Should get topics from both batches, deduplicated
        assert len(topics) == 2
        assert topics[0]["name"] == "Topic A"
        assert topics[1]["name"] == "Topic B"

        # Verify batching was called
        mock_split.assert_called_once()
        mock_is_token_error.assert_called_once_with(token_error)

    @patch("src.maltopic.core.utils.validate_dataframe")
    def test_generate_topics_non_token_error(self, mock_validate, sample_enriched_df):
        """Test that non-token errors are re-raised as RuntimeError."""
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.default_model_name = "gpt-4"

        # Mock a non-token error
        network_error = Exception("Network connection failed")
        m.llm_client.generate.side_effect = network_error

        with pytest.raises(RuntimeError) as excinfo:
            m.generate_topics(
                topic_mining_context="context",
                df=sample_enriched_df,
                enriched_column="response_enriched",
            )

        assert "Error generating topics" in str(excinfo.value)
        assert "Network connection failed" in str(excinfo.value)

    def test_parse_topics_response_success(self):
        """Test successful parsing of topics response."""
        json_response = """[
            {
                "name": "Topic 1",
                "description": "Description 1", 
                "relevance": "High",
                "representative_words": ["word1", "word2"]
            }
        ]"""

        topics = utils.parse_topics_response(json_response)

        assert len(topics) == 1
        assert topics[0]["name"] == "Topic 1"
        assert topics[0]["description"] == "Description 1"

    def test_parse_topics_response_json_error(self):
        """Test handling of invalid JSON in topics response."""
        with pytest.raises(ValueError) as excinfo:
            utils.parse_topics_response("invalid json")

        assert "Failed to parse LLM response as JSON" in str(excinfo.value)

    def test_consolidate_topics(self):
        """Test consolidation of topics from multiple batches."""
        topics = [
            {"name": "Topic A", "description": "About A"},
            {"name": "Topic B", "description": "About B"},
            {"name": "Topic A", "description": "About A again"},  # Duplicate
            {"name": "Topic C", "description": "About C"},
        ]

        consolidated = utils.consolidate_topics(topics)

        # Should remove the duplicate "Topic A"
        assert len(consolidated) == 3
        topic_names = [t["name"] for t in consolidated]
        assert "Topic A" in topic_names
        assert "Topic B" in topic_names
        assert "Topic C" in topic_names

    def test_consolidate_topics_empty(self):
        """Test consolidation with empty input."""
        result = utils.consolidate_topics([])
        assert result == []

    @patch("src.maltopic.core.utils.split_text_into_batches")
    def test_generate_topics_batching_fallback(self, mock_split, sample_enriched_df):
        """Test fallback batching when tiktoken is not available."""
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.default_model_name = "gpt-4"

        # Mock ImportError from split_text_into_batches
        mock_split.side_effect = ImportError("tiktoken not available")

        # Mock token limit error to trigger batching
        token_error = Exception("maximum context length exceeded")
        batch_response = '[{"name": "Topic A", "description": "About A", "relevance": "High", "representative_words": ["word1"]}]'

        m.llm_client.generate.side_effect = [token_error, batch_response]

        with patch("src.maltopic.core.utils.is_token_limit_error", return_value=True):
            topics = m.generate_topics(
                topic_mining_context="context",
                df=sample_enriched_df,
                enriched_column="response_enriched",
            )

        # Should still work with fallback batching
        assert len(topics) == 1
        assert topics[0]["name"] == "Topic A"
