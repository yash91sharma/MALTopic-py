import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.maltopic.utils import (
    consolidate_topics,
    generate_topics_from_text,
    generate_topics_with_batching,
    parse_topics_response,
    validate_dataframe,
    validate_topic_structure,
)


@pytest.fixture
def sample_topics():
    return [
        {
            "name": "Customer Service Quality",
            "description": "Issues related to customer service experience",
            "relevance": "All customer segments",
            "representative_words": ["service", "support", "help", "staff"],
        },
        {
            "name": "Service Quality",
            "description": "Quality of service provided to customers",
            "relevance": "General customer feedback",
            "representative_words": ["quality", "service", "good", "bad"],
        },
    ]


class TestValidateDataFrame:
    # Test that a valid DataFrame with all required columns passes validation
    def test_valid_dataframe_with_required_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        # Should not raise for subset of columns
        assert validate_dataframe(df, ["a", "b"]) is None
        # Should not raise for all columns present
        assert validate_dataframe(df, ["a", "b", "c"]) is None

    # Test that missing required columns raises ValueError and lists missing columns
    @pytest.mark.parametrize(
        "df, required, missing",
        [
            (pd.DataFrame({"x": [1, 2], "y": [3, 4]}), ["x", "y", "z"], ["z"]),
            (pd.DataFrame({"x": [1, 2], "y": [3, 4]}), ["y", "foo"], ["foo"]),
            (pd.DataFrame({"foo": [1, 2]}), ["bar", "baz"], ["bar", "baz"]),
        ],
    )
    def test_missing_required_columns(self, df, required, missing):
        # Should raise ValueError if any required columns are missing
        with pytest.raises(ValueError) as excinfo:
            validate_dataframe(df, required)
        msg = str(excinfo.value)
        assert "missing required columns" in msg
        # Each missing column should be mentioned in the error message
        for col in missing:
            assert col in msg

    # Test that an empty DataFrame raises ValueError
    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(columns=["a", "b"])
        # Should raise ValueError for empty DataFrame
        with pytest.raises(ValueError) as excinfo:
            validate_dataframe(df, ["a"])
        assert "DataFrame is empty" in str(excinfo.value)

    # Test that no required columns means always valid if DataFrame is not empty
    def test_no_required_columns(self):
        df = pd.DataFrame({"a": [1]})
        # Should not raise if required_columns is empty
        assert validate_dataframe(df, []) is None


class TestValidateTopicStructure:
    def test_validate_topic_structure_valid(self, sample_topics):
        # Should not raise exception for valid topics
        validate_topic_structure(sample_topics)

    def test_validate_topic_structure_invalid_type(self):
        invalid_topics = ["not a dict"]  # type: ignore

        with pytest.raises(ValueError) as excinfo:
            validate_topic_structure(invalid_topics)  # type: ignore
        assert "is not a dictionary" in str(excinfo.value)

    def test_validate_topic_structure_missing_fields(self):
        invalid_topics = [{"name": "test"}]  # Missing required fields

        with pytest.raises(ValueError) as excinfo:
            validate_topic_structure(invalid_topics)
        assert "missing required fields" in str(excinfo.value)

    def test_validate_topic_structure_invalid_field_types(self):
        invalid_topics = [
            {
                "name": 123,  # Should be string
                "description": "test",
                "relevance": "test",
                "representative_words": ["word1", "word2"],
            }
        ]

        with pytest.raises(ValueError) as excinfo:
            validate_topic_structure(invalid_topics)
        assert "must be a string" in str(excinfo.value)

    def test_validate_topic_structure_representative_words_as_string(self):
        # representative_words can be a string instead of a list
        valid_topics = [
            {
                "name": "Test Topic",
                "description": "A test topic",
                "relevance": "Test users",
                "representative_words": "word1, word2, word3",  # String format
            }
        ]

        # Should not raise exception
        validate_topic_structure(valid_topics)

    def test_validate_topic_structure_empty_list(self):
        # Empty list should be valid
        validate_topic_structure([])

    def test_validate_topic_structure_multiple_invalid_types(self):
        invalid_topics = [
            {
                "name": "Valid Topic",
                "description": "Valid description",
                "relevance": "Valid relevance",
                "representative_words": ["word1", "word2"],
            },
            {
                "name": 456,  # Invalid type
                "description": ["not", "a", "string"],  # Invalid type
                "relevance": "Valid relevance",
                "representative_words": 789,  # Invalid type
            },
        ]

        with pytest.raises(ValueError) as excinfo:
            validate_topic_structure(invalid_topics)
        # Should catch the first invalid field
        assert "Topic 1" in str(excinfo.value)
        assert "must be a string" in str(excinfo.value)


class TestGenerateTopicsFromText:
    def test_generate_topics_from_text_success(self):
        """Test successful topic generation from text."""
        mock_client = MagicMock()
        mock_client.generate.return_value = '[{"name": "Test Topic", "description": "A test", "relevance": "High", "representative_words": ["test"]}]'

        result = generate_topics_from_text(mock_client, "instructions", "input text")

        assert len(result) == 1
        assert result[0]["name"] == "Test Topic"
        mock_client.generate.assert_called_once_with(
            instructions="instructions", input="input text"
        )


class TestParseTopicsResponse:
    def test_parse_topics_response_success(self):
        """Test successful parsing of topics response."""
        json_response = '[{"name": "Topic 1", "description": "Description 1", "relevance": "High", "representative_words": ["word1", "word2"]}]'

        topics = parse_topics_response(json_response)

        assert len(topics) == 1
        assert topics[0]["name"] == "Topic 1"
        assert topics[0]["description"] == "Description 1"

    def test_parse_topics_response_converts_non_strings(self):
        """Test that non-string fields (except representative_words) are converted to strings."""
        json_response = '[{"name": 123, "description": "Description 1", "relevance": "High", "representative_words": ["word1"]}]'

        topics = parse_topics_response(json_response)

        assert topics[0]["name"] == "123"  # Should be converted to string

    def test_parse_topics_response_json_error(self):
        """Test handling of invalid JSON in topics response."""
        with pytest.raises(ValueError) as excinfo:
            parse_topics_response("invalid json")

        assert "Failed to parse LLM response as JSON" in str(excinfo.value)

    def test_parse_topics_response_processing_error(self):
        """Test handling of processing errors."""
        # This would cause an error during processing
        with pytest.raises(ValueError) as excinfo:
            parse_topics_response('{"invalid": "structure"}')  # Not a list

        assert "Error processing topics" in str(excinfo.value)


class TestConsolidateTopics:
    def test_consolidate_topics_removes_duplicates(self):
        """Test that duplicate topic names are removed."""
        topics = [
            {"name": "Topic A", "description": "About A"},
            {"name": "Topic B", "description": "About B"},
            {"name": "Topic A", "description": "About A again"},  # Duplicate
            {"name": "Topic C", "description": "About C"},
        ]

        consolidated = consolidate_topics(topics)

        assert len(consolidated) == 3
        topic_names = [t["name"] for t in consolidated]
        assert "Topic A" in topic_names
        assert "Topic B" in topic_names
        assert "Topic C" in topic_names

    def test_consolidate_topics_empty_input(self):
        """Test consolidation with empty input."""
        result = consolidate_topics([])
        assert result == []

    def test_consolidate_topics_handles_missing_name(self):
        """Test that topics without names are filtered out."""
        topics = [
            {"name": "Topic A", "description": "About A"},
            {"description": "No name"},  # Missing name
            {"name": "", "description": "Empty name"},  # Empty name
            {"name": "Topic B", "description": "About B"},
        ]

        consolidated = consolidate_topics(topics)

        assert len(consolidated) == 2
        topic_names = [t["name"] for t in consolidated]
        assert "Topic A" in topic_names
        assert "Topic B" in topic_names


class TestGenerateTopicsWithBatching:
    def test_generate_topics_with_batching_success(self):
        """Test successful topic generation with batching."""
        mock_client = MagicMock()

        # Mock responses for each batch
        batch1_response = '[{"name": "Topic A", "description": "About A", "relevance": "High", "representative_words": ["word1"]}]'
        batch2_response = '[{"name": "Topic B", "description": "About B", "relevance": "Medium", "representative_words": ["word2"]}]'

        mock_client.generate.side_effect = [batch1_response, batch2_response]

        labeled_columns = [
            "1: Response 1",
            "2: Response 2",
            "3: Response 3",
            "4: Response 4",
        ]

        with patch("src.maltopic.utils.split_text_into_batches") as mock_split:
            mock_split.return_value = [
                ["1: Response 1", "2: Response 2"],
                ["3: Response 3", "4: Response 4"],
            ]

            result = generate_topics_with_batching(
                mock_client, "instructions", labeled_columns, "context", "gpt-4"
            )

        assert len(result) == 2
        assert result[0]["name"] == "Topic A"
        assert result[1]["name"] == "Topic B"
        assert mock_client.generate.call_count == 2

    def test_generate_topics_with_batching_fallback(self):
        """Test fallback batching when tiktoken is not available."""
        mock_client = MagicMock()

        batch_response = '[{"name": "Topic A", "description": "About A", "relevance": "High", "representative_words": ["word1"]}]'
        mock_client.generate.return_value = batch_response

        labeled_columns = [
            "1: Response 1",
            "2: Response 2",
            "3: Response 3",
            "4: Response 4",
        ]

        with patch("src.maltopic.utils.split_text_into_batches") as mock_split:
            mock_split.side_effect = ImportError("tiktoken not available")

            result = generate_topics_with_batching(
                mock_client, "instructions", labeled_columns, "context", "gpt-4"
            )

        # Should still work with fallback batching
        assert len(result) == 1
        assert result[0]["name"] == "Topic A"
        # With 4 items and batch_size=1, should generate 4 calls
        assert mock_client.generate.call_count == 4

    def test_generate_topics_with_batching_handles_errors(self):
        """Test that batch processing continues even if some batches fail."""
        mock_client = MagicMock()

        # First batch fails, second succeeds
        batch_response = '[{"name": "Topic B", "description": "About B", "relevance": "Medium", "representative_words": ["word2"]}]'
        mock_client.generate.side_effect = [Exception("Batch 1 failed"), batch_response]

        labeled_columns = [
            "1: Response 1",
            "2: Response 2",
            "3: Response 3",
            "4: Response 4",
        ]

        with patch("src.maltopic.utils.split_text_into_batches") as mock_split:
            mock_split.return_value = [
                ["1: Response 1", "2: Response 2"],
                ["3: Response 3", "4: Response 4"],
            ]

            result = generate_topics_with_batching(
                mock_client, "instructions", labeled_columns, "context", "gpt-4"
            )

        # Should only have topics from successful batch
        assert len(result) == 1
        assert result[0]["name"] == "Topic B"
