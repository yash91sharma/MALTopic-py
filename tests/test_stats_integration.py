"""
Integration tests for MALTopic statistics functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from maltopic.core import MALTopic
from maltopic.stats import MALTopicStats


class TestMALTopicStatsIntegration:
    """Test MALTopic integration with statistics tracking."""

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI response with usage statistics."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"

        # Mock usage statistics
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        mock_response.model = "gpt-4"
        mock_response.system_fingerprint = "test-fingerprint"

        return mock_response

    @pytest.fixture
    def maltopic_instance(self):
        """Create a MALTopic instance for testing."""
        return MALTopic(
            api_key="test-key", default_model_name="gpt-4", llm_type="openai"
        )

    def test_stats_initialization(self, maltopic_instance):
        """Test that stats are properly initialized."""
        assert isinstance(maltopic_instance.stats, MALTopicStats)
        assert maltopic_instance.stats.total_calls_made == 0
        assert maltopic_instance.stats.total_tokens_used == 0

    def test_stats_public_methods(self, maltopic_instance):
        """Test that public stats methods are available."""
        # Test get_stats method
        stats = maltopic_instance.get_stats()
        assert isinstance(stats, dict)
        assert "overview" in stats
        assert "averages" in stats

        # Test reset_stats method
        maltopic_instance.reset_stats()
        assert maltopic_instance.stats.total_calls_made == 0

    @patch("maltopic.llms.openai.OpenAI")
    def test_successful_api_call_tracking(
        self, mock_openai_class, maltopic_instance, mock_openai_response
    ):
        """Test that successful API calls are tracked properly."""
        # Setup mock
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client

        # Recreate instance to use mocked client
        maltopic = MALTopic("test-key", "gpt-4", "openai")

        # Make a call that should be tracked
        result = maltopic.llm_client.generate(
            instructions="Test instruction", input="Test input"
        )

        # Verify the call was tracked
        assert maltopic.stats.total_calls_made == 1
        assert maltopic.stats.successful_calls == 1
        assert maltopic.stats.failed_calls == 0
        assert maltopic.stats.total_tokens_used == 150
        assert maltopic.stats.total_input_tokens == 100
        assert maltopic.stats.total_output_tokens == 50
        assert maltopic.stats.success_rate == 100.0

    @patch("maltopic.llms.openai.OpenAI")
    def test_failed_api_call_tracking(self, mock_openai_class, maltopic_instance):
        """Test that failed API calls are tracked properly."""
        # Setup mock to raise an exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        # Recreate instance to use mocked client
        maltopic = MALTopic("test-key", "gpt-4", "openai")

        # Make a call that should fail
        with pytest.raises(Exception):
            maltopic.llm_client.generate(
                instructions="Test instruction", input="Test input"
            )

        # Verify the failed call was tracked
        assert maltopic.stats.total_calls_made == 1
        assert maltopic.stats.successful_calls == 0
        assert maltopic.stats.failed_calls == 1
        assert maltopic.stats.total_tokens_used == 0
        assert maltopic.stats.success_rate == 0.0

    @patch("maltopic.llms.openai.OpenAI")
    def test_enrich_free_text_stats_tracking(
        self, mock_openai_class, mock_openai_response
    ):
        """Test stats tracking during enrich_free_text_with_structured_data."""
        # Setup mock
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client

        maltopic = MALTopic("test-key", "gpt-4", "openai")

        # Create test data
        df = pd.DataFrame(
            {"text": ["Hello world", "Goodbye world"], "category": ["A", "B"]}
        )

        # Call enrich method
        result_df = maltopic.enrich_free_text_with_structured_data(
            survey_context="Test survey",
            free_text_column="text",
            structured_data_columns=["category"],
            df=df,
        )

        # Should have made 2 API calls (one for each row)
        assert maltopic.stats.total_calls_made == 2
        assert maltopic.stats.successful_calls == 2
        assert maltopic.stats.total_tokens_used == 300  # 150 * 2

    @patch("maltopic.llms.openai.OpenAI")
    def test_generate_topics_stats_tracking(
        self, mock_openai_class, mock_openai_response
    ):
        """Test stats tracking during generate_topics."""
        # Mock the response to return valid JSON
        mock_openai_response.choices[
            0
        ].message.content = """[
            {
                "name": "Test Topic",
                "description": "A test topic",
                "relevance": "High",
                "representative_words": ["test", "topic"]
            }
        ]"""

        # Setup mock
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client

        maltopic = MALTopic("test-key", "gpt-4", "openai")

        # Create test data
        df = pd.DataFrame({"text_enriched": ["Enriched text 1", "Enriched text 2"]})

        # Call generate_topics method
        topics = maltopic.generate_topics(
            topic_mining_context="Test context", df=df, enriched_column="text_enriched"
        )

        # Should have made 1 API call
        assert maltopic.stats.total_calls_made == 1
        assert maltopic.stats.successful_calls == 1
        assert maltopic.stats.total_tokens_used == 150

    @patch("maltopic.llms.openai.OpenAI")
    def test_deduplicate_topics_stats_tracking(
        self, mock_openai_class, mock_openai_response
    ):
        """Test stats tracking during deduplicate_topics."""
        # Mock the response to return valid JSON
        mock_openai_response.choices[
            0
        ].message.content = """[
            {
                "name": "Merged Topic",
                "description": "A merged topic",
                "relevance": "High",
                "representative_words": ["merged", "topic"]
            }
        ]"""

        # Setup mock
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client

        maltopic = MALTopic("test-key", "gpt-4", "openai")

        # Test topics
        topics = [
            {
                "name": "Topic 1",
                "description": "First topic",
                "relevance": "High",
                "representative_words": ["first", "topic"],
            },
            {
                "name": "Topic 2",
                "description": "Second topic",
                "relevance": "Medium",
                "representative_words": ["second", "topic"],
            },
        ]

        # Call deduplicate_topics method
        result = maltopic.deduplicate_topics(
            topics=topics, survey_context="Test context"
        )

        # Should have made 1 API call
        assert maltopic.stats.total_calls_made == 1
        assert maltopic.stats.successful_calls == 1
        assert maltopic.stats.total_tokens_used == 150

    def test_stats_persistence_across_calls(self, maltopic_instance):
        """Test that stats persist across multiple method calls."""
        # Mock multiple successful calls
        with patch("maltopic.llms.openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 25
            mock_response.usage.total_tokens = 75
            mock_response.model = "gpt-4"

            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client

            # Recreate instance
            maltopic = MALTopic("test-key", "gpt-4", "openai")

            # Make multiple calls
            for i in range(3):
                maltopic.llm_client.generate(
                    instructions=f"Instruction {i}", input=f"Input {i}"
                )

            # Verify cumulative stats
            assert maltopic.stats.total_calls_made == 3
            assert maltopic.stats.successful_calls == 3
            assert maltopic.stats.total_tokens_used == 225  # 75 * 3
            assert maltopic.stats.average_tokens_per_call == 75.0

    def test_print_stats_output(self, maltopic_instance, capsys):
        """Test that print_stats produces formatted output."""
        # Add some mock data
        maltopic_instance.stats.record_successful_call("gpt-4", 100, 50, 150, 1.0)

        # Call print_stats
        maltopic_instance.print_stats()

        # Check output
        captured = capsys.readouterr()
        assert "MALTopic Library Usage Statistics" in captured.out
        assert "Total Tokens Used: 150" in captured.out
        assert "Success Rate: 100.0%" in captured.out
