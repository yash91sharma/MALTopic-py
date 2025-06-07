"""
Tests for the MALTopic statistics tracking functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch

from maltopic.stats import MALTopicStats, LLMCallStats


class TestMALTopicStats:
    """Test the MALTopicStats class functionality."""

    def test_initial_state(self):
        """Test that stats are initialized correctly."""
        stats = MALTopicStats()

        assert stats.total_tokens_used == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_calls_made == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.success_rate == 0.0
        assert stats.average_tokens_per_call == 0.0
        assert stats.average_response_time == 0.0
        assert stats.uptime > 0

    def test_record_successful_call(self):
        """Test recording a successful LLM call."""
        stats = MALTopicStats()

        stats.record_successful_call(
            model_name="gpt-4",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            response_time=1.5,
            metadata={"test": "data"},
        )

        assert stats.total_tokens_used == 150
        assert stats.total_input_tokens == 100
        assert stats.total_output_tokens == 50
        assert stats.total_calls_made == 1
        assert stats.successful_calls == 1
        assert stats.failed_calls == 0
        assert stats.success_rate == 100.0
        assert stats.average_tokens_per_call == 150.0
        assert stats.average_input_tokens_per_call == 100.0
        assert stats.average_output_tokens_per_call == 50.0
        assert stats.average_response_time == 1.5

    def test_record_failed_call(self):
        """Test recording a failed LLM call."""
        stats = MALTopicStats()

        stats.record_failed_call(
            model_name="gpt-4", error_message="API Error", response_time=0.5
        )

        assert stats.total_tokens_used == 0
        assert stats.total_calls_made == 1
        assert stats.successful_calls == 0
        assert stats.failed_calls == 1
        assert stats.success_rate == 0.0
        assert stats.average_tokens_per_call == 0.0

    def test_mixed_successful_and_failed_calls(self):
        """Test recording both successful and failed calls."""
        stats = MALTopicStats()

        # Add successful calls
        stats.record_successful_call("gpt-4", 100, 50, 150, 1.0)
        stats.record_successful_call("gpt-4", 200, 100, 300, 2.0)

        # Add failed call
        stats.record_failed_call("gpt-4", "Error", 0.5)

        assert stats.total_tokens_used == 450  # 150 + 300
        assert stats.total_calls_made == 3
        assert stats.successful_calls == 2
        assert stats.failed_calls == 1
        assert stats.success_rate == pytest.approx(66.67, rel=1e-2)
        assert stats.average_tokens_per_call == 225.0  # 450 / 2
        assert stats.average_response_time == 1.5  # (1.0 + 2.0) / 2

    def test_model_breakdown(self):
        """Test model-specific statistics breakdown."""
        stats = MALTopicStats()

        # Add calls for different models
        stats.record_successful_call("gpt-4", 100, 50, 150, 1.0)
        stats.record_successful_call("gpt-3.5-turbo", 80, 40, 120, 0.8)
        stats.record_failed_call("gpt-4", "Error", 0.5)

        breakdown = stats.get_model_breakdown()

        assert "gpt-4" in breakdown
        assert "gpt-3.5-turbo" in breakdown

        gpt4_stats = breakdown["gpt-4"]
        assert gpt4_stats["total_calls"] == 2
        assert gpt4_stats["successful_calls"] == 1
        assert gpt4_stats["failed_calls"] == 1
        assert gpt4_stats["total_tokens"] == 150
        assert gpt4_stats["success_rate"] == 50.0

        gpt35_stats = breakdown["gpt-3.5-turbo"]
        assert gpt35_stats["total_calls"] == 1
        assert gpt35_stats["successful_calls"] == 1
        assert gpt35_stats["failed_calls"] == 0
        assert gpt35_stats["total_tokens"] == 120
        assert gpt35_stats["success_rate"] == 100.0

    def test_recent_calls(self):
        """Test retrieving recent calls."""
        stats = MALTopicStats()

        # Add multiple calls
        for i in range(15):
            stats.record_successful_call(f"model-{i}", 10, 5, 15, 0.1)

        recent = stats.get_recent_calls(5)
        assert len(recent) == 5

        # Should get the most recent calls
        assert recent[-1].model_name == "model-14"
        assert recent[0].model_name == "model-10"

    def test_summary(self):
        """Test getting comprehensive summary."""
        stats = MALTopicStats()

        stats.record_successful_call("gpt-4", 100, 50, 150, 1.0)
        stats.record_failed_call("gpt-4", "Error", 0.5)

        summary = stats.get_summary()

        assert "overview" in summary
        assert "averages" in summary
        assert "model_breakdown" in summary
        assert "recent_calls" in summary

        overview = summary["overview"]
        assert overview["total_tokens_used"] == 150
        assert overview["total_calls_made"] == 2
        assert overview["successful_calls"] == 1
        assert overview["failed_calls"] == 1
        assert overview["success_rate_percent"] == 50.0

    def test_reset(self):
        """Test resetting statistics."""
        stats = MALTopicStats()

        # Add some data
        stats.record_successful_call("gpt-4", 100, 50, 150, 1.0)
        stats.record_failed_call("gpt-4", "Error", 0.5)

        # Verify data exists
        assert stats.total_calls_made == 2

        # Reset and verify clean state
        stats.reset()
        assert stats.total_calls_made == 0
        assert stats.total_tokens_used == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0

    def test_print_summary(self, capsys):
        """Test printing formatted summary."""
        stats = MALTopicStats()
        stats.record_successful_call("gpt-4", 100, 50, 150, 1.0)

        stats.print_summary()

        captured = capsys.readouterr()
        assert "MALTopic Library Usage Statistics" in captured.out
        assert "Total Tokens Used: 150" in captured.out
        assert "Total API Calls: 1" in captured.out
        assert "Success Rate: 100.0%" in captured.out


class TestLLMCallStats:
    """Test the LLMCallStats dataclass."""

    def test_creation(self):
        """Test creating LLMCallStats instance."""
        timestamp = time.time()
        stats = LLMCallStats(
            timestamp=timestamp,
            model_name="gpt-4",
            success=True,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            response_time=1.5,
            metadata={"test": "data"},
        )

        assert stats.timestamp == timestamp
        assert stats.model_name == "gpt-4"
        assert stats.success is True
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        assert stats.total_tokens == 150
        assert stats.response_time == 1.5
        assert stats.metadata == {"test": "data"}

    def test_defaults(self):
        """Test default values for LLMCallStats."""
        stats = LLMCallStats(timestamp=time.time(), model_name="gpt-4", success=False)

        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.total_tokens == 0
        assert stats.response_time == 0.0
        assert stats.error_message is None
        assert stats.metadata == {}
