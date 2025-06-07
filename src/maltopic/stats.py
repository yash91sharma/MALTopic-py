"""
Statistics tracking module for MALTopic library.

This module provides comprehensive tracking of LLM usage statistics including
token counts, API calls, success/failure rates, and other relevant metrics.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class LLMCallStats:
    """Individual LLM call statistics."""

    timestamp: float
    model_name: str
    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    response_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MALTopicStats:
    """
    Comprehensive statistics tracking for MALTopic library usage.

    Tracks:
    - Total LLM tokens used (input, output, total)
    - Total LLM calls made
    - Average tokens per call
    - Number of successful/failed LLM calls
    - Response times and other metadata
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics to initial state."""
        self._call_history: list[LLMCallStats] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_tokens = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._total_response_time = 0.0
        self._start_time = time.time()

    def record_llm_call(
        self,
        model_name: str,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        response_time: float = 0.0,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record an LLM API call with its statistics.

        Args:
            model_name: Name of the model used
            success: Whether the call was successful
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            total_tokens: Total tokens used (input + output)
            response_time: Time taken for the API call in seconds
            error_message: Error message if the call failed
            metadata: Additional metadata from the API response
        """
        call_stats = LLMCallStats(
            timestamp=time.time(),
            model_name=model_name,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            response_time=response_time,
            error_message=error_message,
            metadata=metadata or {},
        )

        self._call_history.append(call_stats)

        # Update aggregate statistics
        if success:
            self._successful_calls += 1
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_tokens += total_tokens
            self._total_response_time += response_time
        else:
            self._failed_calls += 1

    def record_successful_call(
        self,
        model_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        response_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a successful LLM call."""
        self.record_llm_call(
            model_name=model_name,
            success=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            response_time=response_time,
            metadata=metadata,
        )

    def record_failed_call(
        self,
        model_name: str,
        error_message: str,
        response_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a failed LLM call."""
        self.record_llm_call(
            model_name=model_name,
            success=False,
            response_time=response_time,
            error_message=error_message,
            metadata=metadata,
        )

    @property
    def total_tokens_used(self) -> int:
        """Total number of tokens used across all successful calls."""
        return self._total_tokens

    @property
    def total_input_tokens(self) -> int:
        """Total number of input tokens used."""
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Total number of output tokens generated."""
        return self._total_output_tokens

    @property
    def total_calls_made(self) -> int:
        """Total number of LLM calls made (successful + failed)."""
        return self._successful_calls + self._failed_calls

    @property
    def successful_calls(self) -> int:
        """Number of successful LLM calls."""
        return self._successful_calls

    @property
    def failed_calls(self) -> int:
        """Number of failed LLM calls."""
        return self._failed_calls

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage (0-100)."""
        if self.total_calls_made == 0:
            return 0.0
        return (self._successful_calls / self.total_calls_made) * 100

    @property
    def average_tokens_per_call(self) -> float:
        """Average number of tokens per successful call."""
        if self._successful_calls == 0:
            return 0.0
        return self._total_tokens / self._successful_calls

    @property
    def average_input_tokens_per_call(self) -> float:
        """Average number of input tokens per successful call."""
        if self._successful_calls == 0:
            return 0.0
        return self._total_input_tokens / self._successful_calls

    @property
    def average_output_tokens_per_call(self) -> float:
        """Average number of output tokens per successful call."""
        if self._successful_calls == 0:
            return 0.0
        return self._total_output_tokens / self._successful_calls

    @property
    def average_response_time(self) -> float:
        """Average response time per successful call in seconds."""
        if self._successful_calls == 0:
            return 0.0
        return self._total_response_time / self._successful_calls

    @property
    def uptime(self) -> float:
        """Total uptime since stats initialization in seconds."""
        return time.time() - self._start_time

    def get_model_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics broken down by model."""
        model_stats = {}

        for call in self._call_history:
            model = call.model_name
            if model not in model_stats:
                model_stats[model] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_response_time": 0.0,
                }

            stats = model_stats[model]
            stats["total_calls"] += 1

            if call.success:
                stats["successful_calls"] += 1
                stats["total_tokens"] += call.total_tokens
                stats["input_tokens"] += call.input_tokens
                stats["output_tokens"] += call.output_tokens
                stats["total_response_time"] += call.response_time
            else:
                stats["failed_calls"] += 1

        # Calculate averages
        for model, stats in model_stats.items():
            if stats["successful_calls"] > 0:
                stats["avg_tokens_per_call"] = (
                    stats["total_tokens"] / stats["successful_calls"]
                )
                stats["avg_response_time"] = (
                    stats["total_response_time"] / stats["successful_calls"]
                )
                stats["success_rate"] = (
                    stats["successful_calls"] / stats["total_calls"]
                ) * 100
            else:
                stats["avg_tokens_per_call"] = 0.0
                stats["avg_response_time"] = 0.0
                stats["success_rate"] = 0.0

        return model_stats

    def get_recent_calls(self, limit: int = 10) -> list[LLMCallStats]:
        """Get the most recent LLM calls."""
        return self._call_history[-limit:] if self._call_history else []

    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all statistics."""
        return {
            "overview": {
                "total_tokens_used": self.total_tokens_used,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_calls_made": self.total_calls_made,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "success_rate_percent": round(self.success_rate, 2),
                "uptime_seconds": round(self.uptime, 2),
            },
            "averages": {
                "avg_tokens_per_call": round(self.average_tokens_per_call, 2),
                "avg_input_tokens_per_call": round(
                    self.average_input_tokens_per_call, 2
                ),
                "avg_output_tokens_per_call": round(
                    self.average_output_tokens_per_call, 2
                ),
                "avg_response_time_seconds": round(self.average_response_time, 2),
            },
            "model_breakdown": self.get_model_breakdown(),
            "recent_calls": [
                {
                    "timestamp": call.timestamp,
                    "model": call.model_name,
                    "success": call.success,
                    "tokens": call.total_tokens,
                    "response_time": call.response_time,
                    "error": call.error_message,
                }
                for call in self.get_recent_calls(5)
            ],
        }

    def print_summary(self):
        """Print a formatted summary of statistics."""
        summary = self.get_summary()
        overview = summary["overview"]
        averages = summary["averages"]

        print("=" * 60)
        print("MALTopic Library Usage Statistics")
        print("=" * 60)

        print(f"\n📊 Overview:")
        print(f"  Total Tokens Used: {overview['total_tokens_used']:,}")
        print(f"  - Input Tokens: {overview['total_input_tokens']:,}")
        print(f"  - Output Tokens: {overview['total_output_tokens']:,}")
        print(f"  Total API Calls: {overview['total_calls_made']}")
        print(f"  - Successful: {overview['successful_calls']}")
        print(f"  - Failed: {overview['failed_calls']}")
        print(f"  Success Rate: {overview['success_rate_percent']}%")
        print(f"  Uptime: {overview['uptime_seconds']:.1f} seconds")

        print(f"\n📈 Averages:")
        print(f"  Avg Tokens per Call: {averages['avg_tokens_per_call']:.1f}")
        print(f"  - Avg Input Tokens: {averages['avg_input_tokens_per_call']:.1f}")
        print(f"  - Avg Output Tokens: {averages['avg_output_tokens_per_call']:.1f}")
        print(f"  Avg Response Time: {averages['avg_response_time_seconds']:.2f}s")

        if summary["model_breakdown"]:
            print(f"\n🤖 Model Breakdown:")
            for model, stats in summary["model_breakdown"].items():
                print(f"  {model}:")
                print(
                    f"    Calls: {stats['total_calls']} (Success: {stats['successful_calls']}, Failed: {stats['failed_calls']})"
                )
                print(
                    f"    Tokens: {stats['total_tokens']:,} (Avg: {stats['avg_tokens_per_call']:.1f})"
                )
                print(f"    Success Rate: {stats['success_rate']:.1f}%")

        print("=" * 60)
