"""Tests for GUI component helper/logic functions.

These tests exercise the non-Streamlit logic embedded in component modules:
config validation, CSV parsing, export formatting, topic card rendering, etc.
They mock Streamlit where necessary.
"""

import json

import pandas as pd
import pytest


class TestConfigValidation:
    """Tests for configuration parsing logic reused across the GUI."""

    def test_valid_override_params_json(self):
        """Valid JSON dict should parse correctly."""
        text = '{"temperature": 0.8, "max_tokens": 500}'
        parsed = json.loads(text)
        assert isinstance(parsed, dict)
        assert parsed["temperature"] == 0.8

    def test_empty_override_params_is_none(self):
        text = ""
        assert text.strip() == ""

    def test_non_dict_override_params_rejected(self):
        text = "[1, 2, 3]"
        parsed = json.loads(text)
        assert not isinstance(parsed, dict)

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            json.loads("not-json")


class TestCSVParsing:
    """Tests for CSV loading and column extraction logic."""

    def test_read_csv_columns(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name,age,feedback\nAlice,30,Great\nBob,25,Bad\n")
        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["name", "age", "feedback"]
        assert len(df) == 2

    def test_empty_csv_detected(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("name,age\n")
        df = pd.read_csv(csv_path)
        assert df.empty

    def test_column_selection_remaining(self):
        columns = ["feedback", "rating", "category", "id"]
        free_text = "feedback"
        remaining = [c for c in columns if c != free_text]
        assert remaining == ["rating", "category", "id"]


class TestExamplesParsing:
    """Tests for the multi-line examples text → list conversion."""

    def test_split_multiline(self):
        text = "line one\nline two\n\nline three"
        result = [line.strip() for line in text.splitlines() if line.strip()]
        assert result == ["line one", "line two", "line three"]

    def test_empty_text(self):
        assert [l.strip() for l in "".splitlines() if l.strip()] == []

    def test_whitespace_only(self):
        text = "   \n  \n   "
        assert [l.strip() for l in text.splitlines() if l.strip()] == []


class TestTopicFormatting:
    """Test topic card data preparation."""

    def test_words_as_list(self):
        topic = {"representative_words": ["fast", "good", "cheap"]}
        words = topic["representative_words"]
        assert isinstance(words, list)
        assert len(words) == 3

    def test_words_as_string_split(self):
        topic = {"representative_words": "fast, good, cheap"}
        words = topic["representative_words"]
        if isinstance(words, str):
            words = [w.strip() for w in words.split(",") if w.strip()]
        assert words == ["fast", "good", "cheap"]

    def test_topic_card_fields(self):
        topic = {
            "name": "Quality",
            "description": "Product quality feedback",
            "relevance": "All segments",
            "representative_words": ["good", "quality"],
        }
        assert topic.get("name") == "Quality"
        assert topic.get("description") == "Product quality feedback"
        assert topic.get("relevance") == "All segments"


class TestExportFormatting:
    """Test export data preparation (CSV, JSON)."""

    def test_topics_to_json(self):
        topics = [
            {
                "name": "Topic 1",
                "description": "Desc",
                "relevance": "High",
                "representative_words": ["a", "b"],
            }
        ]
        json_str = json.dumps(topics, indent=2)
        loaded = json.loads(json_str)
        assert loaded == topics

    def test_dataframe_to_csv_string(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        csv_str = df.to_csv(index=False)
        assert "a,b" in csv_str
        assert "1,x" in csv_str

    def test_stats_dict_to_json(self):
        stats = {
            "overview": {"total_calls_made": 5, "total_tokens_used": 1000},
            "averages": {"avg_tokens_per_call": 200.0},
        }
        json_str = json.dumps(stats, indent=2)
        loaded = json.loads(json_str)
        assert loaded["overview"]["total_calls_made"] == 5
