import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.maltopic.core import MALTopic


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "age": [25, 34],
            "gender": ["Male", "Female"],
            "location": ["New York", "London"],
            "feedback": ["Great service", "App is slow"],
        }
    )


@pytest.fixture
def enriched_df(sample_df):
    df = sample_df.copy()
    df["feedback_enriched"] = [
        "Great service (NY, Male)",
        "App is slow (London, Female)",
    ]
    return df


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
        {
            "name": "Product Features",
            "description": "Feedback about product functionality and features",
            "relevance": "Tech-savvy users",
            "representative_words": ["features", "functionality", "product", "app"],
        },
        {
            "name": "App Performance",
            "description": "Issues with application speed and performance",
            "relevance": "Mobile users",
            "representative_words": ["slow", "fast", "performance", "speed"],
        },
    ]


@pytest.fixture
def unique_topics():
    return [
        {
            "name": "Pricing",
            "description": "Cost and pricing related feedback",
            "relevance": "Budget-conscious customers",
            "representative_words": ["price", "cost", "expensive", "cheap"],
        },
        {
            "name": "User Interface",
            "description": "Feedback about app design and usability",
            "relevance": "All users",
            "representative_words": ["design", "interface", "ui", "layout"],
        },
    ]


class TestMALTopic:
    @patch("src.maltopic.core.MALTopic._select_agent")
    def test_init_sets_attributes(self, mock_select):
        mock_select.return_value = "client"
        m = MALTopic("key", "model", "openai")
        assert m.api_key == "key"
        assert m.default_model_name == "model"
        assert m.llm_type == "openai"
        assert m.llm_client == "client"

    @patch("src.maltopic.core.MALTopic._select_agent")
    def test_init_with_override_model_params(self, mock_select):
        mock_select.return_value = "client"
        override_params = {"temperature": 0.8, "max_tokens": 100}
        m = MALTopic("key", "model", "openai", override_model_params=override_params)
        assert m.api_key == "key"
        assert m.default_model_name == "model"
        assert m.llm_type == "openai"
        assert m.override_model_params == override_params
        assert m.llm_client == "client"

    @patch("src.maltopic.llms.openai.OpenAI")
    def test_select_agent_passes_override_params(self, mock_openai_class):
        """Test that _select_agent passes override_model_params to OpenAIClient"""
        m = MALTopic.__new__(MALTopic)
        m.llm_type = "openai"
        m.override_model_params = {"temperature": 0.7}
        m.stats = MagicMock()
        
        # Mock the OpenAI class constructor
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance
        
        with patch("src.maltopic.llms.openai.OpenAIClient") as mock_openai_client:
            mock_openai_client.return_value = "mocked_client"
            result = m._select_agent("test_key", "test_model")
            
            # Verify OpenAIClient was called with override_model_params
            mock_openai_client.assert_called_once_with(
                "test_key",
                "test_model",
                stats_tracker=m.stats,
                override_model_params={"temperature": 0.7},
            )
            assert result == "mocked_client"

    def test_select_agent_invalid_type(self):
        m = MALTopic.__new__(MALTopic)
        m.llm_type = "foo"
        with pytest.raises(ValueError):
            m._select_agent("k", "m")

    @patch("src.maltopic.core.utils.validate_dataframe")
    @patch(
        "src.maltopic.core.prompts.ENRICH_INST",
        "enrich inst {survey_context} {free_text_column} {structured_data_columns} {examples}",
    )
    def test_enrich_free_text_with_structured_data_success(
        self, mock_validate, sample_df
    ):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.llm_client.generate.side_effect = ["enriched1", "enriched2"]
        df = sample_df.copy()
        out = m.enrich_free_text_with_structured_data(
            survey_context="ctx",
            free_text_column="feedback",
            structured_data_columns=["age", "gender"],
            df=df,
            examples=["ex1", "ex2"],
        )
        assert "feedback_enriched" in out.columns
        assert out["feedback_enriched"].tolist() == ["enriched1", "enriched2"]
        assert m.llm_client.generate.call_count == 2
        mock_validate.assert_called_once_with(df, ["feedback", "age", "gender"])

    @patch("src.maltopic.core.utils.validate_dataframe")
    @patch(
        "src.maltopic.core.prompts.ENRICH_INST",
        "enrich inst {survey_context} {free_text_column} {structured_data_columns} {examples}",
    )
    def test_enrich_handles_empty_and_exception(self, mock_validate, sample_df):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.llm_client.generate.side_effect = [Exception("fail"), "ok"]
        df = sample_df.copy()
        df.loc[0, "feedback"] = None
        out = m.enrich_free_text_with_structured_data(
            survey_context="ctx",
            free_text_column="feedback",
            structured_data_columns=["age"],
            df=df,
        )
        # Check that first row has error message, second row has "ok"
        assert (
            "Error generating enriched text: fail" in out["feedback_enriched"].iloc[0]
        )
        assert out["feedback_enriched"].iloc[1] == "ok"
        mock_validate.assert_called_once_with(df, ["feedback", "age"])

    @patch("src.maltopic.core.utils.validate_dataframe")
    @patch("src.maltopic.core.prompts.TOPIC_INST", "topic inst {survey_context}")
    def test_generate_topics_success(self, mock_validate, enriched_df):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.default_model_name = "gpt-4"
        # LLM returns a JSON string
        m.llm_client.generate.return_value = '[{"id": "1", "name": "A", "description": "desc", "representative_words": ["foo", "bar"]}]'
        topics = m.generate_topics(
            topic_mining_context="ctx",
            df=enriched_df,
            enriched_column="feedback_enriched",
        )
        assert len(topics) == 1
        assert topics[0]["id"] == "1"
        assert topics[0]["name"] == "A"
        assert topics[0]["description"] == "desc"
        assert topics[0]["representative_words"] == ["foo", "bar"]
        mock_validate.assert_called_once_with(enriched_df, ["feedback_enriched"])

    @patch("src.maltopic.core.utils.validate_dataframe")
    @patch("src.maltopic.core.prompts.TOPIC_INST", "topic inst {survey_context}")
    @patch("src.maltopic.core.utils.is_token_limit_error")
    def test_generate_topics_json_decode_error(
        self, mock_is_token_error, mock_validate, enriched_df
    ):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.default_model_name = "gpt-4"
        m.llm_client.generate.return_value = "not json"
        mock_is_token_error.return_value = False  # Not a token error

        with pytest.raises(RuntimeError) as excinfo:
            m.generate_topics(
                topic_mining_context="ctx",
                df=enriched_df,
                enriched_column="feedback_enriched",
            )
        assert "Error generating topics" in str(excinfo.value)
        assert "Failed to parse LLM response as JSON" in str(excinfo.value)
        mock_validate.assert_called_once_with(enriched_df, ["feedback_enriched"])

    @patch("src.maltopic.core.utils.validate_dataframe")
    @patch("src.maltopic.core.prompts.TOPIC_INST", "topic inst {survey_context}")
    @patch("src.maltopic.core.utils.is_token_limit_error")
    def test_generate_topics_other_exception(
        self, mock_is_token_error, mock_validate, enriched_df
    ):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.default_model_name = "gpt-4"
        m.llm_client.generate.return_value = (
            '[{"id": 1, "name": "A", "description": "desc"}]'
        )
        mock_is_token_error.return_value = False
        # id is int, should be converted to str
        topics = m.generate_topics(
            topic_mining_context="ctx",
            df=enriched_df,
            enriched_column="feedback_enriched",
        )
        assert topics[0]["id"] == "1"
        mock_validate.assert_called_once_with(enriched_df, ["feedback_enriched"])

    @patch("src.maltopic.core.utils.validate_dataframe")
    @patch("src.maltopic.core.prompts.TOPIC_INST", "topic inst {survey_context}")
    @patch("src.maltopic.core.utils.is_token_limit_error")
    def test_generate_topics_llm_error(
        self, mock_is_token_error, mock_validate, enriched_df
    ):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.default_model_name = "gpt-4"
        m.llm_client.generate.side_effect = Exception("llm fail")
        mock_is_token_error.return_value = False  # Not a token error

        with pytest.raises(RuntimeError) as excinfo:
            m.generate_topics(
                topic_mining_context="ctx",
                df=enriched_df,
                enriched_column="feedback_enriched",
            )
        assert "Error generating topics" in str(excinfo.value)
        mock_validate.assert_called_once_with(enriched_df, ["feedback_enriched"])

    @patch("src.maltopic.core.prompts.DEDUP_TOPICS_INST", "dedup inst {survey_context}")
    def test_deduplicate_topics_success(self, sample_topics):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()

        # Mock LLM response that merges overlapping topics
        deduplicated_response = json.dumps(
            [
                {
                    "name": "Customer Service Quality",
                    "description": "Comprehensive feedback about service quality and customer experience",
                    "relevance": "All customer segments, general feedback",
                    "representative_words": [
                        "service",
                        "support",
                        "help",
                        "staff",
                        "quality",
                        "good",
                        "bad",
                    ],
                },
                {
                    "name": "Product Features",
                    "description": "Feedback about product functionality and features",
                    "relevance": "Tech-savvy users",
                    "representative_words": [
                        "features",
                        "functionality",
                        "product",
                        "app",
                    ],
                },
                {
                    "name": "App Performance",
                    "description": "Issues with application speed and performance",
                    "relevance": "Mobile users",
                    "representative_words": ["slow", "fast", "performance", "speed"],
                },
            ]
        )

        m.llm_client.generate.return_value = deduplicated_response

        result = m.deduplicate_topics(
            topics=sample_topics, survey_context="Customer feedback survey"
        )

        assert len(result) == 3  # Should merge 2 similar topics into 1
        assert result[0]["name"] == "Customer Service Quality"
        assert "service quality and customer experience" in result[0]["description"]
        assert len(result[0]["representative_words"]) == 7  # Merged words

        # Verify LLM was called with correct parameters
        m.llm_client.generate.assert_called_once()
        call_args = m.llm_client.generate.call_args
        assert "dedup inst Customer feedback survey" in call_args[1]["instructions"]
        assert "Topics to deduplicate:" in call_args[1]["input"]

    def test_deduplicate_topics_empty_list(self):
        m = MALTopic.__new__(MALTopic)
        result = m.deduplicate_topics(topics=[], survey_context="test")
        assert result == []

    def test_deduplicate_topics_single_topic(self, sample_topics):
        m = MALTopic.__new__(MALTopic)
        single_topic = [sample_topics[0]]
        result = m.deduplicate_topics(topics=single_topic, survey_context="test")
        assert result == single_topic
        assert result is not single_topic  # Should return a copy

    def test_deduplicate_topics_invalid_structure(self):
        m = MALTopic.__new__(MALTopic)
        invalid_topics = [
            {"name": "Test1", "description": "Test desc1"},  # Missing required fields
            {"name": "Test2", "description": "Test desc2"},  # Missing required fields
        ]

        with pytest.raises(ValueError) as excinfo:
            m.deduplicate_topics(topics=invalid_topics, survey_context="test")
        assert "missing required fields" in str(excinfo.value)

    @patch("src.maltopic.core.prompts.DEDUP_TOPICS_INST", "dedup inst {survey_context}")
    def test_deduplicate_topics_llm_failure_fallback(self, sample_topics):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.llm_client.generate.side_effect = Exception("LLM API error")

        result = m.deduplicate_topics(topics=sample_topics, survey_context="test")

        # Should return original topics unchanged when LLM fails
        assert len(result) == 4  # Same as input
        assert result == sample_topics  # Exact same content
        assert result is not sample_topics  # But different object (copy)

    @patch("src.maltopic.core.prompts.DEDUP_TOPICS_INST", "dedup inst {survey_context}")
    def test_deduplicate_topics_invalid_llm_response(self, sample_topics):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.llm_client.generate.return_value = "invalid json response"

        result = m.deduplicate_topics(topics=sample_topics, survey_context="test")

        # Should return original topics unchanged when LLM response is invalid
        assert len(result) == 4  # Same as input
        assert result == sample_topics  # Exact same content

    @patch("src.maltopic.core.prompts.DEDUP_TOPICS_INST", "dedup inst {survey_context}")
    def test_deduplicate_topics_maintains_structure(self, sample_topics):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()

        # Mock response with same structure as input
        m.llm_client.generate.return_value = json.dumps(sample_topics)

        result = m.deduplicate_topics(topics=sample_topics, survey_context="test")

        # Verify structure is maintained
        for topic in result:
            assert "name" in topic
            assert "description" in topic
            assert "relevance" in topic
            assert "representative_words" in topic
            assert isinstance(topic["name"], str)
            assert isinstance(topic["description"], str)
            assert isinstance(topic["relevance"], str)
