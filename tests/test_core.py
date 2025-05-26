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


class TestMALTopic:
    @patch("src.maltopic.core.MALTopic._select_agent")
    def test_init_sets_attributes(self, mock_select):
        mock_select.return_value = "client"
        m = MALTopic("key", "model", "openai")
        assert m.api_key == "key"
        assert m.default_model_name == "model"
        assert m.llm_type == "openai"
        assert m.llm_client == "client"

    def test_select_agent_invalid_type(self):
        m = MALTopic.__new__(MALTopic)
        m.llm_type = "foo"
        with pytest.raises(ValueError):
            m._select_agent("k", "m")

    @patch("src.maltopic.core.utils.validate_dataframe")
    @patch(
        "src.maltopic.core.prompts.ENRICH_INST",
        "enrich inst {survey_context} {free_text_column} {free_text_definition} {structured_data_columns} {examples}",
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
            free_text_definition="def",
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
        "enrich inst {survey_context} {free_text_column} {free_text_definition} {structured_data_columns} {examples}",
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
            free_text_definition="def",
            structured_data_columns=["age"],
            df=df,
        )
        # Both rows trigger exception or error message
        assert (
            out["feedback_enriched"].iloc[0] == "Error generating enriched text: fail"
        )
        assert out["feedback_enriched"].iloc[1] == "ok"
        mock_validate.assert_called_once_with(df, ["feedback", "age"])

    @patch("src.maltopic.core.utils.validate_dataframe")
    @patch("src.maltopic.core.prompts.TOPIC_INST", "topic inst {survey_context}")
    def test_generate_topics_success(self, mock_validate, enriched_df):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        # LLM returns a JSON string
        m.llm_client.generate.return_value = '[{"id": "1", "name": "A", "description": "desc", "representative_words": ["foo", "bar"]}]'
        topics = m.generate_topics("ctx", enriched_df, "feedback_enriched")
        assert isinstance(topics, list)
        assert topics[0]["id"] == "1"
        assert topics[0]["name"] == "A"
        assert topics[0]["description"] == "desc"
        assert topics[0]["representative_words"] == "foo, bar"
        mock_validate.assert_called_once_with(enriched_df, ["feedback_enriched"])

    @patch("src.maltopic.core.utils.validate_dataframe")
    @patch("src.maltopic.core.prompts.TOPIC_INST", "topic inst {survey_context}")
    def test_generate_topics_json_decode_error(self, mock_validate, enriched_df):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.llm_client.generate.return_value = "not json"
        with pytest.raises(ValueError) as excinfo:
            m.generate_topics("ctx", enriched_df, "feedback_enriched")
        assert "Failed to parse LLM response as JSON" in str(excinfo.value)
        mock_validate.assert_called_once_with(enriched_df, ["feedback_enriched"])

    @patch("src.maltopic.core.utils.validate_dataframe")
    @patch("src.maltopic.core.prompts.TOPIC_INST", "topic inst {survey_context}")
    def test_generate_topics_other_exception(self, mock_validate, enriched_df):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.llm_client.generate.return_value = (
            '[{"id": 1, "name": "A", "description": "desc"}]'
        )
        # id is int, should be converted to str
        topics = m.generate_topics("ctx", enriched_df, "feedback_enriched")
        assert topics[0]["id"] == "1"
        mock_validate.assert_called_once_with(enriched_df, ["feedback_enriched"])

    @patch("src.maltopic.core.utils.validate_dataframe")
    @patch("src.maltopic.core.prompts.TOPIC_INST", "topic inst {survey_context}")
    def test_generate_topics_llm_error(self, mock_validate, enriched_df):
        m = MALTopic.__new__(MALTopic)
        m.llm_client = MagicMock()
        m.llm_client.generate.side_effect = Exception("llm fail")
        with pytest.raises(RuntimeError) as excinfo:
            m.generate_topics("ctx", enriched_df, "feedback_enriched")
        assert "Error generating topics" in str(excinfo.value)
        mock_validate.assert_called_once_with(enriched_df, ["feedback_enriched"])
