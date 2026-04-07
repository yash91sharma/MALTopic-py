"""Integration tests for the Streamlit GUI app using st.testing.v1.AppTest.

These tests exercise the full application flow with mocked MALTopic instances
to avoid real API calls. They verify page rendering, step navigation,
and the end-to-end pipeline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

APP_FILE = str(Path(__file__).resolve().parent.parent / "src" / "maltopic" / "gui" / "app.py")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_maltopic():
    """Create a mock MALTopic instance with working stats."""
    instance = MagicMock()
    instance.stats.total_calls_made = 3
    instance.stats.total_tokens_used = 500

    instance.enrich_free_text_with_structured_data.side_effect = (
        lambda *, survey_context, free_text_column, structured_data_columns, df, examples=None: (
            df.assign(**{f"{free_text_column}_enriched": ["enriched"] * len(df)})
        )
    )

    instance.generate_topics.return_value = [
        {
            "name": "Topic A",
            "description": "Description A",
            "relevance": "Relevance A",
            "representative_words": ["word1", "word2"],
        },
    ]

    instance.deduplicate_topics.return_value = [
        {
            "name": "Topic A",
            "description": "Description A (merged)",
            "relevance": "Relevance A",
            "representative_words": ["word1", "word2"],
        },
    ]

    instance.get_stats.return_value = {
        "overview": {
            "total_calls_made": 3,
            "total_tokens_used": 500,
            "total_input_tokens": 300,
            "total_output_tokens": 200,
            "successful_calls": 3,
            "failed_calls": 0,
            "success_rate_percent": 100.0,
            "uptime_seconds": 10.0,
        },
        "averages": {
            "avg_tokens_per_call": 166.7,
            "avg_input_tokens_per_call": 100.0,
            "avg_output_tokens_per_call": 66.7,
            "avg_response_time_seconds": 0.5,
        },
        "model_breakdown": {},
    }
    return instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAppLoads:
    """App should render without crashing on initial load."""

    def test_initial_render(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_FILE, default_timeout=10)
        at.run()
        assert not at.exception, f"App crashed on load: {at.exception}"

    def test_shows_config_step_first(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_FILE, default_timeout=10)
        at.run()
        # The header for Step 1: Configure API should be present
        headers = [h.value for h in at.header]
        assert any("Configure" in h for h in headers), f"Expected config header, got {headers}"


class TestStepNavigation:
    """Users should be able to move forward when prerequisites are met."""

    def test_cannot_advance_without_config(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_FILE, default_timeout=10)
        at.run()
        # current_step should be 0
        assert at.session_state["current_step"] == 0

    def test_advance_to_upload_after_config(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_FILE, default_timeout=10)
        at.run()

        # Inject a mock instance directly into session state
        at.session_state["maltopic_instance"] = _make_mock_maltopic()
        at.session_state["current_step"] = 1
        at.run()

        headers = [h.value for h in at.header]
        assert any("Upload" in h for h in headers), f"Expected upload header, got {headers}"


class TestFullPipelineMocked:
    """Walk through the full pipeline with mocked MALTopic."""

    def test_enrichment_step_renders(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_FILE, default_timeout=10)
        at.run()

        mock_inst = _make_mock_maltopic()
        df = pd.DataFrame({
            "feedback": ["Great", "Bad"],
            "rating": [5, 1],
        })

        at.session_state["maltopic_instance"] = mock_inst
        at.session_state["uploaded_df"] = df
        at.session_state["free_text_column"] = "feedback"
        at.session_state["structured_columns"] = ["rating"]
        at.session_state["current_step"] = 2
        at.run()

        headers = [h.value for h in at.header]
        assert any("Enrich" in h for h in headers)

    def test_topics_step_renders(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_FILE, default_timeout=10)
        at.run()

        mock_inst = _make_mock_maltopic()
        df = pd.DataFrame({
            "feedback": ["Great", "Bad"],
            "rating": [5, 1],
            "feedback_enriched": ["enriched1", "enriched2"],
        })

        at.session_state["maltopic_instance"] = mock_inst
        at.session_state["uploaded_df"] = df
        at.session_state["free_text_column"] = "feedback"
        at.session_state["structured_columns"] = ["rating"]
        at.session_state["enriched_df"] = df
        at.session_state["current_step"] = 3
        at.run()

        headers = [h.value for h in at.header]
        assert any("Topic" in h for h in headers)

    def test_dedup_step_renders(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_FILE, default_timeout=10)
        at.run()

        topics = [
            {
                "name": "Topic A",
                "description": "Desc",
                "relevance": "Rel",
                "representative_words": ["w1"],
            }
        ]

        at.session_state["maltopic_instance"] = _make_mock_maltopic()
        at.session_state["topics"] = topics
        at.session_state["current_step"] = 4
        at.run()

        headers = [h.value for h in at.header]
        assert any("Dedup" in h for h in headers)

    def test_results_step_renders(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_FILE, default_timeout=10)
        at.run()

        topics = [
            {
                "name": "Topic A",
                "description": "Desc",
                "relevance": "Rel",
                "representative_words": ["w1"],
            }
        ]

        at.session_state["maltopic_instance"] = _make_mock_maltopic()
        at.session_state["topics"] = topics
        at.session_state["skip_dedup"] = True
        at.session_state["current_step"] = 5
        at.run()

        headers = [h.value for h in at.header]
        assert any("Result" in h for h in headers)
