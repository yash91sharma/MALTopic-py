"""Tests for maltopic.gui.state – session state management logic."""

import pytest

from src.maltopic.gui.state import (
    STEP_LABELS,
    _DEFAULTS,
    can_proceed_to,
    get_enriched_column_name,
    get_examples_list,
    get_final_topics,
    get_parsed_override_params,
    init_state,
    reset_from,
)


@pytest.fixture
def fresh_state():
    """Return a plain dict simulating a clean st.session_state."""
    state = {}
    init_state(state)
    return state


class TestInitState:
    def test_populates_all_defaults(self, fresh_state):
        for key, default in _DEFAULTS.items():
            assert key in fresh_state
            assert fresh_state[key] == default

    def test_does_not_overwrite_existing(self):
        state = {"api_key": "my-key"}
        init_state(state)
        assert state["api_key"] == "my-key"
        # Other keys still populated
        assert "model_name" in state


class TestCanProceedTo:
    def test_step_0_always(self, fresh_state):
        assert can_proceed_to(0, fresh_state) is True

    def test_step_1_requires_instance(self, fresh_state):
        assert can_proceed_to(1, fresh_state) is False
        fresh_state["maltopic_instance"] = "fake"
        assert can_proceed_to(1, fresh_state) is True

    def test_step_2_requires_df_and_columns(self, fresh_state):
        assert can_proceed_to(2, fresh_state) is False
        fresh_state["uploaded_df"] = "df"
        assert can_proceed_to(2, fresh_state) is False
        fresh_state["free_text_column"] = "text"
        assert can_proceed_to(2, fresh_state) is False
        fresh_state["structured_columns"] = ["a"]
        assert can_proceed_to(2, fresh_state) is True

    def test_step_3_requires_enriched_df(self, fresh_state):
        assert can_proceed_to(3, fresh_state) is False
        fresh_state["enriched_df"] = "edf"
        assert can_proceed_to(3, fresh_state) is True

    def test_step_4_requires_topics(self, fresh_state):
        assert can_proceed_to(4, fresh_state) is False
        fresh_state["topics"] = [{"name": "T1"}]
        assert can_proceed_to(4, fresh_state) is True

    def test_step_5_requires_topics_and_dedup_or_skip(self, fresh_state):
        assert can_proceed_to(5, fresh_state) is False
        fresh_state["topics"] = [{"name": "T1"}]
        assert can_proceed_to(5, fresh_state) is False
        fresh_state["skip_dedup"] = True
        assert can_proceed_to(5, fresh_state) is True

    def test_step_5_with_deduped_topics(self, fresh_state):
        fresh_state["topics"] = [{"name": "T1"}]
        fresh_state["deduped_topics"] = [{"name": "T1"}]
        assert can_proceed_to(5, fresh_state) is True

    def test_invalid_step_returns_false(self, fresh_state):
        assert can_proceed_to(99, fresh_state) is False


class TestResetFrom:
    def test_reset_from_0_clears_everything(self, fresh_state):
        fresh_state["maltopic_instance"] = "inst"
        fresh_state["topics"] = [{"name": "T"}]
        reset_from(0, fresh_state)
        assert fresh_state["maltopic_instance"] is None
        assert fresh_state["topics"] is None

    def test_reset_from_2_clears_enrichment_and_downstream(self, fresh_state):
        fresh_state["uploaded_df"] = "df"
        fresh_state["enriched_df"] = "edf"
        fresh_state["topics"] = [{"name": "T"}]
        reset_from(2, fresh_state)
        assert fresh_state["uploaded_df"] == "df"  # Preserved
        assert fresh_state["enriched_df"] is None
        assert fresh_state["topics"] is None

    def test_reset_from_4_clears_only_dedup(self, fresh_state):
        fresh_state["topics"] = [{"name": "T"}]
        fresh_state["deduped_topics"] = [{"name": "T"}]
        fresh_state["skip_dedup"] = True
        reset_from(4, fresh_state)
        assert fresh_state["topics"] == [{"name": "T"}]  # Preserved
        assert fresh_state["deduped_topics"] is None
        assert fresh_state["skip_dedup"] is False

    def test_reset_from_unknown_step_is_noop(self, fresh_state):
        fresh_state["topics"] = [{"name": "T"}]
        reset_from(99, fresh_state)
        assert fresh_state["topics"] == [{"name": "T"}]


class TestParsedOverrideParams:
    def test_empty_string(self, fresh_state):
        assert get_parsed_override_params(fresh_state) is None

    def test_valid_json(self, fresh_state):
        fresh_state["override_params_text"] = '{"temperature": 0.5}'
        result = get_parsed_override_params(fresh_state)
        assert result == {"temperature": 0.5}

    def test_invalid_json(self, fresh_state):
        fresh_state["override_params_text"] = "not json"
        assert get_parsed_override_params(fresh_state) is None

    def test_non_dict_json(self, fresh_state):
        fresh_state["override_params_text"] = "[1, 2, 3]"
        assert get_parsed_override_params(fresh_state) is None


class TestExamplesList:
    def test_empty(self, fresh_state):
        assert get_examples_list(fresh_state) == []

    def test_multiline(self, fresh_state):
        fresh_state["examples_text"] = "example one\nexample two\n\nexample three"
        result = get_examples_list(fresh_state)
        assert result == ["example one", "example two", "example three"]

    def test_whitespace_lines_ignored(self, fresh_state):
        fresh_state["examples_text"] = "  \n  hello  \n  "
        assert get_examples_list(fresh_state) == ["hello"]


class TestEnrichedColumnName:
    def test_with_column(self, fresh_state):
        fresh_state["free_text_column"] = "feedback"
        assert get_enriched_column_name(fresh_state) == "feedback_enriched"

    def test_without_column(self, fresh_state):
        assert get_enriched_column_name(fresh_state) is None


class TestGetFinalTopics:
    def test_prefers_deduped(self, fresh_state):
        fresh_state["topics"] = [{"name": "raw"}]
        fresh_state["deduped_topics"] = [{"name": "deduped"}]
        assert get_final_topics(fresh_state) == [{"name": "deduped"}]

    def test_falls_back_to_raw(self, fresh_state):
        fresh_state["topics"] = [{"name": "raw"}]
        assert get_final_topics(fresh_state) == [{"name": "raw"}]

    def test_returns_none_when_empty(self, fresh_state):
        assert get_final_topics(fresh_state) is None


class TestStepLabels:
    def test_has_six_steps(self):
        assert len(STEP_LABELS) == 6

    def test_all_strings(self):
        for label in STEP_LABELS:
            assert isinstance(label, str)
            assert len(label) > 0
