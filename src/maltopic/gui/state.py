"""Session state manager for the MALTopic GUI.

Provides a typed wrapper around Streamlit's session_state to manage
the full pipeline state: configuration, data, enrichment, topics, and results.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

STEP_LABELS = [
    "Configure API",
    "Upload Data",
    "Enrich Text",
    "Generate Topics",
    "Deduplicate Topics",
    "Results & Export",
]

_DEFAULTS: dict[str, Any] = {
    # Step 0 – Configuration
    "api_key": "",
    "model_name": "gpt-4",
    "llm_type": "openai",
    "override_params_text": "",
    "maltopic_instance": None,
    # Step 1 – Data upload
    "uploaded_df": None,
    "free_text_column": None,
    "structured_columns": [],
    # Step 2 – Enrichment
    "survey_context": "",
    "examples_text": "",
    "enriched_df": None,
    # Step 3 – Topics
    "topic_mining_context": "",
    "topics": None,
    # Step 4 – Deduplication
    "dedup_survey_context": "",
    "deduped_topics": None,
    "skip_dedup": False,
    # Navigation
    "current_step": 0,
}


def init_state(session_state: dict[str, Any]) -> None:
    """Populate session_state with defaults for any keys not yet set."""
    for key, default in _DEFAULTS.items():
        if key not in session_state:
            session_state[key] = default


def can_proceed_to(step: int, session_state: dict[str, Any]) -> bool:
    """Return True if all prerequisites for *step* are satisfied."""
    if step <= 0:
        return True
    if step == 1:
        return session_state.get("maltopic_instance") is not None
    if step == 2:
        return (
            session_state.get("uploaded_df") is not None
            and session_state.get("free_text_column") is not None
            and len(session_state.get("structured_columns") or []) > 0
        )
    if step == 3:
        return session_state.get("enriched_df") is not None
    if step == 4:
        return session_state.get("topics") is not None
    if step == 5:
        topics = session_state.get("topics")
        deduped = session_state.get("deduped_topics")
        skip = session_state.get("skip_dedup", False)
        return topics is not None and (deduped is not None or skip)
    return False


def reset_from(step: int, session_state: dict[str, Any]) -> None:
    """Clear all state that depends on *step* and later.

    For example, ``reset_from(1)`` clears data upload **and** all downstream
    results (enrichment, topics, dedup).
    """
    cascade: dict[int, list[str]] = {
        0: [
            "maltopic_instance",
            "uploaded_df",
            "free_text_column",
            "structured_columns",
            "survey_context",
            "examples_text",
            "enriched_df",
            "topic_mining_context",
            "topics",
            "dedup_survey_context",
            "deduped_topics",
            "skip_dedup",
        ],
        1: [
            "uploaded_df",
            "free_text_column",
            "structured_columns",
            "survey_context",
            "examples_text",
            "enriched_df",
            "topic_mining_context",
            "topics",
            "dedup_survey_context",
            "deduped_topics",
            "skip_dedup",
        ],
        2: [
            "enriched_df",
            "topic_mining_context",
            "topics",
            "dedup_survey_context",
            "deduped_topics",
            "skip_dedup",
        ],
        3: ["topics", "dedup_survey_context", "deduped_topics", "skip_dedup"],
        4: ["deduped_topics", "skip_dedup"],
    }
    for key in cascade.get(step, []):
        session_state[key] = _DEFAULTS[key]


def get_parsed_override_params(session_state: dict[str, Any]) -> dict | None:
    """Parse the override_params_text JSON string. Return None if empty or invalid."""
    import json

    text = (session_state.get("override_params_text") or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return None
        return parsed
    except (json.JSONDecodeError, TypeError):
        return None


def get_examples_list(session_state: dict[str, Any]) -> list[str]:
    """Parse multi-line examples text into a list of non-empty strings."""
    text = (session_state.get("examples_text") or "").strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def get_enriched_column_name(session_state: dict[str, Any]) -> str | None:
    """Return the expected enriched column name, or None."""
    col = session_state.get("free_text_column")
    if col:
        return f"{col}_enriched"
    return None


def get_final_topics(session_state: dict[str, Any]) -> list[dict[str, str]] | None:
    """Return the best available topics (deduped if present, else raw)."""
    deduped = session_state.get("deduped_topics")
    if deduped is not None:
        return deduped
    return session_state.get("topics")
