"""Step 4 – Deduplicate topics (optional)."""

from __future__ import annotations

import json

import streamlit as st

from maltopic.gui import state as S


def render_dedup() -> None:
    """Render the deduplication configuration and execution panel."""
    st.header("Step 5: Deduplicate Topics")
    st.markdown(
        "Optionally merge semantically similar topics. "
        "You can skip this step if you're happy with the generated topics."
    )

    topics = st.session_state.get("topics")
    if topics is None:
        st.warning("No topics available. Please generate topics first.")
        return

    st.info(f"**{len(topics)}** topics available for deduplication.")

    # Skip option
    skip = st.checkbox(
        "Skip deduplication",
        value=st.session_state.get("skip_dedup", False),
        help="Check to keep all topics as-is and proceed directly to results.",
    )
    st.session_state["skip_dedup"] = skip

    if skip:
        st.session_state["deduped_topics"] = None
        if st.button("Continue to Results →", use_container_width=True):
            st.session_state["current_step"] = 5
            st.rerun()
        return

    # Dedup context
    default_ctx = st.session_state.get("survey_context", "")
    dedup_ctx = st.text_area(
        "Survey Context for Deduplication",
        value=st.session_state.get("dedup_survey_context", "") or default_ctx,
        height=80,
        help="Context to guide the LLM when merging similar topics.",
    )
    st.session_state["dedup_survey_context"] = dedup_ctx

    if not dedup_ctx.strip():
        st.warning("Survey context is required for deduplication.")
        return

    if st.button("Deduplicate Topics", use_container_width=True, type="primary"):
        _run_dedup()

    # Display results
    deduped = st.session_state.get("deduped_topics")
    if deduped is not None:
        st.subheader("Deduplication Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Before", len(topics))
        with col2:
            st.metric("After", len(deduped), delta=len(deduped) - len(topics))

        _display_dedup_topics(deduped)

        st.download_button(
            "Download Deduplicated Topics (JSON)",
            data=json.dumps(deduped, indent=2),
            file_name="topics_deduplicated.json",
            mime="application/json",
        )

        if st.button("Continue to Results →", use_container_width=True):
            st.session_state["current_step"] = 5
            st.rerun()


def _run_dedup() -> None:
    """Execute the deduplication step."""
    instance = st.session_state.get("maltopic_instance")
    topics = st.session_state.get("topics")
    dedup_ctx = st.session_state.get("dedup_survey_context", "")

    if instance is None or topics is None:
        st.error("Missing prerequisites.")
        return

    with st.spinner("Deduplicating topics…"):
        try:
            deduped = instance.deduplicate_topics(
                topics=topics,
                survey_context=dedup_ctx,
            )
            st.session_state["deduped_topics"] = deduped
            st.success(
                f"Deduplication complete: {len(topics)} → {len(deduped)} topics"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Deduplication failed: {exc}")


def _display_dedup_topics(topics: list[dict]) -> None:
    """Render deduplicated topic cards."""
    for topic in topics:
        words = topic.get("representative_words", [])
        if isinstance(words, str):
            words = [w.strip() for w in words.split(",") if w.strip()]
        words_html = "".join(f"<span>{w}</span>" for w in words)
        st.markdown(
            f'<div class="topic-card">'
            f'<h4>{topic.get("name", "Unnamed")}</h4>'
            f'<p><strong>Description:</strong> {topic.get("description", "")}</p>'
            f'<p><strong>Relevance:</strong> {topic.get("relevance", "")}</p>'
            f'<div class="words">{words_html}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
