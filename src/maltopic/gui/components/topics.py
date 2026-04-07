"""Step 3 – Generate topics from enriched data."""

from __future__ import annotations

import json

import streamlit as st

from maltopic.gui import state as S


def render_topics() -> None:
    """Render the topic generation configuration and execution panel."""
    st.header("Step 4: Generate Topics")
    st.markdown(
        "Extract latent themes from the enriched responses using LLM-powered topic mining."
    )

    topic_mining_context = st.text_area(
        "Topic Mining Context",
        value=st.session_state.get("topic_mining_context", ""),
        height=120,
        help="Describe what topics you want to extract and why. Be specific.",
        placeholder="e.g. Identify key customer concerns, product themes, and service issues...",
    )
    st.session_state["topic_mining_context"] = topic_mining_context

    enriched_col = S.get_enriched_column_name(st.session_state)
    if enriched_col:
        st.info(f"Using enriched column: **{enriched_col}**")

    enriched_df = st.session_state.get("enriched_df")
    if enriched_df is not None and enriched_col:
        non_null = enriched_df[enriched_col].dropna().shape[0]
        st.caption(f"{non_null:,} non-empty enriched responses to analyse")

    if not topic_mining_context.strip():
        st.warning("Topic mining context is required.")
        return

    if st.button("Generate Topics", use_container_width=True, type="primary"):
        _run_topic_generation()

    # Display results
    topics = st.session_state.get("topics")
    if topics is not None:
        st.subheader(f"Generated Topics ({len(topics)})")
        _display_topics(topics)

        st.download_button(
            "Download Topics (JSON)",
            data=json.dumps(topics, indent=2),
            file_name="topics.json",
            mime="application/json",
        )

        if st.button("Continue to Deduplication →", use_container_width=True):
            st.session_state["current_step"] = 4
            st.rerun()


def _run_topic_generation() -> None:
    """Execute the topic generation step."""
    instance = st.session_state.get("maltopic_instance")
    enriched_df = st.session_state.get("enriched_df")
    enriched_col = S.get_enriched_column_name(st.session_state)
    topic_mining_context = st.session_state.get("topic_mining_context", "")

    if instance is None or enriched_df is None or not enriched_col:
        st.error("Missing prerequisites. Please complete previous steps.")
        return

    with st.spinner("Generating topics — this may take a moment…"):
        try:
            topics = instance.generate_topics(
                topic_mining_context=topic_mining_context,
                df=enriched_df,
                enriched_column=enriched_col,
            )
            st.session_state["topics"] = topics
            S.reset_from(4, st.session_state)
            st.success(f"Generated {len(topics)} topics!")
            st.rerun()
        except Exception as exc:
            st.error(f"Topic generation failed: {exc}")


def _display_topics(topics: list[dict]) -> None:
    """Render topic cards with HTML."""
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
