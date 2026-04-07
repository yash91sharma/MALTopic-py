"""Step 5 – Results dashboard, stats, and export."""

from __future__ import annotations

import json

import streamlit as st

from maltopic.gui import state as S


def render_results() -> None:
    """Render the results dashboard with topics, stats, and export options."""
    st.header("Results & Export")

    # ── Final topics ─────────────────────────────────────
    final_topics = S.get_final_topics(st.session_state)
    if final_topics is None:
        st.warning("No topics available. Please complete the pipeline first.")
        return

    st.subheader(f"Final Topics ({len(final_topics)})")
    _display_topic_cards(final_topics)

    # ── Stats dashboard ──────────────────────────────────
    instance = st.session_state.get("maltopic_instance")
    if instance is not None:
        st.subheader("Usage Statistics")
        stats = instance.get_stats()
        overview = stats.get("overview", {})
        averages = stats.get("averages", {})

        # Stat boxes row
        cols = st.columns(4)
        _stat_box(cols[0], f"{overview.get('total_calls_made', 0)}", "Total Calls")
        _stat_box(
            cols[1], f"{overview.get('total_tokens_used', 0):,}", "Total Tokens"
        )
        _stat_box(
            cols[2],
            f"{overview.get('success_rate_percent', 0):.1f}%",
            "Success Rate",
        )
        _stat_box(
            cols[3],
            f"{averages.get('avg_response_time_seconds', 0):.2f}s",
            "Avg Response Time",
        )

        # Detail breakdown
        with st.expander("Detailed Breakdown"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Token Usage**")
                st.markdown(
                    f"- Input tokens: {overview.get('total_input_tokens', 0):,}\n"
                    f"- Output tokens: {overview.get('total_output_tokens', 0):,}\n"
                    f"- Avg tokens/call: {averages.get('avg_tokens_per_call', 0):.1f}"
                )
            with col_b:
                st.markdown("**Call Stats**")
                st.markdown(
                    f"- Successful: {overview.get('successful_calls', 0)}\n"
                    f"- Failed: {overview.get('failed_calls', 0)}\n"
                    f"- Uptime: {overview.get('uptime_seconds', 0):.1f}s"
                )

            model_breakdown = stats.get("model_breakdown", {})
            if model_breakdown:
                st.markdown("**Per-Model Breakdown**")
                for model, info in model_breakdown.items():
                    st.markdown(
                        f"- **{model}**: {info.get('total_calls', 0)} calls, "
                        f"{info.get('total_tokens', 0):,} tokens"
                    )

    # ── Exports ──────────────────────────────────────────
    st.subheader("Export")
    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        st.download_button(
            "Download Topics (JSON)",
            data=json.dumps(final_topics, indent=2),
            file_name="maltopic_topics.json",
            mime="application/json",
            use_container_width=True,
        )

    enriched_df = st.session_state.get("enriched_df")
    with col_e2:
        if enriched_df is not None:
            st.download_button(
                "Download Enriched CSV",
                data=enriched_df.to_csv(index=False),
                file_name="maltopic_enriched.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with col_e3:
        if instance is not None:
            st.download_button(
                "Download Stats (JSON)",
                data=json.dumps(instance.get_stats(), indent=2, default=str),
                file_name="maltopic_stats.json",
                mime="application/json",
                use_container_width=True,
            )

    # ── Start over ───────────────────────────────────────
    st.markdown("---")
    if st.button("🔄 Start New Analysis", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def _display_topic_cards(topics: list[dict]) -> None:
    """Render topic cards."""
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


def _stat_box(container, value: str, label: str) -> None:
    """Render a styled stat box."""
    container.markdown(
        f'<div class="stat-box">'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-label">{label}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
