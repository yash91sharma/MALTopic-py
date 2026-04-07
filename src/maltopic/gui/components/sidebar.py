"""Sidebar component – step indicator, navigation, quick stats, and reset."""

from __future__ import annotations

import streamlit as st

from maltopic.gui import state as S


def render_sidebar() -> None:
    """Render the sidebar with step indicator and navigation controls."""
    with st.sidebar:
        st.markdown("## MALTopic")
        st.caption("LLM-powered topic mining")

        st.markdown("---")
        st.markdown("#### Pipeline Steps")

        current = st.session_state.get("current_step", 0)
        for idx, label in enumerate(S.STEP_LABELS):
            if idx < current:
                cls = "completed"
                icon = "&#10003;"
            elif idx == current:
                cls = "active"
                icon = "&#9679;"
            else:
                cls = ""
                icon = ""
            st.markdown(
                f'<div class="step-item {cls}">'
                f'<span class="step-dot"></span>'
                f"{idx + 1}. {label}"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Navigation buttons
        col_back, col_next = st.columns(2)
        with col_back:
            if current > 0 and st.button("← Back", use_container_width=True):
                st.session_state["current_step"] = current - 1
                st.rerun()
        with col_next:
            max_step = len(S.STEP_LABELS) - 1
            if current < max_step and S.can_proceed_to(
                current + 1, st.session_state
            ):
                if st.button("Next →", use_container_width=True):
                    st.session_state["current_step"] = current + 1
                    st.rerun()

        st.markdown("---")

        # Quick stats
        instance = st.session_state.get("maltopic_instance")
        if instance is not None:
            stats = instance.stats
            st.caption("Quick Stats")
            st.markdown(
                f"**Calls:** {stats.total_calls_made} &nbsp; "
                f"**Tokens:** {stats.total_tokens_used:,}"
            )

        # Reset
        if st.button("🔄 Start Over", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
