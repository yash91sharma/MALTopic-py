"""MALTopic GUI – main Streamlit application.

This is the file that Streamlit executes.  It sets up page config,
injects custom CSS, initialises session state, renders the sidebar,
and routes to the active pipeline step.
"""

from __future__ import annotations

import streamlit as st

from maltopic.gui import state as S
from maltopic.gui.styles import inject_css
from maltopic.gui.components.sidebar import render_sidebar
from maltopic.gui.components.config import render_config
from maltopic.gui.components.data_upload import render_data_upload
from maltopic.gui.components.enrichment import render_enrichment
from maltopic.gui.components.topics import render_topics
from maltopic.gui.components.dedup import render_dedup
from maltopic.gui.components.results import render_results

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="MALTopic",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ──────────────────────────────────────────
inject_css()

# ── Session state ────────────────────────────────────────
S.init_state(st.session_state)

# ── Privacy notice ────────────────────────────────────────
st.markdown(
    '<div style="background:linear-gradient(135deg,#EEF2FF,#F0FDFA);border-left:4px solid #4F8BF9;'
    "border-radius:8px;padding:0.9rem 1.1rem;margin-bottom:1.2rem;"
    'font-size:0.88rem;color:#1A1A2E;">'
    "<strong>🔒 Your data stays with you.</strong> &nbsp;"
    "MALTopic does <em>not</em> collect, store, or transmit any of your data. "
    "Everything runs locally on your machine. The only exception is that text you "
    "process will be sent to your configured LLM provider (e.g.&nbsp;OpenAI) to "
    "perform enrichment and topic mining — subject to that provider's data policies."
    "</div>",
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────
render_sidebar()

# ── Route to current step ────────────────────────────────
_STEP_RENDERERS = [
    render_config,       # 0
    render_data_upload,  # 1
    render_enrichment,   # 2
    render_topics,       # 3
    render_dedup,        # 4
    render_results,      # 5
]

current_step: int = st.session_state.get("current_step", 0)
if 0 <= current_step < len(_STEP_RENDERERS):
    _STEP_RENDERERS[current_step]()
else:
    st.session_state["current_step"] = 0
    _STEP_RENDERERS[0]()
