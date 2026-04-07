"""Custom CSS theme for the MALTopic GUI – modern, minimal, cool palette."""

THEME_CSS = """
<style>
/* ── Global ──────────────────────────────────────────── */
:root {
    --accent: #4F8BF9;
    --accent-light: #E8F0FE;
    --bg-card: #FFFFFF;
    --bg-page: #F7F8FA;
    --text-primary: #1A1A2E;
    --text-secondary: #6B7280;
    --border: #E5E7EB;
    --success: #10B981;
    --warning: #F59E0B;
    --error: #EF4444;
}

[data-testid="stAppViewContainer"] {
    background-color: var(--bg-page);
}

/* ── Cards ───────────────────────────────────────────── */
.topic-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease;
}
.topic-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.topic-card h4 {
    margin: 0 0 0.5rem 0;
    color: var(--text-primary);
}
.topic-card p {
    margin: 0.25rem 0;
    color: var(--text-secondary);
    font-size: 0.95rem;
}
.topic-card .words {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.5rem;
}
.topic-card .words span {
    background: var(--accent-light);
    color: var(--accent);
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 500;
}

/* ── Sidebar step indicator ──────────────────────────── */
.step-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.45rem 0.6rem;
    border-radius: 8px;
    margin-bottom: 0.25rem;
    font-size: 0.9rem;
    color: var(--text-secondary);
}
.step-item.active {
    background: var(--accent-light);
    color: var(--accent);
    font-weight: 600;
}
.step-item.completed {
    color: var(--success);
}
.step-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--border);
    flex-shrink: 0;
}
.step-item.active .step-dot { background: var(--accent); }
.step-item.completed .step-dot { background: var(--success); }

/* ── Stat boxes ──────────────────────────────────────── */
.stat-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.stat-box .stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
}
.stat-box .stat-label {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-top: 0.2rem;
}

/* ── Misc tweaks ─────────────────────────────────────── */
.stButton>button {
    border-radius: 8px;
}
div[data-testid="stSidebar"] {
    background: #FAFBFC;
}

</style>
"""


def inject_css():
    """Inject the custom CSS into the Streamlit page."""
    import streamlit as st

    st.markdown(THEME_CSS, unsafe_allow_html=True)
