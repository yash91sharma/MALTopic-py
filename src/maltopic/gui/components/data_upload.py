"""Step 1 – CSV upload, preview, and column selection."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from maltopic.gui import state as S


def render_data_upload() -> None:
    """Render the data upload panel with preview and column selectors."""
    st.header("Step 2: Upload Data")
    st.markdown("Upload a CSV file containing your survey / text data.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Maximum 200 MB",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")
            return

        if df.empty:
            st.error("The uploaded CSV is empty.")
            return

        st.session_state["uploaded_df"] = df
        S.reset_from(2, st.session_state)

        st.success(f"Loaded **{len(df):,}** rows × **{len(df.columns)}** columns")

    # Continue with whatever df is in state (survives re-runs)
    df = st.session_state.get("uploaded_df")
    if df is None:
        return

    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Column Selection")

    columns = list(df.columns)

    free_text = st.selectbox(
        "Free-text column",
        options=columns,
        index=columns.index(st.session_state["free_text_column"])
        if st.session_state.get("free_text_column") in columns
        else 0,
        help="The column containing unstructured text responses.",
    )

    remaining = [c for c in columns if c != free_text]
    default_structured = [
        c
        for c in (st.session_state.get("structured_columns") or [])
        if c in remaining
    ]
    structured = st.multiselect(
        "Structured data columns",
        options=remaining,
        default=default_structured,
        help="Select one or more columns with categorical / structured data to enrich the free text.",
    )

    st.session_state["free_text_column"] = free_text
    st.session_state["structured_columns"] = structured

    if free_text and structured:
        st.info(
            f"Free-text: **{free_text}** &nbsp;|&nbsp; Structured: **{', '.join(structured)}**"
        )
        if S.can_proceed_to(2, st.session_state):
            if st.button("Continue to Enrichment →", use_container_width=True):
                st.session_state["current_step"] = 2
                st.rerun()
    else:
        st.warning("Select both a free-text column and at least one structured column to continue.")
