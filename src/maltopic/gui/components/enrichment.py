"""Step 2 – Enrich free text with structured data."""

from __future__ import annotations

import streamlit as st

from maltopic.gui import state as S


def render_enrichment() -> None:
    """Render the enrichment configuration and execution panel."""
    st.header("Step 3: Enrich Free Text")
    st.markdown(
        "The enrichment step uses your LLM to combine free-text responses with "
        "structured data, producing richer input for topic mining."
    )

    survey_context = st.text_area(
        "Survey Context",
        value=st.session_state.get("survey_context", ""),
        height=100,
        help="Describe the purpose and context of the survey. This guides the LLM.",
        placeholder="e.g. A customer satisfaction survey for an e-commerce platform...",
    )
    st.session_state["survey_context"] = survey_context

    examples_text = st.text_area(
        "Few-shot Examples (optional)",
        value=st.session_state.get("examples_text", ""),
        height=120,
        help="One example per line showing the desired enrichment format.",
        placeholder=(
            "e.g.\n"
            "Great service, 5, Electronics -> A satisfied electronics customer rated 5 stars...\n"
            "Slow delivery, 2, Fashion -> A dissatisfied fashion customer rated 2 stars..."
        ),
    )
    st.session_state["examples_text"] = examples_text

    # Show what we're working with
    free_text_col = st.session_state.get("free_text_column")
    structured_cols = st.session_state.get("structured_columns") or []

    st.info(
        f"**Free-text column:** {free_text_col} &nbsp;|&nbsp; "
        f"**Structured columns:** {', '.join(structured_cols)}"
    )

    if not survey_context.strip():
        st.warning("Survey context is required to run enrichment.")
        return

    if st.button("Run Enrichment", use_container_width=True, type="primary"):
        _run_enrichment()

    # Show results if available
    enriched_df = st.session_state.get("enriched_df")
    if enriched_df is not None:
        enriched_col = S.get_enriched_column_name(st.session_state)
        st.subheader("Enrichment Results")

        # Before / after comparison
        if enriched_col and enriched_col in enriched_df.columns:
            preview = enriched_df[[free_text_col, enriched_col]].head(10)
            st.dataframe(preview, use_container_width=True)

            csv_data = enriched_df.to_csv(index=False)
            st.download_button(
                "Download Enriched CSV",
                data=csv_data,
                file_name="enriched_data.csv",
                mime="text/csv",
            )

        if st.button("Continue to Topic Generation →", use_container_width=True):
            st.session_state["current_step"] = 3
            st.rerun()


def _run_enrichment() -> None:
    """Execute the enrichment step with a progress spinner."""
    instance = st.session_state.get("maltopic_instance")
    df = st.session_state.get("uploaded_df")
    free_text_col = st.session_state.get("free_text_column")
    structured_cols = st.session_state.get("structured_columns") or []
    survey_context = st.session_state.get("survey_context", "")
    examples = S.get_examples_list(st.session_state)

    if instance is None or df is None or not free_text_col:
        st.error("Missing prerequisites. Please complete previous steps.")
        return

    with st.spinner(f"Enriching {len(df):,} rows — this may take a while…"):
        try:
            enriched = instance.enrich_free_text_with_structured_data(
                survey_context=survey_context,
                free_text_column=free_text_col,
                structured_data_columns=structured_cols,
                df=df.copy(),
                examples=examples,
            )
            st.session_state["enriched_df"] = enriched
            S.reset_from(3, st.session_state)
            st.success("Enrichment complete!")
            st.rerun()
        except Exception as exc:
            st.error(f"Enrichment failed: {exc}")
