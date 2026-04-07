"""Step 0 – API configuration: key, model, LLM type, override parameters."""

from __future__ import annotations

import json

import streamlit as st

from maltopic.gui import state as S


def render_config() -> None:
    """Render the API configuration form."""
    st.header("Step 1: Configure API")
    st.markdown(
        "Provide your LLM credentials to initialize the MALTopic engine. "
        "All processing happens server-side; your key is never stored to disk."
    )

    with st.form("config_form"):
        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.get("api_key", ""),
            help="Your OpenAI API key (starts with sk-…)",
        )

        model_name = st.text_input(
            "Model Name",
            value=st.session_state.get("model_name", "gpt-4"),
            help="e.g. gpt-4, gpt-4o, o1-preview, gpt-5-mini",
        )

        llm_type = st.selectbox(
            "LLM Provider",
            options=["openai"],
            index=0,
            help="Currently only OpenAI is supported.",
        )

        override_text = st.text_area(
            "Override Model Parameters (optional)",
            value=st.session_state.get("override_params_text", ""),
            height=80,
            help='JSON dict, e.g. {"temperature": 0.8, "max_tokens": 500}. Leave blank for defaults.',
        )

        submitted = st.form_submit_button(
            "Initialize MALTopic", use_container_width=True
        )

    if submitted:
        # Validate
        if not api_key.strip():
            st.error("API key is required.")
            return

        if not model_name.strip():
            st.error("Model name is required.")
            return

        # Parse override params
        override_params = None
        if override_text.strip():
            try:
                override_params = json.loads(override_text.strip())
                if not isinstance(override_params, dict):
                    st.error("Override params must be a JSON object (dict).")
                    return
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON in override params: {exc}")
                return

        # Attempt initialization
        try:
            from maltopic import MALTopic

            instance = MALTopic(
                api_key=api_key.strip(),
                default_model_name=model_name.strip(),
                llm_type=llm_type,
                override_model_params=override_params,
            )
        except Exception as exc:
            st.error(f"Initialization failed: {exc}")
            return

        # Persist state
        st.session_state["api_key"] = api_key.strip()
        st.session_state["model_name"] = model_name.strip()
        st.session_state["llm_type"] = llm_type
        st.session_state["override_params_text"] = override_text
        st.session_state["maltopic_instance"] = instance

        # Reset downstream
        S.reset_from(1, st.session_state)

        st.success("MALTopic initialized successfully!")
        st.session_state["current_step"] = 1
        st.rerun()

    # Show current status
    if st.session_state.get("maltopic_instance") is not None:
        st.info(
            f"Currently connected — model: **{st.session_state.get('model_name')}**, "
            f"provider: **{st.session_state.get('llm_type')}**"
        )
