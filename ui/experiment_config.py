"""
UI components for experiment configuration.
"""

import streamlit as st
from typing import Dict, Any

from constants import (
    SUPPORTED_MODELS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    MIN_FEW_SHOT_EXAMPLES,
    MAX_FEW_SHOT_EXAMPLES,
    DEFAULT_FEW_SHOT_EXAMPLES
)
from api_utils import get_model_pricing, estimate_api_cost
from validation import validate_api_key


def render_api_configuration() -> Dict[str, Any]:
    """
    Render API key input and validation section.
    
    Returns:
        Dict with 'api_key' and 'api_key_valid' keys
    """
    st.markdown("### API Configuration")
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False
    
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key or "",
        help="Enter your OpenAI API key. It will be stored in session state only and never saved.",
        key="api_key_input"
    )
    
    if api_key_input:
        if st.session_state.api_key != api_key_input:
            st.session_state.api_key_valid = False
        
        st.session_state.api_key = api_key_input
        
        if st.button("Validate API Key", key="validate_key"):
            with st.spinner("Validating API key..."):
                is_valid, message = validate_api_key(api_key_input)
                st.session_state.api_key_valid = is_valid
                if is_valid:
                    st.success(message)
                else:
                    st.error(message)
        elif 'api_key_valid' in st.session_state and st.session_state.api_key_valid:
            st.success("API key is valid and connected")
        elif api_key_input and not st.session_state.api_key_valid:
            st.info("Click 'Validate API Key' to verify your API key")
    else:
        st.session_state.api_key = None
        st.session_state.api_key_valid = False
        st.info("Enter your OpenAI API key to continue. Your API key is stored in session state only and never saved.")
    
    return {
        'api_key': st.session_state.api_key,
        'api_key_valid': st.session_state.api_key_valid
    }


def render_model_configuration() -> Dict[str, Any]:
    """
    Render model selection and parameter configuration.
    
    Returns:
        Dict with 'model_name', 'temperature', 'max_tokens' keys
    """
    st.markdown("### Model Configuration")
    
    if 'model_name' not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL
    if 'temperature' not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    
    # Model selection
    default_index = 0
    if 'model_name' in st.session_state and st.session_state.model_name in SUPPORTED_MODELS:
        default_index = SUPPORTED_MODELS.index(st.session_state.model_name)
    
    selected_model = st.selectbox(
        "Model",
        options=SUPPORTED_MODELS,
        index=default_index,
        help="Select the OpenAI model to use for classification"
    )
    st.session_state.model_name = selected_model
    
    pricing = get_model_pricing(selected_model)
    st.caption(
        f"Estimated cost: ${pricing['input']:.2f} per 1M input tokens, "
        f"${pricing['output']:.2f} per 1M output tokens"
    )
    
    # Temperature
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Lower values (0.0-0.2) for consistent outputs, higher values (0.7-1.0) for more creative responses"
    )
    st.session_state.temperature = temperature
    
    # Max tokens
    max_tokens = st.number_input(
        "Max Tokens",
        min_value=10,
        max_value=200,
        value=st.session_state.max_tokens,
        help="Maximum tokens for model response. 50 is sufficient for label names like 'Gayphobia, Transphobia'"
    )
    st.session_state.max_tokens = max_tokens
    
    return {
        'model_name': selected_model,
        'temperature': temperature,
        'max_tokens': max_tokens
    }


def render_experiment_configuration() -> Dict[str, Any]:
    """
    Render experiment settings (few-shot examples, cost estimate).
    
    Returns:
        Dict with 'n_few_shot_examples' and cost estimate info
    """
    st.markdown("### Experiment Configuration")
    
    n_few_shot_examples = st.slider(
        "Few-Shot Examples",
        min_value=MIN_FEW_SHOT_EXAMPLES,
        max_value=MAX_FEW_SHOT_EXAMPLES,
        value=DEFAULT_FEW_SHOT_EXAMPLES,
        step=1,
        help=f"Number of few-shot examples to include in the prompt ({MIN_FEW_SHOT_EXAMPLES}-{MAX_FEW_SHOT_EXAMPLES} examples)"
    )
    
    st.session_state.experiment_config = {
        'n_few_shot_examples': n_few_shot_examples
    }
    
    st.divider()
    
    # Cost estimate
    st.markdown("### Cost Estimate")
    
    estimated_test_size = 300  # Approximate based on 35% split of 862
    
    cost_estimate = estimate_api_cost(
        model_name=st.session_state.model_name,
        num_test_tweets=estimated_test_size,
        num_ablations=1,
        few_shot_examples=n_few_shot_examples
    )
    
    st.metric("Estimated Cost", f"${cost_estimate['total_cost']:.2f}")
    
    with st.expander("Detailed Cost Breakdown", expanded=False):
        st.markdown(f"**Input Cost**: ${cost_estimate['input_cost']:.4f}")
        st.markdown(f"**Output Cost**: ${cost_estimate['output_cost']:.4f}")
        st.markdown(f"**Tokens per Request (Input)**: {cost_estimate['tokens_per_request_input']:,}")
        st.markdown(f"**Tokens per Request (Output)**: {cost_estimate['tokens_per_request_output']:,}")
        st.markdown(f"**Estimated Test Set Size**: ~{estimated_test_size} tweets")
    
    if cost_estimate['total_cost'] > 10:
        st.warning(f"Estimated cost exceeds $10. Total: ${cost_estimate['total_cost']:.2f}")
    
    return {
        'n_few_shot_examples': n_few_shot_examples,
        'cost_estimate': cost_estimate
    }

