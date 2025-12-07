"""
UI components for data loading section.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from constants import (
    CLASS_NAMES,
    CLASS_LABELS,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_FEW_SHOT_EXAMPLES
)
from data_loading import load_homo_mex_dataset, prepare_label_vectors, validate_dataset
from classification import split_train_test
from few_shot import create_single_few_shot_pool
from classification import format_pool_examples


def render_data_loading_section() -> None:
    """Render the data loading and few-shot pool preparation section."""
    st.header("Data Loading & Few-Shot Pool Preparation")
    
    st.markdown(
        """
        Load the HOMO-MEX training dataset. The dataset will be split into train/test sets 
        immediately, and a default few-shot pool will be created for interactive predictions.
        """
    )
    
    if st.button("Load Dataset & Create Few-Shot Pools", type="primary"):
        with st.spinner("Loading dataset, splitting train/test, and creating default few-shot pool..."):
            try:
                full_df = load_homo_mex_dataset()
                full_df = prepare_label_vectors(full_df)
                validation = validate_dataset(full_df)

                # Split dataset immediately
                train_df, test_df = split_train_test(
                    full_df,
                    test_size=DEFAULT_TEST_SIZE,
                    random_seed=DEFAULT_RANDOM_SEED
                )
                
                # Create default few-shot pool
                default_pool = create_single_few_shot_pool(
                    train_df,
                    DEFAULT_FEW_SHOT_EXAMPLES,
                    random_seed=DEFAULT_RANDOM_SEED
                )
                few_shot_examples = format_pool_examples(default_pool)

                st.session_state.full_df = full_df
                st.session_state.train_df = train_df
                st.session_state.test_df = test_df
                st.session_state.label_counts = validation['stats']['label_counts']
                st.session_state.validation_results = validation
                st.session_state.default_few_shot_examples = few_shot_examples

                st.success(
                    f"Dataset loaded successfully! Split: {len(train_df)} training tweets, "
                    f"{len(test_df)} test tweets. Default {DEFAULT_FEW_SHOT_EXAMPLES}-shot pool created."
                )
                
                # Show few-shot example preview
                with st.expander(f"View Default Few-Shot Examples ({DEFAULT_FEW_SHOT_EXAMPLES} total)", expanded=False):
                    for idx, example in enumerate(few_shot_examples[:5]):
                        st.markdown(f"**Example {idx+1}**: {example['labels']}")
                        st.caption(example['text'][:150] + ("..." if len(example['text']) > 150 else ""))
                    st.caption(f"*Showing 5 of {len(few_shot_examples)} examples*")
                
            except FileNotFoundError as e:
                st.error(f"Error: {str(e)}")
                st.info("Please ensure the Excel file is located at: `data/Annotated LGBTQ+ Phobia Tweets.xlsx`")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)
    
    _render_dataset_statistics()


def _render_dataset_statistics() -> None:
    """Render dataset statistics if data is loaded."""
    if 'full_df' not in st.session_state and 'train_df' not in st.session_state:
        return
    
    if 'train_df' in st.session_state:
        display_df = st.session_state.train_df
    else:
        display_df = st.session_state.full_df
    
    st.markdown("### Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tweets", len(display_df))
    with col2:
        st.metric(
            "Multi-Label Examples",
            st.session_state.validation_results['stats']['multi_label_count']
        )
    with col3:
        st.metric("Classes", 5)
    with col4:
        warnings = len(st.session_state.validation_results.get('warnings', []))
        if warnings == 0:
            st.metric("Validation", "Passed")
        else:
            st.metric("Validation", f"{warnings} warnings")
    
    if 'test_df' in st.session_state:
        st.info(
            f"Dataset split: {len(st.session_state.train_df)} training tweets, "
            f"{len(st.session_state.test_df)} test tweets"
        )
    else:
        st.info("Dataset will be split into train/test sets when you run an experiment.")

    st.markdown("**Label Distribution:**")
    label_data = {
        'Class': [CLASS_NAMES[i] for i in range(5)],
        'Label': CLASS_LABELS,
        'Count': [st.session_state.label_counts[label] for label in CLASS_LABELS],
        'Expected': [714, 72, 10, 79, 64]
    }
    label_df = pd.DataFrame(label_data)
    label_df['Match'] = label_df['Count'] == label_df['Expected']
    st.dataframe(label_df, width='stretch', hide_index=True)
    
    if st.session_state.validation_results.get('warnings'):
        with st.expander("Validation Warnings", expanded=False):
            for warning in st.session_state.validation_results['warnings']:
                st.warning(warning)

