import streamlit as st
import pandas as pd
import numpy as np

from constants import CLASS_NAMES, CLASS_LABELS
from data_loading import load_homo_mex_dataset, prepare_label_vectors, validate_dataset
from api_utils import get_model_pricing, estimate_api_cost, validate_api_key
from classification import (
    split_train_test,
    run_classification_experiment
)
from metrics import (
    extract_predictions_from_results,
    calculate_metrics,
    calculate_per_class_metrics,
    calculate_confusion_matrices,
    analyze_label_cooccurrence,
    BERT_BASELINE
)
from visualizations import (
    create_metrics_table,
    plot_all_confusion_matrices
)


def main():
    st.set_page_config(page_title="LGBT+Phobia Detection", layout="wide")

    st.title("Multi-Label LGBT+Phobia Detection in Mexican Spanish")

    with st.container():
        st.markdown(
            """
        This project extends the [HOMO-MEX corpus](https://github.com/juanmvsa/HOMO-MEX) 
        (Vásquez et al., 2023) using few-shot learning with large language models.
        """
        )

    st.header("The Challenge")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        The original HOMO-MEX research identified **fine-grained LGBT+phobia classification** 
        as particularly challenging. Their best BERT model achieved **74% F1-score** on 
        multi-label classification across five categories.
        """
        )

        st.markdown("**Class Distribution:**")
        class_data = {
            "Category": [
                "Gayphobia",
                "Lesbophobia",
                "Biphobia",
                "Transphobia",
                "Other",
            ],
            "Examples": [714, 72, 10, 79, 64],
        }
        class_df = pd.DataFrame(class_data)
        st.dataframe(class_df, width='stretch', hide_index=True)

        st.info(
            "The extreme class imbalance and cultural nuances of Mexican Spanish made this difficult for traditional supervised learning."
        )

    with col2:
        st.metric("BERT F1-Score", "74%")
        st.metric("Training Examples", "862")
        st.metric("Lowest Class", "10 (Biphobia)")
        st.metric("Classes", "5")

    st.header("Our Approach")

    approach_cols = st.columns(3)

    with approach_cols[0]:
        st.markdown(
            """
        **Biphobia Challenge**: With only 10 training examples (1% of dataset), 
BERT achieved just 42% F1. Few-shot learning may help by leveraging 
semantic understanding rather than statistical patterns.
        """
        )

    with approach_cols[1]:
        st.markdown(
            """
        **Rapid Deployment**
        
        No expensive fine-tuning required. Deploy quickly with just a few examples.
        """
        )

    with approach_cols[2]:
        st.markdown(
            """
        **Interpretability**
        
        Explicit prompting provides transparency in classification decisions.
        """
        )

    st.markdown(
        """
    I explore whether **few-shot prompting with GPT-4** can match or exceed BERT 
    fine-tuning performance using only approximately 15 labeled examples instead of 862.
    """
    )

    with st.expander("Key Research Questions", expanded=False):
        st.markdown(
            """
        1. **Sample Efficiency**: Can 15 few-shot examples compete with 862-example BERT training?
        2. **Minority Class Performance**: Does few-shot learning handle low-resource classes 
        (Biphobia, Lesbophobia) better than fine-tuning?
        3. **Cost-Benefit Tradeoff**: How do API costs compare to GPU training time and compute?
        4. **Cross-Class Generalization**: Can the model learn shared patterns across LGBT+phobia types?
        """
        )

    st.divider()

    st.header("Methodology")

    method_cols = st.columns(3)

    with method_cols[0]:
        st.markdown("**Single Multi-Label Prompt**")
        st.code(
            """You are a classifier for LGBT+phobic 
content in Mexican Spanish.

Classify this tweet: "{spanish_tweet}"

Which types of LGBT+phobia does this 
tweet contain?
Options: Gayphobia, Lesbophobia, 
Biphobia, Transphobia, Other

Answer with all labels that apply 
(can be multiple or none).""",
            language="python",
        )
        st.markdown(
            """
        - One API call per tweet
        - Captures label correlations
        - Handles multi-label naturally
        """
        )

    with method_cols[1]:
        st.markdown("**Stratified Sampling**")
        st.markdown(
            """
        - Minimum 2-3 examples per class
        - Fill remaining with common classes
        - Ensures balanced coverage
        """
        )
        st.caption("Handles class imbalance (Biphobia: 10 examples)")

    with method_cols[2]:
        st.markdown("**Prompt Language**")
        st.code(
            """You are a classifier for LGBT+phobic 
content in Mexican Spanish.

Classify this tweet: "{spanish_tweet}"

Labels: Gayphobia, Lesbophobia, 
Biphobia, Transphobia, Other

Answer with all that apply.""",
            language="python",
        )
        with st.expander("Example", expanded=False):
            st.code(
                """You are a classifier for LGBT+phobic 
content in Mexican Spanish.

Classify this tweet: "Los maricones no deberían 
tener los mismos derechos"

Labels: Gayphobia, Lesbophobia, 
Biphobia, Transphobia, Other

Answer: Gayphobia""",
                language="python",
            )
        st.caption("English instructions with Spanish tweet content")

    st.divider()

    st.header("Implementation Details")

    impl_cols = st.columns(3)

    with impl_cols[0]:
        st.markdown("**Model Configuration**")
        st.markdown(
            """
        - **LLM**: GPT-4-turbo (128K context window)
        - **Temperature**: 0.1 (for consistency)
        - **Few-shot examples**: 15 total (2-3 per class)
        - **Prompt language**: English instructions, Spanish tweet content
        """
        )

    with impl_cols[1]:
        st.markdown("**Dataset Split**")
        st.markdown(
            """
        Using the exact train/test split from HOMO-MEX:
        
        - **Training pool**: 862 tweets → Select 15 for few-shot examples
        - **Test set**: 477 tweets (unchanged for direct comparison)
        """
        )

    with impl_cols[2]:
        st.markdown("**Multi-Label Handling**")
        st.markdown(
            """
        Tweets can have 0 to 5 labels simultaneously. We parse LLM output to extract 
        all mentioned categories and convert to binary vectors.
        """
        )
        st.code(
            """Example output: "Gayphobia, Transphobia" 
→ [1, 0, 0, 1, 0]""",
            language="python",
        )

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Validation Process", "Citation"])

    with tab1:
        st.subheader("What We're Working With")

        st.markdown(
            """
        The public data available for this project contains only the **fine-grained classification subset** 
        of the HOMO-MEX corpus, which is why we work with the multi-label classification task 
        (G, L, B, T, O) rather than the binary LGBT+Phobia detection task.
        """
        )

        st.markdown("**HOMO-MEX Corpus Structure:**")

        subset_cols = st.columns(2)

        with subset_cols[0]:
            st.markdown(
                """
            **LGBT+Phobia Detection Subset**
            
            - Labels: LGBT+Phobic (LP), Not LGBT+Phobic (NLP), Irrelevant (I)
            - Total: 11,000 tweets
            - Train: 7,000 | Test: 4,000
            """
            )

        with subset_cols[1]:
            st.markdown(
                """
            **Fine-Grained Classification Subset**
            
            - Labels: G, L, B, T, O (multi-label)
            - Total: 1,339 tweets
            - Train: 862 | Test: 477
            """
            )

        st.warning(
            """
        **Important**: Tweets in the fine-grained subset can have **more than one label** at a time. 
        This multi-label nature means the number of labels can exceed the total number of tweets.
        """
        )

    with tab2:
        st.subheader("Data Validation Methodology")

        st.markdown(
            """
        **Cross-Reference and Validation Process**
        
        The public data provided by the authors consisted of two datasets:
        1. Dataset with actual tweet text
        2. Dataset with only tweet IDs (for terms of service compliance)
        """
        )

        validation_cols = st.columns(3)

        with validation_cols[0]:
            st.metric("CSV Rows", "862")

        with validation_cols[1]:
            st.metric("Excel Rows", "862")

        with validation_cols[2]:
            st.metric("Match Status", "Verified")

        st.success(
            """
        **Data Verification**: Both datasets contain exactly 862 rows, which matches the train partition 
        size reported in the paper. This confirms we are working with the complete training set of the 
        fine-grained classification subset.
        
        Cross-referencing confirmed no data loss or addition between the two datasets.
        """
        )

    with tab3:
        st.subheader("Citation")

        st.markdown(
            """
        **Vásquez, J., Andersen, S. T., Bel-Enguix, G., Gómez-Adorno, H., & Ojeda-Trueba, S.-L.** (2023). 
        Experiments on the HOMO-MEX Corpus for LGBT+phobia Detection. 
        
        In *Proceedings of the 7th Workshop on Online Abuse and Harms (WOAH 2023)*, pages 200-210. 
        Association for Computational Linguistics.
        
        **Repository**: [HOMO-MEX on GitHub](https://github.com/juanmvsa/HOMO-MEX)
        
        **Paper**: [ACL Anthology](https://aclanthology.org/2023.woah-1.20.pdf)
        """
        )
        
    st.divider()
    
    st.header("Evaluation Strategy")
    
    st.markdown(
        """
        We'll compare our few-shot LLM approach against the published BERT results 
        (Table 8 from Vásquez et al., 2023) using:
        """
    )
    
    eval_cols = st.columns(2)
    
    with eval_cols[0]:
        st.markdown("**Metrics**")
        st.markdown(
            """
        - **Per-Class F1 Score**: Primary metric for each category (G/L/B/T/O)
        - **Macro-Average F1**: Overall performance across all classes
        - **Micro-Average F1**: Weighted by class frequency
        """
        )
    
    with eval_cols[1]:
        st.markdown("**Success Criteria**")
        st.markdown(
            """
        - Match or exceed 74% F1-score with 98% fewer training examples
        - Improve F1 on minority classes (L/B/T) compared to BERT
        - Demonstrate cost-effectiveness of few-shot vs. fine-tuning
        """
        )
    
    st.markdown("**Baseline Comparison**")
    comparison_data = {
        "Model": [
            "BERT-multilingual-uncased",
            "Our Few-Shot LLM"
        ],
        "Accuracy": ["78.15%", "TBD"],
        "Precision": ["93.54%", "TBD"],
        "Recall": ["78.15%", "TBD"],
        "F1-Score": ["73.96%", "TBD"]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, width='stretch', hide_index=True)
    
    st.divider()
    
    with st.expander("Run Experiments", expanded=False):
        if 'full_df' not in st.session_state:
            st.warning("Please load the dataset first using the button below.")
        else:
            if 'api_key' not in st.session_state:
                st.session_state.api_key = None
            if 'api_key_valid' not in st.session_state:
                st.session_state.api_key_valid = False
            if 'model_name' not in st.session_state:
                st.session_state.model_name = "gpt-4.1-nano"
            if 'temperature' not in st.session_state:
                st.session_state.temperature = 0.1
            if 'max_tokens' not in st.session_state:
                st.session_state.max_tokens = 50
            if 'experiment_config' not in st.session_state:
                st.session_state.experiment_config = {}
            
            st.markdown("### API Configuration")
            
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
            
            st.divider()
            
            if st.session_state.api_key:
                st.markdown("### Model Configuration")
                
                model_options = ["gpt-4.1-nano"]
                selected_model = st.selectbox(
                    "Model",
                    options=model_options,
                    index=0,
                    help="Select the OpenAI model to use for classification"
                )
                st.session_state.model_name = selected_model
                
                pricing = get_model_pricing(selected_model)
                st.caption(f"Estimated cost: ${pricing['input']:.2f} per 1M input tokens, ${pricing['output']:.2f} per 1M output tokens")
                
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.temperature,
                    step=0.1,
                    help="Lower values (0.0-0.2) for consistent outputs, higher values (0.7-1.0) for more creative responses"
                )
                st.session_state.temperature = temperature
                
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=10,
                    max_value=200,
                    value=st.session_state.max_tokens,
                    help="Maximum tokens for model response. 50 is sufficient for label names like 'Gayphobia, Transphobia'"
                )
                st.session_state.max_tokens = max_tokens
                
                st.divider()
                
                st.markdown("### Experiment Configuration")
                
                n_few_shot_examples = st.slider(
                    "Few-Shot Examples",
                    min_value=5,
                    max_value=20,
                    value=15,
                    step=1,
                    help="Number of few-shot examples to include in the prompt (5-20 examples)"
                )
                
                st.session_state.experiment_config = {
                    'n_few_shot_examples': n_few_shot_examples
                }
                
                st.divider()
                
                st.markdown("### Cost Estimate")
                
                # Estimate test set size (will be determined when dataset is split)
                estimated_test_size = 300  # Approximate based on 35% split of 862
                
                cost_estimate = estimate_api_cost(
                    model_name=selected_model,
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
                
                st.divider()
                
                st.markdown("### Run Experiment")
                
                can_run = (
                    st.session_state.api_key_valid and
                    st.session_state.api_key and
                    st.session_state.model_name and
                    st.session_state.experiment_config.get('n_few_shot_examples') is not None
                )
                
                if not can_run:
                    st.info("Please validate your API key and configure experiment settings before running.")
                
                run_button = st.button(
                    "Run Experiment",
                    type="primary",
                    disabled=not can_run,
                    help="Start the classification experiment with the configured settings"
                )
                
                if run_button and can_run:
                    if 'test_df' not in st.session_state:
                        with st.spinner("Splitting dataset into train/test sets..."):
                            full_df = st.session_state.full_df.copy()
                            train_df, test_df = split_train_test(full_df, test_size=0.35, random_seed=42)
                            st.session_state.train_df = train_df
                            st.session_state.test_df = test_df
                            st.success(f"Dataset split complete: {len(train_df)} training, {len(test_df)} test.")

                    # Create few-shot pool on-demand
                    n_examples = st.session_state.experiment_config.get('n_few_shot_examples', 15)
                    config_name = f"{n_examples}-shot"
                    
                    with st.spinner(f"Creating {config_name} few-shot pool..."):
                        from few_shot import create_single_few_shot_pool
                        from classification import format_pool_examples
                        
                        pool = create_single_few_shot_pool(st.session_state.train_df, n_examples, random_seed=42)
                        few_shot_examples = format_pool_examples(pool)
                        
                        # Verify no contamination
                        train_ids = set(st.session_state.train_df['id'].values)
                        test_ids = set(st.session_state.test_df['id'].values)
                        pool_ids = {ex['tweet_id'] for ex in pool['examples']}
                        
                        overlap = pool_ids.intersection(test_ids)
                        if overlap:
                            st.warning(f"Warning: {len(overlap)} few-shot examples overlap with test set. This should not happen.")
                        else:
                            st.success(f"Few-shot pool created: {len(pool['examples'])} examples. No contamination detected.")

                    # Use full test set
                    test_df_to_use = st.session_state.test_df.copy()

                    st.markdown("### Experiment Progress")
                    progress_container = st.container()

                    with progress_container:
                        try:
                            run_classification_experiment(
                                test_df=test_df_to_use,
                                few_shot_examples=few_shot_examples,
                                api_key=st.session_state.api_key,
                                model_name=st.session_state.model_name,
                                temperature=st.session_state.temperature,
                                max_tokens=st.session_state.max_tokens,
                                config_name=config_name,
                                progress_container=progress_container
                            )

                            st.success("Experiment completed successfully!")

                            if 'experiment_results' in st.session_state:
                                metadata = st.session_state.experiment_results['metadata']
                                st.markdown("### Experiment Summary")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total API Calls", metadata.get('total_api_calls', 0))
                                with col2:
                                    st.metric("Total Cost", f"${metadata.get('total_cost', 0):.4f}")
                                with col3:
                                    st.metric("Success Rate", f"{(metadata.get('total_api_calls', 0) - len(metadata.get('errors', []))) / max(metadata.get('total_api_calls', 1), 1) * 100:.1f}%")
                                with col4:
                                    st.metric("Errors", len(metadata.get('errors', [])))

                                if metadata.get('errors'):
                                    error_count = len(metadata['errors'])
                                    with st.expander(f"View Errors ({error_count} total)", expanded=False):
                                        if error_count <= 20:
                                            # Show all errors if 20 or fewer
                                            error_df = pd.DataFrame(metadata['errors'])
                                            st.dataframe(error_df, width='stretch', hide_index=True)
                                        else:
                                            # Show summary for many errors
                                            st.warning(f"Total errors: {error_count}")
                                            error_types = {}
                                            for err in metadata['errors']:
                                                err_type = err.get('error', 'Unknown')
                                                error_types[err_type] = error_types.get(err_type, 0) + 1
                                            
                                            st.markdown("**Error Summary:**")
                                            for err_type, count in error_types.items():
                                                st.text(f"  • {err_type}: {count}")
                                            
                                            st.markdown("**First 10 errors:**")
                                            error_df = pd.DataFrame(metadata['errors'][:10])
                                            st.dataframe(error_df, width='stretch', hide_index=True)

                        except Exception as e:
                            st.error(f"Experiment failed with error: {str(e)}")
                            st.exception(e)

    st.divider()

    if 'experiment_results' in st.session_state and st.session_state.get('experiment_complete'):
        st.header("Results & Analysis")
        
        experiment_results = st.session_state.experiment_results
        results = experiment_results.get('results', [])
        config_name = experiment_results.get('config', 'Unknown')
        
        if not results:
            st.info("No results available. Please run an experiment first.")
        else:
            # Compute metrics if not already computed
            if 'computed_metrics' not in st.session_state or st.session_state.get('computed_config') != config_name:
                true_labels, pred_labels = extract_predictions_from_results(results)
                metrics = calculate_metrics(true_labels, pred_labels)
                per_class = calculate_per_class_metrics(true_labels, pred_labels)
                metrics['per_class_metrics'] = per_class
                st.session_state.computed_metrics = metrics
                st.session_state.computed_config = config_name
            
            metrics = st.session_state.computed_metrics
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Metrics & Comparison",
                "Confusion Matrices",
                "Example Predictions",
                "Minority Class Analysis",
                "Cost Summary"
            ])
            
            with tab1:
                st.subheader("Metrics & Comparison")
                
                metrics_table = create_metrics_table(metrics, config_name, BERT_BASELINE)
                st.dataframe(metrics_table, width='stretch', hide_index=True)
                
                f1_macro = metrics.get('f1_macro', 0) * 100
                bert_f1 = BERT_BASELINE['f1_macro'] * 100
                improvement = f1_macro - bert_f1
                
                st.markdown("### Key Findings")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Configuration**: {config_name} achieves {f1_macro:.2f}% F1 (Macro)")
                with col2:
                    if improvement > 0:
                        st.success(f"**Improvement over BERT**: +{improvement:.2f} percentage points")
                    else:
                        st.warning(f"**Below BERT baseline**: {improvement:.2f} percentage points")
            
            with tab2:
                st.subheader("Confusion Matrices")
                
                true_labels, pred_labels = extract_predictions_from_results(results)
                conf_matrices = calculate_confusion_matrices(true_labels, pred_labels)
                
                st.plotly_chart(
                    plot_all_confusion_matrices(conf_matrices, config_name),
                    width='stretch'
                )
                
                cooccurrence = analyze_label_cooccurrence(true_labels, pred_labels)
                st.markdown("### Label Co-occurrence Analysis")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Exact Matches", cooccurrence['exact_matches'])
                with col2:
                    st.metric("Correct Multi-Label", cooccurrence['correct_multi_label'])
                with col3:
                    st.metric("Missed Labels", cooccurrence['missed_labels'])
                with col4:
                    st.metric("Extra Labels", cooccurrence['extra_labels'])
            
            with tab3:
                st.subheader("Example Predictions")
                
                filter_type = st.radio(
                    "Filter Examples",
                    options=["All", "Correct", "Incorrect", "Challenging"],
                    horizontal=True
                )
                
                search_term = st.text_input("Search by tweet ID or keyword", "")
                
                filtered_results = []
                for result in results:
                    if not result.get('is_valid', True):
                        continue
                    
                    true_vec = result['true_labels']
                    pred_vec = result['pred_labels']
                    
                    is_correct = np.array_equal(true_vec, pred_vec)
                    is_multi_label = sum(true_vec) > 1
                    is_biphobia = true_vec[2] == 1 if len(true_vec) > 2 else False
                    is_challenging = is_multi_label or is_biphobia
                    
                    if search_term:
                        if search_term.lower() not in str(result['tweet_id']).lower() and search_term.lower() not in result['tweet_text'].lower():
                            continue
                    
                    if filter_type == "Correct" and not is_correct:
                        continue
                    if filter_type == "Incorrect" and is_correct:
                        continue
                    if filter_type == "Challenging" and not is_challenging:
                        continue
                    
                    filtered_results.append(result)
                
                if not filtered_results:
                    st.info(f"No examples found with current filters.")
                else:
                    display_count = min(5, len(filtered_results))
                    for i, result in enumerate(filtered_results[:display_count]):
                        with st.expander(f"Tweet {result['tweet_id']}: {result['true_labels_str']} → {result['pred_labels_str']}"):
                            st.write(f"**Tweet Text**: {result['tweet_text']}")
                            st.write(f"**True Labels**: {result['true_labels_str']}")
                            st.write(f"**Predicted Labels**: {result['pred_labels_str']}")
                            if result.get('raw_response'):
                                st.write(f"**Raw LLM Response**: {result['raw_response']}")
            
            with tab4:
                st.subheader("Minority Class Analysis")
                
                per_class = metrics.get('per_class_metrics', {})
                per_class_data = []
                
                for label in CLASS_LABELS:
                    if label in per_class:
                        class_metrics = per_class[label]
                        bert_metrics = BERT_BASELINE['per_class'].get(label, {})
                        bert_f1 = bert_metrics.get('f1', 0)
                        f1 = class_metrics.get('f1', 0)
                        improvement = (f1 - bert_f1) * 100
                        
                        per_class_data.append({
                            'Class': CLASS_NAMES[CLASS_LABELS.index(label)],
                            'Precision': f"{class_metrics.get('precision', 0) * 100:.2f}%",
                            'Recall': f"{class_metrics.get('recall', 0) * 100:.2f}%",
                            'F1': f"{f1 * 100:.2f}%",
                            'BERT F1': f"{bert_f1 * 100:.2f}%",
                            'Δ Improvement': f"{improvement:+.2f}%"
                        })
                
                if per_class_data:
                    per_class_df = pd.DataFrame(per_class_data)
                    st.dataframe(per_class_df, width='stretch', hide_index=True)
                    
                    st.markdown("### Minority Class Focus: Biphobia, Lesbophobia, Transphobia")
                    
                    minority_classes = ['B', 'L', 'T']
                    for label in minority_classes:
                        class_name = CLASS_NAMES[CLASS_LABELS.index(label)]
                        st.markdown(f"#### {class_name}")
                        
                        class_row = [r for r in per_class_data if r['Class'] == class_name]
                        if class_row:
                            class_df = pd.DataFrame(class_row)
                            st.dataframe(class_df, width='stretch', hide_index=True)
                            
                            class_examples = [r for r in results if r['true_labels'][CLASS_LABELS.index(label)] == 1]
                            
                            if class_examples:
                                st.markdown(f"**All {class_name} examples in test set ({len(class_examples)}):**")
                                for example in class_examples:
                                    with st.expander(f"Tweet {example['tweet_id']}: True={example['true_labels_str']}, Pred={example['pred_labels_str']}"):
                                        st.write(example['tweet_text'])
            
            with tab5:
                st.subheader("Cost Summary")
                
                metadata = experiment_results.get('metadata', {})
                total_cost = metadata.get('total_cost', 0)
                
                st.markdown("### Cost Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Cost", f"${total_cost:.4f}")
                    st.metric("Total API Calls", metadata.get('total_api_calls', 0))
                with col2:
                    st.metric("Total Input Tokens", f"{metadata.get('total_input_tokens', 0):,}")
                    st.metric("Total Output Tokens", f"{metadata.get('total_output_tokens', 0):,}")
    
    st.divider()
    
    st.header("Data Loading & Few-Shot Pool Preparation")
    
    st.markdown(
        """
        Load the HOMO-MEX training dataset. The dataset will be split into train/test sets 
        when you run an experiment, and few-shot examples will be created on-demand based on your configuration.
        """
    )
    
    if st.button("Load Dataset & Create Few-Shot Pools", type="primary"):
        with st.spinner("Loading dataset and creating few-shot pools..."):
            try:
                full_df = load_homo_mex_dataset()
                full_df = prepare_label_vectors(full_df)
                validation = validate_dataset(full_df)

                st.session_state.full_df = full_df
                st.session_state.label_counts = validation['stats']['label_counts']
                st.session_state.validation_results = validation

                st.success("Dataset loaded successfully! The dataset will be split into train/test when you run an experiment.")
                
            except FileNotFoundError as e:
                st.error(f"Error: {str(e)}")
                st.info("Please ensure the Excel file is located at: `data/Annotated LGBTQ+ Phobia Tweets.xlsx`")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if 'full_df' in st.session_state or 'train_df' in st.session_state:
        if 'train_df' in st.session_state:
            display_df = st.session_state.train_df
        else:
            display_df = st.session_state.full_df
        st.markdown("### Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tweets", len(display_df))
        with col2:
            st.metric("Multi-Label Examples", 
                     st.session_state.validation_results['stats']['multi_label_count'])
        with col3:
            st.metric("Classes", 5)
        with col4:
            warnings = len(st.session_state.validation_results.get('warnings', []))
            if warnings == 0:
                st.metric("Validation", "Passed")
            else:
                st.metric("Validation", f"{warnings} warnings")
        
        if 'test_df' in st.session_state:
            st.info(f"Dataset split: {len(st.session_state.train_df)} training tweets, {len(st.session_state.test_df)} test tweets")
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


if __name__ == "__main__":
    main()
