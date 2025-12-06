# Multi-Label LGBT+Phobia Detection in Mexican Spanish

A Streamlit web application for detecting LGBT+phobic content in Mexican Spanish tweets using few-shot learning with GPT-4, extending the HOMO-MEX corpus.

## Features

- **Few-Shot Classification**: Uses 5-20 examples per class for multi-label classification
- **Multi-Label Detection**: Classifies into Gayphobia, Lesbophobia, Biphobia, Transphobia, Other
- **Interactive Testing**: Classify individual tweets or batches without running full experiments
- **Comprehensive Analysis**: Confusion matrices, per-class metrics, example predictions, minority class analysis
- **Cost Tracking**: Real-time API cost monitoring and cost per tweet calculations
- **BERT Comparison**: Direct comparison to published baseline results from HOMO-MEX paper
- **Stratified Sampling**: Ensures balanced few-shot examples across all classes

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add dataset**: Place the HOMO-MEX fine-grained classification dataset at:
   ```
   data/Annotated LGBTQ+ Phobia Tweets.xlsx
   ```

3. **Run app:**
   ```bash
   streamlit run app.py
   ```

4. **Usage flow:**
   - Load dataset → Enter API key → Run experiment (or test individual tweets)
   - View results in comprehensive analysis tabs

## Data Requirements

The app expects the HOMO-MEX fine-grained classification dataset:

- **Format**: Excel file with columns: `id`, `tweet_text`, `G`, `L`, `B`, `T`, `O`
- **Available data**: 862 training tweets (public release only includes training data)
- **Test set**: Created by splitting available data (35% test, ~302 tweets; 65% train, ~560 tweets)
- **5 multi-label classes**: 
  - G: Gayphobia (714 examples)
  - L: Lesbophobia (72 examples)
  - B: Biphobia (10 examples - extreme minority class)
  - T: Transphobia (79 examples)
  - O: Other (64 examples)

## Methodology

### Few-Shot Learning Approach

- **Single Multi-Label Prompt**: One API call per tweet captures all applicable labels simultaneously
- **Stratified Sampling**: Ensures minimum 2-3 examples per class, even for rare classes like Biphobia
- **English Instructions, Spanish Content**: Prompts use English for clarity, tweets remain in Mexican Spanish
- **Temperature**: 0.1 for consistent predictions

### Research Questions

1. **Sample Efficiency**: Can 15 few-shot examples compete with 862-example BERT training?
2. **Minority Class Performance**: Does few-shot learning handle low-resource classes (Biphobia, Lesbophobia) better than fine-tuning?
3. **Cost-Benefit Tradeoff**: How do API costs compare to GPU training time and compute?
4. **Cross-Class Generalization**: Can the model learn shared patterns across LGBT+phobia types?

## Model Configuration

- **Supported Models**: GPT-4-turbo, GPT-4, GPT-3.5-turbo, GPT-4.1-nano
- **Few-Shot Examples**: Configurable (5-20 examples)
- **Temperature**: 0.1 (recommended for consistency)
- **Max Tokens**: 50 (sufficient for label names)

## Evaluation Metrics

- **Per-Class F1 Score**: Primary metric for each category (G/L/B/T/O)
- **Macro-Average F1**: Overall performance across all classes
- **Micro-Average F1**: Weighted by class frequency
- **Confusion Matrices**: Per-class binary classification matrices
- **Label Co-occurrence Analysis**: Multi-label prediction accuracy

## Baseline Comparison

The app compares results against published BERT baseline from Vásquez et al. (2023):

- **BERT F1-Score**: 73.96% (macro-average)
- **BERT Accuracy**: 78.15%
- **BERT Precision**: 93.54%
- **BERT Recall**: 78.15%

## Cost Considerations

- **Single Classification**: ~$0.001 per tweet
- **Test Set (~302 tweets)**: ~$0.30-$1.20 depending on model
- **Model Choice**: 
  - GPT-3.5-turbo: Most cost-effective
  - GPT-4-turbo: Better accuracy, higher cost
  - GPT-4.1-nano: Experimental, lower cost

## Citation

**Vásquez, J., Andersen, S. T., Bel-Enguix, G., Gómez-Adorno, H., & Ojeda-Trueba, S.-L.** (2023). 
Experiments on the HOMO-MEX Corpus for LGBT+phobia Detection. 

In *Proceedings of the 7th Workshop on Online Abuse and Harms (WOAH 2023)*, pages 200-210. 
Association for Computational Linguistics.

**Repository**: [HOMO-MEX on GitHub](https://github.com/juanmvsa/HOMO-MEX)

**Paper**: [ACL Anthology](https://aclanthology.org/2023.woah-1.20.pdf)

## Project Structure

```
lin313_classifier_project/
├── app.py                 # Main Streamlit application
├── classification.py      # Classification pipeline and API calls
├── few_shot.py           # Few-shot example pool creation
├── data_loading.py       # Dataset loading and validation
├── metrics.py            # Metrics calculation and BERT baselines
├── visualizations.py     # Plotly visualizations
├── api_utils.py          # API key validation and cost estimation
├── constants.py          # Class names and labels
└── data/
    └── Annotated LGBTQ+ Phobia Tweets.xlsx
```

## License

This project extends the HOMO-MEX corpus. Please refer to the original repository for dataset licensing information.
