# Implementation Plan: Interactive Testing & Analysis

## Overview
Transform the Streamlit app from informational to interactive, allowing users to run experiments, compare configurations, and analyze results.

---

## Phase 1: Data Loading & Preparation

### 1.1 Load HOMO-MEX Dataset
**Goal**: Load and validate the fine-grained classification data

**Steps**:
1. Read the CSV/Excel file with tweet IDs and labels (G, L, B, T, O)
2. Read the file with actual tweet text
3. Cross-reference and merge on tweet ID
4. Validate that we have 862 training tweets and 477 test tweets
5. Convert multi-label format to binary vectors [G, L, B, T, O]

**Data Structure**:
```
Tweet ID | Tweet Text | G | L | B | T | O
123      | "text..."  | 1 | 0 | 0 | 1 | 0
```

**Validation Checks**:
- Ensure no missing tweet texts
- Verify label counts match paper (714 G, 72 L, 10 B, 79 T, 64 O)
- Check for multi-label examples (tweets with 2+ labels)

### 1.2 Create Few-Shot Example Pools
**Goal**: Prepare stratified samples for different ablation configurations

**Sampling Strategy**:
- **5 examples**: 1 per class (balanced minimum)
- **10 examples**: 2 per class (baseline)
- **15 examples**: 3 per class (recommended)
- **20 examples**: 4 per class (many-shot)

**Stratified Sampling Rules**:
1. For each configuration, ensure every class (G/L/B/T/O) is represented
2. Priority: Select diverse examples (different tweet lengths, styles)
3. For Biphobia (only 10 total): Never use more than 3 in few-shot to preserve test integrity
4. Track which examples are used to avoid test set contamination

**Storage**:
- Create 4 pre-selected few-shot sets (one for each ablation size)
- Store as JSON or in-memory dictionaries
- Include metadata: tweet ID, text, label vector, class distribution

---

## Phase 2: API Key & Model Configuration

### 2.1 User Input Section
**Location**: After "Evaluation Strategy" section, before experiments

**Components**:
1. **API Key Input**
   - Secure text input (password field)
   - Validation: Test with a simple API call
   - Store in session state (not persistent)
   - Show connection status (✓ Connected / ✗ Invalid)

2. **Model Selection**
   - Dropdown: GPT-4-turbo, GPT-4, GPT-3.5-turbo
   - Show estimated costs per configuration
   - Default: GPT-4-turbo (balance of performance/cost)

3. **Temperature Setting**
   - Slider: 0.0 - 1.0
   - Default: 0.1 (for consistency)
   - Explanation tooltip

4. **Max Tokens**
   - Input field: Default 50
   - Explanation: Enough for "Gayphobia, Transphobia, Other"

### 2.2 Experiment Configuration
**Goal**: Let users choose what to run

**Options**:
- [ ] Run full ablation study (5, 10, 15, 20 examples)
- [ ] Run single configuration (choose number of examples)
- [ ] Run on full test set (477 tweets) or sample (50, 100 tweets)

**Cost Estimator**:
- Show projected API costs before running
- Formula: (num_test_tweets × num_ablations × avg_tokens × model_price)
- Warning if cost > $10

---

## Phase 3: Classification Pipeline

### 3.1 Prompt Construction
**For each ablation configuration**:

1. **System Message**:
   ```
   You are a classifier for LGBT+phobic content in Mexican Spanish tweets.
   You will classify tweets into one or more categories.
   ```

2. **Few-Shot Examples**:
   ```
   Here are some examples:
   
   Tweet: "{example_1_text}"
   Labels: {example_1_labels}
   
   Tweet: "{example_2_text}"
   Labels: {example_2_labels}
   
   [... repeat for all few-shot examples ...]
   ```

3. **Classification Instructions**:
   ```
   Now classify this tweet into one or more of these categories:
   - Gayphobia
   - Lesbophobia
   - Biphobia
   - Transphobia
   - Other
   
   Tweet: "{target_tweet}"
   
   Answer with only the applicable label names, separated by commas.
   If none apply, answer "None".
   ```

**Output Parsing**:
- Extract label names from response
- Convert to binary vector: [G, L, B, T, O]
- Handle variations: "gayphobia" vs "Gayphobia" vs "gay-phobia"
- Handle ambiguous outputs: log for review

### 3.2 Batch Processing
**Goal**: Process test set efficiently with progress tracking

**Process**:
1. Initialize results storage (DataFrame)
2. For each ablation configuration (5, 10, 15, 20):
   a. Build prompt with N few-shot examples
   b. For each test tweet:
      - Send API request
      - Parse response
      - Store prediction vector
      - Handle errors (retry logic)
      - Update progress bar
   c. Save intermediate results (in case of failure)
3. Compile all results into comparison DataFrame

**Progress Indicators**:
- Overall progress: "Ablation 2/4: 15 examples"
- Sub-progress: "Processing tweet 123/477"
- Estimated time remaining
- Current API call count & cost

**Error Handling**:
- Rate limiting: exponential backoff
- API errors: retry 3 times, then skip and log
- Parsing errors: flag tweet for manual review
- Save partial results every 50 tweets

---

## Phase 4: Results Analysis & Visualization

### 4.1 Ablation Study Comparison
**Goal**: Show how performance changes with few-shot example count

**Visualizations**:

1. **Line Chart: F1-Score vs Few-Shot Examples**
   - X-axis: 5, 10, 15, 20 examples
   - Y-axis: F1-score
   - Multiple lines: Overall F1, per-class F1 (G, L, B, T, O)
   - Horizontal reference line: BERT baseline (73.96%)
   - Annotations: Mark optimal configuration

2. **Bar Chart: Per-Class Performance**
   - Grouped bars for each class
   - Groups: 5, 10, 15, 20 examples
   - Color-coded: Green if beats BERT, Red if below
   - Show BERT baseline as dotted line

3. **Table: Detailed Metrics**
   ```
   | Config | Accuracy | Precision | Recall | F1 (Macro) | F1 (Micro) |
   |--------|----------|-----------|--------|------------|------------|
   | 5-shot | XX%      | XX%       | XX%    | XX%        | XX%        |
   | 10-shot| XX%      | XX%       | XX%    | XX%        | XX%        |
   | 15-shot| XX%      | XX%       | XX%    | XX%        | XX%        |
   | 20-shot| XX%      | XX%       | XX%    | XX%        | XX%        |
   | BERT   | 78.15%   | 93.54%    | 78.15% | 73.96%     | ---        |
   ```

**Key Findings Section**:
- Identify optimal configuration
- Highlight minority class improvements
- Note diminishing returns (if 20-shot ≈ 15-shot)

### 4.2 Confusion Matrix (Per Configuration)
**Goal**: Show which classes get confused for each ablation setting

**Implementation**:

1. **Multi-Label Confusion Strategy**:
   Since this is multi-label, create **5 separate binary confusion matrices** (one per class):
   
   ```
   Class: Gayphobia
   
                Predicted: G  Predicted: ¬G
   Actual: G    [True Pos]    [False Neg]
   Actual: ¬G   [False Pos]   [True Neg]
   ```

2. **Heatmap Visualization**:
   - 5 small heatmaps (G, L, B, T, O) side-by-side
   - For each ablation configuration (tabs or dropdown selector)
   - Color scale: intensity = count
   - Annotations: numbers in each cell

3. **Co-occurrence Confusion Matrix**:
   Show which label pairs are confused:
   ```
   When true label is [G, T], model predicts:
   - [G, T] correctly: 45 times
   - [G] only: 12 times (missed T)
   - [G, T, O]: 3 times (extra O)
   ```

**Insights to Extract**:
- Most confused class pairs
- Systematic over-prediction (false positives)
- Systematic under-prediction (false negatives)
- How confusion changes with more examples

### 4.3 Example Predictions Showcase
**Goal**: Show actual predictions with explanations

**Structure**:

1. **Correct Predictions**:
   - Show 3-5 examples where model got it right
   - Display: Tweet text, True labels, Predicted labels, Confidence
   - Highlight: Well-handled edge cases (multi-label, subtle phobia)

2. **Incorrect Predictions**:
   - Show 3-5 examples where model failed
   - Display: Tweet text, True labels, Predicted labels, Error type
   - Categories:
     - False Negative (missed a label)
     - False Positive (added wrong label)
     - Complete Miss (predicted None when should be labeled)
   - Include hypothesis for why it failed

3. **Challenging Cases**:
   - Tweets with 3+ labels (multi-label complexity)
   - Biphobia examples (rare class)
   - Irony/sarcasm within LGBT+ community
   - Ambiguous tweets with low annotator agreement

**Interactive Elements**:
- Dropdown: Select ablation configuration to see its predictions
- Filter: Show only correct / only incorrect
- Search: Find specific tweet by ID or keyword

### 4.4 Minority Class Deep Dive
**Goal**: Specifically analyze Biphobia, Lesbophobia, Transphobia

**Dedicated Section**:

1. **Per-Class Performance Table**:
   ```
   | Class | Config | Precision | Recall | F1 | BERT F1 | Δ Improvement |
   |-------|--------|-----------|--------|----|---------|-   --------------|
   | B     | 5-shot | XX%       | XX%    | XX%| 42%     | +XX%          |
   | B     | 10-shot| XX%       | XX%    | XX%| 42%     | +XX%          |
   | B     | 15-shot| XX%       | XX%    | XX%| 42%     | +XX%          |
   | B     | 20-shot| XX%       | XX%    | XX%| 42%     | +XX%          |
   ```

2. **Minority Class Findings**:
   - Does few-shot learning help with Biphobia (10 train examples)?
   - At what point does more examples stop helping?
   - Are minority classes improved more than majority (Gayphobia)?

3. **Error Analysis**:
   - For Biphobia: Show all 3 test examples and predictions
   - Identify patterns in misclassifications
   - Compare to BERT's known weaknesses

---

## Phase 5: Cost-Benefit Analysis

### 5.1 Cost Tracking
**Real-Time During Experiment**:
- Token count per API call
- Running total cost
- Display: "Current run: $X.XX"

**Post-Experiment Summary**:
```
Total API Costs:
- 5-shot:  $X.XX (477 tweets)
- 10-shot: $X.XX (477 tweets)
- 15-shot: $X.XX (477 tweets)
- 20-shot: $X.XX (477 tweets)
Total: $XX.XX
```

### 5.2 Cost-Benefit Comparison
**Visualization**: 2D scatter plot

- X-axis: Cost (log scale if needed)
- Y-axis: F1-Score
- Points: Each ablation configuration + BERT
- Size: Number of training examples needed
- Color: Few-shot (blue) vs BERT (red)

**Pareto Frontier Analysis**:
- Identify dominant configurations (best F1 for cost)
- Show trade-off curve
- Recommendation: Optimal point balancing performance & cost

### 5.3 Deployment Comparison Table
```
| Aspect | Few-Shot (15 examples) | BERT Fine-Tuning |
|--------|------------------------|------------------|
| Training Time | 5 minutes (manual selection) | 2-4 hours (GPU) |
| Training Cost | $0 | $5-20 (compute) |
| Setup Complexity | Low (select examples) | High (training pipeline) |
| Inference Cost | $X per 1000 tweets | $0 (self-hosted) |
| Update Flexibility | Immediate (change examples) | Requires retraining |
| Interpretability | High (see examples) | Low (black box) |
| Best Use Case | Rapid prototyping, low volume | Production, high volume |
```

---

## Phase 6: UI/UX Flow

### 6.1 Section Organization
**Revised App Structure**:

1. **Introduction** (current)
2. **The Challenge** (current)
3. **Our Approach** (current)
4. **Methodology** (current)
5. **Implementation Details** (current)
6. **Dataset Overview** (current tabs)
7. **Evaluation Strategy** (current)

**NEW SECTIONS:**

8. **🔬 Run Experiments** ← NEW
   - API Configuration
   - Experiment Settings
   - Cost Estimate
   - "Run Ablation Study" button

9. **📊 Results & Analysis** ← NEW (appears after running)
   - Ablation Study Comparison
   - Confusion Matrices
   - Example Predictions
   - Minority Class Analysis
   - Cost-Benefit Analysis

10. **💾 Export Results** ← NEW
    - Download predictions CSV
    - Download full analysis report (PDF/Markdown)
    - Download confusion matrices (images)

### 6.2 Interactive Workflow

**State Management**:
- Use `st.session_state` to track:
  - API key validation status
  - Loaded datasets
  - Experiment configuration
  - Results from each ablation run
  - User selections (which config to view)

**Progressive Disclosure**:
1. User sees intro/methodology first
2. "Run Experiments" section collapsed by default
3. After API key entered → Show configuration options
4. After "Run" clicked → Show progress bar
5. After completion → Auto-expand "Results & Analysis"
6. Results section has tabs for different analysis views

**Error States**:
- No API key → Disable run button, show instructions
- Invalid API key → Show error, allow re-entry
- API failure during run → Show progress up to failure, option to resume
- No results yet → Results section hidden or shows placeholder

### 6.3 Comparison Selector
**For all result visualizations**:

Add dropdown/radio buttons:
- "View configuration: [5-shot | 10-shot | 15-shot | 20-shot | All]"
- When "All" selected → Show comparative charts
- When specific selected → Show detailed analysis for that config

**Example Flow**:
1. User runs all 4 ablations
2. "Results & Analysis" section appears
3. Default view: "All" configurations compared
4. User selects "15-shot" from dropdown
5. Page updates to show:
   - Detailed confusion matrix for 15-shot
   - Example predictions from 15-shot
   - Minority class breakdown for 15-shot

---

## Phase 7: Validation & Quality Control

### 7.1 Sanity Checks
**Before displaying results**:
- Verify predictions sum correctly (no impossible label counts)
- Check for parsing failures (unparseable LLM outputs)
- Identify outlier results (e.g., 0% or 100% on any class)
- Flag if results are suspiciously different from BERT

### 7.2 Reproducibility
**Save experiment configuration**:
- Few-shot examples used (by ID)
- Model settings (temperature, max_tokens)
- Timestamp
- Total cost
- Random seed (if applicable)

**Export Options**:
- JSON file with full configuration
- Allows users to reproduce exact results
- Can be uploaded to recreate experiment

### 7.3 Statistical Significance
**Optional Enhancement**:
- Bootstrap confidence intervals for F1 scores
- Statistical test: Is improvement over BERT significant?
- Note: With 477 test tweets, can compute meaningful intervals

---

## Phase 8: Documentation & Help

### 8.1 Tooltips & Help Text
**Throughout app**:
- Hover tooltips on metrics (what is F1-score?)
- Expandable "What does this mean?" sections
- Links to paper for methodology details

### 8.2 Interpretation Guide
**Section after results**:

"**How to Read These Results**"
- Explain multi-label confusion matrices
- How to identify which configuration is best
- What to do if all configs fail (may need more examples, better prompt)
- Limitations of few-shot approach

### 8.3 Troubleshooting
**Common issues**:
- API rate limits → Solution: Use smaller test sample
- Inconsistent parsing → Solution: Check LLM outputs in raw view
- All predictions are "None" → Check prompt formatting
- Poor performance on all configs → May need domain-specific model

---

## Summary: User Journey

### Happy Path:
1. User reads introduction and methodology ✓
2. User enters OpenAI API key → Validated ✓
3. User selects "Run full ablation study" on 100 test tweets (for speed)
4. User clicks "Run Experiment" → Progress bar appears
5. After 5-10 minutes → Results populate
6. User sees:
   - 15-shot performs best (75% F1, beats BERT!)
   - Biphobia improved from 42% → 68% F1
   - Cost: $2.50 total
7. User explores confusion matrix → Sees Transphobia confused with Other
8. User views example predictions → Understands failure modes
9. User downloads CSV of predictions for further analysis
10. User reads conclusion: Few-shot is viable for rapid deployment!

### Key Deliverables:
- Interactive app where users can run experiments themselves
- Comprehensive ablation study (5, 10, 15, 20 examples)
- Visual comparison to BERT baseline
- Confusion matrices for each configuration
- Concrete example predictions (correct and incorrect)
- Cost-benefit analysis
- Exportable results

**Timeline Estimate**: 2-3 weeks of implementation

Ready to start coding? Which phase should we tackle first?