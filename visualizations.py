import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict
from constants import CLASS_LABELS, CLASS_NAMES
from metrics import BERT_BASELINE


def plot_f1_comparison(metrics_by_config: Dict[str, Dict], bert_baseline: float) -> go.Figure:
    """
    Line chart: F1-score vs Few-Shot Examples (5, 10, 15, 20).
    
    Args:
        metrics_by_config: Dict with config names as keys, metrics dicts as values
        bert_baseline: BERT baseline F1 score
        
    Returns:
        plotly.graph_objects.Figure
    """
    config_order = ['5_shot', '10_shot', '15_shot', '20_shot']
    x_positions = [5, 10, 15, 20]
    
    fig = go.Figure()
    
    overall_f1 = []
    for config in config_order:
        if config in metrics_by_config:
            overall_f1.append(metrics_by_config[config].get('f1_macro', 0) * 100)
        else:
            overall_f1.append(None)
    
    fig.add_trace(go.Scatter(
        x=x_positions,
        y=overall_f1,
        mode='lines+markers',
        name='Overall F1 (Macro)',
        line=dict(width=3, color='#1f77b4'),
        marker=dict(size=10)
    ))
    
    for idx, label in enumerate(CLASS_LABELS):
        class_f1 = []
        for config in config_order:
            if config in metrics_by_config:
                per_class = metrics_by_config[config].get('per_class_metrics', {})
                if label in per_class:
                    class_f1.append(per_class[label].get('f1', 0) * 100)
                else:
                    class_f1.append(None)
            else:
                class_f1.append(None)
        
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=class_f1,
            mode='lines+markers',
            name=f'{CLASS_NAMES[idx]} (F1)',
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    fig.add_hline(
        y=bert_baseline * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"BERT Baseline ({bert_baseline * 100:.2f}%)",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="F1-Score vs Few-Shot Examples",
        xaxis_title="Number of Few-Shot Examples",
        yaxis_title="F1-Score (%)",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )
    
    return fig


def plot_per_class_performance(metrics_by_config: Dict[str, Dict], bert_baselines: Dict) -> go.Figure:
    """
    Grouped bar chart: Per-class F1 scores.
    
    Args:
        metrics_by_config: Dict with config names as keys, metrics dicts as values
        bert_baselines: Dict with per-class BERT baseline F1 scores
        
    Returns:
        plotly.graph_objects.Figure
    """
    config_order = ['5_shot', '10_shot', '15_shot', '20_shot']
    config_labels = ['5-shot', '10-shot', '15-shot', '20-shot']
    
    fig = go.Figure()
    
    for config_idx, config in enumerate(config_order):
        if config not in metrics_by_config:
            continue
            
        per_class = metrics_by_config[config].get('per_class_metrics', {})
        f1_scores = []
        colors = []
        
        for label in CLASS_LABELS:
            if label in per_class:
                f1 = per_class[label].get('f1', 0) * 100
                f1_scores.append(f1)
                bert_f1 = bert_baselines.get(label, {}).get('f1', 0) * 100
                colors.append('green' if f1 >= bert_f1 else 'red')
            else:
                f1_scores.append(0)
                colors.append('gray')
        
        fig.add_trace(go.Bar(
            name=config_labels[config_idx],
            x=[CLASS_NAMES[i] for i in range(len(CLASS_LABELS))],
            y=f1_scores,
            marker_color=colors,
            opacity=0.7
        ))
    
    for idx, label in enumerate(CLASS_LABELS):
        bert_f1 = bert_baselines.get(label, {}).get('f1', 0) * 100
        fig.add_hline(
            y=bert_f1,
            line_dash="dot",
            line_color="black",
            line_width=1,
            annotation_text=f"BERT {label}",
            annotation_position="right",
            annotation_font_size=8
        )
    
    fig.update_layout(
        title="Per-Class F1 Performance by Configuration",
        xaxis_title="Class",
        yaxis_title="F1-Score (%)",
        barmode='group',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )
    
    return fig


def create_metrics_table(metrics: Dict, config_name: str, bert_baseline: Dict) -> pd.DataFrame:
    """
    Create comparison table with metrics for single config vs BERT baseline.
    
    Args:
        metrics: Single metrics dict
        config_name: Configuration name (e.g., '15-shot')
        bert_baseline: BERT baseline metrics dict
        
    Returns:
        pd.DataFrame with formatted metrics (2 rows: user config + BERT)
    """
    data = []
    
    # Add user's config row
    data.append({
        'Config': config_name,
        'Accuracy': f"{metrics.get('accuracy', 0) * 100:.2f}%",
        'Precision (Macro)': f"{metrics.get('precision_macro', 0) * 100:.2f}%",
        'Recall (Macro)': f"{metrics.get('recall_macro', 0) * 100:.2f}%",
        'F1 (Macro)': f"{metrics.get('f1_macro', 0) * 100:.2f}%",
        'F1 (Micro)': f"{metrics.get('f1_micro', 0) * 100:.2f}%"
    })
    
    # Add BERT baseline row
    data.append({
        'Config': 'BERT',
        'Accuracy': f"{bert_baseline.get('accuracy', 0) * 100:.2f}%",
        'Precision (Macro)': f"{bert_baseline.get('precision_macro', 0) * 100:.2f}%",
        'Recall (Macro)': f"{bert_baseline.get('recall_macro', 0) * 100:.2f}%",
        'F1 (Macro)': f"{bert_baseline.get('f1_macro', 0) * 100:.2f}%",
        'F1 (Micro)': '---'
    })
    
    return pd.DataFrame(data)


def plot_confusion_matrix_heatmap(conf_matrix: np.ndarray, class_name: str, config_name: str) -> go.Figure:
    """
    Create heatmap for single class confusion matrix.
    
    Args:
        conf_matrix: 2x2 confusion matrix
        class_name: Name of the class (e.g., 'G')
        config_name: Configuration name (e.g., '15-shot')
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted: Not ' + class_name, 'Predicted: ' + class_name],
        y=['Actual: Not ' + class_name, 'Actual: ' + class_name],
        colorscale='Blues',
        text=conf_matrix.astype(int),
        texttemplate='%{text}',
        textfont={"size": 14},
        showscale=True
    ))
    
    fig.update_layout(
        title=f"{class_name} - {config_name}",
        height=300,
        width=400
    )
    
    return fig


def plot_all_confusion_matrices(conf_matrices: Dict[str, np.ndarray], config_name: str) -> go.Figure:
    """
    Create subplot with 5 heatmaps side-by-side (one per class).
    
    Args:
        conf_matrices: Dict with class labels as keys, confusion matrices as values
        config_name: Configuration name (e.g., '15-shot')
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=1,
        cols=5,
        subplot_titles=[CLASS_NAMES[i] for i in range(len(CLASS_LABELS))],
        horizontal_spacing=0.1
    )
    
    for idx, label in enumerate(CLASS_LABELS):
        conf_matrix = conf_matrices[label]
        
        fig.add_trace(
            go.Heatmap(
                z=conf_matrix,
                x=['Pred: ¬' + label, 'Pred: ' + label],
                y=['Act: ¬' + label, 'Act: ' + label],
                colorscale='Blues',
                text=conf_matrix.astype(int),
                texttemplate='%{text}',
                textfont={"size": 12},
                showscale=(idx == 0)
            ),
            row=1,
            col=idx + 1
        )
    
    fig.update_layout(
        title=f"Confusion Matrices for {config_name}",
        height=400,
        width=2000
    )
    
    return fig

