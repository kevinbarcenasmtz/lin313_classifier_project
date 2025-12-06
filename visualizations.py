import pandas as pd
import numpy as np
import plotly.graph_objects as go
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


def plot_metrics_radar(metrics: Dict, bert_baseline: Dict, config_name: str = "Current Config") -> go.Figure:
    """
    Radar chart: Compare overall metrics between config and BERT baseline.
    
    Args:
        metrics: Single metrics dict
        bert_baseline: BERT baseline metrics dict
        config_name: Name of the current configuration
        
    Returns:
        plotly.graph_objects.Figure
    """
    categories = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1\n(Macro)', 'F1\n(Micro)']
    
    config_values = [
        metrics.get('accuracy', 0) * 100,
        metrics.get('precision_macro', 0) * 100,
        metrics.get('recall_macro', 0) * 100,
        metrics.get('f1_macro', 0) * 100,
        metrics.get('f1_micro', 0) * 100
    ]
    
    bert_values = [
        bert_baseline.get('accuracy', 0) * 100,
        bert_baseline.get('precision_macro', 0) * 100,
        bert_baseline.get('recall_macro', 0) * 100,
        bert_baseline.get('f1_macro', 0) * 100,
        0  # BERT doesn't have F1 micro
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=config_values + [config_values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name=config_name,
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=bert_values + [bert_values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name='BERT Baseline',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickmode='linear',
                tick0=0,
                dtick=20
            )
        ),
        showlegend=True,
        title=f"Metrics Comparison: {config_name} vs BERT Baseline",
        height=500
    )
    
    return fig


def plot_per_class_performance(metrics: Dict, bert_baselines: Dict, config_name: str = "Current Config") -> go.Figure:
    """
    Bar chart: Per-class F1 scores for single config vs BERT baseline.
    
    Args:
        metrics: Single metrics dict with per_class_metrics
        bert_baselines: Dict with per-class BERT baseline F1 scores
        config_name: Name of the current configuration
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    per_class = metrics.get('per_class_metrics', {})
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
        name=config_name,
        x=[CLASS_NAMES[i] for i in range(len(CLASS_LABELS))],
        y=f1_scores,
        marker_color=colors,
        opacity=0.7
    ))
    
    # Add BERT baseline lines
    for idx, label in enumerate(CLASS_LABELS):
        bert_f1 = bert_baselines.get(label, {}).get('f1', 0) * 100
        fig.add_hline(
            y=bert_f1,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"BERT {label} ({bert_f1:.1f}%)",
            annotation_position="right",
            annotation_font_size=9,
            opacity=0.7
        )
    
    fig.update_layout(
        title=f"Per-Class F1 Performance: {config_name} vs BERT Baseline",
        xaxis_title="Class",
        yaxis_title="F1-Score (%)",
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


def plot_confusion_matrix_heatmap(
    conf_matrix: np.ndarray, 
    class_name: str, 
    config_name: str,
    colorscale: str = 'Blues',
    text_color: str = 'white',
    annotation_text_color: str = 'white',
    annotation_bg_color: str = 'rgba(0,0,0,0.3)',
    annotation_border_color: str = 'white'
) -> go.Figure:
    """
    Create heatmap for single class confusion matrix.
    
    Args:
        conf_matrix: 2x2 confusion matrix
        class_name: Name of the class (e.g., 'G')
        config_name: Configuration name (e.g., '15-shot')
        colorscale: Plotly colorscale name (default: 'Blues')
        text_color: Color for heatmap text numbers (default: 'white')
        annotation_text_color: Color for annotation text (default: 'white')
        annotation_bg_color: Background color for annotations (default: 'rgba(0,0,0,0.3)')
        annotation_border_color: Border color for annotations (default: 'white')
        
    Returns:
        plotly.graph_objects.Figure
    """
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=str(int(conf_matrix[i, j])),
                    showarrow=False,
                    font=dict(size=24, color=annotation_text_color, family='Arial Black'),
                    bgcolor=annotation_bg_color,
                    bordercolor=annotation_border_color,
                    borderwidth=2
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted: Not ' + class_name, 'Predicted: ' + class_name],
        y=['Actual: Not ' + class_name, 'Actual: ' + class_name],
        colorscale=colorscale,
        text=conf_matrix.astype(int),
        texttemplate='%{text}',
        textfont={"size": 20, "color": text_color},
        showscale=True,
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=dict(
            text=f"{class_name} - {config_name}",
            font=dict(size=18)
        ),
        height=500,
        width=600,
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14))
    )
    
    return fig

