import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter

# Standard enterprise-ready color palette - Colorful and Premium
# Using qualitative G10 for variety and professional depth
COLORS = px.colors.qualitative.Prism
PAGE_BG = '#f0f9ff' # Light blue page background
CHART_LINE_WIDTH = 2

def set_premium_layout(fig, title, y_title, x_title="Duration (Months)", height=500):
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=26, color='#000000') # Scaled for high visibility
        },
        template="plotly_white",
        height=height,
        hovermode="x unified",
        margin=dict(l=80, r=40, t=100, b=80), # Increased margins to prevent axis "overwrite"
        legend=dict(
            orientation="h", # Horizontal legend below if it fits
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=14, color='#000000')
        ),
        xaxis=dict(
            title={'text': x_title, 'font': {'size': 20, 'color': '#000000'}}, # Solid black text
            tickfont=dict(size=16, color='#000000'),
            gridcolor='#f1f5f9',
            linecolor='#000000', # Solid black axis
            showline=True,
            zeroline=False
        ),
        yaxis=dict(
            title={'text': y_title, 'font': {'size': 20, 'color': '#000000'}}, # Solid black text
            tickfont=dict(size=16, color='#000000'),
            gridcolor='#f1f5f9',
            linecolor='#000000', # Solid black axis
            showline=True,
            zeroline=False
        ),
        plot_bgcolor='white'
    )
    return fig

def plot_km_curves(kmf_dict: dict, title: str = "Kaplan-Meier Survival Probability", p_value: float = None):
    fig = go.Figure()
    
    # Matching the specific colors from the reference image
    ref_colors = ['#EAB308', '#64748B', '#EF4444', '#10B981', '#6366F1']
    
    idx = 0
    for label, kmf in kmf_dict.items():
        surv = kmf.survival_function_
        ci = kmf.confidence_interval_
        upper_col = f"{label}_upper_0.95"
        lower_col = f"{label}_lower_0.95"
        
        color = ref_colors[idx % len(ref_colors)]
        rgba_fill = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)"
        
        # 1. Add Survival line
        fig.add_trace(go.Scatter(
            x=surv.index, y=surv[label],
            mode='lines',
            name=f"{label}",
            line=dict(width=3, color=color)
        ))
        
        # 2. CI Area and Median Crosshairs removed per user request

        
        idx += 1
        
    set_premium_layout(fig, title, "Survival Probability")
    
    # Format Y-axis as percentages (0% - 100%)
    fig.update_yaxes(tickformat=".0%", range=[-0.02, 1.05], gridcolor='#f1f5f9')
    fig.update_xaxes(showgrid=True, gridcolor='#f1f5f9', title="Follow-up Time (Months)")
    
    # Add P-value annotation
    if p_value is not None:
        fig.add_annotation(
            x=0.02, y=0.08, # Shifted to bottom-left to avoid overlap with top-right legend
            xref="paper", yref="paper",
            text=f"<b>Log-rank p = {p_value:.4f}</b>",
            showarrow=False,
            font=dict(size=16, color='#1e3a8a'),
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(30,58,138,0.2)",
            borderwidth=1, borderpad=4
        )
    
    return fig

def plot_hazard_curves(naf_dict: dict, title: str = "Cumulative Hazard Function"):
    fig = go.Figure()
    idx = 0
    for label, naf in naf_dict.items():
        haz = naf.cumulative_hazard_
        color = COLORS[(idx + 2) % len(COLORS)] # Varied color
        
        fig.add_trace(go.Scatter(
            x=haz.index, y=haz[label],
            mode='lines',
            name=f"{label}",
            line=dict(width=CHART_LINE_WIDTH, color=color)
        ))
        idx += 1
        
    set_premium_layout(fig, title, "Cumulative Hazard")
    return fig

def plot_forest_cox(cph, display_map=None):
    summary = cph.summary.copy()
    if isinstance(summary.index, pd.MultiIndex):
        summary.index = ['_'.join(map(str, idx)).strip() for idx in summary.index]
    
    if display_map:
        summary.index = [display_map.get(x, x) for x in summary.index]
        
    summary['variable'] = summary.index
    
    lower_col = next((c for f in ['coef lower 95%', 'lower 0.95'] for c in summary.columns if f in c), None)
    upper_col = next((c for f in ['coef upper 95%', 'upper 0.95'] for c in summary.columns if f in c), None)
    
    summary['Exp_Coef'] = summary['exp(coef)']
    
    fig = px.scatter(
        summary, x='coef', y='variable',
        error_x=(summary[upper_col] - summary['coef']).values if upper_col is not None else None,
        error_x_minus=(summary['coef'] - summary[lower_col]).values if lower_col is not None else None,
        color='Exp_Coef',
        color_continuous_scale='Portland',
        labels={'coef': 'Effect Size (Log HR)', 'variable': 'Covariate'}
    )
    
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8", line_width=2)
    set_premium_layout(fig, "Cox Proportional Hazards Model: Hazard Ratio Forest Plot", "Clinical Predictors", "Log Hazard Ratio")
    return fig

def plot_distribution(series, title="In-Time Cohort Distribution"):
    fig = px.histogram(
        series, 
        nbins=20,
        labels={'value': 'Duration (Months)'},
        color_discrete_sequence=[COLORS[0]]
    )
    set_premium_layout(fig, title, "Patient Count", "Duration (Months)")
    fig.update_layout(bargap=0.1)
    return fig

def plot_metric_comparison(results_dict, metric='aic'):
    df_metrics = pd.DataFrame([
        {'Model': name, 'Metric': val[metric]} 
        for name, val in results_dict.items()
    ])
    df_metrics = df_metrics.sort_values('Metric')
    fig = px.bar(
        df_metrics, x='Model', y='Metric', 
        color='Metric',
        color_continuous_scale='Spectral_r', # Colorful scale
        text_auto='.3s'
    )
    set_premium_layout(fig, f"Model Comparison: Statistical {metric.upper()} Fitness Score", "Score (Lower is Better)", "Statistical Model")
    return fig
